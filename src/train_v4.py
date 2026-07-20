# -*- coding: utf-8 -*-
"""train_v4.py

DeepSeek-V4 自作モデル（edu / pro）の学習・推論スクリプト

使い方:
    # 教育版を学習して生成デモまで実行
    python train_v4.py --model edu --steps 300

    # 忠実版（pro）を学習
    python train_v4.py --model pro --steps 300

    # 学習済みチェックポイントから生成のみ実行
    python train_v4.py --model edu --generate --prompt "深層学習"

    # ハイパーパラメータの上書き例（小さくして高速に動作確認）
    python train_v4.py --model pro --steps 50 --context 64 --d-model 128

学習の流れ（1ステップ）:
    1. コーパスからランダムに (context_size + MTP深さ + 1) 長の窓を切り出す
    2. model.pretrain(source) で
         メイン損失 + λ・MTP損失 + バランス損失 を計算
    3. backward
    4. Muon（2次元重み行列） と AdamW（埋め込み・ヘッド等）で更新
    5. model.update_expert_bias() で MoE のバイアス項を更新
       （補助損失フリーのロードバランス戦略）
"""

import argparse
import os
import time

import torch


# ============================================================
# トイコーパス（自己完結のための埋め込みテキスト）
# ============================================================

CORPUS = """
深層学習は、多層のニューラルネットワークを用いてデータから特徴を自動的に学習する手法である。
大規模言語モデルは、大量のテキストから次の単語を予測するように学習される。
注意機構は、系列の中のどの部分に注目すべきかを学習する仕組みである。
自己注意では、系列内の全てのトークンが互いに注意を向け合う。
混合専門家モデルは、入力ごとに一部の専門家だけを活性化して計算量を抑える。
ルーターは各トークンをどの専門家に送るかを決定する。
負荷分散が崩れると、一部の専門家だけが使われて学習が不安定になる。
回転位置埋め込みは、クエリとキーを回転させて相対位置を表現する。
潜在ベクトルへの圧縮により、キーバリューキャッシュの容量を削減できる。
圧縮された表現は、キーとバリューを兼用することでさらに効率化できる。
疎な注意は、重要なトークンだけを選択して計算量を削減する。
インデクサーは、どの圧縮エントリに注意すべきかを高速に判定する。
残差接続は、深いネットワークでも勾配が流れやすくする仕組みである。
ハイパーコネクションは、残差ストリームを複数本に拡張する。
二重確率行列への制約により、残差変換は非膨張になる。
マルチトークン予測は、複数先のトークンも同時に予測して学習効率を高める。
オプティマイザは、勾配を用いてパラメータを更新する装置である。
ニュートン・シュルツ反復は、行列を近似的に直交化する計算法である。
モデルの学習では、損失関数を最小化するようにパラメータを調整する。
テキスト生成では、学習済みモデルが次のトークンを繰り返し予測する。
"""


# ============================================================
# 文字レベルトークナイザ
# ============================================================

class CharTokenizer:
    """文字単位の簡易トークナイザ（<pad>=0, <bos>=1, <eos>=2）"""

    def __init__(self, text):
        chars = sorted(set(text))
        specials = ['<pad>', '<bos>', '<eos>']
        vocab = specials + chars

        self.word2index = {w: i for i, w in enumerate(vocab)}
        self.index2word = {i: w for i, w in enumerate(vocab)}
        self.pad_token_id = self.word2index['<pad>']
        self.bos_token_id = self.word2index['<bos>']
        self.eos_token_id = self.word2index['<eos>']
        self.vocab_size = len(vocab)

    def encode(self, text, add_bos=False, add_eos=False):
        ids = [self.word2index[ch] for ch in text if ch in self.word2index]
        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]
        return ids

    def decode(self, ids):
        specials = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        return ''.join(self.index2word[i] for i in ids if i not in specials)


# ============================================================
# データ準備
# ============================================================

def build_dataset(tokenizer, device):
    """コーパス全体を1本のID列にする（行末に<eos>を挟む）"""
    ids = []
    for line in CORPUS.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        ids.extend(tokenizer.encode(line, add_eos=True))
    data = torch.tensor(ids, dtype=torch.long, device=device)
    return data


def sample_batch(data, batch_size, window):
    """ランダムな位置から window 長の窓を batch_size 個切り出す"""
    n = data.size(0)
    if n <= window:
        # コーパスが短い場合は繰り返して延長
        repeat = window // n + 2
        data = data.repeat(repeat)
        n = data.size(0)
    starts = torch.randint(0, n - window, (batch_size,), device=data.device)
    return torch.stack([data[s: s + window] for s in starts])


# ============================================================
# 学習
# ============================================================

def train(module, args, tokenizer, data, device, cli):
    model = module.DeepSeekCodeV4(args, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'モデル: {cli.model} / パラメータ数: {n_params:,} / device: {device}')

    muon, adamw = module.create_optimizers(
        model, muon_lr=cli.muon_lr, adamw_lr=cli.adamw_lr)

    # source は context_size + MTP深さ + 1 の長さが必要
    window = args.context_size + args.multi_token_depth + 1

    model.train()
    start_time = time.time()
    for step in range(1, cli.steps + 1):
        source = sample_batch(data, cli.batch_size, window)

        loss = model.pretrain(source)

        muon.zero_grad()
        adamw.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        muon.step()
        adamw.step()

        # 補助損失フリーのロードバランス（バイアス項の更新）
        model.update_expert_bias()

        if step % cli.log_every == 0 or step == 1:
            elapsed = time.time() - start_time
            print(f'step {step:5d} | loss {loss.item():.4f} | {elapsed:.1f}s')

    # チェックポイント保存
    ckpt_path = cli.ckpt or f'deepseekcode_v4{cli.model}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'tokenizer_text': CORPUS,
    }, ckpt_path)
    print(f'チェックポイントを保存しました: {ckpt_path}')
    return model


# ============================================================
# 推論（テキスト生成）
# ============================================================

def load_model(module, cli, device):
    ckpt_path = cli.ckpt or f'deepseekcode_v4{cli.model}.pt'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f'{ckpt_path} が見つかりません。先に学習を実行してください。')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = module.Args(**ckpt['args'])
    model = module.DeepSeekCodeV4(args, device=device)
    model.load_state_dict(ckpt['model_state_dict'])
    tokenizer = CharTokenizer(ckpt['tokenizer_text'])
    return model, tokenizer


def run_generation(model, tokenizer, prompt, device, max_new_tokens=100,
                   top_k=3, temperature=0.8):
    print(f'\n--- 生成デモ ---\nプロンプト: {prompt}\n生成結果: {prompt}', end='')
    input_ids = torch.tensor(tokenizer.encode(prompt),
                             dtype=torch.long, device=device)
    model.generate_text(
        input_ids,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='DeepSeek-V4 学習・推論スクリプト')
    parser.add_argument('--model', choices=['edu', 'pro'], default='edu',
                        help='edu: 教育用簡易実装 / pro: 論文忠実実装')
    parser.add_argument('--steps', type=int, default=300, help='学習ステップ数')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--muon-lr', type=float, default=0.02)
    parser.add_argument('--adamw-lr', type=float, default=3e-4)
    parser.add_argument('--log-every', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ckpt', type=str, default=None,
                        help='チェックポイントの保存/読み込みパス')
    parser.add_argument('--generate', action='store_true',
                        help='学習せずチェックポイントから生成のみ実行')
    parser.add_argument('--prompt', type=str, default='深層学習は')
    parser.add_argument('--max-new-tokens', type=int, default=100)
    # モデルサイズの上書き（動作確認の高速化用）
    parser.add_argument('--context', type=int, default=None)
    parser.add_argument('--d-model', type=int, default=None)
    parser.add_argument('--n-layers', type=int, default=None)
    cli = parser.parse_args()

    torch.manual_seed(cli.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # モデル実装の選択
    if cli.model == 'edu':
        import deepseekcode_v4edu as module
    else:
        import deepseekcode_v4pro as module

    if cli.generate:
        # --- 推論のみ ---
        model, tokenizer = load_model(module, cli, device)
        run_generation(model, tokenizer, cli.prompt, device,
                       max_new_tokens=cli.max_new_tokens)
        return

    # --- 学習 ---
    tokenizer = CharTokenizer(CORPUS)
    data = build_dataset(tokenizer, device)
    print(f'コーパス: {data.size(0):,} トークン / 語彙数: {tokenizer.vocab_size}')

    overrides = {'vocab_size': tokenizer.vocab_size}
    if cli.context is not None:
        overrides['context_size'] = cli.context
        overrides['max_seq_len'] = max(cli.context * 4, 256)
    if cli.d_model is not None:
        overrides['d_model'] = cli.d_model
    if cli.n_layers is not None:
        overrides['n_layers'] = cli.n_layers

    args = module.get_default_args(**overrides)

    model = train(module, args, tokenizer, data, device, cli)

    # 学習後にそのまま生成デモ
    run_generation(model, tokenizer, cli.prompt, device,
                   max_new_tokens=cli.max_new_tokens)


if __name__ == '__main__':
    main()
