# -*- coding: utf-8 -*-
"""
test_modules.py — 教材「Expert検証グラフ」用の テスト・学習・可視化 関数集
========================================================================
方針:
  - パラメータ設定（Args）と モデルのインスタンス化（DeepSeekCode）は
    *Google Colab（ノート）側* で行う。
  - ここには「動作テスト」「四則演算の学習・可視化」「日本語まぜこぜの学習・可視化」
    の **処理だけ** を、引数で model / args / データを受け取る関数として置く。
  - すべて引数で受け取るので、Colab 上で任意のコードを直接書いても組み合わせて使える。

前提: 先に deepseek_modules を読み込み、MoE を注入しておくこと。
    import sys; sys.path.append('DeepSeekCode/src')
    from deepseek_modules import *
    import deepseek_modules as dsm
    import test_modules as tm
    dsm.MoE = MoE   # ← ノートで定義した MoE を注入
"""
import os
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from IPython.display import HTML, display

import deepseek_modules as dsm
from deepseek_modules import create_causal_mask, precompute_freqs_cis


# ============================================================
# 共通ヘルパー
# ============================================================
def _moe_layer_ids(model):
    """MoE が入っている層の番号一覧（layer0 は通常 FFN なので含まれない）。"""
    return [li for li, l in enumerate(model.main_model.layers)
            if isinstance(l.feed_forward, dsm.MoE)]


def _forward(model, args, ids):
    """推論用の前向き計算。out['logits'] などを返す。"""
    mm = model.main_model
    seq_len = ids.size(1)
    mask = create_causal_mask(seq_len)
    freqs = precompute_freqs_cis(args)[:seq_len]
    mm.reset_kv_cache()
    with torch.no_grad():
        out = mm(ids, 0, freqs, mask=mask, train=False)
    return out


def _animate(frame_paths, interval_ms=600, figsize=(9, 5), gif_path=None):
    """保存済みの画像フレームをつなげて、セル出力に動画プレーヤーで表示する。
       interval_ms を大きくするほどゆっくり再生。GIF も保存できる。"""
    fig = plt.figure(figsize=figsize)
    plt.axis('off')
    imgs = [plt.imread(p) for p in frame_paths]
    im = plt.imshow(imgs[0])

    def _update(i):
        im.set_data(imgs[i])
        return [im]

    ani = animation.FuncAnimation(fig, _update, frames=len(imgs),
                                  interval=interval_ms, blit=True)
    plt.close(fig)
    if gif_path is not None:
        ani.save(gif_path, writer=animation.PillowWriter(fps=max(1, 1000 // interval_ms)))
        print('GIF を保存:', gif_path)
    display(HTML(ani.to_jshtml()))
    return ani


# ============================================================
# 1) 基本動作テスト
# ============================================================
def integration_test(model, args, batch_size=2, seed=42):
    """前向き計算・学習・生成がエラーなく動くかを確認する。"""
    torch.manual_seed(seed)
    src = torch.randint(0, args.vocab_size,
                        (batch_size, args.context_size + args.multi_token_depth + 1))
    loss = model.pretrain(src)
    print('事前訓練損失:', round(loss.item(), 4))

    ids = torch.randint(0, args.vocab_size, (batch_size, args.context_size))
    log_probs = model.compute_log_prob(ids, train=False)
    print('log_probs:', tuple(log_probs.shape))

    gen = model.generate_ids(torch.randint(0, args.vocab_size, (1, 5)),
                             max_new_tokens=5, top_k=1)
    print('generated:', tuple(gen.shape))
    return {'loss': loss.item()}


def pattern_demo(model, args, tokens=None, num_epochs=100, lr=1e-3, seed=42, show_plot=True):
    """学習用データ tokens を受け取り、パターン学習を行って結果を確認する。

    tokens: 形 (batch, context_size + 1 + multi_token_depth) の LongTensor。
            受講生が Colab 側で手作業で作って渡す想定。
            None のときだけ、既定パターン（1,2,3→6 / 1,2,3→9）を内部生成する。
    """
    torch.manual_seed(seed)
    if tokens is None:
        batch_size = 2
        tokens = torch.randint(0, args.vocab_size,
                               (batch_size, args.context_size + 1 + args.multi_token_depth))
        tokens[0][1:5] = torch.tensor([1, 2, 3, 6])
        tokens[1][10:14] = torch.tensor([1, 2, 3, 9])

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(num_epochs):
        opt.zero_grad()
        loss = model.pretrain(tokens)
        loss.backward()
        opt.step()
    print('loss', round(loss.item(), 4))

    model.eval()  # テスト時はドロップアウトを止めて安定したロジットにする

    # (1) 「7, 4, 1, 2」 の次 → 3 になってほしい（1,2 の続き）
    out1 = _forward(model, args, torch.LongTensor([[7, 4, 1, 2]]))
    last1 = out1['logits'][0][-1]
    next_after_7412 = int(last1.argmax())
    print('「7, 4, 1, 2」の次に予測したトークン:', next_after_7412, '（期待: 3）')

    # (2) 「1, 2, 3」 の次 → 6 か 9 になってほしい
    out2 = _forward(model, args, torch.LongTensor([[1, 2, 3]]))
    last2 = out2['logits'][0][-1]
    next_after_123 = int(last2.argmax())
    print('「1, 2, 3」の次に予測したトークン:', next_after_123, '（期待: 6 または 9）')

    if show_plot:
        # グラフ1（従来どおり残す）: 「7, 4, 1, 2」の次は 3
        plt.figure(figsize=(7, 3))
        plt.bar(range(10), last1[:10].tolist())
        plt.xticks(range(10))
        plt.xlabel('トークン'); plt.ylabel('ロジット')
        plt.title('「7, 4, 1, 2」の次に来るトークン（3 になってほしい）')
        plt.show()

        # グラフ2（追加）: 「1, 2, 3」の次は 6 か 9
        plt.figure(figsize=(7, 3))
        bars = plt.bar(range(10), last2[:10].tolist())
        bars[6].set_color('crimson'); bars[9].set_color('crimson')  # 6 と 9 を強調
        plt.xticks(range(10))
        plt.xlabel('トークン'); plt.ylabel('ロジット')
        plt.title('「1, 2, 3」の次に来るトークン（6 か 9 になってほしい）')
        plt.show()

    return {'loss': loss.item(),
            'next_after_7412': next_after_7412,
            'next_after_123': next_after_123}


# ============================================================
# 2) 四則演算（数値＋記号）
# ============================================================
def build_arith_dataset(context_size=8, mtp_depth=1):
    """四則演算データセットを作って返す（vocab_size=16）。"""
    PLUS, MINUS, MUL, DIV, EQ, PAD = 10, 11, 12, 13, 14, 15
    OPS = ['+', '-', '*', '/']
    OP_ID = {'+': PLUS, '-': MINUS, '*': MUL, '/': DIV}
    OP_NAME = {'+': '足し算 (+)', '-': '引き算 (−)', '*': 'かけ算 (×)', '/': 'わり算 (÷)'}
    seq_total = context_size + mtp_depth + 1

    def make_eqs(op):
        eqs = []
        for a in range(10):
            for b in range(10):
                if op == '+':
                    r = a + b
                elif op == '-':
                    if a < b:
                        continue
                    r = a - b
                elif op == '*':
                    r = a * b
                else:
                    if b == 0 or a % b != 0:
                        continue
                    r = a // b
                eqs.append([a, OP_ID[op], b, EQ, r // 10, r % 10])
        return eqs

    equations_by_op = {op: make_eqs(op) for op in OPS}

    def pad(e):
        return e + [PAD] * (seq_total - len(e))

    data = torch.tensor([pad(e) for op in OPS for e in equations_by_op[op]], dtype=torch.long)
    return SimpleNamespace(data=data, equations_by_op=equations_by_op, OPS=OPS, OP_NAME=OP_NAME,
                           vocab_size=16, context_size=context_size, mtp_depth=mtp_depth,
                           seq_total=seq_total, PAD=PAD, EQUALS_POS=3, pad=pad)


def collect_expert_usage(model, args, ds, position=None, layer=None):
    """演算ごとに、指定位置（既定は「=」）で選ばれた専門家の回数と正解率を返す。"""
    if position is None:
        position = ds.EQUALS_POS
    model.eval()
    mm = model.main_model
    nr = args.n_routed_experts
    layer_ids = _moe_layer_ids(model)
    tl = layer if layer is not None else layer_ids[-1]
    usage = {op: {li: np.zeros(nr) for li in layer_ids} for op in ds.OPS}
    accuracy = {}
    for op in ds.OPS:
        ids = torch.tensor([ds.pad(e) for e in ds.equations_by_op[op]],
                           dtype=torch.long)[:, :args.context_size]
        out = _forward(model, args, ids)
        pred_t = out['logits'][:, 3, :].argmax(-1)
        pred_o = out['logits'][:, 4, :].argmax(-1)
        accuracy[op] = ((pred_t == ids[:, 4]) & (pred_o == ids[:, 5])).float().mean().item()
        for li in layer_ids:
            topk = mm.layers[li].feed_forward.last_topk_indices.view(ids.size(0), ids.size(1), -1)
            chosen = topk[:, position, :].reshape(-1)
            usage[op][li] += torch.bincount(chosen, minlength=nr).numpy()
    return usage, accuracy


def plot_expert_usage_bar(usage, target_layer, n_routed, ds,
                          title=None, save_path=None, show=True):
    """演算ごとに「どの専門家が使われたか」を棒グラフにする。"""
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b', '#e377c2', '#7f7f7f']
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(n_routed)
    width = 0.8 / len(ds.OPS)
    for k, op in enumerate(ds.OPS):
        v = usage[op][target_layer]
        s = v.sum()
        v = v / s * 100 if s > 0 else v
        ax.bar(x + (k - (len(ds.OPS) - 1) / 2) * width, v, width,
               label=ds.OP_NAME[op], color=colors[k % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels([f'専門家{e}' for e in range(n_routed)])
    ax.set_ylabel('選ばれた割合 (%)')
    ax.set_ylim(0, 100)
    ax.set_title(title or f'各演算で使われた専門家（layer {target_layer}）')
    ax.legend(loc='upper right')
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=80)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def train_arith_with_animation(model, args, ds, num_epochs=400, batch_size=64, lr=2e-3,
                               snapshot_every=10, frame_interval_ms=600,
                               frame_dir='arith_frames', seed=42):
    """四則演算を学習しながら、専門家の使われ方の変化を動画にして表示する。"""
    torch.manual_seed(seed)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    layer_ids = _moe_layer_ids(model)
    target_layer = layer_ids[-1]
    nr = args.n_routed_experts
    os.makedirs(frame_dir, exist_ok=True)
    frames = []
    N = ds.data.size(0)

    def snap(epoch):
        usage, acc = collect_expert_usage(model, args, ds)
        model.train()
        macc = sum(acc.values()) / len(acc)
        p = os.path.join(frame_dir, f'a_{len(frames):04d}.png')
        plot_expert_usage_bar(usage, target_layer, nr, ds,
                              title=f'epoch {epoch}  平均正解率 {macc*100:.1f}%',
                              save_path=p, show=False)
        frames.append(p)

    print(f'学習開始（{num_epochs} エポック, データ {N} 問）')
    snap(0)
    for ep in range(1, num_epochs + 1):
        perm = torch.randperm(N)
        el = 0.0
        for i in range(0, N, batch_size):
            b = ds.data[perm[i:i + batch_size]]
            opt.zero_grad()
            loss = model.pretrain(b)
            loss.backward()
            opt.step()
            for layer in model.main_model.layers:
                if isinstance(layer.feed_forward, dsm.MoE):
                    layer.feed_forward.update_expert_bias()
            el += loss.item()
        if ep % snapshot_every == 0 or ep == num_epochs:
            snap(ep)
        if ep % 50 == 0 or ep == num_epochs:
            print(f'  epoch {ep:4d}  loss {el / (N / batch_size):.4f}')
    print(f'学習終了（コマ数 {len(frames)}）')
    _animate(frames, interval_ms=frame_interval_ms, figsize=(9, 5),
             gif_path=os.path.join(frame_dir, 'arith.gif'))
    return {'frame_dir': frame_dir, 'target_layer': target_layer}


def arith_addition_report(usage, target_layer, ds, n_routed):
    """各演算の主担当専門家と、足し算担当を表示する。"""
    print('=== 各演算で一番使われた専門家 ===')
    for op in ds.OPS:
        v = usage[op][target_layer]
        v = v / v.sum() * 100
        b = int(np.argmax(v))
        print(f'{ds.OP_NAME[op]:9s} → 専門家{b} ({v[b]:.0f}%)')
    av = usage['+'][target_layer]
    av = av / av.sum() * 100
    ab = int(np.argmax(av))
    print(f'\n▶ 足し算でいちばん使われたのは専門家{ab}（{av[ab]:.0f}%）')


# ============================================================
# 3) 日本語まぜこぜ（数字・記号・日本語）
# ============================================================
def build_jp_dataset(context_size=10, mtp_depth=1):
    """記号・日本語・質問の3形式をまぜたデータセットを返す（vocab_size=22）。"""
    ID2TOK = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
              '+', '-', '*', '/', '=',
              'たす', 'ひく', 'かける', 'わる', 'は', '？', 'PAD']
    TOK2ID = {t: i for i, t in enumerate(ID2TOK)}
    PAD = TOK2ID['PAD']
    OPS = ['+', '-', '*', '/']
    OP_WORD = {'+': 'たす', '-': 'ひく', '*': 'かける', '/': 'わる'}
    CATEGORIES = ['数字', '記号', '日本語']

    def token_category(t):
        if 0 <= t <= 9:
            return '数字'
        if 10 <= t <= 14:
            return '記号'
        if 15 <= t <= 20:
            return '日本語'
        return None

    seq_total = context_size + mtp_depth + 1

    def valid_pairs(op):
        pairs = []
        for a in range(10):
            for b in range(10):
                if op == '+':
                    r = a + b
                elif op == '-':
                    if a < b:
                        continue
                    r = a - b
                elif op == '*':
                    r = a * b
                else:
                    if b == 0 or a % b != 0:
                        continue
                    r = a // b
                pairs.append((a, b, r))
        return pairs

    def forms(a, op, b, r):
        te, on = r // 10, r % 10
        return [[a, TOK2ID[op], b, TOK2ID['='], te, on],
                [a, TOK2ID[OP_WORD[op]], b, TOK2ID['は'], te, on],
                [a, TOK2ID[OP_WORD[op]], b, TOK2ID['は'], TOK2ID['？'], te, on]]

    def pad(s):
        return s + [PAD] * (seq_total - len(s))

    seqs = [pad(f) for op in OPS for (a, b, r) in valid_pairs(op) for f in forms(a, op, b, r)]
    data = torch.tensor(seqs, dtype=torch.long)
    return SimpleNamespace(data=data, ID2TOK=ID2TOK, TOK2ID=TOK2ID, OPS=OPS, OP_WORD=OP_WORD,
                           CATEGORIES=CATEGORIES, token_category=token_category,
                           vocab_size=len(ID2TOK), context_size=context_size,
                           mtp_depth=mtp_depth, seq_total=seq_total, PAD=PAD, pad=pad)


def collect_usage_by_category(model, args, ds, sequences=None, layer=None):
    """トークンの種類(数字/記号/日本語)ごとに、選ばれた専門家の回数を数える。"""
    model.eval()
    mm = model.main_model
    nr = args.n_routed_experts
    layer_ids = _moe_layer_ids(model)
    tl = layer if layer is not None else layer_ids[-1]
    seqs = ds.data if sequences is None else sequences
    ids = seqs[:, :args.context_size]
    _forward(model, args, ids)
    topk = mm.layers[tl].feed_forward.last_topk_indices.view(ids.size(0), ids.size(1), -1)
    usage = {c: np.zeros(nr) for c in ds.CATEGORIES}
    tn, kn = ids.numpy(), topk.numpy()
    for b in range(ids.size(0)):
        for t in range(ids.size(1)):
            cat = ds.token_category(int(tn[b, t]))
            if cat is None:
                continue
            for e in kn[b, t]:
                usage[cat][int(e)] += 1
    return usage, tl


def separation_score(usage, ds):
    """各カテゴリの「一番使われた専門家の割合」の平均。1に近いほど分業がはっきり。"""
    sc = [usage[c].max() / usage[c].sum() for c in ds.CATEGORIES if usage[c].sum() > 0]
    return float(np.mean(sc)) if sc else 0.0


def plot_category_heatmap(usage, n_routed, ds, title=None, save_path=None, show=True):
    """トークンの種類 × 専門家 のヒートマップ。"""
    M = np.stack([usage[c] / usage[c].sum() * 100 if usage[c].sum() > 0 else usage[c]
                  for c in ds.CATEGORIES])
    fig, ax = plt.subplots(figsize=(1.2 * n_routed + 2, 3.2))
    im = ax.imshow(M, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(range(n_routed))
    ax.set_xticklabels([f'専門家{e}' for e in range(n_routed)])
    ax.set_yticks(range(len(ds.CATEGORIES)))
    ax.set_yticklabels(ds.CATEGORIES)
    for i in range(len(ds.CATEGORIES)):
        for j in range(n_routed):
            ax.text(j, i, f'{M[i, j]:.0f}', ha='center', va='center',
                    color='white' if M[i, j] > 55 else 'black', fontsize=9)
    ax.set_title(title or 'トークンの種類 × 使われた専門家')
    plt.colorbar(im, label='その種類で選ばれた割合 (%)')
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=80)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def show_token_experts(model, args, token_ids, ds, layer=None, title=None):
    """1つの文を、各トークンを担当した専門家の色で塗り分けて表示する。"""
    model.eval()
    mm = model.main_model
    layer_ids = _moe_layer_ids(model)
    tl = layer if layer is not None else layer_ids[-1]
    ids = torch.tensor([token_ids], dtype=torch.long)[:, :args.context_size]
    _forward(model, args, ids)
    topk = mm.layers[tl].feed_forward.last_topk_indices.view(1, ids.size(1), -1)
    experts = topk[0, :, 0].tolist()
    toks = [ds.ID2TOK[int(t)] for t in ids[0].tolist()]
    cmap = plt.get_cmap('tab10')

    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(toks)), 1.9))
    ax.set_xlim(0, len(toks))
    ax.set_ylim(0, 1)
    ax.axis('off')
    used = {}
    for i, (tok, e) in enumerate(zip(toks, experts)):
        if tok == 'PAD':
            continue
        ax.add_patch(Rectangle((i + 0.05, 0.25), 0.9, 0.5,
                               facecolor=cmap(e % 10), alpha=0.75, edgecolor='gray'))
        ax.text(i + 0.5, 0.5, tok, ha='center', va='center', fontsize=16)
        used[e] = cmap(e % 10)
    handles = [Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.75) for c in used.values()]
    ax.legend(handles, [f'専門家{e}' for e in used], loc='upper center',
              ncol=max(1, len(used)), bbox_to_anchor=(0.5, 1.3), fontsize=9, frameon=False)
    ax.set_title(title or ('文: ' + ' '.join(t for t in toks if t != 'PAD')), fontsize=11)
    plt.show()


def train_jp_with_animation(model, args, ds, num_epochs=400, batch_size=64, lr=2e-3,
                            snapshot_every=10, frame_interval_ms=600,
                            frame_dir='jp_frames', sample_size=200, seed=0):
    """日本語まぜこぜを学習しながら、種類ごとの分業の現れ方を動画にして表示する。"""
    torch.manual_seed(seed)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    layer_ids = _moe_layer_ids(model)
    target_layer = layer_ids[-1]
    nr = args.n_routed_experts
    os.makedirs(frame_dir, exist_ok=True)
    frames = []
    N = ds.data.size(0)
    torch.manual_seed(seed + 1)
    sample = ds.data[torch.randperm(N)[:sample_size]]

    def snap(epoch):
        usage, _ = collect_usage_by_category(model, args, ds, sequences=sample)
        model.train()
        sep = separation_score(usage, ds)
        p = os.path.join(frame_dir, f'j_{len(frames):04d}.png')
        plot_category_heatmap(usage, nr, ds,
                              title=f'epoch {epoch}  分離スコア {sep:.2f}',
                              save_path=p, show=False)
        frames.append(p)

    print(f'学習開始（{num_epochs} エポック, データ {N} 問）')
    snap(0)
    for ep in range(1, num_epochs + 1):
        perm = torch.randperm(N)
        el = 0.0
        for i in range(0, N, batch_size):
            b = ds.data[perm[i:i + batch_size]]
            opt.zero_grad()
            loss = model.pretrain(b)
            loss.backward()
            opt.step()
            for layer in model.main_model.layers:
                if isinstance(layer.feed_forward, dsm.MoE):
                    layer.feed_forward.update_expert_bias()
            el += loss.item()
        if ep % snapshot_every == 0 or ep == num_epochs:
            snap(ep)
        if ep % 50 == 0 or ep == num_epochs:
            print(f'  epoch {ep:4d}  loss {el / (N / batch_size):.4f}')
    print(f'学習終了（コマ数 {len(frames)}）')
    _animate(frames, interval_ms=frame_interval_ms, figsize=(1.2 * nr + 2, 3.2),
             gif_path=os.path.join(frame_dir, 'jp.gif'))
    return {'frame_dir': frame_dir, 'target_layer': target_layer}


def jp_category_report(usage, target_layer, ds):
    """各カテゴリの主担当専門家を表示する。"""
    print('各カテゴリの主担当専門家:')
    for c in ds.CATEGORIES:
        v = usage[c]
        print(f'  {c:4s} → 専門家{int(np.argmax(v))}')


# 公開用エイリアス（Colab から MoE 層番号を取得するため）
moe_layer_ids = _moe_layer_ids
