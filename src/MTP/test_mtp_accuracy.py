"""
MTPによる次トークン予測精度向上の確認サンプル

MTP (Multi-Token Prediction) は訓練補助タスクとして機能し、
共有パラメータ（embedding・output_head）により豊かな文脈表現を
学ばせることで、次トークン予測精度を向上させます。

比較条件:
  - MTPなし: lambda_mtp=0.0  → mtp_losses が損失に加算されない
  - MTPあり: lambda_mtp=0.3  → mtp_losses が損失に加算される

推論は両条件とも main_model のみを使用します。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from deepseekcode import DeepSeekCode, Args, create_causal_mask

# ── ハイパーパラメータ ──────────────────────────────
VOCAB_SIZE = 10
D_MODEL    = 16
N_HEADS    = 4
CONTEXT    = 8
MTP_DEPTH  = 1
N_TRAIN    = 400
N_VAL      = 100
N_EPOCHS   = 300
LR         = 3e-3
BATCH_SIZE = 8
SEED       = 42


def make_args(lambda_mtp=0.0):
    return Args(
        vocab_size          = VOCAB_SIZE,
        d_model             = D_MODEL,
        n_layers            = 1,
        context_size        = CONTEXT,
        norm_eps            = 1e-5,
        n_heads             = N_HEADS,
        max_seq_len         = 64,
        rope_theta          = 10000,
        d_c                 = 8,
        d_cQ                = 16,
        d_rope              = 8,
        d_head              = 24,
        n_shared_experts    = 1,
        n_routed_experts    = 3,
        n_activated_experts = 1,
        moe_bias_update_speed = 0.01,
        moe_alpha           = 0.001,
        moe_inter_dim       = 16,
        multi_token_depth   = MTP_DEPTH,
        lambda_mtp          = lambda_mtp,
    )


# ── データセット生成 ────────────────────────────────
def make_dataset(n_samples, seed=0):
    """
    決定論的な次トークン規則をもつ合成データを生成する。

      token[t+1] = (token[t] * 3 + 2) % VOCAB_SIZE

    隣接する複数トークンの依存関係を持つため、
    モデルは前のトークン列全体を活用する必要がある。
    """
    torch.manual_seed(seed)
    length = CONTEXT + MTP_DEPTH + 1  # main + MTP それぞれのターゲットを含む長さ
    seqs = []
    for _ in range(n_samples):
        start = torch.randint(0, VOCAB_SIZE, (1,)).item()
        seq = [start]
        for _ in range(length - 1):
            seq.append((seq[-1] * 3 + 2) % VOCAB_SIZE)
        seqs.append(torch.tensor(seq, dtype=torch.long))
    return torch.stack(seqs)


# ── 評価 ────────────────────────────────────────────
def evaluate(model, val_data):
    """
    main_model のみで次トークン予測を評価する（推論時は MTP 不使用）。
    返り値: (cross_entropy_loss, accuracy)
    """
    model.main_model.eval()
    model.main_model.reset_kv_cache()

    inp  = val_data[:, :CONTEXT]
    tgt  = val_data[:, 1:CONTEXT + 1]
    freq = model.freqs_cis[:CONTEXT]
    mask = create_causal_mask(CONTEXT)

    with torch.no_grad():
        out = model.main_model(inp, 0, freq, mask, train=False)

    logits = out['logits']
    loss   = F.cross_entropy(logits.view(-1, VOCAB_SIZE), tgt.view(-1)).item()
    acc    = (logits.argmax(-1) == tgt).float().mean().item()
    return loss, acc


# ── 訓練 ────────────────────────────────────────────
def train(label, args, train_data, val_data):
    """モデルを訓練して各 50 エポックごとのバリデーション指標を返す。"""
    torch.manual_seed(SEED)
    model = DeepSeekCode(args)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    n     = train_data.size(0)
    log   = []

    for epoch in range(1, N_EPOCHS + 1):
        model.main_model.train()
        for mtp in model.mtp_modules:
            mtp.train()

        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            batch = train_data[perm[i:i + BATCH_SIZE]]
            opt.zero_grad()
            loss = model.pretrain(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if epoch % 50 == 0:
            val_loss, val_acc = evaluate(model, val_data)
            log.append((epoch, val_loss, val_acc))
            print(f"  [{label:10s}] epoch {epoch:3d} | "
                  f"val_loss={val_loss:.4f}  acc={val_acc:.3f}")

    return log


# ── エントリポイント ────────────────────────────────
def main():
    print()
    print("=== MTPによる次トークン予測精度の比較 ===")
    print()
    print(f"データ : token[t+1] = (token[t]*3+2) % {VOCAB_SIZE}  (決定論的規則)")
    print(f"訓練   : {N_TRAIN} サンプル   バリデーション: {N_VAL} サンプル")
    print(f"設定   : context_size={CONTEXT}, multi_token_depth={MTP_DEPTH}, "
          f"epochs={N_EPOCHS}")
    print()

    train_data = make_dataset(N_TRAIN, seed=0)
    val_data   = make_dataset(N_VAL,   seed=1)

    # ── MTP なし ────────────────────────────────────
    print("--- MTPなし (lambda_mtp=0.0) ---")
    print("  訓練損失 = main_loss のみ")
    log_base = train("MTPなし", make_args(lambda_mtp=0.0), train_data, val_data)

    print()

    # ── MTP あり ────────────────────────────────────
    print("--- MTPあり (lambda_mtp=0.3) ---")
    print("  訓練損失 = main_loss + 0.3 * mtp_loss")
    log_mtp  = train("MTPあり", make_args(lambda_mtp=0.3), train_data, val_data)

    # ── 最終比較 ────────────────────────────────────
    print()
    print("=" * 50)
    print("最終結果 (epoch {})".format(N_EPOCHS))
    print("=" * 50)
    final_base = log_base[-1]
    final_mtp  = log_mtp[-1]

    print(f"  {'条件':<16} {'val_loss':>10} {'accuracy':>10}")
    print("  " + "-" * 38)
    print(f"  {'MTPなし (λ=0.0)':<16} {final_base[1]:>10.4f} {final_base[2]:>10.3f}")
    print(f"  {'MTPあり (λ=0.3)':<16} {final_mtp[1]:>10.4f} {final_mtp[2]:>10.3f}")

    delta_loss = final_base[1] - final_mtp[1]
    delta_acc  = (final_mtp[2] - final_base[2]) * 100
    print()
    if delta_loss > 0:
        print(f"  val_loss 改善 : -{delta_loss:.4f} "
              f"({delta_loss / final_base[1] * 100:.1f}% 低下)")
    else:
        print(f"  val_loss 差   : {delta_loss:+.4f}")
    if delta_acc > 0:
        print(f"  accuracy 向上 : +{delta_acc:.1f} ポイント")
    else:
        print(f"  accuracy 差   : {delta_acc:+.1f} ポイント")

    # ── 収束速度の比較 ──────────────────────────────
    print()
    print("エポック別 accuracy 推移:")
    print(f"  {'epoch':>6} | {'MTPなし':>10} | {'MTPあり':>10}")
    print("  " + "-" * 32)
    for (e_b, _, acc_b), (e_m, _, acc_m) in zip(log_base, log_mtp):
        print(f"  {e_b:>6} | {acc_b:>10.3f} | {acc_m:>10.3f}")

    print()
    print("[解説]")
    print("  MTP は next-token 予測に加えて +1 先のトークンも予測する補助タスクです。")
    print("  この追加の勾配信号が共有 embedding・output_head の更新を強化し、")
    print("  各トークン位置でより豊かな文脈表現を学習させることで")
    print("  main_model の次トークン予測精度を向上させます。")
    print()


if __name__ == "__main__":
    main()
