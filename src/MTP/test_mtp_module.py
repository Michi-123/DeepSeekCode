"""
MTPモジュール（Multi-Token Prediction）の動作確認サンプル

DeepSeek-V3 の MTP は訓練時に次トークン予測と並列して
+1〜+N ステップ先のトークンも予測する補助タスクです。

  MainModel  →  hidden_state  →  MTPModule[0]  →  hidden_state  →  MTPModule[1] ...
      ↓                                ↓
  main_loss                       mtp_loss[0]

訓練損失: total = main_loss + λ * Σ mtp_loss[k]

このサンプルでは以下を確認します:
  1. MTPModule の構造とパラメータ共有
  2. フォワードパスの入出力形状
  3. pretrain() による損失の内訳
  4. 勾配が共有パラメータへ正しく伝播すること
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from deepseekcode import DeepSeekCode, Args, create_causal_mask

# ── ハイパーパラメータ ──────────────────────────────
VOCAB_SIZE = 20
D_MODEL    = 16
N_HEADS    = 4
CONTEXT    = 8
MTP_DEPTH  = 1


def make_args(**overrides):
    base = dict(
        vocab_size         = VOCAB_SIZE,
        d_model            = D_MODEL,
        n_layers           = 1,
        context_size       = CONTEXT,
        norm_eps           = 1e-5,
        n_heads            = N_HEADS,
        max_seq_len        = 64,
        rope_theta         = 10000,
        d_c                = 8,     # KV 圧縮次元 (MLA)
        d_cQ               = 16,    # Q 圧縮次元  (d_cQ > d_c)
        d_rope             = 8,     # RoPE 次元
        d_head             = 24,    # ヘッド次元
        n_shared_experts   = 1,
        n_routed_experts   = 3,
        n_activated_experts= 1,
        moe_bias_update_speed = 0.01,
        moe_alpha          = 0.001,
        moe_inter_dim      = 16,
        multi_token_depth  = MTP_DEPTH,
        lambda_mtp         = 0.1,
    )
    base.update(overrides)
    return Args(**base)


# ── チェック 1: モジュール構造 ───────────────────────
def check_module_structure():
    print("=" * 58)
    print("1. MTPModule 構造とパラメータ共有の確認")
    print("=" * 58)

    args  = make_args(multi_token_depth=2)
    model = DeepSeekCode(args)
    mtp   = model.mtp_modules[0]

    print(f"  multi_token_depth : {args.multi_token_depth}")
    print(f"  MTPModule 数       : {len(model.mtp_modules)}")
    print()
    print("  MTPModule[0] の構成パーツ:")
    print(f"    embedding    : {list(mtp.embedding.weight.shape)}"
          "  ← main_model と重み共有")
    print(f"    norm_pres    : RMSNorm({D_MODEL})  現トークン埋め込みを正規化")
    print(f"    norm_prev    : RMSNorm({D_MODEL})  前モジュール隠れ状態を正規化")
    print(f"    projection   : Linear({D_MODEL*2} → {D_MODEL})  結合後の次元削減")
    print(f"    transformer  : TransformerBlock(MLA + MoE)")
    print(f"    output_norm  : RMSNorm({D_MODEL})")
    print(f"    output_head  : {list(mtp.output_head.weight.shape)}"
          "  ← main_model と重み共有")
    print()

    emb_shared  = model.embedding.weight.data_ptr() == mtp.embedding.weight.data_ptr()
    head_shared = model.output_head.weight.data_ptr() == mtp.output_head.weight.data_ptr()
    print(f"  embedding  重み共有 : {emb_shared}")
    print(f"  output_head 重み共有: {head_shared}")
    print()


# ── チェック 2: フォワードパス ───────────────────────
def check_forward_pass():
    print("=" * 58)
    print("2. MTPModule フォワードパスと出力形状の確認")
    print("=" * 58)

    args  = make_args()
    model = DeepSeekCode(args)
    mtp   = model.mtp_modules[0]

    B, T = 2, CONTEXT
    # MTP depth=1 のとき入力は 1 トークン先へシフト済みのシーケンス
    input_ids = torch.randint(0, VOCAB_SIZE, (B, T))
    # h_prev: main_model が出力した隠れ状態
    h_prev    = torch.randn(B, T, D_MODEL)
    # RoPE 位相: MTP offset=1 なので freqs_cis[1:T+1]
    freqs_cis = model.freqs_cis[1:T + 1]

    with torch.no_grad():
        out = mtp(input_ids, h_prev, start_pos=0, freqs_cis=freqs_cis)

    print("  入力:")
    print(f"    input_ids  : {list(input_ids.shape)}  (batch, seq_len)")
    print(f"    h_prev     : {list(h_prev.shape)}  (main_model の隠れ状態)")
    print(f"    freqs_cis  : {list(freqs_cis.shape)}  (RoPE 位相、複素数)")
    print()
    print("  処理フロー:")
    print("    embedding(input_ids) → norm_pres   ← 現トークン")
    print("    h_prev               → norm_prev   ← 前モジュール隠れ状態")
    print("    cat([h_curr, h_prev], dim=-1) → projection")
    print("    → TransformerBlock → output_norm → output_head")
    print()
    print("  出力:")
    print(f"    logits          : {list(out['logits'].shape)}  (+1 先トークンの予測分布)")
    print(f"    hidden_state    : {list(out['hidden_state'].shape)}")
    print(f"    attention_weight: {list(out['attention_weight'].shape)}")
    print()


# ── チェック 3: 損失の内訳 ──────────────────────────
def check_loss_breakdown():
    print("=" * 58)
    print("3. pretrain() による損失計算の確認")
    print("=" * 58)

    args  = make_args(lambda_mtp=0.1)
    model = DeepSeekCode(args)

    # source 形状: (B, context_size + depth + 1)
    source = torch.randint(0, VOCAB_SIZE, (2, CONTEXT + MTP_DEPTH + 1))

    main_out = model._calculate_main_loss(source)
    mtp_out  = model._calculate_mtp_loss(source, main_out['hidden_state'])

    ml   = main_out['main_loss'].item()
    mbl  = main_out['main_balance_loss'].item()
    mtpl = mtp_out['mtp_losses'].item()
    mtpb = mtp_out['mtp_balance_losses'].item()
    lam  = args.lambda_mtp

    print(f"  main_loss          : {ml:.4f}")
    print(f"  main_balance_loss  : {mbl:.6f}  (MoE 負荷分散)")
    print(f"  mtp_losses         : {mtpl:.4f}")
    print(f"  mtp_balance_losses : {mtpb:.6f}  (MoE 負荷分散)")
    print()
    total = ml + lam * mtpl + mbl + lam * mtpb
    print(f"  total_loss = main_loss + λ·mtp_losses + balance 項")
    print(f"             = {ml:.4f} + {lam}·{mtpl:.4f} + {mbl:.6f} + {lam}·{mtpb:.6f}")
    print(f"             ≈ {total:.4f}")
    print()


# ── チェック 4: 勾配伝播 ────────────────────────────
def check_gradient_flow():
    print("=" * 58)
    print("4. 勾配伝播の確認 (λ=0.3)")
    print("=" * 58)

    args  = make_args(lambda_mtp=0.3)
    model = DeepSeekCode(args)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

    source = torch.randint(0, VOCAB_SIZE, (2, CONTEXT + MTP_DEPTH + 1))

    opt.zero_grad()
    loss = model.pretrain(source)
    loss.backward()

    def grad_norm(params):
        return sum(
            p.grad.norm().item() ** 2
            for p in params if p.grad is not None
        ) ** 0.5

    print(f"  main_model  の勾配ノルム  : "
          f"{grad_norm(model.main_model.parameters()):.4f}")
    print(f"  mtp_modules の勾配ノルム  : "
          f"{grad_norm(model.mtp_modules[0].parameters()):.4f}")
    print(f"  共有 embedding の勾配ノルム: "
          f"{model.embedding.weight.grad.norm().item():.4f}")
    print()
    print("  → 共有パラメータには main_loss と mtp_loss の")
    print("    両方から勾配が流れます。")
    print()


# ── エントリポイント ────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    print()
    print("=== MTPModule (Multi-Token Prediction) 動作確認 ===")
    print()
    check_module_structure()
    check_forward_pass()
    check_loss_breakdown()
    check_gradient_flow()
    print("=== 全チェック完了 ===")
