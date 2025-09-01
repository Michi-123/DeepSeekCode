import torch
import torch.nn as nn

#@title MHA
class MHA(nn.Module):
    """
    Multi-Head Attention with RoPE + KVCache support.

    変更点（要約）:
      - KVキャッシュ（self.kv_cache）を追加し、推論時（train=False）のみ使用。
      - RoPE適用後の k, v を (B, S, H, D) 形状のまま KVCache に蓄積 → 取得。
      - 取得後に (B, H, S, D) に転置して Attention を実行。
      - reset_kv_cache() を実装（生成前に各層で呼び出し可能）。
    前提:
      - Args に context_size が存在すること（既存ファイルと同一）。
      - apply_rope(), ScaledDotProductAttention は既存のものを使用。
    """
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.d_model = args.d_model
        self.d_head = args.d_head

        d_model = args.d_model
        # MLA比較用の出力次元（既存実装に合わせる）
        self.fc_q = nn.Linear(d_model, self.n_heads * self.d_head)
        self.fc_k = nn.Linear(d_model, self.n_heads * self.d_head)
        self.fc_v = nn.Linear(d_model, self.n_heads * self.d_head)

        dropout = 0.1
        self.attention = ScaledDotProductAttention(d_model, dropout)
        self.fc = nn.Linear(self.n_heads * self.d_head, d_model)
        self.dropout = nn.Dropout(dropout)

        # Xavier init（既存実装に合わせる）
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc.weight)

        # --- KVCache 追加 ---
        self.context_size = args.context_size
        self.kv_cache = None
        self.registered_batch_size = None

    def reset_kv_cache(self):
        """外部から明示的にキャッシュをクリアするための関数。"""
        if self.kv_cache:
            self.kv_cache.reset()
        self.registered_batch_size = None

    def forward(self, x, freqs_cis, mask=None, train=False):
        """
        x: (N, S, d_model)
        freqs_cis: RoPE 用の複素周波数（precompute_freqs_cis の出力）
        mask: (S, S) など（学習時のみ使用想定）。
        train: 学習時 True / 推論時 False（False のとき KVCache 使用）。
        """
        N, S = x.size(0), x.size(1)
        H = self.n_heads
        D = self.d_head

        # 初期化（バッチサイズが変わった場合などもここで再生成して良い）
        if (not train) and (self.kv_cache is None):
            self.kv_cache = KVCache(self.context_size)

        # 1) Q, K, V を作成
        q = self.fc_q(x)
        k = self.fc_k(x)
        v = self.fc_v(x)

        # 2) (N, S, H, D) へ整形
        q = q.view(N, S, H, D)
        k = k.view(N, S, H, D)
        v = v.view(N, S, H, D)

        # 3) RoPE を Q, K に適用
        #    freqs: (S, D//2) を切り出し、apply_rope は (N, S, H, D) を想定
        freqs = freqs_cis[:S, : D // 2].to(q.device)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        # 4) 推論時のみ KVCache を使用
        if not train:
            # （注意）KVCache は seq 次元(dim=1)で結合する設計
            # ここでは (N, S, H, D) 形状のまま渡す
            self.kv_cache.update(k, v)
            k, v = self.kv_cache.get()  # k, v: (N, S_total, H, D)

        # 5) Attention へ（ScaledDotProductAttention は (N, H, T, D) を想定）
        q = q.transpose(1, 2)  # (N, H, S_q, D)
        k = k.transpose(1, 2)  # (N, H, S_k, D)
        v = v.transpose(1, 2)  # (N, H, S_k, D)

        # 生成時（seq=1）に KVCache を使う場合、mask は (1,1) で問題なし。
        # 学習時は通常の (S,S) マスクをそのまま渡す。
        x, attn_weight = self.attention(q, k, v, mask=mask)

        # 6) ヘッド結合 & 出力
        x = x.transpose(1, 2).contiguous().view(N, -1, H * D)
        x = self.fc(x)
        x = self.dropout(x)
        output = {}
        output["hidden_state"] = x
        output["attn_weight"] = attn_weight
        return output