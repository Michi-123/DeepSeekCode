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
    def __init__(self, args, dropout=0.1):
        super().__init__()
        
        """ 引数の設定 """
        # モデル全体の隠れ状態次元
        self.d_model = args.d_model
        # Attention ヘッド数
        self.n_heads = args.n_heads
        # multi-head dimension
        self.d_h = args.d_head
        # KVキャッシュの最大長
        self.context_size = args.context_size

        # MLA比較用の出力次元（既存実装に合わせる）
        self.fc_q = nn.Linear(self.d_model, self.n_heads * self.d_h)
        self.fc_k = nn.Linear(self.d_model, self.n_heads * self.d_h)
        self.fc_v = nn.Linear(self.d_model, self.n_heads * self.d_h)

        self.attention = ScaledDotProductAttention(self.d_model, dropout)
        self.output_head = nn.Linear(self.n_heads * self.d_h, self.d_model)
        self.dropout = nn.Dropout(dropout)

        # Xavier init（既存実装に合わせる）
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.output_head.weight)

        # KVキャッシュ（初期化は forward 時に行う）
        self.kv_cache = None
        self.registered_batch_size = None

    def reset_kv_cache(self):
        """
        KVキャッシュを明示的にリセットする関数。
        """
        if self.kv_cache:
            self.kv_cache.reset()
        self.registered_batch_size = None

    def forward(self, h, freqs_cis, mask=None, train=False):
        """
        Args:
            h: 入力テンソル (batch_size, seq_len, d_model)
            freqs_cis: RoPE 用の複素周波数（precompute_freqs_cis の出力）
            mask: (seq_len, seq_len) など（学習時のみ使用想定）。
            train: 学習時 True / 推論時 False（False のとき KVCache 使用）。
        """
        batch_size, seq_len, _ = h.shape
        
        # 最初の推論ではキャッシュを初期化
        if (not train) and (self.kv_cache is None):
            self.kv_cache = KVCache(self.context_size)

        # 1) Q, K, V を作成
        q = self.fc_q(h)
        k = self.fc_k(h)
        v = self.fc_v(h)

        # 2) (batch_size, seq_len, n_heads, d_h) へ整形
        q = q.reshape(batch_size, seq_len, self.n_heads, self.d_h)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.d_h)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.d_h)

        # 3) RoPE を Q, K に適用
        #    freqs: (seq_len, d_h//2) を切り出し、apply_rope は (batch_size, seq_len, n_heads, d_h) を想定
        freqs = freqs_cis[:seq_len, : self.d_h // 2].to(q.device)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        # 4) 推論時のみ KVCache を使用
        if not train:
            # （注意）KVCache は seq 次元(dim=1)で結合する設計
            # ここでは (batch_size, seq_len, n_heads, d_h) 形状のまま渡す
            self.kv_cache.update(k, v)
            k, v = self.kv_cache.get()  # k, v: (batch_size, seq_len_total, n_heads, d_h)

        # 5) Attention へ（ScaledDotProductAttention は (batch_size, n_heads, seq_len, d_h) を想定）
        q = q.transpose(1, 2)  # (batch_size, n_heads, seq_len, d_h)
        k = k.transpose(1, 2)  # (batch_size, n_heads, seq_len_k, d_h)
        v = v.transpose(1, 2)  # (batch_size, n_heads, seq_len_k, d_h)

        # 生成時（seq=1）に KVCache を使う場合、mask は (1,1) で問題なし。
        # 学習時は通常の (seq_len,seq_len) マスクをそのまま渡す。
        h, w = self.attention(q, k, v, mask=mask)

        # 6) ヘッド結合 & 出力
        h = h.transpose(1, 2)
        h = h.reshape(batch_size, seq_len, self.n_heads * self.d_h)
        h = self.output_head(h)
        h = self.dropout(h)
        
        output = {}
        output['hidden_state'] = h
        output['attention_weight'] = w
        
        return output