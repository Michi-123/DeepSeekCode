# @title MLA
class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA)

    複数の射影空間（latent Q/K/V）を用いて効率的に情報を抽出するTransformerのアテンションブロック。
    本クラスでは KV キャッシュを外部クラス (KVCache) に分離して再利用可能にした構造になっている。
    """

    def __init__(self, args):
        super().__init__()
        # モデル全体の隠れ状態次元
        self.d_model = args.d_model
        # Attention ヘッド数
        self.n_heads = args.n_heads

        # latent空間の次元数
        self.d_cQ = args.d_cQ  # 10%～30%の圧縮
        self.d_c = args.d_c  # 5%～10%の圧縮

        # multi-head dimension
        self.d_h = args.d_head
        self.d_hR = args.d_rope # ヘッドの次元の25%～50%

        # Q, K, V の射影層（入力 -> latent表現）
        # 入力 -> latent Q
        self.q_down_proj = nn.Linear(self.d_model, self.d_cQ)
        # 入力 -> latent K/V
        self.kv_down_proj = nn.Linear(self.d_model, self.d_c)

        # latent Q に対する正規化
        self.q_norm = RMSNorm(self.d_cQ)
        self.kv_norm = RMSNorm(self.d_c)

        # latent Q -> 再構成された Qc / Qr（2つの役割で別々に使う）
        self.qc_up_proj = nn.Linear(self.d_cQ, self.n_heads * self.d_h)
        self.qr_up_proj = nn.Linear(self.d_cQ, self.n_heads * self.d_hR)

        # 入力から直接得られる Kr（回転対象）
        self.kr_proj = nn.Linear(self.d_model, self.d_hR)
        self.kr_norm = RMSNorm(self.d_hR)

        # latent K/V から再構成された Kc, Vc
        self.kc_up_proj = nn.Linear(self.d_c, self.n_heads * self.d_h)
        self.vc_up_proj = nn.Linear(self.d_c, self.n_heads * self.d_h)

        # KVキャッシュの最大長
        self.context_size = args.context_size

        # 注意計算モジュール（通常は scaled dot-product）
        self.attention = ScaledDotProductAttention(self.d_model)

        self.output_head = nn.Linear(self.n_heads * self.d_h, self.d_model)

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
            freqs_cis: RoPEで使う複素周波数埋め込み
            mask: Attentionマスク
            train: 訓練中かどうか（Falseの場合のみキャッシュを使う）
        """
        batch_size, seq_len, _ = h.shape

        # バッチサイズの変化があった場合、キャッシュを再作成
        if self.kv_cache is None:# or batch_size != self.registered_batch_size:
            self.kv_cache = KVCache(self.context_size)
            # self.registered_batch_size = batch_size

        # --- Query 処理 ---
        # 入力 -> latent Q
        cq = self.q_down_proj(h)
        cq = self.q_norm(cq)
        # latent Q -> 回転用 Qr
        qr = self.qr_up_proj(cq)
        qr = qr.reshape(batch_size, seq_len, self.n_heads, self.d_hR)
        # RoPEによる回転埋め込みをqrにのみ適用
        qr = apply_rope(qr, freqs_cis)
        # latent Q -> 通常の Qc
        qc = self.qc_up_proj(cq)
        qc = qc.reshape(batch_size, seq_len, self.n_heads, self.d_h)

        # 結合された Q（最終的なAttention用）
        q = torch.cat([qc, qr], dim=-1)

        # --- Key-Value 処理 ---
        # Kr（入力から直接）
        kr = self.kr_proj(h)
        kr = self.kr_norm(kr)
        # kr = kr.view(batch_size, seq_len, 1, self.d_hR)
        kr = kr.reshape(batch_size, seq_len, 1, self.d_hR)

        # RoPEを適用をkrにのみ適用
        kr = apply_rope(kr, freqs_cis)

        # latent key/value
        ckv = self.kv_down_proj(h)
        ckv = self.kv_norm(ckv)

        if not train:
            # 推論時のみキャッシュに保存し、全過去トークンと照合
            # KrのValueは使わないのでダミー
            try:
                self.kv_cache.update(kr, ckv)
                # print('kv_cache.keys', self.kv_cache.keys.shape)
            except:
                print('kv_cache.keys', self.kv_cache.keys.shape)
                print('kr', kr.shape)
                print('kv_cache.values', self.kv_cache.values.shape)
                print('ckv', ckv.shape)
                raise

            kr, ckv = self.kv_cache.get()

        # ckv -> kc, vc（Attention用の形式に再構成）
        kc = self.kc_up_proj(ckv)
        vc = self.vc_up_proj(ckv)
        kc = kc.reshape(batch_size, -1, self.n_heads, self.d_h)
        vc = vc.reshape(batch_size, -1, self.n_heads, self.d_h)

        # kr: 回転的、kc: 内容的 Key
        # ここでkrをブロードキャスト
        kr = kr.expand(-1, -1, kc.size(2), -1)

        k = torch.cat([kr, kc], dim=-1)
        # V は通常通り latent V -> Vc
        v = vc

        # --- Attention ---
        q = q.permute(0, 2, 1, 3)  # (B, H, T, D)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # RoPE回転に用いた系列長
        cache_size = kr.size(1)
        output, attn_weights = self.attention(q, k, v, mask, cache_size)

        # 出力を整形して返す
        # output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.n_heads * self.d_h)
        output = self.output_head(output)
        return output
        # return output, attn_weights