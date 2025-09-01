# @title MLA
class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA)

    複数の射影空間（latent Q/K/V）を用いて効率的に情報を抽出するTransformerのアテンションブロック。
    本クラスでは KV キャッシュを外部クラス (KVCache) に分離して再利用可能にした構造になっている。
    """

    def __init__(self, args):
        super().__init__()

        """ 引数の設定 """
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
        # KVキャッシュの最大長
        self.context_size = args.context_size

        # Q, K, V の射影層（入力 -> latent表現）

        """ Queery """

        # 入力 -> latent Q
        self.q_down_proj = nn.Linear(self.d_model, self.d_cQ)

        # latent Q に対する正規化
        self.q_norm = RMSNorm(self.d_cQ)

        # latent Q -> 再構成された Qc / Qr（2つの役割で別々に使う）
        self.qc_up_proj = nn.Linear(self.d_cQ, self.n_heads * self.d_h)
        self.qr_up_proj = nn.Linear(self.d_cQ, self.n_heads * self.d_hR)

        """ Key for RoPE """

        # 入力から直接得られる Kr（回転対象）
        self.kr_proj = nn.Linear(self.d_model, self.d_hR)
        self.kr_norm = RMSNorm(self.d_hR)

        """ Key / Value  """

        # 入力 -> latent K/V
        self.kv_down_proj = nn.Linear(self.d_model, self.d_c)
        # 正則化
        self.kv_norm = RMSNorm(self.d_c)
        # latent K/V から再構成された Kc, Vc
        self.kc_up_proj = nn.Linear(self.d_c, self.n_heads * self.d_h)
        self.vc_up_proj = nn.Linear(self.d_c, self.n_heads * self.d_h)

        # KVキャッシュ（初期化は forward 時に行う）
        self.kv_cache = None


        """ Attention """

        # 注意計算モジュール（scaled dot-product）
        self.attention = ScaledDotProductAttention(self.d_model)
        # 注意計算の出力用の線形変換
        self.output_head = nn.Linear(self.n_heads * self.d_h, self.d_model)


    def reset_kv_cache(self):
        """
        KVキャッシュを明示的にリセットする関数。
        """
        if self.kv_cache:
            self.kv_cache.reset()

    def forward(self, h, freqs_cis, mask=None, train=False):
        """
        Args:
            h: 入力テンソル (batch_size, seq_len, d_model)
            freqs_cis: RoPEで使う複素周波数埋め込み
            mask: Attentionマスク
            train: 訓練中かどうか（Falseの場合のみキャッシュを使う）
        """
        batch_size, seq_len, _ = h.shape

        # 最初の推論ではキャッシュを初期化
        if self.kv_cache is None:
            self.kv_cache = KVCache(self.context_size)

        # --- Query 処理 ---
        # 入力 -> latent Q
        cQ = self.q_down_proj(h)
        cQ = self.q_norm(cQ)
        # latent Q -> 回転用 Qr
        qR = self.qr_up_proj(cQ)
        qR = qR.reshape(batch_size, seq_len, self.n_heads, self.d_hR)
        # RoPEによる回転埋め込みをqRにのみ適用
        qR = apply_rope(qR, freqs_cis)
        # latent Q -> 通常の Qc
        qC = self.qc_up_proj(cQ)
        qC = qC.reshape(batch_size, seq_len, self.n_heads, self.d_h)

        # 結合された Q（最終的なAttention用）
        q = torch.cat([qC, qR], dim=-1)

        # --- Key-Value 処理 ---
        # Kr（入力から直接）
        kR = self.kr_proj(h)
        kR = self.kr_norm(kR)
        kR = kR.reshape(batch_size, seq_len, 1, self.d_hR) # kRは1ヘッド

        # RoPEを適用をkRにのみ適用
        kR = apply_rope(kR, freqs_cis)

        # latent key/value
        cKV = self.kv_down_proj(h)
        cKV = self.kv_norm(cKV)

        if not train:
            # 推論時のみキャッシュに保存し、全過去トークンと照合
            self.kv_cache.update(kR, cKV)
            kR, cKV = self.kv_cache.get()

        # cKV -> kC, vC（Attention用の形式に再構成）
        kC = self.kc_up_proj(cKV)
        vC = self.vc_up_proj(cKV)
        kC = kC.reshape(batch_size, -1, self.n_heads, self.d_h)
        vC = vC.reshape(batch_size, -1, self.n_heads, self.d_h)

        # kR: 回転的、kC: 内容的 Key
        # ここでkRをブロードキャスト
        kR = kR.expand(-1, -1, kC.size(2), -1)

        k = torch.cat([kR, kC], dim=-1)

        # V は通常通り latent V -> Vc
        v = vC

        # --- Attention ---

        q = q.transpose(1, 2) # (B, T, H, D) → (B, H, T, D)
        # q = q.permute(0, 2, 1, 3)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Sacled Dot-Product Attetnionの計算
        h, attn_weight = self.attention(q, k, v, mask)

        # 出力を整形して返す
        h = h.transpose(1, 2)
        h = h.reshape(batch_size, seq_len, self.n_heads * self.d_h) # H*Dでトークンのベクトルに変換
        h = self.output_head(h)

        output = {}
        output['hidden_state'] = h
        output['attn_weight'] = attn_weight

        return output