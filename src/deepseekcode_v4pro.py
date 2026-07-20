# -*- coding: utf-8 -*-
"""deepseekcode_v4pro.py

DeepSeek-V4 忠実実装 (Pro Edition) — 単一GPU・最小パラメータ構成

ベース: Michi-123/DeepSeekCode (DeepSeek-V3 自作実装)
参考論文: DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence
          (arXiv:2606.19348)

論文のアーキテクチャに可能な限り忠実に従った実装:

  ■ ハイブリッドアテンション（CSA層とHCA層の交互配置）
    - CSA (Compressed Sparse Attention):
        * 2系列のKV潜在 (C^a, C^b) を持ち、mトークン毎に1エントリへ圧縮
          （式11-12）:
            S = Softmax_row([Z^a + B^a ; Z^b + B^b])   ※チャネル毎の重み
            C_i^Comp = Σ_j S_j^a ⊙ C_j^a + Σ_j S_j^b ⊙ C_j^b
        * Lightning Indexer（式14-16）が低ランクでインデクサークエリを生成し
            I_{t,s} = Σ_{h=1}^{n_h^I} w_{t,h}^I · ReLU(q_{t,h}^I · k_s^IComp)
          で Top-k の圧縮エントリを選択（式17）
    - HCA (Heavily Compressed Attention):
        * 単一系列を m'トークン毎 (m' >> m) に圧縮（式22-23）:
            S = Softmax_row(Z + B),  C_i^Comp = Σ_j S_j ⊙ C_j
        * 疎選択なし（圧縮エントリ全体に注意）
    - 共有KV MQA: 圧縮エントリが Key と Value を兼用し全ヘッドで共有
    - グループ化出力射影: n_h 個のヘッド出力を g グループに分割して射影
    - クエリは低ランク圧縮を経て生成（式13, 18）:
        c_t^Q = h_t W^DQ,  q_t = c_t^Q W^UQ
    - RoPE は部分適用（d_rope 次元のみ）

  ■ mHC (Manifold-Constrained Hyper-Connections)
    - 残差ストリームを n 本に拡張
    - 混合行列 B_l を Sinkhorn-Knopp 反復 (t_max=20) で二重確率行列
      （Birkhoff多様体）に射影 → ||B_l||_2 <= 1 で残差変換が非膨張

  ■ DeepSeekMoE (V4)
    - 親和性スコア: Sqrt(Softplus(·))（V3のSigmoidから変更）
    - 補助損失フリー戦略（バイアス項更新）
      + シーケンス方向の軽いバランス損失で補強
    - 先頭数層の密FFNをハッシュルーティングMoEに置き換え
    - ルーティング対象ノード数の制約は廃止

  ■ 推論時KVキャッシュ
    - 圧縮エントリそのものをキャッシュ（V4のKVキャッシュ削減の本体）
    - 混合ストレージ形式: RoPE次元はBF16、それ以外はFP8（利用可能な場合）

  ■ Muon オプティマイザ (Algorithm 1)
    - モメンタム累積 → ハイブリッド Newton-Schulz 反復
      （最初の8ステップ: (3.4445, -4.7750, 2.0315) / 最後の2ステップ: (2, -1.5, 0.5)）

  ■ MTP (Multi-Token Prediction): V3と同一構成（論文に明記）

論文に明記されていない箇所は以下の設計判断で補完している（コメントに明記）:
  - CSA/HCA の層配置は「交互」（論文はハイブリッドとだけ記載、既定はCSA開始）
  - RoPEキーの圧縮は、チャネル毎重み S のチャネル平均をスカラー重みとして使用
  - 現在の未完了ブロック内のトークンには生の潜在（系列a）で注意する
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 基本ユーティリティ
# ============================================================

class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def get_default_args(**overrides):
    """単一GPUで動く最小構成（論文の構造を保ちつつ次元を縮小）"""
    args = Args(
        # --- 全体 ---
        vocab_size=256,
        d_model=256,
        n_layers=6,
        n_heads=8,
        context_size=256,
        max_seq_len=1024,
        norm_eps=1e-6,
        rope_theta=10000.0,

        # --- 潜在空間 ---
        d_cQ=96,               # Query潜在次元 (W^DQ の出力)
        d_c=64,                # KV潜在次元 = 圧縮エントリの次元 c
        d_rope=32,             # RoPE部分適用の次元（論文では64）

        # --- ハイブリッドアテンション ---
        layer_pattern=None,    # None → CSA/HCA交互。例: ['csa','csa','hca',...]
        csa_block=4,           # CSA圧縮ブロックサイズ m
        hca_block=16,          # HCA圧縮ブロックサイズ m' (m' >> m)
        indexer_heads=4,       # Lightning Indexer ヘッド数 n_h^I
        indexer_dim=32,        # Lightning Indexer 次元 d^I
        indexer_topk=16,       # 選択する圧縮エントリ数 k
        n_groups=2,            # グループ化出力射影のグループ数 g

        # --- mHC ---
        n_streams=4,           # 残差ストリーム本数 n
        sinkhorn_iters=20,     # Sinkhorn-Knopp 反復回数 t_max（論文値）

        # --- MoE ---
        n_shared_experts=1,
        n_routed_experts=8,
        n_activated_experts=2,
        moe_inter_dim=256,
        moe_bias_update_speed=0.001,
        seq_balance_alpha=0.0001,   # シーケンス方向バランス損失の係数
        n_hash_layers=1,            # ハッシュルーティングMoEに置き換える先頭層数
        n_hash_experts=4,

        # --- KVキャッシュのストレージ形式 ---
        kv_fp8=False,          # True: 圧縮エントリをFP8、RoPE次元をBF16で保存

        # --- MTP ---
        multi_token_depth=1,
        lambda_mtp=0.3,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def precompute_freqs_cis(args, device='cpu'):
    dim = args.d_rope
    indices = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    freqs = 1.0 / (args.rope_theta ** (indices / dim))
    m = torch.arange(args.max_seq_len, dtype=torch.float32, device=device)
    rotation_angles = torch.outer(m, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), rotation_angles)
    return freqs_cis.detach()


def apply_rope(x, freqs_cis):
    """x: (B, T, H, D) — Dの前半・後半ペアを複素数とみなして回転"""
    B, T, H, D = x.shape
    x_complex = torch.view_as_complex(x.float().view(B, T, H, D // 2, 2))
    freqs_cis = freqs_cis[None, :T, None, :]
    rotated = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    return rotated.type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x_normed * self.weight


# ============================================================
# 圧縮KVキャッシュ（V4のKVキャッシュ削減の本体）
# ============================================================

class CompressedKVCache:
    """
    圧縮エントリ + 現在の未完了ブロックの生潜在 を保持するキャッシュ。

    V3.2 が全トークンの潜在をキャッシュするのに対し、V4 は m (m') トークン毎に
    1エントリへ圧縮したものをキャッシュするため、KVキャッシュが約 1/m になる
    （論文: 1Mトークン設定で V3.2 比 10% のKVキャッシュ）。

    ストレージ形式の混合（論文）:
      - RoPE次元 (comp_kR): BF16
      - それ以外 (comp_c) : FP8 (float8_e4m3fn)
      → キャッシュサイズ約50%削減。kv_fp8=False の場合はそのままのdtypeで保持。
    """

    def __init__(self, block_size, raw_keys, kv_fp8=False):
        self.block_size = block_size
        self.raw_keys = raw_keys      # 生バッファに保持するフィールド名のリスト
        self.kv_fp8 = kv_fp8 and hasattr(torch, 'float8_e4m3fn')

        self.comp_c = None            # (B, nb, d_c)  圧縮エントリ（K/V兼用）
        self.comp_kR = None           # (B, nb, d_rope)
        self.raw = {k: None for k in raw_keys}   # 現在ブロックの生データ

    @property
    def n_blocks(self):
        return 0 if self.comp_c is None else self.comp_c.size(1)

    @property
    def raw_len(self):
        first = self.raw[self.raw_keys[0]]
        return 0 if first is None else first.size(1)

    @property
    def total_len(self):
        return self.n_blocks * self.block_size + self.raw_len

    def reset(self):
        self.comp_c = None
        self.comp_kR = None
        self.raw = {k: None for k in self.raw_keys}

    def _store_comp(self, c_comp, kR_comp):
        """混合精度でエントリを保存"""
        if self.kv_fp8:
            c_comp = c_comp.to(torch.float8_e4m3fn)
            kR_comp = kR_comp.to(torch.bfloat16)
        if self.comp_c is None:
            self.comp_c = c_comp
            self.comp_kR = kR_comp
        else:
            self.comp_c = torch.cat([self.comp_c, c_comp], dim=1)
            self.comp_kR = torch.cat([self.comp_kR, kR_comp], dim=1)

    def get_comp(self, dtype):
        """保存形式から計算用dtypeへ戻して返す"""
        if self.comp_c is None:
            return None, None
        return self.comp_c.to(dtype), self.comp_kR.to(dtype)

    def append_token(self, fields, compress_fn):
        """
        1トークン分の生潜在を追加する。
        生バッファが block_size に達していたら、先に圧縮してからバッファを空にする
        （= 生バッファは常に「現在の未完了ブロック」だけを保持し、
           新トークンは必ず生のまま注意対象に含まれる）。
        fields: {name: (B, 1, d)} 生バッファへ追加するテンソル
        compress_fn: raw辞書 {name: (B, m, d)} → (c_comp, kR_comp)
        """
        if self.raw_len == self.block_size:
            c_comp, kR_comp = compress_fn(self.raw)
            self._store_comp(c_comp.detach(), kR_comp.detach())
            self.raw = {k: None for k in self.raw_keys}

        for k in self.raw_keys:
            t = fields[k].detach()
            self.raw[k] = t if self.raw[k] is None else torch.cat([self.raw[k], t], dim=1)

    def populate(self, comp_c, comp_kR, raw_fields):
        """プリフィル後にまとめてキャッシュを構築する"""
        self.reset()
        if comp_c is not None and comp_c.size(1) > 0:
            self._store_comp(comp_c.detach(), comp_kR.detach())
        for k in self.raw_keys:
            t = raw_fields[k]
            self.raw[k] = None if t is None or t.size(1) == 0 else t.detach()


# ============================================================
# ハイブリッドアテンション: CSA / HCA
# ============================================================

class CompressedAttentionBase(nn.Module):
    """CSA / HCA 共通の骨格（クエリ生成・スコア計算・グループ化出力射影）"""

    def __init__(self, args, block_size):
        super().__init__()
        self.d_model = args.d_model
        self.n_heads = args.n_heads
        self.n_groups = args.n_groups
        self.d_cQ = args.d_cQ
        self.d_c = args.d_c
        self.d_rope = args.d_rope
        self.block_size = block_size
        self.context_size = args.context_size
        self.kv_fp8 = args.kv_fp8

        # --- Query: 低ランク圧縮を経て生成（式13, 18） ---
        #   c_t^Q = h_t W^DQ,  q_t = c_t^Q W^UQ
        self.w_dq = nn.Linear(self.d_model, self.d_cQ)
        self.q_norm = RMSNorm(self.d_cQ)
        # 内容クエリ: 圧縮エントリ空間 (d_c) に直接注意する（共有KV MQA）
        self.w_uq = nn.Linear(self.d_cQ, self.n_heads * self.d_c)
        # RoPEクエリ（部分適用）
        self.w_uq_rope = nn.Linear(self.d_cQ, self.n_heads * self.d_rope)

        # --- RoPEキー（1ヘッド共有・部分適用） ---
        self.w_kr = nn.Linear(self.d_model, self.d_rope)
        self.kr_norm = RMSNorm(self.d_rope)

        # --- グループ化出力射影 ---
        #   n_h 個のヘッド出力を g グループに分割し、グループ内で和を取ってから
        #   まとめて射影する（通常の n_h*d_c → d_model より計算コストが低い）
        assert self.n_heads % self.n_groups == 0
        self.w_o = nn.Linear(self.n_groups * self.d_c, self.d_model)

        self.kv_cache = None

    # --- サブクラスが実装するインタフェース ---
    def token_latents(self, h, freqs_cis):
        """トークン毎の潜在フィールド {name: (B,T,d)} を計算する"""
        raise NotImplementedError

    def compress_raw(self, raw):
        """raw辞書 {name: (B,m,d)} → (c_comp (B,d_c), kR_comp (B,d_rope))"""
        raise NotImplementedError

    def compress_sequence(self, fields, nb):
        """完了ブロックをまとめて圧縮 → (B, nb, d_c), (B, nb, d_rope)"""
        raise NotImplementedError

    def raw_content(self, fields):
        """生トークンの内容キー/バリュー（(B,T,d_c)）を返す"""
        raise NotImplementedError

    def cache_keys(self):
        """キャッシュの生バッファに保持するフィールド名"""
        raise NotImplementedError

    def sparse_select(self, h, cQ, c_comp, comp_invisible):
        """CSAのみ: Top-k選択後の不可視マスクを返す。HCAはそのまま返す"""
        return comp_invisible

    # --- 共通処理 ---
    def reset_kv_cache(self):
        if self.kv_cache is not None:
            self.kv_cache.reset()

    def _queries(self, h, freqs_cis):
        B, T, _ = h.shape
        cQ = self.q_norm(self.w_dq(h))
        qC = self.w_uq(cQ).view(B, T, self.n_heads, self.d_c)
        qR = self.w_uq_rope(cQ).view(B, T, self.n_heads, self.d_rope)
        qR = apply_rope(qR, freqs_cis)
        return cQ, qC, qR

    def _attend(self, qC, qR, k_content, k_rope, v, invisible_mask):
        """
        共有KV MQA のスコア計算と重み付き和。
          qC: (B,H,Tq,d_c), qR: (B,H,Tq,d_rope)
          k_content/v: (B,E,d_c), k_rope: (B,E,d_rope)
          invisible_mask: (B,Tq,E) True=禁止
        """
        scale = 1.0 / math.sqrt(self.d_c + self.d_rope)
        score = (torch.matmul(qC, k_content.unsqueeze(1).transpose(-1, -2))
                 + torch.matmul(qR, k_rope.unsqueeze(1).transpose(-1, -2))) * scale
        score = score.masked_fill(invisible_mask.unsqueeze(1),
                                  torch.finfo(score.dtype).min)
        attn = F.softmax(score, dim=-1)
        out = torch.matmul(attn, v.unsqueeze(1))     # (B, H, Tq, d_c)
        return out, attn

    def _project_out(self, out):
        """グループ化出力射影: (B,H,Tq,d_c) → (B,Tq,d_model)"""
        B, H, Tq, _ = out.shape
        g = self.n_groups
        out = out.permute(0, 2, 1, 3)                       # (B,Tq,H,d_c)
        out = out.view(B, Tq, g, H // g, self.d_c).sum(3)   # グループ内で和
        return self.w_o(out.reshape(B, Tq, g * self.d_c))

    def _visibility(self, q_pos, total_len, nb, device):
        """(True=禁止) 圧縮エントリと生トークンの可視性マスクを作る"""
        m = self.block_size
        tb = q_pos // m
        comp_idx = torch.arange(nb, device=device)
        # 完全に過去のブロックのみ可視
        comp_invisible = comp_idx[None, :] >= tb[:, None]
        # 生トークンは同一ブロック内かつ未来でない場合のみ可視
        s = torch.arange(total_len, device=device)
        same_block = (s[None, :] // m) == tb[:, None]
        not_future = s[None, :] <= q_pos[:, None]
        raw_invisible = ~(same_block & not_future)
        return comp_invisible, raw_invisible

    def forward(self, h, freqs_cis, causal_mask=None, train=False):
        B, Tq, _ = h.shape
        device = h.device

        if self.kv_cache is None:
            self.kv_cache = CompressedKVCache(
                self.block_size, self.cache_keys(), kv_fp8=self.kv_fp8)

        cQ, qC, qR = self._queries(h, freqs_cis)
        fields = self.token_latents(h, freqs_cis)

        if train or (self.kv_cache.total_len == 0 and Tq > 1):
            out, attn = self._forward_full(h, cQ, qC, qR, fields,
                                           populate_cache=not train)
        elif Tq == 1:
            out, attn = self._forward_decode(h, cQ, qC, qR, fields)
        else:
            raise RuntimeError(
                '既存キャッシュへの複数トークン追記は未対応です。'
                'reset_kv_cache() 後にプリフィルしてください。')

        output = {}
        output['hidden_state'] = self._project_out(out)
        output['attention_weight'] = attn
        return output

    def _forward_full(self, h, cQ, qC, qR, fields, populate_cache=False):
        """フルシーケンス処理（学習・プリフィル）"""
        B, Tq, _ = h.shape
        device = h.device
        m = self.block_size
        nb = Tq // m

        c_comp, kR_comp = self.compress_sequence(fields, nb)   # (B,nb,d_c/d_rope)
        c_raw = self.raw_content(fields)                       # (B,Tq,d_c)
        kR_raw = fields['kR']                                  # (B,Tq,d_rope)

        q_pos = torch.arange(Tq, device=device)
        comp_invisible, raw_invisible = self._visibility(q_pos, Tq, nb, device)
        comp_invisible = self.sparse_select(
            h, cQ, c_comp, comp_invisible[None].expand(B, -1, -1))

        k_content = torch.cat([c_comp, c_raw], dim=1)
        k_rope = torch.cat([kR_comp, kR_raw], dim=1)
        invisible = torch.cat(
            [comp_invisible, raw_invisible[None].expand(B, -1, -1)], dim=-1)

        out, attn = self._attend(qC.permute(0, 2, 1, 3), qR.permute(0, 2, 1, 3),
                                 k_content, k_rope, k_content, invisible)

        if populate_cache:
            # 完了ブロックは圧縮エントリとして、残りは生バッファとして保存
            raw_fields = {k: v[:, nb * m:] for k, v in fields.items()}
            self.kv_cache.populate(c_comp, kR_comp, raw_fields)

        return out, attn

    def _forward_decode(self, h, cQ, qC, qR, fields):
        """1トークンずつの逐次デコード（圧縮エントリのキャッシュを利用）"""
        B = h.size(0)
        device = h.device

        # 新トークンをキャッシュへ（ブロック完成時は圧縮が走る）
        self.kv_cache.append_token(fields, self.compress_raw_batchfirst)

        comp_c, comp_kR = self.kv_cache.get_comp(h.dtype)
        raw = self.kv_cache.raw
        c_raw = self.raw_content(raw)                          # (B,r,d_c)
        kR_raw = raw['kR']

        # デコード時: 圧縮エントリは全て過去ブロック、生バッファは全て自ブロック
        # → マスク不要（全エントリ可視）。CSAのみ Top-k 選択を適用する。
        if comp_c is not None:
            nb = comp_c.size(1)
            comp_invisible = torch.zeros(B, 1, nb, dtype=torch.bool, device=device)
            comp_invisible = self.sparse_select(h, cQ, comp_c, comp_invisible)
            k_content = torch.cat([comp_c, c_raw], dim=1)
            k_rope = torch.cat([comp_kR, kR_raw], dim=1)
            invisible = torch.cat(
                [comp_invisible,
                 torch.zeros(B, 1, c_raw.size(1), dtype=torch.bool, device=device)],
                dim=-1)
        else:
            k_content, k_rope = c_raw, kR_raw
            invisible = torch.zeros(B, 1, c_raw.size(1), dtype=torch.bool,
                                    device=device)

        return self._attend(qC.permute(0, 2, 1, 3), qR.permute(0, 2, 1, 3),
                            k_content, k_rope, k_content, invisible)

    def compress_raw_batchfirst(self, raw):
        """append_token 用: raw辞書 → ((B,1,d_c), (B,1,d_rope))"""
        c_comp, kR_comp = self.compress_raw(raw)
        return c_comp.unsqueeze(1), kR_comp.unsqueeze(1)


class CSA(CompressedAttentionBase):
    """
    Compressed Sparse Attention（論文 式11-19）

    2系列のKV潜在 C^a, C^b を持ち、mトークン毎に:
        S = Softmax_row([Z^a + B^a ; Z^b + B^b])     … チャネル毎の圧縮重み
        C_i^Comp = Σ_j S_j^a ⊙ C_j^a + Σ_j S_j^b ⊙ C_j^b
    Lightning Indexer で Top-k の圧縮エントリのみを選択して注意する。
    """

    def __init__(self, args):
        super().__init__(args, block_size=args.csa_block)
        m = self.block_size

        # --- 2系列のKV潜在（式11-12の C^a, C^b） ---
        self.w_ca = nn.Linear(self.d_model, self.d_c)
        self.w_cb = nn.Linear(self.d_model, self.d_c)
        self.ca_norm = RMSNorm(self.d_c)
        self.cb_norm = RMSNorm(self.d_c)

        # --- 圧縮重みスコア Z^a, Z^b ∈ R^{n×c}（チャネル毎） ---
        self.w_za = nn.Linear(self.d_model, self.d_c)
        self.w_zb = nn.Linear(self.d_model, self.d_c)
        # 学習されるブロック内位置バイアス B^a, B^b（m×c）
        self.bias_a = nn.Parameter(torch.zeros(m, self.d_c))
        self.bias_b = nn.Parameter(torch.zeros(m, self.d_c))

        # --- Lightning Indexer（式14-16） ---
        self.indexer_heads = args.indexer_heads
        self.indexer_dim = args.indexer_dim
        self.indexer_topk = args.indexer_topk
        # インデクサークエリは Query潜在から低ランク生成: q_t^I = c_t^Q W^IUQ
        self.w_iuq = nn.Linear(self.d_cQ, self.indexer_heads * self.indexer_dim)
        # ヘッド毎の重み w_t^I = h_t W^w
        self.w_iw = nn.Linear(self.d_model, self.indexer_heads)
        # 圧縮エントリのインデクサーキー k_s^IComp
        self.w_ik = nn.Linear(self.d_c, self.indexer_dim)

    def cache_keys(self):
        return ['ca', 'cb', 'za', 'zb', 'kR']

    def token_latents(self, h, freqs_cis):
        kR = self.kr_norm(self.w_kr(h))
        kR = apply_rope(kR.unsqueeze(2), freqs_cis).squeeze(2)
        return {
            'ca': self.ca_norm(self.w_ca(h)),
            'cb': self.cb_norm(self.w_cb(h)),
            'za': self.w_za(h),
            'zb': self.w_zb(h),
            'kR': kR,
        }

    def raw_content(self, fields):
        # 未完了ブロックの生トークンには系列aの潜在で注意する（設計判断）
        return fields['ca']

    def _compression_weights(self, za_blk, zb_blk):
        """
        za_blk, zb_blk: (..., m, d_c)
        戻り値 S: (..., 2m, d_c)  — [Z^a+B^a ; Z^b+B^b] をトークン軸で
        チャネル毎に Softmax_row したもの（式11）
        """
        La = za_blk + self.bias_a
        Lb = zb_blk + self.bias_b
        L = torch.cat([La, Lb], dim=-2)          # (..., 2m, d_c)
        return F.softmax(L, dim=-2)

    def compress_sequence(self, fields, nb):
        B = fields['ca'].size(0)
        m = self.block_size
        if nb == 0:
            z = fields['ca']
            return (z.new_zeros(B, 0, self.d_c), z.new_zeros(B, 0, self.d_rope))

        def blk(x, d):
            return x[:, :nb * m].view(B, nb, m, d)

        ca, cb = blk(fields['ca'], self.d_c), blk(fields['cb'], self.d_c)
        za, zb = blk(fields['za'], self.d_c), blk(fields['zb'], self.d_c)
        kR = blk(fields['kR'], self.d_rope)

        S = self._compression_weights(za, zb)                   # (B,nb,2m,d_c)
        cab = torch.cat([ca, cb], dim=2)                        # (B,nb,2m,d_c)
        c_comp = (S * cab).sum(dim=2)                           # 式12

        # RoPEキーはチャネル平均のスカラー重みで圧縮（設計判断）
        w_scalar = S.mean(dim=-1)                               # (B,nb,2m)
        kR_dup = torch.cat([kR, kR], dim=2)
        kR_comp = (w_scalar.unsqueeze(-1) * kR_dup).sum(dim=2)
        return c_comp, kR_comp

    def compress_raw(self, raw):
        """1ブロック分 {name: (B,m,d)} → ((B,d_c), (B,d_rope))"""
        S = self._compression_weights(raw['za'], raw['zb'])     # (B,2m,d_c)
        cab = torch.cat([raw['ca'], raw['cb']], dim=1)
        c_comp = (S * cab).sum(dim=1)
        w_scalar = S.mean(dim=-1)
        kR_dup = torch.cat([raw['kR'], raw['kR']], dim=1)
        kR_comp = (w_scalar.unsqueeze(-1) * kR_dup).sum(dim=1)
        return c_comp, kR_comp

    def sparse_select(self, h, cQ, c_comp, comp_invisible):
        """
        Lightning Indexer による Top-k 選択（式14-17）。
          I_{t,s} = Σ_h w_{t,h}^I · ReLU(q_{t,h}^I · k_s^IComp)
        可視 かつ Top-k の圧縮エントリのみ残す。
        """
        B, Tq, nb = comp_invisible.shape
        if nb == 0:
            return comp_invisible

        qI = self.w_iuq(cQ).view(B, Tq, self.indexer_heads, self.indexer_dim)
        wI = self.w_iw(h)                                        # (B,Tq,n_h^I)
        kI = self.w_ik(c_comp)                                   # (B,nb,d^I)

        scores = F.relu(torch.einsum('bthd,bnd->bthn', qI, kI))  # 式16のReLU項
        I = (wI.unsqueeze(-1) * scores).sum(dim=2)               # (B,Tq,nb)

        I = I.masked_fill(comp_invisible, torch.finfo(I.dtype).min)
        k_sel = min(self.indexer_topk, nb)
        topk_indices = I.topk(k_sel, dim=-1).indices             # 式17
        keep = torch.zeros_like(comp_invisible)
        keep.scatter_(-1, topk_indices, True)
        return comp_invisible | (~keep)


class HCA(CompressedAttentionBase):
    """
    Heavily Compressed Attention（論文 式22-25）

    単一系列を m'トークン毎 (m' >> m) に、より大胆に圧縮する:
        S = Softmax_row(Z + B),  C_i^Comp = Σ_j S_j ⊙ C_j
    疎選択（Indexer）は行わず、圧縮エントリ全体に注意する。
    """

    def __init__(self, args):
        super().__init__(args, block_size=args.hca_block)
        m = self.block_size

        self.w_c = nn.Linear(self.d_model, self.d_c)
        self.c_norm = RMSNorm(self.d_c)
        self.w_z = nn.Linear(self.d_model, self.d_c)
        self.bias_z = nn.Parameter(torch.zeros(m, self.d_c))

    def cache_keys(self):
        return ['c', 'z', 'kR']

    def token_latents(self, h, freqs_cis):
        kR = self.kr_norm(self.w_kr(h))
        kR = apply_rope(kR.unsqueeze(2), freqs_cis).squeeze(2)
        return {
            'c': self.c_norm(self.w_c(h)),
            'z': self.w_z(h),
            'kR': kR,
        }

    def raw_content(self, fields):
        return fields['c']

    def compress_sequence(self, fields, nb):
        B = fields['c'].size(0)
        m = self.block_size
        if nb == 0:
            z = fields['c']
            return (z.new_zeros(B, 0, self.d_c), z.new_zeros(B, 0, self.d_rope))

        c = fields['c'][:, :nb * m].view(B, nb, m, self.d_c)
        z = fields['z'][:, :nb * m].view(B, nb, m, self.d_c)
        kR = fields['kR'][:, :nb * m].view(B, nb, m, self.d_rope)

        S = F.softmax(z + self.bias_z, dim=2)                   # 式22
        c_comp = (S * c).sum(dim=2)                             # 式23
        w_scalar = S.mean(dim=-1)
        kR_comp = (w_scalar.unsqueeze(-1) * kR).sum(dim=2)
        return c_comp, kR_comp

    def compress_raw(self, raw):
        S = F.softmax(raw['z'] + self.bias_z, dim=1)            # (B,m,d_c)
        c_comp = (S * raw['c']).sum(dim=1)
        w_scalar = S.mean(dim=-1)
        kR_comp = (w_scalar.unsqueeze(-1) * raw['kR']).sum(dim=1)
        return c_comp, kR_comp


# ============================================================
# mHC: Manifold-Constrained Hyper-Connections
# ============================================================

class HyperConnection(nn.Module):
    """
    残差ストリームを n 本に拡張し、混合行列 B_l を二重確率行列
      M ∈ R^{n×n}:  M・1_n = 1_n,  1_n^T・M = 1_n^T,  M >= 0
    （Birkhoff多様体）に制約する。制約には Sinkhorn-Knopp 反復
    (t_max = 20) を使う。二重確率行列は ||B_l||_2 <= 1 を満たすため、
    残差変換が非膨張になり深いモデルでも信号が爆発しない。
    """

    def __init__(self, n_streams, sinkhorn_iters=20):
        super().__init__()
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters

        self.a_in = nn.Parameter(torch.full((n_streams,), 1.0 / n_streams))
        self.b_out = nn.Parameter(torch.ones(n_streams))
        self.B_raw = nn.Parameter(torch.eye(n_streams) * 4.0)

    def mixing_matrix(self):
        M = F.softplus(self.B_raw)
        for _ in range(self.sinkhorn_iters):        # Sinkhorn-Knopp (t_max=20)
            M = M / (M.sum(dim=1, keepdim=True) + 1e-8)
            M = M / (M.sum(dim=0, keepdim=True) + 1e-8)
        return M

    def aggregate(self, X):
        return torch.einsum('btnd,n->btd', X, self.a_in)

    def update(self, X, y):
        M = self.mixing_matrix()
        X = torch.einsum('mn,btnd->btmd', M, X)
        return X + self.b_out.view(1, 1, -1, 1) * y.unsqueeze(2)


# ============================================================
# DeepSeekMoE (V4)
# ============================================================

class Expert(nn.Module):
    def __init__(self, d_model, moe_inter_dim):
        super().__init__()
        self.w1 = nn.Linear(d_model, moe_inter_dim, bias=False)
        self.w2 = nn.Linear(moe_inter_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, moe_inter_dim, bias=False)
        nn.init.normal_(self.w1.weight, std=0.006)
        nn.init.normal_(self.w2.weight, std=0.006)
        nn.init.normal_(self.w3.weight, std=0.006)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """
    DeepSeek-V4 MoE。

    V3からの変更点（論文）:
      - 親和性スコア: Sigmoid(·) → Sqrt(Softplus(·))
      - ルーティング対象ノード数の制約を廃止（単一GPUでは元々該当なし）
      - 補助損失フリー戦略 + シーケンス方向の軽いバランス損失
        （「個別シーケンス内の極端な不均衡を防ぐ」ため、f_i・P_i を
          シーケンス毎に計算してからバッチ平均する）
    """

    def __init__(self, args):
        super().__init__()
        self.d_model = args.d_model
        self.n_shared_experts = args.n_shared_experts
        self.n_routed_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        self.bias_update_speed = args.moe_bias_update_speed
        self.seq_balance_alpha = args.seq_balance_alpha

        self.shared_experts = nn.ModuleList([
            Expert(self.d_model, args.moe_inter_dim)
            for _ in range(self.n_shared_experts)
        ])
        self.routed_experts = nn.ModuleList([
            Expert(self.d_model, args.moe_inter_dim)
            for _ in range(self.n_routed_experts)
        ])

        self.centroids = nn.Parameter(
            torch.randn(self.n_routed_experts, self.d_model) * 0.1)
        self.register_buffer('expert_bias', torch.zeros(self.n_routed_experts))

        self.expected_load = self.n_activated_experts / self.n_routed_experts
        self.register_buffer('step_expert_counts',
                             torch.zeros(self.n_routed_experts, dtype=torch.long))
        self.register_buffer('step_total_tokens', torch.tensor(0, dtype=torch.long))

    def forward(self, x, train, token_ids=None):
        B, S, d = x.shape
        device = x.device
        u = x.reshape(-1, d)

        shared_output = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_output = shared_output + expert(x)

        # --- V4: Sqrt(Softplus) 親和性スコア ---
        logits = u @ self.centroids.T
        affinity_scores = torch.sqrt(F.softplus(logits) + 1e-9)

        # バイアス項は Top-k 選択にのみ使用（補助損失フリー戦略）
        routing_scores = affinity_scores + self.expert_bias
        _vals, topk_indices = torch.topk(
            routing_scores, k=self.n_activated_experts, dim=1)

        gating_scores = affinity_scores.gather(1, topk_indices)
        gating_weights = gating_scores / (gating_scores.sum(1, keepdim=True) + 1e-8)

        routed_output = torch.zeros_like(u)
        expert_counts = torch.bincount(topk_indices.flatten(),
                                       minlength=self.n_routed_experts)
        counts = expert_counts.tolist()
        for expert_id in range(self.n_routed_experts):
            if counts[expert_id] == 0:
                continue
            t, i = torch.where(topk_indices == expert_id)
            routed_output[t] += (self.routed_experts[expert_id](u[t])
                                 * gating_weights[t, i, None])

        hidden_state = shared_output + routed_output.reshape(B, S, d)

        auxiliary_loss = torch.tensor(0.0, device=device)
        if train:
            auxiliary_loss = self._sequence_balance_loss(
                affinity_scores.view(B, S, -1), topk_indices.view(B, S, -1))
            self.step_expert_counts.copy_(expert_counts.detach())
            self.step_total_tokens.fill_(B * S)

        output = {}
        output['hidden_state'] = hidden_state
        output['auxiliary_loss'] = auxiliary_loss
        output['affinity_scores'] = affinity_scores
        output['gating_weights'] = gating_weights
        return output

    def _sequence_balance_loss(self, affinity, topk_indices):
        """
        シーケンス方向のバランス損失（論文: sequence-wise balance loss）。
        f_i, P_i を各シーケンス内で計算し、バッチ方向に平均する。
          affinity: (B, S, N_r),  topk_indices: (B, S, K)
        """
        B, S, N_r = affinity.shape
        K = topk_indices.size(-1)

        one_hot = F.one_hot(topk_indices, num_classes=N_r).float()  # (B,S,K,Nr)
        f = one_hot.sum(dim=(1, 2)) * N_r / (K * S)                 # (B, Nr)

        norm_scores = affinity / (affinity.sum(-1, keepdim=True) + 1e-8)
        P = norm_scores.mean(dim=1)                                 # (B, Nr)

        return self.seq_balance_alpha * (f * P).sum(-1).mean()

    def update_expert_bias(self):
        """訓練ステップ終了時のバイアス更新（補助損失フリー戦略）"""
        if self.step_total_tokens.item() == 0:
            return
        expert_load = self.step_expert_counts.float() / self.step_total_tokens
        with torch.no_grad():
            self.expert_bias[expert_load > self.expected_load] -= self.bias_update_speed
            self.expert_bias[expert_load < self.expected_load] += self.bias_update_speed


class HashMoE(nn.Module):
    """
    ハッシュルーティングMoE（V4: 先頭数層の密FFNを置き換える）。
    トークンIDの固定ハッシュで専門家を決めるため学習ルーターが不要で、
    負荷が決定的に分散される。
    """

    def __init__(self, args):
        super().__init__()
        self.n_experts = args.n_hash_experts
        self.experts = nn.ModuleList([
            Expert(args.d_model, args.moe_inter_dim)
            for _ in range(self.n_experts)
        ])
        self.shared_expert = Expert(args.d_model, args.moe_inter_dim)

    def forward(self, x, train, token_ids=None):
        assert token_ids is not None, 'HashMoE にはトークンIDが必要です'
        B, T, d = x.shape
        u = x.reshape(-1, d)
        ids = (token_ids.reshape(-1) * 2654435761) % self.n_experts

        routed_output = torch.zeros_like(u)
        for expert_id in range(self.n_experts):
            mask = (ids == expert_id)
            if not mask.any():
                continue
            routed_output[mask] = self.experts[expert_id](u[mask])

        hidden_state = self.shared_expert(x) + routed_output.reshape(B, T, d)

        dummy = torch.tensor(0.0, device=x.device)
        return {'hidden_state': hidden_state, 'auxiliary_loss': dummy,
                'affinity_scores': dummy, 'gating_weights': dummy}


# ============================================================
# Transformer block / Main model / MTP
# ============================================================

def resolve_layer_pattern(args):
    """CSA/HCA の層配置。既定は交互（CSA開始）"""
    if args.layer_pattern is not None:
        assert len(args.layer_pattern) == args.n_layers
        return list(args.layer_pattern)
    return ['csa' if i % 2 == 0 else 'hca' for i in range(args.n_layers)]


class TransformerBlock(nn.Module):
    def __init__(self, args, layer_id=0, attn_type='csa', mtp=False):
        super().__init__()
        self.layer_id = layer_id

        self.attention = CSA(args) if attn_type == 'csa' else HCA(args)

        if mtp or layer_id >= args.n_hash_layers:
            self.feed_forward = MoE(args)
        else:
            self.feed_forward = HashMoE(args)

        self.attn_norm = RMSNorm(args.d_model, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.d_model, eps=args.norm_eps)

        # mHC（アテンション用・FFN用）
        self.hc_attn = HyperConnection(args.n_streams, args.sinkhorn_iters)
        self.hc_ffn = HyperConnection(args.n_streams, args.sinkhorn_iters)

    def forward(self, X, start_pos, freqs_cis=None, causal_mask=None,
                train=False, token_ids=None):
        u = self.hc_attn.aggregate(X)
        attention_output = self.attention(self.attn_norm(u), freqs_cis,
                                          causal_mask, train)
        X = self.hc_attn.update(X, attention_output['hidden_state'])

        u = self.hc_ffn.aggregate(X)
        feed_forward_output = self.feed_forward(self.ffn_norm(u), train,
                                                token_ids=token_ids)
        X = self.hc_ffn.update(X, feed_forward_output['hidden_state'])

        output = {}
        output['hidden_state'] = X
        output['affinity_scores'] = feed_forward_output['affinity_scores']
        output['auxiliary_loss'] = feed_forward_output['auxiliary_loss']
        output['attention_weight'] = attention_output['attention_weight']
        return output


class MainModel(nn.Module):
    def __init__(self, embedding, output_head, args):
        super().__init__()
        self.context_size = args.context_size
        self.n_streams = args.n_streams
        self.embedding = embedding

        pattern = resolve_layer_pattern(args)
        self.layers = nn.ModuleList([
            TransformerBlock(args, layer_id, attn_type=pattern[layer_id])
            for layer_id in range(args.n_layers)
        ])
        self.output_norm = RMSNorm(args.d_model, eps=args.norm_eps)
        self.output_head = output_head

    def reset_kv_cache(self):
        for layer in self.layers:
            layer.attention.reset_kv_cache()

    def forward(self, input_ids, start_pos, freqs_cis, causal_mask=None,
                train=False):
        h = self.embedding(input_ids)
        X = h.unsqueeze(2).repeat(1, 1, self.n_streams, 1)

        for layer in self.layers:
            transformer_block_output = layer(
                X, start_pos, freqs_cis, causal_mask,
                train=train, token_ids=input_ids)
            X = transformer_block_output['hidden_state']

        h = self.output_norm(X.mean(dim=2))
        logits = self.output_head(h)

        main_output = {}
        main_output['logits'] = logits
        main_output['hidden_state'] = h
        main_output['affinity_scores'] = transformer_block_output['affinity_scores']
        main_output['auxiliary_loss'] = transformer_block_output['auxiliary_loss']
        main_output['attention_weight'] = transformer_block_output['attention_weight']
        return main_output


class MTPModule(nn.Module):
    """Multi-Token Prediction（V3と同一構成。MTPブロックはCSAを使用）"""

    def __init__(self, embedding, output_head, args):
        super().__init__()
        self.n_streams = args.n_streams
        self.embedding = embedding
        self.norm_pres = RMSNorm(args.d_model, eps=args.norm_eps)
        self.norm_prev = RMSNorm(args.d_model, eps=args.norm_eps)
        self.projection = nn.Linear(args.d_model * 2, args.d_model)
        self.transformer_block = TransformerBlock(args, attn_type='csa', mtp=True)
        self.output_norm = RMSNorm(args.d_model, eps=args.norm_eps)
        self.output_head = output_head

    def forward(self, input_ids, h_prev, start_pos, freqs_cis, causal_mask=None):
        h_curr = self.norm_pres(self.embedding(input_ids))
        h_prev = self.norm_prev(h_prev)
        h = self.projection(torch.cat([h_curr, h_prev], dim=-1))

        X = h.unsqueeze(2).repeat(1, 1, self.n_streams, 1)
        transformer_block_output = self.transformer_block(
            X, start_pos, freqs_cis, causal_mask, train=True, token_ids=input_ids)
        h = self.output_norm(transformer_block_output['hidden_state'].mean(dim=2))
        logits = self.output_head(h)

        mtp_output = {}
        mtp_output['logits'] = logits
        mtp_output['hidden_state'] = h
        mtp_output['attention_weight'] = transformer_block_output['attention_weight']
        mtp_output['affinity_scores'] = transformer_block_output['affinity_scores']
        mtp_output['auxiliary_loss'] = transformer_block_output['auxiliary_loss']
        return mtp_output


# ============================================================
# DeepSeekCode V4 (pro)
# ============================================================

class DeepSeekCodeV4(nn.Module):
    def __init__(self, args, device='cpu'):
        super().__init__()
        self.args = args
        self.device = device
        self.vocab_size = args.vocab_size
        self.context_size = args.context_size
        self.lambda_mtp = args.lambda_mtp

        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.output_head = nn.Linear(args.d_model, args.vocab_size, bias=True)

        self.main_model = MainModel(self.embedding, self.output_head, args)
        self.mtp_modules = nn.ModuleList([
            MTPModule(self.embedding, self.output_head, args)
            for _ in range(args.multi_token_depth)
        ])

        self.to(device)
        self.freqs_cis = precompute_freqs_cis(args, device)
        self.criterion = nn.CrossEntropyLoss()
        self.reset_kv_cache = self.main_model.reset_kv_cache

    # --- 事前学習 ---
    def pretrain(self, source):
        output = self._calculate_main_loss(source)
        main_loss = output['main_loss']
        main_balance_loss = output['main_balance_loss']
        hidden_state = output['hidden_state']

        output = self._calculate_mtp_loss(source, hidden_state)
        total_loss = (main_loss
                      + self.lambda_mtp * output['mtp_losses']
                      + main_balance_loss
                      + output['mtp_balance_losses'])
        return total_loss

    def _calculate_main_loss(self, source):
        main_input_ids = source[:, :self.context_size]
        main_target_ids = source[:, 1:self.context_size + 1]
        main_freqs_cis = self.freqs_cis[:self.context_size]

        self.main_model.reset_kv_cache()
        main_output = self.main_model(main_input_ids, 0, main_freqs_cis,
                                      None, train=True)

        predicted = main_output['logits'].contiguous().view(-1, self.vocab_size)
        target = main_target_ids.contiguous().view(-1)

        output = {}
        output['main_loss'] = self.criterion(predicted, target)
        output['main_balance_loss'] = main_output['auxiliary_loss']
        output['hidden_state'] = main_output['hidden_state']
        return output

    def _calculate_mtp_loss(self, source, hidden_state):
        mtp_losses = 0
        mtp_balance_losses = 0
        for mtp_offset, mtp_module in enumerate(self.mtp_modules):
            mtp_input_ids = source[:, mtp_offset + 1: self.context_size + mtp_offset + 1]
            mtp_target_ids = source[:, mtp_offset + 2: self.context_size + mtp_offset + 2]
            mtp_freqs_cis = self.freqs_cis[mtp_offset + 1: self.context_size + mtp_offset + 1]

            mtp_output = mtp_module(mtp_input_ids, hidden_state, 0, mtp_freqs_cis)
            hidden_state = mtp_output['hidden_state']

            predicted = mtp_output['logits'].contiguous().view(-1, self.vocab_size)
            target = mtp_target_ids.contiguous().view(-1)
            mtp_losses += self.criterion(predicted, target)
            mtp_balance_losses += mtp_output['auxiliary_loss']

        return {'mtp_losses': mtp_losses, 'mtp_balance_losses': mtp_balance_losses}

    def update_expert_bias(self):
        for module in self.modules():
            if isinstance(module, MoE):
                module.update_expert_bias()

    def compute_log_prob(self, input_ids, causal_mask=None, train=False):
        self.reset_kv_cache()
        main_output = self.main_model(input_ids, 0, self.freqs_cis,
                                      None, train=True)
        return F.log_softmax(main_output['logits'], dim=-1)

    # --- 生成（プリフィル + 圧縮キャッシュによる逐次デコード） ---
    @torch.no_grad()
    def generate(self,
                 input_ids,
                 max_new_tokens=20,
                 temperature=1.0,
                 top_k=1,
                 eos_token_id=None,
                 tokenizer=None,
                 delay=0.0):

        def top_k_sampling(logits, top_k, temperature):
            logits = logits / temperature
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)
            return top_k_indices.gather(dim=-1, index=sampled)

        device = input_ids.device
        self.main_model.eval()
        context_size = self.main_model.context_size

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids[:, :context_size]

        generated_ids = input_ids[:, 0:0]
        self.main_model.reset_kv_cache()
        start_pos = 0

        for _ in range(max_new_tokens):
            seq_len = input_ids.size(1)
            input_freqs_cis = self.freqs_cis[start_pos: start_pos + seq_len].to(device)

            # 1回目: プリフィル（フル計算しつつ圧縮キャッシュを構築）
            # 2回目以降: 圧縮エントリのキャッシュを使った逐次デコード
            output = self.main_model(input_ids, start_pos, input_freqs_cis,
                                     None, train=False)
            last_logits = output['logits'][:, -1, :]

            if top_k == 1:
                index = last_logits.argmax(dim=1)
            else:
                index = top_k_sampling(last_logits, top_k, temperature)

            index = index.view(-1, 1)
            input_ids = index.to(device)

            if tokenizer is not None:
                print(tokenizer.index2word[index[0].item()], end="")
                if tokenizer.eos_token_id == index[0].item():
                    break
            else:
                generated_ids = torch.cat([generated_ids, input_ids], dim=1)
                if eos_token_id is not None and index[0].item() == eos_token_id:
                    break

            time.sleep(delay)

            if start_pos == 0:
                start_pos = seq_len
            else:
                start_pos += 1

            if start_pos >= context_size - 1:
                break

        self.main_model.train()
        if tokenizer is None:
            return generated_ids
        print()
        return None

    def generate_ids(self, input_ids, **kwargs):
        return self.generate(input_ids, tokenizer=None, **kwargs)

    def generate_text(self, input_ids, tokenizer, **kwargs):
        return self.generate(input_ids, tokenizer=tokenizer, **kwargs)


# ============================================================
# Muon オプティマイザ（論文 Algorithm 1）
# ============================================================

# ハイブリッド Newton-Schulz の係数（論文値）
NS_COEFFS = [(3.4445, -4.7750, 2.0315)] * 8 + [(2.0, -1.5, 0.5)] * 2


def newton_schulz_hybrid(G, eps=1e-7):
    """
    ハイブリッド Newton-Schulz 反復:
      M_k = a・M_{k-1} + b・(M M^T)M + c・(M M^T)^2 M
    最初の8ステップは高速収束用の係数、最後の2ステップは
    高精度の仕上げ用係数を使う（論文 Algorithm 1）。
    """
    assert G.dim() == 2
    X = G.float()
    X = X / (X.norm() + eps)

    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True

    for a, b, c in NS_COEFFS:
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X

    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon（論文 Algorithm 1）:
      M_t = μ・M_{t-1} + G_t                  … モメンタム累積
      G'  = G_t + μ・M_t  (Nesterov)
      O_t = NewtonSchulz_hybrid(G')            … 直交化
      W_t = W_{t-1} - η・(O_t・scale + λ・W_{t-1})
    2次元の重み行列のみを対象とする（埋め込み等は AdamW を使う）。
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 weight_decay=0.01):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            mu = group['momentum']
            nesterov = group['nesterov']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']

                buf.mul_(mu).add_(g)                       # M_t = μM + G
                d = g.add(buf, alpha=mu) if nesterov else buf

                O = newton_schulz_hybrid(d.reshape(d.size(0), -1))
                O = O.reshape(p.shape).type_as(p)

                # 更新量のRMSを揃える形状スケール
                scale = 0.2 * math.sqrt(max(p.size(0), p.numel() // p.size(0)))

                p.mul_(1 - lr * wd)
                p.add_(O, alpha=-lr * scale)

        return loss


def create_optimizers(model, muon_lr=0.02, adamw_lr=3e-4):
    """2次元重み行列 → Muon / それ以外（埋め込み・ヘッド・ゲート等）→ AdamW"""
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_special = ('embedding' in name or 'output_head' in name
                      or 'centroids' in name)
        if p.ndim == 2 and not is_special:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    muon = Muon(muon_params, lr=muon_lr)
    adamw = torch.optim.AdamW(adamw_params, lr=adamw_lr, weight_decay=0.01)
    return muon, adamw


# ============================================================
# 動作確認用
# ============================================================

if __name__ == '__main__':
    torch.manual_seed(0)
    args = get_default_args(vocab_size=100, d_model=128, n_layers=4,
                            context_size=64, max_seq_len=128,
                            moe_inter_dim=128, indexer_topk=4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = DeepSeekCodeV4(args, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'パラメータ数: {n_params:,}')

    source = torch.randint(0, args.vocab_size,
                           (2, args.context_size + args.multi_token_depth + 1),
                           device=device)
    muon, adamw = create_optimizers(model)
    loss = model.pretrain(source)
    loss.backward()
    muon.step()
    adamw.step()
    model.update_expert_bias()
    print(f'loss = {loss.item():.4f}')

    prompt = torch.randint(0, args.vocab_size, (1, 10), device=device)
    ids = model.generate_ids(prompt, max_new_tokens=15)
    print('generated:', ids.tolist())
