# -*- coding: utf-8 -*-
"""deepseekcode_v4edu.py

DeepSeek-V4 教育用簡易実装 (Educational Edition)

ベース: Michi-123/DeepSeekCode (DeepSeek-V3 自作実装)
参考論文: DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence
          (arXiv:2606.19348)

■ V3 実装からの主な変更点（論文の骨子を単一GPUで動く最小構成に簡略化）
  1. MLA → ハイブリッドアテンション
       - CSA (Compressed Sparse Attention):
           mトークン毎にKVを1エントリに圧縮し、Lightning Indexer で
           Top-k の圧縮エントリだけを選択して注意計算する。
       - HCA (Heavily Compressed Attention):
           m' (m' >> m) トークン毎に、より大胆に圧縮する。疎選択なし。
       - CSA層とHCA層を交互に配置する。
       - 圧縮エントリは Key と Value を兼用する（共有KVのMQA方式）。
       - 出力射影はヘッドをgグループにまとめて計算量を削減。
  2. 残差接続 → mHC (Manifold-Constrained Hyper-Connections)
       残差ストリームを n 本に拡張し、ストリーム混合行列を
       Sinkhorn-Knopp 反復で二重確率行列（Birkhoff多様体）に制約する。
  3. MoE のゲーティング関数を Sigmoid(・) → Sqrt(Softplus(・)) に変更。
  4. 最初の層の密なFFNをハッシュルーティングMoEに置き換え。
  5. Muon オプティマイザ（ハイブリッド Newton-Schulz 反復）。
  6. MTP (Multi-Token Prediction) は V3 と同一構成のまま。

※ 教育用のため、推論時のKVキャッシュは「トークン毎の潜在ベクトルを保持し、
   毎ステップ圧縮し直す」単純方式にしている（本物のV4は圧縮後のエントリを
   キャッシュすることで KVキャッシュを V3.2 比 約10% に削減している）。
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# @title Args
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# @title get_default_args
def get_default_args(**overrides):
    """単一GPU（またはCPU）で動く最小構成のハイパーパラメータ"""
    args = Args(
        # --- 全体 ---
        vocab_size=256,
        d_model=128,
        n_layers=4,           # CSA / HCA を交互に配置
        n_heads=4,
        context_size=128,
        max_seq_len=512,
        norm_eps=1e-6,
        rope_theta=10000.0,

        # --- 潜在空間（V3のMLAと同様の低ランク圧縮） ---
        d_cQ=48,              # Query潜在次元
        d_c=32,               # KV潜在次元（圧縮エントリの次元でもある）
        d_rope=16,            # RoPEを適用する次元（論文では64次元のみに部分適用）

        # --- V4: ハイブリッドアテンション ---
        csa_block=4,          # CSAの圧縮ブロックサイズ m
        hca_block=16,         # HCAの圧縮ブロックサイズ m' (m' >> m)
        indexer_heads=2,      # Lightning Indexer のヘッド数 n_h^I
        indexer_dim=16,       # Lightning Indexer の次元 d^I
        indexer_topk=8,       # 選択する圧縮エントリ数 k
        n_groups=2,           # グループ化出力射影のグループ数 g

        # --- V4: mHC (Manifold-Constrained Hyper-Connections) ---
        n_streams=2,          # 残差ストリームの本数 n（論文相当は4）
        sinkhorn_iters=20,    # Sinkhorn-Knopp の反復回数 t_max

        # --- MoE ---
        n_shared_experts=1,
        n_routed_experts=8,
        n_activated_experts=2,
        moe_inter_dim=128,
        moe_bias_update_speed=0.001,
        aux_loss_alpha=0.0001,
        n_hash_layers=1,      # V4: ハッシュルーティングMoEに置き換える先頭層の数
        n_hash_experts=4,     # ハッシュルーティングMoEの専門家数

        # --- MTP ---
        multi_token_depth=1,
        lambda_mtp=0.3,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


"""# ヘルパー関数"""

# @title create_causal_mask
def create_causal_mask(seq_len, device='cpu') -> torch.Tensor:
    # 形状: (T, T)。未来をTrueでマスク（上三角の+1オフセット）
    # ※ V4のアテンションはブロック構造から独自にマスクを作るため、
    #   本関数は互換性維持のために残している。
    ones = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    return torch.triu(ones, 1)


# @title KVCache
# 汎用的なキー・バリューキャッシュ管理クラス（V3と同じ）
# V4教育版では keys=RoPE適用済みキー(kR), values=KV潜在ベクトル(c) を保持する
class KVCache(nn.Module):
    def __init__(self, context_size):
        super().__init__()
        self.context_size = context_size
        self.keys = None
        self.values = None

    def get(self):
        return self.keys, self.values

    def update(self, k, v):
        if self.keys is None:
            self.keys = k.detach()
            self.values = v.detach()
        else:
            self.keys = torch.cat([self.keys, k.detach()], dim=1)
            self.values = torch.cat([self.values, v.detach()], dim=1)

        if self.keys.size(1) > self.context_size:
            self.keys = self.keys[:, -self.context_size:, ...]
            self.values = self.values[:, -self.context_size:, ...]

    def reset(self):
        self.keys = None
        self.values = None


# @title precompute_freqs_cis
def precompute_freqs_cis(args, device='cpu'):
    dim = args.d_rope
    indices = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    scaled_index = indices / dim
    freqs = 1.0 / (args.rope_theta ** scaled_index)
    m = torch.arange(args.max_seq_len, dtype=torch.float32, device=device)
    rotation_angles = torch.outer(m, freqs)
    abs = torch.ones_like(freqs)
    freqs_cis = torch.polar(abs, rotation_angles)
    return freqs_cis.detach()


# @title apply_rope
def apply_rope(x, freqs_cis):
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    n_heads = x.shape[2]
    d_head = x.shape[3]

    x_reshaped = x.view(batch_size, seq_len, n_heads, d_head // 2, 2)
    x_complex = torch.view_as_complex(x_reshaped.float())
    freqs_cis = freqs_cis[None, :seq_len, None, :]
    rotated_complex = x_complex * freqs_cis
    rotated_embeds = torch.view_as_real(rotated_complex).flatten(3)
    return rotated_embeds.type_as(x)


# @title RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x_normed * self.weight


"""# V4 Attention: CSA / HCA"""

# @title CompressedAttention (CSA / HCA 共通ベース)
class CompressedAttention(nn.Module):
    """
    DeepSeek-V4 の圧縮アテンション（教育用の共通実装）。

    ・block_size(m) トークン毎に KV潜在ベクトル c を1エントリに圧縮する。
        圧縮重み: ブロック内のスコア z をソフトマックスした S
        圧縮エントリ: C_i^Comp = Σ_j S_j * c_j   （論文 式11-12 の簡略版）
    ・各クエリは「過去の完了ブロックの圧縮エントリ」と
      「自分と同じ（未完了）ブロック内の生トークン」に注意を払う。
    ・圧縮エントリは Key と Value を兼用（共有KVのMQA）。
      全ヘッドのクエリが同一のKVエントリ集合を共有する。
    ・use_indexer=True (CSA) の場合、Lightning Indexer で圧縮エントリの
      Top-k だけを残す:
        I_{t,s} = Σ_h w_{t,h}^I ・ ReLU(q_{t,h}^I ・ k_s^IComp)  （論文 式15-16）
    ・出力射影はヘッドを g グループに分割し、グループ内で和をとってから
      射影する（グループ化出力射影）。
    """

    def __init__(self, args, block_size, use_indexer):
        super().__init__()
        self.d_model = args.d_model
        self.n_heads = args.n_heads
        self.n_groups = args.n_groups
        self.d_cQ = args.d_cQ
        self.d_c = args.d_c
        self.d_rope = args.d_rope
        self.block_size = block_size          # 圧縮ブロックサイズ m (または m')
        self.use_indexer = use_indexer        # True: CSA / False: HCA
        self.context_size = args.context_size

        """ Query（低ランク圧縮を経て生成: c_t^Q = h_t W^DQ, q_t = c_t^Q W^UQ） """
        self.q_down_proj = nn.Linear(self.d_model, self.d_cQ)
        self.q_norm = RMSNorm(self.d_cQ)
        # 内容クエリ: 圧縮エントリ空間 (d_c) に直接注意するため d_c 次元/ヘッド
        self.qc_up_proj = nn.Linear(self.d_cQ, self.n_heads * self.d_c)
        # RoPE用クエリ
        self.qr_up_proj = nn.Linear(self.d_cQ, self.n_heads * self.d_rope)

        """ Key/Value 潜在（1系列。圧縮エントリが K と V を兼用する） """
        self.kv_down_proj = nn.Linear(self.d_model, self.d_c)
        self.kv_norm = RMSNorm(self.d_c)

        """ RoPE用キー（1ヘッド共有） """
        self.kr_proj = nn.Linear(self.d_model, self.d_rope)
        self.kr_norm = RMSNorm(self.d_rope)

        """ 圧縮重みスコア（潜在ベクトルからスカラーのスコアを出す） """
        self.comp_score = nn.Linear(self.d_c, 1)

        """ Lightning Indexer (CSAのみ) """
        if use_indexer:
            self.indexer_heads = args.indexer_heads
            self.indexer_dim = args.indexer_dim
            self.indexer_topk = args.indexer_topk
            # インデクサークエリは Query潜在 c^Q から低ランク生成（論文 式14）
            self.indexer_q_proj = nn.Linear(self.d_cQ, self.indexer_heads * self.indexer_dim)
            # ヘッド毎の重み w_t^I は隠れ状態から生成（論文 式15）
            self.indexer_w_proj = nn.Linear(self.d_model, self.indexer_heads)
            # 圧縮エントリからインデクサーキーを生成
            self.indexer_k_proj = nn.Linear(self.d_c, self.indexer_dim)

        """ グループ化出力射影 """
        assert self.n_heads % self.n_groups == 0
        # g グループそれぞれの出力（グループ内で和をとった d_c ベクトル）を
        # まとめて射影する。通常の n_h*d_c → d_model より低コスト。
        self.output_head = nn.Linear(self.n_groups * self.d_c, self.d_model)

        self.dropout = nn.Dropout(0.1)

        # KVキャッシュ（推論時のみ使用。トークン毎の kR と c を保持）
        self.kv_cache = None

    def reset_kv_cache(self):
        if self.kv_cache:
            self.kv_cache.reset()

    # --- 圧縮処理 ---
    def compress(self, c, kR):
        """
        完了ブロックの圧縮エントリを作る。
          c : (B, T_total, d_c)     KV潜在ベクトル
          kR: (B, T_total, d_rope)  RoPE適用済みキー
        戻り値:
          c_comp : (B, nb, d_c)     圧縮エントリ（K/V兼用）
          kR_comp: (B, nb, d_rope)  圧縮エントリのRoPEキー
        """
        B, T, _ = c.shape
        m = self.block_size
        nb = T // m  # 完了ブロック数

        if nb == 0:
            c_comp = c.new_zeros(B, 0, self.d_c)
            kR_comp = kR.new_zeros(B, 0, self.d_rope)
            return c_comp, kR_comp

        # ブロック単位に整形
        c_blocks = c[:, :nb * m].view(B, nb, m, self.d_c)
        kR_blocks = kR[:, :nb * m].view(B, nb, m, self.d_rope)

        # 圧縮重み: ブロック内ソフトマックス（論文の Softmax_row の簡略版）
        z = self.comp_score(c[:, :nb * m]).view(B, nb, m, 1)
        S = F.softmax(z, dim=2)

        # 重み付き和で 1 エントリに圧縮（mトークン → 1エントリ）
        c_comp = (S * c_blocks).sum(dim=2)
        kR_comp = (S * kR_blocks).sum(dim=2)
        return c_comp, kR_comp

    # --- 可視性マスク ---
    def build_masks(self, q_pos, total_len, nb, device):
        """
        q_pos    : (Tq,) 各クエリの絶対位置
        total_len: 系列全体の長さ（キャッシュ含む）
        nb       : 圧縮エントリ数
        戻り値（True = 注意禁止）:
          comp_invisible: (Tq, nb)        圧縮エントリへの可視性
          raw_invisible : (Tq, total_len) 生トークンへの可視性
        """
        m = self.block_size
        tb = q_pos // m  # クエリの所属ブロック番号

        # 圧縮エントリ i はブロックが「完全に過去」の場合のみ可視 (i < tb)
        comp_idx = torch.arange(nb, device=device)
        comp_invisible = comp_idx[None, :] >= tb[:, None]

        # 生トークン s は「同じブロック内」かつ「未来でない」場合のみ可視
        s = torch.arange(total_len, device=device)
        same_block = (s[None, :] // m) == tb[:, None]
        not_future = s[None, :] <= q_pos[:, None]
        raw_invisible = ~(same_block & not_future)

        return comp_invisible, raw_invisible

    def forward(self, h, freqs_cis, causal_mask=None, train=False):
        """
        h        : (B, Tq, d_model) 入力（このステップで処理するトークン）
        freqs_cis: クエリ位置に対応するRoPE回転係数
        causal_mask: 未使用（ブロック構造から内部でマスクを作るため）
        train    : 学習時 True（キャッシュ不使用）
        """
        B, Tq, _ = h.shape
        device = h.device

        if self.kv_cache is None:
            self.kv_cache = KVCache(self.context_size)

        # --- Query 処理（低ランク: h → c^Q → q） ---
        cQ = self.q_norm(self.q_down_proj(h))
        qC = self.qc_up_proj(cQ).view(B, Tq, self.n_heads, self.d_c)
        qR = self.qr_up_proj(cQ).view(B, Tq, self.n_heads, self.d_rope)
        qR = apply_rope(qR, freqs_cis)

        # --- KV潜在 と RoPEキー ---
        c_new = self.kv_norm(self.kv_down_proj(h))          # (B, Tq, d_c)
        kR_new = self.kr_norm(self.kr_proj(h))              # (B, Tq, d_rope)
        kR_new = apply_rope(kR_new.unsqueeze(2), freqs_cis).squeeze(2)

        if train:
            c_all, kR_all = c_new, kR_new
            past_len = 0
        else:
            # 推論時はトークン毎の潜在をキャッシュし、全体を毎回圧縮し直す
            # （教育用の簡略化。実物は圧縮エントリ自体をキャッシュする）
            self.kv_cache.update(kR_new.unsqueeze(2), c_new)
            kR_all, c_all = self.kv_cache.get()
            kR_all = kR_all.squeeze(2)
            past_len = c_all.size(1) - Tq

        total_len = c_all.size(1)
        q_pos = past_len + torch.arange(Tq, device=device)

        # --- 圧縮（mトークン → 1エントリ） ---
        c_comp, kR_comp = self.compress(c_all, kR_all)
        nb = c_comp.size(1)

        # --- 可視性マスク ---
        comp_invisible, raw_invisible = self.build_masks(q_pos, total_len, nb, device)

        # --- Lightning Indexer による Top-k 選択 (CSA のみ) ---
        if self.use_indexer and nb > 0:
            # インデクサースコア: I_{t,s} = Σ_h w_{t,h}^I・ReLU(q_{t,h}^I・k_s^IComp)
            qI = self.indexer_q_proj(cQ).view(B, Tq, self.indexer_heads, self.indexer_dim)
            wI = self.indexer_w_proj(h)                                  # (B, Tq, n_h^I)
            kI = self.indexer_k_proj(c_comp)                             # (B, nb, d^I)
            scores_I = F.relu(torch.einsum('bthd,bnd->bthn', qI, kI))    # (B,Tq,n_h^I,nb)
            I = (wI.unsqueeze(-1) * scores_I).sum(dim=2)                 # (B, Tq, nb)

            # 不可視エントリを除外して Top-k を選ぶ
            I = I.masked_fill(comp_invisible[None, :, :], torch.finfo(I.dtype).min)
            k_sel = min(self.indexer_topk, nb)
            topk_indices = I.topk(k_sel, dim=-1).indices                 # (B, Tq, k)
            keep = torch.zeros(B, Tq, nb, dtype=torch.bool, device=device)
            keep.scatter_(-1, topk_indices, True)

            # 「可視 かつ Top-k」のみ残す
            comp_invisible = comp_invisible[None, :, :] | (~keep)        # (B, Tq, nb)
        else:
            comp_invisible = comp_invisible[None, :, :].expand(B, Tq, nb)

        # --- エントリ列 = [圧縮エントリ; 現在ブロックの生トークン] ---
        k_content = torch.cat([c_comp, c_all], dim=1)     # (B, nb+T, d_c)
        k_rope = torch.cat([kR_comp, kR_all], dim=1)      # (B, nb+T, d_rope)
        v = k_content                                     # 共有KV: KがVを兼用

        # --- スコア計算（MQA: 全ヘッドが同一エントリ集合を共有） ---
        qC_ = qC.permute(0, 2, 1, 3)                      # (B, H, Tq, d_c)
        qR_ = qR.permute(0, 2, 1, 3)                      # (B, H, Tq, d_rope)
        scale = 1.0 / math.sqrt(self.d_c + self.d_rope)
        score = (torch.matmul(qC_, k_content.unsqueeze(1).transpose(-1, -2))
                 + torch.matmul(qR_, k_rope.unsqueeze(1).transpose(-1, -2))) * scale
        # score: (B, H, Tq, nb+T)

        # --- マスク適用 ---
        mask = torch.cat([comp_invisible,
                          raw_invisible[None, :, :].expand(B, Tq, total_len)], dim=-1)
        score = score.masked_fill(mask.unsqueeze(1), torch.finfo(score.dtype).min)

        attention_weight = F.softmax(score, dim=-1)
        attention_weight = self.dropout(attention_weight)

        # --- 重み付き和（Vは圧縮エントリそのもの） ---
        out = torch.matmul(attention_weight, v.unsqueeze(1))   # (B, H, Tq, d_c)

        # --- グループ化出力射影 ---
        out = out.permute(0, 2, 1, 3)                          # (B, Tq, H, d_c)
        g = self.n_groups
        out = out.view(B, Tq, g, self.n_heads // g, self.d_c).sum(dim=3)  # (B,Tq,g,d_c)
        out = out.reshape(B, Tq, g * self.d_c)
        out = self.output_head(out)                            # (B, Tq, d_model)

        output = {}
        output['hidden_state'] = out
        output['attention_weight'] = attention_weight
        return output


# @title CSA
class CSA(CompressedAttention):
    """Compressed Sparse Attention: 圧縮 + Lightning Indexer による Top-k 疎選択"""
    def __init__(self, args):
        super().__init__(args, block_size=args.csa_block, use_indexer=True)


# @title HCA
class HCA(CompressedAttention):
    """Heavily Compressed Attention: より大胆な圧縮 (m' >> m)。疎選択なし"""
    def __init__(self, args):
        super().__init__(args, block_size=args.hca_block, use_indexer=False)


"""# V4: mHC (Manifold-Constrained Hyper-Connections)"""

# @title HyperConnection
class HyperConnection(nn.Module):
    """
    mHC の教育用実装。

    通常の残差結合 x + f(x) を n 本の「残差ストリーム」に拡張し、
      1) 集約: u = Σ_i a_i・X_i          （ストリームからサブレイヤ入力を作る）
      2) 混合: X ← B・X                  （ストリーム同士を混ぜる）
      3) 分配: X ← X + b_i・y            （サブレイヤ出力を各ストリームへ戻す）
    とする。混合行列 B は
      Softplus で非負化 → Sinkhorn-Knopp 反復（t_max=20）
    により二重確率行列（各行・各列の和が1、Birkhoff多様体上の点）に制約する。
    これによりスペクトルノルム ||B||_2 <= 1 が保証され、残差変換が
    非膨張（信号が層を経るごとに爆発しない）になる。
    """

    def __init__(self, n_streams, sinkhorn_iters=20):
        super().__init__()
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters

        # 集約重み a（初期値: 全ストリームの平均 → 通常の残差結合と等価に開始）
        self.a_in = nn.Parameter(torch.full((n_streams,), 1.0 / n_streams))
        # 分配重み b（初期値: 全ストリームに等しく加算）
        self.b_out = nn.Parameter(torch.ones(n_streams))
        # 混合行列の生パラメータ（単位行列寄りに初期化）
        self.B_raw = nn.Parameter(torch.eye(n_streams) * 4.0)

    def mixing_matrix(self):
        """Softplus + Sinkhorn-Knopp で二重確率行列を作る"""
        M = F.softplus(self.B_raw)
        for _ in range(self.sinkhorn_iters):
            M = M / (M.sum(dim=1, keepdim=True) + 1e-8)  # 行正規化
            M = M / (M.sum(dim=0, keepdim=True) + 1e-8)  # 列正規化
        return M

    def aggregate(self, X):
        """X: (B, T, n, d) → サブレイヤ入力 u: (B, T, d)"""
        return torch.einsum('btnd,n->btd', X, self.a_in)

    def update(self, X, y):
        """ストリーム混合 + サブレイヤ出力 y の分配"""
        M = self.mixing_matrix()
        X = torch.einsum('mn,btnd->btmd', M, X)
        X = X + self.b_out.view(1, 1, -1, 1) * y.unsqueeze(2)
        return X


"""# DeepSeekMoE (V4)"""

# @title Expert
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
        gate = F.silu(self.w1(x))
        value = self.w3(x)
        return self.w2(gate * value)


# @title MoE (V4: Sqrt(Softplus) ゲーティング)
class MoE(nn.Module):
    """
    DeepSeek-V4 MoE の教育用実装。

    V3からの変更点:
      - 親和性スコア: Sigmoid(u・e_i) → Sqrt(Softplus(u・e_i))
        （論文: "Sigmoid(·) から Sqrt(Softplus(·)) へ変更"）
      - ルーティング対象ノード数の制約は廃止（本実装は元々未実装なので同じ）
    V3から継続:
      - 補助損失フリーのバイアス項によるロードバランス (update_expert_bias)
      - 軽いバランス損失で補強（論文の「シーケンス方向の均衡損失」に相当）
    """

    def __init__(self, args):
        super().__init__()
        self.d_model = args.d_model
        self.n_shared_experts = args.n_shared_experts
        self.n_routed_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        self.moe_inter_dim = args.moe_inter_dim
        self.bias_update_speed = args.moe_bias_update_speed
        self.aux_loss_alpha = args.aux_loss_alpha

        self.shared_experts = nn.ModuleList([
            Expert(self.d_model, self.moe_inter_dim)
            for _ in range(self.n_shared_experts)
        ])
        self.routed_experts = nn.ModuleList([
            Expert(self.d_model, self.moe_inter_dim)
            for _ in range(self.n_routed_experts)
        ])

        self.centroids = nn.Parameter(
            torch.randn(self.n_routed_experts, self.d_model) * 0.1
        )
        self.register_buffer('expert_bias', torch.zeros(self.n_routed_experts))

        self.expected_load = self.n_activated_experts / self.n_routed_experts
        self.register_buffer('step_expert_counts',
                             torch.zeros(self.n_routed_experts, dtype=torch.long))
        self.register_buffer('step_total_tokens', torch.tensor(0, dtype=torch.long))

    def forward(self, x, train, token_ids=None):
        batch_size, seq_len, d_model = x.shape
        device = x.device
        u = x.reshape(-1, d_model)

        # 共通専門家
        shared_output = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_output = shared_output + expert(x)

        # --- V4: 親和性スコア = Sqrt(Softplus(u・centroids)) ---
        logits = u @ self.centroids.T
        affinity_scores = torch.sqrt(F.softplus(logits) + 1e-9)

        # ルーティング用のスコア（バイアス項はTop-k選択にのみ使う）
        routing_scores = affinity_scores + self.expert_bias

        _topk_values, topk_indices = torch.topk(
            routing_scores, k=self.n_activated_experts, dim=1
        )

        # Top-Kの親和性スコアをゲートスコアとして正規化
        gating_scores = affinity_scores.gather(1, topk_indices)
        gating_sum = gating_scores.sum(dim=1, keepdim=True)
        gating_weights = gating_scores / (gating_sum + 1e-8)

        routed_output = torch.zeros_like(u)
        expert_counts = torch.bincount(
            topk_indices.flatten(), minlength=self.n_routed_experts
        )
        counts = expert_counts.tolist()

        for expert_id in range(self.n_routed_experts):
            if counts[expert_id] == 0:
                continue
            mask = (topk_indices == expert_id)
            t, i = torch.where(mask)
            expert_output = self.routed_experts[expert_id](u[t])
            routed_output[t] += expert_output * gating_weights[t, i, None]

        routed_output = routed_output.reshape(batch_size, seq_len, d_model)
        hidden_state = shared_output + routed_output

        auxiliary_loss = torch.tensor(0.0, device=device)
        if train:
            auxiliary_loss = self._compute_auxiliary_loss(affinity_scores, topk_indices)
            self.step_expert_counts.copy_(expert_counts.detach())
            self.step_total_tokens.fill_(batch_size * seq_len)

        output = {}
        output['hidden_state'] = hidden_state
        output['auxiliary_loss'] = auxiliary_loss
        output['affinity_scores'] = affinity_scores
        output['gating_weights'] = gating_weights
        return output

    def _compute_auxiliary_loss(self, affinity_scores, topk_indices):
        """負荷分散のための軽い補助損失（V3と同形式）"""
        device = affinity_scores.device
        T, _K = topk_indices.shape
        N_r = self.n_routed_experts
        K_r = self.n_activated_experts

        # f_i: 専門家の利用頻度
        one_hot = F.one_hot(topk_indices, num_classes=N_r).float()
        expert_frequency = one_hot.sum(dim=(0, 1)) * N_r / (K_r * T)

        # P_i: 正規化スコアの平均
        affinity_sum = affinity_scores.sum(dim=1, keepdim=True)
        normalized_scores = affinity_scores / (affinity_sum + 1e-8)
        expert_affinity = normalized_scores.mean(dim=0)

        return self.aux_loss_alpha * (expert_frequency * expert_affinity).sum()

    def update_expert_bias(self):
        """訓練ステップ終了時にエキスパートバイアスを更新（補助損失フリー戦略）"""
        if self.step_total_tokens.item() == 0:
            return
        expert_load = self.step_expert_counts.float() / self.step_total_tokens
        overloaded = expert_load > self.expected_load
        underloaded = expert_load < self.expected_load
        with torch.no_grad():
            self.expert_bias[overloaded] -= self.bias_update_speed
            self.expert_bias[underloaded] += self.bias_update_speed


# @title HashMoE (V4: ハッシュルーティングMoE)
class HashMoE(nn.Module):
    """
    V4: 「最初の数層の密FFNをハッシュルーティングを使うMoEに置き換え」の実装。

    トークンIDの固定ハッシュで専門家を決める（学習されるルーターが無い）ため、
    負荷が最初から決定的に分散され、学習初期のルーティング崩壊が起きない。
    """

    def __init__(self, args):
        super().__init__()
        self.n_experts = args.n_hash_experts
        self.experts = nn.ModuleList([
            Expert(args.d_model, args.moe_inter_dim)
            for _ in range(self.n_experts)
        ])
        # 共通専門家も1つ置く（トークン全体で共有される知識のため）
        self.shared_expert = Expert(args.d_model, args.moe_inter_dim)

    def hash_route(self, token_ids):
        # Knuth の乗法ハッシュでトークンIDを専門家IDへ写像
        return (token_ids * 2654435761) % self.n_experts

    def forward(self, x, train, token_ids=None):
        assert token_ids is not None, "HashMoE にはトークンIDが必要です"
        B, T, d = x.shape
        u = x.reshape(-1, d)
        ids = self.hash_route(token_ids.reshape(-1))

        routed_output = torch.zeros_like(u)
        for expert_id in range(self.n_experts):
            mask = (ids == expert_id)
            if not mask.any():
                continue
            routed_output[mask] = self.experts[expert_id](u[mask])

        hidden_state = self.shared_expert(x) + routed_output.reshape(B, T, d)

        dummy = torch.tensor(0.0, device=x.device)
        output = {}
        output['hidden_state'] = hidden_state
        output['auxiliary_loss'] = dummy
        output['affinity_scores'] = dummy
        output['gating_weights'] = dummy
        return output


"""# Transformer block (V4)"""

# @title TransformerBlock
class TransformerBlock(nn.Module):
    """
    V4 Transformerブロック。

    V3からの変更点:
      - アテンション: layer_id が偶数 → CSA / 奇数 → HCA（交互配置）
      - 残差結合: mHC (HyperConnection) に置き換え
        入力/出力は n 本の残差ストリーム (B, T, n, d) で受け渡す
      - FFN: 先頭 n_hash_layers 層は HashMoE、それ以外は MoE
    """

    def __init__(self, args, layer_id=0, mtp=False):
        super().__init__()
        self.layer_id = layer_id

        # --- V4: CSA / HCA の交互配置 ---
        if layer_id % 2 == 0:
            self.attention = CSA(args)
        else:
            self.attention = HCA(args)

        # --- V4: 先頭層は HashMoE、それ以外（とMTP）は MoE ---
        if mtp or layer_id >= args.n_hash_layers:
            self.feed_forward = MoE(args)
        else:
            self.feed_forward = HashMoE(args)

        self.attn_norm = RMSNorm(args.d_model, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.d_model, eps=args.norm_eps)

        # --- V4: mHC（アテンション用とFFN用で別々に持つ） ---
        self.hc_attn = HyperConnection(args.n_streams, args.sinkhorn_iters)
        self.hc_ffn = HyperConnection(args.n_streams, args.sinkhorn_iters)

    def forward(self, X, start_pos, freqs_cis=None, causal_mask=None,
                train=False, token_ids=None):
        """X: 残差ストリーム (B, T, n_streams, d_model)"""

        # --- アテンション・サブレイヤ ---
        u = self.hc_attn.aggregate(X)                 # ストリーム → 1本に集約
        h1 = self.attn_norm(u)
        attention_output = self.attention(h1, freqs_cis, causal_mask, train)
        y = attention_output['hidden_state']
        w = attention_output['attention_weight']
        X = self.hc_attn.update(X, y)                 # 混合 + 分配（mHC残差）

        # --- FFN・サブレイヤ ---
        u = self.hc_ffn.aggregate(X)
        h2 = self.ffn_norm(u)
        feed_forward_output = self.feed_forward(h2, train, token_ids=token_ids)
        y = feed_forward_output['hidden_state']
        X = self.hc_ffn.update(X, y)

        output = {}
        output['hidden_state'] = X
        output['affinity_scores'] = feed_forward_output['affinity_scores']
        output['auxiliary_loss'] = feed_forward_output['auxiliary_loss']
        output['attention_weight'] = w
        return output


"""# DeepSeek Main Modules"""

# @title Main Model
class MainModel(nn.Module):
    def __init__(self, embedding, output_head, args):
        super().__init__()
        self.context_size = args.context_size
        self.n_streams = args.n_streams
        self.embedding = embedding  # 共通重み

        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(args, layer_id))

        self.output_norm = RMSNorm(args.d_model, eps=args.norm_eps)
        self.output_head = output_head  # 共通重み

    def reset_kv_cache(self):
        for layer in self.layers:
            layer.attention.reset_kv_cache()

    def forward(self, input_ids, start_pos, freqs_cis, causal_mask=None, train=False):
        h = self.embedding(input_ids)

        # --- V4: 残差ストリームへ拡張（n本の複製から開始） ---
        X = h.unsqueeze(2).repeat(1, 1, self.n_streams, 1)  # (B, T, n, d)

        for layer in self.layers:
            transformer_block_output = layer(
                X, start_pos, freqs_cis, causal_mask,
                train=train, token_ids=input_ids
            )
            X = transformer_block_output['hidden_state']

        # --- ストリームを平均して1本へ戻す ---
        h = X.mean(dim=2)
        h = self.output_norm(h)
        logits = self.output_head(h)

        main_output = {}
        main_output['logits'] = logits
        main_output['hidden_state'] = h
        main_output['affinity_scores'] = transformer_block_output['affinity_scores']
        main_output['auxiliary_loss'] = transformer_block_output['auxiliary_loss']
        main_output['attention_weight'] = transformer_block_output['attention_weight']
        return main_output


# @title MTP Module
class MTPModule(nn.Module):
    """Multi-Token Prediction（V4でもV3と同一構成: 論文に明記）"""

    def __init__(self, embedding, output_head, args):
        super().__init__()
        self.n_streams = args.n_streams
        self.embedding = embedding
        self.norm_pres = RMSNorm(args.d_model, eps=args.norm_eps)
        self.norm_prev = RMSNorm(args.d_model, eps=args.norm_eps)
        self.projection = nn.Linear(args.d_model * 2, args.d_model)
        self.transformer_block = TransformerBlock(args, mtp=True)
        self.output_norm = RMSNorm(args.d_model, eps=args.norm_eps)
        self.output_head = output_head

    def forward(self, input_ids, h_prev, start_pos, freqs_cis, causal_mask=None):
        h_curr = self.embedding(input_ids)
        h_curr = self.norm_pres(h_curr)
        h_prev = self.norm_prev(h_prev)

        concatenation = torch.cat([h_curr, h_prev], dim=-1)
        h = self.projection(concatenation)

        # 残差ストリームへ拡張してブロックを通す
        X = h.unsqueeze(2).repeat(1, 1, self.n_streams, 1)
        transformer_block_output = self.transformer_block(
            X, start_pos, freqs_cis, causal_mask, train=True, token_ids=input_ids
        )
        h = transformer_block_output['hidden_state'].mean(dim=2)

        h = self.output_norm(h)
        logits = self.output_head(h)

        mtp_output = {}
        mtp_output['logits'] = logits
        mtp_output['hidden_state'] = h
        mtp_output['attention_weight'] = transformer_block_output['attention_weight']
        mtp_output['affinity_scores'] = transformer_block_output['affinity_scores']
        mtp_output['auxiliary_loss'] = transformer_block_output['auxiliary_loss']
        return mtp_output


"""# DeepSeekCode V4 (edu)"""

# @title DeepSeekCodeV4
class DeepSeekCodeV4(nn.Module):
    def __init__(self, args, device='cpu'):
        super().__init__()
        self.args = args
        self.device = device
        self.vocab_size = args.vocab_size
        self.context_size = args.context_size
        self.lambda_mtp = args.lambda_mtp

        """ 共通重み """
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.output_head = nn.Linear(args.d_model, args.vocab_size, bias=True)

        """ 学習モデル """
        self.main_model = MainModel(self.embedding, self.output_head, args)

        self.mtp_modules = nn.ModuleList()
        for _ in range(args.multi_token_depth):
            mtp_module = MTPModule(self.embedding, self.output_head, args)
            self.mtp_modules.append(mtp_module)

        self.to(device)
        self.freqs_cis = precompute_freqs_cis(args, device)
        self.criterion = nn.CrossEntropyLoss()
        self.reset_kv_cache = self.main_model.reset_kv_cache

    # --- 事前学習の損失計算 ---
    def pretrain(self, source):
        output = self._calculate_main_loss(source)
        main_loss = output['main_loss']
        main_balance_loss = output['main_balance_loss']
        hidden_state = output['hidden_state']

        output = self._calculate_mtp_loss(source, hidden_state)
        mtp_losses = output['mtp_losses']
        mtp_balance_losses = output['mtp_balance_losses']

        total_loss = (main_loss + self.lambda_mtp * mtp_losses
                      + main_balance_loss + mtp_balance_losses)
        return total_loss

    def _calculate_main_loss(self, source):
        main_input_ids = source[:, :self.context_size]
        main_target_ids = source[:, 1:self.context_size + 1]
        main_freqs_cis = self.freqs_cis[:self.context_size]

        self.main_model.reset_kv_cache()
        main_output = self.main_model(main_input_ids, 0, main_freqs_cis,
                                      None, train=True)
        main_logits = main_output['logits']

        predicted = main_logits.contiguous().view(-1, self.vocab_size)
        target = main_target_ids.contiguous().view(-1)
        main_loss = self.criterion(predicted, target)

        output = {}
        output['main_loss'] = main_loss
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
            logits = mtp_output['logits']
            hidden_state = mtp_output['hidden_state']

            predicted = logits.contiguous().view(-1, self.vocab_size)
            target = mtp_target_ids.contiguous().view(-1)
            mtp_losses += self.criterion(predicted, target)
            mtp_balance_losses += mtp_output['auxiliary_loss']

        output = {}
        output['mtp_losses'] = mtp_losses
        output['mtp_balance_losses'] = mtp_balance_losses
        return output

    def update_expert_bias(self):
        """全MoE層のエキスパートバイアスを更新（学習ステップ毎に呼ぶ）"""
        for module in self.modules():
            if isinstance(module, MoE):
                module.update_expert_bias()

    def compute_log_prob(self, input_ids, causal_mask=None, train=False):
        self.reset_kv_cache()
        main_output = self.main_model(input_ids, 0, self.freqs_cis,
                                      causal_mask=None, train=train)
        return F.log_softmax(main_output['logits'], dim=-1)

    # --- テキスト生成 ---
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
            sampled_index = torch.multinomial(probs, num_samples=1)
            return top_k_indices.gather(dim=-1, index=sampled_index)

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
            output = self.main_model(input_ids, start_pos, input_freqs_cis,
                                     causal_mask=None, train=False)
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

            # コンテキスト上限に達したら停止（教育用の簡略化）
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


"""# V4: Muon オプティマイザ"""

# @title newton_schulz_hybrid
def newton_schulz_hybrid(G, eps=1e-7):
    """
    ハイブリッド Newton-Schulz 反復（論文 Algorithm 1）。

    勾配（モメンタム）行列を近似的に直交化する:
      M_k = a*M_{k-1} + b*(M M^T)M + c*(M M^T)^2 M
    係数:
      最初の8ステップ: (a, b, c) = (3.4445, -4.7750, 2.0315)  … 高速な立ち上げ
      最後の2ステップ: (a, b, c) = (2, -1.5, 0.5)             … 高精度な仕上げ
    """
    assert G.dim() == 2
    X = G / (G.norm() + eps)  # フロベニウスノルムで正規化

    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True

    for a, b, c in [(3.4445, -4.7750, 2.0315)] * 8 + [(2.0, -1.5, 0.5)] * 2:
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X

    if transposed:
        X = X.T
    return X


# @title Muon
class Muon(torch.optim.Optimizer):
    """
    Muon オプティマイザ（教育用実装）。

    2次元の重み行列に対して:
      1) モメンタム累積: M_t = μ・M_{t-1} + G_t
      2) ハイブリッド Newton-Schulz 反復で M_t を直交化
      3) 行列サイズに応じたスケールで更新
    埋め込み・出力ヘッド・1次元パラメータには AdamW を使うこと
    （create_optimizers 参照）。
    """

    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.01):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
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
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']

                # モメンタム累積: M_t = μ・M_{t-1} + G_t
                buf.mul_(mu).add_(g)

                # Newton-Schulz による直交化
                update = newton_schulz_hybrid(buf.reshape(buf.size(0), -1).float())
                update = update.reshape(p.shape).type_as(p)

                # 行列の形状に応じた学習率スケール（RMSを揃える経験則）
                scale = 0.2 * math.sqrt(max(p.size(0), p.numel() // p.size(0)))

                # 重み減衰 + 更新
                p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr * scale)

        return loss


# @title create_optimizers
def create_optimizers(model, muon_lr=0.02, adamw_lr=3e-4):
    """
    パラメータを Muon / AdamW に振り分ける。
      - 2次元の重み行列（本体）      → Muon
      - 埋め込み・出力ヘッド・
        1次元パラメータ・ルーター等  → AdamW
    """
    muon_params = []
    adamw_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_matrix = (p.ndim == 2)
        is_special = ('embedding' in name or 'output_head' in name
                      or 'centroids' in name)
        if is_matrix and not is_special:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    muon = Muon(muon_params, lr=muon_lr)
    adamw = torch.optim.AdamW(adamw_params, lr=adamw_lr, weight_decay=0.01)
    return muon, adamw


"""# 動作確認用"""

if __name__ == '__main__':
    torch.manual_seed(0)
    args = get_default_args(vocab_size=100, context_size=32, max_seq_len=64)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = DeepSeekCodeV4(args, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'パラメータ数: {n_params:,}')

    # 学習の1ステップ（sourceは context+depth+1 の長さが必要）
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

    # 生成
    prompt = torch.randint(0, args.vocab_size, (1, 5), device=device)
    ids = model.generate_ids(prompt, max_new_tokens=10)
    print('generated:', ids.tolist())
