# -*- coding: utf-8 -*-
"""
DeepSeekCode 共通モジュール
================================
教材ノート「教材_Expert検証グラフ」用の共通部品です。
MoE（混合専門家）の作成・パラメータ更新は *教材ノート側* に置き、
ここには MoE 以外の部品（Norm / Attention / TransformerBlock / MainModel /
MTPModule / DeepSeekCode / Args / 各種ヘルパー）をまとめています。

使い方（教材ノート側）:
    import sys; sys.path.append('DeepSeekCode/src')
    from deepseek_modules import *
    import deepseek_modules as dsm
    # 教材側で MoE を定義したあとに:
    dsm.MoE = MoE     # TransformerBlock が使う MoE を注入する
"""

# MoE は教材ノート側で定義し、dsm.MoE = MoE で注入する（ここでは空の置き場）
MoE = None



# ===== import math =====
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time



# ===== class Args: =====
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)



# ===== # @title create_causal_mask =====
# @title create_causal_mask
def create_causal_mask(seq_len, device='cpu') -> torch.Tensor:
    # 形状: (T, T)。未来をTrueでマスク（上三角の+1オフセット）
    ones = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    return torch.triu(ones, 1)



# ===== # @title create_padding_mask =====
# @title create_padding_mask
def create_padding_mask(attention_mask):
    # attention_mask: (B, T) で 1=トークン, 0=PAD を想定
    # 形状: (B, T)。PAD位置をTrueに
    return (attention_mask == 0)



# ===== # @title KVCache =====
# @title KVCache
# 汎用的なキー・バリューキャッシュ管理クラス
class KVCache(nn.Module):
    def __init__(self, context_size):
        super().__init__()

        # キャッシュの最大長（トークン数）
        self.context_size = context_size

        # キャッシュ
        self.keys = None
        self.values = None

    def get(self):
        return self.keys, self.values

    def update(self, k, v):
        if self.keys is None:
            # 初回は初期化
            self.keys = k.detach()
            self.values = v.detach()
        else:
            # 過去のキャッシュと結合
            self.keys = torch.cat([self.keys, k.detach()], dim=1)
            self.values = torch.cat([self.values, v.detach()], dim=1)

        # 最大長を超える場合は、先頭から削除（FIFO）
        if self.keys.size(1) > self.context_size:
            self.keys = self.keys[:, -self.context_size:, ...] # ...は残りの全ての次元を選択する省略記号(Ellipsis)
            self.values = self.values[:, -self.context_size:, ...]

    def reset(self):
        self.keys = None
        self.values = None



# ===== # @title precompute_freqs_cis =====
# @title precompute_freqs_cis
def precompute_freqs_cis(args, device='cpu'):

    # 各ヘッドの次元数。
    # dim = args.d_model // args.n_heads
    dim = args.d_rope # RoPE用の次元は qC,kC,vC の半分 <-- NG: Set in Hyperparameter

    # 角周波数のインデックス （要device指定）
    indices = torch.arange(0, dim, 2, dtype=torch.float32, device=device)

    # 各インデックスを埋め込み全体の次元数で正規化
    scaled_index = indices / dim

    # 角周波数
    freqs = 1.0 / (args.rope_theta ** scaled_index)

    # トークンの位置インデックスの生成 （要device指定）
    m = torch.arange(args.max_seq_len, dtype=torch.float32, device=device)

    # 回転角の計算
    rotation_angles = torch.outer(m, freqs)

    # 複素数の絶対値を1で作成
    abs = torch.ones_like(freqs)

    # 回転係数の作成: オイラーの公式[cosθ + i*sinθ]を再現
    freqs_cis = torch.polar(abs, rotation_angles)

    # 計算グラフから切断して返却
    return freqs_cis.detach()



# ===== # @title apply_rope =====
# @title apply_rope
# ロータリ位置埋め込みを適用
def apply_rope(x, freqs_cis):
    # xの形状からマルチヘッドの次元を取得
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    n_heads = x.shape[2]
    d_head = x.shape[3]

    # 形状の変形
    x_reshaped = x.view(batch_size, seq_len, n_heads, d_head // 2, 2)
    # x_reshaped = x.view(*x.shape[:-1], -1, 2)) # *は変数x を任意の形状で扱う

    # 複素数に変換
    x_complex = torch.view_as_complex(x_reshaped)

    # 回転係数をxの形状に変形 (1, seq_len, 1, d_head // 2)
    freqs_cis = freqs_cis[None, :seq_len, None, :]
    # freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))

    # 複素数のベクトルに回転を適用する（RoPE処理のコア部分）
    rotated_complex = x_complex * freqs_cis

    # 複素数として計算された結果を実数に戻して、最後の2つの次元をフラット化
    rotated_embeds = torch.view_as_real(rotated_complex).flatten(3)

    return rotated_embeds.type_as(x) # rotated_embeds.to(dtype=x.dtype)



# ===== # @title RMSNorm =====
# @title RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))  # 学習可能なスケーリングパラメータ

    def forward(self, x):
        # 正規化（中心化なし。スピード重視）
        x_normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # 学習可能な重みでスケーリング
        return x_normed * self.weight



# ===== #@title PositionalEncoding =====
#@title PositionalEncoding
class PositionalEncoding(nn.Module):
    def __init__(self, context_size, d_model):
        super(PositionalEncoding, self).__init__()

        # Create a matrix of shape (context_size, d_model) with positional encodings
        pe = torch.zeros(context_size, d_model)

        # for pos in range(context_size):
        #     for i in range(d_model):
        #         if  i % 2 == 0:
        #             pe[pos,i] = math.sin(pos/(10000**((2*i)/d_model)))
        #         else:
        #             pe[pos,i] = math.cos(pos/(10000**((2*(i-1))/d_model)))

        for pos in range(context_size):
            for i in range(0, d_model, 2):
                pe[pos,i]   = math.sin(pos/(10000**(i/d_model)))
                pe[pos,i+1] = math.cos(pos/(10000**(i/d_model)))

        # 学習パラメーターの更新対象から外してクラス変数に確保(重要)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # positional encodingを埋め込みベクトルへ追加します
        return self.pe[:, :x.size(1)].detach()



# ===== # @title ScaledDotProductAttention =====
# @title ScaledDotProductAttention
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, attn_dropout=0.1):
        super().__init__()
        # self.sqrt_d_k = d_model ** 0.5 # sqrt(d_k)と同じ
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None, cache_size=0):
        sqrt_d_k = q.size(-1) ** 0.5
        score = torch.matmul(q, k.transpose(2, 3)) / sqrt_d_k

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)

            # True（= 未来 = 禁止）を −∞ にする
            score = score.masked_fill(mask, torch.finfo(score.dtype).min)

        attention_weight = F.softmax(score, dim=-1)
        # 特定の単語に注意を払いすぎないようにdropoutを適用します
        attention_weight = self.dropout(attention_weight)
        # Weighted value
        attention_output = torch.matmul(attention_weight, v)

        return attention_output, attention_weight



# ===== # @title MLA =====
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
        h, w = self.attention(q, k, v, mask)

        # 出力を整形して返す
        h = h.transpose(1, 2)
        h = h.reshape(batch_size, seq_len, self.n_heads * self.d_h) # H*Dでトークンのベクトルに変換
        h = self.output_head(h)

        output = {}
        output['hidden_state'] = h
        output['attention_weight'] = w

        return output



# ===== #@title MHA =====
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



# ===== # @title FeedForward =====
# @title FeedForward
class FeedForward(nn.Module):
    def __init__(self, args, dropout=0.1):
        super().__init__()

        d_model = args.d_model

        self.fc1 = nn.Linear(d_model, d_model * 4 )
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model * 4, d_model)

        # 全結合層をザビエル方式で初期化
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x, train=False):
        h = self.fc1(x)
        h = F.gelu(h)
        h = self.fc2(h)
        h = self.dropout(h)

        # Auxiliary loss
        dummy_loss = torch.tensor(0.0, device=x.device)

        output = {}
        output['hidden_state'] = h
        output['affinity_scores'] = dummy_loss
        output['auxiliary_loss'] = dummy_loss
        output['attention_weight'] = dummy_loss
        return output



# ===== # @title LayerNorm =====
# @title LayerNorm
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        nn.init.normal_(self.norm.weight, mean=0, std=0.02)

    def forward(self, x):
        return self.norm(x)



# ===== # @title TransformerBlock =====
# @title TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(self, args, layer_id=0, mtp=False):
        super().__init__()

        if args.d_c is None:
            self.attention = MultiHeadAttention(args)
        else:
            self.attention = MLA(args)

        # 最初のTransformerブロックまたはMTPの場合にMoEを使用
        if layer_id > 0 or mtp:
            self.feed_forward = MoE(args)
        else:
            self.feed_forward = FeedForward(args)

        # self.attn_norm = LayerNorm(args.d_model)
        self.attn_norm = RMSNorm(args.d_model, eps=args.norm_eps)
        # self.ffn_norm = LayerNorm(args.d_model)
        self.ffn_norm = RMSNorm(args.d_model, eps=args.norm_eps)
        self.layer_id = layer_id

    def forward(self, x, start_pos, freqs_cis=None, mask=None, train=False): # Added freqs_cis=None

        h1 = self.attn_norm(x)
        attention_output = self.attention(h1, freqs_cis, mask, train) # 出力形式を output 辞書に変更する?
        h1 = attention_output['hidden_state']
        w = attention_output['attention_weight']

        h1 = h1 + x # 残渣結合

        h2 = self.ffn_norm(h1)
        feed_forward_output = self.feed_forward(h2, train)
        h2 = feed_forward_output['hidden_state']

        h = h2 + h1 # 残渣結合

        output = {}
        output['hidden_state'] = h
        output['affinity_scores'] = feed_forward_output['affinity_scores']
        output['auxiliary_loss'] = feed_forward_output['auxiliary_loss']
        output['attention_weight'] = w

        return output



# ===== # @title Main Model =====
# @title Main Model
class MainModel(nn.Module):
    def __init__(self, embedding, output_head, args):
        super().__init__()

        self.context_size = args.context_size
        self.embedding = embedding # 共通重み

        # PositionalEncoding(args.context_size, args.d_model)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(args, layer_id))
        """
        self.layers = torch.nn.ModuleList([
            TransformerBlock(args, layer_id)
            for layer_id in range(args.n_layers)
        ])
        """
        self.output_norm = RMSNorm(args.d_model, eps=args.norm_eps)
        # self.output_norm = LayerNorm(args.d_model)
        self.output_head = output_head # 共通重み

    def reset_kv_cache(self):
        for layer in self.layers:
            layer.attention.reset_kv_cache()

    def forward(self, input_ids, start_pos, freqs_cis, mask=None, train=False):

        # 入力トークンを埋め込みベクトル（隠れ状態）へ変換
        h = self.embedding(input_ids)
        # h = h + self.pe(tokens) # Positional Encoding

        # Initialize total balance loss as a zero tensor with requires_grad=True
        # balance_losses = torch.tensor(0.0, device=tokens.device, requires_grad=True)
        balance_losses = torch.tensor(0.0, device=input_ids.device)  # requires_grad削除

        for layer in self.layers:  # Changed layer_id to layer
            # h, balance_loss, _attn_weight = layer(h, start_pos, freqs_cis, mask, train=train) # Pass train and unpack outputs
            transformer_block_output = layer(h, start_pos, freqs_cis, mask, train=train) # Pass train and unpack outputs

            # 埋め込みベクトル（隠れ状態）→　再帰的にTransformer block へ入力される
            h = transformer_block_output['hidden_state']

        # 埋め込みベクトル（隠れ状態）
        h = self.output_norm(h)

        # 正規化前の生の値（logits）
        logits = self.output_head(h)

        main_output = {}
        main_output['logits'] = logits
        main_output['hidden_state'] = h # 埋め込みベクトル（隠れ状態）
        main_output['affinity_scores'] = transformer_block_output['affinity_scores']
        main_output['auxiliary_loss'] = transformer_block_output['auxiliary_loss']
        main_output['attention_weight'] = transformer_block_output['attention_weight']

        return main_output



# ===== # @title MTP Module =====
# @title MTP Module
class MTPModule(nn.Module):
    def __init__(self, embedding, output_head, args):
        super().__init__()
        self.embedding = embedding # 共通重み
        self.norm_pres = RMSNorm(args.d_model, eps=args.norm_eps)
        self.norm_prev = RMSNorm(args.d_model, eps=args.norm_eps)
        self.projection = nn.Linear(args.d_model * 2, args.d_model) # 単純な線形変換
        self.transformer_block = TransformerBlock(args, mtp=True)
        self.output_norm = RMSNorm(args.d_model, eps=args.norm_eps)
        self.output_head = output_head # 共通重み

    def forward(self, input_ids, h_prev, start_pos, freqs_cis, mask=None):

        # 入力トークンを埋め込みベクトル（隠れ状態）へ変換
        h_curr = self.embedding(input_ids) # Current sequences

        # 正規化
        h_curr = self.norm_pres(h_curr)
        h_prev = self.norm_prev(h_prev) # 直前のモデルが出力した隠れ状態

        # 隠れ状態を結合
        concatenation = torch.cat([h_curr, h_prev], dim=-1)
        h = self.projection(concatenation)

        transformer_block_output = self.transformer_block(h, start_pos, freqs_cis, mask, train=True)

        h = transformer_block_output['hidden_state']
        # affinity_scores = transformer_block_output['affinity_scores']

        h = self.output_norm(h) # 補足（論文にはない）

        logits = self.output_head(h)

        mtp_output = {}
        mtp_output['logits'] = logits
        mtp_output['hidden_state'] = h
        mtp_output['attention_weight'] = transformer_block_output['attention_weight']
        mtp_output['affinity_scores'] = transformer_block_output['affinity_scores']
        mtp_output['auxiliary_loss'] = transformer_block_output['auxiliary_loss']

        return mtp_output



# ===== # @title DeepSeekCode: __init__ =====
# @title DeepSeekCode: __init__
class DeepSeekCode(nn.Module):
    def __init__(self, args, device='cpu'):
        super().__init__()

        self.args = args
        self.device = device
        self.vocab_size = args.vocab_size
        self.context_size = args.context_size
        self.lambda_mtp = args.lambda_mtp

        """ 共通重み """
        # 最初の埋め込み層
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.embedding.to(device)

        # 出力層
        self.output_head = nn.Linear(args.d_model, args.vocab_size, bias=True)
        self.output_head.to(device)

        """ 学習モデル """
        # Main
        self.main_model = MainModel(self.embedding, self.output_head, args)
        self.main_model.to(device)

        # Multi-Token Prediction (MTP)
        self.mtp_modules = nn.ModuleList() # 計算グラフでつなげるためにModuleListを利用
        for _ in range(args.multi_token_depth):
            mtp_module = MTPModule(self.embedding, self.output_head, args)
            self.mtp_modules.append(mtp_module)
        self.mtp_modules.to(device)

        self.freqs_cis = precompute_freqs_cis(args, device)

        # 誤差関数
        self.criterion = nn.CrossEntropyLoss()

        self.reset_kv_cache = self.main_model.reset_kv_cache

    def parameters(self, recurse: bool = True):
        """PyTorch の API に合わせる"""
        params = list(self.main_model.parameters())
        for mtp_module in self.mtp_modules:
            params.extend(list(mtp_module.parameters()))
        return params



# ===== # @title DeepSeekCode: pretrain =====
# @title DeepSeekCode: pretrain
class DeepSeekCode(DeepSeekCode):
    def pretrain(self, source):
        # main_loss, main_balance_loss = self._calculate_main_loss(source)
        output = self._calculate_main_loss(source)
        main_loss = output['main_loss']
        main_balance_loss = output['main_balance_loss']
        hidden_state = output['hidden_state']

        # mtp_losses, mtp_balance_losses = self._calculate_mtp_loss(source, hidden_state)
        output = self._calculate_mtp_loss(source, hidden_state)
        mtp_losses = output['mtp_losses']
        mtp_balance_losses = output['mtp_balance_losses']
        # total_loss = main_loss + self.lambda_mtp * mtp_losses + main_balance_loss + self.lambda_mtp * mtp_balance_losses
        total_loss = main_loss + self.lambda_mtp * mtp_losses + main_balance_loss + mtp_balance_losses

        return total_loss



# ===== # @title DeepSeekCode: _calculate_main_loss =====
# @title DeepSeekCode: _calculate_main_loss
class DeepSeekCode(DeepSeekCode):
    def _calculate_main_loss(self, source):
        main_input_ids = source[:, :self.context_size]
        main_target_ids = source[:, 1:self.context_size + 1]
        main_freqs_cis = self.freqs_cis[:self.context_size]

        # mask = create_attention_mask(self.context_size).to(source.device)
        mask = create_causal_mask(self.context_size).to(source.device)

        self.main_model.reset_kv_cache()

        main_output = self.main_model(main_input_ids, 0, main_freqs_cis, mask, train=True)
        main_logits = main_output['logits']
        # hidden_state = main_output['hidden_state']
        # main_affinity_scores = main_output['affinity_scores']

        predicted = main_logits.contiguous().view(-1, self.vocab_size)
        target = main_target_ids.contiguous().view(-1)

        main_loss = self.criterion(predicted, target)

        output = {}
        output['main_loss'] = main_loss
        output['main_balance_loss'] = main_output['auxiliary_loss']
        output['hidden_state'] = main_output['hidden_state']
        return output



# ===== # @title DeepSeekCode: _calculate_mtp_loss =====
# @title DeepSeekCode: _calculate_mtp_loss
class DeepSeekCode(DeepSeekCode):
    def _calculate_mtp_loss(self, source, hidden_state):
        mtp_losses = 0
        mtp_balance_losses = 0
        for mtp_offset, mtp_module in enumerate(self.mtp_modules):
            mtp_input_ids = source[:, mtp_offset + 1: self.context_size + mtp_offset + 1]
            mtp_target_ids = source[:, mtp_offset + 2: self.context_size + mtp_offset + 2]
            mtp_freqs_cis = self.freqs_cis[mtp_offset + 1: self.context_size + mtp_offset + 1]

            # mask = create_attention_mask(self.context_size).to(source.device)
            mask = create_causal_mask(self.context_size).to(source.device)

            mtp_output = mtp_module(mtp_input_ids, hidden_state, 0, mtp_freqs_cis)
            logits = mtp_output['logits']
            hidden_state = mtp_output['hidden_state']
            # affinity_scores = mtp_output['affinity_scores']
            auxiliary_loss = mtp_output['auxiliary_loss']

            predicted = logits.contiguous().view(-1, self.vocab_size)
            target = mtp_target_ids.contiguous().view(-1)

            mtp_losses += self.criterion(predicted, target)
            # mtp_balance_losses += calculate_balance_loss(affinity_scores, self.args)
            mtp_balance_losses += auxiliary_loss

        output = {}
        output['mtp_losses'] = mtp_losses
        output['mtp_balance_losses'] = mtp_balance_losses

        return output



# ===== # @title DeepSeekCode: compute_log_prob =====
# @title DeepSeekCode: compute_log_prob
class DeepSeekCode(DeepSeekCode):
    def compute_log_prob(self, input_ids, mask=None, train=False):
        start_pos = 0
        self.reset_kv_cache()

        if mask is None:
            mask = create_causal_mask(input_ids.size(1)).to(input_ids.device)
        main_output = self.main_model(input_ids, start_pos, self.freqs_cis, mask=mask, train=train)

        # return main_output['logits']

        # ロジットを対数確率に変換
        # dim=-1 は通常、語彙次元（最後の次元）に対してsoftmaxを適用することを意味します。
        log_probs = F.log_softmax(main_output['logits'], dim=-1)

        return log_probs



# ===== # @title DeepSeekCode: generate =====
# @title DeepSeekCode: generate
class DeepSeekCode(DeepSeekCode):
    def generate_text(self,
            input_ids,
            tokenizer,
            max_new_tokens=20,
            temperature=1.0,
            top_k=1,
            eos_token_id=None,
            mask=None,
            max_cache_size=None,
            delay=0.0):

        self.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_token_id,
            mask=mask,
            max_cache_size=max_cache_size,
            tokenizer=tokenizer,
            delay=0.0)

    def generate_ids(self,
            input_ids,
            max_new_tokens=20,
            temperature=1.0,
            top_k=1,
            eos_token_id=None,
            mask=None,
            max_cache_size=None,
            delay=0.0):
        return self.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_token_id,
            mask=mask,
            max_cache_size=max_cache_size,
            tokenizer=None,
            delay=0.0)

    def generate(self,
            input_ids,
            max_new_tokens=20,
            temperature=1.0,
            top_k=1,
            eos_token_id=None,
            mask=None,
            max_cache_size=None,
            tokenizer=None,
            delay=0.0):

        def reset_buffered_input_ids():
            buffered_input_ids = input_ids[:, 0:0]
            return buffered_input_ids

        def adjust_bactch_dim(input_ids):
            # バッチの次元が無い場合
            if input_ids.dim() == 1:
                # バッチの次元を追加
                input_ids = input_ids.unsqueeze(0)

            return input_ids


        def top_k_sampling(logits, top_k, temperature):
            # 温度スケーリング
            logits = logits / temperature

            # 上位 top_k のインデックスとスコアを取得
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)

            # ソフトマックスで確率分布に変換
            probs = torch.nn.functional.softmax(top_k_logits, dim=-1)

            # トップKの中から確率的にサンプリング
            sampled_index = torch.multinomial(probs, num_samples=1)

            # 元の語彙空間に戻す
            return top_k_indices.gather(dim=-1, index=sampled_index)

        device = input_ids.device

        self.main_model.eval()

        context_size = self.main_model.context_size

        # バッチ次元の適用処理
        input_ids = adjust_bactch_dim(input_ids)

        # 入力トークンをモデルの入力サイズで切り落とす
        input_ids = input_ids[:, :context_size]

        # 生成されたトークンを格納する変数を入力トークンで初期化（バッチの次元を保持して、シーケンスを０にする）
        generated_ids = input_ids[:, 0:0]

        # 最大キャッシュが未設定の場合は
        if max_cache_size is None:
            # モデルのサイズに合わせる（通常はこれで実行） キャッシュの効果を研究したい場合は値を指定する
            max_cache_size = context_size

        # 最大キャッシュに達したときに再入力するためのトークンのバッファー
        buffered_input_ids = reset_buffered_input_ids()

        self.main_model.reset_kv_cache()

        start_pos = 0

        for _ in range(max_new_tokens):

            # 入力シーケンスの長さを取得
            seq_len = input_ids.size(1)

            # 入力用の回転埋め込み行列の切り出し
            input_freqs_cis = self.freqs_cis[start_pos: start_pos + seq_len]
            input_freqs_cis = input_freqs_cis.to(device)

            # mask = create_attention_mask(seq_len)
            mask = create_causal_mask(seq_len).to(device)

            output = self.main_model(input_ids, start_pos, input_freqs_cis, mask=mask, train=False)

            # 最後のトークンを予測するための生値を取得（3階目のテンソールが語彙ベクトル）
            last_logits = output['logits'][:,-1,:]

            if top_k == 1:
                # 最大の値を持つ要素を
                # index = last_logits.argmax(dim=1).item()
                index = last_logits.argmax(dim=1) # fix
            else:
                if 0:
                    # k個の最大値からランダムに1つをサンプリング
                    _topk_values, topk_indices = torch.topk(last_logits, top_k)
                    random_indices_in_topk = torch.randint(0, top_k, (input_ids.size(0),), dtype=torch.long) #fix
                    random_indices_in_topk = random_indices_in_topk.to(device)
                    index = torch.gather(topk_indices, 1, random_indices_in_topk.unsqueeze(1)) #fix
                else:
                    # 最適top_kサンプリング
                    index = top_k_sampling(last_logits, top_k, temperature)

            # 次の入力トークンに変換（KVキャッシュを使うので2回目からは１トークンでよい）

            index = index.view(-1,1)
            input_ids = index.to(device)

            # 入力トークンの退避（最大キャッシュに達したときのために）
            buffered_input_ids = torch.cat((buffered_input_ids, input_ids), dim=1)

            # 生成結果の処理
            if tokenizer is not None:
                print(tokenizer.index2word[index[0].item()] ,end="")
                if tokenizer.eos_token_id ==  index[0].item():
                    break
            else:
                generated_ids = torch.cat([generated_ids, input_ids], dim=1)

            # 視覚効果 # 1.0で1秒
            time.sleep(delay)

            if start_pos == 0:
                # 2回目は、シーケンス長から
                start_pos = seq_len
            else:
                # 3回目からはインクリメント
                start_pos += 1

            # Reset cache
            cache = self.main_model.layers[0].attention.kv_cache
            # if cache.size(1) >= max_cache_size: #NG
            if cache.keys is not None and cache.keys.size(1) >= max_cache_size:
                """
                キャッシュをクリアして、
                これまでに入力されたトークンからモデルのサイズの半分の長さだけ利用して最初の入力トークンとする。
                バッファーはクリアする。
                最初の入力と同じになるので位置情報も0に初期化する。
                """
                self.main_model.reset_kv_cache()
                input_ids = buffered_input_ids[:, -context_size//2:]
                buffered_input_ids = reset_buffered_input_ids().to(device)
                start_pos = 0
                """a
                非常にtrickyな操作であるが、過剰適合モデルで効果を確認できる。
                例えば最初に数トークン入力すれば、続きの文章を最後まで生成してくれる。
                ただし、このような特殊な処理は、キャッシュをクリアしてしまうので、最初の文章の意味は覚えていない。
                DeepSeekやChatGPTは巨大なcontext_sizeを持つので、この処理は不要。
                """

        if tokenizer is None:
            return generated_ids
        else:
            print()
            return None



# ===== #@title pad_mask_after_eos =====
#@title pad_mask_after_eos
def pad_mask_after_eos(generated_ids, eos_id):
    modified_token_ids = generated_ids.clone()  # 元のテンソルを直接変更しないようにクローンを作成

    # 各バッチをループして処理
    for i in range(generated_ids.size(0)):  # バッチサイズ分ループ
        row = generated_ids[i]

        # eos_idが含まれる位置を探す
        eos_positions = (row == eos_id).nonzero(as_tuple=True)[0]

        # eos_idが見つかった場合のみ処理
        if eos_positions.numel() > 0:
            first_eos_pos = eos_positions[0].item()  # 最初のeosの位置

            # 最初のeosの次の位置以降を0で埋める
            modified_token_ids[i, first_eos_pos + 1:] = 0

    return modified_token_ids

# MHA の別名（TransformerBlock が d_c is None のとき参照する。通常の設定では未使用）
MultiHeadAttention = MHA
