# -*- coding: utf-8 -*-
"""# ライブラリーのインポート"""

#@title import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init



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


    if hasattr(args, "rope_scaling_factor") and args.rope_scaling_factor > 1.0:
        scale = math.log(args.max_seq_len) / math.log(args.max_seq_len // args.rope_scaling_factor)
        m = m * scale
        
    # 回転角の計算
    rotation_angles = torch.outer(m, freqs)

    # 複素数の絶対値を1で作成
    abs = torch.ones_like(freqs)

    # 回転係数の作成: オイラーの公式[cosθ + i*sinθ]を再現
    freqs_cis = torch.polar(abs, rotation_angles)

    # 計算グラフから切断して返却
    return freqs_cis.detach()


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


