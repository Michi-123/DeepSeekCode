# -*- coding: utf-8 -*-
"""# ライブラリーのインポート"""

#@title import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# @title create_causal_mask
def create_causal_mask(seq_len, device='cpu') -> torch.Tensor:
    # 形状: (T, T)。未来をTrueでマスク（上三角の+1オフセット）
    ones = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    return torch.triu(ones, 1)

    
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
   
     
"""
RMSNormがnn.Linearを使わずにnn.Parameterをweightに使用しているのは、
この正規化レイヤーが各次元に対して単純なスケーリング係数を学習するためです。
nn.Parameterは、この重みのような個別の学習可能なテンソルのために設計されています。
一方、nn.Linearは完全な行列乗算とバイアス操作であり、
RMSNormで行われるような単純なスケーリング操作には必要ありません。
"""

    
from IPython.display import HTML, display

""" 改行処理 """
def set_css(*args, **kwargs):
  display(HTML('''
  <style>
    pre {
        white-space: pre-wrap;
    }
  </style>
  '''))
get_ipython().events.register('pre_run_cell', set_css)

