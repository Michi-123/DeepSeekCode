# -*- coding: utf-8 -*-
"""# ライブラリーのインポート"""

#@title import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init



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

