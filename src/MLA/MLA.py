import torch
import torch.nn.functional as F
from torch import nn


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

        attn_weights = F.softmax(score, dim=-1)
        # 特定の単語に注意を払いすぎないようにdropoutを適用します
        attn_weights = self.dropout(attn_weights)
        # Weighted value
        attention_output = torch.matmul(attn_weights, v)

        return attention_output, attn_weights
        