#@title TransformerBlock
import torch
import torch.nn.functional as F
from torch import nn

# -*- coding: utf-8 -*-
"""# ライブラリーのインポート"""

#@title import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# -*- coding: utf-8 -*-
"""# ライブラリーのインポート"""

#@title import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init



# @title Args
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


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
   




class TransformerBlock(nn.Module):
    def __init__(self, args, MHA):
        super().__init__()
        
        self.norm1 = RMSNorm(args.d_model)
        self.attn = MHA(args)
        self.norm2 = RMSNorm(args.d_model)
        self.ffn  = FeedForward(args)

    def forward(self, x, freqs_cis, mask=None, train=False):
    
        output = self.attn(self.norm1(x), freqs_cis, mask=mask, train=train)
        x = x + output['hidden_state']
        output = self.ffn(self.norm2(x))
        x = x + output['hidden_state'] # Add the hidden_state from the ffn output
        
        return x

#@title Transformer
class Transformer(nn.Module):
    def __init__(self, args, MHA):
    
        super().__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)

        self.layers = torch.nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args, MHA))
        # 内包表記を使う場合
        # self.blocks = nn.ModuleList([
        #     TransformerBlock(args)
        # for _ in range(n_layers)])

        self.norm = RMSNorm(args.d_model) # Use RMSNorm and args.d_model
        self.output_head = nn.Linear(args.d_model, args.vocab_size, bias=True) # Use args.d_model and args.vocab_size

    def forward(self, input_ids, freqs_cis, mask=None, train=False):
        # input_ids: [B, T]
        x = self.embedding(input_ids)  # [B, T, D]

        for layer in self.layers:
            x = layer(x, freqs_cis, mask, train=train)

        x = self.norm(x)
        logits = self.output_head(x)  # [B,T,V]
        return logits

    def generate(self, text, freqs_cis):

        input_ids = encode_text(text, stoi).to(device)
        input_ids = input_ids.unsqueeze(0)  # バッチ次元を追加


        seq_len = input_ids.size(1)


        generated_ids = []
        start_pos = 0

        input_freqs_cis = freqs_cis[start_pos : start_pos + seq_len]

        for i in range(max_seq_len):
            seq_len = input_ids.size(0)
            mask = create_causal_mask(seq_len, device)
            x = self.embedding(input_ids)  # [B, T, D]
            for layer in self.layers:
                x = layer(x, input_freqs_cis, mask=mask, train=False)
            x = self.norm(x)
            logits = self.output_head(x)  # [B,T,V]

            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            generated_ids.append(next_token_id[0].item())
            input_ids = next_token_id.unsqueeze(0)

            start_pos += seq_len

            input_freqs_cis = freqs_cis[start_pos: start_pos + seq_len]

            print(decode_ids(next_token_id, itos), end="")



# @title FeedForward
class FeedForward(nn.Module):
    def __init__(self, args, dropout=0.1):
        # super(FeedForward, self).__init__()
        super().__init__()

        d_model = args.d_model
        d_hidden = 8 // 3 * d_model

        self.fc1 = nn.Linear(d_model, d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_hidden, d_model)

        # 全結合層をザビエル方式で初期化
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x, train=False):
        h = self.fc1(x)
        h = F.relu(h)
        h = self.fc2(h)
        h = self.dropout(h)

        # balance_loss = torch.tensor(0.0, requires_grad=False, device=x.device) # Set requires_grad=False

        output = {}
        output['hidden_state'] = h
        output['affinity_scores'] = None
        return output


