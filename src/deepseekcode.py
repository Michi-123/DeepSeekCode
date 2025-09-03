import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


d_model = 8 
n_layers = 1 
context_size = 20 
norm_eps = 1e-5 


n_heads = 4 
max_seq_len = 1000 #@param{type:"integer"}
rope_theta = 10000 #@param{type:"number"}

use_MLA = True #@param{type:"boolean"}

if use_MLA:
    #@markdown MLA
    # MLA
    d_c    = 16 #@param{type:"integer"}
    d_cQ   = 32 #@param{type:"integer"}
    # d_cQ > d_c
    d_rope = 16 #@param{type:"integer"}
    d_head = 48 #@param{type:"integer"}

else:
    #  MHA
    d_rope = d_model // n_heads
    d_head = d_model // n_heads
    d_c = None
    d_cQ = None


n_shared_experts = 1 #@param{type:"integer"}
n_routed_experts = 3 #@param{type:"integer"}
n_activated_experts = 1 #@param{type:"integer"}
moe_bias_update_speed= 0.01 #@param{type:"number"}
moe_alpha= 0.001 #@param{type:"number"}
moe_inter_dim = 8 // 3 * d_model

multi_token_depth = 1 #@param{type:"integer"}
lambda_mtp = 0.1 #@param{type:"number"}



class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def create_causal_mask(seq_len, device='cpu') -> torch.Tensor:
    ones = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    return torch.triu(ones, 1)


def create_padding_mask(attention_mask):
    return (attention_mask == 0)

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

def apply_rope(x, freqs_cis):
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    n_heads = x.shape[2]
    d_head = x.shape[3]

    x_reshaped = x.view(batch_size, seq_len, n_heads, d_head // 2, 2)
    x_complex = torch.view_as_complex(x_reshaped)

    freqs_cis = freqs_cis[None, :seq_len, None, :]
    rotated_complex = x_complex * freqs_cis
    rotated_embeds = torch.view_as_real(rotated_complex).flatten(3)

    return rotated_embeds.type_as(x) 

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))  # 学習可能なスケーリングパラメータ

    def forward(self, x):
        x_normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x_normed * self.weight




class PositionalEncoding(nn.Module):
    def __init__(self, context_size, d_model):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(context_size, d_model)

        for pos in range(context_size):
            for i in range(0, d_model, 2):
                pe[pos,i]   = math.sin(pos/(10000**(i/d_model)))
                pe[pos,i+1] = math.cos(pos/(10000**(i/d_model)))

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)].detach()

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, attn_dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None, cache_size=0):
        sqrt_d_k = q.size(-1) ** 0.5
        score = torch.matmul(q, k.transpose(2, 3)) / sqrt_d_k

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)

            score = score.masked_fill(mask, torch.finfo(score.dtype).min)

        attention_weight = F.softmax(score, dim=-1)
        attention_weight = self.dropout(attention_weight)
        # Weighted value
        attention_output = torch.matmul(attention_weight, v)

        return attention_output, attention_weight

class MLA(nn.Module):
    def __init__(self, args):
        super().__init__()

        """ 引数の設定 """
        self.d_model = args.d_model
        self.n_heads = args.n_heads
        self.d_cQ = args.d_cQ  # 10%～30%の圧縮
        self.d_c = args.d_c  # 5%～10%の圧縮
        self.d_h = args.d_head
        self.d_hR = args.d_rope # ヘッドの次元の25%～50%
        self.context_size = args.context_size
        self.q_down_proj = nn.Linear(self.d_model, self.d_cQ)

        self.q_norm = RMSNorm(self.d_cQ)

        self.qc_up_proj = nn.Linear(self.d_cQ, self.n_heads * self.d_h)
        self.qr_up_proj = nn.Linear(self.d_cQ, self.n_heads * self.d_hR)

        self.kr_proj = nn.Linear(self.d_model, self.d_hR)
        self.kr_norm = RMSNorm(self.d_hR)

        self.kv_down_proj = nn.Linear(self.d_model, self.d_c)
        self.kv_norm = RMSNorm(self.d_c)
        self.kc_up_proj = nn.Linear(self.d_c, self.n_heads * self.d_h)
        self.vc_up_proj = nn.Linear(self.d_c, self.n_heads * self.d_h)

        self.kv_cache = None

        self.attention = ScaledDotProductAttention(self.d_model)
        self.output_head = nn.Linear(self.n_heads * self.d_h, self.d_model)


    def reset_kv_cache(self):
        if self.kv_cache:
            self.kv_cache.reset()

    def forward(self, h, freqs_cis, mask=None, train=False):
        batch_size, seq_len, _ = h.shape

        if self.kv_cache is None:
            self.kv_cache = KVCache(self.context_size)

        cQ = self.q_down_proj(h)
        cQ = self.q_norm(cQ)
        qR = self.qr_up_proj(cQ)
        qR = qR.reshape(batch_size, seq_len, self.n_heads, self.d_hR)
        qR = apply_rope(qR, freqs_cis)
        qC = self.qc_up_proj(cQ)
        qC = qC.reshape(batch_size, seq_len, self.n_heads, self.d_h)
        q = torch.cat([qC, qR], dim=-1)

        kR = self.kr_proj(h)
        kR = self.kr_norm(kR)
        kR = kR.reshape(batch_size, seq_len, 1, self.d_hR) # kRは1ヘッド

        kR = apply_rope(kR, freqs_cis)

        cKV = self.kv_down_proj(h)
        cKV = self.kv_norm(cKV)

        if not train:
            self.kv_cache.update(kR, cKV)
            kR, cKV = self.kv_cache.get()

        kC = self.kc_up_proj(cKV)
        vC = self.vc_up_proj(cKV)
        kC = kC.reshape(batch_size, -1, self.n_heads, self.d_h)
        vC = vC.reshape(batch_size, -1, self.n_heads, self.d_h)

        kR = kR.expand(-1, -1, kC.size(2), -1)

        k = torch.cat([kR, kC], dim=-1)

        v = vC

        q = q.transpose(1, 2) # (B, T, H, D) → (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        h, w = self.attention(q, k, v, mask)

        # 出力を整形して返す
        h = h.transpose(1, 2)
        h = h.reshape(batch_size, seq_len, self.n_heads * self.d_h) # H*Dでトークンのベクトルに変換
        h = self.output_head(h)

        output = {}
        output['hidden_state'] = h
        output['attention_weight'] = w

        return output


import torch
import torch.nn as nn

#@title MHA
class MHA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.d_model = args.d_model
        self.d_head = args.d_head

        d_model = args.d_model
        self.fc_q = nn.Linear(d_model, self.n_heads * self.d_head)
        self.fc_k = nn.Linear(d_model, self.n_heads * self.d_head)
        self.fc_v = nn.Linear(d_model, self.n_heads * self.d_head)

        dropout = 0.1
        self.attention = ScaledDotProductAttention(d_model, dropout)
        self.fc = nn.Linear(self.n_heads * self.d_head, d_model)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc.weight)

        self.context_size = args.context_size
        self.kv_cache = None
        self.registered_batch_size = None

    def reset_kv_cache(self):
        if self.kv_cache:
            self.kv_cache.reset()
        self.registered_batch_size = None

    def forward(self, x, freqs_cis, mask=None, train=False):
        N, S = x.size(0), x.size(1)
        H = self.n_heads
        D = self.d_head

        if (not train) and (self.kv_cache is None):
            self.kv_cache = KVCache(self.context_size)

        q = self.fc_q(x)
        k = self.fc_k(x)
        v = self.fc_v(x)

        q = q.view(N, S, H, D)
        k = k.view(N, S, H, D)
        v = v.view(N, S, H, D)

        freqs = freqs_cis[:S, : D // 2].to(q.device)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        if not train:
            self.kv_cache.update(k, v)
            k, v = self.kv_cache.get()  # k, v: (N, S_total, H, D)

        q = q.transpose(1, 2)  # (N, H, S_q, D)
        k = k.transpose(1, 2)  # (N, H, S_k, D)
        v = v.transpose(1, 2)  # (N, H, S_k, D)

        x, attn_weight = self.attention(q, k, v, mask=mask)

        x = x.transpose(1, 2).contiguous().view(N, -1, H * D)
        x = self.fc(x)
        x = self.dropout(x)
        output = {}
        output["hidden_state"] = x
        output["attn_weight"] = attn_weight
        return output

class Expert(nn.Module):
    def __init__(self, dim, inter_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)  # ゲート射影
        self.w2 = nn.Linear(inter_dim, dim)  # 圧縮
        self.w3 = nn.Linear(dim, inter_dim)  # 復元

        nn.init.normal_(self.w1.weight, std=0.02)
        nn.init.normal_(self.w2.weight, std=0.02)
        nn.init.normal_(self.w3.weight, std=0.02)
        nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w2.bias)
        nn.init.zeros_(self.w3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim = args.d_model
        self.n_shared_experts = args.n_shared_experts  
        self.n_routed_experts = args.n_routed_experts  
        self.top_k = args.n_activated_experts  # 選択する専門家数 (Top-K)
        self.bias_update_speed = args.moe_bias_update_speed  

        self.shared_experts = nn.ModuleList(
            [Expert(args.d_model, args.moe_inter_dim) for _ in range(self.n_shared_experts)]
        )
        self.routed_experts = nn.ModuleList(
            [Expert(args.d_model, args.moe_inter_dim) for _ in range(self.n_routed_experts)]
        )

        self.centroids = nn.Parameter(torch.randn(self.n_routed_experts, args.d_model))
        self.biases = nn.Parameter(torch.zeros(self.n_routed_experts))

        self.init_expert_freqs()

        self.output = {}

    def init_expert_freqs(self):
        self.expert_freqs = {idx: 0 for idx in range(self.n_routed_experts)}

    def update_biases(self, g_i_t, topk_indices):
        expert_load = torch.zeros(self.n_routed_experts, device=g_i_t.device)

        for k in range(self.top_k):
            expert_indices = topk_indices[:, :, k]
            weights = g_i_t[:, :, k]

            for i in range(self.n_routed_experts):
                mask = (expert_indices == i)
                expert_load[i] += weights[mask].sum()

        mean_load = expert_load.mean()

        for i in range(self.n_routed_experts):
            if expert_load[i] > mean_load:  
                self.biases.data[i] -= self.bias_update_speed  
            else:  
                self.biases.data[i] += self.bias_update_speed  

    def forward(self, input_embeddings, train=False):
        
        shared_outputs = []
        for expert in self.shared_experts:
            shared_outputs.append(expert(input_embeddings))  
        
        shared_output = torch.stack(shared_outputs, dim=2).mean(dim=2)  


        affinity_scores = torch.sigmoid(input_embeddings @ self.centroids.T)
        expanded_biases = self.biases.unsqueeze(0).unsqueeze(0)
        biased_scores = affinity_scores + expanded_biases

        topk_values, topk_indices = torch.topk(biased_scores, self.top_k, dim=-1)
        selected_affinity_scores = torch.gather(affinity_scores, -1, topk_indices)
        gating_weights = F.softmax(selected_affinity_scores, dim=-1)

        batch_size, seq_len = input_embeddings.shape[:2]

        flat_topk_indices = topk_indices.view(-1)
        flat_gating_weights = gating_weights.view(-1)

        expanded_input_embeddings = input_embeddings.unsqueeze(2).expand(-1, -1, self.top_k, -1)
        flat_input_embeddings = expanded_input_embeddings.contiguous().view(-1, self.dim)

        used_experts = torch.unique(flat_topk_indices)
        expert_outputs = {}
        for expert_idx in used_experts:
            if not train:
                self.expert_freqs[expert_idx.item()] += 1
            mask = (flat_topk_indices == expert_idx)
            if mask.any():
                expert_outputs[int(expert_idx)] = self.routed_experts[expert_idx](flat_input_embeddings[mask])

        routed_output = torch.zeros_like(flat_input_embeddings)
        for expert_idx in used_experts:
            mask = (flat_topk_indices == expert_idx)
            if mask.any():
                corresponding_weights = flat_gating_weights[mask]
                weighted_expert_output = expert_outputs[int(expert_idx)] * corresponding_weights.unsqueeze(-1)
                routed_output[mask] = weighted_expert_output

        reshaped_routed_output = routed_output.view(batch_size, seq_len, self.top_k, -1)
        routed_output = reshaped_routed_output.sum(dim=2)

        output_embeddings = input_embeddings + shared_output + routed_output

        if train:
            self.update_biases(gating_weights, topk_indices)

        self.output['hidden_state'] = output_embeddings
        self.output['affinity_scores'] = affinity_scores

        return self.output

def calculate_balance_loss(affinity_scores, args):
    s_i_t = affinity_scores  
    batch_size, seq_len, num_routed_experts = s_i_t.shape
    top_k = args.n_activated_experts

    topk_values, topk_indices = torch.topk(s_i_t, top_k, dim=2)  
    topk_mask = torch.zeros_like(s_i_t, dtype=torch.bool)
    topk_mask.scatter_(2, topk_indices, True)

    total_tokens = batch_size * seq_len
    f_i = (topk_mask.sum(dim=(0, 1)).float() * num_routed_experts / (top_k * total_tokens))  

    denom = s_i_t.sum(dim=2, keepdim=True) + 1e-9  
    s_i_t_normalized = s_i_t / denom  

    P_i = s_i_t_normalized.mean(dim=(0, 1))  # [num_routed_experts]

    balance_loss = args.moe_alpha * (f_i * P_i).sum()

    return balance_loss


import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert_2(nn.Module):
    def __init__(self, d_model, intermediate_dim):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, d_model, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class MoE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim = args.d_model
        self.n_shared_experts = args.n_shared_experts  # 共有専門家の数
        self.n_routed_experts = args.n_routed_experts  # ルーティング専門家の数
        self.top_k = args.n_activated_experts  # 選択する専門家数 (Top-K)
        self.bias_update_speed = args.moe_bias_update_speed  # バイアス更新速度 γ
        self.alpha = getattr(args, 'moe_alpha', 0.001)  # バランシングロスの重み α

        # 専門家ネットワーク
        self.shared_experts = nn.ModuleList(
            [Expert(args.d_model, args.moe_inter_dim) for _ in range(self.n_shared_experts)]
        )
        self.routed_experts = nn.ModuleList(
            [Expert(args.d_model, args.moe_inter_dim) for _ in range(self.n_routed_experts)]
        )

        # ルーティングパラメータ
        self.centroids = nn.Parameter(torch.randn(self.n_routed_experts, args.d_model))
        self.biases = nn.Parameter(torch.zeros(self.n_routed_experts))

        self.init_expert_freqs()
        self.output = {}

    def init_expert_freqs(self):
        self.expert_freqs = {idx: 0 for idx in range(self.n_routed_experts)}

    def update_biases(self, gating_weights, topk_indices):
        expert_load = torch.zeros(self.n_routed_experts, device=gating_weights.device)

        for k in range(self.top_k):
            expert_indices = topk_indices[:, :, k]  # [batch, seq]
            weights = gating_weights[:, :, k]       # [batch, seq]

            for i in range(self.n_routed_experts):
                mask = (expert_indices == i)
                expert_load[i] += weights[mask].sum()

        mean_load = expert_load.mean()

        for i in range(self.n_routed_experts):
            if expert_load[i] > mean_load:  # 過負荷の場合
                self.biases.data[i] -= self.bias_update_speed
            else:  # 低負荷/未使用の場合
                self.biases.data[i] += self.bias_update_speed

    def compute_balance_loss(self, affinity_scores, topk_indices):
        batch_size, seq_len, num_experts = affinity_scores.shape
        total_tokens = batch_size * seq_len

        f_i = torch.zeros(num_experts, device=affinity_scores.device)
        for i in range(num_experts):
            mask = (topk_indices == i).any(dim=-1)  # [batch, seq]
            f_i[i] = mask.sum().float() / total_tokens

        normalized_scores = F.softmax(affinity_scores, dim=-1)
        P_i = normalized_scores.mean(dim=(0, 1))  # [num_experts]

        balance_loss = self.alpha * (f_i * P_i).sum()

        return balance_loss

    def forward(self, input_embeddings, train=False):
        batch_size, seq_len = input_embeddings.shape[:2]

        shared_outputs = []
        for expert in self.shared_experts:
            shared_outputs.append(expert(input_embeddings))
        shared_output = torch.stack(shared_outputs, dim=2).mean(dim=2)

        affinity_scores = torch.sigmoid(input_embeddings @ self.centroids.T)

        expanded_biases = self.biases.unsqueeze(0).unsqueeze(0)  # [1, 1, num_experts]
        biased_scores = affinity_scores + expanded_biases

        topk_values, topk_indices = torch.topk(biased_scores, self.top_k, dim=-1)

        selected_affinity_scores = torch.gather(affinity_scores, -1, topk_indices)
        gating_weights = F.softmax(selected_affinity_scores, dim=-1)
        flat_topk_indices = topk_indices.view(-1)
        flat_gating_weights = gating_weights.view(-1)

        expanded_input_embeddings = input_embeddings.unsqueeze(2).expand(-1, -1, self.top_k, -1)
        flat_input_embeddings = expanded_input_embeddings.contiguous().view(-1, self.dim)

        used_experts = torch.unique(flat_topk_indices)
        expert_outputs = {}

        for expert_idx in used_experts:
            if not train:
                self.expert_freqs[expert_idx.item()] += 1

            mask = (flat_topk_indices == expert_idx)
            if mask.any():
                expert_outputs[int(expert_idx)] = self.routed_experts[expert_idx](flat_input_embeddings[mask])

        # 各専門家の出力を重み付きで集約
        routed_output = torch.zeros_like(flat_input_embeddings)
        for expert_idx in used_experts:
            mask = (flat_topk_indices == expert_idx)
            if mask.any():
                corresponding_weights = flat_gating_weights[mask]
                weighted_expert_output = expert_outputs[int(expert_idx)] * corresponding_weights.unsqueeze(-1)
                routed_output[mask] = weighted_expert_output

        reshaped_routed_output = routed_output.view(batch_size, seq_len, self.top_k, -1)
        routed_output = reshaped_routed_output.sum(dim=2)

        output_embeddings = input_embeddings + shared_output + routed_output

        balance_loss = None
        if train:
            self.update_biases(gating_weights, topk_indices)

            balance_loss = self.compute_balance_loss(affinity_scores, topk_indices)

        self.output = {
            'hidden_state': output_embeddings,
            'affinity_scores': affinity_scores,
            'gating_weights': gating_weights,
            'topk_indices': topk_indices
        }

        if balance_loss is not None:
            self.output['balance_loss'] = balance_loss

        return self.output

    def get_expert_usage_stats(self):
        """専門家使用統計の取得"""
        total_usage = sum(self.expert_freqs.values())
        if total_usage == 0:
            return {i: 0.0 for i in range(self.n_routed_experts)}

        return {i: freq / total_usage for i, freq in self.expert_freqs.items()}

    def reset_expert_freqs(self):
        """専門家使用頻度のリセット"""
        self.expert_freqs = {idx: 0 for idx in range(self.n_routed_experts)}


class FeedForward(nn.Module):
    def __init__(self, args, dropout=0.1):
        # super(FeedForward, self).__init__()
        super().__init__()

        d_model = args.d_model

        self.fc1 = nn.Linear(d_model, d_model * 4 )
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model * 4, d_model)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x, train=False):
        h = self.fc1(x)
        h = F.gelu(h)
        h = self.fc2(h)
        h = self.dropout(h)

        # balance_loss = torch.tensor(0.0, requires_grad=False, device=x.device) # Set requires_grad=False

        output = {}
        output['hidden_state'] = h
        output['affinity_scores'] = None
        return output

# @title LayerNorm
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        nn.init.normal_(self.norm.weight, mean=0, std=0.02)

    def forward(self, x):
        return self.norm(x)


# @title TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(self, args, layer_id=0):
        super().__init__()

        if args.d_c is None:
            self.attention = MultiHeadAttention(args)
        else:
            self.attention = MLA(args)

        # self.feed_forward = FeedForward(args)
        self.feed_forward = MoE(args)
        # self.attn_norm = LayerNorm(args.d_model)
        self.attn_norm = RMSNorm(args.d_model, eps=args.norm_eps)
        # self.ffn_norm = LayerNorm(args.d_model)
        self.ffn_norm = RMSNorm(args.d_model, eps=args.norm_eps)
        self.layer_id = layer_id

    def forward(self, x, start_pos, freqs_cis=None, mask=None, train=False): # Added freqs_cis=None

        h1 = self.attn_norm(x)
        output = self.attention(h1, freqs_cis, mask, train) # 出力形式を output 辞書に変更する?
        h1 = output['hidden_state']
        w = output['attention_weight']

        h1 = h1 + x # 残渣結合

        h2 = self.ffn_norm(h1)
        feed_forward_output = self.feed_forward(h2, train)
        h2 = feed_forward_output['hidden_state']

        h = h2 + h1 # 残渣結合

        output = {}
        output['hidden_state'] = h
        output['affinity_scores'] = feed_forward_output['affinity_scores']
        output['attention_weight'] = w
        return output


# @title MainModel
class MainModel(nn.Module):
    def __init__(self, embedding, output_head, args):
        super().__init__()

        self.context_size = args.context_size
        self.embedding = embedding # 共通重み


        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(args, layer_id))
        self.output_norm = RMSNorm(args.d_model, eps=args.norm_eps)
        self.output_head = output_head # 共通重み

    def reset_kv_cache(self):
        for layer in self.layers:
            layer.attention.reset_kv_cache()

    def forward(self, input_ids, start_pos, freqs_cis, mask=None, train=False):

        h = self.embedding(input_ids)
        balance_losses = torch.tensor(0.0, device=input_ids.device)  # requires_grad削除

        for layer in self.layers:  # Changed layer_id to layer
            transformer_block_output = layer(h, start_pos, freqs_cis, mask, train=train) # Pass train and unpack outputs

            h = transformer_block_output['hidden_state']

        h = self.output_norm(h)

        logits = self.output_head(h)

        main_output = {}
        main_output['logits'] = logits
        main_output['hidden_state'] = h # 埋め込みベクトル（隠れ状態）
        main_output['affinity_scores'] = transformer_block_output['affinity_scores']
        main_output['attention_weight'] = transformer_block_output['attention_weight']

        return main_output

class MTPModule(nn.Module):
    def __init__(self, embedding, output_head, args):
        super().__init__()
        self.embedding = embedding # 共通重み
        self.norm_pres = RMSNorm(args.d_model, eps=args.norm_eps)
        self.norm_prev = RMSNorm(args.d_model, eps=args.norm_eps)
        self.projection = nn.Linear(args.d_model * 2, args.d_model) # 単純な線形変換
        self.transformer_block = TransformerBlock(args)
        self.output_norm = RMSNorm(args.d_model, eps=args.norm_eps)
        self.output_head = output_head # 共通重み

    def forward(self, input_ids, h_prev, start_pos, freqs_cis, mask=None):
        debug('h_prev', h_prev)

        h_curr = self.embedding(input_ids) # Current sequences

        h_curr = self.norm_pres(h_curr)
        h_prev = self.norm_prev(h_prev) # 直前のモデルが出力した隠れ状態

        concatenation = torch.cat([h_curr, h_prev], dim=-1)
        h = self.projection(concatenation)

        transformer_block_output = self.transformer_block(h, start_pos, freqs_cis, mask, train=True)

        h = transformer_block_output['hidden_state']

        h = self.output_norm(h) # 補足（論文にはない）

        logits = self.output_head(h)

        mtp_output = {}
        mtp_output['logits'] = logits
        mtp_output['hidden_state'] = h
        mtp_output['attention_weight'] = transformer_block_output['attention_weight']
        mtp_output['affinity_scores'] = transformer_block_output['affinity_scores']

        return mtp_output


def 参考_calculate_main_logits(main_model, tokens, start_pos, freqs_cis, args):
    input_ids = tokens[:, :args.context_size]
    seq_len = input_ids.size(1)  # input_idsが context_size未満である場合もあるため

    input_freqs_cis = freqs_cis[start_pos:start_pos + seq_len]

    mask = create_causal_mask(seq_len)
    mask = mask.to(device)

    main_model.reset_kv_cache()

    main_output = main_model(input_ids, start_pos, input_freqs_cis, mask, train=False)
    main_logits = main_output['logits']

    return main_logits


class DeepSeekCode(nn.Module):
    def __init__(self, args, device='cpu'):
        super().__init__()

        self.args = args
        self.device = device
        self.vocab_size = args.vocab_size
        self.context_size = args.context_size
        self.lambda_mtp = args.lambda_mtp

        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.embedding.to(device)

        self.output_head = nn.Linear(args.d_model, args.vocab_size, bias=True)
        self.output_head.to(device)

        self.main_model = MainModel(self.embedding, self.output_head, args)
        self.main_model.to(device)

        self.mtp_modules = nn.ModuleList() # 計算グラフでつなげるためにModuleListを利用
        for _ in range(args.multi_token_depth):
            mtp_module = MTPModule(self.embedding, self.output_head, args)
            self.mtp_modules.append(mtp_module)
        self.mtp_modules.to(device)

        self.freqs_cis = precompute_freqs_cis(args, device)

        self.criterion = nn.CrossEntropyLoss()

        self.reset_kv_cache = self.main_model.reset_kv_cache

    def parameters(self, recurse: bool = True):
        params = list(self.main_model.parameters())
        for mtp_module in self.mtp_modules:
            params.extend(list(mtp_module.parameters()))
        return params

class DeepSeekCode(DeepSeekCode):
    def pretrain(self, source):
        output = self._calculate_main_loss(source)
        main_loss = output['main_loss']
        main_balance_loss = output['main_balance_loss']
        hidden_state = output['hidden_state']

        output = self._calculate_mtp_loss(source, hidden_state)
        mtp_losses = output['mtp_losses']
        mtp_balance_losses = output['mtp_balance_losses']
        total_loss = main_loss + self.lambda_mtp * mtp_losses + main_balance_loss + self.lambda_mtp * mtp_balance_losses
        return total_loss

class DeepSeekCode(DeepSeekCode):
    def _calculate_main_loss(self, source):
        main_input_ids = source[:, :self.context_size]
        main_target_ids = source[:, 1:self.context_size + 1]
        main_freqs_cis = self.freqs_cis[:self.context_size]

        mask = create_causal_mask(self.context_size).to(source.device)

        self.main_model.reset_kv_cache()

        main_output = self.main_model(main_input_ids, 0, main_freqs_cis, mask, train=True)
        main_logits = main_output['logits']
        hidden_state = main_output['hidden_state']
        main_affinity_scores = main_output['affinity_scores']

        predicted = main_logits.contiguous().view(-1, self.vocab_size)
        target = main_target_ids.contiguous().view(-1)

        debug('predicted', predicted.shape)
        debug('target',target.shape)

        main_loss = self.criterion(predicted, target)
        main_balance_loss = calculate_balance_loss(main_affinity_scores, self.args)

        output = {}
        output['main_loss'] = main_loss
        output['main_balance_loss'] = main_balance_loss
        output['hidden_state'] = hidden_state
        return output

class DeepSeekCode(DeepSeekCode):
    def _calculate_mtp_loss(self, source, hidden_state):
        mtp_losses = 0
        mtp_balance_losses = 0
        for mtp_offset, mtp_module in enumerate(self.mtp_modules):
            mtp_input_ids = source[:, mtp_offset + 1: self.context_size + mtp_offset + 1]
            mtp_target_ids = source[:, mtp_offset + 2: self.context_size + mtp_offset + 2]
            mtp_freqs_cis = self.freqs_cis[mtp_offset + 1: self.context_size + mtp_offset + 1]

            mask = create_causal_mask(self.context_size).to(source.device)

            mtp_output = mtp_module(mtp_input_ids, hidden_state, 0, mtp_freqs_cis)
            logits = mtp_output['logits']
            hidden_state = mtp_output['hidden_state']
            affinity_scores = mtp_output['affinity_scores']

            predicted = logits.contiguous().view(-1, self.vocab_size)
            target = mtp_target_ids.contiguous().view(-1)

            mtp_losses += self.criterion(predicted, target)
            mtp_balance_losses += calculate_balance_loss(affinity_scores, self.args)

        output = {}
        output['mtp_losses'] = mtp_losses
        output['mtp_balance_losses'] = mtp_balance_losses

        return output

class DeepSeekCode(DeepSeekCode):
    def compute_log_prob(self, input_ids, mask=None, train=False):
        start_pos = 0
        self.reset_kv_cache()

        if mask is None:
            mask = create_causal_mask(input_ids.size(1)).to(input_ids.device)
        main_output = self.main_model(input_ids, start_pos, self.freqs_cis, mask=mask, train=train)

        log_probs = F.log_softmax(main_output['logits'], dim=-1)

        return log_probs

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
            if input_ids.dim() == 1:

                input_ids = input_ids.unsqueeze(0)

            return input_ids


        def top_k_sampling(logits, top_k, temperature):
            logits = logits / temperature

            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)

            probs = torch.nn.functional.softmax(top_k_logits, dim=-1)

            sampled_index = torch.multinomial(probs, num_samples=1)

            return top_k_indices.gather(dim=-1, index=sampled_index)

        device = input_ids.device

        self.main_model.eval()

        context_size = self.main_model.context_size

        input_ids = adjust_bactch_dim(input_ids)

        input_ids = input_ids[:, :context_size]

        generated_ids = input_ids[:, 0:0]

        if max_cache_size is None:
            max_cache_size = context_size

        buffered_input_ids = reset_buffered_input_ids()

        self.main_model.reset_kv_cache()

        start_pos = 0

        for _ in range(max_new_tokens):

            seq_len = input_ids.size(1)

            input_freqs_cis = self.freqs_cis[start_pos: start_pos + seq_len]
            input_freqs_cis = input_freqs_cis.to(device)
            mask = create_causal_mask(seq_len).to(device)

            output = self.main_model(input_ids, start_pos, input_freqs_cis, mask=mask, train=False)

            last_logits = output['logits'][:,-1,:]

            if top_k == 1:
                index = last_logits.argmax(dim=1) # fix
            else:
                if 0:
                    _topk_values, topk_indices = torch.topk(last_logits, top_k)
                    random_indices_in_topk = torch.randint(0, top_k, (input_ids.size(0),), dtype=torch.long) #fix
                    random_indices_in_topk = random_indices_in_topk.to(device)
                    index = torch.gather(topk_indices, 1, random_indices_in_topk.unsqueeze(1)) #fix
                else:
                    index = top_k_sampling(last_logits, top_k, temperature)


            index = index.view(-1,1)
            input_ids = index.to(device)

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
                self.main_model.reset_kv_cache()
                input_ids = buffered_input_ids[:, -context_size//2:]
                buffered_input_ids = reset_buffered_input_ids().to(device)
                start_pos = 0

        if tokenizer is None:
            return generated_ids
        else:
            print()
            return None

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

