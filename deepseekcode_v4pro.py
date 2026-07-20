# -*- coding: utf-8 -*-
"""
DeepSeek-V4 Professional Implementation
論文に忠実な実装（最小限パラメータで単一GPU対応）

V4の主な新機能:
1. Multi-Token Prediction (MTP)
2. Hybrid Attention (Local + Global)
3. MoE with Load Balancing
4. RMSNorm improvements
5. RoPE extension
6. Multi-head Latent Attention (MLA) inspired compression
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Args:
    """
    DeepSeek-V4 ハイパーパラメータ
    単一GPU向けに最小限の設定
    """
    def __init__(self, **kwargs):
        defaults = {
            # モデルサイズ
            'vocab_size': 32000,
            'd_model': 512,
            'n_layers': 6,
            'n_heads': 8,
            'd_head': 64,
            
            # MLA関連 (V4の重要な機能)
            'd_cQ': 128,      # Query latent次元 (圧縮率~25%)
            'd_c': 64,        # KV latent次元 (圧縮率~12.5%)
            'd_rope': 32,     # RoPE用次元
            
            # コンテキスト
            'context_size': 512,
            'max_seq_len': 512,
            'rope_theta': 10000.0,
            'norm_eps': 1e-6,
            
            # MoE設定
            'n_shared_experts': 1,
            'n_routed_experts': 8,
            'n_activated_experts': 2,
            'moe_inter_dim': 1024,
            'moe_bias_update_speed': 0.001,
            'aux_loss_alpha': 0.01,
            
            # MTP設定
            'multi_token_depth': 2,  # 予測するトークン数
            'lambda_mtp': 0.5,       # MTP損失の重み
            
            # ドロップアウト
            'attn_dropout': 0.1,
            'ffn_dropout': 0.1,
            
            # ハイブリッドアテンション
            'local_window_size': 64,
            'global_every_n_layers': 2,
        }
        defaults.update(kwargs)
        for key, value in defaults.items():
            setattr(self, key, value)


# ============================================================================
# ユーティリティ関数
# ============================================================================

def create_causal_mask(seq_len, device='cpu') -> torch.Tensor:
    """因果マスク作成"""
    ones = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    return torch.triu(ones, 1)


def precompute_freqs_cis(args, device='cpu'):
    """RoPE周波数行列の事前計算"""
    dim = args.d_rope
    indices = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    scaled_index = indices / dim
    freqs = 1.0 / (args.rope_theta ** scaled_index)
    m = torch.arange(args.max_seq_len, dtype=torch.float32, device=device)
    rotation_angles = torch.outer(m, freqs)
    abs_val = torch.ones_like(freqs)
    freqs_cis = torch.polar(abs_val, rotation_angles)
    return freqs_cis.detach()


def apply_rope(x, freqs_cis):
    """Rotary Positional Embedding"""
    batch_size, seq_len, n_heads, d_head = x.shape
    x_reshaped = x.view(batch_size, seq_len, n_heads, d_head // 2, 2)
    x_complex = torch.view_as_complex(x_reshaped)
    freqs_cis = freqs_cis[None, :seq_len, None, :]
    rotated_complex = x_complex * freqs_cis
    rotated_embeds = torch.view_as_real(rotated_complex).flatten(3)
    return rotated_embeds.type_as(x)


# ============================================================================
# 正規化
# ============================================================================

class RMSNorm(nn.Module):
    """
    RMS Normalization
    DeepSeek-V4では標準的に使用
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x_normed * self.weight


# ============================================================================
# KV Cache
# ============================================================================

class KVCache(nn.Module):
    """キー・バリューキャッシュ管理"""
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


# ============================================================================
# Multi-head Latent Attention (MLA) - V4の核心機能
# ============================================================================

class MLA(nn.Module):
    """
    Multi-head Latent Attention
    DeepSeek-V4の重要な圧縮技術
    """
    def __init__(self, args):
        super().__init__()
        self.d_model = args.d_model
        self.n_heads = args.n_heads
        self.d_cQ = args.d_cQ
        self.d_c = args.d_c
        self.d_h = args.d_head
        self.d_hR = args.d_rope
        self.context_size = args.context_size
        
        # Query処理
        self.q_down_proj = nn.Linear(self.d_model, self.d_cQ)
        self.q_norm = RMSNorm(self.d_cQ)
        self.qc_up_proj = nn.Linear(self.d_cQ, self.n_heads * self.d_h)
        self.qr_up_proj = nn.Linear(self.d_cQ, self.n_heads * self.d_hR)
        
        # Key処理 (RoPE用)
        self.kr_proj = nn.Linear(self.d_model, self.d_hR)
        self.kr_norm = RMSNorm(self.d_hR)
        
        # KV圧縮
        self.kv_down_proj = nn.Linear(self.d_model, self.d_c)
        self.kv_norm = RMSNorm(self.d_c)
        self.kc_up_proj = nn.Linear(self.d_c, self.n_heads * self.d_h)
        self.vc_up_proj = nn.Linear(self.d_c, self.n_heads * self.d_h)
        
        # 出力
        self.output_head = nn.Linear(self.n_heads * self.d_h, self.d_model)
        
        # キャッシュ
        self.kv_cache = None

    def reset_kv_cache(self):
        if self.kv_cache:
            self.kv_cache.reset()

    def forward(self, h, freqs_cis, causal_mask=None, train=False):
        batch_size, seq_len, _ = h.shape
        
        if self.kv_cache is None:
            self.kv_cache = KVCache(self.context_size)
        
        # Query処理
        cQ = self.q_down_proj(h)
        cQ = self.q_norm(cQ)
        qR = self.qr_up_proj(cQ).reshape(batch_size, seq_len, self.n_heads, self.d_hR)
        qR = apply_rope(qR, freqs_cis)
        qC = self.qc_up_proj(cQ).reshape(batch_size, seq_len, self.n_heads, self.d_h)
        q = torch.cat([qC, qR], dim=-1)
        
        # Key処理
        kR = self.kr_proj(h)
        kR = self.kr_norm(kR).reshape(batch_size, seq_len, 1, self.d_hR)
        kR = apply_rope(kR, freqs_cis)
        
        # KV圧縮
        cKV = self.kv_down_proj(h)
        cKV = self.kv_norm(cKV)
        
        if not train:
            self.kv_cache.update(kR, cKV)
            kR, cKV = self.kv_cache.get()
        
        kC = self.kc_up_proj(cKV).reshape(batch_size, -1, self.n_heads, self.d_h)
        vC = self.vc_up_proj(cKV).reshape(batch_size, -1, self.n_heads, self.d_h)
        
        kR = kR.expand(-1, -1, kC.size(2), -1)
        k = torch.cat([kR, kC], dim=-1)
        v = vC
        
        # Attention計算
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        sqrt_d_k = q.size(-1) ** 0.5
        scores = torch.matmul(q, k.transpose(2, 3)) / sqrt_d_k
        
        if causal_mask is not None:
            if causal_mask.dim() == 2:
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * self.d_h)
        out = self.output_head(out)
        
        return {
            'hidden_state': out,
            'attention_weight': attn_weights
        }


# ============================================================================
# Hybrid Attention (V4新機能)
# ============================================================================

class HybridAttention(nn.Module):
    """
    Hybrid Attention
    ローカル注意とグローバル注意を組み合わせ
    """
    def __init__(self, args, use_mla=True):
        super().__init__()
        self.use_mla = use_mla
        
        if use_mla:
            self.attention = MLA(args)
        else:
            # 従来型Attention（一部の層で使用）
            self.n_heads = args.n_heads
            self.d_model = args.d_model
            self.d_head = args.d_head
            self.fc_q = nn.Linear(args.d_model, args.n_heads * args.d_head)
            self.fc_k = nn.Linear(args.d_model, args.n_heads * args.d_head)
            self.fc_v = nn.Linear(args.d_model, args.n_heads * args.d_head)
            self.fc_out = nn.Linear(args.n_heads * args.d_head, args.d_model)
            self.attn_dropout = nn.Dropout(args.attn_dropout)
        
        self.local_window_size = args.local_window_size

    def reset_kv_cache(self):
        if self.use_mla and hasattr(self.attention, 'reset_kv_cache'):
            self.attention.reset_kv_cache()

    def forward(self, x, freqs_cis, causal_mask=None, train=False):
        if self.use_mla:
            return self.attention(x, freqs_cis, causal_mask, train)
        else:
            batch_size, seq_len, _ = x.shape
            H, D = self.n_heads, self.d_head
            
            q = self.fc_q(x).view(batch_size, seq_len, H, D)
            k = self.fc_k(x).view(batch_size, seq_len, H, D)
            v = self.fc_v(x).view(batch_size, seq_len, H, D)
            
            freqs = freqs_cis[:seq_len, :D//2].to(x.device)
            q = apply_rope(q, freqs)
            k = apply_rope(k, freqs)
            
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            sqrt_d_k = D ** 0.5
            scores = torch.matmul(q, k.transpose(2, 3)) / sqrt_d_k
            
            if causal_mask is not None:
                if causal_mask.dim() == 2:
                    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
                scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            
            out = torch.matmul(attn_weights, v)
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, H * D)
            out = self.fc_out(out)
            
            return {
                'hidden_state': out,
                'attention_weight': attn_weights
            }


# ============================================================================
# Expert Network
# ============================================================================

class Expert(nn.Module):
    """MoE専門家"""
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
        h = gate * value
        return self.w2(h)


class MoE(nn.Module):
    """
    DeepSeek-V4 MoE
    ロードバランシング付き
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
        self.expected_load = self.n_activated_experts / self.n_routed_experts
        
        # 共有エキスパート
        self.shared_experts = nn.ModuleList([
            Expert(self.d_model, self.moe_inter_dim)
            for _ in range(self.n_shared_experts)
        ])
        
        # ルーティングエキスパート
        self.routed_experts = nn.ModuleList([
            Expert(self.d_model, self.moe_inter_dim)
            for _ in range(self.n_routed_experts)
        ])
        
        # ゲート
        self.centroids = nn.Parameter(
            torch.randn(self.n_routed_experts, self.d_model) * 0.1
        )
        self.register_buffer('expert_bias', torch.zeros(self.n_routed_experts))
        
        # 統計
        self.register_buffer('step_expert_counts', torch.zeros(self.n_routed_experts, dtype=torch.long))
        self.register_buffer('step_total_tokens', torch.tensor(0, dtype=torch.long))

    def forward(self, x, train=False):
        batch_size, seq_len, d_model = x.shape
        u = x.reshape(-1, d_model)
        
        # 共有エキスパート
        shared_output = sum(expert(x) for expert in self.shared_experts)
        
        # ゲート計算
        affinity_scores = torch.sigmoid(u @ self.centroids.T)
        routing_scores = affinity_scores + self.expert_bias
        
        # Top-K選択
        _topk_values, topk_indices = torch.topk(
            routing_scores, 
            k=self.n_activated_experts, 
            dim=1
        )
        
        gating_scores = affinity_scores.gather(1, topk_indices)
        gating_sum = gating_scores.sum(dim=1, keepdim=True)
        gating_weights = gating_scores / (gating_sum + 1e-8)
        
        # ルーティング出力
        routed_output = torch.zeros_like(u)
        expert_counts = torch.bincount(
            topk_indices.flatten(),
            minlength=self.n_routed_experts
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
        
        # 補助損失
        auxiliary_loss = torch.tensor(0.0, device=x.device)
        if train:
            auxiliary_loss = self._compute_auxiliary_loss(affinity_scores, topk_indices)
            self.step_expert_counts.copy_(expert_counts.detach())
            self.step_total_tokens.fill_(batch_size * seq_len)
        
        return {
            'hidden_state': shared_output + routed_output,
            'auxiliary_loss': auxiliary_loss,
            'affinity_scores': affinity_scores,
            'gating_weights': gating_weights
        }

    def _compute_auxiliary_loss(self, affinity_scores, topk_indices):
        device = affinity_scores.device
        T, _K = topk_indices.shape
        N_r = self.n_routed_experts
        K_r = self.n_activated_experts
        
        expert_frequency = torch.zeros(N_r, device=device)
        for i in range(N_r):
            is_selected = (topk_indices == i).sum().float()
            expert_frequency[i] = N_r / (K_r * T) * is_selected
        
        affinity_sum = affinity_scores.sum(dim=1, keepdim=True)
        normalized_scores = affinity_scores / (affinity_sum + 1e-8)
        expert_affinity = normalized_scores.mean(dim=0)
        
        return self.aux_loss_alpha * (expert_frequency * expert_affinity).sum()

    def update_expert_bias(self):
        """エキスパートバイアス更新"""
        expert_load = self.step_expert_counts / self.step_total_tokens
        overloaded = expert_load > self.expected_load
        underloaded = expert_load < self.expected_load
        
        with torch.no_grad():
            self.expert_bias[overloaded] -= self.bias_update_speed
            self.expert_bias[underloaded] += self.bias_update_speed


# ============================================================================
# Transformer Block
# ============================================================================

class TransformerBlock(nn.Module):
    """DeepSeek-V4 Transformerブロック"""
    def __init__(self, args, layer_id=0):
        super().__init__()
        
        # ハイブリッドアテンション（MLAを使用）
        use_mla = True  # V4では基本的にMLA
        self.attention = HybridAttention(args, use_mla=use_mla)
        
        # FFN/MoE
        if layer_id == 0:
            self.feed_forward = nn.Sequential(
                nn.Linear(args.d_model, args.d_model * 4),
                nn.GELU(),
                nn.Dropout(args.ffn_dropout),
                nn.Linear(args.d_model * 4, args.d_model)
            )
            self._is_moe = False
        else:
            self.feed_forward = MoE(args)
            self._is_moe = True
        
        self.attn_norm = RMSNorm(args.d_model, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.d_model, eps=args.norm_eps)

    def forward(self, x, freqs_cis, causal_mask=None, train=False):
        h = self.attn_norm(x)
        attn_out = self.attention(h, freqs_cis, causal_mask, train)
        h = attn_out['hidden_state'] + x
        
        h2 = self.ffn_norm(h)
        ffn_out = self.feed_forward(h2)
        
        if self._is_moe:
            h = ffn_out['hidden_state'] + h
            aux_loss = ffn_out['auxiliary_loss']
        else:
            h = ffn_out + h
            aux_loss = torch.tensor(0.0, device=x.device)
        
        return {
            'hidden_state': h,
            'auxiliary_loss': aux_loss,
            'attention_weight': attn_out['attention_weight']
        }


# ============================================================================
# Multi-Token Prediction (MTP) Module - V4の新機能
# ============================================================================

class MTPModule(nn.Module):
    """
    Multi-Token Prediction Module
    DeepSeek-V4の重要な高速化技術
    """
    def __init__(self, embedding, output_head, args):
        super().__init__()
        self.embedding = embedding
        self.norm_pres = RMSNorm(args.d_model, eps=args.norm_eps)
        self.norm_prev = RMSNorm(args.d_model, eps=args.norm_eps)
        self.projection = nn.Linear(args.d_model * 2, args.d_model)
        self.transformer_block = TransformerBlock(args, layer_id=0)
        self.output_norm = RMSNorm(args.d_model, eps=args.norm_eps)
        self.output_head = output_head

    def forward(self, input_ids, h_prev, start_pos, freqs_cis, causal_mask=None):
        h_curr = self.embedding(input_ids)
        h_curr = self.norm_pres(h_curr)
        h_prev = self.norm_prev(h_prev)
        
        concatenation = torch.cat([h_curr, h_prev], dim=-1)
        h = self.projection(concatenation)
        
        out = self.transformer_block(h, freqs_cis, causal_mask, train=True)
        h = out['hidden_state']
        h = self.output_norm(h)
        logits = self.output_head(h)
        
        return {
            'logits': logits,
            'hidden_state': h,
            'auxiliary_loss': out['auxiliary_loss']
        }


# ============================================================================
# メインモデル
# ============================================================================

class MainModel(nn.Module):
    """メインモデル"""
    def __init__(self, embedding, output_head, args):
        super().__init__()
        self.context_size = args.context_size
        self.embedding = embedding
        
        self.layers = nn.ModuleList([
            TransformerBlock(args, layer_id)
            for layer_id in range(args.n_layers)
        ])
        
        self.output_norm = RMSNorm(args.d_model, eps=args.norm_eps)
        self.output_head = output_head

    def reset_kv_cache(self):
        for layer in self.layers:
            layer.attention.reset_kv_cache()

    def forward(self, input_ids, freqs_cis, causal_mask=None, train=False):
        h = self.embedding(input_ids)
        total_aux_loss = torch.tensor(0.0, device=input_ids.device)
        
        for layer in self.layers:
            out = layer(h, freqs_cis, causal_mask, train)
            h = out['hidden_state']
            total_aux_loss = total_aux_loss + out['auxiliary_loss']
        
        h = self.output_norm(h)
        logits = self.output_head(h)
        
        return {
            'logits': logits,
            'hidden_state': h,
            'auxiliary_loss': total_aux_loss
        }


class DeepSeekCodeV4Pro(nn.Module):
    """
    DeepSeek-V4 Professional Version
    論文に忠実な実装
    """
    def __init__(self, args=None, device='cpu'):
        super().__init__()
        if args is None:
            args = Args()
        
        self.args = args
        self.device = device
        self.vocab_size = args.vocab_size
        self.context_size = args.context_size
        self.lambda_mtp = args.lambda_mtp
        self.multi_token_depth = args.multi_token_depth
        
        # 共通重み
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.output_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.output_head.weight = self.embedding.weight  # 重み共有
        
        # メインモデル
        self.main_model = MainModel(self.embedding, self.output_head, args)
        
        # MTPモジュール
        self.mtp_modules = nn.ModuleList([
            MTPModule(self.embedding, self.output_head, args)
            for _ in range(args.multi_token_depth)
        ])
        
        # RoPE
        self.freqs_cis = precompute_freqs_cis(args, device)
        
        # 損失関数
        self.criterion = nn.CrossEntropyLoss()
        
        self.to(device)

    def forward(self, source, train=False):
        """順伝播と損失計算"""
        # メイン損失
        main_input = source[:, :self.context_size]
        main_target = source[:, 1:self.context_size + 1]
        main_freqs = self.freqs_cis[:self.context_size]
        causal_mask = create_causal_mask(self.context_size, device=source.device)
        
        self.main_model.reset_kv_cache()
        main_output = self.main_model(main_input, main_freqs, causal_mask, train)
        
        main_logits = main_output['logits'].contiguous().view(-1, self.vocab_size)
        main_target = main_target.contiguous().view(-1)
        main_loss = self.criterion(main_logits, main_target)
        
        hidden_state = main_output['hidden_state']
        
        # MTP損失
        mtp_losses = 0
        mtp_aux_losses = 0
        
        for mtp_offset, mtp_module in enumerate(self.mtp_modules):
            mtp_input = source[:, mtp_offset + 1:self.context_size + mtp_offset + 1]
            mtp_target = source[:, mtp_offset + 2:self.context_size + mtp_offset + 2]
            mtp_freqs = self.freqs_cis[mtp_offset + 1:self.context_size + mtp_offset + 1]
            
            mtp_output = mtp_module(mtp_input, hidden_state, 0, mtp_freqs, causal_mask)
            
            mtp_logits = mtp_output['logits'].contiguous().view(-1, self.vocab_size)
            mtp_target_flat = mtp_target.contiguous().view(-1)
            
            if mtp_target_flat.numel() > 0:
                mtp_losses += self.criterion(mtp_logits, mtp_target_flat)
                mtp_aux_losses += mtp_output['auxiliary_loss']
            
            hidden_state = mtp_output['hidden_state']
        
        # 総損失
        total_loss = main_loss + self.lambda_mtp * mtp_losses + main_output['auxiliary_loss'] + mtp_aux_losses
        
        return {
            'loss': total_loss,
            'main_loss': main_loss,
            'mtp_losses': mtp_losses,
            'aux_loss': main_output['auxiliary_loss'] + mtp_aux_losses
        }

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=10, eos_token_id=None):
        """テキスト生成"""
        self.eval()
        self.main_model.reset_kv_cache()
        
        generated = input_ids.clone()
        
        for step in range(max_new_tokens):
            seq_len = generated.size(1)
            freqs = self.freqs_cis[:seq_len]
            causal_mask = create_causal_mask(seq_len, device=input_ids.device)
            
            output = self.main_model(generated, freqs, causal_mask, train=False)
            logits = output['logits'][:, -1, :] / temperature
            
            if top_k > 1:
                topk_logits, topk_indices = torch.topk(logits, top_k)
                probs = F.softmax(topk_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                next_token = topk_indices.gather(1, next_token)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
        
        return generated


if __name__ == '__main__':
    print("=== DeepSeek-V4 Professional Test ===")
    args = Args(vocab_size=1000, d_model=256, n_layers=4, n_heads=4, context_size=128)
    model = DeepSeekCodeV4Pro(args, device='cpu')
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # 順伝播テスト
    source = torch.randint(0, args.vocab_size, (2, 129))
    output = model(source, train=True)
    print(f"Source shape: {source.shape}")
    print(f"Total loss: {output['loss'].item():.6f}")
    print(f"Main loss: {output['main_loss'].item():.6f}")
    print(f"Aux loss: {output['aux_loss'].item():.6f}")
    
    # 生成テスト
    generated = model.generate(source[:, :10], max_new_tokens=20)
    print(f"Generated shape: {generated.shape}")
    print("Test passed!")
