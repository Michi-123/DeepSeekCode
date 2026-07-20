--- deepseekcode_v4edu.py (原始)


+++ deepseekcode_v4edu.py (修改后)
# -*- coding: utf-8 -*-
"""
DeepSeek-V4 Educational Implementation
簡易版：教育・学習用に単純化したDeepSeek-V4実装

V3からV4への主な変更点:
1. Hybrid Attention (ローカル+グローバル)
2. 簡易MoEルーティング
3. RMSNormの改善
4. RoPEの拡張
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Args:
    """ハイパーパラメータ設定"""
    def __init__(self, **kwargs):
        # デフォルト値（単一GPU向けに小規模）
        defaults = {
            'vocab_size': 32000,
            'd_model': 512,
            'n_layers': 4,
            'n_heads': 8,
            'd_head': 64,
            'd_rope': 64,  # d_head と同じ値にする
            'context_size': 512,
            'max_seq_len': 512,
            'rope_theta': 10000.0,
            'norm_eps': 1e-6,
            'n_shared_experts': 1,
            'n_routed_experts': 4,
            'n_activated_experts': 2,
            'moe_inter_dim': 1024,
            'attn_dropout': 0.1,
            'ffn_dropout': 0.1,
        }
        defaults.update(kwargs)
        for key, value in defaults.items():
            setattr(self, key, value)


# ============================================================================
# ヘルパー関数
# ============================================================================

def create_causal_mask(seq_len, device='cpu') -> torch.Tensor:
    """因果マスクの作成"""
    ones = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    return torch.triu(ones, 1)


def precompute_freqs_cis(args, device='cpu'):
    """RoPE用の周波数行列を事前計算"""
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
    """Rotary Positional Embedding の適用"""
    batch_size, seq_len, n_heads, d_head = x.shape

    # freqs_cis が 2 次元 (seq_len, dim) の場合
    if freqs_cis.dim() == 2:
        freqs_cis = freqs_cis[:seq_len]  # シーケンス長に合わせる

    x_reshaped = x.view(batch_size, seq_len, n_heads, d_head // 2, 2)
    x_complex = torch.view_as_complex(x_reshaped)

    # freqs_cis を適切な形状に変形 (1, seq_len, 1, dim)
    if freqs_cis.dim() == 1:
        freqs_cis = freqs_cis[None, :].expand(seq_len, -1)[None, :, None, :]
    elif freqs_cis.dim() == 2:
        freqs_cis = freqs_cis[None, :, None, :]

    rotated_complex = x_complex * freqs_cis
    rotated_embeds = torch.view_as_real(rotated_complex).flatten(3)
    return rotated_embeds.type_as(x)



# ============================================================================
# 正規化レイヤー
# ============================================================================

class RMSNorm(nn.Module):
    """RMS Normalization (DeepSeek-V4では改善版を採用)"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x_normed * self.weight


# ============================================================================
# V4の新機能: Hybrid Attention
# ============================================================================

class HybridAttention(nn.Module):
    """
    DeepSeek-V4のHybrid Attention
    ローカル注意とグローバル注意を組み合わせ
    """
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.d_model = args.d_model
        self.d_head = args.d_head
        self.d_rope = args.d_rope
        self.attn_dropout = nn.Dropout(args.attn_dropout)

        # Q, K, V投影
        self.fc_q = nn.Linear(args.d_model, args.n_heads * args.d_head)
        self.fc_k = nn.Linear(args.d_model, args.n_heads * args.d_head)
        self.fc_v = nn.Linear(args.d_model, args.n_heads * args.d_head)

        # 出力投影
        self.fc_out = nn.Linear(args.n_heads * args.d_head, args.d_model)

        # ローカルウィンドウサイズ（V4の新機能）
        self.local_window_size = 64

        # Xavier初期化
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)

    def forward(self, x, freqs_cis, causal_mask=None, train=False):
        """
        Hybrid Attention forward pass
        """
        batch_size, seq_len, _ = x.shape
        H, D = self.n_heads, self.d_head

        # Q, K, Vの計算
        q = self.fc_q(x).view(batch_size, seq_len, H, D)
        k = self.fc_k(x).view(batch_size, seq_len, H, D)
        v = self.fc_v(x).view(batch_size, seq_len, H, D)

        # RoPE適用
        freqs = freqs_cis[:seq_len, :D//2].to(x.device)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        # 転置: (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled Dot-Product Attention
        sqrt_d_k = D ** 0.5
        scores = torch.matmul(q, k.transpose(2, 3)) / sqrt_d_k

        # マスク処理
        if causal_mask is not None:
            if causal_mask.dim() == 2:
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)

        # ソフトマックスとドロップアウト
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 出力計算
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, H * D)
        out = self.fc_out(out)

        output = {
            'hidden_state': out,
            'attention_weight': attn_weights
        }
        return output


# ============================================================================
# V4のMoE: 簡易ルーティング
# ============================================================================

class Expert(nn.Module):
    """MoEの専門家ネットワーク"""
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


class SimpleMoE(nn.Module):
    """
    DeepSeek-V4用 簡易MoE
    教育用にルーティングを単純化
    """
    def __init__(self, args):
        super().__init__()
        self.d_model = args.d_model
        self.n_shared_experts = args.n_shared_experts
        self.n_routed_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        self.moe_inter_dim = args.moe_inter_dim

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

        # ゲートネットワーク（簡易版）
        self.gate = nn.Linear(self.d_model, self.n_routed_experts, bias=False)

        # 補助損失用バッファ
        self.register_buffer('step_expert_counts', torch.zeros(self.n_routed_experts))
        self.register_buffer('step_total_tokens', torch.tensor(0.0))

    def forward(self, x, train=False):
        batch_size, seq_len, d_model = x.shape
        u = x.reshape(-1, d_model)

        # 共有エキスパートの出力
        shared_output = sum(expert(x) for expert in self.shared_experts)

        # ゲートスコア計算
        gate_scores = F.softmax(self.gate(u), dim=-1)

        # Top-K選択
        topk_values, topk_indices = torch.topk(
            gate_scores,
            k=self.n_activated_experts,
            dim=1
        )

        # 正規化
        gating_sum = topk_values.sum(dim=1, keepdim=True)
        gating_weights = topk_values / (gating_sum + 1e-8)

        # ルーティング出力
        routed_output = torch.zeros_like(u)
        expert_counts = torch.zeros(self.n_routed_experts, device=x.device)

        for expert_id in range(self.n_routed_experts):
            mask = (topk_indices == expert_id)
            t, i = torch.where(mask)
            if len(t) > 0:
                expert_output = self.routed_experts[expert_id](u[t])
                routed_output[t] += expert_output * gating_weights[t, i, None]
                expert_counts[expert_id] = mask.sum()

        routed_output = routed_output.reshape(batch_size, seq_len, d_model)

        # 補助損失
        auxiliary_loss = torch.tensor(0.0, device=x.device)
        if train and self.step_total_tokens > 0:
            expected_load = self.n_activated_experts / self.n_routed_experts
            expert_load = expert_counts / self.step_total_tokens
            auxiliary_loss = (expert_load - expected_load).pow(2).mean()
            self.step_expert_counts.copy_(expert_counts.detach())
            self.step_total_tokens.fill_(batch_size * seq_len)

        output = {
            'hidden_state': shared_output + routed_output,
            'auxiliary_loss': auxiliary_loss,
            'affinity_scores': gate_scores,
            'gating_weights': gating_weights
        }
        return output


# ============================================================================
# Transformer Block
# ============================================================================

class TransformerBlock(nn.Module):
    """DeepSeek-V4 Transformerブロック"""
    def __init__(self, args, layer_id=0):
        super().__init__()
        self.attention = HybridAttention(args)

        # 最初の層はFFN、それ以降はMoE
        if layer_id == 0:
            self.feed_forward = nn.Sequential(
                nn.Linear(args.d_model, args.d_model * 4),
                nn.GELU(),
                nn.Dropout(args.ffn_dropout),
                nn.Linear(args.d_model * 4, args.d_model)
            )
            self._is_moe = False
        else:
            self.feed_forward = SimpleMoE(args)
            self._is_moe = True

        self.attn_norm = RMSNorm(args.d_model, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.d_model, eps=args.norm_eps)

    def forward(self, x, freqs_cis, causal_mask=None, train=False):
        # Attention
        h = self.attn_norm(x)
        attn_out = self.attention(h, freqs_cis, causal_mask, train)
        h = attn_out['hidden_state'] + x

        # FFN/MoE
        h2 = self.ffn_norm(h)
        ffn_out = self.feed_forward(h2)

        if self._is_moe:
            h = ffn_out['hidden_state'] + h
            aux_loss = ffn_out['auxiliary_loss']
        else:
            h = ffn_out + h
            aux_loss = torch.tensor(0.0, device=x.device)

        output = {
            'hidden_state': h,
            'auxiliary_loss': aux_loss,
            'attention_weight': attn_out['attention_weight']
        }
        return output


# ============================================================================
# メインモデル
# ============================================================================

class DeepSeekCodeV4Edu(nn.Module):
    """
    DeepSeek-V4 Educational Version
    単一GPUで動作する簡易実装
    """
    def __init__(self, args=None, device='cpu'):
        super().__init__()
        if args is None:
            args = Args()

        self.args = args
        self.device = device
        self.vocab_size = args.vocab_size
        self.context_size = args.context_size

        # 埋め込み
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)

        # レイヤー
        self.layers = nn.ModuleList([
            TransformerBlock(args, layer_id)
            for layer_id in range(args.n_layers)
        ])

        # 出力
        self.output_norm = RMSNorm(args.d_model, eps=args.norm_eps)
        self.output_head = nn.Linear(args.d_model, args.vocab_size, bias=False)

        # 重み共有（オプション）
        self.output_head.weight = self.embedding.weight

        # RoPE
        self.freqs_cis = precompute_freqs_cis(args, device)

        # 損失関数
        self.criterion = nn.CrossEntropyLoss()

        self.to(device)

    def forward(self, input_ids, train=False):
        """順伝播"""
        batch_size, seq_len = input_ids.shape
        causal_mask = create_causal_mask(seq_len, device=input_ids.device)
        freqs_cis = self.freqs_cis[:seq_len]

        # 埋め込み
        h = self.embedding(input_ids)

        # Transformer layers
        total_aux_loss = torch.tensor(0.0, device=input_ids.device)
        for layer in self.layers:
            out = layer(h, freqs_cis, causal_mask, train)
            h = out['hidden_state']
            total_aux_loss = total_aux_loss + out['auxiliary_loss']

        # 出力
        h = self.output_norm(h)
        logits = self.output_head(h)

        return {
            'logits': logits,
            'hidden_state': h,
            'auxiliary_loss': total_aux_loss
        }

    def compute_loss(self, input_ids, target_ids, train=False):
        """損失計算"""
        output = self(input_ids, train=train)
        logits = output['logits'].contiguous().view(-1, self.vocab_size)
        target = target_ids.contiguous().view(-1)

        ce_loss = self.criterion(logits, target)
        total_loss = ce_loss + output['auxiliary_loss']

        return {
            'loss': total_loss,
            'ce_loss': ce_loss,
            'aux_loss': output['auxiliary_loss']
        }

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=10, eos_token_id=None):
        """テキスト生成"""
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            seq_len = generated.size(1)
            output = self(generated)
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
    # テスト実行
    print("=== DeepSeek-V4 Educational Test ===")
    args = Args(vocab_size=1000, d_model=256, n_layers=2, n_heads=4, context_size=128)
    model = DeepSeekCodeV4Edu(args, device='cpu')

    # パラメータ数表示
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # 順伝播テスト
    input_ids = torch.randint(0, args.vocab_size, (2, 64))
    output = model(input_ids, train=True)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {output['logits'].shape}")
    print(f"Auxiliary loss: {output['auxiliary_loss'].item():.6f}")

    # 生成テスト
    generated = model.generate(input_ids[:, :10], max_new_tokens=20)
    print(f"Generated shape: {generated.shape}")
    print("Test passed!")