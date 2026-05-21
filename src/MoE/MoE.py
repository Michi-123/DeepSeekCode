# DeepSeekMoE_refactored.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """Simple FFN-style expert (論文の細部実装はプロジェクトに合わせて調整してください)"""
    def __init__(self, d_model, inter_dim):
        super().__init__()
        # 論文のFFNは GELU や SwiGLU 系が多いですが、ここは SiLU + 2-layer を使用
        self.fc1 = nn.Linear(d_model, inter_dim, bias=True)
        self.act = F.silu
        self.fc2 = nn.Linear(inter_dim, d_model, bias=True)

    def forward(self, x):
        # x: [N, d_model]
        return self.fc2(self.act(self.fc1(x)))


class DeepSeekMoE(nn.Module):
    """
    DeepSeekMoE (論文準拠に近い実装)
    - Shared experts を全トークンに加算
    - Routed experts を Top-K で選択し、選択内正規化した gating を掛け合わせて出力を合成
    - Auxiliary-loss-free bias update（stepごとに実行）
    - Sequence-wise balance loss (α * sum(f_i * P_i))
    """
    def __init__(self, args):
        super().__init__()
        self.d_model = args.d_model
        self.n_shared = args.n_shared_experts
        self.n_routed = args.n_routed_experts
        self.top_k = args.n_activated_experts
        self.gamma = args.moe_bias_update_speed  # bias update speed γ
        # V3は alpha を非常に小さくすることを推奨
        self.alpha = getattr(args, "moe_alpha", 1e-6)

        # experts
        self.shared_experts = nn.ModuleList([Expert(self.d_model, args.moe_inter_dim) for _ in range(self.n_shared)])
        self.routed_experts = nn.ModuleList([Expert(self.d_model, args.moe_inter_dim) for _ in range(self.n_routed)])

        # routing parameters
        # centroids (paper: e_i centroid vectors used to compute sigmoid(u^T e_i))
        self.centroids = nn.Parameter(torch.randn(self.n_routed, self.d_model) * (1.0 / (self.d_model ** 0.5)))
        # biases used only for selection (aux-loss-free)
        self.biases = nn.Parameter(torch.zeros(self.n_routed))

        # usage stats (optional, for monitoring)
        self.register_buffer("_expert_usage_counts", torch.zeros(self.n_routed, dtype=torch.long))
        self._reset_usage()

    def _reset_usage(self):
        self._expert_usage_counts.zero_()

    def compute_affinity(self, inputs):
        """
        inputs: [B, T, d_model]
        return s_i_t: [B, T, n_routed] (sigmoid affinities)
        """
        # Flatten for matmul: (B*T, d)
        B, T, D = inputs.shape
        x_flat = inputs.view(-1, D)  # [B*T, D]
        # affinity logits: x_flat @ centroids.T -> [B*T, n_routed]
        logits = x_flat @ self.centroids.t()
        s = torch.sigmoid(logits)  # [B*T, n_routed]
        return s.view(B, T, self.n_routed)  # [B, T, n_routed]

    def update_biases_step(self, affinity_scores, topk_indices):
        """
        bias update per training step (aux-loss-free):
        - compute load per expert on the whole batch (sum of gating-selection counts)
        - expected_load = total_tokens * top_k / n_routed
        - if load > expected -> biases[i] -= gamma else biases[i] += gamma
        """
        # affinity_scores: [B, T, n_routed] (before bias)
        B, T, n = affinity_scores.shape
        device = affinity_scores.device
        total_tokens = B * T
        # Count how many times each expert was selected (Top-K selection)
        # topk_indices: [B, T, top_k]
        # flatten
        flat_idx = topk_indices.view(-1)  # [(B*T*top_k)]
        counts = torch.bincount(flat_idx, minlength=self.n_routed).to(dtype=torch.float32, device=device)
        # Each selection corresponds to one (token, expert) assignment; expected total selections = total_tokens * top_k
        expected_count_per_expert = (total_tokens * self.top_k) / max(1, self.n_routed)
        # Update biases in-place (paper: decrease if overloaded, increase if underloaded)
        with torch.no_grad():
            overloaded = counts > expected_count_per_expert
            self.biases.data[overloaded] -= self.gamma
            self.biases.data[~overloaded] += self.gamma

    def compute_sequence_balance_loss(self, affinity_scores, topk_indices):
        """
        Compute L_bal = α * sum_i ( f_i * P_i )
        - f_i: fraction (per-batch) of tokens for which expert i is in Top-K (scaled as in paper)
        - P_i: average normalized score per expert (normalized over experts per token then mean over tokens)
        """
        B, T, n = affinity_scores.shape
        device = affinity_scores.device
        total_tokens = B * T

        # 1) compute f_i:
        # topk_indices: [B, T, top_k]
        topk_mask = torch.zeros_like(affinity_scores, dtype=torch.bool, device=device)
        topk_mask.scatter_(2, topk_indices, True)  # True where selected
        # Count tokens where expert i is in Top-K
        counts = topk_mask.any(dim=-1).sum(dim=(0, 1)).to(dtype=torch.float32, device=device)  # [n]
        # f_i as fraction of tokens (paper's eqn uses normalization factor; here we adopt fraction)
        f_i = counts / float(total_tokens)  # [n]

        # 2) compute P_i:
        # normalize affinity_scores across experts for each token (eq.19)
        denom = affinity_scores.sum(dim=2, keepdim=True) + 1e-12
        s_normalized = affinity_scores / denom  # [B, T, n]
        P_i = s_normalized.mean(dim=(0, 1))  # mean over tokens -> [n]

        bal = self.alpha * (f_i * P_i).sum()
        return bal

    def forward(self, inputs, train=False):
        """
        inputs: [B, T, d_model]
        returns dict with:
         - hidden_state: [B, T, d_model]
         - affinity_scores: [B, T, n_routed] (raw sigmoid affinities)
         - gating_weights: [B, T, top_k]
         - topk_indices: [B, T, top_k]
         - balance_loss (if train)
        """
        B, T, D = inputs.shape
        assert D == self.d_model, "input dim mismatch"

        # 1) Shared experts: sum their outputs (論文は加算)
        if self.n_shared > 0:
            # apply each shared expert to all tokens (vectorized)
            # inputs_flat: [B*T, D]
            inputs_flat = inputs.view(-1, D)
            shared_sum = torch.zeros_like(inputs_flat)
            for expert in self.shared_experts:
                shared_sum = shared_sum + expert(inputs_flat)
            shared_output = shared_sum.view(B, T, D)  # [B, T, D]
        else:
            shared_output = torch.zeros_like(inputs)

        # 2) Routed experts: compute affinities and Top-K selection
        affinity = self.compute_affinity(inputs)  # [B, T, n_routed] (sigmoid)
        # For selection only, add biases (broadcast)
        biased = affinity + self.biases.view(1, 1, -1)  # [B, T, n_routed]

        # Top-K selection on biased scores
        topk_vals, topk_idx = torch.topk(biased, self.top_k, dim=-1)  # [B, T, top_k]
        # Build gating weights as normalized original s values among selected ones (paper: normalize among selected affinity scores)
        # gather selected original affinity values
        selected_s = torch.gather(affinity, 2, topk_idx)  # [B, T, top_k]
        # make g'_i,t = s_i if selected else 0; then normalize across selected
        # normalization across top_k dimension
        sum_selected = selected_s.sum(dim=-1, keepdim=True) + 1e-12
        gating = selected_s / sum_selected  # [B, T, top_k]

        # 3) Prepare per-selection inputs and compute corresponding expert outputs
        # Expand inputs to match top_k selections: [B, T, top_k, D] -> flatten to [B*T*top_k, D]
        expanded_inputs = inputs.unsqueeze(2).expand(-1, -1, self.top_k, -1).contiguous()
        flat_inputs = expanded_inputs.view(-1, D)  # [B*T*top_k, D]

        flat_topk_idx = topk_idx.view(-1)  # [B*T*top_k]
        flat_gating = gating.view(-1)      # [B*T*top_k]

        # Find unique expert indices used to avoid calling all experts
        used_experts = torch.unique(flat_topk_idx)

        # Compute outputs for used experts in a batched way
        device = inputs.device
        routed_flat_output = torch.zeros_like(flat_inputs)  # will store weighted outputs

        for e_idx in used_experts:
            e = int(e_idx.item())
            mask = (flat_topk_idx == e)
            if not mask.any():
                continue
            idx_positions = torch.nonzero(mask, as_tuple=False).squeeze(1)
            selected_inputs = flat_inputs[idx_positions]  # [N_e, D]
            expert_out = self.routed_experts[e](selected_inputs)  # [N_e, D]
            # multiply by corresponding gating weights
            gw = flat_gating[idx_positions].unsqueeze(-1)  # [N_e, 1]
            routed_flat_output[idx_positions] = expert_out * gw  # weighted

            # update usage counter for monitoring if needed
            if train:
                # accumulate selection counts (coarse)
                self._expert_usage_counts[e] += mask.sum().long()

        # Reshape back: [B, T, top_k, D] then sum over top_k
        routed_output = routed_flat_output.view(B, T, self.top_k, D).sum(dim=2)  # [B, T, D]

        # 4) Final output: input + shared + routed (論文の式)
        output_hidden = inputs + shared_output + routed_output  # [B, T, D]

        result = {
            "hidden_state": output_hidden,
            "affinity_scores": affinity,
            "gating_weights": gating,
            "topk_indices": topk_idx
        }

        if train:
            # update biases according to batch-wise loads
            self.update_biases_step(affinity, topk_idx)
            # compute sequence-wise balance loss (with very small alpha)
            bal_loss = self.compute_sequence_balance_loss(affinity, topk_idx)
            result["balance_loss"] = bal_loss

        return result

    def get_usage(self):
        total = int(self._expert_usage_counts.sum().item())
        if total == 0:
            return {i: 0.0 for i in range(self.n_routed)}
        return {i: (self._expert_usage_counts[i].item() / total) for i in range(self.n_routed)}

    def reset_usage(self):
        self._expert_usage_counts.zero_()
