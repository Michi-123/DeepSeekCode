#@title MoE (Fix)
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """Individual expert network for MoE"""
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
    """
    DeepSeekMoE implementation following DeepSeek-V2/V3 papers
    with auxiliary-loss-free load balancing strategy
    """
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
        """専門家使用頻度の初期化"""
        self.expert_freqs = {idx: 0 for idx in range(self.n_routed_experts)}

    def update_biases(self, gating_weights, topk_indices):
        """
        Auxiliary-loss-free load balancing strategy from DeepSeek-V3
        過負荷の専門家のバイアスを下げ、未使用の専門家のバイアスを上げる
        """
        # 各専門家の負荷を計算
        expert_load = torch.zeros(self.n_routed_experts, device=gating_weights.device)

        for k in range(self.top_k):
            expert_indices = topk_indices[:, :, k]  # [batch, seq]
            weights = gating_weights[:, :, k]       # [batch, seq]

            # ベクトル化された負荷計算
            for i in range(self.n_routed_experts):
                mask = (expert_indices == i)
                expert_load[i] += weights[mask].sum()

        mean_load = expert_load.mean()

        # バイアス更新（修正版）
        # 過負荷の専門家: バイアスを下げて選ばれにくくする
        # 未使用/低負荷の専門家: バイアスを上げて選ばれやすくする
        for i in range(self.n_routed_experts):
            if expert_load[i] > mean_load:  # 過負荷の場合
                self.biases.data[i] -= self.bias_update_speed
            else:  # 低負荷/未使用の場合
                self.biases.data[i] += self.bias_update_speed

    def compute_balance_loss(self, affinity_scores, topk_indices):
        """
        DeepSeek論文に基づくバランスロス計算
        L_balance = α * Σ(f_i * P_i)
        """
        batch_size, seq_len, num_experts = affinity_scores.shape
        total_tokens = batch_size * seq_len

        # f_i: 専門家iに割り当てられたトークンの割合
        f_i = torch.zeros(num_experts, device=affinity_scores.device)
        for i in range(num_experts):
            # 専門家iがTop-Kに選ばれたトークン数をカウント
            mask = (topk_indices == i).any(dim=-1)  # [batch, seq]
            f_i[i] = mask.sum().float() / total_tokens

        # P_i: 専門家iの平均ゲート確率
        # 全専門家に対してsoftmax正規化を適用
        normalized_scores = F.softmax(affinity_scores, dim=-1)
        P_i = normalized_scores.mean(dim=(0, 1))  # [num_experts]

        # バランスロス: α * Σ(f_i * P_i)
        balance_loss = self.alpha * (f_i * P_i).sum()

        return balance_loss

    def forward(self, input_embeddings, train=False):
        """
        MoE forward pass following DeepSeek-V2/V3 formulation:
        h'_t = h_t + Σ(shared_experts) + Σ(G_j,t * routed_expert_j(h_t))
        """
        batch_size, seq_len = input_embeddings.shape[:2]

        # 共有専門家の処理 (全トークンがすべての共有専門家を利用)
        shared_outputs = []
        for expert in self.shared_experts:
            shared_outputs.append(expert(input_embeddings))
        shared_output = torch.stack(shared_outputs, dim=2).mean(dim=2)

        # ルーティング専門家の処理 (Top-K 選択)
        # アフィニティスコア計算（DeepSeek-V3ではsigmoidを使用）
        affinity_scores = torch.sigmoid(input_embeddings @ self.centroids.T)

        # バイアス項を加えたスコア（Top-K選択に使用）
        # auxiliary-loss-free戦略に基づく
        expanded_biases = self.biases.unsqueeze(0).unsqueeze(0)  # [1, 1, num_experts]
        biased_scores = affinity_scores + expanded_biases

        # Top-K選択（バイアス付きスコアを使用）
        topk_values, topk_indices = torch.topk(biased_scores, self.top_k, dim=-1)

        # ゲーティング重み（元のアフィニティスコアから計算）
        # バイアスは選択のみに使用し、重み計算には元のスコアを使用
        selected_affinity_scores = torch.gather(affinity_scores, -1, topk_indices)
        gating_weights = F.softmax(selected_affinity_scores, dim=-1)
        # V2の場合
        # Gating weights: softmax over ALL affinity scores, then gather selected ones
        # all_gating_weights = F.softmax(affinity_scores, dim=-1)  # Normalize over ALL experts
        # gating_weights = torch.gather(all_gating_weights, -1, topk_indices)

        # 効率的な専門家計算のための前処理
        # インデックスとゲーティング重みを1次元に平坦化
        flat_topk_indices = topk_indices.view(-1)
        flat_gating_weights = gating_weights.view(-1)

        # 入力埋め込みをTop-K分だけ複製してから平坦化
        expanded_input_embeddings = input_embeddings.unsqueeze(2).expand(-1, -1, self.top_k, -1)
        flat_input_embeddings = expanded_input_embeddings.contiguous().view(-1, self.dim)

        # 使用される専門家の特定と計算
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

        # 平坦化された出力を元の形状に戻し、Top-K次元で合計
        reshaped_routed_output = routed_output.view(batch_size, seq_len, self.top_k, -1)
        routed_output = reshaped_routed_output.sum(dim=2)

        # 最終出力 (入力 + 共有専門家 + ルーティング専門家)
        # DeepSeek論文の式: h'_t = h_t + shared + routed
        output_embeddings = input_embeddings + shared_output + routed_output

        # 訓練時の処理
        balance_loss = None
        if train:
            # Auxiliary-loss-free load balancing
            self.update_biases(gating_weights, topk_indices)
            
            # 補完的なバランスロス計算
            balance_loss = self.compute_balance_loss(affinity_scores, topk_indices)

        # 出力の準備
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


# 使用例とテスト用のダミークラス
class Args:
    def __init__(self):
        self.d_model = 512
        self.n_shared_experts = 2
        self.n_routed_experts = 8
        self.n_activated_experts = 2
        self.moe_inter_dim = 1024
        self.moe_bias_update_speed = 0.001
        self.moe_alpha = 0.001


def test_moe():
    """MoE実装のテスト"""
    args = Args()
    moe = MoE(args)
    
    # テスト用の入力
    batch_size, seq_len = 2, 4
    input_embeddings = torch.randn(batch_size, seq_len, args.d_model)
    
    # Forward pass
    output = moe(input_embeddings, train=True)
    
    print(f"Input shape: {input_embeddings.shape}")
    print(f"Output shape: {output['hidden_state'].shape}")
    print(f"Balance loss: {output['balance_loss'].item():.6f}")
    print(f"Expert usage: {moe.get_expert_usage_stats()}")
    
    return moe, output


if __name__ == "__main__":
    test_moe()