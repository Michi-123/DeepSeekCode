# -*- coding: utf-8 -*-
"""
DeepSeek-V4 学習・推論スクリプト

使用方法:
    # 教育版の学習
    python deepseekcode_v4_train.py --model edu --mode train
    
    # 教育版の推論
    python deepseekcode_v4_train.py --model edu --mode infer
    
    # プロ版の学習
    python deepseekcode_v4_train.py --model pro --mode train
    
    # プロ版の推論
    python deepseekcode_v4_train.py --model pro --mode infer
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np


# モデルのインポート
from deepseekcode_v4edu import DeepSeekCodeV4Edu, Args as EduArgs
from deepseekcode_v4pro import DeepSeekCodeV4Pro, Args as ProArgs


# ============================================================================
# データセット
# ============================================================================

class SimpleTextDataset(Dataset):
    """
    簡易テキストデータセット
    実際にはランダムなトークン列を生成（デモ用）
    """
    def __init__(self, vocab_size, seq_length, num_samples=1000):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples
        
        # ランダムなデータ生成（デモ用）
        # 実際にはここにトークナイザーとテキストデータの読み込みを実装
        self.data = torch.randint(0, vocab_size, (num_samples, seq_length + 1))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


class DummyTokenizer:
    """ダミートークナイザー"""
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.eos_token_id = 2  # 仮のEOSトークン
    
    def encode(self, text):
        # テキストを適当なIDに変換（デモ用）
        return [min(ord(c), self.vocab_size - 1) for c in text[:100]]
    
    def decode(self, token_ids):
        # IDをテキストに変換（デモ用）
        return ''.join(chr(min(id, 127)) for id in token_ids if id < 128)


# ============================================================================
# 学習クラス
# ============================================================================

class Trainer:
    """DeepSeek-V4 学習マネージャー"""
    def __init__(self, model, args, device='cpu'):
        self.model = model
        self.args = args
        self.device = device
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.max_steps
        )
        
        # MoEモデルの場合、バイアス更新用
        self.has_moe = hasattr(args, 'n_routed_experts') and args.n_routed_experts > 0
    
    def train_step(self, batch):
        """1ステップの学習"""
        self.model.train()
        batch = batch.to(self.device)
        
        self.optimizer.zero_grad()
        
        # 順伝播
        if isinstance(self.model, DeepSeekCodeV4Pro):
            output = self.model(batch, train=True)
            loss = output['loss']
        else:
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]
            output = self.model.compute_loss(input_ids, target_ids, train=True)
            loss = output['loss']
        
        # 逆伝播
        loss.backward()
        
        # グラディエントクリッピング
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # オプティマイザーステップ
        self.optimizer.step()
        self.scheduler.step()
        
        # MoEの場合、エキスパートバイアスを更新
        if self.has_moe and isinstance(self.model, DeepSeekCodeV4Pro):
            for layer in self.model.main_model.layers:
                if hasattr(layer.feed_forward, 'update_expert_bias'):
                    layer.feed_forward.update_expert_bias()
        
        return loss.item()
    
    def train(self, dataloader, max_steps=None):
        """学習ループ"""
        self.model.train()
        
        total_loss = 0
        step_count = 0
        
        start_time = time.time()
        
        for epoch in range(self.args.epochs):
            print(f"\n=== Epoch {epoch + 1}/{self.args.epochs} ===")
            
            for batch_idx, batch in enumerate(dataloader):
                if max_steps and step_count >= max_steps:
                    break
                
                loss = self.train_step(batch)
                total_loss += loss
                step_count += 1
                
                # ログ出力
                if step_count % self.args.log_interval == 0:
                    avg_loss = total_loss / self.args.log_interval
                    elapsed = time.time() - start_time
                    steps_per_sec = self.args.log_interval / elapsed
                    
                    print(f"Step {step_count}: Loss = {avg_loss:.6f}, "
                          f"Speed = {steps_per_sec:.2f} steps/sec")
                    
                    total_loss = 0
                    start_time = time.time()
        
        print("\n=== Training Complete ===")


# ============================================================================
# 推論クラス
# ============================================================================

class Inferencer:
    """DeepSeek-V4 推論マネージャー"""
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, temperature=0.8, top_k=50):
        """テキスト生成"""
        self.model.eval()
        
        # プロンプトのエンコード
        if isinstance(prompt, str):
            input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
        else:
            input_ids = torch.tensor([prompt], device=self.device)
        
        # 生成
        generated = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # デコード
        output_text = self.tokenizer.decode(generated[0].tolist())
        
        return output_text, generated[0].tolist()


# ============================================================================
# メイン処理
# ============================================================================

def create_model(model_type, device):
    """モデル作成"""
    if model_type == 'edu':
        args = EduArgs(
            vocab_size=1000,
            d_model=256,
            n_layers=4,
            n_heads=4,
            context_size=128,
            n_routed_experts=4,
            n_activated_experts=2
        )
        model = DeepSeekCodeV4Edu(args, device=device)
    else:  # pro
        args = ProArgs(
            vocab_size=1000,
            d_model=256,
            n_layers=4,
            n_heads=4,
            context_size=128,
            n_routed_experts=8,
            n_activated_experts=2,
            multi_token_depth=2
        )
        model = DeepSeekCodeV4Pro(args, device=device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_type.upper()}")
    print(f"Total parameters: {total_params:,}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='DeepSeek-V4 Training/Inference')
    parser.add_argument('--model', type=str, default='edu', choices=['edu', 'pro'],
                        help='Model type (edu: educational, pro: professional)')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'infer'],
                        help='Mode (train or infer)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Maximum training steps')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval')
    parser.add_argument('--prompt', type=str, default='Hello',
                        help='Prompt for inference')
    parser.add_argument('--max_new_tokens', type=int, default=30,
                        help='Maximum new tokens to generate')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # モデル作成
    model = create_model(args.model, device)
    
    if args.mode == 'train':
        # 学習モード
        print("\n=== Starting Training ===")
        
        # データセット作成
        train_dataset = SimpleTextDataset(
            vocab_size=1000,
            seq_length=128,
            num_samples=500
        )
        
        dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True
        )
        
        # トレーナー作成
        trainer = Trainer(model, args, device=device)
        
        # 学習開始
        trainer.train(dataloader, max_steps=args.max_steps)
        
        # モデル保存
        torch.save({
            'model_state_dict': model.state_dict(),
            'args': args,
        }, f'deepseek_v4_{args.model}_checkpoint.pth')
        print("Model saved to deepseek_v4_{}_checkpoint.pth".format(args.model))
    
    else:
        # 推論モード
        print("\n=== Starting Inference ===")
        
        # チェックポイントの読み込み（あれば）
        checkpoint_path = f'deepseek_v4_{args.model}_checkpoint.pth'
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from {checkpoint_path}")
        except FileNotFoundError:
            print(f"No checkpoint found at {checkpoint_path}, using random weights")
        
        # トークナイザー作成
        tokenizer = DummyTokenizer(vocab_size=1000)
        
        # 推論実行
        inferencer = Inferencer(model, tokenizer, device=device)
        
        print(f"\nPrompt: {args.prompt}")
        output_text, token_ids = inferencer.generate(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=0.8,
            top_k=50
        )
        
        print(f"Generated: {output_text}")
        print(f"Token IDs: {token_ids[:20]}...")


if __name__ == '__main__':
    main()
