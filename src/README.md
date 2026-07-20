# DeepSeekCode V4 — 自作 DeepSeek-V3 の V4 対応版

[Michi-123/DeepSeekCode](https://github.com/Michi-123/DeepSeekCode/blob/deepseek/src/deepseekcode.py)
（DeepSeek-V3 論文の教育用自作実装）を、
**DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence**
([arXiv:2606.19348](https://arxiv.org/pdf/2606.19348)) のアーキテクチャへ修正したもの。
いずれも**単一GPUのローカルマシン（CPUでも可）**で動く最小構成。

## ファイル構成

| ファイル | 内容 |
|---|---|
| `deepseekcode_v4edu.py` | **教育用の簡易実装**。V3版のコードスタイルを踏襲し、V4の各要素を読みやすく簡略化 |
| `deepseekcode_v4pro.py` | **論文に忠実な実装**。2系列圧縮・チャネル毎の圧縮重み・圧縮エントリのKVキャッシュ・FP8/BF16混合ストレージ等を含む（パラメータは最小構成） |
| `train_v4.py` | 学習・推論スクリプト（トイコーパス内蔵で自己完結） |

## V3 実装からの変更点と論文の対応

| # | V4 の新要素 | edu | pro |
|---|---|---|---|
| 1 | **CSA** (Compressed Sparse Attention): mトークン毎にKVを1エントリへ圧縮し、Lightning Indexer `I_{t,s}=Σ_h w_{t,h}^I·ReLU(q_{t,h}^I·k_s^IComp)` で Top-k 選択（式11-19） | 1系列・スカラー圧縮重み | 2系列 C^a/C^b・チャネル毎重み `Softmax_row([Z^a+B^a; Z^b+B^b])`（式11-12） |
| 2 | **HCA** (Heavily Compressed Attention): m'≫m でさらに強圧縮、疎選択なし（式22-25） | ○ | ○ |
| 3 | CSA/HCA の**交互配置**（ハイブリッドアテンション） | 偶数層CSA/奇数層HCA | `layer_pattern` で任意指定可（既定は交互） |
| 4 | **共有KV MQA**（圧縮エントリがKeyとValueを兼用）+ **グループ化出力射影**（gグループ） | ○ | ○ |
| 5 | クエリの低ランク生成 `c_t^Q = h_t W^DQ`, `q_t = c_t^Q W^UQ`（式13,18）、RoPE部分適用 | ○ | ○ |
| 6 | **mHC**: 残差ストリームn本 + 混合行列を Sinkhorn-Knopp 反復(t_max=20)で二重確率行列（Birkhoff多様体）に制約 → `‖B_l‖₂≤1` で非膨張 | n=2 | n=4 |
| 7 | MoE ゲーティング `Sigmoid → Sqrt(Softplus)` | ○ | ○ |
| 8 | 先頭層の密FFNを**ハッシュルーティングMoE**へ置換 | ○ | ○ |
| 9 | バランス損失 | V3形式の軽い補助損失 | **シーケンス方向**バランス損失（シーケンス毎に f_i·P_i を計算しバッチ平均） |
| 10 | **Muon** オプティマイザ: ハイブリッド Newton-Schulz 反復（前半8回 (3.4445, -4.7750, 2.0315) / 後半2回 (2, -1.5, 0.5)、Algorithm 1） | ○ | ○（Nesterov付き） |
| 11 | 推論時KVキャッシュ | トークン毎潜在を保持し毎回再圧縮（簡略化） | **圧縮エントリ自体をキャッシュ**（V4のKV削減の本体）。`kv_fp8=True` で RoPE次元=BF16 / その他=FP8 の混合ストレージ |
| 12 | **MTP** (Multi-Token Prediction) | V3と同一構成（論文に「V3と同一」と明記） | 同左 |

論文に明記のない箇所（層配置の比率、RoPEキーの圧縮方法、未完了ブロックの扱い等）は
各ファイル冒頭のdocstringに設計判断として明記した。

## 使い方

必要なもの: Python 3.9+ / PyTorch 2.x（GPUなしでも動作）

```bash
# 教育版を学習 → そのまま生成デモ
python train_v4.py --model edu --steps 300

# 忠実版（pro）を学習
python train_v4.py --model pro --steps 300

# 学習済みチェックポイントから推論のみ
python train_v4.py --model edu --generate --prompt "深層学習は"

# 高速動作確認（モデルを小さく）
python train_v4.py --model pro --steps 50 --context 64 --d-model 128

# 各実装単体のスモークテスト
python deepseekcode_v4edu.py
python deepseekcode_v4pro.py
```

### Google Colab の場合

```python
!git clone <このディレクトリを置いたリポジトリ>  # または各ファイルをアップロード
%cd v4
!python train_v4.py --model edu --steps 300
```

## 学習の流れ（train_v4.py）

1. トイコーパス（内蔵の日本語テキスト）を文字レベルでトークン化
2. 毎ステップ `context_size + MTP深さ + 1` 長の窓をランダムに切り出す
3. `model.pretrain(source)` = メイン損失 + λ·MTP損失 + バランス損失
4. backward 後、**Muon**（2次元重み行列）と **AdamW**（埋め込み・出力ヘッド・ルーター・1次元パラメータ）で更新
5. `model.update_expert_bias()` で補助損失フリーのロードバランス（バイアス項±γ）

## 注意事項

- 本コードは教材目的の最小実装であり、論文の 1.6T (Pro) / 284B (Flash) 級の
  性能・効率を再現するものではない。百万トークンコンテキストも扱わない
  （構造だけを忠実に縮小再現している）。
- `kv_fp8=True` は `torch.float8_e4m3fn` が使える PyTorch (2.1+) でのみ有効。
  非対応環境では自動的に通常精度で保持する。
