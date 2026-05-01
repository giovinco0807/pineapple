# OFC Pineapple AI - PolicyNet枝刈りパイプライン

## 概要

PolicyNet（行動模倣学習済み）を使ってT0の配置候補を232→50に枝刈りし、
Rust MCソルバーで高精度評価を行うパイプライン。

## アーキテクチャ

```
[Stage 1: Python Pre-Filter]
  ランダム5枚ハンド生成
  → PolicyNet推論（522dim入力 → 250出力）
  → Top-50配置を選択
  → JSON出力

[Stage 2: Rust MC Evaluation]
  JSON読み込み
  → 50配置のみMC評価（Imperfect-Info Nested MC）
  → JSONL出力（全配置のEV付き）

[Stage 3: Python Training Data Conversion]
  JSONL → convert_t0_cfr.py → BC学習用データ
  → 24x augmentation (suit permutation)
```

## ファイル構成

| ファイル | 役割 |
|---|---|
| `ai/training/generate_filtered_t0.py` | Stage 1: PolicyNet pre-filter |
| `ai/rust_solver/cfr_solver/src/main.rs` | `T0BatchFiltered` CLIコマンド |
| `ai/rust_solver/cfr_solver/src/t0_eval.rs` | `run_batch_filtered()` MC評価 |
| `ai/training/convert_t0_cfr.py` | Stage 3: BC学習データ変換 |
| `ai/models/t0_bc/bc_policy_best.pt` | 学習済みPolicyNet（522→250） |

## 使い方

### Stage 1: Python Pre-Filter
```bash
cd ofc-pineapple
python ai/training/generate_filtered_t0.py \
  --n-hands 500 \
  --top-k 50 \
  --output ai/data/filtered_t0.json \
  --seed 42
```

### Stage 2: Rust Filtered Evaluation
```bash
cd ofc-pineapple/ai/rust_solver
cargo build --release
./target/release/cfr_solver t0-batch-filtered \
  --input ../../ai/data/filtered_t0.json \
  --samples 30 \
  --output ../../ai/data/t0_filtered_train.jsonl \
  --nesting "10,6,3" \
  --seed 42
```

### Stage 3: Training Data Conversion
```bash
python ai/training/convert_t0_cfr.py \
  --input ai/data/t0_filtered_train.jsonl \
  --output ai/data/t0_filtered_bc.jsonl
```

## パラメータ設計

### サンプル数・ネスティング

| 設定 | 旧（全232配置） | 新（50配置） | 効果 |
|---|---|---|---|
| `--samples` | 30 | 30 | 同一（深度で精度向上） |
| `--nesting` | `5,3,2` | `10,6,3` | ~2x深度向上 |
| 計算量/ハンド | 232 × 30 = 6,960タスク | 50 × 30 = 1,500タスク | **4.6x削減** |
| 実効精度 | nesting計 5×3×2=30 | nesting計 10×6×3=180 | **6x向上** |

> **注意**: `--nesting "10,6,3"` は1サンプルあたりの内部展開が5,3,2の6倍になるため、
> 実際のwall-clock timeは旧版と同等か若干増加する可能性あり。
> GCPでベンチマーク後に微調整する。

### GCP実行見積もり

| 項目 | 値 |
|---|---|
| ハンド数 | 500 |
| 24x augmentation後 | 12,000サンプル |
| 推定時間/ハンド | ~40-50秒（ローカルPC） |
| GCP (64core) | ~10-15秒/ハンド |
| 推定総時間 | ~2-3時間 |

## カード表記の対応

| Python | Rust | 例 |
|---|---|---|
| 通常カード | 同一 | `Ah`, `Ts`, `2c` |
| Joker 1 | `X1` → `JK` | Python `X1` = Rust `JK` |
| Joker 2 | `X2` → `JK` | Python `X2` = Rust `JK` |

## 重要な実装詳細

### カード順の一致
Python側の`action_to_rust_placement()`はdealt_cardsのインデックス順（0→4）で
カードを列挙する。これはRust側の`format_placement()`が`hand[0]..hand[4]`の
順でカードを列挙するのと一致させるため。

### 出力JSONLフォーマット
旧`T0Batch`と完全互換。`convert_t0_cfr.py`でそのまま変換可能。
```json
{
  "hand_idx": 0,
  "hand": "2s Td 7s 6c 8d",
  "type": "NoPair_HiT",
  "n_placements": 50,
  "n_samples": 30,
  "nesting": "[10, 6, 3]",
  "placements": [
    {"p": "Top[] Mid[7s 6c 8d] Bot[2s Td]", "ev": 19.923},
    ...
  ]
}
```

---

## 今後のロードマップ

### Phase 2a: T0高精度データ収集（← 今ここ）
- [x] PolicyNet pre-filter実装 (`generate_filtered_t0.py`)
- [x] Rust `T0BatchFiltered` コマンド実装
- [x] ローカルテスト完了（5ハンド、50/232マッチ確認）
- [ ] GCPで500ハンド生成
- [ ] convert_t0_cfr.py で変換 → PolicyNet再学習

### Phase 2b: T1-T3 PolicyNet学習
- [ ] T1用データ収集パイプライン（T0配置後の状態からT1配置を評価）
- [ ] T2, T3用データ収集パイプライン
- [ ] 各ターン用PolicyNet学習
- [ ] 各ターンでも枝刈り → 高精度MC評価のサイクルを回す

### Phase 3: セルフプレイ強化
- [ ] 全ターン(T0-T3)のPolicyNet統合
- [ ] セルフプレイ対戦ループ構築
- [ ] PPO or MCTS + PolicyNetでの強化学習
- [ ] ELO評価によるモデル選択

### Phase 4: デプロイ
- [ ] ONNX変換 → Rustネイティブ推論
- [ ] リアルタイムAIプレイヤーへの統合
