# T0 Value Network 改善ロードマップ

## 目的

T0（初手5枚配置）で ~150通りの配置候補をValue Networkで即座にスコアリングし、
有望な候補だけをRustソルバーで精密評価することで**ソルバー全体を大幅に高速化**する。

---

## 現状

```
Rustソルバー (nesting=[3,2,1], samples=50)
  ├── T0: 150配置を全てMC評価 → 1手あたり ~130秒
  ├── T1-T4: 各ターンもMC評価
  └── 合計: 高コスト
```

**ボトルネック**: T0の150配置を全部深く評価するのが重い

---

## Phase 1: VN v1 — ソルバーEVで教師あり学習 ← **今ここ**

### やること
1. Rustソルバー (s50) で500手のT0データを生成（GCP 10台フリート）
2. 全配置のEVを教師データとしてVNを訓練
3. VNをソルバーに組み込み、Top-K枝刈りを実装

### データ
- 500手 × ~150配置 = **~75,000サンプル**
- 入力: 522次元の盤面状態（配置後）
- 教師: ソルバーが計算したEV

### 期待される効果
| 指標 | 値 |
|------|---|
| Top-20にTop-1が含まれる確率 | ~90% |
| ソルバー速度改善 | 130s → ~20s（**6-7倍高速化**）|
| VN推論コスト | < 1ms（GPU） / < 5ms（CPU） |

### ソルバー統合方法（2つの選択肢）

**A. Python前処理 → Rust本体**
```
Python: 150配置 → VN推論 → Top-20のインデックスをRustに渡す
Rust:   Top-20だけMC評価
```
- 実装が簡単
- Python↔Rust間のオーバーヘッドあり

**B. ONNX推論をRustに組み込み**
```
Rust内部: 150配置 → ONNX Runtime推論 → Top-20 → MC評価
```
- オーバーヘッドなし
- ONNX Runtimeのクレート追加が必要

### 状態
- [x] データ生成パイプライン構築
- [x] JSONL→NPZ変換スクリプト (`convert_t0_to_npz.py`)
- [x] 訓練テスト成功 (7手で corr=0.79)
- [ ] GCPフリートのデータ収集完了待ち
- [ ] 本番訓練 (500手, 200エポック)
- [ ] ソルバー統合

---

## Phase 2: VN v2 — 好循環ループ（VN枝刈り → 高精度ソルバー → より良いVN）

### 考え方

VN v1で枝刈りしたソルバーは速い。
速いソルバーなら**より多くの手・より高精度な設定**でデータを生成できる。
そのデータでVN v2を訓練すれば、さらに精度が上がる。

```
┌─────────────────────────────────────────────────┐
│                                                 │
│  VN v1 (s50, 500手)                             │
│    ↓ 枝刈り: 150→20配置                         │
│  ソルバーが7倍速 (130s → 20s)                    │
│    ↓ 余った計算予算を活用                         │
│  高精度データ生成 (s200, 2000手)                  │
│    ↓                                            │
│  VN v2 訓練 (~300K samples)                     │
│    ↓ 枝刈り精度UP: 150→10配置                    │
│  ソルバーがさらに高速 (20s → ~10s)               │
│    ↓                                            │
│  もう一周回せる...                               │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 具体的な手順

1. Phase 1のVN v1をソルバーに組み込む
2. VN枝刈り付きソルバーで新データを生成
   - `samples=200`（精度4倍UP）
   - `nesting=[4,3,2]` にアップグレードも検討
   - 2000手（量4倍UP）
3. VN v2を訓練
4. 精度検証 → 必要ならもう1周

### 期待される効果
| 指標 | v1 | v2 |
|------|---|---|
| 訓練データ量 | 75K | 300K |
| ソルバーEV精度 | s50 | s200 |
| Top-10命中率 | ~80% | ~95% |
| ソルバー速度 | ~20s/手 | ~10s/手 |

### コスト・期間
- GCP計算: VN枝刈り後は1手~20sなので、2000手 ≈ 11時間
- 10台フリートで ~1時間のwall-time
- **GCPコスト: ~$5**
- 訓練: GPU 30分

---

## Phase 3: VN v3 — フルゲーム Self-Play

### 考え方

Phase 1-2はソルバーのMC近似EVに依存している。
Self-Playでは**実際にゲームを最後まで打ち、本当のスコアを教師信号にする**。
これにより「T0の配置がT3以降にどう影響するか」という**長期的視野**を学習できる。

```
フルゲーム自己対戦 (T0→T1→...→T4→スコア計算)
  ├── T0: VN v2 + RolloutEvaluator で配置選択
  ├── T1-T4: PolicyNet (BC) + RolloutEvaluator で配置選択
  └── ゲーム結果: 「このT0配置 → 最終スコア+25」を記録

数千ゲーム分の (各ターン盤面, 最終スコア) ペアで VN v3 を訓練
```

### アーキテクチャ: AlphaZero式イテレーションループ

```
┌──────────────────────────────────────────────────┐
│  Iteration N                                     │
│                                                  │
│  1. Self-Play (500ゲーム)                         │
│     └── PolicyNet + VN + RolloutEvaluator        │
│         で両プレイヤーを制御                       │
│     └── 全ターンの (state, action, reward) 記録   │
│                                                  │
│  2. VN訓練                                       │
│     └── 全ターンのデータで ValueNetworkV3 更新    │
│     └── turn-conditioned heads で T0-T4 同時学習  │
│                                                  │
│  3. PolicyNet訓練                                 │
│     └── Self-Playの行動分布 (visit-count) を模倣  │
│     └── KL-div loss                              │
│                                                  │
│  4. 評価                                          │
│     └── 新モデル vs 旧モデル 100ゲーム対戦        │
│     └── 勝率55%+なら新モデルを採用                │
│                                                  │
│  → Iteration N+1                                 │
└──────────────────────────────────────────────────┘
```

### 既存コード資産（全て実装済み）

| ファイル | 役割 | 状態 |
|---------|------|------|
| `ai/mcts/self_play.py` | フルゲーム実行 + 軌跡記録 | ✅ `play_hand_with_mcts()` |
| `ai/mcts/rollout_evaluator.py` | MC rolloutベース行動選択 | ✅ 1653行、VN prefilter対応 |
| `ai/mcts/mcts.py` | IS-MCTS + Progressive Widening | ✅ PolicyNet prior + VN leaf eval |
| `ai/training/train_selfplay.py` | イテレーションループ | ✅ KL-div policy + MSE value |
| `ai/training/generate_data.py` | ヒューリスティック自己対戦 | ✅ ベースライン生成 |
| `ai/models/networks.py` | PolicyNet + ValueNetworkV3 | ✅ turn-conditioned |
| `ai/training/config.py` | ハイパーパラメータ | ✅ `TrainingConfig` |
| `ai/engine/game_engine.py` | スコアリング + ゲーム進行 | ✅ `GameEngine.compute_result()` |

### Self-Play 行動選択の3つの方式

| 方式 | 速度 | 強さ | 用途 |
|------|------|------|------|
| **A. RolloutEvaluator** | ~2s/手 | 中-高 | Phase 3 メイン |
| **B. MCTS + VN** | ~5s/手 | 高 | Phase 4 |
| **C. PolicyNet直接推論** | ~1ms/手 | 低-中 | 大量生成用 |

**Phase 3推奨**: 方式A（RolloutEvaluator）を使い、VN v2で事前フィルタすれば
1ゲーム（T0-T4 × 2プレイヤー = 10決定点）を **~20秒** で完了。
1000ゲーム ≈ **6時間**（ローカルGPUで実行可能）。

### 実装計画

#### Step 1: データ収集ループ

```python
# 既存の generate_self_play_data() を拡張
for game in range(num_games):
    # RolloutEvaluator (VN prefilter付き) で両者をプレイ
    steps, result = play_hand_with_evaluator(evaluator)

    for step in steps:
        # (state_522, turn, action_idx, reward, bust, fl) を保存
        save_trajectory(step, result)
```

`self_play.py` の `play_hand_with_mcts()` を `play_hand_with_evaluator()` に
分岐させるだけ。RolloutEvaluatorは `select_action()` を持つので差し替えは容易。

#### Step 2: Multi-Task VN訓練

```python
# 全ターン混合データで ValueNetworkV3 を訓練
# turn-embedding により各ターンの特性を自動学習
#   T0: 配置の初期品質を学ぶ
#   T1-T3: 中盤のドロー完成度を学ぶ
#   T4: 最終配置のバスト確率を学ぶ
train_value.py --data selfplay_all_turns.npz --epochs 200
```

#### Step 3: PolicyNet訓練

```python
# Self-Playの行動分布を教師として PolicyNet を更新
# 既存の train_selfplay.py がそのまま使える
#   policy_loss = KL(MCTS分布 || PolicyNet出力)
#   value_loss  = MSE(reward, VN予測)
```

#### Step 4: ゲートつき更新

```python
# 新モデル vs 旧モデル の対戦評価
# 既存の ai/mcts/evaluate.py が使える
win_rate = evaluate_models(new_policy, new_vn, old_policy, old_vn, 100)
if win_rate > 0.55:
    accept(new_model)  # 新モデル採用
```

### 教師信号の設計

| データ | Phase 1-2 (ソルバー) | Phase 3 (Self-Play) |
|--------|---------------------|---------------------|
| Value | ソルバーMC EV | ゲーム最終スコア |
| Bust | 盤面構造から判定 | 実際にバストしたか |
| FL | 盤面構造から判定 | 実際にFL入りしたか |
| Policy | なし（VNのみ） | 行動選択分布 |
| ターン | T0のみ | T0-T4全ターン |

### 鶏と卵問題の解決策

T1-T4のPolicyNetが弱いと、Self-Playの結果がノイジーになる問題：

1. **Phase 2のVN v2をプレイアウトに使う**
   - RolloutEvaluator は既にVN prefilter対応
   - `vn_truncate_depth` で途中からVN推定に切り替え可能
2. **BC事前学習でPolicyNetを初期化**
   - `generate_data.py` のヒューリスティック対戦で10万サンプル生成（数分）
   - BCで大まかな方策を学習 → Self-Playの出発点にする
3. **段階的温度下降**
   - 初期イテレーション: temperature=1.5（探索重視）
   - 後半: temperature=0.5（活用重視）

### 期待される効果

| 指標 | v2 (ソルバー) | v3 (Self-Play) |
|------|-------------|----------------|
| 教師データの質 | MC近似EV | 実ゲーム結果 |
| 学習対象ターン | T0のみ | **T0-T4全ターン** |
| 長期的視野 | なし | **T0→T4の因果関係** |
| Top-5命中率 | ~85% | ~99% |
| ソルバー速度 | ~10s/手 | ~5s/手 |
| 実行環境 | GCP | ローカルGPU |
| 所要期間 | +1-2日 | +1週間 |

---

## Phase 4: MCTS + VN 統合 — AlphaZero方式

### 考え方

Phase 3のRolloutEvaluator方式をさらに進化させ、
**MCTS内でVNを直接使って葉ノードを評価**する（AlphaZeroスタイル）。

```
MCTS (IS-MCTS + Progressive Widening)
  ├── PolicyNet → 候補行動のprior確率
  ├── VN v3 → 葉ノードの局面評価
  ├── UCB1(Q + c * P * √N / n) でノード選択
  └── visit-count分布 → 行動確率

Rollout不要 → 1000倍高速な self-play
```

### 既存MCTSエンジンとの統合

`ai/mcts/mcts.py` は既にこの構造を持っている：

```python
class MCTS:
    def _evaluate(self, node, root_obs):
        # VN で葉ノードを評価（既に実装済み）
        pred = self.value_net(state_tensor)
        raw_value = pred['value'].item() * self.score_std + self.score_mean
        return max(-1.0, min(1.0, raw_value / 30.0))
```

**改善点**:
1. VN v3の精度が十分なら、rolloutなしでMCTSが正しく機能する
2. 1シミュレーション = VN推論1回 (~0.01ms) vs rollout (~2s)
3. 200シミュレーション × 10決定点 = 1ゲーム **~0.5秒**

### Self-Playの加速

| 方式 | 1ゲーム所要時間 | 1万ゲーム |
|------|--------------|----------|
| RolloutEvaluator (Phase 3) | ~20s | ~56時間 |
| **MCTS + VN (Phase 4)** | ~0.5s | **~1.4時間** |

→ Phase 4ではSelf-Playを**40倍高速**に回せる
→ 1日で数万ゲームのデータを生成 → VNがさらに賢くなる好循環

### AlphaZero式フルループ

```
┌──────────────────────────────────────────┐
│  MCTS Self-Play (10,000ゲーム/日)        │
│    ↓                                    │
│  全ターンの (state, π, z) を蓄積        │
│    ↓                                    │
│  PolicyNet + VN を同時更新              │
│    policy: π_θ → π_MCTS (KL-div)       │
│    value:  v_θ → z (MSE)               │
│    ↓                                    │
│  新モデル vs 旧モデル (ゲート評価)       │
│    ↓                                    │
│  採用 → 次のイテレーション              │
└──────────────────────────────────────────┘
```

### 前提条件
- Phase 3で鍛えたVN v3が「枝刈りなしでも正しくノード評価できる精度」に到達
- PolicyNetがMCTS priorとして有効に機能
- IS-MCTSの不完全情報対応が正しく動作（既に実装済み）

---

## 比較まとめ

| | Phase 1 (v1) | Phase 2 (v2) | Phase 3 (v3) | Phase 4 (v4) |
|---|---|---|---|---|
| **方法** | ソルバーs50 | 好循環ループ | Self-Play (Rollout) | Self-Play (MCTS+VN) |
| **データソース** | 500手 T0 | 2000手 T0 | 数千ゲーム全ターン | 数万ゲーム全ターン |
| **学ぶもの** | MC近似EV | 高精度MC EV | 実ゲーム結果 | 実ゲーム結果 + MCTS分布 |
| **対象ターン** | T0のみ | T0のみ | T0-T4 | T0-T4 |
| **行動選択** | — | — | RolloutEvaluator | MCTS + VN |
| **実装コスト** | 低 | 低 | 中 | 中-高 |
| **所要期間** | ~1日 | +1-2日 | +1週間 | +2週間 |
| **計算コスト** | GCP ~$8 | GCP ~$5 | ローカルGPU 6h | ローカルGPU 2h |
| **ソルバー速度** | ~20s/手 | ~10s/手 | ~5s/手 | ~2s/手 |

---

## 推奨プラン

```
今日        Phase 1: データ収集完了 → VN v1訓練 → 精度検証
明日        Phase 1: ソルバー統合 (Python前処理方式)
            Phase 2: VN v1枝刈りで高精度データ生成開始
翌日        Phase 2: VN v2訓練 → 精度検証
1週間後     Phase 3: Self-Play (RolloutEvaluator)
              → BCでPolicyNet初期化
              → VN v2 prefilter付き RolloutEvaluator で1000ゲーム
              → VN v3 + PolicyNet v1 訓練
2週間後     Phase 4: MCTS + VN Self-Play
              → VN v3精度が十分なら rollout不要に
              → 10,000ゲーム/日のハイスループットループ
              → VN v4 + PolicyNet v2
```

**Phase 1→2は連続実行。Phase 3は1週間程度で到達可能。Phase 4はVN精度次第。**

---

## 長期ビジョン

```
現在のRustソルバー (130s/手, ドメイン枝刈り)
  ↓ Phase 1-2: VN枝刈り
高速ソルバー (10s/手, VN枝刈り)
  ↓ Phase 3: Self-Play学習
汎用エージェント (VN + PolicyNet, 全ターン対応)
  ↓ Phase 4: MCTS + VN
AlphaZero級エージェント (リアルタイム推論で最善手)
  ↓ 将来: 対人戦 WebApp統合
リアルタイムAI対戦 (< 1秒/手)
```
