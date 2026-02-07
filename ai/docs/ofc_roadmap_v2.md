# OFC Pineapple AI — 開発ロードマップ v2

---

## 全体像

```
[Phase 1] Webアプリでプレイログ収集（RYO 300-500ハンド）
    │                ↑ ログ形式の確認・変換スクリプト
    │                │
    ↓                │ 並行作業
[Phase 2] ロールアウト評価器の実装・テスト ←── MCTSを捨ててこれに置き換え
    │
    ↓
[Phase 3] BC再訓練（RYOデータ → PolicyNet）
    │
    ↓
[Phase 4] Self-Play（ロールアウト × 自己対戦でPolicyNet強化）
    │
    ↓
[Phase 5] RYO vs AI 対戦・チューニング
```

---

## 現在の手持ち（完成済み）

| ファイル | 役割 | 状態 |
|----------|------|------|
| `ai/engine/game_engine.py` | スコアリング、バスト判定、FL判定 | ✅ |
| `ai/engine/encoding.py` | 490次元の状態ベクトル | ✅ |
| `ai/engine/action_space.py` | Turn 0: 最大232通り、Turn 1-8: 9通り | ✅ |
| `ai/training/generate_data_fast.py` | ヒューリスティックデータ生成 | ✅ バグ修正済 |
| `ai/training/preprocess_fast.py` | JSONL → numpy変換 | ✅ 衝突修正済 |
| `ai/training/behavior_cloning.py` | BC訓練 | ✅ |
| `ai/mcts/evaluate.py` | モデル同士の対戦評価 | ✅ デッキペアリング済 |
| `ai/mcts/mcts.py` | MCTS | ❌ **廃止** → ロールアウトに置き換え |
| Webアプリ（frontend + backend） | 対人戦プレイ | ✅ |

---

## Phase 1: RYOのプレイデータ収集（Week 1-2）

### なぜ最優先か

今のBCは「低いカードをtop、高いカードをbottom」を学んだだけ。
RYOが無意識にやっている判断（フラッシュドロー追い、FL狙いの
リスク管理、相手ボードに応じた配置変更）は一切入っていない。
RYOのデータなしにSelf-Playを始めても、弱いスタート地点から
局所最適に嵌まるだけ。

### やること

**1. Webアプリのログ形式を確認**

BC訓練に必要な情報が全てログに含まれているか確認する：

```json
{
  "turn_log": {
    "turn": 3,
    "player": 0,
    "is_btn": true,
    "board_self": {"top": ["Ah"], "middle": ["8s","9s","Ts"], "bottom": ["Kd","Kc"]},
    "board_opponent": {"top": [], "middle": ["Td","Jd"], "bottom": ["As","Ks","Qs"]},
    "dealt_cards": ["5h", "X1", "Tc"],
    "discards_self": ["3d", "7c"],
    "action": {
      "placements": [["5h","middle"], ["Tc","bottom"]],
      "discard": "X1"
    }
  },
  "hand_result": {
    "busted": [false, true],
    "royalties": [{"top":0,"middle":4,"bottom":6,"total":10}, ...],
    "fl_entry": [false, false],
    "raw_score": [16, -16]
  }
}
```

足りない項目があれば、バックエンドのログ出力を修正する。

**2. ログ変換スクリプト**（新規実装が必要な場合）

Webアプリの保存形式（SQLite or 独自JSON）→ 上記JSONL形式に変換。

**3. プレイ**

- 2タブで自分 vs 自分で対戦
- **真剣にプレイする**（適当に置くとゴミデータになる）
- 目標: 300ハンド（最低）、500ハンド（理想）
- 1ハンド ≈ 3-5分 → 300ハンド ≈ 15-25時間
- 1日2時間で約2週間

---

## Phase 2: ロールアウト評価器の実装（Week 1-2、Phase 1と並行）

### 考え方

各ターンで候補アクション（Turn 1-8なら9通り）それぞれについて、
残りのゲームを何百回もランダムにシミュレートする。
平均スコアが最も高いアクションが最善手。

```
Turn 3の例:

手札: [Jh, 7c, 3d]
ボード: top[Ah] / middle[8s,9s,Ts] / bottom[Kd,Kc]

候補A: Jh→bottom, 7c→middle, 捨て3d
  → 残り6ターンを1000回シミュレート → 平均スコア: +4.2

候補B: Jh→middle, 7c→top, 捨て3d
  → 残り6ターンを1000回シミュレート → 平均スコア: +1.8

候補C: Jh→bottom, 3d→top, 捨て7c
  → 残り6ターンを1000回シミュレート → 平均スコア: +5.1

→ 候補Cを選択（KKJの形 + middleのストレートドローを温存）
```

### 新規ファイル: `ai/mcts/rollout_evaluator.py`

```python
class RolloutEvaluator:

    def __init__(self, policy_net, n_rollouts=200, top_k=20):
        self.policy_net = policy_net    # プレイアウト用
        self.n_rollouts = n_rollouts    # 各候補の試行回数
        self.top_k = top_k             # Turn 0の候補絞り込み

    def select_action(self, obs):
        """
        1. 候補アクションを列挙
           - Turn 0: 最大232通り → PolicyNetで上位K個に絞る
           - Turn 1-8: 9通り → 全候補を評価
        2. 各候補をN回ロールアウトして平均スコアを計算
        3. 平均スコア最大のアクションを返す
        """

    def _rollout(self, obs, action):
        """
        1つのロールアウト:
        1. アクションを適用してボード更新
        2. unseen cards（デッキ残り + 相手手札）をシャッフル
        3. 残りターンをPolicyNetで自分と相手の両方をプレイ
        4. ハンド完了 → game_engine.pyでスコア計算
        5. スコアを返す
        """
```

### 速度見積もり（GPU）

| ターン | 候補数 | ロールアウト | 所要時間 |
|--------|--------|-------------|----------|
| Turn 0 | 20（絞り込み後） | ×200回 | ~4秒 |
| Turn 1-8 | 9 | ×200回 | ~2秒 |

対人戦の20秒制限に余裕で収まる。
Self-Play用にはロールアウト100回に減らせば半分の時間。

### テスト

- ロールアウト評価器 vs PolicyNet単体（greedy）を対戦
- ロールアウト側が明確に勝つことを確認（数ターン先を見ているため）
- ロールアウト回数を変えて精度 vs 速度のトレードオフを把握

---

## Phase 3: BC再訓練（Week 2-3）

### やること

RYOのプレイデータで PolicyNet と ValueNet を訓練し直す。

```bash
# 1. 前処理
python ai/training/preprocess_fast.py data/ryo_hands.jsonl --output data/processed_ryo

# 2. BC訓練
python ai/training/behavior_cloning.py --data data/processed_ryo --epochs 200

# 3. 評価（旧ヒューリスティックBC vs 新RYO BC）
python ai/mcts/evaluate.py --model_a checkpoints/ryo_bc.pt --model_b checkpoints/heuristic_bc.pt
```

### 精度の目安

| 指標 | ヒューリスティックBC | RYO BC（期待値） |
|------|---------------------|------------------|
| Top-1 | 85%（偽の高精度） | 40-60% |
| Top-3 | 99% | 80-90% |
| 対戦勝率 | 基準 | 60-70% |

Top-1が下がるのは正常。人間のプレイは同じ局面でも複数の正解がある。
重要なのは**対戦勝率が上がること**。

---

## Phase 4: Self-Play（Week 3-5）

### 概要

ロールアウト評価器を使って自己対戦し、PolicyNetを強化する。

### 修正ファイル: `ai/training/train_selfplay.py`

MCTS呼び出しを RolloutEvaluator に差し替える。

### 訓練ループ（1イテレーション）

```
1. データ収集: 200ゲーム自己対戦
   - 各ターンでRolloutEvaluatorが全候補を評価
   - 記録: (状態, 各アクションの平均スコア, ゲーム結果)

2. PolicyNet更新
   - 教師: ロールアウトで得た「各アクションの平均スコア」→ ソフトマックスで分布に変換
   - PolicyNetがロールアウトの判断を内在化する
   - → 次のイテレーションではPolicyNetが賢くなっている
   - → ロールアウトのプレイアウトも賢くなる
   - → さらに良い教師データが生まれる（好循環）

3. 評価
   - 新モデル vs 旧モデル（デッキペアリング400ハンド）
   - 勝率55%以上なら採用、そうでなければ棄却
```

### なぜ強くなるか

```
イテレーション 0:  PolicyNet = RYOの模倣
                   ロールアウト = RYOの模倣で1000回シミュレーション
                   → RYOより少し強い（計算ミスがないだけ）

イテレーション 20: PolicyNet = 改善済み
                   ロールアウト = 改善済みポリシーで1000回シミュレーション
                   → さらに強い

イテレーション 50: PolicyNet = かなり最適化
                   ロールアウト = 最適化ポリシーでシミュレーション
                   → RYOを超える可能性
```

### 所要時間

```
1イテレーション:
  データ収集: 200ゲーム × 9ターン × 2秒/ターン ≈ 60分
  訓練 + 評価: 15分
  合計: ~75分

50イテレーション ≈ 63時間 ≈ GPU 3日
100イテレーション ≈ 125時間 ≈ GPU 5日
```

---

## Phase 5: 評価・チューニング（Week 5+）

### RYO vs AI 直接対戦

WebアプリにAIプレイヤーを接続して、RYOが実際に対戦する。

### チューニング項目

- ロールアウト回数（100 / 200 / 500）
- Turn 0の候補絞り込み数（10 / 20 / 30）
- Self-Playの温度パラメータ
- 弱点が見つかれば追加データ収集 → 再訓練

---

## 新規実装ファイル一覧

| # | ファイル | 内容 | いつ |
|---|----------|------|------|
| 1 | `ai/training/collect_human_data.py` | Webアプリログ → BC用JSONL変換 | Phase 1 |
| 2 | `ai/mcts/rollout_evaluator.py` | ロールアウトベースのアクション評価器 | Phase 2 |
| 3 | `ai/player/rollout_player.py` | WebSocket AIプレイヤー（対人戦用） | Phase 5 |

### 修正が必要な既存ファイル

| ファイル | 修正内容 | いつ |
|----------|----------|------|
| `ai/training/train_selfplay.py` | MCTS → RolloutEvaluator に差し替え | Phase 4 |
| バックエンド（ログ出力） | BC訓練に必要な項目が揃っているか確認 | Phase 1 |

### 廃止

| ファイル | 理由 |
|----------|------|
| `ai/mcts/mcts.py` | OFCでは木構造が不要。ロールアウトで十分 |

---

## タイムライン

```
Week 1  ┬─ [Phase 1] プレイ開始（毎日2時間）
        └─ [Phase 2] rollout_evaluator.py 実装・テスト

Week 2  ┬─ [Phase 1] プレイ継続（300ハンド到達）
        ├─ [Phase 2] ロールアウト vs Greedy で動作確認
        └─ [Phase 3] BC再訓練（RYOデータ）

Week 3  ┬─ [Phase 1] プレイ継続（500ハンド目標）
        └─ [Phase 4] Self-Play開始（GPU常時稼働）

Week 4  ── [Phase 4] Self-Play継続（50イテレーション到達）

Week 5  ── [Phase 5] RYO vs AI 対戦・チューニング
```

### 最短: 3週間（300ハンド + 50イテレーション）
### 現実的: 4-5週間（500ハンド + 100イテレーション + 調整）

---

## 最初にやること（今日）

1. Webアプリのログに必要項目が揃っているか確認
2. 足りなければバックエンド修正
3. プレイ開始
4. 並行して `rollout_evaluator.py` の設計・実装に着手
