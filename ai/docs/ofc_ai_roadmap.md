# OFC Pineapple AI — 開発ロードマップ

## 現状まとめ

### 完成しているもの
- ゲームエンジン (`game_engine.py`) — スコアリング、ロイヤリティ、バスト判定、FL判定
- 状態エンコーディング (`encoding.py`) — 490次元ベクトル
- アクション空間 (`action_space.py`) — Turn 0: 最大232通り、Turn 1-8: 9通り
- PolicyNet / ValueNet (`networks.py`) — MLP構造
- BC訓練パイプライン (`behavior_cloning.py`) — 訓練済み（ヒューリスティック教師）
- データ生成 (`generate_data_fast.py`) — バグ修正済み
- 前処理 (`preprocess_fast.py`) — 衝突修正済み
- 評価 (`evaluate.py`) — デッキペアリング実装済み
- Webアプリ（フロントエンド + バックエンド） — 対人戦プレイ可能

### 捨てるもの
- `mcts.py` — OFCではMCTSの木構造が不要。シンプルロールアウトに置き換え

### 新しく作るもの
- `rollout_evaluator.py` — ロールアウトベースのアクション評価器
- `collect_human_data.py` — Webアプリのログをjsonlに変換（なければ）
- `train_selfplay.py` の修正 — MCTS → ロールアウト方式に変更

---

## 全体フロー

```
Phase 1: RYOのプレイデータ収集（1-2週間）
    ↓
Phase 2: BC訓練（RYOデータで再訓練）（1日）
    ↓
Phase 3: ロールアウト評価器の実装・テスト（2-3日）
    ↓
Phase 4: Self-Play訓練（1-2週間）
    ↓
Phase 5: 評価・チューニング（継続）
```

---

## Phase 1: RYOのプレイデータ収集

### 目的
弱いヒューリスティックの代わりに、RYOの実際の判断パターンを教師データにする。
フラッシュドロー追い、FL狙いのタイミング、相手に応じた配置変更 — これらは
ヒューリスティックでは表現できない。

### やること
1. Webアプリで2タブを開いて自分 vs 自分で対戦
2. 全ターンのプレイログが自動でDB/JSONLに保存される
3. 目標: **最低300ハンド、理想500ハンド**

### 必要な確認
- Webアプリのログ出力がBC訓練に必要な形式を含んでいるか確認：
  - `turn` (ターン番号)
  - `player` (seat)
  - `board_self` (配置前のボード状態)
  - `board_opponent` (相手ボード)
  - `dealt_cards` (配られたカード)
  - `action` (placements + discard)
  - `hand_result` (busted, royalties, fl_entry, raw_score)

### ログ変換（必要なら新規実装）
```
Webアプリ DB/ログ → JSONL（generate_data_fast.pyと同じ形式）
```
ファイル: `ai/training/collect_human_data.py`

### 時間見積もり
- 1ハンド ≈ 3-5分（考えながらプレイ）
- 300ハンド ≈ 15-25時間
- 1日2時間で1-2週間

### 並行作業
データ収集中に Phase 3（ロールアウト評価器の実装）を進める。

---

## Phase 2: BC再訓練（RYOデータ）

### やること
1. RYOのプレイログを前処理
   ```bash
   python ai/training/preprocess_fast.py data/ryo_300hands.jsonl --output data/processed_ryo
   ```

2. BC訓練
   ```bash
   python ai/training/behavior_cloning.py --data data/processed_ryo --epochs 200
   ```

3. 品質確認
   - Top-1精度は40-60%で十分（人間のプレイは状況依存で多様なため）
   - Top-3精度が80%以上あれば良い
   - 旧モデル（ヒューリスティックBC）vs 新モデル（RYO BC）を対戦させて確認

### 期待される改善
| 指標 | ヒューリスティックBC | RYO BC |
|------|---------------------|--------|
| フラッシュ/ストレート追い | ❌ | ✅ |
| 相手ボードを見た判断 | ❌ | ✅ |
| FL狙いのリスク管理 | ❌ | ✅ |
| バスト回避 | △（単純ルール） | ✅ |

### 所要時間
- 前処理: 数分
- BC訓練: 1-2時間（GPU）
- 評価: 30分

---

## Phase 3: ロールアウト評価器の実装

### 概要
MCTSを捨てて、シンプルなロールアウトでアクションを評価する。

### 新規ファイル: `ai/mcts/rollout_evaluator.py`

### アルゴリズム

```
入力: 現在のObservation（ボード状態 + 手札）
出力: 最善のアクション

1. 候補アクションを列挙
   - Turn 0: 最大232通り → PolicyNetで上位K個に絞る（K=20）
   - Turn 1-8: 最大9通り → 全候補を評価

2. 各候補アクションについて:
   a. アクションを仮適用 → ボード更新
   b. N回ロールアウト（N=100〜1000）:
      - 見えていないカード（unseen cards）をシャッフル
      - 残りターンをPolicyNetで最後までプレイ
      - 相手もPolicyNetでプレイ（推定）
      - ハンド完了 → スコア計算
   c. 平均スコアを記録

3. 平均スコアが最も高いアクションを選択
```

### 実装の詳細

```python
class RolloutEvaluator:
    def __init__(self, policy_net, n_rollouts=200, top_k=20, device="cpu"):
        self.policy_net = policy_net
        self.n_rollouts = n_rollouts
        self.top_k = top_k  # Turn 0用
        self.device = device

    def select_action(self, obs: Observation) -> Tuple[int, Action]:
        """最善のアクションを選択"""
        valid_actions = self._get_valid_actions(obs)

        # Turn 0はPolicyNetで候補を絞る
        if obs.turn == 0 and len(valid_actions) > self.top_k:
            candidates = self._filter_top_k(obs, valid_actions)
        else:
            candidates = list(enumerate(valid_actions))

        # 各候補をロールアウトで評価
        best_idx, best_score = -1, float('-inf')
        for orig_idx, action in candidates:
            avg_score = self._evaluate_action(obs, action)
            if avg_score > best_score:
                best_score = avg_score
                best_idx = orig_idx

        return best_idx, valid_actions[best_idx]

    def _evaluate_action(self, obs, action) -> float:
        """1つのアクションをN回ロールアウトして平均スコアを返す"""
        total_score = 0

        for _ in range(self.n_rollouts):
            # unseen cardsを作成
            unseen = self._get_unseen_cards(obs, action)
            random.shuffle(unseen)

            # アクション適用後のボードからスタート
            board = self._apply_action(obs.board_self, action)

            # 相手のボードは現在の状態から
            opp_board = obs.board_opponent.copy()

            # 残りターンをPolicyNetでプレイアウト
            score = self._playout(board, opp_board, unseen, obs.turn)
            total_score += score

        return total_score / self.n_rollouts

    def _playout(self, my_board, opp_board, unseen, current_turn) -> float:
        """残りターンを最後までプレイして最終スコアを返す"""
        card_idx = 0

        for turn in range(current_turn + 1, 9):
            # 自分に3枚配る
            if card_idx + 3 > len(unseen):
                break
            my_cards = unseen[card_idx:card_idx + 3]
            card_idx += 3

            # PolicyNetで自分のアクション選択
            if not my_board.is_complete():
                my_obs = Observation(board_self=my_board, ...)
                action = self._policy_select(my_obs, my_cards)
                self._apply_action_inplace(my_board, action)

            # 相手に3枚配る
            if card_idx + 3 > len(unseen):
                break
            opp_cards = unseen[card_idx:card_idx + 3]
            card_idx += 3

            # PolicyNetで相手のアクション選択
            if not opp_board.is_complete():
                opp_obs = Observation(board_self=opp_board, ...)
                action = self._policy_select(opp_obs, opp_cards)
                self._apply_action_inplace(opp_board, action)

        # 最終スコア計算
        return compute_final_score(my_board, opp_board)
```

### 速度最適化

| 手法 | 効果 |
|------|------|
| PolicyNet推論をバッチ化 | 10倍速 |
| Turn 0のtop_kを調整 | K=10なら半分の時間 |
| ロールアウト数を用途別に変更 | 対人戦: 500回、Self-Play: 100回 |
| GPU使用 | CPU比5-10倍速 |

### 速度見積もり（GPU）

```
1回のロールアウト:
  残りターン数(平均5) × PolicyNet推論2回(自分+相手) = 10回のforward
  10回 × 0.1ms = 1ms/ロールアウト

Turn 1-8（9候補 × 200ロールアウト）:
  9 × 200 × 1ms = 1.8秒/ターン

Turn 0（20候補 × 200ロールアウト）:
  20 × 200 × 1ms = 4秒/ターン
```

20秒の制限時間内に余裕で収まる。

### テスト方法
1. ロールアウト評価器 vs PolicyNet単体を対戦させる
2. ロールアウト評価器が明らかに勝つはず（数ターン先を見ているため）
3. ロールアウト回数を変えて精度vs速度のトレードオフを確認

---

## Phase 4: Self-Play訓練

### 概要
ロールアウト評価器を使って自己対戦し、PolicyNetとValueNetを強化する。

### 修正が必要なファイル: `ai/training/train_selfplay.py`
- MCTS呼び出しをRolloutEvaluator呼び出しに変更
- 訓練データの形式は同じ（状態, 行動分布, ゲーム結果）

### 訓練ループ

```
イテレーション 1-100:

  1. データ収集（自己対戦）
     - 現在のPolicyNet + RolloutEvaluator で 200ゲーム対戦
     - 各ターンで記録:
       - 状態ベクトル（490次元）
       - 各アクション候補の平均スコア → ソフトマックスで行動分布に変換
       - ゲーム最終結果（raw_score）

  2. PolicyNet更新
     - 教師: ロールアウトで得た行動分布（ソフトターゲット）
     - 損失: KLダイバージェンス
     - → PolicyNetがロールアウトの判断を内在化

  3. ValueNet更新
     - 教師: 各状態での実際のゲーム結果
     - 損失: MSE
     - → ValueNetが「この状態は有利/不利」を学習

  4. 評価
     - 新モデル vs 旧モデルを 200ゲーム対戦（デッキペアリング）
     - 勝率55%以上 or 平均スコア差 > +0.5 なら新モデル採用
     - そうでなければ旧モデルを維持

  5. チェックポイント保存
```

### Self-Playの効果

```
イテレーション 0（BC直後）:
  PolicyNet = RYOの模倣
  ロールアウト = RYOの模倣 × 200回シミュレーション
  → RYOより少し強い（計算ミスがない分）

イテレーション 10:
  PolicyNet = 改善されたRYO
  ロールアウト = 改善されたRYO × 200回シミュレーション
  → さらに強い

イテレーション 50:
  PolicyNet = かなり最適化された戦略
  ロールアウト = 最適化された戦略 × 200回シミュレーション
  → 人間を超える可能性
```

### ポイント: Self-Playが機能する条件
- **PolicyNetの初期品質が重要** → だからRYOのデータでBCする
- **ロールアウト回数が十分** → 100回以上で安定
- **評価が正確** → デッキペアリング400ハンドで判定

### 所要時間

```
1イテレーション:
  200ゲーム × 9ターン × 1.8秒/ターン = 約55分（データ収集）
  PolicyNet訓練: 5分
  ValueNet訓練: 5分
  評価200ゲーム: 10分
  合計: 約75分/イテレーション

100イテレーション = 約125時間 = 5日（GPU常時稼働）

現実的には:
  50イテレーションで十分な強さに到達する可能性あり → 2-3日
```

---

## Phase 5: 評価・チューニング

### Elo Rating
- 各イテレーションのモデルにEloを付与
- RYO自身の推定Elo（1500基準）と比較
- 目標: RYOのElo + 200以上

### パラメータチューニング
- ロールアウト回数: 100, 200, 500 で比較
- Top-K（Turn 0）: 10, 20, 30 で比較
- Self-Playの温度パラメータ
- 訓練のlearning rate、バッチサイズ

### RYO vs AI対戦
- Webアプリに AI プレイヤーを接続
- RYOが実際にAIと対戦して強さを体感
- AIの弱点を発見 → 追加データ収集 → 再訓練

---

## 新規実装ファイル一覧

| ファイル | 内容 | 優先度 |
|----------|------|--------|
| `ai/mcts/rollout_evaluator.py` | ロールアウトベースのアクション評価器 | **高** |
| `ai/training/collect_human_data.py` | Webアプリログ → JSONL変換 | **高** |
| `ai/training/train_selfplay.py` | Self-Play訓練（MCTS→ロールアウト） | **中** |
| `ai/player/ai_player.py` | WebSocket AIプレイヤー（対人戦用） | **低**（Phase 5） |

### 修正が必要な既存ファイル

| ファイル | 修正内容 |
|----------|----------|
| `ai/mcts/mcts.py` | 削除 or 保守のみ（使わない） |
| `ai/training/behavior_cloning.py` | RYOデータ用にデータローダー調整（必要なら） |

---

## タイムライン

```
Week 1-2: Phase 1（データ収集）+ Phase 3（ロールアウト実装）
  - RYOが毎日2時間プレイ → 300ハンド蓄積
  - 並行してrollout_evaluator.pyを実装・テスト

Week 2: Phase 2（BC再訓練）
  - 300ハンド溜まったらBC再訓練
  - ヒューリスティックBC vs RYO BC で対戦確認

Week 2-4: Phase 4（Self-Play）
  - GPU常時稼働で50-100イテレーション
  - 途中で評価しながら進捗確認

Week 4: Phase 5（評価・対戦）
  - RYO vs AI の直接対戦
  - チューニング・追加訓練
```

### 最短ケース（全てうまくいった場合）
- 3週間でRYOに勝てるAIが完成

### 現実的ケース
- 4-5週間で安定してRYOに勝てるレベル
