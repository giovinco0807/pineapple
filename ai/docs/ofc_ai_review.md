# OFC Pineapple AI — コードレビュー・修正事項

## 概要

全ファイルを通したレビュー結果。優先度順に修正すべき問題点と具体的な修正案をまとめる。

---

## 修正一覧（優先度順）

| 優先度 | ファイル | 問題 | 影響 |
|--------|----------|------|------|
| **高** | `preprocess_fast.py` | Turn 0 ハッシュ衝突 | BC学習ラベルが破損 |
| **高** | `generate_data_fast.py` | FL判定閾値が間違い | fl_entry教師ラベルが不正 |
| **高** | `generate_data_fast.py` | raw_scoreが常に0（両者ノーバスト時） | 報酬信号の欠落 |
| **中** | `generate_data_fast.py` | heuristic_turnのランダムdiscard | ノイズの学習 |
| **中** | `mcts.py` | 深さ2以降が無意味 | MCTSが実質1手先評価のみ |
| **中** | `mcts.py` | composite valueの係数が未調整 | Value評価の精度 |
| **低** | `evaluate.py` | 50ハンドでは分散が大きすぎる | 評価が不安定 |
| **低** | `evaluate.py` | デッキペアリング未実装 | 運の影響が大きい |
| **低** | `game_engine.py` | フルハウス判定のエッジケース | ユニットテスト推奨 |

---

## 1. `preprocess_fast.py` — アクションインデックスの衝突【高】

### 問題

`bucket_size = 10` に `_stable_hash % 10` でマッピングしているが、同一パーティション内のカード組み合わせ数がバケットサイズを超える。

```python
bucket_size = 10
sub_idx = _stable_hash(cards_key) % bucket_size
return min(partition_id * bucket_size + sub_idx, MAX_ACTIONS - 1)
```

例：パーティション (1,2,2) の場合
- top に置く1枚の選び方: C(5,1) = 5
- 残り4枚から mid に置く2枚: C(4,2) = 6
- 残り2枚は bot: 1通り
- 合計: 5 × 6 = **30通り** → 10スロットに入れるので**衝突率約67%**

### 影響

異なるアクションが同じインデックスにマッピングされ、BCが「異なる手を同じラベル」として学習する。Top-1精度85%の一部はこの衝突による偽の正解。

### 修正案

`action_space.py` の `get_initial_actions` と同じ列挙順序でインデックスを決定的に割り当てる。

```python
def action_to_index(action: dict, turn: int, dealt_cards: list = None, board: dict = None) -> int:
    placements = action["placements"]

    if turn == 0:
        from ai.engine.encoding import Board
        from ai.engine.action_space import get_initial_actions

        # 教師データのアクションを正規化
        by_pos = {"top": [], "middle": [], "bottom": []}
        for card, pos in placements:
            by_pos[pos].append(card)
        target = (
            tuple(sorted(by_pos["top"])),
            tuple(sorted(by_pos["middle"])),
            tuple(sorted(by_pos["bottom"])),
        )

        # get_initial_actions と同じ順序で列挙し、一致するインデックスを返す
        b = Board.from_dict(board) if board else Board()
        all_actions = get_initial_actions(dealt_cards, b)
        for idx, a in enumerate(all_actions):
            a_pos = {"top": [], "middle": [], "bottom": []}
            for card, pos in a.placements:
                a_pos[pos].append(card)
            a_key = (
                tuple(sorted(a_pos["top"])),
                tuple(sorted(a_pos["middle"])),
                tuple(sorted(a_pos["bottom"])),
            )
            if a_key == target:
                return min(idx, MAX_ACTIONS - 1)
        return 0  # fallback
    else:
        # Turn 1-8: 9通りなので問題なし
        positions = ["top", "middle", "bottom"]
        pos0 = placements[0][1] if len(placements) > 0 else "top"
        pos1 = placements[1][1] if len(placements) > 1 else "top"
        return positions.index(pos0) * 3 + positions.index(pos1)
```

> **注意:** この修正により前処理速度が低下する（毎回アクション列挙が発生）。高速化が必要なら、カード→パーティション内の辞書順インデックスを計算するロジックを別途実装する。

---

## 2. `generate_data_fast.py` — FL判定の閾値が不正【高】

### 問題

```python
if royalties[seat]["top"] >= 3:  # QQ+
    fl_entry[seat] = True
```

Top royalty 3 は **88のペア**（8 - 5 = 3）。QQのロイヤリティは **7**（12 - 5 = 7）。

ロイヤリティ対応表：
| ペア | ランク値 | ロイヤリティ (rank - 5) |
|------|----------|------------------------|
| 66   | 6        | 1                      |
| 77   | 7        | 2                      |
| 88   | 8        | **3** ← 現在の閾値    |
| 99   | 9        | 4                      |
| TT   | 10       | 5                      |
| JJ   | 11       | 6                      |
| **QQ** | 12     | **7** ← 正しい閾値    |
| KK   | 13       | 8                      |
| AA   | 14       | 9                      |

### 影響

88, 99, TT, JJ のペアでも FL entry = True になり、教師データの `fl_entry` ラベルが大量に誤判定される。

### 修正案

```python
# 方法1: 閾値を修正
if royalties[seat]["top"] >= 7:  # QQ+
    fl_entry[seat] = True

# 方法2: game_engine.py の check_fl_entry を直接使う（より正確）
from ai.engine.game_engine import check_fl_entry
fl, cards = check_fl_entry(b["top"])
fl_entry[seat] = fl
```

---

## 3. `generate_data_fast.py` — raw_scoreが常に0【高】

### 問題

```python
if not busted[0] and not busted[1]:
    raw_score = [0, 0]  # Simplified
```

両者ノーバストの場合にスコアが常に0。ライン勝敗、スクープボーナス、ロイヤリティ差分が全て無視されている。

### 影響

`raw_score` を参照する訓練パイプライン（例：Self-Playの報酬計算）で、両者ノーバスト時の報酬信号が完全に欠落する。

### 修正案

`game_engine.py` の `compute_result` と同じロジックでライン比較を実装。

```python
if not busted[0] and not busted[1]:
    line_vals = [{}, {}]
    for seat in [0, 1]:
        b = boards[seat]
        line_vals[seat] = {
            "top": quick_eval(b["top"], 3),
            "middle": quick_eval(b["middle"], 5),
            "bottom": quick_eval(b["bottom"], 5),
        }

    line_total = 0
    for line in ["top", "middle", "bottom"]:
        if line_vals[0][line] > line_vals[1][line]:
            line_total += 1
        elif line_vals[0][line] < line_vals[1][line]:
            line_total -= 1

    scoop_bonus = 3 if abs(line_total) == 3 else 0
    p0 = line_total
    p0 += scoop_bonus if line_total > 0 else (-scoop_bonus if line_total < 0 else 0)
    p0 += royalties[0]["total"] - royalties[1]["total"]
    raw_score = [p0, -p0]
```

---

## 4. `generate_data_fast.py` — ランダムdiscard【中】

### 問題

```python
# Sometimes discard mid instead of lowest (add randomness)
if random.random() < 0.3:
    discard = sorted_cards[1]
    remaining = [sorted_cards[0], sorted_cards[2]]
```

30%の確率で中間カードを捨てる。これは「戦略的多様性」ではなく「ノイズ」。

### 影響

BCがこのランダム行動も正解として学習し、不必要に悪い手を覚える。

### 修正案

ランダム性を除去し、戦略的な分岐のみ残す。

```python
def heuristic_turn(cards: list, board: dict) -> tuple:
    sorted_cards = sorted(cards, key=card_rank)

    # ペアが作れる場合はペアを残す
    ranks = [card_rank(c) for c in sorted_cards]
    board_ranks = [card_rank(c) for c in
                   board["top"] + board["middle"] + board["bottom"]
                   if c not in ("X1", "X2")]

    # ボード上のカードとペアになるカードを優先して残す
    pair_candidates = [c for c in sorted_cards if card_rank(c) in board_ranks]
    if pair_candidates:
        keep = pair_candidates[0]
        others = [c for c in sorted_cards if c != keep]
        discard = min(others, key=card_rank)
        remaining = [c for c in sorted_cards if c != discard]
    else:
        discard = sorted_cards[0]  # 最低ランクを捨てる
        remaining = sorted_cards[1:]

    # ... 以下配置ロジック
```

---

## 5. `mcts.py` — 木の深さが実質1【中】

### 問題

OFCでは1ターンに1回のアクション。ルートの子ノード展開後、`_expand_leaf` で子ノードをさらに展開しようとするが：

- Turn 1-8: 3枚中2枚を配置済み → `get_turn_actions` は3枚を要求するので `AssertionError` か空リスト
- Turn 0: 5枚全て配置済み → 再列挙は無意味

**深さ2以降のノードは有効なアクションを持てず、木は実質深さ1。**

### 修正案

#### 方向A: 深さ1で割り切る（推奨・短期）

`_expand_leaf` の呼び出しを削除し、simulations をルートの子ノード間のbandit問題として回す。

```python
def _evaluate(self, node: MCTSNode, root_obs: Observation) -> float:
    obs = self._build_node_observation(node, root_obs)
    if obs is None:
        return 0.0

    state_vec = encode_state(obs)
    state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)

    with torch.no_grad():
        pred = self.value_net(state_tensor)

    royalty = pred["royalty_ev"].item()
    bust = pred["bust_prob"].item()
    fl = pred["fl_prob"].item()
    value = royalty * 0.5 - bust * 6.0 + fl * 10.0
    return max(-1.0, min(1.0, value / 20.0))
    # _expand_leaf を呼ばない → 深さ1固定
```

この場合、木探索というよりは **Policy prior + Value evaluationのUCB-based action selection** になる。BC + ValueNet の品質が十分なら、これだけでも効果的。

#### 方向B: Determinizationで将来ターンをシミュレート（長期）

各simulationで以下を行う：

1. unseen_cardsからランダムに次ターンの3枚をサンプリング
2. そのカードで新しいアクション候補を列挙
3. PolicyNetで選択してボードを更新
4. 繰り返し（ハンド完了 or 深さ上限まで）
5. 最終ボードをValueNetで評価

これは本格的なIS-MCTS実装で、大幅な改修が必要。

```
search()
  └── simulation (×200)
        ├── select child at root (UCB)
        ├── apply action → board updated
        ├── sample next 3 cards from unseen
        ├── get_turn_actions → policy select → apply
        ├── ... repeat until hand complete or depth limit
        └── evaluate final board → backpropagate
```

---

## 6. `mcts.py` — Composite valueの係数が未調整【中】

### 問題

```python
value = royalty * 0.5 - bust * 6.0 + fl * 10.0
```

この係数はヒューリスティックで、実際のゲーム結果との相関が未検証。

### 修正案

意味的に整合した係数に変更：

```python
# bust時のペナルティ: -6（スクープ負け） - 相手royalty平均(~5) ≒ -11
# FL entry ボーナス: FL成功時の追加期待スコア ≒ +8
# royalty: そのまま加算（正規化は外側で）
value = royalty - bust * 11.0 + fl * 8.0
```

自己対戦データが溜まったら、回帰分析で最適係数を求める：

```python
# 回帰で係数を求める例
from sklearn.linear_model import LinearRegression
# X: [royalty_ev, bust_prob, fl_prob] (各ハンドのBC予測)
# y: actual_game_result (実際のスコア)
reg = LinearRegression().fit(X, y)
# → reg.coef_ が最適係数
```

---

## 7. `evaluate.py` — 評価の統計的信頼性【低】

### 7a. サンプル数が少ない

```python
# train_selfplay.py から
win_rate = evaluate_models(..., num_games=50, ...)
```

OFCは1ハンドのスコア分散が大きい（ロイヤリティで±50が発生）。50ハンドではノイズに埋もれる。

**修正案:** 最低200ハンド、理想的には500ハンド。勝率だけでなく平均スコア差も判定基準に使う。

```python
avg_margin = (score_a - score_b) / max(num_games, 1)
is_improved = avg_margin > 0.5  # 1ハンドあたり0.5点以上有利なら改善と判定
```

### 7b. デッキペアリング未実装

各ゲームでランダムにデッキを生成しており、運の影響が大きい。

**修正案:** 同じデッキ順序で席を入れ替えて2ゲームプレイする「ペアリング」方式。

```python
for game_idx in range(num_games):
    deck = list(ALL_CARDS)
    random.shuffle(deck)

    # 同じデッキでA=seat0, B=seat1
    result_1 = play_eval_hand_with_deck(policy_a, policy_b, deck, seat_a=0)

    # 同じデッキでA=seat1, B=seat0
    result_2 = play_eval_hand_with_deck(policy_a, policy_b, deck, seat_a=1)

    # 両結果を集計 → 運の要素が相殺される
```

---

## 8. `game_engine.py` — フルハウス判定のエッジケース【低】

### 問題

ジョーカー絡みのフルハウス判定にエッジケースがある可能性。現在の条件分岐：

```python
# Case 1: AAA KK (natural trips + natural pair)
if best_count >= 3 and len(rank_counts) == 2:
    return 7000 + ...

# Case 2: AAA + 2 jokers
if best_count >= 3 and len(rank_counts) == 1 and jokers >= 2:
    return 7000 + ...

# Case 3: AA KK + joker → AAA KK
if best_count >= 2 and jokers >= 1 and len(rank_counts) == 2:
    return 7000 + ...
```

### 影響

大きなバグは確認できていないが、以下のエッジケースのユニットテストを推奨：

```python
# テストすべきケース
assert evaluate_hand(["Ah", "Ad", "Ac", "Kh", "Kd"], 5) >= 7000  # AAA KK → FH
assert evaluate_hand(["Ah", "Ad", "Kh", "Kd", "X1"], 5) >= 7000  # AA KK + joker → FH
assert evaluate_hand(["Ah", "Ad", "Ac", "X1", "X2"], 5) >= 7000  # AAA + 2 jokers → FH
assert evaluate_hand(["Ah", "Ad", "Kh", "Qs", "X1"], 5) < 7000   # AA K Q + joker → trips, not FH
assert evaluate_hand(["Ah", "Kh", "Qh", "Jh", "X1"], 5) >= 9000  # flush + straight + joker → SF
assert evaluate_hand(["Ah", "Ad", "Ac", "Ad", "Kh"], 5) >= 8000  # impossible but test gracefully
```

---

## 今後の開発ロードマップ

### Phase 1: データ品質修正（今すぐ）
1. `preprocess_fast.py` のアクションインデックス衝突を修正
2. `generate_data_fast.py` のFL判定・raw_score・ランダムdiscardを修正
3. データを再生成して BC を再訓練

### Phase 2: MCTS整備（BC再訓練後）
4. `mcts.py` を方向A（深さ1）で簡素化
5. Composite value の係数を調整
6. `evaluate.py` の信頼性向上（ハンド数増加 + ペアリング）

### Phase 3: Self-Play開始
7. 修正済みのBCモデル + 整備済みMCTS で自己対戦
8. ValueNetの `value` ヘッドが実データで学習開始
9. Composite value → 学習済み value head への切り替え

### Phase 4: 本格探索（長期）
10. Determinization 実装（方向B）
11. IS-MCTS で複数ターン先の探索
12. Elo rating による継続的な強さ測定
