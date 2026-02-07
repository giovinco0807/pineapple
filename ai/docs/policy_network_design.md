# Policy Network + 探索 実装設計

## 概要

**目標**: NNで候補を絞り込み、Exhaustiveで精密評価

```
従来: 100万通り全探索 → 遅い
新方式: NN → 1000候補 → 1000通り精密評価 → 高速 + 高精度
```

---

## アーキテクチャ

### Phase 1: Top予測NN

```
入力: 14枚の手札 (14 × 18 = 252次元)
      ↓
    [Policy Network]
      ↓
出力: Top候補スコア (C(14,3) = 364通り)
      ↓
    上位k個を選択
```

### Phase 2: Middle予測NN

```
入力: 残り11枚 + 選択されたTop (11×18 + 3×18 = 252次元)
      ↓
    [Policy Network]
      ↓
出力: Middle候補スコア (C(11,5) = 462通り)
      ↓
    上位k個を選択
```

### Phase 3: Exhaustive評価

```
Top k個 × Middle k個 × Bottom全パターン
= k × k × C(6,5) 
= k² × 6 通りのみ評価

k=10なら: 10 × 10 × 6 = 600通り
k=50なら: 50 × 50 × 6 = 15,000通り
```

---

## NN設計

### 入力表現

```python
def encode_hand(cards: List[Card]) -> np.ndarray:
    """
    各カード: 18次元
    - ranks: 13次元 (one-hot)
    - suits: 4次元 (one-hot)
    - joker: 1次元 (binary)
    """
    features = []
    for card in cards:
        rank_vec = one_hot(card.rank, 13)
        suit_vec = one_hot(card.suit, 4)
        joker_vec = [1 if card.is_joker else 0]
        features.extend(rank_vec + suit_vec + joker_vec)
    return np.array(features)
```

### 出力表現

```python
class TopPolicyNetwork(nn.Module):
    def __init__(self, input_dim=252, hidden_dim=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 364),  # C(14,3)候補
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.fc(x)
```

---

## 訓練データ

### 既存ソルバーデータを使用

```python
# fl_joker0_v2.jsonl から
{
    "hand": [...],      # 入力
    "top": [...],       # 正解Top (インデックスに変換)
    "middle": [...],    # 正解Middle
    "bottom": [...]     # 正解Bottom
}
```

### ラベル生成

```python
def hand_to_top_label(hand, top):
    """
    正解Topのインデックスを計算
    C(14,3) = 364通りのうち、どれが正解か
    """
    from itertools import combinations
    
    all_tops = list(combinations(range(14), 3))
    top_indices = set(hand.index(c) for c in top)
    
    for i, combo in enumerate(all_tops):
        if set(combo) == top_indices:
            return i
    
    return -1  # エラー
```

---

## 推論フロー

```python
def solve_with_policy_network(hand, policy_net, k=50):
    """
    1. Top候補を予測
    2. 上位k個に絞る
    3. 各Topについて、Middle候補を予測
    4. 上位k個に絞る
    5. 残りをExhaustiveで評価
    """
    # 1. Top予測
    hand_enc = encode_hand(hand)
    top_scores = policy_net.predict_top(hand_enc)
    top_k_indices = top_scores.argsort()[-k:]
    
    best_placement = None
    best_score = -inf
    
    for top_idx in top_k_indices:
        top = get_top_by_index(hand, top_idx)
        remaining = [c for c in hand if c not in top]
        
        # 2. Middle予測 (または全探索)
        for middle in combinations(remaining, 5):
            rest = [c for c in remaining if c not in middle]
            
            # 3. Bottom全探索
            for bottom in combinations(rest, 5):
                score = evaluate_placement(top, middle, bottom)
                if score > best_score:
                    best_score = score
                    best_placement = (top, middle, bottom)
    
    return best_placement
```

---

## 期待効果

| 方法 | 探索数 | 時間 |
|------|-------|------|
| Exhaustive | 1,009,008 | 90秒 |
| Pruning付き | ~300,000 | 26秒 |
| **Policy k=50** | 15,000 | **~1秒** |
| **Policy k=10** | 600 | **~0.05秒** |

---

## 実装ステップ

1. [ ] TopPolicyNetwork実装
2. [ ] 訓練データ (Top正解ラベル) 生成
3. [ ] Top予測モデル訓練
4. [ ] solve_with_policy_network実装
5. [ ] 精度検証（Exhaustiveと比較）
6. [ ] MiddlePolicyNetwork追加（オプション）

---

## 注意点

- k が小さすぎると最適解を逃す
- k=50なら約98%の精度を期待
- 訓練データの品質が重要（Exhaustiveで生成したデータを使用）
