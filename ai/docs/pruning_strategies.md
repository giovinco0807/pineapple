# 枝刈り（Pruning）戦略

## 現状の問題

14枚のExhaustive Search:
- C(14,3) × C(11,5) × C(6,5) = **約100万通り**
- 90秒/ハンド

目標: **10万通り以下**に削減 → 10秒以下

---

## 枝刈り戦略

### 1. バースト早期検出

**ルール**: Top ≤ Middle ≤ Bottom の強さ順

```python
# Topが決まった時点でチェック
if top == [A♠, K♠, Q♠]:  # ハイカード AKQ
    # MiddleがAKQより弱いペアだと即バースト確定
    # → その分岐を全て枝刈り
```

**実装案**:
```python
def can_complete_without_bust(top, middle_cards, bottom_cards):
    """
    部分配置から、バーストせずに完成可能か判定
    """
    # Topが決まっている場合
    if len(top) == 3:
        top_strength = evaluate_top(top)
        
        # 残りカードで作れる最強のMiddle
        best_possible_middle = get_best_possible_hand(middle_cards, 5)
        
        if best_possible_middle < top_strength:
            return False  # バースト確定
    
    return True
```

**削減効果**: 約20-30%

---

### 2. 支配配置の除外

**定義**: 配置Aが配置Bを支配 = AはどのDiscardでもBより良い

```
例:
  A: Top=[Q,Q,2] Middle=[A,K,J,T,9] Bottom=[...]
  B: Top=[Q,2,3] Middle=[A,K,J,T,9] Bottom=[...]

A > B なのでBを探索する必要なし
```

**実装**: 同じMiddle/Bottomの組み合わせで、Topだけ違う場合、弱いTopを除外

**削減効果**: 約10-20%

---

### 3. ロイヤリティ下限枝刈り

**アイデア**: 現在のベスト解より明らかに劣る分岐を除外

```python
best_so_far = 15  # 現在のベストスコア

def should_prune(partial_placement):
    # この配置から達成可能な最大ロイヤリティ
    max_possible = estimate_max_royalties(partial_placement)
    
    if max_possible < best_so_far:
        return True  # 枝刈り
    
    return False
```

**削減効果**: 探索順序に依存（良い順で探索すれば効果大）

---

### 4. FL Stay不可能枝刈り

**FL Stay条件**:
- Bottom: Quads or Straight Flush
- Top: Trips

```python
def can_stay_in_FL(remaining_cards, top, bottom):
    # Tripsがもう作れない
    if len(top) == 3 and not is_trips(top):
        if not can_make_quads_or_sf(remaining_cards, bottom):
            return False  # FL Stay不可能
    
    return True
```

**注意**: FL Stay不可能でもロイヤリティ最大化は価値あり

**削減効果**: FL Stay優先の場合に有効

---

### 5. 同型配置の統合

**アイデア**: 同じランク構成のカードを同一視

```
[A♠, K♠, Q♠] と [A♥, K♥, Q♥] は同じ価値
（フラッシュ可能性を除く）
```

**実装**: カードをランクでグループ化し、代表1つだけ探索

**削減効果**: 約50% (大きい！)

---

## 実装優先順位

| 戦略 | 削減効果 | 実装難易度 | 優先度 |
|------|---------|-----------|--------|
| バースト早期検出 | 20-30% | 低 | ⭐⭐⭐⭐⭐ |
| 同型配置統合 | ~50% | 中 | ⭐⭐⭐⭐⭐ |
| ロイヤリティ下限 | 変動 | 中 | ⭐⭐⭐⭐ |
| 支配配置除外 | 10-20% | 中 | ⭐⭐⭐ |
| FL Stay不可能 | 変動 | 低 | ⭐⭐⭐ |

---

## 次のステップ

1. **バースト早期検出**を実装
2. **同型配置統合**を実装
3. 効果測定（探索数とスコアの比較）
