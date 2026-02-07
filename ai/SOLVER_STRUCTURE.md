# FL Solver V2 ヒューリスティック構造

## 概要

14枚のカードから最適なFantasylandの配置を探索するソルバー。
パターンベースのヒューリスティックで高速に解を見つける。

---

## PHASE 1: FL Stay Patterns
**目的**: FL残留条件（Bottom = Quads+, SF, RF）を満たす解を探索

### Pattern A: FL Stay Bottom Fixed
```
1. find_fl_stay_bottoms() で Quads/Straight Flush/Royal Flush を探索
2. 残りカードで以下を試行:
   a) 66+ Pair を Top に配置 → 全Middle組み合わせを試行
   b) Trips を Top に配置 → 全Middle組み合わせを試行
   c) Flush を Middle に配置 → 全Top組み合わせを試行
   d) Straight を Middle に配置 → 全Top組み合わせを試行
```

### Pattern B: Trips Top Fixed
```
1. find_all_trips() で全Tripsを探索
2. optimize_bottom_middle() でBottom/Middle最適化
3. FL Stay有無に関わらず候補に追加（高ロイヤリティを逃さない）
```

---

## PHASE 2: Non-FL Stay Patterns
**目的**: FL残留しないがロイヤリティの高い解を探索

### Pattern C: Strong Bottom Fixed (Pair〜Full House)
```
1. find_all_bottoms(min_rank=PAIR) で強いBottomを探索
2. 残りから 66+ Pair を探して Top に配置
3. 全Middle組み合わせを試行
```

### Pattern D: 66+ Pair Top Fixed
```
1. find_pairs_66_plus() で全66+ペアを探索
2. ペアを Top に配置（キッカーは最後に決定）
3. 残りから Bottom → Middle → キッカー の順に選択
```

### Pattern F: Flush Bottom/Middle
```
1. find_flushes() で全Flushを探索
2. パターンA: Flush を Bottom に配置 → 全Middle/Top組み合わせを試行
3. パターンB: Flush を Middle に配置 → 全Bottom/Top組み合わせを試行
```

### Pattern G: Straight Bottom
```
1. find_straights() で全Straightを探索
2. Straight を Bottom に配置 → 全Middle/Top組み合わせを試行
```

### Pattern H: Flush + Straight Combo
```
1. Flush Bottom + Straight Middle の組み合わせを探索
2. Straight Bottom + Flush Middle の組み合わせを探索
```

### Pattern I: Double Flush
```
1. find_flushes() で1つ目のFlushを選択
2. 残りカードから2つ目のFlushを探索
3. 両方の順序を試行:
   - Flush1 Bottom + Flush2 Middle
   - Flush2 Bottom + Flush1 Middle
```

### Pattern E: Fallback
```
※上記パターンで解が見つからない場合のみ実行
1. find_all_bottoms() で全Bottomを探索
2. 全Middle/Top組み合わせを試行（限定的）
```

---

## Final Selection
```
1. FL候補 + 非FL候補 を統合
2. ソート基準: (royalties, can_stay) の降順
   - ロイヤリティ最高を優先
   - 同点の場合はFL Stay優先
3. 上位 max_solutions 個を返す
```

---

## ヘルパー関数

| 関数名 | 説明 |
|--------|------|
| `find_fl_stay_bottoms()` | Quads/SF/RF を探索 |
| `find_all_trips()` | 全Trips（ジョーカー含む）を探索 |
| `find_pairs_66_plus()` | 66以上のペアを探索 |
| `find_all_bottoms()` | 指定ランク以上の全Bottom組み合わせ |
| `find_flushes()` | 全Flush（ジョーカー含む）を探索 |
| `find_straights()` | 全Straight（ジョーカー含む）を探索 |
| `optimize_bottom_middle()` | 固定Topに対するBottom/Middle最適化 |
| `evaluate_placement()` | 配置の評価（バスト判定、ロイヤリティ計算） |

---

## ロイヤリティ表

### Top (3枚)
| 役 | ロイヤリティ |
|----|-------------|
| 66〜AA Pair | 1〜9 |
| 222〜AAA Trips | 10〜22 |

### Middle (5枚)
| 役 | ロイヤリティ |
|----|-------------|
| Three of a Kind | 2 |
| Straight | 4 |
| Flush | 8 |
| Full House | 12 |
| Four of a Kind | 20 |
| Straight Flush | 30 |
| Royal Flush | 50 |

### Bottom (5枚)
| 役 | ロイヤリティ |
|----|-------------|
| Straight | 2 |
| Flush | 4 |
| Full House | 6 |
| Four of a Kind | 10 |
| Straight Flush | 15 |
| Royal Flush | 25 |

---

## FL Stay 条件
- **Top が Three of a Kind** または
- **Bottom が Four of a Kind 以上**
