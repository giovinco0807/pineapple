# FL Solver v3 設計書

## 概要

Fantasylandの最適配置を高速に求めるソルバー。
FL Stay条件を優先的に探索し、枝刈りで探索空間を削減。

---

## FL Stay条件（復習）

| 位置 | 役 | ロイヤリティ |
|------|-----|-------------|
| Bottom | Quads | 10点 |
| Bottom | Straight Flush | 15点 |
| Bottom | Royal Flush | 25点 |
| Top | Trips | 10-22点 |

---

## 全体フロー

```
入力: 14-17枚のカード

Phase A: Bottom FL Stay狙い
    ↓
A >= 41? → Yes → 終了（Aが最適解）
    ↓ No
Phase B: Top FL Stay狙い
    ↓
max(A,B) >= 41? → Yes → 終了
    ↓ No
Phase C: FL Stayなし（ロイヤリティ最大化）
    ↓
最終解 = max(A, B, C)
```

---

## Phase A: Bottom FL Stay狙い

### A1: Royal Flush検証

```
1. Bottomに配置可能なRoyal Flushを列挙
   - 各スート(s,h,d,c)について AKQJT があるか確認
   - ジョーカーで補完可能な場合も含む
   - ジョーカーなしで作れる場合はジョーカーを使わない

2. 各Royal Flushについて:
   a. Bottomに配置
   b. 残りカードでTopを検証（高ロイヤリティ順）
      - AAA(22点) → KKK(21点) → ... → 66(1点) → ロイヤリティなし
   c. 各Topについて:
      - 残りカードでMiddleを検証（高ロイヤリティ順）
      - 制約: Middle >= Top（役の強さ）
      - バーストしない最高点を記録

3. 最高点を A_royal として記録
```

### A2: Straight Flush検証

```
1. Bottomに配置可能なStraight Flushを列挙
   - 各スート × 各開始ランク（A5432〜KQJT9）
   - 注意: QJT98 と JT987 など複数パターンがある場合は全て検証
   - ジョーカー使用ルールはRoyalと同様

2. 各Straight Flushについて:
   a. Bottomに配置
   b. Top/Middleを A1 と同様に検証

3. 最高点を A_sf として記録
```

### A3: Quads検証

```
1. Bottomに配置可能なQuadsを列挙
   - 同じランク4枚 + キッカー1枚
   - キッカーは残りカードから最適なものを選択
   - ジョーカーで補完可能な場合も含む

2. 各Quadsについて:
   a. Bottomに配置
   b. Top/Middleを A1 と同様に検証

3. 最高点を A_quads として記録
```

### Phase A結果

```
A = max(A_royal, A_sf, A_quads)

枝刈り: A >= 41 → Phase B, Cをスキップ
理由: Phase Bの最高点は40点（Top AAA 22 + Mid FH 12 + Bot FH 6）
```

---

## Phase B: Top FL Stay狙い

### B1: Top Trips検証

```
1. Topに配置可能なTripsを列挙
   - AAA(22点) → KKK(21点) → ... → 222(10点)
   - ジョーカーで補完可能な場合も含む

2. 各Tripsについて:
   a. Topに配置
   b. 残りカードで役の組み合わせを検証:
   
      優先順位（トータルロイヤリティ順）:
      
      i.  Trips以上が2つ作れるか確認
          - Bottom Quads(10) + Middle FH(12) = 22点
          - Bottom FH(6) + Middle Quads(20) = 26点 ← ただしこれはバースト
          - Bottom SF(15) + Middle FH(12) = 27点
          等
      
      ii. 作れる場合、ロイヤリティ順に配置
          注意: 役の強さ制約 Bottom >= Middle >= Top
          
          例外ケース:
          - (FH, Straight) = 6+4=10点
          - (FH, Trips) = 6+2=8点
          - (Flush, Flush) = 4+8=12点 ← こちらが高い
          
      iii. バーストしない最高点を記録

3. 最高点を B として記録
```

### Phase B結果

```
現時点の最高 = max(A, B)

枝刈り: すでに A >= 41 なら Phase B 自体をスキップ済み
```

---

## Phase C: FL Stayなし（ロイヤリティ最大化）

FL Stay条件を満たせない場合のロイヤリティ最大化。

### C1: Full House検証

```
最高点: 27点（Bot FH 6 + Mid FH 12 + Top AA 9）

1. Bottomに配置可能なFull Houseを全て列挙
   - AAAKK, AAAQQ, KKKAA, KKKQQ, ... 全パターン

2. 各Full Houseについて:
   a. Bottomに配置
   b. 残りカードでMiddleを検証（高ロイヤリティ順）
      - 制約: Middle <= Bottom（役の強さ）
      - FH(12) → Flush(8) → Straight(4) → Trips(2) → ...
   c. 各Middleについて:
      - 残りカードでTopを検証（高ロイヤリティ順）
      - 制約: Top <= Middle（役の強さ）
      - AA(9) → KK(8) → ... → 66(1) → なし
   d. バーストしない最高点を記録

3. 最高点を C1 として記録

枝刈り:
- C1 == 27 → 即終了（最高点達成）
- C1 >= 22 → C2以降スキップ（C2最大は21点）
```

### C2: Flush検証

```
最高点: 21点（Bot Flush 4 + Mid Flush 8 + Top AA 9）

条件: C1 < 22 の場合のみ実行

1. Bottomに配置可能なFlushを全て列挙
   - 同スート5枚の全組み合わせ

2. 各Flushについて:
   a. Bottomに配置
   b. Middle/TopをC1と同様に検証
      - 制約: Middle <= Bottom（役の強さ）
   
3. 最高点を C2 として記録

枝刈り:
- C2 == 21 → 即終了
- max(C1, C2) >= 15 → C3以降スキップ（C3最大は15点）
```

### C3: Straight検証

```
最高点: 15点（Bot Str 2 + Mid Str 4 + Top AA 9）

条件: max(C1, C2) < 15 の場合のみ実行

1. Bottomに配置可能なStraightを全て列挙
   - A2345 〜 AKQJT

2. 各Straightについて:
   a. Bottomに配置
   b. Middle/Topを同様に検証

3. 最高点を C3 として記録

枝刈り:
- C3 == 15 → 即終了
- max(C1, C2, C3) >= 11 → C4スキップ（C4最大は9点）
```

### C4: Pairのみ

```
最高点: 9点（Top AA のみ）

条件: max(C1, C2, C3) < 11 の場合のみ実行

1. 作れるペアを数える

2. 5ペアの場合:
   - Top: 最もロイヤリティが高いペア（AA優先）
   - Middle: TwoPair
   - Bottom: TwoPair
   - 制約: Bottom >= Middle >= Top を満たすよう調整

3. 4ペアの場合:
   - Top: 2番目に強いペア
   - Middle: 最も強いペア（OnePair）
   - Bottom: 残りの TwoPair
   - 理由: Topにロイヤリティあり、Middleペアにはなし

4. 3ペア以下の場合:
   - 最適に配置してバースト回避

5. 最高点を C4 として記録
```

### Phase C結果

```
C = max(C1, C2, C3, C4)
```

---

## 最終結果

```
最適解 = max(A, B, C)
```

---

## 枝刈りまとめ

| 条件 | アクション |
|------|-----------|
| A >= 41 | Phase B, C スキップ |
| C1 == 27 | C2, C3, C4 スキップ |
| C1 >= 22 | C2, C3, C4 スキップ |
| C2 == 21 | C3, C4 スキップ |
| max(C1,C2) >= 15 | C3, C4 スキップ |
| C3 == 15 | C4 スキップ |
| max(C1,C2,C3) >= 11 | C4 スキップ |

---

## ジョーカールール

1. **温存優先**: ジョーカーなしで役が作れる場合は使わない
2. **上位役優先**: より高いロイヤリティの役に使う
3. **複数ジョーカー**: 2枚ある場合は別々の役に使うことも検討

---

## 役の強さ比較（バースト判定用）

```
Royal Flush > Straight Flush > Quads > Full House > Flush > 
Straight > Trips > Two Pair > One Pair > High Card
```

同じ役の場合はランクで比較。

---

## 計算量の見積もり

### 最悪ケース（14枚、枝刈りなし）
```
Phase A: 少数（役が作れる場合のみ）
Phase B: C(14,3) × 内部探索 ≈ 数千
Phase C: C(14,5) × C(9,5) × C(4,3) ≈ 数万
```

### 枝刈りあり
```
多くのケースで Phase A または B で終了
Phase C も早期終了が多い
実測: 0.1秒以下を目標
```

---

## 実装優先度

1. [ ] 役判定関数（各役が作れるか + 全パターン列挙）
2. [ ] Phase A実装
3. [ ] Phase B実装
4. [ ] Phase C実装
5. [ ] 枝刈り実装
6. [ ] テスト・ベンチマーク
