# OFC Pineapple AI ロードマップ

## 最終目標

**OFC Pineapple 完全AI** - 通常プレイとFantasyland両方で最適なプレイができるAI

---

## Phase 1: Fantasyland AI（現在）

### 1.1 ソルバー開発 ✅
- [x] Exhaustive Search実装
- [x] Beam Search実装
- [x] ジョーカー対応
- [x] FL Stay条件の優先

### 1.2 RL環境構築 ✅
- [x] FantasylandEnv実装
- [x] Action Masking（バースト防止）
- [x] 報酬設計（ロイヤリティ + FL Stayボーナス）

### 1.3 訓練 🔄 進行中
- [x] 14枚モデル訓練
- [x] 17枚モデル訓練
- [ ] Exhaustive Search版訓練
- [ ] 全枚数（14-17）モデル統合

### 1.4 EV計算
- [ ] 各枚数の平均ロイヤリティ測定
- [ ] 各枚数のFL Stay率測定
- [ ] EV計算式の実装

```
EV(n枚) = 平均ロイヤリティ(n) + Stay率(n) × EV(次のFL枚数)
```

---

## Phase 2: 通常プレイAI

### 2.1 環境構築
- [ ] OFCEnv実装（1手ずつ配置）
- [ ] 相手のハンド観測
- [ ] Pineappleルール（3枚から2枚選択）

### 2.2 報酬設計
- [ ] ロイヤリティ報酬
- [ ] FLエントリー報酬 = **FL EV**（Phase 1で計算）
- [ ] スクープボーナス

```python
reward = royalties + scoop_bonus + (FL_entry × FL_EV)
```

### 2.3 訓練
- [ ] Self-play訓練
- [ ] 対ランダムAI訓練
- [ ] 対人間データでの調整

---

## Phase 3: 統合と最適化

### 3.1 統合
- [ ] 通常プレイ → FL遷移のシームレス化
- [ ] FL → 通常プレイの遷移
- [ ] エンドツーエンドの評価

### 3.2 最適化
- [ ] NN推論速度最適化（<10ms目標）
- [ ] メモリ使用量最適化
- [ ] WebAssembly対応（ブラウザ実行）

### 3.3 評価と改善
- [ ] 人間プレイヤーとの対戦評価
- [ ] 既存AIとの比較
- [ ] 弱点分析と改善

---

## 現在の進捗

```
Phase 1: ████████░░ 80%
Phase 2: ░░░░░░░░░░ 0%
Phase 3: ░░░░░░░░░░ 0%
```

---

## 次のステップ

1. **今日**: Exhaustive Search版訓練の完了
2. **明日**: EVテーブル作成（14-17枚）
3. **今週**: Behavior Cloningでソルバー知識をNNに転移
4. **来週**: Phase 2開始 - 通常プレイ環境構築
