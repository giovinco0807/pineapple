# T0-BB evaluator 審判値追訓練 — 設計 (2026-08-30)

## 背景 (なぜ方策ではなくevaluatorか)

- 8/29のpgateで方策argmax直配信は **−0.72/hand [−0.80, −0.65]** の大敗 (20,000ディール4シード)。
  柵(方策)と審査員(evaluator)はどちらも外せない。
- 一方、mine3↔mine4の再採掘119対で、**確定誤りの再発23件中22件は審判の正解が方策top-8柵内**。
  誤答を選び続けているのはevaluator。採掘→方策再訓練では本番に届かない。
- AAトップ病理 (ジョーカーをA/K/Q横に即置き) だけで確定誤り556件中129件・損失の24%。

## 設計の制約 (Explore調査 2026-08-30)

1. evaluator = `T4FirstEvaluator` 207次元MLP(512,256,128)、`train_t4_first_evaluator.py`、
   SmoothL1(EV回帰)、dev regret選抜。**fine-tune経路なし** (流儀はy-swap/コーパス結合+全再訓練)。
2. **目的関数の不一致**: 教師ラベルは own-hand EV (2026-08-14転換、相手除外)。
   審判値(sel_means)は実対戦のfull-game settle平均。値の直接混入・y-swapは単位が合わない。
3. 特徴量エンコードはRust `--joint-outlook` (samples=400, arrangements=32) を共有。
   採掘ルート(空盤面+5枚)のペアエンコードはローカルで可能。

## 設計: 序列移植 stage-B (方策v2で実証済みのレシピをevaluatorに移す)

配信で使われるのはevaluatorの**柵内argmax=序列だけ**。序列は目的関数に依存しない。
値回帰ではなく、審判の候補順位をペアワイズで教える。

1. **エンコード** (ローカル, 無料):
   material 4,040ルート × 審判候補 (mine1-3: 中央値7、mine4: 2) ≈ 23k行を
   Rust joint-outlook (400×32, 訓練時仕様) で207次元化。empty-board joint=8零則そのまま。
2. **訓練** (新スクリプト `train_t0_eval_correction.py`, ローカルGPU, 無料):
   - `hu/t0_bb.bin`の元チェックポイント(`evaluator_best.pt`)から初期化 (要 --init-checkpoint 追加)。
   - 損失 = ①元コーパス(t0_bb_enc_both fit)10%リプレイのSmoothL1アンカー (壊さない)
     + ②審判候補ペアワイズhinge: 序列 (a≻b) を margin重み付きで
     (scored/escalated=全重量、select-onlyペア=低重量)。
   - 方策v1の教訓: サブセットのみの損失は分布外で暴走 → アンカー必須。
   - torch-seed固定・複数シード (8/29の分散の教訓)。
3. **オフライン検査**: 保留した監査ルートで「柵内argmaxがref_pickに一致する率」と回収率。
   複数シードで安定して改善したものだけ次へ。
4. **ミラーゲート** (~$8-10, 要承認): 修正evaluator vs 現行 (models_ship_20260830)。
   同柵・同方策・同ランカー。B腕のhu/t0_bb.binのみ差し替え。

## リスクと対策

- **23k行は元コーパス(百万行規模)に比べ極小** → 全再訓練ではなくstage-B fine-tuneにする理由。
- **mine4の順位表が2候補に痩せている** (halvingバグ、修正済み) → mine4行はverdictペアのみ寄与。
  次の採掘波から完全な順位表が入る。
- **T0-BBの1ノードだけの差し替え**なので他街は無風。ゲートの差分はT0-BB先手のみに由来。

## 状態

- [ ] エンコーダ配線 (材料→207次元npz)
- [ ] train_t0_eval_correction.py
- [ ] オフライン検査
- [ ] ゲート (承認待ち)
