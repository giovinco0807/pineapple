# ジョーカー版AI (`ai/`) 引き継ぎ書 — トラック全体

2026-08-21 作成。**このリポジトリの `ai/` 配下すべて**を対象にした地図。
今夜のHU作業だけの詳細は [`hu_handover_20260821.md`](hu_handover_20260821.md) にある。

> **注意書き**: 本書のうち、HU梯子 (§3.1) と測定規律 (§7) は作成者が直接手を動かして
> 検証した範囲。それ以外は**ファイル調査と各docの冒頭による分類**であり、
> 全33件を精読した結果ではない。各節から正典docへ飛んで確認すること。

---

## 0. まず知るべき3つ

1. **このリポジトリは2トラックある。** `ai/` = **ジョーカー2枚あり**、
   `regular-ofc-pineapple/` = **ジョーカー無し**。別プロジェクトと思ってよい。
   docs も別 (`ai/docs/` 33件 / `regular-ofc-pineapple/docs/` 99件)。
2. **GCPの quota を共有している。** `ofc-lg-*` という名前のVMは**レギュラー版の作業**。
   絶対に削除・停止しない。ジョーカー版のVMは `hu-*` プレフィックス。
3. **`MEMORY.md` 冒頭の2026年3月の記述 (BC Policy + Value Network + MCTS、
   `backend/ai_player.py`) は legacy。** 現行の主戦場はHU梯子とFL14で、
   設計思想が入れ替わっている (§2)。

---

## 1. このトラックが解こうとしていること

**ジョーカー2枚入り標準OFC Pineapple のヘッズアップを、T0〜T4の全ターン・
BB(先行)/BTN(後攻)の両席で、相手の非公開カードを使わずに近似均衡で打つ。**

正典: [`all_turn_ai_goal_and_milestones.md`](all_turn_ai_goal_and_milestones.md)。

**行動順と盤面形状の契約**: [`hu_all_turns_contract.md`](hu_all_turns_contract.md)。
`position_contract_version = "bb_first_v1"` を持たない成果物は legacy 扱いで、
**正典ランタイムが黙って読み込んではならない**と明記されている。新しいモデルを
足すときは必ずこのフィールドを確認すること。

局面の呼び方: **T0** = 初手5枚配置、**T1〜T3** = 3枚引いて2枚置き1枚捨て、
**T4** = 最終ストリート。**FL** = ファンタジーランド (14〜17枚一括配置)。

---

## 2. 歴史の3層 — 何が現役で何が化石か

### 第1層 (〜2026-03): AlphaZero風 — **legacy**

BC Policy + Value Network + MCTS。`ai/mcts/`, `ai/models/`, `ai/player/`,
`ai/rl/`, `ai/policy_network*.py` (多くは削除済み)。`backend/ai_player.py` が
その本番入口だった。**`MEMORY.md` 冒頭の表 (v1/v2/v3、VN v4、bottom-up BC) は
この層の記録。** 現行の判断根拠にしてはいけない。

放棄した理由は記録されている: 自己対戦VNは corr 0.79 で頭打ち、
OFCのチャンスノード分岐が C(30,3)≈4060 で完全MCTSが成立しない。

### 第2層 (2026-05〜07): bottom-up 厳密解 + 教師蒸留

T4から厳密に解き、上の街へ教師を積み上げる方式。Rustソルバ群
(`ai/rust_solver/`) と `ai/tutor/` の大半がここで生まれた。
`ai/config/` の大量の JSON (t2_*, t3_*) はこの層のプール定義・ゲート定義。

**この層の資産は現役**。特に `t4_first_exact` クレートと `ai/tutor/` の
エンコーダ/学習器は、そのまま第3層の土台になっている。

### 第3層 (2026-08〜、現行): HU梯子 と FL14

**設計の正典は [`hu_regen_distill_ladder_20260818.md`](hu_regen_distill_ladder_20260818.md)。
一行の要約が書いてある:**

> **地面は厳密解、ラベルは消耗品、モデルは中間生成物。**

---

## 3. 現在アクティブな2系統

### 3.1 HU梯子 (通常ハンド) — **今夜の作業**

**厳密V4 → T3教師 → 境界ネット(V3s/V2s/V1s) → 街×席ごとのBC評価器8本。**

- 現チャンピオン = **gen-2** (on-policy lap 1)。
- 今夜 lap 2 のラベルが完走し、**どの重みを出荷するかを対戦で決める直前で停止中**。
- 詳細・次の一手・コマンドはすべて
  [`hu_handover_20260821.md`](hu_handover_20260821.md) と
  [`hu_gen2_onpolicy_20260819.md`](hu_gen2_onpolicy_20260819.md) (§1-19)。

**中核のRust**: `ai/rust_solver/t4_first_exact/` (22ファイル)。
`hu_match.rs` (対戦・トレース・ミラー・深掘り再現)、`hu_encode.rs` (207次元ペア符号化)、
`t3_first_hu.rs` / `t3_second_hu.rs` (T3教師)、`v4_first.rs` (厳密な地面)。

### 3.2 FL14 (ファンタジーランド) — 別系統、停止中

14枚FLをレギュラー版の方法で作り直す系統。
計画 [`fl14_joker_plan_20260811.md`](fl14_joker_plan_20260811.md)、
lap 3 の契約 [`fl14_lap3_start_20260814.md`](fl14_lap3_start_20260814.md)。

**重要な前提** ([[fl14-best-response-rebuild]]): 旧「対FL」資産は
**相手が静的**という前提で作られており無効。FLのbest responseはヒーローの利得の
2/3を取り返す。

**基盤のRust**: `ai/rust_solver/fl_solver/` (9ファイル)。
`frontier::best_response` が厳密解。

**HU梯子との接点**: `ai/config/fl_ev.json` の定数。FL進入の価値を通常ハンドの
評価に流し込むので、**片方を変えるともう片方の全ラベルの意味が変わる** (§6)。

---

## 4. コードの地図

| 場所 | 中身 | 状態 |
|---|---|---|
| `ai/rust_solver/t4_first_exact/` | HU梯子の中核。対戦・教師・符号化 (22 rs) | **現役** |
| `ai/rust_solver/fl_solver/` | FL厳密解・frontier (9 rs) | **現役** |
| `ai/rust_solver/prob_engine/` | 組合せ列挙による確率計算 | 現役 (第2層) |
| `ai/rust_solver/{t3,t4}_exact_solver`, `*_generator` | 第2層の厳密解・教師生成 | 保守のみ |
| `ai/rust_solver/cfr_solver/`, `backward/` | CFR実験・後退帰納 | 実験、[`t3_public_cfr_plan.md`](t3_public_cfr_plan.md) |
| `ai/tutor/` (211 py) | 教師生成・符号化・学習・評価。**現行作業の主戦場** | **現役** |
| `ai/tutor/fleet/` | GCP艦隊 (投入・回収・補充・startupスクリプト) | **現役** |
| `ai/tutor/verdict/` | 世代間判定の測定道具 (今夜追加) | **現役** |
| `ai/training/` (138 py) | 第2層の学習器群 (action-value reranker 等) | 大半は保守のみ |
| `ai/engine/` | ゲームエンジン (action_space, encoding, game_engine) | 現役 (共有) |
| `ai/mcts/`, `ai/models/`, `ai/player/`, `ai/rl/`, `ai/cfr/` | 第1層 | **legacy** |
| `ai/tmp_worker_*/`, `ai/logs/`, `ai/reports/` | 実行時の残骸 | 掃除可 |

**現行作業で実際に触るのは `ai/tutor/`, `ai/tutor/fleet/`, `ai/tutor/verdict/`,
`ai/rust_solver/t4_first_exact/`, `ai/config/fl_ev.json` の5箇所にほぼ収まる。**

### 主要モジュール (HU梯子)

| モジュール | 役割 |
|---|---|
| `ai.tutor.hu_street_teacher` | 街教師。207次元 `PairEncoder` の定義もここ |
| `ai.tutor.encode_hu_teacher` / `encode_boundary` | 教師出力→学習用npz |
| `ai.tutor.train_t4_first_evaluator` | 評価器の学習 (`T4FirstEvaluator`) |
| `ai.tutor.export_t4_first_evaluator` | `.pt` → `.bin` (Rustが読む平坦形式) |
| `ai.tutor.export_value_npz` | 境界ネット → `.npz` (教師の葉として使う) |
| `ai.tutor.extract_onpolicy_requests` | トレース → 街ごとのroot要求 |
| `ai.tutor.analyze_selfplay_flev` | **fl_ev の逆算 (自作するな、§6)** |
| `ai.tutor.verdict.*` | 世代間判定 (§7) |

---

## 5. データと成果物の置き場

| 場所 | 中身 |
|---|---|
| `D:/ofc_data/hu/` | HU梯子の作業領域。ラベル・エンコード済npz・学習済みモデル |
| `D:/ofc_data/` (直下) | 第2層以前の資産 (value_v4c 等) |
| `gs://pokerhu-ofc-solver-485418-training/hu-street/` | 艦隊のバケツ |
| └ `artifacts/` | 入力 (requests, value npz, models tar.gz) |
| └ `src/`, `bin/<binstamp>/` | ソース tar と検証済みバイナリ (sha256 で照合) |
| └ `runs/<run-id>/labels|enc|matches|partial|progress|logs/` | 出力 |
| `ai/models/` | 第1層のチェックポイント (legacy) |

**命名の規約**: モデルディレクトリは `{街}_{系統}_s{シード}`
(例 `t1_bb_onpol_s20260815`)、エンコードは `{街}_enc_{系統}`。
**系統タグ** = `onpol` (lap1) / `lap2` / `both` (連結) / `c16` (gen-1)。

---

## 6. 契約と設定 — 変えると全部の意味が変わるもの

### `ai/config/fl_ev.json`

FL進入の価値。**通常ハンドの全ラベルがこの定数の下で計算されている。**
現行 `reward_mode: direct`、`{14:5.28, 15:16.61, 16:38.76, 17:70.07}`。

- **実測値は 14:6.57 / 15:16.66 / 16:38.51 / 17:75.79** (gen-2自己対戦60,000ハンド)。
  幅15・16は収束済み (全進入の93%)。**幅14だけ固定点反復の途中** (0→3.17→5.28→6.57)。
- **更新は世代境界でのみ。** lap の途中で変えると、ラベルとそれを生んだ方策の
  価格が食い違う。fl_solver 側のハードコード定数も同時に直すこと。
- **逆算は `ai.tutor.analyze_selfplay_flev` を使う。自作するな。**
  自作してセッション打ち切りの連鎖を捨て、「幅16は10点過大」と誤報告した前科がある
  (捨てられるのは大勝ちの直後 = 最も価値の高い連鎖)。

### `hu_all_turns_contract.md`

行動順と決定盤面の形状。`position_contract_version` を持たない成果物は
正典ランタイムに load させない。

### `ai/config/promotion_gate_*.json`, `m3_behavior_*.json`

第2層の昇格ゲート定義。運用手順は
[`m3_behavior_calibration_production_runbook.md`](m3_behavior_calibration_production_runbook.md)。

---

## 7. 測定規律 — このプロジェクトが最も高く買った知識

**この節は暗記に値する。ここに書いてある失敗はすべて実際に起きた。**

### 7.1 忠実度は勝敗に転写されない (4例)

学習指標の改善が対戦成績に**繋がらなかった**事例が4件記録されている
([[leaf-truncation-costs]]、[[m7-t1-relabel]]、正典doc §2.1、§13)。
**判定は対戦のみ。regret や MAE で出荷を決めてはいけない。**

### 7.2 物差しを揃えろ、そして「誰の分布か」を問え

今夜の核心 (正典doc §17-19):

- モデルの「自分のdev」の値は**その dev でチェックポイントを選んだ後の値**。
  比較に使えない。`ai.tutor.verdict.cross_eval` は対角にドットを打って可視化する。
- 別々のコーパスの数値を並べるのは**別々の物差し**を比べること。3回踏んだ。
- コーパスを束ねて切り直すと**片方の検証行がもう片方の訓練行になる**。
  実測で79〜80%漏れていた。**印は「自分のdevより他人のdevで良い点が出る」。**
- そして最後に: **その共通の物差しは誰の分布か。** プールtestで連結が6街全勝、
  現分布だけで見ると lap2単独が4街で勝つ。**算術的にどちらも正しい。**

道具: `ai.tutor.verdict.{cross_eval,street_verdict,split_verdict,build_union_fit}`。

### 7.3 対戦の判定単位はシードセット

セット内CIはFL連鎖とスタック継続でハンド独立性が壊れ、**狭く出すぎる**。
**最低2セット、理想4セット以上でプール判定。** m7/m7b では2セットのCIが重ならず
符号も逆で、どちらも単体なら「有意」に読めた。

### 7.4 ミラー対戦と、その厳密ゼロの両義性

`--hu-mirror` = 同一ディールを席交換で2回 (デュプリケート・ブリッジ方式)。
配牌運が差分から消え、誤差棒が1.7倍鋭くなる (sd 43→14)。

- **同一アーム同士は全ディール厳密ゼロ** — これが受け入れ試験。
- **異なるアームで厳密ゼロなら「差なし」ではなく「モデルが読まれていない」。**
  T3街が丸ごと無効だったバグ (正典doc §10、価値 +0.286) はこれで見つかった。

### 7.5 一つの街で測って全街に一般化するな

蒸留は T3 で18%改善、T0 で42%悪化した。効果量に対する**検出力を先に見積もれ** —
T3-BBの全振れ幅は 0.286 しかなく、CI±0.12 では全幅の42%までしか見えない。

---

## 8. GCP運用

**ゾーン計画**: europe-west4(10) → us-west1(31) → us-central1(30)。
`--placement-offset` は**先頭からの切り詰め (回転ではない)**。
レギュラー版が us-west1 を埋めているときは **offset 41 で us-central1 直行**。

**罠 (すべて実際に踏んだ)**:

1. `--dry-run` の quota は**作成時の実容量ではない**。全offsetで「入る」と答えて
   実際は40台中11台しか作れなかった。**起動直後に必ず実測。**
2. `--skip-done` は quota チェックの**後**に効く。要求シャード数がまるごと
   入らないと **SystemExit** (警告ではなく死)。
3. 判断はすべて**公開済みシャード番号**で。台数・ラベル数・生死は代用にならない。
4. **マッチ艦隊にリーパーは付いていない。** 完走VMが quota を占有して残りが
   起動できない (tr3で90分・$10)。投入と同時に張る。
5. 補充の完了判定は**最終成果物 (npz)**。ラベル完了で監視を外すと
   「エンコード45/60で永久待機」を踏む。
6. トレースは**チャンクごとに書く** (`--trace-chunk`)。tr2は22シャード中13を
   プリエンプトで失った。粒度ではなく**書き出しの頻度**が本体。
7. **記録された故障は構成込みで読む。** [[gcp-zone-reachability]] の
   「europe-west4-a は GCSに届かない」は今夜の艦隊には当てはまらなかった。
8. python の `write_text` は Windows で CRLF 化し bash を壊す。パッチ後は LF 正規化。
9. `shell=True` は gcloud のパスの空白で壊れる。argv リストで渡す。

**実測 Spot 価格** ([[gcp-spot-pricing-measured]]): c4-standard-8 は
東京 $0.293/h、us-west1 $0.237/h、europe-west4 $0.165/h。

---

## 9. `ai/docs/` 索引 (33件)

**現行 (HU梯子)**
| doc | 位置づけ |
|---|---|
| `hu_handover_20260821.md` | 今夜の引き継ぎ。**再開時はここから** |
| `hu_gen2_onpolicy_20260819.md` | **正典 §1-19**。gen-2 lap の全記録 |
| `hu_regen_distill_ladder_20260818.md` | **方法論の正典**。「地面は厳密解、ラベルは消耗品」 |
| `hu_all_turns_contract.md` | 行動順・盤面形状の契約 (2026-07-12〜) |

**FL14**
`fl14_joker_plan_20260811` / `fl14_improvements_20260813` / `fl14_two_laps_20260814` /
`fl14_lap3_start_20260814` / `fl14_lap3_t2_regret_20260814` / `fl14_lap3_t3_teacher_20260814`

**FLソルバ基盤**
`fl_solver_v3_design` / `fl_solver_benchmark` / `bust_prevention_algorithm` /
`pruning_strategies` / `speedup_approaches`

**設計判断・ロードマップ (第2層)**
`engine_unification_decision_20260728` (accepted, B-phased) /
`roadmap_improvement_proposal_20260728` / `all_turn_ai_goal_and_milestones` /
`teacher_speedup_design_20260801` / `rules_contract_audit_20260731` /
`value_chain_milestones_20260731` / `bottom_up_iteration_loop_20260729`

**m3 較正**
`m3_behavior_calibration_production_runbook` / `m3_behavior_posterior_calibration`

**CFR**
`t3_public_cfr_plan`

**legacy (2026-05〜06、第1〜2層初期)**
`oracle_first_strength_plan_20260602` / `t1_t2_strength_status_20260602` /
`top1_5s_refinement_strategy_20260531` / `bottomup_to_rl_roadmap` /
`rl_environment_spec` / `ofc_computation_roadmap` / `policynet_pruning_pipeline` /
`vn_improvement_roadmap`

---

## 10. 宿題 (優先順)

1. **lap 2 の出荷判定** — 3アームミラー。[`hu_handover_20260821.md`](hu_handover_20260821.md) §2。
2. **fl_ev(14) を 6.57 へ** — 世代境界で、fl_solver のハードコードと同時に。
3. **T1/T2 の蒸留** — 未測定。T3で効きT0で悪化した中間が不明。
4. **T0 蒸留の再測定** — コーパスが2.3倍になった条件で。
5. **FL専用シミュレータ** — fl_ev測定の高速化。幅指定でFL開始・ヒーローは
   `fl_solver::frontier::best_response` の厳密解。全試行がFL観測で幅14なら100倍効率。
   循環注意 (ステイ判断が fl_ev を使う) で2-3回反復が要る。
6. **掃除** — `ai/tmp_worker_*/` 10ディレクトリ、`ai/ai/`、`ai/logs/` の残骸。

---

## 11. オーナーとの仕事の仕方

- **日本語**。
- **データ駆動のみ。** ルールベースの手当ては明確に拒否されている
  (「あまりルールは使いたくないです」)。
- **課金操作は事前承認。** 艦隊投入・大規模対戦は必ず伺う。
- **報告間隔は指示に従う** (この夜は10分→15分に変更された)。
- **通知の数値を鵜呑みにして報告しない。** 節目の数字は実測で裏を取る。
- **オーナーは測定の穴を突いてくる。**「悪化したらダメでしょ」「データ量ならlapする
  必要は？」はどちらも結論の土台を崩す正しい指摘だった。**反論する前に測り直す。**
