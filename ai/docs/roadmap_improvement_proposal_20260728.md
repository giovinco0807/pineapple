# 全ターンAIロードマップ改善提案 (2026-07-28)

本書は`ai/docs/all_turn_ai_goal_and_milestones.md`（2026-07-13版）に対する改善提案で
ある。既存の完成条件、`promotion_gate_v2`、M0〜M7の定義は変更しない。各提案の採否
は個別に決定し、採用時のみ正典ドキュメントへ反映する。対象はジョーカー2枚込み
54枚HUトラックである。

## 背景となる観察

1. M0〜M3で正しさ検証（hash binding、readback、tamper検出）は高水準に達したが、
   「新路線が既存baselineより強い」ことを示す測定はまだ1件もない。強さの測定は
   M3b distillation完了後まで計画上存在しない。
2. `ai/reports/joker_rule_migration_20260711/`の分析では、旧T3方策のmean regretは
   0.0290、真に劣るTop1は3.385%に留まる。T3単独の伸び代は小さい可能性があり、
   伸び代の大きいターンがどこかは未測定である。
3. regularトラック（`regular-ofc-pineapple/`）はM3.0で両席exact T4を先に実装し、
   first-seat p95 42.91ms、fresh 100-paired counterfactualで+0.295 EV/hand
   （95% CI [+0.1204, +0.4696]）という定量根拠を早期に取得した。jokerトラックは
   T4 BTNのみexactで、T4 BBはpublic mixture solveの対象に残っている。
4. behavior calibrationの元となるT1/T2 checkpointは旧学習契約のため
   `promotion_eligible=false`である。「良いT1/T2 behaviorには下流T3/T4方策が必要、
   T3/T4を解くにはT1/T2 behaviorが必要」という循環の反復回数・収束基準は未定義。
5. Fantasyland EVは方策依存の内生値だが、固定定数（version/hash付き）として均衡を
   解いている。定数の誤差がFL進入判断（T0/T1方策）へ与える影響は未測定。
6. 2トラックでT4 exact、CFR、gate、Rust engineを重複実装している。regular側の
   RL-ready engine（scalar/batch parity、packed batch約8,149 decisions/s）の方が
   新しく、情報漏洩対策も監査済みである。

## 提案1: 早期strength probe（ターン別伸び代の測定）

M4以降へ投資する前に、「候補コンポーネントを1ターンだけ差し替えたpaired
self-play」をアルゴリズム検証専用のprobeとして定義する。共通乱数、seat swap、
raw count/sum/sum_squaresからのSE独立再導出という既存holdout evaluatorの様式に
従い、promotion証跡としては使わない（`algorithm_validation_only=true`）。

- 受け入れ条件: paired 100〜1,000ハンド、95% CI、tail loss記録、oracle/PIMC不使用。
- 期待する出力: ターン別（T4のみ差替→T3追加→…）のEV改善の分解。投資配分の根拠。
- 概算: 実装1〜2日＋ローカル計算。
- **第0弾（T4 BBコンポーネントregret probe）実施済み（2026-07-28）**:
  レガシーmyopic T4規則（`best_t4_completion(opponent=None)`）のexact規則に対する
  取りこぼしは60 rootでfire率3.3%、mean regret約0.016〜0.024点/hand、max 0.62点。
  T3移行分析（旧方策mean regret 0.029）と合わせ、**T3/T4の伸び代は合計0.1点/hand
  オーダーと小さく、伸び代はT0〜T2に集中**という提案1の作業仮説を裏付けた。
  M4〜M6への投資優先を支持。詳細は
  `ai/reports/t4_bb_exact_vs_myopic_probe_20260728/README.md`。
  残作業はpaired self-play形式の本体probe（T0〜T2差し替え分解）。

## 提案2: T4 BBのexact化（本提案と同時に実装）

宣言されたuniform exchangeable restart belief（未見26枚から相手最終3枚ドローを
一様抽出）の下で、T4 BB（`t4_first`）の全合法配置を
`max_a E_uniform-deal[相手exact best responseの負値]`で直接解くresolverを追加する。
列挙部品は`ai/tutor/exact_late.py`の`exact_t4_opponent_response_distribution`として
既存であり、`t4_btn_exact_resolver.py`と同一のfail-closed契約様式で包む。

これによりT4はBB/BTN両席がsolver不要になり、MCCFR/教師生成の対象が実質T3のみに
縮小する。regularトラックM3.0と同じ近似宣言を明示する: これは宣言belief下のexact
であり、相手behaviorに条件付いたBayes posterior、Nash証明、未見局面への一般化では
ない。`promotion_eligible=false`、`serving_changed=false`を固定する。

- 受け入れ条件: 全合法手列挙、全C(26,3)=2,600ドロー列挙、相手側exact best
  response、hidden情報不使用、fail-closed source/runtime binding、独立再計算
  テスト通過。
- 実装: `ai/tutor/t4_bb_exact_resolver.py`（Python reference）。Rust高速化は
  必要になった時点で既存public CFR T4 leaf batch経路を流用する後続作業とする。

## 提案3: behavior感度分析（校正循環の収束見積を先に取る）

locked calibration完了後、calibrated behaviorへ温度・混合摂動を与えた複数条件で
T3 solveを実行し、結果方策のTV距離とroot EV差を測定する。感度が小さければ固定点
反復の回数削減の根拠、大きければcalibration精度目標の根拠になる。M7でぶっつけ
本番の固定点反復を行う前に、収束見積を安価に取るのが目的である。

- 受け入れ条件: 摂動groupごとに独立seed、TV/EV差のSE付きレポート、
  `promotion_eligible=false`維持。
- 概算: 2〜3日＋計算時間。

## 提案4: FL EV感度分析

`ai/config/fl_ev.json`を±20%した3条件で同一rootのT3（可能ならT4含む）を解き、
方策TV距離とFL関連行動（FL狙い配置の選択率）の変化を測定する。結果に応じて
M7の凍結条件へ「FL EV感度が閾値内であることの証明」または「FL EV再推定を含む
固定点反復」のいずれかを追加する。感度が小さいならそれ自体が現行固定値で凍結
してよい根拠になる。

- 概算: 1〜2日＋計算時間。
- **T4パイロット実施済み（2026-07-28）**: FL-live・バスト不能fixture 36 root ×
  スケール{0.8, 1.0, 1.2}で、トップ行動変化0/36、baseline行動regret厳密に0
  （EV水準は最大±8.7点シフト）。T4層は±20%比例誤差に頑健。ただし現行configは
  14枚FL entryのEVが0.0のため比例スケーリングでは14枚entryの誤評価を検出できず、
  T0/T1層では加法摂動の併用が必要。詳細は
  `ai/reports/t4_bb_fl_ev_sensitivity_20260728/README.md`。残作業はT3 solveの
  TV距離測定と早期ターンの加法摂動。

## 提案5: serving engine一本化のdecision record

M4着手前に「最終servingエンジンはどちらのトラックか」を決定し、decision record
として残す。候補は次の2つ。

- A: jokerトラックの現行スタックを続行。regular側の成果は参照のみ。
- B: regular側RL-ready engineへjokerルール（54枚デッキ＋canonical制約置換評価器）
  を移植し、`include_jokers`相当のフラグで一本化。移植は1〜2週だが、以後の
  T4 exact/CFR/RL基盤/性能最適化の二重実装が消える。

どちらでも成立するが、未決定のままM4〜M7を進めると片側が捨て作業になる。

## 提案6: 運用の更新

- 見積の基準日を更新する。「残り4〜6週」は2026-07-13起点であり、以後jokerトラック
  は停止している。トラック間の優先度（どちらを先に凍結するか）を明文化する。
- gate/audit機構の強化は現水準で凍結し、以後の実装リソースは教師生成・学習・
  strength測定へ充てる。新しいgateはgate versionの昇格が必要な変更に限る。

## 実装順序の提案

1. 提案2: T4 BB exact resolver（本書と同時に実装、下記）
2. 提案5: engine一本化のdecision record（0.5日、計算不要）
3. M3a残作業: natural評価とlocked calibrationの完了（既存計画どおり）
4. 提案3・4: 感度分析2件（calibration完了後、M7設計の入力として）
5. 提案1: strength probe（T4両席exact＋T3候補が揃った時点で第1回）
6. 判断: probeの結果でM4以降の投資配分を決定

## 本提案と同時に実装したもの

- `ai/tutor/t4_bb_exact_resolver.py`: T4 BB（`t4_first`）のuniform-deal exact
  resolver。`t4_btn_exact_resolver.py`と同一のfail-closed契約様式。
- `tests/test_t4_bb_exact_resolver.py`: 独立brute-force再計算との一致、
  改ざん・runtime drift拒否、hidden情報非使用の検証。

非claim: 本resolverはアルゴリズムコンポーネントであり、promotion、serving変更、
未見局面への一般化、Bayes posterior、Nash証明のいずれもclaimしない。
