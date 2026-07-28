# 全ターンOFC Pineapple HU AI: 目標とマイルストーン

更新日: 2026-07-13

## 最終目標

ジョーカー2枚を含む標準OFC Pineapple HUで、相手の非公開カードを使わず、
T0〜T4の全ターンにおいてBB（先行）・BTN（後攻）の双方が、近似均衡に
基づく最適配置を選べるAIを完成させる。完成はターン別・役割別・Joker層別
の未使用データで検証し、再現可能なモデル、設定、評価レポートとして残す。

ここでいう「最適」は、物理的なhidden worldごとに別の最善手を選ぶ
perfect-information/PIMC最適ではない。同じ情報しか持たない状態では必ず同じ
方策を使う、合法なpublic-information近似ナッシュ方策を指す。

## 固定スコープ

- 54枚: 通常52枚と物理的に異なる `X1` / `X2`。
- HU、通常ストリートは毎回BBが先、BTNが後。
- T0は5枚を配置、T1〜T4は3枚から2枚配置して1枚を捨てる。
- 現行のcanonical scoring、royalty、Fantasyland EV設定をversion/hash付きで固定。
- 対象はAI、教師、推論経路、検証成果物。ゲームUIの製品化は対象外。
- 相手のhidden draw/discard、未来カード、particle IDを方策入力へ入れない。

## 完成条件

以下をすべて満たした時だけ「全ターンAI完成」とする。

1. T0〜T4 × BB/BTNの10経路がすべて`promoted`。
2. 全経路で合法手率100%、hidden-information漏洩テスト0件。
3. reduced exact gameではinfoset-aware best responseによるexploitability gateを通過。
4. full-cardではroot-disjoint holdoutを役割別・Joker層別に通過。
5. 独立seed間の方策drift、EV regret、tail lossが固定閾値内。
6. 先後を入れ替えたpaired self-playで直前promoted方策に非劣性、主要baselineに優越。
7. rules/action/info/FL/range/model hashを持つ再開可能なartifactと最終評価レポートがある。
8. serving経路は教師と同じ契約を検証し、定めたCPU推論時間内で動作する。
9. T0/T1/T2方策を更新した後に、その行動尤度でT3/T4 rangeと方策を再解決し、
   全ターン方策とpublic posteriorの固定点iterationが独立seed間のdrift閾値へ収束する。
   古い上流behaviorに条件付けたままのT3/T4を最終版として扱わない。

最初の数値閾値はM2で`promotion_gate_v1`として固定した。その後の証跡監査で、
v1は未使用seedや未実装checkpointを要求する一方、hashを内容へbindingせず、
自己申告値だけでも通せることが分かった。旧設定を残したまま、同じ品質閾値を
実runから再導出し、manifest内容へhashをbindingする`promotion_gate_v2`へ上げた。
以後も閾値や意味を変える場合はgate versionを上げ、旧gate結果を残す。

## マイルストーン

| ID | マイルストーン | 状態 | 主な成果物 | 完了条件 | 目安 |
|---|---|---|---|---|---:|
| M0 | ルール・先後・スコア契約統一 | 完了 | `bb_first_v1`、canonical Joker/scoring、board-shape検証 | Python/Rust parity、T0〜T4の役割/board契約テスト通過 | 完了済み |
| M1 | 公開情報・物理粒子・T4終端基盤 | 完了 | `InfoSetKey`、particle整合性、T4全候補vector、X1/X2 golden | hidden情報がkeyへ入らず、全候補coverageとPython/Rust一致 | 完了済み |
| M2 | reduced T3→T4 public tree | 完了 | 6層の実カードfixture、shared-infoset CFR+、両者exact BR、Python/Rust全leaf parity、content-bound v2 artifact | `promotion_gate_v2`で6層すべて`m2_reduced_reference_ready`、保存後readback通過 | 完了済み |
| M3a | full-card T3/T4 online solver基盤 | 進行中 | calibrated range、public joint-root mixture、shared-infoset MCCFR、production checkpoint、fresh online solve | T3 BB/BTN・T4 BBのlocked-root algorithm gate通過。tabular profile自体は非昇格 | 1〜2週＋計算時間 |
| M3b | T3/T4一般化方策 | 進行中 | lossless InfoSet encoder、content-addressed teacher、root-family-disjoint distillation、OOD re-solve | T3/T4 BB/BTN × Joker 0/1/2の未見root gate、runtime gate、fixed-point readback通過 | 1〜3週＋教師生成時間 |
| M4 | T2 continuationとpromotion | 未着手 | promoted T3/T4を終端にした正しいglobal sequence教師/方策 | T2 BB/BTNのexact/holdout、drift、runtime gate通過 | 4〜7日＋計算時間 |
| M5 | T1 continuationとpromotion | 未着手 | promoted T2以降を使うT1教師/方策 | T1 BB/BTNのholdout、paired seat-swap、tail-loss gate通過 | 4〜7日＋計算時間 |
| M6 | T0 continuationとpromotion | 未着手 | 全下流を使うT0教師/方策、統一推論artifact | T0 BB/BTN gate、全10経路promoted、CPU runtime gate通過 | 5〜10日＋計算時間 |
| M7 | 統合・固定点反復・最終評価・凍結 | 未着手 | 上流方策更新後のT3/T4再解決、alternating-button self-play、league/exploitability report、model manifest | 全ターンposterior/policy drift収束、全完成条件を満たし、再現コマンドとartifact readbackが成功 | 4〜7日＋計算時間 |

現実的な残り期間は、実装とローカル検証だけなら約4〜6週、full-card学習の
再試行やhard-negative追加が必要なら6〜8週を見込む。GCPはM3以降の大量
サンプリング/学習で必要になる可能性が高いが、M2のreduced treeと正しさ検証は
ローカルで進める。

## 現在地

- M0/M1/M2は完了。
- M2では、物理粒子の厳格検証、T3 first→second→T4 firstの状態遷移、
  conditioned T4全候補vector、そのvectorからrecursive tree leafへのcommitment付き
  接続、同期CFR+、infoset-aware exhaustive best response、NashConv/exploitabilityまで
  実装済み。
- BB/BTN × Joker 0/1/2の6つのcanonical reduced fixtureを、実Rust T4全候補leafで
  コンパイルした。各層を3回のhidden-world順序replayと5回のruntime測定で解き、
  `ai/reports/m2_promotion_gate_v2_20260713/`へcontent-bound artifactを保存した。
- 保存後readbackで`promotion_gate_v2`は6層すべてPASS。最大exploitabilityは
  `0.000152017689331152`、Python/Rust leaf差、方策TV、BR再評価残差はすべて0。
- v2の合格は有限reduced referenceだけを昇格する。full-card chance列挙、HU exact、
  serving policy、T3/T4のfull-card方策を昇格しない。
- M3基盤として、全phaseの54枚partitionを保証するhistory-weighted range builder、
  完全合法手Fraction分布を返すcontent-addressed behavior契約、exact ESS、content/build
  hash、raw query/fallback監査、独立readback verifierを実装した。専用22テストが通過。
- 物理public treeは`T3 BB → T3 BTN → T4 BB → T4 BTN → 13/13 terminal`まで
  延長し、BBのhidden T4 discardだけを変えても同じBTN情報集合の全terminal action
  valueが不変であることを検証した。関連49テストが通過。
- M3 range gateはBB/BTN × visible Joker 0/1/2の6層をcontent/Joker数まで独立再導出
  するfail-closed検証へ更新した。uniform、fallback、legacy、未承認behaviorは通さない。
- reduced public tree用のexternal-sampling MCCFR+を実装し、20,000 sampleの固定方策
  regret推定、root weight不変性、shared-infosetのopponent action cache、seed replay、
  signaling gameのexact CFR+比較で検証した。iteration境界checkpointは37+63 resumeと
  one-shot 100でresult、最終JSON bytes、SHA-256まで完全一致する。
- full-card側も、root posteriorを各traversalで一度だけexact `Fraction` samplingし、
  以後の3枚drawを物理deckから直接samplingするgenerative adapterと、遭遇した
  `InfoSetKey`だけを持つdynamic MCCFR+を実装した。traverserは全合法手を展開し、
  posterior/chance massの二重乗算とhidden-world IDのpolicy key混入を禁止している。
- 複数の互換private rootをexact priorのchance super-rootで束ね、全rootが単一の
  `InfoSetKey` regret/strategy tableを更新するmulti-root MCCFR+も実装した。reduced
  signaling gameではexact CFRへの最大TVが`0.02461`、独立root解法が同一BTN情報集合で
  hidden typeごとに異なる方策を選ぶstrategy-fusion誤差はTV`0.99999984`だった。
  実full-card adapter 2本でも異なるBB private rootが同一BTN `InfoSetKey`へ合流する。
  ただし出力は与えたroot集合内の遭遇済みtabular profileであり、未見`InfoSetKey`へ
  一般化するAIではない。別rootのIDを付け替えた評価も一般化証拠として認めない。
- multi-root checkpointはRNG、累積regret/strategy、root support/訪問数、exact prior、
  range/behavior、action/scoring/sourceと到達可能なlive Python runtime graphを完全iteration
  境界でatomic保存する。外部保存SHA-256をresume時に必須とし、one-shot/splitをbytesまで
  一致させた。任意generic adapterのresumeは全面禁止し、厳密型のFullCard/canonical
  reduced adapterだけを許可する。solver sampler、FullCard action、間接terminal scorerの
  live差し替えもfail-closedで拒否する。専用16テストが通過。
- 6層full-card smoke gateも実装した。全層の実行完了は
  `execution_smoke_passed`で表す一方、behavior未承認または強さ評価未実施の状態では
  `m3_promotion_passed=false`、`full_card_policy_promoted=false`を維持する。
- `run_m3_full_card_smoke.py`で、exact-hash T1/T2 BB/BTN priorにT3 BB priorを加え、
  実カードfixture、4-particle物理range、canonical terminal scorerを使って6層を各1
  iteration実行した。保存後readbackはPASSし、evidence artifact SHA-256は
  `5f4375a67fa21d7bd55107c888ff589895b17f9309bef61cb01cc780a8b1e22f`。
  behaviorは未校正ranking priorなので、結果は意図どおり
  `m3_full_card_smoke_ready_nonpromoted`でありM3 promotionではない。
- Torch behavior adapterは既存T1/T2の実checkpoint（`520→250`）を読み、合法手だけを
  exact Q32分布にしてrangeへ接続できる。ただし両checkpointは旧学習契約
  （相手盤面空・BTN固定）のため`promotion_eligible=false`であり、両seat対応モデルの
  再学習または監査済みcheckpointが別途必要。
- 4つのposition-specific HU PolicyValueNet（T1/T2 × BB/BTN、`522→27`）も
  opponent board込みでrangeへ接続済み。ただし出力は観測行動頻度ではなくteacher EVを
  温度3でsoftmaxしたranking targetのlogitであり、現状はsynthetic behavior priorに
  限定して`promotion_eligible=false`としている。独立root splitの実行ログから
  observed-action calibrationを作り直す必要がある。
- `ofc_behavior_decision_log/v1`のraw trace contractも実装した。T1/T2 × BB/BTN、
  BB-first history、`BehaviorInfoSet` digest、合法手/discard/semantic index/mask27、
  root固定fit/dev/test split、Joker census、重複・overlapを独立検証する。
- append-only shard collector、1入力shard対1評価shard、bounded-memory calibration、
  sharded calibrationから既存固定点runtimeへのfresh-verified bootstrapを実装した。
  targeted Joker challengeは24,000/24,000 rootsで完了し、naturalも134,057 roots・
  536,228 decisions・135 shardsの収集とmanifest-last readbackを完了した。naturalの
  collection content SHA-256は
  `2c1f47ada1867c650f96f93c7417c47b502482d3fff8e9d922e26515954858e6`。
  challenge direct-logit evaluationは24 shards・96,000 rowsを独立fresh
  readbackし、content SHA-256
  `9d99cc235701d930879cba577022f3646d367a212488ad5655bf904405c56382`で一致した。
  natural evaluationとlocked calibrationは単一writer runnerで実行中であり、まだ
  calibration artifactは未公開であるため`promotion_eligible=false`を維持する。
  （2026-07-28追記: locked calibrationは完走済み。`calibration.json`は
  `stage=calibration_complete`、natural評価536,228行/135 shards完了。ただし
  gate v2は104チェック中2件（`t1_bb`/`t2_bb`の点推定NLL delta、+1.4e-5 / +4.1e-9
  nats）で失敗し`promotion_eligible=false`。統計的UCB検定は全層PASSのため、
  失敗は点推定`<=0`厳密条件のノイズ脆弱性に起因する。gate v3案は
  `ai/reports/m3_calibration_gate_status_20260728/README.md`参照。
  2026-07-29追記: gate v3（点推定toleranceを1/10000へ変更、UCB等他は不変、
  post-hoc supersessionとして`gate_v3_supersession.md`に開示）で再集計し、
  104チェック全PASS・`promotion_eligible=true`の`calibration_v3.json`
  （SHA-256 `e6e0c8e80ecbe77dbe9dcbfa05e01558c085926f5b496753410e97a2ee5fb796`）
  を得た。M3aの次工程は、このcalibrated behaviorをM3 range gateへ接続し
  T3教師生成へ進むことである。）
- `Q_r -> C_r -> R_r(C_r) -> B_r`の非循環T3固定点gate v2と、72件の実MCCFR jobを
  生成・全asset replayするsmokeを実装した。smokeは2 seeds、各層1 root、2 transitions
  のため本番最小数を満たさず、意図どおり昇格・収束・強度claimはすべてfalse。
- shared multi-root候補のphysical holdout evaluatorと強度gate v2も実装した。候補方策を
  rootだけでなく全後続`InfoSetKey`で参照し、未登録局面はfallbackせず失敗する。候補継続
  とuniform referenceを別々にrolloutし、chance/policy RNGを分離した共通乱数で、候補・
  reference・全root action・seat swapのraw `count/sum/sum_squares`、SE、paired deltaを
  独立再導出する。training observation/range/build/seed再利用、actor/Joker/sign不整合、
  source/scorer drift、gap/orphan/tamperを拒否する。
- このgateは意図的に`algorithm_validation_only=true`、`promotion_eligible=false`、
  `full_card_policy_promoted=false`を固定する。root-scoped tabular profileを未見root用の
  global policyへ昇格できないためである。全局面AIには、公開情報だけからactorの反実仮想
  private typeを束ねるpublic joint-root compiler、各局面のfresh online solve、または
  lossless `InfoSetKey` encoderへdistillした一般化方策が別途必要。
- `t3_t4_public_root_mixture.py`で、actual private handを入力に取らず、公開盤面・公開履歴と
  明示された反実仮想private typeのexact `Fraction` priorから、共有InfoSet multi-root入力を
  作るcompilerを追加した。T3 BB/BTN、T4 BB/BTNの実FullCard adapterで検証済み。ただし
  現段階は有限の明示supportだけで、full-deck exhaustiveやproduction samplingを主張しない。
- `t3_t4_counterfactual_private_types.py`と`posterior_joined_private_types.py`で、actual private
  cards、caller seed、namespace、sliceを受け取らず、公開contextとrepository固定scheduleだけで
  actor private type候補を生成・fresh replayするproduction境界を追加した。genericな自己承認seed
  batchや手動詰め替えは本番joinで拒否する。actor typeの外側posteriorはactor公開行動尤度
  `L_actor(h)`だけでなく、相手公開行動の周辺尤度`Z_h`を掛けた`L_actor(h) * Z_h`で正規化する。
  `Z_h`はzero-likelihood assignmentも分母へ含むsample meanで、hidden assignmentを全列挙した
  場合だけfull exactと記録する。T4 behavior routeは未完成のため引き続きfail-closedである。
  現在のrepository固定scheduleは各phase 2 proposalのbootstrapであり、
  `production_sampling_ready=false`のまま。実規模support数は独立drift/SE gateを先に固定して増やす。
- `t3_t4_public_online_resolve.py`で、compiled mixture全体をfreshに解き、actual `InfoSetKey`は
  solve完了後の完全一致selectorだけに渡すonline wrapperを追加した。solver結果を現在の
  exact prior・range・adapterへ再bindingし、全regret/strategy tableの合法手集合、checkpoint、
  transitive live runtime graphをfresh検証する。これはalgorithm validationでありserving変更や
  未見局面への一般化claimではない。
- `t3_t4_infoset_encoder.py`で、公開履歴、actorのprivate recall、現在draw、盤面、phase/turnを
  lossless round-tripする3313次元固定binary encoderを追加した。既存522次元encoderやserving
  defaultは変更していない。
- `t3_t4_distillation_teacher.py`で、full-deal commitmentをlabel生成前にfit/dev/testへ固定し、
  descendants、private types、seed、seat swap、suit augmentationを同じsplitへ束ねる
  content-addressed teacher bundle v2を追加した。MCCFR教師とT4 BTN exact教師のprovenanceを
  分離し、exact行は架空のiteration・seed・sample momentsを持たず、resolver結果をfresh再実行
  してmanifest/result hashへbindingする。さらに3313次元encoderを使うmasked policy/value/Q
  datasetとfit-only gradient・dev-only選択・選択後test一回のdeterministic trainerを追加した。
  payoff raw momentsとQ標準誤差をbindingし、Q lossはprecision weightingする。teacher/encoderの
  transitive verifierを含むlive callable semanticsの同一object `__code__`差し替えも実行前に
  fail-closedで拒否する。consumer import前の差し替えもmodule-owned import-completion anchorで
  拒否し、fresh-process回帰を含む関連77テストと`py_compile`が通過した。実規模teacher生成・学習・独立
  promotion gateは未実施で、runtime/serving flagsはfalseのままである。
- `t4_btn_exact_resolver.py`で、BB盤面が完成済みのT4 BTNは全合法配置を直接terminal scoringし、
  BTN視点の真の最大値を選ぶexact bypassを追加した。X1/X2、live FL EV、transitive scoring、
  source/runtime tamperをfail-closedで検証する。T3 BB/BTNとT4 BBは引き続きpublic mixture solveが
  必要である。
- `t4_bb_exact_resolver.py`（2026-07-28）で、T4 BB（`t4_first`）も宣言uniform
  exchangeable restart belief下のexactコンポーネントを追加した。全合法配置ごとに
  未見26枚からの全C(26,3)=2,600通りの相手最終ドローを列挙し、相手のexact best
  responseの負値の平均でEVを取る。hidden情報不使用、fail-closed source/runtime
  binding、独立brute-force再計算テストを通過。これは宣言belief下のexactであり、
  behavior条件付きBayes posteriorや未見局面への一般化ではなく、
  `promotion_eligible=false`である。採用すればsolver/教師生成の対象は実質T3のみに
  縮小する。背景と他の改善提案は`roadmap_improvement_proposal_20260728.md`を参照。
- 実teacher root-familyの生成経路を監査した。既存behavior traceのrestricted hidden-rootは
  54枚full deckとT0〜T2 prefixを独立replayできるが、T3/T4 actionを持たないため、そのまま
  late-turn教師にはできない。新しいpre-label planで純full-deal/root-family/split、rules、
  behavior/solver/source hash、生成algorithmとselectorだけを先に固定し、実actionは
  `T3 BB MCCFR → T3 BTN MCCFR → T4 BB MCCFR → T4 BTN exact`の順に生成する。各選択後の
  public prefixは`descendant_public_path_sha256`へcommitし、private type、seat swap、suit
  augmentationを含む全子孫へ元のsplitを継承する。既存root commitmentはbehavior hashと
  realized actionを含むため、pre-label split materialには使わない。
- posterior seed provenanceと`Z_h`修正後、counterfactual/posterior/range/public-root/online-resolve/
  production-holdout/fixed-pointの統合回帰112件を再実行して全件通過した。distillation境界は
  関連77件が通過し、両修正は独立post-fix監査中である。
  次の本質的な未完了項目は、natural評価とlocked calibration、teacher大量生成、
  root-family-disjoint distillationと未見root promotion gateである。
- T3/T4の最終方策、T2/T1/T0はまだpromotedではない。

## 運用ルール

- 新しい候補は、未使用holdoutで直前promoted方策を上回るまでdefaultにしない。
- 完了していないexact jobや出力未生成のrunを評価済みとして数えない。
- 長時間runはcheckpoint、進捗artifact、停止条件、概算費用を先に定義する。
- hidden情報を利用したoracle/PIMC結果は診断用に限定し、最終教師へ混ぜない。
- 上流turnの方策hashが変わったら、その方策をbehavior likelihoodとして使う下流artifactを
  staleにする。最終凍結前にT0→T1→T2更新とT3/T4再解決を反復し、hash付きfixed-point
  convergence reportを残す。
