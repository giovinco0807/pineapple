# Regular OFC Pineapple HU AI 引継ぎ (2026-07-21)

2026年7月21日時点で、コード・設定・成果物・Git状態を再確認した引継ぎ文書。
そのまま他のAIへ渡せる。

検証は次のread-onlyスクリプトで自動照合できる:

```powershell
powershell -File scripts\Test-HuJointPolicyHandoff20260721.ps1 -RunTests
```

## 0. 絶対に守る制約(最初に読むこと)

- `current`を変更しない
- baseline profile/modelを削除・上書きしない
- full replacementを早期有効化しない
- 相手private discardをpolicy入力・label featureへ混ぜない
- teacher EVを実戦EVとして報告しない
- teacher EV LCBをruntime gateへ直接使わない
- top1 accuracyだけで採用しない
- candidate選択と最終評価のRNGを分離する
- `--seed-stride`とseed provenanceを記録する
- locked holdoutでthresholdを再探索しない
- 失敗したlocked seedを追加seedで救済しない
- aggregate EVで片seat・片opponentの悪化を隠さない
- lifecycle canary合格前にmulti-VM fanoutしない
- Cloud retry・attempt1・第三VMを勝手に行わない
- Cloud費用はphaseごとに別途認可を取る
- dirty worktreeを整理しない(`git reset --hard`、`git clean`、
  checkoutによる巻き戻し、無断削除は禁止。大量の未コミット変更は
  ユーザーの作業である)
- Step12c runnerを再実行しない(identityは消費済みterminal)

## 1. リポジトリ

- ローカル:
  `C:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\regular-ofc-pineapple`
- GitHub:
  `https://github.com/giovinco0807/pineapple`
- branch:
  `codex/regular-ofc-pineapple`
- 引継ぎbaseline HEAD:
  `623e3948cb70b05be2958f0b1c63b6f02c7fb756`
  (2026-07-21のR0b/Step12d/引継ぎ整備コミットがこの上に積まれる。
  検証スクリプトはbaselineが現HEADの祖先であることを確認する)
- dirty worktree (2026-07-21、本引継ぎ作成前時点):
  - tracked modified: 70件
  - untracked: 1,000 entries(ディレクトリ折り畳み表示)、
    `-uall`では1,018ファイル
  - 本引継ぎ作成で新規untrackedが3件追加された:
    本文書、Step12d決定文書、検証スクリプト
    (なおロードマップ`hu_joint_policy_full_hand_rl_milestones_20260718.md`
    自体もuntrackedであり、2026-07-21に冒頭へ注意バナーを追記済み)
- `src/ofc_regular/ai_profiles.py` SHA-256:
  `d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3`

注意: untracked成果物はGit保護がない。ユーザー自身によるWIPブランチへの
コミットまたは外部バックアップを推奨する(AIが勝手に行ってはならない)。

## 2. ルールと最終目標

対象はHeads-Up Regular OFC Pineapple。

- 52枚、ジョーカーなし
- T0: 5枚を全配置
- T1~T4: 3枚から2枚配置、1枚を非公開で捨てる
- 最終盤面: top 3 / middle 5 / bottom 5
- 相手の捨て札は見えない。自分の捨て札だけ認識可能
- 通常royalty、foul、scoopを含むHU score
- middle trips royaltyは2点
- FL entry/stayは14枚
- 現在のFL bootstrap EV: `10.227020614683454`

最終目標は、先攻・後攻の両プレイヤーが同じfull handでT0からT4まで
対戦するpolicyである。「最適」は次を区別すること。

1. 特定相手へのbest response
2. 強いself-play policy
3. 経験的に低exploitabilityなpolicy
4. 数学的に証明されたNash/完全最適解

目標は2と3。4は主張しない。

## 3. 既存固定AIチェーン

明示的に`stage19_p0`を選択した場合のrollback chain:

```text
T0 stage19_p0
  -> T1 stage18_p1
  -> T2 stage9f_p2
  -> T3 stage7_m5_r10
  -> T4 existing exact final-turn path
```

状態:

- `stage19_p0`: T0先攻限定selective override。100,000 paired seeds、
  EV/hand `+0.0173028`、95% CI `[+0.0082918,+0.0263139]`。後攻T0未完成
- `stage18_p1`: T1先攻限定selective override。後攻T1未完成
- `stage9f_p2`: T2先攻・後攻selective override
- `stage7_m5_r10`: T3 rollback baseline。margins `5.0 / 10.0`
- `current`: この固定チェーンとは別の既存stage9d mapping。
  joint-policy作業では変更していない

重要な注意:

- 旧T2 Stage9fとT3 Stage7/referenceの学習cacheには、実現した相手
  private discardを含むlegacy feature lineageがある。P0/P1もその
  continuationを教師・評価に使用していた。
- したがって旧EVや旧Go判定はlegacy baselineとしては残すが、新しい
  hidden-discard-safe joint policyのpromotion証拠には使えない。
- baseline、model、profileは削除・上書き・黙って再学習してはならない。

## 4. T4の現在地

M3.0 exact T4 componentはローカル実装済みで、明示opt-in component
としての検証は通っている。

- 後攻: 完成した相手盤面に対し、heroの全合法配置をterminal exact評価
- 先攻: unknown 24枚から全2,024 opponent dealを列挙し、各dealで
  相手の全合法応答を評価。一様exchangeable beliefの下でexact
- 全legal ActionKeyと各action EVを返す
- illegal action、unknown field、相手private discard、realized deck
  tailを拒否
- Python/Rust、scalar/batch、ActionKey mapping、カード順序のparity確認済み
- 1,000-state pilot: first p95/p99 `42.91/54.53 ms`、second p99
  `1.70 ms`、batch `156.04 roots/s`

release runtimeのR0bは、当初No-Goだったが**2026-07-21に解消(Go)**。
経緯:

- 歴史的contractがpinしているDLLが現在欠落
- rebuild candidateはexact/parity等に合格したが、second-seat p99が
  `2.3487 ms > 2.0 ms`
- 解決策: (1) 元のpinned DLLを復旧してSHAを確認、または
  (2) 新version contractとfresh seedsで検証をやり直す。
  (2)の場合は閾値2.0msの根拠(ウォームアップ条件・計測環境)を
  新contractで明文化すること。どちらを選ぶかはユーザー判断
- **2026-07-21調査結果**: (1)はローカルでは不可能と判明。元DLLは
  2026-07-18のm30_build再ビルドで上書き破壊された。C:/D:全域・
  ゴミ箱・git履歴・r0 snapshotに一致コピーなし。残る可能性は
  VSS shadow copies(要管理者権限)と外部バックアップのみ
- **2026-07-21 v2検証ラダー結果**: (2)を事前登録・実行
  (`configs/hu_joint_policy_m30_t4_runtime_v2_candidate.json`)。
  pilot100は全ゲート合格(7/18のNo-Goは外れ値と確定)、pilot1000は
  後攻p99 total latencyのみ不合格(3.046ms > 2.0ms)。native p99は
  1.274msでゲート内、超過はPythonラッパーのオーバーヘッドスパイク
- **2026-07-21 v3ラダー(ユーザー認可済み案A)で全ゲート合格 →
  R0bはvalidationレベルでGo**。後攻2.0msゲートをnative latencyへ
  適用し、total p99 ≤ 10msの健全性上限を追加(事前登録:
  `configs/hu_joint_policy_m30_t4_runtime_v3_candidate.json`、
  validator: `src/ofc_regular/validate_hu_m30_t4_runtime_v3.py`、
  ユニットテスト8件)。pilot1000: native p99 0.937ms、total p99
  2.32ms、batch 500 roots/s、semantic全合格。候補runtime
  `hu_m30_t4_exact_both_seats_v3`(DLL SHA `faf1ee...`)。歴史的
  M3.0 contractは無変更。**2026-07-21 ユーザー認可によりrelease
  review完了 → R0b Go**。release contract:
  `configs/hu_joint_policy_m30_t4_runtime_v3.json`
  (`complete_component_go_explicit_opt_in`。named profile追加・
  current昇格・full replacementは引き続き別途ユーザー判断)。詳細は
  `docs/hu_joint_policy_m31_t3_step12d_entrypoint_decision_20260721.md`
  のR0bセクション参照
- T4 componentの数理実装が完成していることと、release runtimeが
  有効化可能なことを混同しない

## 5. T3の現在地

通常T3は、hidden-card particleに対するcommon-random Monte Carloで
全legal actionを比較し、各T4 childはexact評価する。つまり:

- T4 child: exact
- T3 root: Monte Carlo
- full-deck T3全体の数学的exactではない

Candidate02 performance-development:

- 100 paired hands / 200 roots、semantic parity 100%
- first p95/max: `86.05/88.01秒`、second p95: `1.048秒`
- first-seat speedup: `2.700x`、peak RSS: 約124 MB

実装済み基盤: Rust `hu_m3_engine`、information-set-safe observation、
canonical ActionKey、candidate/reference分離、selection/evaluation RNG
分離、common random futures、restart-safe checkpoint/heartbeat、
immutable package、Windows/Linux feature encoder parity、
generation-pinned upload/receive、VM/disk/IAM cleanup、
tamper・collision・unknown fieldのfail-closed。

未完成: independent performance-lock、fresh quality、9,000 paired
labels、`StreetPolicyNetV1`、runtime safety calibration、
population/ABR評価、T3実戦promotion。

Candidate02の速度結果はperformance-developmentであり、AI強度や
promotionの証拠ではない。

## 6. 最新のStep12c結果

Step12c candidate/reference 2-VM lifecycle canaryはNo-Goで安全に閉じた。

認可されていた構成: candidate/reference各`c4-standard-8` 1台、
attempt0のみ。attempt1、自動retry、3台目は禁止。

結果:

1. corrected token barrierは通過
2. Token Creator削除後のzero readbackも成功
3. success receiptへ追加された2フィールドをPhase2 validatorの
   allowlistが認識しなかった
4. 最初のPhase2 IAM mutation前にfail closed
5. Compute insert前なのでVMは起動されていない

修正済み:

- Phase2 validatorが新しい2フィールドとnested evidenceを厳密検証
- `c4-standard-8` memory契約を誤った`32768 MiB`から実値`30720 MiB`へ修正

独立closeout: VM GET `[404,404]`、disk GET `[404,404]`、controller
service account不在、Phase2 IAM 8 bindingsすべて0、result objects 0、
private key/access token保存 0、Step12b tree不変、profile hash不変。

closeout receipt:
`outputs/hu_joint_policy/m31_t3_step6d/step12c_pair_v1_actual/phase2_schema_failure_closeout_receipt.json`

- receipt内部の`receipt_sha256`フィールド(canonical self-hash):
  `31e675593ca168fc899a6dc34083a6f54c8258aabbb62dcd71906d56d77161ef`
- receiptファイル自体のSHA-256:
  `322eea21eb52bff767539127f29677951a074cf87aa4b0f37eace172ffd4235f`

両者は別の値である。`Get-FileHash`で照合するのは後者。

最新のStep12b/Step12c回帰: `275 passed`(2026-07-21に再実行で確認済み)

Step12cは安全性・cleanupの証拠であり、性能・品質・学習・AI強度の
証拠ではない。

## 7. 現在のブロッカー

### Step12c・Step12d identity消費済み

Step12cに加え、**Step12d attempt0も2026-07-21に実行されNo-Go**
(Phase2冒頭のread-only GETが`iam_https_transport_failed`、
最初のIAM mutation前にfail-closed、VM insert 0回)。token barrierは
成功し**Step12c schema修正はliveで実証済み**。独立GET検証でcloudの
完全クリーンを確定。両stepのsigner、nonce、contract、prefix、
output root、SA、VM/disk名、confirmation tokenはすべてterminalで
再利用禁止。**次のcanaryはStep12e。ローカル実装・テスト済み**
(read-only policy GET限定の有界retry: transport障害のみ・最大3回・
backoff 2s/8s・mutation非retry。runner:
`scripts/run_hu_m31_t3_step6d_rearm2_diagnostic_step12e_pair_v1.py`、
token `EXECUTE_STEP12E_DIRECT_V2_EXACT_PAIR_ATTEMPT0`)。
Cloud実行にはfresh explicit authorizationが必要。監査:
`docs/hu_joint_policy_m31_t3_step12d_transport_failure_20260721.md`
決定文書:
`docs/hu_joint_policy_m31_t3_step12d_entrypoint_decision_20260721.md`

### 次のCloud実行には新しい認可が必要

ローカル実装・テストは進められるが、Cloudでcandidate/reference pairを
起動する前にユーザーのfresh explicit authorizationが必要。

### 20 VM構成はquota不足

Tokyo C4 quota 24 vCPUに対し、旧計画は`20 × c4-standard-16 = 320
vCPU`で起動不能。現quotaでは次waveは最大`3 × c4-standard-8`。
**2026-07-21にユーザー認可の下で24→128 vCPUの増枠申請を提出済み**
(quota preference `c4-cpus-asia-northeast1-128`、traceId
`d0cd568c-fbee-4c16-acd2-9122a4aa9cd1`、審査中)。fanout設計の確定は
canary合格後。

### R0bは解消済み(2026-07-21 Go)

exact-T4 Windows release runtimeはv3 release contract
(`configs/hu_joint_policy_m30_t4_runtime_v3.json`)でGo。
T3 runtime promotion / full-hand releaseのR0bブロッカーは解消
(セクション4参照)。

## 8. 次に実施する作業

最優先はStep12d fresh 2-VM lifecycle canary。詳細な必須要件
(producer/validator round-tripテスト、machine-type preflight照合、
disjointnessテスト等)は
`docs/hu_joint_policy_m31_t3_step12d_entrypoint_decision_20260721.md`
に固定済み。

1. Step12b/Step12c成果物をimmutableなterminal evidenceとして保持
2. 新しいStep12d entrypointを追加
3. signer、nonce、contract、prefix、output、SA、VM/disk名を全て新規生成
4. Step12b/12cとの完全非一致をtestで固定
5. dry-runではCloud adapterを構築しないことを確認
6. fixture、tamper、cleanup、resume、receiver testsを追加
7. Step12b~12d回帰とprofile hashを確認
8. ユーザーからfresh authorizationを得る
9. candidate/reference各1台、attempt0のみ実行
10. heartbeat、DONE、self-delete、receive、VM/disk/IAM zeroを独立検証
11. 合格時だけperformance-lockのquota/fanoutを再設計

## 9. M3.1 T3完成条件

Performance-lock:

- fresh 100 paired / 200 roots、semantic parity 100%
- first p95 ≤ 150秒、first p99/max ≤ 240秒、second p95 ≤ 5秒
- candidate/reference speedup ≥ 1.55x、RSS ≤ 0.8 GiB
- missing/censored/mixed binary/image/allocation = 0

Fresh quality:

- fresh 50 paired / 100 roots、別5 pairedで`8/128/4/0` confirmation
- regret mean/p95/p99/max ≤ `0.75/3/6/15`
- hidden truth、unknown field、ActionKey drift、RNG overlap、missing = 0

Dataset:

| Role | Paired hands |
|---|---:|
| train | 6,000 |
| safety-fit | 1,000 |
| threshold-lock | 1,000 |
| diagnostic holdout | 1,000 |

- 各splitの10%だけhigh-precision confirmation。confirmation対象は
  結果を見る前に固定
- 最初の25-paired shardがresume/receive/validateまで合格してからfanout

Model:

- card/zone order-invariant encoder
- public-history/belief encoder
- semantic ActionKey encoder
- policy/value/Q/delta/uncertainty/safety heads
- first/second共有backbone、seat embeddingとseat別calibration
- interference gate不合格時だけsmall seat adapter

Runtime override:

- candidateとbaselineのActionKeyが異なる
- `predicted delta - downside p95 - ensemble disagreement > 0`
- seat別safe probabilityがlocked threshold以上
- schema/action mapping/model hashが一致

非発火時はbaselineのAction objectをそのまま返し、policy RNG、future
deck、trajectoryを一切変えない。

Promotion:

- 5 opponent policies、3以上のABR families
- valid overrides ≥ 300、各seat ≥ 100
- gain/overrideとEV/handのCI95 lower > 0
- false positive ≤ 0.30、各seat ≤ 0.35
- loss p95/p99/max ≤ `25/40/50`
- opponent別mean ≥ -0.005、CI lower ≥ -0.02
- non-fire trajectory mismatch = 0
- ABR worst-response ≥ -0.01 point/hand
- 合格しても新しいopt-in profileのみ

## 10. RL-readyとその後

依存順:

```text
Step12d lifecycle canary
  -> quota/fanout再設計
  -> T3 performance-lock
  -> T3 fresh quality
  -> 9,000 labels/model/promotion
  -> RLA full-hand schema
  -> RLB Rust mechanics
  -> RLC PyO3 batch
  -> RLD replay/population
  -> RLE RL-entry pilot
  -> T2 curriculum
  -> T1 both-seat curriculum
  -> T0 both-seat curriculum
  -> explicit 14-card FL
  -> joint league/CFR/ABR
  -> empirical low-exploitability opt-in profile
```

RLはstreet別の別ゲームではない。常に両プレイヤーがT0~T4を通して
対戦するfull handで、後段をfreezeしてT2→T1→T0の順に改善する。

現在未実装: `rust/hu_rl_engine`、`hu_rl_contract.py`、
`hu_rl_reference.py`、`street_policy_net_v1.py`、
`hu_search_teacher_v1.py`、`hu_expert_iteration_v1.py`、
`hu_population_league_v1.py`、`hu_abr_v1.py`

RL API予定: `reset_batch`、`observe_batch`、`legal_actions_batch`、
`step_batch`、`snapshot_batch/restore_batch`、`search_label_batch`、
`rollout_population_batch`

ActorObservationに含める: hero board、opponent public board、hero
private discards、dealt cards、seat/order/turn、FL/scoring context、
public history、observationから構築したbelief

含めない: opponent private discard、realized deck tail、world
identity、audit truth

## 11. 学習方式

採用方式:

- search teacher付きExpert Iteration
- population/league full-hand self-play
- cycle 2から低weight V-trace
- ReBeL型public-history belief
- late-street restricted MCCFR
- learned approximate best response

避ける方式:

- sparse terminal rewardだけのPPO/DQNを最初から実行
- full-game Deep CFR
- 1種類の相手へのbest responseを最終policyとすること

各streetで2 Expert Iteration cycles:

1. balanced full-hand population roots
2. searchでQ/delta/uncertainty/hard negatives
3. policy/value/Q/safetyへdistill
4. disagreement/tail/false-positive hard negatives追加
5. cycle 2だけV-trace追加
6. fresh population/ABR seedsでlocked評価

予定scale: T2 18,000 / T1 22,000 / T0 34,000 roots(各pilot後)。
T0は232 actions粗評価 → top24+baseline search → top8 high-precision MC。

FLは環境へ14-card modeを最初から持たせる。初期normal-hand学習は固定EV
でbootstrap可能だが、joint promotion前にentry/stayをexplicit
transitionへ変更する。

M6は最大3 league iterations。ABR-gap 95% upperがM5比20%以上改善し、
absolute empirical proxy ≤ 0.10 point/handを要求する。2回連続plateau
ならNo-Goで終了。

## 12. 最初に実行するread-only確認

自動照合(推奨):

```powershell
Set-Location C:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\regular-ofc-pineapple
powershell -File scripts\Test-HuJointPolicyHandoff20260721.ps1 -RunTests
```

期待結果: `ALL CHECKS PASSED`(回帰は`275 passed`、約4分)

手動確認する場合はスクリプト内の期待値を参照。

## 13. 主要資料

- 最新ロードマップ:
  `docs/hu_joint_policy_full_hand_rl_milestones_20260718.md`
  (冒頭・末尾に過去Step時点の記述が残る。冒頭のバナーとI1 Step12c
  チェックリスト、Step12c/12d専用文書を優先)
- Step12d決定文書:
  `docs/hu_joint_policy_m31_t3_step12d_entrypoint_decision_20260721.md`
- Step12c監査:
  `docs/hu_joint_policy_m31_t3_step12c_phase2_schema_failure_20260721.md`
- Step12c closeout receipt:
  `outputs/hu_joint_policy/m31_t3_step6d/step12c_pair_v1_actual/phase2_schema_failure_closeout_receipt.json`
- T4 completion audit:
  `docs/hu_joint_policy_m30_t4_completion_audit.md`
- T4 runtime contract:
  `configs/hu_joint_policy_m30_t4_runtime.json`
- 既存AI handoff:
  `docs/regular_ofc_ai_handoff_status.md`
- AI profile registry:
  `src/ofc_regular/ai_profiles.py`
- 引継ぎ検証スクリプト:
  `scripts/Test-HuJointPolicyHandoff20260721.ps1`

## 14. 見積もり

現在の計画値(認可ではない):

| Phase | 期間 | Cloud予算目安 |
|---|---:|---:|
| M3.1残り | 3~6日 | 最大$500 |
| RL-ready | 5~10日 | $20~100 |
| T2 | 4~8日 | $250~700 |
| T1 | 5~10日 | $600~1,500 |
| T0 | 7~14日 | $900~2,500 |
| FL | 2~5日 | $100~500 |
| joint league/ABR | 2~4週間 | $2,000~6,000 |

全gateが初回合格なら7~12週間。再設計を含む現実的な研究期間は
10~16週間。累計Cloud目安は約`$4,000~12,000`だが、これは認可ではない。

現時点の最も正確な一文:

> T4 exact componentとT3 search/Cloud lifecycle基盤はかなり完成して
> いるが、hidden-discard-safe T3実戦promotion、full-hand RL環境、
> T2/T1/T0両seat joint policy、FL統合、低exploitability評価はまだ
> 未完成である。
