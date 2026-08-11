# Regular OFC HU Joint Policy — 2026-07-22 進捗

## 結論

現在は、T0〜T4を両seatで対戦させる強化学習へ入る前の
`RL-ready correctness / performance foundation`を実装している。

直近の順序は次の通り。

1. RLB: Python/Rust scalar full-hand parity 10,000局
2. RLC: batch/PyO3 packed boundary、再開、性能
3. T3: candidate/reference tail pair、performance lock、quality
4. T3 label/model/promotion
5. 全局面RL environment、population self-play

数学的に証明された完全最適解を作ったという段階ではない。目標は、
強い自己対戦policyと、複数相手および近似best responseで測定した
低exploitability policyである。

## 保持しているbaseline

- T0: `stage19_p0`
- T1: `stage18_p1`
- T2: `stage9f_p2`
- T3: `stage7_m5_r10`
- T4: exact runtime
- `current`は変更しない
- 既存baselineを削除しない
- 十分なlocked評価前にfull replacementを有効化しない

2026-07-22確認時の`src/ofc_regular/ai_profiles.py` SHA-256:

`d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3`

## 完了したもの

### T4 / T3 teacher基盤

- T4 exact runtime
- T3 Step6d contract、runner、独立validator
- Candidate02 historical performance-development full100は
  100 paired / 200 rootsでportable parity 1.0
- historical candidate first-seat p95は約86.05秒、second-seat p95は約1.05秒
- historical candidate/reference geometric mean speedupは約2.70

historical full100はcandidate freezeの根拠であり、今後のfresh lockまたは
quality結果の代用にはしない。

### RL scalar/reference contract

- 情報集合からopponent private discardとrealized deck tailを除外
- canonical `ActionKey`、最大232 legal actions、illegal action拒否
- Python reference full handとRust scalar full handの比較器
- privileged correctness artifactをpolicy/replay/training不適格として分離

旧`rlb_scalar_parity_10000_v1`は、途中のbinary rebuildとruntime provenance不足の
ため探索用のまま残し、合格artifactとして使用しない。

### RLC batch/PyO3

- scalar/batchの同値なRust engine
- batch stepの全lane atomic commit
- deterministic lowest-lane error
- snapshot lineageとhidden-discard redaction
- packed observation / legal actions / selected actions / step outcome
- `actor_decision_batch_packed()`で1回のnative observationからobservationと
  legal mappingを同時生成
- fresh CPython 3.13 release wheelで関連RL tests 132件合格
- separate/combined packedは128/512 lanesの全10 decisionでbyte-exact
- Python allocation削減で512-lane warmed medianを約24.2%改善
- 100-lane fresh subprocessのseed＋ActionKey prefix replayが中断なし実行とbyte-identical

ローカルhash-bound benchmarkのcombined packedは、128 lanesで約5,366 decisions/s、
512 lanesで約8,149 decisions/sだった。20,000 actor decisions/s gateは未達なので
`below_rate_target_no_go`として保持し、C4上の同条件測定へ進む。

durable artifacts:

- `outputs/hu_joint_policy/rl_ready/rlc_native_batch_diagnostic_20260722_001/receipt.json`
- `outputs/hu_joint_policy/rl_ready/rlc_replay_resume_smoke_20260722_001/receipt.json`

後者はproduction checkpointではない。同一seedからfresh envを再生成し、prefixの
ActionKey履歴を再生する限定的な復旧smokeである。serializable/cross-process snapshotは
引き続き未完成。

### T3 performance-development v2 live preflight

GET専用collectorを実装し、実GCP環境で次を確認した。

- project: `ofc-solver-485418`
- zone: `asia-northeast1-b`
- machine: `c4-standard-16` = 16 vCPU / 61,440 MiB
- Debian 12 image: READY
- C4 quota: limit 128 / usage 0
- regional Spot CPU quota: limit 468 / usage 32
- current Tokyo C4 Spot price: `$0.53146 / VM-hour`
- planned VM/disk names: collision 0
- planned result/identity GCS prefix: collision 0
- cloud query: 21件、すべてGET
- VM作成、IAM変更、GCS書込み、package作成、launch: 0

local artifacts:

`outputs/hu_joint_policy/m31_t3_step6d/perfdev_v2_live_preflight_20260722_001/`

collection SHA-256:

`9bdfad952607c6358093afc49c7daf9014f59cbca259319c4fc23a3ce8c89d34`

このreceiptは`launch_authorized=false`であり、VM起動許可ではない。

## 2026-07-22追加完了

### RLB provenance v3 / 10,000局 parity

adversarial auditで実証した、別`source_root`のconfigをhashしながら元checkoutの
Python oracleを実行できる問題をv3で修正した。

- loaded module、producer CLI、merge CLIの実体とfrozen source rootを照合
- configのFL EVと実効`DEFAULT_FL_EV`を完全照合
- reference scoring contextを同じcanonical scoring identityへ固定
- Python executableとNumPy distribution/module identityを固定
- spawn workerごとにrun contractを再構築して照合
- v1/v2 shardを拒否

fresh 10,000 full-hand parityの結果:

- hands: 10,000
- decisions: 100,000
- shards: 20 × 500
- unique requests: 10,000
- mismatch / missing / duplicate: 0
- run contract: `ea463f3be7dbf321c0e2d54b250e31379812e9bdcd07b1f3886882ce170bbd33`
- merge receipt: `82f22589e206e0d781a8a5d1ae12dfe945223a46d15c41af320b48d168a7b7df`

artifact:

`outputs/hu_joint_policy/rl_ready/rlb_scalar_parity_10000_v3/merge_receipt.json`

artifactはhidden oracle stateをseedから再構築可能なprivileged correctness auditであり、
policy input、replay、trainingには使用しない。

## 2026-07-22追加進捗

### T3 local byte-locked package

live preflight合格後、cloud mutation前に次を固定するpackage builderを完成した。

- candidate/reference/feature libraryのsource・packaged bytes
- full100 plan
- 全100 root JSONとaggregate/topology digest
- preregistered 10 tail hands
- seed schedule
- rearm2とのroot/package/seed/namespace非重複
- startup、role manifests、tooling source

focused contract/preflight/package testsは53件合格。package合格時点でも
`cloud_executable=false`、`launch_authorized=false`を維持する。

現在は次のcloud execution層を実装中。

- candidate/reference専用の2 VMだけを扱うfresh startup/controller
- fixed 10 tail、`1 process × 16 Rayon`、TTL、heartbeat、resume、result manifest
- immutable sourceとLinux CPython 3.11 wheelhouseのcontent-addressed staging
- staging後のfresh namespace/IAM再確認とone-shot launch authorization
- result完全検証後だけのowned cleanup

Linux wheelhouseはD:へ23 locked wheels、約247.5 MiBで生成済み。実環境のdedicated
worker service accountは存在するが、現在のbucket/project IAM bindingは0のため、
run-prefix限定のobject get/create bindingを別の明示lifecycleとして実装中である。
このIAM gateが通るまでVM launchはfail closedとする。

## 次のGo/No-Go

### RLC

1. combined packed pathとseparate packed pathの全10 decision byte parity
2. scalar/batch/card permutation parity
3. invalid actionの全lane rollback
4. seed＋ActionKey replay後のpublic aggregate byte identity
5. hidden truth、opponent private discard、deck tail漏洩0
6. hash-bound benchmarkをローカルとC4で実行
7. production checkpoint / serializable snapshotは別gateで完成させる

### T3 tail pair

1. local byte-locked package合格
2. fresh read-only preflight再確認
3. candidate/reference別VM、各`1 process × 16 Rayon`
4. 同一10 tail roots、別process・別VM
5. portable parity 10/10
6. RSS、first/second latency、speedup gate
7. result artifact受領後にVMを終了

tail pair待ち時間には、RLC resume/replay/performance作業を進める。

## 禁止事項

- opponent private discardをpolicy入力に入れない
- teacher EVまたはLCBをruntime gateとして直接使わない
- model top1 accuracyだけでpromotionしない
- candidate選択MCと評価MCを共有しない
- nonfire時にbaseline Action object、RNG、trajectoryを変えない
- holdoutでthresholdを再探索しない
- fresh artifactが揃う前にVM statusだけで完了扱いしない
- `current`または既存named profileを暗黙変更しない
