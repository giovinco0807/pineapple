# HU Turn1 Stage2 With Stage9f P2

Status: T1 remains `No-Go` for runtime/production. Candidate-subset teacher
generation and TopK+confirm execution are valid, but the current MC32
candidate plus safe-selector runtime did not produce stable positive
seat-swap EV. T1/T0 should not be broadened until T2 is fixed and a new T1
teacher/gate path is validated.

## Purpose

Restart HU Turn1 work after accepting T2 `stage9f_p2` as the fixed P2
continuation for experiments.

This is not a continuation of the failed leaf31 TopK gate. The leaf31 fire-count
diagnostic reached `71` valid overrides but had EV/hand `-0.0101` and realized
per-fire delta `-0.1844`, so that exact runtime line is No-Go.

## Fixed Continuation

- T2 profile: `stage9f_p2`
- T2 runtime:
  `k3/mc16/d0/se0/confirm32/cse2/csemax2/fcd0/scd0/pd0/seat=first+second/bygate_delta`
- T2 mode: selective override only
- T3 profile: `stage7_m5_r10`
- T3 mode: selective override only
- FL EV source: `configs/fl_ev_regular_2k.json`

## Teacher Path

Use `ofc_regular.hu_turn1_teacher_pilot`.

Initial smoke:

```powershell
python -m ofc_regular.hu_turn1_teacher_pilot `
  --samples 2 `
  --future-samples 1 `
  --max-actions 2 `
  --seed 2026062511 `
  --profile stage9f_p2 `
  --opponent-profile stage9f_p2 `
  --opening-lookahead-samples 1 `
  --source-bucket stage2_stage9f_p2_smoke `
  --collect-topk-log `
  --output outputs/hu_turn1_stage2_stage9f_p2/smoke2_mc1_max2/hu_turn1_stage2_smoke.jsonl `
  --summary-output outputs/hu_turn1_stage2_stage9f_p2/smoke2_mc1_max2/summary.json
```

Analyze:

```powershell
python -m ofc_regular.analyze_hu_turn1_pilot `
  --input outputs/hu_turn1_stage2_stage9f_p2/smoke2_mc1_max2/hu_turn1_stage2_smoke.jsonl `
  --output-dir outputs/hu_turn1_stage2_stage9f_p2/smoke2_mc1_max2/analysis `
  --summary-output outputs/hu_turn1_stage2_stage9f_p2/smoke2_mc1_max2/analysis/summary.json
```

Smoke pass conditions:

- records equal requested samples,
- profile and opponent profile are `stage9f_p2`,
- T2 continuation is `stage9f_p2`,
- T3 continuation is `stage7_m5_r10`,
- no invalid action rows,
- action scores and SE are finite,
- `visible_dead_cards`, `hero_private_discards`, and
  `opponent_private_discards` remain present.

Observed smoke:

- output:
  `outputs/hu_turn1_stage2_stage9f_p2/smoke2_mc1_max2/hu_turn1_stage2_smoke.jsonl`
- summary:
  `outputs/hu_turn1_stage2_stage9f_p2/smoke2_mc1_max2/summary.json`
- analysis:
  `outputs/hu_turn1_stage2_stage9f_p2/smoke2_mc1_max2/analysis/summary.json`
- records:
  `2`
- profile / opponent:
  `stage9f_p2 / stage9f_p2`
- T2 continuation:
  `stage9f_p2`
- T3 continuation:
  `stage7_m5_r10`
- invalid action/state rows:
  `0 / 0`
- best action not legal:
  `0`
- seat split:
  `first=1`, `second=1`
- topk decisions / overrides during continuation:
  `8 / 0`
- seconds/sample:
  `2.35`

Decision: teacher smoke pass. This is execution-path evidence only, not model
quality evidence.

All-action speed smoke:

```powershell
python -m ofc_regular.hu_turn1_teacher_pilot `
  --samples 1 `
  --future-samples 1 `
  --max-actions 0 `
  --seed 2026062512 `
  --profile stage9f_p2 `
  --opponent-profile stage9f_p2 `
  --opening-lookahead-samples 1 `
  --source-bucket stage2_stage9f_p2_allactions_speed `
  --collect-topk-log `
  --output outputs/hu_turn1_stage2_stage9f_p2/speed1_mc1_allactions/hu_turn1_stage2_speed.jsonl `
  --summary-output outputs/hu_turn1_stage2_stage9f_p2/speed1_mc1_allactions/summary.json
```

Observed:

- records: `1`
- future samples: `1`
- max actions: `0` (all legal actions)
- legal actions: `27`
- seconds/sample: `13.71`
- T2 decisions inside rollout: `54`
- T2 choose_action time: `12.72s`
- TopK decisions / overrides: `54 / 2`
- analyzer: invalid rows `0`, action mismatches `0`, best action legal.

Decision: all-action path is valid, but broad local generation is too slow.
`1000 records x MC16` is roughly `61` local worker hours from this smoke, so
the next step should be a small Spot VM pilot before broad 1k.

## Broad Dataset

Do not start the broad set until the smaller GCP pilot passes.

Pilot100 GCP dry run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\Start-GcpHuTurn1PilotRun.ps1 `
  -RunName regular-hu-t1-stage2-stage9f-p2-pilot100-mc16-dryrun `
  -TotalSamples 100 `
  -ShardSamples 5 `
  -BaseSeed 2026062701 `
  -VmCount 10 `
  -FutureSamples 16 `
  -MaxActions 0 `
  -Profile stage9f_p2 `
  -OpponentProfile stage9f_p2 `
  -SourceBucket stage2_stage9f_p2_pilot100_mc16 `
  -CollectTopkLog `
  -DryRun
```

Dry-run result:

- total samples: `100`
- shard samples: `5`
- total shards: `20`
- VM count: `10`
- expected waves: `2`
- future samples: `16`
- max actions: `0`
- profile / opponent: `stage9f_p2 / stage9f_p2`
- source bucket: `stage2_stage9f_p2_pilot100_mc16`

Pilot100 start command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\Start-GcpHuTurn1PilotRun.ps1 `
  -RunName regular-hu-t1-stage2-stage9f-p2-pilot100-mc16-20260625 `
  -TotalSamples 100 `
  -ShardSamples 5 `
  -BaseSeed 2026062701 `
  -VmCount 10 `
  -FutureSamples 16 `
  -MaxActions 0 `
  -Profile stage9f_p2 `
  -OpponentProfile stage9f_p2 `
  -SourceBucket stage2_stage9f_p2_pilot100_mc16 `
  -CollectTopkLog `
  -CreateInstances
```

Receive and aggregate:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\Receive-GcpHuTurn1PilotRun.ps1 `
  -RunName regular-hu-t1-stage2-stage9f-p2-pilot100-mc16-20260625 `
  -OutputDir outputs/hu_turn1_stage2_stage9f_p2/gcp_pilot100_mc16
```

Pilot100 pass conditions:

- records `100/100`,
- missing shards `0`,
- profile/opponent `stage9f_p2`,
- T2 continuation `stage9f_p2`,
- T3 continuation `stage7_m5_r10`,
- invalid action/state rows `0`,
- best action legal,
- `visible_dead_cards`, `hero_private_discards`, and
  `opponent_private_discards` present,
- first/second split not badly skewed,
- runtime is feasible enough to justify broad 1k.

Observed all-action Pilot100 attempt:

- run:
  `regular-hu-t1-stage2-stage9f-p2-pilot100-mc16-20260625-001`
- output:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_pilot100_mc16_partial/`
- completed shards:
  `2 / 20`
- records:
  `10`
- missing shards:
  `18`
- residual running instances after abort:
  `0`
- mean seconds/sample:
  `117.22`
- max seconds/sample:
  `129.07`
- mean action count:
  `26.1`
- T2 choose_action seconds:
  `1032.37s`
- invalid action/state rows:
  `0 / 0`
- T2 continuation:
  `stage9f_p2`
- T3 continuation:
  `stage7_m5_r10`

Decision: all-action `stage9f_p2` MC16 teacher is too slow to broaden directly.
The run was intentionally stopped after partial evidence and all residual VMs
were deleted.

## Candidate-Subset Teacher Path

`ofc_regular.hu_turn1_teacher_pilot` now supports a candidate selector:

```powershell
python -m ofc_regular.hu_turn1_teacher_pilot `
  --samples 1 `
  --future-samples 1 `
  --max-actions 0 `
  --candidate-model models/hu_turn1_stage1_stage9f_p2_full2k_hgb_leaf3_lr01_l2_1.pkl `
  --candidate-topk 5 `
  --seed 2026062513 `
  --profile stage9f_p2 `
  --opponent-profile stage9f_p2 `
  --opening-lookahead-samples 1 `
  --source-bucket stage2_stage9f_p2_candidate_top5_smoke `
  --collect-topk-log `
  --output outputs/hu_turn1_stage2_stage9f_p2/candidate_top5_smoke_mc1/hu_turn1_stage2_candidate_top5.jsonl `
  --summary-output outputs/hu_turn1_stage2_stage9f_p2/candidate_top5_smoke_mc1/summary.json
```

Local Top5 smoke:

- records:
  `1`
- future samples:
  `1`
- candidate topK:
  `5`
- seconds/sample:
  `5.00`
- evaluated actions:
  `5`
- invalid rows:
  `0`

GCP Top5 candidate pilot:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\Start-GcpHuTurn1PilotRun.ps1 `
  -RunName regular-hu-t1-stage2-stage9f-p2-top5-candidate20-mc16-20260625-001 `
  -TotalSamples 20 `
  -ShardSamples 2 `
  -BaseSeed 2026062801 `
  -VmCount 5 `
  -FutureSamples 16 `
  -MaxActions 0 `
  -Profile stage9f_p2 `
  -OpponentProfile stage9f_p2 `
  -SourceBucket stage2_stage9f_p2_candidate_top5_mc16 `
  -CandidateModel models/hu_turn1_stage1_stage9f_p2_full2k_hgb_leaf3_lr01_l2_1.pkl `
  -CandidateTopk 5 `
  -CollectTopkLog `
  -CreateInstances
```

Observed Top5 candidate20:

- output:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_candidate20_top5_mc16/`
- records:
  `20 / 20`
- missing shards:
  `0`
- residual VMs:
  `0`
- future samples:
  `16`
- candidate topK:
  `5`
- mean seconds/sample:
  `28.16`
- max seconds/sample:
  `46.06`
- mean action count:
  `5.0`
- invalid action/state rows:
  `0 / 0`
- first/second:
  `10 / 10`
- T2 continuation:
  `stage9f_p2`
- T3 continuation:
  `stage7_m5_r10`

Decision: candidate-subset execution path is viable.

GCP Top10 candidate pilot:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\Start-GcpHuTurn1PilotRun.ps1 `
  -RunName regular-hu-t1-stage2-stage9f-p2-top10-candidate20-mc16-20260625-001 `
  -TotalSamples 20 `
  -ShardSamples 2 `
  -BaseSeed 2026062901 `
  -VmCount 5 `
  -FutureSamples 16 `
  -MaxActions 0 `
  -Profile stage9f_p2 `
  -OpponentProfile stage9f_p2 `
  -SourceBucket stage2_stage9f_p2_candidate_top10_mc16 `
  -CandidateModel models/hu_turn1_stage1_stage9f_p2_full2k_hgb_leaf3_lr01_l2_1.pkl `
  -CandidateTopk 10 `
  -CollectTopkLog `
  -CreateInstances
```

Observed Top10 candidate20:

- output:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_candidate20_top10_mc16/`
- records:
  `20 / 20`
- missing shards:
  `0`
- residual running instances:
  `0`
- future samples:
  `16`
- candidate TopK:
  `10`
- mean / max seconds per sample:
  `38.01 / 82.58`
- mean action count:
  `10.0`
- TopK decisions / overrides during continuation:
  `6400 / 55`
- T2 choose_action seconds:
  `672.80s`
- invalid action/state rows:
  `0 / 0`
- seat split:
  `first=10`, `second=10`
- T2 continuation:
  `stage9f_p2`
- T3 continuation:
  `stage7_m5_r10`

Decision: Top10 candidate-subset generation is execution-clean and about 3x
faster than the all-action partial run, but it is still not complete teacher
coverage evidence.

Coverage audit on the 10 completed all-action records:

| K | teacher-best recall | avg TopK regret | max TopK regret |
|---|---:|---:|---:|
| 1 | `3/10` | `2.3991` | `7.1959` |
| 3 | `4/10` | `2.0838` | `6.8068` |
| 5 | `6/10` | `1.2454` | `5.3209` |
| 8 | `8/10` | `0.7912` | `5.3209` |
| 10 | `9/10` | `0.5321` | `5.3209` |

Decision: Top5 is too narrow for broad training. Top10 is viable as an execution
path, but this audit still has one large miss at Top10/Top15/Top20. Do not treat
Top5 or Top10 candidate labels as complete all-action teacher labels unless the
training/evaluation explicitly treats them as incomplete candidate-subset labels.

All-action audit40 MC4:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\Start-GcpHuTurn1PilotRun.ps1 `
  -RunName regular-hu-t1-stage2-stage9f-p2-allaction-audit40-mc4-20260625-001 `
  -TotalSamples 40 `
  -ShardSamples 2 `
  -BaseSeed 2026063001 `
  -VmCount 10 `
  -FutureSamples 4 `
  -MaxActions 0 `
  -Profile stage9f_p2 `
  -OpponentProfile stage9f_p2 `
  -SourceBucket stage2_stage9f_p2_allaction_audit40_mc4 `
  -CollectTopkLog `
  -CreateInstances
```

Notes:

- The initial local launcher timed out after creating 6 VMs. The same run was
  resumed with `-SkipExistingInstances`.
- Two VMs stalled during startup package installation and were deleted/recreated.
- Final run completed cleanly with no residual RUNNING instances.

Observed all-action audit40:

- output:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_allaction_audit40_mc4/`
- records:
  `40 / 40`
- missing shards:
  `0`
- future samples:
  `4`
- mean / max seconds per sample:
  `35.59 / 64.06`
- mean action count:
  `26.1`
- invalid action/state rows:
  `0 / 0`
- seat split:
  `first=20`, `second=20`
- T2/T3 continuation:
  `stage9f_p2 / stage7_m5_r10`

Candidate coverage on audit40:

```powershell
python -m ofc_regular.analyze_hu_turn1_candidate_coverage `
  --input outputs\hu_turn1_stage2_stage9f_p2\gcp_allaction_audit40_mc4\hu_turn1_stage1_pilot.jsonl `
  --candidate-model models\hu_turn1_stage1_stage9f_p2_full2k_hgb_leaf3_lr01_l2_1.pkl `
  --ks 1,3,5,8,10,12,15,20,27 `
  --output-dir outputs\hu_turn1_stage2_stage9f_p2\gcp_allaction_audit40_mc4\coverage `
  --summary-output outputs\hu_turn1_stage2_stage9f_p2\gcp_allaction_audit40_mc4\coverage\coverage_summary.json
```

| K | teacher-best recall | avg TopK regret | max TopK regret |
|---|---:|---:|---:|
| 1 | `5/40` | `6.3539` | `15.3635` |
| 3 | `15/40` | `2.9784` | `13.3635` |
| 5 | `21/40` | `1.9199` | `10.3635` |
| 8 | `27/40` | `1.1114` | `10.3635` |
| 10 | `29/40` | `0.7585` | `7.0568` |
| 12 | `31/40` | `0.6383` | `7.0568` |
| 15 | `33/40` | `0.3605` | `7.0568` |
| 20 | `35/40` | `0.1403` | `3.0568` |
| 27 | `40/40` | `0.0000` | `0.0000` |

Decision: current T1 candidate model coverage is No-Go for broad
candidate-subset teacher generation. Even Top15 misses `7 / 40` teacher-best
actions and can lose `7.06` EV on this MC4 audit. The next step should improve
the candidate generator or change the teacher objective; scaling Top10/Top15
labels now would bake in avoidable misses.

Multi-model union candidate selector:

- Implemented in `src/ofc_regular/hu_turn1_teacher_pilot.py`:
  `--candidate-models ... --candidate-topk K --candidate-union-cap N`.
- Implemented in `src/ofc_regular/analyze_hu_turn1_candidate_coverage.py`:
  coverage can now evaluate multi-model TopK union candidate sets.
- GCP launcher support:
  `scripts/Start-GcpHuTurn1PilotRun.ps1` accepts `-CandidateModels` and
  `-CandidateUnionCap`.
- Test coverage:
  union selector preserves original legal action index mapping.

Existing model comparison on all-action audit40:

| candidate | K/cap | teacher-best recall | avg TopK regret | max TopK regret |
|---|---:|---:|---:|---:|
| current full2k | `15` | `33/40` | `0.3605` | `7.0568` |
| stage1b accept2 gray5 | `15` | `35/40` | `0.0938` | `1.4432` |
| stage1b accept2 gray5 | `18` | `36/40` | `0.0375` | `1.2500` |
| stage1b accept2 gray5 | `25` | `40/40` | `0.0000` | `0.0000` |
| stage1e accept2 runtime313 | `20` | `39/40` | `0.0000` | `0.0000` |
| current + stage1j union top10 | cap `20` | `36/40` | `0.1341` | `2.0000` |
| current + stage1j + stage1b union top15 | cap `20` | `38/40` | `0.0313` | `1.2500` |

Interpretation:

- The union path works, but simple union does not clearly beat the simpler
  `stage1b accept2 gray5` candidate model for the next teacher probe.
- `stage1b Top15` is the best speed/coverage tradeoff from this audit.
- Since it was selected after looking at audit40, do not broaden directly from
  this evidence. It needs a fresh non-overlapping all-action coverage audit.

Stage1b Top15 MC16 candidate smoke:

- unbalanced run:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_candidate20_stage1b_top15_mc16/`
- records:
  `20 / 20`
- mean / max seconds per sample:
  `81.17 / 159.49`
- mean action count:
  `15.0`
- invalid action/state rows:
  `0 / 0`
- seat split:
  `first=20`, `second=0`
- decision:
  execution-clean but not representative because `ShardSamples=1` captured only
  the first player's T1 spot in each shard.

Balanced Stage1b Top15 MC16 candidate smoke:

- run:
  `regular-hu-t1-stage2-stage9f-p2-stage1b-top15-mc16-balanced20-20260625-001`
- output:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_candidate20_stage1b_top15_mc16_balanced/`
- records:
  `20 / 20`
- missing shards:
  `0`
- future samples:
  `16`
- candidate model:
  `models/hu_turn1_stage1b_candidate_hgb_accept2_gray5_weighted.pkl`
- candidate TopK:
  `15`
- mean / max seconds per sample:
  `75.71 / 140.11`
- mean action count:
  `15.0`
- invalid action/state rows:
  `0 / 0`
- seat split:
  `first=10`, `second=10`
- T2/T3 continuation:
  `stage9f_p2 / stage7_m5_r10`

Decision: `stage1b Top15` is a viable candidate-subset teacher execution path,
but not yet a broad teacher source. Next gate is a fresh heldout all-action
audit, then compare `stage1b Top15/18/20` coverage on that unseen audit.

Fresh all-action audit40 MC4:

- run:
  `regular-hu-t1-stage2-stage9f-p2-allaction-fresh-audit40-mc4-20260625-001`
- output:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_allaction_fresh_audit40_mc4/`
- records:
  `40 / 40`
- missing shards:
  `0`
- mean / max seconds per sample:
  `30.62 / 55.67`
- mean legal actions:
  `26.63`
- seat split:
  `first=20`, `second=20`
- invalid action/state rows:
  `0 / 0`

Fresh audit candidate coverage:

| candidate | K/cap | teacher-best recall | avg TopK regret | max TopK regret |
|---|---:|---:|---:|---:|
| current full2k | `20` | `36/40` | `0.1577` | `4.3068` |
| stage1b accept2 gray5 | `15` | `36/40` | `0.2188` | `5.1932` |
| stage1b accept2 gray5 | `18` | `37/40` | `0.1423` | `5.1932` |
| stage1b accept2 gray5 | `25` | `40/40` | `0.0000` | `0.0000` |
| stage1e accept2 runtime313 | `20` | `39/40` | `0.0000` | `0.0000` |
| stage1b + stage1e union | top15 cap `18` | `40/40` | `0.0000` | `0.0000` |

Interpretation:

- Fresh audit does not validate `stage1b Top15/18` for broad teacher generation;
  it still has a large tail miss.
- `stage1e Top20` and `stage1b+stage1e union Top18` are the best observed
  candidates on the fresh audit.
- The union Top18 result is not yet robust enough by itself because earlier
  audit evidence was weaker. Do not launch broad 1k as complete teacher labels
  unless another heldout confirms it, or the training explicitly treats the
  candidate subset as incomplete labels.

Stage1b + Stage1e union Top18 MC16 candidate smoke:

- run:
  `regular-hu-t1-stage2-stage9f-p2-union-b-e-top18-mc16-20260625-001`
- output:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_candidate20_union_stage1b_stage1e_top18_mc16/`
- records:
  `20 / 20`
- missing shards:
  `0`
- future samples:
  `16`
- candidate models:
  `models/hu_turn1_stage1b_candidate_hgb_accept2_gray5_weighted.pkl`,
  `models/hu_turn1_stage1e_candidate_hgb_accept2_gray5_runtime313_weighted.pkl`
- candidate TopK / union cap:
  `15` per model, cap `18`
- mean / max seconds per sample:
  `99.98 / 168.51`
- mean action count:
  `17.2`
- seat split:
  `first=10`, `second=10`
- invalid action/state rows:
  `0 / 0`
- continuation:
  `stage9f_p2 / stage7_m5_r10`

Decision: union Top18 is execution-clean, but it is slower than `stage1b Top15`
and coverage is not robust across both all-action audits. Broad T1 teacher
generation remains `No-Go`.

Stage1e Top20 MC16 candidate smoke:

- run:
  `regular-hu-t1-stage2-stage9f-p2-stage1e-top20-mc16-20260625-001`
- output:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_candidate20_stage1e_top20_mc16/`
- records:
  `20 / 20`
- missing shards:
  `0`
- future samples:
  `16`
- candidate model:
  `models/hu_turn1_stage1e_candidate_hgb_accept2_gray5_runtime313_weighted.pkl`
- candidate TopK:
  `20`
- mean / max seconds per sample:
  `108.98 / 257.36`
- mean action count:
  `20.0`
- seat split:
  `first=10`, `second=10`
- invalid action/state rows:
  `0 / 0`
- continuation:
  `stage9f_p2 / stage7_m5_r10`

Decision: `stage1e Top20` is execution-clean and has the strongest fresh-audit
coverage among single models, but it is slower than union Top18 and far slower
than `stage1b Top15`. It should not be broadened as complete all-action teacher
labels without another heldout or an explicit incomplete-label training design.

Current T1 Stage2 candidate decision:

- `stage1b Top15`: fastest viable path, but undercovers fresh audit.
- `stage1e Top20`: best single-model coverage, but slow.
- `stage1b+stage1e union Top18`: best fresh-audit coverage at slightly fewer
  actions than stage1e Top20, but not robust across both audits and still slow.
- Broad T1 teacher: still `No-Go`.
- Next useful step: either run one more non-overlapping all-action audit for
  `stage1e Top20` and union Top18, or change training to treat candidate-subset
  rows as incomplete labels instead of complete all-action labels.

Broad1k target after Pilot100 pass:

- records: `1000`
- future samples: `16`
- max actions: `0` (all legal actions)
- source bucket: `stage2_stage9f_p2_natural_mc16`

This broad set is for candidate-generator training and refinement target mining,
not production adoption.

## Refinement

Mine selected targets from the broad set:

- high best-action SE,
- low margin,
- baseline miss,
- TopK candidate miss,
- first/second balance.

Relabel selected targets with MC32 before the next training pass.

## Model Direction

Train candidate generators first:

- HGB,
- ExtraTrees,
- pairwise logistic,
- listwise torch.

Do not adopt a direct T1 policy unless it beats the current baseline on holdout
top1 regret and passes runtime seat-swap. TopK candidate coverage alone is only
diagnostic.

## Runtime Evaluation

Any T1 runtime candidate must use independent TopK confirm and report performance
from realized fired whole-game paired deltas.

Confirm delta is gate diagnostic only. It must not be used as the performance
claim.

Initial acceptance gates:

- non-fired final mismatch count is `0`,
- at least `50` valid fired decisions in a diagnostic,
- realized per-fire CI lower bound is positive on non-overlapping seeds,
- aggregate EV/hand is non-negative,
- first/second split is not one-sided negative,
- p95 fired loss is acceptable.

## Current Decision

- T2 `stage9f_p2`: accepted for post-acceptance experiments.
- T1 leaf31 TopK: No-Go.
- T1 Stage2: teacher smoke and all-action speed smoke pass.
- All-action `stage9f_p2` MC16 teacher: No-Go on speed.
- Candidate-subset teacher: execution pass.
- Top5 candidate subset: undercovers teacher best; do not broaden as-is.
- Top10 candidate subset: GCP execution pass, clean schema, mean `38.01s/sample`.
- Candidate coverage: No-Go for current model after all-action audit40 MC4.
- Next gate: improve the T1 candidate generator, or design training around
  incomplete candidate-subset labels. Do not launch broad 1k from current TopK
  labels.
- Production/current replacement: No-Go.

## Union Top20 Cap22 Pilot100

Latest candidate-subset pilot:

- candidate models:
  `models/hu_turn1_stage1b_candidate_hgb_accept2_gray5_weighted.pkl`,
  `models/hu_turn1_stage1e_candidate_hgb_accept2_gray5_runtime313_weighted.pkl`
- selection:
  Top20 per model, union cap `22`
- run:
  `regular-hu-t1-stage2-stage9f-p2-union-b-e-top20-cap22-pilot100-mc16-20260625-001`
- merged output:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_candidate100_union_stage1b_stage1e_top20_cap22_mc16/hu_turn1_stage1_pilot.jsonl`
- merge summary:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_candidate100_union_stage1b_stage1e_top20_cap22_mc16/rescue_merge_summary.json`
- analysis:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_candidate100_union_stage1b_stage1e_top20_cap22_mc16/analysis_summary.json`
- records:
  `100`
- future samples:
  `16`
- action count:
  mean `21.63`, min/max `20 / 22`
- invalid action/state rows:
  `0 / 0`
- actions truncated:
  `100 / 100`
- seat split:
  `first=60`, `second=40`, expected from 20 shards of 5 records
- T2/T3 continuation:
  `stage9f_p2` / `stage7_m5_r10`

Rescue notes:

- Original Spot pilot completed `13 / 20` shards.
- `rescue2` supplied `27` exact-skip partial rows.
- `rescue4` supplied the final `8` missing rows using `--fast-skip-records`
  and on-demand VMs.
- The old `rescue_sXXX` outputs are excluded because they generated only the
  first T1 row for each missing seed and skewed the seat distribution.
- Fast skip preserves the target T1 state by replaying prior actual actions,
  but uses fresh MC futures for the evaluated row. Use it for rescue teacher
  rows, not for exact MC-future parity checks.

Updated decision:

- `union Top20 cap22` is execution-clean and is the current best
  candidate-subset teacher path.
- The pilot100 set is suitable for a training smoke that explicitly handles
  incomplete/candidate-set labels.
- It is not all-action teacher evidence and does not justify T1 runtime
  adoption or production.
- Next gate: train/evaluate a small candidate/listwise model on this pilot100
  set, then run heldout all-action coverage before broad scaling.

### T1 Stage2 candidate100 training smoke

Training smoke input:

- `outputs/hu_turn1_stage2_stage9f_p2/gcp_candidate100_union_stage1b_stage1e_top20_cap22_mc16/hu_turn1_stage1_pilot.jsonl`

Common split:

- seed: `2026062699`
- train / holdout: `80 / 20`
- objective: candidate-set only. These are truncated candidate labels, not
  complete all-action labels.

Artifacts:

- HGB score-regressor:
  `models/hu_turn1_stage2_candidate100_commonseed_hgb_regressor.pkl`
- pairwise logistic:
  `models/hu_turn1_stage2_candidate100_commonseed_pairwise_logistic.pkl`
- Torch listwise:
  `models/hu_turn1_stage2_candidate100_commonseed_listwise_torch.pt`
- comparison:
  `outputs/training/hu_turn1_stage2_candidate100_commonseed_model_comparison.json`

Common-holdout results:

| model | top1 avg regret | top3 best avg regret | top5 best avg regret | top10 best avg regret | top3 accepted recall | top5 accepted recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stage1b accept2 gray5 | `4.0964` | `1.9305` | `1.2714` | `0.4601` | `0.35` | `0.50` |
| stage1e accept2 runtime313 | `2.9335` | `1.3794` | `0.7023` | `0.0851` | `0.35` | `0.60` |
| stage2 HGB regressor | `1.7541` | `0.8175` | `0.6535` | `0.0716` | `0.50` | `0.55` |
| stage2 pairwise logistic | `3.7662` | `1.6138` | `0.5834` | `0.3115` | `0.25` | `0.65` |
| stage2 listwise torch | `2.5541` | `1.6388` | `1.0696` | `0.2656` | `0.35` | `0.50` |

Implementation fix:

- `DecisionFunctionAsScoreEstimator` was moved into the importable
  `hu_turn3_model` module so pairwise models trained through `python -m` are
  loadable in fresh Python processes.

Decision:

- `stage2 HGB regressor` is the only candidate from this smoke worth carrying
  forward.
- `pairwise_logistic` and `listwise_torch` should not be broadened from this
  pilot100 result. Listwise strongly overfit the 80-sample train split.
- This is still not runtime or production evidence.

Post-smoke all-action coverage refresh:

- The pilot100 HGB regressor should not be used directly as the broad candidate
  selector: as a single model it needs Top25 to become clean on the checked
  all-action audits.
- The original `stage1b + stage1e` union Top20 cap22 label source remains the
  correct broad-data generator:

| audit | cap22 recall | cap22 avg regret | cap22 max regret |
| --- | ---: | ---: | ---: |
| original audit40 | `0.95` | `0.0` | `0.0` |
| fresh audit40 | `1.00` | `0.0` | `0.0` |
| heldout2 audit40 | `0.975` | `0.0` | `0.0` |

Updated next gate:

- Broad `1k` MC16 candidate-set teacher generation is `Go`, using
  `stage1b + stage1e` Top20 per model with union cap `22`.
- The broad data remains incomplete-label training data only.  It is not
  runtime or production evidence.

### T1 Stage2 broad1k candidate-set teacher

Broad `1k` MC16 generation completed.

- run:
  `regular-hu-t1-stage2-stage9f-p2-union-b-e-top20-cap22-broad1000-mc16-20260625-001`
- merged output:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_candidate1000_union_stage1b_stage1e_top20_cap22_mc16/hu_turn1_stage1_pilot.jsonl`
- aggregate summary:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_candidate1000_union_stage1b_stage1e_top20_cap22_mc16/summary.json`
- analysis:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_candidate1000_union_stage1b_stage1e_top20_cap22_mc16/analysis_summary.json`
- records:
  `1000 / 1000`
- shards:
  `200 / 200`
- future samples:
  `16`
- candidate source:
  `stage1b + stage1e`, Top20 per model, union cap `22`
- mean action count:
  `21.559`
- invalid action/state rows:
  `0 / 0`
- seat split:
  `first=600`, `second=400`
- T2/T3 continuation:
  `stage9f_p2` / `stage7_m5_r10`

Generation notes:

- Initial `e2-highcpu-4` Spot generation stalled at `59 / 200` shards with
  memory-pressure symptoms.
- Completed GCS shard outputs were kept; stalled instances were deleted.
- Rescue used `e2-standard-4` Spot VMs, with a final single-shard rescue for
  shard `196`.

Decision:

- The broad1k data is clean candidate-set training data.
- It is still incomplete-label data, not all-action teacher evidence.
- T1 runtime, production, and P2-style fixing remain `No-Go`.

### T1 Stage2 broad1k HGB training

The broad1k HGB candidate-set regressor completed.

- model:
  `models/hu_turn1_stage2_candidate1000_union_top20_cap22_hgb_regressor.pkl`
- metrics:
  `outputs/training/hu_turn1_stage2_candidate1000_union_top20_cap22_hgb_regressor/metrics.json`
- split:
  seed `2026062701`, train/holdout `800 / 200`

Candidate-set holdout:

| model | top1 avg regret | top3 best avg regret | top5 best avg regret | top3 accepted recall | top5 accepted recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage1e baseline | `3.8236` | `2.1083` | `1.3641` | `0.415` | `0.545` |
| Stage2 broad1k HGB | `1.9634` | `0.7852` | `0.4409` | `0.635` | `0.775` |

All-action coverage audits:

| selector | original audit40 | fresh audit40 | heldout2 audit40 |
| --- | ---: | ---: | ---: |
| broad1k HGB single model | Top25 clean | Top22 clean | Top27 clean |
| stage1b + stage1e + broad1k HGB union, Top20/model | Top22 EV-regret clean (`recall=0.975`) | Top20 clean | Top25 clean |

Interpretation:

- The broad1k HGB model is a real candidate-set improvement over Stage1e on
  the same 200-row holdout.
- It should be carried forward as a candidate-set model, not as a direct T1
  runtime policy.
- A larger all-action audit is still needed before using it as the main T1
  candidate selector.

### T1 Stage2 audit100 and MC32 candidate pilot

The larger all-action audit and a higher-MC candidate-set pilot were completed
with `stage9f_p2` as the fixed T2 continuation and `stage7_m5_r10` as the fixed
T3 continuation.

All-action audit100:

- output:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_allaction_audit100_mc4/hu_turn1_stage1_pilot.jsonl`
- records:
  `100 / 100`
- future samples:
  `4`
- invalid action/state rows:
  `0 / 0`
- best action missing / not legal:
  `0 / 0`
- seat split:
  `first=50`, `second=50`
- mean legal actions:
  `26.46`

Coverage against audit100:

| selector | checked K/cap | recall | avg topK regret | max topK regret |
| --- | ---: | ---: | ---: | ---: |
| broad1k HGB single model | Top20 | `0.98` | `0.0306` | `2.5568` |
| broad1k HGB single model | Top25 | `0.99` | `0.0050` | `0.5000` |
| `stage1b + stage1e + broad1k HGB` union | Top20/model cap25 | `1.00` | `0.0000` | `0.0000` |

Candidate-set MC32 pilot:

- run:
  `regular-hu-t1-stage2-stage9f-p2-union-b-e-hgb-top20-cap25-pilot200-mc32-20260626-001`
- output:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_candidate200_union_stage1b_stage1e_hgb_top20_cap25_mc32/hu_turn1_stage1_pilot.jsonl`
- records / shards:
  `200 / 100`
- future samples:
  `32`
- invalid action/state rows:
  `0 / 0`
- best action missing / not legal:
  `0 / 0`
- mean action count:
  `23.195`
- actions truncated:
  `182 / 200`
- mean action SE:
  `2.408`
- score gap:
  mean `1.486`, median `0.882`
- average generation time:
  `236.17s/sample`

Training checks:

| model | train source | holdout top1 avg regret | holdout top3 best avg regret | holdout top5 best avg regret |
| --- | --- | ---: | ---: | ---: |
| broad1k HGB baseline | 1k MC16 | `1.6714` | `0.5349` | `0.2749` |
| candidate200 MC32 HGB | 200 MC32 | `1.4908` | `0.6483` | `0.2047` |
| candidate1200 HGB | 1k MC16 + 200 MC32 | `1.8042` | `0.5789` | `0.2533` |

Audit100 coverage for the newly trained models:

| selector | checked K/cap | recall | avg topK regret | max topK regret |
| --- | ---: | ---: | ---: | ---: |
| candidate200 MC32 HGB single model | Top20 | `0.93` | `0.0656` | `2.2500` |
| candidate1200 HGB single model | Top20 | `0.94` | `0.0231` | `1.1932` |
| `stage1b + stage1e + candidate200 MC32 HGB` union | Top20/model cap25 | `0.99` | `0.0119` | `1.1932` |
| `stage1b + stage1e + candidate1200 HGB` union | Top20/model cap25 | `0.99` | `0.0119` | `1.1932` |

Decision:

- Do not replace the broad1k HGB selector with candidate200 or candidate1200.
- The best checked candidate source remains
  `stage1b + stage1e + broad1k HGB` Top20/model with cap25.
- The next useful T1 step is not another small retrain; it is a larger or
  higher-MC teacher pass using the same clean union selector, followed by
  candidate-model training and a runtime TopK+confirm evaluation.
- T1 runtime, production, and T0 dependency work remain `No-Go` until T1
  candidate generation is stable under higher-MC labels.

### T1 Stage2 broad1000 MC32 union pass

The larger MC32 candidate-set teacher pass using the clean audit100 selector
completed on GCP.

- run:
  `regular-hu-t1-stage2-stage9f-p2-union-b-e-hgb-top20-cap25-broad1000-mc32-20260626-001`
- output:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_candidate1000_union_stage1b_stage1e_broad1k_hgb_top20_cap25_mc32/hu_turn1_stage1_pilot.jsonl`
- records / shards:
  `1000 / 500`
- future samples:
  `32`
- invalid action/state rows:
  `0 / 0`
- best action missing / not legal:
  `0 / 0`
- seat split:
  `first=500`, `second=500`
- mean action count:
  `23.266`
- actions truncated:
  `936 / 1000`
- mean action SE:
  `2.368`
- score gap:
  mean `1.265`, median `0.787`
- average generation time:
  `226.39s/sample`

The new MC32 HGB regressor was trained from these 1000 rows.

- model:
  `models/hu_turn1_stage2_candidate1000_union_b_e_broad1k_top20_cap25_mc32_hgb_regressor.pkl`
- metrics:
  `outputs/training/hu_turn1_stage2_candidate1000_union_b_e_broad1k_top20_cap25_mc32_hgb_regressor/metrics.json`
- split:
  seed `20260626`, train/holdout `800 / 200`

Same-split holdout versus the old broad1k HGB baseline:

| model | holdout top1 avg regret | holdout top3 best avg regret | holdout top5 best avg regret |
| --- | ---: | ---: | ---: |
| old broad1k HGB baseline | `1.4518` | `0.5627` | `0.2605` |
| broad1000 MC32 HGB | `1.3228` | `0.4153` | `0.1969` |

Audit100 all-action coverage:

| selector | checked K/cap | recall | avg topK regret | max topK regret |
| --- | ---: | ---: | ---: | ---: |
| broad1000 MC32 HGB single model | Top20 | `0.97` | `0.0381` | `1.7500` |
| broad1000 MC32 HGB single model | Top25 | `0.99` | `0.0000` | `0.0000` |
| `stage1b + stage1e + broad1000 MC32 HGB` union | Top20/model cap20 | `0.98` | `0.0119` | `1.1932` |
| `stage1b + stage1e + broad1000 MC32 HGB` union | Top20/model cap25 | `1.00` | `0.0000` | `0.0000` |
| `stage1b + stage1e + old broad1k HGB + broad1000 MC32 HGB` union | Top20/model cap25 | `1.00` | `0.0000` | `0.0000` |

Interpretation:

- The MC32-trained model improves same-split holdout regret over the old
  broad1k HGB baseline.
- As a single model, it does not dominate the old broad1k HGB on audit100.
- In the union selector it slightly improves Top20/Top22 coverage, but cap25 is
  already clean for the old checked selector.
- This is a useful candidate-generator update, not a production runtime proof.

Runtime smoke with the new MC32 model:

- profile:
  `stage9f_p2_hu_t1_topk_confirm` versus `stage9f_p2`
- candidate model override:
  `models/hu_turn1_stage2_candidate1000_union_b_e_broad1k_top20_cap25_mc32_hgb_regressor.pkl`
- config:
  `k3/mc8/d0/confirm16/cse1/pd0/seat=first+second`
- games:
  `20` paired seeds, `40` hands
- output:
  `outputs/evals/hu_turn1_stage2_stage9f_p2_topk_confirm_smoke_mc32_candidate/`

Smoke results:

| metric | value |
| --- | ---: |
| average score / hand | `+1.3114` |
| 95% CI | `[-0.4983, +3.1210]` |
| T1 decisions | `40` |
| valid T1 overrides | `4` |
| realized T1 per-fire delta mean | `+7.5568` |
| realized T1 per-fire CI95 | `[-7.5280, +22.6415]` |
| T1 non-fired counterfactual nonzero | `2` |

Runtime smoke interpretation:

- The runtime path loads and logs correctly with the MC32 model.
- The sample is far too small for a strength claim; only four valid T1 fires
  were observed.
- Confirm delta remains a gate diagnostic only; strength must be judged from
  realized seat-swap counterfactual deltas.
- The next validation should be sized by fired decisions, not by raw games.
  A practical next gate is at least `50` valid T1 fires for one or two configs,
  then compare realized per-fire delta and estimated EV/decision.

### T1 Stage2 MC32 topk-confirm 500 paired seed canary

A larger runtime canary was run on GCP with the MC32 candidate model.

- run:
  `regular-hu-t1-stage2-topk-confirm-mc32-smoke500-20260626-001`
- output:
  `outputs/evals/hu_turn1_stage2_topk_confirm_mc32_smoke500/`
- profile A/B:
  `stage9f_p2_hu_t1_topk_confirm` versus `stage9f_p2`
- candidate model:
  `models/hu_turn1_stage2_candidate1000_union_b_e_broad1k_top20_cap25_mc32_hgb_regressor.pkl`
- config:
  `k3/mc8/d0/confirm16/cse1/pd0/seat=first+second`
- games:
  `500` paired seeds, `1000` hands
- shards:
  `10 / 10`, missing `0`, failed `0`

Whole-match summary:

| metric | value |
| --- | ---: |
| average score / hand for T1 candidate | `-0.0650` |
| T1 decisions | `1000` |
| T1 realized decisions | `958` |
| T1 valid overrides | `42` |
| T1 valid override rate | `0.0438` |

T1 fired decision audit:

| metric | value |
| --- | ---: |
| realized per-fire delta mean | `-2.2543` |
| realized per-fire CI95 | `[-5.5765, +1.0679]` |
| estimated EV / decision | `-0.0988` |
| estimated EV / decision CI95 | `[-0.2445, +0.0468]` |
| losses / wins / zeros | `14 / 8 / 20` |
| p95 / p99 / max loss | `26.0770 / 27.4070 / 28.2270` |
| confirm delta mean on fired | `+3.3533` |
| confirm SE mean on fired | `2.2476` |
| non-fired counterfactual nonzero | `14` |
| non-fired final mismatch | `0` |
| runtime p95 | `28.37s` |

No-override reasons:

| reason | count |
| --- | ---: |
| `topk_empty` | `563` |
| `mc_best_is_baseline` | `197` |
| `below_confirm_delta` | `120` |
| `below_confirm_se` | `78` |
| `override_fired` | `42` |

Decision:

- `k3/mc8/d0/confirm16/cse1/pd0` with the MC32 candidate model is `No-Go`.
- The 500-seed run confirms that confirm delta is not an unbiased performance
  metric. Fired rows show positive confirm delta but negative realized
  per-fire delta.
- Tail risk is too large: p95 loss is about `26` points.
- Fire count is close to the intended minimum but still below `50`, and the
  observed direction is negative, so scaling this config is not justified.
- Next T1 work should add a safety gate before more large validation:
  `safe_selector_threshold`, stricter confirm such as `cse1.5` or `cse2`, or
  a dedicated T1 fire/veto head trained from realized fired losses.

### T1 Stage2 safe selector runtime attempts

A safety selector was added to veto risky TopK+confirm fires. This improved
some small samples but did not survive larger non-overlapping validation.

Runtime355 safe selector:

- training input:
  `outputs/training/hu_turn1_safe_selector_stage2_runtime355/teacher_regret313_plus_runtime42.jsonl`
- rows / positives / negatives:
  `355 / 110 / 245`
- feature mode:
  `candidate_delta_plus_meta`
- model:
  `models/hu_turn1_safe_selector_stage2_runtime355_accept0p25_gray1_candidate_delta_plus_meta.pkl`
- best model / validation AP:
  `hist_gradient / 0.4496`

Runtime355 validation sequence:

| run | config | valid overrides | realized per-fire mean | CI95 | EV/decision | p95/max loss | decision |
| --- | --- | ---: | ---: | --- | ---: | --- | --- |
| `outputs/evals/hu_turn1_stage2_topk_confirm_safe015_runtime355_smoke500/` | `safe0.15/seat=first+second` | `33` | `-0.5220` | `[-4.3800, +3.3360]` | `-0.0178` | `24.227 / 34.227` | `No-Go` |
| `outputs/evals/hu_turn1_stage2_topk_confirm_first_safe015_runtime355_2k/` | `safe0.15/seat=first` | `49` | `+2.7829` | `[+0.3359, +5.2299]` | `+0.0345` | `7.400 / 11.000` | promising, needed larger validation |
| `outputs/evals/hu_turn1_stage2_topk_confirm_first_safe015_runtime355_5k/` | `safe0.15/seat=first` | `144` | `-0.2617` | `[-2.1059, +1.5826]` | `-0.0038` | `24.227 / 51.454` | `No-Go` |

The 5k run invalidated the 2k positive read. It also showed that the selector
failed on high-tail first-seat losses despite `non_fired_final_mismatch_count =
0`.

Runtime499 safe selector:

- added the 5k fired rows as runtime proxy labels:
  `32` wins, `30` losses, `82` zero/gray rows
- dataset:
  `outputs/training/hu_turn1_safe_selector_stage2_runtime499/teacher_regret313_plus_runtime42_plus_first5k144.jsonl`
- rows / positives / negatives:
  `499 / 142 / 357`
- model:
  `models/hu_turn1_safe_selector_stage2_runtime499_accept0p25_gray1_candidate_delta_plus_meta.pkl`
- best model / validation AP:
  `extra_trees / 0.3024`
- posthoc on the old 5k fired set:
  `outputs/training/hu_turn1_safe_selector_stage2_runtime499/posthoc_first5k/posthoc_threshold_summary.csv`

The posthoc table looked better in-sample at `safe0.25+`, but a
non-overlapping 1k canary did not validate it:

- run:
  `regular-hu-t1-stage2-topk-confirm-first-safe025-runtime499-1k-20260626-001`
- output:
  `outputs/evals/hu_turn1_stage2_topk_confirm_first_safe025_runtime499_1k/`
- config:
  `k3/mc8/d0/confirm16/cse1/pd0/safe0.25/seat=first`
- valid overrides:
  `35`
- realized per-fire delta:
  `-0.4701`, CI95 `[-3.8624, +2.9222]`
- estimated EV / decision:
  `-0.0084`, CI95 `[-0.0688, +0.0520]`
- losses / wins / zeros:
  `7 / 7 / 21`
- p95 / p99 / max loss:
  `22.8270 / 24.8870 / 25.2270`
- whole-match average score / hand:
  `-0.0280`

Posthoc thresholding inside this same 1k run showed stricter thresholds do not
rescue the model:

| threshold | fires | mean delta | CI95 | losses / wins / zeros |
| ---: | ---: | ---: | --- | --- |
| `0.25` | `35` | `-0.4701` | `[-3.8624, +2.9222]` | `7 / 7 / 21` |
| `0.30` | `21` | `-2.7575` | `[-7.2212, +1.7061]` | `6 / 3 / 12` |
| `0.35` | `8` | `-9.4601` | `[-17.8146, -1.1057]` | `4 / 0 / 4` |
| `0.40` | `4` | `-13.3635` | `[-26.3279, -0.3991]` | `3 / 0 / 1` |

Decision:

- The current T1 TopK+confirm+safe-selector family is `No-Go`.
- Runtime selector confidence is not calibrated: stricter selector thresholds
  can keep worse losses.
- Do not run larger T1/T0 teacher generation from this runtime line.
- Next useful T1 work should happen after the T2 continuation is fixed, then
  regenerate T1 teacher/gate labels against that fixed continuation. A likely
  better direction is an all-action or near-all-action T1 teacher on selected
  buckets, not another threshold-only safe selector sweep.

### T1 all-action restart after runtime selector No-Go

After runtime355/runtime499 failed to validate, the T1 path was restarted from
all-action teacher generation with fixed downstream continuation:

- T2 continuation:
  `stage9f_p2`
- T3 continuation:
  `stage7_m5_r10`
- FL EV source:
  `configs/fl_ev_regular_2k.json`
- teacher mode:
  all legal T1 actions, `max_actions=0`

Local restart smoke:

```powershell
python -m ofc_regular.hu_turn1_teacher_pilot `
  --samples 2 `
  --future-samples 1 `
  --max-actions 0 `
  --seed 2026069001 `
  --profile stage9f_p2 `
  --opponent-profile stage9f_p2 `
  --opening-lookahead-samples 1 `
  --source-bucket stage2_stage9f_p2_allaction_restart_smoke_mc1 `
  --collect-topk-log `
  --output outputs/hu_turn1_stage2_stage9f_p2/restart_allaction_smoke2_mc1/hu_turn1_stage2_restart_smoke.jsonl `
  --summary-output outputs/hu_turn1_stage2_stage9f_p2/restart_allaction_smoke2_mc1/summary.json
```

Local smoke result:

- records:
  `2`
- action count:
  `27 / 27`
- invalid action/state rows:
  `0 / 0`
- best action not legal:
  `0`
- T2 continuation:
  `stage9f_p2`
- T3 continuation:
  `stage7_m5_r10`
- seconds/sample:
  `7.48`
- dominant cost:
  T2 continuation, `12.98s / 14.65s rollout`

Small GCP restart:

- run:
  `regular-hu-t1-stage2-restart-allaction40-mc4-20260626-001`
- output:
  `outputs/hu_turn1_stage2_stage9f_p2/restart_allaction40_mc4/gcp_full20/`
- samples:
  `40`
- shards:
  `20 / 20`
- future samples:
  `4`
- max actions:
  `0`
- profile / opponent:
  `stage9f_p2 / stage9f_p2`
- source bucket:
  `stage2_stage9f_p2_restart_allaction40_mc4`
- residual running instances:
  `0`

Aggregate summary:

| metric | value |
| --- | ---: |
| records | `40` |
| completed shards | `20 / 20` |
| mean seconds/sample | `30.77` |
| max seconds/sample | `50.87` |
| mean action count | `26.025` |
| topk continuation decisions | `8328` |
| topk continuation overrides | `76` |
| T2 choose_action seconds | `1118.67` |
| rollout seconds | `1228.21` |
| T2 duplicate raw/unique | `8328 / 8328` |

Analysis summary:

| metric | value |
| --- | ---: |
| total actions | `1041` |
| invalid state rows | `0` |
| invalid action rows | `0` |
| action count mismatches | `0` |
| actions truncated | `0` |
| best action missing | `0` |
| best action not legal | `0` |
| best score mismatches | `0` |
| score gap mismatches | `0` |
| duplicate sample ids | `0` |
| duplicate state keys | `0` |
| first / second | `20 / 20` |
| score gap mean / median / p95 | `1.9219 / 0.8750 / 8.0799` |
| best-action SE mean / p90 / max | `7.0793 / 9.8430 / 12.2159` |

Implementation note:

- `aggregate_hu_turn1_pilot` now globalizes `sample_id` as
  `shard * 1_000_000 + local_row_index`.
- The original shard-local value is preserved as `local_sample_id`.
- Aggregated records also include `aggregate_shard` and
  `aggregate_local_row_index`.
- This fixes duplicate sample ids across shards before any broad T1 cache,
  split, or replay audit.

Decision:

- All-action T1 teacher generation is schema-clean with current
  `stage9f_p2` / `stage7_m5_r10` continuation.
- MC4 is too noisy for final labels: best-action SE mean is about `7.08`.
- Runtime is feasible only on Spot VM. At MC4, this run cost about
  `30.77s/sample` on `e2-highcpu-4`; MC16 all-action remains expensive.
- Next T1 step should not be another TopK+confirm runtime sweep. The next
  useful step is to use this clean all-action path to build a small
  higher-quality teacher/candidate-generator dataset, then train a candidate
  generator or listwise scorer before any new runtime seat-swap.
- T0 remains blocked behind a validated T1/T2 continuation.

Candidate model coverage on this all-action40 set:

- input:
  `outputs/hu_turn1_stage2_stage9f_p2/restart_allaction40_mc4/gcp_full20/hu_turn1_stage1_pilot.jsonl`
- broad MC32 model:
  `models/hu_turn1_stage2_candidate1000_union_b_e_broad1k_top20_cap25_mc32_hgb_regressor.pkl`
- MC16/MC32 model:
  `models/hu_turn1_stage2_candidate1200_union_top20_mc16_mc32_hgb_regressor.pkl`

Single-model coverage:

| model | k | recall | avg topK regret | max topK regret |
| --- | ---: | ---: | ---: | ---: |
| broad MC32 | `1` | `0.225` | `4.5278` | `22.9203` |
| broad MC32 | `5` | `0.775` | `0.5077` | `6.0568` |
| broad MC32 | `10` | `0.825` | `0.2827` | `6.0568` |
| broad MC32 | `15` | `0.925` | `0.0000` | `0.0000` |
| MC16/MC32 | `1` | `0.325` | `2.8977` | `10.8635` |
| MC16/MC32 | `5` | `0.650` | `0.6091` | `6.0568` |
| MC16/MC32 | `10` | `0.850` | `0.2875` | `6.0568` |
| MC16/MC32 | `15` | `0.950` | `0.0000` | `0.0000` |

Union coverage:

| union config | k | recall | avg topK regret | max topK regret |
| --- | ---: | ---: | ---: | ---: |
| two models, each top5 | `5` | `0.725` | `0.6841` | `8.3068` |
| two models, each top5 | `8` | `0.775` | `0.4452` | `6.0568` |
| two models, each top10 | `8` | `0.825` | `0.3264` | `6.0568` |
| two models, each top10 | `10` | `0.875` | `0.1952` | `6.0568` |

Coverage interpretation:

- Current candidate models are usable only as broad candidate generators, not
  final T1 policies.
- TopK smaller than about `10-15` is too narrow for safe T1 relabeling.
- Top15 showing zero topK regret on only 40 MC4 rows is encouraging but not
  adoption evidence; MC4 labels have high SE.
- The next teacher generation should either keep all actions for selected
  high-value buckets or use a wide candidate union (`top15+`) before high-MC
  relabeling.

## Candidate-Union MC8 80-State Pilot

Purpose:

- Verify that a wider candidate-union T1 teacher path runs cleanly before
  spending on a larger high-MC T1 teacher.
- This is not production evidence and not a final T1 model selection run.

Run:

- GCP run:
  `regular-hu-t1-stage2-candunion15cap20-mc8-80-20260626-001`
- output:
  `outputs/hu_turn1_stage2_stage9f_p2/candidate_union15_cap20_mc8_80/gcp_full40/`
- samples:
  `80`
- shards:
  `40 / 40`
- future samples:
  `8`
- candidate models:
  - `models/hu_turn1_stage2_candidate1000_union_b_e_broad1k_top20_cap25_mc32_hgb_regressor.pkl`
  - `models/hu_turn1_stage2_candidate1200_union_top20_mc16_mc32_hgb_regressor.pkl`
- candidate topK / union cap:
  `15 / 20`
- profile / opponent:
  `stage9f_p2 / stage9f_p2`
- T3 continuation:
  `stage7_m5_r10`

Aggregate summary:

| metric | value |
| --- | ---: |
| records | `80` |
| completed shards | `40 / 40` |
| missing shards | `0` |
| mean seconds/sample | `38.50` |
| max seconds/sample | `99.66` |
| mean action count | `16.95` |
| topk continuation decisions | `21696` |
| topk continuation overrides | `156` |
| T2 choose_action seconds | `2762.22` |
| rollout seconds | `3073.67` |
| T2 duplicate raw/unique | `21696 / 21696` |
| sample id globalized | `true` |

Teacher analysis:

| metric | value |
| --- | ---: |
| total actions | `1356` |
| invalid state rows | `0` |
| invalid action rows | `0` |
| action count mismatches | `0` |
| actions truncated | `80` |
| best action missing | `0` |
| best action not legal | `0` |
| best score mismatches | `0` |
| score gap mismatches | `0` |
| duplicate sample ids | `0` |
| duplicate state keys | `0` |
| first / second | `40 / 40` |
| action count mean / p95 / max | `16.95 / 20 / 20` |
| score gap mean / median / p95 | `1.4743 / 0.9233 / 4.0568` |
| best-action SE mean / p90 / max | `5.4788 / 7.3944 / 9.1058` |

Training smoke:

| model | train top5 regret | holdout top5 regret | holdout top5 recall | decision |
| --- | ---: | ---: | ---: | --- |
| `hu_turn1_stage2_candidate80_union15_cap20_mc8_hgb_regressor.pkl` | `0.3026` | `0.9716` | `0.600` | candidate-generator smoke only |
| `hu_turn1_stage2_candidate80_union15_cap20_mc8_listwise_torch.pt` | `0.0000` | `1.0932` | `0.550` | candidate-generator smoke only |
| baseline broad MC32 | `0.7291` | `0.6571` | `0.750` | still stronger on this holdout |

Coverage on all-action40 reference:

| model / union | k | recall | avg topK regret | max topK regret |
| --- | ---: | ---: | ---: | ---: |
| existing 2-model union | `5` | `0.725` | `0.6841` | `8.3068` |
| existing 2-model union | `10` | `0.875` | `0.1952` | `6.0568` |
| existing 2-model union | `15` | `0.925` | `0.0000` | `0.0000` |
| existing 2-model union | `20` | `1.000` | `0.0000` | `0.0000` |
| new MC8 HGB80 | `5` | `0.550` | `1.3210` | `10.8635` |
| new MC8 HGB80 | `10` | `0.725` | `0.7571` | `10.3635` |
| new MC8 HGB80 | `20` | `0.950` | `0.0000` | `0.0000` |
| new MC8 listwise80 | `5` | `0.375` | `2.3878` | `16.1135` |
| new MC8 listwise80 | `10` | `0.600` | `0.9321` | `11.1135` |
| new MC8 listwise80 | `20` | `0.925` | `0.1952` | `6.5568` |
| existing 2 + new 2 union | `10` | `0.825` | `0.2688` | `3.5000` |
| existing 2 + new 2 union | `15` | `0.950` | `0.0000` | `0.0000` |
| existing 2 + new 2 union | `20` | `1.000` | `0.0000` | `0.0000` |

Decision:

- The candidate-union MC8 teacher path is schema-clean and operational.
- The new 80-row MC8-trained models are not stronger than the existing broad
  candidate generators. They are not adoption candidates.
- MC8 still has high SE and is too noisy for final T1 teacher labels.
- The existing 2-model union remains the better candidate generator for the
  next teacher run.
- Next step for T1 completion should be a larger, higher-MC
  candidate-union teacher, not another small model trained from 80 noisy rows.
  A practical next run is candidate union `top15 cap20`, `MC16` or a two-stage
  `MC8 -> selected MC32` relabel, with at least several hundred states.

## Candidate-Union MC16 400-State Pass

Purpose:

- Test whether a larger `top15 cap20` candidate-union teacher with MC16 can
  train a better T1 candidate generator.
- Keep T2 continuation fixed at `stage9f_p2` and T3 continuation fixed at
  `stage7_m5_r10`.
- Treat this as candidate-generator evidence only, not runtime or production
  evidence.

Run:

- GCP run:
  `regular-hu-t1-stage2-candunion15cap20-mc16-400-20260626-001`
- output:
  `outputs/hu_turn1_stage2_stage9f_p2/candidate_union15_cap20_mc16_400/gcp_full200/`
- samples:
  `400`
- shards:
  `200 / 200`
- future samples:
  `16`
- candidate models:
  - `models/hu_turn1_stage2_candidate1000_union_b_e_broad1k_top20_cap25_mc32_hgb_regressor.pkl`
  - `models/hu_turn1_stage2_candidate1200_union_top20_mc16_mc32_hgb_regressor.pkl`
- candidate topK / union cap:
  `15 / 20`
- profile / opponent:
  `stage9f_p2 / stage9f_p2`

Aggregate and analysis summary:

| metric | value |
| --- | ---: |
| records | `400` |
| completed shards | `200 / 200` |
| missing shards | `0` |
| mean seconds/sample | `81.86` |
| max seconds/sample | `179.42` |
| mean action count | `16.8375` |
| topk continuation decisions | `215520` |
| topk continuation overrides | `1764` |
| T2 choose_action seconds | `29494.45` |
| rollout seconds | `32708.53` |
| total actions | `6735` |
| invalid state/action rows | `0 / 0` |
| action count mismatches | `0` |
| actions truncated | `399` |
| best action missing / not legal | `0 / 0` |
| duplicate sample ids / state keys | `0 / 0` |
| first / second | `200 / 200` |
| score gap mean / median / p95 | `1.5526 / 0.9858 / 4.7307` |
| best-action SE mean / p90 / max | `3.8465 / 4.8133 / 5.9512` |

Training results:

| model | holdout top1 regret | holdout top3 regret | holdout top5 regret | holdout top5 recall | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `hu_turn1_stage2_candidate400_union15_cap20_mc16_hgb_regressor.pkl` | `2.1497` | `0.8538` | `0.4665` | `0.6625` | no adopt |
| `hu_turn1_stage2_candidate400_union15_cap20_mc16_listwise_torch.pt` | `2.6125` | `0.7961` | `0.4852` | `0.6875` | no adopt |
| baseline broad MC32 | - | - | `0.3295-0.3604` | `0.750-0.775` | still stronger |

Coverage on the all-action40 reference:

| model / union | k | recall | avg TopK regret | max TopK regret |
| --- | ---: | ---: | ---: | ---: |
| existing 2-model union | `5` | `0.725` | `0.6841` | `8.3068` |
| existing 2-model union | `10` | `0.875` | `0.1952` | `6.0568` |
| existing 2-model union | `15` | `0.925` | `0.0000` | `0.0000` |
| existing 2-model union | `20` | `1.000` | `0.0000` | `0.0000` |
| MC16/400 HGB | `5` | `0.600` | `0.7966` | `6.0568` |
| MC16/400 HGB | `10` | `0.800` | `0.2361` | `6.0568` |
| MC16/400 HGB | `20` | `0.975` | `0.0250` | `0.0250` |
| MC16/400 listwise | `5` | `0.575` | `1.2696` | `8.3068` |
| MC16/400 listwise | `10` | `0.800` | `0.4091` | `6.0568` |
| MC16/400 listwise | `20` | `0.875` | `0.1063` | `6.0568` |
| existing 2 + MC16/400 models | `5` | `0.675` | `0.7403-0.9293` | `8.3068` |
| existing 2 + MC16/400 models | `10` | `0.825-0.850` | `0.3250-0.3264` | `6.0568` |

Decision:

- The MC16/400 teacher generation is schema-clean and balanced.
- MC16 labels are still noisy enough that new models do not beat the existing
  2-model union on practical top5/top10 coverage.
- The existing 2-model union remains the best candidate generator.

## MC32 All-Action Refinement Relabel

Purpose:

- Select 120 high-value or uncertain states from the MC16/400 teacher and
  relabel them with stronger all-action MC32 labels.
- Use this as diagnostic/augmentation evidence, not broad relabel evidence.

Target extraction:

- target file:
  `outputs/hu_turn1_stage2_stage9f_p2/candidate_union15_cap20_mc16_400/refinement_targets/targets120.jsonl`
- targets:
  `120`
- seat split:
  `first=59`, `second=61`
- reason counts:
  `high_model_regret=60`, `high_action_se=20`, `high_score_gap=20`,
  `low_score_gap=20`, `random_cover=20`, `fill_cover=2`

Relabel run:

- GCP run:
  `regular-hu-t1-stage2-refine120-mc32-20260626-001`
- output:
  `outputs/hu_turn1_stage2_stage9f_p2/candidate_union15_cap20_mc16_400/refinement_mc32_partial_final/`
- expected shards:
  `60`
- completed shards:
  `41`
- usable records:
  `68`
- skipped / timed out targets:
  `14 / 14`
- failed targets:
  `0`
- mean relabel seconds/record:
  `220.55`
- T2 choose_action seconds:
  `13334.22`
- best action changed:
  `55 / 68` (`0.8088`)
- residual GCP instances:
  `0`

Merge:

- merged teacher:
  `outputs/hu_turn1_stage2_stage9f_p2/candidate_union15_cap20_mc16_400/refinement_mc32_partial_final/hu_turn1_stage1_pilot_mc32_refined.jsonl`
- base records:
  `400`
- refinement records:
  `68`
- replaced records:
  `68`
- missing refinement keys:
  `0`
- label source counts:
  `base_fast_t2=332`, `stage9f_p2_refinement=68`

Refined training:

| model | holdout top1 regret | holdout top3 regret | holdout top5 regret | holdout top5 recall | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `hu_turn1_stage2_candidate400_mc16_plus68_mc32_refined_hgb_regressor.pkl` | `2.2158` | `0.8249` | `0.2825` | `0.7750` | no adopt |
| `hu_turn1_stage2_candidate400_mc16_plus68_mc32_refined_listwise_torch.pt` | `2.1909` | `0.8087` | `0.4364` | `0.7250` | no adopt |
| baseline broad MC32 on same splits | `1.5469-1.5961` | `0.5001-0.5399` | `0.2774-0.3206` | `0.7875-0.8375` | still stronger |

Refined coverage on the all-action40 reference:

| model / union | k | recall | avg TopK regret | max TopK regret |
| --- | ---: | ---: | ---: | ---: |
| existing 2-model union | `5` | `0.725` | `0.6841` | `8.3068` |
| existing 2-model union | `10` | `0.875` | `0.1952` | `6.0568` |
| refined HGB single | `5` | `0.700` | `0.6903` | `6.0568` |
| refined HGB single | `10` | `0.850` | `0.2938` | `6.0568` |
| refined listwise single | `5` | `0.625` | `1.0230` | `8.3068` |
| refined listwise single | `10` | `0.800` | `0.5202` | `6.0568` |
| existing 2 + refined 2 union | `5` | `0.675` | `0.9293` | `8.3068` |
| existing 2 + refined 2 union | `10` | `0.825` | `0.3250` | `6.0568` |

Decision:

- The 68 MC32 all-action relabels are useful diagnostics: the MC16 best action
  changed on about `81%` of completed relabels.
- The completed relabel rows are not enough to train a better standalone T1
  candidate generator.
- Naive union with the refined models worsens top5/top10 coverage on the
  all-action40 reference.
- Broad all-action MC32 refinement is too expensive in the current continuation
  stack; the slowest/hardest targets timed out.
- Do not adopt the refined HGB/listwise models.
- Next T1 work should switch to candidate-subset high-MC relabeling: use the
  existing 2-model union to pick a wide candidate set, then apply higher MC only
  to those candidates and the baseline/all-action sentinel cases. This should
  preserve most of the relabel benefit without paying all-action MC32 cost.

## Candidate-Subset Refinement Relabel Support

Implementation:

- `ofc_regular.relabel_hu_turn1_refinement_targets` now accepts:
  - `--candidate-model`
  - `--candidate-models`
  - `--candidate-topk`
  - `--candidate-union-cap`
- `scripts/Start-GcpHuTurn1RefinementRelabelRun.ps1` passes the same selector
  options through dry-run, manifest, VM metadata, startup command, and package
  model inclusion.
- Candidate-subset relabel records retain:
  - `candidate_models`
  - `candidate_topk`
  - `candidate_union_cap`
  - `candidate_selector.selected_indices`
  - per-action `original_index`

Local smoke:

```powershell
python -m ofc_regular.relabel_hu_turn1_refinement_targets `
  --input outputs/hu_turn1_stage2_stage9f_p2/candidate_union15_cap20_mc16_400/refinement_targets/targets120.jsonl `
  --output outputs/hu_turn1_stage2_stage9f_p2/candidate_union15_cap20_mc16_400/refinement_candidate_subset_smoke/hu_turn1_refinement_relabel_smoke.jsonl `
  --summary-output outputs/hu_turn1_stage2_stage9f_p2/candidate_union15_cap20_mc16_400/refinement_candidate_subset_smoke/summary.json `
  --profile stage9f_p2 `
  --opponent-profile stage9f_p2 `
  --future-samples 1 `
  --max-targets 1 `
  --seed 2026070201 `
  --opening-lookahead-samples 1 `
  --candidate-models `
    models/hu_turn1_stage2_candidate1000_union_b_e_broad1k_top20_cap25_mc32_hgb_regressor.pkl `
    models/hu_turn1_stage2_candidate1200_union_top20_mc16_mc32_hgb_regressor.pkl `
  --candidate-topk 15 `
  --candidate-union-cap 20
```

Observed smoke:

| metric | value |
| --- | ---: |
| records | `1` |
| future samples | `1` |
| candidate model count | `2` |
| candidate topK / union cap | `15 / 20` |
| total legal actions | `24` |
| evaluated action count | `16` |
| actions truncated | `true` |
| relabel seconds | `10.50` |
| T2 choose_action seconds | `7.32` |
| best action changed | `1 / 1` |

GCP dry-run:

```powershell
.\scripts\Start-GcpHuTurn1RefinementRelabelRun.ps1 `
  -RunName regular-hu-t1-refine-candsubset-dryrun `
  -TargetInput outputs/hu_turn1_stage2_stage9f_p2/candidate_union15_cap20_mc16_400/refinement_targets/targets120.jsonl `
  -TotalTargets 2 `
  -ShardTargets 1 `
  -BaseSeed 2026070201 `
  -VmCount 1 `
  -FutureSamples 2 `
  -MaxActions 0 `
  -Profile stage9f_p2 `
  -OpponentProfile stage9f_p2 `
  -CandidateModels `
    models/hu_turn1_stage2_candidate1000_union_b_e_broad1k_top20_cap25_mc32_hgb_regressor.pkl,models/hu_turn1_stage2_candidate1200_union_top20_mc16_mc32_hgb_regressor.pkl `
  -CandidateTopk 15 `
  -CandidateUnionCap 20 `
  -DryRun
```

Dry-run result includes the candidate model paths, `candidate_topk=15`, and
`candidate_union_cap=20`.

Next gate:

- Run a small GCP candidate-subset relabel, for example `20-40` targets at
  MC32 or MC64 with `candidate top15 cap20`.
- Compare speed and label-change rate against the all-action MC32 relabel.
- If the candidate-subset relabel is stable and much faster, use it for a
  larger high-MC T1 refinement pass. Keep runtime/production T1 as `No-Go`
  until a new model improves all-action coverage and then validates in
  counterfactual seat-swap.

## Candidate-Subset MC32 20-Target GCP Probe

Purpose:

- Confirm the candidate-subset refinement relabel path on Spot VM.
- Compare speed against the all-action MC32 relabel.
- Check whether the existing 2-model candidate subset contains the all-action
  MC32 best action on overlapping completed targets.

Run:

- GCP run:
  `regular-hu-t1-refine-candsubset20-mc32-20260626-001`
- output:
  `outputs/hu_turn1_stage2_stage9f_p2/candidate_union15_cap20_mc16_400/refinement_candidate_subset20_mc32/`
- target input:
  `outputs/hu_turn1_stage2_stage9f_p2/candidate_union15_cap20_mc16_400/refinement_targets/targets120.jsonl`
- requested targets:
  `20`
- completed targets:
  `18`
- missing shards:
  `0`, `10`
- residual GCP instances:
  `0`
- future samples:
  `32`
- candidate models:
  - `models/hu_turn1_stage2_candidate1000_union_b_e_broad1k_top20_cap25_mc32_hgb_regressor.pkl`
  - `models/hu_turn1_stage2_candidate1200_union_top20_mc16_mc32_hgb_regressor.pkl`
- candidate topK / union cap:
  `15 / 20`

Aggregate result:

| metric | value |
| --- | ---: |
| records | `18` |
| skipped / timed out / failed | `0 / 0 / 0` |
| mean relabel seconds/record | `151.36` |
| median relabel seconds/record | `145.19` |
| max relabel seconds/record | `252.76` |
| T2 choose_action seconds | `2445.67` |
| mean evaluated actions | `16.5` |
| mean total legal actions | `26.5` |
| evaluated/legal ratio | `0.6245` |
| best action changed vs MC16 source | `11 / 18` (`0.6111`) |
| seat split | `first=9`, `second=9` |

Comparison to all-action MC32 partial:

| check | value |
| --- | ---: |
| all-action MC32 mean seconds/record | `220.55` |
| candidate-subset MC32 mean seconds/record | `151.36` |
| speedup | about `31%` lower wall time per record |
| common targets with all-action MC32 | `14` |
| same best action on common targets | `8 / 14` |
| all-action best contained in candidate subset | `14 / 14` |

Interpretation:

- Candidate-subset relabel is materially faster than all-action MC32, but not
  fast enough to treat high-MC broad relabel as cheap.
- The key coverage check passed on the overlapping completed targets: every
  all-action MC32 best action was present in the candidate subset. The best
  action mismatch is therefore independent-MC ordering noise, not candidate
  generator omission, for this small probe.
- Because 2 shards disappeared without DONE/status, the next larger run should
  continue to use small shard sizes and allow partial aggregation. Missing
  shards can be restarted, but this probe already answers the speed/coverage
  question.
- Next practical gate: run a larger candidate-subset high-MC relabel on `80-120`
  targets, preferably MC32 first, then train only if coverage against all-action
  sentinel targets remains clean.

## Candidate-Subset MC32 80-Target Relabel

Purpose:

- Test whether a larger candidate-subset MC32 relabel can replace the too-slow
  all-action MC32 refinement path.
- Train candidate generators from the relabeled rows only if the new labels
  improve holdout or all-action coverage.

Run:

- GCP run:
  `regular-hu-t1-refine-candsubset80-mc32-20260626-001`
- output:
  `outputs/hu_turn1_stage2_stage9f_p2/candidate_union15_cap20_mc16_400/refinement_candidate_subset80_mc32/`
- requested targets:
  `80`
- completed targets:
  `75`
- missing shards:
  `3`, `23`, `43`, `61`, `63`
- residual GCP instances:
  `0`
- future samples:
  `32`
- candidate models:
  - `models/hu_turn1_stage2_candidate1000_union_b_e_broad1k_top20_cap25_mc32_hgb_regressor.pkl`
  - `models/hu_turn1_stage2_candidate1200_union_top20_mc16_mc32_hgb_regressor.pkl`
- candidate topK / union cap:
  `15 / 20`

Aggregate relabel result:

| metric | value |
| --- | ---: |
| records | `75` |
| skipped / timed out / failed | `0 / 0 / 0` |
| mean relabel seconds/record | `159.45` |
| median relabel seconds/record | `147.63` |
| max relabel seconds/record | `455.05` |
| T2 choose_action seconds | `10765.65` |
| mean evaluated actions | `16.96` |
| mean total legal actions | `26.28` |
| evaluated/legal ratio | `0.6467` |
| best action changed vs MC16 source | `61 / 75` (`0.8133`) |
| seat split | `first=36`, `second=39` |

Overlap against the all-action MC32 partial relabel:

| check | value |
| --- | ---: |
| common targets with all-action MC32 | `52` |
| same best action on common targets | `15 / 52` (`0.2885`) |
| all-action best contained in candidate subset | `50 / 52` (`0.9615`) |
| all-action best subset rank mean | `4.18` |

Interpretation:

- Candidate-subset MC32 is faster than all-action MC32 (`159s` vs `221s`
  per completed record), but T2 continuation still dominates runtime.
- Top15/cap20 is not a perfect relabel set: `2 / 52` overlapping all-action
  best actions were outside the candidate subset.
- The high best-action-change rate confirms that MC16 labels are noisy for
  hard T1 targets, but the candidate subset must be wide enough before broad
  relabeling is trusted.

Merged teacher:

- base teacher:
  `outputs/hu_turn1_stage2_stage9f_p2/candidate_union15_cap20_mc16_400/gcp_full200/hu_turn1_stage1_pilot.jsonl`
- refinement:
  `outputs/hu_turn1_stage2_stage9f_p2/candidate_union15_cap20_mc16_400/refinement_candidate_subset80_mc32/hu_turn1_refinement_relabel.jsonl`
- merged output:
  `outputs/hu_turn1_stage2_stage9f_p2/candidate_union15_cap20_mc16_400/refinement_candidate_subset80_mc32/hu_turn1_stage1_pilot_candidate_subset_mc32_refined.jsonl`
- base records / refinement records / replaced records:
  `400 / 75 / 75`
- label source counts:
  `base_fast_t2=325`, `stage9f_p2_refinement=75`

Training from the 75-refined teacher:

| model | holdout top1 regret | holdout top3 regret | holdout top5 regret | holdout top5 recall | baseline holdout top5 regret | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `hu_turn1_stage2_candidate400_mc16_plus75_candsubset_mc32_refined_hgb_regressor.pkl` | `1.6734` | `0.6979` | `0.3315` | `0.7500` | `0.2440` | no adopt |
| `hu_turn1_stage2_candidate400_mc16_plus75_candsubset_mc32_refined_listwise_torch.pt` | `2.0623` | `0.5439` | `0.2744` | `0.6875` | `0.2213` | no adopt |

Coverage on the all-action40 reference:

| model / union | k | recall | avg TopK regret | max TopK regret |
| --- | ---: | ---: | ---: | ---: |
| existing 2-model union | `5` | `0.725` | `0.6841` | `8.3068` |
| existing 2-model union | `10` | `0.875` | `0.1952` | `6.0568` |
| existing 2-model union | `15` | `0.925` | `0.0000` | `0.0000` |
| refined HGB single | `5` | `0.700` | `0.7077` | `6.0568` |
| refined HGB single | `10` | `0.825` | `0.3813` | `6.0568` |
| refined HGB single | `15` | `0.925` | `0.0000` | `0.0000` |
| refined listwise single | `5` | `0.575` | `1.2508` | `11.1135` |
| refined listwise single | `10` | `0.700` | `0.4764` | `7.3635` |
| refined listwise single | `15` | `0.875` | `0.0938` | `1.5000` |
| existing 2 + refined 2 union | `5` | `0.675` | `0.7543` | `6.0568` |
| existing 2 + refined 2 union | `10` | `0.825` | `0.3875` | `6.0568` |
| existing 2 + refined 2 union | `15` | `0.925` | `0.0000` | `0.0000` |

Decision:

- Do not adopt the `plus75` HGB or listwise model.
- Do not add the `plus75` models to the current 2-model union. The union
  worsens top5/top10 coverage on the all-action40 sentinel.
- Keep the existing 2-model union as the current T1 candidate generator.
- Candidate-subset relabeling remains a useful execution path, but Top15/cap20
  is not safe enough for broad relabel. A next candidate-subset pass should use
  a wider set such as Top20/cap25 or add all-action sentinel checks before
  training.
- T1 runtime/production remains `No-Go`; T0 remains blocked behind a stronger
  validated T1 continuation.

## Candidate-Subset Top20/Cap25 Probe

Rationale:

- Top15/cap20 was faster but missed `2 / 52` all-action MC32 best actions on
  overlapping targets in the 80-target probe.
- The next candidate-subset relabel should test a wider set before any larger
  high-MC relabel or training run.

Implementation fix:

- `relabel_hu_turn1_refinement_targets` now calls
  `evaluate_turn1_action_subset` instead of the legacy wrapper so relabel
  outputs overwrite stale source metadata:
  - `candidate_selector`
  - `total_legal_actions`
  - `evaluated_action_count`
- Regression coverage:
  `test_relabel_record_overwrites_stale_candidate_selector_metadata`
- Fixed local smoke confirmed:
  `candidate_topk=20`, `candidate_union_cap=25`, selector `topk=20`,
  selector `union_cap=25`, `action_count=24`, `evaluated_action_count=24`,
  `total_legal_actions=24`.

GCP probe:

- GCP run:
  `regular-hu-t1-refine-candsubset20-top20cap25-mc32-20260626-001`
- output:
  `outputs/hu_turn1_stage2_stage9f_p2/candidate_union15_cap20_mc16_400/refinement_candidate_subset20_top20cap25_mc32/`
- requested targets:
  `20`
- completed targets:
  `17`
- missing shards:
  `0`, `3`, `12`
- residual GCP instances:
  `0`
- future samples:
  `32`
- candidate topK / union cap:
  `20 / 25`

Important caveat:

- This GCP probe was launched before the stale `candidate_selector` metadata
  fix. The action list and top-level `candidate_topk/candidate_union_cap` show
  the Top20/cap25 execution, but the nested `candidate_selector` field is stale
  and should not be used for analysis from this run.
- The labels are still useful for action-list coverage diagnostics. Any larger
  Top20/cap25 run should use the fixed code path.

Aggregate result:

| metric | value |
| --- | ---: |
| records | `17` |
| skipped / timed out / failed | `0 / 0 / 0` |
| mean relabel seconds/record | `197.56` |
| median relabel seconds/record | `202.50` |
| max relabel seconds/record | `311.27` |
| T2 choose_action seconds | `3015.05` |
| mean evaluated actions | `21.53` |
| mean total legal actions | `26.65` |
| evaluated/legal ratio | `0.8086` |
| best action changed vs MC16 source | `13 / 17` (`0.7647`) |
| seat split | `first=9`, `second=8` |

Overlap against the all-action MC32 partial relabel:

| check | value |
| --- | ---: |
| common targets with all-action MC32 | `12` |
| same best action on common targets | `6 / 12` (`0.5000`) |
| all-action best contained in candidate subset | `12 / 12` (`1.0000`) |
| all-action best subset rank mean | `3.25` |

Interpretation:

- Top20/cap25 has better coverage than Top15/cap20 on this small overlap:
  `12 / 12` all-action best actions were included.
- It is slower than Top15/cap20 (`198s` vs `151s` mean per record in the
  20-target probes), but still cheaper than all-action MC32 and much safer for
  candidate omission.
- Next T1 high-MC relabel should use fixed-code Top20/cap25, not Top15/cap20,
  unless a larger sentinel audit shows the wider set is unnecessary.
- Do not train from the stale-selector GCP run as a canonical artifact. Use it
  as coverage evidence only, or regenerate after the metadata fix.

## Fixed Top20/Cap25 MC32 80-Target Relabel

Purpose:

- Re-run the Top20/cap25 candidate-subset relabel after fixing stale
  `candidate_selector` metadata.
- Verify whether the wider candidate subset preserves all-action MC32 best
  actions on overlapping sentinel targets.
- Check whether mixing the fixed relabel rows into the 400-row MC16 base teacher
  is enough to train a stronger T1 candidate generator.

Run:

- GCP run:
  `regular-hu-t1-refine-candsubset80-top20cap25-fixed-mc32-20260626-001`
- output:
  `outputs/hu_turn1_stage2_stage9f_p2/candidate_union15_cap20_mc16_400/refinement_candidate_subset80_top20cap25_fixed_mc32/`
- requested targets:
  `80`
- completed targets:
  `65`
- missing shards:
  `1`, `15`, `19`, `21`, `31`, `35`, `39`, `41`, `51`, `55`, `59`,
  `61`, `71`, `75`, `79`
- residual GCP instances:
  `0`
- future samples:
  `32`
- candidate topK / union cap:
  `20 / 25`

Aggregate result:

| metric | value |
| --- | ---: |
| records | `65` |
| skipped / timed out / failed | `0 / 0 / 0` |
| mean relabel seconds/record | `207.75` |
| median relabel seconds/record | `205.89` |
| p90 relabel seconds/record | `324.33` |
| max relabel seconds/record | `446.08` |
| T2 choose_action seconds | `12190.87` |
| mean evaluated actions | `21.69` |
| mean total legal actions | `26.22` |
| evaluated/legal ratio | `0.8299` |
| best action changed vs MC16 source | `51 / 65` (`0.7846`) |
| selector metadata consistent | `true` |
| seat split | `first=33`, `second=32` |

Overlap against the all-action MC32 partial relabel:

| check | value |
| --- | ---: |
| common targets with all-action MC32 | `46` |
| same best action on common targets | `14 / 46` (`0.3043`) |
| all-action best contained in candidate subset | `46 / 46` (`1.0000`) |
| all-action best subset rank mean | `5.22` |
| all-action best subset rank max | `17` |

Interpretation:

- Top20/cap25 fixed-code relabel solved the observed candidate omission on this
  sentinel overlap: every all-action MC32 best action was present.
- The larger set costs more than Top15/cap20 (`208s` vs `159s` per completed
  record in the 80-target probes), but the coverage gain is worth it for
  canonical relabel generation.
- Best action changes on about `78%` of hard targets, confirming MC16 labels are
  too noisy in the current T1 hard-target pool.

Merged teacher:

- merged output:
  `outputs/hu_turn1_stage2_stage9f_p2/candidate_union15_cap20_mc16_400/refinement_candidate_subset80_top20cap25_fixed_mc32/hu_turn1_stage1_pilot_top20cap25_mc32_refined.jsonl`
- base records / refinement records / replaced records:
  `400 / 65 / 65`
- label source counts:
  `base_fast_t2=335`, `stage9f_p2_refinement=65`

Training from the 65-refined teacher:

| model | holdout top1 regret | holdout top3 regret | holdout top5 regret | holdout top5 recall | baseline holdout top5 regret | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `hu_turn1_stage2_candidate400_mc16_plus65_top20cap25_mc32_refined_hgb_regressor.pkl` | `2.2138` | `0.8637` | `0.4029` | `0.7375` | `0.3544` | no adopt |
| `hu_turn1_stage2_candidate400_mc16_plus65_top20cap25_mc32_refined_listwise_torch.pt` | `1.8934` | `0.7003` | `0.4437` | `0.7250` | `0.2126` | no adopt |

Coverage on the all-action40 reference:

| model / union | k | recall | avg TopK regret | max TopK regret |
| --- | ---: | ---: | ---: | ---: |
| existing 2-model union | `5` | `0.725` | `0.6841` | `8.3068` |
| existing 2-model union | `10` | `0.875` | `0.1952` | `6.0568` |
| existing 2-model union | `15` | `0.925` | `0.0000` | `0.0000` |
| fixed Top20/cap25 refined HGB single | `5` | `0.625` | `0.6591` | `6.0568` |
| fixed Top20/cap25 refined HGB single | `10` | `0.775` | `0.5139` | `6.0568` |
| fixed Top20/cap25 refined HGB single | `15` | `0.900` | `0.2139` | `5.0568` |
| fixed Top20/cap25 refined listwise single | `5` | `0.575` | `0.8091` | `6.0568` |
| fixed Top20/cap25 refined listwise single | `10` | `0.800` | `0.1625` | `2.0000` |
| fixed Top20/cap25 refined listwise single | `15` | `0.900` | `0.1125` | `2.0000` |
| existing 2 + fixed refined 2 union | `5` | `0.650` | `0.9682` | `6.0568` |
| existing 2 + fixed refined 2 union | `10` | `0.825` | `0.3688` | `6.0568` |
| existing 2 + fixed refined 2 union | `15` | `0.925` | `0.0000` | `0.0000` |

Decision:

- Do not adopt the fixed Top20/cap25 `plus65` HGB or listwise model.
- Do not add these refined models to the current 2-model union; the union still
  worsens top5/top10 all-action40 coverage.
- Keep the existing 2-model union as the active T1 candidate generator.
- Top20/cap25 is now the preferred relabel candidate subset for future
  high-MC labels because the all-action best inclusion was `46 / 46`.
- The failure is not candidate-subset coverage; it is data volume/generalization.
  Training from a 400-row base with only 65 high-MC replacement labels is too
  small and overfits/noises out.
- Next T1 step: build a larger canonical Top20/cap25 MC32 dataset rather than
  continuing to patch the 400-row MC16 base. A practical next gate is
  `1k-2k` candidate-subset MC32 teacher rows with all-action sentinel audits,
  then train a fresh candidate generator and compare against the current
  2-model union.

## Canonical Top20/Cap25 MC32 1k Teacher Run

Purpose:

- Build a fresh T1 candidate-subset MC32 teacher instead of patching the old
  400-row MC16 base.
- Use the current fixed continuation:
  - T2 profile: `stage9f_p2`
  - T3 continuation: Stage7 `m5_r10`
  - candidate models:
    - `models/hu_turn1_stage2_candidate1000_union_b_e_broad1k_top20_cap25_mc32_hgb_regressor.pkl`
    - `models/hu_turn1_stage2_candidate1200_union_top20_mc16_mc32_hgb_regressor.pkl`
  - candidate topK / union cap: `20 / 25`
- Check whether a larger clean Top20/cap25 MC32 teacher can train a better T1
  candidate generator than the existing 2-model union.

Run:

- GCP run:
  `regular-hu-t1-stage2-top20cap25-mc32-1k-20260626-001`
- final output:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_top20cap25_mc32_1k_final/`
- requested / completed / missing:
  `1000 / 1000 / 0`
- future samples:
  `32`
- candidate topK / union cap:
  `20 / 25`
- Spot recovery:
  - restarted missing worker columns during the main run
  - final 23 missing shards were relaunched as one-shard workers with
    `VmCount=1000`
- residual GCP instances after final receive:
  `0`

Aggregate result:

| metric | value |
| --- | ---: |
| records | `1000` |
| completed shards | `1000 / 1000` |
| missing shards | `0` |
| mean seconds/sample | `220.33` |
| max seconds/sample | `563.56` |
| mean evaluated actions | `21.907` |
| T2 choose_action seconds | `196521.32` |
| rollout seconds | `220171.68` |
| sample_id globalized | `true` |
| T2 state duplicate rate | `0.0` |

Training from the 1k teacher:

| model | holdout top1 regret | holdout top3 regret | holdout top5 regret | holdout top5 recall | baseline holdout top1/top3/top5 regret | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `hu_turn1_stage2_top20cap25_mc32_1k_hgb_regressor.pkl` | `1.3290` | `0.3704` | `0.1506` | `0.8250` | `1.4990 / 0.4469 / 0.1482` | diagnostic only |
| `hu_turn1_stage2_top20cap25_mc32_1k_listwise_torch.pt` | `1.6517` | `0.5489` | `0.2997` | `0.7550` | `1.3309 / 0.4709 / 0.1339` | no adopt |

Interpretation:

- The 1k HGB model improves holdout top1/top3 regret versus the older single
  baseline model, while top5 is roughly tied/slightly worse.
- The 1k listwise model still overfits: train regret is excellent, but holdout
  top1/top3/top5 are worse than the baseline model.

Coverage on the all-action40 reference:

| model / union | k | recall | avg TopK regret | max TopK regret |
| --- | ---: | ---: | ---: | ---: |
| existing 2-model union | `5` | `0.725` | `0.6841` | `8.3068` |
| existing 2-model union | `10` | `0.875` | `0.1952` | `6.0568` |
| existing 2-model union | `15` | `0.925` | `0.0000` | `0.0000` |
| existing 2-model union | `20` | `1.000` | `0.0000` | `0.0000` |
| 1k HGB single | `5` | `0.725` | `0.6313` | `6.0568` |
| 1k HGB single | `10` | `0.850` | `0.1611` | `4.4432` |
| 1k HGB single | `15` | `0.950` | `0.0000` | `0.0000` |
| 1k HGB single | `20` | `1.000` | `0.0000` | `0.0000` |
| 1k listwise single | `5` | `0.675` | `0.8932` | `6.5000` |
| 1k listwise single | `10` | `0.875` | `0.0813` | `1.7500` |
| 1k listwise single | `15` | `0.975` | `0.0000` | `0.0000` |
| 1k listwise single | `20` | `1.000` | `0.0000` | `0.0000` |
| existing 2 + 1k HGB | `10` | `0.875` | `0.2625` | `6.0568` |
| existing 2 + 1k HGB | `20` | `0.975` | `0.0000` | `0.0000` |
| existing 2 + 1k listwise | `10` | `0.875` | `0.2202` | `6.0568` |
| existing 2 + 1k listwise | `20` | `1.000` | `0.0000` | `0.0000` |
| existing 2 + 1k HGB + 1k listwise | `10` | `0.900` | `0.2625` | `6.0568` |
| existing 2 + 1k HGB + 1k listwise | `20` | `1.000` | `0.0000` | `0.0000` |

Decision:

- Do not replace the active T1 candidate generator yet.
- Keep the existing 2-model union as the active runtime/teacher candidate
  generator for now.
- The 1k HGB model is useful evidence that clean Top20/cap25 MC32 data improves
  holdout top1/top3, but adding it to the current union worsens all-action40
  ranking at k10/k20.
- The 1k listwise model is not reliable despite a good all-action40 k10 result;
  its holdout metrics show overfit.
- T1 is not complete. The current bottleneck is no longer simple candidate
  inclusion at `k=20`; the existing union and new single models already cover
  all-action40 at k20. The remaining issue is ranking/order and generalization
  at smaller k, plus validation on a larger independent all-action sentinel.
- Next recommended gate:
  1. Build a larger independent all-action sentinel set (`100-200` states) so
     coverage differences are not decided by only 40 records.
  2. Re-score the candidate-union selection rule: adding models can push out
     good actions, so union ordering needs calibration, not just more models.
  3. If the larger sentinel confirms the 1k HGB/listwise models improve k10
     regret, train a calibrated candidate-ranker/merger instead of blindly
     appending models to the union.

## Independent All-Action200 MC4 Sentinel

Purpose:

- Validate T1 candidate-generator coverage on a larger non-overlapping
  all-action sentinel before spending on broader T1 relabel/training.
- Keep continuation fixed:
  - T2 profile: `stage9f_p2`
  - T3 continuation: Stage7 `m5_r10`
  - FL EV source: `configs/fl_ev_regular_2k.json`
- Test whether adding the 1k HGB/listwise models and calibrated union ordering
  improves practical TopK coverage.

Run:

- GCP run:
  `regular-hu-t1-stage2-allaction200-mc4-20260626-001`
- output:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_allaction200_mc4/`
- requested / completed / missing:
  `200 / 200 / 0`
- shards:
  `100 / 100`
- future samples:
  `4`
- max actions:
  `0` (all legal actions)
- profile / opponent:
  `stage9f_p2 / stage9f_p2`
- T3 continuation:
  `stage7_m5_r10`
- residual GCP instances after completion:
  `0`

Aggregate result:

| metric | value |
| --- | ---: |
| records | `200` |
| mean seconds/sample | `35.28` |
| max seconds/sample | `67.73` |
| mean legal actions | `26.385` |
| topk decisions | `42216` |
| topk overrides | `452` |
| T2 choose_action seconds | `6445.79` |
| T2 state duplicate rate | `0.0` |

Teacher analysis:

| metric | value |
| --- | ---: |
| invalid action/state rows | `0 / 0` |
| best action not legal | `0` |
| seat split | `first=100`, `second=100` |
| action count mean / max | `26.385 / 27` |
| score gap mean / median | `2.3019 / 1.2784` |
| best action SE mean | `7.2722` |

The sentinel is schema-clean and balanced. MC4 is still noisy, so it is used as
candidate-coverage evidence, not final high-confidence EV labeling.

Coverage summary on all-action200:

| selector | union mode | k10 recall / avg regret | k15 recall / avg regret | k20 recall / avg regret / max | k25 |
| --- | --- | ---: | ---: | ---: | ---: |
| existing 2-model union | `min_rank` | `0.770 / 0.7389` | `0.885 / 0.2166` | `0.970 / 0.0793 / 11.8635` | `1.000 / 0.0000` |
| 1k HGB single | `min_rank` | `0.800 / 0.5470` | `0.900 / 0.2834` | `0.955 / 0.1470 / 11.8635` | `1.000 / 0.0000` |
| 1k listwise single | `min_rank` | `0.785 / 0.5661` | `0.875 / 0.2231` | `0.950 / 0.0449 / 3.3635` | `1.000 / 0.0000` |
| existing 2 + 1k HGB | `min_rank` | `0.790 / 0.7036` | `0.890 / 0.2269` | `0.960 / 0.1064 / 11.8635` | `1.000 / 0.0000` |
| existing 2 + 1k listwise | `min_rank` | `0.790 / 0.5076` | `0.880 / 0.1598` | `0.960 / 0.0534 / 3.8068` | `0.995 / 0.0050` |
| existing 2 + HGB + listwise | `min_rank` | `0.800 / 0.5151` | `0.885 / 0.1610` | `0.965 / 0.0599 / 3.8068` | `1.000 / 0.0000` |
| existing 2 + HGB + listwise | `max_z_score` | `0.810 / 0.4673` | `0.890 / 0.1913` | `0.970 / 0.0396 / 3.5568` | `1.000 / 0.0000` |

Implementation note:

- `ofc_regular.analyze_hu_turn1_candidate_coverage` now supports union ordering
  modes: `min_rank`, `rank_sum`, `reciprocal_rank_sum`, `mean_score`,
  `max_score`, `mean_z_score`, and `max_z_score`.
- Default remains `min_rank` for backward compatibility.
- The teacher generation path and GCP launcher now accept
  `--candidate-union-mode` / `-CandidateUnionMode`, so `max_z_score` can be used
  in the next candidate-subset relabel run.

Interpretation:

- The existing 2-model union remains safe at k25, but it has a severe k20 tail
  miss on this larger sentinel: max regret `11.8635`.
- The 1k listwise model is still not reliable as a standalone model, but adding
  it as a diversity source reduces k10/k15/k20 average regret and cuts the k20
  tail materially.
- The best observed diagnostic selector is `existing 2 + HGB + listwise` with
  `max_z_score`: k10 avg regret `0.4673`, k20 avg regret `0.0396`, k20 max
  regret `3.5568`, and clean k25.
- This is not enough to call T1 complete. It is enough to move from "candidate
  inclusion is unresolved" to "candidate inclusion at k25 is clean; next problem
  is candidate ordering/ranking and high-MC label quality."

Decision:

- T1 remains `No-Go` for runtime/production.
- Keep T2 `stage9f_p2` and T3 `stage7_m5_r10` fixed for T1 continuation.
- Do not replace T1 policy from this evidence.
- Next T1 work should implement/use the calibrated union ordering in candidate
  generation, then run a larger candidate-subset MC32/MC64 relabel using the
  4-model candidate pool. The all-action200 sentinel should remain the coverage
  guardrail for that new generator.

## T1 GCP Seat-Split Guardrail

`build_turn1_pilot_samples` emits one T1 row per player. A GCP shard launched
with `-ShardSamples 1` and no record skip stops after the first row of each
generated hand, which is normally `player=0` / `seat=first`. That run can be
schema-clean but still unusable for balanced T1 training.

Operational rule:

- use `-ShardSamples 2` for paired first+second rows from the same generated
  hand; or
- if a first-only run already exists, run the complement with
  `-RecordSkipBase 1 -FastSkipRecords`, then merge first+second before training.

The T1 pilot analyzer now reports `seat_distribution` and writes a
`single_seat_only` warning when only one of `first` / `second` is present.

## Union4 Max-Z Top20/Cap25 MC32 Pilot400

The first candidate-subset MC32 pilot was launched with `-ShardSamples 1`, which
made it first-seat only. A matching second-seat complement was generated with
`-RecordSkipBase 1 -FastSkipRecords`, then the two files were merged with unique
sample ids.

Runs:

- first-only:
  `regular-hu-t1-stage2-union4-maxz-top20cap25-mc32-pilot200-20260626-001`
- second-only:
  `regular-hu-t1-stage2-union4-maxz-top20cap25-mc32-pilot200-second-20260626-001`
- balanced output:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_union4_maxz_top20cap25_mc32_pilot200_balanced/hu_turn1_stage1_pilot.jsonl`

Balanced analysis:

| metric | value |
| --- | ---: |
| records | `400` |
| seat split | `first=200`, `second=200` |
| invalid action/state rows | `0 / 0` |
| best action not legal | `0` |
| duplicate sample ids | `0` |
| future samples | `32` |
| action count mean / max | `23.2175 / 25` |
| truncated rows | `377` |
| score gap mean / median | `1.4060 / 0.8736` |
| best action SE mean | `2.6688` |

Speed:

| side | mean seconds/sample |
| --- | ---: |
| first-only | `208.42` |
| second-only | `238.99` |

Candidate models trained on this balanced pilot:

| model | holdout top10 regret | holdout top15 regret | holdout top20 regret | decision |
| --- | ---: | ---: | ---: | --- |
| HGB regressor | `0.0463` | `0.0000` | `0.0000` | diagnostic only; low-k overfits |
| listwise torch | `0.1927` | `0.0647` | `0.0000` | diagnostic only; low-k overfits |
| pairwise logistic | `0.0806` | `0.0040` | `0.0000` | diagnostic only |

External all-action200 MC4 coverage with pilot400 models added to the existing
4-model `max_z_score` union:

| selector | k10 recall / avg regret | k15 avg regret | k20 recall / avg regret / max |
| --- | ---: | ---: | ---: |
| existing4 | `0.810 / 0.4673` | `0.1913` | `0.970 / 0.0396 / 3.5568` |
| existing4 + pilot400 HGB | `0.790 / 0.5686` | `0.2176` | `0.955 / 0.0396 / 3.5568` |
| existing4 + pilot400 listwise | `0.785 / 0.4724` | `0.1102` | `0.970 / 0.0396 / 3.5568` |
| existing4 + pilot400 pairwise | `0.780 / 0.5033` | `0.1673` | `0.960 / 0.0218 / 3.3635` |

Interpretation:

- The balanced MC32 pilot is schema-clean and usable for T1 candidate-subset
  research.
- Training on only 400 rows is not enough to replace the existing T1 candidate
  generator. The new models overfit low-k ranking.
- Existing 4-model `max_z_score` Top20/cap25 remains the safest next generation
  path.
- Pilot400 models may be useful as diagnostics or later ensemble members, but
  they need more balanced MC32/MC64 data before adoption.
- T1 remains `No-Go` for runtime/production.

Next step:

- Generate a larger balanced candidate-subset relabel set using the existing
  4-model `max_z_score` Top20/cap25 selector.
- Use `-ShardSamples 2` for paired rows, or explicitly launch first/second
  complements and merge them with unique sample ids.

## Union4 Max-Z Top20/Cap25 MC32 Balanced2k

Purpose:

- Build the larger balanced candidate-subset MC32 relabel set requested by the
  Pilot400 gate.
- Keep T2 fixed at `stage9f_p2` and T3 fixed at Stage7 `m5_r10`.
- Use `-ShardSamples 2` so first/second T1 rows are produced together.

Run:

- run name:
  `regular-hu-t1-stage2-union4-maxz-top20cap25-mc32-balanced2k-20260626-001`
- output:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_union4_maxz_top20cap25_mc32_balanced2k/hu_turn1_stage1_pilot.jsonl`
- records:
  `2000`
- shards:
  `1000 / 1000`
- recovery:
  spot launch partially failed on zone stockout; missing workers were restarted,
  and the final stragglers were completed on regular `e2-standard-4` VMs in
  `us-central1-f`.

Balanced analysis:

| metric | value |
| --- | ---: |
| records | `2000` |
| seat split | `first=1000`, `second=1000` |
| invalid action/state rows | `0 / 0` |
| best action not legal | `0` |
| duplicate sample/state keys | `0 / 0` |
| future samples | `32` |
| action count mean / max | `23.2355 / 25` |
| truncated rows | `1874` |
| score gap mean / median | `1.3782 / 0.8508` |
| best action SE mean | `2.6980` |
| mean seconds/sample | `239.26` |

Candidate models trained on Balanced2k:

| model | holdout top1 regret | holdout top10 regret | holdout top15 regret | holdout top20 regret | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| HGB regressor | `1.2887` | `0.0534` | `0.0227` | `0.0075` | useful candidate-union member |
| listwise torch | `1.3943` | `0.0548` | `0.0175` | `0.0055` | useful candidate-union member |
| pairwise logistic | `1.5841` | `0.0505` | `0.0169` | `0.0023` | do not add; weak external ordering |
| extra trees | `1.6376` | `0.0894` | `0.0371` | `0.0170` | no adopt |

External all-action200 MC4 coverage:

| selector | k10 recall / avg regret | k15 recall / avg regret | k20 recall / avg regret / max |
| --- | ---: | ---: | ---: |
| existing4 max-z | `0.810 / 0.4673` | `0.890 / 0.1913` | `0.970 / 0.0396 / 3.5568` |
| existing4 + Balanced2k HGB | `0.795 / 0.4895` | `0.890 / 0.1901` | `0.980 / 0.0396 / 3.5568` |
| existing4 + Balanced2k listwise | `0.815 / 0.4565` | `0.905 / 0.1573` | `0.970 / 0.0218 / 3.3635` |
| existing4 + Balanced2k HGB + listwise | `0.815 / 0.4652` | `0.910 / 0.1485` | `0.980 / 0.0218 / 3.3635` |
| existing4 + all Balanced2k models | `0.815 / 0.4652` | `0.910 / 0.1217` | `0.965 / 0.0321 / 3.3635` |

Interpretation:

- Balanced2k is schema-clean and large enough to produce useful T1 candidate
  generator members.
- The best next candidate-generation pool is existing4 plus Balanced2k HGB and
  Balanced2k listwise, using `max_z_score`, Top20 per model, cap25.
- Do not include Balanced2k pairwise in the active pool; it improves some k15
  average regret but reduces k20 recall on the external sentinel.
- T1 remains `No-Go` for runtime/production. This improves candidate-subset
  generation, not the final T1 policy.

6-model union ordering check on all-action200:

| union mode | k10 recall / avg regret | k15 recall / avg regret | k20 recall / avg regret / max |
| --- | ---: | ---: | ---: |
| `max_z_score` | `0.815 / 0.4652` | `0.910 / 0.1485` | `0.980 / 0.0218 / 3.3635` |
| `min_rank` | `0.825 / 0.4955` | `0.905 / 0.1526` | `0.965 / 0.0306 / 3.3635` |
| `rank_sum` | `0.800 / 0.6531` | `0.890 / 0.2203` | `0.975 / 0.0334 / 3.3635` |
| `reciprocal_rank_sum` | `0.805 / 0.5597` | `0.885 / 0.1876` | `0.970 / 0.0334 / 3.3635` |
| `max_score` | `0.790 / 0.6315` | `0.910 / 0.1214` | `0.975 / 0.0396 / 3.3635` |
| `mean_z_score` | `0.810 / 0.6153` | `0.880 / 0.2722` | `0.970 / 0.0361 / 3.3635` |
| `mean_score` | `0.795 / 0.6665` | `0.880 / 0.2609` | `0.965 / 0.0461 / 3.3635` |

Decision: keep `max_z_score` for the next relabel. `max_score` has slightly
better k15 average regret, but `max_z_score` is materially better at k20, which
is the guardrail for candidate omission.

Next step:

- Use the 6-model pool
  `(existing4 + Balanced2k HGB + Balanced2k listwise)` for the next T1
  relabel/evaluation gate.
- Validate with a larger or higher-MC sentinel before calling T1 candidate
  ordering solved.

6-model teacher-path smoke:

- output:
  `outputs/hu_turn1_stage2_stage9f_p2/smoke_6model_maxz_top20cap25_mc1/hu_turn1_stage1_pilot.jsonl`
- records / seat split:
  `2`, `first=1`, `second=1`
- invalid action/state rows:
  `0 / 0`
- continuation:
  T2 `stage9f_p2`, T3 `stage7_m5_r10`
- result:
  the 6-model pool loads and executes cleanly with Top20/cap25 `max_z_score`.

## 6-Model Max-Z Top20/Cap25 MC64 Balanced1k

Purpose:

- Check whether higher-MC labels from the current 6-model candidate pool improve
  T1 candidate ordering.
- Keep T2 fixed at `stage9f_p2` and T3 fixed at Stage7 `m5_r10`.
- Use the same `max_z_score`, Top20 per model, cap25 candidate pool that passed
  the external all-action200 sentinel.

Run:

- run name:
  `regular-hu-t1-stage2-6model-maxz-top20cap25-mc64-balanced1k-20260627-001`
- output:
  `outputs/hu_turn1_stage2_stage9f_p2/gcp_6model_maxz_top20cap25_mc64_balanced1k/hu_turn1_stage1_pilot.jsonl`
- records:
  `1000`
- shards:
  `500 / 500`
- recovery:
  spot launch hit zone stockout and stragglers; missing shards were retried,
  with the final eight shards completed on on-demand `e2-standard-4` workers.

Balanced analysis:

| metric | value |
| --- | ---: |
| records | `1000` |
| seat split | `first=500`, `second=500` |
| invalid action/state rows | `0 / 0` |
| best action not legal | `0` |
| duplicate sample/state keys | `0 / 0` |
| future samples | `64` |
| action count mean / max | `23.778 / 25` |
| truncated rows | `913` |
| score gap mean / median | `1.2921 / 0.7234` |
| best action SE mean | `1.9014` |
| mean seconds/sample | `473.11` |

Candidate models trained on MC64 Balanced1k:

| model | holdout top1 regret | holdout top10 regret | holdout top15 regret | holdout top20 regret | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| HGB regressor | `0.9420` | `0.0307` | `0.0053` | `0.0030` | diagnostic only; no external k20 gain |
| listwise torch | `1.2442` | `0.0399` | `0.0049` | `0.0043` | diagnostic only; no external k20 gain |

External all-action200 MC4 coverage:

| selector | k10 recall / avg regret | k15 recall / avg regret | k20 recall / avg regret / max |
| --- | ---: | ---: | ---: |
| existing 6-model max-z | `0.815 / 0.4652` | `0.910 / 0.1485` | `0.980 / 0.0218 / 3.3635` |
| 6-model + MC64 HGB | `0.810 / 0.4790` | `0.905 / 0.1548` | `0.980 / 0.0218 / 3.3635` |
| 6-model + MC64 listwise | `0.805 / 0.4112` | `0.910 / 0.1420` | `0.980 / 0.0218 / 3.3635` |
| 6-model + both MC64 models | `0.805 / 0.4224` | `0.900 / 0.1582` | `0.980 / 0.0218 / 3.3635` |

External k20 misses are unchanged after adding the MC64 models:

| sample_id | seat | top20 regret | note |
| ---: | --- | ---: | --- |
| `21000001` | second | `1.0000` | small miss |
| `49000000` | first | `0.0000` | tied best omitted |
| `63000001` | second | `0.0000` | tied best omitted |
| `66000000` | first | `3.3635` | meaningful miss; next target |

The four miss rows were then relabeled as all-action MC64 targets on GCP:

- run name:
  `regular-hu-t1-stage2-k20-miss4-mc64-relabel-20260627-001`
- output:
  `outputs/hu_turn1_stage2_stage9f_p2/k20_miss4_relabel_mc64_allactions_gcp/hu_turn1_refinement_relabel.jsonl`
- records / skipped:
  `4 / 0`
- best action changed vs MC4 source:
  `4 / 4`
- relabeled score gap mean / max:
  `0.2912 / 0.6563`
- mean relabel seconds:
  `638.44`

Coverage after MC64 relabel:

| selector | k5 recall / avg regret | k20 recall / avg regret / max |
| --- | ---: | ---: |
| existing 6-model max-z | `1.000 / 0.0000` | `1.000 / 0.0000 / 0.0000` |
| 6-model + both MC64 models | `1.000 / 0.0000` | `1.000 / 0.0000 / 0.0000` |

Interpretation:

- MC64 Balanced1k is schema-clean and useful as diagnostic training data.
- The new HGB/listwise models do not improve the primary guardrail
  `k20 recall/avg regret` on the external all-action200 sentinel.
- Keep the active candidate-generation pool as the existing 6-model
  `max_z_score` Top20/cap25 set.
- Do not add the MC64 Balanced1k models to the active pool yet.
- The external all-action200 k20 misses were not structural candidate-pool
  omissions after higher-MC relabeling. They were MC4/near-tie artifacts:
  every source best action changed at MC64, and the existing 6-model pool
  covered all relabeled best actions by k5.
- The next useful T1 work is not another broad model add. It should either
  raise the external sentinel MC level or move to T1 runtime confirmation with
  the existing 6-model pool.
- T1 remains `No-Go` for runtime/production.

## 6-Model Runtime TopK Confirm Wiring

Purpose:

- Move the active 6-model candidate pool from teacher/coverage analysis into
  the playable `stage9f_p2_hu_t1_topk_confirm` runtime path.
- Preserve single-model behavior for older T1 tests and validation runs.
- Match the teacher-side candidate pool semantics: per-model Top20 union,
  cap25, ordered by `max_z_score`.

Implementation:

- `evaluate_matchups` now accepts `--hu-turn1-stage1-models` as a candidate
  model pool for `stage9f_p2_hu_t1_topk_confirm`.
- `ModelPaths` / `ModelBundle` can load a tuple of HU T1 candidate models.
- `HuTurn1TopKConfirmPolicy` accepts `hu_turn1_candidate_models` and supports
  union modes:
  `min_rank`, `rank_sum`, `reciprocal_rank_sum`, `mean_score`, `max_score`,
  `mean_z_score`, and `max_z_score`.
- T1 config supports:
  `ctopk`, `cap`, and `union`.
- In union mode, `pd` is not used as a candidate-extraction filter. This is
  intentional: union z-score is not a baseline delta, and the teacher pool did
  not filter union candidates by baseline score difference.

Runtime smoke:

```powershell
python -m ofc_regular.evaluate_matchups --profile-a stage9f_p2_hu_t1_topk_confirm --profile-b stage9f_p2 --games 1 --seed 2026062801 --seed-stride 1000000 --opening-lookahead-samples 1 --prediction-threads 1 --hu-turn1-stage1-models models/hu_turn1_stage2_candidate1000_union_b_e_broad1k_top20_cap25_mc32_hgb_regressor.pkl models/hu_turn1_stage2_candidate1200_union_top20_mc16_mc32_hgb_regressor.pkl models/hu_turn1_stage2_top20cap25_mc32_1k_hgb_regressor.pkl models/hu_turn1_stage2_top20cap25_mc32_1k_listwise_torch.pt models/hu_turn1_stage2_union4_maxz_top20cap25_mc32_balanced2k_hgb_regressor.pkl models/hu_turn1_stage2_union4_maxz_top20cap25_mc32_balanced2k_listwise_torch.pt --hu-turn1-topk-config "k25/mc1/d0/confirm1/cse0/pd0/seat=first+second/ctopk20/cap25/union=max_z_score" --hu-turn1-decision-output outputs/evals/hu_turn1_stage2_stage9f_p2_topk_confirm_6model_smoke/decision_log.jsonl --output outputs/evals/hu_turn1_stage2_stage9f_p2_topk_confirm_6model_smoke/summary.json
```

Smoke result:

| metric | value |
| --- | ---: |
| paired seeds / hands | `1 / 2` |
| HU T1 decisions | `2` |
| candidate model count | `6` |
| union mode | `max_z_score` |
| observed union sizes | `24`, `26` |
| evaluated action counts | `24`, `26` |
| non-fired delta max abs | `0.0` |

Decision:

- Runtime wiring is valid: the 6-model pool loads, forms the union, and passes
  the union candidates into Stage A MC.
- This is not quality evidence. T1 remains `No-Go` until a seat-swap gate shows
  positive realized fired whole-game delta with non-fired cancellation intact.

Next gate:

- Run a small non-overlapping seat-swap with the active 6-model runtime pool.
- Suggested first validation config:
  `k25/mc8/d0/confirm16/cse1/pd0/seat=first+second/ctopk20/cap25/union=max_z_score`.
- If fire count is too low, loosen confirm only after verifying non-fired
  cancellation remains exact.

Small local validation:

```powershell
python -m ofc_regular.evaluate_matchups --profile-a stage9f_p2_hu_t1_topk_confirm --profile-b stage9f_p2 --games 5 --seed 2026062802 --seed-stride 1000000 --opening-lookahead-samples 1 --prediction-threads 1 --hu-turn1-stage1-models models/hu_turn1_stage2_candidate1000_union_b_e_broad1k_top20_cap25_mc32_hgb_regressor.pkl models/hu_turn1_stage2_candidate1200_union_top20_mc16_mc32_hgb_regressor.pkl models/hu_turn1_stage2_top20cap25_mc32_1k_hgb_regressor.pkl models/hu_turn1_stage2_top20cap25_mc32_1k_listwise_torch.pt models/hu_turn1_stage2_union4_maxz_top20cap25_mc32_balanced2k_hgb_regressor.pkl models/hu_turn1_stage2_union4_maxz_top20cap25_mc32_balanced2k_listwise_torch.pt --hu-turn1-topk-config "k25/mc4/d0/confirm8/cse1/pd0/seat=first+second/ctopk20/cap25/union=max_z_score" --hu-turn1-decision-output outputs/evals/hu_turn1_stage2_stage9f_p2_topk_confirm_6model_small5/decision_log.jsonl --output outputs/evals/hu_turn1_stage2_stage9f_p2_topk_confirm_6model_small5/summary.json
```

Small local result:

| metric | value |
| --- | ---: |
| paired seeds / hands | `5 / 10` |
| HU T1 decisions | `10` |
| overrides | `1` |
| non-fired delta max abs | `0.0` |
| no-override reasons | `below_confirm_delta=6`, `below_confirm_se=2`, `mc_best_is_baseline=1` |
| avg / max runtime latency per T1 decision | `29.07s / 41.87s` |
| elapsed | `294.23s` |

Interpretation:

- Non-fired cancellation is still exact.
- The 6-model runtime can fire an override.
- The run is far too small for quality and too slow to scale locally.
- Next validation should use Spot VM.

GCP support:

- `scripts/Start-GcpHuTurn2Stage9fProfileCanaryRun.ps1` now accepts
  `-HuTurn1Stage1Models` and passes the model pool to `evaluate_matchups` as
  `--hu-turn1-stage1-models`.
- Dry run passed with phase `hu_t1_stage2_6model_runtime_seatswap`:

```powershell
.\scripts\Start-GcpHuTurn2Stage9fProfileCanaryRun.ps1 `
  -RunName regular-hu-t1-stage2-6model-runtime-dryrun `
  -TotalGames 10 `
  -ShardGames 5 `
  -VmCount 2 `
  -ProfileA stage9f_p2_hu_t1_topk_confirm `
  -ProfileB stage9f_p2 `
  -Phase hu_t1_stage2_6model_runtime_seatswap `
  -HuTurn1TopkConfig 'k25/mc4/d0/confirm8/cse1/pd0/seat=first+second/ctopk20/cap25/union=max_z_score' `
  -HuTurn1Stage1Models @(
    'models/hu_turn1_stage2_candidate1000_union_b_e_broad1k_top20_cap25_mc32_hgb_regressor.pkl',
    'models/hu_turn1_stage2_candidate1200_union_top20_mc16_mc32_hgb_regressor.pkl',
    'models/hu_turn1_stage2_top20cap25_mc32_1k_hgb_regressor.pkl',
    'models/hu_turn1_stage2_top20cap25_mc32_1k_listwise_torch.pt',
    'models/hu_turn1_stage2_union4_maxz_top20cap25_mc32_balanced2k_hgb_regressor.pkl',
    'models/hu_turn1_stage2_union4_maxz_top20cap25_mc32_balanced2k_listwise_torch.pt'
  ) `
  -DryRun
```

Next Spot VM gate:

- Start with `k25/mc4/confirm8` to measure fresh fire rate and latency at scale.
- Move to `k25/mc8/confirm16` only if fire count is adequate and non-fired
  cancellation stays exact.
- T1 is still not complete; this is the first playable 6-model runtime path.
