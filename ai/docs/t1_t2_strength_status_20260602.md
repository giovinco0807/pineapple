# T1/T2 Strength Status 2026-06-02

## Current conclusion

The system is stronger than the original pure model baseline, but the strength is uneven.

- T2 is close to the 5 second gameplay target.
- T1 still blocks reliable gameplay.
- T1 Pool20/Sync selection is mostly good enough on many sets, but hard T1 cases fail at the final selection step.

## Verified metrics

### T2 current runtime

Source summary:

- `ai/data/hybrid_t1t2_active_20260531/eval_hybrid_refine_local500_t2_mc300_active61_plus_t2resid200_turnmodel_t2sync10/summary.json`

Known current result:

- decisions: 254
- final top1: 96.85%
- p95 latency: 2294.9 ms
- average regret: 0

### T1 normal source-neighbor set

Source summary:

- `ai/data/hybrid_t1t2_active_20260531/eval_t1_source_neighbor136_mc1000_k10_flsafe_aux_t1gap1mc50ft_k5_20260602/summary.json`

Known current result:

- final top1: 71.32%
- average regret: 0.2693
- pool recall: 99.26%
- sync recall: 92.65%
- p95 latency: 4507.7 ms

### T1 hard sets

Current runtime summaries:

- `ai/data/hybrid_t1t2_active_20260531/eval_t1_residual17_blocker_rows_currentbest_20260602/summary.json`
- `ai/data/hybrid_t1t2_active_20260531/eval_t1_hardcase31_blocker_rows_currentbest_20260602/summary.json`
- `ai/data/hybrid_t1t2_active_20260531/eval_t1_finalgate37_blocker_rows_currentbest_20260602/summary.json`

Results:

| set | decisions | final top1 | avg regret | pool recall | sync recall | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| residual17 | 17 | 41.18% | 0.6636 | 100.00% | 100.00% | 4827.8 |
| hardcase31 | 31 | 0.00% | 1.2403 | 100.00% | 93.55% | 4290.5 |
| finalgate37 | 37 | 24.32% | 0.8618 | 97.30% | 72.97% | 4643.1 |

## Experiments from this pass

### New final selector with finalgate37 data

Selector:

- `ai/data/hybrid_t1t2_active_20260531/t1_final_selector_blocker_source136_residual200_finalgate37_margin025_regret100_e600_20260602.json`

LOSO arbitration report:

- `ai/data/hybrid_t1t2_active_20260531/t1_selector_arbitration_loso_current_vs_src136_resid200_finalgate37_e600_20260602.json`

Result:

- Do not promote.
- Average heldout regret improved, but residual17 regressed.
- residual17 top1 delta: -5.88 percentage points.
- residual17 regret delta: +0.3074.

### New sync selector with finalgate37 data

Selector:

- `ai/data/hybrid_t1t2_active_20260531/t1_sync_selector_source136_residual200_finalgate37_e2000_20260602.json`

Runtime report:

- `ai/data/hybrid_t1t2_active_20260531/eval_t1_finalgate37_newsync_source136_residual200_finalgate37_e2000_20260602/summary.json`

Result on finalgate37:

| config | final top1 | avg regret | pool recall | sync recall | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| current sync | 24.32% | 0.8618 | 97.30% | 72.97% | 4643.1 |
| new sync | 21.62% | 0.8665 | 97.30% | 86.49% | 4271.6 |

Interpretation:

- New sync selector improves sync recall strongly.
- It does not improve final top1 because the final selector fails to pick the teacher action even when it is refined.
- Do not promote as default yet.

### New final selector trained on new-sync rows

Selector:

- `ai/data/hybrid_t1t2_active_20260531/t1_final_selector_source136_residual200_newsync_finalgate37_margin025_regret100_e600_20260602.json`

Result:

- Do not promote.
- residual17 holdout selector top1: 29.41%
- hardcase31 holdout selector top1: 9.68%

## Failure decomposition

The key blocker is no longer simply candidate generation.

| set/config | sync hit but final miss | pool hit but sync miss | pool miss | final hit |
| --- | ---: | ---: | ---: | ---: |
| finalgate37 current sync | 19 | 9 | 1 | 9 |
| finalgate37 new sync | 24 | 4 | 1 | 8 |
| residual17 current | 10 | 0 | 0 | 7 |
| hardcase31 current | 29 | 2 | 0 | 0 |

For hardcase31, 29 of 31 decisions already have the teacher-best action in the sync/refined candidate set, but final selection still misses all 31. This is a final selection/generalization failure.

## Next direction

1. Keep T2 fixed except for regression checks.
2. Keep current T1 runtime defaults for now.
3. Do not promote the new final selectors.
4. Use the new sync selector as an experimental candidate only; it proves sync recall can improve but needs a stronger final selector.
5. Next work should target T1 final selection on sync-hit/final-miss rows:
   - extract rows where `teacher_best_in_sync=true` and `final_top1_hit=false`;
   - train/evaluate a pairwise or setwise selector focused on teacher-vs-runtime-chosen action;
   - include opponent board/dead-card features and FL type features;
   - validate with LOSO across source136, residual200, residual17, hardcase31, and finalgate37;
   - promote only if no heldout set regresses in regret and hardcase/finalgate improve.

## New target set for the next training pass

`ai/tutor/collect_runtime_active_targets.py` now supports the reason `sync_hit_final_miss`.

Generated targets:

- `ai/data/hybrid_t1t2_active_20260531/targets_t1_sync_hit_final_miss_20260602.jsonl`
- `ai/data/hybrid_t1t2_active_20260531/teacher_t1_sync_hit_final_miss_20260602.jsonl`

Counts:

- total written: 29
- finalgate37_newsync: 15
- residual17_current: 10
- hardcase31_current: 4

These are the most direct examples for the next T1 final-selector training pass because the teacher-best action was already inside the sync/refined candidate set, yet the runtime final selection still missed it.

## Pairwise final-selector pass

Added tools:

- `ai/tutor/train_t1_pairwise_final_selector.py`
- `ai/tutor/evaluate_t1_selector_rule_grid.py`

The first pairwise attempt with `pair-mode=runtime_best` failed because it only trained teacher-vs-runtime-best pairs and allowed unrelated third candidates to receive extreme scores.

The useful version was:

- `ai/data/hybrid_t1t2_active_20260531/t1_pairwise_final_selector_src136_resid200_newsync_finalgate37_allpairs_fmiss_e1000_20260602.json`

Training setup:

- train rows: source136 currentbest, residual200 currentbest, finalgate37 newsync
- group filter: final misses only
- pair mode: all candidate pairs against teacher-best
- pair groups: 65
- pair count: 585

Offline result:

- hardcase31 selector-row Top1 improved from 9.68% baseline selector to 58.06%.
- residual17 Top1 stayed 47.06%, but average regret was worse, so the pairwise selector must not replace the current selector directly.

## Pairwise arbitration rule

Added runtime policy:

- `pairwise_refined_delta_ge_m086_current_bust_ge_0026`

Policy:

- use current final selector as the default pick;
- score the same refined candidates with the pairwise selector;
- switch only if:
  - challenger action differs from current action;
  - `challenger.refined_score - current.refined_score >= -0.8597222222222314`;
  - `current.predicted_bust >= 0.02634526789188385`.

Rule-grid report:

- `ai/data/hybrid_t1t2_active_20260531/t1_rule_grid_current_vs_pairwise_allpairs_fmiss_guard_source_residual_target_hard_finalgate_20260602.json`

Guard sets:

- source136
- residual200
- residual17

Target sets:

- hardcase31
- finalgate37

The top rule improved all guard sets in offline selector-row replay while improving hardcase/finalgate.

Runtime checks:

| set/config | final top1 | avg regret | pool recall | sync recall | p95 ms | arbitration applied |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| source136 current | 70.59% | 0.3594 | 99.26% | 92.65% | 4520.4 | 0 |
| source136 pairwise arbitration | 71.32% | 0.2369 | 99.26% | 92.65% | 4460.5 | 13 |
| residual200 current | 62.50% | 0.3979 | 100.00% | 91.50% | 4087.7 | 0 |
| residual200 pairwise arbitration | 65.00% | 0.3517 | 100.00% | 91.50% | 3997.4 | 22 |
| hardcase31 current | 0.00% | 1.2403 | 100.00% | 93.55% | 4290.5 | 0 |
| hardcase31 pairwise arbitration | 9.68% | 0.9307 | 100.00% | 93.55% | 3931.1 | 3 |
| finalgate37 current sync | 24.32% | 0.8618 | 97.30% | 72.97% | 4643.1 | 0 |
| finalgate37 newsync only | 21.62% | 0.8665 | 97.30% | 86.49% | 4271.6 | 0 |
| finalgate37 newsync + pairwise arbitration | 32.43% | 0.5853 | 97.30% | 86.49% | 4385.9 | 7 |
| residual17 current | 41.18% | 0.6636 | 100.00% | 100.00% | 4827.8 | 0 |
| residual17 pairwise arbitration | 41.18% | 0.6636 | 100.00% | 100.00% | 4661.4 | 0 |
| runtime112 current | 47.32% | 0.7875 | 100.00% | 93.75% | 4238.8 | 0 |
| runtime112 pairwise arbitration | 49.11% | 0.7479 | 100.00% | 93.75% | 4250.0 | 9 |

### Broad MC50 smoke check

Generated a fixed broad T1 smoke slice:

- `ai/data/hybrid_t1t2_active_20260531/teacher_t1_broad_selfplay_mc50_gap1p0_first100_20260602.jsonl`

This slice contains the first 100 T1 rows from:

- `ai/data/hybrid_t1t2_active_20260531/teacher_t1t2_mixed_broad500_diverse1000_mc50_gap1p0_allactions_20260602.jsonl`

The labels are `mc50`, so this is a noisy overfit/regression smoke test, not a promotion-quality accuracy benchmark.

| set/config | final top1 | avg regret | pool recall | sync recall | p95 ms | time violations | arbitration applied |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| broad MC50 first100 current | 31.00% | 3.3832 | 96.00% | 88.00% | 4308.5 | 0 | 0 |
| broad MC50 first100 pairwise arbitration | 32.00% | 3.2524 | 96.00% | 88.00% | 4321.1 | 0 | 8 |

Pairwise arbitration changed the final action on 4 of 100 rows. All 4 changes improved the MC50 teacher score, with total changed-row score delta +13.0766.

### MC1000 relabel probe for broad MC50 misses

Collected high-priority relabel targets from the pairwise broad MC50 smoke:

- `ai/data/hybrid_t1t2_active_20260531/targets_t1_broad_mc50_first100_pairwise_misses_20260602.jsonl`
- `ai/data/hybrid_t1t2_active_20260531/teacher_t1_broad_mc50_first100_pairwise_misses_20260602.jsonl`

Counts:

- written targets: 68
- final misses: 68
- sync misses: 12
- pool misses: 4

Ran a local MC1000 probe on the first 10 targets:

- chunks: `ai/data/hybrid_t1t2_active_20260531/chunks_t1_broad_mc50_misses_mc1000_probe10_20260602`
- merged output: `ai/data/hybrid_t1t2_active_20260531/teacher_t1_broad_mc50_misses_mc1000_probe10_20260602.jsonl`

Runtime:

- 10 records took 447 seconds on local CPU with 2 workers.
- The remaining 58 records took 2725 seconds on local CPU with 2 workers.
- Full 68-target relabel took about 52 minutes including the probe run.

Label stability:

- `ai/data/hybrid_t1t2_active_20260531/audit_t1_broad_mc50_misses_probe10_mc50_vs_mc1000_20260602.json`
- matched rows: 9
- MC50/MC1000 Top1 same: 2 / 9
- MC50/MC1000 Top1 changed: 7 / 9
- MC50 Top1 inside MC1000 Top3: 2 / 9
- MC1000 Top1 inside MC50 Top3: 3 / 9

This confirms MC50 labels are too noisy for direct T1 training. MC50 is useful for finding suspicious positions, but promotion-quality training/evaluation should use MC1000 or better.

Runtime on the 10 MC1000 relabeled targets:

| set/config | final top1 | avg regret | pool recall | sync recall | p95 ms | time violations | arbitration applied |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MC1000 probe10 current | 80.00% | 0.0956 | 100.00% | 100.00% | 4474.0 | 0 | 0 |
| MC1000 probe10 pairwise arbitration | 80.00% | 0.0956 | 100.00% | 100.00% | 4526.1 | 0 | 0 |

The probe suggests many broad MC50 "misses" disappear under MC1000. The remaining two misses are final-selection misses, not pool/sync misses.

### MC1000 relabel result for all 68 broad MC50 misses

Completed all 68 MC1000 relabels:

- first 10: `ai/data/hybrid_t1t2_active_20260531/teacher_t1_broad_mc50_misses_mc1000_probe10_20260602.jsonl`
- remaining 58: `ai/data/hybrid_t1t2_active_20260531/teacher_t1_broad_mc50_misses_mc1000_remaining58_20260602.jsonl`
- merged 68: `ai/data/hybrid_t1t2_active_20260531/teacher_t1_broad_mc50_misses_mc1000_all68_20260602.jsonl`

Full MC50-vs-MC1000 stability:

- report: `ai/data/hybrid_t1t2_active_20260531/audit_t1_broad_mc50_misses_all68_mc50_vs_mc1000_20260602.json`
- common states: 68
- matched action keys: 60
- MC50/MC1000 Top1 same: 12 / 60
- MC50/MC1000 Top1 changed: 48 / 60
- MC50 Top1 inside MC1000 Top3: 23 / 60
- MC1000 Top1 inside MC50 Top3: 22 / 60

This is strong evidence that MC50 Top1 labels should not be used as direct training labels for T1.

Runtime on all 68 MC1000 relabeled targets:

| set/config | final top1 | avg regret | pool recall | sync recall | p95 ms | time violations | arbitration applied |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MC1000 all68 current | 50.00% | 0.7047 | 98.53% | 91.18% | 4241.8 | 0 | 0 |
| MC1000 all68 pairwise arbitration | 50.00% | 1.0011 | 98.53% | 91.18% | 4415.0 | 0 | 6 |
| MC1000 all68 refined_score selection | 51.47% | 0.6739 | 98.53% | 91.18% | 4172.0 | 0 | 0 |

Pairwise arbitration on all68:

- changed final action: 4 / 68
- better changes: 2
- worse changes: 1
- same-score changes: 1
- total changed-row teacher score delta: -20.1573

The single bad pairwise switch was large enough to wipe out previous gains. Do not promote `pairwise_refined_delta_ge_m086_current_bust_ge_0026` as a default.

MC1000-confirmed current misses:

- current final misses: 34 / 68
- sync misses: 6 / 68
- pool misses: 1 / 68
- sync-hit/final-miss: 28 / 68

High-confidence active targets extracted with teacher margin >= 0.25, sims >= 1000, and regret >= 0.1:

- `ai/data/hybrid_t1t2_active_20260531/targets_t1_broad_mc1000_all68_current_misses_margin025_20260602.jsonl`
- `ai/data/hybrid_t1t2_active_20260531/teacher_t1_broad_mc1000_all68_current_misses_margin025_20260602.jsonl`

Counts:

- written targets: 14
- final misses: 14
- sync-hit/final-miss: 12
- sync misses: 2
- pool misses: 0 after margin/regret filtering

### Selector retraining with broad68

Trained a broad68-augmented pairwise selector:

- `ai/data/hybrid_t1t2_active_20260531/t1_pairwise_final_selector_src136_resid200_newsync_finalgate37_broad68_allpairs_fmiss_e1000_20260602.json`

Training inputs:

- source136 current final misses
- residual200 current final misses
- finalgate37 new-sync final misses
- broad MC1000 all68 current final misses

Result:

- Do not promote.
- It improves hardcase31 selector-row Top1, but it regresses runtime112 and broad68 average regret against the current baseline selector.
- On broad68 selector rows, plain `refined_score` is stronger than both the current selector and the newly trained pairwise selector.
- On runtime112 selector rows, the current baseline selector remains stronger than `refined_score`, so `refined_score` cannot become the global T1 default.

### Current selector vs refined_score guard

Added a manual pseudo-selector for offline arbitration:

- `ai/data/hybrid_t1t2_active_20260531/t1_refined_score_selector_20260602.json`

Rule-grid report:

- `ai/data/hybrid_t1t2_active_20260531/t1_rule_grid_current_vs_refined_score_guard_existing_target_broad68_20260602.json`

Offline top rule:

- switch from current selector to `refined_score` if:
  - `challenger_model_rank >= 6`
  - `selector_conflict_margin >= -0.8622868813378028`

Offline selector-row replay showed no guard-set regret regression and improved broad68, so this was implemented as an experimental runtime policy:

- `refined_score_model_rank_ge6_conflict_ge_m086`

Runtime on broad68:

| set/config | final top1 | avg regret | pool recall | sync recall | p95 ms | time violations | arbitration applied |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MC1000 all68 current | 50.00% | 0.7047 | 98.53% | 91.18% | 4241.8 | 0 | 0 |
| MC1000 all68 refined_score guard | 51.47% | 0.7118 | 98.53% | 91.18% | 4357.5 | 0 | 2 |

The runtime guard changed 2 decisions: 1 better and 1 worse, total changed-row teacher score delta -0.4843.

Result:

- Do not promote.
- It improves Top1 by one decision on broad68, but violates the regret non-regression rule.
- The offline selector-row rule is not sufficient; runtime override/rescue interactions need to be included in future gate training.

### Runtime current vs refined_score guard search across six guard sets

Added matching runtime `refined_score` evaluations for the main guard sets:

- `ai/data/hybrid_t1t2_active_20260531/eval_t1_source136_refined_score_20260602`
- `ai/data/hybrid_t1t2_active_20260531/eval_t1_residual200_refined_score_20260602`
- `ai/data/hybrid_t1t2_active_20260531/eval_t1_residual17_refined_score_20260602`
- `ai/data/hybrid_t1t2_active_20260531/eval_t1_hardcase31_refined_score_20260602`
- `ai/data/hybrid_t1t2_active_20260531/eval_t1_finalgate37_refined_score_20260602`

Current selector vs plain `refined_score`:

| set | current top1 | current regret | refined_score top1 | refined_score regret | result |
| --- | ---: | ---: | ---: | ---: | --- |
| runtime112 | 47.32% | 0.7875 | 44.64% | 0.8739 | worse |
| source136 | 70.59% | 0.3594 | 63.97% | 0.3466 | lower Top1, lower regret |
| residual200 | 62.50% | 0.3979 | 62.00% | 0.3892 | slightly lower Top1, lower regret |
| residual17 | 41.18% | 0.6636 | 29.41% | 1.2876 | worse |
| hardcase31 | 0.00% | 1.2403 | 6.45% | 1.2619 | higher Top1, worse regret |
| finalgate37 | 24.32% | 0.8618 | 16.22% | 1.0286 | worse |
| broad68 | 50.00% | 0.7047 | 51.47% | 0.6739 | better |

Strict runtime rule search:

- report: `ai/data/hybrid_t1t2_active_20260531/t1_runtime_rule_grid_current_vs_refined_score_guard6_target_broad68_20260602.json`
- guard sets: `runtime112`, `source136`, `residual200`, `residual17`, `hardcase31`, `finalgate37`
- target set: `broad68`
- non-regression rule: no average regret increase on any guard set

Best surviving rule:

- switch to `refined_score` if `t1_predicted_bust_delta_delta <= -0.10457902774214745`
- guard behavior: zero switches on every guard set, so no guard regression
- broad68 behavior: 1 switch, Top1 50.00% -> 51.47%, regret 0.7047 -> 0.6868

Result:

- Do not promote.
- The only strict-safe rule is too weak: it improves one broad68 row and learns almost nothing reusable.
- The promising two-feature rule from the runtime112+broad68 search does not survive the broader guard sets.
- To make T1 materially stronger, the next move is not another hand-written runtime gate. It is more MC1000+ high-confidence final-selection data and then retraining a final selector from those rows.

Current interpretation:

- T1 is stronger than the raw model because Pool20 plus recursive refinement and selectors lift Top1 substantially under the 5 second budget.
- The previous pairwise arbitration policy is no longer safe after the MC1000 all68 check.
- Plain `refined_score` is not a safe global default.
- Keep the code default arbitration as `none`; do not promote pairwise or refined_score arbitration yet.
- The broad MC50 result shows the model can look weak under noisy labels; MC1000 relabeling is required before training from these rows.
- For T1, the current bottleneck on reliable MC1000 rows is mostly final selection, not Pool20 candidate generation.
- Current strict-safe runtime gates are too weak to matter.
- Future selection gains should come from more MC1000+ final-selection data, then a learned final selector validated on actual runtime results.

Deprecated experimental command additions:

- `--t1-arbitration-selector ai/data/hybrid_t1t2_active_20260531/t1_pairwise_final_selector_src136_resid200_newsync_finalgate37_allpairs_fmiss_e1000_20260602.json`
- `--t1-arbitration-policy pairwise_refined_delta_ge_m086_current_bust_ge_0026`

Next checkpoint:

1. Do not use MC50 Top1 labels for T1 training.
2. Train from the 14 MC1000-confirmed high-confidence T1 misses plus existing MC1000 final-selection misses.
3. Add more MC1000+ high-confidence T1 final-selection rows before trying another selector promotion.
4. Build any future guarded selection policy from runtime result rows, including override/rescue effects, not just selector-row replay.
5. Promote only if average regret does not increase on any guard set and p95 remains below 5 seconds.

### MC1000 final-selection miss consolidation

Collected a broader high-confidence T1 final-selection miss set from existing MC1000 runtime evaluations:

- targets: `ai/data/hybrid_t1t2_active_20260531/targets_t1_final_selection_mc1000_guardsets_margin025_regret010_20260602.jsonl`
- teacher labels: `ai/data/hybrid_t1t2_active_20260531/teacher_t1_final_selection_mc1000_guardsets_margin025_regret010_20260602.jsonl`
- summary: `ai/data/hybrid_t1t2_active_20260531/targets_t1_final_selection_mc1000_guardsets_margin025_regret010_20260602.summary.json`

Filters:

- turn 1 only
- reasons: `final_miss`, `sync_hit_final_miss`
- teacher sims >= 1000
- teacher margin >= 0.25
- runtime regret >= 0.1
- no suit expansion

Result:

- written targets: 90
- sync-hit/final-miss: 78
- source tags:
  - runtime112: 59
  - source136: 11
  - residual200: 12
  - broad68: 8
- residual17, hardcase31, and finalgate37 contributed no additional unique rows after the filters and dedupe.

Runtime selector-row generation on these 90 rows:

- output: `ai/data/hybrid_t1t2_active_20260531/eval_t1_final_selection90_currentbest_20260602`

| set/config | final top1 | avg regret | pool recall | sync recall | p95 ms | time violations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| final-selection90 current | 1.11% | 1.5838 | 100.00% | 86.67% | 4549.4 | 0 |

This confirms the key T1 failure mode: the teacher-best action is usually already in Pool20, and often in the refined/sync set, but the final selector chooses the wrong candidate.

### Selector retraining smoke from final-selection90

Pairwise selector trained only on final-selection90:

- `ai/data/hybrid_t1t2_active_20260531/t1_pairwise_final_selector_finalsel90_fmiss_runtimebest_e1000_20260602.json`

Result:

- Do not promote.
- The model collapsed on selector-row replay: train/heldout selector Top1 was 0%, and every guard eval was far worse than the current baseline selector.

Listwise selector trained only on final-selection90:

- `ai/data/hybrid_t1t2_active_20260531/t1_final_selector_finalsel90_listwise_margin025_heldout25_20260602.json`

Result:

- Do not promote.
- It fits the final-selection90 rows, but badly regresses normal source/residual/broad selector-row sets.

Mixed listwise selector trained on source136 + residual200 + final-selection90:

- `ai/data/hybrid_t1t2_active_20260531/t1_final_selector_src136_resid200_finalsel90_listwise_margin025_heldout15_e500_20260602.json`

Selector-row comparison against the current baseline selector:

| eval set | current baseline top1/regret | mixed selector top1/regret | result |
| --- | ---: | ---: | --- |
| runtime112 | 50.46% / 0.6911 | 49.54% / 0.7035 | worse |
| residual17 | 47.06% / 0.5768 | 41.18% / 0.9923 | worse |
| hardcase31 | 9.68% / 1.1682 | 29.03% / 0.8227 | better |
| finalgate37 | 20.83% / 1.5949 | 41.67% / 0.6505 | better |
| broad68 | 64.44% / 0.4148 | 62.22% / 0.3946 | lower Top1, lower regret |

Result:

- Do not promote.
- The mixed selector learns useful corrections for hardcase31 and finalgate37, but it regresses runtime112 and residual17. A single global linear selector is too blunt for these failure modes.

Updated direction:

1. Keep the current final selector as the default.
2. Use final-selection90 as a hard-case training/evaluation pack, not as a standalone replacement set.
3. Next selector work should either:
   - train a gated/mixture selector that applies only to hardcase-like and finalgate-like rows, or
   - use constrained training where no guard set is allowed to regress.
4. More MC1000 data is still useful, but it needs to preserve normal-position coverage. Adding only miss rows makes the selector over-correct.

### Runtime strict mixed-selector gate

Implemented two explicit T1 arbitration policies in `ai/tutor/hybrid_t1t2.py`:

- `mixed_kk_delta_le_0043_model_delta_le_1261`
- `mixed_kk_delta_le_0043_model_delta_le_1111_conflict_ge_m0483`

The wider gate fixed several hard cases but regressed broad68 runtime average regret, so it should not be promoted.

The stricter gate keeps the same challenger selector but only switches when:

- `predicted_kk_delta <= 0.043167173862457275`
- `model_score_delta <= 1.110815`
- `selector_conflict_margin >= -0.48343`

Strict gate runtime comparison against the current default 5 second T1 config:

| set | current Top1/regret/p95 | strict gate Top1/regret/p95 | Top1 delta | regret delta | strict time violations |
| --- | ---: | ---: | ---: | ---: | ---: |
| runtime112 | 47.32% / 0.7875 / 4239ms | 48.21% / 0.7163 / 4272ms | +0.89pt | -0.0712 | 0 |
| source136 | 70.59% / 0.3594 / 4520ms | 69.12% / 0.2308 / 4701ms | -1.47pt | -0.1287 | 2 |
| residual200 | 62.50% / 0.3979 / 4088ms | 64.50% / 0.3687 / 4042ms | +2.00pt | -0.0292 | 0 |
| residual17 | 41.18% / 0.6636 / 4828ms | 47.06% / 0.5879 / 4616ms | +5.88pt | -0.0757 | 0 |
| broad68 | 50.00% / 0.7047 / 4242ms | 54.41% / 0.6671 / 4109ms | +4.41pt | -0.0376 | 0 |
| hardcase31 | 0.00% / 1.2403 / 4290ms | 12.90% / 0.8993 / 4058ms | +12.90pt | -0.3410 | 0 |
| finalgate37 | 24.32% / 0.8618 / 4643ms | 29.73% / 0.6763 / 4225ms | +5.41pt | -0.1855 | 0 |
| final-selection90 | 1.11% / 1.5838 / 4549ms | 13.33% / 1.3152 / 4414ms | +12.22pt | -0.2686 | 0 |

Result:

- Keep the project default arbitration as `none` for now.
- The strict gate is the best T1 runtime challenger so far: it lowers average regret on every tested set and improves Top1 on 7 of 8 sets.
- Do not call it fully promoted yet because source136 Top1 drops by 1.47pt and the Python runtime still has a few wall-clock outliers near or above 5 seconds.
- Next speed check should test `sync_exact_k=8` or `9` with the strict gate to preserve the accuracy gain while reducing worst-case latency.
- Next quality check should collect the remaining strict-gate misses as MC1000+ final-selection training rows, while preserving normal-position rows to avoid over-correction.

### Sync exact K speed check

Tested the strict gate with lower T1 sync refinement counts.

Source136 speed-risk set:

| config | Top1 | regret | p95 | max | time violations | sync recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current default | 70.59% | 0.3594 | 4520ms | 5022ms | 0 | 92.65% |
| strict gate, k=10 | 69.12% | 0.2308 | 4701ms | 5511ms | 2 | 92.65% |
| strict gate, k=9 | 69.12% | 0.2589 | 3888ms | 5015ms | 0 | 91.91% |
| strict gate, k=8 | 66.91% | 0.3295 | 3592ms | 5037ms | 0 | 89.71% |

Representative checks:

| set/config | Top1 | regret | p95 | sync recall |
| --- | ---: | ---: | ---: | ---: |
| runtime112 current | 47.32% | 0.7875 | 4239ms | 93.75% |
| runtime112 strict k=10 | 48.21% | 0.7163 | 4272ms | 93.75% |
| runtime112 strict k=9 | 45.54% | 0.7929 | 3834ms | 89.29% |
| runtime112 strict k=9 adaptive model-rank-1 to k=10 | 45.54% | 0.7700 | 3779ms | 90.18% |
| broad68 current | 50.00% | 0.7047 | 4242ms | 91.18% |
| broad68 strict k=10 | 54.41% | 0.6671 | 4109ms | 91.18% |
| broad68 strict k=9 | 52.94% | 0.6741 | 3688ms | 88.24% |
| final-selection90 current | 1.11% | 1.5838 | 4549ms | 86.67% |
| final-selection90 strict k=10 | 13.33% | 1.3152 | 4414ms | 86.67% |
| final-selection90 strict k=9 | 14.44% | 1.1866 | 3842ms | 84.44% |

Result:

- Do not adopt fixed `sync_exact_k=8`; it loses too much recall.
- Do not adopt fixed `sync_exact_k=9` globally; it improves latency but regresses runtime112 Top1 and regret relative to the current default.
- The current best quality candidate remains strict gate with `sync_exact_k=10`.
- For true "never above 5 seconds" behavior, the next implementation should be latency control around T1 recursive refinement, not a global K reduction. Because T1 refinement is currently a batch call, this likely means adding a pre-call latency headroom estimate or splitting the last candidate(s) into a smaller second batch.

Collected remaining strict-gate misses:

- targets: `ai/data/hybrid_t1t2_active_20260531/targets_t1_strict_gate_remaining_misses_mc1000_margin025_regret010_20260602.jsonl`
- teacher labels: `ai/data/hybrid_t1t2_active_20260531/teacher_t1_strict_gate_remaining_misses_mc1000_margin025_regret010_20260602.jsonl`
- summary: `ai/data/hybrid_t1t2_active_20260531/targets_t1_strict_gate_remaining_misses_mc1000_margin025_regret010_20260602.summary.json`

Counts:

- written / teacher_written: 85
- sync-hit/final-miss: 73
- tags:
  - runtime112 strict: 58
  - source136 strict: 9
  - residual200 strict: 9
  - broad68 strict: 9

These are the next high-value T1 final-selection rows. Before retraining, convert/evaluate them into selector rows and mix them with normal-position rows; training only on remaining misses would over-correct the final selector.

### Remaining85 selector retraining attempt

Converted the remaining strict-gate miss pack into selector rows:

- runtime eval: `ai/data/hybrid_t1t2_active_20260531/eval_t1_strict_remaining85_mixed_arbitration_strict_gate_20260602`
- selector rows: `ai/data/hybrid_t1t2_active_20260531/eval_t1_strict_remaining85_mixed_arbitration_strict_gate_20260602/selector_rows.jsonl`

This set is intentionally hard:

- decisions: 85
- final Top1 under strict gate: 0.00%
- Pool recall: 100.00%
- Sync recall: 85.88%
- p95: 4866ms

Trained a mixed listwise selector from normal rows plus final/hard rows plus remaining85:

- selector: `ai/data/hybrid_t1t2_active_20260531/t1_final_selector_runtime_source_resid_broad_finalsel_remain85_listwise_margin025_heldout15_e700_20260602.json`
- train inputs:
  - runtime112 current selector rows
  - source136 current selector rows
  - residual200 current selector rows
  - broad68 current selector rows
  - final-selection90 current selector rows
  - remaining85 strict selector rows

Selector-row gate search against the current selector found a strong-looking rule:

- policy shape: switch to the new selector when `current_margin_under_current <= 0.664098569143615`
- guard sets all improved in selector-row replay
- target sets all improved in selector-row replay

Selector-row deltas for this rule:

| set | Top1 delta | regret delta |
| --- | ---: | ---: |
| runtime112 | +2.94pt | -0.0608 |
| source136 | +5.69pt | -0.1006 |
| residual200 | +3.89pt | -0.0134 |
| residual17 | +11.76pt | -0.1348 |
| broad68 | +3.28pt | -0.0782 |
| hardcase31 | +17.24pt | -0.2449 |
| finalgate37 | +14.81pt | -0.5791 |
| final-selection90 | +9.21pt | -0.0861 |
| remaining85 | +5.63pt | -0.0416 |

Runtime implementation:

- policy: `remain85_current_margin_le_0664`
- implemented in `ai/tutor/hybrid_t1t2.py`
- exposed in `ai/tutor/evaluate_hybrid_refinement_teacher.py`

Runtime results were not good enough:

| set | current/strict reference | remain85 selector gate | result |
| --- | ---: | ---: | --- |
| runtime112 | strict k=10: 48.21% / 0.7163 | 47.32% / 0.7985 | worse than strict, roughly current Top1 |
| broad68 | strict k=10: 54.41% / 0.6671 | 50.00% / 0.9853 | bad regression |
| remaining85 | strict k=10: 0.00% / 1.4817 | 7.06% / 1.4076 | improves hard pack, but not enough |

Result:

- Do not promote the remaining85 selector or the `remain85_current_margin_le_0664` policy.
- This confirms selector-row replay is not sufficient once runtime override/rescue effects are included.
- The best current runtime candidate remains strict gate with `sync_exact_k=10`.
- Next selector training should optimize directly against runtime outputs or include override-gate interaction features, rather than only replaying selector rows.

### Runtime-output gate search

Extended `ai/tutor/evaluate_t1_runtime_rule_grid.py` to expose numeric `t1_arbitration_details` fields, so runtime rule searches can use actual arbitration features rather than only override features.

Compared completed runtime outputs:

- current: strict gate with `sync_exact_k=10`
- challenger: remaining85 selector gate
- guard sets: runtime112, broad68
- target set: remaining85
- output: `ai/data/hybrid_t1t2_active_20260531/t1_runtime_rule_grid_strict_vs_remain85_gate_guard_runtime_broad_target_remaining85_20260602.json`

Baseline runtime comparison:

| set | strict gate | remaining85 selector gate | challenger result |
| --- | ---: | ---: | --- |
| runtime112 | 48.21% / 0.7163 | 47.32% / 0.7985 | worse |
| broad68 | 54.41% / 0.6671 | 50.00% / 0.9853 | much worse |
| remaining85 | 0.00% / 1.4817 | 7.06% / 1.4076 | better but small |

Best safe runtime rule found:

- switch only when `t1_predicted_fl_delta_delta <= -0.06118202209472656` and actions differ
- runtime112: Top1 +1.79pt, regret -0.0248
- broad68: no switches, no regression
- remaining85: Top1 +2.35pt, regret -0.0327

Result:

- Do not promote this rule.
- The gain is too small, and the rule compares two completed runtime policies; implementing it synchronously would require evaluating multiple arbitration paths or adding a second arbitration selector path.
- The useful conclusion is negative but important: remaining85 retraining is not the next lever. The current best remains strict gate with `sync_exact_k=10`.
- Next practical lever is latency control: keep strict gate quality while reducing occasional 5s wall-clock outliers.

### T1 chunked refinement latency control

Implemented optional chunked T1 refinement in `ai/tutor/hybrid_t1t2.py`:

- new config/CLI: `t1_refinement_first_batch_k`
- new config/CLI: `t1_refinement_tail_min_remaining_ms`
- default remains unchanged: `first_batch_k=0` keeps the previous single-batch behavior
- chunked mode evaluates the first N T1 sync candidates first, then evaluates the tail only if enough wall-clock budget remains

Tested strict gate with:

- `sync_exact_k=10`
- `t1_refinement_first_batch_k=9`
- `t1_refinement_tail_min_remaining_ms=1200`
- same selector/arbitration/gate config as the prior strict gate runs

Results:

| set | strict k10 Top1/regret/p95/max/violations | chunk9 tail1200 Top1/regret/p95/max/violations | result |
| --- | ---: | ---: | --- |
| source136 | 69.12% / 0.2308 / 4701ms / 5511ms / 2 | 69.12% / 0.2308 / 4375ms / 5037ms / 0 | same quality, better tail latency |
| runtime112 | 48.21% / 0.7163 / 4272ms / 5035ms / 0 | 48.21% / 0.6419 / 4330ms / 5025ms / 0 | same Top1, lower regret, p95 slightly worse |
| broad68 | 54.41% / 0.6671 / 4109ms / 5026ms / 0 | 54.41% / 0.6671 / 4542ms / 5019ms / 0 | same quality, slower p95 |

Interpretation:

- This is a useful latency-safety option, not a pure speedup.
- It preserves strict k10 quality on the tested sets and removes the source136 5s violations.
- It adds a second process launch for positions where the tail is evaluated, so average/p95 can worsen on already-safe sets.
- Do not globally promote it as the default yet. Use it where hard 5s wall-clock safety matters, and next test a smaller tail threshold/grid before deciding.

### T1 hard 5s profile

Added one more T1 timeout-control knob:

- new config/CLI: `t1_refinement_timeout_headroom_ms`
- it subtracts a fixed safety margin from the subprocess timeout so Python-side output parsing does not push wall-clock time past the 5s budget
- default remains unchanged: `0`

Tested a stricter profile:

- `sync_exact_k=10`
- `t1_refinement_first_batch_k=8`
- `t1_refinement_tail_min_remaining_ms=1500`
- `t1_refinement_timeout_headroom_ms=150`

Results:

| set | strict k10 Top1/regret/p95/max/actual>5s | hard5 profile Top1/regret/p95/max/actual>5s | exact count profile |
| --- | ---: | ---: | --- |
| source136 | 69.12% / 0.2308 / 4701ms / 5511ms / 5 | 69.12% / 0.2435 / 4340ms / 4882ms / 0 | 117x10, 17x8, 2x0 |
| runtime112 | 48.21% / 0.7163 / 4272ms / 5035ms / 3 | 48.21% / 0.6832 / 4283ms / 4868ms / 0 | 104x10, 7x8, 1x0 |
| broad68 | 54.41% / 0.6671 / 4109ms / 5026ms / 1 | 54.41% / 0.6543 / 4291ms / 4865ms / 0 | 57x10, 10x8, 1x0 |

Compared with `chunk9/tail1200/headroom150` on source136:

- `chunk9` stayed under 5s, but Top1 dropped to 68.38% and regret worsened to 0.2747.
- `chunk8/tail1500/headroom150` kept Top1 at 69.12% and had lower regret.

Recommendation:

- Keep the previous strict k10 profile as the quality reference.
- Use `chunk8/tail1500/headroom150` when the product path must respect a hard 5s wall-clock limit.
- The cost is that some positions only refine 8 of 10 sync candidates, so this is not the final strength solution. The next strength lever is still more T1/T2 hard-case teacher data plus final-selector training.

### Hard-5s selector retraining attempt

Trained a new T1 final/arbitration selector on hard-5s runtime rows plus existing hard sets:

- output: `ai/data/hybrid_t1t2_active_20260531/t1_final_selector_hard5_src_runtime_broad_resid_finalsel_hardcase_listwise_margin025_heldout15_e700_20260602.json`
- training rows:
  - `eval_t1_source136_strict_gate_chunk8_tail1500_headroom150_20260602/selector_rows.jsonl`
  - `eval_t1_runtime112_strict_gate_chunk8_tail1500_headroom150_20260602/selector_rows.jsonl`
  - `eval_t1_broad68_strict_gate_chunk8_tail1500_headroom150_20260602/selector_rows.jsonl`
  - `eval_t1_residual200_mixed_arbitration_strict_gate_20260602/selector_rows.jsonl`
  - `eval_t1_final_selection90_mixed_arbitration_strict_gate_20260602/selector_rows.jsonl`
  - `eval_t1_hardcase31_mixed_arbitration_strict_gate_20260602/selector_rows.jsonl`
- train groups after margin: 487
- heldout groups: 73
- heldout selector Top1/regret: 63.01% / 0.3853

Runtime evaluation used the hard-5s profile:

- `sync_exact_k=10`
- `t1_refinement_first_batch_k=8`
- `t1_refinement_tail_min_remaining_ms=1500`
- `t1_refinement_timeout_headroom_ms=150`
- old current final selector unchanged
- arbitration selector changed to the new hard5-mix selector
- arbitration policy unchanged: `mixed_kk_delta_le_0043_model_delta_le_1111_conflict_ge_m0483`

Direct replacement results:

| set | hard5 current Top1/regret/max | hard5 new-arb Top1/regret/max | result |
| --- | ---: | ---: | --- |
| source136 | 69.12% / 0.2435 / 4882ms | 71.32% / 0.2279 / 4902ms | better |
| runtime112 | 48.21% / 0.6832 / 4868ms | 50.89% / 0.6778 / 4877ms | better |
| broad68 | 54.41% / 0.6543 / 4865ms | 52.94% / 0.6541 / 4884ms | Top1 regression |

Conclusion:

- Do not globally replace the hard5 arbitration selector with this new one yet.
- The new selector improves source136 and runtime112, but broad68 Top1 drops by 1.47pt.
- This is still progress: hard5-mix training found useful corrections, but it needs a safe gate before promotion.

Runtime-output gate search:

- compared current hard5 vs new-arb hard5 runtime outputs
- guard sets: source136, runtime112, broad68
- target sets: source136, runtime112
- output: `ai/data/hybrid_t1t2_active_20260531/t1_runtime_rule_grid_hard5_current_vs_newarb_guard3_target_src_runtime_20260602.json`

Best safe completed-output rule:

- switch only when:
  - `t1_predicted_kk_delta_delta <= -1.5966406863299198e-07`
  - `t1_premium_fl_delta_sum_delta <= 0.0`

Rule replay result:

| set | switched | Top1 | regret | delta vs current |
| --- | ---: | ---: | ---: | --- |
| source136 | 4/136 | 70.59% | 0.2353 | +1.47pt, -0.0083 |
| runtime112 | 5/112 | 52.68% | 0.6197 | +4.46pt, -0.0635 |
| broad68 | 0/68 | 54.41% | 0.6543 | no change |

Do not implement this rule directly yet:

- It compares two completed runtime policies, so a literal implementation would require a second arbitration path or duplicate policy evaluation.
- The next step should be to derive a single-pass gate from the same cases using features available inside `_t1_arbitration_candidate`, or extend the runtime to compute both old/new arbitration scores cheaply without rerunning refinement.

### Hard-5s single-pass dual arbitration gate

Implemented the cheap version of the completed-output gate in the runtime path:

- primary selector: `t1_final_selector_src136_resid200_finalsel90_listwise_margin025_heldout15_e500_20260602.json`
- challenger selector: `t1_final_selector_hard5_src_runtime_broad_resid_finalsel_hardcase_listwise_margin025_heldout15_e700_20260602.json`
- primary policy: `mixed_kk_delta_le_0043_model_delta_le_1111_conflict_ge_m0483`
- dual gate policy: `new_if_kk_and_premium_delta_nonpositive`

Implementation details:

- `_t1_arbitration_candidate` now accepts `score_key`, so primary and challenger selector scores can coexist on the same refined candidates.
- `_t1_dual_arbitration_candidate` compares the primary arbitration result against the challenger arbitration result without rerunning Rust, recursive refinement, or shortlist construction.
- The gate accepts the challenger only when:
  - `predicted_kk_delta_delta <= -1.5966406863299198e-07`
  - `premium_fl_delta_sum_delta <= 0.0`
- `HybridConfig`, `hybrid_t1t2.py`, and `evaluate_hybrid_refinement_teacher.py` now expose:
  - `t1_arbitration_challenger_selector`
  - `t1_arbitration_dual_gate_policy`

Hard-5s runtime evaluation:

| set | current Top1/regret/max | dual-gate Top1/regret/p95/max | delta | time violations |
| --- | ---: | ---: | ---: | ---: |
| source136 | 69.12% / 0.2435 / 4882ms | 70.59% / 0.2353 / 4319ms / 4879ms | +1.47pt, -0.0083 | 0 |
| runtime112 | 48.21% / 0.6832 / 4868ms | 52.68% / 0.6197 / 4440ms / 4878ms | +4.46pt, -0.0635 | 0 |
| broad68 | 54.41% / 0.6543 / 4865ms | 54.41% / 0.6667 / 4359ms / 4872ms | +0.00pt, +0.0124 | 0 |
| residual200 | 64.50% / 0.3687 / 5028ms | 65.50% / 0.3097 / 4176ms / 4883ms | +1.00pt, -0.0590 | 0 |
| final-selection90 | 13.33% / 1.3152 / 5023ms | 18.89% / 1.1207 / 4408ms / 4546ms | +5.56pt, -0.1945 | 0 |
| hardcase31 | 12.90% / 0.8993 / 4394ms | 16.13% / 0.8722 / 4289ms / 4410ms | +3.23pt, -0.0271 | 0 |

Interpretation:

- This is promotable as the current hard-5s T1 candidate because it improves Top1 on 5 of 6 evaluated sets and keeps every measured max below 5 seconds.
- The broad68 Top1 regression from direct challenger replacement is avoided, but broad68 average regret worsens slightly. Keep broad68 in every promotion guard.
- The weak absolute numbers on final-selection90 and hardcase31 mean the model is stronger, not solved.
- The next strength work should not be another hand-written rule unless a new clear failure split appears. Collect more MC1000+ T1 hard-case rows from dual-gate misses, mix them with normal rows, then retrain the final/arbitration selector.

Current operational recommendation:

- Use the dual-gate hard-5s profile for T1 product-path experiments.
- Keep `sync_exact_k=10`, `t1_refinement_first_batch_k=8`, `t1_refinement_tail_min_remaining_ms=1500`, and `t1_refinement_timeout_headroom_ms=150` as the wall-clock-safe profile.
- Continue treating Top1 as the target and Top15/20 as an internal refinement pool, not as success.

### Dual-gate miss target extraction

Converted remaining dual-gate final Top1 misses into active-learning targets:

- output directory: `ai/data/hybrid_t1t2_active_20260531/active_targets_dualgate_misses_20260602/`
- per-set target files:
  - `source136_targets.jsonl`: 19
  - `runtime112_targets.jsonl`: 53
  - `broad68_targets.jsonl`: 14
  - `residual200_targets.jsonl`: 34
  - `final_selection90_targets.jsonl`: 73
  - `hardcase31_targets.jsonl`: 26
- filter: T1 only, `min_teacher_margin=0.25`, `min_regret=0.1`, `refinement_tag=t1_hard5_dualgate`
- merged output: `all_dualgate_miss_targets.jsonl`
- merged summary: 219 read, 80 written, 139 duplicates

Interpretation:

- The high duplicate count is useful: the same failure patterns are recurring across source, final-selection, and hardcase sets.
- These 80 deduped targets are the next compact T1 hard-case seed set.
- Next training should mix this file with normal source/runtime selector rows; training only on these 80 hard cases would likely over-correct broad positions.

### Dual-gate selector-row retraining smoke

Trained a listwise selector on the six dual-gate selector-row sets:

- output: `ai/data/hybrid_t1t2_active_20260531/t1_final_selector_hard5_dualgate_all6_listwise_margin025_heldout15_e700_20260602.json`
- train inputs:
  - source136 dual-gate selector rows
  - runtime112 dual-gate selector rows
  - broad68 dual-gate selector rows
  - residual200 dual-gate selector rows
  - final-selection90 dual-gate selector rows
  - hardcase31 dual-gate selector rows
- groups after margin: 490
- train/heldout split: 416 / 74
- heldout selector Top1/regret: 62.16% / 0.6850

Selector-row replay against `refined_score`:

| set | selector Top1/regret | refined_score Top1/regret | verdict |
| --- | ---: | ---: | --- |
| source136 | 73.79% / 0.4433 | 74.76% / 0.3590 | worse |
| runtime112 | 49.55% / 0.8036 | 40.54% / 0.8715 | better |
| broad68 | 64.44% / 0.5474 | 68.89% / 0.2817 | much worse |
| residual200 | 72.73% / 0.5663 | 72.73% / 0.4183 | worse regret |
| final-selection90 | 34.44% / 1.0294 | 16.67% / 1.1891 | better |
| hardcase31 | 35.48% / 0.7379 | 6.45% / 1.2619 | better |

Conclusion:

- Do not promote this selector directly.
- It improves the intended hard sets but over-corrects broad/source/residual rows, exactly the failure mode we expected from a small hard-case-heavy set.
- The right next move is stronger data, not another global selector replacement: run MC1000+ generation from `all_dualgate_miss_targets.jsonl`, then retrain with those new rows plus normal-position rows.

### Local MC1000 relabeling of dual-gate misses

Generated MC1000 labels locally for all 80 deduped dual-gate miss targets:

- input: `active_targets_dualgate_misses_20260602/all_dualgate_miss_targets.jsonl`
- output directory: `ai/data/hybrid_t1t2_active_20260531/teacher_dualgate_misses_mc1000_local80_20260602/`
- merged output: `merged.jsonl`
- records: 80/80 written
- eval mode: `mc1000`
- chunking: 40 chunks of 2 records
- local runtime:
  - 2-record batch probe: 133.8s
  - 2-record non-batch/2-worker probe: 137.3s, not faster
  - 10-record local slice: 550.8s
  - full remaining run: 4059.7s after reusing the first 10 records

Runtime evaluation of the current hard-5s dual-gate policy on these 80 new MC1000 labels:

- output: `eval_t1_dualgate_local80_mc1000_hard5_dualgate_20260602`

| metric | value |
| --- | ---: |
| decisions | 80 |
| model Top1 | 22.50% |
| final Top1 | 0.00% |
| model avg regret | 2.1778 |
| final avg regret | 1.4500 |
| pool recall | 100.00% |
| sync recall | 85.00% |
| overrides | 53 |
| improved overrides | 28 |
| worse overrides | 24 |
| p95 / max | 4319ms / 4868ms |
| time violations | 0 |

Interpretation:

- This is a real final-selection/override failure: the best action is in the pool for every target, but the current runtime final selector drives Top1 to zero on this active-hardcase distribution.
- The runtime still lowers average regret versus raw model Top1, but it destroys the 18 model Top1 hits that were already correct.
- This is not primarily a candidate-generation problem for T1 on these targets; it is a final choice/gating problem.

Selector-row analysis on the same 80 targets:

| policy | Top1 | avg regret | max regret |
| --- | ---: | ---: | ---: |
| model_top1 | 22.50% | 2.1778 | 13.1449 |
| runtime_best | 0.00% | 1.4500 | 8.8366 |
| refined_score_top1 | 11.25% | 1.2270 | 6.9669 |
| refined_plus_0.5x_model | 26.25% | 1.0395 | 6.9669 |
| refined_plus_1x_model | 27.50% | 1.0385 | 6.9669 |

Runtime check for forced `refined_plus_model`:

- output: `eval_t1_dualgate_local80_mc1000_refined_plus_model1_force_20260602`
- config difference: `t1_selection_policy=refined_plus_model`, `t1_selection_model_weight=1.0`, no arbitration, no override gate, no sparse rescue

| metric | current hard5 dual-gate | forced refined_plus_model |
| --- | ---: | ---: |
| Top1 | 0.00% | 27.50% |
| avg regret | 1.4500 | 1.0146 |
| max regret | 8.8366 | 6.9669 |
| p95 / max | 4319ms / 4868ms | 4269ms / 4621ms |
| time violations | 0 | 0 |

Mixed-selector retraining with local80:

- output: `t1_final_selector_hard5_dualgate_local80_mix7_listwise_margin025_heldout15_e700_20260602.json`
- train inputs: six dual-gate selector-row sets plus local80 selector rows
- groups after margin: 556
- train/heldout split: 473 / 83
- heldout selector Top1/regret: 59.04% / 0.6938

Selector-row replay of the mixed selector:

| set | selector Top1/regret | refined_score Top1/regret | verdict |
| --- | ---: | ---: | --- |
| source136 | 72.82% / 0.5342 | 74.76% / 0.3590 | worse |
| runtime112 | 51.35% / 0.8190 | 40.54% / 0.8715 | mixed, lower regret not enough |
| broad68 | 57.78% / 0.6937 | 68.89% / 0.2817 | much worse |
| residual200 | 69.23% / 0.6650 | 72.73% / 0.4183 | worse |
| final-selection90 | 34.44% / 1.0532 | 16.67% / 1.1891 | better |
| hardcase31 | 35.48% / 0.8220 | 6.45% / 1.2619 | better |
| local80 | 24.05% / 1.2527 | 11.39% / 1.2081 | Top1 better, regret worse |

Conclusion:

- Do not promote the mixed selector. It learns the hardcase correction but over-corrects normal/broad/residual distributions.
- Do not globally replace current hard-5s dual-gate with `refined_plus_model`; current runtime remains better on source136, runtime112, and residual200 selector-row replay.
- The next implementable improvement is a local80-aware gate between current runtime and `refined_plus_model`, or a gate that preserves model Top1 when the override pattern resembles local80. The gate must be validated on source136/runtime112/broad68/residual200/final-selection90/hardcase31/local80 before runtime promotion.

### Runtime-best vs refined-plus-model selector-row gate search

Added an offline policy-gate search:

- script: `ai/tutor/evaluate_t1_selector_row_policy_gate.py`
- current policy: saved `runtime_best` from each `selector_rows.jsonl`
- challenger policy: `refined_plus_model` with model weight `1.0`
- evaluated sets:
  - source136
  - runtime112
  - broad68
  - residual200
  - final-selection90
  - hardcase31
  - local80 MC1000 dual-gate misses

Strict guard search:

- output: `t1_policy_gate_runtime_vs_refinedplus_targets4_20260602.json`
- guard sets: source136/runtime112/residual200
- target sets: local80/broad68/final-selection90/hardcase31
- guard tolerance: no Top1 or regret degradation
- best rule: switch to `refined_plus_model` when `model_score_delta <= -5.177628517150879`

Strict result:

| set | switches | Top1 delta | regret delta |
| --- | ---: | ---: | ---: |
| source136 | 0 | 0.0000 | 0.0000 |
| runtime112 | 0 | 0.0000 | 0.0000 |
| broad68 | 1 | +0.0147 | -0.1300 |
| residual200 | 0 | 0.0000 | 0.0000 |
| final-selection90 | 1 | +0.0111 | -0.0982 |
| hardcase31 | 0 | 0.0000 | 0.0000 |
| local80 | 1 | +0.0125 | -0.1105 |

Interpretation:

- A fully safe one/two-condition gate exists, but it only switches one high-confidence case.
- This is not enough to materially improve T1 final Top1.

Mid-tolerance diagnostic search:

- output: `t1_policy_gate_runtime_vs_refinedplus_targets4_mid_20260602.json`
- guard tolerance: regret <= `0.05`, Top1 drop <= `0.02`
- best rule: switch to `refined_plus_model` when `predicted_trips_delta <= -1.1638671821856406e-06`

Mid-tolerance result:

| set | switches | Top1 delta | regret delta |
| --- | ---: | ---: | ---: |
| source136 | 13 | -0.0074 | +0.0160 |
| runtime112 | 16 | 0.0000 | +0.0398 |
| broad68 | 12 | 0.0000 | -0.0692 |
| residual200 | 23 | -0.0050 | -0.0130 |
| final-selection90 | 19 | +0.0556 | -0.0525 |
| hardcase31 | 6 | +0.0323 | +0.1353 |
| local80 | 17 | +0.1250 | -0.1983 |

Interpretation:

- The mid-tolerance gate meaningfully improves local80 and final-selection90.
- It still degrades source136/residual200 Top1 slightly and worsens hardcase31 regret.
- Do not promote this rule directly into runtime.
- The next step should be full runtime validation only for candidate gates that meet stricter guard behavior, or use these rows as additional final-selection training data instead of a hand rule.

### Guard-weighted final-selector retraining

Tried feeding local80 miss rows back into final-selector training.

Pairwise selector attempts:

- `t1_final_selector_pairwise_local80_finalmiss_margin025_e900_20260602.json`
- `t1_final_selector_pairwise_mix7_finalmiss_margin025_e900_20260602.json`

Result:

- Both pairwise selectors failed badly.
- Heldout Top1 was `0.00%`, with heldout regret around `8`.
- Per-set evaluation also collapsed to near-zero Top1 on normal sets.

Interpretation:

- Pairwise training only on teacher-vs-runtime misses does not constrain scores for the rest of the candidate pool.
- It can learn the pair correction while assigning very high scores to unrelated bad candidates.
- Do not use this pairwise selector path unless the trainer is changed to include all-candidate negatives or listwise regularization.

Guard-weighted listwise selectors:

- `t1_final_selector_listwise_mix7_guard2_margin025_e300_20260602.json`
- `t1_final_selector_listwise_mix7_guard2_margin025_regret1_e300_20260602.json`

Training mix:

- source136/runtime112/broad68/residual200 repeated twice
- final-selection90/hardcase31/local80 included once
- margin >= `0.25`
- 300 epochs

Selector-row result for the regret-weighted guard2 selector:

| set | Top1 | avg regret |
| --- | ---: | ---: |
| source136 | 77.67% | 0.2431 |
| runtime112 | 51.35% | 0.6551 |
| broad68 | 68.89% | 0.3084 |
| residual200 | 78.32% | 0.3340 |
| final-selection90 | 28.89% | 0.8913 |
| hardcase31 | 16.13% | 0.8951 |
| local80 | 17.72% | 1.1024 |

This selector-row profile is better than the earlier mixed selector on regret, but still not strictly safe versus the current runtime on runtime112/residual/hardcase.

Runtime validation:

1. With existing arbitration/override stack:

- output: `eval_t1_local80_guard2regret_selector_hard5_dualgate_20260602`
- local80 Top1: `0.00% -> 5.00%`
- local80 avg regret: `1.4500 -> 1.3921`
- p95/max: `4231ms / 4525ms`
- time violations: `0`

2. Pure guard2 regret selector, with arbitration/rescue removed and override forced through:

- local80 output: `eval_t1_local80_guard2regret_selector_noarb_force_20260602`
- runtime112 output: `eval_t1_runtime112_guard2regret_selector_noarb_force_20260602`

| set | current Top1/regret | guard2 no-arb Top1/regret | verdict |
| --- | ---: | ---: | --- |
| local80 | 0.00% / 1.4500 | 18.75% / 1.1449 | improves |
| runtime112 | 52.68% / 0.6197 | 51.79% / 0.6815 | worsens |

Latency stayed under 5 seconds:

| set | p95 | max | violations |
| --- | ---: | ---: | ---: |
| local80 | 4291ms | 4463ms | 0 |
| runtime112 | 4358ms | 4873ms | 0 |

Conclusion:

- local80 failures can be learned back into a listwise final selector.
- The existing arbitration/override stack suppresses much of that improvement.
- Removing arbitration/override improves local80 significantly, but it weakens runtime112 guard performance.
- Do not promote the new selector or no-arb path globally yet.
- The next engineering target is not more pairwise training. It is a new final-selection stack:
  - keep the guard2 regret selector as a challenger
  - add an arbitration gate that switches to it only when the current stack is in the local80-like failure mode
  - validate that gate in runtime on source136/runtime112/residual200 before promotion

### Final challenger gate runtime experiment

Implemented an experimental post-stack challenger path:

- config selector: `t1_final_challenger_selector`
- config policy: `t1_final_challenger_gate_policy`
- runtime fields:
  - `t1_final_challenger_details`
  - `t1_final_challenger_applied`
- code path: after normal T1 final selector, arbitration, override gate, and bottom-sparse rescue
- challenger selector: `t1_final_selector_listwise_mix7_guard2_margin025_regret1_e300_20260602.json`

The first policy was derived from current-vs-guard2 runtime result rows:

- `guard2_bust_delta_ge_m066_rank_gap_delta_ge_m1`
- condition:
  - challenger selected candidate bust is not much safer/worse than current: `best_predicted_bust_delta >= -0.06600075960159302`
  - challenger model-rank gap does not move down by more than one: `model_rank_gap_delta >= -1.0`

Runtime validation:

| set | current Top1/regret | challenger-gate Top1/regret | applied | p95 / max | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| local80 | 0.00% / 1.4500 | 12.50% / 1.1740 | 15 | 4044ms / 4512ms | improves |
| runtime112 | 52.68% / 0.6197 | 54.46% / 0.6372 | 10 | 4319ms / 4880ms | Top1 up, regret worse |
| source136 | 70.59% / 0.2353 | 70.59% / 0.3065 | 20 | 4405ms / 4884ms | regret worse |

Conclusion:

- The final challenger path works mechanically and stays under 5 seconds.
- It materially improves local80 hard failures.
- It is not safe enough for global runtime because source136 regret worsens by about `+0.0712`.
- Do not promote this policy.

A stricter direct policy was also tested:

- `guard2_bust_rank_refined_delta_ge_m0778`
- extra condition: `refined_score_delta >= -0.7777777777777772`

Runtime validation:

| set | current Top1/regret | refined-gate Top1/regret | applied | p95 / max | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| local80 | 0.00% / 1.4500 | 10.00% / 1.1475 | 10 | 4197ms / 4793ms | improves |
| runtime112 | 52.68% / 0.6197 | 33.04% / 2.0996 | 2 | 4962ms / 4996ms | fails |
| source136 | 70.59% / 0.2353 | 52.21% / 1.3227 | 7 | 4956ms / 4989ms | fails |

Interpretation:

- The stricter direct policy did not match the offline two-stage replay behavior.
- Some severe misses came from no-refine fallback during time-budget timeouts, and the direct policy also accepted bad normal-distribution switches.
- Do not use this policy.

Current state:

- Best safe production candidate remains the previous hard-5s dual-gate stack.
- Best hardcase improvement candidate is the final challenger path, but it needs a learned gate or broader guard validation before promotion.
- Next useful work:
  - generate current-vs-guard2 no-arb runtime rows for more guard sets, especially residual200 and broad68
  - train a logistic gate on runtime-level features, not selector-row-only features
  - optimize/refactor T1 recursive refinement to reduce timeout-driven no-refine fallback before adding more final-selection complexity
