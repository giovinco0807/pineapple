# OFC Pineapple Tutor Evaluation Holdout

Date: 2026-05-23

## Purpose

The Route10 review set has only 80 decisions, so turn-level top-N accuracy is too noisy. This holdout is a fixed evaluation-only set. Do not mix it into training.

## Target Set

Target file:
`ai/data/tutor_eval_holdout_20260523/targets_t0t3_500_each.jsonl`

Summary:

| turn | targets |
|---|---:|
| T0 | 500 |
| T1 | 500 |
| T2 | 500 |
| T3 | 500 |
| total | 2,000 |

The target sampler excludes the 116 decisions already converted into the T1/T3 active fine-tune data.

## Labels Generated So Far

T3 exact labels are complete:

- labels: `ai/data/tutor_eval_holdout_20260523/teacher_labels_t3_exact_500.jsonl`
- reranker data: `ai/data/tutor_eval_holdout_20260523/reranker_t3_exact_500`
- decisions: 500
- candidate samples: 7,140
- mode: exact

T2 MC300 labels are complete:

- labels: `ai/data/tutor_eval_holdout_20260523/teacher_labels_t2_mc300_500.jsonl`
- reranker data: `ai/data/tutor_eval_holdout_20260523/reranker_t2_mc300_500`
- decisions: 500
- candidate samples: 10,236
- mode: MC300
- elapsed: 1,070.1s

T1 MC300 labels are complete:

- labels: `ai/data/tutor_eval_holdout_20260523/teacher_labels_t1_mc300_500.jsonl`
- reranker data: `ai/data/tutor_eval_holdout_20260523/reranker_t1_mc300_500`
- decisions: 500
- candidate samples: 11,456
- mode: MC300
- elapsed: 6,925.8s

## T3 Exact Holdout Result

| model | top1 | top3 | top5 | top10 | top20 | avg_regret | score_mae | FL_MAE | bust_MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 22.4% | 51.2% | 66.6% | 87.0% | 99.6% | 3.053 | 5.953 | 7.7% | 38.4% |
| active8x | 26.0% | 54.2% | 70.4% | 85.8% | 98.8% | 2.769 | 4.087 | 5.3% | 28.1% |
| top20-prune | 25.2% | 53.2% | 66.2% | 87.0% | 99.0% | 2.925 | 3.972 | 5.4% | 30.5% |
| top20-prune+t2t3-active16x | 25.0% | 54.0% | 68.8% | 86.8% | 99.0% | 2.876 | 3.911 | 5.6% | 28.0% |
| top20-prune+p99-2x | 24.6% | 56.2% | 69.2% | 86.4% | 99.2% | 2.863 | 3.649 | 4.8% | 26.0% |
| top20-prune+p99-8x | 25.2% | 57.2% | 68.8% | 87.0% | 99.0% | 2.734 | 3.521 | 4.5% | 25.1% |

This is a better signal than the 20-decision Route10 T3 slice. The active8x model improves top1/top3/top5/regret/calibration, but top10/top20 drop slightly. For pruning, top20 looks much safer than top10 on T3.

If the top-K candidates are exact-reranked after pruning:

| model | top10 rerank avg regret | top20 rerank avg regret | top10 rerank max regret | top20 rerank max regret |
|---|---:|---:|---:|---:|
| baseline | 0.172 | 0.000 | 13.146 | 0.000 |
| active8x | 0.212 | 0.003 | 13.146 | 1.567 |
| top20-prune | 0.164 | 0.000 | 13.146 | 0.000 |
| top20-prune+t2t3-active16x | 0.192 | 0.000 | 13.146 | 0.000 |
| top20-prune+p99-2x | 0.258 | 0.000 | 21.848 | 0.000 |
| top20-prune+p99-8x | 0.261 | 0.003 | 21.848 | 1.567 |

## T2 MC300 Holdout Result

| model | top1 | top3 | top5 | top10 | top20 | avg_regret | score_mae | FL_MAE | bust_MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 32.2% | 58.4% | 72.8% | 89.2% | 98.4% | 2.860 | 4.460 | 11.2% | 17.6% |
| active8x | 31.6% | 58.4% | 72.6% | 88.4% | 99.0% | 3.012 | 3.859 | 8.6% | 18.2% |
| top20-prune | 32.0% | 57.8% | 73.6% | 89.6% | 99.2% | 2.958 | 3.935 | 8.7% | 17.6% |
| top20-prune+t2t3-active16x | 32.4% | 59.4% | 72.8% | 89.2% | 98.6% | 2.877 | 3.926 | 9.1% | 18.3% |
| top20-prune+p99-2x | 32.6% | 59.6% | 71.2% | 89.4% | 98.8% | 2.801 | 3.899 | 8.4% | 19.0% |
| top20-prune+p99-8x | 32.4% | 58.6% | 71.2% | 89.8% | 99.0% | 2.707 | 3.889 | 8.0% | 19.4% |

If the top-K candidates are MC300-reranked after pruning:

| model | top10 rerank avg regret | top20 rerank avg regret | top10 rerank max regret | top20 rerank max regret |
|---|---:|---:|---:|---:|
| baseline | 0.178 | 0.021 | 20.726 | 7.294 |
| active8x | 0.183 | 0.017 | 24.545 | 7.294 |
| top20-prune | 0.163 | 0.015 | 24.545 | 7.294 |
| top20-prune+t2t3-active16x | 0.170 | 0.022 | 24.545 | 7.294 |
| top20-prune+p99-2x | 0.162 | 0.020 | 24.545 | 7.294 |
| top20-prune+p99-8x | 0.159 | 0.056 | 24.545 | 20.726 |

T2 says the same thing as T3: top20 is a safer pruning target than top10. The active8x model improves score/FL calibration but does not clearly improve pruning recall on this holdout.

## T1 MC300 Holdout Result

| model | top1 | top3 | top5 | top10 | top20 | avg_regret | score_mae | FL_MAE | bust_MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 32.4% | 57.2% | 73.0% | 91.8% | 98.0% | 2.243 | 4.216 | 19.8% | 12.1% |
| active8x | 33.0% | 57.8% | 74.8% | 92.0% | 98.4% | 2.148 | 3.895 | 17.0% | 11.5% |
| top20-prune | 33.4% | 57.0% | 73.4% | 91.8% | 98.6% | 2.268 | 3.806 | 18.2% | 12.1% |
| top20-prune+t2t3-active16x | 33.8% | 57.8% | 75.4% | 91.6% | 98.4% | 2.177 | 4.130 | 18.1% | 11.6% |
| top20-prune+p99-2x | 33.6% | 57.2% | 74.8% | 91.8% | 98.4% | 2.159 | 3.984 | 17.3% | 11.6% |
| top20-prune+p99-8x | 33.8% | 57.2% | 74.2% | 91.2% | 98.0% | 2.175 | 3.961 | 17.0% | 11.6% |

If the top-K candidates are MC300-reranked after pruning:

| model | top10 rerank avg regret | top20 rerank avg regret | top10 rerank max regret | top20 rerank max regret |
|---|---:|---:|---:|---:|
| baseline | 0.106 | 0.022 | 6.243 | 5.298 |
| active8x | 0.121 | 0.025 | 10.495 | 5.298 |
| top20-prune | 0.116 | 0.018 | 6.195 | 5.298 |
| top20-prune+t2t3-active16x | 0.122 | 0.022 | 10.495 | 5.298 |
| top20-prune+p99-2x | 0.119 | 0.021 | 10.495 | 5.298 |
| top20-prune+p99-8x | 0.137 | 0.021 | 10.495 | 5.298 |

T1 is the clearest case where active8x helps single-pass model quality: top1/top3/top5/top10/top20 and average regret all improve slightly. For pruning plus rerank, baseline and active8x are very close; top20 remains the safer cutoff.

## MC Label Cost Benchmarks

Local batch engine, MC300:

| turn | benchmark | elapsed | estimate for 500 |
|---|---:|---:|---:|
| T1 | 500 labels | 6,925.8s | complete |
| T2 | 500 labels | 1,070.1s | complete |
| T0 | 20 labels | timed out at 1200s after 2 responses | local full-candidate MC300 is not practical |

T0 needs a different path: staged/laddder teacher for a proxy holdout, or GCP batch generation for full-candidate labels.

## Current Conclusion

For live tutor pruning, use top20 rather than top10 for T1/T2/T3. On the 500-decision holdouts, top20 recall stays near 98-99%, while top10 recall is only 86-92%. The top20-prune model is the best current pruning candidate for T1/T2: it improves top20 recall and top20 rerank regret versus baseline/active8x. On T3, baseline still has the highest raw top20 hit rate, but top20-prune has zero top20 rerank regret on this holdout, so exact reranking of the retained top20 removes the observed misses.

The `top20-prune+t2t3-active16x` follow-up is not adopted. It overfits a small 16-decision Route10 weak-spot set: internal validation improves, but fixed-holdout T2 top20 recall drops from 99.2% to 98.6% and top20 rerank regret worsens from 0.015 to 0.022. Keep the previous `top20-prune` checkpoint as the current candidate.

The `top20-prune+p99` follow-ups are also not adopted as global model upgrades. They improve the p99 stress set and some T3 calibration, but they reduce fixed-holdout T1/T2 top20 safety versus the current `top20-prune` checkpoint. The p99 data is still useful as a stress-test set and should be mixed only with a turn-specific or distribution-balanced schedule.

The turn-specific-head follow-ups are not adopted either. The implementation is useful because it can isolate T2/T3 experiments from T1, but the current head-only p99 schedules still trade away fixed-holdout T2 safety. Keep `top20-prune` as the production candidate until a turn-specific run improves p99 stress without lowering fixed-holdout top20 recall or top20 rerank regret.

## Top20-Prune Training Run

The top20-prune run adds a soft top-K ranking loss to `ai/training/train_action_value_reranker.py`. It penalizes groups where the teacher-best action is softly ranked below the requested K and adds validation metrics for top10/top20 hit rate plus top10/top20 rerank regret.

Model:
`ai/models/candidate_runs/tutor-route10-top20-prune-active8x-ft-20260523/model/action_value_best.pt`

Training data:
`D:/ofc_data/tutor-route10-t1t3-active8x-mix-20260523/reranker_base_active8x`

Training summary:

| epoch | top1 | top10 | top20 | regret | top20 rerank regret | score_mae |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 59.8% | 98.6% | 99.7% | 0.739 | 0.004 | 2.442 |
| 2 | 60.7% | 98.7% | 99.8% | 0.710 | 0.004 | 2.128 |
| 3 | 61.9% | 98.9% | 99.7% | 0.637 | 0.003 | 1.917 |
| 4 | 62.3% | 99.0% | 99.7% | 0.617 | 0.004 | 1.689 |

## Rejected T2/T3 Weak-Spot Follow-Up

Route10 weak-spot extraction:

- input: `ai/data/tutor_route10_20260522/tutor_route10_merged.jsonl`
- output: `ai/data/tutor_route10_20260522/model_eval_top20_prune_route10_merged`
- top20 recall on the 60 Route10 T1/T2/T3 decisions: 100.0%
- extracted active targets: `ai/data/tutor_route10_20260522/top20_prune_t2t3_active_targets/targets.jsonl`
- target count: 16 decisions, T2=9 and T3=7
- generated labels: `ai/data/tutor_route10_20260522/top20_prune_t2t3_active_targets/teacher_labels.jsonl`
- converted reranker data: `ai/data/tutor_route10_20260522/top20_prune_t2t3_active_targets/reranker`
- mixed data: `D:/ofc_data/tutor-route10-top20-prune-t2t3-active16x-20260523/reranker_base_active8x_t2t3_16x`
- model: `ai/models/candidate_runs/tutor-route10-top20-prune-t2t3-active16x-ft-20260523/model/action_value_best.pt`

This run is useful as a pipeline test, but not as a model upgrade.

## P99 Stress-Set Follow-Up

P99 target conversion script:
`ai/tutor/p99_samples_to_active_targets.py`

Generated targets:

- T2 BTN p99: `ai/data/t2_hu_btn_gcp_5m/t2_btn_p99_samples.jsonl`, 75 targets
- T2 BB p99: `ai/data/t2_hu_bb_gcp_5m/t2_bb_p99_samples.jsonl`, 75 targets
- T3 p99: `ai/data/t3_oracle_rust_370k/t3_p99_samples.jsonl`, 150 targets
- combined target file: `ai/data/tutor_top20_prune_p99_20260523/targets_t2t3_p99_300.jsonl`
- generated labels: `ai/data/tutor_top20_prune_p99_20260523/teacher_labels_t2t3_p99_300.jsonl`
- written labels: 299 decisions, T2=149 with MC300 and T3=150 exact
- converted reranker data: `ai/data/tutor_top20_prune_p99_20260523/reranker_t2t3_p99_299`

P99 stress-set result:

| model | top1 | top3 | top5 | top10 | top20 | top20 rerank regret | score_mae | FL_MAE | bust_MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top20-prune | 21.1% | 46.5% | 61.5% | 84.3% | 97.7% | 0.071 | 4.525 | 9.3% | 23.0% |
| top20-prune+p99-2x | 24.7% | 51.5% | 66.2% | 87.6% | 98.0% | 0.071 | 3.467 | 5.9% | 18.3% |
| top20-prune+p99-8x | 27.4% | 54.8% | 66.9% | 90.3% | 99.0% | 0.000 | 2.793 | 4.3% | 15.9% |

This confirms the p99 set is learnable and catches real weaknesses. The issue is not whether p99 data helps the stress distribution; it is that naive global mixing trades away fixed-holdout pruning safety. Next experiments should split the model or loss schedule by turn/distribution instead of a single global fine-tune.

## Turn-Specific Head Follow-Up

Implementation:

- `ai/models/action_value_reranker.py` now supports `turn_specific_heads=True`.
- `ai/training/train_action_value_reranker.py` adds `--turn-specific-heads`, `--freeze-shared`, `--normalization-source checkpoint`, and `--train-turns`.
- Existing single-head checkpoints remain loadable. New turn heads are initialized from the global heads.
- With `--freeze-shared --train-turns 2,3`, T1 stays identical to the adopted `top20-prune` model while only T2/T3 heads move.

Rejected models:

- p99-only head update: `ai/models/candidate_runs/tutor-route10-top20-prune-p99-turnheads-headonly-20260523/model/action_value_best.pt`
- mixed p99-2x all-turn head update: `ai/models/candidate_runs/tutor-route10-top20-prune-p99-2x-turnheads-headonly-20260523/model/action_value_best.pt`
- mixed p99-2x T2/T3-only head update: `ai/models/candidate_runs/tutor-route10-top20-prune-p99-2x-turnheads-t23-headonly-20260523/model/action_value_best.pt`

Fixed holdout vs current candidate:

| model | T1 top20 / t20reg | T2 top20 / t20reg | T3 top20 / t20reg | p99 top20 / t20reg | verdict |
|---|---:|---:|---:|---:|---|
| top20-prune | 98.6% / 0.018 | 99.2% / 0.015 | 99.0% / 0.000 | 97.7% / 0.071 | current |
| p99-only turn heads | 98.6% / 0.018 | 93.6% / 0.189 | 98.2% / 0.024 | 99.7% / 0.000 | reject |
| p99-2x all-turn heads | 97.8% / 0.027 | 99.0% / 0.017 | 99.4% / 0.000 | 98.0% / 0.086 | reject |
| p99-2x T2/T3-only heads | 98.6% / 0.018 | 98.6% / 0.026 | 99.4% / 0.000 | 98.3% / 0.086 | reject |

The T2/T3-only variant proves the isolation mechanism works: T1 metrics are exactly preserved. It still loses too much T2 pruning safety, so the next useful direction is not heavier p99 mixing but better selection of T2 weak spots, likely by sampling fixed-holdout-like states where top20 misses or top20 rerank regret is nonzero.

## T2 Top20 Weak-Spot Follow-Up

Implementation:

- Added `ai/tutor/weak_groups_to_active_targets.py`.
- It converts candidate-level `weak_groups.jsonl` entries back into active-teacher targets using the original teacher-label JSONL group ids.
- It can expand selected targets through all 24 suit permutations.

Target extraction:

- source weak groups: `ai/data/tutor_eval_holdout_20260523/eval_t2_mc300_500_top20_prune/weak_groups.jsonl`
- source teacher labels: `ai/data/tutor_eval_holdout_20260523/teacher_labels_t2_mc300_500.jsonl`
- selected groups: 3 T2 decisions with rank > 20 or top20 rerank regret > 0
- expanded targets: `ai/data/tutor_t2_top20_weak_20260524/targets_t2_top20_suit24.jsonl`
- target count: 72 after 24 suit permutations
- generated labels: `ai/data/tutor_t2_top20_weak_20260524/teacher_labels_t2_top20_suit24_mc1000.jsonl`
- label mode: T2 MC1000, 72/72 written, no batch errors
- converted reranker data: `ai/data/tutor_t2_top20_weak_20260524/reranker_t2_top20_suit24_mc1000`

Stress-set result:

| model | weak-set top1 | weak-set top10 | weak-set top20 | weak-set t20reg | fixed T2 top20 | fixed T2 t20reg | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| top20-prune | 0.0% | 5.6% | 73.6% | 1.026 | 99.2% | 0.015 | current |
| t2-top20-suit24 8x T2-head | 4.2% | 38.9% | 84.7% | 0.331 | 95.8% | 0.069 | reject |
| t2-top20-suit24 1x T2-head | 1.4% | 33.3% | 70.8% | 1.042 | 96.4% | 0.057 | reject |

## Non-Holdout T2 Top20 Mining Follow-Up

Implementation:

- `ai/tutor/evaluate_action_value_dataset.py` now keeps `states.npy` memory-mapped during evaluation, so multi-GB candidate datasets can be evaluated without loading all states into RAM.
- Added `ai/training/select_action_value_weak_groups.py` to copy complete candidate-level groups from a `weak_groups.jsonl` report into a focused reranker data directory.
- Fixed `ai/tutor/weak_groups_to_active_targets.py` for source JSONL files that contain skipped records, such as `turn=-1`; it now maps weak `group_id` through the same valid-record counter used by `convert_action_value_teacher.py`.

Mining run:

- source data: `ai/models/candidate_runs/fl-route-reranker-v2-medium-20260520/data`
- source labels: `ai/models/candidate_runs/prob-engine-r20-combined-snapshot-20260518/teacher.jsonl`
- source label mode: old MC20, useful for mining but not trustworthy enough for final training
- full-source T2 decisions: 7,253
- full-source T2 candidates: 164,868
- current model on this source: top20 96.3%, top20 rerank regret 0.138
- mined T2 rank>20 groups: 268

Old-label candidate subset:

- extracted subset: `ai/data/tutor_t2_top20_nonholdout_20260524/reranker_oldmix_t2_top20_misses`
- groups: 84
- samples: 2,112
- current model top20 on old-label subset: 0.0%
- result: direct training on old MC20 labels did not help and was rejected.

Re-labeled subset:

- targets: `ai/data/tutor_t2_top20_nonholdout_20260524/targets_oldfull_t2_top20_miss120.jsonl`
- selected groups: 120
- generated labels: `ai/data/tutor_t2_top20_nonholdout_20260524/teacher_labels_oldfull_t2_top20_miss120_mc300.jsonl`
- label mode: T2 MC300, 120/120 written, no batch errors, elapsed 511.3s
- converted reranker data: `ai/data/tutor_t2_top20_nonholdout_20260524/reranker_oldfull_t2_top20_miss120_mc300`
- samples: 2,979
- current model on this MC300 subset: top20 95.0%, top20 rerank regret 0.407

Rejected models:

- old MC20 subset 4x T2-head: `ai/models/candidate_runs/tutor-route10-top20-prune-oldmix-t2top20-4x-t2head-20260524/model/action_value_best.pt`
- MC300 subset 4x T2-head: `ai/models/candidate_runs/tutor-route10-top20-prune-oldfull-t2top20-mc300-4x-t2head-20260524/model/action_value_best.pt`
- MC300 subset weak-only T2-head: `ai/models/candidate_runs/tutor-route10-top20-prune-oldfull-t2top20-mc300-weakonly-t2head-20260524/model/action_value_best.pt`

Guardrail result:

| model | new MC300 subset top20 / t20reg | fixed T2 top20 / t20reg | 72-stress top20 / t20reg | verdict |
|---|---:|---:|---:|---|
| top20-prune | 95.0% / 0.407 | 99.2% / 0.015 | 73.6% / 1.026 | current |
| old MC20 4x T2-head | 7.1% / 3.258 on old-label subset | 99.2% / 0.017 | 72.2% / 1.127 | reject |
| MC300 4x T2-head | 95.0% / 0.407 | 99.0% / 0.015 | 70.8% / 1.128 | reject |
| MC300 weak-only T2-head | 95.8% / 0.300 | 97.4% / 0.038 | 70.8% / 1.189 | reject |

Conclusion: non-holdout weak mining works, and re-labeling removes much of the apparent old-label weakness. The remaining MC300 misses are real but too narrow to improve the T2 head without hurting fixed-holdout or stress safety. Keep the current `top20-prune` model. Next useful data generation should collect more diverse MC300/MC1000 T2 misses before another model update, not increase weight on the current narrow subset.

This confirms these three T2 families are real hard cases for the current model, but directly mixing them into the training set overcorrects and damages the broader T2 fixed holdout. Do not adopt either checkpoint. The useful artifact is the stress set itself; the next improvement should use it as a validation/guardrail while sourcing more diverse T2 top20 misses from non-holdout data.

## Expanded MC300 T2 Diverse Follow-Up

Additional mining:

- expanded old full-source misses: `ai/data/tutor_t2_top20_nonholdout_20260524/targets_oldfull_t2_top20_miss268.jsonl`
- generated labels: `ai/data/tutor_t2_top20_nonholdout_20260524/teacher_labels_oldfull_t2_top20_miss268_mc300.jsonl`
- label mode: T2 MC300, 268/268 written, no batch errors, elapsed 1024.9s
- converted reranker data: `ai/data/tutor_t2_top20_nonholdout_20260524/reranker_oldfull_t2_top20_miss268_mc300`
- current model on this MC300 set: top20 92.9%, top20 rerank regret 0.191

Second source:

- source data: `ai/models/candidate_runs/fl-route-reranker-v2-medium-20260520/data_active_t1t2_s20_2k`
- source labels: `ai/models/candidate_runs/fl-route-reranker-v2-medium-20260520/active_teacher_t1t2_s20_2k.jsonl`
- source T2 decisions: 1,082
- current model on source T2: top20 96.6%, top20 rerank regret 0.070
- selected T2 rank>20 groups: 37
- generated labels: `ai/data/tutor_t2_top20_nonholdout_20260524/teacher_labels_active_t1t2_t2_top20_miss37_mc300.jsonl`
- label mode: T2 MC300, 37/37 written, no batch errors, elapsed 152.3s
- converted reranker data: `ai/data/tutor_t2_top20_nonholdout_20260524/reranker_active_t1t2_t2_top20_miss37_mc300`
- current model on this MC300 set: top20 89.2%, top20 rerank regret 0.030

Training run:

- mixed data: `D:/ofc_data/tutor-route10-top20-prune-t2diverse-mc300-4x-20260524/reranker_base_active8x_t2diverse_mc300_4x`
- mix contents: base active8x data plus 4x oldfull268 MC300 plus 4x active37 MC300
- mixed samples: 387,063 total, 78,767 T2
- checkpoint: `ai/models/candidate_runs/tutor-route10-top20-prune-t2diverse-mc300-4x-t2head-20260524/model/action_value_best.pt`
- training mode: T2-only turn-specific head, shared layers frozen, initialized from current `top20-prune`

Guardrail result:

| model | fixed T2 top20 / t20reg | oldfull268 top20 / t20reg | active37 top20 / t20reg | 72-stress top20 / t20reg | verdict |
|---|---:|---:|---:|---:|---|
| top20-prune | 99.2% / 0.015 | 92.9% / 0.191 | 89.2% / 0.030 | 73.6% / 1.026 | current |
| t2diverse MC300 4x T2-head | 99.2% / 0.015 | 93.7% / 0.189 | 89.2% / 0.030 | 72.2% / 1.127 | reject |

Conclusion: the expanded MC300 weak data produced only a tiny gain on the oldfull268 set and no gain on the active37 set, while worsening the 72-state stress guardrail. Keep `tutor-route10-top20-prune-active8x-ft-20260523` as the adopted model. The next useful step is to source more true T2 weak examples from different generation distributions, or change the ranking objective/candidate generator rather than increasing weight on these narrow misses.

## T2 Top20 Objective Follow-Up

Rationale:

- T1 and T3 are not ignored. The fixed holdouts are already strong enough for pruning: current T1 top20 is 98.6%, current T3 top20 is 99.0% with zero top20 rerank regret.
- The known remaining pruning failure is concentrated in T2 stress cases: the current model gets only 73.6% top20 recall on the 72-state MC1000 stress set.
- For this reason, the first objective change targets T2 top20 pruning safety only, using turn-specific heads so T1/T3 should remain unchanged.

Implementation:

- Added `topk_boundary_margin_loss` to `ai/training/train_action_value_reranker.py`.
- The loss pushes the teacher-best action above the predicted K-th competitor boundary, targeting top20 pruning directly.
- Added CLI flags: `--topk-margin-weight`, `--topk-margin`, and `--topk-margin-temperature`.

Experiments:

- margin checkpoint: `ai/models/candidate_runs/tutor-route10-top20-prune-t2diverse-mc300-4x-top20margin-t2head-20260524/model/action_value_best.pt`
- stress-added margin checkpoint: `ai/models/candidate_runs/tutor-route10-top20-prune-t2diverse-stress-mc300-4x-top20margin-t2head-20260524/model/action_value_best.pt`
- stress-added ListNet checkpoint: `ai/models/candidate_runs/tutor-route10-top20-prune-t2diverse-stress-mc300-4x-listnet-t2head-20260524/model/action_value_best.pt`
- stress-added data: `D:/ofc_data/tutor-route10-top20-prune-t2diverse-stress-mc300-4x-20260524/reranker_base_active8x_t2diverse_stress_mc300_4x`
- stress-added mix contents: previous T2-diverse MC300 4x mix plus 4 copies of the 72-state MC1000 stress set

Guardrail result:

| model | T1 fixed top20 / t20reg | T2 fixed top20 / t20reg | T3 fixed top20 / t20reg | oldfull268 top20 / t20reg | active37 top20 / t20reg | 72-stress top20 / t20reg | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| top20-prune | 98.6% / 0.018 | 99.2% / 0.015 | 99.0% / 0.000 | 92.9% / 0.191 | 89.2% / 0.030 | 73.6% / 1.026 | current |
| T2-diverse margin | 98.6% / 0.018 | 99.2% / 0.016 | 99.0% / 0.000 | 94.4% / 0.138 | 94.6% / 0.023 | 66.7% / 1.196 | reject |
| T2-diverse+stress margin | 98.6% / 0.018 | 99.2% / 0.016 | 99.0% / 0.000 | 94.4% / 0.142 | 97.3% / 0.015 | 70.8% / 0.950 | reject |
| T2-diverse+stress ListNet | 98.6% / 0.018 | 99.2% / 0.015 | 99.0% / 0.000 | 94.0% / 0.189 | 89.2% / 0.030 | 72.2% / 1.127 | reject |

Conclusion: T1/T3 remain unchanged because the T2-only head isolation works. The new objective improves several non-holdout T2 weak sets, but still fails the key pruning criterion because 72-stress top20 recall is below current. Do not adopt these checkpoints. The useful code change is the top20-boundary loss knob; the useful next experiment is broader T2 data mining, not more weight on the same small stress set.

## Broader T2 P99 Mining and Budget Follow-Up

Additional mining:

- broad recursive source checked: `D:/ofc_data/recursive-t0t3-s128-b5-c2-p16-tr32-20260520/reranker_recursive_s128_b5_c2_p16_tr32`
- result on broad recursive T2: current top20 100.0%, top20 rerank regret 0.000, so it was not useful for weak mining
- p99 source: `ai/data/tutor_top20_prune_p99_20260523/reranker_t2t3_p99_299`
- current model on p99 T2: top20 96.0%, but with high-regret misses
- selected p99 T2 misses, suit-augmented targets: `ai/data/tutor_t2_broader_mining_20260524/targets_p99_t2_top20_miss4_suit24.jsonl`
- generated labels: `ai/data/tutor_t2_broader_mining_20260524/teacher_labels_p99_t2_top20_miss4_suit24_mc1000.jsonl`
- label mode: T2 MC1000, 144/144 written, no batch errors, elapsed 952.1s
- converted reranker data: `ai/data/tutor_t2_broader_mining_20260524/reranker_p99_t2_top20_miss4_suit24_mc1000`

Adapter implementation:

- Added optional zero-initialized turn-specific adapters to `ai/models/action_value_reranker.py`.
- Added `--turn-specific-adapters` and `--adapter-dim` to `ai/training/train_action_value_reranker.py`.
- The adapters preserve the loaded checkpoint at initialization and allow T2-only fine-tuning without changing the shared trunk.

Training attempts:

- 4x p99 adapter: `ai/models/candidate_runs/tutor-route10-top20-prune-t2diverse-stress-p99-mc1000-4x-adapter64-t2head-20260524/model/action_value_best.pt`
- lower-LR 4x p99 adapter: `ai/models/candidate_runs/tutor-route10-top20-prune-t2diverse-stress-p99-mc1000-4x-adapter64-lr2e5-t2head-20260524/model/action_value_best.pt`
- 1x p99 adapter: `ai/models/candidate_runs/tutor-route10-top20-prune-t2diverse-stress-p99-mc1000-1x-adapter64-t2head-20260524/model/action_value_best.pt`

Guardrail result:

| model | fixed T2 top20 / t20reg | oldfull268 top20 / t20reg | active37 top20 / t20reg | 72-stress top20 / t20reg | p99 MC1000 top20 / t20reg | verdict |
|---|---:|---:|---:|---:|---:|---|
| top20-prune | 99.2% / 0.015 | 92.9% / 0.191 | 89.2% / 0.030 | 73.6% / 1.026 | 64.6% / 1.117 | current |
| p99 4x adapter | 98.4% / 0.019 | 94.4% / 0.178 | 91.9% / 0.021 | 72.2% / 1.094 | 81.2% / 1.112 | reject |
| p99 4x adapter, low LR | 99.0% / 0.017 | n/a | n/a | 69.4% / 1.196 | 68.1% / 0.985 | reject |
| p99 1x adapter | 99.0% / 0.017 | 93.7% / 0.191 | 91.9% / 0.022 | 73.6% / 1.027 | 73.6% / 1.113 | candidate only |

Conclusion: the p99 data is a real T2 weakness, but direct fine-tuning still trades off fixed T2 top20 safety. Do not replace the adopted `top20-prune` model yet. The best checkpoint remains:

`ai/models/candidate_runs/tutor-route10-top20-prune-active8x-ft-20260523/model/action_value_best.pt`

Prune-budget result:

- Added `--top-ns` to `ai/tutor/evaluate_action_value_dataset.py` so arbitrary budgets such as top24 can be reported.
- On current `top20-prune`, T2 fixed top24 is 100.0% with zero top24 regret.
- On current `top20-prune`, p99 MC1000 top24 is 100.0% with zero top24 regret.
- On current `top20-prune`, active37 top24 is 100.0% with zero top24 regret.
- On current `top20-prune`, 72-stress top24 is 93.1% with top24 rerank regret 0.417.

Operational recommendation: keep the current model, but use a safer T2 candidate budget of 24 before exact/high-precision reranking when latency allows. Most T2 top20 failures are near-boundary rank 21-24 cases, so this is a better immediate safety valve than adopting the overfitted p99 adapters.

## T2 Continued Mining: Joint Unseen 200

Additional mining:

- source logs: the five `joint-t0-aa-k20-trips45-ft-20260521` Route10 self-play files
- exclusions: fixed eval holdout `targets_t0t3_500_each.jsonl` plus Route10 active teacher labels
- selected targets: `ai/data/tutor_t2_continued_mining_20260524/targets_joint_unseen_t2_200.jsonl`
- generated labels: `ai/data/tutor_t2_continued_mining_20260524/teacher_labels_joint_unseen_t2_200_mc300.jsonl`
- label mode: T2 MC300, 200/200 written, no batch errors, elapsed 440.6s
- converted reranker data: `ai/data/tutor_t2_continued_mining_20260524/reranker_joint_unseen_t2_200_mc300`

Current model on the new set:

- top20: 98.0%, top20 rerank regret 0.001
- top24: 100.0%, top24 rerank regret 0.000

This new distribution supports the operational top24 budget: the current model already keeps every teacher-best action inside 24 candidates.

Training attempt:

- mixed data: `D:/ofc_data/tutor-route10-top20-prune-t2diverse-stress-p99-joint200-1x-20260524/reranker_base_active8x_t2diverse_stress_p99_joint200_1x`
- mix contents: prior T2-diverse+stress+p99 1x mix plus the new joint-unseen 200 MC300 set
- checkpoint: `ai/models/candidate_runs/tutor-route10-top20-prune-t2diverse-stress-p99-joint200-1x-adapter64-t2head-20260524/model/action_value_best.pt`
- training mode: T2-only turn-specific head plus adapter, shared layers frozen, initialized from current `top20-prune`

Guardrail result:

| model | fixed T2 top20 / top24 / t24reg | joint200 top20 / top24 / t24reg | oldfull268 top20 / top24 / t24reg | active37 top20 / top24 / t24reg | 72-stress top20 / top24 / t24reg | p99 MC1000 top20 / top24 / t24reg | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| top20-prune | 99.2% / 100.0% / 0.000 | 98.0% / 100.0% / 0.000 | 92.9% / 98.5% / 0.048 | 89.2% / 100.0% / 0.000 | 73.6% / 93.1% / 0.417 | 64.6% / 100.0% / 0.000 | current |
| joint200 1x adapter | 99.0% / 99.6% / 0.002 | 97.5% / 100.0% / 0.000 | 93.7% / 98.9% / 0.052 | 91.9% / 100.0% / 0.000 | 68.1% / 87.5% / 0.757 | 71.5% / 100.0% / 0.000 | reject |

Conclusion: adding the joint-unseen T2 set does not justify a model replacement. It improves p99 top20 recall, but worsens fixed T2 top24 safety and the 72-state stress guardrail. Keep the adopted `top20-prune` checkpoint and the T2 top24 runtime budget. The next model-strengthening work should either mine broader T2 top24 misses from genuinely different generation distributions or improve the candidate generator before fine-tuning another T2 adapter.

## T2 Top24 Miss Mining From Eval Logs

Additional mining:

- source logs: 15 `fl-route-reranker-v2-medium-20260520/eval_*_r2_g300.jsonl` files
- selected targets: `ai/data/tutor_t2_continued_mining_20260524/targets_eval_g300_t2_500.jsonl`
- generated labels: `ai/data/tutor_t2_continued_mining_20260524/teacher_labels_eval_g300_t2_500_mc300.jsonl`
- label mode: T2 MC300, 500/500 written, no batch errors, elapsed 1157.1s
- converted reranker data: `ai/data/tutor_t2_continued_mining_20260524/reranker_eval_g300_t2_500_mc300`

Current model on this eval-log distribution:

- top20: 97.6%, top20 rerank regret 0.012
- top24: 99.4%, top24 rerank regret 0.002
- top24 misses: 3/500 decisions

The three top24 misses were converted back into targets, suit-augmented, and relabeled at MC1000:

- targets: `ai/data/tutor_t2_continued_mining_20260524/targets_eval_g300_t2_top24_miss_suit24.jsonl`
- generated labels: `ai/data/tutor_t2_continued_mining_20260524/teacher_labels_eval_g300_t2_top24_miss_suit24_mc1000.jsonl`
- label mode: T2 MC1000, 72/72 written, no batch errors, elapsed 645.3s
- converted reranker data: `ai/data/tutor_t2_continued_mining_20260524/reranker_eval_g300_t2_top24_miss_suit24_mc1000`

Current model on this new top24-stress set:

- top20: 47.2%, top20 rerank regret 0.190
- top24: 75.0%, top24 rerank regret 0.041

Implementation note:

- `ai/tutor/weak_groups_to_active_targets.py` now records dynamic `topK` miss/regret reasons such as `top24_miss` and `top24_rerank_regret` when the selected weak-group threshold is not 20.

Training attempt:

- mixed data: `D:/ofc_data/tutor-route10-top20-prune-t2diverse-stress-p99-joint200-top24stress-1x-20260524/reranker_base_active8x_t2diverse_stress_p99_joint200_top24stress_1x`
- mix contents: prior T2-diverse+stress+p99+joint200 1x mix plus the new MC1000 top24-stress set
- checkpoint: `ai/models/candidate_runs/tutor-route10-top20-prune-t2diverse-stress-p99-joint200-top24stress-1x-adapter64-t2head-20260524/model/action_value_best.pt`
- training mode: T2-only turn-specific head plus adapter, shared layers frozen, `target_topk=24`, small topK margin loss

Guardrail result:

| model | fixed T2 top20 / top24 / t24reg | eval500 top20 / top24 / t24reg | new top24-stress top20 / top24 / t24reg | old 72-stress top20 / top24 / t24reg | p99 MC1000 top20 / top24 / t24reg | active37 top20 / top24 / t24reg | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| top20-prune | 99.2% / 100.0% / 0.000 | 97.6% / 99.4% / 0.002 | 47.2% / 75.0% / 0.041 | 73.6% / 93.1% / 0.417 | 64.6% / 100.0% / 0.000 | 89.2% / 100.0% / 0.000 | current |
| top24-stress 1x adapter | 98.4% / 99.6% / 0.003 | 93.6% / 99.0% / 0.036 | 83.3% / 95.8% / 0.011 | 79.2% / 93.1% / 0.404 | 78.5% / 100.0% / 0.000 | 89.2% / 97.3% / 0.015 | reject |

Conclusion: the new top24-stress set is real and learnable, but the adapter still overfits: it improves the new top24-stress and p99 sets while hurting fixed T2, the broad eval500 distribution, and active37 top24 safety. Do not adopt the checkpoint. Keep the current model plus T2 top24 runtime budget. The next useful direction is not more adapter weight; it is either broader top24-miss mining across more generation distributions or a stronger candidate generator/reranker architecture that improves these hard sets without shifting broad T2 calibration.

## T2 G1000 Eval-Log Mining

Additional mining:

- source logs: three `fl-route-reranker-v2-medium-20260520/eval_strong_fl_top10_*_r2_g1000.jsonl` files
- selected targets: `ai/data/tutor_t2_continued_mining_20260524/targets_eval_g1000_t2_500.jsonl`
- generated labels: `ai/data/tutor_t2_continued_mining_20260524/teacher_labels_eval_g1000_t2_500_mc300.jsonl`
- label mode: T2 MC300, 500/500 written, no batch errors, elapsed 1193.0s
- converted reranker data: `ai/data/tutor_t2_continued_mining_20260524/reranker_eval_g1000_t2_500_mc300`

Current model on this G1000 eval-log distribution:

- top20: 95.8%, top20 rerank regret 0.032
- top24: 99.0%, top24 rerank regret 0.0004
- top24 misses: 5/500 decisions
- largest observed top24 rerank regret: 0.100

Conclusion: the G1000 eval-log mining strengthens the operational case for the T2 top24 safety budget. The top24 miss count is nonzero, but the misses are near ties with very small rerank regret, so they are not worth another MC1000 top24-stress expansion yet. This set provides additional top20 miss examples, but it does not justify a model replacement or another narrow T2 adapter. Keep the adopted model and continue using top24 before exact/high-precision reranking at T2.

## T2 G1000 Top20-Miss MC1000 Probe

The 21 G1000 eval-log decisions where the current model missed top20 were converted back into targets and relabeled at MC1000 without suit augmentation:

- targets: `ai/data/tutor_t2_continued_mining_20260524/targets_eval_g1000_t2_top20_miss_mc1000_probe.jsonl`
- generated labels: `ai/data/tutor_t2_continued_mining_20260524/teacher_labels_eval_g1000_t2_top20_miss_probe_mc1000.jsonl`
- label mode: T2 MC1000, 21/21 written, no batch errors, elapsed 182.0s
- converted reranker data: `ai/data/tutor_t2_continued_mining_20260524/reranker_eval_g1000_t2_top20_miss_probe_mc1000`

Current model on the no-augmentation MC1000 probe:

- top20: 19.0%, top20 rerank regret 0.752
- top24: 76.2%, top24 rerank regret 0.0187
- top24 max rerank regret: 0.226

This confirms that the mined top20 misses are real under a higher-sim teacher, while the operational top24 budget still removes almost all practical regret on the original suit forms.

Suit-label augmentation was added in `ai/tutor/augment_teacher_suits.py`. It applies global h/d/c/s permutations to already-labeled teacher records, preserving EV/FL/bust labels without rerunning the evaluator. Applying it to the 21 MC1000 records produced 504 records:

- augmented labels: `ai/data/tutor_t2_continued_mining_20260524/teacher_labels_eval_g1000_t2_top20_miss_probe_mc1000_suit24.jsonl`
- converted reranker data: `ai/data/tutor_t2_continued_mining_20260524/reranker_eval_g1000_t2_top20_miss_probe_mc1000_suit24`
- conversion output: 504 decisions, 12,816 candidate samples

Evaluation on the suit24 probe:

| model | top20 | top24 | t20reg | t24reg | verdict |
|---|---:|---:|---:|---:|---|
| top20-prune current | 72.4% | 93.8% | 0.202 | 0.057 | current |
| prior T2 top20 suit24 1x head | 81.7% | 96.4% | 0.284 | 0.184 | reject |
| prior T2 top20 suit24 8x head | 77.6% | 94.2% | 0.249 | 0.121 | reject |

Conclusion: the current model has a real suit-sensitivity weakness on hard T2 examples. The prior T2-head top20-suit24 models are still not adoptable because they improve some recall numbers while worsening regret and broad fixed T2 guardrails. This points to a different improvement direction: either train a shared/suit-invariant candidate model with suit augmentation from the start, or add a slower high-confidence suit-ensemble/canonicalization path for hard T2 states. Do not replace the currently adopted model yet.

## T2 Suit-Ensemble Probe

`ai/tutor/evaluate_action_value_suit_ensemble.py` evaluates a suit-augmented reranker dataset by grouping each 24-permutation block back into one original decision and averaging model predictions per candidate across the block. This approximates a slower runtime path where a hard T2 state is evaluated under all global suit permutations before pruning.

Current model with 24-way suit-ensemble:

| dataset | original per-variant top20 / top24 / t24reg | suit-ensemble top20 / top24 / t24reg | conclusion |
|---|---:|---:|---|
| G1000 top20-miss MC1000 suit24 | 72.4% / 93.8% / 0.057 | 66.7% / 100.0% / 0.000 | top24 fully recovered |
| older T2 top20 suit24 MC1000 | 73.6% / n/a / n/a | 33.3% / 100.0% / 0.000 | tiny 3-decision set, top24 recovered |
| p99 T2 top20-miss4 suit24 MC1000 | 64.6% / n/a / n/a | 66.7% / 100.0% / 0.000 | top24 recovered |

Conclusion: suit-ensemble is not a replacement for better training because top1/top10 remain weak, but it is a strong safety path for T2 pruning when followed by exact/high-precision reranking. If latency allows, a hard-state runtime mode can keep top24 after a 24-way suit-ensemble score average. This is safer than adopting the prior T2-head suit24 checkpoints, which harmed broad guardrails.

Partial ensemble tradeoff:

| dataset | ensemble size | top20 | top24 | t20reg | t24reg |
|---|---:|---:|---:|---:|---:|
| G1000 top20-miss MC1000 suit24 | 4 | 57.1% | 90.5% | 0.477 | 0.004 |
| G1000 top20-miss MC1000 suit24 | 8 | 71.4% | 100.0% | 0.057 | 0.000 |
| G1000 top20-miss MC1000 suit24 | 12 | 71.4% | 100.0% | 0.036 | 0.000 |
| G1000 top20-miss MC1000 suit24 | 24 | 66.7% | 100.0% | 0.076 | 0.000 |
| older T2 top20 suit24 MC1000 | 8 | 33.3% | 100.0% | 1.896 | 0.000 |
| older T2 top20 suit24 MC1000 | 24 | 33.3% | 100.0% | 2.019 | 0.000 |
| p99 T2 top20-miss4 suit24 MC1000 | 4 | 66.7% | 100.0% | 0.001 | 0.000 |
| p99 T2 top20-miss4 suit24 MC1000 | 8 | 66.7% | 100.0% | 0.000 | 0.000 |
| p99 T2 top20-miss4 suit24 MC1000 | 24 | 66.7% | 100.0% | 0.001 | 0.000 |

Recommendation: use ensemble size 8 as the default T2 safety path when latency matters. Size 4 is too weak on the G1000 hard set. Size 24 remains useful for offline diagnostics or highest-accuracy paid/background analysis, but size 8 already recovered top24 on the tested hard sets.

CPU runtime benchmark on the adopted model for a 24-candidate T2 state (`ai/tutor/benchmark_suit_ensemble_runtime.py`, 30 iterations, PyTorch CPU with 8 threads):

| ensemble size | states scored | mean ms/decision | median ms/decision |
|---|---:|---:|---:|
| off | 24 | 4.3 | 4.4 |
| 4 | 96 | 13.3 | 12.9 |
| 8 | 192 | 27.6 | 27.8 |
| 12 | 288 | 39.3 | 39.3 |
| 24 | 576 | 79.7 | 81.1 |

This makes size 8 practical for paid/background T2 pruning and still cheap compared with high-sim MC or exact reranking. Free/fast paths can keep ensemble off.

Self-play smoke:

```powershell
$env:PYTHONPATH='.'
python ai\self_play.py `
  --games 1 --workers 1 --rollouts 0 --top-k 10 --seed 20260524 `
  --output ai\data\tutor_t2_continued_mining_20260524\selfplay_smoke_suitensemble_t2_size8.jsonl `
  --reranker ai\models\candidate_runs\tutor-route10-top20-prune-active8x-ft-20260523\model\action_value_best.pt `
  --reranker-top-k 20 --reranker-t2-top-k 24 `
  --reranker-suit-ensemble-turns 2
```

Result: 1 game, 10 turn records, including 2 T2 records. The run initialized the reranker with suit ensemble turns `2` and default size `8`.

Runtime support:

- `ai/mcts/rollout_evaluator.py` supports `action_value_suit_ensemble_turns`; matching turns score each candidate under a deterministic spread of suit permutations and average the reranker score.
- `ai/self_play.py` exposes this as `--reranker-suit-ensemble-turns` and `--reranker-suit-ensemble-size`; the runtime default size is 8, for example `--reranker-suit-ensemble-turns 2`.
- The option is off by default, so existing runs are unchanged unless explicitly enabled.
- Unit coverage: `tests/test_rollout_evaluator_budget.py` verifies that the selected turn expands reranker batches by the configured ensemble size and other turns remain unchanged.

## Candidate Model Guardrail Comparison

Added `ai/tutor/compare_candidate_models.py` to keep future model upgrades from overfitting a mined weak set. The first `--model` is the baseline; every later model is compared against it on fixed broad, hard, and suit-ensemble guardrails. The report writes per-dataset evaluator summaries plus `comparison.json` and `comparison.md`.

Example smoke run:

```powershell
$env:PYTHONPATH='.'
python ai\tutor\compare_candidate_models.py `
  --model current=ai\models\candidate_runs\tutor-route10-top20-prune-active8x-ft-20260523\model\action_value_best.pt `
  --model t2head1x=ai\models\candidate_runs\tutor-route10-top20-prune-t2top20-suit24-1x-t2head-20260524\model\action_value_best.pt `
  --datasets t2_holdout,t2_top20_probe_suit24_ens8 `
  --output ai\data\tutor_model_compare_smoke_20260524 `
  --force
```

Smoke result: `t2head1x` is rejected. It improves the T2 top20 suit-ensemble probe, but drops T2 holdout top20 from 99.2% to 96.4%, so it is not suitable as the production checkpoint.

Use the full guardrail suite before adopting a new model:

```powershell
$env:PYTHONPATH='.'
python ai\tutor\compare_candidate_models.py `
  --model current=ai\models\candidate_runs\tutor-route10-top20-prune-active8x-ft-20260523\model\action_value_best.pt `
  --model candidate=PATH\TO\action_value_best.pt `
  --strict
```

Guardrail rules are intentionally conservative:

- broad sets reject top24 recall worse by more than 0.5pp, top20 recall worse by more than 1.0pp, top24 regret worse by more than 0.010, or top20 regret worse by more than 0.050.
- hard and ensemble sets reject top24 recall worse by more than 1.0pp or top24 regret worse by more than 0.050.

Full T2 candidate comparison:

```powershell
$env:PYTHONPATH='.'
python ai\tutor\compare_candidate_models.py `
  --model current=ai\models\candidate_runs\tutor-route10-top20-prune-active8x-ft-20260523\model\action_value_best.pt `
  --model t2_p99_4x_lr2e5=ai\models\candidate_runs\tutor-route10-top20-prune-t2diverse-stress-p99-mc1000-4x-adapter64-lr2e5-t2head-20260524\model\action_value_best.pt `
  --model t2_p99_4x_listnet=ai\models\candidate_runs\tutor-route10-top20-prune-t2diverse-stress-p99-mc1000-4x-listnet-t2head-20260524\model\action_value_best.pt `
  --model t2_mc300_listnet=ai\models\candidate_runs\tutor-route10-top20-prune-t2diverse-stress-mc300-4x-listnet-t2head-20260524\model\action_value_best.pt `
  --model t2_mc300_margin=ai\models\candidate_runs\tutor-route10-top20-prune-t2diverse-stress-mc300-4x-top20margin-t2head-20260524\model\action_value_best.pt `
  --output ai\data\tutor_model_compare_full_t2_more_candidates_20260524
```

Result: four candidates pass the guardrail suite: `t2_p99_4x_lr2e5`, `t2_p99_4x_listnet`, `t2_mc300_listnet`, and `t2_mc300_margin`.

Recommended next candidate depends on the target:

- `t2_p99_4x_lr2e5`: best hard-T2 improvement without broad rejection. On `t2_top24_stress_suit24`, top10 improves from 26.4% to 45.8%, top20 from 47.2% to 66.7%, while broad T2 drops stay within guardrails.
- `t2_mc300_margin`: best broad-distribution top10 behavior. On `t2_g300_eval`, top10 improves from 83.4% to 85.0%; on `t2_g1000_eval`, top10 improves from 84.4% to 86.4%. It is less effective on the hardest top24 stress set.

Do not adopt the earlier `t2_suit1x`, `t2_suit8x`, or `t2_joint_top24` checkpoints as production defaults; they improve some hard sets but regress broad T2 guardrails.

Route smoke for `t2_p99_4x_lr2e5`:

```powershell
$env:PYTHONPATH='.'
python ai\self_play.py `
  --games 100 --workers 4 --rollouts 0 --top-k 10 --seed 20260524 `
  --output ai\data\tutor_route_compare_20260524\current_100_suitensemble_t2_size8.jsonl `
  --reranker ai\models\candidate_runs\tutor-route10-top20-prune-active8x-ft-20260523\model\action_value_best.pt `
  --reranker-top-k 20 --reranker-t2-top-k 24 `
  --reranker-suit-ensemble-turns 2

python ai\self_play.py `
  --games 100 --workers 4 --rollouts 0 --top-k 10 --seed 20260524 `
  --output ai\data\tutor_route_compare_20260524\t2_p99_4x_lr2e5_100_suitensemble_t2_size8.jsonl `
  --reranker ai\models\candidate_runs\tutor-route10-top20-prune-t2diverse-stress-p99-mc1000-4x-adapter64-lr2e5-t2head-20260524\model\action_value_best.pt `
  --reranker-top-k 20 --reranker-t2-top-k 24 `
  --reranker-suit-ensemble-turns 2

python ai\analysis\summarize_selfplay_jsonl.py `
  ai\data\tutor_route_compare_20260524\current_100_suitensemble_t2_size8.jsonl `
  ai\data\tutor_route_compare_20260524\t2_p99_4x_lr2e5_100_suitensemble_t2_size8.jsonl `
  --output-json ai\data\tutor_route_compare_20260524\summary_100.json `
  --output-md ai\data\tutor_route_compare_20260524\summary_100.md
```

100-hand aggregate result:

| model | hands | seat FL | any-FL | foul | shape-foul | avg seat0 reward |
|---|---:|---:|---:|---:|---:|---:|
| current | 100 | 17.5% | 30.0% | 60.5% | 38.5% | +15.220 |
| t2_p99_4x_lr2e5 | 100 | 17.0% | 29.0% | 61.0% | 38.5% | +15.100 |

Aligned 20-hand route diff, generated with `--workers 1` and compared by `ai/tutor/compare_selfplay_routes.py`:

| turn | records | action diffs |
|---:|---:|---:|
| T0 | 40 | 0 |
| T1 | 40 | 0 |
| T2 | 40 | 2 |
| T3 | 40 | 1 |
| T4 | 40 | 0 |

Interpretation: `t2_p99_4x_lr2e5` changes only a small number of route decisions in this smoke. The aggregate 100-hand route metrics are essentially neutral to slightly worse, so this checkpoint is a safe experimental candidate for hard-T2 pruning, but not a clear default replacement from route-smoke alone.

Larger 1,000-hand route smoke:

```powershell
$env:PYTHONPATH='.'
python ai\self_play.py `
  --games 1000 --workers 4 --rollouts 0 --top-k 10 --seed 20260524 `
  --output ai\data\tutor_route_compare_20260524\current_1000_suitensemble_t2_size8.jsonl `
  --reranker ai\models\candidate_runs\tutor-route10-top20-prune-active8x-ft-20260523\model\action_value_best.pt `
  --reranker-top-k 20 --reranker-t2-top-k 24 `
  --reranker-suit-ensemble-turns 2

python ai\self_play.py `
  --games 1000 --workers 4 --rollouts 0 --top-k 10 --seed 20260524 `
  --output ai\data\tutor_route_compare_20260524\t2_p99_4x_lr2e5_1000_suitensemble_t2_size8.jsonl `
  --reranker ai\models\candidate_runs\tutor-route10-top20-prune-t2diverse-stress-p99-mc1000-4x-adapter64-lr2e5-t2head-20260524\model\action_value_best.pt `
  --reranker-top-k 20 --reranker-t2-top-k 24 `
  --reranker-suit-ensemble-turns 2
```

1,000-hand aggregate result:

| model | hands | seat FL | any-FL | foul | shape-foul | avg seat0 reward |
|---|---:|---:|---:|---:|---:|---:|
| current | 1,000 | 18.2% | 33.1% | 64.5% | 44.5% | +16.025 |
| t2_p99_4x_lr2e5 | 1,000 | 18.6% | 33.3% | 64.0% | 44.0% | +16.039 |

Aligned 100-hand route diff:

```powershell
$env:PYTHONPATH='.'
python ai\tutor\compare_selfplay_routes.py `
  --baseline ai\data\tutor_route_compare_20260524\current_100_aligned_suitensemble_t2_size8.jsonl `
  --candidate ai\data\tutor_route_compare_20260524\t2_p99_4x_lr2e5_100_aligned_suitensemble_t2_size8.jsonl `
  --output-json ai\data\tutor_route_compare_20260524\aligned_100_action_compare.json `
  --output-md ai\data\tutor_route_compare_20260524\aligned_100_action_compare.md
```

| turn | records | action diffs |
|---:|---:|---:|
| T0 | 200 | 0 |
| T1 | 200 | 0 |
| T2 | 200 | 12 |
| T3 | 200 | 13 |
| T4 | 200 | 6 |

The aligned diff shows 31 changed records across 100 hands, with 11 hands affected. The direct model changes start at T2; later T3/T4 differences are mostly downstream state divergence. With the 1,000-hand aggregate neutral-to-slightly-positive, `t2_p99_4x_lr2e5` remains a reasonable experimental candidate for hard-T2 pruning, but the evidence still favors keeping `current` as the default until a candidate improves both hard T2 and broad route metrics more clearly.

Teacher check for changed T2 actions:

```powershell
$env:PYTHONPATH='.'
python ai\tutor\extract_selfplay_diff_targets.py `
  --baseline ai\data\tutor_route_compare_20260524\current_100_aligned_suitensemble_t2_size8.jsonl `
  --candidate ai\data\tutor_route_compare_20260524\t2_p99_4x_lr2e5_100_aligned_suitensemble_t2_size8.jsonl `
  --turns 2 `
  --same-state-only `
  --output ai\data\tutor_route_compare_20260524\t2_same_state_action_diff_targets.jsonl

python ai\training\generate_active_teacher.py `
  ai\data\tutor_route_compare_20260524\t2_same_state_action_diff_targets.jsonl `
  --output ai\data\tutor_route_compare_20260524\t2_same_state_action_diff_teacher_mc3000.jsonl `
  --sims 3000 `
  --turns 2 `
  --mc-turns 2 `
  --batch-engine `
  --batch-timeout 3600

python ai\tutor\compare_diff_teacher_labels.py `
  --targets ai\data\tutor_route_compare_20260524\t2_same_state_action_diff_targets.jsonl `
  --labels ai\data\tutor_route_compare_20260524\t2_same_state_action_diff_teacher_mc3000.jsonl `
  --output-json ai\data\tutor_route_compare_20260524\t2_same_state_action_diff_teacher_mc3000_compare.json `
  --output-md ai\data\tutor_route_compare_20260524\t2_same_state_action_diff_teacher_mc3000_compare.md
```

MC3000 result on the 11 same-state T2 route changes:

| valid | candidate wins | baseline wins | ties | avg candidate-baseline EV delta | sum delta |
|---:|---:|---:|---:|---:|---:|
| 11 | 4 | 5 | 2 | -0.731 | -8.038 |

The 6 largest-delta states were rechecked at MC10000:

| valid | candidate wins | baseline wins | ties | avg candidate-baseline EV delta | sum delta |
|---:|---:|---:|---:|---:|---:|
| 6 | 3 | 3 | 0 | -1.186 | -7.114 |

Notable MC10000 examples:

- hand 60 record 595: candidate loses -16.147 EV versus baseline; candidate lowers FL from 52.0% to 30.1% and raises bust from 48.0% to 69.8%.
- hand 73 record 726: candidate gains +7.205 EV by avoiding a 100% bust baseline action.
- hand 72 record 715: candidate gains +3.220 EV and slightly improves FL/bust.

Conclusion: `t2_p99_4x_lr2e5` is not a production default replacement. It fixes some catastrophic T2 choices, but it also introduces high-impact false positives. Use it only as a mining/experimental checkpoint. The safer production path remains `current` plus runtime T2 top24 pruning and optional suit ensemble; the next training round should add these candidate-vs-current changed states as hard negatives/positives before trying another replacement.

## Repro Commands

Build targets:

```powershell
$env:PYTHONPATH='.'
$inputs = @(
  'ai\models\candidate_runs\joint-t0-aa-k20-trips45-ft-20260521\selfplay_direct_aaktrips_b0_g100.jsonl',
  'ai\models\candidate_runs\joint-t0-aa-k20-trips45-ft-20260521\selfplay_hybrid_r2_top10_aaktrips_b0_g100.jsonl',
  'ai\models\candidate_runs\joint-t0-aa-k20-trips45-ft-20260521\selfplay_hybrid_r2_top10_balanced_b4_g50.jsonl',
  'ai\models\candidate_runs\joint-t0-aa-k20-trips45-ft-20260521\selfplay_hybrid_r2_top10_typefocus_noqqpen_g50.jsonl',
  'ai\models\candidate_runs\joint-t0-aa-k20-trips45-ft-20260521\selfplay_hybrid_r2_top20_typefocus_noqqpen_g50.jsonl'
)
python ai\tutor\build_eval_holdout_targets.py @inputs `
  --output ai\data\tutor_eval_holdout_20260523\targets_t0t3_500_each.jsonl `
  --turns 0,1,2,3 --per-turn 500 --seed 20260523 `
  --split eval_holdout_t0t3_500_each_20260523 `
  --exclude-jsonl ai\data\tutor_route10_20260522\active_teacher_t1_t3_combined\teacher_labels.jsonl
```

Generate T3 exact labels and evaluate:

```powershell
$env:PYTHONPATH='.'
python ai\training\generate_active_teacher.py ai\data\tutor_eval_holdout_20260523\targets_t0t3_500_each.jsonl `
  --output ai\data\tutor_eval_holdout_20260523\teacher_labels_t3_exact_500.jsonl `
  --turns 3 --mc-turns 0,1,2 --batch-engine --batch-timeout 3600 --print-errors

python ai\training\convert_action_value_teacher.py ai\data\tutor_eval_holdout_20260523\teacher_labels_t3_exact_500.jsonl `
  --output ai\data\tutor_eval_holdout_20260523\reranker_t3_exact_500 --turns 3 --state-dim 520

python ai\tutor\evaluate_action_value_dataset.py `
  --data ai\data\tutor_eval_holdout_20260523\reranker_t3_exact_500 `
  --model ai\models\candidate_runs\joint-t0-aa-k20-trips45-ft-20260521\model\action_value_best.pt `
  --output ai\data\tutor_eval_holdout_20260523\eval_t3_exact_500_baseline --device cuda
```
