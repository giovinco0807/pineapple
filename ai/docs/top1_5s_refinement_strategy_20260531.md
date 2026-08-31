# Top1 + 5s Refinement Strategy

Date: 2026-05-31

## Goal

The target product behavior is: while the user is playing, return the best action
within 5 seconds per turn. The ideal metric is final Top1 correctness.

Model Top1 alone is not yet strong enough, so the practical target is:

1. Improve model Top1 with hard Top1-miss data.
2. Keep enough candidates in the shortlist so the true best action is not lost.
3. Use the 5 second budget to refine the shortlist and correct model Top1 misses.

## Current Diagnostic Snapshot

Model:
`ai/models/candidate_runs/tutor-route10-hybrid-t1t2-hard-mc300500-20260531/model/action_value_best.pt`

Holdout/proxy results:

| turn | dataset | Top1 | Top3 | Top10 | Top15 | Top20 | Top24 | Top64 | notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| T0 | proxy MC30 top128 | 1.95% | 6.25% | 18.20% | 25.55% | 32.00% | 36.90% | 69.45% | Top128 is 100%; T0 is not ready for tight pruning |
| T1 | fixed MC300 | 35.20% | 59.40% | 92.20% | 97.80% | 99.60% | 100.00% | n/a | Top24 is safe on this holdout, but not meaningful pruning when actions are already small |
| T2 | fixed MC300 | 32.40% | 59.40% | 90.40% | 97.60% | 99.60% | 100.00% | n/a | Top15/20 are useful shortlist sizes; Top24 usually keeps almost everything |
| T3 | fixed exact | 26.20% | 54.80% | 87.00% | 93.40% | 98.60% | 100.00% | n/a | T3 labels are exact, so Top1 misses are safe active-loop targets |

## Hard Targets Created

Diagnostic active target files:

- `ai/data/hybrid_t1t2_active_20260531/targets_top1_miss_t0_proxy_mc300500_all_holdout_diagnostic.jsonl`
- `ai/data/hybrid_t1t2_active_20260531/targets_top1_miss_t1_mc300500_holdout_diagnostic.jsonl`
- `ai/data/hybrid_t1t2_active_20260531/targets_top1_miss_t2_mc300500_holdout_diagnostic.jsonl`

Counts:

- T0: 1,961 Top1-miss targets from proxy holdout.
- T1: 324 Top1-miss targets from fixed holdout.
- T2: 338 Top1-miss targets from fixed holdout.

These are diagnostic holdout-derived targets. If they are used for training,
the holdout must be replaced before reporting final generalization metrics.

MC Top1 labels are noisy when the best and second-best candidates are close.
`ai/tutor/weak_groups_to_active_targets.py` now supports teacher-margin
filtering:

- `--min-teacher-margin`: require a minimum best-vs-second-best teacher score
  gap.
- `--margin-applies-to estimated`: apply that filter to MC/estimated labels
  while preserving exact labels.

High-confidence diagnostic target files:

- `ai/data/hybrid_t1t2_active_20260531/targets_top1_miss_t1_mc300500_margin1_holdout_diagnostic.jsonl`
- `ai/data/hybrid_t1t2_active_20260531/targets_top1_miss_t2_mc300500_margin1_holdout_diagnostic.jsonl`
- `ai/data/hybrid_t1t2_active_20260531/targets_top1_miss_t0_proxy_mc30_margin2_holdout_diagnostic.jsonl`
- `ai/data/hybrid_t1t2_active_20260531/targets_top1_miss_t3_exact_holdout_diagnostic.jsonl`

High-confidence counts:

- T1 MC300 margin >= 1.0: 116 written, 208 skipped as low margin.
- T2 MC300 margin >= 1.0: 106 written, 232 skipped as low margin.
- T0 MC30 margin >= 2.0: 300 written, 1,661 skipped as low margin.
- T3 exact: 369 written. No MC-margin filtering needed because labels are exact.

The output target records include `teacher_eval_mode`, `teacher_is_exact`,
`teacher_score`, `teacher_second_score`, `teacher_margin`, and `teacher_sims`.
These fields should be preserved through future teacher generation logs so the
training set can distinguish exact labels from estimated labels.

## Local Teacher Smoke

The active-teacher pipeline was verified locally:

- T1: 5 MC30 records, 111 reranker samples.
- T2: 5 MC30 records, 84 reranker samples.
- T0: 1 MC30 record, 232 reranker samples.

Observed local teacher speed:

- T1 MC30: 5 records in 6.2s.
- T2 MC30: 5 records in 0.8s.
- T0 MC30: 1 record in 33.0s.

T0 is the bottleneck. Large T0 data generation should not be done locally with
full candidates unless the T0 candidate set is reduced first or the Rust path is
made substantially faster.

## Practical 5s Policy

T3/T4:

- Use exact evaluation where available.

T2:

- Model all legal actions.
- Keep a meaningful shortlist, e.g. Top10 to Top15 plus insurance candidates.
- Spend the 5 second budget refining the shortlist, not all 24 actions.

T1:

- Model all legal actions.
- Keep Top10 to Top15 plus insurance candidates.
- Use shallow refinement only when time remains; deeper recursive refinement is
  background data generation, not synchronous gameplay.

T0:

- Current model cannot safely prune to 20 to 24.
- Need a stronger T0-specific candidate generator before exact/refinement can be
  useful inside 5 seconds.
- The next useful target is not Top24; it is to make Top32/Top64 recall much
  higher while reducing average regret, then tighten the pool.

## Next Experiment

For model strengthening:

1. Generate non-holdout Top1-miss hard data, especially T0.
2. Train with turn-balanced sampling so T0 does not drown out T1/T2 or vice versa.
3. Report by turn:
   - Top1, Top3, Top5, Top10, Top15, Top20, Top24, Top32, Top64.
   - Average regret and max regret.
   - 5 second refinement override rate.
4. Replace any holdout that was mined into training.

## 2026-05-31 Follow-Up

The target extractor now supports recursive teacher logs via
`--teacher-format recursive`. This lets us mine Top1 misses from datasets
created by `convert_recursive_teacher_to_reranker.py`, where one source JSONL
line contains multiple T0-T3 decisions.

Non-holdout recursive source checked:

- data: `D:/ofc_data/joint-t0-aa-k20-trips45-top30-s100-b10-h120-20260521/reranker_joint`
- teacher source: `D:/ofc_data/joint-t0-aa-k20-trips45-top30-s100-b10-h120-20260521/merged.valid.jsonl`

Current model on that non-holdout distribution:

| turn | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | avg regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| T0 | 50.0% | 82.5% | 93.7% | 99.3% | 99.7% | 100.0% | 0.685 |
| T1 | 54.3% | 86.8% | 92.7% | 99.7% | 100.0% | 100.0% | 0.680 |
| T2 | 56.0% | 85.1% | 93.4% | 99.3% | 100.0% | 100.0% | 0.727 |
| T3 | 46.0% | 75.5% | 90.1% | 100.0% | 100.0% | 100.0% | 0.521 |

High-confidence non-holdout Top1-miss targets mined:

- `targets_nonholdout_joint_t0_top1_margin2.jsonl`: 14 written, 137 skipped as low margin.
- `targets_nonholdout_joint_t1_top1_margin1.jsonl`: 42 written, 96 skipped as low margin.
- `targets_nonholdout_joint_t2_top1_margin1.jsonl`: 36 written, 97 skipped as low margin.

Large exact late-turn dataset checked:

- data: `ai/data/t0_proxy_holdout_20260529/exact_late_reranker_t3t4_10000_20260529`
- teacher: `ai/data/t0_proxy_holdout_20260529/exact_late_teacher_t3t4_10000_20260529.jsonl`
- records: T3=20,000, T4=20,000.

Current model on exact late-turn 10k:

| turn | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | avg regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| T3 | 24.8% | 49.6% | 63.8% | 86.0% | 94.2% | 99.0% | 2.076 |
| T4 | 30.8% | 76.6% | 92.9% | 100.0% | 100.0% | 100.0% | 1.838 |

Severe exact Top1 misses extracted:

- `targets_top1_miss_t3t4_exact_10k_severe.jsonl`: 5,000 written
  - T3: 3,211
  - T4: 1,789

Two short T3/T4 adapter experiments were run from the current model:

1. `tutor-route10-exact-t3t4-adapter-20260531`
   - Training distribution: improves Top10/Top20.
   - Fixed T3 holdout: Top1 drops from 26.2% to 23.0%; not a Top1 upgrade.
2. `tutor-route10-exact-t3t4-top1-adapter-20260531`
   - Exact late 10k distribution: Top1 improves from 27.8% to 34.5%.
   - Fixed T3 holdout: Top1 drops from 26.2% to 23.6%; not deployable.

Conclusion: exact T3/T4 data is useful, but the adapter overfits the late-turn
10k distribution when trained directly. Do not adopt either adapter globally.
Use the exact data for active-loop mining and exact/refinement safety, then
train with a mixed distribution or stricter split before treating it as a model
upgrade.

## 2026-05-31 Mixed Exact T3/T4 Fine-Tune

A mixed fine-tune was run to avoid the direct adapter overfit:

- model: `tutor-route10-mixed-exact-t3t4-120k-ft-20260531`
- data: current active T1/T2/base mix plus 120k sampled exact T3/T4 samples.
- init: `tutor-route10-hybrid-t1t2-hard-mc300500-20260531`.

Fixed holdout deltas versus the current model:

| turn | Top1 delta | Top3 delta | Top10 delta | Top15 delta | Top20 delta | avg regret delta | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| T1 MC300 | -0.2 pp | +1.4 pp | +1.2 pp | +0.2 pp | +0.2 pp | -0.054 | pruning improves, Top1 does not |
| T2 MC300 | +2.0 pp | +2.8 pp | +2.0 pp | +0.0 pp | +0.0 pp | +0.048 | Top1 improves, but high-regret outlier worsens |
| T3 exact fixed | -0.4 pp | -2.6 pp | +1.0 pp | +1.0 pp | +0.4 pp | +0.063 | shortlist improves slightly, Top1/regret worsens |

Exact late-turn 10k evaluation:

| turn | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | avg regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| T3 | 28.7% | 55.7% | 70.0% | 90.0% | 96.2% | 99.5% | 1.798 |
| T4 | 33.0% | 78.1% | 93.4% | 100.0% | 100.0% | 100.0% | 1.634 |

Compared with the current model on the same exact late-turn 10k set:

- T3 Top1 improves from 24.8% to 28.7%, and T3 Top10 improves from 86.0% to 90.0%.
- T4 Top1 improves from 30.8% to 33.0%, and T4 Top10 stays 100%.
- However, fixed T3 holdout Top1/regret get worse, and T2 has a larger max-regret outlier.

Conclusion: this mixed model is not a clean global replacement. It is useful
evidence that exact T3/T4 data can improve shortlist recall, but it still cannot
be trusted as the final Top1 answer. The product path remains:

1. Model scores all legal actions.
2. Model keeps a shortlist that is large enough to avoid dropping the true best.
3. The synchronous 5 second budget reranks the shortlist with exact or partial
   exact evaluation.
4. Top1 misses after refinement become the next active-loop teacher targets.

For the user's goal, "Top1 100%" should mean final post-refinement Top1, not raw
model Top1. Raw model Top1 is still an optimization target, but not the only
safety mechanism.

## 2026-05-31 T2 5s Refinement Smoke

Runtime evaluation tooling was added:

- script: `ai/tutor/evaluate_hybrid_refinement_teacher.py`
- purpose: compare raw model Top1 against post-refinement Top1 on teacher JSONL
  records.

Bug fixed in `ai/tutor/hybrid_t1t2.py`:

- Before: T2 partial-exact scores and raw model scores were mixed in the same
  sort key, so an unrefined background candidate could become the returned
  `best`.
- After: when any T2 candidate has a refined score, the synchronous `best` is
  selected only from refined candidates. Background candidates stay in the
  response for logging/UI.

Local MC300 smoke set:

- input: `ai/data/hybrid_t1t2_active_20260531/local_smoke_t1t2_10_mc300.jsonl`
- model: `tutor-route10-hybrid-t1t2-hard-mc300500-20260531`
- output: `ai/data/hybrid_t1t2_active_20260531/eval_hybrid_refine_local_mc300_10_current_scale_fix`

Result:

| scope | decisions | model Top1 | final Top1 | model avg regret | final avg regret | p95 ms | overrides | improved | worse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 10 | 30.0% | 30.0% | 3.612 | 2.662 | 5021 | 4 | 3 | 1 |
| T1 | 4 | 50.0% | 50.0% | 0.488 | 0.488 | 13 | 0 | 0 | 0 |
| T2 | 6 | 16.7% | 16.7% | 5.695 | 4.112 | 5017 | 4 | 3 | 1 |

Interpretation:

- The current T2 partial-exact path lowers regret, so it is directionally useful.
- It does not yet improve Top1 on this small smoke set.
- One T2 case got worse because only a few sampled T3 draws overestimated a
  challenger action.
- Re-labeling that case at MC3000 kept the original model Top1 as best
  (`7.514` vs challenger `6.893`), confirming the partial-exact sample was too
  noisy for a hard Top1 override.
- T2 full exact was attempted for one position and did not finish in 5 minutes,
  so full exact is not viable synchronously.
- Existing Rust MC300 all-action labels for the six T2 smoke positions took
  3.3s to 4.4s each. This suggests a Rust MC refinement path may be more stable
  inside 5 seconds than the current Python-driven T3 exact sampling path.

Next T2 refinement experiment:

1. Keep the model shortlist/pool logic for training and logging.
2. Add a Rust-backed T2 MC refinement option that evaluates either all legal
   T2 actions at MC300 when time permits, or model TopK when a direct subset API
   is available.
3. Compare three synchronous policies on the same teacher set:
   - model only
   - current T3 exact-partial
   - Rust MC300 refinement
4. Promote only a policy that improves final Top1 or final regret without
   introducing high-regret override failures.

That Rust-backed TopK MC refinement option was added as:

- `--t2-refinement mc_board`
- `--t2-mc-sims 300`

It applies each synchronized T2 candidate to the board and calls Rust
`prob_engine` batch `board_mc` for the post-action board. This keeps refinement
on the model-selected TopK instead of all legal actions.

MC300 smoke comparison on the same 10 local records:

- output: `ai/data/hybrid_t1t2_active_20260531/eval_hybrid_refine_local_mc300_10_current_mcboard`

| scope | decisions | model Top1 | final Top1 | model avg regret | final avg regret | p95 ms | overrides | improved | worse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 10 | 30.0% | 50.0% | 3.612 | 1.885 | 394 | 4 | 4 | 0 |
| T2 | 6 | 16.7% | 50.0% | 5.695 | 2.816 | 394 | 4 | 4 | 0 |

Larger noisy MC30 smoke:

- input: `teacher_t1t2_neighbors_100_mc30.jsonl`
- output: `ai/data/hybrid_t1t2_active_20260531/eval_hybrid_refine_neighbors100_mc30_current_mcboard`

| scope | decisions | model Top1 | final Top1 | model avg regret | final avg regret | p95 ms | overrides | improved | worse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 100 | 30.0% | 38.0% | 3.086 | 2.442 | 473 | 26 | 19 | 6 |
| T2 | 50 | 34.0% | 50.0% | 2.775 | 1.489 | 673 | 26 | 19 | 6 |

The MC30 comparison should not be treated as final truth because those teacher
labels are intentionally noisy. The important signal is speed and average-regret
direction. The next high-confidence check should use MC300 or better teacher
records, and exact T3/T4 labels where applicable.

## 2026-05-31 MC300 Local500 Refinement Check

A larger local MC300 teacher set already existed:

- teacher: `ai/data/hybrid_t1t2_active_20260531/local_t1t2_500_mc300.jsonl`
- records: 500 total, T1=246, T2=254
- generation cost: 5,771.7s locally with Rust `prob_engine` batch mode

Model-only baseline:

- output: `ai/data/hybrid_t1t2_active_20260531/eval_hybrid_refine_local500_mc300_current_model_only`

| scope | decisions | Top1 | avg regret | max regret | p95 ms |
|---|---:|---:|---:|---:|---:|
| overall | 500 | 34.6% | 2.233 | 40.350 | 5 |
| T1 | 246 | 34.6% | 2.188 | 17.067 | 5 |
| T2 | 254 | 34.6% | 2.275 | 40.350 | 6 |

MC board refinement on all 500 records:

- output: `ai/data/hybrid_t1t2_active_20260531/eval_hybrid_refine_local500_mc300_current_mcboard`

| scope | decisions | model Top1 | final Top1 | final avg regret | final max regret | p95 ms | overrides | improved | worse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 500 | 34.6% | 47.0% | 1.488 | 17.067 | 677 | 118 | 118 | 0 |
| T2 | 254 | 34.6% | 59.1% | 0.809 | 13.489 | 792 | 118 | 118 | 0 |

On the first 50 MC300 records, three policies were compared:

| policy | T2 decisions | T2 final Top1 | T2 avg regret | T2 max regret | T2 p95 ms | improved overrides | worse overrides |
|---|---:|---:|---:|---:|---:|---:|---:|
| model only | 26 | 26.9% | 3.184 | 16.312 | 9 | 0 | 0 |
| exact_partial | 26 | 42.3% | 1.831 | 10.850 | 5,032 | 11 | 2 |
| mc_board | 26 | 65.4% | 0.850 | 5.923 | 705 | 16 | 0 |

Conclusion:

- For T2, `mc_board` is currently a better synchronous refinement path than
  the existing Python-driven T3 `exact_partial`: it is faster, improves Top1
  more, and had no bad overrides on the MC300 checks above.
- This does not make raw model Top1 good enough. It makes the 5-second final
  answer materially stronger while active learning improves the model.

Active-loop targets were extracted from the 500-record MC300 run after
`mc_board` refinement:

- output: `ai/data/hybrid_t1t2_active_20260531/targets_runtime_final_miss_local500_mc300_mcboard_margin1.jsonl`
- filter: final Top1 miss, teacher margin >= 1.0, regret >= 0.1, no suit augmentation
- written: 85 targets
  - T1: 62
  - T2: 23

An MC3000 relabel smoke was attempted locally:

- output: `ai/data/hybrid_t1t2_active_20260531/teacher_runtime_final_miss_local500_mc3000_10.jsonl`
- request: first 10 active-loop targets
- result: timed out after 10 minutes with 6 records written

Conclusion: MC3000 relabeling is too slow locally for meaningful batches. Keep
local MC3000 only for spot checks. Larger high-confidence relabeling should run
as a resumable batch job, preferably on GCP, after the target extraction and
policy are stable.

## 2026-05-31 Top1 Target and T2 Runtime-Miss Fine Tune

The product target is final Top1 correctness. Top15/Top20 is only an internal
pool for safe pruning and refinement; it is not an acceptable final answer by
itself.

For T2, the remaining misses after `mc_board` refinement were high-value active
learning targets:

- source targets: `ai/data/hybrid_t1t2_active_20260531/targets_runtime_final_miss_local500_mc300_mcboard_margin1.jsonl`
- T2 targets: 23
- local MC3000 relabel output: `ai/data/hybrid_t1t2_active_20260531/teacher_runtime_final_miss_t2_mc3000_23_20260531.jsonl`
- relabel runtime: 23/23 completed in 468s locally
- suit augmentation: 23 records -> 552 records
- converted samples: 11,664
- mixed training data: `ai/data/hybrid_t1t2_active_20260531/reranker_mix_current_plus_t2_mc3000_runtime_miss_suit24_20260531`
- fine-tuned model: `ai/models/candidate_runs/tutor-route10-t2-runtime-miss-mc3000-suit24-ft-20260531/model/action_value_best.pt`

Fixed T2 MC300 holdout improved:

| model | Top1 | Top3 | Top5 | Top10 | Top15 | avg regret | Top10 rerank max regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| current hard MC300/500 | 32.4% | 59.4% | 72.6% | 90.4% | 97.6% | 2.736 | 24.545 |
| + T2 runtime-miss MC3000 suit24 FT | 37.0% | 65.0% | 79.0% | 94.8% | 98.0% | 2.547 | 3.336 |

T1 was not trained directly, but shared weights changed slightly:

| model | T1 Top1 | T1 Top10 | T1 avg regret | T1 score MAE | T1 FL MAE |
|---|---:|---:|---:|---:|---:|
| current hard MC300/500 | 35.2% | 92.2% | 2.124 | 3.144 | 0.107 |
| + T2 runtime-miss MC3000 suit24 FT | 35.4% | 95.0% | 2.044 | 3.640 | 0.142 |

The operational 5-second path improved more clearly:

- output: `ai/data/hybrid_t1t2_active_20260531/eval_hybrid_refine_local500_mc300_t2_runtime_miss_mc3000_suit24_ft_mcboard`
- refinement: T2 `mc_board`, 300 sims per synchronized candidate
- time budget violations: 0

| scope | decisions | model Top1 | final Top1 | final avg regret | final max regret | p95 ms | overrides | improved | worse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall current | 500 | 34.6% | 47.0% | 1.488 | 17.067 | 677 | 118 | 118 | 0 |
| overall + FT | 500 | 37.6% | 52.4% | 1.210 | 17.067 | 668 | 124 | 124 | 0 |
| T2 current | 254 | 34.6% | 59.1% | 0.809 | 13.489 | 792 | 118 | 118 | 0 |
| T2 + FT | 254 | 39.0% | 68.1% | 0.448 | 12.091 | 776 | 124 | 124 | 0 |

Remaining final Top1 misses were extracted for the next active-learning loop:

- output: `ai/data/hybrid_t1t2_active_20260531/targets_runtime_final_miss_local500_mc300_mcboard_after_t2mc3000ft_margin1.jsonl`
- written: 71 targets
  - T1: 59
  - T2: 12

Next action:

1. Do not chase Top15 as the final metric. Keep it only as the prune/refine
   safety pool.
2. Relabel the remaining T2 final misses at higher confidence first; there are
   only 12 after this fine tune.
3. For T1, local MC3000 takes about 186s per record, so use either a smaller
   local probe ladder or a resumable GCP batch.
4. Add a T1 `mc_board` or recursive shallow refinement path, because current
   synchronous refinement only helps T2.

## 2026-05-31 Second T2 Runtime-Miss Loop

The remaining T2 final misses after the first fine tune were relabeled locally:

- source targets: `ai/data/hybrid_t1t2_active_20260531/targets_runtime_final_miss_local500_mc300_mcboard_after_t2mc3000ft_margin1.jsonl`
- T2 targets: 12
- local MC3000 relabel output: `ai/data/hybrid_t1t2_active_20260531/teacher_runtime_final_miss_after_t2mc3000ft_t2_mc3000_12_20260531.jsonl`
- relabel runtime: 12/12 completed in 226.5s locally
- suit augmentation: 12 records -> 288 records
- converted samples: 5,184
- mixed training data: `ai/data/hybrid_t1t2_active_20260531/reranker_mix_current_plus_t2_mc3000_runtime_miss_35_suit24_20260531`
- fine-tuned model: `ai/models/candidate_runs/tutor-route10-t2-runtime-miss-mc3000-35-suit24-ft2-20260531/model/action_value_best.pt`

Fixed T2 MC300 holdout after the second loop:

| model | Top1 | Top3 | Top5 | Top10 | Top15 | avg regret | Top10 rerank max regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| current hard MC300/500 | 32.4% | 59.4% | 72.6% | 90.4% | 97.6% | 2.736 | 24.545 |
| + first T2 runtime-miss FT | 37.0% | 65.0% | 79.0% | 94.8% | 98.0% | 2.547 | 3.336 |
| + second T2 runtime-miss FT | 38.4% | 66.8% | 81.4% | 95.2% | 98.2% | 2.543 | 3.336 |

Operational 5-second path after the second loop:

- output: `ai/data/hybrid_t1t2_active_20260531/eval_hybrid_refine_local500_mc300_t2_runtime_miss_mc3000_35_suit24_ft2_mcboard`
- time budget violations: 0

| scope | model | model Top1 | final Top1 | final avg regret | final max regret | p95 ms | improved overrides | worse overrides |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| overall | current | 34.6% | 47.0% | 1.488 | 17.067 | 677 | 118 | 0 |
| overall | first FT | 37.6% | 52.4% | 1.210 | 17.067 | 668 | 124 | 0 |
| overall | second FT | 39.2% | 54.2% | 1.180 | 17.067 | 644 | 121 | 0 |
| T2 | current | 34.6% | 59.1% | 0.809 | 13.489 | 792 | 118 | 0 |
| T2 | first FT | 39.0% | 68.1% | 0.448 | 12.091 | 776 | 124 | 0 |
| T2 | second FT | 40.9% | 70.5% | 0.392 | 12.091 | 790 | 121 | 0 |

Remaining final Top1 misses after the second loop:

- output: `ai/data/hybrid_t1t2_active_20260531/targets_runtime_final_miss_local500_mc300_mcboard_after_t2mc3000ft2_margin1.jsonl`
- written: 70 targets
  - T1: 59
  - T2: 11

Conclusion:

- The active loop is working for T2: model Top1, final Top1, and regret all
  improved while staying comfortably inside the 5-second limit.
- Returns are already diminishing on this small local validation set. T2 is no
  longer the only blocker; T1 has 59 high-confidence remaining misses and no
  synchronous refinement path yet.
- To push final Top1 toward 100%, the next highest-leverage work is T1
  refinement plus higher-confidence labels, not only more T2 micro-fine-tuning.

## 2026-05-31 T1 Synchronous Recursive Refinement Prototype

T1 now has an optional 5-second refinement path:

- Python option: `--t1-refinement recursive_mc`
- T1 post-action board is evaluated by Rust `prob_engine` batch mode
  `board_recursive_mc`.
- Current tested setting: `--t1-mc-sims 32 --t1-recursive-beam 5
  --t1-recursive-child-sims 4`
- Rust batch JSONL was extended to accept `recursive_beam` and
  `recursive_child_sims` per request.

Small T1-only probes on the first 100 input lines, 47 T1 decisions:

| T1 policy | final Top1 | final avg regret | final max regret | p95 ms | overrides | improved | worse |
|---|---:|---:|---:|---:|---:|---:|---:|
| model only | 44.7% | 1.828 | 17.067 | ~17 | 0 | 0 | 0 |
| recursive 16 / beam5 / child2 | 51.1% | 1.399 | 10.203 | 482 | 17 | 9 | 8 |
| recursive 32 / beam5 / child2 | 55.3% | 0.992 | 9.657 | 891 | 18 | 13 | 5 |
| recursive 64 / beam5 / child2 | 55.3% | 0.947 | 9.657 | 1,774 | 18 | 13 | 5 |
| recursive 32 / beam5 / child4 | 57.4% | 1.122 | 10.203 | 2,155 | 15 | 12 | 3 |
| recursive 64 / beam5 / child4 | 55.3% | 0.940 | 9.657 | 4,148 | 16 | 12 | 4 |

The best small-probe Top1 setting was `32 / beam5 / child4`, while `64`
variants pushed closer to the 5-second budget without improving Top1.

Full local500 run with T1 recursive plus T2 `mc_board`:

- output: `ai/data/hybrid_t1t2_active_20260531/eval_hybrid_refine_local500_mc300_t1_recursive32_child4_t2_mcboard_ft2`
- model: `ai/models/candidate_runs/tutor-route10-t2-runtime-miss-mc3000-35-suit24-ft2-20260531/model/action_value_best.pt`
- time budget violations: 0

| scope | model Top1 | final Top1 | final avg regret | final max regret | p95 ms | overrides | improved | worse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall, no T1 refine | 39.2% | 54.2% | 1.180 | 17.067 | 644 | 121 | 121 | 0 |
| overall, T1 recursive + T2 mc | 39.2% | 59.4% | 0.707 | 12.091 | 1,581 | 227 | 204 | 23 |
| T1, no T1 refine | 37.4% | 37.4% | 1.994 | 17.067 | 15 | 0 | 0 | 0 |
| T1, recursive 32/5/4 | 37.4% | 48.0% | 1.033 | 11.833 | 1,837 | 106 | 83 | 23 |
| T2, mc_board | 40.9% | 70.5% | 0.392 | 12.091 | 811 | 121 | 121 | 0 |

Remaining final Top1 misses after T1 recursive + T2 mc:

- output: `ai/data/hybrid_t1t2_active_20260531/targets_runtime_final_miss_local500_mc300_t1rec32child4_t2mcboard_ft2_margin1.jsonl`
- written: 54 targets
  - T1: 43
  - T2: 11

Conclusion:

- T1 recursive refinement is fast enough locally: p95 1.84s for T1 and no
  5-second violations.
- It materially improves Top1 and regret, but unlike T2 it still has bad
  overrides. That makes T1 the next active-learning target.
- Before promoting T1 recursive refinement as a production default, the bad
  override cases should be relabeled with stronger teachers and used for
  training or override gating.

## 2026-05-31 T1 Bad-Override Active Targets

The T1 bad overrides from the full local500 run are now explicitly extractable:

- script: `ai/tutor/hybrid_refinement_misses_to_targets.py`
- new filters:
  - `--override-worse-only`
  - `--min-override-delta`
- output: `ai/data/hybrid_t1t2_active_20260531/targets_t1_bad_override_local500_mc300_t1rec32child4_ft2.jsonl`
- written: 23 T1 bad-override targets

A reusable comparison script was added:

- script: `ai/tutor/compare_runtime_targets_to_teacher.py`
- purpose: compare the runtime model action and runtime final action against a
  stronger relabeled teacher file.

Local MC1000 probe on the first 3 T1 bad overrides:

- output: `ai/data/hybrid_t1t2_active_20260531/teacher_t1_bad_override_mc1000_probe3_20260531.jsonl`
- runtime: 3/3 completed in 178.6s
- per-target time: 43.7s, 72.4s, 60.8s
- comparison output: `ai/data/hybrid_t1t2_active_20260531/compare_t1_bad_override_mc1000_probe3_20260531.jsonl`

Comparison against the stronger MC1000 labels:

| records | model found | final found | model best | final best | model > final | final > model | avg model regret | avg final regret |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 3 | 3 | 0 | 0 | 2 | 1 | 5.317 | 5.553 |

Interpretation:

- Two of the first three bad overrides are still bad under MC1000.
- One flips direction, confirming that MC300 can mislabel close T1 decisions.
- In all three cases, neither model Top1 nor runtime final was the MC1000 best
  action, so the target should also improve candidate ranking, not only
  override gating.

The MC1000 probe was suit-augmented and converted successfully:

- augmented teacher: `ai/data/hybrid_t1t2_active_20260531/teacher_t1_bad_override_mc1000_probe3_20260531_suit24.jsonl`
- converted data: `ai/data/hybrid_t1t2_active_20260531/reranker_t1_bad_override_mc1000_probe3_suit24_20260531`
- records: 72
- samples: 1,872

Next action:

1. Relabel all 23 T1 bad overrides at MC1000 locally if a 20-30 minute run is
   acceptable, or run them as a resumable GCP batch.
2. Convert and add them as a T1-focused active-loop fine tune.
3. Re-evaluate T1 recursive refinement and bad overrides. Promote T1 recursive
   only if bad override count drops materially or a reliable guard is learned.

## 2026-05-31 T1 Bad-Override MC1000 Fine Tune

All 23 T1 bad overrides were relabeled locally at MC1000:

- input targets: `ai/data/hybrid_t1t2_active_20260531/targets_t1_bad_override_local500_mc300_t1rec32child4_ft2.jsonl`
- chunk dir: `ai/data/hybrid_t1t2_active_20260531/chunk_probe_t1_bad_override_mc1000`
- merged teacher: `ai/data/hybrid_t1t2_active_20260531/teacher_t1_bad_override_mc1000_23_20260531.jsonl`
- completed: 23/23
- first 3 chunks were reused from the probe via `.done` markers
- remaining chunk times ranged from 17.0s to 69.8s

MC1000 comparison for the 23 bad overrides:

- comparison: `ai/data/hybrid_t1t2_active_20260531/compare_t1_bad_override_mc1000_23_20260531.jsonl`

| records | model best | final best | model > final | final > model | avg model regret | avg final regret |
|---:|---:|---:|---:|---:|---:|---:|
| 23 | 6 | 2 | 13 | 10 | 1.335 | 1.741 |

Interpretation:

- Most bad overrides are real under MC1000, but 10/23 reverse direction from
  the MC300 judgment. That confirms MC300 should not be used as a hard Top1
  truth source for close T1 decisions.
- In many cases neither runtime model Top1 nor runtime final is the MC1000
  best, so this also targets candidate ranking quality.

The MC1000 labels were added to training:

- augmented teacher: `ai/data/hybrid_t1t2_active_20260531/teacher_t1_bad_override_mc1000_23_20260531_suit24.jsonl`
- augmented records: 552
- converted data: `ai/data/hybrid_t1t2_active_20260531/reranker_t1_bad_override_mc1000_23_suit24_20260531`
- converted samples: 12,960
- mixed data: `ai/data/hybrid_t1t2_active_20260531/reranker_mix_current_plus_t2_35_and_t1_bad_override_mc1000_23_suit24_20260531`
- fine-tuned model: `ai/models/candidate_runs/tutor-route10-t1-badoverride-mc1000-23-suit24-ft-20260531/model/action_value_best.pt`
- init: `tutor-route10-t2-runtime-miss-mc3000-35-suit24-ft2-20260531`
- train turns: T1 only

Fixed holdout effects:

| scope | previous model Top1 | new model Top1 | previous Top10 | new Top10 | previous Top15 | new Top15 |
|---|---:|---:|---:|---:|---:|---:|
| T1 MC300 holdout | 36.6% | 35.6% | 94.4% | 94.8% | 98.4% | 99.2% |
| T2 MC300 holdout | 38.4% | 38.2% | 95.2% | 95.6% | 98.2% | 99.0% |

Operational 5-second path:

- output: `ai/data/hybrid_t1t2_active_20260531/eval_hybrid_refine_local500_mc300_t1_badoverride_mc1000_23_suit24_ft_t1rec32child4_t2mcboard`
- time budget violations: 0

| scope | previous final Top1 | new final Top1 | previous avg regret | new avg regret | previous bad overrides | new bad overrides | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 59.4% | 60.4% | 0.707 | 0.626 | 23 | 19 | 1,527 |
| T1 | 48.0% | 51.6% | 1.033 | 0.878 | 23 | 19 | 1,748 |
| T2 | 70.5% | 68.9% | 0.392 | 0.383 | 0 | 0 | 753 |

Remaining targets after this fine tune:

- final misses: `ai/data/hybrid_t1t2_active_20260531/targets_runtime_final_miss_local500_mc300_after_t1_badoverride_ft_margin1.jsonl`
  - written: 48
  - T1: 37
  - T2: 11
- bad overrides: `ai/data/hybrid_t1t2_active_20260531/targets_t1_bad_override_after_t1_badoverride_ft.jsonl`
  - written: 19 T1 bad overrides

Conclusion:

- The T1 bad-override loop improved the operational metric, especially T1
  final Top1 and regret.
- It slightly regressed T2 final Top1, so this checkpoint should stay a
  candidate, not an unconditional production replacement.
- The next loop should use the new 19 T1 bad overrides plus the 37 T1 final
  misses, preferably with stronger labels and a model-update recipe that
  preserves T2 behavior better.

## 2026-05-31 T1 Union49 and Joint T1/T2 Low-LR Fine Tune

The next T1 active set combined:

- 37 T1 final misses after the bad-override fine tune
- 19 T1 bad overrides after the bad-override fine tune
- 49 unique T1 states after de-duplication
- 30 new states required MC1000 relabeling because 19 were already covered

Artifacts:

- target union: `ai/data/hybrid_t1t2_active_20260531/targets_t1_final_miss_plus_bad_override_after_t1_badoverride_ft_union49.jsonl`
- new MC1000 target subset: `ai/data/hybrid_t1t2_active_20260531/targets_t1_final_miss_plus_bad_override_after_t1_badoverride_ft_new30_mc1000_needed.jsonl`
- merged MC1000 teacher: `ai/data/hybrid_t1t2_active_20260531/teacher_t1_finalmiss_badoverride_union49_mc1000_20260531.jsonl`
- augmented teacher: `ai/data/hybrid_t1t2_active_20260531/teacher_t1_finalmiss_badoverride_union49_mc1000_20260531_suit24.jsonl`
- converted data: `ai/data/hybrid_t1t2_active_20260531/reranker_t1_finalmiss_badoverride_union49_mc1000_suit24_20260531`
- mixed data: `ai/data/hybrid_t1t2_active_20260531/reranker_mix_current_plus_t2_35_and_t1_union49_mc1000_suit24_20260531`

MC1000 comparison on the 49 states:

| records | model best | final best | model > final | final > model | tie | avg model regret | avg final regret |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 49 | 8 | 2 | 13 | 19 | 17 | 3.502 | 2.862 |

Two update recipes were tested:

| model | train recipe | overall final Top1 | T1 final Top1 | T2 final Top1 | overall avg regret | T1 avg regret | T2 avg regret | bad overrides |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `tutor-route10-t1-union49-mc1000-suit24-turnadapter-freeze-20260531` | T1 adapters/head only, shared frozen | 59.8% | 48.8% | 70.5% | 0.688 | 0.993 | 0.392 | 20 |
| `tutor-route10-t1-union49-mc1000-suit24-joint-t12-lr1e6-20260531` | joint T1/T2, lr 1e-6 | 61.2% | 50.4% | 71.7% | 0.650 | 0.914 | 0.394 | 24 |

Interpretation:

- The joint low-LR model is the best overall 5-second Top1 checkpoint so far
  on the local500 MC300 operational set.
- The earlier bad-override model still has the best T1-only final Top1
  measured here: 51.6% T1 final Top1 with 19 bad overrides.
- The joint model improves T2 to 71.7% final Top1 but increases T1 bad
  overrides to 24, so it is better as a broad checkpoint than as the final T1
  answer.
- Pool20/24 is only a safety pool for refinement and data collection. It is
  not the target behavior. The target behavior remains Top1 correctness; the
  pool exists only to avoid throwing away the true action before refinement.

Remaining targets from the joint checkpoint:

- final misses: `ai/data/hybrid_t1t2_active_20260531/targets_runtime_final_miss_local500_mc300_after_t1_union49_joint_t12_lr1e6_margin1.jsonl`
  - written: 49
  - T1: 39
  - T2: 10
- T1 bad overrides: `ai/data/hybrid_t1t2_active_20260531/targets_t1_bad_override_after_t1_union49_joint_t12_lr1e6.jsonl`
  - written: 24

Next action:

- Prioritize the 24 T1 bad overrides, because these are cases where the 5s
  refinement actively makes Top1 worse.
- Then relabel the 49 final misses with stronger labels and train with a recipe
  that preserves the T2 gain while reducing T1 bad overrides.
- For the product path, treat Top1 as the target and Top15/20 as a temporary
  internal safety mechanism, not as acceptable final accuracy.

## 2026-05-31 T1 Override Guard

Top1 correctness is the target. The shortlist pool is not the answer; it only
keeps the true action alive for refinement and logging. For T1, recursive MC
can improve many positions but can also override a correct or better model Top1
because the refinement samples are still noisy. A small T1 override guard was
added:

- config: `t1_override_margin`
- behavior: on T1, keep model Top1 unless the refined best candidate beats
  model Top1 by at least this refined-score margin
- default: `0.0`, preserving existing behavior unless explicitly enabled
- extra logging: `model_top1_refined_score`, `best_refined_score`,
  `refined_override_delta`

The latest joint checkpoint was re-evaluated with delta logging:

- output: `ai/data/hybrid_t1t2_active_20260531/eval_hybrid_refine_local500_mc300_t1_union49_joint_t12_lr1e6_t1rec32child4_t2mcboard_with_deltas`

Offline threshold replay on the same refined candidates showed:

| T1 override margin | overall Top1 | T1 Top1 | T2 Top1 | overall regret | T1 regret | bad overrides |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 61.2% | 50.4% | 71.7% | 0.650 | 0.914 | 24 |
| 0.05 | 61.6% | 51.2% | 71.7% | 0.647 | 0.908 | 20 |
| 0.20 | 61.4% | 50.8% | 71.7% | 0.644 | 0.903 | 16 |
| 0.50 | 61.0% | 50.0% | 71.7% | 0.635 | 0.883 | 13 |
| 1.00 | 61.0% | 50.0% | 71.7% | 0.646 | 0.907 | 9 |

Actual runtime evaluation with `t1_override_margin=0.05`:

- output: `ai/data/hybrid_t1t2_active_20260531/eval_hybrid_refine_local500_mc300_t1_union49_joint_t12_lr1e6_t1rec32child4_t2mcboard_margin005`
- time budget violations: 0

| scope | final Top1 | avg regret | overrides | improved | worse | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| overall | 61.6% | 0.647 | 218 | 198 | 20 | 1,557 |
| T1 | 51.2% | 0.908 | 103 | 83 | 20 | 1,810 |
| T2 | 71.7% | 0.394 | 115 | 115 | 0 | 785 |

Remaining targets under the guarded setting:

- final misses: `ai/data/hybrid_t1t2_active_20260531/targets_runtime_final_miss_local500_mc300_after_t1_union49_joint_margin005.jsonl`
  - written: 49
  - T1: 39
  - T2: 10
- T1 bad overrides: `ai/data/hybrid_t1t2_active_20260531/targets_t1_bad_override_after_t1_union49_joint_margin005.jsonl`
  - written: 20

Conclusion:

- `t1_override_margin=0.05` is the current best Top1 setting on this local500
  MC300 operational check.
- Larger margins reduce bad overrides further but give back too many real
  improvements, so they are better as a regret-control experiment than as the
  Top1-first setting.
- Next data generation should target the remaining 20 guarded T1 bad overrides
  plus the 49 final misses with stronger labels.

## 2026-05-31 Margin005 Active61 Loop

The guarded `margin=0.05` runtime left:

- 49 final misses
- 20 T1 bad overrides
- 61 unique states after de-duplication

Most of these states were already covered by MC1000 labels. Only 4 new T1
states required local MC1000 generation:

- missing target input: `ai/data/hybrid_t1t2_active_20260531/targets_runtime_final_miss_local500_mc300_after_t1_union49_joint_margin005_mc1000_missing.jsonl`
- chunk dir: `ai/data/hybrid_t1t2_active_20260531/chunk_finalmiss_after_margin005_missing4_mc1000`
- new teacher: `ai/data/hybrid_t1t2_active_20260531/teacher_finalmiss_after_margin005_missing4_mc1000_20260531.jsonl`
- completed: 4/4
- chunk times: 111.3s and 98.8s

The merged active teacher set:

- teacher: `ai/data/hybrid_t1t2_active_20260531/teacher_margin005_finalmiss49_badoverride20_mc1000_20260531.jsonl`
- unique records: 61
- T1: 51
- T2: 10
- suit24 teacher: `ai/data/hybrid_t1t2_active_20260531/teacher_margin005_finalmiss49_badoverride20_mc1000_20260531_suit24.jsonl`
- suit24 records: 1,464
- converted data: `ai/data/hybrid_t1t2_active_20260531/reranker_margin005_finalmiss49_badoverride20_mc1000_suit24_20260531`
- converted samples: 34,704
- mixed data: `ai/data/hybrid_t1t2_active_20260531/reranker_mix_current_plus_margin005_active61_mc1000_suit24_20260531`
- mixed samples: 243,291

MC1000 comparison:

| target set | records | model best | final best | model > final | final > model | tie | avg model regret | avg final regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bad overrides | 20 | 6 | 2 | 12 | 8 | 0 | 0.987 | 1.863 |
| final misses | 49 | 5 | 1 | 7 | 19 | 23 | 4.485 | 3.420 |

Interpretation:

- The bad override set still mostly says "do not override this T1 model Top1".
- The final-miss set is mixed: many misses are not simply bad overrides, and
  refinement often helps compared with the raw model.

Fine tune:

- model: `ai/models/candidate_runs/tutor-route10-margin005-active61-mc1000-suit24-joint-t12-lr5e7-20260531/model/action_value_best.pt`
- init: `tutor-route10-t1-union49-mc1000-suit24-joint-t12-lr1e6-20260531`
- train turns: T1/T2
- lr: 5e-7

Fixed MC300 holdout:

| scope | previous Top1 | new Top1 | previous Top10 | new Top10 | previous Top15 | new Top15 |
|---|---:|---:|---:|---:|---:|---:|
| T1 | 37.0% | 36.4% | 94.8% | 94.6% | 98.8% | 98.8% |
| T2 | 39.0% | 38.4% | 95.4% | 95.4% | 98.8% | 98.8% |

Operational 5-second path with `t1_override_margin=0.05`:

- output: `ai/data/hybrid_t1t2_active_20260531/eval_hybrid_refine_local500_mc300_margin005_active61_joint_lr5e7_t1rec32child4_t2mcboard_margin005`
- time budget violations: 0

| scope | previous final Top1 | new final Top1 | previous avg regret | new avg regret | previous bad overrides | new bad overrides | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 61.6% | 62.2% | 0.647 | 0.615 | 20 | 21 | 1,575 |
| T1 | 51.2% | 52.4% | 0.908 | 0.849 | 20 | 21 | 1,776 |
| T2 | 71.7% | 71.7% | 0.394 | 0.388 | 0 | 0 | 802 |

Remaining targets:

- final misses: `ai/data/hybrid_t1t2_active_20260531/targets_runtime_final_miss_local500_mc300_after_margin005_active61_joint_lr5e7.jsonl`
  - written: 46
  - T1: 37
  - T2: 9
- T1 bad overrides: `ai/data/hybrid_t1t2_active_20260531/targets_t1_bad_override_after_margin005_active61_joint_lr5e7.jsonl`
  - written: 21

Conclusion:

- This is the current best operational checkpoint by Top1 and average regret.
- The regression is concentrated in T1 bad overrides, so the next loop should
  target the 21 bad overrides specifically, and should also test a slightly
  stronger T1 guard or a learned override gate instead of relying on model
  fine-tuning alone.

## 2026-05-31 Active61 Guard Retune

After the active61 fine tune, the T1 override margin was re-tuned because the
model changed. A no-guard run was generated to capture `refined_override_delta`:

- no-guard output: `ai/data/hybrid_t1t2_active_20260531/eval_hybrid_refine_local500_mc300_margin005_active61_joint_lr5e7_t1rec32child4_t2mcboard_margin0`

Offline replay over the same refined candidates:

| T1 override margin | overall Top1 | T1 Top1 | T2 Top1 | overall regret | T1 regret | bad overrides | overrides |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 61.8% | 51.6% | 71.7% | 0.618 | 0.855 | 25 | 235 |
| 0.03 | 62.2% | 52.4% | 71.7% | 0.616 | 0.852 | 22 | 230 |
| 0.04 | 62.4% | 52.8% | 71.7% | 0.615 | 0.849 | 21 | 229 |
| 0.05 | 62.2% | 52.4% | 71.7% | 0.615 | 0.849 | 21 | 228 |
| 0.10 | 62.0% | 52.0% | 71.7% | 0.615 | 0.849 | 18 | 222 |
| 0.50 | 61.6% | 51.2% | 71.7% | 0.603 | 0.824 | 13 | 209 |
| 1.00 | 61.6% | 51.2% | 71.7% | 0.620 | 0.860 | 8 | 196 |

Actual runtime evaluation with `t1_override_margin=0.04`:

- output: `ai/data/hybrid_t1t2_active_20260531/eval_hybrid_refine_local500_mc300_margin005_active61_joint_lr5e7_t1rec32child4_t2mcboard_margin004`
- time budget violations: 0

| scope | final Top1 | avg regret | overrides | improved | worse | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| overall | 62.4% | 0.615 | 229 | 207 | 21 | 1,562 |
| T1 | 52.8% | 0.849 | 108 | 86 | 21 | 1,762 |
| T2 | 71.7% | 0.388 | 121 | 121 | 0 | 778 |

Remaining targets under `margin=0.04`:

- final misses: `ai/data/hybrid_t1t2_active_20260531/targets_runtime_final_miss_local500_mc300_after_margin005_active61_joint_lr5e7_margin004.jsonl`
  - written: 46
  - T1: 37
  - T2: 9
- T1 bad overrides: `ai/data/hybrid_t1t2_active_20260531/targets_t1_bad_override_after_margin005_active61_joint_lr5e7_margin004.jsonl`
  - written: 21

Conclusion:

- `t1_override_margin=0.04` is now the best Top1-first runtime setting for the
  active61 checkpoint.
- If prioritizing lower regret over Top1, margins around `0.5` are worth
  testing separately, but they give back Top1 accuracy.
- The next active loop should target the 21 bad overrides plus 46 final misses,
  ideally with a learned override gate or stronger T1 labels rather than only
  more low-LR fine-tuning.

## 2026-05-31 Margin004 Active59 Loop

The `margin=0.04` runtime left:

- 46 final misses
- 21 T1 bad overrides
- 59 unique states after de-duplication

Only 2 bad-override states were missing MC1000 labels. They were generated
locally:

- missing target input: `ai/data/hybrid_t1t2_active_20260531/targets_t1_bad_override_after_margin005_active61_joint_lr5e7_margin004_mc1000_missing.jsonl`
- chunk dir: `ai/data/hybrid_t1t2_active_20260531/chunk_badoverride_after_margin004_missing2_mc1000`
- new teacher: `ai/data/hybrid_t1t2_active_20260531/teacher_badoverride_after_margin004_missing2_mc1000_20260531.jsonl`
- completed: 2/2
- chunk time: 36.8s

The merged active teacher set:

- teacher: `ai/data/hybrid_t1t2_active_20260531/teacher_margin004_finalmiss46_badoverride21_mc1000_20260531.jsonl`
- unique records: 59
- T1: 50
- T2: 9
- suit24 teacher: `ai/data/hybrid_t1t2_active_20260531/teacher_margin004_finalmiss46_badoverride21_mc1000_20260531_suit24.jsonl`
- suit24 records: 1,416
- converted data: `ai/data/hybrid_t1t2_active_20260531/reranker_margin004_finalmiss46_badoverride21_mc1000_suit24_20260531`
- converted samples: 33,048
- mixed data: `ai/data/hybrid_t1t2_active_20260531/reranker_mix_current_plus_margin004_active59_mc1000_suit24_20260531`
- mixed samples: 276,339

MC1000 comparison:

| target set | records | model best | final best | model > final | final > model | tie | avg model regret | avg final regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bad overrides | 21 | 6 | 2 | 13 | 8 | 0 | 0.921 | 1.830 |
| final misses | 46 | 5 | 1 | 7 | 17 | 22 | 4.206 | 3.304 |

Fine tune:

- model: `ai/models/candidate_runs/tutor-route10-margin004-active59-mc1000-suit24-joint-t12-lr3e7-20260531/model/action_value_best.pt`
- init: `tutor-route10-margin005-active61-mc1000-suit24-joint-t12-lr5e7-20260531`
- train turns: T1/T2
- lr: 3e-7

Fixed MC300 holdout:

| scope | active61 Top1 | active59 Top1 | active61 Top10 | active59 Top10 | active61 Top15 | active59 Top15 |
|---|---:|---:|---:|---:|---:|---:|
| T1 | 36.4% | 37.2% | 94.6% | 94.8% | 98.8% | 98.8% |
| T2 | 38.4% | 38.4% | 95.4% | 95.4% | 98.8% | 99.0% |

Operational 5-second path with `t1_override_margin=0.04`:

- output: `ai/data/hybrid_t1t2_active_20260531/eval_hybrid_refine_local500_mc300_margin004_active59_joint_lr3e7_t1rec32child4_t2mcboard_margin004`
- time budget violations: 0

| scope | active61 final Top1 | active59 final Top1 | active61 avg regret | active59 avg regret | active61 bad overrides | active59 bad overrides | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 62.4% | 61.8% | 0.615 | 0.623 | 21 | 19 | 1,548 |
| T1 | 52.8% | 52.4% | 0.849 | 0.841 | 21 | 19 | 1,740 |
| T2 | 71.7% | 70.9% | 0.388 | 0.411 | 0 | 0 | 817 |

Conclusion:

- Active59 reduces bad overrides and T1 regret, but it gives back too much
  Top1 and T2 performance for the current Top1-first objective.
- Do not promote active59 as the current best model.
- Current best Top1-first checkpoint remains:
  `tutor-route10-margin005-active61-mc1000-suit24-joint-t12-lr5e7-20260531`
  with `t1_override_margin=0.04`.
- The next useful direction is not simply another low-LR fine tune. It should
  be a learned T1 override gate, or stronger T1 labels for the bad-override
  cases.

## 2026-05-31 Learned T1 Override Gate

A lightweight learned T1 override gate was added to the hybrid evaluator. It is
optional and disabled by default.

Implementation:

- evaluator: `ai/tutor/hybrid_t1t2.py`
- training script: `ai/tutor/train_t1_override_gate.py`
- evaluation wiring: `ai/tutor/evaluate_hybrid_refinement_teacher.py`
- config:
  - `t1_override_gate`
  - `t1_override_gate_threshold`
- emitted diagnostics:
  - `t1_override_features`
  - `t1_override_gate_probability`
  - `t1_override_gate_accepted`
  - `t1_override_reject_reason`

Gate training data:

- runtime features: `ai/data/hybrid_t1t2_active_20260531/eval_hybrid_refine_local500_mc300_margin005_active61_joint_lr5e7_t1rec32child4_t2mcboard_margin0_gatefeatures/results.jsonl`
- source input: `ai/data/hybrid_t1t2_active_20260531/local_t1t2_500_mc300.jsonl`
- rows output: `ai/data/hybrid_t1t2_active_20260531/t1_override_gate_active61_mc1000_20260531.rows.jsonl`
- gate output: `ai/data/hybrid_t1t2_active_20260531/t1_override_gate_active61_mc1000_20260531.json`
- training rows: 43
- accept labels: 26
- reject labels: 17
- recommended threshold: 0.45

The gate was trained against MC1000 relabeled rows, not MC300 hit/miss labels.
The label is whether the recursive-refinement final action scored higher than
the model Top1 under the stronger teacher.

Gate training summary:

| threshold | accepted | TP | FP | TN | FN | accuracy | accepted gain |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.35 | 37 | 26 | 11 | 6 | 0 | 74.4% | 50.616 |
| 0.45 | 29 | 24 | 5 | 12 | 2 | 83.7% | 59.229 |
| 0.50 | 25 | 21 | 4 | 13 | 5 | 79.1% | 54.829 |
| 0.70 | 15 | 15 | 0 | 17 | 11 | 74.4% | 48.343 |

Operational 5-second path with active61, `t1_override_margin=0.04`, and gate
threshold `0.45`:

- output: `ai/data/hybrid_t1t2_active_20260531/eval_hybrid_refine_local500_mc300_active61_margin004_gate045`
- time budget violations: 0

| scope | no gate final Top1 | gate final Top1 | no gate avg regret | gate avg regret | no gate bad overrides | gate bad overrides | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 62.4% | 62.2% | 0.615 | 0.611 | 21 | 10 | 1,549 |
| T1 | 52.8% | 52.4% | 0.849 | 0.841 | 21 | 10 | 1,816 |
| T2 | 71.7% | 71.7% | 0.388 | 0.388 | 0 | 0 | 842 |

Conclusion:

- The learned gate is useful for a lower-regret / fewer-bad-overrides mode.
- It is not the current Top1-first setting because it gives back 0.2pp overall
  Top1 and 0.4pp T1 Top1 on this check.
- Current best Top1-first runtime remains active61 with `t1_override_margin=0.04`
  and no learned gate.
- The next Top1-first work should add stronger T1 labels or improve the
  recursive refinement itself. The learned gate can remain as an optional
  safety mode.

## 2026-05-31 T1 Recursive Refinement Sweep

The next Top1-first experiment improved the T1 recursive refinement itself
instead of changing the model. The active61 checkpoint was held fixed:

- model: `ai/models/candidate_runs/tutor-route10-margin005-active61-mc1000-suit24-joint-t12-lr5e7-20260531/model/action_value_best.pt`
- base runtime setting: `t1_override_margin=0.04`, T2 `mc_board` 300 sims

Small T1-only sweep on the first 120 records:

| setting | decisions | T1 final Top1 | T1 model Top1 | T1 regret | overrides | bad overrides | p95 ms | max ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rec16 child4 | 57 | 59.7% | 47.4% | 0.977 | 20 | 5 | 1,007 | 1,170 |
| rec32 child4 | 57 | 61.4% | 47.4% | 0.874 | 18 | 2 | 2,127 | 2,473 |
| rec64 child4 | 57 | 63.2% | 47.4% | 0.706 | 19 | 3 | 4,282 | 4,801 |
| rec32 child8 | 57 | 63.2% | 47.4% | 0.994 | 16 | 2 | 5,022 | 5,026 |
| rec64 child8 | 57 | 54.4% | 47.4% | 1.302 | 14 | 5 | 5,032 | 5,038 |

`rec64 child4` was the best Top1/regret candidate while staying inside the
5-second target in the small sweep.

Full local500 evaluation with `t1_mc_sims=64`, `t1_recursive_child_sims=4`:

- output: `ai/data/hybrid_t1t2_active_20260531/eval_hybrid_refine_local500_mc300_active61_t1rec64child4_t2mcboard_margin004`
- time budget violations: 0

| scope | rec32 child4 final Top1 | rec64 child4 final Top1 | rec32 child4 regret | rec64 child4 regret | rec32 child4 bad overrides | rec64 child4 bad overrides | p95 ms | max ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 62.4% | 62.6% | 0.615 | 0.585 | 21 | 23 | 3,064 | 5,019 |
| T1 | 52.8% | 53.3% | 0.849 | 0.789 | 21 | 23 | 3,604 | 5,019 |
| T2 | 71.7% | 71.7% | 0.388 | 0.388 | 0 | 0 | 745 | 1,082 |

Remaining targets:

- final misses: `ai/data/hybrid_t1t2_active_20260531/targets_runtime_final_miss_local500_mc300_after_active61_t1rec64child4_margin004.jsonl`
  - written: 46
  - T1: 37
  - T2: 9
- T1 bad overrides: `ai/data/hybrid_t1t2_active_20260531/targets_t1_bad_override_after_active61_t1rec64child4_margin004.jsonl`
  - written: 23

Conclusion:

- `t1_mc_sims=64`, `t1_recursive_child_sims=4` is the current best Top1-first
  runtime setting on this local500 MC300 check.
- It improves Top1 and regret without violating the 5-second budget, but it is
  close to the limit: max observed time was about 5.02s.
- It increases T1 bad overrides from 21 to 23, so a follow-up should retune the
  T1 guard/gate for the rec64 setting or improve T1 teacher labels.

## 2026-05-31 Top1-First Pool Audit

The product target is final Top1 correctness. A Top15/Top20 pool is not the
answer; it is only an internal safety set so the 5-second refinement does not
throw away the true best action before it has a chance to rerank candidates.

Offline replay of the rec64 child4 no-guard run showed that Top1-first margin
settings `0.03` to `0.06` are tied on the local500 MC300 operational set:

| setting | overall Top1 | T1 Top1 | T2 Top1 | overall regret | T1 regret |
|---|---:|---:|---:|---:|---:|
| rec64 child4, margin 0.03 | 62.6% | 53.3% | 71.7% | 0.585 | 0.789 |
| rec64 child4, margin 0.04 | 62.6% | 53.3% | 71.7% | 0.585 | 0.789 |
| rec64 child4, margin 0.50 | 62.4% | 52.8% | 71.7% | 0.575 | 0.768 |

The larger margin is useful for lower regret / fewer bad overrides, but it is
not the Top1-first setting because it gives back one correct decision on this
500-position check. The current Top1-first runtime remains:

- model: `ai/models/candidate_runs/tutor-route10-margin005-active61-mc1000-suit24-joint-t12-lr5e7-20260531/model/action_value_best.pt`
- runtime: T1 `recursive_mc`, `t1_mc_sims=64`, `t1_recursive_child_sims=4`
- guard: `t1_override_margin=0.04`
- T2: `mc_board`, `t2_mc_sims=300`

Shortlist-size audit on the same active61 model confirms why `Pool20/24` is not
the final goal, but also why shrinking too aggressively is unsafe today:

| pool setting | T1 pool recall | T1 misses | T2 pool recall | T2 misses | note |
|---|---:|---:|---:|---:|---|
| Top10 only | 94.6% | 27 | 95.4% | 23 | too many true-best actions are pruned |
| Top10 + 5 insurance | 97.8% | 11 | 98.2% | 9 | better, still not safe enough |
| Top15 + 5 insurance, max20 | 99.2% | 4 | 99.6% | 2 | current safety setting |

Implication:

- If we force the pool down to 10 now, Top1 can never reach 100% because the
  true best action is sometimes removed before refinement.
- If we keep all 20-24 candidates, it is not real pruning, but it preserves
  the true best action while the model and refinement are still weak.
- The path to Top1 100% is not to accept Pool20 as success; it is to use Pool20
  temporarily, generate exact/stronger labels for the misses, retrain until
  Top10/Top15 recall approaches 100%, then tighten the pool.

## 2026-05-31 Active62 Rec64 Relabel Loop

The current best runtime after the rec64 child4 refinement left 46 final misses
and 23 T1 bad overrides. These were de-duplicated into one MC1000 relabeling
set:

- target input: `ai/data/hybrid_t1t2_active_20260531/targets_active61_rec64_finalmiss_badoverride_unique_20260531.jsonl`
- unique targets: 62
- T1: 53
- T2: 9
- chunk dir: `ai/data/hybrid_t1t2_active_20260531/chunk_active61_rec64_finalmiss_badoverride_mc1000`
- teacher: `ai/data/hybrid_t1t2_active_20260531/teacher_active61_rec64_finalmiss_badoverride_mc1000_20260531.jsonl`
- completed: 62/62
- eval mode: MC1000
- total local generation time: about 43.7 minutes

The relabeled MC1000 comparison confirms that MC300 misses are mixed rather
than cleanly "model wrong" or "refinement wrong":

| scope | records | model best | final best | model > final | final > model | tie | avg model regret | avg final regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 62 | 7 | 5 | 16 | 25 | 21 | 3.255 | 2.337 |
| T1 | 53 | 7 | 5 | 15 | 21 | 17 | 2.912 | 2.298 |
| T2 | 9 | 0 | 0 | 1 | 4 | 4 | 5.271 | 2.564 |

The labels were added to training:

- suit24 teacher: `ai/data/hybrid_t1t2_active_20260531/teacher_active61_rec64_finalmiss_badoverride_mc1000_20260531_suit24.jsonl`
- suit24 records: 1,488
- converted data: `ai/data/hybrid_t1t2_active_20260531/reranker_active61_rec64_finalmiss_badoverride_mc1000_suit24_20260531`
- converted samples: 34,848
- mixed data: `ai/data/hybrid_t1t2_active_20260531/reranker_mix_current_plus_active62_rec64_mc1000_suit24_20260531`
- mixed samples: 278,139

Fine tune:

- model: `ai/models/candidate_runs/tutor-route10-active62-rec64-mc1000-suit24-joint-t12-lr2e7-20260531/model/action_value_best.pt`
- init: `tutor-route10-margin005-active61-mc1000-suit24-joint-t12-lr5e7-20260531`
- train turns: T1/T2
- lr: 2e-7

Fixed MC300 holdout:

| scope | active61 Top1 | active62 Top1 | active61 Top10 | active62 Top10 | active61 Top15 | active62 Top15 |
|---|---:|---:|---:|---:|---:|---:|
| T1 | 36.4% | 36.8% | 94.6% | 94.6% | 98.8% | 98.8% |
| T2 | 38.4% | 38.8% | 95.4% | 95.4% | 98.8% | 99.0% |

Operational 5-second path with rec64 child4 and `t1_override_margin=0.04`:

- output: `ai/data/hybrid_t1t2_active_20260531/eval_hybrid_refine_local500_mc300_active62_rec64_lr2e7_t1rec64child4_t2mcboard_margin004`
- time budget violations: 0

| scope | active61 final Top1 | active62 final Top1 | active61 avg regret | active62 avg regret | active61 bad overrides | active62 bad overrides | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 62.6% | 62.2% | 0.585 | 0.591 | 23 | 23 | 2,967 |
| T1 | 53.3% | 52.8% | 0.789 | 0.782 | 23 | 23 | 3,355 |
| T2 | 71.7% | 71.3% | 0.388 | 0.407 | 0 | 0 | 803 |

Conclusion:

- Active62 slightly improves model-only fixed holdout, but it does not improve
  the 5-second operational Top1 path.
- Do not promote active62. The current Top1-first runtime remains active61 with
  rec64 child4 and `t1_override_margin=0.04`.
- Directly retraining on the current MC300 runtime misses has diminishing
  returns. The next data loop should target nearby self-play states around the
  Top15+insurance pool misses, so the model learns the pattern instead of just
  memorizing the local500 failures.

Pool-miss neighbor targets were prepared for that next loop:

- input misses: `ai/data/hybrid_t1t2_active_20260531/audit_active61_top15_ins5_pool20/*.misses.jsonl`
- output: `ai/data/hybrid_t1t2_active_20260531/targets_active61_pool20_miss_neighbors_20260531.jsonl`
- written: 720
- T1: 359
- T2: 361
- source: neighboring self-play rows from a 250-line window around the pool
  misses, excluding the miss lines themselves

At the observed local MC1000 speed, all 720 records would be several hours on
this machine. They should be generated in restartable chunks, or on GCP once
the local pipeline has been smoke-tested.

Local MC1000 smoke for the 720 neighbor targets:

- chunk dir: `ai/data/hybrid_t1t2_active_20260531/chunk_active61_pool20_miss_neighbors_mc1000`
- merged output: `ai/data/hybrid_t1t2_active_20260531/teacher_active61_pool20_miss_neighbors_mc1000_20260531.jsonl`
- chunk size: 8
- total chunks: 90
- completed chunks: 6/90
- merged records: 48/720
- T1: 22
- T2: 26
- skipped: 0
- batch errors: 0
- eval mode: MC1000

Observed chunk times:

| chunk | records | elapsed |
|---:|---:|---:|
| 0 | 8 | 312.2s |
| 1 | 8 | 246.6s |
| 2 | 8 | 133.5s |
| 3 | 8 | 210.4s |
| 4 | 8 | 199.5s |
| 5 | 8 | 97.8s |

Current local estimate:

- done: 48/720 = 6.7%
- remaining: 672 records / 84 chunks
- measured average: about 25.0s per record, or about 200s per chunk
- remaining local time: about 4.7 hours if the remaining mix is similar

Resume command:

```powershell
python -m ai.tutor.run_active_teacher_chunks --input ai\data\hybrid_t1t2_active_20260531\targets_active61_pool20_miss_neighbors_20260531.jsonl --output-dir ai\data\hybrid_t1t2_active_20260531\chunk_active61_pool20_miss_neighbors_mc1000 --merged-output ai\data\hybrid_t1t2_active_20260531\teacher_active61_pool20_miss_neighbors_mc1000_20260531.jsonl --turns 1,2 --mc-turns 1,2 --sims 1000 --chunk-size 8 --workers 1 --progress-every 1 --engine-path ai\rust_solver\target\release\prob_engine.exe --batch-engine --batch-timeout 1800 --chunk-timeout 1800 --print-errors --keep-going
```

The command is restart-safe. Existing `.done` chunks are skipped and the merged
JSONL is rebuilt from completed chunks after each run.

Full local MC1000 generation completed:

- completed chunks: 90/90
- merged records: 720/720
- T1: 359
- T2: 361
- skipped: 0
- batch errors: 0
- stderr: empty

The generated labels were expanded and converted:

- suit24 teacher: `ai/data/hybrid_t1t2_active_20260531/teacher_active61_pool20_miss_neighbors_mc1000_20260531_suit24.jsonl`
- suit24 records: 17,280
- converted data: `ai/data/hybrid_t1t2_active_20260531/reranker_active61_pool20_miss_neighbors_mc1000_suit24_20260531`
- converted samples: 376,944

A full mixed dataset with all neighbor labels could not be created locally
because the C: drive had about 1GB free and the mixed memmaps exceeded that.
The failed partial mixed directory was removed. To support disk-safe active
loops, `ai/training/sample_action_value_data.py` now has:

- `--extra-samples N`
- behavior: sample at most `N` rows from each extra dataset, preserving complete
  decision groups
- default: `0`, which keeps the previous behavior of using all extra rows

### Neighbor720 Fine Tune Results

Three local follow-ups were tested from the active61 checkpoint:

1. Extra-only neighbor720, lr `1e-7`
2. Extra-only neighbor720, lr `3e-8`
3. Balanced sample mix: active61 base 100k + neighbor extra 100k, lr `2e-7`

Fixed MC300 holdout:

| model | T1 Top1 | T1 Top10 | T1 Top15 | T2 Top1 | T2 Top10 | T2 Top15 |
|---|---:|---:|---:|---:|---:|---:|
| active61 current | 36.4% | 94.6% | 98.8% | 38.4% | 95.4% | 98.8% |
| neighbor720 extra lr1e-7 | 36.8% | 94.6% | 98.8% | 38.2% | 95.4% | 99.0% |
| neighbor720 extra lr3e-8 | 36.8% | 94.8% | 98.8% | 38.2% | 95.4% | 98.8% |
| neighbor100k balanced lr2e-7 | 36.8% | 94.6% | 98.8% | 38.4% | 95.2% | 99.0% |

Operational 5-second path with rec64 child4 and `t1_override_margin=0.04`:

| model | overall Top1 | T1 Top1 | T2 Top1 | overall regret | T1 regret | T2 regret | bad overrides | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| active61 current | 62.6% | 53.3% | 71.7% | 0.585 | 0.789 | 0.388 | 23 | 3,064 |
| neighbor720 extra lr1e-7 | 61.4% | 51.6% | 70.9% | 0.598 | 0.804 | 0.397 | 23 | 2,975 |
| neighbor720 extra lr3e-8 | 62.6% | 53.3% | 71.7% | 0.582 | 0.782 | 0.388 | 22 | 2,963 |
| neighbor100k balanced lr2e-7 | 62.2% | 52.8% | 71.3% | 0.582 | 0.782 | 0.388 | 22 | 2,917 |

Conclusion:

- None of the neighbor720 follow-ups improves Top1 beyond the active61 current
  best of 62.6% on the local500 MC300 operational check.
- `neighbor720 extra lr3e-8` ties Top1 and slightly improves average regret and
  bad overrides. It is a useful low-regret tie-breaker candidate, but not a
  clear Top1-first promotion.
- `neighbor100k balanced lr2e-7` shows that disk-safe mixing works, but it
  still loses 0.4pp overall Top1.
- Current Top1-first checkpoint remains active61 with rec64 child4 and
  `t1_override_margin=0.04`.
- Next Top1-first progress likely requires stronger labels on the remaining
  runtime miss set, a better T1 override decision model, or improving the
  recursive refinement rather than simply adding more MC1000 neighbor states.

## 2026-06-01 Neighbor720 T1 Override Gate

The neighbor720 MC1000 labels were more useful for learning the T1 override
decision than for directly fine-tuning the action-value model. A new gate was
trained from T1-only runtime features:

- runtime feature eval: `ai/data/hybrid_t1t2_active_20260531/eval_neighbor720_mc1000_active61_t1rec64_margin0_gatefeatures`
- input / strong teacher: `ai/data/hybrid_t1t2_active_20260531/teacher_active61_pool20_miss_neighbors_mc1000_20260531.jsonl`
- model: active61 current
- runtime: T1 rec64 child4, no margin guard
- T1 decisions evaluated: 359
- model Top1: 41.5%
- final Top1 after no-guard refinement: 57.1%
- overrides: 161
- improved: 133
- worse: 26

Gate training:

- output: `ai/data/hybrid_t1t2_active_20260531/t1_override_gate_neighbor720_mc1000_rec64_20260601.json`
- rows: 156 override rows
- accept labels: 128
- reject labels: 28
- recommended threshold by accepted gain: 0.40

Offline replay on the local500 no-guard rec64 result showed the best Top1-first
setting was:

- `t1_override_margin=0.04`
- `t1_override_gate=t1_override_gate_neighbor720_mc1000_rec64_20260601.json`
- `t1_override_gate_threshold=0.40`

Actual operational 5-second evaluation:

- output: `ai/data/hybrid_t1t2_active_20260531/eval_hybrid_refine_local500_mc300_active61_rec64_margin004_neighbor_gate040`
- time budget violations: 0

| scope | previous best Top1 | gate Top1 | previous regret | gate regret | previous bad overrides | gate bad overrides | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 62.6% | 63.0% | 0.585 | 0.582 | 23 | 21 | 2,943 |
| T1 | 53.3% | 54.1% | 0.789 | 0.783 | 23 | 21 | 3,322 |
| T2 | 71.7% | 71.7% | 0.388 | 0.388 | 0 | 0 | 812 |

This is the new Top1-first runtime best on the local500 MC300 operational
check. The gate improves T1 by rejecting a few bad recursive overrides while
preserving most beneficial ones.

Remaining active-loop targets under the new best:

- final misses: `ai/data/hybrid_t1t2_active_20260531/targets_runtime_final_miss_local500_mc300_after_active61_rec64_gate040.jsonl`
  - written: 45
  - T1: 36
  - T2: 9
- T1 bad overrides: `ai/data/hybrid_t1t2_active_20260531/targets_t1_bad_override_after_active61_rec64_gate040.jsonl`
  - written: 6

Current Top1-first runtime:

- model: `ai/models/candidate_runs/tutor-route10-margin005-active61-mc1000-suit24-joint-t12-lr5e7-20260531/model/action_value_best.pt`
- T1 runtime: rec64 child4
- T1 guard: `t1_override_margin=0.04`
- T1 gate: `ai/data/hybrid_t1t2_active_20260531/t1_override_gate_neighbor720_mc1000_rec64_20260601.json`
- T1 gate threshold: `0.40`
- T2 runtime: `mc_board`, 300 sims

## 2026-06-01 Residual45 MC1000 Active Loop

The 45 remaining local500 misses under the current Top1-first runtime were
relabelled with MC1000. The 6 T1 bad overrides were duplicates of the same
target states, so the unique relabel set stayed at 45:

- merged targets: `ai/data/hybrid_t1t2_active_20260531/targets_residual_after_active61_rec64_gate040_unique_20260601.jsonl`
- source rows read: 51
- unique targets: 45
- T1: 36
- T2: 9
- duplicates removed: 6
- chunk output: `ai/data/hybrid_t1t2_active_20260531/chunk_residual_after_active61_rec64_gate040_mc1000_20260601`
- teacher: `ai/data/hybrid_t1t2_active_20260531/teacher_residual_after_active61_rec64_gate040_mc1000_20260601.jsonl`
- chunk status: 9/9 done, 45/45 records, no skips
- suit24 teacher: `ai/data/hybrid_t1t2_active_20260531/teacher_residual_after_active61_rec64_gate040_mc1000_20260601_suit24.jsonl`
- suit24 records: 1,080
- converted data: `ai/data/hybrid_t1t2_active_20260531/reranker_residual_after_active61_rec64_gate040_mc1000_suit24_20260601`
- converted samples: 25,920

Relabel comparison against the old runtime choices showed that these are mostly
real misses, not just MC300 noise:

- records compared: 45
- old model action is MC1000 best: 2
- old final action is MC1000 best: 2
- final better than model: 18
- model better than final: 6
- tie: 21
- average model regret: 4.279
- average final regret: 3.002

Two follow-ups were tested and neither is promoted:

1. New T1 override gate trained from the current local500 runtime rows plus the
   residual45 MC1000 labels.
2. Small model fine-tunes using active61 base 100k + residual45 all samples.

Gate result:

- gate: `ai/data/hybrid_t1t2_active_20260531/t1_override_gate_active61_rec64_gate040_residual45_mc1000_20260601.json`
- rows: 71
- accept labels: 48
- reject labels: 23
- recommended threshold: 0.45
- operational output: `ai/data/hybrid_t1t2_active_20260531/eval_hybrid_refine_local500_mc300_active61_rec64_margin004_residual45_gate045`
- overall Top1: 62.4%
- T1 Top1: 52.8%
- T2 Top1: 71.7%
- regret: 0.586

Model fine-tune results with the existing neighbor720 gate at threshold 0.40:

| model | overall Top1 | T1 Top1 | T2 Top1 | overall regret | bad overrides | p95 ms | promote |
|---|---:|---:|---:|---:|---:|---:|---|
| current active61 + neighbor gate | 63.0% | 54.1% | 71.7% | 0.582 | 21 | 2,943 | yes |
| residual45 mix lr1e-7 | 62.8% | 54.1% | 71.3% | 0.583 | 21 | 2,930 | no |
| residual45 mix lr3e-8 | 62.8% | 54.1% | 71.3% | 0.583 | 21 | 2,950 | no |

Conclusion:

- The residual45 MC1000 labels are valuable evidence: most current misses remain
  misses under stronger labels.
- A small residual-only active-loop correction is not enough to move model Top1;
  model Top1 stayed at 39.8% on the local500 MC300 check.
- The current runtime best remains active61 + rec64 child4 +
  neighbor720 gate threshold 0.40.
- Next progress should target either a larger balanced residual dataset, a
  stronger T2-specific relabel/refinement loop, or a richer T1 override model
  than the current linear gate.

## 2026-06-01 T2 Residual Neighbor100 Turn-Routed Model

The next successful step was T2-specific rather than another all-turn
fine-tune. `build_hybrid_active_targets.py` now accepts both old audit misses
with `source_record` and runtime target/miss rows with top-level `source` /
`source_line`, so current runtime misses can drive neighbor mining directly.

T2 neighbor target generation:

- source misses: `ai/data/hybrid_t1t2_active_20260531/targets_runtime_final_miss_local500_mc300_after_active61_rec64_gate040.jsonl`
- output: `ai/data/hybrid_t1t2_active_20260531/targets_t2_residual_neighbors_w120_unique_20260601.jsonl`
- turn: T2 only
- window: 120 source rows around current misses
- available T2 neighbors before cap: 684
- selected unique targets: 300

Local MC1000 was first run for 100 targets, then extended to 200 targets:

- chunk dir: `ai/data/hybrid_t1t2_active_20260531/chunk_t2_residual_neighbors_w120_mc1000_20260601`
- chunk status: 20/30 done
- teacher: `ai/data/hybrid_t1t2_active_20260531/teacher_t2_residual_neighbors_w120_mc1000_20260601.jsonl`
- records completed: 200/300
- skips: 0
- observed speed: about 57-79 seconds per 10 T2 targets

First 100-target converted data:

- suit24 teacher: `ai/data/hybrid_t1t2_active_20260531/teacher_t2_residual_neighbors_w120_mc1000_100_20260601_suit24.jsonl`
- suit24 records: 2,400
- converted data: `ai/data/hybrid_t1t2_active_20260531/reranker_t2_residual_neighbors_w120_mc1000_100_suit24_20260601`
- converted samples: 48,888
- mixed data: `ai/data/hybrid_t1t2_active_20260531/reranker_mix_active61_100k_plus_t2resid100_mc1000_suit24_20260601`
- mixed samples: 149,090
- base samples: 100,202
- extra T2 samples: 48,888

Second 200-target converted data:

- suit24 teacher: `ai/data/hybrid_t1t2_active_20260531/teacher_t2_residual_neighbors_w120_mc1000_200_20260601_suit24.jsonl`
- suit24 records: 4,800
- converted data: `ai/data/hybrid_t1t2_active_20260531/reranker_t2_residual_neighbors_w120_mc1000_200_suit24_20260601`
- converted samples: 98,160
- mixed data: `ai/data/hybrid_t1t2_active_20260531/reranker_mix_active61_100k_plus_t2resid200_100k_mc1000_suit24_20260601`
- mixed samples: 198,362
- base samples: 100,202
- extra T2 samples: 98,160

Models:

- 100-target checkpoint: `ai/models/candidate_runs/tutor-route10-active61-t2resid100-mc1000-suit24-ft-t2only-lr1e7-20260601/model/action_value_best.pt`
- 200-target checkpoint: `ai/models/candidate_runs/tutor-route10-active61-t2resid200-mc1000-suit24-ft-t2only-lr1e7-20260601/model/action_value_best.pt`
- init: current active61 checkpoint
- train turns: T2 only
- lr: `1e-7`

The T2-only model improved T2 runtime but slightly hurt T1 if used globally.
To avoid that cross-turn regression, the evaluator/runtime path now supports
per-turn model overrides:

- `ai/tutor/hybrid_t1t2.py`: `make_action_value_evaluator(..., model_paths_by_turn={2: ...})`
- CLI: `--turn-model 2=path/to/action_value_best.pt`
- `RolloutEvaluator` already supported `action_value_nets_by_turn`; this change
  wires it into the hybrid tutor path.

Operational 5-second result with turn routing:

- base model: current active61
- T2 model override: T2 residual checkpoint
- time budget violations: 0

| model | overall Top1 | T1 Top1 | T2 Top1 | overall regret | T2 regret | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| previous best | 63.0% | 54.1% | 71.7% | 0.582 | 0.388 | 2,943 |
| T2 residual100 routed | 63.2% | 54.1% | 72.0% | 0.582 | 0.387 | 2,979 |
| T2 residual200 routed | 63.4% | 54.1% | 72.4% | 0.580 | 0.382 | 3,030 |

The 200-target routed model is the new Top1-first runtime best on local500
MC300, with the important
qualification that the high-margin residual miss target set is unchanged:

- new residual targets: `ai/data/hybrid_t1t2_active_20260531/targets_runtime_final_miss_local500_mc300_after_active61_plus_t2resid200_turnmodel.jsonl`
- written: 44
- T1: 36
- T2: 8

Current Top1-first runtime:

- base model: `ai/models/candidate_runs/tutor-route10-margin005-active61-mc1000-suit24-joint-t12-lr5e7-20260531/model/action_value_best.pt`
- T2 turn model: `ai/models/candidate_runs/tutor-route10-active61-t2resid200-mc1000-suit24-ft-t2only-lr1e7-20260601/model/action_value_best.pt`
- T1 runtime: rec64 child4
- T1 guard: `t1_override_margin=0.04`
- T1 gate: `ai/data/hybrid_t1t2_active_20260531/t1_override_gate_neighbor720_mc1000_rec64_20260601.json`
- T1 gate threshold: `0.40`
- T2 runtime: `mc_board`, 300 sims

Next step:

- Complete more of the remaining 100 T2 neighbor MC1000 targets, then retrain
  another T2-only turn model.
- In parallel, mine a larger T1 bad-override / residual-neighbor set because
  T1 remains the main overall bottleneck.

## 2026-06-01 T2 Sync10 Refinement Breakthrough

The largest T2 gain did not come from the 300-target model. It came from using
the available T2 latency budget. T2 with 3 refined candidates was spending only
about 0.8s p95, so the runtime now supports a separate T2 refinement width:

- `HybridConfig.t2_sync_exact_k`
- CLI: `--t2-sync-exact-k N`
- default `0`, which preserves the old `--sync-exact-k` behavior
- T1 still uses the base `--sync-exact-k`, so T1 latency does not increase

T2-only latency/quality sweep using the 200-target T2 turn model:

| T2 refined candidates | T2 Top1 | T2 regret | p95 ms | max ms | time violations |
|---:|---:|---:|---:|---:|---:|
| 3 | 72.4% | 0.382 | 812 | 1,159 | 0 |
| 5 | 85.4% | 0.116 | 1,204 | 1,673 | 0 |
| 10 | 96.9% | 0.000 | 2,295 | 3,308 | 0 |
| 15 | 96.9% | 0.000 | 2,737 | 3,579 | 0 |

The 10-candidate setting is the best tradeoff. It reaches the same T2 Top1 as
15 candidates, with lower latency.

Operational 5-second result:

- output: `ai/data/hybrid_t1t2_active_20260531/eval_hybrid_refine_local500_mc300_active61_plus_t2resid200_turnmodel_t2sync10`
- base model: current active61
- T2 turn model: residual200 T2-only checkpoint
- T1 refined candidates: 3
- T2 refined candidates: 10
- time budget violations: 0

| scope | previous best Top1 | sync10 Top1 | previous regret | sync10 regret | p95 ms |
|---|---:|---:|---:|---:|---:|
| overall | 63.4% | 75.8% | 0.580 | 0.385 | 2,985 |
| T1 | 54.1% | 54.1% | 0.783 | 0.783 | 3,439 |
| T2 | 72.4% | 96.9% | 0.382 | 0.000 | 2,218 |

The residual target set confirms that T2 is no longer the current bottleneck
under local500 MC300:

- residual targets: `ai/data/hybrid_t1t2_active_20260531/targets_runtime_final_miss_local500_mc300_after_active61_plus_t2resid200_t2sync10.jsonl`
- written: 36
- T1: 36
- T2: 0
- T1 bad overrides: `ai/data/hybrid_t1t2_active_20260531/targets_t1_bad_override_after_active61_plus_t2resid200_t2sync10.jsonl`
- T1 bad override targets: 6

The remaining T2 neighbor run was completed for completeness:

- chunk status: 30/30 done
- teacher records: 300/300
- skips: 0
- 300-target T2 model: `ai/models/candidate_runs/tutor-route10-active61-t2resid300-mc1000-suit24-ft-t2only-lr1e7-20260601/model/action_value_best.pt`
- T2-only sync3 check: 72.4%, not better than the 200-target model

Current Top1-first runtime:

- base model: `ai/models/candidate_runs/tutor-route10-margin005-active61-mc1000-suit24-joint-t12-lr5e7-20260531/model/action_value_best.pt`
- T2 turn model: `ai/models/candidate_runs/tutor-route10-active61-t2resid200-mc1000-suit24-ft-t2only-lr1e7-20260601/model/action_value_best.pt`
- T1 runtime: rec64 child4
- T1 refined candidates: 3
- T1 guard: `t1_override_margin=0.04`
- T1 gate: `ai/data/hybrid_t1t2_active_20260531/t1_override_gate_neighbor720_mc1000_rec64_20260601.json`
- T1 gate threshold: `0.40`
- T2 runtime: `mc_board`, 300 sims
- T2 refined candidates: 10

Next step:

- Shift active-loop work to T1: mine T1 residual neighbors and bad overrides,
  relabel with MC1000, then improve either the T1 model or the T1 recursive
  override gate.
- Keep T2 sync10 as the runtime default candidate; more T2 model-only work has
  lower priority until T1 catches up.

## 2026-06-01 T1 Sync6 + T2 Sync10 Runtime

The product target remains final Top1 correctness. Top15/Top20 pools are only
internal safety pools for pruning and refinement; they are not a success metric.

A wider T1 synchronous refinement was tested while keeping the T2 sync10 path:

- output: `ai/data/hybrid_t1t2_active_20260531/eval_hybrid_refine_local500_mc300_active61_t1sync6s32_plus_t2resid200_t2sync10`
- base model: current active61
- T2 turn model: residual200 T2-only checkpoint
- T1 refined candidates: 6
- T1 recursive MC: 32 sims, beam 5, child sims 4
- T1 guard: `t1_override_margin=0.04`
- T1 gate: neighbor720 gate threshold `0.40`
- T2 refined candidates: 10
- T2 MC board sims: 300
- time budget violations: 0

| scope | decisions | model Top1 | final Top1 | final regret | p95 ms | max ms |
|---|---:|---:|---:|---:|---:|---:|
| overall | 500 | 40.2% | 77.8% | 0.236 | 3,036 | 4,872 |
| T1 | 246 | 38.2% | 58.1% | 0.480 | 3,461 | 4,872 |
| T2 | 254 | 42.1% | 96.9% | 0.000 | 2,243 | 3,410 |

Compared with the previous T2 sync10 runtime, this improves overall final Top1
from 75.8% to 77.8% by spending more of the T1 budget. T2 stays at 96.9% with
zero measured MC300 regret. T1 remains the bottleneck.

New residual targets after this runtime:

- final misses: `ai/data/hybrid_t1t2_active_20260531/targets_runtime_final_miss_local500_mc300_after_active61_t1sync6s32_plus_t2resid200_t2sync10.jsonl`
  - written: 21
  - T1: 21
  - T2: 0
- T1 bad overrides: `ai/data/hybrid_t1t2_active_20260531/targets_t1_bad_override_after_active61_t1sync6s32_plus_t2resid200_t2sync10.jsonl`
  - written: 6

Current Top1-first runtime candidate:

- base model: `ai/models/candidate_runs/tutor-route10-margin005-active61-mc1000-suit24-joint-t12-lr5e7-20260531/model/action_value_best.pt`
- T2 turn model: `ai/models/candidate_runs/tutor-route10-active61-t2resid200-mc1000-suit24-ft-t2only-lr1e7-20260601/model/action_value_best.pt`
- T1 runtime: rec32 child4, 6 refined candidates
- T1 guard: `t1_override_margin=0.04`
- T1 gate: `ai/data/hybrid_t1t2_active_20260531/t1_override_gate_neighbor720_mc1000_rec64_20260601.json`
- T1 gate threshold: `0.40`
- T2 runtime: `mc_board`, 300 sims, 10 refined candidates

Next step:

- Build a larger T1 residual-neighbor dataset from the 21 final misses and 6
  bad overrides.
- Relabel those states with MC1000 or stronger labels.
- Train either a T1-only turn-routed model or a richer T1 override gate.
- Keep measuring final Top1 under the same 5 second runtime, because raw model
  Top1 alone is not enough for gameplay.

T1 residual-neighbor target generation was started:

- merged residual targets: `ai/data/hybrid_t1t2_active_20260531/targets_t1_residual_after_active61_t1sync6s32_t2sync10_unique_20260601.jsonl`
  - read: 27
  - written unique: 21
  - duplicates removed: 6
- neighbor targets: `ai/data/hybrid_t1t2_active_20260531/targets_t1_residual_neighbors_w180_1000_20260601.jsonl`
  - T1 available: 654
  - written: 654
  - window: 180 source rows
  - miss lines included

Local MC1000 smoke:

- chunk dir: `ai/data/hybrid_t1t2_active_20260531/chunk_t1_residual_neighbors_w180_mc1000_20260601`
- teacher: `ai/data/hybrid_t1t2_active_20260531/teacher_t1_residual_neighbors_w180_mc1000_20260601.jsonl`
- completed: 20/20 smoke records
- skipped: 0
- elapsed by 5-record chunk: 268s, 186s, 196s, 264s
- average speed: about 45.7 seconds per T1 record
- estimate for all 654 local records: about 8.3 hours on this machine

The pipeline is restart-safe. If continuing locally, use a smaller first
training slice such as 100 to 200 records before spending the full 654-record
cost. If full coverage is needed quickly, run this chunk job on GCP.

The 100-record local slice was completed and tested:

- chunk status: 20/20 done
- teacher: `ai/data/hybrid_t1t2_active_20260531/teacher_t1_residual_neighbors_w180_mc1000_20260601.jsonl`
  - records: 100
  - skipped: 0
  - T1 MC1000: 100
- suit24 teacher: `ai/data/hybrid_t1t2_active_20260531/teacher_t1_residual_neighbors_w180_mc1000_100_20260601_suit24.jsonl`
  - records: 2,400
- converted data: `ai/data/hybrid_t1t2_active_20260531/reranker_t1_residual_neighbors_w180_mc1000_100_suit24_20260601`
  - samples: 54,432
- mixed data: `ai/data/hybrid_t1t2_active_20260531/reranker_mix_active61_100k_plus_t1resid100_mc1000_suit24_20260601`
  - total samples: 154,634
  - base samples: 100,202
  - extra T1 samples: 54,432

Two T1-only turn-routed fine-tunes were trained from active61:

- `tutor-route10-active61-t1resid100-mc1000-suit24-ft-t1only-lr1e7-20260601`
- `tutor-route10-active61-t1resid100-mc1000-suit24-ft-t1only-lr3e8-20260601`

Operational 5-second evaluation kept the same runtime policy:

- base model: active61
- T1 turn model: residual100 candidate
- T2 turn model: residual200 candidate
- T1 refined candidates: 6
- T2 refined candidates: 10
- T1 recursive MC: 32 sims, beam 5, child sims 4
- T2 MC board sims: 300

| model | overall Top1 | T1 Top1 | T2 Top1 | overall regret | T1 regret | bad overrides | p95 ms | violations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| active61 + T2resid200 + T1sync6 | 77.8% | 58.1% | 96.9% | 0.236 | 0.480 | 24 | 3,036 | 0 |
| + T1resid100 lr1e-7 | 77.8% | 58.1% | 96.9% | 0.233 | 0.473 | 22 | 3,087 | 0 |
| + T1resid100 lr3e-8 | 77.8% | 58.1% | 96.9% | 0.241 | 0.490 | 23 | 3,140 | 0 |

Conclusion:

- The first 100 T1 MC1000 residual-neighbor records did not improve final
  Top1.
- The lr1e-7 T1 turn model is a low-regret tie-breaker candidate: same Top1,
  lower regret, and fewer bad overrides.
- It is not a Top1-first breakthrough. The remaining final misses are still
  21 high-margin T1 states, and T2 remains at zero high-margin final misses on
  this local500 MC300 check.

Residual targets after the lr1e-7 T1 model:

- final misses: `ai/data/hybrid_t1t2_active_20260531/targets_runtime_final_miss_local500_mc300_after_t1resid100_lr1e7_t1sync6s32_plus_t2resid200_t2sync10.jsonl`
  - written: 21
  - T1: 21
  - T2: 0
- T1 bad overrides: `ai/data/hybrid_t1t2_active_20260531/targets_t1_bad_override_after_t1resid100_lr1e7_t1sync6s32_plus_t2resid200_t2sync10.jsonl`
  - written: 6

Next step:

- Continue the T1 residual-neighbor MC1000 generation from 100 toward 200 or
  the full 654 records.
- The 100-record result suggests this needs broader pattern coverage, not just
  a tiny residual slice.
- If staying local, extend to 200 first and retrain. If using cloud, complete
  all 654 records and train a larger T1-only turn model plus a richer T1
  override gate.

The 200-record local extension was completed:

- chunk status: 40/40 done
- teacher: `ai/data/hybrid_t1t2_active_20260531/teacher_t1_residual_neighbors_w180_mc1000_20260601.jsonl`
  - records: 200
  - skipped: 0
  - T1 MC1000: 200
- suit24 teacher: `ai/data/hybrid_t1t2_active_20260531/teacher_t1_residual_neighbors_w180_mc1000_200_20260601_suit24.jsonl`
  - records: 4,800
- converted data: `ai/data/hybrid_t1t2_active_20260531/reranker_t1_residual_neighbors_w180_mc1000_200_suit24_20260601`
  - samples: 107,928
- mixed data: `ai/data/hybrid_t1t2_active_20260531/reranker_mix_active61_100k_plus_t1resid200_100k_mc1000_suit24_20260601`
  - total samples: 200,204
  - base samples: 100,202
  - extra T1 samples: 100,002

T1-only turn-routed fine-tune:

- model: `ai/models/candidate_runs/tutor-route10-active61-t1resid200-mc1000-suit24-ft-t1only-lr1e7-20260601/model/action_value_best.pt`
- init: active61
- train turns: T1 only
- lr: `1e-7`

Operational 5-second result:

| model | overall Top1 | T1 Top1 | T2 Top1 | overall regret | T1 regret | bad overrides | p95 ms | violations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| active61 + T2resid200 + T1sync6 | 77.8% | 58.1% | 96.9% | 0.236 | 0.480 | 24 | 3,036 | 0 |
| + T1resid100 lr1e-7 | 77.8% | 58.1% | 96.9% | 0.233 | 0.473 | 22 | 3,087 | 0 |
| + T1resid200 lr1e-7 | 77.6% | 57.7% | 96.9% | 0.241 | 0.490 | 23 | 3,345 | 0 |

Conclusion:

- Extending the same T1 residual-neighbor slice from 100 to 200 records did not
  improve Top1. It slightly worsened the 5-second operational result.
- The best current Top1-first runtime remains active61 + T2resid200 +
  T1sync6/T2sync10. The T1resid100 lr1e-7 model is only a low-regret tie, not a
  Top1 upgrade.
- The remaining high-margin local500 misses are still T1-only:
  - final misses after T1resid200: `ai/data/hybrid_t1t2_active_20260531/targets_runtime_final_miss_local500_mc300_after_t1resid200_lr1e7_t1sync6s32_plus_t2resid200_t2sync10.jsonl`
  - written: 21
  - T1: 21
  - T2: 0
  - bad overrides: `ai/data/hybrid_t1t2_active_20260531/targets_t1_bad_override_after_t1resid200_lr1e7_t1sync6s32_plus_t2resid200_t2sync10.jsonl`
  - written: 6

Next step:

- Stop treating simple T1 residual fine-tuning as the main lever.
- Improve the T1 synchronous decision policy instead:
  - compare wider T1 refine widths under the 5s budget,
  - train a richer T1 override/gate model from runtime features,
  - and only then add more MC1000 labels if the new policy exposes different
    residual miss patterns.
- If more teacher generation is needed, prefer the full 654-record set on GCP
  rather than another small local slice.

T1 sync width was also tested directly:

- output: `ai/data/hybrid_t1t2_active_20260531/eval_hybrid_refine_local500_mc300_active61_t1sync8s24_plus_t2resid200_t2sync10`
- T1 refined candidates: 8
- T1 recursive MC: 24 sims, beam 5, child sims 4
- T2 refined candidates: 10
- T2 MC board sims: 300

| model/runtime | overall Top1 | T1 Top1 | T2 Top1 | overall regret | T1 regret | bad overrides | p95 ms | violations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| active61 + T2resid200 + T1sync6/s32 | 77.8% | 58.1% | 96.9% | 0.236 | 0.480 | 24 | 3,036 | 0 |
| active61 + T2resid200 + T1sync8/s24 | 77.4% | 57.3% | 96.9% | 0.256 | 0.520 | 25 | 3,875 | 3 |

Conclusion:

- Simply widening T1 synchronous refinement does not improve Top1. It worsens
  regret, bad overrides, and latency.
- The T1 problem is now more about override decision quality than candidate
  count.
- The next useful implementation target is a richer T1 override/gate model that
  decides whether to trust recursive refinement, rather than spending more
  budget on more low-confidence T1 candidates.

## 2026-06-01 T1 Gate Threshold Update

The T1 gate path was extended with deterministic derived features:

- `ai/tutor/hybrid_t1t2.py`
  - adds `enrich_override_gate_features(...)`
  - runtime gate features now include interaction/risk-adjusted features.
- `ai/tutor/train_t1_override_gate.py`
  - writes the derived features into new rows.
  - can train from prebuilt `--rows-input` JSONL files, so multiple MC1000
    gate datasets can be combined.

The current T1 residual200 MC1000 teacher set was evaluated with gate disabled:

- output: `ai/data/hybrid_t1t2_active_20260531/eval_t1_residual200_mc1000_active61_t1sync6s32_margin0_gatefeatures`
- decisions: 200
- model Top1: 38.0%
- no-gate refined Top1: 61.5%
- overrides: 120
- improved: 101
- worse: 18
- p95: 3,279 ms
- time violations: 0

Rows were generated from that run:

- `ai/data/hybrid_t1t2_active_20260531/t1_override_gate_t1resid200_mc1000_rec32_derived_20260601.rows.jsonl`
- rows: 118
- accept labels: 99
- reject labels: 19

A combined derived-feature gate was trained from:

- neighbor720 MC1000 gate rows
- residual45 MC1000 gate rows
- T1 residual200 MC1000 gate rows

Combined gate:

- `ai/data/hybrid_t1t2_active_20260531/t1_override_gate_combined_neighbor720_residual45_t1resid200_derived_20260601.json`
- rows: 344
- accept labels: 274
- reject labels: 70
- recommended threshold: 0.55

However, the combined derived gate did not generalize to the local500 MC300
operational check:

| gate | threshold | overall Top1 | T1 Top1 | T2 Top1 | overall regret | T1 regret | bad overrides | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neighbor720 old | 0.40 | 77.8% | 58.1% | 96.9% | 0.236 | 0.480 | 24 | 3,036 |
| combined derived | 0.55 | 77.2% | 56.9% | 96.9% | 0.248 | 0.504 | 23 | 3,159 |

The useful improvement came from replaying the existing neighbor720 gate on the
current T1 sync6/s32 no-gate local500 run. Threshold `0.45` was better than
`0.40`:

| gate | threshold | overall Top1 | T1 Top1 | T2 Top1 | overall regret | T1 regret | bad overrides | p95 ms | violations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| neighbor720 old | 0.40 | 77.8% | 58.1% | 96.9% | 0.236 | 0.480 | 24 | 3,036 | 0 |
| neighbor720 old | 0.45 | 78.0% | 58.5% | 96.9% | 0.226 | 0.459 | 23 | 3,099 | 0 |

New current Top1-first runtime:

- base model: `ai/models/candidate_runs/tutor-route10-margin005-active61-mc1000-suit24-joint-t12-lr5e7-20260531/model/action_value_best.pt`
- T2 turn model: `ai/models/candidate_runs/tutor-route10-active61-t2resid200-mc1000-suit24-ft-t2only-lr1e7-20260601/model/action_value_best.pt`
- T1 runtime: rec32 child4, 6 refined candidates
- T1 guard: `t1_override_margin=0.04`
- T1 gate: `ai/data/hybrid_t1t2_active_20260531/t1_override_gate_neighbor720_mc1000_rec64_20260601.json`
- T1 gate threshold: `0.45`
- T2 runtime: `mc_board`, 300 sims, 10 refined candidates

Remaining residuals under the new best:

- final misses: `ai/data/hybrid_t1t2_active_20260531/targets_runtime_final_miss_local500_mc300_after_active61_t1sync6s32_plus_t2resid200_t2sync10_gate045.jsonl`
  - written: 20
  - T1: 20
  - T2: 0
- T1 bad overrides: `ai/data/hybrid_t1t2_active_20260531/targets_t1_bad_override_after_active61_t1sync6s32_plus_t2resid200_t2sync10_gate045.jsonl`
  - written: 5

Disk cleanup:

- Removed older, non-promoted mixed memmap directories from previous experiments.
- Kept teacher JSONL files and model checkpoints.
- Kept the active61 base mixed dataset for future training.

Next step:

- Mine the 20 remaining high-confidence T1 misses under gate045.
- Build a T1-specific policy/gate dataset around those states.
- If more labels are needed, run the full 654-record T1 residual-neighbor MC1000
  job on GCP rather than continuing local slices.

## 2026-06-01 Top1-First Clarification

The product target is strict final Top1 correctness. Top-N recall is only an
internal safety metric for pruning and refinement. In particular, a Pool20/24
result is not a final answer, and Pool24 is not meaningful pruning when the
turn has about 24 legal actions. The runtime should report and optimize final
Top1 first, then use regret and Top-N recall as supporting diagnostics.

Guaranteeing Top1 100% requires one of these to be true:

1. all legal candidates can be evaluated exactly within the 5 second turn
   budget, or
2. the model shortlist never drops the true best candidate and the 5 second
   refinement can rank that shortlist correctly.

Current status on local500 MC300:

| runtime | overall final Top1 | T1 final Top1 | T2 final Top1 | p95 ms | violations |
|---|---:|---:|---:|---:|---:|
| active61 + T2resid200 + T1sync6/s32 + gate045 | 78.0% | 58.5% | 96.9% | 3,099 | 0 |

The remaining local500 high-confidence misses are T1-only. That means the next
Top1-first work should not focus on T2 or on wider Pool24 pruning. It should
focus on T1 final decision quality.

The 20 residual T1 misses under gate045 were relabeled with MC1000:

- input: `targets_residual_after_active61_t1sync6s32_t2sync10_gate045_unique_20260601.jsonl`
- teacher: `teacher_residual_after_active61_t1sync6s32_t2sync10_gate045_mc1000_20260601.jsonl`
- records: 20/20 completed, all T1.

On those 20 hard states, disabling the gate and trusting recursive refinement
improves average regret but does not solve Top1:

| set | decisions | model Top1 | no-gate refined Top1 | model avg regret | final avg regret | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| T1 residual20 MC1000 | 20 | 20.0% | 5.0% | 2.793 | 1.998 | 3,010 |

This shows why "accept refinement if score improves" is not enough for strict
Top1. A move can reduce regret while still not matching the teacher Top1.

A combined Top1-label gate was trained from four MC1000 row sets:

- neighbor720 Top1 rows
- residual45 Top1 rows
- T1 residual200 Top1 rows
- residual20 gate045 Top1 rows

The resulting gate did not beat the existing neighbor720 gate on the local500
offline replay:

| gate | best replay threshold | T1 Top1 | T1 avg regret | accepted overrides |
|---|---:|---:|---:|---:|
| neighbor720 old | 0.45 | 58.5% | 0.459 | 128 |
| combined Top1-label | 0.20 | 57.7% | 0.518 | 124 |

Decision: do not promote the combined Top1-label gate. Keep
`t1_override_gate_neighbor720_mc1000_rec64_20260601.json` at threshold `0.45`
as the current best local runtime.

Next Top1-first path:

- build more exact/high-confidence T1 teacher rows around the 20 residual
  states, but include neighboring boards/actions so the gate sees successful
  and failed overrides in the same local pattern family;
- train/evaluate the gate by replay Top1, not accepted-score gain;
- separately test whether a T1 exact/recursive Rust path can evaluate all legal
  candidates inside 5 seconds. If yes, the model becomes a speed aid. If no, the
  model remains necessary for shortlist safety.

## 2026-06-01 MC Label Stability Guard

MC300 labels are not reliable enough to be treated as hard Top1 truth. A new
audit script was added:

- `ai/tutor/audit_teacher_label_stability.py`

It matches weak and strong teacher JSONL records by state or source and reports
whether Top1 labels agree. The current T1 residual20 check compared:

- weak: `local_t1t2_500_mc300.jsonl`
- strong: `teacher_residual_after_active61_t1sync6s32_t2sync10_gate045_mc1000_20260601.jsonl`

Result:

| comparison | matched | Top1 same | Top1 changed | weak Top1 in strong Top3 | strong Top1 in weak Top3 |
|---|---:|---:|---:|---:|---:|
| MC300 vs MC1000 T1 residual20 | 20 | 18 | 2 | 20 | 20 |

Both changed labels had MC300 margin above 1.0, so teacher margin alone is not
enough. For active-loop Top1 training, estimated labels should require both:

- enough margin, and
- enough simulations or exact evaluation.

Target extractors now support simulation-count filtering:

- `ai/tutor/weak_groups_to_active_targets.py`
  - `--min-teacher-sims`
  - `--teacher-sims-applies-to estimated|all`
- `ai/tutor/hybrid_refinement_misses_to_targets.py`
  - `--min-teacher-sims`
  - `--teacher-sims-applies-to estimated|all`

Validation:

- Applying `--min-teacher-sims 1000` to the MC300 local500 gate045 misses writes
  zero targets:
  - 20 high-margin misses are skipped by simulation count.
- Applying the same guard to the MC1000 residual20 no-gate run writes 17 T1
  targets:
  - `targets_t1_residual20_mc1000_final_miss_min1000_20260601.jsonl`
  - 2 skipped by low MC1000 margin, 1 skipped because refinement already hit
    Top1.

Decision: future Top1 active-loop training should not use MC300-only final-miss
targets as hard labels. MC300 can still identify suspicious states, but those
states must be relabeled with MC1000+ or exact before being promoted into a
Top1 training set.

## 2026-06-01 T1 Residual17 Active-Loop Smoke

The 17 MC1000 high-confidence residual targets were promoted back to teacher
JSONL and converted into action-value data:

- target filter: `ai/tutor/filter_teacher_by_targets.py`
- promoted teacher:
  `ai/data/hybrid_t1t2_active_20260531/teacher_t1_residual17_mc1000_top1_active_20260601.jsonl`
- converted data:
  `ai/data/hybrid_t1t2_active_20260531/reranker_t1_residual17_mc1000_top1_active_20260601`

Conversion result:

- teacher records: 17
- candidate samples: 435
- skipped candidates: 0
- score mean/std: +5.660 / 6.845
- bust mean: 27.6%
- FL mean: 11.9%

Current active61 model on those 17 hard states:

| model | Top1 | Top3 | Top5 | Top10 | Top20 | avg regret | max regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| active61 | 17.6% | 47.1% | 64.7% | 88.2% | 100.0% | 3.033 | 11.082 |

This confirms the same pattern as the runtime checks: the true action is still
inside a wide pool, but raw model Top1 is weak on the residual T1 states.

Two lightweight local fine-tunes were tested from active61:

1. Base 80k + residual17 once:
   - data:
     `reranker_mix_active61_base80k_plus_t1resid17_mc1000_top1_20260601`
   - model:
     `tutor-route10-active61-t1resid17-mc1000-top1-localft-20260601`
2. Base 80k + residual17 repeated 20x:
   - data:
     `reranker_mix_active61_base80k_plus_t1resid17_mc1000_top1_x20_20260601`
   - model:
     `tutor-route10-active61-t1resid17-mc1000-top1-x20-localft-20260601`

Residual17 evaluation:

| model | residual weighting | Top1 | Top3 | Top10 | avg regret | score MAE | FL MAE | bust MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| active61 | 0x | 17.6% | 47.1% | 88.2% | 3.033 | 3.027 | 8.4% | 10.6% |
| localft | 1x | 23.5% | 41.2% | 94.1% | 3.041 | 3.139 | 7.4% | 10.1% |
| localft | 20x | 17.6% | 41.2% | 94.1% | 3.072 | 2.965 | 6.9% | 9.9% |

Decision: do not promote either local fine-tune. Hard residual weighting alone
does not solve Top1. It slightly improves calibration and Top10, but Top3 and
regret do not move in the right direction.

Useful code/data changes from this smoke:

- `ai/tutor/filter_teacher_by_targets.py` can now promote active targets back
  into teacher JSONL for conversion/training.
- `ai/training/sample_action_value_data.py` now supports `--extra-repeat` for
  deliberate hard-state upweighting.

Next T1 active-loop step:

- generate neighboring MC1000 T1 states around the residual17 patterns instead
  of repeating the same 17 decisions;
- include both successful and failed override cases from the same pattern
  family;
- evaluate against residual17 and a fresh T1 holdout before replacing the
  current runtime model/gate.

## 2026-06-01 Top1 Target and Focused T1 Source Neighbors

The product target is Top1 correctness. Top3/Top15/Top20/Top24 are internal
safety or refinement pools only; they are not success metrics for the answer
shown to the player. A 24-of-24 pool is not pruning and should only be accepted
as a fallback/safety diagnostic, not as the target runtime behavior.

Because repeating the same 17 residual decisions did not improve Top1, a more
focused neighbor extractor was added:

- script: `ai/tutor/build_source_neighbor_targets.py`
- input seed targets:
  `ai/data/hybrid_t1t2_active_20260531/targets_t1_residual20_mc1000_final_miss_min1000_20260601.jsonl`
- output:
  `ai/data/hybrid_t1t2_active_20260531/targets_t1_residual17_source_neighbors_w50_k8_20260601.jsonl`

Extraction result:

- seed records: 17
- source files: 5
- turns: T1 only
- window: +/-50 source rows
- max per seed: 8 nearest neighbors
- written: 136 T1 targets
- seeds with neighbor: 17/17

Compatibility smoke:

- command path: `ai.training.generate_active_teacher`
- output:
  `ai/data/hybrid_t1t2_active_20260531/teacher_t1_residual17_source_neighbors_w50_k8_smoke_mc1_20260601.jsonl`
- result: 1/1 written, skipped 0

Next MC1000 command should be chunked/resume-safe because local T1 MC1000 has
previously cost about 45 seconds per record:

```powershell
python -m ai.tutor.run_active_teacher_chunks `
  --input ai\data\hybrid_t1t2_active_20260531\targets_t1_residual17_source_neighbors_w50_k8_20260601.jsonl `
  --output-dir ai\data\hybrid_t1t2_active_20260531\chunk_t1_residual17_source_neighbors_w50_k8_mc1000_20260601 `
  --merged-output ai\data\hybrid_t1t2_active_20260531\teacher_t1_residual17_source_neighbors_w50_k8_mc1000_20260601.jsonl `
  --turns 1 --mc-turns 1 --sims 1000 --chunk-size 2 --workers 1
```

This batch is aimed at improving model Top1 on the same local pattern families
that currently miss. After relabeling, train/evaluate a T1-only candidate
against:

- residual17 MC1000 Top1 recall/regret,
- local500 5-second operational Top1,
- bad override count,
- a fresh T1 holdout so the model does not overfit these 17 seeds.

## 2026-06-01 Focused T1 Neighbor MC1000 Slice

The focused source-neighbor batch was started locally in resume-safe chunks:

- chunk dir:
  `ai/data/hybrid_t1t2_active_20260531/chunk_t1_residual17_source_neighbors_w50_k8_mc1000_20260601`
- merged teacher:
  `ai/data/hybrid_t1t2_active_20260531/teacher_t1_residual17_source_neighbors_w50_k8_mc1000_20260601.jsonl`
- completed: 10/68 chunks
- records: 20/136
- skipped: 0
- turn/eval: T1 MC1000
- observed elapsed: 895.736s total, 44.787s per record
- candidate count: min 12, max 27, avg 22.5
- MC1000 best-vs-second margin: min 0.004, avg 1.674, max 6.940
- low-margin labels below 0.25: 5/20

The 20-record slice was suit-augmented and converted:

- suit24 teacher:
  `ai/data/hybrid_t1t2_active_20260531/teacher_t1_residual17_source_neighbors_w50_k8_mc1000_20_20260601_suit24.jsonl`
  - records: 480
- converted data:
  `ai/data/hybrid_t1t2_active_20260531/reranker_t1_residual17_source_neighbors_w50_k8_mc1000_20_suit24_20260601`
  - samples: 10,800

Two diagnostic fine-tunes were tested:

1. Base80k + new source-neighbor20 x5:
   - mixed data:
     `ai/data/hybrid_t1t2_active_20260531/reranker_mix_active61_base80k_plus_t1srcnbr20_mc1000_suit24_x5_20260601`
   - model:
     `ai/models/candidate_runs/tutor-route10-active61-t1srcnbr20-mc1000-suit24-x5-ft-t1only-lr1e7-20260601/model/action_value_best.pt`
2. Extra-only diagnostic, lr=1e-6:
   - model:
     `ai/models/candidate_runs/tutor-route10-active61-t1srcnbr20-mc1000-suit24-extraonly-diagnostic-lr1e6-20260601/model/action_value_best.pt`

Evaluation:

| model | eval set | Top1 | Top3 | Top10 | avg regret | score MAE | FL MAE | bust MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| active61 | source-neighbor20 | 15.0% | 50.0% | 95.0% | 2.699 | 3.151 | 10.7% | 9.5% |
| source-neighbor20 x5 | source-neighbor20 | 15.0% | 50.0% | 95.0% | 2.699 | 3.095 | 10.2% | 9.4% |
| extra-only diagnostic | source-neighbor20 | 35.0% | 70.0% | 100.0% | 2.424 | 2.928 | 6.5% | 8.9% |
| active61 | residual17 | 17.6% | 47.1% | 88.2% | 3.033 | 3.027 | 8.4% | 10.6% |
| source-neighbor20 x5 | residual17 | 17.6% | 47.1% | 88.2% | 3.033 | 3.042 | 8.1% | 10.6% |
| extra-only diagnostic | residual17 | 5.9% | 35.3% | 94.1% | 3.540 | 3.370 | 6.8% | 10.1% |

Decision:

- Do not promote either fine-tune.
- The extra-only model proves the slice is learnable, but it overfits and hurts
  the held residual17 states.
- The base-mixed model is too weak to move Top1.
- Continue MC1000 labeling toward the full 136 focused neighbors before another
  production-candidate T1 turn model.
- Low-margin MC1000 labels should either be filtered or downweighted when
  training a Top1-focused model, because 5/20 labels have margin below 0.25.

The focused batch was then extended locally:

- completed: 20/68 chunks
- records: 40/136
- skipped: 0
- turn/eval: T1 MC1000
- observed elapsed across merged records: 1,892.739s total, 47.318s per record
- candidate count: min 12, max 27, avg 23.32
- MC1000 best-vs-second margin: min 0.000, avg 2.652, max 15.460
- low-margin labels below 0.25: 6/40
- low-margin labels below 0.10: 4/40
- best-action FL avg: 27.96%
- best-action bust avg: 21.78%

The 40-record slice was also suit-augmented and converted:

- suit24 teacher:
  `ai/data/hybrid_t1t2_active_20260531/teacher_t1_residual17_source_neighbors_w50_k8_mc1000_40_20260601_suit24.jsonl`
  - records: 960
- converted data:
  `ai/data/hybrid_t1t2_active_20260531/reranker_t1_residual17_source_neighbors_w50_k8_mc1000_40_suit24_20260601`
  - samples: 22,392
  - skipped: 0
  - score mean/std: +4.758 / 6.748
  - bust mean: 24.6%
  - FL mean: 11.3%

Decision after 40 records:

- Still do not train a new production-candidate model from this small slice.
- The correct next step is to continue the resume-safe MC1000 job toward the
  full 136 focused neighbors, then train with low-margin filtering or
  downweighting.

The focused source-neighbor batch was completed locally:

- completed: 68/68 chunks
- records: 136/136
- skipped: 0
- turn/eval: T1 MC1000
- observed elapsed across merged records: 6,627.872s total, 48.734s per record
- candidate count: min 12, max 27, avg 23.25
- MC1000 best-vs-second margin: min 0.000, avg 2.307, max 15.460
- low-margin labels below 0.25: 32/136
- low-margin labels below 0.10: 23/136
- zero-margin labels: 4/136
- best-action FL avg: 25.23%
- best-action bust avg: 22.36%

Generated datasets:

- full suit24 teacher:
  `ai/data/hybrid_t1t2_active_20260531/teacher_t1_residual17_source_neighbors_w50_k8_mc1000_136_20260601_suit24.jsonl`
  - records: 3,264
- full converted data:
  `ai/data/hybrid_t1t2_active_20260531/reranker_t1_residual17_source_neighbors_w50_k8_mc1000_136_suit24_20260601`
  - samples: 75,888
- high-confidence margin>=0.25 teacher:
  `ai/data/hybrid_t1t2_active_20260531/teacher_t1_residual17_source_neighbors_w50_k8_mc1000_margin025_20260601.jsonl`
  - records: 104
- high-confidence suit24 converted data:
  `ai/data/hybrid_t1t2_active_20260531/reranker_t1_residual17_source_neighbors_w50_k8_mc1000_margin025_104_suit24_20260601`
  - samples: 60,192

T1-only fine-tunes from active61:

| model | training data | eval set | Top1 | Top3 | Top10 | avg regret | score MAE | FL MAE | bust MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| active61 | baseline | source-neighbor136 | 33.8% | 68.4% | 93.4% | 2.158 | 3.035 | 9.4% | 10.2% |
| t1srcnbr136 lr1e-7 | base80k + source-neighbor136 | source-neighbor136 | 34.6% | 68.4% | 93.4% | 2.133 | 2.982 | 8.9% | 10.1% |
| t1srcnbr104 margin025 lr1e-7 | base80k + margin>=0.25 source-neighbor104 | source-neighbor136 | 34.6% | 68.4% | 93.4% | 2.133 | 2.992 | 9.0% | 10.1% |
| t1srcnbr136 x3 lr3e-7 | base80k + source-neighbor136 x3 | source-neighbor136 | 34.6% | 69.1% | 93.4% | 2.151 | 2.848 | 7.6% | 9.9% |
| extra-only diagnostic lr1e-6 | source-neighbor136 only | source-neighbor136 | 39.0% | 77.2% | 97.1% | 1.853 | 2.953 | 5.6% | 9.5% |
| active61 | baseline | residual17 | 17.6% | 47.1% | 88.2% | 3.033 | 3.027 | 8.4% | 10.6% |
| t1srcnbr136 lr1e-7 | base80k + source-neighbor136 | residual17 | 17.6% | 47.1% | 88.2% | 3.033 | 3.063 | 8.1% | 10.5% |
| t1srcnbr104 margin025 lr1e-7 | base80k + margin>=0.25 source-neighbor104 | residual17 | 17.6% | 47.1% | 88.2% | 3.033 | 3.059 | 8.1% | 10.6% |
| t1srcnbr136 x3 lr3e-7 | base80k + source-neighbor136 x3 | residual17 | 17.6% | 47.1% | 94.1% | 2.858 | 3.199 | 7.4% | 10.3% |
| extra-only diagnostic lr1e-6 | source-neighbor136 only | residual17 | 23.5% | 47.1% | 100.0% | 3.077 | 3.531 | 6.8% | 9.6% |

Decision after 136 records:

- Do not promote the mixed T1 fine-tunes as production models. They barely move
  Top1 on the focused source-neighbor set and do not improve residual17 Top1.
- The extra-only diagnostic proves that the focused data is learnable, but it is
  not a safe production model because residual17 regret and score MAE worsen.
- The strongest practical signal is Top10 coverage, not immediate Top1:
  extra-only reaches residual17 Top10=100%, and x3 reaches residual17 Top10=94.1%.
- To approach final Top1 correctness under the 5s constraint, the next lever is
  T1 candidate/rerank policy: keep the correct action inside Top10 and use a
  better refinement/gate decision, rather than expecting a plain reranker
  fine-tune to make Top1 100% by itself.

## 2026-06-01 T1 Auxiliary Shortlist Check

An auxiliary T1 shortlist path was added to test whether the source-neighbor
extra-only diagnostic model can help candidate selection without replacing the
production scorer:

- code: `HybridConfig.t1_aux_shortlist_k`
- CLI: `--t1-aux-shortlist-model`, `--t1-aux-shortlist-k`
- behavior: keep the production model ranks, but union in auxiliary TopK
  candidates for T1 refinement and log `aux_model_rank`.

Residual17 MC1000 check:

| runtime | final Top1 | final avg regret | pool recall | sync recall | p95 ms |
|---|---:|---:|---:|---:|---:|
| base sync6/gate045 | 0.0% | 2.345 | 100.0% | 64.7% | 3,591 |
| aux Top10 sync6/gate045 | 5.9% | 2.108 | 100.0% | 70.6% | 3,765 |
| aux Top10 sync10/no gate | 29.4% | 1.092 | 100.0% | 94.1% | 5,019 |

The auxiliary model improves hard residual coverage, but sync10 is already on
the edge of the 5 second budget.

Normal local100 T1 MC300 check:

| runtime | final Top1 | final avg regret | pool recall | sync recall | p95 ms | worse overrides |
|---|---:|---:|---:|---:|---:|---:|
| base sync6/gate045 | 74.5% | 0.289 | 100.0% | 91.5% | 4,099 | 1 |
| aux Top10 sync6/gate045 | 70.2% | 0.496 | 100.0% | 85.1% | 4,240 | 1 |

Decision:

- Do not promote the auxiliary shortlist path to the default runtime.
- It is useful as a diagnostic/pruning experiment for hard residual states, but
  it hurts the normal T1 distribution.
- For strict final Top1, the missing piece is not merely Pool20/Top24 coverage.
  The synchronous reranker must rank the refined candidates correctly within 5
  seconds.

Next Top1-first target:

1. Keep the current best runtime as the default.
2. Use auxiliary ranks only for offline miss mining.
3. Improve T1 refinement/gating with labels that directly say whether the
   refinement top action matches the teacher Top1, not only whether its sampled
   score is higher.
4. Treat MC300 as a smoke signal; use MC1000+ or exact labels for Top1 training.

## 2026-06-01 T1 Gate Top1 Objective Replay

The T1 override gate trainer now supports threshold selection by final Top1
instead of accepted score gain:

- code: `ai/tutor/train_t1_override_gate.py`
- new options: `--threshold-objective top1`, `--threshold-objective top1_then_regret`,
  and `--threshold-step`.
- rows now track `teacher_best_score` for future regret-aware threshold
  selection.

A replay tool was added so gate thresholds can be tested on saved no-gate
runtime rows without rerunning recursive refinement:

- code: `ai/tutor/replay_t1_override_gate.py`
- purpose: apply a gate and threshold to saved `results.jsonl` rows and report
  final Top1/regret after accepting or rejecting each T1 override.

Combined Top1-objective gate:

- output:
  `ai/data/hybrid_t1t2_active_20260531/t1_override_gate_combined_top1_objective_20260601.json`
- training rows: 356
- labels: accept 149, reject 207
- recommended threshold by training Top1 objective: 0.19

Replay on local500 T1 no-gate MC300 rows:

| gate | threshold | T1 Top1 | T1 avg regret | accepted | bad Top1 accepts | missed Top1 rejects |
|---|---:|---:|---:|---:|---:|---:|
| neighbor720 old | 0.45 | 58.5% | 0.459 | 128 | 10 | 0 |
| neighbor720 old | 0.58 | 58.9% | 0.466 | 123 | 8 | 1 |
| combined Top1-objective | 0.20 | 58.1% | 0.515 | 120 | 9 | 2 |

Short actual local100 T1 MC300 check for neighbor720 threshold 0.58:

| threshold | decisions | final Top1 | final avg regret | p95 ms | violations |
|---:|---:|---:|---:|---:|---:|
| 0.45 | 47 | 74.5% | 0.289 | 4,099 | 0 |
| 0.58 | 47 | 74.5% | 0.352 | 4,094 | 0 |

Decision:

- Do not promote the combined Top1-objective gate; it does not beat the old
  neighbor720 gate on local500 replay.
- Do not replace threshold 0.45 yet. Threshold 0.58 gives one extra Top1 on
  local500 replay but worse regret, and the local100 actual check is tied on
  Top1 with worse regret.
- The useful durable change is the tooling: future T1 gates can now be selected
  by Top1 directly and replayed cheaply before spending time on full recursive
  runtime evaluation.

## 2026-06-01 MC1000-Safe Runtime Active Targets

A runtime target collector was added for the Top1 active loop:

- code: `ai/tutor/collect_runtime_active_targets.py`
- purpose: merge multiple runtime result files, require MC1000+ or exact
  teachers, deduplicate states, and emit both active targets and matching
  teacher JSONL.
- default safety: `min_teacher_sims=1000`, `min_teacher_margin=0.25`.

The collector was run over three existing T1 MC1000 runtime evaluations:

- neighbor720 MC1000 gate-feature set
- T1 residual200 MC1000 gate-feature set
- T1 residual20 MC1000 gate045 set

Output:

- targets:
  `ai/data/hybrid_t1t2_active_20260531/targets_t1_mc1000_runtime_active_loop_margin025_20260601.jsonl`
- teacher:
  `ai/data/hybrid_t1t2_active_20260531/teacher_t1_mc1000_runtime_active_loop_margin025_20260601.jsonl`

Extraction result:

| item | count |
|---|---:|
| source runtime rows | 579 |
| written target states | 112 |
| matching teacher records | 112 |
| skipped low teacher margin | 181 |
| skipped no requested reason | 273 |
| duplicate states skipped | 13 |
| neighbor720 targets | 68 |
| residual200 targets | 33 |
| residual20 targets | 11 |
| runtime final Top1 misses | 112 |
| bad overrides among targets | 20 |

The 112 teacher records were suit-augmented and converted:

- suit24 teacher:
  `ai/data/hybrid_t1t2_active_20260531/teacher_t1_mc1000_runtime_active_loop_margin025_20260601_suit24.jsonl`
  - records: 2,688
- reranker data:
  `ai/data/hybrid_t1t2_active_20260531/reranker_t1_mc1000_runtime_active_loop_margin025_suit24_20260601`
  - samples: 64,800

A diagnostic T1-only fine-tune was run:

- mixed data:
  `ai/data/hybrid_t1t2_active_20260531/reranker_mix_active61_base80k_plus_t1_runtime112_mc1000_margin025_suit24_20260601`
- model:
  `ai/models/candidate_runs/tutor-route10-active61-t1runtime112-mc1000-margin025-suit24-ft-t1only-lr1e7-20260601/model/action_value_best.pt`

Evaluation:

| eval set | model | Top1 | Top3 | Top10 | avg regret | score MAE |
|---|---|---:|---:|---:|---:|---:|
| residual17 MC1000 | active61 | 17.6% | 47.1% | 88.2% | 3.033 | 3.027 |
| residual17 MC1000 | runtime112 FT | 17.6% | 47.1% | 88.2% | 3.033 | 3.043 |
| source-neighbor136 MC1000 | active61 | 33.8% | 68.4% | 93.4% | 2.158 | 3.035 |
| source-neighbor136 MC1000 | runtime112 FT | 34.6% | 68.4% | 93.4% | 2.125 | 3.004 |
| runtime112 MC1000 | active61 | 13.4% | 34.8% | 88.4% | 2.987 | 3.450 |
| runtime112 MC1000 | runtime112 FT | 13.4% | 35.7% | 89.3% | 2.984 | 3.425 |

Decision:

- Do not promote the runtime112 fine-tune. It slightly improves Top3/Top10 and
  calibration on the mined runtime set, but it does not improve Top1.
- The current bottleneck is still T1 final selection, not just adding more
  similarly-shaped MC1000 examples to the scorer.
- The new target/teacher data is useful as a standing high-confidence
  active-loop set for future gate/selector work.

Next Top1-first target:

1. Keep mining MC1000+ final misses with the collector, but avoid treating
   MC300-only rows as hard Top1 truth.
2. Use the runtime112 set for training/evaluating a selector that chooses among
   model Top1, recursive-refined Top1, and possibly auxiliary shortlist
   candidates.
3. If the selector cannot beat the current gate, improve T1 refinement itself:
   more stable sampling, better child beam, or a Rust-backed T1 recursive path.

## 2026-06-01 T1 Selector Rows and Sync Oracle Audit

Top1 correctness remains the product target. The desired end state is strict
final Top1 = 100%, not "Top15/Top20 contains the answer". The pool is only a
safety net that lets the 5-second runtime inspect enough candidates.

Candidate-level selector export was added:

- code: `ai/tutor/evaluate_hybrid_refinement_teacher.py`
- new option: `--write-selector-rows`
- output: `selector_rows.jsonl`, one row per candidate with teacher rank,
  pool/sync/refined membership, model score, refined score, and action text.

A selector analysis tool was added:

- code: `ai/tutor/analyze_t1_selector_rows.py`
- purpose: measure pool oracle, sync oracle, refined oracle, and simple
  candidate-selection policies without rerunning recursive refinement.

The T1 runtime can now optionally choose the final refined candidate with a
model/refined-score blend:

- code: `ai/tutor/hybrid_t1t2.py`
- new options:
  - `--t1-selection-policy refined_score`
  - `--t1-selection-policy refined_plus_model`
  - `--t1-selection-model-weight <float>`

This is intentionally experimental and is not promoted by default.

Selector audit results:

| set | decisions | pool oracle Top1 | sync oracle Top1 | refined oracle Top1 | model Top1 | refined-score Top1 | best simple selector |
|---|---:|---:|---:|---:|---:|---:|---|
| T1 residual20 MC1000 | 20 | 100.0% | 70.0% | 70.0% | 20.0% | 5.0% | model Top1, 20.0% |
| T1 runtime112 MC1000 | 112 | 100.0% | 71.4% | 70.5% | 13.4% | 24.1% | refined - 1.0 * model, 32.1% |

Actual runtime check for the experimental `refined_plus_model` selector on
residual20:

| setting | decisions | final Top1 | final avg regret | p95 ms | violations |
|---|---:|---:|---:|---:|---:|
| refined_score default | 20 | 5.0% | 1.998 | 2,973 | 0 |
| refined - 1.0 * model | 20 | 10.0% | 2.049 | 2,961 | 0 |

Decision:

- Do not promote `refined_plus_model` yet. It improves one in-sample selector
  set, but it does not beat model Top1 on residual20.
- The important diagnosis is the upper bound split:
  - Pool20 already contains the MC1000 teacher Top1 on these hard T1 sets.
  - Sync/refined candidates contain the teacher Top1 only about 70%-71% of the
    time with `sync_exact_k=6`.
  - Therefore Top1 100% cannot be reached by a selector alone; the 5-second
    candidate-selection/refinement stage must bring teacher Top1 into the
    sync/refined set much more often.
- The next implementation target is a Top1-oriented T1 sync candidate policy:
  choose which 6-10 candidates receive recursive refinement based on predicted
  teacher-best probability, not only model rank or sampled refined score.

## 2026-06-01 T1 Sync Candidate Selector Pilot

A first Top1-oriented sync candidate selector was added.  It is trained from
`selector_rows.jsonl`, using only features available before recursive
refinement starts:

- code: `ai/tutor/train_t1_sync_selector.py`
- runtime support: `ai/tutor/hybrid_t1t2.py`
- evaluation support: `ai/tutor/evaluate_hybrid_refinement_teacher.py`
- runtime options:
  - `--t1-sync-selection-policy selector`
  - `--t1-sync-selector <selector.json>`

Trained selector:

- train rows:
  `ai/data/hybrid_t1t2_active_20260531/eval_t1_runtime112_mc1000_selector_rows_20260601/selector_rows.jsonl`
- output:
  `ai/data/hybrid_t1t2_active_20260531/t1_sync_selector_runtime112_train_residual20_eval_20260601.json`

Offline sync-recall check:

| eval set | policy | recall@3 | recall@6 | recall@8 | recall@10 |
|---|---|---:|---:|---:|---:|
| runtime112 train | model rank | 34.8% | 71.4% | 79.5% | 88.4% |
| runtime112 train | sync selector | 47.3% | 74.1% | 83.0% | 90.2% |
| residual20 eval | model rank | 45.0% | 70.0% | 85.0% | 90.0% |
| residual20 eval | sync selector | 35.0% | 65.0% | 90.0% | 100.0% |

Actual residual20 MC1000 runtime checks:

| setting | sync recall | final Top1 | final avg regret | p95 ms | max ms |
|---|---:|---:|---:|---:|---:|
| selector sync8, sims32 child4 | 90.0% | 20.0% | 1.223 | 3,960 | 5,038 |
| selector sync9, sims32 child4 | 95.0% | 20.0% | 1.223 | 4,512 | 5,018 |
| selector sync10, sims32 child4 | 100.0% | 30.0% | 0.911 | 5,018 | 5,042 |
| selector sync10, sims32 child4, final blend -1 | 100.0% | 35.0% | 0.962 | 5,024 | 5,031 |
| selector sync10, sims24 child4, final blend -1 | 100.0% | 20.0% | 1.073 | 4,099 | 5,010 |
| selector sync10, sims32 child3, final blend -1 | 100.0% | 40.0% | 0.876 | 3,792 | 4,597 |

Decision:

- This is the first setting that improves all three relevant T1 hard-set
  signals at once: sync recall, final Top1, and runtime.
- Do not declare it production-best from only 20 positions.  It needs a broader
  local500/MC1000 validation before replacing the current operational setting.
- The result confirms the right direction for Top1 100%:
  1. Put teacher Top1 into the sync/refined set as close to 100% as possible.
  2. Train a stronger final selector/gate so the refined set chooses teacher
     Top1 instead of merely reducing regret.
  3. Feed the remaining misses back into the MC1000+ active loop.

## 2026-06-01 T1 Sync Selector Wider Validation and Final Selector Pilot

The sync selector setting was validated beyond the 20-position pilot:

- setting:
  `sync10 + t1_mc_sims=32 + child_sims=3 + sync selector`
- initial final policy in the runtime run:
  `refined_score - 1.0 * model_score`

Runtime checks:

| eval set | decisions | pool recall | sync recall | final Top1 | final avg regret | p95 ms | max ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| residual17 MC1000 | 17 | 100.0% | 100.0% | 41.2% | 1.002 | 4,595 | 4,595 |
| source-neighbor136 MC1000 | 136 | 99.3% | 89.0% | 47.8% | 1.184 | 4,220 | 5,024 |

The wider source-neighbor run shows the sync selector is useful but not enough:

- It improves final Top1 over raw model Top1 from 33.8% to 47.8%.
- It keeps p95 under 5 seconds.
- But sync recall is only 89.0%, so the true teacher Top1 is still absent from
  the 5-second refined set in about 11% of these positions.
- One position is already missing from Pool20, so Top1 100% is impossible for
  this pool on the full source-neighbor136 set until the upstream model/pool
  improves.

Replaying final-selection policies over the same refined candidates showed that
the `-1.0 * model_score` blend is not generally reliable:

| eval set | refined-score Top1 | refined + 0.25 model Top1 | refined - 1.0 model Top1 |
|---|---:|---:|---:|
| residual17 MC1000 | 29.4% | 29.4% | 41.2% |
| residual20 MC1000 | 30.0% | 30.0% | 40.0% |
| source-neighbor136 MC1000 | 60.3% | 62.5% | 47.8% |

A T1 final selector was added so the final choice can be learned after
refinement:

- code: `ai/tutor/train_t1_final_selector.py`
- runtime support:
  - `--t1-selection-policy selector`
  - `--t1-final-selector <selector.json>`

Final selector experiments:

| selector training rows | eval set | selector Top1 | baseline refined-score Top1 | verdict |
|---|---|---:|---:|---|
| runtime112 | residual20 | 45.0% | 30.0% | improves hard residual |
| runtime112 | residual17 | 47.1% | 29.4% | improves hard residual |
| runtime112 | source-neighbor136 | 44.4% | 60.9% | not general enough |
| source-neighbor136 | residual20 | 25.0% | 30.0% | overfits source-neighbor |
| source-neighbor136 | residual17 | 23.5% | 29.4% | overfits source-neighbor |
| runtime112 + source-neighbor136 | residual20 | 35.0% | 30.0% | modest improvement |
| runtime112 + source-neighbor136 | residual17 | 41.2% | 29.4% | modest improvement |

Decision:

- Do not promote any final selector yet.  They are distribution-sensitive.
- Keep the T1 sync selector as a promising refinement-candidate path, but treat
  it as experimental until it improves a broader heldout without worsening max
  regret.
- The next Top1-first work should collect final-selection misses from both
  residual-style hard states and source-neighbor-style states, then train the
  final selector on a mixed heldout protocol instead of optimizing one slice.

## 2026-06-01 T1 Final Selector Heldout and Active Miss Collection

The final selector trainer now supports safer Top1 training/evaluation:

- code: `ai/tutor/train_t1_final_selector.py`
- added `--min-teacher-margin` to filter out low-confidence MC1000 Top1 labels.
- added `--heldout-fraction` and `--heldout-seed` for group-level heldout
  evaluation.

MC1000 margin audit on available T1 final-selector rows:

| set | groups | teacher Top1 in refined | margin >= 0.25 | margin >= 0.5 |
|---|---:|---:|---:|---:|
| runtime112 | 112 | 79 | 112 | 88 |
| source-neighbor136 | 136 | 119 | 104 | 90 |
| residual20 | 20 | 20 | 18 | 15 |
| residual17 | 17 | 17 | 17 | 15 |

Mixed final selector heldout run:

- train rows:
  - runtime112 selector rows
  - source-neighbor136 selector rows
  - residual20 selector rows
- filter: `min_teacher_margin=0.25`
- split: 25% group-level heldout, seed 7
- output:
  `ai/data/hybrid_t1t2_active_20260531/t1_final_selector_mixed_margin025_heldout25_20260601.json`

| split/eval | policy | Top1 | avg regret | max regret |
|---|---|---:|---:|---:|
| train fit | refined_score | 54.8% | 0.659 | 7.629 |
| train fit | final selector | 61.6% | 0.544 | 7.629 |
| heldout | refined_score | 56.3% | 0.580 | 5.972 |
| heldout | final selector | 60.4% | 0.740 | 11.332 |
| residual17 external | refined_score | 29.4% | 1.288 | 5.972 |
| residual17 external | final selector | 47.1% | 0.610 | 2.859 |

Decision:

- Still do not promote the final selector.  Heldout Top1 improves, but heldout
  average regret and max regret get worse.  That is not acceptable for a
  gameplay answer selector.
- The result is still useful: it shows the selector can learn some residual
  hard cases, but it needs regret-aware training or a gate before runtime
  promotion.

A final-selection miss collector was added:

- code: `ai/tutor/collect_t1_final_selector_misses.py`
- purpose: convert final-selection misses from `selector_rows.jsonl` back into
  teacher/target JSONL for the next active loop.

Collected high-confidence source-neighbor136 misses:

| policy | margin filter | groups | hits | misses | output teacher |
|---|---:|---:|---:|---:|---|
| refined_score | >= 0.25 | 103 | 74 | 29 | `teacher_t1_final_refined_score_misses_srcnbr136_margin025_20260601.jsonl` |
| mixed final selector | >= 0.25 | 103 | 72 | 31 | `teacher_t1_final_selector_misses_srcnbr136_margin025_20260601.jsonl` |

Next Top1-first target:

1. Use the refined-score miss set first; it is the current more stable baseline.
2. Merge it with residual-style final-selection misses.
3. Train a regret-aware final gate/selector that improves Top1 without
   increasing heldout max regret.

## 2026-06-01 Regret-Aware T1 Final Selector

The final selector trainer now supports a regret-aware objective:

- code: `ai/tutor/train_t1_final_selector.py`
- added `--regret-weight`
- added `--regret-cap`
- added `--regret-scale`

The loss is now:

`Top1 cross entropy + regret_weight * expected_teacher_regret`

where teacher regret is `teacher_best_score - candidate_teacher_score`, with an
optional cap.  This keeps the model aligned with Top1 while discouraging
large-regret selections.

Residual-style final-selection misses were collected and merged with the
source-neighbor final-selection misses:

| source | margin filter | groups | hits | misses |
|---|---:|---:|---:|---:|
| source-neighbor136 refined_score | >= 0.25 | 103 | 74 | 29 |
| residual17 refined_score | >= 0.25 | 17 | 5 | 12 |
| residual20 refined_score | >= 0.25 | 18 | 6 | 12 |

Merged active-target output:

- `ai/data/hybrid_t1t2_active_20260531/targets_t1_final_refined_score_misses_mixed_margin025_20260601.jsonl`
- read: 53
- written after state dedupe: 41
- duplicates: 12

Regret-aware selector runs used:

- train rows:
  - runtime112 selector rows
  - source-neighbor136 selector rows
  - residual20 selector rows
- external eval:
  - residual17 selector rows
- filter: `min_teacher_margin=0.25`
- split: 25% group heldout, seed 7
- cap: `regret_cap=8`

Heldout comparison:

| final selector | heldout Top1 | heldout avg regret | heldout max regret | residual17 Top1 | residual17 avg regret | residual17 max regret |
|---|---:|---:|---:|---:|---:|---:|
| refined_score baseline | 56.3% | 0.580 | 5.972 | 29.4% | 1.288 | 5.972 |
| no regret term | 60.4% | 0.740 | 11.332 | 47.1% | 0.610 | 2.859 |
| regret weight 0.25 | 62.5% | 0.714 | 11.332 | 52.9% | 0.545 | 2.859 |
| regret weight 0.50 | 62.5% | 0.714 | 11.332 | 52.9% | 0.545 | 2.859 |
| regret weight 1.00 | 66.7% | 0.431 | 3.144 | 47.1% | 0.577 | 2.859 |

Best heldout candidate:

- `ai/data/hybrid_t1t2_active_20260531/t1_final_selector_mixed_margin025_heldout25_regret100_20260601.json`

Runtime smoke/full residual17 check with regret-weight 1.0 selector:

| setting | decisions | sync recall | final Top1 | final avg regret | final max regret | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| sync selector + final blend -1 | 17 | 100.0% | 41.2% | 1.002 | 5.523 | 4,595 |
| sync selector + regret-aware final selector | 17 | 100.0% | 41.2% | 0.708 | 2.859 | 4,603 |

Decision:

- This is a real improvement over the previous blend on residual17 because it
  keeps Top1 while cutting both average and max regret.
- It is not yet a global promotion because source-neighbor136 has not been
  rerun end-to-end with this final selector.  Existing selector-row replay says
  the regret-aware selector is promising, but runtime confirmation is still
  required before replacing the operational final policy.
- The next highest-value check is a full source-neighbor136 runtime run with:
  `sync10 + t1_mc_sims=32 + child_sims=3 + regret-aware final selector`.

## 2026-06-01 Source-Neighbor136 Regret-Aware Runtime Confirmation

The full source-neighbor136 MC1000 runtime was rerun end-to-end with the
regret-aware final selector:

- output:
  `ai/data/hybrid_t1t2_active_20260531/eval_t1_source_neighbor136_mc1000_sync_selector_k10_child3_regret_final_selector_20260601`
- sync policy:
  `sync10 + t1_mc_sims=32 + child_sims=3 + t1_sync_selector`
- final policy:
  `t1_final_selector_mixed_margin025_heldout25_regret100_20260601`

Comparison against the previous `refined_score - 1.0 * model_score` final
blend:

| setting | decisions | final Top1 | final avg regret | final max regret | sync recall | p95 ms | max ms | worse overrides |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| old final blend -1 | 136 | 47.8% | 1.184 | 21.270 | 89.0% | 4,220 | 5,024 | 36 |
| regret-aware final selector | 136 | 61.8% | 0.555 | 11.763 | 89.0% | 4,257 | 5,025 | 16 |

This confirms the regret-aware final selector improves the wider
source-neighbor distribution as well as residual17.  The remaining bottleneck
is now split:

- Pool recall is 99.3%, so at least one source-neighbor136 position is still
  unrecoverable from the current Pool20.
- Sync/refined oracle recall is 89.0%/87.5%, so about 11%-12.5% of positions
  still cannot be fixed by any final selector inside the current 5-second
  refined set.
- Within the refined set, final selection is much better but still not perfect:
  final Top1 is 61.8% while refined-oracle is 87.5%.

Selector-row replay over the same refined candidates still shows that a simple
`refined_score + 0.25 * model_score` would be slightly higher on this one set:

| final policy replay | Top1 | avg regret | max regret |
|---|---:|---:|---:|
| regret-aware final selector | 61.8% | 0.555 | 11.763 |
| refined_score | 60.3% | 0.555 | 11.763 |
| refined_score + 0.25 model_score | 62.5% | 0.526 | 11.763 |

However, the simple blend was weak on the residual hard sets, while the
regret-aware selector improved residual17 regret materially.  Treat the
regret-aware selector as the stronger promotion candidate, but keep the blend
as a comparison baseline.

High-confidence remaining misses were collected for the next active loop:

- output teacher:
  `ai/data/hybrid_t1t2_active_20260531/teacher_t1_final_regret_selector_misses_srcnbr136_margin025_20260601.jsonl`
- output targets:
  `ai/data/hybrid_t1t2_active_20260531/targets_t1_final_regret_selector_misses_srcnbr136_margin025_20260601.jsonl`
- margin filter: `>= 0.25`
- groups: 103
- hits: 72
- misses: 31

Decision:

- Promote the regret-aware final selector to the current T1 experimental
  runtime candidate for MC1000-safe testing.
- Do not call the T1 problem solved: Top1 is still bounded by Pool20/sync
  recall, and the remaining high-confidence final misses should feed the next
  active loop.
