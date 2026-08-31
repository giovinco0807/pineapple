# T3 EV Precision Status 2026-06-14

## Summary

T3 exact teacher generation and EV-focused retraining were moved to a D-drive
pipeline.  The run used the current FL rewards:

- QQ: 0
- KK: 10.7
- AA: 29.9
- trips: 63.5

This is not enough yet for safe Top10 pruning.  Hard-negative retraining helped,
but even on the same 20k exact dataset, Top10 still has EV-loss tail cases.

After that, a broader branch-expanded 99,903-row T3 exact dataset was generated
from T0/T1/T2 routes on D drive.  This materially improved Top20 safety on the
internal dataset, but Top10/Top15 still leak large EV-loss tail cases.

## Code Added

- `ai/tutor/run_t3_ev_precision_pipeline.py`
  - Selects T3 inputs.
  - Runs Rust `t3_exact_solver`.
  - Converts exact output to teacher JSONL.
  - Converts teacher JSONL to candidate-level reranker arrays.
  - Trains separate BB/BTN models.
  - Evaluates TopK + exact-rerank EV loss.
- `ai/tutor/mine_t3_ev_loss_misses.py`
  - Mines TopK EV-loss misses.
  - Writes per-K miss JSONL files.
  - Writes `group_sample_weights.npy` for hard-negative ranking batches.
- `ai/tutor/build_branch_expanded_t3_targets.py`
  - Added `--max-routes-per-position` so T3 inputs can be sampled from many
    root deals instead of filling the dataset from only the first few roots.

## Artifacts

- Smoke run:
  - `D:\ofc-pineapple-data\t3_ev_precision_20260614\smoke`
- 20k run:
  - `D:\ofc-pineapple-data\t3_ev_precision_20260614\run20k`
- Base models:
  - `D:\ofc-pineapple-data\t3_ev_precision_20260614\run20k\models\t3-ev-exact-bb\action_value_best.pt`
  - `D:\ofc-pineapple-data\t3_ev_precision_20260614\run20k\models\t3-ev-exact-btn\action_value_best.pt`
- Hard-negative models:
  - `D:\ofc-pineapple-data\t3_ev_precision_20260614\run20k\models\t3-ev-hardneg-bb\action_value_best.pt`
  - `D:\ofc-pineapple-data\t3_ev_precision_20260614\run20k\models\t3-ev-hardneg-btn\action_value_best.pt`
- Evaluation:
  - `D:\ofc-pineapple-data\t3_ev_precision_20260614\run20k\eval\t3_ev_exact_models.json`
  - `D:\ofc-pineapple-data\t3_ev_precision_20260614\run20k\eval\t3_ev_hardneg_models.json`
- Misses:
  - `D:\ofc-pineapple-data\t3_ev_precision_20260614\run20k\misses`
- Branch-expanded 100k run:
  - `D:\ofc-pineapple-data\t3_ev_precision_20260614\branch100k`
  - `D:\ofc-pineapple-data\t3_ev_precision_20260614\branch100k_exact_run1`
- Branch-expanded hard-negative models:
  - `D:\ofc-pineapple-data\t3_ev_precision_20260614\branch100k_exact_run1\models\t3-ev-hardneg-bb\action_value_best.pt`
  - `D:\ofc-pineapple-data\t3_ev_precision_20260614\branch100k_exact_run1\models\t3-ev-hardneg-btn\action_value_best.pt`
- External holdout checks:
  - `D:\ofc-pineapple-data\t3_ev_precision_20260614\external_holdout20k`
  - `D:\ofc-pineapple-data\t3_ev_precision_20260614\external_holdout5k_diverse`
  - `D:\ofc-pineapple-data\t3_ev_precision_20260614\external_holdout2k_seed20260706`

## 20k Runtime

- Rust exact: 499.2s
- Rust exact to teacher: 14.5s
- Teacher to reranker arrays: 55.8s
- BB base training: 168.2s
- BTN base training: 170.4s
- Base model evaluation: 91.5s

The 20k exact run produced:

- T3 rows: 20,000
- Candidate samples: 269,745
- Output size under `run20k`: about 1.25GB

## Base Model Result

Evaluated on the same 20k exact rows:

| Pool | Recall | Mean EV Loss | Max EV Loss | EV Loss > 0.1 | EV Loss > 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top5 | 86.61% | 0.1619 | 35.8772 | 1,814 | 791 |
| Top10 | 98.01% | 0.0196 | 23.0238 | 244 | 103 |
| Top15 | 99.63% | 0.0055 | 17.5788 | 47 | 23 |
| Top20 | 99.97% | 0.0004 | 4.9391 | 5 | 1 |

Top1 is still weak:

- Top1: 40.27%
- Top3: 67.22%
- Top5: 79.95%
- Top10: 94.74%

## Hard-Negative Result

Top10 EV-loss misses from the base model were mined:

- Top10 loss >= 0.1: 244 rows
- `group_sample_weights.npy` was written with 244 weighted groups.

After hard-negative retraining:

| Pool | Recall | Mean EV Loss | Max EV Loss | EV Loss > 0.1 | EV Loss > 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top5 | 87.86% | 0.1380 | 62.1759 | 1,636 | 686 |
| Top10 | 98.62% | 0.0101 | 11.4407 | 155 | 51 |
| Top15 | 99.81% | 0.0009 | 3.0120 | 23 | 7 |
| Top20 | 99.995% | 0.00008 | 1.5705 | 1 | 1 |

Top1 improved but is still far from usable as a final decision:

- Top1: 42.07%
- Top3: 68.14%
- Top5: 80.65%
- Top10: 95.00%

## Branch-Expanded 99,903 Runtime

Input rows were generated from broader routes:

- T0 model: top50 branches
- T1/T2 model: top10 branches
- Positions: BB and BTN
- Opponent visible board and known discards are included in the generated T3
  input rows.

Runtime:

- Branch input generation: 644.1s
- Rust exact: 3241.1s
- Rust exact to teacher: 70.8s
- Teacher to reranker arrays: 414.7s
- BB base training: 970.2s
- BTN base training: 851.9s
- Base model evaluation: 498.8s
- Hard-negative BB training: 796.4s
- Hard-negative BTN training: 781.3s
- Hard-negative evaluation: 461.8s

The run produced:

- T3 rows: 99,903
- Candidate samples: 1,575,240

## Branch-Expanded Base Result

Evaluated on the same 99,903 exact rows:

| Pool | Recall | Mean EV Loss | Max EV Loss | EV Loss > 0.1 | EV Loss > 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top5 | 90.33% | 0.0738 | 26.0343 | 5,339 | 1,879 |
| Top10 | 98.90% | 0.0125 | 21.6478 | 581 | 298 |
| Top15 | 99.83% | 0.0031 | 21.6478 | 111 | 61 |
| Top20 | 99.991% | 0.000059 | 2.2721 | 4 | 3 |

## Branch-Expanded Hard-Negative Result

Top10 loss >= 0.1 rows from the branch-expanded base model were mined and
weighted:

- Top10 loss >= 0.1: 581 rows
- `group_sample_weights.npy` was written with max weight 12.0.

After hard-negative retraining:

| Pool | Recall | Mean EV Loss | Max EV Loss | EV Loss > 0.1 | EV Loss > 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top5 | 91.25% | 0.0684 | 31.3883 | 5,104 | 1,800 |
| Top10 | 99.02% | 0.0069 | 16.1721 | 462 | 169 |
| Top15 | 99.87% | 0.0006 | 12.3953 | 42 | 16 |
| Top20 | 99.997% | 0.0000004 | 0.0378 | 0 | 0 |

Top1 remains far from a final decision model:

- Top1: 47.20%
- Top3: 75.23%
- Top5: 86.01%
- Top10: 96.14%

Position split after hard-negative retraining:

- BB Top20: recall 99.994%, max EV loss 0.0378
- BTN Top20: recall 100.000%, max EV loss 0.0000

## External Holdout Result

The first external 20k holdout was generated with `--max-outputs 20000` and no
per-root cap.  This filled from only about two root deals, which confirmed the
distribution problem but is too narrow as a final validation set:

- Rows: 20,000
- Top10 recall: 91.71%, max EV loss 15.4912, EV loss > 0.1: 794
- Top15 recall: 97.06%, max EV loss 9.8608, EV loss > 0.1: 266
- Top20 recall: 99.72%, max EV loss 3.2206, EV loss > 0.1: 30

Then `--max-routes-per-position` was added and a 50-root diversified holdout was
generated:

- Rows: 5,000
- Roots: 50
- Rows per root: 100
- Candidate samples: 78,903
- Generation time: 435.8s
- Rust exact + conversion + evaluation: 163.9s

Hard-negative model on this diversified external holdout:

| Pool | Recall | Mean EV Loss | Max EV Loss | EV Loss > 0.1 | EV Loss > 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top5 | 69.22% | 0.2067 | 17.4375 | 624 | 255 |
| Top10 | 91.36% | 0.0577 | 13.8295 | 179 | 85 |
| Top15 | 97.68% | 0.0220 | 13.8295 | 52 | 29 |
| Top20 | 99.74% | 0.0027 | 3.8945 | 6 | 6 |

This is the important result: the internal 99,903-row score was too optimistic.
The model is learning those route families, but it is not yet robust across
many root deals.

## Many-Root 20k Training Result

To fix the T0-card distribution problem, a new many-root training set was built
with `--max-routes-per-position 50`:

- Path: `D:\ofc-pineapple-data\t3_ev_precision_20260614\manyroot20k_train`
- Exact/training path:
  `D:\ofc-pineapple-data\t3_ev_precision_20260614\manyroot20k_exact_run1`
- Roots: 200
- Rows per root: 100
- T3 rows: 20,000
- Target-view T0 5-card patterns: 400
- Candidate samples: 315,315
- Input generation time: 1645.7s

Base many-root internal result:

| Pool | Recall | Mean EV Loss | Max EV Loss | EV Loss > 0.1 | EV Loss > 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top5 | 85.08% | 0.1348 | 24.8451 | 1,819 | 664 |
| Top10 | 98.10% | 0.0196 | 24.8451 | 244 | 109 |
| Top15 | 99.69% | 0.0031 | 7.0395 | 45 | 21 |
| Top20 | 99.985% | 0.0002 | 1.8549 | 2 | 2 |

Hard-negative retraining mined Top10 EV-loss misses:

- Top10 loss >= 0.1: 244 rows
- Top15 threshold misses: 49 rows
- Top20 threshold misses: 2 rows

Hard-negative many-root internal result:

| Pool | Recall | Mean EV Loss | Max EV Loss | EV Loss > 0.1 | EV Loss > 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top5 | 85.20% | 0.1366 | 24.8451 | 1,767 | 673 |
| Top10 | 98.21% | 0.0148 | 11.2159 | 216 | 82 |
| Top15 | 99.79% | 0.0015 | 8.3599 | 26 | 9 |
| Top20 | 99.995% | 0.0001 | 2.7833 | 1 | 1 |

Many-root hard-negative model on the earlier 50-root external holdout:

| Pool | Recall | Mean EV Loss | Max EV Loss | EV Loss > 0.1 | EV Loss > 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top5 | 76.86% | 0.1701 | 13.8295 | 582 | 222 |
| Top10 | 94.22% | 0.0491 | 10.2134 | 153 | 71 |
| Top15 | 98.60% | 0.0142 | 10.2134 | 36 | 20 |
| Top20 | 99.96% | 0.00004 | 0.1765 | 1 | 0 |

Compared with the earlier 10-root 100k hard-negative model on this same
external holdout:

- Top10 recall improved from 91.36% to 94.22%.
- Top20 EV loss > 0.1 dropped from 6 rows to 1 row.
- Top20 max EV loss dropped from 3.8945 to 0.1765.

The many-root direction is therefore better for generalization, even with fewer
total rows.  It still does not make Top10 safe.

## Random-T0 10k Training Result

The generator was updated to sample branches immediately after T0 as well as
after T1/T2.  This makes it practical to increase T0 root diversity with small
per-root branch counts.

Random-T0 input:

- Path: `D:\ofc-pineapple-data\t3_ev_precision_20260614\manyroot_random_t0_10k`
- Exact/training path:
  `D:\ofc-pineapple-data\t3_ev_precision_20260614\random_t0_10k_exact_run1`
- Roots: 1,000
- Rows per root: 10
- T3 rows: 10,000
- Target-view T0 5-card patterns: 2,000
- Candidate samples: 155,718
- Input generation time: 2185.5s

Base random-T0 internal result:

| Pool | Recall | Mean EV Loss | Max EV Loss | EV Loss > 0.1 | EV Loss > 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top5 | 83.28% | 0.1659 | 32.2592 | 999 | 380 |
| Top10 | 98.06% | 0.0190 | 20.9579 | 113 | 49 |
| Top15 | 99.69% | 0.0046 | 20.9579 | 20 | 10 |
| Top20 | 99.95% | 0.0023 | 20.9579 | 5 | 2 |

After hard-negative retraining:

| Pool | Recall | Mean EV Loss | Max EV Loss | EV Loss > 0.1 | EV Loss > 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top5 | 83.94% | 0.1378 | 31.7189 | 927 | 351 |
| Top10 | 98.41% | 0.0127 | 20.9579 | 85 | 33 |
| Top15 | 99.79% | 0.0036 | 20.9579 | 11 | 3 |
| Top20 | 99.98% | 0.0004 | 2.9458 | 2 | 1 |

Random-T0 hard-negative model on the earlier 50-root external holdout:

| Pool | Recall | Mean EV Loss | Max EV Loss | EV Loss > 0.1 | EV Loss > 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top5 | 77.42% | 0.1952 | 14.1833 | 637 | 258 |
| Top10 | 94.32% | 0.0475 | 14.1833 | 153 | 69 |
| Top15 | 99.00% | 0.0072 | 4.9805 | 25 | 13 |
| Top20 | 99.92% | 0.0008 | 2.6266 | 2 | 2 |

Compared with many-root20k on the same external holdout:

- Top10 is about flat: 94.22% to 94.32%.
- Top15 improved: EV loss > 0.1 from 36 rows to 25 rows.
- Top20 got worse in the rare tail: EV loss > 0.1 from 1 row to 2 rows, max
  loss from 0.1765 to 2.6266.

Interpretation: increasing random T0 coverage is directionally useful, but
10k rows with only 10 rows per root is too thin.  The next dataset should keep
high T0 diversity while increasing rows per root enough to cover T1/T2/T3
branch variation.

## External EV-Loss Reduction With Model Union

Single-model Top10/Top15 is still not safe enough.  A practical way to drive
external EV loss toward zero is to union candidate pools from multiple
specialists, then exact-rerank the union.

Added:

- `ai/tutor/evaluate_t3_exact_teacher_union.py`

Union sources:

- 10-root 100k hard-negative model
- many-root 20k hard-negative model
- random-T0 10k hard-negative model

On `external_holdout5k_diverse`:

| Union Pool | Recall | Mean EV Loss | Max EV Loss | EV Loss > 0.1 | Avg Pool | Max Pool |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Top5 each | 90.06% | 0.0688 | 13.2314 | 259 | 7.60 | 13 |
| Top8 each | 96.56% | 0.0317 | 10.5182 | 100 | 10.51 | 18 |
| Top10 each | 98.26% | 0.0197 | 5.7539 | 59 | 11.94 | 20 |
| Top15 each | 99.80% | 0.0012 | 2.8764 | 5 | 14.38 | 21 |
| Top20 each | 100.00% | 0.0000 | 0.0000 | 0 | 15.72 | 21 |

A fresh external holdout was then generated:

- Path: `D:\ofc-pineapple-data\t3_ev_precision_20260614\external_holdout5k_seed20260704`
- Roots: 250
- Rows per root: 20
- T3 rows: 5,000
- Candidate samples: 78,903

On this fresh holdout:

| Union Pool | Recall | Mean EV Loss | Max EV Loss | EV Loss > 0.1 | Avg Pool | Max Pool |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Top5 each | 90.48% | 0.0693 | 17.7657 | 247 | 7.58 | 14 |
| Top8 each | 97.06% | 0.0178 | 5.6764 | 70 | 10.56 | 18 |
| Top10 each | 98.40% | 0.0118 | 5.6764 | 40 | 12.00 | 20 |
| Top15 each | 99.76% | 0.0030 | 5.6764 | 8 | 14.40 | 21 |
| Top20 each | 100.00% | 0.0000 | 0.0000 | 0 | 15.72 | 21 |
| Top21 each | 100.00% | 0.0000 | 0.0000 | 0 | 15.78 | 21 |

Interpretation:

- For T3, model union Top20 + exact rerank currently eliminates EV loss on two
  external 5k holdouts.
- The average exact-rerank pool is only about 16 candidates, because the three
  models overlap heavily.
- Top15 union is close but still leaks rare EV-loss cases.
- Top10 union is not safe yet.

This suggests the near-term runtime path should use union Top20 for T3 while
training continues to shrink the pool toward Top15/Top10.

## Instant Model-Only Check

If the runtime must answer instantly from model scores only, the exact-rerank
result above is not enough evidence.  I added:

- `ai/tutor/evaluate_t3_exact_teacher_score_ensemble.py`

This evaluates raw model-score ranking against the exact T3 teacher data, with
no exact rerank.

On the fresh external holdout, the existing three-model score average was:

| Metric | Value |
| --- | ---: |
| Top1 recall | 33.68% |
| Top3 recall | 60.28% |
| Top5 recall | 73.66% |
| Top10 recall | 91.84% |
| Top15 recall | 97.64% |
| Top20 recall | 99.72% |
| Top1 mean EV loss | 0.7936 |
| Top1 p95 EV loss | 4.0701 |
| Top1 p99 EV loss | 10.1665 |
| Top1 max EV loss | 40.0653 |
| Top1 EV loss > 1.0 | 945 / 5,000 |

Worst Top1 miss:

- Position: BTN
- Dealt: `7c 2d 4s`
- Model action: `4s->middle; 7c->middle; discard 2d`
- Exact action: `2d->middle; 7c->bottom; discard 4s`
- EV loss: `40.0653`

I then trained a many-root20k Top1-loss specialist with `target_topk=1` and
`selection_metric=regret`:

- BB: `D:\ofc-pineapple-data\t3_ev_precision_20260614\manyroot20k_exact_run1\models\t3-ev-top1-bb\action_value_best.pt`
- BTN: `D:\ofc-pineapple-data\t3_ev_precision_20260614\manyroot20k_exact_run1\models\t3-ev-top1-btn\action_value_best.pt`

Internal validation improved to roughly 45-47% Top1, but external holdout
generalization got worse:

| Model scoring setup | Top1 recall | Top3 recall | Top10 recall | Top1 mean EV loss | Top1 max EV loss | EV loss > 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing 3-model average | 33.68% | 60.28% | 91.84% | 0.7936 | 40.0653 | 945 |
| many-root20k Top1 specialist only | 28.86% | 53.82% | 90.62% | 0.9986 | 38.0798 | 1,125 |
| Existing 3 + Top1 specialist average | 32.84% | 59.12% | 92.26% | 0.8511 | 38.0798 | 988 |

Conclusion: the Top1 specialist should not be adopted.  It appears to overfit
the 20k many-root train/dev distribution and does not solve instant model-only
decision quality.

## Action-Aware T3 Model Check

The action-aware 617-dim input adds 97 explicit action features on top of the
520-dim post-action board state.  These features include discard rank/suit,
semantic action id, row placement counts, placed-card row/rank counts, and
before/after row slot ratios.

Important code fix:

- `ai/tutor/benchmark_t2_t3_value_model.py`
- `ai/tutor/diagnose_t3_model_from_t2_miss.py`

The T3 inference path previously always encoded 520-dim states.  It now reads
`model.input_dim` and uses `adapt_np_state_with_action`, so 617-dim models are
actually evaluated with their action features.  520-dim checkpoints remain
compatible.

The 617-dim many-root20k models were trained here:

- `D:\ofc-pineapple-data\t3_ev_precision_20260614\manyroot20k_exact_run1\models\t3-ev-action617-bb\action_value_best.pt`
- `D:\ofc-pineapple-data\t3_ev_precision_20260614\manyroot20k_exact_run1\models\t3-ev-action617-btn\action_value_best.pt`

Model-only 617-dim single-model result on the fresh external holdout:

| Pool | Recall | Mean EV Loss | Max EV Loss | EV Loss > 0.1 | EV Loss > 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top1 | 35.70% | 1.1430 | 40.7514 | 2322 | 1232 |
| Top10 | 95.44% | 0.0284 | 9.9239 | 136 | 41 |
| Top15 | 99.02% | 0.0054 | 7.2072 | 23 | 8 |
| Top20 | 99.88% | 0.0010 | 2.5611 | 4 | 2 |

Interpretation:

- The action-aware model improves candidate-pool quality, especially Top15.
- It does not solve model-only Top1.  Top1 EV loss is still too large.

## Four-Model Ensemble After Action Features

I added `--method` to `ai/tutor/evaluate_t3_exact_teacher_score_ensemble.py`
so score aggregation can be compared:

- `score_mean`: raw denormalized EV score average
- `score_zmean`: per-model, per-position z-scored score average
- `rank_mean`: average rank
- `reciprocal_rank`: reciprocal-rank voting

Four sources:

- 10-root 100k hard-negative
- many-root20k hard-negative
- random-T0 10k hard-negative
- many-root20k action-aware 617-dim

Model-only score ensemble on the fresh external holdout:

| Method | Top1 Recall | Top1 Mean Loss | Top10 Mean Loss | Top15 Mean Loss | Top20 Mean Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| score_mean | 36.72% | 0.8203 | 0.0324 | 0.0099 | 0.0015 |
| score_zmean | 36.62% | 0.8329 | 0.0269 | 0.0098 | 0.0004 |
| rank_mean | 34.70% | 0.8615 | 0.0268 | 0.0098 | 0.0009 |
| reciprocal_rank | 35.60% | 0.8890 | 0.0287 | 0.0073 | 0.0009 |

This reaches mean EV loss below `0.01` for model-only Top15, but not for Top10
or Top1.

Four-model union + exact rerank on the fresh external holdout:

| Union Pool | Recall | Mean EV Loss | Max EV Loss | EV Loss > 0.1 | Avg Pool | Max Pool |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Top5 each | 93.54% | 0.0506 | 13.5252 | 190 | 8.20 | 15 |
| Top8 each | 98.02% | 0.0121 | 4.9798 | 57 | 11.10 | 19 |
| Top10 each | 99.06% | 0.0068 | 4.9798 | 25 | 12.49 | 20 |
| Top12 each | 99.52% | 0.0035 | 3.0052 | 13 | 13.52 | 21 |
| Top15 each | 99.88% | 0.0006 | 1.3254 | 4 | 14.74 | 21 |
| Top20 each | 100.00% | 0.0000 | 0.0000 | 0 | 15.76 | 21 |

This reaches mean EV loss below `0.01` for union Top10 + exact rerank.  It does
not eliminate the tail at Top10; Top20 is still the only zero-loss pool on this
holdout.

## Shape-Insurance Candidate Pool

After inspecting Top10 union misses, the main pattern was model over-selection
of top placements while exact-best often placed low/medium cards into
middle/bottom.  I added a deterministic shape-insurance rule that uses only the
candidate action shape, not teacher EV:

- prefer actions with no top placement
- prefer two bottom placements
- prefer low cards into bottom/middle
- penalize high-card top placements

Code:

- `ai/tutor/evaluate_t3_exact_teacher_union.py`
- `ai/tutor/t3_runtime.py`
- `ai/config/t3_ev_precision_pool_20260614.json`

The runtime path now also fixes 617-dim action-aware inference.  It builds
states per model input dimension, so 520-dim and 617-dim checkpoints can be
mixed correctly in one union.

Four-model union Top10 + exact rerank, with shape insurance:

| Holdout | Insurance | Recall | Mean EV Loss | Max EV Loss | EV Loss > 0.1 | Avg Pool | Max Pool |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed20260704 | 0 | 99.06% | 0.0068 | 4.9798 | 25 | 12.49 | 20 |
| seed20260704 | 1 | 99.38% | 0.0022 | 2.6455 | 12 | 13.08 | 21 |
| seed20260704 | 2 | 99.58% | 0.0010 | 2.2241 | 8 | 13.61 | 21 |
| seed20260704 | 3 | 99.66% | 0.0008 | 2.2241 | 6 | 14.12 | 21 |
| seed20260704 | 4 | 99.40% | 0.0024 | 2.6455 | 12 | 13.49 | 21 |
| diverse | 2 | 99.30% | 0.0055 | 5.7539 | 20 | 13.55 | 21 |
| diverse | 3 | 99.48% | 0.0023 | 2.4769 | 13 | 14.07 | 21 |
| diverse | 4 | 99.30% | 0.0047 | 4.9805 | 20 | 13.45 | 21 |

Adding a fourth general insurance candidate does not improve Top10; it worsens
the Top10 tail on both external holdouts.  Category-diversified insurance was
also tested and rejected because it increased Top10 EV loss on both holdouts.

Top15 each + 4 general insurance candidates is the better precision mode.  One
seed tail remained without a near-full expansion guard, so safer mode now
expands to all legal actions when the pool is within 3 actions of full:

| Holdout | Pool | Recall | Mean EV Loss | Max EV Loss | EV Loss > 0.1 | Avg Pool | Max Pool |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| seed20260704 | Top15+4+gap3 | 100.00% | 0.0000 | 0.0000 | 0 | 15.71 | 21 |
| diverse | Top15+4+gap3 | 100.00% | 0.0000 | 0.0000 | 0 | 15.69 | 21 |
| seed20260706 | Top15+4+gap3 | 100.00% | 0.0000 | 0.0000 | 0 | 15.58 | 21 |

A fresh 2,000-row diversified holdout was generated after the safer policy was
chosen:

- Path: `D:\ofc-pineapple-data\t3_ev_precision_20260614\external_holdout2k_seed20260706`
- Seed: `20260706`
- Roots: 100
- Rows: 2,000
- T0 model topK: 50
- T1/T2 model topK: 10
- Max routes per position: 10
- Input generation: 309.1s on CUDA
- Teacher candidates: 31,293

The older narrow `external_holdout20k` first 5,000 rows also passed
Top15+4+gap3 with mean/max EV loss `0.0000`, but that subset is mostly one
position/root family and should not be treated as a final robustness proof.

Runtime smoke:

- Fast command: `python -B -m ai.tutor.t3_runtime --config ai\config\t3_ev_precision_pool_20260614.json --pool-policy fast --limit 3 ...`
- Fast output: `D:\ofc-pineapple-data\t3_ev_precision_20260614\runtime_smoke_t3_pool_20260614_fast_policy.jsonl`
- Fast result: 3/3 positions completed through Rust exact rerank with `k=10`, `insurance=2`, `mode=general`
- Fast pool examples: 15/21, 12/12, 11/12
- Fast elapsed examples: 376 ms, 42 ms, 42 ms
- Safer command: `python -B -m ai.tutor.t3_runtime --config ai\config\t3_ev_precision_pool_20260614.json --pool-policy safer --limit 3 ...`
- Safer output: `D:\ofc-pineapple-data\t3_ev_precision_20260614\runtime_smoke_t3_pool_20260614_safer_gap3_policy.jsonl`
- Safer result: 3/3 positions completed through Rust exact rerank with `k=15`, `insurance=4`, `mode=general`, `gap=3`
- Safer pool examples: 21/21, 12/12, 12/12
- Safer elapsed examples: 364 ms, 44 ms, 41 ms

Current recommendation:

- For practical low-latency T3: use four-model union Top10 + 2 shape-insurance
  candidates + exact rerank.
- If rare tail reduction matters more than the smallest pool, use Top15 + 4
  general shape-insurance candidates + near-full gap 3 + exact rerank.
- For strict zero-loss evidence on the current holdouts, still use Top20 union.

## Interpretation

The direction is correct for exact-rerank safety, but the model is not strong
enough yet for instant model-only decisions or small TopK pruning.

- Hard-negative weighting reduces Top10 and Top20 loss tails.
- Top10 is still not safe for exact rerank.
- Top15 + 4 general insurance + near-full gap 3 is zero-loss on the two
  external 5k holdouts plus the fresh 2k holdout tested here.
- Union Top20 + exact rerank is EV-safe on two external 5k holdouts, but this
  works partly because T3 has only about 16 legal candidates on average.
- Top1 model-only is not close to 99.9%; current external Top1 recall is only
  about one third and still has very large EV-loss misses.
- Mean EV loss `<= 0.01` is now reached by model-only Top15 and by four-model
  union Top10 + exact rerank on the fresh external holdout.
- Four-model union Top10 + 2 shape-insurance candidates reduces mean EV loss
  to `0.0010` on seed20260704 and `0.0055` on the diverse holdout.
- Four-model union Top15 + 4 shape-insurance candidates + near-full gap 3
  reduces mean and max EV loss to `0.0000` on both external 5k holdouts and
  the fresh external 2k holdout.
- Max/tail EV loss is solved on these two holdouts by Top15+4+gap3 and by
  union Top20 + exact rerank; broader holdouts are still needed before treating
  smaller pools as universally safe.
- The 100k branch-expanded input is a large improvement over the initial 20k
  run, but still not enough to generalize.
- Future training data must use many root deals, not full expansion from a few
  roots.

## Next Work

The next useful step is to rebuild the model-only training setup, not to keep
training the same objective on the same data:

1. Keep model-union Top20 + exact rerank as the safe T3 fallback path.
2. Build a larger many-root/random-T0 exact T3 dataset while preserving root
   diversity.
3. Train a model-only EV predictor with an architecture/objective aimed at
   calibrated action EV, not only TopK recall.
4. Add explicit action features if the current 520-dim state-only encoding is
   insufficient for placement/discard sensitivity.
5. Mine high-loss Top1 misses from the training split and upweight them, but
   verify on untouched many-root external holdouts before adopting.
6. Track model-only Top1 EV loss, not just TopK recall.
7. Keep Top15+4+gap3 as the current precision target and continue mining
   Top10 tails before trying to shrink exact fallback further.

If model-only Top1 remains weak after more data, the next code change should be
an action-aware model input or a set/listwise architecture that compares all
legal candidates jointly before choosing one action.
