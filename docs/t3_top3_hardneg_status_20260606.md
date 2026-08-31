# T3 Top3 Hard-Negative Status 2026-06-06

## Goal

Push T3 action-value reranker quality toward reliable Top3 pruning. The practical service target is still
`shortlist -> exact rerank`, so Top3 is monitored as a model-strength metric rather than a hard runtime guarantee.

## Data

- Base exact teacher: `D:\ofc-pineapple-data\branch_t0t2_top10_100k_20260605\merged\branch_t0t2_top10_t3_exact_unique100k_teacher.jsonl`
- Reranker data: `D:\ofc-pineapple-data\branch_t0t2_top10_100k_20260605\merged\reranker_unique100k`
- Hard-negative split:
  - train mining: `worker_10`
  - external holdout: `worker_11`
- Augmented train data:
  - `D:\ofc-pineapple-data\branch_t0t2_top10_100k_20260605\hardneg_worker10_train_worker11_holdout\augmented_top3_regret005_repeat8\reranker`
  - base records: 100,000
  - added worker_10 Top3 misses with regret > 0.05, repeated 8x
  - augmented records: 117,432
  - candidate samples: 1,885,374

## Trained Models

- BB hard-negative model:
  - `D:\ofc-pineapple-data\branch_t0t2_top10_100k_20260605\models\t3-top10-100k-hardneg-w10-bb-20260606\action_value_best.pt`
  - internal final: Top1 53.0%, Top10 99.3%, Top15 99.9%, Top20 100.0%
- BTN hard-negative model:
  - `D:\ofc-pineapple-data\branch_t0t2_top10_100k_20260605\models\t3-top10-100k-hardneg-w10-btn-20260606\action_value_best.pt`
  - internal final: Top1 56.1%, Top3 89.3%, Top5 96.4%, Top10 99.7%, Top15 100.0%, Top20 100.0%

## Worker 11 Holdout

BB, 5,003 groups:

| model | Top1 | Top3 | Top5 | Top10 | Top15 |
| --- | ---: | ---: | ---: | ---: | ---: |
| old hardneg | 36.2% | 71.7% | 86.2% | 96.0% | 99.1% |
| new100k | 39.0% | 70.0% | 83.6% | 95.5% | 98.8% |
| hardneg_w10 | 38.2% | 71.5% | 85.5% | 95.9% | 98.9% |
| old + hardneg 0.5/0.5 | 38.6% | 73.2% | 86.7% | 96.3% | 99.0% |
| best weight grid | 39.4% | 73.5% | 86.6% | 96.1% | - |

BTN, 4,997 groups:

| model | Top1 | Top3 | Top5 | Top10 | Top15 |
| --- | ---: | ---: | ---: | ---: | ---: |
| old hardneg | 35.0% | 68.5% | 83.1% | 97.0% | 99.5% |
| new100k | 35.8% | 66.4% | 82.7% | 96.9% | 99.4% |
| hardneg_w10 | 38.7% | 69.3% | 85.1% | 97.2% | 99.5% |
| old + hardneg 0.5/0.5 | 38.5% | 70.5% | 85.1% | 97.6% | 99.7% |
| best weight grid | 39.9% | 71.1% | 85.7% | 97.5% | - |

## Multi-Model Union Candidate Pool

Using old, new100k, and hardneg_w10 as three independent rankers, then taking the union of each model's TopK:

| position | union K | recall | misses | avg pool | max pool |
| --- | ---: | ---: | ---: | ---: | ---: |
| BB | 3 | 80.21% | 990 | 3.9 | 8 |
| BB | 5 | 91.37% | 432 | 6.1 | 11 |
| BB | 10 | 97.46% | 127 | 10.5 | 16 |
| BB | 15 | 99.26% | 37 | 13.4 | 20 |
| BB | 20 | 99.98% | 1 | 15.6 | 21 |
| BTN | 3 | 79.43% | 1028 | 4.2 | 8 |
| BTN | 5 | 90.79% | 460 | 6.5 | 12 |
| BTN | 10 | 98.64% | 68 | 11.0 | 17 |
| BTN | 15 | 99.82% | 9 | 13.8 | 20 |
| BTN | 20 | 99.98% | 1 | 15.7 | 21 |

## Interpretation

Top3 did improve, but not nearly enough to treat model-only Top3 as safe. Existing models have similar blind spots; ensembling improves Top3 only modestly. The strong operational path is to keep a Top15/Top20 candidate pool and exact-rerank it. That is close to perfect coverage on this holdout and still small enough for T3 exact refinement.

## Top5-Miss Follow-Up Data

To move the target from Top3 misses to a more practical Top5 boundary, a second augmented dataset was built from worker_10 misses where the teacher-best action ranked worse than 5 and regret was greater than 0.05.

- Output teacher:
  - `D:\ofc-pineapple-data\branch_t0t2_top10_100k_20260605\hardneg_worker10_train_worker11_holdout\augmented_top5_regret005_repeat12\branch_t0t2_top10_t3_exact_unique100k_plus_worker10_top5miss_repeat12_teacher.jsonl`
- Reranker data:
  - `D:\ofc-pineapple-data\branch_t0t2_top10_100k_20260605\hardneg_worker10_train_worker11_holdout\augmented_top5_regret005_repeat12\reranker`
- Selected unique Top5 misses:
  - BB: 511, repeated 12x for 6,132 added records
  - BTN: 800, repeated 12x for 9,600 added records
- Total records: 115,732
- Total candidate samples: 1,866,846

This dataset should be trained with `--target-topk 5` and selected by the Top5 metric. It is intended to improve Top5/Top8 pruning quality, not to chase model-only Top3.

## Top5-Miss Training Result

Top5-miss models were fine-tuned from the Top3 hard-negative checkpoints with `--target-topk 5`.

- BB:
  - `D:\ofc-pineapple-data\branch_t0t2_top10_100k_20260605\models\t3-top10-100k-top5miss-w10-bb-20260606\action_value_best.pt`
  - internal final: Top1 56.3%, Top3 90.4%, Top5 97.2%, Top10 99.6%
- BTN:
  - `D:\ofc-pineapple-data\branch_t0t2_top10_100k_20260605\models\t3-top10-100k-top5miss-w10-btn-20260606\action_value_best.pt`
  - internal final: Top1 60.6%, Top3 91.5%, Top5 97.6%, Top10 99.8%

Worker_11 holdout shows the Top5 fine-tune is not a standalone replacement:

| position | model | Top3 | Top5 | Top8 | Top10 |
| --- | --- | ---: | ---: | ---: | ---: |
| BB | old | 71.7% | 86.2% | 93.3% | 96.0% |
| BB | Top3 hardneg | 71.5% | 85.5% | 93.6% | 95.9% |
| BB | Top5 fine-tune | 71.8% | 84.8% | 93.3% | 95.5% |
| BB | old + Top5 0.5/0.5 | 73.9% | 86.9% | 94.0% | 96.4% |
| BTN | old | 68.5% | 83.1% | 93.8% | 97.0% |
| BTN | Top3 hardneg | 69.3% | 85.1% | 94.9% | 97.2% |
| BTN | Top5 fine-tune | 67.2% | 83.8% | 94.2% | 96.9% |
| BTN | old + Top3 + Top5 | 70.5% | 85.6% | 95.1% | 97.5% |

Taking the union of each model's TopK candidates, now using old, new100k, Top3 hardneg, and Top5 fine-tune:

| position | union K | recall | misses | avg pool | max pool |
| --- | ---: | ---: | ---: | ---: | ---: |
| BB | 3 | 80.87% | 957 | 4.0 | 8 |
| BB | 5 | 91.57% | 422 | 6.2 | 12 |
| BB | 8 | 95.88% | 206 | 9.0 | 15 |
| BB | 10 | 97.50% | 125 | 10.6 | 16 |
| BTN | 3 | 80.15% | 992 | 4.3 | 8 |
| BTN | 5 | 91.01% | 449 | 6.6 | 12 |
| BTN | 8 | 97.18% | 141 | 9.5 | 15 |
| BTN | 10 | 98.68% | 66 | 11.1 | 18 |

The Top5 fine-tune adds a little diversity to union candidate pools, but it does not solve Top5 pruning alone. Do not replace the serving model with the Top5 checkpoint by itself. If used, use it as an auxiliary candidate source in a union pool.

## Set/Listwise Reranker Follow-Up

A candidate-set model was trained so each decision group is scored jointly rather than one candidate at a time. The goal was not to replace exact reranking, but to see whether a different architecture reduces Top5 candidate-pool misses.

- Model code:
  - `ai/models/action_value_set_reranker.py`
  - `ai/training/train_action_value_set_reranker.py`
- External evaluation code added:
  - `ai/training/evaluate_action_value_set_reranker.py`
  - `ai/training/evaluate_action_value_candidate_pool.py` now accepts `--set-checkpoints`.
- Training data:
  - `D:\ofc-pineapple-data\branch_t0t2_top10_100k_20260605\hardneg_worker10_train_worker11_holdout\augmented_top5_regret005_repeat12\reranker`
- BB checkpoint:
  - `D:\ofc-pineapple-data\branch_t0t2_top10_100k_20260605\models\t3-set-top5-w10-bb-20260606\action_value_set_best.pt`
  - 20 epochs on RTX 2060 SUPER, 417 seconds, best epoch 18
  - internal final: Top1 45.0%, Top3 85.1%, Top5 96.9%, Top10 99.9%
- BTN checkpoint:
  - `D:\ofc-pineapple-data\branch_t0t2_top10_100k_20260605\models\t3-set-top5-w10-btn-20260606\action_value_set_best.pt`
  - 20 epochs on RTX 2060 SUPER, 402 seconds, best epoch 11
  - internal final: Top1 40.3%, Top3 77.9%, Top5 93.8%, Top10 99.8%

Worker_11 shows the set model is not a standalone replacement:

| position | model | Top1 | Top3 | Top5 | Top8 | Top10 | Top15 | Top20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BB | set/listwise only | 20.6% | 56.2% | 75.7% | 91.1% | 94.8% | 99.1% | 100.0% |
| BTN | set/listwise only | 24.6% | 53.0% | 71.8% | 88.2% | 93.9% | 99.0% | 100.0% |

However, it is useful as a fifth independent candidate source. Re-running the same worker_11 candidate-pool evaluation with old, new100k, Top3 hardneg, Top5 fine-tune, plus the set/listwise model gives:

| position | pool | Top3 recall | Top5 recall | Top8 recall | Top10 recall | Top15 recall | avg pool at K=5 | max pool at K=5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BB | union4 baseline | 80.6% | 91.5% | 95.7% | 97.5% | 99.3% | 6.2 | 12 |
| BB | union4 + set | 85.0% | 94.4% | 98.0% | 98.8% | 99.7% | 7.2 | 13 |
| BTN | union4 baseline | 80.2% | 91.2% | 97.5% | 98.7% | 99.8% | 6.5 | 12 |
| BTN | union4 + set | 82.9% | 93.2% | 98.2% | 99.3% | 99.9% | 7.7 | 14 |

Conclusion: the set/listwise model is valuable as a diversity source for shortlist construction. It should not become the primary scorer yet. The practical serving path is now:

1. Build a candidate pool from the existing action-value rankers plus the set/listwise ranker.
2. Use per-model Top5 for a roughly 7-8 candidate pool when speed matters.
3. Use per-model Top8/Top10 when a near-perfect T3 pool is needed before exact reranking.
4. Continue mining misses from `union4 + set` rather than from any single model.

## Pool-Miss Specialist Follow-Up

The next pass mined worker_10 misses from the `union4 + set` candidate pool, then built weighted data that keeps the original arrays as hardlinks and only rewrites candidate/group weights.

- Worker_10 pool rows:
  - BB: `D:\ofc-pineapple-data\branch_t0t2_top10_100k_20260605\hardneg_worker10_train_worker11_holdout\worker10_eval\pool_bb_union4_plus_set\rows.jsonl`
  - BTN: `D:\ofc-pineapple-data\branch_t0t2_top10_100k_20260605\hardneg_worker10_train_worker11_holdout\worker10_eval\pool_btn_union4_plus_set\rows.jsonl`
- Weighted data:
  - BB: `D:\ofc-pineapple-data\branch_t0t2_top10_100k_20260605\hardneg_worker10_train_worker11_holdout\poolmiss_union5_top5sec3_weighted_bb`
  - BTN: `D:\ofc-pineapple-data\branch_t0t2_top10_100k_20260605\hardneg_worker10_train_worker11_holdout\poolmiss_union5_top5sec3_weighted_btn`
- Weighted miss counts:
  - BB: Top5 misses 235, secondary Top3 misses 278
  - BTN: Top5 misses 12, secondary Top3 misses 90
- Specialist checkpoints:
  - BB: `D:\ofc-pineapple-data\branch_t0t2_top10_100k_20260605\models\t3-set-poolmiss-union5-bb-20260606\action_value_set_best.pt`
  - BTN: `D:\ofc-pineapple-data\branch_t0t2_top10_100k_20260605\models\t3-set-poolmiss-union5-btn-20260606\action_value_set_best.pt`

The specialists are weak as standalone models on worker_11, so they should not replace the broad set/listwise checkpoint:

| position | model | Top1 | Top3 | Top5 | Top8 | Top10 | Top15 | Top20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BB | pool-miss specialist only | 13.7% | 39.4% | 58.7% | 78.5% | 87.5% | 97.2% | 99.7% |
| BTN | pool-miss specialist only | 13.0% | 36.4% | 53.2% | 73.2% | 81.7% | 91.6% | 99.6% |

As a candidate source, however, the specialists add useful diversity:

| position | pool | Top3 recall | Top5 recall | Top8 recall | Top10 recall | Top12 recall | Top15 recall | avg pool at K=5 | max pool at K=5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BB | union4 + set | 85.0% | 94.4% | 98.0% | 98.8% | 99.2% | 99.7% | 7.2 | 13 |
| BB | union4 + set + poolmiss | 87.4% | 95.6% | 98.4% | 99.1% | 99.4% | 99.8% | 8.4 | 14 |
| BTN | union4 + set | 82.9% | 93.2% | 98.2% | 99.3% | 99.8% | 99.9% | 7.7 | 14 |
| BTN | union4 + set + poolmiss | 84.5% | 94.4% | 98.7% | 99.7% | 100.0% | 100.0% | 9.1 | 17 |

Conclusion: the pool-miss specialist is worthwhile only as an auxiliary candidate source. It moves Top5 recall by about +1.2 points for both positions, but it also increases the candidate pool by roughly 1-1.4 candidates at K=5. For runtime:

1. Use `union4 + set` Top5 when the pool must stay very small.
2. Use `union4 + set + poolmiss` Top5 when a 8-9 candidate pool is acceptable.
3. Use Top8 or Top10 before exact reranking when correctness matters more than pool size.

To push Top3 toward 100%, the next work should not be another broad repeat of the same training. It should be:

1. Mine high-regret Top3 misses from a fresh holdout distribution.
2. Generate more exact labels specifically around those miss patterns.
3. Train a stronger listwise/set reranker or add candidate-relative features.
4. Keep `Top15/Top20 + exact rerank` as the product path while Top3 remains below 95%.

## EV-Loss Objective

The product objective is now stricter than recall: minimize the EV loss caused by pruning. The evaluator now reports EV-loss mean, p95, p99, max, and threshold counts for each candidate-pool K. The pool-miss data builder can also filter misses by minimum EV loss, so future hard-negative data focuses on costly misses instead of harmless near-ties.

The current safest T3 serving preset is saved at:

- `ai/config/t3_ev_safe_pool_20260606.json`

It uses six candidate sources per position:

- four action-value checkpoints: old 20k+, new100k, Top3 hard-negative, Top5-miss fine-tune
- two set/listwise checkpoints: broad set model and pool-miss specialist

The serving rule for T3 should be:

1. Build the union pool from each source's Top20.
2. Exact-rerank that pool.
3. Fall back to full T3 exact if the pool cannot be built.

This is intentionally more conservative than Top10/Top15. Top15 looked almost perfect by recall, but still left rare high-EV misses. On worker_11 external holdout, Top20 removed those misses:

| position | union K | recall | mean EV loss | p99 EV loss | max EV loss | avg pool | max pool |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BB | 10 | 99.12% | 0.003954 | 0.000000 | 5.645869 | 12.6 | 20 |
| BB | 12 | 99.42% | 0.001508 | 0.000000 | 2.706825 | 13.7 | 21 |
| BB | 15 | 99.80% | 0.000564 | 0.000000 | 1.411799 | 14.9 | 21 |
| BB | 20 | 100.00% | 0.000000 | 0.000000 | 0.000000 | 15.9 | 21 |
| BTN | 10 | 99.66% | 0.003490 | 0.000000 | 11.521933 | 13.2 | 21 |
| BTN | 12 | 99.96% | 0.002721 | 0.000000 | 11.521933 | 14.2 | 21 |
| BTN | 15 | 99.98% | 0.000415 | 0.000000 | 2.072696 | 15.3 | 21 |
| BTN | 20 | 100.00% | 0.000000 | 0.000000 | 0.000000 | 15.9 | 21 |

Worker_10 replay after adding the pool-miss specialist has essentially no remaining training signal: BB Top3 has only a 0.0189 max EV loss and Top5+ is zero; BTN Top3+ is zero. Further improvement should therefore come from fresh exact-labeled states, not more repeats of the same worker_10 misses.

Operationally, this changes the target:

1. Use Top20 + exact rerank when EV safety matters.
2. Use Top10/Top12 only as faster modes with known rare tail risk.
3. Mine new hard negatives by EV loss, not by rank miss alone.
4. Judge future models by candidate-pool EV loss first, recall second.

## 2026-06-07 EV-Loss Specialist Retrain

The next specialist pass was trained from high-cost worker_11 pool misses rather than raw rank misses.

- New config:
  - `ai/config/t3_ev_loss_pool_20260607.json`
- Weighted data:
  - BB: `D:\ofc-pineapple-data\branch_t0t2_top10_100k_20260605\hardneg_worker10_train_worker11_holdout\poolmiss_evloss_w11_top10sec5_bb`
  - BTN: `D:\ofc-pineapple-data\branch_t0t2_top10_100k_20260605\hardneg_worker10_train_worker11_holdout\poolmiss_evloss_w11_top10sec5_btn`
- Mining rule:
  - primary: Top10 miss with EV loss >= 0.1
  - secondary: Top5 miss with EV loss >= 0.25
  - confusers: existing union Top10 candidates
- Mined groups:
  - BB: 15 primary, 71 secondary
  - BTN: 7 primary, 80 secondary
- New specialists:
  - BB: `D:\ofc-pineapple-data\branch_t0t2_top10_100k_20260605\models\t3-set-evloss-w11-top10-bb-20260607\action_value_set_best.pt`
  - BTN: `D:\ofc-pineapple-data\branch_t0t2_top10_100k_20260605\models\t3-set-evloss-w11-top10-btn-20260607\action_value_set_best.pt`

Internal training results:

| position | epochs | val Top3 | val Top5 | val Top10 | val Top15 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BB | 24 | 77.0% | 89.8% | 100.0% | 100.0% |
| BTN | 22 | 74.7% | 91.8% | 99.4% | 100.0% |

Adding these specialists as a seventh candidate source improves the replayed worker_11 pool:

| position | union K | recall | mean EV loss | p99 EV loss | max EV loss | avg pool | max pool |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BB | 3 | 93.9% | 0.014 | 0.071 | 11.862 | 6.6 | 13 |
| BB | 5 | 99.1% | 0.000 | 0.000 | 0.232 | 9.1 | 18 |
| BB | 8 | 100.0% | 0.000 | 0.000 | 0.000 | 11.8 | 21 |
| BB | 10 | 100.0% | 0.000 | 0.000 | 0.000 | 13.1 | 21 |
| BTN | 3 | 94.9% | 0.007 | 0.101 | 4.785 | 7.3 | 15 |
| BTN | 5 | 99.3% | 0.000 | 0.000 | 1.097 | 10.1 | 18 |
| BTN | 8 | 100.0% | 0.000 | 0.000 | 0.056 | 12.7 | 21 |
| BTN | 10 | 100.0% | 0.000 | 0.000 | 0.000 | 13.8 | 21 |

Worker_10 sanity check did not reveal a regression:

| position | union K | recall | max EV loss | avg pool | max pool |
| --- | ---: | ---: | ---: | ---: | ---: |
| BB | 3 | 99.9% | 0.019 | 7.2 | 14 |
| BB | 5 | 100.0% | 0.000 | 9.8 | 18 |
| BB | 10 | 100.0% | 0.000 | 14.1 | 21 |
| BTN | 3 | 100.0% | 0.000 | 7.6 | 14 |
| BTN | 5 | 100.0% | 0.000 | 10.5 | 18 |
| BTN | 10 | 100.0% | 0.000 | 14.0 | 21 |

Important caveat: worker_11 is now replay/training evidence for this specialist, not a clean holdout. The next proof needs fresh exact-labeled T3 states, then the same union7 evaluation. Until that fresh holdout passes, use union7 Top10 as a promising fast candidate pool, not as a guarantee.

## Code Note

`ai/training/train_action_value_reranker.py` now prints and writes Top3/Top5 metrics in training logs and summaries so Top3 is visible during future runs.

## 2026-06-07 Fresh T3 EV-Loss Specialist

Worker_11 is no longer treated as a clean holdout after the previous EV-loss specialist. A fresh exact-labeled T3 split was created on D drive from branch-expanded T0 Top50 / T1-T2 Top10 data:

- Fresh root:
  - `D:\ofc-pineapple-data\t3_evloss_fresh_20260607`
- New config:
  - `ai/config/t3_ev_loss_fresh_pool_20260607.json`
- Sources:
  - BB: `D:\ofc-pineapple-data\branch_expanded_t0top50_t1t2top10_20260605\teacher\bb_root1_t3_teacher.jsonl`
  - BTN: `D:\ofc-pineapple-data\branch_expanded_t0top50_t1t2top10_20260605\teacher\btn_root1_t3_teacher.jsonl`

Fresh split:

| position | split | records | avg candidates | mean best EV |
| --- | --- | ---: | ---: | ---: |
| BB | mine_train | 2,995 | 16.12 | 5.115 |
| BB | dev | 998 | 16.15 | 5.153 |
| BB | final_holdout | 999 | 16.26 | 4.588 |
| BTN | mine_train | 3,000 | 17.46 | 0.707 |
| BTN | dev | 1,000 | 17.51 | 0.690 |
| BTN | final_holdout | 1,000 | 17.46 | 0.687 |

Baseline union7 on this fresh final holdout still had costly TopK misses:

| position | union K | max EV loss | avg pool |
| --- | ---: | ---: | ---: |
| BB | 5 | 15.382 | 9.23 |
| BB | 8 | 4.441 | 12.11 |
| BB | 10 | 3.324 | 13.57 |
| BB | 20 | 0.000 | 16.26 |
| BTN | 5 | 1.238 | 10.78 |
| BTN | 8 | 0.693 | 13.52 |
| BTN | 10 | 0.693 | 14.86 |
| BTN | 20 | 0.000 | 17.46 |

Mining was done only from `mine_train`. The Top10 EV-loss miss set was too sparse, especially for BTN, so the planned fallback was used:

- primary: Top8 miss with EV loss >= 0.05
- secondary: Top5 miss with EV loss >= 0.25
- confusers: existing union7 pool candidates

Weighted miss data:

| position | primary groups | secondary groups | weighted candidates | confusers boosted |
| --- | ---: | ---: | ---: | ---: |
| BB | 62 | 52 | 1,857 | 1,743 |
| BTN | 15 | 9 | 423 | 399 |

New fresh specialists:

- BB: `D:\ofc-pineapple-data\t3_evloss_fresh_20260607\models\t3-set-evloss-fresh-top8-bb-20260607\action_value_set_best.pt`
- BTN: `D:\ofc-pineapple-data\t3_evloss_fresh_20260607\models\t3-set-evloss-fresh-top8-btn-20260607\action_value_set_best.pt`

Internal training results:

| position | best epoch | val Top3 | val Top5 | val Top10 |
| --- | ---: | ---: | ---: | ---: |
| BB | 22 | 72.2% | 90.3% | 99.7% |
| BTN | 18 | 75.0% | 90.3% | 99.3% |

Adding the fresh specialists as the eighth candidate source gives this fresh final-holdout result:

| position | union K | recall | mean EV loss | p99 EV loss | max EV loss | avg pool | max pool |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BB | 5 | 99.3% | 0.001446 | 0.000000 | 0.539555 | 10.1 | 17 |
| BB | 8 | 99.9% | 0.000062 | 0.000000 | 0.061846 | 12.8 | 20 |
| BB | 10 | 100.0% | 0.000000 | 0.000000 | 0.000000 | 14.1 | 21 |
| BB | 20 | 100.0% | 0.000000 | 0.000000 | 0.000000 | 16.3 | 21 |
| BTN | 5 | 99.0% | 0.000500 | 0.000000 | 0.446634 | 11.0 | 17 |
| BTN | 8 | 99.8% | 0.000000 | 0.000000 | 0.000000 | 13.7 | 20 |
| BTN | 10 | 100.0% | 0.000000 | 0.000000 | 0.000000 | 15.0 | 21 |
| BTN | 20 | 100.0% | 0.000000 | 0.000000 | 0.000000 | 17.5 | 21 |

Worker replay also stayed clean at Top10:

| replay | position | Top5 max EV loss | Top8 max EV loss | Top10 max EV loss | Top10 avg pool |
| --- | --- | ---: | ---: | ---: | ---: |
| worker10 | BB | 0.000000 | 0.000000 | 0.000000 | 14.6 |
| worker10 | BTN | 0.000000 | 0.000000 | 0.000000 | 14.4 |
| worker11 | BB | 0.232036 | 0.000000 | 0.000000 | 13.7 |
| worker11 | BTN | 1.096718 | 0.056377 | 0.000000 | 14.5 |

Conclusion: the current practical T3 path is now `union8 Top10 -> exact rerank`. On the fresh final holdout this reduced Top10 max EV loss to zero while keeping average pool size around 14-15. Top5 remains too risky for a correctness-first mode; it is useful only as a speed mode with known rare EV-loss tails.

## Runtime Wiring

The fresh union8 pool is now available through a runtime helper:

- Python API:
  - `ai.tutor.t3_runtime.T3UnionCandidatePool`
  - `ai.tutor.t3_runtime.evaluate_t3_position`
- CLI:
  - `python -m ai.tutor.t3_runtime --input INPUT.jsonl --output OUTPUT.jsonl --pool-k 10 --rust-timeout-s 5`
- Default config:
  - `ai/config/t3_ev_loss_fresh_pool_20260607.json`

The Rust exact solver was extended to accept optional `candidate_actions` in each input record. When `candidate_actions` is present, it evaluates only that subset and reports both `legal_actions` and `evaluated_actions`. If omitted, behavior remains unchanged and all legal actions are evaluated.

Runtime smoke outputs:

- BB: `D:\ofc-pineapple-data\t3_evloss_fresh_20260607\runtime_smoke\bb_limit5_union8_top10.jsonl`
- BTN: `D:\ofc-pineapple-data\t3_evloss_fresh_20260607\runtime_smoke\btn_limit5_union8_top10.jsonl`

Smoke result, fresh final holdout, five positions each:

| position | n | mean elapsed ms | max elapsed ms | mean pool | max pool | all under 5s |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| BB | 5 | 377.38 | 952.40 | 15.2 | 20 | yes |
| BTN | 5 | 359.96 | 896.51 | 14.2 | 20 | yes |

These smoke numbers include the first in-process model load for each position. A long-running service should keep `T3UnionCandidatePool` alive and reuse the loaded models.

## T2 Hybrid Hook

`ai.tutor.hybrid_t1t2` can now use the T3 union runtime inside T2 `exact_partial` refinement:

- CLI switch: `--t2-refinement exact_partial --t2-exact-backend t3_union`
- T3 pool controls: `--t3-pool-config ai/config/t3_ev_loss_fresh_pool_20260607.json --t3-pool-k 10`
- Runtime config: `ai/config/t1t2_t3_union_runtime_20260607.json`

The old T2 partial exact path remains available as `--t2-exact-backend full_exact`, so existing batch checks and historical comparisons are not changed.

T2 smoke, one local position:

| input | elapsed ms | legal actions | T2 pool | sync candidates | T3 samples | mean T3 pool | max T3 pool | backend |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ai/data/hybrid_t1t2_active_20260531/local_smoke_line8_only.jsonl` | 1366.28 | 24 | 20 | 3 | 3 | 12.67 | 17 | `t3_union` |

Best smoke action:

- `As -> middle`
- `Td -> bottom`
- discard `4c`
- refined score `15.728913`
