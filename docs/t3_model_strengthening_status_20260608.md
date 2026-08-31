# T3 Model Strengthening Status - 2026-06-08

## Why

The T2 capped exact pilot showed that the T2 model-ranked best action can lose a large amount of EV after exact refinement.  The first hypothesis to test was whether this is caused by weak T3 continuation scoring.

## Diagnostic

Script:

```powershell
python -B -m ai.tutor.diagnose_t3_model_from_t2_miss `
  --source D:\ofc-pineapple-data\t2_t0t1_top10_t3_model_20260608\root1_full_s2\t2_t3_model_rows.jsonl `
  --misses D:\ofc-pineapple-data\t2_t0t1_top10_t3_model_20260608\exact_cap50_limit1\t2_oracle_cap50_limit1.misses.jsonl `
  --output D:\ofc-pineapple-data\t2_t0t1_top10_t3_model_20260608\t3_diagnose_cap50_miss0_s5_teacher\t3_diagnosis.jsonl `
  --teacher-output D:\ofc-pineapple-data\t2_t0t1_top10_t3_model_20260608\t3_diagnose_cap50_miss0_s5_teacher\t3_exact_teacher.jsonl `
  --miss-index 0 `
  --draw-limit 5 `
  --max-t2-actions 2 `
  --device cpu `
  --rust-solver ai/rust_solver/target/release/t3_exact_solver.exe `
  --rust-timeout-s 10 `
  --exact-top-n 100
```

Output:

- `D:\ofc-pineapple-data\t2_t0t1_top10_t3_model_20260608\t3_diagnose_cap50_miss0_s5_teacher\t3_diagnosis.jsonl`
- `D:\ofc-pineapple-data\t2_t0t1_top10_t3_model_20260608\t3_diagnose_cap50_miss0_s5_teacher\t3_exact_teacher.jsonl`

Result against the existing BB T3 single model:

- rows: 10
- Top1: 40%
- Top3: 70%
- EV loss mean: 1.1804
- EV loss max: 3.3288
- EV loss > 0.1: 6/10
- EV loss > 1.0: 5/10

Worst case:

- dealt: `4d 6c 8s`
- model: `4d->bottom; 8s->middle; discard 6c`
- exact: `4d->bottom; 8s->bottom; discard 6c`
- EV loss: 3.3288

This confirms that at least part of the T2 weakness is caused by T3 model ranking error.

## Teacher Data

The diagnostic script now supports `--teacher-output`.  It writes exact T3 teacher JSONL compatible with `ai.training.convert_action_value_teacher`.

Conversion:

```powershell
python -B -m ai.training.convert_action_value_teacher `
  D:\ofc-pineapple-data\t2_t0t1_top10_t3_model_20260608\t3_diagnose_cap50_miss0_s5_teacher\t3_exact_teacher.jsonl `
  --output D:\ofc-pineapple-data\t2_t0t1_top10_t3_model_20260608\t3_diagnose_cap50_miss0_s5_teacher\reranker_t3_exact_hardcase `
  --turns 3 `
  --state-dim 520 `
  --regular-max-candidates 0
```

Converted data:

- records: 10
- samples: 210
- skipped: 0
- position: BB only
- score mean/std: +4.550 / 7.393
- bust mean: 10.9%
- FL mean: 5.5%

## Smoke Fine-Tune

This is not a production model.  It only verifies that the hard-case data can correct the observed miss.

Output model:

- `D:\ofc-pineapple-data\t2_t0t1_top10_t3_model_20260608\models\t3-bb-hardcase-miss0-smoke-20260608\action_value_best.pt`

Same 10 hard-case rows after smoke fine-tune:

- Top1: 100%
- Top3: 100%
- EV loss mean: 0
- EV loss max: 0

## Larger Local Hard-Case Smoke

To confirm the signal was not limited to one tiny sample, the cap20 pilot misses were expanded with:

- miss indices: all 2 misses
- T2 actions per miss: 4
- sampled T3 draws per T2 action: 10
- T3 exact rows: 80
- converted samples: 1,680

Output:

- `D:\ofc-pineapple-data\t2_t0t1_top10_t3_model_20260608\t3_diagnose_cap20_misses_all_s10_a4\t3_exact_teacher.jsonl`
- `D:\ofc-pineapple-data\t2_t0t1_top10_t3_model_20260608\t3_diagnose_cap20_misses_all_s10_a4\reranker_t3_exact_hardcase`

Existing BB T3 single model on these 80 hard-case rows:

- Top1: 30.0%
- Top3: 61.25%
- Top5: 83.75%
- Top10: 93.75%
- Top15: 97.5%
- EV loss mean: 1.8749
- EV loss p95: 6.1965
- EV loss max: 29.6353
- EV loss > 1.0: 30/80

Worst case:

- board top: `2d 2c`
- board middle: `3h 6d 7h`
- board bottom: `4s Kd Kh Kc`
- dealt: `3c 3d Qh`
- model: `3d->middle; Qh->top; discard 3c`
- exact: `3c->middle; 3d->middle; discard Qh`
- EV loss: 29.6353

Smoke model:

- `D:\ofc-pineapple-data\t2_t0t1_top10_t3_model_20260608\models\t3-bb-hardcase-cap20-s10-a4-smoke-20260608\action_value_best.pt`

Same 80 rows after smoke fine-tune:

- Top1: 83.75%
- Top3: 96.25%
- Top5: 100%
- Top10: 100%
- Top15: 100%
- EV loss mean: 0.0962
- EV loss p95: 0.9154
- EV loss max: 2.2920
- EV loss > 1.0: 3/80

This confirms that exact T3 hard cases can train the model in the desired direction.  It is still not a production result because these 80 rows are not a clean holdout.

## Next Step

Scale this from 10 T3 positions to a clean split:

- `mine_train`: many T2 miss continuations and random T3 positions
- `dev`: threshold and model selection
- `final_holdout`: never train on it

Use the same EV-loss metrics, not recall alone:

- Top1/Top3/Top5
- EV loss mean
- EV loss p95/p99
- EV loss max
- count over 0.1 / 0.5 / 1.0

## Balanced Multi-Root Local Run

The first source generator attempt with `--roots 5 --max-records-per-position 100` still filled all records from root 0.  To avoid the fixed-root bias, `ai.tutor.build_t2_t3_model_teacher_from_t0t1_topk` now supports:

```powershell
--max-records-per-root-per-position 20
```

Balanced source output:

- `D:\ofc-pineapple-data\t3_exact_from_t2_rows_20260608\source_multi_root5_balanced_rows200\t2_t3_model_rows.jsonl`
- roots: 5
- rows: 200
- per root: 20 BB + 20 BTN

T3 exact teacher output:

- root: `D:\ofc-pineapple-data\t3_exact_from_t2_rows_20260608\balanced_rows200_a2_d2`
- mine_train: 560 T3 records / 9,180 candidates
- dev: 120 T3 records / 1,926 candidates
- final_holdout: 120 T3 records / 2,064 candidates

Balanced specialist models:

- BB: `D:\ofc-pineapple-data\t3_exact_from_t2_rows_20260608\models\t3-bb-balanced200-a2-d2-20260608\action_value_best.pt`
- BTN: `D:\ofc-pineapple-data\t3_exact_from_t2_rows_20260608\models\t3-btn-balanced200-a2-d2-20260608\action_value_best.pt`

Final holdout, single-model Top1:

| Model | Top1 | Top3 | EV loss mean | EV loss p95 | EV loss max |
| --- | ---: | ---: | ---: | ---: | ---: |
| old T3 | 42.5% | 68.3% | 1.3645 | 8.6681 | 29.5689 |
| balanced200 | 48.3% | 75.0% | 1.3047 | 8.1758 | 29.5689 |

Final holdout, old + balanced200 union with exact rerank:

| Per-source TopK | Recall | EV loss mean | EV loss max | Avg pool | Max pool |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 83.3% | 0.3380 | 10.3905 | 3.77 | 6 |
| 5 | 92.5% | 0.1231 | 7.3513 | 6.05 | 8 |
| 10 | 97.5% | 0.0236 | 1.5287 | 11.02 | 15 |
| 20 | 100.0% | 0.0000 | 0.0000 | 16.88 | 21 |

Config:

- `ai/config/t3_balanced200_pool_20260608.json`

Conclusion: the balanced200 specialist improves Top1/Top3, but it is not safe as a replacement.  The practical safe direction is still a union pool plus exact rerank.  On this local balanced holdout, old + balanced200 Top20 recovered the exact best with zero EV loss.

## Balanced400 Multi-Root Local Run

The next local run doubled the balanced source size while keeping root balance:

```powershell
python -B -m ai.tutor.build_t2_t3_model_teacher_from_t0t1_topk `
  --output D:\ofc-pineapple-data\t3_exact_from_t2_rows_20260608\source_multi_root10_balanced_rows400\t2_t3_model_rows.jsonl `
  --roots 10 `
  --position both `
  --target-top-k 10 `
  --opponent-top-k 1 `
  --opponent-t2-top-k 1 `
  --draw-limit 1 `
  --max-records-per-position 200 `
  --max-records-per-root-per-position 20 `
  --device cpu
```

Source output:

- `D:\ofc-pineapple-data\t3_exact_from_t2_rows_20260608\source_multi_root10_balanced_rows400\t2_t3_model_rows.jsonl`
- roots: 10
- rows: 400
- per root: 20 BB + 20 BTN
- average T2 candidates: 23.95
- elapsed: 42.7s

T3 exact teacher output:

- root: `D:\ofc-pineapple-data\t3_exact_from_t2_rows_20260608\balanced_rows400_a2_d2`
- mine_train: 1,120 T3 records / 18,510 candidates
- dev: 240 T3 records / 4,044 candidates
- final_holdout: 240 T3 records / 3,966 candidates
- exact generation elapsed: 324.5s

Balanced400 specialist models:

- BB: `D:\ofc-pineapple-data\t3_exact_from_t2_rows_20260608\models\t3-bb-balanced400-a2-d2-20260608\action_value_best.pt`
- BTN: `D:\ofc-pineapple-data\t3_exact_from_t2_rows_20260608\models\t3-btn-balanced400-a2-d2-20260608\action_value_best.pt`

Training was local CPU only:

- BB: 181s, stopped by max-seconds at epoch 50
- BTN: 182s, stopped by max-seconds at epoch 53

Final holdout, single-model Top1:

| Model | Top1 | Top3 | Top5 | Top10 | EV loss mean | EV loss p95 | EV loss max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old T3 | 38.8% | 70.8% | 81.2% | 93.3% | 1.7495 | 10.0419 | 30.4204 |
| balanced400 | 43.8% | 74.6% | 86.2% | 95.4% | 1.5509 | 9.2833 | 30.4204 |

The balanced400 specialist improves the single-model ranking, but it still has high-loss Top1 failures.  It is not a standalone answer model.

Final holdout, old + balanced400 union with exact rerank:

| Per-source TopK | Recall | EV loss mean | EV loss max | Avg pool | Max pool |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 81.2% | 0.2729 | 11.1542 | 3.60 | 5 |
| 5 | 90.0% | 0.1500 | 11.1542 | 5.73 | 8 |
| 10 | 96.7% | 0.0282 | 3.2808 | 10.50 | 15 |
| 15 | 99.2% | 0.0111 | 2.4899 | 13.73 | 18 |
| 20 | 100.0% | 0.0000 | 0.0000 | 16.17 | 21 |

Final holdout, old + balanced200 + balanced400 union with exact rerank:

| Per-source TopK | Recall | EV loss mean | EV loss max | Avg pool | Max pool |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 90.8% | 0.1522 | 11.1542 | 4.06 | 6 |
| 5 | 94.6% | 0.0996 | 11.1542 | 6.33 | 10 |
| 10 | 99.2% | 0.0012 | 0.1789 | 11.08 | 16 |
| 15 | 99.6% | 0.0007 | 0.1789 | 14.09 | 20 |
| 20 | 100.0% | 0.0000 | 0.0000 | 16.25 | 21 |

Config:

- `ai/config/t3_balanced400_pool_20260608.json`

New evaluation command:

```powershell
python -B -m ai.tutor.evaluate_t3_union_pool `
  --teacher D:\ofc-pineapple-data\t3_exact_from_t2_rows_20260608\balanced_rows400_a2_d2\final_holdout\t3_exact_teacher.jsonl `
  --model old=OLD_BB.pt,OLD_BTN.pt `
  --model balanced200=BALANCED200_BB.pt,BALANCED200_BTN.pt `
  --model balanced400=BALANCED400_BB.pt,BALANCED400_BTN.pt `
  --topks 3,5,10,15,20
```

Conclusion: more balanced local data did improve the T3 model, especially Top1/Top3/Top5.  It did not make model-only Top1 safe.  The best current route for gameplay remains model union pruning followed by exact rerank.  For correctness-sensitive T3 serving, use per-source Top20.  Per-source Top10 is much faster and nearly safe on this final holdout, but it still has a small 0.1789 EV tail.

## Balanced400 Hard-Negative Fine-Tune

The next step mined misses from the balanced400 `mine_train` split only.  `dev` and `final_holdout` were not used for training.

Mining source:

- teacher: `D:\ofc-pineapple-data\t3_exact_from_t2_rows_20260608\balanced_rows400_a2_d2\mine_train\t3_exact_teacher.jsonl`
- converted data: `D:\ofc-pineapple-data\t3_exact_from_t2_rows_20260608\balanced_rows400_a2_d2\mine_train\reranker_t3_exact`
- pool rows: `D:\ofc-pineapple-data\t3_exact_from_t2_rows_20260608\hardneg_balanced400_three_source_20260608\mine_train\rows.jsonl`

Three-source mine_train pool before hard-negative training:

| Per-source TopK | Recall | EV loss mean | EV loss max | Avg pool | Max pool |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 91.4% | 0.0874 | 11.0684 | 4.04 | 7 |
| 5 | 96.3% | 0.0255 | 6.6055 | 6.30 | 11 |
| 8 | 98.3% | 0.0081 | 2.4594 | 9.38 | 15 |
| 10 | 99.2% | 0.0053 | 2.4594 | 11.06 | 17 |
| 15 | 99.8% | 0.0001 | 0.0941 | 14.06 | 21 |
| 20 | 100.0% | 0.0000 | 0.0000 | 16.26 | 21 |

Weighted hard-negative data:

```powershell
python -B -m ai.training.create_pool_miss_weighted_action_value_data `
  --source D:\ofc-pineapple-data\t3_exact_from_t2_rows_20260608\balanced_rows400_a2_d2\mine_train\reranker_t3_exact `
  --pool-rows D:\ofc-pineapple-data\t3_exact_from_t2_rows_20260608\hardneg_balanced400_three_source_20260608\mine_train\rows.jsonl `
  --output D:\ofc-pineapple-data\t3_exact_from_t2_rows_20260608\hardneg_balanced400_three_source_20260608\weighted_mine_train_top10sec5 `
  --miss-k 10 `
  --min-ev-loss 0.05 `
  --secondary-miss-k 5 `
  --secondary-min-ev-loss 0.25
```

Weighted stats:

- primary Top10 miss groups: 7
- secondary Top5 miss groups: 11
- weighted candidates: 210
- confusers boosted: 192

Hard-negative specialist models:

- BB: `D:\ofc-pineapple-data\t3_exact_from_t2_rows_20260608\models\t3-bb-balanced400-hardneg-top10sec5-20260608\action_value_best.pt`
- BTN: `D:\ofc-pineapple-data\t3_exact_from_t2_rows_20260608\models\t3-btn-balanced400-hardneg-top10sec5-20260608\action_value_best.pt`

Training was local GPU:

- device: NVIDIA GeForce RTX 2060 SUPER
- BB: 242s, stopped by max-seconds
- BTN: 241s, stopped by max-seconds

Final holdout, balanced400 vs hardneg single-model:

| Model | Top1 | Top3 | Top5 | Top10 | EV loss mean | EV loss p95 | EV loss max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced400 | 43.8% | 74.6% | 86.2% | 95.4% | 1.5509 | 9.2833 | 30.4204 |
| hardneg | 51.7% | 80.8% | 88.8% | 96.2% | 1.3085 | 8.0772 | 31.4311 |

Hardneg improves Top1/Top3/Top5 substantially.  It still is not a standalone answer model, and its own Top10 has a bad tail, so it should be used only as an additional candidate source in a union pool.

Final holdout, old + balanced200 + balanced400 + hardneg union with exact rerank:

| Per-source TopK | Recall | EV loss mean | EV loss max | Avg pool | Max pool |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 92.9% | 0.1377 | 11.1542 | 4.54 | 9 |
| 5 | 95.8% | 0.0527 | 7.2297 | 7.01 | 14 |
| 8 | 98.8% | 0.0018 | 0.1789 | 10.09 | 15 |
| 10 | 99.2% | 0.0012 | 0.1789 | 11.80 | 17 |
| 15 | 99.6% | 0.0007 | 0.1789 | 14.67 | 21 |
| 20 | 100.0% | 0.0000 | 0.0000 | 16.39 | 21 |

Config:

- `ai/config/t3_balanced400_hardneg_pool_20260608.json`

Conclusion: this round made the model stronger as a candidate source.  The practical improvement is that Top8 now reaches the same final-holdout max EV loss as the prior three-source Top10, with a smaller average pool.  It did not eliminate the last small final-holdout Top10/Top15 tail.  The next meaningful step is to generate more fresh, balanced `mine_train` data around the remaining tail patterns rather than training on final_holdout.

## External Root1000 Holdout

The balanced400 and hard-negative results above were internal local holdouts from the same generation run family.  To check whether the improvement generalizes, a fresh external-style local holdout was generated with a different root range and seed:

```powershell
python -B -m ai.tutor.build_t2_t3_model_teacher_from_t0t1_topk `
  --output D:\ofc-pineapple-data\t3_external_holdout_20260608\source_root1000_seed20260609_rows400\t2_t3_model_rows.jsonl `
  --roots 10 `
  --root-start 1000 `
  --seed 20260609 `
  --position both `
  --target-top-k 10 `
  --opponent-top-k 1 `
  --opponent-t2-top-k 1 `
  --draw-limit 1 `
  --max-records-per-position 200 `
  --max-records-per-root-per-position 20 `
  --device cuda
```

Source output:

- `D:\ofc-pineapple-data\t3_external_holdout_20260608\source_root1000_seed20260609_rows400\t2_t3_model_rows.jsonl`
- roots: 1000-1009
- rows: 400
- per root: 20 BB + 20 BTN
- average T2 candidates: 23.71
- elapsed: 58.3s

T3 exact teacher output:

- root: `D:\ofc-pineapple-data\t3_external_holdout_20260608\root1000_seed20260609_rows400_a2_d2`
- mine_train: 1,120 T3 records / 18,642 candidates
- dev: 240 T3 records / 3,966 candidates
- final_holdout: 240 T3 records / 3,822 candidates
- exact generation elapsed: 347.9s

External final_holdout, single-model Top1:

| Model | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | EV loss mean | EV loss p95 | EV loss max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old T3 | 38.8% | 62.9% | 77.9% | 93.3% | 98.3% | 99.6% | 1.8880 | 9.3673 | 105.1386 |
| balanced400 | 38.8% | 67.1% | 80.8% | 94.6% | 98.3% | 99.6% | 1.9188 | 9.2453 | 105.1386 |
| hardneg | 39.2% | 65.8% | 79.6% | 94.2% | 97.5% | 99.6% | 1.9462 | 9.1439 | 105.1386 |

On this external holdout, the new single models do not clearly beat the old model.  The internal hard-negative gain should therefore be treated as a useful candidate-source improvement, not as proof of a standalone stronger answer model.

External final_holdout, old + balanced200 + balanced400 + hardneg union with exact rerank:

| Per-source TopK | Recall | EV loss mean | EV loss p99 | EV loss max | EV loss > 0.1 | Avg pool | Max pool |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 76.7% | 0.6247 | 5.8626 | 96.8311 | 30 | 3.93 | 7 |
| 5 | 90.4% | 0.4747 | 2.1779 | 96.8311 | 13 | 6.21 | 11 |
| 8 | 97.1% | 0.0369 | 0.2250 | 6.0782 | 4 | 9.31 | 16 |
| 10 | 97.5% | 0.0367 | 0.2250 | 6.0782 | 4 | 10.78 | 19 |
| 15 | 99.6% | 0.0012 | 0.0000 | 0.2936 | 1 | 13.64 | 20 |
| 20 | 100.0% | 0.0000 | 0.0000 | 0.0000 | 0 | 15.71 | 21 |

The external result is stricter than the internal balanced400 holdout.  Top20 union + exact rerank still recovered the exact best with zero EV loss, but Top15 is not strictly safe and Top8/Top10 still have non-trivial rare misses.  The next data round should mine fresh external-style misses instead of training again on the previous balanced400 final_holdout: Top5 misses include the large-loss BB `5d 6s 7h` trips-FL spot, while Top8/Top10 misses include the BTN `3d Qc X2` joker/FL-risk spot.
