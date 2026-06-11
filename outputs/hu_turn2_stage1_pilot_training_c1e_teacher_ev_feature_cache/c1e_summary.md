# Gate C1e Teacher-EV Feature Cache Summary

Gate C1e is calibration-only. It does not authorize 50k teacher, T1, C2-small, or production training.

- execution status: `hard_pass`
- blockers: `none`
- warnings: `unbiased_rows_lt_20k`
- strong pass: `False`
- total rows: `20000`
- unbiased rows: `4000`
- enriched rows: `16000`
- replay-ready rows: `20000/20000`
- missing replay rows: `0`
- elapsed seconds: `79.80`

## Source / Position

| c1e_split | source | position | rows |
|---|---|---|---:|
| enriched | difficult_spots | first | 3934 |
| enriched | difficult_spots | second | 4066 |
| enriched | near_threshold_spots | first | 2002 |
| enriched | near_threshold_spots | second | 1998 |
| enriched | random_off_policy | first | 2000 |
| enriched | random_off_policy | second | 2000 |
| unbiased | policy_on_distribution | first | 2000 |
| unbiased | policy_on_distribution | second | 2000 |

## Candidate Fire Estimate

| candidate | split | current fires | projected C1e fires | current FP rate |
|---|---|---:|---:|---:|
| best_diagnostic_m2p5_r0_g0p7 | all | 225 | 403.1 | 0.711 |
| confidence_lcb164_m2p5_r0_g0p7 | all | 36 | 22.5 | 0.000 |
| confidence_lcb196_m2p5_r0_g0p7 | all | 33 | 20.6 | 0.000 |
| global_conservative_m3_r0_g0p9 | all | 20 | 12.5 | 0.500 |
| source_filtered_lcb196 | all | 14 | 8.8 | 0.000 |
| position_specific_second_strict | all | 11 | 6.9 | 0.000 |

## Decision

- production training: `No-Go`
- T1 training: `No-Go`
- 50k teacher: `No-Go`
- C2-small heldout: `No-Go`
- C1f expanded calibration: `Go`
