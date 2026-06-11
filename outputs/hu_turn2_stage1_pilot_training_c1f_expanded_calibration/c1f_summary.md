# HU Turn2 Stage8 Gate C1f Expanded Calibration

C1f is calibration-only. It does not authorize C2-small, 50k teacher, T1, production training, or production runtime changes.

- C1e rows loaded: `20000`
- C1e replay-ready rows: `20000`
- elapsed seconds: `16.87`
- C1f expanded calibration: `completed`
- C2-small seat-swap: `Go`
- C2-small blockers: `none`
- runtime caveat: `LCB filters in this analyzer use MC512 teacher gain LCB; they are calibration filters, not directly deployable runtime gates.`

## Candidate Metrics

| candidate | fires | unbiased | enriched | first | second | FP rate | avg gain | p95 loss | p99 loss | max loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| best_diagnostic_m2p5_r0_g0p7 | 978 | 233 | 745 | 412 | 566 | 0.018 | 5.2349 | 0.0000 | 0.9634 | 5.7725 |
| confidence_lcb196_m2p5_r0_g0p7 | 913 | 217 | 696 | 387 | 526 | 0.000 | 5.5729 | 0.0000 | 0.0000 | 0.0000 |
| confidence_lcb164_m2p5_r0_g0p7 | 932 | 227 | 705 | 395 | 537 | 0.000 | 5.4978 | 0.0000 | 0.0000 | 0.0000 |
| source_filtered_lcb196 | 765 | 217 | 548 | 387 | 378 | 0.000 | 5.9146 | 0.0000 | 0.0000 | 0.0000 |
| confidence_lcb196_m2p25_r0_g0p7 | 1084 | 286 | 798 | 482 | 602 | 0.000 | 5.1697 | 0.0000 | 0.0000 | 0.0000 |
| confidence_lcb164_m2p25_r0_g0p7 | 1118 | 303 | 815 | 502 | 616 | 0.000 | 5.0709 | 0.0000 | 0.0000 | 0.0000 |
| confidence_lcb196_m2p0_r0_g0p75 | 1260 | 320 | 940 | 575 | 685 | 0.000 | 4.8362 | 0.0000 | 0.0000 | 0.0000 |
| confidence_lcb164_m2p0_r0_g0p75 | 1300 | 341 | 959 | 596 | 704 | 0.000 | 4.7452 | 0.0000 | 0.0000 | 0.0000 |

## Recommended Seat-Swap Candidates

| candidate | status | blockers | fires | avg gain | FP rate |
|---|---|---|---:|---:|---:|
| confidence_lcb196_m2p5_r0_g0p7 | go |  | 913 | 5.5729 | 0.000 |
| confidence_lcb164_m2p5_r0_g0p7 | go |  | 932 | 5.4978 | 0.000 |
| source_filtered_lcb196 | go |  | 765 | 5.9146 | 0.000 |
| confidence_lcb164_m2p0_r0_g0p75 | go |  | 1300 | 4.7452 | 0.000 |

## Decisions

- production training: `No-Go`
- T1 training: `No-Go`
- 50k teacher: `No-Go`
- C2-small heldout: `Go`
