# HU Turn2 Stage8 C2-Small Validation

C2-small is validation-only. It does not authorize 50k teacher, T1, production training, or production runtime changes.

Teacher-oracle filters use MC512 teacher-EV LCB and are not runtime-deployable gates. Runtime proxy filters use only model/runtime fields.

`reference_margin_raw` is a T2 baseline/reference score margin on the T2 model scale. It is not comparable to the T3 Stage7 `hu_turn3_reference_min_margin=10.0` gate.

- C2-small: `Go`
- blockers: `none`
- seat_swap_run: `True`
- elapsed seconds: `496.30`

## Teacher Oracle Heldout

| candidate | fires | FP rate | avg gain | unbiased fires | enriched fires | first | second |
|---|---:|---:|---:|---:|---:|---:|---:|
| confidence_lcb196_m2p5_r0_g0p7 | 202 | 0.0000 | 5.4958 | 70 | 132 | see position csv | see position csv |
| confidence_lcb164_m2p5_r0_g0p7 | 208 | 0.0000 | 5.3892 | 72 | 136 | see position csv | see position csv |
| source_filtered_lcb196 | 164 | 0.0000 | 5.9199 | 70 | 94 | see position csv | see position csv |
| confidence_lcb164_m2p0_r0_g0p75 | 312 | 0.0000 | 4.5040 | 109 | 203 | see position csv | see position csv |
| best_diagnostic_m2p5_r0_g0p7 | 236 | 0.0551 | 4.7011 | 75 | 161 | see position csv | see position csv |

## Runtime Proxy Candidates

| proxy | fires | FP rate | avg gain | unbiased | first | second | overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_m2.5_g0.9 | 223 | 0.0448 | 4.8652 | 75 | 92 | 131 | 0.8987 |
| A_m2.5_g0.85 | 228 | 0.0482 | 4.8072 | 75 | 94 | 134 | 0.8957 |
| A_m2.75_g0.9 | 187 | 0.0428 | 5.3080 | 66 | 79 | 108 | 0.7713 |
| A_m2.75_g0.85 | 192 | 0.0469 | 5.2277 | 66 | 81 | 111 | 0.7699 |

## Seat-Swap Small

| config | EV/hand | CI low | CI high | override rate |
|---|---:|---:|---:|---:|
| m2.5_r0_g0.9 | 0.0173 | -0.0035 | 0.0382 | 0.0033 |
| m2.5_r0_g0.85 | 0.0173 | -0.0035 | 0.0382 | 0.0039 |
| m2.75_r0_g0.9 | 0.0173 | -0.0035 | 0.0382 | 0.0033 |
| m2.75_r0_g0.85 | 0.0173 | -0.0035 | 0.0382 | 0.0039 |

## Decisions

- C3 / larger seat-swap: `Go`
- 50k teacher: `No-Go`
- T1 training: `No-Go`
- production training: `No-Go`
