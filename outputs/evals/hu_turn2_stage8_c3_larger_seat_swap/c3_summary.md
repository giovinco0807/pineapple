# HU Turn2 Stage8 C3 Larger Seat-Swap Validation

C3 is validation-only. It does not authorize 50k teacher, T1, production training, or production runtime changes.

- T3 continuation: `Stage7_candidate_A_m5_r10`
- Stage8 mode: `HU T2 selective override`, not full replacement
- seed_stride: `1000000`
- games_per_seed: `1000`
- completed_shards: `20`

## Seat-Swap Results

| config | EV/hand | CI low | CI high | seeds | paired | overrides | override rate | teacher avg gain | FP | p95 loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m2.5_r0_g0.9 | -0.0058 | -0.0315 | 0.0199 | 5 | 5000 | 45 | 0.0045 | 4.3530 | 0.0496 | 0.0000 |
| m2.75_r0_g0.9 | -0.0058 | -0.0315 | 0.0199 | 5 | 5000 | 42 | 0.0042 | 4.6650 | 0.0495 | 0.0000 |
| m2.5_r0_g0.925 | -0.0058 | -0.0315 | 0.0199 | 5 | 5000 | 40 | 0.0040 | 4.3694 | 0.0500 | 0.0173 |
| m2.5_r0_g0.7 | -0.0069 | -0.0299 | 0.0161 | 5 | 5000 | 81 | 0.0081 | 4.1743 | 0.0556 | 0.2592 |

## Decision

- C3 larger seat-swap: `No-Go`
- production: `No-Go`
- 50k teacher: `No-Go`
- T1 training: `No-Go`
- next step: `revise runtime proxy gate; use high-MC only as diagnostic audit for fired/top-loss states`
