# HU T2 Stage8b C3 Larger Seat-Swap Validation

C3 is validation-only. It does not authorize 50k teacher generation, T1 training, production training, P2 fixed status, or production runtime changes.

## Inputs

- T3 continuation: `Stage7_candidate_A m5_r10` fixed
- T3 runtime: `hu_turn3_min_margin=5.0`, `hu_turn3_reference_min_margin=10.0`
- T2 model: `models/hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt`
- Main candidate: `m2.75_r0_g0.95_seatfirst_k3`
- Seed stride: `1000000`
- Stage8b mode: HU T2 selective override, not full replacement

## Seat-Swap Results

| config | role | paired seeds | EV/hand | CI low | CI high | overrides | override rate |
|---|---|---:|---:|---:|---:|---:|---:|
| `m2.75_r0_g0.95_seatfirst_k3` | main C3 | 7,500 | 0.0007 | -0.0435 | 0.0448 | 138 | 0.0092 |
| `m2.75_r0_g0.9_seatfirst_k3` | looser gate comparison | 3,000 | 0.0052 | -0.0866 | 0.0969 | 69 | 0.0115 |
| `m2.5_r0_g0.95_seatfirst_k3` | looser margin comparison | 3,000 | -0.0015 | -0.0923 | 0.0892 | 70 | 0.0117 |

## Seed Split

Main candidate `m2.75_r0_g0.95_seatfirst_k3`:

| seed | paired seeds | EV/hand | overrides |
|---:|---:|---:|---:|
| 2026062001 | 1,500 | -0.0501 | 21 |
| 2026062002 | 1,500 | 0.0043 | 36 |
| 2026062003 | 1,500 | 0.0785 | 34 |
| 2026062004 | 1,500 | 0.0082 | 28 |
| 2026062005 | 1,500 | -0.0375 | 19 |

The main candidate is effectively breakeven with mixed seed direction. The comparison candidates also have wide CIs and do not establish robust positive EV.

## Runtime Notes

- The first-only guard worked: second-position T2 overrides were blocked by `seat_not_allowed`.
- Runtime override rate stayed around 0.9% to 1.2%.
- Seat-swap output did not provide teacher EV / high-MC gain for fired overrides; those fields are therefore not used as promotion evidence in this C3 decision.
- `c3_failure_top30.jsonl` is a runtime audit sample ranked by poor `candidate_seat_score`, not a high-MC false-positive label set.

## Decision

- C3 execution: `Pass`
- C3 decision: `No-Go`
- `m2.75_r0_g0.95_seatfirst_k3` reproducibly positive: `No`
- first-only T2 override production value: `Not proven`
- second-position Stage8b override: `Keep disabled`
- selected MC4096/8192 refinement: `Diagnostic only`
- 50k teacher: `No-Go`
- T1 training: `No-Go`
- production / P2 fixed: `No-Go`
