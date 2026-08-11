# HU T1 Stage18 P1 Completion Audit

Date: 2026-07-11 JST

Scope: Regular OFC heads-up T1 first-seat selective override.

Conclusion: `stage18_p1` is accepted as the fixed first-seat T1 P1
continuation for downstream T0 work. It is not a second-seat policy, it is not
a full replacement policy, and it does not replace the global `current`
profile.

## Accepted Runtime

- Profile: `stage18_p1`
- Candidate model:
  `models/hu_turn1_stage18_first1801_mc32_hgb_abs_aug3.pkl`
- Safe selector:
  `models/hu_turn1_stage18_highmc_safe_selector_a_meta_only.pkl`
- Runtime:
  `k5/mc8/d3/confirm32/cse1.5/pd1.5/seat=first/safe0.7`
- Fallback and T2 continuation: `stage9f_p2`
- T3 continuation: `stage7_m5_r10`
- Mode: selective override only
- Allowed seat: `first`
- Rollback/off profile: `stage9f_p2`

## Completion Criteria

| Requirement | Evidence | Status |
|---|---|---|
| Candidate and selector artifacts exist | Both accepted model files under `models/` | Pass |
| Explicit fixed profile exists | `stage18_p1` with `runtime_status=p1_fixed` | Pass |
| Runtime is locked | P1 profile ignores the validation CLI config override | Pass |
| First seat only | `seat=first`; second seat logs `seat_not_allowed` and keeps fallback | Pass |
| Selective override only | Baseline `stage9f_p2` remains default/fallback | Pass |
| T2 continuation fixed | `stage9f_p2` | Pass |
| T3 continuation fixed | `stage7_m5_r10`, thresholds `5.0 / 10.0` | Pass |
| Fresh preregistered validation | C6 used non-overlapping seeds and no threshold adaptation | Pass |
| Minimum fired coverage | `340 >= 250` | Pass |
| Realized per-fire gain | `+3.5103`, 95% CI `[+1.9683, +5.0523]` | Pass |
| EV per decision | `+0.002389`, 95% CI `[+0.001339, +0.003438]` | Pass |
| Whole-hand EV | `+0.002387` | Pass |
| Non-fired cancellation | nonzero/mismatch/unknown all `0` | Pass |
| Invalid overrides | `0` | Pass |
| Fired tail limits | p95/p99/max `24.227 / 30.837 / 39.227` | Pass |
| Missing-model safety | P1 model and selector use safe loads and fall back to `stage9f_p2` | Pass |
| Canonical verifier | C6 `acceptance.json` has `passed=true` | Pass |
| Current-code wiring smoke | 20/20 P1 rows locked; second seat 10/10 fallback | Pass |
| Full regression suite | `697 passed` | Pass |

## C6 Evidence

Run:
`regular-hu-t1-stage18-c6-safe07-cse15-d3-pd15-250k-20260711-001`

Canonical small artifacts:
`outputs/evals/hu_turn1_stage18_c6_safe07_cse15_d3_pd15_250k_final/`

- paired seeds: `250,000`
- T1 decisions: `500,000`
- valid decisions: `499,660`
- realized fires: `340`
- valid fire rate: `0.0680%`
- realized per-fire delta: `+3.5103`
- realized per-fire 95% CI: `[+1.9683, +5.0523]`
- estimated EV/decision: `+0.002389`
- estimated EV/decision 95% CI: `[+0.001339, +0.003438]`
- whole-hand average: `+0.002387`
- fired losses/wins/zeros: `70 / 155 / 115`
- p95/p99/max loss: `24.227 / 30.837 / 39.227`
- invalid overrides: `0`
- non-fired cancellation errors: `0`

Confirm-MC estimates were used only by the runtime gate. Performance claims
above use realized seat-swap counterfactual deltas.

## Post-Wiring Smoke

Artifact: `outputs/evals/hu_turn1_stage18_p1_smoke10/`

- paired seeds / hands / T1 decisions: `10 / 20 / 20`
- runtime profile/status: `stage18_p1 / p1_fixed` on `20 / 20` rows
- config ID: locked C6 config on `20 / 20` rows
- first / second rows: `10 / 10`
- second-seat `seat_not_allowed`: `10 / 10`
- replay-ready: `20 / 20`
- non-fired counterfactual nonzero: `0`
- non-fired final mismatch: `0`
- matchup difference: `0`, as expected with no smoke overrides

Current full test suite: `697 passed`.

## Scope Boundary

P1 completion means the validated first-seat selective override is fixed for
downstream T0 experiments. The following remain explicitly outside this
acceptance:

- second-seat T1 override,
- T1 full replacement,
- automatic replacement of `current`,
- using confirm-MC delta as a performance estimate.

Rollback is immediate: select `stage9f_p2`, or use the
`stage18_p1_off` preset.
