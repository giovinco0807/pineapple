# HU T0 Stage19 P0 Completion Audit

Date: 2026-07-12 JST

Scope: Regular OFC heads-up T0 first-seat selective override.

Conclusion: `stage19_p0` is accepted as the fixed first-seat P0 policy over the
existing `stage18_p1` chain. It is explicit opt-in, not a second-seat policy,
not a full replacement, and does not replace `current`.

## Accepted Runtime

- Profile: `stage19_p0`
- Candidate model:
  `models/hu_turn0_stage3_top60_mc32_1000_hgb_baseline-delta_sew1_aug3.pkl`
- Safe selector:
  `models/hu_turn0_stage19_safe_selector_highmc_candidate_delta_plus_meta.pkl`
- Candidate TopK: `60`
- First-seat minimum predicted margin: `0.5`
- First-seat safe-selector threshold: `0.6`
- Allowed seat: `first`
- Fallback and T1 continuation: `stage18_p1`
- T2 continuation: `stage9f_p2`
- T3 continuation: `stage7_m5_r10`
- FL EV: `10.227020614683454`
- Visibility: hidden opponent discards; own private discards only
- Rollback profile: `stage18_p1`

## Fresh Acceptance Evidence

Run: `hu-t0-stage19-selector-holdout-v2-100k-20260712-001`

- Fresh paired seeds: `100,000`
- T0 decision events: `200,000`
- Realized fires: `3,747`
- Fire rate: `1.8735%` over both seat events, `3.747%` on first seat
- Whole-game EV: `+0.017303/hand`
- Whole-game 95% CI: `[+0.008292, +0.026314]`
- Realized gain per fire: `+0.9236`
- Per-fire 95% CI: `[+0.4434, +1.4037]`
- First-seat EV: `+0.034606`, 95% CI `[+0.016584, +0.052628]`
- Second-seat fires / EV: `0 / 0`
- Non-fired nonzero counterfactuals: `0`
- Duplicate events: `0`
- Seed-set mismatch: none
- p95 / p99 realized loss: `25.227 / 35.767`
- Maximum single-future loss, report only: `59.454`
- Formal preregistered verifier: `Go`

Canonical artifacts:

- `outputs/evals/hu_turn0_stage19_selector_holdout_v2_100k/summary.json`
- `outputs/evals/hu_turn0_stage19_selector_holdout_v2_100k/acceptance/acceptance.json`

## Tail Audit

The maximum single-future loss is sample-size dependent and is not a hard
policy gate. Safety is instead locked by p99 realized loss plus independent
MC512 replay of the prior holdout's worst 30 realized losses.

- Replayed records: `30 / 30`
- Common random futures: pass
- Action mapping: pass
- Mean MC512 candidate delta: `+0.8255`
- Negative-label rate: `13.33%`, below the preregistered `20%` ceiling

Artifact:
`outputs/evals/hu_turn0_stage19_selector_holdout_50k/replay/top30_tail_mc512_aggregate/summary.json`

## Post-Wiring Smoke

Artifact: `outputs/evals/hu_turn0_stage19_p0_smoke10/`

- paired seeds / hands / T0 decisions: `10 / 20 / 20`
- first / second rows: `10 / 10`
- runtime profile/status: `stage19_p0 / p0_fixed` on `20 / 20` rows
- second-seat disabled fallback: `10 / 10`
- replay-ready rows: `20 / 20`
- non-fired counterfactual nonzero / final mismatch: `0 / 0`
- smoke matchup delta: `0`, as expected with no smoke overrides

Final regression: `779 passed`.

Final GCP audit: project `ofc-solver-485418` had `0` RUNNING instances.

## Safety And Scope

The opening model remains the default action. T0 overrides only when all of
the following pass: first seat, candidate differs from baseline, predicted
margin at least `0.5`, selector probability at least `0.6`, finite prediction,
and legal placement.

If either Stage19 model cannot load, the entire T0 layer stays off and the
policy is exactly `stage18_p1`. Feature, prediction, selector, threshold, or
legality failure also keeps the Stage18 opening action.

Out of scope:

- second-seat T0 override,
- T0 full replacement,
- automatic replacement of `current`.

Rollback is immediate: select `stage18_p1` or use preset `stage19_p0_off`.
