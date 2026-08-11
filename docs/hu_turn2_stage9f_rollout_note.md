# HU T2 Stage9f Preflight Rollout Note

Status: validation-only preflight. This does not authorize production, P2 fixed
status, 50k teacher generation, or T1 training.

## Runtime

- Preset config:
  `configs/hu_turn2_stage9f_cse1p5_firstseat_preflight.json`
- Canary matrix:
  `configs/hu_turn2_stage9f_canary_presets.json`
- Model:
  `models/hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt`
- Runtime config string:
  `k3/mc16/d0/se0/confirm32/cse1.5/pd0/seat=first/bygate_delta`
- Validation AI profile:
  `stage9f_cse1p5_firstseat`
- T3 continuation:
  `stage7_m5_r10`
- Scope:
  first-seat HU T2 selective override only.

Stage9f is not a full replacement policy. Baseline T2 remains the default and
fallback policy.

The `current` AI profile is not changed. The Stage9f hook is available only
through the explicit `stage9f_cse1p5_firstseat` validation profile.

## Evidence

- Larger C4 validation:
  `outputs/evals/hu_turn2_stage9f_c4_larger_mc16c32_aggregate/`
- Tail-loss audit:
  `outputs/evals/hu_turn2_stage9f_c4_cse1p5_tail_loss_audit/`
- Fired-loss MC512 replay:
  `outputs/evals/hu_turn2_stage9f_c4_cse1p5_tail_loss_replay_mc512/`
- `csemax3` guard validation:
  `outputs/evals/hu_turn2_stage9f_c4_csemax3_aggregate/`
- Preflight smoke:
  `outputs/evals/hu_turn2_stage9f_preflight_cse1p5_smoke20/`
- Model-load failure fallback smoke:
  `outputs/evals/hu_turn2_stage9f_preflight_missing_model_fallback_smoke/`
- AI profile smoke:
  `outputs/evals/hu_turn2_stage9f_profile_smoke1/`
- AI profile TopK log smoke:
  `outputs/evals/hu_turn2_stage9f_profile_log_smoke5/`
- AI profile canary 100 audit:
  `outputs/evals/hu_turn2_stage9f_profile_canary100/`
- AI profile realized-delta canary 100 after T3 m5/r10 wiring fix:
  `outputs/evals/hu_turn2_stage9f_profile_canary100_realized_m5r10/`

Primary metric is realized fired whole-game paired delta. Confirm-MC delta is a
runtime gate diagnostic only.

## Current Decision

- `stage9f_cse1p5_firstseat`: validation pass, current best first-seat T2
  candidate.
- `stage9f_cse1p5_csemax2p5_firstseat`: experiment-only tail guard
  candidate derived from the 3500-paired profile canary. It is not a production
  default and needs independent seed validation.
- `stage9f_cse2_csemax2p5_firstseat`: stricter experiment-only tail guard
  candidate derived from the csemax2.5 independent run. It is not yet
  independently validated.
- `stage9f_cse2_csemax2_firstseat`: stricter experiment-only tail guard
  candidate independently validated on one 3500-paired profile canary with
  positive EV and per-fire CI lower bounds. It remains validation-only, not
  production.
- `stage9f_cse2_csemax2_bothseat`: current validation-default candidate after
  the mixed first+second 20x6000 GCP canary. It enables both seats with
  `fcd0/scd0`, has an explicit validation profile, and remains validation-only.
- `stage9f_cse2_csemax2_rank1_firstseat`: rejected rank guard. It looked good
  in the cse2+csemax2.0 retrospective sweep but failed partial independent
  validation with a negative realized per-fire delta and a large tail loss.
- `stage9f_cse1p5_csemax3_firstseat`: rejected guard.
- second seat: canary-validated only through `stage9f_cse2_csemax2_bothseat`.
- production / P2 fixed: No-Go.
- 50k teacher: No-Go.
- T1: No-Go.

## Safety

Fallback to baseline T2 on:

- model load failure or missing model object,
- prediction failure,
- feature generation failure,
- illegal candidate,
- NaN or infinite prediction,
- seat not allowed,
- threshold not met,
- Stage A or confirm MC rejection.

Missing Stage8b model fallback is an explicit preflight mode only. The default
CLI behavior is fatal if the candidate model cannot be loaded; use
`--allow-missing-stage8b-model-fallback` only for fallback smoke verification.

Runtime logs must preserve:

- `dead_cards`,
- `visible_dead_cards`,
- hero/opponent private discards,
- baseline action,
- final action,
- `override_fired`,
- `no_override_reason`,
- confirm delta and confirm SE,
- realized whole-game paired delta when available.

For profile-based matchup evaluation, pass `--topk-decision-output` to
`ofc_regular.evaluate_matchups`. This writes the Stage9f TopK decision records
from the explicit validation profile without changing `current`.

Use `ofc_regular.analyze_hu_turn2_stage9f_profile_canary` to summarize those
decision logs. The analyzer reports runtime plumbing metrics only; it does not
replace realized fired whole-game delta validation.

## Rollback

Use `stage9f_off` from `configs/hu_turn2_stage9f_canary_presets.json`.

Operationally, rollback means:

- do not pass a Stage9f TopK config to the HU T2 TopK runtime, or
- use the off preset and baseline Turn2 model only.

No model files need to be deleted. Stage7 T3 continuation remains independent
and should stay at `stage7_m5_r10` unless a separate T3 rollback is required.

## Preflight Command

```powershell
python -m ofc_regular.evaluate_hu_turn2_stage8b_topk_mc_rerank `
  --games-per-seed 20 `
  --seeds 2026065501 `
  --configs "k3/mc16/d0/se0/confirm32/cse1.5/pd0/seat=first/bygate_delta" `
  --t3-continuation stage7_m5_r10 `
  --hu-turn2-stage8b-model models/hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt `
  --output-dir outputs/evals/hu_turn2_stage9f_preflight_cse1p5_smoke20 `
  --write-decision-log
```

Expected smoke properties:

- command completes,
- `t3_continuation_policy = Stage7_candidate_A_m5_r10`,
- non-fired cancellation is clean,
- second-seat decisions show `seat_not_allowed`,
- production / P2 fixed remains No-Go.

## Model-Load Failure Fallback Smoke

```powershell
python -m ofc_regular.evaluate_hu_turn2_stage8b_topk_mc_rerank `
  --games-per-seed 3 `
  --seeds 2026065502 `
  --configs "k3/mc16/d0/se0/confirm32/cse1.5/pd0/seat=first/bygate_delta" `
  --t3-continuation stage7_m5_r10 `
  --hu-turn2-stage8b-model models/does_not_exist_stage9f.pt `
  --allow-missing-stage8b-model-fallback `
  --output-dir outputs/evals/hu_turn2_stage9f_preflight_missing_model_fallback_smoke `
  --write-decision-log `
  --device cpu `
  --prediction-threads 1
```

Observed result:

- command completed,
- `stage8b_model_loaded = False`,
- `stage8b_model_load_failed = True`,
- `no_override_reason` counts were `model_load_failed = 3` and
  `seat_not_allowed = 3`,
- `override_fired = False` for all `6` decisions,
- non-fired cancellation stayed clean.

## AI Profile Smoke

```powershell
python -m ofc_regular.evaluate_matchups `
  --profile-a stage9f_cse1p5_firstseat `
  --profile-b stage7_m5_r10 `
  --games 1 `
  --seed 2026065601 `
  --opening-lookahead-samples 1 `
  --prediction-threads 1 `
  --trace-limit 1 `
  --trace-output outputs/evals/hu_turn2_stage9f_profile_smoke1/trace.jsonl `
  --output outputs/evals/hu_turn2_stage9f_profile_smoke1/summary.json
```

Observed result:

- command completed with real model files loaded,
- `profile_a = stage9f_cse1p5_firstseat`,
- `profile_b = stage7_m5_r10`,
- `paired_seeds = 1`, `hands = 2`,
- `avg_score_per_hand_for_a = 0.0`.

This is a wiring smoke only, not performance evidence.

## AI Profile TopK Log Smoke

```powershell
python -m ofc_regular.evaluate_matchups `
  --profile-a stage9f_cse1p5_firstseat `
  --profile-b stage7_m5_r10 `
  --games 5 `
  --seed 2026065602 `
  --opening-lookahead-samples 1 `
  --prediction-threads 1 `
  --trace-limit 2 `
  --trace-output outputs/evals/hu_turn2_stage9f_profile_log_smoke5/trace.jsonl `
  --topk-decision-output outputs/evals/hu_turn2_stage9f_profile_log_smoke5/topk_decisions.jsonl `
  --output outputs/evals/hu_turn2_stage9f_profile_log_smoke5/summary.json
```

Observed result:

- command completed,
- `paired_seeds = 5`, `hands = 10`,
- `topk_decisions_written = 10`,
- no overrides fired in this tiny smoke,
- `no_override_reason` counts were `seat_not_allowed = 5`, `topk_empty = 3`,
  `mc_best_is_baseline = 1`, and `below_confirm_se = 1`.

This is log plumbing evidence only, not performance evidence.

## AI Profile Canary 100

The original profile canary at
`outputs/evals/hu_turn2_stage9f_profile_canary100/` is superseded. It exposed a
profile wiring bug: the Stage9f profile used Stage7 m5/r10 for T2 confirm
rollouts but did not pass the same m5/r10 thresholds to the actual T3 runtime.
That caused non-fired counterfactual deltas to be nonzero.

```powershell
python -m ofc_regular.evaluate_matchups `
  --profile-a stage9f_cse1p5_firstseat `
  --profile-b stage7_m5_r10 `
  --games 100 `
  --seed 2026065701 `
  --opening-lookahead-samples 1 `
  --prediction-threads 1 `
  --trace-limit 0 `
  --topk-decision-output outputs/evals/hu_turn2_stage9f_profile_canary100/topk_decisions.jsonl `
  --output outputs/evals/hu_turn2_stage9f_profile_canary100/summary.json `
  --progress-every 25

python -m ofc_regular.analyze_hu_turn2_stage9f_profile_canary `
  --decisions outputs/evals/hu_turn2_stage9f_profile_canary100/topk_decisions.jsonl `
  --matchup-summary outputs/evals/hu_turn2_stage9f_profile_canary100/summary.json `
  --output-dir outputs/evals/hu_turn2_stage9f_profile_canary100/audit
```

Observed runtime audit:

- `100` paired seeds / `200` hands completed locally in about `31s`,
- `200` TopK decisions written,
- `4` overrides, all first-seat,
- first-seat override rate `4 / 100 = 4.0%`,
- second-seat override rate `0 / 100`,
- replay-ready `200 / 200`,
- no missing replay fields,
- p95 runtime latency `530.82ms`.

No-override / decision buckets:

- `seat_not_allowed = 100`,
- `topk_empty = 63`,
- `mc_best_is_baseline = 14`,
- `below_confirm_delta = 11`,
- `below_confirm_se = 8`,
- `override_fired = 4`.

This confirms profile runtime wiring and logging. It is not a replacement for
C4-style realized fired whole-game delta validation.

## AI Profile Realized-Delta Canary 100

After wiring the actual T3 runtime thresholds to Stage7 m5/r10, the same seed
range was rerun:

```powershell
python -m ofc_regular.evaluate_matchups `
  --profile-a stage9f_cse1p5_firstseat `
  --profile-b stage7_m5_r10 `
  --games 100 `
  --seed 2026065701 `
  --opening-lookahead-samples 1 `
  --prediction-threads 1 `
  --trace-limit 0 `
  --topk-decision-output outputs/evals/hu_turn2_stage9f_profile_canary100_realized_m5r10/topk_decisions.jsonl `
  --output outputs/evals/hu_turn2_stage9f_profile_canary100_realized_m5r10/summary.json `
  --progress-every 25

python -m ofc_regular.analyze_hu_turn2_stage9f_profile_canary `
  --decisions outputs/evals/hu_turn2_stage9f_profile_canary100_realized_m5r10/topk_decisions.jsonl `
  --matchup-summary outputs/evals/hu_turn2_stage9f_profile_canary100_realized_m5r10/summary.json `
  --output-dir outputs/evals/hu_turn2_stage9f_profile_canary100_realized_m5r10/audit
```

Observed runtime audit:

- `100` paired seeds / `200` hands completed locally in about `30s`,
- `200` TopK decisions written,
- `4` overrides, all first-seat,
- realized per-fire delta `+10.3068`,
- estimated EV/decision `+0.2061`,
- non-fired nonzero realized deltas `0`,
- replay-ready `200 / 200`,
- no missing replay fields,
- p95 runtime latency `530.59ms`.

This run is a profile-runtime correctness canary. The fired sample is still too
small for production evidence, but the non-fired cancellation invariant now
holds.

## AI Profile Realized-Delta Canary 1250

After the T3 m5/r10 wiring fix, a larger local profile canary was run to collect
more fired decisions:

```powershell
python -m ofc_regular.evaluate_matchups `
  --profile-a stage9f_cse1p5_firstseat `
  --profile-b stage7_m5_r10 `
  --games 1250 `
  --seed 2026065702 `
  --opening-lookahead-samples 1 `
  --prediction-threads 1 `
  --trace-limit 0 `
  --topk-decision-output outputs/evals/hu_turn2_stage9f_profile_canary1250_realized_m5r10/topk_decisions.jsonl `
  --output outputs/evals/hu_turn2_stage9f_profile_canary1250_realized_m5r10/summary.json `
  --progress-every 250

python -m ofc_regular.analyze_hu_turn2_stage9f_profile_canary `
  --decisions outputs/evals/hu_turn2_stage9f_profile_canary1250_realized_m5r10/topk_decisions.jsonl `
  --matchup-summary outputs/evals/hu_turn2_stage9f_profile_canary1250_realized_m5r10/summary.json `
  --output-dir outputs/evals/hu_turn2_stage9f_profile_canary1250_realized_m5r10/audit
```

Observed runtime audit:

- `1250` paired seeds / `2500` hands completed locally in about `416s`,
- `2500` TopK decisions written,
- `38` overrides, all first-seat,
- override rate `1.52%` overall and `3.04%` on first-seat decisions,
- realized per-fire delta `+2.8014`,
- estimated EV/decision `+0.0426`,
- non-fired nonzero realized deltas `0`,
- replay-ready `2500 / 2500`,
- no missing replay fields,
- p95 runtime latency `567.03ms` overall and `756.90ms` for first-seat
  decisions.

This still falls short of the preferred `>=50` fired-decision target, so it is
not production evidence. It does confirm that the explicit AI profile now keeps
the non-fired cancellation invariant while producing positive realized fired
whole-game deltas on a larger local canary.

## AI Profile Realized-Delta Canary 3500

A follow-up local run targeted at least `100` fired decisions:

```powershell
python -m ofc_regular.evaluate_matchups `
  --profile-a stage9f_cse1p5_firstseat `
  --profile-b stage7_m5_r10 `
  --games 3500 `
  --seed 2026065703 `
  --opening-lookahead-samples 1 `
  --prediction-threads 1 `
  --trace-limit 0 `
  --topk-decision-output outputs/evals/hu_turn2_stage9f_profile_canary3500_realized_m5r10/topk_decisions.jsonl `
  --output outputs/evals/hu_turn2_stage9f_profile_canary3500_realized_m5r10/summary.json `
  --progress-every 500

python -m ofc_regular.analyze_hu_turn2_stage9f_profile_canary `
  --decisions outputs/evals/hu_turn2_stage9f_profile_canary3500_realized_m5r10/topk_decisions.jsonl `
  --matchup-summary outputs/evals/hu_turn2_stage9f_profile_canary3500_realized_m5r10/summary.json `
  --output-dir outputs/evals/hu_turn2_stage9f_profile_canary3500_realized_m5r10/audit
```

Observed runtime audit:

- `3500` paired seeds / `7000` hands completed locally in about `1057s`,
- `7000` TopK decisions written,
- `109` overrides, all first-seat,
- override rate `1.56%` overall and `3.11%` on first-seat decisions,
- whole-hand EV/hand `+0.0230`, 95% CI `[-0.0103, +0.0562]`,
- realized per-fire delta `+1.4741`,
- realized per-fire 95% CI `[-0.6510, +3.5992]`,
- positive / negative / zero fired deltas: `31 / 21 / 57`,
- p95 / max fired loss: `18.2270 / 36.2270`,
- non-fired nonzero realized deltas `0`,
- replay-ready `7000 / 7000`,
- no missing replay fields,
- p95 runtime latency `533.74ms` overall and `641.80ms` for first-seat
  decisions.

Additional artifact:

- `outputs/evals/hu_turn2_stage9f_profile_canary3500_realized_m5r10/audit/profile_canary_fired_losses_top30.jsonl`

Decision:

- This is a useful runtime-canary improvement over the 1250-paired run because
  it reaches `109` fired decisions and preserves exact non-fired cancellation.
- It is still not production evidence: the per-fire CI crosses zero and the
  fired tail loss remains heavy.
- Retrospective guard sweep on the same fired set prefers `confirm_se<=2.5`:
  `81` fires, estimated EV/decision `+0.0244`, per-fire `+2.1128`, per-fire
  95% CI `[-0.0141, +4.2397]`, p95/max loss `8.0 / 30.2270`.
- Added experiment-only profile `stage9f_cse1p5_csemax2p5_firstseat` for
  independent validation of that guard. Simply scaling the unguarded profile is
  unlikely to be enough for production.
- Wiring smoke for that experiment-only profile:
  `outputs/evals/hu_turn2_stage9f_csemax2p5_profile_smoke1/` completed `1`
  paired seed / `2` hands, wrote `2` TopK decision rows, and confirmed
  `runtime_profile = stage9f_cse1p5_csemax2p5_firstseat`,
  `max_confirm_se = 2.5`, and
  `t3_continuation_policy = Stage7_candidate_A_m5_r10`.

## csemax2.5 Profile Canary 3500

The `stage9f_cse1p5_csemax2p5_firstseat` guard was then validated on
non-overlapping seeds:

```powershell
python -m ofc_regular.evaluate_matchups `
  --profile-a stage9f_cse1p5_csemax2p5_firstseat `
  --profile-b stage7_m5_r10 `
  --games 3500 `
  --seed 2026065901 `
  --opening-lookahead-samples 1 `
  --prediction-threads 1 `
  --trace-limit 0 `
  --topk-decision-output outputs/evals/hu_turn2_stage9f_csemax2p5_profile_canary3500_realized_m5r10/topk_decisions.jsonl `
  --output outputs/evals/hu_turn2_stage9f_csemax2p5_profile_canary3500_realized_m5r10/summary.json `
  --progress-every 500
```

Observed runtime audit:

- `3500` paired seeds / `7000` hands completed locally in about `1319s`,
- `64` overrides, all first-seat,
- whole-hand EV/hand `+0.0104`, 95% CI `[-0.0098, +0.0306]`,
- realized per-fire delta `+1.1406`,
- realized per-fire 95% CI `[-1.0687, +3.3500]`,
- positive / negative / zero fired deltas: `23 / 11 / 30`,
- p95 / max fired loss: `12.0 / 26.2270`,
- non-fired nonzero realized deltas `0`,
- replay-ready `7000 / 7000`,
- p95 runtime latency `615.77ms` overall and `951.44ms` for first-seat
  decisions.

Interpretation:

- `csemax2.5` did reduce tail risk versus the unguarded profile, but it also
  reduced fire count and EV contribution.
- It is not a production candidate.
- Retrospective guard sweep on this run prefers adding `confirm_z>=2.0`,
  equivalent to `cse2 + csemax2.5`: `46` fires, estimated EV/decision
  `+0.0200`, per-fire `+3.0365`, per-fire 95% CI `[+0.7919, +5.2812]`,
  p95/max loss `5.0 / 12.0`.
- Added experiment-only profile `stage9f_cse2_csemax2p5_firstseat` for the next
  independent validation. Because it is selected from this run, it must be
  tested on non-overlapping seeds before any adoption decision.
- Wiring smoke for `stage9f_cse2_csemax2p5_firstseat`:
  `outputs/evals/hu_turn2_stage9f_cse2_csemax2p5_profile_smoke1/` completed
  `1` paired seed / `2` hands, wrote `2` TopK decision rows, and confirmed
  `confirm_se_multiplier = 2.0`, `max_confirm_se = 2.5`, and
  `t3_continuation_policy = Stage7_candidate_A_m5_r10`.

## cse2+csemax2.5 Profile Canary 3500

The `stage9f_cse2_csemax2p5_firstseat` profile was then evaluated on
non-overlapping seeds:

```powershell
python -m ofc_regular.evaluate_matchups `
  --profile-a stage9f_cse2_csemax2p5_firstseat `
  --profile-b stage7_m5_r10 `
  --games 3500 `
  --seed 2026066101 `
  --opening-lookahead-samples 1 `
  --prediction-threads 1 `
  --trace-limit 0 `
  --topk-decision-output outputs/evals/hu_turn2_stage9f_cse2_csemax2p5_profile_canary3500_realized_m5r10/topk_decisions.jsonl `
  --output outputs/evals/hu_turn2_stage9f_cse2_csemax2p5_profile_canary3500_realized_m5r10/summary.json `
  --progress-every 500
```

Observed runtime audit:

- `3500` paired seeds / `7000` hands completed locally in about `1282s`,
- `46` overrides, all first-seat,
- whole-hand EV/hand `+0.0134`, 95% CI `[-0.0086, +0.0353]`,
- realized per-fire delta `+2.0316`,
- realized per-fire 95% CI `[-1.3005, +5.3637]`,
- positive / negative / zero fired deltas: `14 / 4 / 28`,
- p95 / max fired loss: `5.0 / 36.2270`,
- non-fired nonzero realized deltas `0`,
- replay-ready `7000 / 7000`,
- p95 runtime latency `589.55ms` overall and `859.51ms` for first-seat
  decisions.

Interpretation:

- The stricter `cse2+csemax2.5` profile lowers negative fire count, but the
  EV contribution is still small and the per-fire CI still crosses zero.
- A single severe loss remains.
- Retrospective sweep on this run points to `confirm_se<=2.0`
  (`cse2+csemax2.0`): `33` fires, estimated EV/decision `+0.0180`,
  per-fire `+3.8085`, per-fire 95% CI `[+0.5267, +7.0903]`, and p95/max loss
  `0.0 / 2.0`.
- Added experiment-only profile `stage9f_cse2_csemax2_firstseat` for the next
  independent validation. Because it is selected from this run, it must be
  tested on non-overlapping seeds before any adoption decision.
- Wiring smoke for `stage9f_cse2_csemax2_firstseat`:
  `outputs/evals/hu_turn2_stage9f_cse2_csemax2_profile_smoke1/` completed `1`
  paired seed / `2` hands, wrote `2` TopK decision rows, and confirmed
  `confirm_se_multiplier = 2.0`, `max_confirm_se = 2.0`,
  `no_override_reason = above_confirm_se` on the first-seat row, and
  `t3_continuation_policy = Stage7_candidate_A_m5_r10`.

## cse2+csemax2.0 Profile Canary 3500

The `stage9f_cse2_csemax2_firstseat` profile was then evaluated on
non-overlapping seeds:

```powershell
python -m ofc_regular.evaluate_matchups `
  --profile-a stage9f_cse2_csemax2_firstseat `
  --profile-b stage7_m5_r10 `
  --games 3500 `
  --seed 2026066301 `
  --opening-lookahead-samples 1 `
  --prediction-threads 1 `
  --trace-limit 0 `
  --topk-decision-output outputs/evals/hu_turn2_stage9f_cse2_csemax2_profile_canary3500_realized_m5r10/topk_decisions.jsonl `
  --output outputs/evals/hu_turn2_stage9f_cse2_csemax2_profile_canary3500_realized_m5r10/summary.json `
  --progress-every 500
```

Observed runtime audit:

- `3500` paired seeds / `7000` hands completed locally in about `1260s`,
- `39` overrides, all first-seat,
- whole-hand EV/hand `+0.0174`, 95% CI `[+0.0030, +0.0319]`,
- realized per-fire delta `+3.1258`,
- realized per-fire 95% CI `[+0.6907, +5.5610]`,
- positive / negative / zero fired deltas: `14 / 2 / 23`,
- p95 / max fired loss: `2.0 / 19.2270`,
- non-fired nonzero realized deltas `0`,
- replay-ready `7000 / 7000`,
- p95 runtime latency `592.63ms` overall and `844.84ms` for first-seat
  decisions.

Artifacts:

- `outputs/evals/hu_turn2_stage9f_cse2_csemax2_profile_canary3500_realized_m5r10/audit/profile_canary_summary.md`
- `outputs/evals/hu_turn2_stage9f_cse2_csemax2_profile_canary3500_realized_m5r10/tail_loss_audit/tail_loss_guard_sweep.csv`
- `outputs/evals/hu_turn2_stage9f_cse2_csemax2_profile_canary3500_realized_m5r10/tail_loss_audit/tail_loss_top50.jsonl`

Interpretation:

- This is the first Stage9f profile canary in this sequence where both
  whole-hand EV/hand and realized per-fire CI have positive lower bounds.
- Non-fired cancellation stayed exact, so the result is not inflated by
  background hand noise.
- It is still validation-only: the fired sample is `39`, second-seat is not
  enabled, and production/P2 fixed remains No-Go.
- Tail risk is much lower than unguarded `cse1.5`, but a `19.2270` loss remains.
- Retrospective guard sweep on this run points to `rank<=1`: `21` fires,
  estimated EV/decision `+0.0131`, per-fire `+4.3766`, per-fire 95% CI
  `[+1.1565, +7.5967]`, and p95/max loss `0.0 / 2.0`.
- Because `rank<=1` is selected from this run, it needs independent validation
  before adoption. The current next step is a non-overlapping `rank<=1`
  canary, not production.
- Added experiment-only profile `stage9f_cse2_csemax2_rank1_firstseat`.
- Wiring smoke:
  `outputs/evals/hu_turn2_stage9f_cse2_csemax2_rank1_profile_smoke1/`
  completed `1` paired seed / `2` hands, wrote `2` TopK decision rows, and
  confirmed `candidate_ev_rank_max = 1`, `confirm_se_multiplier = 2.0`,
  `max_confirm_se = 2.0`, and
  `t3_continuation_policy = Stage7_candidate_A_m5_r10`.

## cse2+csemax2.0 rank1 Partial Independent Validation

The `rank<=1` guard was then started on non-overlapping seeds:

```powershell
python -u -m ofc_regular.evaluate_matchups `
  --profile-a stage9f_cse2_csemax2_rank1_firstseat `
  --profile-b stage7_m5_r10 `
  --games 10000 `
  --seed 2026066701 `
  --opening-lookahead-samples 1 `
  --prediction-threads 1 `
  --trace-limit 0 `
  --topk-decision-output outputs/evals/hu_turn2_stage9f_cse2_csemax2_rank1_profile_canary10000_realized_m5r10/topk_decisions.jsonl `
  --output outputs/evals/hu_turn2_stage9f_cse2_csemax2_rank1_profile_canary10000_realized_m5r10/summary.json `
  --progress-every 500
```

The run was stopped early after the independent sample contradicted the
retrospective tail-risk claim:

- `4047` paired seeds / `8094` decisions reached,
- `28` first-seat overrides,
- estimated EV/decision from realized fired deltas `-0.0014`,
- realized per-fire delta `-0.4010`,
- realized per-fire 95% CI `[-2.9542, +2.1523]`,
- positive / negative / zero fired deltas: `8 / 5 / 15`,
- p95 / max fired loss: `22.0270 / 24.2270`,
- non-fired nonzero realized deltas `0`.

Artifact:

- `outputs/evals/hu_turn2_stage9f_cse2_csemax2_rank1_profile_canary10000_realized_m5r10/partial_abort_summary.md`

Decision:

- `rank<=1` is rejected. It was a same-run retrospective artifact, not a stable
  tail guard.
- Do not continue this 10000-paired run.
- The broader `stage9f_cse2_csemax2_firstseat` profile remains the better
  validation candidate. Next work should either run a larger non-overlapping
  validation of that profile or train a new tail-risk guard from independent
  fired-loss rows.

## cse2+csemax2.0 Profile Canary 10000

The broader `stage9f_cse2_csemax2_firstseat` profile was then evaluated on a
larger non-overlapping seed range:

```powershell
python -u -m ofc_regular.evaluate_matchups `
  --profile-a stage9f_cse2_csemax2_firstseat `
  --profile-b stage7_m5_r10 `
  --games 10000 `
  --seed 2026066901 `
  --opening-lookahead-samples 1 `
  --prediction-threads 1 `
  --trace-limit 0 `
  --topk-decision-output outputs/evals/hu_turn2_stage9f_cse2_csemax2_profile_canary10000_realized_m5r10/topk_decisions.jsonl `
  --output outputs/evals/hu_turn2_stage9f_cse2_csemax2_profile_canary10000_realized_m5r10/summary.json `
  --progress-every 500
```

Observed runtime audit:

- `10000` paired seeds / `20000` hands completed locally in about `3739s`,
- `90` overrides, all first-seat,
- whole-hand EV/hand `+0.0129`, 95% CI `[+0.0036, +0.0222]`,
- realized per-fire delta `+2.8571`,
- realized per-fire 95% CI `[+0.8641, +4.8500]`,
- positive / negative / zero fired deltas: `31 / 7 / 52`,
- p95 / max fired loss: `8.0 / 31.2270`,
- non-fired nonzero realized deltas `0`,
- replay-ready `20000 / 20000`,
- p95 runtime latency `596.27ms`.

Artifacts:

- `outputs/evals/hu_turn2_stage9f_cse2_csemax2_profile_canary10000_realized_m5r10/audit/profile_canary_summary.md`
- `outputs/evals/hu_turn2_stage9f_cse2_csemax2_profile_canary10000_realized_m5r10/tail_loss_audit/tail_loss_guard_sweep.csv`
- `outputs/evals/hu_turn2_stage9f_cse2_csemax2_profile_canary10000_realized_m5r10/tail_loss_audit/tail_loss_top.jsonl`

Interpretation:

- The positive signal reproduced at larger size: both whole-hand EV/hand and
  realized per-fire CI lower bounds are above zero.
- Non-fired cancellation stayed exact, so the aggregate EV is attributable to
  fired hands.
- This is still validation-only. Production/P2 fixed remains No-Go because the
  fired tail still includes `7` losses, p95 loss `8.0`, and max loss `31.2270`.
- The same-run guard sweep again makes simple retrospective filters look
  attractive, but independent validation already rejected `rank<=1`; do not
  promote `rank<=1` from any same-run sweep.
- `confirm_z>=2.5` is diagnostic only: in this run it kept `53` fires with
  estimated EV/decision `+0.0076`, per-fire `+2.8576`, p95/max loss
  `0.0 / 26.2270`. It lowers fire count and still leaves a severe max loss, so
  it needs independent validation before use.
- Next useful work is to train or validate a new tail-risk guard from
  independent fired-loss rows, or run selected high-MC replay on the worst
  losses. Do not move to T1, 50k teacher, production, or P2 fixed from this
  result alone.

## Tail-Guard Target Extraction

The canonical next-input target set was generated from the `10000` canary only
so it does not mix overlapping earlier canary seeds:

```powershell
python -m ofc_regular.prepare_hu_turn2_stage9f_tail_guard_targets `
  --input-dir outputs/evals/hu_turn2_stage9f_cse2_csemax2_profile_canary10000_realized_m5r10 `
  --config-id stage9f_cse2_csemax2_firstseat `
  --output-dir outputs/training/hu_turn2_stage9f_cse2_csemax2_tail_guard_targets_10000 `
  --loss-limit 200 `
  --positive-limit 100 `
  --zero-limit 100 `
  --boundary-limit 100 `
  --severe-loss-threshold 8.0 `
  --safe-positive-threshold 2.0 `
  --confirm-z-threshold 2.0 `
  --confirm-z-boundary-width 0.35
```

Output:

- `outputs/training/hu_turn2_stage9f_cse2_csemax2_tail_guard_targets_10000/stage9f_tail_guard_targets.jsonl`
- `outputs/training/hu_turn2_stage9f_cse2_csemax2_tail_guard_targets_10000/stage9f_tail_guard_target_summary.md`
- `outputs/training/hu_turn2_stage9f_cse2_csemax2_tail_guard_targets_10000/stage9f_tail_guard_target_manifest.json`

Counts:

- `114` targets, `114 / 114` replay-ready,
- source fired realized rows: `90`,
- tail losses: `7`, including `5` severe losses at threshold `8.0`,
- positive controls: `28`,
- zero controls: `52`,
- confirm-z boundary targets: `27`.

This artifact is for tail-risk guard training or selected high-MC replay. It is
not production evidence and does not change the No-Go status for production,
P2 fixed, T1, or 50k teacher.

## Tail-Guard Replay Runner

Added `ofc_regular.replay_hu_turn2_stage9f_tail_guard_targets` to replay these
runtime-log targets with independent common future decks. The runner:

- reconstructs the T2 board and legal action list from each target,
- verifies that the logged baseline and fired candidate actions map back to the
  same legal action indices,
- can run `--readiness-only`,
- writes `replay_results.partial.jsonl` and `progress.json` incrementally, and
- supports `--resume` for Spot VM / interrupted runs.

Readiness command:

```powershell
python -m ofc_regular.replay_hu_turn2_stage9f_tail_guard_targets `
  --readiness-only `
  --targets outputs/training/hu_turn2_stage9f_cse2_csemax2_tail_guard_targets_10000/stage9f_tail_guard_targets.jsonl `
  --output-dir outputs/evals/hu_turn2_stage9f_tail_guard_replay_readiness_10000
```

Readiness result:

- `114 / 114` targets ready,
- action mapping recovered for all targets,
- output: `outputs/evals/hu_turn2_stage9f_tail_guard_replay_readiness_10000/`.

Selected high-MC replay of the `7` tail-loss targets:

```powershell
python -m ofc_regular.replay_hu_turn2_stage9f_tail_guard_targets `
  --targets outputs/training/hu_turn2_stage9f_cse2_csemax2_tail_guard_targets_10000/stage9f_tail_guard_targets.jsonl `
  --target-group tail_loss `
  --output-dir outputs/evals/hu_turn2_stage9f_tail_guard_replay_tail_loss_mc512 `
  --mc-samples 512 `
  --opening-lookahead-samples 1 `
  --prediction-threads 1 `
  --batched-continuation-batch-size 8192 `
  --stage3-feature-encoder-mode rust_direct
```

Result:

- `7 / 7` replay successes,
- high-MC gain mean `+1.3560`,
- min / max gain `-1.9255 / +3.9536`,
- high-MC losses `1 / 7`,
- LCB95-positive rows `5 / 7`,
- sign flips vs input realized deltas `6 / 7`.

Interpretation:

- Most large realized whole-hand losses from the seat-swap canary do not remain
  negative under independent MC512. They are still valuable as tail-risk
  examples, but not all should become hard negatives.
- One row remains negative at MC512 and should be treated as the first true
  hard-negative seed for the next guard.
- The MC512 replay rows were converted into training labels with
  `ofc_regular.label_hu_turn2_stage9f_tail_guard_replay`:
  `outputs/training/hu_turn2_stage9f_tail_guard_labels_tail_loss_mc512/`.
  Label counts are `1` hard negative, `5` safe positives, and `1` gray row.
  These labels are inputs for a future tail-risk guard; they are not runtime
  thresholds and not production evidence.
- The full `114` target replay should use Spot VM or the new partial/resume
  path if run locally:

```powershell
python -m ofc_regular.replay_hu_turn2_stage9f_tail_guard_targets `
  --targets outputs/training/hu_turn2_stage9f_cse2_csemax2_tail_guard_targets_10000/stage9f_tail_guard_targets.jsonl `
  --output-dir outputs/evals/hu_turn2_stage9f_tail_guard_replay_all_mc512 `
  --mc-samples 512 `
  --opening-lookahead-samples 1 `
  --prediction-threads 1 `
  --batched-continuation-batch-size 8192 `
  --stage3-feature-encoder-mode rust_direct `
  --resume `
  --progress-every 1
```

The full `114` targets were then replayed on GCP Spot VM:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\Start-GcpHuTurn2Stage9fTailGuardReplayRun.ps1 `
  -RunName regular-hu-t2-stage9f-tail-guard-replay-20260622-221333 `
  -ChunkSize 8 `
  -VmCount 10 `
  -CreateInstances
```

Receive command:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\Receive-GcpHuTurn2Stage9fTailGuardReplayRun.ps1 `
  -RunName regular-hu-t2-stage9f-tail-guard-replay-20260622-221333 `
  -OutputDir outputs/evals/hu_turn2_stage9f_tail_guard_replay_all_mc512_gcp
```

Result:

- output: `outputs/evals/hu_turn2_stage9f_tail_guard_replay_all_mc512_gcp/`,
- shards: `15 / 15` complete,
- replay rows: `114 / 114`,
- raw high-MC labels: `8` hard negatives, `72` safe positives, `34` gray,
- duplicated source rows across target buckets: `25`,
- deduped event labels: `89` rows = `5` hard negatives, `57` safe positives,
  `27` gray.

Interpretation:

- The training input should use the deduped label file:
  `outputs/evals/hu_turn2_stage9f_tail_guard_replay_all_mc512_gcp/labels/stage9f_tail_guard_labels_dedup.csv`.
- Raw `114` labels remain useful for audit/source-bucket accounting, but should
  not be used directly for model training because some events appear in more
  than one target group.
- These labels are still guard-training data only. Production/P2 fixed, 50k
  teacher, and T1 remain `No-Go`.

## Stage9g Tail-Guard Smoke

The full MC512 replay labels were converted into Stage9g risk-head training rows:

```powershell
python -m ofc_regular.prepare_hu_turn2_stage9g_tail_guard_training `
  --output-dir outputs/training/hu_turn2_stage9g_tail_guard_training
```

Training input:

- rows excluding gray: `62`,
- hard negatives: `5`,
- safe/non-loss controls: `57`,
- source labels:
  `outputs/evals/hu_turn2_stage9f_tail_guard_replay_all_mc512_gcp/labels/stage9f_tail_guard_labels_dedup.csv`.

Three smoke models were trained with `train_hu_turn2_stage8c_risk_head`:

| feature mode | all AP | all AUC | all top1 precision | all top3 precision | runtime threshold status |
|---|---:|---:|---:|---:|---|
| `preconfirm_meta_only` | `0.4137` | `0.6982` | `1.000` | `0.667` | `No-Go` |
| `opportunity_proxy_plus_preconfirm_meta` | `0.3884` | `0.6737` | `1.000` | `0.333` | `No-Go` |
| `hu_delta_plus_preconfirm_meta` | `0.5967` | `0.9404` | `1.000` | `0.667` | `No-Go` |

Interpretation:

- `hu_delta_plus_preconfirm_meta` is the best next offline ranking candidate:
  it has the strongest all-row AP/AUC and ranks at least one hard-negative row
  at the top.
- It is not a runtime guard yet. The predicted probabilities are still clustered
  near `0.5`, and thresholds `>= 0.7` select zero rows in the smoke.
- The current data has only `5` deduped hard negatives. Large Stage9g training
  should wait until at least `50` deduped hard negatives are available, with
  `100+` preferred.

Config:

- `configs/hu_turn2_stage9g_tail_guard_training_plan.json`

Decision:

- Stage9g label prep and smoke training: `Go`.
- Stage9g runtime guard: `No-Go`.
- Stage9g large training: `No-Go until more hard negatives`.
- Production/P2 fixed, 50k teacher, and T1: still `No-Go`.

Next useful step:

- Collect more selected high-MC labels for Stage9g from Stage9f
  `cse2/csemax2` fired losses, near-confirm-boundary rows, rank/confirm
  disagreements, large-gain positive controls, and zero controls.
  Use GCP Spot VM if the selected replay is too slow locally.

### Expanded Stage9g Labels

Additional mixed Stage9f runtime logs were mined with event-key dedupe before
replay:

```powershell
python -m ofc_regular.prepare_hu_turn2_stage9f_tail_guard_targets `
  --input-dir outputs/evals/hu_turn2_stage9f_cse2_csemax2_profile_canary10000_realized_m5r10 `
  --input-dir outputs/evals/hu_turn2_stage9f_cse2_csemax2_profile_canary3500_realized_m5r10 `
  --input-dir outputs/evals/hu_turn2_stage9f_cse2_csemax2p5_profile_canary3500_realized_m5r10 `
  --input-dir outputs/evals/hu_turn2_stage9f_csemax2p5_profile_canary3500_realized_m5r10 `
  --input-dir outputs/evals/hu_turn2_stage9f_profile_canary3500_realized_m5r10 `
  --input-dir outputs/evals/hu_turn2_stage9f_profile_canary1250_realized_m5r10 `
  --output-dir outputs/training/hu_turn2_stage9g_more_tail_guard_targets_stage9f_mixed `
  --target-name stage9g_more_tail_guard_targets.jsonl `
  --loss-limit 500 `
  --positive-limit 500 `
  --zero-limit 250 `
  --boundary-limit 500 `
  --dedupe-event-key
```

Target result:

- source rows: `50,500`,
- source fired-realized rows: `386`,
- event-deduped replay targets: `317`,
- replay-ready: `317 / 317`,
- duplicate source rows removed before replay: `118`,
- target groups:
  - tail loss `44`,
  - severe tail loss `26`,
  - positive control `62`,
  - zero control `112`,
  - confirm-z boundary `99`.

GCP MC512 replay:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\Start-GcpHuTurn2Stage9fTailGuardReplayRun.ps1 `
  -RunName regular-hu-t2-stage9g-more-tailguard-replay-20260623-001 `
  -Targets outputs\training\hu_turn2_stage9g_more_tail_guard_targets_stage9f_mixed\stage9g_more_tail_guard_targets.jsonl `
  -ChunkSize 8 `
  -VmCount 10 `
  -CreateInstances
```

The first start timed out locally after launching VM indices `0..4`, so VM
indices `5..9` and then start shard `5` were restarted with the same run name.
All output is in:

- `outputs/evals/hu_turn2_stage9g_more_tail_guard_replay_stage9f_mixed_gcp/`

Replay result:

- shards: `40 / 40`,
- replay rows: `317 / 317`,
- failures: `0`,
- labels: `21` hard negatives, `195` safe positives, `101` gray.

Training rows:

```powershell
python -m ofc_regular.prepare_hu_turn2_stage9g_tail_guard_training `
  --targets outputs/training/hu_turn2_stage9g_more_tail_guard_targets_stage9f_mixed/stage9g_more_tail_guard_targets.jsonl `
  --labels outputs/evals/hu_turn2_stage9g_more_tail_guard_replay_stage9f_mixed_gcp/labels/stage9f_tail_guard_labels_dedup.csv `
  --output-dir outputs/training/hu_turn2_stage9g_tail_guard_training_stage9f_mixed_mc512
```

- rows excluding gray: `216`,
- hard negatives: `21`,
- safe controls: `195`.

New smoke results:

| feature mode | rows | hard neg | all AP | all AUC | test AP | test AUC | threshold 0.6 precision | threshold 0.7 selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `hu_delta_plus_preconfirm_meta` | `216` | `21` | `0.8389` | `0.9516` | `0.5111` | `0.8161` | `0.9000` | `0` |
| `preconfirm_meta_only` | `216` | `21` | `0.2508` | `0.7282` | `0.2578` | `0.6207` | `0.0000` | `0` |

Interpretation:

- `hu_delta_plus_preconfirm_meta` remains the Stage9g candidate for offline
  tail-risk ranking.
- This is a real improvement over the `5` hard-negative smoke, but still not a
  production/runtime guard. The validation split is weak and the hard-negative
  count is only `21`.
- Large Stage9g training remains `No-Go` until at least `50` deduped hard
  negatives are available; `100+` is preferred.

## Both-Seat Stage9f Per-Fire Canary

After separate first-seat and second-seat Stage9f checks, the same
`cse2+csemax2.0` two-stage MC gate was validated with both seats enabled.

Run:

- GCP run:
  `regular-hu-t2-stage9f-bothseat-canary-chunked-20260623-001`
- Output:
  `outputs/evals/hu_turn2_stage9f_bothseat_canary_chunked_20x6000_m5r10/`
- Config:
  `k3/mc16/d0/se0/confirm32/cse2/csemax2/fcd0/scd0/pd0/seat=first+second/bygate_delta`
- Model:
  `models/hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt`
- T3 continuation:
  `stage7_m5_r10`
- Seeds:
  `2026067401..2026067420`
- Scale:
  `20` shards, `6000` games per seed, target `60` realized overrides per seed.
- Spot recovery:
  shard `14` was requeued after repeated `us-central1-b` failures and completed
  in `us-east1-b`.

Aggregate result:

- paired seeds / hands: `69638`,
- realized fires: `1200`,
- estimated EV/hand: `+0.02370`,
- realized per-fire delta: `+2.73199`,
- realized per-fire 95% CI: `[+2.22675, +3.23723]`,
- confirm diagnostic mean: `+3.54814`,
- confirm-realized gap: `+0.81615`,
- losses: `126`,
- p95 / max fired loss: `6.0 / 34.2270`,
- non-fired cancellation:
  `non_fired_nonzero_count = 0`, `non_fired_delta_sum = 0`,
  `max_non_fired_delta_abs = 0`.

Position split:

- first seat:
  `617` fires, estimated EV/hand `+0.02542`, per-fire `+2.85102`,
  per-fire CI `[+2.13321, +3.56884]`, p95/max loss `6.0 / 31.2270`.
- second seat:
  `583` fires, estimated EV/hand `+0.02197`, per-fire `+2.60602`,
  per-fire CI `[+1.89534, +3.31669]`, p95/max loss `6.0 / 34.2270`.

Interpretation:

- Both seats can be enabled together in validation without degrading either
  position in this canary.
- This supersedes the older `second seat not validated` exclusion.
- Confirm-MC delta remains gate diagnostic only. Performance claims use realized
  fired whole-game paired deltas.
- `stage9f_cse2_csemax2_bothseat` is now the validation-default preset and an
  explicit validation AI profile in `configs/hu_turn2_stage9f_canary_presets.json`.
- This is still not a production/P2 decision. A longer independent C4
  validation plus runtime preflight is required before promotion.

Decision:

- Stage9f both-seat validation candidate: `Go`.
- Stage9f production/P2 fixed: `No-Go`.
- 50k teacher: `No-Go`.
- T1 training: `No-Go`.

## Both-Seat Stage9f Larger C4 Validation

The both-seat `cse2+csemax2.0` two-stage MC gate was then validated on a larger
independent GCP run.

Run:

- GCP run:
  `regular-hu-t2-stage9f-bothseat-c4-chunked-20260623-001`
- Output:
  `outputs/evals/hu_turn2_stage9f_bothseat_c4_chunked_40x12000_m5r10/`
- Config:
  `k3/mc16/d0/se0/confirm32/cse2/csemax2/fcd0/scd0/pd0/seat=first+second/bygate_delta`
- Model:
  `models/hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt`
- T3 continuation:
  `stage7_m5_r10`
- Seeds:
  `2026067601..2026067640`
- Scale:
  `40` shards, `12000` games per seed, target `100` realized overrides per
  seed.
- Spot recovery:
  several shards were requeued after spot preemption; all `40 / 40` shards
  completed.

Aggregate result:

- paired seeds / hands: `218116`,
- realized fires: `4000`,
- override rate: `0.9215%`,
- estimated EV/hand from fired deltas: `+0.02580`,
- seed-mean EV/hand: `+0.02614`,
- seed-mean 95% CI: `[+0.02355, +0.02874]`,
- realized per-fire delta: `+2.79986`,
- realized per-fire 95% CI: `[+2.51686, +3.08286]`,
- confirm diagnostic mean: `+3.76829`,
- confirm-realized gap: `+0.96843`,
- losses: `448`,
- p95 / max fired loss: `7.0 / 37.2270`,
- non-fired cancellation:
  `non_fired_nonzero_count = 0`, `non_fired_delta_sum = 0`,
  `max_non_fired_delta_abs = 0`.

Position split:

- first seat:
  `1956` fires, estimated EV/hand `+0.02453`, per-fire `+2.72177`,
  per-fire CI `[+2.32263, +3.12091]`, p95/max loss `8.0 / 36.2270`.
- second seat:
  `2044` fires, estimated EV/hand `+0.02707`, per-fire `+2.87458`,
  per-fire CI `[+2.47349, +3.27568]`, p95/max loss `7.0 / 37.2270`.

Interpretation:

- The both-seat Stage9f validation signal reproduced at larger scale.
- Both positions have positive per-fire lower bounds and comparable EV
  contribution.
- Non-fired cancellation stayed exact, so the aggregate EV estimate is
  attributable to fired hands.
- Confirm-MC delta remains a gate diagnostic only. Performance claims use
  realized fired whole-game paired deltas.
- Tail losses are not gone; p95 and max fired loss still require runtime
  preflight and rollback-safe deployment discipline before any production
  promotion.

Decision:

- Stage9f both-seat C4 validation: `Pass`.
- `stage9f_cse2_csemax2_bothseat`: current validation-default candidate and
  explicit validation AI profile.
- Production / P2 fixed: still `No-Go`.
- 50k teacher: still `No-Go`.
- T1 training: still `No-Go`.
- Next step: runtime preflight for the explicit both-seat validation profile,
  followed by a production-readiness review. Do not promote full replacement.

## Both-Seat Stage9f Runtime Preflight

The C4 candidate was wired into an explicit validation AI profile:
`stage9f_cse2_csemax2_bothseat`.

Preflight config:

- `configs/hu_turn2_stage9f_cse2_csemax2_bothseat_preflight.json`

Profile smoke:

```powershell
python -m ofc_regular.evaluate_matchups `
  --profile-a stage9f_cse2_csemax2_bothseat `
  --profile-b stage7_m5_r10 `
  --games 20 `
  --seed 2026067701 `
  --opening-lookahead-samples 1 `
  --prediction-threads 1 `
  --trace-limit 0 `
  --topk-decision-output outputs/evals/hu_turn2_stage9f_bothseat_profile_preflight_smoke20/topk_decisions.jsonl `
  --output outputs/evals/hu_turn2_stage9f_bothseat_profile_preflight_smoke20/summary.json `
  --progress-every 10
```

Analyzer:

```powershell
python -m ofc_regular.analyze_hu_turn2_stage9f_profile_canary `
  --decisions outputs/evals/hu_turn2_stage9f_bothseat_profile_preflight_smoke20/topk_decisions.jsonl `
  --matchup-summary outputs/evals/hu_turn2_stage9f_bothseat_profile_preflight_smoke20/summary.json `
  --output-dir outputs/evals/hu_turn2_stage9f_bothseat_profile_preflight_smoke20/audit
```

Observed profile smoke:

- `20` paired seeds / `40` decisions completed,
- `topk_decisions_written = 40`,
- `runtime_profile = stage9f_cse2_csemax2_bothseat`,
- first/second decisions: `20 / 20`,
- override count: `1`, on second seat,
- realized per-fire delta: `+2.0`,
- non-fired nonzero realized deltas: `0`,
- replay-ready: `40 / 40`,
- missing replay fields: `0`,
- p95 runtime latency: `470.90ms`.

Missing-model fallback smoke:

```powershell
python -m ofc_regular.evaluate_hu_turn2_stage8b_topk_mc_rerank `
  --games-per-seed 3 `
  --seeds 2026067702 `
  --configs "k3/mc16/d0/se0/confirm32/cse2/csemax2/fcd0/scd0/pd0/seat=first+second/bygate_delta" `
  --t3-continuation stage7_m5_r10 `
  --hu-turn2-stage8b-model models/does_not_exist_stage9f_bothseat.pt `
  --allow-missing-stage8b-model-fallback `
  --output-dir outputs/evals/hu_turn2_stage9f_bothseat_preflight_missing_model_fallback_smoke `
  --write-decision-log `
  --device cpu `
  --prediction-threads 1
```

Observed fallback smoke:

- command completed,
- `stage8b_model_loaded = False`,
- `stage8b_model_load_failed = True`,
- decision count: `6`,
- override count: `0`,
- `no_override_reason = model_load_failed` for all `6` decisions,
- allowed seats were `first+second`,
- non-fired cancellation stayed clean.

Decision:

- Both-seat runtime preflight: `Pass`.
- Runtime profile remains validation-only.
- Production / P2 fixed: still `No-Go`.
- 50k teacher: still `No-Go`.
- T1 training: still `No-Go`.
- Next step: production-readiness review and rollout plan, including explicit
  off preset rollback and canary monitoring thresholds.
- Production-readiness review:
  `docs/hu_turn2_stage9f_production_readiness_review.md`.

## Both-Seat Stage9f Limited Profile Canary

After the runtime preflight smoke, the explicit validation profile was run
through the normal matchup path on non-overlapping local seeds.

Initial 1000-paired run:

```powershell
python -m ofc_regular.evaluate_matchups `
  --profile-a stage9f_cse2_csemax2_bothseat `
  --profile-b stage7_m5_r10 `
  --games 1000 `
  --seed 2026067801 `
  --opening-lookahead-samples 1 `
  --prediction-threads 1 `
  --trace-limit 0 `
  --topk-decision-output outputs/evals/hu_turn2_stage9f_bothseat_profile_limited_canary1000/topk_decisions.jsonl `
  --output outputs/evals/hu_turn2_stage9f_bothseat_profile_limited_canary1000/summary.json `
  --progress-every 250
```

Observed 1000-paired result:

- `1000` paired seeds / `2000` decisions,
- `20` realized overrides,
- whole-hand EV/hand `+0.04234`, 95% CI `[+0.00188, +0.08280]`,
- realized per-fire delta `+4.23405`,
- realized per-fire CI `[+0.53735, +7.93075]`,
- non-fired nonzero realized deltas `0`,
- replay-ready `2000 / 2000`,
- p95 latency `618.77ms`.

Follow-up 2500-paired run:

```powershell
python -u -m ofc_regular.evaluate_matchups `
  --profile-a stage9f_cse2_csemax2_bothseat `
  --profile-b stage7_m5_r10 `
  --games 2500 `
  --seed 2026067901 `
  --opening-lookahead-samples 1 `
  --prediction-threads 1 `
  --trace-limit 0 `
  --topk-decision-output outputs/evals/hu_turn2_stage9f_bothseat_profile_limited_canary2500/topk_decisions.jsonl `
  --output outputs/evals/hu_turn2_stage9f_bothseat_profile_limited_canary2500/summary.json `
  --progress-every 500
```

Analyzer:

```powershell
python -m ofc_regular.analyze_hu_turn2_stage9f_profile_canary `
  --decisions outputs/evals/hu_turn2_stage9f_bothseat_profile_limited_canary2500/topk_decisions.jsonl `
  --matchup-summary outputs/evals/hu_turn2_stage9f_bothseat_profile_limited_canary2500/summary.json `
  --output-dir outputs/evals/hu_turn2_stage9f_bothseat_profile_limited_canary2500/audit
```

Observed 2500-paired result:

- `2500` paired seeds / `5000` decisions,
- `52` realized overrides,
- override rate `1.04%`,
- whole-hand EV/hand `+0.03767`,
- whole-hand 95% CI `[+0.00967, +0.06567]`,
- realized per-fire delta `+3.62235`,
- realized per-fire CI `[+1.08872, +6.15598]`,
- positive / negative / zero fired deltas: `14 / 2 / 36`,
- p95 / max fired loss: `0.0 / 12.0`,
- non-fired nonzero realized deltas `0`,
- replay-ready `5000 / 5000`,
- p95 latency `566.27ms`.

Seat split:

- first seat:
  `30` fires, per-fire `+2.31513`,
  CI `[+0.09561, +4.53466]`, p95/max loss `0.0 / 0.0`.
- second seat:
  `22` fires, per-fire `+5.40491`,
  CI `[+0.25089, +10.55894]`, p95/max loss `8.0 / 12.0`.

Interpretation:

- The explicit profile path reproduces the positive C4 signal on a limited
  canary with at least `50` fired decisions.
- Non-fired cancellation and replay readiness remain clean.
- First and second seats both remain positive, but second-seat fired losses
  are still the main monitoring item.
- This supports limited canary readiness. It still does not authorize production
  default, P2 fixed status, 50k teacher generation, or T1 training.

Decision:

- Stage9f both-seat limited profile canary: `Pass`.
- `stage9f_cse2_csemax2_bothseat`: limited-canary candidate.
- Production / P2 fixed: still `No-Go`.
- 50k teacher: still `No-Go`.
- T1 training: still `No-Go`.

## Both-Seat Stage9f Production-Like Canary and Guarded Preset

The production-like GCP profile canary completed under the explicit
`stage9f_cse2_csemax2_bothseat` validation profile.

Artifact:

```text
outputs/evals/hu_turn2_stage9f_bothseat_profile_production_like_canary30000/
```

Observed result:

- `30000` paired seeds / `60000` decisions,
- `547` realized fires,
- override rate `0.918%`,
- estimated EV/decision `+0.02386`,
- realized per-fire delta `+2.61699`,
- realized per-fire CI `[+1.87247, +3.36150]`,
- first-seat CI low `+1.58773`,
- second-seat CI low `+1.53843`,
- non-fired nonzero realized deltas `0`,
- replay-ready `60000 / 60000`,
- p95 / max fired loss `7.0 / 31.2270`,
- p95 runtime latency `895.28ms`.
- latency p95 components: Stage A `406.10ms`, confirm `440.55ms`,
  overhead `24.80ms`.

Interpretation:

- The production-like canary clears the promotion-review evidence bar.
- Both seats are positive with positive per-seat CI lower bounds.
- Non-fired cancellation and replay readiness remain clean.
- Latency is the remaining warning: p95 is above the `800ms` warning threshold
  but below the `1500ms` rollback threshold.

Guarded preset:

```text
configs/hu_turn2_stage9f_cse2_csemax2_bothseat_guarded_production_preset.json
```

The guarded preset is ready but not enabled by default:

- `production_default=false`
- `production_p2_fixed=false`
- `requires_explicit_enable=true`
- full replacement disabled
- rollback preset `stage9f_off`

Verifier:

```powershell
python -m ofc_regular.verify_hu_turn2_stage9f_guarded_preset `
  --output-dir outputs/evals/hu_turn2_stage9f_bothseat_profile_production_like_canary30000/guarded_preset_verification
```

Observed verifier result:

- status `pass`,
- promotion decisions `60000`,
- realized fires `547`,
- aggregate and first/second per-fire CI lower bounds positive,
- replay-ready `60000 / 60000`,
- non-fired nonzero deltas `0`,
- latency p95 `895.28ms` warning, below rollback.
- latency warning is driven by MC work, not fixed runtime overhead.

Decision:

- Guarded production preset: `Ready`.
- Production default: still `No-Go` until explicitly enabled.
- P2 fixed: still `No-Go` until explicitly accepted.
- 50k teacher: still `No-Go`.
- T1 training: still `No-Go`.
