# HU T2 Stage9f P2 Acceptance Review

Status: P2 accepted for post-acceptance experiments; current and production
default not changed.

This review covers the Regular OFC HU T2 selective override candidate:

- evidence profile: `stage9f_cse2_csemax2_bothseat`
- fixed P2 profile: `stage9f_p2`
- model:
  `models/hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt`
- runtime:
  `k3/mc16/d0/se0/confirm32/cse2/csemax2/fcd0/scd0/pd0/seat=first+second/bygate_delta`
- T3 continuation: `stage7_m5_r10`
- rollback: `stage9f_off`

Stage9f is selective override only. It is not a full replacement T2 policy.
The `stage9f_p2` profile is the accepted both-seat T2 P2 runtime for
post-acceptance experiments. It uses `runtime_status=p2_fixed` and keeps T3
continuation fixed to `stage7_m5_r10`.

## Evidence

### C4 Validation

Artifact:
`outputs/evals/hu_turn2_stage9f_bothseat_c4_chunked_40x12000_m5r10/`

- paired hands: `218116`
- realized fires: `4000`
- estimated EV/hand: `+0.02580`
- per-fire delta: `+2.79986`
- per-fire 95% CI: `[+2.51686, +3.08286]`
- first / second per-fire delta: `+2.72177 / +2.87458`
- non-fired cancellation: clean
- p95 / max fired loss: `7.0 / 37.2270`

### Production-Like Canary

Artifact:
`outputs/evals/hu_turn2_stage9f_bothseat_profile_production_like_canary30000/`

- decisions: `60000`
- realized fires: `547`
- override rate: `0.918%`
- estimated EV/decision: `+0.02386`
- per-fire delta: `+2.61699`
- per-fire 95% CI: `[+1.87247, +3.36150]`
- first / second CI low: `+1.58773 / +1.53843`
- non-fired cancellation: clean
- replay-ready rows: `60000 / 60000`
- p95 / max fired loss: `7.0 / 31.2270`
- p95 latency: `895.28ms`

### Guarded Canary

Artifact:
`outputs/evals/hu_turn2_stage9f_guarded_production_canary/`

- verifier: `pass`
- paired seeds / hands / decisions: `10000 / 20000 / 20000`
- realized fires: `174`
- override rate: `0.87%`
- estimated EV/decision: `+0.02892`
- per-fire delta: `+3.32417`
- per-fire 95% CI: `[+1.87929, +4.76905]`
- first / second CI low: `+0.93869 / +1.64419`
- non-fired cancellation: clean
- replay-ready rows: `20000 / 20000`
- p95 / p99 / max fired loss: `6.0 / 18.2270 / 30.2270`
- p95 latency: `907.40ms`
- latency p95 components:
  Stage A `414.64ms`, confirm `451.19ms`, overhead `25.15ms`

## Acceptance Decision

Stage9f is available as the fixed T2 P2 profile via `stage9f_p2`.

Reasons:

- It is positive in C4 validation, production-like profile canary, and the
  guarded canary.
- Aggregate, first-seat, and second-seat per-fire CI lows are positive.
- Non-fired cancellation is clean.
- Replay logging is complete.
- Tail loss stays below rollback thresholds.
- Runtime p95 latency is above the warning threshold but below rollback, and
  overhead is small. The latency is dominated by Stage A and confirm MC, which
  is expected for this runtime.

## Applied Scope

Accepted:

- fixed T2 P2 experiment profile: `stage9f_p2`
- T3 continuation for this P2 profile: `stage7_m5_r10`
- selective override only; no full replacement

Still not applied:

- production default: not enabled
- `current` profile replacement: not enabled
- 50k teacher: blocked unless more data is explicitly needed after P2
- full replacement: blocked

`stage9f_p2` is intentionally separate from `current`, because the acceptance
evidence used Stage9f T2 with T3 continuation `stage7_m5_r10`. Replacing
`current` would also change the existing Stage9d T3 default path and should be
handled as a separate, explicit rollout decision.

## Production Enable Preview

P2 acceptance does not flip production. If a separate production-default enable
is approved, the minimal runtime direction is:

- file: `configs/hu_turn2_stage9f_canary_presets.json`
- preset: `stage9f_cse2_csemax2_bothseat`
- set `production_default=true`
- keep `validation_default=true`
- keep full replacement disabled
- keep rollback preset `stage9f_off`

Do not change the Stage9f runtime string without revalidating quality and
latency. Reduced-MC latency variants were smoke-tested and were not promoted.

## Rollback

Rollback is immediate:

- use preset `stage9f_off`
- or omit the Stage9f TopK config and use baseline Turn2

Rollback triggers remain:

- missing replay fields
- non-fired nonzero realized deltas
- aggregate or seat-level per-fire CI low <= 0
- p95 fired loss > `12`
- max fired loss >= `40`
- p95 latency >= `1500ms`
- illegal candidate override
- NaN/inf prediction
- model load failure outside explicit fallback smoke

## Next Step

Use `stage9f_p2` for post-acceptance T2/T1 experiments. The existing T1
leaf31 TopK line remains No-Go based on its fire-count diagnostic, so the next
T1 work should use a revised objective/model/gate rather than extending that
line. Production default or `current` replacement still requires a separate
intentional change and a short post-enable smoke.
