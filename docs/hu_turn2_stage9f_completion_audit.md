# HU T2 Stage9f Completion Audit

Date: 2026-06-25 JST

Scope: Regular OFC heads-up T2 selective override.

Conclusion: `stage9f_p2` is complete enough to be the fixed T2 P2 continuation
for downstream T1/T0 work. It is not a full replacement policy, and production
default remains a separate explicit enable decision.

## Accepted Runtime

- Fixed profile: `stage9f_p2`
- Evidence profile: `stage9f_cse2_csemax2_bothseat`
- Model:
  `models/hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt`
- Runtime:
  `k3/mc16/d0/se0/confirm32/cse2/csemax2/fcd0/scd0/pd0/seat=first+second/bygate_delta`
- T3 continuation: `stage7_m5_r10`
- Mode: selective override only
- Rollback: `stage9f_off`

## Completion Criteria

| Requirement | Evidence | Status |
|---|---|---|
| Model artifact exists | `models/hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt` | Pass |
| Explicit P2 profile exists | `stage9f_p2` in `src/ofc_regular/ai_profiles.py` with `runtime_status=p2_fixed` | Pass |
| Selective override, not full replacement | `full_replacement_enabled=false` in P2 and guarded preset configs | Pass |
| T3 continuation fixed | `stage7_m5_r10`, `hu_turn3_min_margin=5.0`, `hu_turn3_reference_min_margin=10.0` | Pass |
| Large independent validation positive | C4 per-fire CI low `+2.51686` over `4000` fires | Pass |
| Profile-path canary positive | production-like canary per-fire CI low `+1.87247` over `547` fires | Pass |
| Both seats positive | production-like first/second CI lows `+1.58773 / +1.53843` | Pass |
| Non-fired cancellation clean | C4, production-like canary, guarded canary, and smoke all have `0` non-fired nonzero deltas | Pass |
| Replay logging complete | production-like canary `60000 / 60000`, guarded canary `20000 / 20000` replay-ready | Pass |
| Tail loss below rollback | production-like p95/max loss `7.0 / 31.2270`, rollback max `<40` | Pass |
| Latency below rollback | p95 latency `895.28ms` and `907.40ms`, rollback threshold `1500ms` | Pass with warning |
| Missing model fallback safe | guarded readiness docs include missing-model fallback smoke with zero overrides | Pass |
| Current-code verifier passes | `verify_hu_turn2_stage9f_guarded_preset` status `pass` | Pass |
| Current-code regression tests pass | `42 passed` for Stage9f profile/preset/analyzer tests | Pass |
| Wiring smoke passes | `outputs/evals/hu_turn2_stage9f_p2_post_goal_smoke20/` | Pass |

## Primary Evidence

### C4 Both-Seat Validation

Artifact:
`outputs/evals/hu_turn2_stage9f_bothseat_c4_chunked_40x12000_m5r10/`

- paired hands: `218116`
- realized fires: `4000`
- estimated EV/hand: `+0.02580`
- realized per-fire delta: `+2.79986`
- realized per-fire 95% CI: `[+2.51686, +3.08286]`
- first / second per-fire delta: `+2.72177 / +2.87458`
- non-fired cancellation: clean
- p95 / max fired loss: `7.0 / 37.2270`

### Production-Like Profile Canary

Artifact:
`outputs/evals/hu_turn2_stage9f_bothseat_profile_production_like_canary30000/`

- decisions: `60000`
- realized fires: `547`
- override rate: `0.918%`
- estimated EV/decision: `+0.02386`
- realized per-fire delta: `+2.61699`
- realized per-fire 95% CI: `[+1.87247, +3.36150]`
- first / second per-fire CI lows: `+1.58773 / +1.53843`
- replay-ready rows: `60000 / 60000`
- non-fired cancellation: clean
- p95 / max fired loss: `7.0 / 31.2270`
- p95 latency: `895.28ms`

### Guarded Canary

Artifact:
`outputs/evals/hu_turn2_stage9f_guarded_production_canary/`

- verifier status: `pass`
- decisions: `20000`
- realized fires: `174`
- override rate: `0.87%`
- estimated EV/decision: `+0.02892`
- realized per-fire 95% CI: `[+1.87929, +4.76905]`
- first / second per-fire CI lows: `+0.93869 / +1.64419`
- replay-ready rows: `20000 / 20000`
- non-fired cancellation: clean
- p95 / p99 / max fired loss: `6.0 / 18.2270 / 30.2270`
- p95 latency: `907.40ms`

### Post-Goal Wiring Smoke

Artifact:
`outputs/evals/hu_turn2_stage9f_p2_post_goal_smoke20/`

- profile A/B: `stage9f_p2` / `stage7_m5_r10`
- paired seeds / hands / decisions: `20 / 40 / 40`
- overrides: `1`
- realized fired delta: `+8.0`
- non-fired nonzero realized deltas: `0`
- replay-ready rows: `40 / 40`
- p95 latency: `598.21ms`

## Verification Commands

```powershell
python -m ofc_regular.verify_hu_turn2_stage9f_guarded_preset `
  --output-dir outputs/evals/hu_turn2_stage9f_guarded_production_canary/guarded_preset_verification_goal_audit
```

Observed result:

- status: `pass`
- failures: `0`
- promotion decisions: `60000`
- realized fires: `547`
- replay-ready: `60000`
- non-fired nonzero count: `0`
- p95 latency warning remains, but below rollback

```powershell
python -m pytest `
  tests/test_ai_profiles.py `
  tests/test_hu_turn2_stage9f_runtime_presets.py `
  tests/test_aggregate_hu_turn2_stage9f_profile_canary.py `
  tests/test_hu_turn2_stage9f_profile_canary_analysis.py `
  tests/test_hu_turn2_stage9f_tail_loss_analysis.py `
  -q -p no:cacheprovider
```

Observed result:

- `42 passed`

## Decision

T2 is complete for the requested model-building objective as a fixed P2
continuation:

- Use `stage9f_p2` for downstream T1/T0 work.
- Keep `stage7_m5_r10` as the fixed T3 continuation.
- Do not use confirm-MC delta as the performance metric.
- Do not treat old Stage8/Stage8b direct runtime gates as production evidence.

Still separate from this completion:

- flipping production default,
- replacing `current`,
- enabling full replacement,
- generating 50k more T2 teacher data.

If production default is requested later, use the guarded preset and run a
post-enable smoke rather than changing the runtime string.
