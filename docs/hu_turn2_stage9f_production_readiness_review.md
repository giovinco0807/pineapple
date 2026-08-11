# HU T2 Stage9f Production-Readiness Review

Status: guarded production preset ready, not enabled by default.

This review covers only the Regular OFC HU T2 Stage9f selective override
candidate:

- evidence profile: `stage9f_cse2_csemax2_bothseat`
- fixed P2 profile: `stage9f_p2`
- model:
  `models/hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt`
- runtime:
  `k3/mc16/d0/se0/confirm32/cse2/csemax2/fcd0/scd0/pd0/seat=first+second/bygate_delta`
- T3 continuation: `stage7_m5_r10`

Stage9f is not a full replacement policy. Baseline T2 remains the default and
fallback action unless the selective override passes the TopK and independent
confirm-MC gates.
The `stage9f_p2` profile exposes the accepted T2 runtime as a deliberate
selection path with `runtime_status=p2_fixed`; it does not replace `current` and
does not flip the production default preset.

## Evidence

### C4 Both-Seat Validation

Artifact:
`outputs/evals/hu_turn2_stage9f_bothseat_c4_chunked_40x12000_m5r10/`

- shards: `40 / 40`
- paired hands: `218116`
- realized fires: `4000`
- estimated EV/hand: `+0.02580`
- seed-mean EV/hand: `+0.02614`
- seed-mean 95% CI: `[+0.02355, +0.02874]`
- realized per-fire delta: `+2.79986`
- realized per-fire 95% CI: `[+2.51686, +3.08286]`
- non-fired cancellation: clean
- first-seat per-fire delta: `+2.72177`
- second-seat per-fire delta: `+2.87458`
- p95 / max fired loss: `7.0 / 37.2270`

Interpretation: the validation signal is strong and reproduced at larger scale.
Both positions are positive. The remaining blocker is not average EV; it is
runtime rollout discipline and tail-loss containment.

### Runtime Profile Preflight

Artifact:
`outputs/evals/hu_turn2_stage9f_bothseat_profile_preflight_smoke20/`

- profile route completed with `stage9f_cse2_csemax2_bothseat`
- `40` decisions logged
- first/second decisions: `20 / 20`
- replay-ready rows: `40 / 40`
- missing replay fields: `0`
- non-fired cancellation: clean
- p95 runtime latency: `470.90ms`

Artifact:
`outputs/evals/hu_turn2_stage9f_bothseat_preflight_missing_model_fallback_smoke/`

- missing Stage8b model fallback completed with explicit fallback flag
- model loaded: `False`
- model load failed: `True`
- decisions: `6`
- overrides: `0`
- no-override reason: `model_load_failed` for all decisions

Interpretation: the explicit validation profile is wired, both seats are enabled,
runtime logs preserve replay data, and the fallback path does not fire unsafe
overrides when the candidate model is missing.

### Limited Profile Canary

Artifacts:

- `outputs/evals/hu_turn2_stage9f_bothseat_profile_limited_canary1000/`
- `outputs/evals/hu_turn2_stage9f_bothseat_profile_limited_canary2500/`

The 2500-paired canary is the current profile-path monitoring evidence:

- paired seeds / decisions: `2500 / 5000`
- realized fires: `52`
- override rate: `1.04%`
- whole-hand EV/hand: `+0.03767`
- whole-hand 95% CI: `[+0.00967, +0.06567]`
- realized per-fire delta: `+3.62235`
- realized per-fire 95% CI: `[+1.08872, +6.15598]`
- positive / negative / zero fired deltas: `14 / 2 / 36`
- p95 / max fired loss: `0.0 / 12.0`
- non-fired cancellation: clean
- replay-ready rows: `5000 / 5000`
- p95 runtime latency: `566.27ms`

Seat split:

- first seat:
  `30` fires, per-fire `+2.31513`, CI `[+0.09561, +4.53466]`,
  p95/max loss `0.0 / 0.0`.
- second seat:
  `22` fires, per-fire `+5.40491`, CI `[+0.25089, +10.55894]`,
  p95/max loss `8.0 / 12.0`.

Interpretation: the normal profile path reproduces the C4 signal on a limited
canary and reaches the minimum `50` fired-decision evidence target. Second-seat
tail losses remain the main monitoring item.

### Production-Like Profile Canary

Artifact:
`outputs/evals/hu_turn2_stage9f_bothseat_profile_production_like_canary30000/`

- paired seeds / decisions: `30000 / 60000`
- realized fires: `547`
- override rate: `0.918%`
- realized per-fire delta: `+2.61699`
- realized per-fire 95% CI: `[+1.87247, +3.36150]`
- estimated EV/decision: `+0.02386`
- non-fired cancellation: clean
- replay-ready rows: `60000 / 60000`
- p95 / max fired loss: `7.0 / 31.2270`
- p95 runtime latency: `895.28ms`
- latency p95 components:
  Stage A `406.10ms`, confirm `440.55ms`, overhead `24.80ms`

Seat split:

- first seat:
  `280` realized fires, per-fire `+2.58359`,
  CI `[+1.58773, +3.57945]`, p95/max loss `5.0 / 31.2270`.
- second seat:
  `267` realized fires, per-fire `+2.65201`,
  CI `[+1.53843, +3.76559]`, p95/max loss `8.0 / 29.2270`.

Interpretation: this run clears the promotion-review evidence bar:
more than `500` realized fires, positive aggregate and per-seat per-fire lower
bounds, clean non-fired cancellation, and complete replay-ready logging. The
only warning is latency: p95 is above the `800ms` warning threshold but below
the `1500ms` rollback threshold.

## Decision

- C4 validation: Go.
- Runtime preflight: Go.
- Limited canary profile: Go.
- Limited canary observed positive: Go.
- Production-like canary: Go.
- Promotion review: Go.
- Guarded production preset: Ready.
- Production default: No-Go until explicitly flipped.
- P2 fixed profile: available as `stage9f_p2`.
- `current` replacement: No-Go until explicitly accepted.
- Full replacement: No-Go.
- 50k teacher generation: No-Go.
- T1 training: No-Go.

## Guarded Production Preset

The guarded opt-in preset is:

```text
configs/hu_turn2_stage9f_cse2_csemax2_bothseat_guarded_production_preset.json
```

It records the promotion evidence and runtime safety thresholds but still keeps:

- `production_default=false`
- `production_p2_fixed=false`
- `requires_explicit_enable=true`
- full replacement disabled
- rollback to `stage9f_off`

This means Stage9f can be selected deliberately for a guarded production canary,
but it has not been made the automatic default policy.

Guarded canary plan:

```text
configs/hu_turn2_stage9f_guarded_production_canary_plan.json
```

Recommended guarded canary scale:

- `10000` paired seeds / `20000` decisions
- `10` shards of `1000` paired seeds
- profile A `stage9f_cse2_csemax2_bothseat`
- profile B `stage7_m5_r10`
- rollback preset `stage9f_off`

Observed dry run:

- run name:
  `regular-hu-t2-stage9f-guarded-production-canary-dryrun-20260623-001`
- execution: `dry_run`
- create instances: `false`
- project / bucket:
  `ofc-solver-485418` / `pokerhu-ofc-solver-485418-training`
- machine type / VM count: `e2-highcpu-4` / `10`
- total games / hands / decisions: `10000 / 20000 / 20000`
- shard games / shards: `1000 / 10`
- base seed / stride: `2026069001 / 1000000`
- shard seeds:
  `2026069001`, `2027069001`, `2028069001`, `2029069001`,
  `2030069001`, `2031069001`, `2032069001`, `2033069001`,
  `2034069001`, `2035069001`

Dry-run interpretation: the GCP canary launcher resolves the intended project,
bucket, shard layout, profiles, and seed spacing without creating instances.
The actual canary is ready to start only after explicit approval.

Active GCP run:

- run name:
  `regular-hu-t2-stage9f-guarded-production-canary-20260623-001`
- started at: `2026-06-23T07:14:22Z`
- profile A / B:
  `stage9f_cse2_csemax2_bothseat` / `stage7_m5_r10`
- total games / hands / decisions: `10000 / 20000 / 20000`
- shards / VM count: `10 / 10`
- machine type: `e2-highcpu-4`
- initial status: all `10` VM instances were `RUNNING`
- representative serial log:
  setup completed, `rust_direct_available=true`, and the shard loop started

This run is still a guarded canary. It does not change production default,
does not fix P2, and does not unblock 50k teacher generation or T1 training by
itself.

Observed guarded canary result:

- status: `pass`
- completed shards: `10 / 10`
- Spot retries: shards `5`, `3`, and `4` were restarted and completed
- running instances after completion: `0`
- paired seeds / hands / decisions: `10000 / 20000 / 20000`
- realized fires: `174`
- override rate: `0.87%`
- estimated EV/decision: `+0.02892`
- realized per-fire delta: `+3.32417`
- realized per-fire 95% CI: `[+1.87929, +4.76905]`
- first-seat per-fire CI low: `+0.93869`
- second-seat per-fire CI low: `+1.64419`
- non-fired cancellation: clean (`0` non-fired nonzero deltas)
- replay-ready rows: `20000 / 20000`
- p95 / p99 / max fired loss: `6.0 / 18.2270 / 30.2270`
- p95 runtime latency: `907.40ms`
- latency p95 components:
  Stage A `414.64ms`, confirm `451.19ms`, overhead `25.15ms`

Guarded canary verifier:

- status: `pass`
- output:
  `outputs/evals/hu_turn2_stage9f_guarded_production_canary/guarded_preset_verification/`
- warning:
  p95 latency exceeds the `800ms` warning threshold but remains below the
  `1500ms` rollback threshold

Decision after canary:

- guarded production preset: `Ready`
- production default: `No-Go until explicitly enabled`
- P2 fixed: `No-Go until explicitly accepted`
- 50k teacher: `No-Go`
- T1 training: `No-Go`

P2 acceptance review:

- `docs/hu_turn2_stage9f_p2_acceptance_review.md`
- config artifact:
  `configs/hu_turn2_stage9f_p2_acceptance_candidate.json`
- status: P2 acceptance ready, not applied

Verification wrapper:

```powershell
.\scripts\Verify-HuTurn2Stage9fGuardedProductionCanary.ps1 `
  -OutputDir outputs/evals/hu_turn2_stage9f_guarded_production_canary
```

Verifier:

```powershell
python -m ofc_regular.verify_hu_turn2_stage9f_guarded_preset `
  --output-dir outputs/evals/hu_turn2_stage9f_bothseat_profile_production_like_canary30000/guarded_preset_verification
```

Observed verifier result:

- status: `pass`
- failures: `0`
- warning: p95 latency `895.28ms` exceeds the `800ms` warning threshold
  but remains below the `1500ms` rollback threshold
- warning detail: the p95 latency is dominated by Stage A and confirm MC
  (`406.10ms` and `440.55ms` respectively), not fixed overhead
- output:
  `outputs/evals/hu_turn2_stage9f_bothseat_profile_production_like_canary30000/guarded_preset_verification/`

The wrapper was also smoke-tested against the production-like canary output and
completed successfully.

## Limited Canary Requirements

Use only the explicit validation profile:

```text
stage9f_cse2_csemax2_bothseat
```

Rollback-safe rollout config:

```text
configs/hu_turn2_stage9f_cse2_csemax2_bothseat_limited_canary_rollout.json
```

Required runtime invariants:

- T3 continuation remains `stage7_m5_r10`.
- Stage9f is selective override only.
- Baseline T2 is fallback/default.
- Confirm-MC delta is gate diagnostic only.
- Performance reporting uses realized fired whole-game paired deltas.
- Runtime logs must include `dead_cards`, `visible_dead_cards`,
  private discards, baseline action, final action, `override_fired`,
  `no_override_reason`, confirm delta/SE, and realized delta when available.
- Non-fired cancellation must remain clean in evaluation.

Stop / rollback triggers:

- missing replay fields,
- non-fired nonzero deltas,
- model load failure outside explicit fallback smoke,
- illegal candidate overrides,
- NaN/inf predictions,
- material increase in p95 fired loss,
- repeated severe fired losses above the C4 max-loss band,
- first/second split turns negative in canary monitoring.

Rollback:

- use `stage9f_off` from
  `configs/hu_turn2_stage9f_canary_presets.json`,
- or do not pass a Stage9f TopK config and use baseline Turn2.

## Next Work

1. If explicitly approved, enable the guarded production preset for a narrow
   canary. Keep rollback to `stage9f_off`.
2. Continue monitoring
   first/second split, p95/max fired loss, missing replay fields, non-fired
   cancellation, and p95 latency.
3. Treat latency as the main remaining runtime warning. The overhead p95 is
   small (`24.80ms`); latency is driven by Stage A and confirm MC. Any further
   speed change must be treated as a quality/latency tradeoff and revalidated.
4. T2 is now accepted as `stage9f_p2` for post-acceptance experiments, but
   production default remains off.
5. Keep 50k teacher blocked unless a later experiment shows additional broad
   teacher data is needed.
6. Resume T1 only with a revised objective/model/gate. The existing leaf31
   TopK line is No-Go after its fire-count diagnostic.

## Latency Variant Probe

Config:

```text
configs/hu_turn2_stage9f_latency_variant_probe.json
```

Artifacts:

- `outputs/evals/hu_turn2_stage9f_latency_variant_smoke80/`
- `outputs/evals/hu_turn2_stage9f_latency_variant_smoke300/`

The reduced `mc12/confirm24` variant improved p95 latency in the 300-paired
smoke (`419.75ms` vs `538.34ms` for the current guarded config in the direct
TopK evaluator), but the fired signal was weaker:

- current `mc16/confirm32`: `9` fires, per-fire `+3.69`, EV/hand `+0.055`
- reduced `mc12/confirm24`: `5` fires, per-fire `-3.85`, EV/hand `-0.032`

This is not enough evidence to reject the reduced variant permanently, but it
is enough to avoid replacing the current guarded preset. If latency must be
reduced, run a larger non-overlapping quality/latency canary before changing the
production candidate.
