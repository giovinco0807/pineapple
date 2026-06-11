# Stage7 Candidate A Production Rollout

Stage7_candidate_A is enabled only as a HU T3 selective override.

Default runtime:

- `hu_turn3_stage7_enabled = true`
- `hu_turn3_min_margin = 5.0`
- `hu_turn3_reference_min_margin = 10.0`

Stage3 remains the fallback/default policy. Stage7 full replacement is not enabled.

`margin0.25` is explicitly excluded from production because prior validation was negative.

If Stage7 fails any runtime safety check, action falls back to Stage3:

- model load failure
- feature generation failure
- illegal candidate action
- NaN/inf prediction
- support/gate/self-regret fallback if those gates are enabled

## Runtime Presets

- Production default: `configs/hu_turn3_stage7_m5_r10_production.json`
- Canary matrix: `configs/hu_turn3_stage7_canary_presets.json`
- Rollback/off: `configs/hu_turn3_stage7_off.json`

Production default is `m5_r10`. `m4_r10` is canary-only. `m3_r10` and `m3_r12` are experiment-only.

## Golden Eval Command

```powershell
.\scripts\Run-Stage7ProductionGoldenEval.ps1 -GamesPerSeed 1000 -PredictionThreads 1
```

Golden seeds:

- `2026060904`
- `2026063501`
- `2026068201`

Expected direction: `m5_r10` should stay positive versus Stage3, with override rate and false-positive/tail-loss metrics close to the shortlist1000 evaluation artifacts.

## Rollback

Use the `stage7_off` profile/config, or pass `--disable-hu-turn3-stage7` for CLI smoke runs. For explicit model-set eval, pass `--disable-hu-turn3-stage7-a` and/or omit the `--hu-turn3-a` model path.
