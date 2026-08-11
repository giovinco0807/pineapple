# HU T3 Stage9d Promotion Candidate Rollout Note

## Status

Stage9d second3seed accept-gated runtime is the current HU T3 default profile,
not a full replacement. Stage7 m5/r10 remains the explicit rollback profile.

## Baseline Rollback

- profile: `stage7_m5_r10`
- config: `configs/hu_turn3_stage7_m5_r10_production.json`
- model: `models/hu_turn3_stage7_reference_override_cached_rank_wide.pt`
- `hu_turn3_min_margin = 5.0`
- `hu_turn3_reference_min_margin = 10.0`

## Candidate Runtime

- profile: `current`
- alias profile: `stage9d_p07_relaxed_both`
- config: `configs/hu_turn3_stage9d_second3seed_gate_p07_relaxed_both_promotion_candidate.json`
- canary matrix: `configs/hu_turn3_stage9d_canary_presets.json`
- HU T3 model: `models/hu_turn3_joint_exact_stage9_mixed1200_mc128_extra_trees.pkl`
- support model: `models/hu_turn3_stage9d_mixed_runtime_hn245_mc128_extra_trees.pkl`
- gate model: `models/hu_turn3_stage9_accept_gate_rf_265fired_runtimepred_second3seed.pkl`
- reference model: `models/hu_turn3_stage3_mc32_500k_plus_m8_12_f128_100k_w2_cached_rank_wide.pt`
- `hu_turn3_min_margin = 0.5`
- `hu_turn3_reference_min_margin = 0.0`
- `hu_turn3_min_support_margin = 0.5`
- `hu_turn3_min_model_score = 5.0`
- `hu_turn3_min_gate_probability = 0.7`
- both seats enabled

If the candidate primary model, support model, gate model, or reference model is
not available through the named profile loader, `current` falls back to Stage7
m5/r10 rather than running a partial Stage9d policy.

## Evidence

Fresh non-overlapping 10-seed paired seat-swap:

- artifact: `outputs/evals/hu_turn3_stage9d_accept_gate_second3seed_t07_relaxed_both_fresh/fresh10_summary.md`
- paired seeds: `10000`
- hands: `20000`
- EV/hand: `+0.031500`
- 95% CI: `[+0.022321, +0.040679]`
- fires: `95`
- first/second fires: `55/40`
- all ten seed means were positive

Fresh10 fired-state MC512 audit:

- artifact: `outputs/evals/hu_turn3_stage9d_second3seed_fresh10_fire_replay_mc512/summary.md`
- fired states: `95`
- mean delta: `+4.525119`
- 95% CI: `[+3.549927, +5.500311]`
- positive/negative/gray: `85/2/8`
- worst MC512 delta: `-2.703103`

## Evaluation Command

```powershell
python -m ofc_regular.evaluate_matchups `
  --profile-a current `
  --profile-b stage7_m5_r10 `
  --games 1000 `
  --seed 2026066201 `
  --prediction-threads 1 `
  --output outputs/evals/hu_turn3_stage9d_profile_smoke/summary.json `
  --trace-output outputs/evals/hu_turn3_stage9d_profile_smoke/trace.jsonl `
  --trace-limit 20
```

Use `--profile-a stage7_m5_r10 --profile-b current` as the reverse spot check
when validating a rollout.

## Rollback

Use `profile=stage7_m5_r10` or `configs/hu_turn3_stage7_m5_r10_production.json`.
Do not delete the Stage7 model or Stage7 config while Stage9d is being canaried.

## Notes

- Stage9d is still selective override only.
- Do not use the older both-seat p0.7 gate without the second3seed gate; that
  path had a heavy realized tail.
- Teacher/regret evidence is secondary. Fresh paired seat-swap and fired MC512
  replay are the adoption evidence.
