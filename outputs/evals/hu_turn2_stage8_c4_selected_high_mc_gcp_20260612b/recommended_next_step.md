# Recommended Next Step

- T2 Stage8 production: `No-Go`
- P2 fixation: `No-Go`
- 50k teacher: `defer`
- T1: `defer`
- high-MC successes: `50`
- high-MC failures: `0`
- max_replay_states setting: `50`

## Diagnosis Counts

- `reference_margin_bad_gate`: `21`
- `underfire`: `10`
- `model_ranking_error`: `9`
- `needs_more_samples`: `8`
- `false_positive_gate`: `1`
- `low_margin_noise`: `1`

## Decision

Primary blocker looks like underfire. Re-sweep thresholds on high-MC labels, then run a larger seat-swap validation.

## Next Command

```powershell
python -m ofc_regular.audit_hu_turn2_stage8_high_mc `
  --mc-samples 4096 `
  --max-replay-states 50 `
  --selection-strategy stratified `
  --include-fired 15 `
  --include-suspected-false-positive 15 `
  --include-missed-positive 15 `
  --include-near-fired 5 `
  --prediction-threads 1
```
