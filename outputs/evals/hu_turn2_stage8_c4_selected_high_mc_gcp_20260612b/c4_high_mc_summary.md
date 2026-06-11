# HU Turn2 Stage8 C4 Selected High-MC Replay

C4 is analysis-only. It does not authorize production training, T1 training, or a 50k teacher run.

- run_name: `regular-hu-t2-c4-highmc-20260612b`
- mc_samples: `4096`
- target_replay_states: `50`
- high-MC successes: `50`
- high-MC failures: `0`
- positive candidate-vs-baseline delta: `44`
- lower95 positive candidate-vs-baseline delta: `41`
- false_positive_gate: `1`

## Diagnosis Counts

- `reference_margin_bad_gate`: `21`
- `underfire`: `10`
- `model_ranking_error`: `9`
- `needs_more_samples`: `8`
- `false_positive_gate`: `1`
- `low_margin_noise`: `1`

## Decision

- C4 selected high-MC replay: `No-Go`
- production training: `No-Go`
- T1 training: `No-Go`
- 50k teacher: `No-Go`
- C2/C3 threshold is still evaluation-only until C4 and another larger seat-swap validation are clean.
