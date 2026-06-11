# HU Turn2 Stage8 Reference Index

This repository snapshot is intended to make the Regular OFC HU Turn2 Stage8
work reviewable from GitHub.

## Included in Git

- Source code, tests, scripts, Rust encoder code, configs, and docs.
- Stage7 production selective override configs and rollout note.
- HU Turn2 C1e teacher-EV cache audit artifacts:
  - `outputs/hu_turn2_stage1_pilot_training_c1e_teacher_ev_feature_cache/`
- HU Turn2 C1f expanded calibration artifacts:
  - `outputs/hu_turn2_stage1_pilot_training_c1f_expanded_calibration/`
- HU Turn2 C2-small heldout / seat-swap artifacts:
  - `outputs/evals/hu_turn2_stage8_c2_small/`

## Intentionally Not Included

Large generated raw data and local build products remain ignored:

- `outputs/hu_turn2_stage8_20k_mc512/` (~7.5 GB locally)
- `models/`
- `target/`
- `.pytest_cache/`

The large 20k MC512 teacher output is excluded because it is too large for
normal GitHub review. The included C1e cache contains the teacher-EV fields
needed for C1f/C2 gate inspection.

## Current Gate State

- C1e teacher-EV feature cache: hard pass.
- C1f expanded calibration: pass.
- C2-small heldout / seat-swap: pass for moving to C3, not production.
- 50k teacher: no-go for now.
- T1 training: no-go for now.
- Production T2 runtime: no-go for now.

## Fixed T3 Continuation

T2 evaluation uses the fixed Stage7 Candidate A continuation policy:

- Stage7 model: `models/hu_turn3_stage7_reference_override_cached_rank_wide.pt`
- `hu_turn3_min_margin = 5.0`
- `hu_turn3_reference_min_margin = 10.0`
- Stage3 HU margin10 fallback remains the default.
- Stage7 is a HU T3 selective override only, not a full replacement.
