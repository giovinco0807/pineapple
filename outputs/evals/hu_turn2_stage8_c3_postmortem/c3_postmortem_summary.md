# HU Turn2 Stage8 C3 No-Go Postmortem

C3 is a decision No-Go. This postmortem is diagnostic only and does not authorize production, 50k teacher, T1, or P2 fixation.

- T3 continuation: `Stage7_candidate_A_m5_r10` fixed
- Stage8 mode: `HU T2 selective override`, not full replacement
- runtime decisions read: `40000`
- output directory: `outputs\evals\hu_turn2_stage8_c3_postmortem`

## C3 Result

| config | EV/hand | CI low | CI high | overrides | override rate | teacher avg gain | FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| m2.5_r0_g0.7 | -0.0069 | -0.0299 | 0.0161 | 81 | 0.0081 | 4.1743 | 0.0556 |
| m2.5_r0_g0.9 | -0.0058 | -0.0315 | 0.0199 | 45 | 0.0045 | 4.3530 | 0.0496 |
| m2.5_r0_g0.925 | -0.0058 | -0.0315 | 0.0199 | 40 | 0.0040 | 4.3694 | 0.0500 |
| m2.75_r0_g0.9 | -0.0058 | -0.0315 | 0.0199 | 42 | 0.0042 | 4.6650 | 0.0495 |

## Interpretation

The C1f teacher oracle LCB filters were strong on teacher EV, but the C3 runtime proxy gate did not reproduce that signal in larger seat-swap. The current threshold family also underfires: the best candidate overrides less than 1% of T2 decisions, so even good individual teacher gains have little aggregate EV impact.

## Produced Artifacts

- `c3_fired_overlap.csv`
- `c3_no_override_reason.csv`
- `c3_fired_state_quality.csv`
- `c3_top_loss_audit_states.jsonl`
- `oracle_vs_proxy_gap.csv`
- `safe_override_label_design.md`
- `stage8b_training_plan.md`
- `recommended_next_step.md`
