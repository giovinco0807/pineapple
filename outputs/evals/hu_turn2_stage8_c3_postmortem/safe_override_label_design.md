# Stage8b Safe Override Label Design

C1f showed that teacher-EV LCB filters are strong, but C3 showed the current runtime proxy does not reproduce that advantage in seat-swap EV. Stage8b should learn a deployable confidence target instead of using teacher LCB directly at runtime.

## Labels

- `teacher_delta_lcb_196 = teacher_delta_candidate_vs_baseline - 1.96 * SE_delta_candidate_vs_baseline`
- `teacher_delta_lcb_164 = teacher_delta_candidate_vs_baseline - 1.64 * SE_delta_candidate_vs_baseline`
- `safe_lcb196_label = 1` when `teacher_delta_lcb_196 > 0`, otherwise 0 for confident negatives and low-weight gray for uncertain rows
- `safe_lcb164_label = 1` when `teacher_delta_lcb_164 > 0`, otherwise 0 for confident negatives and low-weight gray for uncertain rows
- `hard_negative_label = 1` when current runtime proxy confidence is high but teacher or high-MC delta is `<= 0`
- `gray_label = 1` for near-zero or high-SE rows that should receive low gate-loss weight

## Runtime Safety

Teacher EV, teacher gain LCB, MC512 EV per action, and actual bucket labels remain analysis-only. Production runtime should use only model outputs and runtime-observable state/action fields.

## Required Cache Columns

- `teacher_delta_candidate_vs_baseline`
- `SE_delta_candidate_vs_baseline`
- `teacher_delta_lcb_196`
- `teacher_delta_lcb_164`
- `safe_lcb196_label`
- `safe_lcb164_label`
- `hard_negative_label`
- `gray_label`
- `hard_negative_source`

## Selected High-MC Use

MC4096/8192 should be used to upgrade labels for C3 fired losses, near-threshold rows, oracle-positive/proxy-missed rows, and high-confidence proxy false positives. It is diagnostic and training-label work, not production approval.
