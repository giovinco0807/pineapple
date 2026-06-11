# Gate C1e Go/No-Go For C1f

- C1e hard pass: `True`
- C1e strong pass: `False`
- blockers: `none`
- warnings: `unbiased_rows_lt_20k`
- C1f expanded calibration: `Go`
- C2-small heldout: `No-Go`
- 50k teacher: `No-Go`
- T1 training: `No-Go`
- production training: `No-Go`

C1f requires a 20k-30k teacher-EV cache with 100% replay-ready rows. Unbiased/enriched target misses are reported as warnings so the expanded calibration can still measure source bias explicitly.
