# T2 Exact Pilot - 2026-06-08

## Purpose

Check whether the new T2 rows generated from T0/T1 Top10 branches can be promoted from T3-model labels to Rust T2 capped-exact labels.

Input:

`D:/ofc-pineapple-data/t2_t0t1_top10_t3_model_20260608/root1_full_s2/t2_t3_model_rows.jsonl`

## Compatibility Fix

`ai/tutor/run_t2_exact_oracle.py` now reads candidate scores from the new T3-model fields:

- `t3_model_ev`
- `model_t3_value_score`
- `model_t3_priority_score`

It also reads `model_t3_value_bust` and `model_t3_value_fl` for comparison reports.

## cap20 Limit2

Command:

```powershell
python -m ai.tutor.run_t2_exact_oracle `
  --input D:\ofc-pineapple-data\t2_t0t1_top10_t3_model_20260608\root1_full_s2\t2_t3_model_rows.jsonl `
  --out-dir D:\ofc-pineapple-data\t2_t0t1_top10_t3_model_20260608\exact_cap20_limit2 `
  --limit 2 `
  --top-n 24 `
  --t2-draw-limit 20 `
  --binary ai/rust_solver/target/release/t3_exact_solver.exe `
  --miss-top-k 5
```

Results:

- Records: 2
- Same Top1 vs T3-model source: 0/2
- Avg elapsed: 82.6s / record
- Avg estimated full T2 elapsed from cap: 29,479.8s / record
- Avg EV regret of T3-model Top1: 12.488
- Max EV regret of T3-model Top1: 15.841

Output:

`D:/ofc-pineapple-data/t2_t0t1_top10_t3_model_20260608/exact_cap20_limit2/t2_oracle_cap20_limit2.jsonl`

## cap50 Limit1

Command:

```powershell
python -m ai.tutor.run_t2_exact_oracle `
  --input D:\ofc-pineapple-data\t2_t0t1_top10_t3_model_20260608\root1_full_s2\t2_t3_model_rows.jsonl `
  --out-dir D:\ofc-pineapple-data\t2_t0t1_top10_t3_model_20260608\exact_cap50_limit1 `
  --limit 1 `
  --top-n 24 `
  --t2-draw-limit 50 `
  --binary ai/rust_solver/target/release/t3_exact_solver.exe `
  --miss-top-k 5
```

Results:

- Records: 1
- Same Top1 vs T3-model source: 0/1
- Elapsed: 168.2s / record
- Estimated full T2 elapsed from cap: 24,016.4s / record
- EV regret of T3-model Top1: 14.725

Output:

`D:/ofc-pineapple-data/t2_t0t1_top10_t3_model_20260608/exact_cap50_limit1/t2_oracle_cap50_limit1.jsonl`

## Example: Record 0

Position: BB

Dealt: `Kc 2c 7s`

Board:

- Top: `2d Ac`
- Middle: `3h 7h`
- Bottom: `4s Kd Kh`

Opponent:

- Top: `2s`
- Middle: `3s 6s 6h`
- Bottom: `Js Ts Ks`

T3-model source, draw limit 2:

- Action: `7s->middle, Kc->middle; discard 2c`
- Score: 21.182
- FL: 27.2%
- Bust: 16.2%

cap20 best:

- Action: `2c->middle, 7s->middle; discard Kc`
- Score: 25.903
- FL: 40.8%
- Bust: 5.1%

cap50 best:

- Action: `7s->middle, Kc->bottom; discard 2c`
- Score: 22.200
- FL: 28.5%
- Bust: 5.3%

cap50 second:

- Action: `2c->middle, 7s->middle; discard Kc`
- Score: 22.193

The cap50 top two actions are essentially tied, but the T3-model source Top1 falls to rank 5 under cap50 with a large EV loss.

## Conclusion

The broad T2 generation path is working, but `draw_limit=2` is too noisy to use directly as a T2 teacher label. Use it for cheap candidate discovery and hard-case mining only.

For actual T2 training labels:

1. Generate many T2 rows cheaply with the T0/T1 Top10 -> T3-model path.
2. Promote high-loss or close-call rows to Rust capped exact.
3. Treat `cap20/cap50` agreement, not raw model Top1, as the first stable-label filter.
4. Only send a small selected subset toward higher caps or full exact because local full T2 exact is hours per record.
