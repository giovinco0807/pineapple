# T2 T3-union runtime benchmark

This benchmark measures the local serving path for T2:

1. Score all legal T2 actions with the action-value model.
2. Keep the hybrid shortlist pool.
3. Refine the sync candidates by sampling T3 deals.
4. For each sampled T3 deal, build a T3 union candidate pool and exact-rerank it with the Rust solver.

Script:

```powershell
python -m ai.tutor.benchmark_t2_t3_union_runtime `
  --limit 100 `
  --device cpu `
  --t3-device cpu `
  --output-dir D:/ofc-pineapple-data/t3_evloss_fresh_20260607/t2_t3_union_benchmark_local500_limit100_forcedguard_floor
```

Default input:

```text
ai/data/hybrid_t1t2_active_20260531/local_t1t2_500_mc300.jsonl
```

Latest local CPU smoke, 2026-06-07, with the cheap forced-bust guard:

- Count: 100 T2 positions.
- Latency mean/p50/p95/max: 745.3 / 723.8 / 1089.3 / 1192.5 ms.
- Under 5 seconds: 100/100.
- T2 candidate pool mean/max: 17.48 / 20.
- T3 union pool mean-of-means/max: 13.71 / 21.
- Refinement errors: 0.

The forced-bust guard is conservative.  It only removes T2 candidates when a
cheap bottom-fixed feasibility check proves that no legal completion is
available.  If the check would require too much search, it leaves the candidate
in the pool so the runtime stays under budget.

Output files:

- `results.jsonl`: full source payload, runtime result, candidate list, and teacher comparison.
- `rows.jsonl`: compact per-position metrics and best action.
- `teacher_misses.jsonl`: positions where the selected action loses EV against the available teacher labels.
- `summary.json`: machine-readable aggregate.
- `summary.md`: short human-readable aggregate.

Important caveat: the teacher comparison in this input is MC300 over full T2
actions, while the fast runtime currently refines only sampled T3 deals. Use it
as a rough regression signal, not as the final T2 exact quality metric.
