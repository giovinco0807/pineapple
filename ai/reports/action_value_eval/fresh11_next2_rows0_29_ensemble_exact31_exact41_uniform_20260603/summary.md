# Action-Value Reranker Evaluation

- label: `fresh11_next2_rows0_29_ensemble_exact31_exact41_uniform`
- checkpoint: `ensemble`
- checkpoints: 2
- data: `ai\data\hybrid_t1t2_active_20260531\reranker_t2_fresh_next2_rows0_29_stable11_cap200_cap500_20260603`
- groups: 11
- samples: 213
- score MAE/corr: 2.972 / 0.771
- bust MAE: 0.161
- FL MAE: 0.044
- teacher rank mean/p95: 2.82 / 7.0
- full-bust chosen when avoidable: 0

## Group Recall

| K | recall | rerank regret |
|---:|---:|---:|
| 1 | 54.5% | 0.296 |
| 3 | 72.7% | 0.046 |
| 5 | 72.7% | 0.046 |
| 10 | 100.0% | 0.000 |
| 15 | 100.0% | 0.000 |
| 20 | 100.0% | 0.000 |
| 24 | 100.0% | 0.000 |
