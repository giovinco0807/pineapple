# Action-Value Reranker Evaluation

- label: `holdout3_next2_rows10_19_ensemble_exact31_exact41_uniform`
- checkpoint: `ensemble`
- checkpoints: 2
- data: `ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next2_rows10_19_misses3_cap200_cap500_20260603`
- groups: 3
- samples: 60
- score MAE/corr: 1.207 / 0.857
- bust MAE: 0.152
- FL MAE: 0.017
- teacher rank mean/p95: 1.33 / 1.9
- full-bust chosen when avoidable: 0

## Group Recall

| K | recall | rerank regret |
|---:|---:|---:|
| 1 | 66.7% | 0.131 |
| 3 | 100.0% | 0.000 |
| 5 | 100.0% | 0.000 |
| 10 | 100.0% | 0.000 |
| 15 | 100.0% | 0.000 |
| 20 | 100.0% | 0.000 |
| 24 | 100.0% | 0.000 |
