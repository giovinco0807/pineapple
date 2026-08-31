# Action-Value Reranker Evaluation

- label: `holdout5_next2_rows20_29_ensemble_exact31_exact41_uniform`
- checkpoint: `ensemble`
- checkpoints: 2
- data: `ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next2_rows20_29_misses5_cap200_cap500_20260603`
- groups: 5
- samples: 96
- score MAE/corr: 3.440 / 0.841
- bust MAE: 0.192
- FL MAE: 0.045
- teacher rank mean/p95: 4.80 / 7.0
- full-bust chosen when avoidable: 0

## Group Recall

| K | recall | rerank regret |
|---:|---:|---:|
| 1 | 20.0% | 0.572 |
| 3 | 40.0% | 0.102 |
| 5 | 40.0% | 0.102 |
| 10 | 100.0% | 0.000 |
| 15 | 100.0% | 0.000 |
| 20 | 100.0% | 0.000 |
| 24 | 100.0% | 0.000 |
