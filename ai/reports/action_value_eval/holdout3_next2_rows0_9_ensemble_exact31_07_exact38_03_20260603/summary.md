# Action-Value Reranker Evaluation

- label: `holdout3_next2_rows0_9_ensemble_exact31_07_exact38_03`
- checkpoint: `ensemble`
- checkpoints: 2
- data: `ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next2_rows0_9_misses4_cap200_cap500_20260603`
- groups: 3
- samples: 57
- score MAE/corr: 5.904 / 0.455
- bust MAE: 0.133
- FL MAE: 0.071
- teacher rank mean/p95: 7.33 / 12.5
- full-bust chosen when avoidable: 0

## Group Recall

| K | recall | rerank regret |
|---:|---:|---:|
| 1 | 33.3% | 0.122 |
| 3 | 33.3% | 0.122 |
| 5 | 33.3% | 0.122 |
| 10 | 66.7% | 0.000 |
| 15 | 100.0% | 0.000 |
| 20 | 100.0% | 0.000 |
| 24 | 100.0% | 0.000 |
