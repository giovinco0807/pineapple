# Action-Value Candidate Pool Evaluation

- label: `holdout5_next2_rows20_29_pool_exact31_exact41`
- checkpoints: 2
- data: `ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next2_rows20_29_misses5_cap200_cap500_20260603`
- groups: 5
- samples: 96

## Union Pool Recall

| per-model K | recall | rerank regret | avg pool | max pool |
|---:|---:|---:|---:|---:|
| 1 | 20.0% | 0.572 | 1.2 | 2 |
| 3 | 80.0% | 0.036 | 3.8 | 5 |
| 5 | 80.0% | 0.009 | 6.2 | 7 |
| 10 | 100.0% | 0.000 | 10.4 | 11 |
| 15 | 100.0% | 0.000 | 14.4 | 17 |
