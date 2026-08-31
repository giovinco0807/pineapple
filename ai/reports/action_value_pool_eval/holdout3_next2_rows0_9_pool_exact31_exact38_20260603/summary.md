# Action-Value Candidate Pool Evaluation

- label: `holdout3_next2_rows0_9_pool_exact31_exact38`
- checkpoints: 2
- data: `ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next2_rows0_9_misses4_cap200_cap500_20260603`
- groups: 3
- samples: 57

## Union Pool Recall

| per-model K | recall | rerank regret | avg pool | max pool |
|---:|---:|---:|---:|---:|
| 1 | 33.3% | 0.122 | 1.7 | 2 |
| 3 | 33.3% | 0.122 | 3.7 | 4 |
| 5 | 33.3% | 0.122 | 5.7 | 7 |
| 10 | 66.7% | 0.000 | 11.0 | 13 |
| 15 | 100.0% | 0.000 | 14.7 | 16 |
