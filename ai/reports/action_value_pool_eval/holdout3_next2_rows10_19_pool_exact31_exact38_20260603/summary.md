# Action-Value Candidate Pool Evaluation

- label: `holdout3_next2_rows10_19_pool_exact31_exact38`
- checkpoints: 2
- data: `ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next2_rows10_19_misses3_cap200_cap500_20260603`
- groups: 3
- samples: 60

## Union Pool Recall

| per-model K | recall | rerank regret | avg pool | max pool |
|---:|---:|---:|---:|---:|
| 1 | 66.7% | 0.131 | 1.3 | 2 |
| 3 | 100.0% | 0.000 | 3.3 | 4 |
| 5 | 100.0% | 0.000 | 5.3 | 6 |
| 10 | 100.0% | 0.000 | 11.0 | 12 |
| 15 | 100.0% | 0.000 | 15.3 | 17 |
