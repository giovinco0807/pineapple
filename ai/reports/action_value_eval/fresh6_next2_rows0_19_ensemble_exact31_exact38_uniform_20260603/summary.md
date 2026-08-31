# Action-Value Reranker Evaluation

- label: `fresh6_next2_rows0_19_ensemble_exact31_exact38_uniform`
- checkpoint: `ensemble`
- checkpoints: 2
- data: `ai\data\hybrid_t1t2_active_20260531\reranker_t2_fresh_next2_rows0_19_stable6_cap200_cap500_20260603`
- groups: 6
- samples: 117
- score MAE/corr: 3.463 / 0.608
- bust MAE: 0.142
- FL MAE: 0.042
- teacher rank mean/p95: 4.50 / 11.8
- full-bust chosen when avoidable: 0

## Group Recall

| K | recall | rerank regret |
|---:|---:|---:|
| 1 | 33.3% | 0.155 |
| 3 | 66.7% | 0.061 |
| 5 | 66.7% | 0.061 |
| 10 | 83.3% | 0.000 |
| 15 | 100.0% | 0.000 |
| 20 | 100.0% | 0.000 |
| 24 | 100.0% | 0.000 |
