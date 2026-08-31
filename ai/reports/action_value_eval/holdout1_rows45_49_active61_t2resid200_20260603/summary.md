# Action-Value Reranker Evaluation

- label: `holdout1_rows45_49_active61_t2resid200`
- checkpoint: `ai\models\candidate_runs\tutor-route10-active61-t2resid200-mc1000-suit24-ft-t2only-lr1e7-20260601\model\action_value_best.pt`
- data: `ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next_rows45_49_misses2_cap200_cap500_20260603`
- groups: 1
- samples: 12
- score MAE/corr: 3.057 / -0.302
- bust MAE: 0.457
- FL MAE: 0.000
- teacher rank mean/p95: 10.00 / 10.0
- full-bust chosen when avoidable: 0

## Group Recall

| K | recall | rerank regret |
|---:|---:|---:|
| 1 | 0.0% | 0.418 |
| 3 | 0.0% | 0.418 |
| 5 | 0.0% | 0.014 |
| 10 | 100.0% | 0.000 |
| 15 | 100.0% | 0.000 |
| 20 | 100.0% | 0.000 |
| 24 | 100.0% | 0.000 |
