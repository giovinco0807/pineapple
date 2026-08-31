# Action-Value Reranker Evaluation

- label: `holdout1_rows25_34_active61_t2resid200`
- checkpoint: `ai\models\candidate_runs\tutor-route10-active61-t2resid200-mc1000-suit24-ft-t2only-lr1e7-20260601\model\action_value_best.pt`
- data: `ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next_rows25_34_misses3_cap200_cap500_20260603`
- groups: 1
- samples: 24
- score MAE/corr: 2.638 / 0.403
- bust MAE: 0.240
- FL MAE: 0.017
- teacher rank mean/p95: 5.00 / 5.0
- full-bust chosen when avoidable: 0

## Group Recall

| K | recall | rerank regret |
|---:|---:|---:|
| 1 | 0.0% | 0.357 |
| 3 | 0.0% | 0.357 |
| 5 | 100.0% | 0.000 |
| 10 | 100.0% | 0.000 |
| 15 | 100.0% | 0.000 |
| 20 | 100.0% | 0.000 |
| 24 | 100.0% | 0.000 |
