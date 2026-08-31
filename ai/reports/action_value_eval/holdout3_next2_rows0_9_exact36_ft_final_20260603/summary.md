# Action-Value Reranker Evaluation

- label: `holdout3_next2_rows0_9_exact36_ft_final`
- checkpoint: `ai\models\candidate_runs\t2-stable-oracle36-cap500-ft-from-exact31-20260603\model\action_value_final.pt`
- data: `ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next2_rows0_9_misses4_cap200_cap500_20260603`
- groups: 3
- samples: 57
- score MAE/corr: 5.610 / 0.523
- bust MAE: 0.130
- FL MAE: 0.067
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
