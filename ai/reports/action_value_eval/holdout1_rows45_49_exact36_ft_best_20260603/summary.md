# Action-Value Reranker Evaluation

- label: `holdout1_rows45_49_exact36_ft_best`
- checkpoint: `ai\models\candidate_runs\t2-stable-oracle36-cap500-ft-from-exact31-20260603\model\action_value_best.pt`
- data: `ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next_rows45_49_misses2_cap200_cap500_20260603`
- groups: 1
- samples: 12
- score MAE/corr: 2.338 / 0.292
- bust MAE: 0.112
- FL MAE: 0.001
- teacher rank mean/p95: 6.00 / 6.0
- full-bust chosen when avoidable: 0

## Group Recall

| K | recall | rerank regret |
|---:|---:|---:|
| 1 | 0.0% | 0.407 |
| 3 | 0.0% | 0.407 |
| 5 | 0.0% | 0.000 |
| 10 | 100.0% | 0.000 |
| 15 | 100.0% | 0.000 |
| 20 | 100.0% | 0.000 |
| 24 | 100.0% | 0.000 |
