# Action-Value Reranker Evaluation

- label: `holdout1_rows45_49_exact31_ft_final`
- checkpoint: `ai\models\candidate_runs\t2-stable-oracle31-cap500-ft-from-t2resid200-20260603\model\action_value_final.pt`
- data: `ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next_rows45_49_misses2_cap200_cap500_20260603`
- groups: 1
- samples: 12
- score MAE/corr: 1.722 / -0.341
- bust MAE: 0.109
- FL MAE: 0.001
- teacher rank mean/p95: 9.00 / 9.0
- full-bust chosen when avoidable: 0

## Group Recall

| K | recall | rerank regret |
|---:|---:|---:|
| 1 | 0.0% | 0.418 |
| 3 | 0.0% | 0.407 |
| 5 | 0.0% | 0.407 |
| 10 | 100.0% | 0.000 |
| 15 | 100.0% | 0.000 |
| 20 | 100.0% | 0.000 |
| 24 | 100.0% | 0.000 |
