# Action-Value Reranker Evaluation

- label: `holdout3_next2_rows0_9_exact31_ft_final`
- checkpoint: `ai\models\candidate_runs\t2-stable-oracle31-cap500-ft-from-t2resid200-20260603\model\action_value_final.pt`
- data: `ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next2_rows0_9_misses4_cap200_cap500_20260603`
- groups: 3
- samples: 57
- score MAE/corr: 6.068 / 0.416
- bust MAE: 0.139
- FL MAE: 0.076
- teacher rank mean/p95: 7.00 / 12.4
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
