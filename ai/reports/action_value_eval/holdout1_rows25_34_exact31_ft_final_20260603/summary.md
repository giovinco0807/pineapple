# Action-Value Reranker Evaluation

- label: `holdout1_rows25_34_exact31_ft_final`
- checkpoint: `ai\models\candidate_runs\t2-stable-oracle31-cap500-ft-from-t2resid200-20260603\model\action_value_final.pt`
- data: `ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next_rows25_34_misses3_cap200_cap500_20260603`
- groups: 1
- samples: 24
- score MAE/corr: 1.029 / 0.629
- bust MAE: 0.172
- FL MAE: 0.016
- teacher rank mean/p95: 4.00 / 4.0
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
