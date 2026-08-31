# Action-Value Reranker Evaluation

- label: `holdout3_next2_rows10_19_exact41_ft_final`
- checkpoint: `ai\models\candidate_runs\t2-stable-oracle41-cap500-ft-from-exact38-20260603\model\action_value_final.pt`
- data: `ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next2_rows10_19_misses3_cap200_cap500_20260603`
- groups: 3
- samples: 60
- score MAE/corr: 1.520 / 0.837
- bust MAE: 0.148
- FL MAE: 0.016
- teacher rank mean/p95: 1.67 / 2.0
- full-bust chosen when avoidable: 0

## Group Recall

| K | recall | rerank regret |
|---:|---:|---:|
| 1 | 33.3% | 0.187 |
| 3 | 100.0% | 0.000 |
| 5 | 100.0% | 0.000 |
| 10 | 100.0% | 0.000 |
| 15 | 100.0% | 0.000 |
| 20 | 100.0% | 0.000 |
| 24 | 100.0% | 0.000 |
