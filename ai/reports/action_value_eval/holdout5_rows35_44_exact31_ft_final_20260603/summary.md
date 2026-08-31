# Action-Value Reranker Evaluation

- label: `holdout5_rows35_44_exact31_ft_final`
- checkpoint: `ai\models\candidate_runs\t2-stable-oracle31-cap500-ft-from-t2resid200-20260603\model\action_value_final.pt`
- data: `ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next_rows35_44_misses5_cap200_cap500_20260603`
- groups: 5
- samples: 111
- score MAE/corr: 1.796 / 0.645
- bust MAE: 0.217
- FL MAE: 0.035
- teacher rank mean/p95: 4.00 / 7.8
- full-bust chosen when avoidable: 1

## Group Recall

| K | recall | rerank regret |
|---:|---:|---:|
| 1 | 40.0% | 0.226 |
| 3 | 60.0% | 0.046 |
| 5 | 60.0% | 0.046 |
| 10 | 100.0% | 0.000 |
| 15 | 100.0% | 0.000 |
| 20 | 100.0% | 0.000 |
| 24 | 100.0% | 0.000 |
