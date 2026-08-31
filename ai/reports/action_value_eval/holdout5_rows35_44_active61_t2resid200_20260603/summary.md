# Action-Value Reranker Evaluation

- label: `holdout5_rows35_44_active61_t2resid200`
- checkpoint: `ai\models\candidate_runs\tutor-route10-active61-t2resid200-mc1000-suit24-ft-t2only-lr1e7-20260601\model\action_value_best.pt`
- data: `ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next_rows35_44_misses5_cap200_cap500_20260603`
- groups: 5
- samples: 111
- score MAE/corr: 2.550 / 0.558
- bust MAE: 0.307
- FL MAE: 0.035
- teacher rank mean/p95: 4.40 / 7.6
- full-bust chosen when avoidable: 1

## Group Recall

| K | recall | rerank regret |
|---:|---:|---:|
| 1 | 40.0% | 0.685 |
| 3 | 40.0% | 0.224 |
| 5 | 40.0% | 0.178 |
| 10 | 100.0% | 0.000 |
| 15 | 100.0% | 0.000 |
| 20 | 100.0% | 0.000 |
| 24 | 100.0% | 0.000 |
