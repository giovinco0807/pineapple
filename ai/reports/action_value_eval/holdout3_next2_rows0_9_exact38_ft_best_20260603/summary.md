# Action-Value Reranker Evaluation

- label: `holdout3_next2_rows0_9_exact38_ft_best`
- checkpoint: `ai\models\candidate_runs\t2-stable-oracle38-cap500-ft-from-exact36-20260603\model\action_value_best.pt`
- data: `ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next2_rows0_9_misses4_cap200_cap500_20260603`
- groups: 3
- samples: 57
- score MAE/corr: 5.555 / 0.537
- bust MAE: 0.131
- FL MAE: 0.065
- teacher rank mean/p95: 7.33 / 11.7
- full-bust chosen when avoidable: 0

## Group Recall

| K | recall | rerank regret |
|---:|---:|---:|
| 1 | 33.3% | 2.268 |
| 3 | 33.3% | 0.122 |
| 5 | 33.3% | 0.122 |
| 10 | 66.7% | 0.000 |
| 15 | 100.0% | 0.000 |
| 20 | 100.0% | 0.000 |
| 24 | 100.0% | 0.000 |
