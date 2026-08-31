# Action-Value Reranker Evaluation

- label: `holdout1_rows25_34_broad500_mc50`
- checkpoint: `ai\models\candidate_runs\t1t2-broad500-mc50-ft-20260602\model\action_value_best.pt`
- data: `ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next_rows25_34_misses3_cap200_cap500_20260603`
- groups: 1
- samples: 24
- score MAE/corr: 3.533 / 0.251
- bust MAE: 0.179
- FL MAE: 0.016
- teacher rank mean/p95: 8.00 / 8.0
- full-bust chosen when avoidable: 0

## Group Recall

| K | recall | rerank regret |
|---:|---:|---:|
| 1 | 0.0% | 0.357 |
| 3 | 0.0% | 0.357 |
| 5 | 0.0% | 0.357 |
| 10 | 100.0% | 0.000 |
| 15 | 100.0% | 0.000 |
| 20 | 100.0% | 0.000 |
| 24 | 100.0% | 0.000 |
