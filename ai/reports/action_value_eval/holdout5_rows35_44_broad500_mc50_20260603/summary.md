# Action-Value Reranker Evaluation

- label: `holdout5_rows35_44_broad500_mc50`
- checkpoint: `ai\models\candidate_runs\t1t2-broad500-mc50-ft-20260602\model\action_value_best.pt`
- data: `ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next_rows35_44_misses5_cap200_cap500_20260603`
- groups: 5
- samples: 111
- score MAE/corr: 4.000 / 0.408
- bust MAE: 0.246
- FL MAE: 0.033
- teacher rank mean/p95: 3.60 / 5.8
- full-bust chosen when avoidable: 0

## Group Recall

| K | recall | rerank regret |
|---:|---:|---:|
| 1 | 20.0% | 3.684 |
| 3 | 40.0% | 2.143 |
| 5 | 80.0% | 0.000 |
| 10 | 100.0% | 0.000 |
| 15 | 100.0% | 0.000 |
| 20 | 100.0% | 0.000 |
| 24 | 100.0% | 0.000 |
