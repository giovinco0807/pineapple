# Action-Value Reranker Evaluation

- label: `holdout5_next2_rows20_29_exact31_ft_final`
- checkpoint: `ai\models\candidate_runs\t2-stable-oracle31-cap500-ft-from-t2resid200-20260603\model\action_value_final.pt`
- checkpoints: 1
- data: `ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next2_rows20_29_misses5_cap200_cap500_20260603`
- groups: 5
- samples: 96
- score MAE/corr: 3.625 / 0.836
- bust MAE: 0.186
- FL MAE: 0.045
- teacher rank mean/p95: 4.40 / 7.8
- full-bust chosen when avoidable: 1

## Group Recall

| K | recall | rerank regret |
|---:|---:|---:|
| 1 | 20.0% | 0.573 |
| 3 | 60.0% | 0.106 |
| 5 | 60.0% | 0.021 |
| 10 | 100.0% | 0.000 |
| 15 | 100.0% | 0.000 |
| 20 | 100.0% | 0.000 |
| 24 | 100.0% | 0.000 |
