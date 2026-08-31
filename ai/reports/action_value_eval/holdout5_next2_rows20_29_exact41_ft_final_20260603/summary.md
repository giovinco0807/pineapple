# Action-Value Reranker Evaluation

- label: `holdout5_next2_rows20_29_exact41_ft_final`
- checkpoint: `ai\models\candidate_runs\t2-stable-oracle41-cap500-ft-from-exact38-20260603\model\action_value_final.pt`
- checkpoints: 1
- data: `ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next2_rows20_29_misses5_cap200_cap500_20260603`
- groups: 5
- samples: 96
- score MAE/corr: 3.265 / 0.841
- bust MAE: 0.202
- FL MAE: 0.045
- teacher rank mean/p95: 4.20 / 7.6
- full-bust chosen when avoidable: 0

## Group Recall

| K | recall | rerank regret |
|---:|---:|---:|
| 1 | 20.0% | 0.572 |
| 3 | 60.0% | 0.090 |
| 5 | 60.0% | 0.090 |
| 10 | 100.0% | 0.000 |
| 15 | 100.0% | 0.000 |
| 20 | 100.0% | 0.000 |
| 24 | 100.0% | 0.000 |
