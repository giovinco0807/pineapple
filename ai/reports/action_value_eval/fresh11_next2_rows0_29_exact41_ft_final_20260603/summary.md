# Action-Value Reranker Evaluation

- label: `fresh11_next2_rows0_29_exact41_ft_final`
- checkpoint: `ai\models\candidate_runs\t2-stable-oracle41-cap500-ft-from-exact38-20260603\model\action_value_final.pt`
- checkpoints: 1
- data: `ai\data\hybrid_t1t2_active_20260531\reranker_t2_fresh_next2_rows0_29_stable11_cap200_cap500_20260603`
- groups: 11
- samples: 213
- score MAE/corr: 2.518 / 0.795
- bust MAE: 0.164
- FL MAE: 0.050
- teacher rank mean/p95: 2.64 / 7.0
- full-bust chosen when avoidable: 0

## Group Recall

| K | recall | rerank regret |
|---:|---:|---:|
| 1 | 45.5% | 0.311 |
| 3 | 81.8% | 0.041 |
| 5 | 81.8% | 0.041 |
| 10 | 100.0% | 0.000 |
| 15 | 100.0% | 0.000 |
| 20 | 100.0% | 0.000 |
| 24 | 100.0% | 0.000 |
