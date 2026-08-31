# Action-Value Reranker Evaluation

- label: `fresh11_next2_rows0_29_exact31_ft_final`
- checkpoint: `ai\models\candidate_runs\t2-stable-oracle31-cap500-ft-from-t2resid200-20260603\model\action_value_final.pt`
- checkpoints: 1
- data: `ai\data\hybrid_t1t2_active_20260531\reranker_t2_fresh_next2_rows0_29_stable11_cap200_cap500_20260603`
- groups: 11
- samples: 213
- score MAE/corr: 3.556 / 0.672
- bust MAE: 0.167
- FL MAE: 0.045
- teacher rank mean/p95: 4.27 / 10.5
- full-bust chosen when avoidable: 1

## Group Recall

| K | recall | rerank regret |
|---:|---:|---:|
| 1 | 36.4% | 0.329 |
| 3 | 63.6% | 0.081 |
| 5 | 63.6% | 0.043 |
| 10 | 90.9% | 0.000 |
| 15 | 100.0% | 0.000 |
| 20 | 100.0% | 0.000 |
| 24 | 100.0% | 0.000 |
