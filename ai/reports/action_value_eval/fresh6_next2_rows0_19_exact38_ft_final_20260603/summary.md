# Action-Value Reranker Evaluation

- label: `fresh6_next2_rows0_19_exact38_ft_final`
- checkpoint: `ai\models\candidate_runs\t2-stable-oracle38-cap500-ft-from-exact36-20260603\model\action_value_final.pt`
- checkpoints: 1
- data: `ai\data\hybrid_t1t2_active_20260531\reranker_t2_fresh_next2_rows0_19_stable6_cap200_cap500_20260603`
- groups: 6
- samples: 117
- score MAE/corr: 3.552 / 0.632
- bust MAE: 0.141
- FL MAE: 0.040
- teacher rank mean/p95: 4.50 / 11.2
- full-bust chosen when avoidable: 0

## Group Recall

| K | recall | rerank regret |
|---:|---:|---:|
| 1 | 33.3% | 1.228 |
| 3 | 66.7% | 0.061 |
| 5 | 66.7% | 0.061 |
| 10 | 83.3% | 0.000 |
| 15 | 100.0% | 0.000 |
| 20 | 100.0% | 0.000 |
| 24 | 100.0% | 0.000 |
