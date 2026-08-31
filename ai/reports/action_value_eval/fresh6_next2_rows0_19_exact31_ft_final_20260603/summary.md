# Action-Value Reranker Evaluation

- label: `fresh6_next2_rows0_19_exact31_ft_final`
- checkpoint: `ai\models\candidate_runs\t2-stable-oracle31-cap500-ft-from-t2resid200-20260603\model\action_value_final.pt`
- checkpoints: 1
- data: `ai\data\hybrid_t1t2_active_20260531\reranker_t2_fresh_next2_rows0_19_stable6_cap200_cap500_20260603`
- groups: 6
- samples: 117
- score MAE/corr: 3.499 / 0.578
- bust MAE: 0.151
- FL MAE: 0.045
- teacher rank mean/p95: 4.17 / 11.5
- full-bust chosen when avoidable: 0

## Group Recall

| K | recall | rerank regret |
|---:|---:|---:|
| 1 | 50.0% | 0.126 |
| 3 | 66.7% | 0.061 |
| 5 | 66.7% | 0.061 |
| 10 | 83.3% | 0.000 |
| 15 | 100.0% | 0.000 |
| 20 | 100.0% | 0.000 |
| 24 | 100.0% | 0.000 |
