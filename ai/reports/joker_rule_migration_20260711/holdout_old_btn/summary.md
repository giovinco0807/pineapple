# Action-Value Reranker Evaluation

- label: `old_btn_holdout`
- checkpoint: `ai/models/candidate_runs/t3-jokerfix-20k-btn-20260605/action_value_best.pt`
- checkpoints: 1
- data: `ai\data\joker_rule_migration_20260711\holdout\reranker_t3_holdout`
- groups: 200
- samples: 3,411
- score MAE/corr: 1.976 / 0.426
- bust MAE: 0.296
- FL MAE: 0.037
- teacher rank mean/p95: 4.42 / 14.0
- full-bust chosen when avoidable: 43

## Group Recall

| K | recall | rerank regret |
|---:|---:|---:|
| 1 | 30.0% | 1.237 |
| 3 | 54.0% | 0.529 |
| 5 | 70.5% | 0.209 |
| 10 | 92.5% | 0.005 |
| 15 | 96.5% | 0.000 |
| 20 | 100.0% | 0.000 |
