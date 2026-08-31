# Action-Value Reranker Evaluation

- label: `old_bb_holdout`
- checkpoint: `ai/models/candidate_runs/t3-jokerfix-20k-bb-20260605/action_value_best.pt`
- checkpoints: 1
- data: `ai\data\joker_rule_migration_20260711\holdout\reranker_t3_holdout`
- groups: 200
- samples: 3,513
- score MAE/corr: 2.354 / 0.585
- bust MAE: 0.249
- FL MAE: 0.046
- teacher rank mean/p95: 4.72 / 15.0
- full-bust chosen when avoidable: 48

## Group Recall

| K | recall | rerank regret |
|---:|---:|---:|
| 1 | 30.0% | 1.478 |
| 3 | 57.0% | 0.667 |
| 5 | 66.0% | 0.474 |
| 10 | 90.0% | 0.119 |
| 15 | 96.0% | 0.005 |
| 20 | 99.5% | 0.005 |
