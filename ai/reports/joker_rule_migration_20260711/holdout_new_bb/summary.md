# Action-Value Reranker Evaluation

- label: `new_bb_holdout`
- checkpoint: `ai/models/candidate_runs/t3-canonical-joker-20k-bb-20260711/action_value_best.pt`
- checkpoints: 1
- data: `ai\data\joker_rule_migration_20260711\holdout\reranker_t3_holdout`
- groups: 200
- samples: 3,513
- score MAE/corr: 1.207 / 0.590
- bust MAE: 0.250
- FL MAE: 0.045
- teacher rank mean/p95: 4.25 / 12.0
- full-bust chosen when avoidable: 47

## Group Recall

| K | recall | rerank regret |
|---:|---:|---:|
| 1 | 33.5% | 1.154 |
| 3 | 60.0% | 0.504 |
| 5 | 70.5% | 0.371 |
| 10 | 90.5% | 0.101 |
| 15 | 97.5% | 0.010 |
| 20 | 99.0% | 0.005 |
