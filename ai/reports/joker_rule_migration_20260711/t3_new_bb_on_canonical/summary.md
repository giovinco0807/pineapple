# Action-Value Reranker Evaluation

- label: `new_bb`
- checkpoint: `ai/models/candidate_runs/t3-canonical-joker-20k-bb-20260711/action_value_best.pt`
- checkpoints: 1
- data: `ai\data\joker_rule_migration_20260711\t3_full\reranker_t3_canonical`
- groups: 10,000
- samples: 133,839
- score MAE/corr: 1.390 / 0.836
- bust MAE: 0.219
- FL MAE: 0.032
- teacher rank mean/p95: 3.02 / 9.0
- full-bust chosen when avoidable: 685

## Group Recall

| K | recall | rerank regret |
|---:|---:|---:|
| 1 | 38.4% | 0.857 |
| 3 | 70.4% | 0.228 |
| 5 | 84.8% | 0.088 |
| 10 | 98.1% | 0.010 |
| 15 | 99.8% | 0.001 |
| 20 | 100.0% | 0.000 |
