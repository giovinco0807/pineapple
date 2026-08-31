# Action-Value Reranker Evaluation

- label: `old_bb`
- checkpoint: `ai/models/candidate_runs/t3-jokerfix-20k-bb-20260605/action_value_best.pt`
- checkpoints: 1
- data: `ai\data\joker_rule_migration_20260711\t3_full\reranker_t3_canonical`
- groups: 10,000
- samples: 133,839
- score MAE/corr: 2.008 / 0.820
- bust MAE: 0.261
- FL MAE: 0.029
- teacher rank mean/p95: 3.07 / 9.0
- full-bust chosen when avoidable: 651

## Group Recall

| K | recall | rerank regret |
|---:|---:|---:|
| 1 | 38.5% | 0.867 |
| 3 | 70.0% | 0.235 |
| 5 | 84.5% | 0.088 |
| 10 | 97.6% | 0.007 |
| 15 | 99.5% | 0.002 |
| 20 | 100.0% | 0.000 |
