# Action-Value Reranker Evaluation

- label: `new_btn`
- checkpoint: `ai/models/candidate_runs/t3-canonical-joker-20k-btn-20260711/action_value_best.pt`
- checkpoints: 1
- data: `ai\data\joker_rule_migration_20260711\t3_full\reranker_t3_canonical`
- groups: 10,000
- samples: 135,906
- score MAE/corr: 1.134 / 0.885
- bust MAE: 0.184
- FL MAE: 0.024
- teacher rank mean/p95: 2.72 / 7.0
- full-bust chosen when avoidable: 638

## Group Recall

| K | recall | rerank regret |
|---:|---:|---:|
| 1 | 41.0% | 0.677 |
| 3 | 74.0% | 0.168 |
| 5 | 88.3% | 0.058 |
| 10 | 99.1% | 0.004 |
| 15 | 100.0% | 0.001 |
| 20 | 100.0% | 0.000 |
