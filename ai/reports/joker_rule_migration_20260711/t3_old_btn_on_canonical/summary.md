# Action-Value Reranker Evaluation

- label: `old_btn`
- checkpoint: `ai/models/candidate_runs/t3-jokerfix-20k-btn-20260605/action_value_best.pt`
- checkpoints: 1
- data: `ai\data\joker_rule_migration_20260711\t3_full\reranker_t3_canonical`
- groups: 10,000
- samples: 135,906
- score MAE/corr: 1.832 / 0.807
- bust MAE: 0.270
- FL MAE: 0.028
- teacher rank mean/p95: 3.11 / 9.0
- full-bust chosen when avoidable: 729

## Group Recall

| K | recall | rerank regret |
|---:|---:|---:|
| 1 | 37.4% | 0.900 |
| 3 | 69.5% | 0.252 |
| 5 | 83.8% | 0.110 |
| 10 | 97.7% | 0.010 |
| 15 | 99.6% | 0.001 |
| 20 | 100.0% | 0.000 |
