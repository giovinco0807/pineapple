# Action-Value Reranker Evaluation

- label: `new_btn_holdout`
- checkpoint: `ai/models/candidate_runs/t3-canonical-joker-20k-btn-20260711/action_value_best.pt`
- checkpoints: 1
- data: `ai\data\joker_rule_migration_20260711\holdout\reranker_t3_holdout`
- groups: 200
- samples: 3,411
- score MAE/corr: 1.427 / 0.520
- bust MAE: 0.265
- FL MAE: 0.040
- teacher rank mean/p95: 4.09 / 13.0
- full-bust chosen when avoidable: 43

## Group Recall

| K | recall | rerank regret |
|---:|---:|---:|
| 1 | 28.5% | 1.196 |
| 3 | 61.5% | 0.158 |
| 5 | 73.0% | 0.090 |
| 10 | 92.5% | 0.013 |
| 15 | 96.5% | 0.008 |
| 20 | 100.0% | 0.000 |
