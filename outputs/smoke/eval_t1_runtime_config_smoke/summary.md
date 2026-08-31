# Hybrid Refinement Teacher Evaluation

- input: `ai\data\hybrid_t1t2_active_20260531\splits_t1_holdout_20260603\t1_mc300_lines001_100.jsonl`
- model: `ai/models/candidate_runs/t1-runtime112-top40regret-x20-plus-currenthard15-x20-ft-20260604/model/action_value_best.pt`
- time budget: 5000 ms

| scope | decisions | model top1 | final top1 | final zero-regret | pool | sync | sync zero-regret | delta | model regret | final regret | p95 ms | overrides | improved | worse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 1 | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% | 100.0% | +0.0% | 0.349 | 0.198 | 3683.8 | 1 | 1 | 0 |
| T1 | 1 | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% | 100.0% | +0.0% | 0.349 | 0.198 | 3683.8 | 1 | 1 | 0 |
