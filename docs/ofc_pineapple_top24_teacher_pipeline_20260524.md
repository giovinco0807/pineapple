# OFC Pineapple Top24 Teacher Pipeline

This pipeline collects scalable silver teacher data by pruning every turn to
the current action-value model's top 24 candidates, then audits a sample
against full-candidate teacher labels.

## 1. Generate Top24 Self-Play States

```powershell
$RUN="tutor_top24_prune_20260524"
$RERANKER="ai/models/candidate_runs/tutor-route10-top20-prune-active8x-ft-20260523/model/action_value_best.pt"

python ai/self_play.py `
  --games 1000 `
  --workers 10 `
  --seed 20260524 `
  --game-start 0 `
  --rollouts 300 `
  --top-k 24 `
  --deep-all-turns `
  --record-action-scores `
  --reranker $RERANKER `
  --reranker-top-k 24 `
  --reranker-t2-top-k 24 `
  --output "ai/data/$RUN/selfplay_top24_mc300.jsonl"
```

Each `turn_log` records `n_actions`, `evaluated_actions`, `pruned_top_k`,
`eval_mode`, `sims`, `estimated`, `top_k_indices`, and `action_evs`.

## 2. Extract Training Targets

```powershell
python ai/tutor/extract_selfplay_targets.py `
  "ai/data/$RUN/selfplay_top24_mc300.jsonl" `
  --output "ai/data/$RUN/targets_all.jsonl" `
  --turns 0,1,2,3,4 `
  --require-pruned
```

## 3. Label Only The Pruned Candidates

This evaluates only the saved top24 candidates, producing candidate-level
`EV / FL / QQ / KK / AA / trips / bust` labels compatible with
`convert_action_value_teacher.py`.

```powershell
python ai/tutor/generate_pruned_teacher.py `
  "ai/data/$RUN/targets_all.jsonl" `
  --output "ai/data/$RUN/teacher_top24_mc300.jsonl" `
  --sims 300 `
  --turns 0,1,2,3,4 `
  --workers 10
```

Then convert to reranker training arrays:

```powershell
$env:PYTHONPATH="."
python ai/training/convert_action_value_teacher.py `
  "ai/data/$RUN/teacher_top24_mc300.jsonl" `
  --output "ai/data/$RUN/reranker_top24_mc300" `
  --turns 0,1,2,3,4 `
  --t0-max-candidates 0 `
  --regular-max-candidates 0
```

## 4. Audit Top24 Miss Rate

Sample 5-10% of states, generate full-candidate teacher labels, then audit.

```powershell
python ai/tutor/extract_selfplay_targets.py `
  "ai/data/$RUN/selfplay_top24_mc300.jsonl" `
  --output "ai/data/$RUN/audit_targets_10pct.jsonl" `
  --turns 0,1,2,3,4 `
  --sample-rate 0.10 `
  --require-pruned `
  --seed 20260524

$env:PYTHONPATH="."
python ai/training/generate_active_teacher.py `
  "ai/data/$RUN/audit_targets_10pct.jsonl" `
  --output "ai/data/$RUN/audit_full_labels_mc3000.jsonl" `
  --sims 3000 `
  --mc-turns 0,1,2 `
  --workers 10 `
  --progress-every 100

python ai/tutor/audit_pruned_teacher.py `
  --targets "ai/data/$RUN/audit_targets_10pct.jsonl" `
  --labels "ai/data/$RUN/audit_full_labels_mc3000.jsonl" `
  --output "ai/data/$RUN/audit_top24_vs_full_mc3000.json"
```

Acceptance rule for using the data as high-confidence training input:

- overall top24 hit rate: 99%+
- each turn top24 hit rate: 98%+
- all misses are written to `.misses.jsonl` and should be recycled as hard
  examples in the next training run

