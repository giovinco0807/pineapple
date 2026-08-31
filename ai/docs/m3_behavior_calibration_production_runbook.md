# M3 behavior calibration: locked local production runbook

This run is local-only. It does not require GCP. The production inputs are
the preregistered natural and targeted-Joker shard trees under
`ai/data/m3_behavior_calibration_production_20260713`.

## Immutable pre-label locks

- Collection plan:
  `ai/reports/m3_behavior_collection_plan_20260713/plan.json`
  - `plan_sha256`:
    `4aaa2d5720c4475fc9ecad5b9779ab1cededb567b8c88636fcc7f810392878af`
- Explicit gate config:
  `ai/config/m3_behavior_temperature_gate_v2_20260713.json`
  - embedded `gate_config_sha256`:
    `4f90642b47740785ea767aa80ce88055b429769756b657441299449c1e708569`
  - file SHA-256:
    `22dfa9de6a8fa65ab071bcaa2d7920fecca2150d6f138857850254a2adfa77fe`
- Natural collection:
  - 134,057 roots, 1,000 roots per full shard;
  - seed namespace
    `m3-behavior-calibration-production-natural-20260713-v1`;
  - sampling mode `natural_uniform_shuffle`.
- Joker challenge collection:
  - 24,000 roots, 1,000 roots per shard;
  - seed namespace
    `m3-behavior-calibration-production-joker-grid-20260713-v1`;
  - sampling mode `targeted_joker_challenge`;
  - challenge ID `m3-joker-challenge-v1`;
  - deterministic role/Joker cycle.

The gate file was fixed and content-hashed before the natural locked-test
evaluation. The
pipeline rejects a non-canonical file, an invalid self-hash, any difference
from `build_temperature_gate_config()`, or any difference from the gate
embedded in the collection plan before creating an evaluation directory.

## Exact collection commands

These commands are recorded for reproducibility. Do not launch a second
collector against a directory while its current collector is alive.

```powershell
$repo = 'C:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple'
$run = Join-Path $repo 'ai\data\m3_behavior_calibration_production_20260713'

C:\Python313\python.exe -u -m ai.tutor.collect_hu_behavior_trace_shards `
  --workspace-root $repo `
  --output-dir (Join-Path $run 'natural') `
  --total-root-target 134057 `
  --shard-size 1000 `
  --seed-namespace m3-behavior-calibration-production-natural-20260713-v1 `
  --root-sampling-mode natural_uniform_shuffle `
  --resume

C:\Python313\python.exe -u -m ai.tutor.collect_hu_behavior_trace_shards `
  --workspace-root $repo `
  --output-dir (Join-Path $run 'challenge') `
  --total-root-target 24000 `
  --shard-size 1000 `
  --seed-namespace m3-behavior-calibration-production-joker-grid-20260713-v1 `
  --root-sampling-mode targeted_joker_challenge `
  --challenge-id m3-joker-challenge-v1 `
  --resume
```

## Required end-to-end command

Run this only after both collection manifests say `collection_complete=true`.
It verifies the gate and plan first, verifies every collection shard against
the preregistered root commitments, authenticates any existing evaluation
prefix, appends every remaining evaluation shard, and finally writes
`calibration.json` atomically.

```powershell
$repo = 'C:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple'
$run = Join-Path $repo 'ai\data\m3_behavior_calibration_production_20260713'

C:\Python313\python.exe -u -m ai.tutor.run_m3_behavior_calibration_pipeline `
  --workspace-root $repo `
  --run-dir $run `
  --plan (Join-Path $repo 'ai\reports\m3_behavior_collection_plan_20260713\plan.json') `
  --gate-config (Join-Path $repo 'ai\config\m3_behavior_temperature_gate_v2_20260713.json') `
  --output (Join-Path $run 'calibration.json') `
  --scratch-dir (Join-Path $run 'calibration_scratch')
```

For an intentionally bounded invocation, add
`--max-new-evaluation-shards N`. A bounded invocation may return
`pipeline_complete=false`; rerunning the same command re-verifies the full
published prefix and continues. Without that option, all remaining shards are
attempted in one invocation, while every shard still publishes its top marker
before the next shard starts.

The only successful terminal stage is `calibration_complete`. The expected
artifact is:

`ai/data/m3_behavior_calibration_production_20260713/calibration.json`

An existing artifact is never overwritten. It is accepted only after a fresh
rebuild from all four immutable collection/evaluation trees and must contain
the same explicit gate config.

## Underlying CLI equivalents

The raw evaluator command is:

```powershell
C:\Python313\python.exe -u -m ai.tutor.evaluate_hu_behavior_trace_shards `
  --workspace-root $repo `
  --input-dir (Join-Path $run 'natural') `
  --output-dir (Join-Path $run 'natural_evaluation')
```

The raw calibration command is:

```powershell
C:\Python313\python.exe -u -m ai.tutor.behavior_temperature_calibration_shards `
  --workspace-root $repo `
  --natural-collection-dir (Join-Path $run 'natural') `
  --natural-evaluation-dir (Join-Path $run 'natural_evaluation') `
  --challenge-collection-dir (Join-Path $run 'challenge') `
  --challenge-evaluation-dir (Join-Path $run 'challenge_evaluation') `
  --gate-config (Join-Path $repo 'ai\config\m3_behavior_temperature_gate_v2_20260713.json') `
  --scratch-dir (Join-Path $run 'calibration_scratch') `
  --output (Join-Path $run 'calibration.json')
```

The direct evaluator does not accept a gate-config argument, so it is not the
locked production entry point. Use the end-to-end runner above; it establishes
the explicit gate lock before calling the evaluator.
