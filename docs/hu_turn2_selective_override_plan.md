# HU T2 Selective Override Plan

## Baseline

Current T2 baseline is `models/turn2_stage8.pkl`, the self-board Turn2 action-value model used at 7-card boards.

Legacy/original open-discard T3 continuation was:

- policy: `Stage7_candidate_A_m5_r10`
- model: `models/hu_turn3_stage7_reference_override_cached_rank_wide.pt`
- `hu_turn3_min_margin = 5.0`
- `hu_turn3_reference_min_margin = 10.0`
- fallback inside that legacy selective override: Stage3 HU margin10 policy
- full replacement: disabled
- margin0.25: excluded

Current hidden-discard caution:

- The plan above was written before the hidden-discard information-model fix.
- New hidden-discard T2 teacher generation must not silently assume Stage7
  `m5_r10` is fixed.
- The current `hu_turn2_teacher_data` default is
  `--t3-continuation stage3_reference_default`, which records
  `continuation_policy_T3 = Stage3_HU_reference_default`.
- Stage7 `m5_r10` is still available only by explicitly passing
  `--t3-continuation stage7_m5_r10`, and those artifacts should be labeled as
  Stage7 opt-in / legacy-continuation experiments.

## Teacher Generation

Initial target:

- states: 50,000
- all legal actions
- MC rollouts: 4,096
- common random future decks: required
- missing target: 0

Bucket mix:

- 40% natural self-play
- 30% teacher disagreement
- 10% high-regret mistakes
- 10% low-margin near ties
- 5% high-margin
- 5% random off-policy

Each teacher row stores full action EVs, standard errors,
baseline/reference/fallback actions, deltas, explicit T3 continuation metadata,
scoring/royalty versions, and source bucket. Hidden-discard artifacts should use
`Stage3_HU_reference_default` unless a run is intentionally labeled as Stage7
`m5_r10` opt-in.

## Commands

Smoke:

```powershell
python -m ofc_regular.hu_turn2_teacher_data --samples 1 --future-samples 4 --seed 2026061001 --source-bucket natural --prediction-threads 1 --t3-continuation stage3_reference_default --output outputs/hu_turn2_stage1_smoke_mc4.jsonl
```

Shard:

```powershell
.\scripts\Run-HuTurn2TeacherShard.ps1 -SourceBucket natural -Samples 100 -FutureSamples 4096 -Seed 2026061101 -PredictionThreads 1 -T3Continuation stage3_reference_default -Output outputs/hu_turn2_stage1/chunks/hu_turn2_teacher_natural_0000.jsonl
```

Full local chunk plan:

```powershell
.\scripts\Run-HuTurn2TeacherChunksParallel.ps1 -TotalSamples 50000 -Chunks 100 -FutureSamples 4096 -MaxParallel 4 -PredictionThreads 1 -T3Continuation stage3_reference_default
```

Merge chunks:

```powershell
Get-Content outputs/hu_turn2_stage1/chunks/*.jsonl | Set-Content outputs/hu_turn2_stage1/hu_turn2_teacher_merged.jsonl
```

Feature cache:

```powershell
.\scripts\Run-HuTurn2FeatureCache.ps1
```

Initial EV-head training:

```powershell
.\scripts\Run-HuTurn2Training.ps1 -Device cuda
```

## GCP Execution Plan

Use Spot VMs with small shards. Recommended first production run:

- 20 VMs
- 16 vCPU each
- 100 shards total
- 500 states per shard
- write JSONL per shard
- sync every completed shard to Cloud Storage
- retry missing shard files only

Spot VM preemption is expected. Shards must be append-free final files; incomplete shard files are discarded and rerun.

## Evaluation Plan

Teacher holdout:

- avg_regret
- p95/p99 regret
- top1/top3
- margin bucket regret
- predicted_delta vs teacher_delta calibration

Production eval:

- paired seat-swap vs current T2 baseline
- seed-level EV/hand
- aggregate EV/hand and 95% CI
- override_rate
- avg_gain_on_override
- false_positive_override_rate
- p95/p99/max loss

Threshold sweep:

- `hu_turn2_min_margin`: 3, 5, 8, 10
- `hu_turn2_reference_min_margin`: 8, 10, 15, 20
- conservative focus: 5/15, 8/10, 8/15, 10/15

## Adoption

Do not enable full replacement. Promote only if selective override is positive versus current T2 baseline, false positives are low, p95/p99 loss is acceptable, and multiple golden seeds point in the same direction.
