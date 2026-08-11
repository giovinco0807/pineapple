# M4.3 final population readiness

Date: 2026-07-13

## Audit result

`evaluate_hu_m4_population.py` already measured the required realized-hand
quantities correctly: same-seed candidate/baseline counterfactuals, physical
seat swap, explicit four-opponent population, first/second summaries, realized
gain per override, false positives, p95/p99/max loss, and full-trajectory
non-fire cancellation. It never resolves `current`.

It was not sufficient for the final Spot run by itself. It was monolithic,
opened its record output in truncate mode, had no completed-shard checkpoint,
and had no strict merger. The existing M4.2 model Spot script is also not a
valid substitute: it freezes the old gate40 teacher/source hashes, uses
`negative_regret_ranker_v2`, consumes the old 20/10/10 inputs, and does not run
the final population evaluation.

The readiness additions are:

- `summarize_hu_m4_population_records`: validates an exact opponent/seed/seat
  grid and recomputes all final statistics from raw records. First/second EV
  confidence intervals use hand seed clusters, and gain/override uses a
  seed-clustered ratio influence function rather than treating four opponents
  on the same deal as independent.
- `merge_hu_m4_population_shards.py`: verifies every shard summary against its
  records, rejects gaps/duplicates/mixed model thresholds, and recomputes the
  final confidence intervals after merging. It never averages shard CIs.
- `Start/Get/Receive-GcpHuM43Attempt02PopulationRun.ps1`: immutable, Spot, 20 small
  shards, heartbeat, completed-shard checkpoint, missing-shard resume,
  self-delete, hash-verified receive, and `DONE` uploaded last.
- `validate_hu_m43_attempt02_acceptance.py`: Attempt02 requires the v4 model and
  training manifest,
  sealed data contract, pre-holdout model/threshold freeze, one-shot receipt,
  consumption marker, frozen population plan, raw merged records, merge
  manifest, Spot receipt/run manifest, and fresh population evaluation in one
  fail-closed hash/content chain. It recomputes the evaluation from records and
  verifies non-fire trajectory digests, score deltas, policy seeds, shard
  provenance, and every byte hash. M4/M4.1/M4.2 compatibility remains.

No profile is activated and no `current` mapping is changed by any of these
commands.

## Frozen population size

One paired population seed creates four second-seat override opportunities:
one for each required opponent. For an independently estimated fire-rate lower
bound `r_low`, the minimum seed count is:

`ceil(300 / (4 * r_low))`.

The bounded pilot's minimum positive threshold-lock signal is 10 fires in 30
states. Its two-sided 95% Wilson lower bound is about 0.1923, implying 391
paired seeds under the formula. The frozen plan uses 1,000 paired seeds, so it
needs only a 7.5% realized population fire rate to obtain 300 overrides. The
count is fixed before population outcomes; post-hoc extension and optional
stopping are forbidden. Fewer than 300 fires is a completed No-Go.

The fixed work is 8,000 candidate records, 8,000 matched baseline records, and
16,000 terminal hand traces. The prior local smoke took 13.1858 seconds for two
paired seeds, projecting about 1 hour 50 minutes locally. Twenty 50-seed Spot
shards project about 5.5 minutes of evaluation per worker plus VM/bootstrap
time. The schedule starts at 6,106,071,901 with stride 1,000,003. The frozen
plan lists 19 M4-through-Attempt02 teacher/correctness schedules (755 unique
hand seeds) plus six prior population-smoke seeds; the validator requires that
exact list and found zero overlap across 1,000 population seeds.

## Exact final Go/No-Go inputs

The validator must receive all twelve immutable inputs:

1. frozen Attempt02 v4 `model.pkl`;
2. Attempt02 v4 `training_manifest.json`;
3. sealed Attempt02 `data_contract.json`;
4. pre-holdout `freeze_manifest.json`;
5. one-shot `locked_receipt.json`;
6. canonical global `M43_LOCKED_CONSUMED.json`;
7. `configs/hu_joint_policy_m43_population.json`;
8. merged fresh population `evaluation.json`;
9. merged `records.jsonl`;
10. `merge_manifest.json`;
11. verified Spot `receipt.json`;
12. frozen local `population_run_manifest.json` for the same run.

The final gates remain: zero invalid/non-fire mismatches; at least 300 valid
overrides; realized gain/override CI95 low above zero; paired seat-swap and
second-seat delta CI95 lows above zero; exact first-seat cancellation; false
positive rate at most 0.30; loss p95/p99/max at most 25/40/50; and every
opponent point/CI low at least -0.005/-0.02. Teacher metrics and top-1 accuracy
remain diagnostic only.

## Commands after a positive one-shot pilot

Use a unique run name and the actual frozen artifact paths:

```powershell
$run = "regular-hu-m43-population-20260713-xxxx"
.\scripts\Start-GcpHuM43Attempt02PopulationRun.ps1 `
  -RunName $run `
  -ModelPath <model.pkl> `
  -TrainingManifestPath <training_manifest.json> `
  -DataContractPath <data_contract.json> `
  -FreezeManifestPath <freeze_manifest.json> `
  -LockedReceiptPath <locked_receipt.json> `
  -ConsumptionMarkerPath <M43_LOCKED_CONSUMED.json>

# Canary shard.
.\scripts\Start-GcpHuM43Attempt02PopulationRun.ps1 `
  -RunName $run -ResumeExisting -CreateInstances -StartShards 0 `
  -ModelPath <model.pkl> -TrainingManifestPath <training_manifest.json> `
  -DataContractPath <data_contract.json> -FreezeManifestPath <freeze_manifest.json> `
  -LockedReceiptPath <locked_receipt.json> -ConsumptionMarkerPath <M43_LOCKED_CONSUMED.json>

.\scripts\Get-GcpHuM43Attempt02PopulationRunStatus.ps1 -RunName $run

# Fan out the remaining missing shards after the canary is verified.
.\scripts\Start-GcpHuM43Attempt02PopulationRun.ps1 `
  -RunName $run -ResumeExisting -CreateInstances -SkipExistingInstances `
  -ModelPath <model.pkl> -TrainingManifestPath <training_manifest.json> `
  -DataContractPath <data_contract.json> -FreezeManifestPath <freeze_manifest.json> `
  -LockedReceiptPath <locked_receipt.json> -ConsumptionMarkerPath <M43_LOCKED_CONSUMED.json>

.\scripts\Receive-GcpHuM43Attempt02PopulationRun.ps1 -RunName $run
```

Then run the immutable decision:

```powershell
$out = "outputs/hu_joint_policy/m43_attempt02_population/$run"
$env:PYTHONPATH = (Resolve-Path src).Path
python -m ofc_regular.validate_hu_m43_attempt02_acceptance accept `
  --model <model.pkl> `
  --training-manifest <training_manifest.json> `
  --data-contract <data_contract.json> `
  --freeze-manifest <freeze_manifest.json> `
  --locked-receipt <locked_receipt.json> `
  --consumption-marker <M43_LOCKED_CONSUMED.json> `
  --population-plan configs/hu_joint_policy_m43_population.json `
  --evaluation "$out/evaluation.json" `
  --records "$out/records.jsonl" `
  --merge-manifest "$out/merge_manifest.json" `
  --spot-receipt "$out/receipt.json" `
  --run-manifest "outputs/gcp_runs/$run/population_run_manifest.json" `
  --config-output "$out/acceptance_config.json" `
  --status-output "$out/acceptance_status.json"
```

Only `status == complete_go` authorizes creating a new explicit opt-in profile.
It still does not authorize changing `current`.
