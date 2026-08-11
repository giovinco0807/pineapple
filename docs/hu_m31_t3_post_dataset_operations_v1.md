# M3.1 post-dataset operations v1

This document describes the executable path after the immutable 9,000-paired
dataset has passed its merge gate. It does not authorize cloud work, change
`current`, or register a profile before the locked population/ABR gate passes.

## Readiness boundary

Implemented and source-replayed:

- CUDA/CPU `StreetPolicyNetV1` training with one-epoch atomic checkpoints;
- restart from the last complete core/risk epoch;
- threshold-lock and diagnostic holdout without weight or threshold research;
- rich-threshold to locked-promotion compatibility bridge;
- content-addressed runtime closure containing model, source, legacy models,
  and exact-T4 native runtime;
- deterministic three-checkpoint ABR distillation and artifact-bound loaders;
- exact 260-item / 13,000-row locked evaluation grid;
- pending-shard inspection, bounded resume, atomic shard publication, exact
  merge/gate closeout, and dormant opt-in authorization.

Implemented but not yet executed at production scale:

- the real `abr_development` search-teacher generator and resumable
  shard/merge controller. It requires the accepted Candidate02 binary plus a
  separately hash-pinned diagnostic binary whose aggregate terminal
  decomposition is bit-exact with every accepted action Q. A 50-pair pilot
  must pass before the exact 250-pair run; hand-written/synthetic values are
  rejected by the production CLI.

## 1. CUDA training

Create and externally freeze the run config once:

```powershell
$env:PYTHONPATH = "src"
python -m ofc_regular.hu_m31_t3_street_policy_training_cli_v1 `
  write-default-config `
  --output D:\ofc-m31\post-dataset\street-policy-run-config.json
```

Calculate and record its SHA-256, then train explicitly on the local GPU:

```powershell
python -m ofc_regular.hu_m31_t3_street_policy_training_cli_v1 `
  train `
  --plan D:\ofc-m31\dataset\dataset-plan.json `
  --merge D:\ofc-m31\dataset\dataset-merge.json `
  --expected-merge-sha256 <dataset-merge-file-sha256> `
  --shard-map D:\ofc-m31\dataset\absolute-shard-map.json `
  --config D:\ofc-m31\post-dataset\street-policy-run-config.json `
  --expected-config-sha256 <run-config-file-sha256> `
  --output-root D:\ofc-m31\post-dataset\street-policy-run `
  --device cuda:0
```

Reissuing the same command replays every completed epoch and resumes at the
first absent epoch. A changed dataset, config, device contract, checkpoint, or
receipt fails closed.

The promotion inputs are:

- final bundle:
  `street-policy-run\risk_epoch_<risk_epochs>\bundle`;
- rich threshold:
  `street-policy-run\calibration\threshold_lock.json`;
- diagnostic-only report:
  `street-policy-run\calibration\diagnostic_report.json`.

## 2. Threshold bridge, runtime closure, and promotion plan

Create the compact compatibility lock:

```powershell
python -m ofc_regular.hu_m31_t3_locked_promotion_artifact_bridge_v1 `
  --checkpoint-bundle-directory <final-risk-bundle> `
  --training-threshold-lock <rich-threshold-lock> `
  --model-artifact-id street-policy-net-v1-m31-t3 `
  --output-threshold-lock D:\ofc-m31\post-dataset\promotion-threshold-lock.json
```

Create the runtime closure with the accepted host-target exact-T4 library:

```powershell
python -m ofc_regular.hu_m31_t3_promotion_runtime_closure_v1 `
  create `
  --repository-root <repository-root> `
  --candidate-checkpoint-bundle <final-risk-bundle> `
  --training-threshold-lock <rich-threshold-lock> `
  --exact-t4-native-library <accepted-exact-t4-library> `
  --output D:\ofc-m31\post-dataset\runtime-closure.tar
```

Build the locked promotion plan with
`hu_m31_t3_locked_promotion_cli_v1 build-plan`, pinning the model manifest,
compatibility lock, unchanged `src/ofc_regular/ai_profiles.py`, and runtime
closure file hashes. Then prepare the 260-item execution plan:

```powershell
python -m ofc_regular.hu_m31_t3_locked_promotion_execution_v1 `
  prepare `
  --promotion-plan D:\ofc-m31\post-dataset\promotion-plan.json `
  --output D:\ofc-m31\post-dataset\execution-plan.json
```

## 3. Three real ABR checkpoints

After the disjoint real search-teacher run freezes its canonical raw example
object, normalize it:

```powershell
python scripts/run_hu_m31_t3_abr_v1.py freeze-dataset `
  --promotion-plan D:\ofc-m31\post-dataset\promotion-plan.json `
  --expected-promotion-plan-sha256 <plan-file-sha256> `
  --candidate-bundle-directory <final-risk-bundle> `
  --expected-candidate-manifest-sha256 <manifest-file-sha256> `
  --raw-examples D:\ofc-m31\abr\raw-search-teacher.json `
  --expected-raw-examples-sha256 <raw-file-sha256> `
  --teacher-receipt D:\ofc-m31\abr\teacher-receipt.json `
  --expected-teacher-receipt-sha256 <teacher-receipt-file-sha256> `
  --output D:\ofc-m31\abr\development-dataset.json
```

Atomically train/freeze all three artifact-bound checkpoints:

```powershell
python scripts/run_hu_m31_t3_abr_v1.py train-bundle `
  --promotion-plan D:\ofc-m31\post-dataset\promotion-plan.json `
  --expected-promotion-plan-sha256 <plan-file-sha256> `
  --candidate-bundle-directory <final-risk-bundle> `
  --expected-candidate-manifest-sha256 <manifest-file-sha256> `
  --development-dataset D:\ofc-m31\abr\development-dataset.json `
  --expected-development-dataset-sha256 <dataset-file-sha256> `
  --raw-examples D:\ofc-m31\abr\raw-search-teacher.json `
  --expected-raw-examples-sha256 <raw-file-sha256> `
  --teacher-receipt D:\ofc-m31\abr\teacher-receipt.json `
  --expected-teacher-receipt-sha256 <teacher-receipt-file-sha256> `
  --output-directory D:\ofc-m31\abr\three-policy-bundle
```

If the final bundle already exists, the command validates every checkpoint,
manifest, source model, semantic probe, and hash and returns
`resume_complete`. It also replays the exact 250-pair/500-root, seat-balanced,
three-family teacher lineage and writes
`production_build_receipt.json`. It never silently completes a partial
destination.

The runtime composition bound into that receipt is:

- T0: explicit `stage19_p0`;
- T1: its pinned `stage18_p1` continuation;
- T2: its pinned `stage9f_p2` continuation;
- T3: the selected artifact-bound learned response;
- T4: the accepted exact solver.

Only `greedy_search_response` is interpreted as a direct HU-score approximate
best response. `foul_pressure_response` and `royalty_denial_response` are
exploitative stress families. Their combined floor is robustness evidence,
not a NashConv or exploitability upper bound.

## 4. Resumable 260-shard evaluation

Read-only status:

```powershell
python scripts/run_hu_m31_t3_post_dataset_controller_v1.py status `
  --promotion-plan D:\ofc-m31\post-dataset\promotion-plan.json `
  --execution-plan D:\ofc-m31\post-dataset\execution-plan.json `
  --shard-directory D:\ofc-m31\locked-eval\shards
```

`run-pending` builds the real artifact-bound candidate, five named opponents,
three frozen ABRs, and exact-T4 runtime once, then runs only the next bounded
pending items. Each completed shard becomes visible with one create-only hard
link, so interruption cannot leave a partial JSON file in the shard directory.
Use `--work-id` to assign deterministic disjoint work items to independent
workers, or omit it to consume plan order.

`run-pending` additionally requires
`--expected-abr-production-build-receipt-sha256`; this prevents a synthetic
unit fixture or an incomplete pilot bundle from entering locked evaluation.

No cloud call is made by this controller. A cloud launcher may invoke this
same command inside a prepared worker, but cloud lifecycle and transport need
their own explicit authorization.

## 5. Merge, gate, and opt-in boundary

After status reports exactly 260 complete items and 13,000 accepted rows:

```powershell
python scripts/run_hu_m31_t3_post_dataset_controller_v1.py closeout `
  --promotion-plan D:\ofc-m31\post-dataset\promotion-plan.json `
  --execution-plan D:\ofc-m31\post-dataset\execution-plan.json `
  --shard-directory D:\ofc-m31\locked-eval\shards `
  --merge-output D:\ofc-m31\locked-eval\merge.json `
  --gate-output D:\ofc-m31\locked-eval\gate.json
```

Only a fully passing gate can create the dormant registration authorization:

```powershell
python scripts/run_hu_m31_t3_post_dataset_controller_v1.py authorize-opt-in `
  --promotion-plan D:\ofc-m31\post-dataset\promotion-plan.json `
  --execution-plan D:\ofc-m31\post-dataset\execution-plan.json `
  --shard-directory D:\ofc-m31\locked-eval\shards `
  --merge D:\ofc-m31\locked-eval\merge.json `
  --gate D:\ofc-m31\locked-eval\gate.json `
  --output D:\ofc-m31\locked-eval\opt-in-authorization.json
```

That artifact authorizes exactly
`stage7_m31_street_policy_v1_opt_in_candidate`; it does not edit
`ai_profiles.py`, change `current`, activate runtime use, or enable full
replacement. The one explicit registry change is a separate final action after
reviewing the real gate.
