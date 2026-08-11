# M3.1 post-training Spot fanout v1

This transport is only for two CPU-heavy workloads after the T3 model exists:

1. real ABR teacher labels: 50 paired-root pilot, then a new exact 250-pair
   production run;
2. locked promotion: 260 immutable work items and 13,000 rows.

It never resolves or edits `current`, edits `ai_profiles.py`, accepts synthetic
ABR values, or gives a worker opponent-private discards.

## Safety boundary

- One ABR pair or one locked work ID is one atomic cloud job.
- A wave has at most eight `c4-standard-16` Spot VMs (128 C4 vCPU).
- A Spot interruption retries only that job under `a01`; `a00` is immutable.
- Inputs and results use create-only GCS writes; the result manifest is last.
- A real job-zero smoke on ext4/xfs is mandatory before cloud authorization.
- The ABR runtime is one create-only bundle. It freezes the full Python runtime
  source, accepted Rust search source, exact 11 real behavior models, exact
  23-wheel offline wheelhouse, accepted engine, and diagnostic engine.
- The packager deserializes all 11 required model fields before creating the
  output. Optional-loader fallback, missing models, and placeholder models fail.
- `ABR_RUNTIME_BUNDLE_READY.json` is written last. Contract construction
  replays its manifest and requires every explicit artifact path to belong to
  that same bundle.
- Accepted and diagnostic ABR native libraries are separately SHA-pinned.
- Locked evaluation separately pins runtime closure, threshold lock,
  policy/model registry, ABR archive, `bundle.json`, and production receipt.
- Each cloud phase needs the exact run name, controller token, and sentinel.
- OAuth bearer and expiry are environment-only. At least 2,700 seconds must
  remain; neither bearer nor bearer hash is stored.
- Cleanup validates exact instance specs/provider IDs, deletes only listed VMs
  and boot disks, proves absence, removes exact temporary IAM, and keeps GCS.
- Lost-launch recovery validates the complete planned VM and IAM identities
  before deleting anything.

## ABR pilot preparation

First build the real runtime closure from WSL. The Ubuntu VHD is stored on
`D:`, so `/home/wner/...` is D-backed ext4. Do not use `/mnt/d/...` as the
output root: that mount is `9p`, not ext4. The raw wheelhouse may remain under
`/mnt/d` because it is an immutable input.

```bash
export REPO=/mnt/c/Users/Owner/.gemini/antigravity/scratch/ofc-pineapple/regular-ofc-pineapple
export BUNDLE=/home/wner/ofc-m31-postdata/abr-runtime-bundle-v1
export DIAGNOSTIC_SHA256=9b84b0b2f28142c1a5b7ad0bbfc1e6b5e48cacd457605f666c9104a8bcf48fd8

PYTHONPATH="$REPO/src" python \
  "$REPO/scripts/run_hu_m31_t3_post_training_gcp_v1.py" \
  package-abr-runtime \
  --repository-root "$REPO" \
  --models-root /home/wner/ofc-m31-postdata/live-snapshot-v2/models \
  --raw-wheelhouse /mnt/d/ofc-gcp-runs/regular-hu-m31-c02-f100wv2-20260723-009/phase-a/outer_package/content/wheelhouse/wheelhouse.zip \
  --raw-wheelhouse-manifest /mnt/d/ofc-gcp-runs/regular-hu-m31-c02-f100wv2-20260723-009/phase-a/outer_package/content/wheelhouse/wheelhouse_manifest.json \
  --accepted-library /home/wner/ofc-m31-fqv1-wsl-20260724-005/bootstrap/libofc_hu_m3_engine.so \
  --diagnostic-library /mnt/d/ofc-m31-postdata-build/abr-diag-v1-20260724-002/target/release/libofc_hu_m3_engine.so \
  --expected-diagnostic-library-sha256 "$DIAGNOSTIC_SHA256" \
  --output-root "$BUNDLE"
```

Successful packaging ends at
`ready_for_real_one_pair_local_smoke`; it does not authorize or launch cloud
work.

Prepare the existing teacher plan:

```bash
python -m ofc_regular.hu_m31_t3_abr_teacher_v1 prepare \
  --run-directory /ext4/abr-pilot-plan \
  --pair-count 50 \
  --accepted-library /ext4/native/accepted/libofc_hu_m3_engine.so \
  --diagnostic-library /ext4/native/diagnostic/libofc_hu_m3_engine.so \
  --expected-diagnostic-library-sha256 "$DIAGNOSTIC_SHA256"
```

Freeze the cloud-neutral contract:

```bash
python scripts/run_hu_m31_t3_post_training_gcp_v1.py build-abr-contract \
  --run-name m31-t3-abr-pilot50-v1 \
  --bucket pokerhu-ofc-solver-485418-training \
  --mode pilot50 \
  --abr-plan /ext4/abr-pilot-plan/PLAN.json \
  --runtime-bundle-manifest "$BUNDLE/ABR_RUNTIME_BUNDLE_MANIFEST.json" \
  --runtime-archive "$BUNDLE/runtime/runtime.tar.gz" \
  --wheelhouse-archive "$BUNDLE/wheelhouse/wheelhouse.zip" \
  --accepted-library "$BUNDLE/native/accepted/libofc_hu_m3_engine.so" \
  --diagnostic-library "$BUNDLE/native/diagnostic/libofc_hu_m3_engine.so" \
  --output /ext4/abr-pilot50/contract.json
```

Run one real pair locally. Both directories must be new and on the same
ext4/xfs filesystem:

```bash
python scripts/run_hu_m31_t3_post_training_gcp_v1.py local-smoke \
  --contract /ext4/abr-pilot50/contract.json \
  --staging-root /ext4/abr-pilot50/smoke-staging \
  --output-root /ext4/abr-pilot50/smoke-output \
  --receipt /ext4/abr-pilot50/smoke-receipt.json
```

The canonical provider config contains the exact Debian image ID and exactly
eight existing worker service accounts:

```json
{"current_profile_changed":false,"guest_os_features":["GVNIC","UEFI_COMPATIBLE"],"image_id":"<image-id>","image_self_link":"https://www.googleapis.com/compute/v1/projects/debian-cloud/global/images/<exact-image>","schema":"hu_m31_t3_post_training_provider_config_v1","worker_service_accounts":["<worker00>","<worker01>","<worker02>","<worker03>","<worker04>","<worker05>","<worker06>","<worker07>"]}
```

Authorize and prepare without contacting GCP:

```bash
python scripts/run_hu_m31_t3_post_training_gcp_v1.py authorize-cloud-plan \
  --contract /ext4/abr-pilot50/contract.json \
  --smoke-receipt /ext4/abr-pilot50/smoke-receipt.json \
  --provider-config /ext4/abr-pilot50/provider-config.json \
  --output /ext4/abr-pilot50/cloud-plan.json

export OFC_M31_POST_TRAINING_CONTROLLER_TOKEN="$(python -c 'import uuid;print(uuid.uuid4())')"
python scripts/run_hu_m31_t3_post_training_gcp_v1.py prepare-controller \
  --cloud-plan /ext4/abr-pilot50/cloud-plan.json \
  --output-root /ext4/abr-pilot50/controller
```

## One cloud wave

Refresh OAuth immediately before every live phase:

```bash
export OFC_M31_POST_TRAINING_OAUTH_TOKEN='<fresh bearer>'
export OFC_M31_POST_TRAINING_OAUTH_EXPIRES_AT_UNIX='<unix expiry>'
python scripts/run_hu_m31_t3_post_training_gcp_v1.py oauth-preflight \
  --expires-at-unix "$OFC_M31_POST_TRAINING_OAUTH_EXPIRES_AT_UNIX"
```

Print and copy the exact phase sentinel, then launch exactly once:

```bash
python scripts/run_hu_m31_t3_post_training_gcp_v1.py sentinel \
  --output-root /ext4/abr-pilot50/controller --phase execute
export OFC_M31_POST_TRAINING_PHASE_SENTINEL='<exact printed sentinel>'
export OFC_M31_POST_TRAINING_LAUNCH_NONCE="$(python -c 'import uuid;print(uuid.uuid4())')"
python scripts/run_hu_m31_t3_post_training_gcp_v1.py execute \
  --output-root /ext4/abr-pilot50/controller \
  --confirm-run-name m31-t3-abr-pilot50-v1 \
  --allow-cloud-mutation
```

Never retry `execute`. Repeat the sentinel step separately for `poll`,
`cleanup`, and `receive`, then accept:

```bash
python scripts/run_hu_m31_t3_post_training_gcp_v1.py poll \
  --output-root /ext4/abr-pilot50/controller \
  --confirm-run-name m31-t3-abr-pilot50-v1 --allow-cloud-read
python scripts/run_hu_m31_t3_post_training_gcp_v1.py cleanup \
  --output-root /ext4/abr-pilot50/controller \
  --confirm-run-name m31-t3-abr-pilot50-v1 --allow-cloud-mutation
python scripts/run_hu_m31_t3_post_training_gcp_v1.py receive \
  --output-root /ext4/abr-pilot50/controller \
  --confirm-run-name m31-t3-abr-pilot50-v1 --allow-cloud-read
python scripts/run_hu_m31_t3_post_training_gcp_v1.py accept \
  --output-root /ext4/abr-pilot50/controller
```

After all 50 jobs:

```bash
python scripts/run_hu_m31_t3_post_training_gcp_v1.py finalize \
  --output-root /ext4/abr-pilot50/controller
```

The 50-pair result is diagnostic only. Build a new `production250` contract,
repeat the real one-pair smoke, then finish all 250 before training an ABR used
by locked promotion.

## Locked promotion contract

After the candidate model, threshold, real production ABR, and runtime closure
exist:

```bash
python scripts/run_hu_m31_t3_post_training_gcp_v1.py build-locked-contract \
  --run-name m31-t3-locked-promotion-v1 \
  --bucket pokerhu-ofc-solver-485418-training \
  --promotion-plan /ext4/locked/promotion-plan.json \
  --execution-plan /ext4/locked/execution-plan.json \
  --runtime-archive /ext4/packages/runtime.tar.gz \
  --wheelhouse-archive /ext4/packages/wheelhouse.zip \
  --closure-package /ext4/locked/runtime-closure.tar.gz \
  --compatibility-threshold-lock /ext4/locked/threshold-lock.json \
  --policy-registry /ext4/locked/policy-registry.json \
  --abr-bundle-archive /ext4/locked/abr_bundle.tar.gz \
  --expected-closure-sha256 "$CLOSURE_SHA256" \
  --expected-abr-bundle-archive-sha256 "$ABR_ARCHIVE_SHA256" \
  --expected-abr-bundle-file-sha256 "$ABR_BUNDLE_JSON_SHA256" \
  --expected-abr-production-build-receipt-sha256 "$ABR_BUILD_RECEIPT_SHA256" \
  --output /ext4/locked/contract.json
```

Run the same real job-zero smoke before Spot. Finalization requires exactly 260
valid shards and 13,000 rows. Scientific closeout and dormant profile
registration remain in the existing post-dataset controller.

## Focused regression

```powershell
pytest -q `
  tests/test_hu_m31_t3_post_training_runtime_v1.py `
  tests/test_hu_m31_t3_post_training_gcp_v1.py `
  tests/test_hu_m31_t3_abr_teacher_v1.py `
  tests/test_hu_m31_t3_abr_cli_v1.py `
  tests/test_hu_m31_t3_post_dataset_controller_v1.py `
  tests/test_hu_m31_t3_locked_promotion_execution_v1.py `
  tests/test_hu_m31_t3_locked_promotion_cli_v1.py `
  tests/test_hu_m31_t3_locked_promotion_runner_v1.py
```

No command here changes `current` or registers a named profile.
