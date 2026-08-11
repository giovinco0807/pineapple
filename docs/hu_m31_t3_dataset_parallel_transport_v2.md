# M3.1 dataset parallel transport v2

This is a delivery-only replacement for the v1 GCP fanout. It does not
define a new dataset.

Current production decision: **No-Go**. The live C4-family quota is 128
vCPUs, while this transport requires 320 plus a 35-vCPU reserve. M3.1
production therefore uses v1 with 8 VMs and 45 waves. Service accounts
`08..19` must not be created for the selected v1 run.

## Frozen scientific boundary

- `hu_m31_t3_dataset_contract_v1`: unchanged 9,000 paired hands, 360 shards,
  split/seed schedule, and merge.
- `hu_m31_t3_dataset_executor_v1`: unchanged 25-pair shard executor.
- `hu_m31_t3_dataset_portable_worker_v1`: unchanged worker and shard bytes.
- smoke shard `train-0000`: still completed locally before cloud fanout.
- cloud shards: the same remaining 359 v1 shard descriptors and attempt object
  identities.

The v2 plan records hashes of all four v1 source modules and the exact v1
transport plan. Validation fails closed if any scientific source or shard
descriptor changes. The v2 byte-producer and merge symbols are direct aliases
of the v1 functions.

## Parallel change

- maximum concurrent VMs: 20
- machine: one `c4-standard-16` Spot VM per shard
- maximum concurrent vCPUs: 320
- waves: 18, with 20 shards in each of the first 17 waves, then 19
- required post-launch quota reserve: 35 vCPUs

The read-only preflight requires all of:

- OAuth TTL of at least 45 minutes;
- exact existing service-account pool
  `ofc-f100-worker-00..19@ofc-solver-485418.iam.gserviceaccount.com`;
- live regional C4, regional Spot, and global vCPU quota/inventory headroom;
- at least 35 vCPUs left after the selected wave;
- selected VM and disk names absent;
- exact image and Cloud NAT readback.

## Worker service accounts

Only workers `00..07` are currently known to exist. The separate
`hu_m31_t3_dataset_worker_sa_plan_v2` artifact lists `08..19` as missing.
It has `service_account_creation_authorized=false` and performs no cloud
mutation. Provisioning requires a separate authorization and a fresh
read-only inventory receipt. The parallel provider itself never creates a
service account.

## Source relocation

Run preparation inside Linux/WSL. `relocate-ext4` copies every immutable
source plus the smoke shard to a create-only ext4 directory, verifies every
byte, and emits a versioned relocation receipt. All later plan/controller
validation replays the ext4 filesystem type and exact inventory.

```powershell
python scripts/prepare_hu_m31_t3_dataset_gcp_v2.py write-sa-plan `
  --output <run>\worker-sa-provisioning-plan-v2.json

python scripts/prepare_hu_m31_t3_dataset_gcp_v2.py relocate-ext4 `
  <all frozen source arguments> `
  --destination-root <wsl-ext4-run-root> `
  --output <wsl-ext4-run-root>\source-ext4-relocation-v2.json

python scripts/prepare_hu_m31_t3_dataset_gcp_v2.py build-plan `
  <the relocated source arguments> `
  --run-name <run-name> --bucket <bucket> `
  --source-relocation-receipt <receipt> `
  --output <run>\transport-plan-v2.json
```

`run_hu_m31_t3_dataset_gcp_v2.py` then provides `prepare`, `plan-next`,
`preflight-next`, `stage-next`, `install-iam-next`, `execute-next`,
`poll-next`, `cleanup-next`, `receive-next`, and `accept-next`.
Every cloud phase requires the exact run name, explicit allow flag, controller
token, and content-addressed phase sentinel. OAuth and launch nonce values
remain environment-only.

Partial VM creation is atomic at the controller boundary: every selected name
is reconciled, only exact owned instances/disks are deleted, temporary worker
IAM is removed, absence is proven, and a no-go lifecycle opens the second
attempt. Wildcard deletion and unrelated-resource mutation are prohibited.
