# M3.1 T3 dataset production readiness

## Transport decision

The selected production transport is the existing v1 schedule:

- 8 `c4-standard-16` Spot VMs at most;
- 128 vCPUs at most;
- 359 cloud shards in 45 waves after the local `train-0000` smoke shard;
- unchanged v1 dataset contract, shard executor, portable worker, and merge;
- existing worker service accounts `ofc-f100-worker-00` through `07`.

The live attempt003 quota receipt reported C4-family `128/0`, Spot
`468/0`, and global vCPU `512/0` in `asia-northeast1`. Therefore parallel-20
v2 needs 355 C4 vCPUs including its 35-vCPU reserve, has a 227-vCPU deficit,
and is No-Go under the current quota. Accounts `08` through `19` are not
needed and their creation remains unauthorized.

The v2 implementation remains a future transport only. It must not be used
until a fresh live quota and exact 20-account inventory pass its existing
read-only preflight.

## Supervisor boundary

`run_hu_m31_t3_dataset_supervisor_v1.py` wraps the existing v1 controller
without changing its scientific or cloud contracts. It holds a create-only
exclusive lock and processes one wave at a time:

1. replay the pinned source hashes, including `ai_profiles.py`;
2. force-refresh a non-persisted OAuth token through
   `gcloud config config-helper --force-auth-refresh --format=json` and require
   TTL at least 2,700 s;
3. execute the current wave once through the idempotent controller;
4. poll for a bounded number of observations;
5. delete only exact owned VMs/disks and remove only exact temporary IAM;
6. source-replay received objects;
7. write an immutable, source-replayed checkpoint intent;
8. accept the lifecycle into the append-only attempt ledger;
9. write an immutable supervisor checkpoint before continuing.

An incomplete wave is cleaned and recorded as a failed attempt, so the v1
ledger selects the next attempt for the same shards. A partial launch is
cleaned and stops. The supervisor refuses to auto-retry that contaminated
controller; recovery requires an explicit new controller decision. A stale
`ACTIVE_SUPERVISOR.lock` also fails closed and is never removed automatically.

The nominal schedule is 45 waves. Because v1 permits two attempts per shard,
the unattended safety cap is 90 accepted lifecycles; it stops earlier at
`complete` or immediately at `no_go_attempts_exhausted`. This prevents a
single Spot retry from silently ending `run-all` after only 45 lifecycles.

If the process stops before acceptance, the next invocation replays the
intent and all referenced receipts, then performs acceptance without reopening
cloud access. If the controller ledger was committed but the final supervisor
checkpoint was not written, the next invocation backfills that checkpoint and
returns without launching the next wave. Tampered intent inputs fail closed.

No access token is written to a receipt. The supervisor never changes
`current`, never creates service accounts, never deletes by wildcard, and
never launches more than one eight-VM wave.

The force refresh runs before every wave launch. Poll, cleanup, and receive
reuse the memory-only lease and refresh it again only when less than 900 s
remain.

Before cloud access, every wave now writes a create-only compute-cost
reservation. The supervisor uses integer micro-USD, a conservative
`$0.500000` per VM-hour guard, and the existing six-hour VM maximum. After
exact VM/disk absence and temporary-IAM removal, it settles the reservation
using the pre-launch-to-cleanup interval. Unsettled reservations retain their
full six-hour charge on resume. A permanent `$25` non-compute reserve leaves
`$475` for compute, so the total M3.1 guard cannot exceed `$500`. Launch must
stop if the next reservation would cross the cap. The operator must recheck
the live Spot price is no greater than `$0.500000` and confirm both frozen
values on the command line.

The relocated Ubuntu VHD is physically D-backed, but the production
controller must still use native ext4 paths under `/home/wner/...` for temp,
staging, source replay, supervisor state, and shard work. `/mnt/d` is not a
production work filesystem: it is used only for create-only export of final
receipts and final artifacts. This avoids the 9p metadata and small-file
penalty while keeping the large VHD off C:.

## Exact attempt005-to-dataset command sequence

`regular-hu-m31-t3-fqv1-wsl-20260724-005` is the only active
fresh-quality candidate for this sequence. This name is not evidence of a
pass. Dataset production remains No-Go until all 15 fresh-quality jobs are
accepted, owned compute/disks and temporary IAM are absent, the same-Linux
closeout finishes, and its source-replayed 25-paired smoke gate passes.

Do not substitute `attempt004`, and do not manually point `prepare` at a set
of loose artifacts. `hu_m31_t3_same_linux_closeout_v1` owns the one valid
preparation path: it rebuilds the fresh-quality gate, runs the 25-paired smoke,
and prepares the 359-shard controller on the same persistent Linux
filesystem. The raw controller `prepare` wrapper is disabled.

Run the following cloud-neutral closeout in that native Linux controller
context. Every path below must be an absolute regular path inside
`controller_filesystem_root`; fill the placeholders from the accepted
attempt005 lifecycle only.

```bash
set -euo pipefail

repo='<absolute attempt005 repository path inside the controller filesystem>'
controller_filesystem_root='<absolute persistent Linux controller root>'
closeout_root="$controller_filesystem_root/dataset-closeout-v1"
fresh_quality_run_name='regular-hu-m31-t3-fqv1-wsl-20260724-005'
dataset_run_name='regular-hu-m31-t3-dataset-v1-20260724-001'
bucket='pokerhu-ofc-solver-485418-training'

fresh_quality_gcp_plan='<absolute accepted attempt005 bridge_plan.json>'
fresh_quality_ledger='<absolute accepted attempt005 attempt_ledger.json>'
accepted_results='<absolute directory containing all 15 accepted results>'
fresh_quality_provider_plan='<absolute attempt005 provider_plan.json>'
provider_config="$controller_filesystem_root/provider-config-v1.json"
local_shards="$closeout_root/fanout_shards"

cd "$repo"
python scripts/prepare_hu_m31_t3_dataset_gcp_v1.py \
  write-provider-config-from-fq-plan \
  --fresh-quality-provider-plan "$fresh_quality_provider_plan" \
  --output "$provider_config"

export OFC_M31_DATASET_CONTROLLER_TOKEN="$(
  python -c 'import uuid; print(uuid.uuid4())'
)"
python -m ofc_regular.hu_m31_t3_same_linux_closeout_v1 \
  --controller-filesystem-root "$controller_filesystem_root" \
  --output-root "$closeout_root" \
  --repository-root "$repo" \
  --fresh-quality-gcp-plan "$fresh_quality_gcp_plan" \
  --fresh-quality-ledger "$fresh_quality_ledger" \
  --accepted-results-directory "$accepted_results" \
  --dataset-run-name "$dataset_run_name" \
  --dataset-bucket "$bucket" \
  --provider-config "$provider_config"
```

Freeze the three independently supplied file digests after closeout. The
binding command replays the closeout, the retained attempt005 plan, both
gates, the completed `train-0000` shard, the transport plan, and controller
contract. It fails closed on a run-name mismatch, missing/non-pass receipt,
or digest change.

```bash
ready="$closeout_root/SAME_LINUX_CLOSEOUT_READY.json"
smoke_gate="$closeout_root/dataset/dataset_smoke_gate.json"
binding_root="$controller_filesystem_root/dataset-production-binding-v1"
binding="$binding_root/dataset-source-binding.json"
source_receipt="$binding_root/supervisor-source-receipt.json"
supervisor_root="$controller_filesystem_root/dataset-supervisor-v1"
controller_root="$closeout_root/controller"

test -f "$ready"
test -f "$smoke_gate"
test -f "$fresh_quality_gcp_plan"
mkdir -p "$binding_root" "$supervisor_root"

closeout_file_sha256="$(sha256sum "$ready" | awk '{print $1}')"
fresh_quality_plan_file_sha256="$(
  sha256sum "$fresh_quality_gcp_plan" | awk '{print $1}'
)"
smoke_gate_file_sha256="$(sha256sum "$smoke_gate" | awk '{print $1}')"

python scripts/prepare_hu_m31_t3_dataset_gcp_v1.py \
  write-source-binding \
  --same-linux-closeout "$ready" \
  --expected-closeout-file-sha256 "$closeout_file_sha256" \
  --expected-fresh-quality-plan-file-sha256 \
    "$fresh_quality_plan_file_sha256" \
  --expected-smoke-gate-file-sha256 "$smoke_gate_file_sha256" \
  --expected-fresh-quality-run-name "$fresh_quality_run_name" \
  --expected-dataset-run-name "$dataset_run_name" \
  --output "$binding"

python scripts/run_hu_m31_t3_dataset_supervisor_v1.py source-check \
  --output "$source_receipt"
python scripts/run_hu_m31_t3_dataset_gcp_v1.py plan-next \
  --output-root "$controller_root"
python scripts/run_hu_m31_t3_dataset_gcp_v1.py status \
  --output-root "$controller_root"
```

Everything above is cloud-neutral. The following is the sole production
mutation command. The supervisor requires the immutable source binding and
replays it before opening cloud access. Run it only after separately
reviewing `plan-next`, the source receipt, the v1 selection receipt, and fresh
live inventory:

```bash
python scripts/run_hu_m31_t3_dataset_supervisor_v1.py run-all \
  --controller-root "$controller_root" \
  --supervisor-root "$supervisor_root" \
  --confirm-run-name "$dataset_run_name" \
  --source-binding "$binding" \
  --confirm-total-cost-cap-usd 500.000000 \
  --confirm-spot-rate-guard-usd-per-vm-hour 0.500000 \
  --max-poll-attempts 720 \
  --poll-interval-seconds 30 \
  --allow-cloud-mutation
```

The terminal steps are local and source-replay the complete 360-shard set:

```bash
python scripts/run_hu_m31_t3_dataset_gcp_v1.py status \
  --output-root "$controller_root"
python scripts/run_hu_m31_t3_dataset_gcp_v1.py finalize \
  --output-root "$controller_root"
```

`finalize` must remain closed unless `resume_status` is `complete`,
`complete_shard_count` is 359 for cloud shards, and
`exhausted_shard_count` is zero.

## Current readiness result

- v1 selection logic: implemented;
- v2 No-Go under C4=128: implemented and tested;
- account `08..19` creation authorization: false;
- cloud-neutral v1 provider-config generator: implemented;
- source/current hash receipt: implemented;
- same-Linux closeout/attempt005/dataset-run/25-pair exact source binding:
  implemented;
- production supervisor requires that binding before cloud access:
  implemented;
- loose-artifact raw controller preparation through the production wrapper:
  disabled;
- one-wave supervisor, bounded poll, exact cleanup, receive, accept, and
  checkpoint: implemented;
- local fake-cloud tests cover low TTL, exclusive lock, incomplete cleanup,
  partial launch cleanup, unwritten-but-exact IAM reconciliation, orphan disk
  cleanup, abort retry refusal, pre-accept crash recovery,
  post-ledger/pre-checkpoint recovery, and tampered recovery evidence;
- attempt003-backed cloud-neutral selection receipt:
  `D:\ofc-gcp-runs\regular-hu-m31-dataset-transport-readiness-20260724-001\transport-selection-v1.json`,
  file SHA-256
  `8c8ac62b16684c44592228bf08e9a021ef844460c6d58543ee32d96e2d3f0fd3`;
- the earlier pinned source receipt with file SHA-256
  `bcbb8076c90d6428743e2df564a76579ba414f72c64a6a46bdc3bcb9a746f27d`
  predates the source-binding gate and is superseded; regenerate it before
  launch;
- live dataset launch: intentionally not started;
- observed attempt005 local snapshot at this audit had zero accepted jobs and
  no `SAME_LINUX_CLOSEOUT_READY.json`;
- remaining external prerequisites: attempt005 terminal accepted closeout,
  passed 25-pair smoke, and a successfully source-replayed binding receipt.
