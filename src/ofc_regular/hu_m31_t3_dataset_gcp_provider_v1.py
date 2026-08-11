"""Live GCP provider for the M3.1 359-shard post-smoke fanout.

All mutations are behind an injected transport.  Plans are cloud-neutral and
source-replayed by :mod:`hu_m31_t3_dataset_gcp_transport_v1`; this module only
stages exact bytes, installs wave-scoped worker IAM, creates at most eight
exact C4 Spot instances, observes create-only heartbeat/checkpoint objects,
deletes only measured owned instances, proves VM/disk absence, removes IAM,
and hands an object mirror to the independent receiver.

Importing or building a provider plan performs no cloud operation.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import os
import re
import time
import uuid
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import hu_m31_t3_dataset_contract_v1 as dataset
from . import hu_m31_t3_dataset_gcp_transport_v1 as bridge
from . import hu_m31_t3_step6d_fresh_quality_gcp_provider_v1 as quality_provider
from . import hu_rl_c4_gcp_lifecycle as c4_gcp


PROVIDER_PLAN_SCHEMA = "hu_m31_t3_dataset_gcp_provider_plan_v1"
STAGE_RECEIPT_SCHEMA = "hu_m31_t3_dataset_gcp_stage_receipt_v1"
IAM_RECEIPT_SCHEMA = "hu_m31_t3_dataset_gcp_worker_iam_v1"
LAUNCH_RECEIPT_SCHEMA = "hu_m31_t3_dataset_gcp_launch_receipt_v1"
POLL_RECEIPT_SCHEMA = "hu_m31_t3_dataset_gcp_poll_receipt_v1"
IAM_CLEANUP_SCHEMA = "hu_m31_t3_dataset_gcp_worker_iam_cleanup_v1"

PROJECT = quality_provider.PROJECT
REGION = quality_provider.REGION
ZONE = quality_provider.ZONE
MACHINE_TYPE = quality_provider.MACHINE_TYPE
BOOT_DISK_TYPE = quality_provider.BOOT_DISK_TYPE
BOOT_DISK_INTERFACE = quality_provider.BOOT_DISK_INTERFACE
BOOT_DISK_SIZE_GB = quality_provider.BOOT_DISK_SIZE_GB
NETWORK = quality_provider.NETWORK
SUBNETWORK = quality_provider.SUBNETWORK
NIC_TYPE = quality_provider.NIC_TYPE
OAUTH_SCOPE = quality_provider.OAUTH_SCOPE
NAT_ROUTER_NAME = quality_provider.NAT_ROUTER_NAME
NAT_NAME = quality_provider.NAT_NAME

MAX_RUN_SECONDS = 21_600
WATCHDOG_SECONDS = 21_300
IAM_TTL_SECONDS = 25_200
ABSENCE_POLL_ATTEMPTS = 60
ABSENCE_POLL_INTERVAL_SECONDS = 2.0

_SHA = re.compile(r"^[0-9a-f]{64}$")
_PROVIDER_ID = re.compile(r"^[1-9][0-9]*$")
_SERVICE_ACCOUNT = re.compile(
    rf"^[a-z][a-z0-9-]{{4,28}}[a-z0-9]@{re.escape(PROJECT)}"
    r"\.iam\.gserviceaccount\.com$"
)
_IMAGE_LINK = re.compile(
    r"^https://www\.googleapis\.com/compute/v1/projects/"
    r"debian-cloud/global/images/[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?$"
)


class DatasetProviderError(RuntimeError):
    """Live cloud state is incomplete, ambiguous, or outside the plan."""


class DatasetCloudTransport(Protocol):
    def get_object_metadata(
        self, *, bucket: str, object_name: str
    ) -> Mapping[str, Any] | None: ...

    def get_object_bytes(
        self, *, bucket: str, object_name: str, generation: str | None = None
    ) -> bytes | None: ...

    def put_object_new(
        self, *, bucket: str, object_name: str, payload: bytes, content_type: str
    ) -> Mapping[str, Any]: ...

    def get_instance(self, *, instance_name: str) -> Mapping[str, Any] | None: ...

    def create_instance(
        self, *, instance_spec: Mapping[str, Any], request_id: str
    ) -> Mapping[str, Any]: ...

    def get_zone_operation(self, *, operation_name: str) -> Mapping[str, Any]: ...

    def delete_instance(
        self, *, instance_name: str, request_id: str
    ) -> Mapping[str, Any]: ...

    def get_disk_optional(
        self, *, disk_name: str
    ) -> Mapping[str, Any] | None: ...

    def get_bucket_iam_policy(self, *, bucket: str) -> Mapping[str, Any]: ...

    def set_bucket_iam_policy(
        self, *, bucket: str, policy: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def get_service_account(self, *, email: str) -> Mapping[str, Any]: ...

    def test_service_account_act_as(self, *, email: str) -> bool: ...

    def get_region_quota(self, *, region: str) -> Mapping[str, Any]: ...

    def get_image(self, *, self_link: str) -> Mapping[str, Any]: ...

    def get_router(
        self, *, region: str, router_name: str
    ) -> Mapping[str, Any]: ...

    def get_cloud_quota(self, *, quota_id: str) -> Mapping[str, Any]: ...

    def list_instances(self) -> Sequence[Mapping[str, Any]]: ...

    def get_machine_type_url(
        self, *, self_link: str
    ) -> Mapping[str, Any]: ...


class GcpDatasetRestAdapter(quality_provider.GcpQualityRestAdapter):
    """Concrete REST adapter; OAuth credentials remain only in memory."""


def canonical_bytes(value: Any) -> bytes:
    return bridge.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return bridge.canonical_sha256(value)


def _self_digest(value: Mapping[str, Any], field: str) -> str:
    copied = deepcopy(dict(value))
    copied.pop(field, None)
    return canonical_sha256(copied)


def _uuid_for(*parts: str) -> str:
    raw = hashlib.sha256("\0".join(parts).encode("ascii")).hexdigest()
    return str(uuid.UUID(raw[:32], version=4))


def _utc(epoch_seconds: int) -> str:
    return (
        dt.datetime.fromtimestamp(epoch_seconds, tz=dt.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )


def _plain_file(path: str | Path, label: str) -> Path:
    source = Path(path).resolve()
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    return source


def _content_type(path: Path) -> str:
    if path.suffix == ".json":
        return "application/json"
    if path.suffix in {".gz", ".zip", ".whl"}:
        return "application/octet-stream"
    return "application/octet-stream"


def _content_entry(
    *,
    kind: str,
    path: Path,
    object_name: str,
    shard_id: str | None,
    relative_path: str,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "local_path": str(path),
        "object_name": object_name,
        "sha256": bridge.sha256_file(path),
        "bytes": path.stat().st_size,
        "content_type": _content_type(path),
        "shard_id": shard_id,
        "relative_path": relative_path,
    }


def _resume_files(
    *,
    bridge_plan: Mapping[str, Any],
    selected: Mapping[str, Any],
    local_shard_root: Path,
) -> list[Path]:
    count = selected["resume_completed_pair_count"]
    if count == 0:
        return []
    directory = local_shard_root / selected["shard_id"]
    resume = dataset.inspect_shard_resume(
        plan=dataset.build_dataset_plan(),
        shard_id=selected["shard_id"],
        shard_directory=directory,
    )
    if (
        resume["completed_pair_count"] != count
        or resume["done_present"]
        or not resume["safe_to_resume"]
    ):
        raise ValueError("local resume shard differs from attempt ledger")
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if any(path.is_symlink() for path in directory.rglob("*")):
        raise ValueError("local resume shard contains a symlink")
    return files


def build_provider_plan(
    *,
    transport_plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    wave_request: Mapping[str, Any],
    image_self_link: str,
    image_id: str,
    guest_os_features: Sequence[str],
    worker_service_accounts: Sequence[str],
    local_shard_root: str | Path,
) -> dict[str, Any]:
    plan = bridge.validate_transport_plan(transport_plan, replay_sources=True)
    checked_ledger = bridge.validate_attempt_ledger(plan, ledger)
    checked_resume = bridge.validate_resume_plan(plan, checked_ledger, resume)
    request = bridge.validate_wave_request(
        plan, checked_ledger, checked_resume, wave_request
    )
    if (
        plan["project"] != PROJECT
        or plan["region"] != REGION
        or plan["zone"] != ZONE
        or _IMAGE_LINK.fullmatch(image_self_link) is None
        or _PROVIDER_ID.fullmatch(str(image_id)) is None
    ):
        raise ValueError("dataset provider GCP target or image changed")
    features = sorted(set(guest_os_features))
    accounts = list(worker_service_accounts)
    if (
        len(accounts) != bridge.MAX_CONCURRENT_VMS
        or len(set(accounts)) != len(accounts)
        or any(_SERVICE_ACCOUNT.fullmatch(email) is None for email in accounts)
        or not features
        or len(features) != len(guest_os_features)
    ):
        raise ValueError("dataset provider requires eight unique safe workers")
    selected = request["selected_attempts"]
    workers = [
        {
            "slot": index,
            "shard_id": row["shard_id"],
            "attempt_id": row["attempt_id"],
            "instance_id": row["instance_id"],
            "service_account": accounts[index],
        }
        for index, row in enumerate(selected)
    ]
    base = (
        f"m31-dataset/{plan['execution_identity_sha256']}/content/"
        f"{plan['source_identity']['content_sha256'][:20]}/"
    )
    entries = []
    for record in plan["content_sources"]:
        path = _plain_file(record["source_path"], record["kind"])
        entries.append(
            _content_entry(
                kind=record["kind"],
                path=path,
                object_name=f"{base}static/{record['kind']}/{record['filename']}",
                shard_id=None,
                relative_path=f"static/{record['kind']}/{record['filename']}",
            )
        )
    local_root = Path(local_shard_root).resolve()
    for row in selected:
        for path in _resume_files(
            bridge_plan=plan,
            selected=row,
            local_shard_root=local_root,
        ):
            directory = local_root / row["shard_id"]
            relative = path.relative_to(directory).as_posix()
            entries.append(
                _content_entry(
                    kind="resume_file",
                    path=path,
                    object_name=(
                        f"{base}resume/{row['shard_id']}/"
                        f"{row['resume_completed_pair_count']:06d}/{relative}"
                    ),
                    shard_id=row["shard_id"],
                    relative_path=f"resume/{relative}",
                )
            )
    titles = [
        f"ofc-ds-read-{plan['execution_identity_sha256'][:12]}-"
        f"w{request['wave_index']:02d}",
        *[
            f"ofc-ds-create-{plan['execution_identity_sha256'][:12]}-"
            f"w{request['wave_index']:02d}-s{index:02d}"
            for index in range(len(workers))
        ],
    ]
    startup = worker_startup_script()
    core = {
        "schema": PROVIDER_PLAN_SCHEMA,
        "status": "provider_plan_ready_cloud_not_mutated",
        "transport_plan_sha256": plan["plan_sha256"],
        "ledger_sha256": checked_ledger["ledger_sha256"],
        "resume_sha256": checked_resume["resume_sha256"],
        "wave_request_sha256": request["request_sha256"],
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_index": request["wave_index"],
        "project": PROJECT,
        "region": REGION,
        "zone": ZONE,
        "bucket": plan["bucket"],
        "image": {
            "self_link": image_self_link,
            "id": image_id,
            "guest_os_features": features,
        },
        "network": {
            "network": NETWORK,
            "subnetwork": SUBNETWORK,
            "nic_type": NIC_TYPE,
            "external_ipv4": False,
            "cloud_nat_required": True,
            "nat_router_name": NAT_ROUTER_NAME,
            "nat_name": NAT_NAME,
        },
        "workers": workers,
        "content_entries": entries,
        "local_shard_root": str(local_root),
        "claim_object": (
            f"m31-dataset/{plan['execution_identity_sha256']}/claims/"
            f"{request['request_sha256']}.json"
        ),
        "iam_contract": {
            "viewer_role": "roles/storage.objectViewer",
            "creator_role": "roles/storage.objectCreator",
            "members": [
                f"serviceAccount:{row['service_account']}" for row in workers
            ],
            "reader_prefix": base,
            "condition_titles": titles,
            "ttl_seconds": IAM_TTL_SECONDS,
        },
        "runtime_contract": {
            "machine_type": MACHINE_TYPE,
            "vcpus_per_vm": bridge.VCPUS_PER_VM,
            "max_vm_count": len(workers),
            "max_concurrent_vms": bridge.MAX_CONCURRENT_VMS,
            "max_run_seconds": MAX_RUN_SECONDS,
            "watchdog_seconds": WATCHDOG_SECONDS,
            "boot_disk_type": BOOT_DISK_TYPE,
            "boot_disk_interface": BOOT_DISK_INTERFACE,
            "boot_disk_size_gb": BOOT_DISK_SIZE_GB,
            "provisioning_model": "SPOT",
            "instance_termination_action": "DELETE",
            "startup_has_network_install": False,
            "checkpoint_pair_granularity": 1,
        },
        "selected_attempts": deepcopy(selected),
        "startup_script_sha256": hashlib.sha256(
            startup.encode("utf-8")
        ).hexdigest(),
        "quality_and_smoke_source_replayed": True,
        "cloud_launch_authorized": True,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return {**core, "provider_plan_sha256": canonical_sha256(core)}


def validate_provider_plan(
    value: Mapping[str, Any],
    *,
    transport_plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    wave_request: Mapping[str, Any],
) -> dict[str, Any]:
    plan = bridge.validate_transport_plan(transport_plan, replay_sources=True)
    checked_ledger = bridge.validate_attempt_ledger(plan, ledger)
    checked_resume = bridge.validate_resume_plan(plan, checked_ledger, resume)
    request = bridge.validate_wave_request(
        plan, checked_ledger, checked_resume, wave_request
    )
    provider = deepcopy(dict(value))
    if provider.get("provider_plan_sha256") != _self_digest(
        provider, "provider_plan_sha256"
    ):
        raise ValueError("dataset provider plan digest changed")
    workers = provider.get("workers")
    entries = provider.get("content_entries")
    if (
        not isinstance(workers, list)
        or not isinstance(entries, list)
        or len(workers) != request["selected_count"]
        or len(workers) > bridge.MAX_CONCURRENT_VMS
    ):
        raise ValueError("dataset provider worker/content grid changed")
    expected_workers = [
        {
            "slot": index,
            "shard_id": row["shard_id"],
            "attempt_id": row["attempt_id"],
            "instance_id": row["instance_id"],
            "service_account": workers[index]["service_account"],
        }
        for index, row in enumerate(request["selected_attempts"])
    ]
    seen_objects: set[str] = set()
    for entry in entries:
        path = _plain_file(entry["local_path"], "dataset provider content")
        if (
            entry["object_name"] in seen_objects
            or bridge.sha256_file(path) != entry["sha256"]
            or path.stat().st_size != entry["bytes"]
        ):
            raise ValueError("dataset provider content changed")
        seen_objects.add(entry["object_name"])
    if (
        provider.get("schema") != PROVIDER_PLAN_SCHEMA
        or provider.get("status") != "provider_plan_ready_cloud_not_mutated"
        or provider.get("transport_plan_sha256") != plan["plan_sha256"]
        or provider.get("ledger_sha256") != checked_ledger["ledger_sha256"]
        or provider.get("resume_sha256") != checked_resume["resume_sha256"]
        or provider.get("wave_request_sha256") != request["request_sha256"]
        or provider.get("execution_identity_sha256")
        != plan["execution_identity_sha256"]
        or provider.get("wave_index") != request["wave_index"]
        or provider.get("project") != PROJECT
        or provider.get("region") != REGION
        or provider.get("zone") != ZONE
        or provider.get("bucket") != plan["bucket"]
        or provider.get("selected_attempts") != request["selected_attempts"]
        or workers != expected_workers
        or any(
            _SERVICE_ACCOUNT.fullmatch(row["service_account"]) is None
            for row in workers
        )
        or provider.get("runtime_contract", {}).get("machine_type")
        != MACHINE_TYPE
        or provider["runtime_contract"].get("max_concurrent_vms")
        != bridge.MAX_CONCURRENT_VMS
        or provider["runtime_contract"].get("max_vm_count") != len(workers)
        or provider["runtime_contract"].get("checkpoint_pair_granularity") != 1
        or provider.get("startup_script_sha256")
        != hashlib.sha256(worker_startup_script().encode("utf-8")).hexdigest()
        or provider.get("quality_and_smoke_source_replayed") is not True
        or provider.get("cloud_launch_authorized") is not True
        or provider.get("cloud_mutated") is not False
        or provider.get("current_profile_changed") is not False
    ):
        raise ValueError("dataset provider plan binding changed")
    return provider


def _validate_context(
    *,
    provider_plan: Mapping[str, Any],
    transport_plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    wave_request: Mapping[str, Any],
) -> dict[str, Any]:
    return validate_provider_plan(
        provider_plan,
        transport_plan=transport_plan,
        ledger=ledger,
        resume=resume,
        wave_request=wave_request,
    )


def worker_startup_script() -> str:
    """Network-install-free worker with create-only pair checkpoints."""

    return r"""#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=/var/lib/ofc-m31-dataset-v1
STAGING=$ROOT/staging
WORK=$ROOT/work
LOG=/var/log/ofc-m31-dataset-v1
META='http://metadata.google.internal/computeMetadata/v1/instance/attributes'
HEADER='Metadata-Flavor: Google'
mkdir -p "$STAGING" "$WORK" "$LOG"
chmod 0700 "$ROOT" "$STAGING" "$WORK" "$LOG"
exec >>"$LOG/startup.log" 2>&1
meta() { curl -fsS -H "$HEADER" "$META/$1"; }
BUCKET="$(meta ds-bucket)"
SHARD_ID="$(meta ds-shard-id)"
ATTEMPT_ID="$(meta ds-attempt-id)"
PREFIX="$(meta ds-object-prefix)"
PLAN_SHA="$(meta ds-plan-sha256)"
BINDINGS_B64="$(meta ds-content-bindings-b64)"
WATCHDOG="$(meta ds-watchdog-seconds)"
( sleep "$WATCHDOG"; shutdown -h now ) &
WATCHDOG_PID=$!
trap 'kill "$WATCHDOG_PID" >/dev/null 2>&1 || true' EXIT

python3 - "$STAGING" "$BUCKET" "$BINDINGS_B64" <<'PY'
import base64,hashlib,json,os,pathlib,sys,urllib.parse,urllib.request
root=pathlib.Path(sys.argv[1]).resolve(); bucket=sys.argv[2]
raw=base64.b64decode(sys.argv[3],validate=True); rows=json.loads(raw)
canon=lambda v: json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
if raw!=canon(rows) or not isinstance(rows,list): raise SystemExit("bad bindings")
token=json.load(urllib.request.urlopen(urllib.request.Request(
 "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
 headers={"Metadata-Flavor":"Google"})))["access_token"]
for row in rows:
 if set(row)!={"relative","object","generation","sha256","bytes"}: raise SystemExit("binding fields")
 rel=pathlib.PurePosixPath(row["relative"])
 if rel.is_absolute() or ".." in rel.parts: raise SystemExit("unsafe binding")
 url=("https://storage.googleapis.com/download/storage/v1/b/"+urllib.parse.quote(bucket,safe="")+
      "/o/"+urllib.parse.quote(row["object"],safe="")+"?alt=media&generation="+row["generation"])
 data=urllib.request.urlopen(urllib.request.Request(url,headers={"Authorization":"Bearer "+token}),timeout=600).read()
 if len(data)!=row["bytes"] or hashlib.sha256(data).hexdigest()!=row["sha256"]: raise SystemExit("content drift")
 path=root.joinpath(*rel.parts); path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("xb") as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())
PY

python3 - "$STAGING" <<'PY'
import pathlib,shutil,sys,tarfile,zipfile
root=pathlib.Path(sys.argv[1])
def untar(source,target,top):
 with tarfile.open(source,"r:*") as a:
  for m in a.getmembers():
   p=pathlib.PurePosixPath(m.name)
   if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0]!=top or m.issym() or m.islnk(): raise SystemExit("unsafe tar")
  a.extractall(target)
def unzip(source,target,top):
 with zipfile.ZipFile(source) as a:
  for n in a.namelist():
   p=pathlib.PurePosixPath(n)
   if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0]!=top: raise SystemExit("unsafe zip")
  a.extractall(target)
untar(next((root/"static/smoke_shard_archive").iterdir()),root,"smoke_shard")
untar(next((root/"static/runtime_archive").iterdir()),root,"runtime")
unzip(next((root/"static/wheelhouse_archive").iterdir()),root,"wheelhouse")
resume=root/"resume"; shard=root/"shard"
shard.mkdir()
if resume.exists():
 for p in resume.rglob("*"):
  if p.is_file():
   q=shard/p.relative_to(resume); q.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,q)
PY

export PYTHONPATH="$STAGING/runtime/src"
python3 -m venv --system-site-packages "$ROOT/venv"
"$ROOT/venv/bin/pip" install --no-index --find-links "$STAGING/wheelhouse" "$STAGING"/wheelhouse/* >/dev/null
cd "$STAGING/runtime"
PLAN="$(find "$STAGING/static/dataset_plan" -type f -maxdepth 1)"
FQ="$(find "$STAGING/static/fresh_quality_gate" -type f -maxdepth 1)"
SG="$(find "$STAGING/static/smoke_gate" -type f -maxdepth 1)"
PA="$(find "$STAGING/static/portable_authorization" -type f -maxdepth 1)"
LIB="$(find "$STAGING/static/candidate_library" -type f -maxdepth 1)"

python3 - "$STAGING" "$ROOT" "$BUCKET" "$PREFIX" "$PLAN_SHA" "$SHARD_ID" "$ATTEMPT_ID" "$PLAN" "$FQ" "$SG" "$PA" "$LIB" <<'PY'
import datetime,hashlib,json,os,pathlib,subprocess,sys,threading,urllib.parse,urllib.request
staging,root=map(pathlib.Path,sys.argv[1:3]); bucket,prefix,plan_sha,shard,attempt=sys.argv[3:8]
plan,fq,sg,pa,lib=map(pathlib.Path,sys.argv[8:13]); out=staging/"shard"; smoke=staging/"smoke_shard"
canon=lambda v: json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
def token():
 return json.load(urllib.request.urlopen(urllib.request.Request(
  "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
  headers={"Metadata-Flavor":"Google"})))["access_token"]
def put(obj,data):
 url=("https://storage.googleapis.com/upload/storage/v1/b/"+urllib.parse.quote(bucket,safe="")+
      "/o?uploadType=media&ifGenerationMatch=0&name="+urllib.parse.quote(obj,safe=""))
 req=urllib.request.Request(url,data=data,method="POST",headers={"Authorization":"Bearer "+token(),"Content-Type":"application/octet-stream"})
 return json.load(urllib.request.urlopen(req,timeout=600))
uploaded={}
heartbeat_tick=0
def pair_count():
 return len(list((out/"pairs").glob("pair_*.json"))) if (out/"pairs").exists() else 0
def heartbeat(completed):
 global heartbeat_tick
 tick=heartbeat_tick; heartbeat_tick+=1
 hb={"schema":"hu_m31_t3_dataset_gcp_heartbeat_v1","plan_sha256":plan_sha,"shard_id":shard,
     "attempt_id":attempt,"sequence":tick,"completed_pair_count":completed,"create_only":True,
     "observed_at_utc":datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
     "current_profile_changed":False}
 hb["heartbeat_sha256"]=hashlib.sha256(canon(hb)).hexdigest()
 put(prefix+f"/heartbeats/{tick:06d}.json",canon(hb))
def run_with_heartbeat(cmd,completed):
 heartbeat(completed); stop=threading.Event()
 def loop():
  while not stop.wait(60): heartbeat(completed)
 thread=threading.Thread(target=loop,daemon=True); thread.start()
 try: subprocess.run(cmd,check=True)
 finally: stop.set(); thread.join()
def publish():
 files=[]
 for path in sorted(p for p in out.rglob("*") if p.is_file()):
  rel=path.relative_to(out).as_posix(); data=path.read_bytes(); obj=prefix+"/files/"+rel
  if rel not in uploaded: uploaded[rel]=put(obj,data)["generation"]
  files.append({"relative_path":rel,"object_name":obj,"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)})
 count=pair_count(); complete=(out/"SHARD_DONE.json").is_file()
 cp={"schema":"hu_m31_t3_dataset_gcp_checkpoint_manifest_v1","plan_sha256":plan_sha,"shard_id":shard,
     "attempt_id":attempt,"sequence":count,"completed_pair_count":count,"complete":complete,"files":files,
     "checkpoint_published_after_files":True,"create_only":True,"teacher_values_are_realized_match_ev":False,
     "current_profile_changed":False}
 cp["checkpoint_sha256"]=hashlib.sha256(canon(cp)).hexdigest()
 put(prefix+f"/checkpoints/{count:06d}.json",canon(cp)); return count,complete
cmd=[str(root/"venv/bin/python"),"-m","ofc_regular.hu_m31_t3_dataset_portable_worker_v1","run",
 "--plan",str(plan),"--shard-id",shard,"--shard-directory",str(out),"--portable-authorization",str(pa),
 "--fresh-quality-gate",str(fq),"--smoke-gate",str(sg),"--smoke-shard-directory",str(smoke),
 "--library",str(lib),"--max-new-pairs","0"]
run_with_heartbeat(cmd,pair_count()); count,complete=publish()
while not complete:
 cmd[-1]="1"; run_with_heartbeat(cmd,count); count,complete=publish()
PY
kill "$WATCHDOG_PID" >/dev/null 2>&1 || true
shutdown -h now
"""


def _metadata_record(
    *,
    entry: Mapping[str, Any],
    metadata: Mapping[str, Any],
    payload: bytes,
    created: bool,
) -> dict[str, Any]:
    generation = str(metadata.get("generation", ""))
    if (
        metadata.get("name") != entry["object_name"]
        or _PROVIDER_ID.fullmatch(generation) is None
    ):
        raise DatasetProviderError("GCS staged object identity changed")
    return {
        "kind": entry["kind"],
        "object_name": entry["object_name"],
        "generation": generation,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "created": created,
    }


def stage_content(
    *,
    provider_plan: Mapping[str, Any],
    transport_plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    wave_request: Mapping[str, Any],
    transport: DatasetCloudTransport,
) -> dict[str, Any]:
    provider = _validate_context(
        provider_plan=provider_plan,
        transport_plan=transport_plan,
        ledger=ledger,
        resume=resume,
        wave_request=wave_request,
    )
    records = []
    created_any = False
    for entry in provider["content_entries"]:
        payload = _plain_file(entry["local_path"], "dataset content").read_bytes()
        if (
            hashlib.sha256(payload).hexdigest() != entry["sha256"]
            or len(payload) != entry["bytes"]
        ):
            raise ValueError("dataset content changed before stage")
        metadata = transport.get_object_metadata(
            bucket=provider["bucket"], object_name=entry["object_name"]
        )
        created = False
        if metadata is None:
            metadata = transport.put_object_new(
                bucket=provider["bucket"],
                object_name=entry["object_name"],
                payload=payload,
                content_type=entry["content_type"],
            )
            created = True
            created_any = True
        generation = str(metadata.get("generation", ""))
        observed = transport.get_object_bytes(
            bucket=provider["bucket"],
            object_name=entry["object_name"],
            generation=generation,
        )
        if observed != payload:
            raise DatasetProviderError("staged dataset object bytes changed")
        records.append(
            _metadata_record(
                entry=entry,
                metadata=metadata,
                payload=payload,
                created=created,
            )
        )
    core = {
        "schema": STAGE_RECEIPT_SCHEMA,
        "status": "all_content_hash_replayed",
        "provider_plan_sha256": provider["provider_plan_sha256"],
        "content_records": records,
        "record_count": len(records),
        "all_bytes_replayed": True,
        "create_only_or_identical_reuse": True,
        "cloud_mutated": created_any,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def validate_stage_receipt(
    value: Mapping[str, Any], *, provider_plan: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = deepcopy(dict(value))
    if (
        receipt.get("receipt_sha256") != _self_digest(receipt, "receipt_sha256")
        or receipt.get("schema") != STAGE_RECEIPT_SCHEMA
        or receipt.get("status") != "all_content_hash_replayed"
        or receipt.get("provider_plan_sha256")
        != provider_plan["provider_plan_sha256"]
        or receipt.get("record_count") != len(provider_plan["content_entries"])
        or receipt.get("all_bytes_replayed") is not True
        or receipt.get("create_only_or_identical_reuse") is not True
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("dataset stage receipt changed")
    for row, entry in zip(
        receipt["content_records"],
        provider_plan["content_entries"],
        strict=True,
    ):
        if (
            row["kind"] != entry["kind"]
            or row["object_name"] != entry["object_name"]
            or row["sha256"] != entry["sha256"]
            or row["bytes"] != entry["bytes"]
            or _PROVIDER_ID.fullmatch(str(row["generation"])) is None
        ):
            raise ValueError("dataset stage object binding changed")
    return receipt


def _policy_parts(value: Mapping[str, Any]) -> tuple[list[Any], str, int]:
    bindings = value.get("bindings", [])
    etag = value.get("etag")
    version = value.get("version", 1)
    if (
        not isinstance(bindings, list)
        or not isinstance(etag, str)
        or not etag
        or not isinstance(version, int)
    ):
        raise DatasetProviderError("bucket IAM policy is incomplete")
    return deepcopy(bindings), etag, max(3, version)


def _iam_bindings(
    provider: Mapping[str, Any], *, expires_at_utc: str
) -> list[dict[str, Any]]:
    resource = f"projects/_/buckets/{provider['bucket']}/objects/"
    contract = provider["iam_contract"]
    rows = [
        {
            "role": contract["viewer_role"],
            "members": list(contract["members"]),
            "condition": {
                "title": contract["condition_titles"][0],
                "description": "M3.1 dataset immutable content read",
                "expression": (
                    f"resource.name.startsWith('{resource}"
                    f"{contract['reader_prefix']}') && "
                    f"request.time < timestamp('{expires_at_utc}')"
                ),
            },
        }
    ]
    for index, (worker, selected) in enumerate(
        zip(provider["workers"], provider["selected_attempts"], strict=True)
    ):
        rows.append(
            {
                "role": contract["creator_role"],
                "members": [f"serviceAccount:{worker['service_account']}"],
                "condition": {
                    "title": contract["condition_titles"][index + 1],
                    "description": "M3.1 dataset create-only checkpoint publish",
                    "expression": (
                        f"resource.name.startsWith('{resource}"
                        f"{selected['object_prefix']}/') && "
                        f"request.time < timestamp('{expires_at_utc}')"
                    ),
                },
            }
        )
    return rows


def _condition_title(row: Any) -> str | None:
    if not isinstance(row, Mapping):
        return None
    condition = row.get("condition")
    if not isinstance(condition, Mapping):
        return None
    title = condition.get("title")
    return title if isinstance(title, str) else None


def _existing_iam_expiry(
    provider: Mapping[str, Any],
    bindings: Sequence[Any],
    *,
    now_unix_seconds: int,
) -> str | None:
    """Return an exact pre-existing wave binding expiry, or fail closed.

    This is the only reconciliation path for a process crash after the bucket
    policy mutation succeeded but before its create-only receipt was written.
    A partial, changed, duplicate, or nearly-expired binding set is never
    adopted.
    """

    titles = set(provider["iam_contract"]["condition_titles"])
    matching = [row for row in bindings if _condition_title(row) in titles]
    if not matching:
        return None
    if (
        len(matching) != len(titles)
        or {_condition_title(row) for row in matching} != titles
    ):
        raise FileExistsError("dataset worker IAM title set is partial or duplicated")
    expiries: set[str] = set()
    for row in matching:
        condition = row.get("condition")
        expression = (
            condition.get("expression")
            if isinstance(condition, Mapping)
            else None
        )
        if not isinstance(expression, str):
            raise PermissionError("dataset worker IAM condition changed")
        match = re.search(r"timestamp\('([^']+)'\)", expression)
        if match is None:
            raise PermissionError("dataset worker IAM expiry changed")
        expiries.add(match.group(1))
    if len(expiries) != 1:
        raise PermissionError("dataset worker IAM expiries diverged")
    expires = next(iter(expiries))
    try:
        expiry_epoch = int(
            dt.datetime.strptime(expires, "%Y-%m-%dT%H:%M:%SZ")
            .replace(tzinfo=dt.timezone.utc)
            .timestamp()
        )
    except ValueError as exc:
        raise PermissionError("dataset worker IAM expiry is invalid") from exc
    if expiry_epoch < now_unix_seconds + MAX_RUN_SECONDS:
        raise PermissionError("dataset worker IAM reconciliation window expired")
    expected = _iam_bindings(provider, expires_at_utc=expires)
    if len(matching) != len(expected) or any(row not in matching for row in expected):
        raise PermissionError("dataset worker IAM binding changed")
    return expires


def install_worker_iam(
    *,
    provider_plan: Mapping[str, Any],
    transport_plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    wave_request: Mapping[str, Any],
    transport: DatasetCloudTransport,
    now_unix_seconds: int,
) -> dict[str, Any]:
    provider = _validate_context(
        provider_plan=provider_plan,
        transport_plan=transport_plan,
        ledger=ledger,
        resume=resume,
        wave_request=wave_request,
    )
    for worker in provider["workers"]:
        account = transport.get_service_account(email=worker["service_account"])
        if (
            account.get("email") != worker["service_account"]
            or _PROVIDER_ID.fullmatch(str(account.get("uniqueId", ""))) is None
            or transport.test_service_account_act_as(
                email=worker["service_account"]
            )
            is not True
        ):
            raise PermissionError("dataset worker identity check failed")
    expires = _utc(now_unix_seconds + IAM_TTL_SECONDS)
    policy = dict(transport.get_bucket_iam_policy(bucket=provider["bucket"]))
    bindings, etag, version = _policy_parts(policy)
    reconciled_expiry = _existing_iam_expiry(
        provider, bindings, now_unix_seconds=now_unix_seconds
    )
    response_reconciled = reconciled_expiry is not None
    if response_reconciled:
        expires = reconciled_expiry
        additions = _iam_bindings(provider, expires_at_utc=expires)
    else:
        additions = _iam_bindings(provider, expires_at_utc=expires)
        desired = {
            **{key: value for key, value in policy.items() if key != "bindings"},
            "version": version,
            "etag": etag,
            "bindings": [*bindings, *additions],
        }
        updated = dict(
            transport.set_bucket_iam_policy(bucket=provider["bucket"], policy=desired)
        )
        observed = dict(
            transport.get_bucket_iam_policy(bucket=provider["bucket"])
        )
        updated_bindings, updated_etag, _ = _policy_parts(updated)
        observed_bindings, observed_etag, _ = _policy_parts(observed)
        if (
            updated_etag != observed_etag
            or any(row not in updated_bindings for row in additions)
            or any(row not in observed_bindings for row in additions)
        ):
            raise DatasetProviderError("dataset IAM readback changed")
    core = {
        "schema": IAM_RECEIPT_SCHEMA,
        "status": "exact_temporary_worker_bindings_installed",
        "provider_plan_sha256": provider["provider_plan_sha256"],
        "bucket": provider["bucket"],
        "condition_titles": list(
            provider["iam_contract"]["condition_titles"]
        ),
        "members": provider["iam_contract"]["members"],
        "expires_at_utc": expires,
        "bindings_installed": len(additions),
        "readback_exact": True,
        "response_reconciled": response_reconciled,
        "cloud_mutated": not response_reconciled,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def validate_iam_receipt(
    value: Mapping[str, Any], *, provider_plan: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = deepcopy(dict(value))
    if (
        receipt.get("receipt_sha256")
        != _self_digest(receipt, "receipt_sha256")
        or receipt.get("schema") != IAM_RECEIPT_SCHEMA
        or receipt.get("status")
        != "exact_temporary_worker_bindings_installed"
        or receipt.get("provider_plan_sha256")
        != provider_plan["provider_plan_sha256"]
        or receipt.get("bucket") != provider_plan["bucket"]
        or set(receipt.get("condition_titles", []))
        != set(provider_plan["iam_contract"]["condition_titles"])
        or receipt.get("members") != provider_plan["iam_contract"]["members"]
        or receipt.get("bindings_installed")
        != len(provider_plan["workers"]) + 1
        or receipt.get("readback_exact") is not True
        or not isinstance(receipt.get("response_reconciled"), bool)
        or receipt.get("cloud_mutated")
        is not (not receipt["response_reconciled"])
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("dataset IAM receipt changed")
    return receipt


def _content_bindings(
    provider: Mapping[str, Any],
    stage_receipt: Mapping[str, Any],
    shard_id: str,
) -> list[dict[str, Any]]:
    stage = validate_stage_receipt(stage_receipt, provider_plan=provider)
    records = {
        row["object_name"]: row for row in stage["content_records"]
    }
    result = []
    for entry in provider["content_entries"]:
        if entry["shard_id"] not in (None, shard_id):
            continue
        record = records[entry["object_name"]]
        result.append(
            {
                "relative": entry["relative_path"],
                "object": entry["object_name"],
                "generation": record["generation"],
                "sha256": record["sha256"],
                "bytes": record["bytes"],
            }
        )
    return result


def _instance_spec(
    *,
    provider: Mapping[str, Any],
    worker: Mapping[str, Any],
    selected: Mapping[str, Any],
    stage_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    bindings = _content_bindings(
        provider, stage_receipt, selected["shard_id"]
    )
    metadata = {
        "ds-bucket": provider["bucket"],
        "ds-shard-id": selected["shard_id"],
        "ds-attempt-id": selected["attempt_id"],
        "ds-object-prefix": selected["object_prefix"],
        "ds-plan-sha256": provider["transport_plan_sha256"],
        "ds-content-bindings-b64": base64.b64encode(
            canonical_bytes(bindings)
        ).decode("ascii"),
        "ds-watchdog-seconds": str(WATCHDOG_SECONDS),
        "startup-script": worker_startup_script(),
    }
    wave = provider["wave_index"]
    labels = {
        "ofc-owner": provider["execution_identity_sha256"][:32],
        "ofc-plan": provider["provider_plan_sha256"][:32],
        "ofc-wave": f"w{wave:02d}",
    }
    return {
        "name": selected["instance_id"],
        "machineType": (
            f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/zones/"
            f"{ZONE}/machineTypes/{MACHINE_TYPE}"
        ),
        "labels": labels,
        "scheduling": {
            "provisioningModel": "SPOT",
            "instanceTerminationAction": "DELETE",
            "automaticRestart": False,
            "onHostMaintenance": "TERMINATE",
            "maxRunDuration": {"seconds": str(MAX_RUN_SECONDS), "nanos": 0},
        },
        "disks": [
            {
                "boot": True,
                "autoDelete": True,
                "type": "PERSISTENT",
                "interface": BOOT_DISK_INTERFACE,
                "deviceName": selected["instance_id"],
                "initializeParams": {
                    "sourceImage": provider["image"]["self_link"],
                    "diskSizeGb": str(BOOT_DISK_SIZE_GB),
                    "diskName": selected["instance_id"],
                    "labels": labels,
                    "diskType": (
                        f"https://www.googleapis.com/compute/v1/projects/"
                        f"{PROJECT}/zones/{ZONE}/diskTypes/{BOOT_DISK_TYPE}"
                    ),
                },
            }
        ],
        "networkInterfaces": [
            {
                "network": (
                    f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
                    f"global/networks/{NETWORK}"
                ),
                "subnetwork": (
                    f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
                    f"regions/{REGION}/subnetworks/{SUBNETWORK}"
                ),
                "nicType": NIC_TYPE,
                "accessConfigs": [],
            }
        ],
        "serviceAccounts": [
            {"email": worker["service_account"], "scopes": [OAUTH_SCOPE]}
        ],
        "metadata": {
            "items": [
                {"key": key, "value": value}
                for key, value in sorted(metadata.items())
            ]
        },
        "deletionProtection": False,
        "canIpForward": False,
    }


def _metadata_map(instance: Mapping[str, Any]) -> dict[str, str]:
    metadata = instance.get("metadata")
    items = metadata.get("items") if isinstance(metadata, Mapping) else None
    if not isinstance(items, list):
        raise DatasetProviderError("GCE dataset instance metadata changed")
    result: dict[str, str] = {}
    for row in items:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("key"), str)
            or not isinstance(row.get("value"), str)
            or row["key"] in result
        ):
            raise DatasetProviderError("GCE dataset instance metadata changed")
        result[row["key"]] = row["value"]
    return result


def _validate_owned_instance(
    instance: Mapping[str, Any], *, expected_spec: Mapping[str, Any]
) -> tuple[str, str, str]:
    """Validate against the dataset spec, including its six-hour duration.

    The fresh-quality provider has the same resource shape but a 90-minute
    ``MAX_RUN_SECONDS`` constant.  Reusing its validator would reject every
    correctly configured dataset VM, so all variable fields are compared to
    the exact provider plan instead of another phase's globals.
    """

    name = expected_spec["name"]
    disks = instance.get("disks")
    interfaces = instance.get("networkInterfaces")
    accounts = instance.get("serviceAccounts")
    scheduling = instance.get("scheduling")
    expected_scheduling = expected_spec["scheduling"]
    expected_disk = expected_spec["disks"][0]
    expected_interface = expected_spec["networkInterfaces"][0]
    expected_account = expected_spec["serviceAccounts"][0]
    if (
        instance.get("name") != name
        or instance.get("machineType") != expected_spec["machineType"]
        or instance.get("labels") != expected_spec["labels"]
        or instance.get("deletionProtection", False) is not False
        or instance.get("canIpForward", False) is not False
        or not isinstance(scheduling, Mapping)
        or scheduling.get("provisioningModel")
        != expected_scheduling["provisioningModel"]
        or scheduling.get("instanceTerminationAction")
        != expected_scheduling["instanceTerminationAction"]
        or scheduling.get("automaticRestart")
        != expected_scheduling["automaticRestart"]
        or scheduling.get("onHostMaintenance")
        != expected_scheduling["onHostMaintenance"]
        or scheduling.get("maxRunDuration")
        != expected_scheduling["maxRunDuration"]
        or _metadata_map(instance)
        != {
            row["key"]: row["value"]
            for row in expected_spec["metadata"]["items"]
        }
        or not isinstance(disks, list)
        or len(disks) != 1
        or disks[0].get("boot") is not expected_disk["boot"]
        or disks[0].get("autoDelete") is not expected_disk["autoDelete"]
        or disks[0].get("interface") != expected_disk["interface"]
        or disks[0].get("deviceName") != expected_disk["deviceName"]
        or not isinstance(interfaces, list)
        or len(interfaces) != 1
        or interfaces[0].get("nicType") != expected_interface["nicType"]
        or interfaces[0].get("network") != expected_interface["network"]
        or interfaces[0].get("subnetwork") != expected_interface["subnetwork"]
        or interfaces[0].get("accessConfigs", [])
        != expected_interface["accessConfigs"]
        or not isinstance(accounts, list)
        or len(accounts) != 1
        or accounts[0].get("email") != expected_account["email"]
        or accounts[0].get("scopes") != expected_account["scopes"]
    ):
        raise PermissionError(
            "GCE dataset instance differs from exact owned specification"
        )
    provider_id = str(instance.get("id", ""))
    source = disks[0].get("source")
    status = instance.get("status")
    if (
        _PROVIDER_ID.fullmatch(provider_id) is None
        or not isinstance(source, str)
        or not source
        or status
        not in {
            "PROVISIONING",
            "STAGING",
            "RUNNING",
            "STOPPING",
            "SUSPENDING",
            "SUSPENDED",
            "REPAIRING",
            "TERMINATED",
        }
    ):
        raise DatasetProviderError(
            "GCE dataset provider identity or status changed"
        )
    return provider_id, source.rsplit("/", 1)[-1], str(status)


def _validate_owned_disk(
    disk: Mapping[str, Any], *, expected_spec: Mapping[str, Any]
) -> str:
    expected = expected_spec["disks"][0]["initializeParams"]
    provider_id = str(disk.get("id", ""))
    if (
        disk.get("name") != expected_spec["name"]
        or _PROVIDER_ID.fullmatch(provider_id) is None
        or disk.get("type") != expected["diskType"]
        or str(disk.get("sizeGb")) != str(expected["diskSizeGb"])
        or disk.get("sourceImage") != expected["sourceImage"]
        or disk.get("labels") != expected["labels"]
    ):
        raise PermissionError(
            "GCE dataset boot disk differs from exact owned specification"
        )
    return provider_id


def _claim(
    provider: Mapping[str, Any],
    *,
    transport: DatasetCloudTransport,
    raw_nonce: str,
) -> dict[str, Any]:
    try:
        parsed = uuid.UUID(raw_nonce)
    except (ValueError, AttributeError) as exc:
        raise ValueError("dataset launch nonce must be UUIDv4") from exc
    if parsed.version != 4 or str(parsed) != raw_nonce:
        raise ValueError("dataset launch nonce must be canonical UUIDv4")
    payload_value = {
        "schema": "hu_m31_t3_dataset_gcp_launch_claim_v1",
        "provider_plan_sha256": provider["provider_plan_sha256"],
        "wave_request_sha256": provider["wave_request_sha256"],
        "nonce_sha256": hashlib.sha256(raw_nonce.encode("ascii")).hexdigest(),
        "create_only": True,
    }
    payload = canonical_bytes(payload_value)
    metadata = transport.get_object_metadata(
        bucket=provider["bucket"], object_name=provider["claim_object"]
    )
    reconciled = metadata is not None
    if metadata is None:
        metadata = transport.put_object_new(
            bucket=provider["bucket"],
            object_name=provider["claim_object"],
            payload=payload,
            content_type="application/json",
        )
    generation = str(metadata.get("generation", ""))
    observed = transport.get_object_bytes(
        bucket=provider["bucket"],
        object_name=provider["claim_object"],
        generation=generation,
    )
    if observed != payload or _PROVIDER_ID.fullmatch(generation) is None:
        raise DatasetProviderError("dataset launch claim changed")
    return {
        "object_name": provider["claim_object"],
        "generation": generation,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "nonce_sha256": payload_value["nonce_sha256"],
        "response_reconciled": reconciled,
    }


def execute_wave(
    *,
    provider_plan: Mapping[str, Any],
    transport_plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    wave_request: Mapping[str, Any],
    stage_receipt: Mapping[str, Any],
    iam_receipt: Mapping[str, Any],
    transport: DatasetCloudTransport,
    raw_nonce: str,
    observed_at_utc: str,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Perform the only instance-create boundary; never called implicitly."""

    provider = _validate_context(
        provider_plan=provider_plan,
        transport_plan=transport_plan,
        ledger=ledger,
        resume=resume,
        wave_request=wave_request,
    )
    stage = validate_stage_receipt(stage_receipt, provider_plan=provider)
    if (
        provider.get("cloud_launch_authorized") is not True
        or provider.get("quality_and_smoke_source_replayed") is not True
        or len(provider["workers"]) > bridge.MAX_CONCURRENT_VMS
    ):
        raise PermissionError("dataset provider plan does not authorize launch")
    installed_iam = validate_iam_receipt(
        iam_receipt, provider_plan=provider
    )
    quality_provider._validate_image(provider, transport)  # type: ignore[arg-type]
    quality_provider._validate_network(provider, transport)  # type: ignore[arg-type]
    quota = quality_provider.read_quota(  # type: ignore[arg-type]
        provider_plan=provider,
        transport=transport,
        observed_at_utc=observed_at_utc,
    )
    claim = _claim(provider, transport=transport, raw_nonce=raw_nonce)
    workers = {row["shard_id"]: row for row in provider["workers"]}
    rows = []
    for selected in provider["selected_attempts"]:
        worker = workers[selected["shard_id"]]
        spec = _instance_spec(
            provider=provider,
            worker=worker,
            selected=selected,
            stage_receipt=stage,
        )
        request_id = _uuid_for(
            provider["provider_plan_sha256"],
            selected["shard_id"],
            selected["attempt_id"],
            "create",
        )
        instance = transport.get_instance(
            instance_name=selected["instance_id"]
        )
        created = False
        if instance is None:
            try:
                operation = transport.create_instance(
                    instance_spec=spec, request_id=request_id
                )
            except c4_gcp.ResponseLostError:
                instance = transport.get_instance(
                    instance_name=selected["instance_id"]
                )
                if instance is None:
                    operation = transport.create_instance(
                        instance_spec=spec, request_id=request_id
                    )
                else:
                    operation = None
            if operation is not None:
                quality_provider._wait_operation(  # type: ignore[arg-type]
                    transport,
                    initial=operation,
                    operation_type="insert",
                    target_name=selected["instance_id"],
                    sleep=sleep,
                )
            instance = transport.get_instance(
                instance_name=selected["instance_id"]
            )
            created = True
        if instance is None:
            raise DatasetProviderError("created dataset instance is absent")
        provider_id, disk_name, status = _validate_owned_instance(
            instance, expected_spec=spec
        )
        disk = transport.get_disk_optional(disk_name=disk_name)
        if disk is None:
            raise DatasetProviderError("dataset boot disk is absent")
        disk_id = _validate_owned_disk(disk, expected_spec=spec)
        rows.append(
            {
                "shard_id": selected["shard_id"],
                "attempt_id": selected["attempt_id"],
                "instance_id": selected["instance_id"],
                "request_id": request_id,
                "spec_sha256": canonical_sha256(spec),
                "provider_instance_id": provider_id,
                "provider_boot_disk_id": disk_id,
                "observed_status": status,
                "created": created,
            }
        )
    core = {
        "schema": LAUNCH_RECEIPT_SCHEMA,
        "provider_plan_sha256": provider["provider_plan_sha256"],
        "wave_request_sha256": provider["wave_request_sha256"],
        "claim": claim,
        "stage_receipt": stage,
        "iam_receipt": installed_iam,
        "quota_receipt": quota,
        "rows": rows,
        "selected_shard_count": len(provider["workers"]),
        "created_instance_count": sum(row["created"] for row in rows),
        "at_most_eight_c4": len(rows) <= bridge.MAX_CONCURRENT_VMS,
        "one_shot_claim_consumed": True,
        "unlisted_vm_created": 0,
        "cloud_mutated": True,
        "current_profile_changed": False,
    }
    result = {**core, "receipt_sha256": canonical_sha256(core)}
    return validate_launch_receipt(result, provider_plan=provider)


def validate_launch_receipt(
    value: Mapping[str, Any],
    *,
    provider_plan: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = deepcopy(dict(value))
    provider = deepcopy(dict(provider_plan))
    if (
        receipt.get("receipt_sha256")
        != _self_digest(receipt, "receipt_sha256")
        or receipt.get("schema") != LAUNCH_RECEIPT_SCHEMA
        or receipt.get("provider_plan_sha256")
        != provider["provider_plan_sha256"]
        or receipt.get("wave_request_sha256")
        != provider["wave_request_sha256"]
        or receipt.get("selected_shard_count") != len(provider["workers"])
        or receipt.get("created_instance_count")
        != sum(row.get("created") is True for row in receipt.get("rows", []))
        or receipt.get("at_most_eight_c4") is not True
        or receipt.get("one_shot_claim_consumed") is not True
        or receipt.get("unlisted_vm_created") != 0
        or receipt.get("cloud_mutated") is not True
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("dataset launch receipt changed")
    stage = validate_stage_receipt(
        receipt.get("stage_receipt", {}), provider_plan=provider
    )
    validate_iam_receipt(
        receipt.get("iam_receipt", {}), provider_plan=provider
    )
    claim = receipt.get("claim")
    quota = receipt.get("quota_receipt")
    rows = receipt.get("rows")
    if (
        not isinstance(claim, Mapping)
        or claim.get("object_name") != provider["claim_object"]
        or _PROVIDER_ID.fullmatch(str(claim.get("generation", ""))) is None
        or _SHA.fullmatch(str(claim.get("sha256", ""))) is None
        or not isinstance(quota, Mapping)
        or quota.get("sufficient") is not True
        or not isinstance(rows, list)
        or len(rows) != len(provider["selected_attempts"])
    ):
        raise ValueError("dataset launch receipt evidence changed")
    workers = {row["shard_id"]: row for row in provider["workers"]}
    for row, selected in zip(
        rows, provider["selected_attempts"], strict=True
    ):
        worker = workers[selected["shard_id"]]
        expected_spec = _instance_spec(
            provider=provider,
            worker=worker,
            selected=selected,
            stage_receipt=stage,
        )
        if (
            row.get("shard_id") != selected["shard_id"]
            or row.get("attempt_id") != selected["attempt_id"]
            or row.get("instance_id") != selected["instance_id"]
            or row.get("request_id")
            != _uuid_for(
                provider["provider_plan_sha256"],
                selected["shard_id"],
                selected["attempt_id"],
                "create",
            )
            or row.get("spec_sha256") != canonical_sha256(expected_spec)
            or _PROVIDER_ID.fullmatch(
                str(row.get("provider_instance_id", ""))
            )
            is None
            or _PROVIDER_ID.fullmatch(
                str(row.get("provider_boot_disk_id", ""))
            )
            is None
            or not isinstance(row.get("created"), bool)
        ):
            raise ValueError("dataset launch row changed")
    return receipt


def _read_remote(
    transport: DatasetCloudTransport,
    *,
    bucket: str,
    object_name: str,
) -> tuple[dict[str, Any], bytes] | None:
    metadata = transport.get_object_metadata(
        bucket=bucket, object_name=object_name
    )
    if metadata is None:
        return None
    generation = str(metadata.get("generation", ""))
    payload = transport.get_object_bytes(
        bucket=bucket, object_name=object_name, generation=generation
    )
    if (
        payload is None
        or metadata.get("name") != object_name
        or _PROVIDER_ID.fullmatch(generation) is None
    ):
        raise DatasetProviderError("dataset remote object changed")
    return (
        {
            "object": object_name,
            "generation": generation,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        },
        payload,
    )


def _latest_sequence(
    transport: DatasetCloudTransport,
    *,
    bucket: str,
    pattern: str,
    maximum_sequence: int,
) -> tuple[int, dict[str, Any], bytes] | None:
    latest = None
    for sequence in range(maximum_sequence + 1):
        observed = _read_remote(
            transport,
            bucket=bucket,
            object_name=pattern % sequence,
        )
        if observed is None:
            break
        record, payload = observed
        latest = (sequence, record, payload)
    return latest


def _attempt_observation(
    *,
    provider: Mapping[str, Any],
    selected: Mapping[str, Any],
    transport: DatasetCloudTransport,
) -> dict[str, Any]:
    latest_checkpoint = _latest_sequence(
        transport,
        bucket=provider["bucket"],
        pattern=selected["checkpoint_object_format"],
        maximum_sequence=dataset.SHARD_PAIR_COUNT,
    )
    latest_heartbeat = _latest_sequence(
        transport,
        bucket=provider["bucket"],
        pattern=selected["heartbeat_object_format"],
        maximum_sequence=bridge.MAX_HEARTBEAT_SEQUENCE,
    )
    checkpoint_record = None
    completed = 0
    status = "failed"
    if latest_checkpoint is not None:
        sequence, record, payload = latest_checkpoint
        try:
            manifest = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DatasetProviderError("remote checkpoint is not JSON") from exc
        if payload != canonical_bytes(manifest):
            raise DatasetProviderError("remote checkpoint is not canonical")
        manifest = bridge.validate_checkpoint_manifest(
            manifest,
            plan=provider["_transport_plan"],
            shard_id=selected["shard_id"],
            attempt_id=selected["attempt_id"],
        )
        completed = sequence
        file_objects = []
        for file in manifest["files"]:
            observed = _read_remote(
                transport,
                bucket=provider["bucket"],
                object_name=file["object_name"],
            )
            if observed is None:
                raise DatasetProviderError(
                    "checkpoint references a missing dataset file"
                )
            file_record, _ = observed
            if (
                file_record["sha256"] != file["sha256"]
                or file_record["bytes"] != file["bytes"]
            ):
                raise DatasetProviderError("checkpoint file binding changed")
            file_objects.append(file_record)
        checkpoint_record = {
            **record,
            "manifest": manifest,
            "file_objects": file_objects,
        }
        status = "ready" if manifest["complete"] else "checkpointed"
    heartbeat_record = None
    if latest_heartbeat is not None:
        _sequence, record, payload = latest_heartbeat
        value = json.loads(payload)
        if payload != canonical_bytes(value):
            raise DatasetProviderError("remote heartbeat is not canonical")
        value = bridge.validate_heartbeat(
            value,
            plan=provider["_transport_plan"],
            shard_id=selected["shard_id"],
            attempt_id=selected["attempt_id"],
        )
        heartbeat_record = {**record, "value": value}
    return {
        "shard_id": selected["shard_id"],
        "attempt_id": selected["attempt_id"],
        "instance_id": selected["instance_id"],
        "status": status,
        "completed_pair_count": completed,
        "checkpoint": checkpoint_record,
        "heartbeat": heartbeat_record,
        "owned_compute_absent": True,
        "worker_iam_removed": True,
    }


def poll_wave(
    *,
    provider_plan: Mapping[str, Any],
    transport_plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    wave_request: Mapping[str, Any],
    transport: DatasetCloudTransport,
) -> dict[str, Any]:
    checked = _validate_context(
        provider_plan=provider_plan,
        transport_plan=transport_plan,
        ledger=ledger,
        resume=resume,
        wave_request=wave_request,
    )
    provider = {**checked, "_transport_plan": transport_plan}
    rows = []
    for selected in provider["selected_attempts"]:
        observed = _attempt_observation(
            provider=provider,
            selected=selected,
            transport=transport,
        )
        instance = transport.get_instance(instance_name=selected["instance_id"])
        rows.append(
            {
                "shard_id": selected["shard_id"],
                "attempt_id": selected["attempt_id"],
                "instance_present": instance is not None,
                "latest_completed_pair_count": observed[
                    "completed_pair_count"
                ],
                "checkpoint_present": observed["checkpoint"] is not None,
                "heartbeat_present": observed["heartbeat"] is not None,
                "complete": observed["status"] == "ready",
            }
        )
    core = {
        "schema": POLL_RECEIPT_SCHEMA,
        "provider_plan_sha256": provider_plan["provider_plan_sha256"],
        "rows": rows,
        "selected_shard_count": len(rows),
        "complete_count": sum(row["complete"] for row in rows),
        "read_only": True,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def validate_poll_receipt(
    value: Mapping[str, Any], *, provider_plan: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = deepcopy(dict(value))
    rows = receipt.get("rows")
    if (
        receipt.get("receipt_sha256")
        != _self_digest(receipt, "receipt_sha256")
        or receipt.get("schema") != POLL_RECEIPT_SCHEMA
        or receipt.get("provider_plan_sha256")
        != provider_plan["provider_plan_sha256"]
        or not isinstance(rows, list)
        or receipt.get("selected_shard_count") != len(rows)
        or receipt.get("complete_count")
        != sum(row.get("complete") is True for row in rows)
        or receipt.get("read_only") is not True
        or receipt.get("cloud_mutated") is not False
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("dataset poll receipt changed")
    return receipt


def remove_worker_iam(
    *,
    provider_plan: Mapping[str, Any],
    transport_plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    wave_request: Mapping[str, Any],
    iam_receipt: Mapping[str, Any],
    transport: DatasetCloudTransport,
) -> dict[str, Any]:
    provider = _validate_context(
        provider_plan=provider_plan,
        transport_plan=transport_plan,
        ledger=ledger,
        resume=resume,
        wave_request=wave_request,
    )
    installed = validate_iam_receipt(
        iam_receipt, provider_plan=provider
    )
    policy = dict(transport.get_bucket_iam_policy(bucket=provider["bucket"]))
    bindings, etag, version = _policy_parts(policy)
    expected = _iam_bindings(
        provider, expires_at_utc=installed["expires_at_utc"]
    )
    titles = set(provider["iam_contract"]["condition_titles"])
    matching = [row for row in bindings if _condition_title(row) in titles]
    if matching and (
        len(matching) != len(expected)
        or any(row not in matching for row in expected)
    ):
        raise PermissionError("dataset worker IAM binding changed before cleanup")
    response_reconciled = not matching
    unrelated = bindings if response_reconciled else [
        row for row in bindings if row not in expected
    ]
    if not response_reconciled:
        desired = {
            **{key: value for key, value in policy.items() if key != "bindings"},
            "version": version,
            "etag": etag,
            "bindings": unrelated,
        }
        transport.set_bucket_iam_policy(bucket=provider["bucket"], policy=desired)
    observed = dict(
        transport.get_bucket_iam_policy(bucket=provider["bucket"])
    )
    observed_bindings, _, _ = _policy_parts(observed)
    if observed_bindings != unrelated:
        raise DatasetProviderError("dataset worker IAM absence was not proven")
    core = {
        "schema": IAM_CLEANUP_SCHEMA,
        "provider_plan_sha256": provider["provider_plan_sha256"],
        "iam_receipt_sha256": installed["receipt_sha256"],
        "bindings_removed": len(expected),
        "readback_absent": True,
        "unrelated_bindings_preserved": True,
        "response_reconciled": response_reconciled,
        "cloud_mutated": not response_reconciled,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def validate_iam_cleanup_receipt(
    value: Mapping[str, Any],
    *,
    provider_plan: Mapping[str, Any],
    iam_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = deepcopy(dict(value))
    installed = validate_iam_receipt(
        iam_receipt, provider_plan=provider_plan
    )
    if (
        receipt.get("receipt_sha256")
        != _self_digest(receipt, "receipt_sha256")
        or receipt.get("schema") != IAM_CLEANUP_SCHEMA
        or receipt.get("provider_plan_sha256")
        != provider_plan["provider_plan_sha256"]
        or receipt.get("iam_receipt_sha256")
        != installed["receipt_sha256"]
        or receipt.get("bindings_removed")
        != len(provider_plan["workers"]) + 1
        or receipt.get("readback_absent") is not True
        or receipt.get("unrelated_bindings_preserved") is not True
        or not isinstance(receipt.get("response_reconciled"), bool)
        or receipt.get("cloud_mutated")
        is not (not receipt["response_reconciled"])
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("dataset IAM cleanup receipt changed")
    return receipt


def cleanup_wave(
    *,
    provider_plan: Mapping[str, Any],
    transport_plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    wave_request: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
    transport: DatasetCloudTransport,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    provider = _validate_context(
        provider_plan=provider_plan,
        transport_plan=transport_plan,
        ledger=ledger,
        resume=resume,
        wave_request=wave_request,
    )
    launch = validate_launch_receipt(
        launch_receipt, provider_plan=provider
    )
    launched_by_shard = {
        row["shard_id"]: row for row in launch["rows"]
    }
    workers = {row["shard_id"]: row for row in provider["workers"]}
    stage = launch["stage_receipt"]
    for selected in provider["selected_attempts"]:
        instance = transport.get_instance(instance_name=selected["instance_id"])
        if instance is None:
            continue
        spec = _instance_spec(
            provider=provider,
            worker=workers[selected["shard_id"]],
            selected=selected,
            stage_receipt=stage,
        )
        provider_id, disk_name, _status = _validate_owned_instance(
            instance, expected_spec=spec
        )
        disk = transport.get_disk_optional(disk_name=disk_name)
        if disk is None:
            raise PermissionError(
                "dataset boot disk disappeared before exact cleanup"
            )
        disk_id = _validate_owned_disk(disk, expected_spec=spec)
        launched = launched_by_shard[selected["shard_id"]]
        if (
            launched["provider_instance_id"] != provider_id
            or launched["provider_boot_disk_id"] != disk_id
        ):
            raise PermissionError(
                "dataset provider identity changed before cleanup"
            )
        request_id = _uuid_for(
            provider["provider_plan_sha256"],
            selected["shard_id"],
            selected["attempt_id"],
            "delete",
        )
        try:
            operation = transport.delete_instance(
                instance_name=selected["instance_id"], request_id=request_id
            )
        except c4_gcp.ResponseLostError:
            if (
                transport.get_instance(
                    instance_name=selected["instance_id"]
                )
                is None
            ):
                operation = None
            else:
                operation = transport.delete_instance(
                    instance_name=selected["instance_id"],
                    request_id=request_id,
                )
        if operation is not None:
            quality_provider._wait_operation(  # type: ignore[arg-type]
                transport,
                initial=operation,
                operation_type="delete",
                target_name=selected["instance_id"],
                sleep=sleep,
            )
    for attempt in range(ABSENCE_POLL_ATTEMPTS):
        remaining = [
            row
            for row in provider["selected_attempts"]
            if transport.get_instance(instance_name=row["instance_id"])
            is not None
            or transport.get_disk_optional(disk_name=row["instance_id"])
            is not None
        ]
        if not remaining:
            break
        if attempt + 1 < ABSENCE_POLL_ATTEMPTS:
            sleep(ABSENCE_POLL_INTERVAL_SECONDS)
    else:
        raise TimeoutError("dataset VM/disk absence was not proven")
    iam_cleanup = remove_worker_iam(
        provider_plan=provider,
        transport_plan=transport_plan,
        ledger=ledger,
        resume=resume,
        wave_request=wave_request,
        iam_receipt=launch["iam_receipt"],
        transport=transport,
    )
    observed_provider = {**provider, "_transport_plan": transport_plan}
    rows = [
        _attempt_observation(
            provider=observed_provider,
            selected=selected,
            transport=transport,
        )
        for selected in provider["selected_attempts"]
    ]
    core = {
        "schema": bridge.LIFECYCLE_SCHEMA,
        "plan_sha256": transport_plan["plan_sha256"],
        "ledger_sha256": ledger["ledger_sha256"],
        "resume_sha256": resume["resume_sha256"],
        "request_sha256": wave_request["request_sha256"],
        "wave_index": wave_request["wave_index"],
        "selected_attempt_count": len(rows),
        "attempt_rows": rows,
        "max_concurrent_vms": bridge.MAX_CONCURRENT_VMS,
        "create_only_gcs": True,
        "exact_owned_cleanup": True,
        "owned_vm_disk_absent": True,
        "worker_iam_removed_before_receive": True,
        "receiver_handoff_ready": True,
        "wildcard_delete_used": False,
        "unrelated_resource_touched": False,
        "gcs_evidence_deleted": False,
        "provider_plan_sha256": provider["provider_plan_sha256"],
        "launch_receipt_sha256": launch["receipt_sha256"],
        "iam_cleanup_receipt_sha256": iam_cleanup["receipt_sha256"],
        "current_profile_changed": False,
    }
    lifecycle = {**core, "receipt_sha256": canonical_sha256(core)}
    return bridge.validate_lifecycle_receipt(
        lifecycle,
        plan=transport_plan,
        ledger=ledger,
        resume=resume,
        wave_request=wave_request,
    )


def mirror_lifecycle_objects(
    *,
    lifecycle_receipt: Mapping[str, Any],
    bucket: str,
    output_root: str | Path,
    transport: DatasetCloudTransport,
) -> dict[str, Any]:
    """Create a local immutable mirror consumed by the neutral receiver."""

    root = Path(output_root)
    if root.exists() and (not root.is_dir() or root.is_symlink()):
        raise ValueError("dataset object mirror root is unsafe")
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for row in lifecycle_receipt["attempt_rows"]:
        checkpoint = row["checkpoint"]
        if checkpoint is None:
            continue
        objects = [
            {
                "object": checkpoint["object"],
                "generation": checkpoint["generation"],
                "sha256": checkpoint["sha256"],
                "bytes": checkpoint["bytes"],
            },
            *checkpoint["file_objects"],
        ]
        if row["heartbeat"] is not None:
            objects.append(
                {
                    key: row["heartbeat"][key]
                    for key in ("object", "generation", "sha256", "bytes")
                }
            )
        for record in objects:
            target = root.joinpath(*PurePosixPath(record["object"]).parts)
            payload = transport.get_object_bytes(
                bucket=bucket,
                object_name=record["object"],
                generation=record["generation"],
            )
            if (
                payload is None
                or len(payload) != record["bytes"]
                or hashlib.sha256(payload).hexdigest() != record["sha256"]
            ):
                raise DatasetProviderError("dataset mirror source changed")
            if target.exists():
                if target.read_bytes() != payload:
                    raise FileExistsError("dataset mirror object conflicts")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            records.append(deepcopy(record))
    return {
        "schema": "hu_m31_t3_dataset_gcp_object_mirror_v1",
        "lifecycle_receipt_sha256": lifecycle_receipt["receipt_sha256"],
        "record_count": len(records),
        "records_sha256": canonical_sha256(records),
        "create_only_local_mirror": True,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }


__all__ = [
    "DatasetCloudTransport",
    "DatasetProviderError",
    "GcpDatasetRestAdapter",
    "IAM_CLEANUP_SCHEMA",
    "IAM_RECEIPT_SCHEMA",
    "LAUNCH_RECEIPT_SCHEMA",
    "POLL_RECEIPT_SCHEMA",
    "PROVIDER_PLAN_SCHEMA",
    "STAGE_RECEIPT_SCHEMA",
    "build_provider_plan",
    "canonical_bytes",
    "canonical_sha256",
    "cleanup_wave",
    "execute_wave",
    "install_worker_iam",
    "mirror_lifecycle_objects",
    "poll_wave",
    "remove_worker_iam",
    "stage_content",
    "validate_iam_cleanup_receipt",
    "validate_iam_receipt",
    "validate_launch_receipt",
    "validate_poll_receipt",
    "validate_provider_plan",
    "validate_stage_receipt",
    "worker_startup_script",
]
