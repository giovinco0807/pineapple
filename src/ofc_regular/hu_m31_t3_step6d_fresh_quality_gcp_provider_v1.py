"""Live GCP provider adapter for the M3.1 fresh-quality 8+7 bridge.

The scientific contract and retry state live in
``hu_m31_t3_step6d_fresh_quality_gcp_bridge_v1``.  This module is the narrow
provider side: it stages hash-pinned content, installs two run/wave-scoped
bucket-IAM bindings, creates only the selected C4 Spot instances, polls exact
objects, deletes only those instances, proves VM/disk absence, removes the
temporary IAM bindings, and emits the bridge lifecycle receipt.

Credentials are accepted only by ``GcpQualityRestAdapter`` in memory.  They are
never accepted by a plan/receipt function and never serialized.  All mutating
functions require an injected transport, which keeps the contracts testable
without a network.  This module cannot alter an AI profile.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import uuid
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import hu_m31_t3_step6d_fresh_quality_gcp_bridge_v1 as bridge
from . import hu_m31_t3_step6d_fresh_quality_transport_v1 as quality_transport
from . import hu_m31_t3_step6d_full100_wave_gce_adapter_v2 as full100_gce
from . import hu_rl_c4_gcp_lifecycle as c4_gcp


PROVIDER_PLAN_SCHEMA = "hu_m31_t3_step6d_fresh_quality_gcp_provider_plan_v1"
STAGE_RECEIPT_SCHEMA = "hu_m31_t3_step6d_fresh_quality_gcp_content_stage_v1"
IAM_RECEIPT_SCHEMA = "hu_m31_t3_step6d_fresh_quality_gcp_worker_iam_v1"
LAUNCH_RECEIPT_SCHEMA = "hu_m31_t3_step6d_fresh_quality_gcp_launch_v1"
POLL_RECEIPT_SCHEMA = "hu_m31_t3_step6d_fresh_quality_gcp_poll_v1"
IAM_CLEANUP_SCHEMA = "hu_m31_t3_step6d_fresh_quality_gcp_worker_iam_cleanup_v1"

PROJECT = full100_gce.PROJECT
REGION = full100_gce.REGION
ZONE = full100_gce.ZONE
MACHINE_TYPE = full100_gce.MACHINE_TYPE
BOOT_DISK_TYPE = full100_gce.BOOT_DISK_TYPE
BOOT_DISK_INTERFACE = full100_gce.BOOT_DISK_INTERFACE
BOOT_DISK_SIZE_GB = full100_gce.BOOT_DISK_SIZE_GB
NETWORK = full100_gce.NETWORK
SUBNETWORK = full100_gce.SUBNETWORK
NIC_TYPE = full100_gce.NIC_TYPE
OAUTH_SCOPE = full100_gce.OAUTH_SCOPE
NAT_ROUTER_NAME = "ofc-t3-nat-router-asia-northeast1"
NAT_NAME = "ofc-t3-nat-asia-northeast1"
C4_QUOTA_ID = "CPUS-PER-VM-FAMILY-per-project-region"
GLOBAL_QUOTA_ID = "CPUS-ALL-REGIONS-per-project"

MAX_RUN_SECONDS = 5_400
WATCHDOG_SECONDS = 5_100
IAM_TTL_SECONDS = 7_200
OPERATION_POLL_ATTEMPTS = 120
OPERATION_POLL_INTERVAL_SECONDS = 2.0
INSTANCE_POLL_ATTEMPTS = 90
INSTANCE_POLL_INTERVAL_SECONDS = 2.0
ABSENCE_POLL_ATTEMPTS = 60
ABSENCE_POLL_INTERVAL_SECONDS = 2.0

_SHA = re.compile(r"^[0-9a-f]{64}$")
_SERVICE_ACCOUNT = re.compile(
    rf"^[a-z][a-z0-9-]{{4,28}}[a-z0-9]@{re.escape(PROJECT)}"
    r"\.iam\.gserviceaccount\.com$"
)
_IMAGE_LINK = re.compile(
    r"^https://www\.googleapis\.com/compute/v1/projects/"
    r"debian-cloud/global/images/[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?$"
)
_PROVIDER_ID = re.compile(r"^[1-9][0-9]*$")
_OPERATION = re.compile(r"^[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?$")
_UTC = re.compile(
    r"^(?:19|20)[0-9]{2}-(?:0[1-9]|1[0-2])-"
    r"(?:0[1-9]|[12][0-9]|3[01])T(?:[01][0-9]|2[0-3]):"
    r"[0-5][0-9]:[0-5][0-9]Z$"
)

_PROVIDER_PLAN_KEYS = frozenset(
    {
        "schema",
        "status",
        "bridge_plan_sha256",
        "ledger_sha256",
        "resume_sha256",
        "wave_request_sha256",
        "run_name",
        "execution_identity_sha256",
        "project",
        "region",
        "zone",
        "bucket",
        "image",
        "network",
        "workers",
        "content_entries",
        "claim_object",
        "iam_contract",
        "runtime_contract",
        "selected_attempts",
        "startup_script_sha256",
        "cloud_mutated",
        "current_profile_changed",
        "provider_plan_sha256",
    }
)
_IMAGE_KEYS = frozenset({"self_link", "id", "guest_os_features"})
_NETWORK_KEYS = frozenset(
    {
        "network",
        "subnetwork",
        "nic_type",
        "external_ipv4",
        "cloud_nat_required",
        "nat_router_name",
        "nat_name",
    }
)
_WORKER_KEYS = frozenset(
    {"slot", "job_id", "attempt_id", "instance_id", "service_account"}
)
_CONTENT_KEYS = frozenset(
    {"kind", "local_path", "object_name", "sha256", "bytes", "content_type"}
)
_IAM_CONTRACT_KEYS = frozenset(
    {
        "viewer_role",
        "creator_role",
        "members",
        "reader_prefix",
        "creator_prefix",
        "condition_titles",
        "ttl_seconds",
    }
)
_RUNTIME_KEYS = frozenset(
    {
        "machine_type",
        "vcpus_per_vm",
        "max_run_seconds",
        "watchdog_seconds",
        "boot_disk_type",
        "boot_disk_interface",
        "boot_disk_size_gb",
        "provisioning_model",
        "instance_termination_action",
        "startup_has_network_install",
    }
)
_STAGE_KEYS = frozenset(
    {
        "schema",
        "status",
        "provider_plan_sha256",
        "content_records",
        "record_count",
        "all_bytes_replayed",
        "create_only_or_identical_reuse",
        "cloud_mutated",
        "current_profile_changed",
        "receipt_sha256",
    }
)
_OBJECT_RECORD_KEYS = frozenset(
    {"kind", "object_name", "generation", "sha256", "bytes", "created"}
)
_IAM_KEYS = frozenset(
    {
        "schema",
        "status",
        "provider_plan_sha256",
        "wave_request_sha256",
        "bucket",
        "policy_etag_before_sha256",
        "policy_etag_after_sha256",
        "condition_titles",
        "members",
        "expires_at_utc",
        "bindings_installed",
        "readback_exact",
        "cloud_mutated",
        "current_profile_changed",
        "receipt_sha256",
    }
)
_LAUNCH_KEYS = frozenset(
    {
        "schema",
        "status",
        "provider_plan_sha256",
        "bridge_plan_sha256",
        "ledger_sha256",
        "resume_sha256",
        "wave_request_sha256",
        "run_name",
        "execution_identity_sha256",
        "wave_index",
        "claim",
        "stage_receipt",
        "iam_receipt",
        "quota_receipt",
        "rows",
        "selected_job_count",
        "created_instance_count",
        "create_complete",
        "one_shot_consumed",
        "unlisted_vm_created",
        "cloud_mutated",
        "current_profile_changed",
        "receipt_sha256",
    }
)
_CLAIM_KEYS = frozenset(
    {
        "object_name",
        "generation",
        "sha256",
        "bytes",
        "nonce_sha256",
        "response_reconciled",
    }
)
_QUOTA_KEYS = frozenset(
    {
        "source",
        "metrics",
        "required_vcpus",
        "sufficient",
        "observed_at_utc",
    }
)
_LAUNCH_ROW_KEYS = frozenset(
    {
        "job_id",
        "attempt_id",
        "instance_id",
        "service_account",
        "request_id",
        "spec_sha256",
        "provider_instance_id",
        "provider_boot_disk_id",
        "observed_status",
        "created",
    }
)
_POLL_KEYS = frozenset(
    {
        "schema",
        "status",
        "provider_plan_sha256",
        "launch_receipt_sha256",
        "rows",
        "done_count",
        "selected_job_count",
        "read_only",
        "cloud_mutated",
        "current_profile_changed",
        "receipt_sha256",
    }
)
_POLL_ROW_KEYS = frozenset(
    {"job_id", "attempt_id", "instance_id", "instance_status", "done_generation"}
)
_IAM_CLEANUP_KEYS = frozenset(
    {
        "schema",
        "status",
        "provider_plan_sha256",
        "iam_receipt_sha256",
        "bucket",
        "condition_titles",
        "bindings_removed",
        "readback_absent",
        "unrelated_bindings_preserved",
        "cloud_mutated",
        "current_profile_changed",
        "receipt_sha256",
    }
)


class QualityProviderError(RuntimeError):
    """Provider state is incomplete, ambiguous, or outside the frozen plan."""


class QualityCloudTransport(Protocol):
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
    ) -> Mapping[str, Any] | None: ...

    def get_disk_optional(self, *, disk_name: str) -> Mapping[str, Any] | None: ...

    def delete_disk(
        self, *, disk_name: str, request_id: str
    ) -> Mapping[str, Any] | None: ...

    def get_bucket_iam_policy(self, *, bucket: str) -> Mapping[str, Any]: ...

    def set_bucket_iam_policy(
        self, *, bucket: str, policy: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def get_service_account(self, *, email: str) -> Mapping[str, Any]: ...

    def test_service_account_act_as(self, *, email: str) -> bool: ...

    def get_region_quota(self, *, region: str) -> Mapping[str, Any]: ...

    def get_image(self, *, self_link: str) -> Mapping[str, Any]: ...

    def get_router(self, *, region: str, router_name: str) -> Mapping[str, Any]: ...

    def get_cloud_quota(self, *, quota_id: str) -> Mapping[str, Any]: ...

    def list_instances(self) -> Sequence[Mapping[str, Any]]: ...

    def get_machine_type_url(self, *, self_link: str) -> Mapping[str, Any]: ...


def canonical_bytes(value: Any) -> bytes:
    return bridge.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return bridge.canonical_sha256(value)


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(f"{label} fields changed: missing={missing}, extra={extra}")


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _self_digest(value: Mapping[str, Any], field: str) -> str:
    payload = deepcopy(dict(value))
    payload.pop(field, None)
    return canonical_sha256(payload)


def _uuid_for(*parts: str) -> str:
    raw = hashlib.sha256("\0".join(parts).encode("ascii")).hexdigest()
    return str(uuid.UUID(raw[:32], version=4))


def _utc(epoch_seconds: int) -> str:
    return (
        dt.datetime.fromtimestamp(epoch_seconds, tz=dt.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )


def _content_type(path: Path) -> str:
    if path.suffix == ".json":
        return "application/json"
    if path.suffix == ".sh":
        return "text/x-shellscript"
    if path.suffix in {".zip", ".whl"}:
        return "application/zip"
    return "application/octet-stream"


def _plain_file(path: str | Path, label: str) -> Path:
    target = Path(path).resolve()
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    return target


def _content_entry(
    *, kind: str, path: Path, object_name: str
) -> dict[str, Any]:
    return {
        "kind": kind,
        "local_path": str(path),
        "object_name": object_name,
        "sha256": bridge.sha256_file(path),
        "bytes": path.stat().st_size,
        "content_type": _content_type(path),
    }


def build_provider_plan(
    *,
    bridge_plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    wave_request: Mapping[str, Any],
    image_self_link: str,
    image_id: str,
    guest_os_features: Sequence[str],
    worker_service_accounts: Sequence[str],
) -> dict[str, Any]:
    plan = bridge.validate_gcp_plan(bridge_plan, replay_sources=True)
    checked_ledger = bridge.validate_attempt_ledger(plan, ledger)
    checked_resume = bridge.validate_resume_plan(plan, checked_ledger, resume)
    request = bridge.validate_wave_request(
        plan, checked_ledger, checked_resume, wave_request
    )
    if (
        plan["project"] != PROJECT
        or plan["region"] != REGION
        or plan["zone"] != ZONE
        or not isinstance(image_self_link, str)
        or _IMAGE_LINK.fullmatch(image_self_link) is None
        or not isinstance(image_id, str)
        or _PROVIDER_ID.fullmatch(image_id) is None
    ):
        raise ValueError("fresh-quality live GCP target or image changed")
    features = sorted(set(guest_os_features))
    if (
        not features
        or len(features) != len(guest_os_features)
        or any(not isinstance(item, str) or not item for item in features)
    ):
        raise ValueError("fresh-quality image guest features are invalid")
    accounts = list(worker_service_accounts)
    selected = request["selected_attempts"]
    if (
        len(accounts) != bridge.MAX_CONCURRENT_VMS
        or len(set(accounts)) != len(accounts)
        or any(
            not isinstance(email, str) or _SERVICE_ACCOUNT.fullmatch(email) is None
            for email in accounts
        )
        or len(selected) > len(accounts)
    ):
        raise ValueError("exact eight fresh-quality worker accounts are required")
    workers = [
        {
            "slot": index,
            "job_id": row["job_id"],
            "attempt_id": row["attempt_id"],
            "instance_id": row["instance_id"],
            "service_account": accounts[index],
        }
        for index, row in enumerate(selected)
    ]
    staging = Path(plan["source_paths"]["staging_directory"]).resolve()
    launch = plan["launch_manifest"]
    launch_path = _plain_file(
        plan["source_paths"]["launch_manifest"], "fresh-quality launch manifest"
    )
    content_base = (
        f"{plan['artifact_contract']['content_prefix']}"
        f"{plan['launch_manifest_sha256'][:20]}/"
    )
    entries = [
        _content_entry(
            kind="launch_manifest",
            path=launch_path,
            object_name=f"{content_base}launch.json",
        )
    ]
    for field in (
        "quality_package",
        "runtime_source",
        "runtime_source_manifest",
        "wheelhouse",
        "wheelhouse_manifest",
        "startup",
    ):
        record = launch[field]
        relative = PurePosixPath(record["path"])
        path = _plain_file(
            staging.joinpath(*relative.parts), f"fresh-quality {field}"
        )
        if (
            bridge.sha256_file(path) != record["sha256"]
            or path.stat().st_size != record["bytes"]
        ):
            raise ValueError(f"fresh-quality staged {field} changed")
        entries.append(
            _content_entry(
                kind=field,
                path=path,
                object_name=f"{content_base}{relative.as_posix()}",
            )
        )
    titles = [
        f"ofc-fq-read-{plan['execution_identity_sha256'][:12]}-"
        f"w{checked_resume['resume_wave_index']}",
        *[
            f"ofc-fq-create-{plan['execution_identity_sha256'][:12]}-"
            f"w{checked_resume['resume_wave_index']}-s{index:02d}"
            for index in range(len(workers))
        ],
    ]
    startup = worker_startup_script()
    core = {
        "schema": PROVIDER_PLAN_SCHEMA,
        "status": "provider_plan_ready_cloud_not_mutated",
        "bridge_plan_sha256": plan["plan_sha256"],
        "ledger_sha256": checked_ledger["ledger_sha256"],
        "resume_sha256": checked_resume["resume_sha256"],
        "wave_request_sha256": request["request_sha256"],
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
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
        "claim_object": (
            f"{plan['artifact_contract']['claim_prefix']}"
            f"{request['request_sha256']}.json"
        ),
        "iam_contract": {
            "viewer_role": "roles/storage.objectViewer",
            "creator_role": "roles/storage.objectCreator",
            "members": [f"serviceAccount:{row['service_account']}" for row in workers],
            "reader_prefix": plan["artifact_contract"]["content_prefix"],
            "creator_prefix": plan["artifact_contract"]["result_prefix"],
            "condition_titles": titles,
            "ttl_seconds": IAM_TTL_SECONDS,
        },
        "runtime_contract": {
            "machine_type": MACHINE_TYPE,
            "vcpus_per_vm": bridge.VCPUS_PER_VM,
            "max_run_seconds": MAX_RUN_SECONDS,
            "watchdog_seconds": WATCHDOG_SECONDS,
            "boot_disk_type": BOOT_DISK_TYPE,
            "boot_disk_interface": BOOT_DISK_INTERFACE,
            "boot_disk_size_gb": BOOT_DISK_SIZE_GB,
            "provisioning_model": "SPOT",
            "instance_termination_action": "DELETE",
            "startup_has_network_install": False,
        },
        "selected_attempts": deepcopy(selected),
        "startup_script_sha256": hashlib.sha256(startup.encode("utf-8")).hexdigest(),
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    result = {**core, "provider_plan_sha256": canonical_sha256(core)}
    return validate_provider_plan(
        result,
        bridge_plan=plan,
        ledger=checked_ledger,
        resume=checked_resume,
        wave_request=request,
    )


def validate_provider_plan(
    value: Mapping[str, Any],
    *,
    bridge_plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    wave_request: Mapping[str, Any],
) -> dict[str, Any]:
    plan = bridge.validate_gcp_plan(bridge_plan, replay_sources=True)
    checked_ledger = bridge.validate_attempt_ledger(plan, ledger)
    checked_resume = bridge.validate_resume_plan(plan, checked_ledger, resume)
    request = bridge.validate_wave_request(
        plan, checked_ledger, checked_resume, wave_request
    )
    provider = deepcopy(dict(value))
    _exact_keys(provider, _PROVIDER_PLAN_KEYS, "fresh-quality provider plan")
    if provider.get("provider_plan_sha256") != _self_digest(
        provider, "provider_plan_sha256"
    ):
        raise ValueError("fresh-quality provider plan digest changed")
    _exact_keys(provider.get("image", {}), _IMAGE_KEYS, "provider image")
    _exact_keys(provider.get("network", {}), _NETWORK_KEYS, "provider network")
    _exact_keys(
        provider.get("iam_contract", {}), _IAM_CONTRACT_KEYS, "provider IAM contract"
    )
    _exact_keys(
        provider.get("runtime_contract", {}), _RUNTIME_KEYS, "provider runtime contract"
    )
    workers = provider.get("workers")
    entries = provider.get("content_entries")
    if not isinstance(workers, list) or not isinstance(entries, list):
        raise ValueError("fresh-quality provider collections are missing")
    for row in workers:
        if not isinstance(row, Mapping):
            raise ValueError("fresh-quality provider worker is not an object")
        _exact_keys(row, _WORKER_KEYS, "fresh-quality provider worker")
    seen_objects: set[str] = set()
    for row in entries:
        if not isinstance(row, Mapping):
            raise ValueError("fresh-quality content entry is not an object")
        _exact_keys(row, _CONTENT_KEYS, "fresh-quality content entry")
        local = _plain_file(row["local_path"], "fresh-quality provider content")
        if (
            row["object_name"] in seen_objects
            or bridge.sha256_file(local) != row["sha256"]
            or local.stat().st_size != row["bytes"]
        ):
            raise ValueError("fresh-quality provider content changed")
        seen_objects.add(row["object_name"])
    expected_selected = request["selected_attempts"]
    expected_members = [
        f"serviceAccount:{row['service_account']}" for row in workers
    ]
    expected_titles = [
        f"ofc-fq-read-{plan['execution_identity_sha256'][:12]}-"
        f"w{checked_resume['resume_wave_index']}",
        *[
            f"ofc-fq-create-{plan['execution_identity_sha256'][:12]}-"
            f"w{checked_resume['resume_wave_index']}-s{index:02d}"
            for index in range(len(workers))
        ],
    ]
    iam = provider["iam_contract"]
    runtime = provider["runtime_contract"]
    if (
        provider.get("schema") != PROVIDER_PLAN_SCHEMA
        or provider.get("status") != "provider_plan_ready_cloud_not_mutated"
        or provider.get("bridge_plan_sha256") != plan["plan_sha256"]
        or provider.get("ledger_sha256") != checked_ledger["ledger_sha256"]
        or provider.get("resume_sha256") != checked_resume["resume_sha256"]
        or provider.get("wave_request_sha256") != request["request_sha256"]
        or provider.get("run_name") != plan["run_name"]
        or provider.get("execution_identity_sha256")
        != plan["execution_identity_sha256"]
        or provider.get("project") != PROJECT
        or provider.get("region") != REGION
        or provider.get("zone") != ZONE
        or provider.get("bucket") != plan["bucket"]
        or _IMAGE_LINK.fullmatch(str(provider["image"].get("self_link", ""))) is None
        or _PROVIDER_ID.fullmatch(str(provider["image"].get("id", ""))) is None
        or provider["image"].get("guest_os_features")
        != sorted(set(provider["image"].get("guest_os_features", [])))
        or provider.get("network")
        != {
            "network": NETWORK,
            "subnetwork": SUBNETWORK,
            "nic_type": NIC_TYPE,
            "external_ipv4": False,
            "cloud_nat_required": True,
            "nat_router_name": NAT_ROUTER_NAME,
            "nat_name": NAT_NAME,
        }
        or provider.get("selected_attempts") != expected_selected
        or len(workers) != len(expected_selected)
        or len({row["service_account"] for row in workers}) != len(workers)
        or any(
            _SERVICE_ACCOUNT.fullmatch(str(row["service_account"])) is None
            for row in workers
        )
        or any(
            worker["slot"] != index
            or worker["job_id"] != expected_selected[index]["job_id"]
            or worker["attempt_id"] != expected_selected[index]["attempt_id"]
            or worker["instance_id"] != expected_selected[index]["instance_id"]
            for index, worker in enumerate(workers)
        )
        or [row["kind"] for row in entries]
        != [
            "launch_manifest",
            "quality_package",
            "runtime_source",
            "runtime_source_manifest",
            "wheelhouse",
            "wheelhouse_manifest",
            "startup",
        ]
        or iam
        != {
            "viewer_role": "roles/storage.objectViewer",
            "creator_role": "roles/storage.objectCreator",
            "members": expected_members,
            "reader_prefix": plan["artifact_contract"]["content_prefix"],
            "creator_prefix": plan["artifact_contract"]["result_prefix"],
            "condition_titles": expected_titles,
            "ttl_seconds": IAM_TTL_SECONDS,
        }
        or runtime
        != {
            "machine_type": MACHINE_TYPE,
            "vcpus_per_vm": bridge.VCPUS_PER_VM,
            "max_run_seconds": MAX_RUN_SECONDS,
            "watchdog_seconds": WATCHDOG_SECONDS,
            "boot_disk_type": BOOT_DISK_TYPE,
            "boot_disk_interface": BOOT_DISK_INTERFACE,
            "boot_disk_size_gb": BOOT_DISK_SIZE_GB,
            "provisioning_model": "SPOT",
            "instance_termination_action": "DELETE",
            "startup_has_network_install": False,
        }
        or provider.get("startup_script_sha256")
        != hashlib.sha256(worker_startup_script().encode("utf-8")).hexdigest()
        or provider.get("cloud_mutated") is not False
        or provider.get("current_profile_changed") is not False
    ):
        raise ValueError("fresh-quality provider plan binding changed")
    return provider


def worker_startup_script() -> str:
    """Return the network-install-free startup used by every quality worker."""

    return r"""#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=/var/lib/ofc-fresh-quality-v1
STAGING=$ROOT/staging
LOG=/var/log/ofc-fresh-quality-v1
META='http://metadata.google.internal/computeMetadata/v1/instance/attributes'
HEADER='Metadata-Flavor: Google'
mkdir -p "$ROOT" "$STAGING" "$LOG"
chmod 0700 "$ROOT" "$STAGING" "$LOG"
exec >>"$LOG/startup.log" 2>&1

meta() { curl -fsS -H "$HEADER" "$META/$1"; }
token() {
  curl -fsS -H "$HEADER" \
    'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token' |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])'
}
enc() { python3 -c 'import sys,urllib.parse; print(urllib.parse.quote(sys.argv[1],safe=""))' "$1"; }
sha() { sha256sum "$1" | awk '{print $1}'; }

BUCKET="$(meta fq-bucket)"
JOB_ID="$(meta fq-job-id)"
ARTIFACT_PREFIX="$(meta fq-artifact-prefix)"
BINDINGS_B64="$(meta fq-content-bindings-b64)"
LAUNCH_RELATIVE="$(meta fq-launch-relative)"
EXPECTED_STARTUP_SHA="$(meta fq-local-startup-sha256)"
WATCHDOG="$(meta fq-watchdog-seconds)"

( sleep "$WATCHDOG"; shutdown -h now ) &
WATCHDOG_PID=$!
trap 'kill "$WATCHDOG_PID" >/dev/null 2>&1 || true' EXIT

python3 - "$STAGING" "$BUCKET" "$BINDINGS_B64" <<'PY'
import base64, hashlib, json, os, pathlib, sys, urllib.parse, urllib.request
root=pathlib.Path(sys.argv[1]).resolve()
bucket=sys.argv[2]
raw=base64.b64decode(sys.argv[3], validate=True)
bindings=json.loads(raw.decode("ascii"))
if raw != json.dumps(bindings,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii"):
    raise SystemExit("content bindings are not canonical")
if not isinstance(bindings,list) or not bindings:
    raise SystemExit("content bindings are missing")
token=json.load(urllib.request.urlopen(urllib.request.Request(
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
    headers={"Metadata-Flavor":"Google"})))["access_token"]
seen=set()
for row in bindings:
    if set(row)!={"relative","object","generation","sha256","bytes"}:
        raise SystemExit("content binding fields changed")
    rel=pathlib.PurePosixPath(row["relative"])
    if rel.is_absolute() or ".." in rel.parts or rel.as_posix()!=row["relative"] or row["relative"] in seen:
        raise SystemExit("unsafe or duplicate content path")
    seen.add(row["relative"])
    url=("https://storage.googleapis.com/download/storage/v1/b/"
         +urllib.parse.quote(bucket,safe="")+"/o/"
         +urllib.parse.quote(row["object"],safe="")+"?alt=media&generation="
         +urllib.parse.quote(row["generation"],safe=""))
    req=urllib.request.Request(url,headers={"Authorization":"Bearer "+token})
    payload=urllib.request.urlopen(req,timeout=600).read()
    if hashlib.sha256(payload).hexdigest()!=row["sha256"] or len(payload)!=row["bytes"]:
        raise SystemExit("downloaded content changed")
    target=root.joinpath(*rel.parts)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("xb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
PY

LOCAL_STARTUP="$STAGING/$(B="$BINDINGS_B64" python3 -c 'import json,os; b=json.loads(__import__("base64").b64decode(os.environ["B"]).decode()); print(next(x["relative"] for x in b if x["relative"].endswith(".sh")))')"
[[ -f "$LOCAL_STARTUP" && ! -L "$LOCAL_STARTUP" ]]
[[ "$(sha "$LOCAL_STARTUP")" == "$EXPECTED_STARTUP_SHA" ]]
chmod 0700 "$LOCAL_STARTUP"
bash "$LOCAL_STARTUP" "$STAGING" "$STAGING/$LAUNCH_RELATIVE" "$JOB_ID" "$ROOT/work" run

python3 - "$ROOT/work/output" "$BUCKET" "$ARTIFACT_PREFIX" <<'PY'
import hashlib, json, os, pathlib, sys, urllib.parse, urllib.request
output=pathlib.Path(sys.argv[1]).resolve(strict=True)
bucket=sys.argv[2]; prefix=sys.argv[3]
done_path=output/"DONE.json"
raw=done_path.read_bytes(); done=json.loads(raw.decode("ascii"))
canonical=lambda v: json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
if raw!=canonical(done) or done.get("status")!="complete_validated_quality_job" or done.get("done_published_last") is not True:
    raise SystemExit("local DONE is invalid")
paths=[]
for record in done.get("task_records",[]):
    rel=pathlib.PurePosixPath(record["path"])
    if rel.is_absolute() or ".." in rel.parts or rel.parts[:1]!=("tasks",):
        raise SystemExit("unsafe task record")
    paths.append((output.joinpath(*rel.parts), prefix+"/tasks/"+rel.name))
result_rel=pathlib.PurePosixPath(done["result_path"])
if result_rel.is_absolute() or ".." in result_rel.parts:
    raise SystemExit("unsafe result path")
paths.append((output.joinpath(*result_rel.parts),prefix+"/result.json"))
paths.append((done_path,prefix+"/DONE.json"))
token=json.load(urllib.request.urlopen(urllib.request.Request(
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
    headers={"Metadata-Flavor":"Google"})))["access_token"]
for path,obj in paths:
    payload=path.read_bytes()
    url=("https://storage.googleapis.com/upload/storage/v1/b/"
         +urllib.parse.quote(bucket,safe="")+"/o?uploadType=media&ifGenerationMatch=0&name="
         +urllib.parse.quote(obj,safe=""))
    req=urllib.request.Request(url,data=payload,method="POST",headers={
        "Authorization":"Bearer "+token,"Content-Type":"application/octet-stream"})
    urllib.request.urlopen(req,timeout=600).read()
PY

kill "$WATCHDOG_PID" >/dev/null 2>&1 || true
shutdown -h now
"""


class GcpQualityRestAdapter(c4_gcp.GcpRestTransport):
    """Concrete OAuth REST adapter; the token remains only in this object."""

    def __init__(
        self,
        *,
        access_token: str,
        zone: str = ZONE,
        request: Callable[..., c4_gcp.HttpResponse] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        token_provider: Callable[[], str] | None = None,
    ) -> None:
        # A quality wave polls for far longer than an access token lives, so a
        # refresher is accepted here; without one a 401 stays fatal.
        # `zone` defaults to the historical target; a multi-region label fleet
        # names one of the transport's approved zones instead, and every
        # zone-scoped URL below reads self.zone so the two cannot drift.
        super().__init__(
            access_token=access_token,
            project=PROJECT,
            zone=zone,
            request=request,
            sleep=sleep,
            token_provider=token_provider,
        )

    def delete_instance(
        self, *, instance_name: str, request_id: str
    ) -> Mapping[str, Any] | None:
        url = (
            f"https://compute.googleapis.com/compute/v1/projects/{PROJECT}/"
            f"zones/{self.zone}/instances/"
            f"{urllib.parse.quote(instance_name, safe='')}?"
            + urllib.parse.urlencode({"requestId": request_id})
        )
        try:
            response = self._call("DELETE", url, allow_404=True)
        except (TimeoutError, urllib.error.URLError) as exc:
            raise c4_gcp.ResponseLostError(
                "GCE instance delete response was lost"
            ) from exc
        return (
            None
            if response is None
            else self._json(response, "GCE instance delete")
        )

    def get_disk_optional(self, *, disk_name: str) -> Mapping[str, Any] | None:
        url = (
            f"https://compute.googleapis.com/compute/v1/projects/{PROJECT}/"
            f"zones/{self.zone}/disks/{urllib.parse.quote(disk_name, safe='')}"
        )
        response = self._call("GET", url, allow_404=True)
        return None if response is None else self._json(response, "GCE disk get")

    def delete_disk(
        self, *, disk_name: str, request_id: str
    ) -> Mapping[str, Any] | None:
        url = (
            f"https://compute.googleapis.com/compute/v1/projects/{PROJECT}/"
            f"zones/{self.zone}/disks/"
            f"{urllib.parse.quote(disk_name, safe='')}?"
            + urllib.parse.urlencode({"requestId": request_id})
        )
        try:
            response = self._call("DELETE", url, allow_404=True)
        except (TimeoutError, urllib.error.URLError) as exc:
            raise c4_gcp.ResponseLostError(
                "GCE disk delete response was lost"
            ) from exc
        return None if response is None else self._json(response, "GCE disk delete")

    def get_bucket_iam_policy(self, *, bucket: str) -> Mapping[str, Any]:
        url = (
            "https://storage.googleapis.com/storage/v1/b/"
            + urllib.parse.quote(bucket, safe="")
            + "/iam?"
            + urllib.parse.urlencode({"optionsRequestedPolicyVersion": "3"})
        )
        return self._json(self._call("GET", url), "GCS bucket IAM get")

    def set_bucket_iam_policy(
        self, *, bucket: str, policy: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        url = (
            "https://storage.googleapis.com/storage/v1/b/"
            + urllib.parse.quote(bucket, safe="")
            + "/iam"
        )
        return self._json(
            self._call("PUT", url, payload=canonical_bytes(policy)),
            "GCS bucket IAM set",
        )

    def get_service_account(self, *, email: str) -> Mapping[str, Any]:
        url = (
            f"https://iam.googleapis.com/v1/projects/{PROJECT}/serviceAccounts/"
            + urllib.parse.quote(email, safe="")
        )
        return self._json(self._call("GET", url), "worker service account get")

    def test_service_account_act_as(self, *, email: str) -> bool:
        url = (
            f"https://iam.googleapis.com/v1/projects/{PROJECT}/serviceAccounts/"
            + urllib.parse.quote(email, safe="")
            + ":testIamPermissions"
        )
        result = self._json(
            self._call(
                "POST",
                url,
                payload=canonical_bytes(
                    {"permissions": ["iam.serviceAccounts.actAs"]}
                ),
            ),
            "worker service account actAs test",
        )
        return result.get("permissions") == ["iam.serviceAccounts.actAs"]

    def get_region_quota(self, *, region: str) -> Mapping[str, Any]:
        url = (
            f"https://compute.googleapis.com/compute/v1/projects/{PROJECT}/regions/"
            + urllib.parse.quote(region, safe="")
        )
        return self._json(self._call("GET", url), "GCE region quota")

    def get_image(self, *, self_link: str) -> Mapping[str, Any]:
        if _IMAGE_LINK.fullmatch(self_link) is None:
            raise ValueError("image self-link changed")
        return self._json(self._call("GET", self_link), "GCE image get")

    def get_router(self, *, region: str, router_name: str) -> Mapping[str, Any]:
        url = (
            f"https://compute.googleapis.com/compute/v1/projects/{PROJECT}/regions/"
            f"{urllib.parse.quote(region, safe='')}/routers/"
            f"{urllib.parse.quote(router_name, safe='')}"
        )
        return self._json(self._call("GET", url), "GCE Cloud NAT router get")

    def get_cloud_quota(self, *, quota_id: str) -> Mapping[str, Any]:
        url = (
            f"https://cloudquotas.googleapis.com/v1/projects/{PROJECT}/locations/"
            "global/services/compute.googleapis.com/quotaInfos/"
            + urllib.parse.quote(quota_id, safe="")
        )
        return self._json(self._call("GET", url), "Cloud Quotas quotaInfo get")

    def list_instances(self) -> Sequence[Mapping[str, Any]]:
        url = (
            f"https://compute.googleapis.com/compute/v1/projects/{PROJECT}/"
            "aggregated/instances"
        )
        result: list[dict[str, Any]] = []
        token: str | None = None
        while True:
            query = {"maxResults": "500"}
            if token is not None:
                query["pageToken"] = token
            page = self._json(
                self._call("GET", url + "?" + urllib.parse.urlencode(query)),
                "GCE aggregated instances",
            )
            items = page.get("items", {})
            if not isinstance(items, Mapping):
                raise QualityProviderError("GCE aggregated instance items changed")
            for scope, scoped in items.items():
                rows = scoped.get("instances", []) if isinstance(scoped, Mapping) else []
                if not isinstance(rows, list):
                    raise QualityProviderError("GCE aggregated instance rows changed")
                for row in rows:
                    if not isinstance(row, Mapping):
                        raise QualityProviderError("GCE aggregated instance changed")
                    result.append({**deepcopy(dict(row)), "_scope": str(scope)})
            token_value = page.get("nextPageToken")
            if token_value is None:
                break
            if not isinstance(token_value, str) or not token_value:
                raise QualityProviderError("GCE instance pagination token changed")
            token = token_value
        return result

    def get_machine_type_url(self, *, self_link: str) -> Mapping[str, Any]:
        if not isinstance(self_link, str) or not self_link.startswith(
            f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/zones/"
        ):
            raise ValueError("GCE machine-type self-link changed")
        return self._json(self._call("GET", self_link), "GCE machine type get")


def _validate_provider_context(
    *,
    provider_plan: Mapping[str, Any],
    bridge_plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    wave_request: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan = bridge.validate_gcp_plan(bridge_plan, replay_sources=True)
    checked_ledger = bridge.validate_attempt_ledger(plan, ledger)
    checked_resume = bridge.validate_resume_plan(plan, checked_ledger, resume)
    request = bridge.validate_wave_request(
        plan, checked_ledger, checked_resume, wave_request
    )
    provider = validate_provider_plan(
        provider_plan,
        bridge_plan=plan,
        ledger=checked_ledger,
        resume=checked_resume,
        wave_request=request,
    )
    return provider, plan, checked_ledger, checked_resume, request


def _metadata_record(
    *,
    kind: str,
    object_name: str,
    metadata: Mapping[str, Any],
    payload: bytes,
    created: bool,
) -> dict[str, Any]:
    generation = str(metadata.get("generation", ""))
    if (
        metadata.get("name") != object_name
        or _PROVIDER_ID.fullmatch(generation) is None
    ):
        raise QualityProviderError("GCS object metadata identity changed")
    return {
        "kind": kind,
        "object_name": object_name,
        "generation": generation,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "created": created,
    }


def stage_content(
    *,
    provider_plan: Mapping[str, Any],
    bridge_plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    wave_request: Mapping[str, Any],
    transport: QualityCloudTransport,
) -> dict[str, Any]:
    provider, _plan, _ledger, _resume, _request = _validate_provider_context(
        provider_plan=provider_plan,
        bridge_plan=bridge_plan,
        ledger=ledger,
        resume=resume,
        wave_request=wave_request,
    )
    records: list[dict[str, Any]] = []
    any_created = False
    for entry in provider["content_entries"]:
        payload = _plain_file(entry["local_path"], "provider content").read_bytes()
        if (
            hashlib.sha256(payload).hexdigest() != entry["sha256"]
            or len(payload) != entry["bytes"]
        ):
            raise ValueError("fresh-quality provider content changed before stage")
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
            any_created = True
        generation = str(metadata.get("generation", ""))
        observed = transport.get_object_bytes(
            bucket=provider["bucket"],
            object_name=entry["object_name"],
            generation=generation,
        )
        if observed != payload:
            raise QualityProviderError(
                "staged fresh-quality object differs from immutable local bytes"
            )
        records.append(
            _metadata_record(
                kind=entry["kind"],
                object_name=entry["object_name"],
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
        "cloud_mutated": any_created,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def validate_stage_receipt(
    value: Mapping[str, Any], *, provider_plan: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = deepcopy(dict(value))
    _exact_keys(receipt, _STAGE_KEYS, "fresh-quality stage receipt")
    if receipt.get("receipt_sha256") != _self_digest(receipt, "receipt_sha256"):
        raise ValueError("fresh-quality stage receipt digest changed")
    records = receipt.get("content_records")
    if not isinstance(records, list):
        raise ValueError("fresh-quality staged content records are missing")
    expected = provider_plan["content_entries"]
    if len(records) != len(expected):
        raise ValueError("fresh-quality staged content cardinality changed")
    for row, entry in zip(records, expected, strict=True):
        if not isinstance(row, Mapping):
            raise ValueError("fresh-quality staged record is not an object")
        _exact_keys(row, _OBJECT_RECORD_KEYS, "fresh-quality staged object")
        if (
            row.get("kind") != entry["kind"]
            or row.get("object_name") != entry["object_name"]
            or row.get("sha256") != entry["sha256"]
            or row.get("bytes") != entry["bytes"]
            or _PROVIDER_ID.fullmatch(str(row.get("generation", ""))) is None
            or not isinstance(row.get("created"), bool)
        ):
            raise ValueError("fresh-quality staged object binding changed")
    if (
        receipt.get("schema") != STAGE_RECEIPT_SCHEMA
        or receipt.get("status") != "all_content_hash_replayed"
        or receipt.get("provider_plan_sha256")
        != provider_plan["provider_plan_sha256"]
        or receipt.get("record_count") != len(records)
        or receipt.get("all_bytes_replayed") is not True
        or receipt.get("create_only_or_identical_reuse") is not True
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("fresh-quality stage receipt boundary changed")
    return receipt


def observe_staged_content(
    *, provider_plan: Mapping[str, Any], transport: QualityCloudTransport
) -> dict[str, Any]:
    records = []
    for entry in provider_plan["content_entries"]:
        metadata = transport.get_object_metadata(
            bucket=provider_plan["bucket"], object_name=entry["object_name"]
        )
        if metadata is None:
            raise QualityProviderError("abort cleanup cannot recover staged content")
        generation = str(metadata.get("generation", ""))
        payload = transport.get_object_bytes(
            bucket=provider_plan["bucket"],
            object_name=entry["object_name"],
            generation=generation,
        )
        if (
            payload is None
            or hashlib.sha256(payload).hexdigest() != entry["sha256"]
            or len(payload) != entry["bytes"]
        ):
            raise QualityProviderError("abort cleanup staged content changed")
        records.append(
            _metadata_record(
                kind=entry["kind"],
                object_name=entry["object_name"],
                metadata=metadata,
                payload=payload,
                created=False,
            )
        )
    core = {
        "schema": STAGE_RECEIPT_SCHEMA,
        "status": "all_content_hash_replayed",
        "provider_plan_sha256": provider_plan["provider_plan_sha256"],
        "content_records": records,
        "record_count": len(records),
        "all_bytes_replayed": True,
        "create_only_or_identical_reuse": True,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    receipt = {**core, "receipt_sha256": canonical_sha256(core)}
    return validate_stage_receipt(receipt, provider_plan=provider_plan)


def _policy_parts(value: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str, int]:
    bindings = value.get("bindings", [])
    etag = value.get("etag")
    version = value.get("version", 1)
    if (
        not isinstance(bindings, list)
        or not isinstance(etag, str)
        or not etag
        or not isinstance(version, int)
        or isinstance(version, bool)
    ):
        raise QualityProviderError("bucket IAM policy is incomplete")
    return deepcopy(bindings), etag, max(3, version)


def _binding_fingerprints(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted(canonical_sha256(row) for row in rows)


def _iam_bindings(
    provider: Mapping[str, Any], *, expires_at_utc: str
) -> list[dict[str, Any]]:
    contract = provider["iam_contract"]
    titles = contract["condition_titles"]
    bucket_resource = f"projects/_/buckets/{provider['bucket']}/objects/"
    bindings = [
        {
            "role": contract["viewer_role"],
            "members": list(contract["members"]),
            "condition": {
                "title": titles[0],
                "description": "M3.1 fresh-quality immutable content read",
                "expression": (
                    f"resource.name.startsWith('{bucket_resource}"
                    f"{contract['reader_prefix']}') && "
                    f"request.time < timestamp('{expires_at_utc}')"
                ),
            },
        },
    ]
    for index, (worker, selected) in enumerate(
        zip(provider["workers"], provider["selected_attempts"], strict=True)
    ):
        bindings.append(
            {
            "role": contract["creator_role"],
            "members": [f"serviceAccount:{worker['service_account']}"],
            "condition": {
                "title": titles[index + 1],
                "description": "M3.1 fresh-quality create-only result publish",
                "expression": (
                    f"resource.name.startsWith('{bucket_resource}"
                    f"{selected['artifact_prefix']}/') && "
                    f"request.time < timestamp('{expires_at_utc}')"
                ),
            },
            }
        )
    return bindings


def install_worker_iam(
    *,
    provider_plan: Mapping[str, Any],
    transport: QualityCloudTransport,
    now_unix_seconds: int,
) -> dict[str, Any]:
    provider = deepcopy(dict(provider_plan))
    if provider.get("schema") != PROVIDER_PLAN_SCHEMA:
        raise ValueError("fresh-quality provider plan is invalid")
    for worker in provider["workers"]:
        account = transport.get_service_account(email=worker["service_account"])
        if (
            account.get("email") != worker["service_account"]
            or not isinstance(account.get("uniqueId"), str)
            or _PROVIDER_ID.fullmatch(account["uniqueId"]) is None
            or transport.test_service_account_act_as(
                email=worker["service_account"]
            )
            is not True
        ):
            raise PermissionError("fresh-quality worker identity/actAs check failed")
    expires = _utc(now_unix_seconds + provider["iam_contract"]["ttl_seconds"])
    policy = dict(transport.get_bucket_iam_policy(bucket=provider["bucket"]))
    bindings, etag_before, version = _policy_parts(policy)
    titles = set(provider["iam_contract"]["condition_titles"])
    if any(
        isinstance(row, Mapping)
        and isinstance(row.get("condition"), Mapping)
        and row["condition"].get("title") in titles
        for row in bindings
    ):
        raise FileExistsError("fresh-quality worker IAM title already exists")
    additions = _iam_bindings(provider, expires_at_utc=expires)
    desired = {
        **{key: deepcopy(value) for key, value in policy.items() if key != "bindings"},
        "version": version,
        "etag": etag_before,
        "bindings": [*bindings, *additions],
    }
    installed = dict(
        transport.set_bucket_iam_policy(bucket=provider["bucket"], policy=desired)
    )
    installed_bindings, etag_after, _ = _policy_parts(installed)
    observed = dict(transport.get_bucket_iam_policy(bucket=provider["bucket"]))
    observed_bindings, observed_etag, _ = _policy_parts(observed)
    if (
        etag_after != observed_etag
        or any(addition not in installed_bindings for addition in additions)
        or any(addition not in observed_bindings for addition in additions)
    ):
        raise QualityProviderError("fresh-quality worker IAM readback changed")
    core = {
        "schema": IAM_RECEIPT_SCHEMA,
        "status": "exact_temporary_worker_bindings_installed",
        "provider_plan_sha256": provider["provider_plan_sha256"],
        "wave_request_sha256": provider["wave_request_sha256"],
        "bucket": provider["bucket"],
        "policy_etag_before_sha256": hashlib.sha256(
            etag_before.encode("utf-8")
        ).hexdigest(),
        "policy_etag_after_sha256": hashlib.sha256(
            etag_after.encode("utf-8")
        ).hexdigest(),
        "condition_titles": list(provider["iam_contract"]["condition_titles"]),
        "members": list(provider["iam_contract"]["members"]),
        "expires_at_utc": expires,
        "bindings_installed": len(additions),
        "readback_exact": True,
        "cloud_mutated": True,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def validate_iam_receipt(
    value: Mapping[str, Any], *, provider_plan: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = deepcopy(dict(value))
    _exact_keys(receipt, _IAM_KEYS, "fresh-quality IAM receipt")
    if receipt.get("receipt_sha256") != _self_digest(receipt, "receipt_sha256"):
        raise ValueError("fresh-quality IAM receipt digest changed")
    if (
        receipt.get("schema") != IAM_RECEIPT_SCHEMA
        or receipt.get("status") != "exact_temporary_worker_bindings_installed"
        or receipt.get("provider_plan_sha256")
        != provider_plan["provider_plan_sha256"]
        or receipt.get("wave_request_sha256")
        != provider_plan["wave_request_sha256"]
        or receipt.get("bucket") != provider_plan["bucket"]
        or receipt.get("condition_titles")
        != provider_plan["iam_contract"]["condition_titles"]
        or receipt.get("members") != provider_plan["iam_contract"]["members"]
        or receipt.get("bindings_installed")
        != len(provider_plan["workers"]) + 1
        or receipt.get("readback_exact") is not True
        or receipt.get("cloud_mutated") is not True
        or receipt.get("current_profile_changed") is not False
        or _UTC.fullmatch(str(receipt.get("expires_at_utc", ""))) is None
    ):
        raise ValueError("fresh-quality IAM receipt boundary changed")
    return receipt


def read_quota(
    *,
    provider_plan: Mapping[str, Any],
    transport: QualityCloudTransport,
    observed_at_utc: str,
) -> dict[str, Any]:
    region = transport.get_region_quota(region=provider_plan["region"])
    quotas = region.get("quotas")
    if not isinstance(quotas, list):
        raise QualityProviderError("regional quota readback is missing")
    by_metric = {
        str(row.get("metric")): row
        for row in quotas
        if isinstance(row, Mapping)
    }
    spot_matches = [
        by_metric[key] for key in ("PREEMPTIBLE_CPUS", "SPOT_CPUS") if key in by_metric
    ]
    if len(spot_matches) != 1:
        raise QualityProviderError("live regional Spot quota is missing or ambiguous")
    c4_quota = transport.get_cloud_quota(quota_id=C4_QUOTA_ID)
    global_quota = transport.get_cloud_quota(quota_id=GLOBAL_QUOTA_ID)
    c4_dimensions = c4_quota.get("dimensionsInfos")
    c4_matches = (
        [
            row
            for row in c4_dimensions
            if isinstance(row, Mapping)
            and row.get("dimensions") == {"region": REGION, "vm_family": "C4"}
        ]
        if isinstance(c4_dimensions, list)
        else []
    )
    global_dimensions = global_quota.get("dimensionsInfos")
    if (
        c4_quota.get("quotaId") != C4_QUOTA_ID
        or c4_quota.get("metric")
        != "compute.googleapis.com/cpus_per_vm_family"
        or c4_quota.get("isPrecise") is not True
        or len(c4_matches) != 1
        or not isinstance(c4_matches[0].get("details"), Mapping)
        or global_quota.get("quotaId") != GLOBAL_QUOTA_ID
        or global_quota.get("metric") != "compute.googleapis.com/cpus_all_regions"
        or global_quota.get("isPrecise") is not True
        or not isinstance(global_dimensions, list)
        or len(global_dimensions) != 1
        or not isinstance(global_dimensions[0], Mapping)
        or global_dimensions[0].get("dimensions") not in (None, {})
        or not isinstance(global_dimensions[0].get("details"), Mapping)
    ):
        raise QualityProviderError("Cloud Quotas C4/global identity changed")
    try:
        c4_limit = int(c4_matches[0]["details"]["value"])
        global_limit = int(global_dimensions[0]["details"]["value"])
    except (KeyError, TypeError, ValueError) as exc:
        raise QualityProviderError("Cloud Quotas limit changed") from exc
    instances = list(transport.list_instances())
    machine_cpus: dict[str, int] = {}
    global_usage = 0
    c4_usage = 0
    spot_inventory_usage = 0
    active_statuses = {
        "PROVISIONING",
        "STAGING",
        "RUNNING",
        "STOPPING",
        "SUSPENDING",
        "SUSPENDED",
    }
    for instance in instances:
        if not isinstance(instance, Mapping) or instance.get("status") not in active_statuses:
            continue
        machine_link = instance.get("machineType")
        scope = str(instance.get("_scope", ""))
        if not isinstance(machine_link, str):
            raise QualityProviderError("instance usage machine type changed")
        if machine_link not in machine_cpus:
            machine = transport.get_machine_type_url(self_link=machine_link)
            guest_cpus = machine.get("guestCpus")
            if (
                not isinstance(guest_cpus, int)
                or isinstance(guest_cpus, bool)
                or guest_cpus <= 0
                or machine.get("selfLink") != machine_link
            ):
                raise QualityProviderError("machine-type vCPU readback changed")
            machine_cpus[machine_link] = guest_cpus
        cpus = machine_cpus[machine_link]
        global_usage += cpus
        in_target_region = scope.startswith(f"zones/{REGION}-")
        if in_target_region and machine_link.rsplit("/", 1)[-1].startswith("c4-"):
            c4_usage += cpus
        scheduling = instance.get("scheduling")
        if (
            in_target_region
            and isinstance(scheduling, Mapping)
            and scheduling.get("provisioningModel") == "SPOT"
        ):
            spot_inventory_usage += cpus
    spot = spot_matches[0]
    spot_limit = int(float(spot.get("limit", -1)))
    spot_usage = int(float(spot.get("usage", -1)))
    if spot_usage < spot_inventory_usage:
        raise QualityProviderError("regional Spot quota usage is below inventory")
    required = len(provider_plan["workers"]) * bridge.VCPUS_PER_VM
    metrics = []
    for metric, limit, usage in (
        ("c4_family_vcpus", c4_limit, c4_usage),
        ("spot_vcpus", spot_limit, spot_usage),
        ("global_vcpus", global_limit, global_usage),
    ):
        metrics.append(
            {
                "metric": metric,
                "limit_vcpus": limit,
                "usage_vcpus": usage,
                "headroom_vcpus": limit - usage,
                "sufficient": limit > 0 and limit >= usage >= 0 and limit - usage >= required,
            }
        )
    sufficient = all(row["sufficient"] for row in metrics)
    receipt = {
        "source": "compute_regions_get",
        "metrics": metrics,
        "required_vcpus": required,
        "sufficient": sufficient,
        "observed_at_utc": observed_at_utc,
    }
    _exact_keys(receipt, _QUOTA_KEYS, "fresh-quality quota receipt")
    if not sufficient:
        raise PermissionError("live regional C4 quota headroom is insufficient")
    return receipt


def _validate_image(
    provider: Mapping[str, Any], transport: QualityCloudTransport
) -> None:
    observed = transport.get_image(self_link=provider["image"]["self_link"])
    features = observed.get("guestOsFeatures")
    observed_features = sorted(
        row.get("type")
        for row in features
        if isinstance(row, Mapping) and isinstance(row.get("type"), str)
    ) if isinstance(features, list) else []
    if (
        observed.get("id") != provider["image"]["id"]
        or observed.get("selfLink") != provider["image"]["self_link"]
        or observed_features != provider["image"]["guest_os_features"]
        or observed.get("status") != "READY"
    ):
        raise PermissionError("active GCE image identity changed")


def _validate_network(
    provider: Mapping[str, Any], transport: QualityCloudTransport
) -> None:
    router = transport.get_router(
        region=provider["region"], router_name=NAT_ROUTER_NAME
    )
    nats = router.get("nats")
    matching = (
        [
            row
            for row in nats
            if isinstance(row, Mapping) and row.get("name") == NAT_NAME
        ]
        if isinstance(nats, list)
        else []
    )
    if (
        router.get("name") != NAT_ROUTER_NAME
        or not str(router.get("network", "")).endswith(
            f"/global/networks/{NETWORK}"
        )
        or len(matching) != 1
        or matching[0].get("natIpAllocateOption") != "AUTO_ONLY"
        or matching[0].get("sourceSubnetworkIpRangesToNat")
        != "ALL_SUBNETWORKS_ALL_IP_RANGES"
    ):
        raise PermissionError("launch-time Cloud NAT path changed")


def _content_bindings(
    provider: Mapping[str, Any], stage_receipt: Mapping[str, Any]
) -> list[dict[str, Any]]:
    stage = validate_stage_receipt(stage_receipt, provider_plan=provider)
    by_kind = {row["kind"]: row for row in stage["content_records"]}
    launch_paths = provider["content_entries"]
    result: list[dict[str, Any]] = []
    for entry in launch_paths:
        record = by_kind[entry["kind"]]
        relative = (
            "launch.json"
            if entry["kind"] == "launch_manifest"
            else Path(entry["local_path"]).name
        )
        # Preserve the launch-manifest relative paths.  The local startup
        # resolves every record from the staging root.
        if entry["kind"] != "launch_manifest":
            launch_entry = next(
                row
                for row in provider["content_entries"]
                if row["kind"] == entry["kind"]
            )
            local = Path(launch_entry["local_path"])
            staging = Path(
                next(
                    row["local_path"]
                    for row in provider["content_entries"]
                    if row["kind"] == "launch_manifest"
                )
            ).parent
            try:
                relative = local.relative_to(staging).as_posix()
            except ValueError as exc:
                raise ValueError("provider content escaped staging") from exc
        result.append(
            {
                "relative": relative,
                "object": record["object_name"],
                "generation": record["generation"],
                "sha256": record["sha256"],
                "bytes": record["bytes"],
            }
        )
    if len({row["relative"] for row in result}) != len(result):
        raise ValueError("provider content relative path is duplicated")
    return result


def _instance_spec(
    *,
    provider: Mapping[str, Any],
    worker: Mapping[str, Any],
    selected: Mapping[str, Any],
    stage_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    bindings = _content_bindings(provider, stage_receipt)
    bindings_raw = canonical_bytes(bindings)
    local_startup_relative = next(
        row["relative"] for row in bindings if row["relative"].endswith(".sh")
    )
    local_startup = next(
        entry for entry in provider["content_entries"] if entry["kind"] == "startup"
    )
    metadata = {
        "fq-bucket": provider["bucket"],
        "fq-job-id": selected["job_id"],
        "fq-artifact-prefix": selected["artifact_prefix"],
        "fq-content-bindings-b64": base64.b64encode(bindings_raw).decode("ascii"),
        "fq-launch-relative": "launch.json",
        "fq-local-startup-relative": local_startup_relative,
        "fq-local-startup-sha256": local_startup["sha256"],
        "fq-watchdog-seconds": str(WATCHDOG_SECONDS),
        "startup-script": worker_startup_script(),
    }
    return {
        "name": selected["instance_id"],
        "machineType": (
            f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/zones/"
            f"{ZONE}/machineTypes/{MACHINE_TYPE}"
        ),
        "labels": {
            "ofc-owner": provider["execution_identity_sha256"][:32],
            "ofc-plan": provider["provider_plan_sha256"][:32],
            "ofc-wave": f"w{selected['wave_index']}",
        },
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
                    "labels": {
                        "ofc-owner": provider["execution_identity_sha256"][:32],
                        "ofc-plan": provider["provider_plan_sha256"][:32],
                        "ofc-wave": f"w{selected['wave_index']}",
                    },
                    "diskType": (
                        f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
                        f"zones/{ZONE}/diskTypes/{BOOT_DISK_TYPE}"
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
            {
                "email": worker["service_account"],
                "scopes": [OAUTH_SCOPE],
            }
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
    rows = instance.get("metadata", {}).get("items", [])
    if not isinstance(rows, list):
        raise QualityProviderError("GCE instance metadata is missing")
    result: dict[str, str] = {}
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("key"), str)
            or not isinstance(row.get("value"), str)
            or row["key"] in result
        ):
            raise QualityProviderError("GCE instance metadata changed")
        result[row["key"]] = row["value"]
    return result


def _validate_owned_instance(
    instance: Mapping[str, Any], *, expected_spec: Mapping[str, Any]
) -> tuple[str, str, str]:
    name = expected_spec["name"]
    disks = instance.get("disks")
    interfaces = instance.get("networkInterfaces")
    accounts = instance.get("serviceAccounts")
    scheduling = instance.get("scheduling")
    if (
        instance.get("name") != name
        or instance.get("machineType") != expected_spec["machineType"]
        or instance.get("labels") != expected_spec["labels"]
        or instance.get("deletionProtection", False) is not False
        or instance.get("canIpForward", False) is not False
        or not isinstance(scheduling, Mapping)
        or scheduling.get("provisioningModel") != "SPOT"
        or scheduling.get("instanceTerminationAction") != "DELETE"
        or scheduling.get("automaticRestart") is not False
        or scheduling.get("onHostMaintenance") != "TERMINATE"
        or not isinstance(scheduling.get("maxRunDuration"), Mapping)
        or scheduling["maxRunDuration"].get("seconds") != str(MAX_RUN_SECONDS)
        or scheduling["maxRunDuration"].get("nanos", 0) != 0
        or _metadata_map(instance)
        != {
            row["key"]: row["value"]
            for row in expected_spec["metadata"]["items"]
        }
        or not isinstance(disks, list)
        or len(disks) != 1
        or disks[0].get("boot") is not True
        or disks[0].get("autoDelete") is not True
        or disks[0].get("interface") != BOOT_DISK_INTERFACE
        or disks[0].get("deviceName") != name
        or not isinstance(interfaces, list)
        or len(interfaces) != 1
        or interfaces[0].get("nicType") != NIC_TYPE
        or interfaces[0].get("network")
        != expected_spec["networkInterfaces"][0]["network"]
        or interfaces[0].get("subnetwork")
        != expected_spec["networkInterfaces"][0]["subnetwork"]
        or interfaces[0].get("accessConfigs", []) != []
        or not isinstance(accounts, list)
        or len(accounts) != 1
        or accounts[0].get("email")
        != expected_spec["serviceAccounts"][0]["email"]
        or accounts[0].get("scopes")
        != expected_spec["serviceAccounts"][0]["scopes"]
    ):
        raise PermissionError("GCE instance differs from exact owned specification")
    provider_id = str(instance.get("id", ""))
    source = disks[0].get("source")
    if (
        _PROVIDER_ID.fullmatch(provider_id) is None
        or not isinstance(source, str)
        or not source
    ):
        raise QualityProviderError("GCE provider instance/disk identity is missing")
    status = instance.get("status")
    if status not in {
        "PROVISIONING",
        "STAGING",
        "RUNNING",
        "STOPPING",
        "SUSPENDING",
        "SUSPENDED",
        "REPAIRING",
        "TERMINATED",
    }:
        raise QualityProviderError("GCE instance status changed")
    return provider_id, source.rsplit("/", 1)[-1], status


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
        raise PermissionError("GCE boot disk differs from exact owned specification")
    return provider_id


def _wait_operation(
    transport: QualityCloudTransport,
    *,
    initial: Mapping[str, Any],
    operation_type: str,
    target_name: str,
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    name = initial.get("name")
    if not isinstance(name, str) or _OPERATION.fullmatch(name) is None:
        raise QualityProviderError("GCE zone operation name changed")
    current = dict(initial)
    for attempt in range(OPERATION_POLL_ATTEMPTS):
        if current.get("status") == "DONE":
            if current.get("error"):
                raise QualityProviderError(
                    f"GCE {operation_type} operation completed with an error"
                )
            if current.get("operationType") not in (None, operation_type):
                raise QualityProviderError("GCE operation type changed")
            target = current.get("targetLink")
            if target is not None and (
                not isinstance(target, str)
                or target.rsplit("/", 1)[-1] != target_name
            ):
                raise QualityProviderError("GCE operation target changed")
            return current
        if attempt + 1 < OPERATION_POLL_ATTEMPTS:
            sleep(OPERATION_POLL_INTERVAL_SECONDS)
            current = dict(transport.get_zone_operation(operation_name=name))
    raise TimeoutError(f"GCE {operation_type} operation did not complete")


def _claim(
    *,
    provider: Mapping[str, Any],
    transport: QualityCloudTransport,
    raw_nonce: str,
    now_unix_seconds: int,
) -> dict[str, Any]:
    try:
        parsed = uuid.UUID(raw_nonce)
    except (ValueError, AttributeError) as exc:
        raise ValueError("operation nonce must be a canonical UUIDv4") from exc
    if parsed.version != 4 or str(parsed) != raw_nonce:
        raise ValueError("operation nonce must be a canonical UUIDv4")
    object_name = provider["claim_object"]
    if (
        transport.get_object_metadata(
            bucket=provider["bucket"], object_name=object_name
        )
        is not None
    ):
        raise FileExistsError("fresh-quality wave claim already exists")
    payload = canonical_bytes(
        {
            "schema": "hu_m31_t3_step6d_fresh_quality_gcp_claim_v1",
            "provider_plan_sha256": provider["provider_plan_sha256"],
            "wave_request_sha256": provider["wave_request_sha256"],
            "nonce_sha256": hashlib.sha256(raw_nonce.encode("ascii")).hexdigest(),
            "claimed_at_utc": _utc(now_unix_seconds),
            "create_only": True,
        }
    )
    reconciled = False
    try:
        metadata = transport.put_object_new(
            bucket=provider["bucket"],
            object_name=object_name,
            payload=payload,
            content_type="application/json",
        )
    except c4_gcp.ResponseLostError:
        metadata = transport.get_object_metadata(
            bucket=provider["bucket"], object_name=object_name
        )
        if metadata is None:
            raise QualityProviderError(
                "wave claim response was lost and claim is absent"
            )
        reconciled = True
    generation = str(metadata.get("generation", ""))
    observed = transport.get_object_bytes(
        bucket=provider["bucket"],
        object_name=object_name,
        generation=generation,
    )
    if observed != payload:
        raise QualityProviderError("wave claim readback differs")
    return {
        "object_name": object_name,
        "generation": generation,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "nonce_sha256": hashlib.sha256(raw_nonce.encode("ascii")).hexdigest(),
        "response_reconciled": reconciled,
    }


def execute_wave(
    *,
    provider_plan: Mapping[str, Any],
    bridge_plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    wave_request: Mapping[str, Any],
    raw_nonce: str,
    transport: QualityCloudTransport,
    now_unix_seconds: int,
    observed_at_utc: str,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    provider, plan, checked_ledger, checked_resume, request = _validate_provider_context(
        provider_plan=provider_plan,
        bridge_plan=bridge_plan,
        ledger=ledger,
        resume=resume,
        wave_request=wave_request,
    )
    if _UTC.fullmatch(observed_at_utc) is None:
        raise ValueError("fresh-quality execution observation time changed")
    stage = stage_content(
        provider_plan=provider,
        bridge_plan=plan,
        ledger=checked_ledger,
        resume=checked_resume,
        wave_request=request,
        transport=transport,
    )
    _validate_image(provider, transport)
    _validate_network(provider, transport)
    quota = read_quota(
        provider_plan=provider,
        transport=transport,
        observed_at_utc=observed_at_utc,
    )
    selected_by_job = {row["job_id"]: row for row in request["selected_attempts"]}
    workers_by_job = {row["job_id"]: row for row in provider["workers"]}
    specs: dict[str, dict[str, Any]] = {}
    for job_id, selected in selected_by_job.items():
        spec = _instance_spec(
            provider=provider,
            worker=workers_by_job[job_id],
            selected=selected,
            stage_receipt=stage,
        )
        if transport.get_instance(instance_name=selected["instance_id"]) is not None:
            raise FileExistsError("selected fresh-quality VM already exists")
        # Orphan boot disk names are the instance names for this exact spec.
        if transport.get_disk_optional(disk_name=selected["instance_id"]) is not None:
            raise FileExistsError("selected fresh-quality boot disk already exists")
        specs[job_id] = spec
    claim = _claim(
        provider=provider,
        transport=transport,
        raw_nonce=raw_nonce,
        now_unix_seconds=now_unix_seconds,
    )
    iam = install_worker_iam(
        provider_plan=provider,
        transport=transport,
        now_unix_seconds=now_unix_seconds,
    )
    rows: list[dict[str, Any]] = []
    for job_id, selected in selected_by_job.items():
        try:
            spec = specs[job_id]
            worker = workers_by_job[job_id]
            request_id = _uuid_for(
                provider["provider_plan_sha256"],
                job_id,
                selected["attempt_id"],
                "create",
            )
            created = True
            try:
                initial = transport.create_instance(
                    instance_spec=spec, request_id=request_id
                )
            except c4_gcp.ResponseLostError:
                instance = transport.get_instance(
                    instance_name=selected["instance_id"]
                )
                if instance is None:
                    raise QualityProviderError(
                        "GCE create response was lost and exact VM is absent"
                    )
            else:
                _wait_operation(
                    transport,
                    initial=initial,
                    operation_type="insert",
                    target_name=selected["instance_id"],
                    sleep=sleep,
                )
                instance = transport.get_instance(
                    instance_name=selected["instance_id"]
                )
                if instance is None:
                    raise QualityProviderError("created GCE instance is absent")
            provider_id, disk_name, status = _validate_owned_instance(
                instance, expected_spec=spec
            )
            disk = transport.get_disk_optional(disk_name=disk_name)
            if disk is None:
                raise QualityProviderError(
                    "created GCE boot disk identity is absent"
                )
            disk_id = _validate_owned_disk(disk, expected_spec=spec)
            rows.append(
                {
                    "job_id": job_id,
                    "attempt_id": selected["attempt_id"],
                    "instance_id": selected["instance_id"],
                    "service_account": worker["service_account"],
                    "request_id": request_id,
                    "spec_sha256": canonical_sha256(spec),
                    "provider_instance_id": provider_id,
                    "provider_boot_disk_id": disk_id,
                    "observed_status": status,
                    "created": created,
                }
            )
        except (
            QualityProviderError,
            c4_gcp.HuRlC4GcpLifecycleError,
            c4_gcp.ResponseLostError,
            PermissionError,
            TimeoutError,
            ValueError,
        ):
            # The one-shot claim is consumed.  Return a durable partial receipt
            # so cleanup can inspect and delete every exact selected name; the
            # missing jobs advance only to a01 after cleanup/receive.
            break
    complete = len(rows) == len(request["selected_attempts"])
    core = {
        "schema": LAUNCH_RECEIPT_SCHEMA,
        "status": (
            "exact_selected_quality_wave_created"
            if complete
            else "partial_selected_quality_wave_created"
        ),
        "provider_plan_sha256": provider["provider_plan_sha256"],
        "bridge_plan_sha256": plan["plan_sha256"],
        "ledger_sha256": checked_ledger["ledger_sha256"],
        "resume_sha256": checked_resume["resume_sha256"],
        "wave_request_sha256": request["request_sha256"],
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_index": checked_resume["resume_wave_index"],
        "claim": claim,
        "stage_receipt": stage,
        "iam_receipt": iam,
        "quota_receipt": quota,
        "rows": rows,
        "selected_job_count": len(request["selected_attempts"]),
        "created_instance_count": len(rows),
        "create_complete": complete,
        "one_shot_consumed": True,
        "unlisted_vm_created": 0,
        "cloud_mutated": True,
        "current_profile_changed": False,
    }
    receipt = {**core, "receipt_sha256": canonical_sha256(core)}
    return validate_launch_receipt(
        receipt,
        provider_plan=provider,
        bridge_plan=plan,
        ledger=checked_ledger,
        resume=checked_resume,
        wave_request=request,
    )


def validate_launch_receipt(
    value: Mapping[str, Any],
    *,
    provider_plan: Mapping[str, Any],
    bridge_plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    wave_request: Mapping[str, Any],
) -> dict[str, Any]:
    provider, plan, checked_ledger, checked_resume, request = _validate_provider_context(
        provider_plan=provider_plan,
        bridge_plan=bridge_plan,
        ledger=ledger,
        resume=resume,
        wave_request=wave_request,
    )
    receipt = deepcopy(dict(value))
    _exact_keys(receipt, _LAUNCH_KEYS, "fresh-quality launch receipt")
    if receipt.get("receipt_sha256") != _self_digest(receipt, "receipt_sha256"):
        raise ValueError("fresh-quality launch receipt digest changed")
    claim = receipt.get("claim")
    quota = receipt.get("quota_receipt")
    rows = receipt.get("rows")
    if (
        not isinstance(claim, Mapping)
        or not isinstance(quota, Mapping)
        or not isinstance(rows, list)
    ):
        raise ValueError("fresh-quality launch receipt evidence is missing")
    _exact_keys(claim, _CLAIM_KEYS, "fresh-quality launch claim")
    _exact_keys(quota, _QUOTA_KEYS, "fresh-quality quota receipt")
    _sha(claim.get("sha256"), "fresh-quality claim")
    if (
        claim.get("object_name") != provider["claim_object"]
        or _PROVIDER_ID.fullmatch(str(claim.get("generation", ""))) is None
        or not isinstance(claim.get("bytes"), int)
        or claim["bytes"] <= 0
        or _sha(claim.get("nonce_sha256"), "fresh-quality nonce") is None
        or not isinstance(claim.get("response_reconciled"), bool)
    ):
        raise ValueError("fresh-quality claim binding changed")
    validate_stage_receipt(receipt.get("stage_receipt", {}), provider_plan=provider)
    validate_iam_receipt(receipt.get("iam_receipt", {}), provider_plan=provider)
    selected = request["selected_attempts"]
    selected_by_job = {row["job_id"]: row for row in selected}
    workers_by_job = {row["job_id"]: row for row in provider["workers"]}
    stage = receipt["stage_receipt"]
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("fresh-quality launch row is not an object")
        _exact_keys(row, _LAUNCH_ROW_KEYS, "fresh-quality launch row")
        selected_row = selected_by_job.get(row.get("job_id"))
        worker = workers_by_job.get(row.get("job_id"))
        if (
            selected_row is None
            or worker is None
            or row["job_id"] in seen
            or row.get("attempt_id") != selected_row["attempt_id"]
            or row.get("instance_id") != selected_row["instance_id"]
            or row.get("service_account") != worker["service_account"]
            or row.get("request_id")
            != _uuid_for(
                provider["provider_plan_sha256"],
                row["job_id"],
                row["attempt_id"],
                "create",
            )
            or row.get("spec_sha256")
            != canonical_sha256(
                _instance_spec(
                    provider=provider,
                    worker=worker,
                    selected=selected_row,
                    stage_receipt=stage,
                )
            )
            or _PROVIDER_ID.fullmatch(str(row.get("provider_instance_id", "")))
            is None
            or _PROVIDER_ID.fullmatch(str(row.get("provider_boot_disk_id", "")))
            is None
            or row.get("observed_status")
            not in {"PROVISIONING", "STAGING", "RUNNING", "STOPPING", "TERMINATED"}
            or row.get("created") is not True
        ):
            raise ValueError("fresh-quality launch row escaped selected inventory")
        seen.add(row["job_id"])
    complete = len(rows) == len(selected)
    if (
        receipt.get("schema") != LAUNCH_RECEIPT_SCHEMA
        or receipt.get("status")
        != (
            "exact_selected_quality_wave_created"
            if complete
            else "partial_selected_quality_wave_created"
        )
        or receipt.get("provider_plan_sha256")
        != provider["provider_plan_sha256"]
        or receipt.get("bridge_plan_sha256") != plan["plan_sha256"]
        or receipt.get("ledger_sha256") != checked_ledger["ledger_sha256"]
        or receipt.get("resume_sha256") != checked_resume["resume_sha256"]
        or receipt.get("wave_request_sha256") != request["request_sha256"]
        or receipt.get("run_name") != plan["run_name"]
        or receipt.get("execution_identity_sha256")
        != plan["execution_identity_sha256"]
        or receipt.get("wave_index") != checked_resume["resume_wave_index"]
        or receipt.get("selected_job_count") != len(selected)
        or receipt.get("created_instance_count") != len(rows)
        or receipt.get("create_complete") is not complete
        or receipt.get("one_shot_consumed") is not True
        or receipt.get("unlisted_vm_created") != 0
        or receipt.get("cloud_mutated") is not True
        or receipt.get("current_profile_changed") is not False
        or quota.get("required_vcpus") != len(selected) * bridge.VCPUS_PER_VM
        or quota.get("sufficient") is not True
    ):
        raise ValueError("fresh-quality launch receipt boundary changed")
    return receipt


def poll_wave(
    *,
    provider_plan: Mapping[str, Any],
    bridge_plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    wave_request: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
    transport: QualityCloudTransport,
) -> dict[str, Any]:
    provider, plan, checked_ledger, checked_resume, request = _validate_provider_context(
        provider_plan=provider_plan,
        bridge_plan=bridge_plan,
        ledger=ledger,
        resume=resume,
        wave_request=wave_request,
    )
    launch = validate_launch_receipt(
        launch_receipt,
        provider_plan=provider,
        bridge_plan=plan,
        ledger=checked_ledger,
        resume=checked_resume,
        wave_request=request,
    )
    selected_by_job = {row["job_id"]: row for row in request["selected_attempts"]}
    workers_by_job = {row["job_id"]: row for row in provider["workers"]}
    launch_by_job = {row["job_id"]: row for row in launch["rows"]}
    rows = []
    for job_id, selected in selected_by_job.items():
        launched = launch_by_job.get(job_id)
        instance_status = "ABSENT"
        if launched is not None:
            instance = transport.get_instance(instance_name=selected["instance_id"])
            if instance is not None:
                spec = _instance_spec(
                    provider=provider,
                    worker=workers_by_job[job_id],
                    selected=selected,
                    stage_receipt=launch["stage_receipt"],
                )
                provider_id, _disk, instance_status = _validate_owned_instance(
                    instance, expected_spec=spec
                )
                if provider_id != launched["provider_instance_id"]:
                    raise PermissionError("polled GCE instance provider ID changed")
        attempt = next(
            row
            for row in plan["jobs"]
            if row["job_id"] == job_id
        )["attempts"][
            0 if selected["attempt_id"] == "a00" else 1
        ]
        done = transport.get_object_metadata(
            bucket=provider["bucket"], object_name=attempt["done_object"]
        )
        generation = None if done is None else str(done.get("generation", ""))
        if generation is not None and _PROVIDER_ID.fullmatch(generation) is None:
            raise QualityProviderError("fresh-quality DONE generation changed")
        rows.append(
            {
                "job_id": job_id,
                "attempt_id": selected["attempt_id"],
                "instance_id": selected["instance_id"],
                "instance_status": instance_status,
                "done_generation": generation,
            }
        )
    done_count = sum(row["done_generation"] is not None for row in rows)
    core = {
        "schema": POLL_RECEIPT_SCHEMA,
        "status": (
            "all_selected_done_ready"
            if done_count == len(rows)
            else "selected_wave_in_progress_or_failed"
        ),
        "provider_plan_sha256": provider["provider_plan_sha256"],
        "launch_receipt_sha256": launch["receipt_sha256"],
        "rows": rows,
        "done_count": done_count,
        "selected_job_count": len(rows),
        "read_only": True,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def validate_poll_receipt(
    value: Mapping[str, Any],
    *,
    provider_plan: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = deepcopy(dict(value))
    _exact_keys(receipt, _POLL_KEYS, "fresh-quality poll receipt")
    if receipt.get("receipt_sha256") != _self_digest(receipt, "receipt_sha256"):
        raise ValueError("fresh-quality poll receipt digest changed")
    rows = receipt.get("rows")
    if not isinstance(rows, list):
        raise ValueError("fresh-quality poll rows are missing")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("fresh-quality poll row is not an object")
        _exact_keys(row, _POLL_ROW_KEYS, "fresh-quality poll row")
    done_count = sum(row["done_generation"] is not None for row in rows)
    if (
        receipt.get("schema") != POLL_RECEIPT_SCHEMA
        or receipt.get("status")
        != (
            "all_selected_done_ready"
            if done_count == len(rows)
            else "selected_wave_in_progress_or_failed"
        )
        or receipt.get("provider_plan_sha256")
        != provider_plan["provider_plan_sha256"]
        or receipt.get("launch_receipt_sha256")
        != launch_receipt["receipt_sha256"]
        or receipt.get("done_count") != done_count
        or receipt.get("selected_job_count") != len(rows)
        or receipt.get("read_only") is not True
        or receipt.get("cloud_mutated") is not False
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("fresh-quality poll receipt boundary changed")
    return receipt


def remove_worker_iam(
    *,
    provider_plan: Mapping[str, Any],
    iam_receipt: Mapping[str, Any],
    transport: QualityCloudTransport,
) -> dict[str, Any]:
    provider = deepcopy(dict(provider_plan))
    installed = validate_iam_receipt(iam_receipt, provider_plan=provider)
    policy = dict(transport.get_bucket_iam_policy(bucket=provider["bucket"]))
    bindings, etag, version = _policy_parts(policy)
    expected = _iam_bindings(
        provider, expires_at_utc=installed["expires_at_utc"]
    )
    if any(binding not in bindings for binding in expected):
        raise PermissionError("fresh-quality worker IAM binding is missing or changed")
    unrelated_before = [row for row in bindings if row not in expected]
    desired = {
        **{key: deepcopy(value) for key, value in policy.items() if key != "bindings"},
        "version": version,
        "etag": etag,
        "bindings": unrelated_before,
    }
    updated = dict(
        transport.set_bucket_iam_policy(bucket=provider["bucket"], policy=desired)
    )
    updated_bindings, updated_etag, _ = _policy_parts(updated)
    observed = dict(transport.get_bucket_iam_policy(bucket=provider["bucket"]))
    observed_bindings, observed_etag, _ = _policy_parts(observed)
    if (
        updated_etag != observed_etag
        or _binding_fingerprints(updated_bindings)
        != _binding_fingerprints(unrelated_before)
        or _binding_fingerprints(observed_bindings)
        != _binding_fingerprints(unrelated_before)
        or any(binding in observed_bindings for binding in expected)
    ):
        raise QualityProviderError("fresh-quality worker IAM cleanup readback changed")
    core = {
        "schema": IAM_CLEANUP_SCHEMA,
        "status": "exact_temporary_worker_bindings_removed",
        "provider_plan_sha256": provider["provider_plan_sha256"],
        "iam_receipt_sha256": installed["receipt_sha256"],
        "bucket": provider["bucket"],
        "condition_titles": installed["condition_titles"],
        "bindings_removed": len(expected),
        "readback_absent": True,
        "unrelated_bindings_preserved": True,
        "cloud_mutated": True,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def remove_worker_iam_without_receipt(
    *,
    provider_plan: Mapping[str, Any],
    transport: QualityCloudTransport,
) -> dict[str, Any]:
    """Remove an exact title-bound policy after a lost prelaunch receipt."""

    provider = deepcopy(dict(provider_plan))
    policy = dict(transport.get_bucket_iam_policy(bucket=provider["bucket"]))
    bindings, etag, version = _policy_parts(policy)
    titles = set(provider["iam_contract"]["condition_titles"])
    matching = [
        row
        for row in bindings
        if isinstance(row, Mapping)
        and isinstance(row.get("condition"), Mapping)
        and row["condition"].get("title") in titles
    ]
    if matching:
        if {row["condition"]["title"] for row in matching} != titles:
            raise PermissionError("partial fresh-quality IAM title set is unsafe")
        expiries = []
        for row in matching:
            expression = str(row["condition"].get("expression", ""))
            found = re.findall(
                r"request\.time < timestamp\('([0-9TZ:-]+)'\)", expression
            )
            if len(found) != 1:
                raise PermissionError("fresh-quality IAM expiry expression changed")
            expiries.append(found[0])
        if len(set(expiries)) != 1:
            raise PermissionError("fresh-quality IAM expiries diverged")
        expected = _iam_bindings(provider, expires_at_utc=expiries[0])
        if _binding_fingerprints(matching) != _binding_fingerprints(expected):
            raise PermissionError("fresh-quality IAM binding content changed")
        unrelated = [row for row in bindings if row not in matching]
        desired = {
            **{key: deepcopy(value) for key, value in policy.items() if key != "bindings"},
            "version": version,
            "etag": etag,
            "bindings": unrelated,
        }
        transport.set_bucket_iam_policy(bucket=provider["bucket"], policy=desired)
        observed = dict(
            transport.get_bucket_iam_policy(bucket=provider["bucket"])
        )
        observed_bindings, _observed_etag, _ = _policy_parts(observed)
        if _binding_fingerprints(observed_bindings) != _binding_fingerprints(
            unrelated
        ):
            raise QualityProviderError("abort IAM cleanup readback changed")
        removed = len(matching)
        mutated = True
    else:
        unrelated = bindings
        removed = 0
        mutated = False
    core = {
        "schema": IAM_CLEANUP_SCHEMA,
        "status": "exact_temporary_worker_bindings_removed",
        "provider_plan_sha256": provider["provider_plan_sha256"],
        "iam_receipt_sha256": "0" * 64,
        "bucket": provider["bucket"],
        "condition_titles": list(provider["iam_contract"]["condition_titles"]),
        "bindings_removed": removed,
        "readback_absent": True,
        "unrelated_bindings_preserved": True,
        "cloud_mutated": mutated,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def validate_iam_cleanup_receipt(
    value: Mapping[str, Any],
    *,
    provider_plan: Mapping[str, Any],
    iam_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    installed = validate_iam_receipt(
        iam_receipt, provider_plan=provider_plan
    )
    receipt = deepcopy(dict(value))
    _exact_keys(receipt, _IAM_CLEANUP_KEYS, "fresh-quality IAM cleanup receipt")
    if receipt.get("receipt_sha256") != _self_digest(receipt, "receipt_sha256"):
        raise ValueError("fresh-quality IAM cleanup receipt digest changed")
    if (
        receipt.get("schema") != IAM_CLEANUP_SCHEMA
        or receipt.get("status") != "exact_temporary_worker_bindings_removed"
        or receipt.get("provider_plan_sha256")
        != provider_plan["provider_plan_sha256"]
        or receipt.get("iam_receipt_sha256") != installed["receipt_sha256"]
        or receipt.get("bucket") != provider_plan["bucket"]
        or receipt.get("condition_titles") != installed["condition_titles"]
        or receipt.get("bindings_removed") != len(provider_plan["workers"]) + 1
        or receipt.get("readback_absent") is not True
        or receipt.get("unrelated_bindings_preserved") is not True
        or receipt.get("cloud_mutated") is not True
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("fresh-quality IAM cleanup receipt boundary changed")
    return receipt


def validate_abort_iam_cleanup_receipt(
    value: Mapping[str, Any],
    *,
    provider_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate title-bound cleanup when the install receipt was not persisted."""

    provider = deepcopy(dict(provider_plan))
    _exact_keys(provider, _PROVIDER_PLAN_KEYS, "fresh-quality provider plan")
    if (
        provider.get("schema") != PROVIDER_PLAN_SCHEMA
        or provider.get("provider_plan_sha256")
        != _self_digest(provider, "provider_plan_sha256")
    ):
        raise ValueError("fresh-quality provider plan digest changed")
    receipt = deepcopy(dict(value))
    _exact_keys(receipt, _IAM_CLEANUP_KEYS, "fresh-quality abort IAM cleanup receipt")
    if receipt.get("receipt_sha256") != _self_digest(receipt, "receipt_sha256"):
        raise ValueError("fresh-quality abort IAM cleanup receipt digest changed")
    removed = receipt.get("bindings_removed")
    if (
        receipt.get("schema") != IAM_CLEANUP_SCHEMA
        or receipt.get("status") != "exact_temporary_worker_bindings_removed"
        or receipt.get("provider_plan_sha256")
        != provider["provider_plan_sha256"]
        or receipt.get("iam_receipt_sha256") != "0" * 64
        or receipt.get("bucket") != provider["bucket"]
        or receipt.get("condition_titles")
        != provider["iam_contract"]["condition_titles"]
        or removed not in {0, len(provider["workers"]) + 1}
        or receipt.get("readback_absent") is not True
        or receipt.get("unrelated_bindings_preserved") is not True
        or receipt.get("cloud_mutated") is not (removed != 0)
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("fresh-quality abort IAM cleanup receipt boundary changed")
    return receipt


def _read_object_record(
    transport: QualityCloudTransport,
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
    if (
        metadata.get("name") != object_name
        or _PROVIDER_ID.fullmatch(generation) is None
    ):
        raise QualityProviderError("fresh-quality result object metadata changed")
    payload = transport.get_object_bytes(
        bucket=bucket, object_name=object_name, generation=generation
    )
    if payload is None:
        raise QualityProviderError("fresh-quality result object bytes are absent")
    return (
        {
            "name": object_name,
            "generation": generation,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        },
        payload,
    )


def _collect_attempt_objects(
    *,
    provider: Mapping[str, Any],
    plan: Mapping[str, Any],
    selected: Mapping[str, Any],
    transport: QualityCloudTransport,
) -> dict[str, Any]:
    job = next(row for row in plan["jobs"] if row["job_id"] == selected["job_id"])
    attempt = next(
        row for row in job["attempts"] if row["attempt_id"] == selected["attempt_id"]
    )
    done_read = _read_object_record(
        transport,
        bucket=provider["bucket"],
        object_name=attempt["done_object"],
    )
    if done_read is None:
        return {
            "job_id": selected["job_id"],
            "attempt_id": selected["attempt_id"],
            "instance_id": selected["instance_id"],
            "status": "failed",
            "result_object": None,
            "done_object": None,
            "task_objects": [],
        }
    done_record, done_payload = done_read
    try:
        done = json.loads(done_payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualityProviderError("remote fresh-quality DONE is not JSON") from exc
    if (
        not isinstance(done, dict)
        or done_payload != canonical_bytes(done)
        or done.get("schema") != quality_transport.DONE_SCHEMA
        or done.get("status") != "complete_validated_quality_job"
        or done.get("job_id") != selected["job_id"]
        or done.get("phase") != job["phase"]
        or done.get("job_manifest_sha256") != job["job_manifest_sha256"]
        or done.get("done_published_last") is not True
        or done.get("current_profile_changed") is not False
        or done.get("training_eligible") is not False
        or done.get("promotion_evidence") is not False
    ):
        raise QualityProviderError("remote fresh-quality DONE contract changed")
    result_read = _read_object_record(
        transport,
        bucket=provider["bucket"],
        object_name=attempt["result_object"],
    )
    if result_read is None:
        raise QualityProviderError("fresh-quality DONE exists without its result")
    result_record, _result_payload = result_read
    if (
        result_record["sha256"] != done.get("result_sha256")
        or result_record["bytes"] != done.get("result_bytes")
    ):
        raise QualityProviderError("fresh-quality DONE/result binding changed")
    task_records = done.get("task_records")
    expected_count = 10 if job["phase"] == "primary" else 2
    if (
        not isinstance(task_records, list)
        or len(task_records) != expected_count
        or done.get("completed_root_count") != expected_count
    ):
        raise QualityProviderError("fresh-quality task object count changed")
    tasks: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for task in task_records:
        if not isinstance(task, Mapping):
            raise QualityProviderError("fresh-quality DONE task record changed")
        relative = task.get("path")
        if (
            not isinstance(relative, str)
            or re.fullmatch(r"tasks/root_[0-9]{3}\.json", relative) is None
        ):
            raise QualityProviderError("fresh-quality task path changed")
        object_name = f"{attempt['task_object_prefix']}{PurePosixPath(relative).name}"
        if object_name in seen_names:
            raise QualityProviderError("fresh-quality task object is duplicated")
        seen_names.add(object_name)
        read = _read_object_record(
            transport,
            bucket=provider["bucket"],
            object_name=object_name,
        )
        if read is None:
            raise QualityProviderError("fresh-quality DONE task object is absent")
        record, _payload = read
        if (
            record["sha256"] != task.get("sha256")
            or record["bytes"] != task.get("bytes")
        ):
            raise QualityProviderError("fresh-quality DONE task binding changed")
        tasks.append(record)
    # A GCS generation identifies one immutable object version; it is not a
    # cross-object sequence number and therefore cannot prove publish order.
    # DONE-last is instead enforced by the hash-pinned worker startup: every
    # task/result create is awaited before the create-only DONE upload begins.
    return {
        "job_id": selected["job_id"],
        "attempt_id": selected["attempt_id"],
        "instance_id": selected["instance_id"],
        "status": "ready",
        "result_object": result_record,
        "done_object": done_record,
        "task_objects": tasks,
    }


def cleanup_wave(
    *,
    provider_plan: Mapping[str, Any],
    bridge_plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    wave_request: Mapping[str, Any],
    launch_receipt: Mapping[str, Any] | None,
    transport: QualityCloudTransport,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    provider, plan, checked_ledger, checked_resume, request = _validate_provider_context(
        provider_plan=provider_plan,
        bridge_plan=bridge_plan,
        ledger=ledger,
        resume=resume,
        wave_request=wave_request,
    )
    launch = (
        None
        if launch_receipt is None
        else validate_launch_receipt(
            launch_receipt,
            provider_plan=provider,
            bridge_plan=plan,
            ledger=checked_ledger,
            resume=checked_resume,
            wave_request=request,
        )
    )
    workers = {row["job_id"]: row for row in provider["workers"]}
    selected = {row["job_id"]: row for row in request["selected_attempts"]}
    launched = (
        {}
        if launch is None
        else {row["job_id"]: row for row in launch["rows"]}
    )
    recovered_stage: dict[str, Any] | None = (
        None if launch is None else launch["stage_receipt"]
    )
    for job_id, selected_row in selected.items():
        instance = transport.get_instance(instance_name=selected_row["instance_id"])
        if instance is None:
            continue
        if recovered_stage is None:
            recovered_stage = observe_staged_content(
                provider_plan=provider, transport=transport
            )
        spec = _instance_spec(
            provider=provider,
            worker=workers[job_id],
            selected=selected_row,
            stage_receipt=recovered_stage,
        )
        provider_id, disk_name, _status = _validate_owned_instance(
            instance, expected_spec=spec
        )
        launch_row = launched.get(job_id)
        disk_value = transport.get_disk_optional(disk_name=disk_name)
        if disk_value is None:
            raise PermissionError("cleanup GCE boot disk disappeared before delete")
        disk_id = _validate_owned_disk(disk_value, expected_spec=spec)
        if launch_row is not None and (
            launch_row["provider_instance_id"] != provider_id
            or launch_row["provider_boot_disk_id"] != disk_id
        ):
            raise PermissionError("cleanup GCE provider identity changed")
        request_id = _uuid_for(
            provider["provider_plan_sha256"],
            job_id,
            selected_row["attempt_id"],
            "delete",
        )
        try:
            initial = transport.delete_instance(
                instance_name=selected_row["instance_id"],
                request_id=request_id,
            )
        except c4_gcp.ResponseLostError:
            if (
                transport.get_instance(instance_name=selected_row["instance_id"])
                is not None
            ):
                initial = transport.delete_instance(
                    instance_name=selected_row["instance_id"],
                    request_id=request_id,
                )
            else:
                initial = None
        if initial is not None:
            _wait_operation(
                transport,
                initial=initial,
                operation_type="delete",
                target_name=selected_row["instance_id"],
                sleep=sleep,
            )
    for attempt in range(ABSENCE_POLL_ATTEMPTS):
        remaining_instances = [
            row
            for row in request["selected_attempts"]
            if transport.get_instance(instance_name=row["instance_id"]) is not None
        ]
        if not remaining_instances:
            break
        if attempt + 1 < ABSENCE_POLL_ATTEMPTS:
            sleep(ABSENCE_POLL_INTERVAL_SECONDS)
    else:
        raise TimeoutError("owned fresh-quality VM absence was not proven")

    # A lost/partial insert or delayed auto-delete can leave the exact boot
    # disk after its VM is absent.  Validate the full immutable disk spec and
    # provider ID before deleting that selected-name orphan.
    for job_id, selected_row in selected.items():
        disk_value = transport.get_disk_optional(
            disk_name=selected_row["instance_id"]
        )
        if disk_value is None:
            continue
        if recovered_stage is None:
            recovered_stage = observe_staged_content(
                provider_plan=provider, transport=transport
            )
        spec = _instance_spec(
            provider=provider,
            worker=workers[job_id],
            selected=selected_row,
            stage_receipt=recovered_stage,
        )
        disk_id = _validate_owned_disk(disk_value, expected_spec=spec)
        launch_row = launched.get(job_id)
        if (
            launch_row is not None
            and launch_row["provider_boot_disk_id"] != disk_id
        ):
            raise PermissionError("cleanup orphan boot disk provider ID changed")
        request_id = _uuid_for(
            provider["provider_plan_sha256"],
            job_id,
            selected_row["attempt_id"],
            "delete-disk",
        )
        try:
            initial = transport.delete_disk(
                disk_name=selected_row["instance_id"],
                request_id=request_id,
            )
        except c4_gcp.ResponseLostError:
            if (
                transport.get_disk_optional(
                    disk_name=selected_row["instance_id"]
                )
                is not None
            ):
                initial = transport.delete_disk(
                    disk_name=selected_row["instance_id"],
                    request_id=request_id,
                )
            else:
                initial = None
        if initial is not None:
            _wait_operation(
                transport,
                initial=initial,
                operation_type="delete",
                target_name=selected_row["instance_id"],
                sleep=sleep,
            )
    for attempt in range(ABSENCE_POLL_ATTEMPTS):
        remaining = [
            row
            for row in request["selected_attempts"]
            if transport.get_instance(instance_name=row["instance_id"]) is not None
            or transport.get_disk_optional(disk_name=row["instance_id"]) is not None
        ]
        if not remaining:
            break
        if attempt + 1 < ABSENCE_POLL_ATTEMPTS:
            sleep(ABSENCE_POLL_INTERVAL_SECONDS)
    else:
        raise TimeoutError("owned fresh-quality VM/disk absence was not proven")
    if launch is None:
        iam_cleanup = remove_worker_iam_without_receipt(
            provider_plan=provider,
            transport=transport,
        )
        validate_abort_iam_cleanup_receipt(
            iam_cleanup,
            provider_plan=provider,
        )
    else:
        iam_cleanup = remove_worker_iam(
            provider_plan=provider,
            iam_receipt=launch["iam_receipt"],
            transport=transport,
        )
        validate_iam_cleanup_receipt(
            iam_cleanup,
            provider_plan=provider,
            iam_receipt=launch["iam_receipt"],
        )
    attempt_rows = [
        _collect_attempt_objects(
            provider=provider,
            plan=plan,
            selected=row,
            transport=transport,
        )
        for row in request["selected_attempts"]
    ]
    lifecycle_core = {
        "schema": bridge.LIFECYCLE_SCHEMA,
        "status": "exact_wave_terminal_cleanup_and_receiver_handoff_ready",
        "plan_sha256": plan["plan_sha256"],
        "ledger_sha256": checked_ledger["ledger_sha256"],
        "resume_sha256": checked_resume["resume_sha256"],
        "request_sha256": request["request_sha256"],
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_index": checked_resume["resume_wave_index"],
        "attempt_rows": attempt_rows,
        "launch_create_only": True,
        "one_vm_per_job": True,
        "unlisted_vm_created": 0,
        "wildcard_delete_used": False,
        "unrelated_resource_touched": False,
        "owned_compute_absent": True,
        "worker_iam_removed": True,
        "receiver_handoff_ready": True,
        "current_profile_changed": False,
    }
    lifecycle = {
        **lifecycle_core,
        "receipt_sha256": canonical_sha256(lifecycle_core),
    }
    return bridge.validate_lifecycle_receipt(
        plan,
        checked_ledger,
        checked_resume,
        request,
        lifecycle,
    )


class GcsGenerationReader:
    """Read-only adapter consumed by the source-replaying bridge receiver."""

    def __init__(
        self,
        *,
        transport: QualityCloudTransport,
        expected_records: Sequence[Mapping[str, Any]],
    ) -> None:
        self._transport = transport
        self._records = {
            str(row["name"]): deepcopy(dict(row)) for row in expected_records
        }
        if len(self._records) != len(expected_records):
            raise ValueError("receiver expected object record is duplicated")

    def read_object(self, *, bucket: str, object_name: str) -> Mapping[str, Any]:
        expected = self._records.get(object_name)
        if expected is None:
            raise PermissionError("receiver attempted an unlisted object read")
        metadata = self._transport.get_object_metadata(
            bucket=bucket, object_name=object_name
        )
        if (
            metadata is None
            or metadata.get("name") != object_name
            or str(metadata.get("generation")) != expected["generation"]
        ):
            raise QualityProviderError("receiver GCS generation changed")
        payload = self._transport.get_object_bytes(
            bucket=bucket,
            object_name=object_name,
            generation=expected["generation"],
        )
        if (
            payload is None
            or hashlib.sha256(payload).hexdigest() != expected["sha256"]
            or len(payload) != expected["bytes"]
        ):
            raise QualityProviderError("receiver GCS bytes changed")
        return {**expected, "payload": payload}


def receive_wave(
    *,
    bridge_plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    wave_request: Mapping[str, Any],
    lifecycle_receipt: Mapping[str, Any],
    transport: QualityCloudTransport,
    output_directory: str | Path,
    accepted_results_directory: str | Path | None = None,
) -> dict[str, Any]:
    plan = bridge.validate_gcp_plan(bridge_plan, replay_sources=True)
    checked_ledger = bridge.validate_attempt_ledger(plan, ledger)
    checked_resume = bridge.validate_resume_plan(plan, checked_ledger, resume)
    request = bridge.validate_wave_request(
        plan, checked_ledger, checked_resume, wave_request
    )
    lifecycle = bridge.validate_lifecycle_receipt(
        plan, checked_ledger, checked_resume, request, lifecycle_receipt
    )
    records = [
        object_record
        for row in lifecycle["attempt_rows"]
        if row["status"] == "ready"
        for object_record in [
            row["result_object"],
            row["done_object"],
            *row["task_objects"],
        ]
    ]
    reader = GcsGenerationReader(
        transport=transport, expected_records=records
    )
    return bridge.receive_ready_jobs(
        plan=plan,
        ledger=checked_ledger,
        resume=checked_resume,
        request=request,
        lifecycle_receipt=lifecycle,
        reader=reader,
        output_directory=output_directory,
        accepted_results_directory=accepted_results_directory,
    )


__all__ = [
    "GcpQualityRestAdapter",
    "GcsGenerationReader",
    "IAM_CLEANUP_SCHEMA",
    "IAM_RECEIPT_SCHEMA",
    "LAUNCH_RECEIPT_SCHEMA",
    "POLL_RECEIPT_SCHEMA",
    "PROVIDER_PLAN_SCHEMA",
    "QualityCloudTransport",
    "QualityProviderError",
    "STAGE_RECEIPT_SCHEMA",
    "build_provider_plan",
    "cleanup_wave",
    "execute_wave",
    "install_worker_iam",
    "poll_wave",
    "read_quota",
    "receive_wave",
    "remove_worker_iam",
    "remove_worker_iam_without_receipt",
    "stage_content",
    "validate_abort_iam_cleanup_receipt",
    "validate_iam_cleanup_receipt",
    "validate_iam_receipt",
    "validate_launch_receipt",
    "validate_poll_receipt",
    "validate_provider_plan",
    "validate_stage_receipt",
    "worker_startup_script",
]
