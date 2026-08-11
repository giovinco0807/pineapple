"""GCP provider boundary for the M3.1 parallel-20 dataset transport.

Planning and preflight are read-only.  The provider assumes twenty existing,
unique worker service accounts; it never creates service accounts.  A launch
requires a fresh OAuth-TTL receipt, live C4/Spot/global quota and inventory
headroom, exact staged bytes, and wave-scoped IAM.  Any partial create is
reconciled and deleted before a no-go receipt is returned.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import os
import re
import time
import urllib.parse
import uuid
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import hu_m31_t3_dataset_gcp_provider_v1 as provider_v1
from . import hu_m31_t3_dataset_gcp_transport_v1 as transport_v1
from . import hu_m31_t3_dataset_gcp_transport_v2 as bridge
from . import hu_m31_t3_step6d_fresh_quality_gcp_provider_v1 as quality_provider
from . import hu_rl_c4_gcp_lifecycle as c4_gcp


PROVIDER_PLAN_SCHEMA = "hu_m31_t3_dataset_gcp_provider_plan_v2"
PREFLIGHT_RECEIPT_SCHEMA = "hu_m31_t3_dataset_gcp_preflight_receipt_v2"
STAGE_RECEIPT_SCHEMA = "hu_m31_t3_dataset_gcp_stage_receipt_v2"
IAM_RECEIPT_SCHEMA = "hu_m31_t3_dataset_gcp_worker_iam_v2"
IAM_CLEANUP_SCHEMA = "hu_m31_t3_dataset_gcp_worker_iam_cleanup_v2"
LAUNCH_RECEIPT_SCHEMA = "hu_m31_t3_dataset_gcp_launch_receipt_v2"
CLEANUP_RECEIPT_SCHEMA = "hu_m31_t3_dataset_gcp_compute_cleanup_v2"
POLL_RECEIPT_SCHEMA = "hu_m31_t3_dataset_gcp_poll_receipt_v2"
MIRROR_RECEIPT_SCHEMA = "hu_m31_t3_dataset_gcp_mirror_receipt_v2"
MIN_POST_LAUNCH_RESERVE_VCPUS = 35

PROJECT = provider_v1.PROJECT
REGION = provider_v1.REGION
ZONE = provider_v1.ZONE
MACHINE_TYPE = provider_v1.MACHINE_TYPE
BOOT_DISK_TYPE = provider_v1.BOOT_DISK_TYPE
BOOT_DISK_INTERFACE = provider_v1.BOOT_DISK_INTERFACE
BOOT_DISK_SIZE_GB = provider_v1.BOOT_DISK_SIZE_GB
NETWORK = provider_v1.NETWORK
SUBNETWORK = provider_v1.SUBNETWORK
NIC_TYPE = provider_v1.NIC_TYPE
OAUTH_SCOPE = provider_v1.OAUTH_SCOPE
NAT_ROUTER_NAME = provider_v1.NAT_ROUTER_NAME
NAT_NAME = provider_v1.NAT_NAME
MAX_RUN_SECONDS = provider_v1.MAX_RUN_SECONDS
WATCHDOG_SECONDS = provider_v1.WATCHDOG_SECONDS
IAM_TTL_SECONDS = provider_v1.IAM_TTL_SECONDS
ABSENCE_POLL_ATTEMPTS = provider_v1.ABSENCE_POLL_ATTEMPTS
ABSENCE_POLL_INTERVAL_SECONDS = provider_v1.ABSENCE_POLL_INTERVAL_SECONDS

_SERVICE_ACCOUNT = re.compile(
    rf"^[a-z][a-z0-9-]{{4,28}}[a-z0-9]@{re.escape(PROJECT)}"
    r"\.iam\.gserviceaccount\.com$"
)
_IMAGE_LINK = provider_v1._IMAGE_LINK  # type: ignore[attr-defined]
_PROVIDER_ID = provider_v1._PROVIDER_ID  # type: ignore[attr-defined]
EXPECTED_WORKER_SERVICE_ACCOUNTS = tuple(
    f"ofc-f100-worker-{index:02d}@{PROJECT}.iam.gserviceaccount.com"
    for index in range(bridge.MAX_CONCURRENT_VMS)
)


class DatasetProviderV2Error(RuntimeError):
    """Cloud state is incomplete, ambiguous, or outside the v2 plan."""


class ParallelDatasetCloudTransport(Protocol):
    def get_oauth_token_ttl_seconds(self) -> int: ...

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

    def delete_instance(
        self, *, instance_name: str, request_id: str
    ) -> Mapping[str, Any] | None: ...

    def get_disk_optional(
        self, *, disk_name: str
    ) -> Mapping[str, Any] | None: ...

    def get_zone_operation(self, *, operation_name: str) -> Mapping[str, Any]: ...

    def get_bucket_iam_policy(self, *, bucket: str) -> Mapping[str, Any]: ...

    def set_bucket_iam_policy(
        self, *, bucket: str, policy: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def get_service_account(self, *, email: str) -> Mapping[str, Any]: ...

    def test_service_account_act_as(self, *, email: str) -> bool: ...

    def get_region_quota(self, *, region: str) -> Mapping[str, Any]: ...

    def get_cloud_quota(self, *, quota_id: str) -> Mapping[str, Any]: ...

    def list_instances(self) -> Sequence[Mapping[str, Any]]: ...

    def get_machine_type_url(
        self, *, self_link: str
    ) -> Mapping[str, Any]: ...

    def get_image(self, *, self_link: str) -> Mapping[str, Any]: ...

    def get_router(
        self, *, region: str, router_name: str
    ) -> Mapping[str, Any]: ...


class GcpParallelDatasetRestAdapter(provider_v1.GcpDatasetRestAdapter):
    """Concrete REST adapter with non-persisted OAuth token introspection."""

    def get_oauth_token_ttl_seconds(self) -> int:
        # Google tokeninfo does not expose a bearer-only form.  The URL and
        # response stay in memory and are never copied into a receipt.
        url = (
            "https://oauth2.googleapis.com/tokeninfo?access_token="
            + urllib.parse.quote(self._token, safe="")  # type: ignore[attr-defined]
        )
        response = self._request(  # type: ignore[attr-defined]
            "GET", url, {"Accept": "application/json"}, None, 30
        )
        if response.status < 200 or response.status >= 300:
            raise DatasetProviderV2Error(
                f"OAuth tokeninfo failed with HTTP {response.status}"
            )
        try:
            value = json.loads(response.body.decode("utf-8"))
            ttl = int(value["expires_in"])
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise DatasetProviderV2Error("OAuth token TTL readback changed") from exc
        if ttl < 0:
            raise DatasetProviderV2Error("OAuth token TTL is negative")
        return ttl


def canonical_bytes(value: Any) -> bytes:
    return bridge.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return bridge.canonical_sha256(value)


def _self_digest(value: Mapping[str, Any], field: str) -> str:
    copied = deepcopy(dict(value))
    copied.pop(field, None)
    return canonical_sha256(copied)


def _utc(epoch_seconds: int) -> str:
    return (
        dt.datetime.fromtimestamp(epoch_seconds, tz=dt.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )


def _uuid_for(*parts: str) -> str:
    raw = hashlib.sha256("\0".join(parts).encode("ascii")).hexdigest()
    return str(uuid.UUID(raw[:32], version=4))


def _plain_file(path: str | Path, label: str) -> Path:
    source = Path(path).resolve()
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    return source


def _content_type(path: Path) -> str:
    return "application/json" if path.suffix == ".json" else "application/octet-stream"


def _resume_files(selected: Mapping[str, Any], local_root: Path) -> list[Path]:
    return provider_v1._resume_files(  # type: ignore[attr-defined]
        bridge_plan={},
        selected=selected,
        local_shard_root=local_root,
    )


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
    state = bridge.validate_attempt_ledger(plan, ledger)
    resumed = bridge.validate_resume_plan(plan, state, resume)
    request = bridge.validate_wave_request(plan, state, resumed, wave_request)
    pool = list(worker_service_accounts)
    features = sorted(set(guest_os_features))
    if (
        pool != list(EXPECTED_WORKER_SERVICE_ACCOUNTS)
        or len(pool) != bridge.MAX_CONCURRENT_VMS
        or len(set(pool)) != len(pool)
        or any(_SERVICE_ACCOUNT.fullmatch(email) is None for email in pool)
        or len(features) != len(guest_os_features)
        or not features
        or _IMAGE_LINK.fullmatch(image_self_link) is None
        or _PROVIDER_ID.fullmatch(str(image_id)) is None
    ):
        raise ValueError("parallel20 provider requires 20 unique existing workers")
    workers = [
        {
            "slot": index,
            "shard_id": selected["shard_id"],
            "attempt_id": selected["attempt_id"],
            "instance_id": selected["instance_id"],
            "service_account": pool[index],
        }
        for index, selected in enumerate(request["selected_attempts"])
    ]
    base = (
        f"m31-dataset/{plan['execution_identity_sha256']}/content/"
        f"{plan['source_identity']['content_sha256'][:20]}/"
    )
    entries: list[dict[str, Any]] = []
    for record in plan["content_sources"]:
        path = _plain_file(record["source_path"], record["kind"])
        entries.append(
            {
                "kind": record["kind"],
                "local_path": str(path),
                "object_name": f"{base}static/{record['kind']}/{record['filename']}",
                "sha256": bridge.sha256_file(path),
                "bytes": path.stat().st_size,
                "content_type": _content_type(path),
                "shard_id": None,
                "relative_path": f"static/{record['kind']}/{record['filename']}",
            }
        )
    local_root = Path(local_shard_root).resolve()
    for selected in request["selected_attempts"]:
        directory = local_root / selected["shard_id"]
        for path in _resume_files(selected, local_root):
            relative = path.relative_to(directory).as_posix()
            entries.append(
                {
                    "kind": "resume_file",
                    "local_path": str(path),
                    "object_name": (
                        f"{base}resume/{selected['shard_id']}/"
                        f"{selected['resume_completed_pair_count']:06d}/{relative}"
                    ),
                    "sha256": bridge.sha256_file(path),
                    "bytes": path.stat().st_size,
                    "content_type": _content_type(path),
                    "shard_id": selected["shard_id"],
                    "relative_path": f"resume/{relative}",
                }
            )
    prefix = (
        f"ofc-ds2-{plan['execution_identity_sha256'][:10]}-"
        f"w{request['wave_index']:02d}"
    )
    condition_titles = [
        f"{prefix}-read",
        *[f"{prefix}-create-s{index:02d}" for index in range(len(workers))],
    ]
    core = {
        "schema": PROVIDER_PLAN_SCHEMA,
        "status": "parallel20_provider_ready_cloud_not_mutated",
        "transport_plan_sha256": plan["plan_sha256"],
        "scientific_v1_plan_sha256": plan["scientific_v1_lock"][
            "transport_v1_plan_sha256"
        ],
        "source_ext4_relocation_receipt_sha256": plan[
            "source_ext4_relocation_receipt_sha256"
        ],
        "ledger_sha256": state["ledger_sha256"],
        "resume_sha256": resumed["resume_sha256"],
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
        "worker_service_account_pool": pool,
        "workers": workers,
        "selected_attempts": deepcopy(request["selected_attempts"]),
        "content_entries": entries,
        "local_shard_root": str(local_root),
        "claim_object": (
            f"m31-dataset-v2/{plan['execution_identity_sha256']}/claims/"
            f"{request['request_sha256']}.json"
        ),
        "iam_contract": {
            "viewer_role": "roles/storage.objectViewer",
            "creator_role": "roles/storage.objectCreator",
            "members": [
                f"serviceAccount:{row['service_account']}" for row in workers
            ],
            "reader_prefix": base,
            "condition_titles": condition_titles,
            "ttl_seconds": IAM_TTL_SECONDS,
        },
        "runtime_contract": {
            "machine_type": MACHINE_TYPE,
            "vcpus_per_vm": bridge.VCPUS_PER_VM,
            "max_vm_count": len(workers),
            "max_concurrent_vms": bridge.MAX_CONCURRENT_VMS,
            "required_vcpus": len(workers) * bridge.VCPUS_PER_VM,
            "minimum_post_launch_reserve_vcpus": (
                MIN_POST_LAUNCH_RESERVE_VCPUS
            ),
            "one_vm_per_shard": True,
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
        "startup_script_sha256": hashlib.sha256(
            worker_startup_script().encode("utf-8")
        ).hexdigest(),
        "quality_and_smoke_source_replayed": True,
        "cloud_launch_authorized": True,
        "cloud_mutated": False,
        "service_accounts_created": False,
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
    state = bridge.validate_attempt_ledger(plan, ledger)
    resumed = bridge.validate_resume_plan(plan, state, resume)
    request = bridge.validate_wave_request(plan, state, resumed, wave_request)
    provider = deepcopy(dict(value))
    workers = provider.get("workers")
    pool = provider.get("worker_service_account_pool")
    entries = provider.get("content_entries")
    if (
        provider.get("provider_plan_sha256")
        != _self_digest(provider, "provider_plan_sha256")
        or provider.get("schema") != PROVIDER_PLAN_SCHEMA
        or provider.get("status") != "parallel20_provider_ready_cloud_not_mutated"
        or provider.get("transport_plan_sha256") != plan["plan_sha256"]
        or provider.get("scientific_v1_plan_sha256")
        != plan["scientific_v1_lock"]["transport_v1_plan_sha256"]
        or provider.get("ledger_sha256") != state["ledger_sha256"]
        or provider.get("resume_sha256") != resumed["resume_sha256"]
        or provider.get("wave_request_sha256") != request["request_sha256"]
        or not isinstance(pool, list)
        or pool != list(EXPECTED_WORKER_SERVICE_ACCOUNTS)
        or len(pool) != bridge.MAX_CONCURRENT_VMS
        or len(set(pool)) != len(pool)
        or any(_SERVICE_ACCOUNT.fullmatch(email) is None for email in pool)
        or not isinstance(workers, list)
        or len(workers) != request["selected_count"]
        or len(workers) > bridge.MAX_CONCURRENT_VMS
        or not isinstance(entries, list)
        or provider.get("selected_attempts") != request["selected_attempts"]
        or provider.get("runtime_contract", {}).get("required_vcpus")
        != len(workers) * bridge.VCPUS_PER_VM
        or provider.get("runtime_contract", {}).get(
            "minimum_post_launch_reserve_vcpus"
        )
        != MIN_POST_LAUNCH_RESERVE_VCPUS
        or provider.get("runtime_contract", {}).get("one_vm_per_shard") is not True
        or provider.get("startup_script_sha256")
        != hashlib.sha256(worker_startup_script().encode("utf-8")).hexdigest()
        or provider.get("cloud_mutated") is not False
        or provider.get("service_accounts_created") is not False
        or provider.get("current_profile_changed") is not False
    ):
        raise ValueError("parallel20 provider plan binding changed")
    expected_workers = [
        {
            "slot": index,
            "shard_id": row["shard_id"],
            "attempt_id": row["attempt_id"],
            "instance_id": row["instance_id"],
            "service_account": pool[index],
        }
        for index, row in enumerate(request["selected_attempts"])
    ]
    if workers != expected_workers:
        raise ValueError("parallel20 worker mapping changed")
    seen: set[str] = set()
    for entry in entries:
        path = _plain_file(entry["local_path"], "parallel20 content")
        if (
            entry["object_name"] in seen
            or bridge.sha256_file(path) != entry["sha256"]
            or path.stat().st_size != entry["bytes"]
        ):
            raise ValueError("parallel20 provider content changed")
        seen.add(entry["object_name"])
    return provider


def worker_startup_script() -> str:
    """The exact v1 byte-producing worker is retained."""

    return provider_v1.worker_startup_script()


def build_readonly_preflight(
    *,
    provider_plan: Mapping[str, Any],
    transport_plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    wave_request: Mapping[str, Any],
    transport: ParallelDatasetCloudTransport,
    observed_at_utc: str,
) -> dict[str, Any]:
    provider = validate_provider_plan(
        provider_plan,
        transport_plan=transport_plan,
        ledger=ledger,
        resume=resume,
        wave_request=wave_request,
    )
    oauth_ttl = transport.get_oauth_token_ttl_seconds()
    if (
        not isinstance(oauth_ttl, int)
        or isinstance(oauth_ttl, bool)
        or oauth_ttl < bridge.MIN_OAUTH_TTL_SECONDS
    ):
        raise PermissionError("OAuth token TTL is below 45 minutes")
    identities = []
    for email in provider["worker_service_account_pool"]:
        account = transport.get_service_account(email=email)
        if (
            account.get("email") != email
            or _PROVIDER_ID.fullmatch(str(account.get("uniqueId", ""))) is None
            or transport.test_service_account_act_as(email=email) is not True
        ):
            raise PermissionError("parallel20 worker identity preflight failed")
        identities.append(
            {"email": email, "unique_id": str(account["uniqueId"]), "act_as": True}
        )
    if len({row["unique_id"] for row in identities}) != bridge.MAX_CONCURRENT_VMS:
        raise PermissionError("parallel20 worker provider IDs are not unique")
    quality_provider._validate_image(provider, transport)  # type: ignore[arg-type]
    quality_provider._validate_network(provider, transport)  # type: ignore[arg-type]
    quota = quality_provider.read_quota(  # type: ignore[arg-type]
        provider_plan=provider,
        transport=transport,
        observed_at_utc=observed_at_utc,
    )
    required = len(provider["workers"]) * bridge.VCPUS_PER_VM
    if quota.get("required_vcpus") != required or quota.get("sufficient") is not True:
        raise PermissionError("parallel20 live quota receipt is insufficient")
    post_launch = []
    for metric in quota.get("metrics", []):
        remaining = int(metric["headroom_vcpus"]) - required
        post_launch.append(
            {
                "metric": metric["metric"],
                "remaining_vcpus": remaining,
                "minimum_reserve_vcpus": MIN_POST_LAUNCH_RESERVE_VCPUS,
                "sufficient": remaining >= MIN_POST_LAUNCH_RESERVE_VCPUS,
            }
        )
    if len(post_launch) != 3 or not all(row["sufficient"] for row in post_launch):
        raise PermissionError("parallel20 post-launch vCPU reserve is below 35")
    conflicts = []
    for selected in provider["selected_attempts"]:
        instance = transport.get_instance(instance_name=selected["instance_id"])
        disk = transport.get_disk_optional(disk_name=selected["instance_id"])
        if instance is not None or disk is not None:
            conflicts.append(selected["instance_id"])
    if conflicts:
        raise PermissionError("parallel20 selected instance inventory is not absent")
    core = {
        "schema": PREFLIGHT_RECEIPT_SCHEMA,
        "status": "parallel20_readonly_preflight_passed",
        "provider_plan_sha256": provider["provider_plan_sha256"],
        "wave_request_sha256": provider["wave_request_sha256"],
        "observed_at_utc": observed_at_utc,
        "oauth_ttl_seconds": oauth_ttl,
        "oauth_minimum_ttl_seconds": bridge.MIN_OAUTH_TTL_SECONDS,
        "oauth_token_persisted": False,
        "service_account_count": len(identities),
        "service_accounts": identities,
        "service_accounts_created": False,
        "quota_receipt": quota,
        "required_vcpus": required,
        "post_launch_reserve": post_launch,
        "minimum_post_launch_reserve_vcpus": MIN_POST_LAUNCH_RESERVE_VCPUS,
        "selected_instance_conflicts": [],
        "selected_instance_disk_absent": True,
        "live_regional_c4_headroom_checked": True,
        "live_spot_headroom_checked": True,
        "live_instance_inventory_checked": True,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def validate_readonly_preflight(
    value: Mapping[str, Any], *, provider_plan: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = deepcopy(dict(value))
    workers = provider_plan["workers"]
    if (
        receipt.get("receipt_sha256") != _self_digest(receipt, "receipt_sha256")
        or receipt.get("schema") != PREFLIGHT_RECEIPT_SCHEMA
        or receipt.get("status") != "parallel20_readonly_preflight_passed"
        or receipt.get("provider_plan_sha256")
        != provider_plan["provider_plan_sha256"]
        or receipt.get("wave_request_sha256")
        != provider_plan["wave_request_sha256"]
        or receipt.get("oauth_ttl_seconds", -1) < bridge.MIN_OAUTH_TTL_SECONDS
        or receipt.get("oauth_minimum_ttl_seconds")
        != bridge.MIN_OAUTH_TTL_SECONDS
        or receipt.get("oauth_token_persisted") is not False
        or receipt.get("service_account_count") != bridge.MAX_CONCURRENT_VMS
        or receipt.get("service_accounts_created") is not False
        or receipt.get("required_vcpus") != len(workers) * bridge.VCPUS_PER_VM
        or receipt.get("minimum_post_launch_reserve_vcpus")
        != MIN_POST_LAUNCH_RESERVE_VCPUS
        or not isinstance(receipt.get("post_launch_reserve"), list)
        or len(receipt["post_launch_reserve"]) != 3
        or any(
            row.get("sufficient") is not True
            for row in receipt["post_launch_reserve"]
        )
        or receipt.get("quota_receipt", {}).get("sufficient") is not True
        or receipt.get("selected_instance_conflicts") != []
        or receipt.get("selected_instance_disk_absent") is not True
        or receipt.get("live_regional_c4_headroom_checked") is not True
        or receipt.get("live_spot_headroom_checked") is not True
        or receipt.get("live_instance_inventory_checked") is not True
        or receipt.get("cloud_mutated") is not False
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("parallel20 preflight receipt changed")
    return receipt


def stage_content(
    *,
    provider_plan: Mapping[str, Any],
    transport: ParallelDatasetCloudTransport,
) -> dict[str, Any]:
    records = []
    mutated = False
    for entry in provider_plan["content_entries"]:
        payload = _plain_file(entry["local_path"], "parallel20 content").read_bytes()
        if (
            hashlib.sha256(payload).hexdigest() != entry["sha256"]
            or len(payload) != entry["bytes"]
        ):
            raise ValueError("parallel20 staged source changed")
        metadata = transport.get_object_metadata(
            bucket=provider_plan["bucket"], object_name=entry["object_name"]
        )
        created = False
        if metadata is None:
            metadata = transport.put_object_new(
                bucket=provider_plan["bucket"],
                object_name=entry["object_name"],
                payload=payload,
                content_type=entry["content_type"],
            )
            created = True
            mutated = True
        generation = str(metadata.get("generation", ""))
        observed = transport.get_object_bytes(
            bucket=provider_plan["bucket"],
            object_name=entry["object_name"],
            generation=generation,
        )
        if observed != payload or _PROVIDER_ID.fullmatch(generation) is None:
            raise DatasetProviderV2Error("parallel20 staged object changed")
        records.append(
            {
                "kind": entry["kind"],
                "object_name": entry["object_name"],
                "generation": generation,
                "sha256": entry["sha256"],
                "bytes": entry["bytes"],
                "created": created,
            }
        )
    core = {
        "schema": STAGE_RECEIPT_SCHEMA,
        "status": "parallel20_content_hash_replayed",
        "provider_plan_sha256": provider_plan["provider_plan_sha256"],
        "content_records": records,
        "record_count": len(records),
        "create_only_or_identical_reuse": True,
        "cloud_mutated": mutated,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def validate_stage_receipt(
    value: Mapping[str, Any], *, provider_plan: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = deepcopy(dict(value))
    records = receipt.get("content_records")
    if (
        receipt.get("receipt_sha256") != _self_digest(receipt, "receipt_sha256")
        or receipt.get("schema") != STAGE_RECEIPT_SCHEMA
        or receipt.get("provider_plan_sha256")
        != provider_plan["provider_plan_sha256"]
        or not isinstance(records, list)
        or len(records) != len(provider_plan["content_entries"])
        or receipt.get("record_count") != len(records)
        or receipt.get("create_only_or_identical_reuse") is not True
        or not isinstance(receipt.get("cloud_mutated"), bool)
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("parallel20 stage receipt changed")
    for row, entry in zip(records, provider_plan["content_entries"], strict=True):
        if (
            row.get("object_name") != entry["object_name"]
            or row.get("sha256") != entry["sha256"]
            or row.get("bytes") != entry["bytes"]
            or _PROVIDER_ID.fullmatch(str(row.get("generation", ""))) is None
        ):
            raise ValueError("parallel20 staged object binding changed")
    return receipt


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
                "description": "M3.1 parallel20 immutable content read",
                "expression": (
                    f"resource.name.startsWith('{resource}{contract['reader_prefix']}') "
                    f"&& request.time < timestamp('{expires_at_utc}')"
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
                    "description": "M3.1 parallel20 create-only checkpoint",
                    "expression": (
                        f"resource.name.startsWith('{resource}{selected['object_prefix']}/') "
                        f"&& request.time < timestamp('{expires_at_utc}')"
                    ),
                },
            }
        )
    return rows


def _policy_parts(policy: Mapping[str, Any]) -> tuple[list[Any], str, int]:
    bindings = policy.get("bindings", [])
    etag = policy.get("etag")
    version = policy.get("version", 1)
    if (
        not isinstance(bindings, list)
        or not isinstance(etag, str)
        or not etag
        or not isinstance(version, int)
    ):
        raise DatasetProviderV2Error("parallel20 bucket IAM policy is incomplete")
    return deepcopy(bindings), etag, max(3, version)


def install_worker_iam(
    *,
    provider_plan: Mapping[str, Any],
    transport: ParallelDatasetCloudTransport,
    now_unix_seconds: int,
) -> dict[str, Any]:
    expires = _utc(now_unix_seconds + IAM_TTL_SECONDS)
    additions = _iam_bindings(provider_plan, expires_at_utc=expires)
    policy = dict(transport.get_bucket_iam_policy(bucket=provider_plan["bucket"]))
    bindings, etag, version = _policy_parts(policy)
    titles = set(provider_plan["iam_contract"]["condition_titles"])
    existing = [
        row
        for row in bindings
        if isinstance(row, Mapping)
        and isinstance(row.get("condition"), Mapping)
        and row["condition"].get("title") in titles
    ]
    reconciled = existing == additions
    if existing and not reconciled:
        raise FileExistsError("parallel20 worker IAM is partial or changed")
    if not reconciled:
        desired = {
            **{key: value for key, value in policy.items() if key != "bindings"},
            "version": version,
            "etag": etag,
            "bindings": [*bindings, *additions],
        }
        transport.set_bucket_iam_policy(bucket=provider_plan["bucket"], policy=desired)
    observed = dict(
        transport.get_bucket_iam_policy(bucket=provider_plan["bucket"])
    )
    observed_bindings, _observed_etag, _version = _policy_parts(observed)
    if any(row not in observed_bindings for row in additions):
        raise DatasetProviderV2Error("parallel20 IAM readback changed")
    core = {
        "schema": IAM_RECEIPT_SCHEMA,
        "status": "parallel20_temporary_worker_bindings_installed",
        "provider_plan_sha256": provider_plan["provider_plan_sha256"],
        "bucket": provider_plan["bucket"],
        "condition_titles": list(provider_plan["iam_contract"]["condition_titles"]),
        "expires_at_utc": expires,
        "bindings_installed": len(additions),
        "response_reconciled": reconciled,
        "cloud_mutated": not reconciled,
        "service_accounts_created": False,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def validate_iam_receipt(
    value: Mapping[str, Any], *, provider_plan: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = deepcopy(dict(value))
    if (
        receipt.get("receipt_sha256") != _self_digest(receipt, "receipt_sha256")
        or receipt.get("schema") != IAM_RECEIPT_SCHEMA
        or receipt.get("provider_plan_sha256")
        != provider_plan["provider_plan_sha256"]
        or set(receipt.get("condition_titles", []))
        != set(provider_plan["iam_contract"]["condition_titles"])
        or receipt.get("bindings_installed") != len(provider_plan["workers"]) + 1
        or receipt.get("service_accounts_created") is not False
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("parallel20 IAM receipt changed")
    return receipt


def remove_worker_iam(
    *,
    provider_plan: Mapping[str, Any],
    iam_receipt: Mapping[str, Any],
    transport: ParallelDatasetCloudTransport,
) -> dict[str, Any]:
    installed = validate_iam_receipt(iam_receipt, provider_plan=provider_plan)
    policy = dict(transport.get_bucket_iam_policy(bucket=provider_plan["bucket"]))
    bindings, etag, version = _policy_parts(policy)
    titles = set(installed["condition_titles"])
    retained = [
        row
        for row in bindings
        if not (
            isinstance(row, Mapping)
            and isinstance(row.get("condition"), Mapping)
            and row["condition"].get("title") in titles
        )
    ]
    removed = len(bindings) - len(retained)
    if removed not in (0, len(titles)):
        raise PermissionError("parallel20 IAM cleanup set is partial")
    mutated = removed > 0
    if mutated:
        desired = {
            **{key: value for key, value in policy.items() if key != "bindings"},
            "version": version,
            "etag": etag,
            "bindings": retained,
        }
        transport.set_bucket_iam_policy(bucket=provider_plan["bucket"], policy=desired)
    observed = transport.get_bucket_iam_policy(bucket=provider_plan["bucket"])
    observed_bindings, _etag, _version = _policy_parts(observed)
    if any(
        isinstance(row, Mapping)
        and isinstance(row.get("condition"), Mapping)
        and row["condition"].get("title") in titles
        for row in observed_bindings
    ):
        raise DatasetProviderV2Error("parallel20 IAM cleanup readback changed")
    core = {
        "schema": IAM_CLEANUP_SCHEMA,
        "status": "parallel20_worker_iam_absent",
        "provider_plan_sha256": provider_plan["provider_plan_sha256"],
        "iam_receipt_sha256": installed["receipt_sha256"],
        "bindings_removed": removed,
        "readback_absent": True,
        "unrelated_bindings_preserved": True,
        "cloud_mutated": mutated,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def _content_bindings(
    provider: Mapping[str, Any],
    stage: Mapping[str, Any],
    shard_id: str,
) -> list[dict[str, Any]]:
    receipt = validate_stage_receipt(stage, provider_plan=provider)
    by_object = {row["object_name"]: row for row in receipt["content_records"]}
    return [
        {
            "relative": entry["relative_path"],
            "object": entry["object_name"],
            "generation": by_object[entry["object_name"]]["generation"],
            "sha256": entry["sha256"],
            "bytes": entry["bytes"],
        }
        for entry in provider["content_entries"]
        if entry["shard_id"] in (None, shard_id)
    ]


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
        # Checkpoint transport metadata stays v1 so existing receiver bytes and
        # schemas remain valid; the v2 lifecycle binds it independently.
        "ds-plan-sha256": provider["scientific_v1_plan_sha256"],
        "ds-content-bindings-b64": base64.b64encode(
            canonical_bytes(bindings)
        ).decode("ascii"),
        "ds-watchdog-seconds": str(WATCHDOG_SECONDS),
        "startup-script": worker_startup_script(),
    }
    labels = {
        "ofc-owner": provider["execution_identity_sha256"][:32],
        "ofc-plan": provider["provider_plan_sha256"][:32],
        "ofc-wave": f"w{provider['wave_index']:02d}",
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
            {"email": worker["service_account"], "scopes": [OAUTH_SCOPE]}
        ],
        "metadata": {
            "items": [
                {"key": key, "value": value} for key, value in sorted(metadata.items())
            ]
        },
        "deletionProtection": False,
        "canIpForward": False,
    }


def _claim(
    provider: Mapping[str, Any],
    *,
    transport: ParallelDatasetCloudTransport,
    raw_nonce: str,
) -> dict[str, Any]:
    try:
        parsed = uuid.UUID(raw_nonce)
    except (ValueError, AttributeError) as exc:
        raise ValueError("parallel20 launch nonce must be UUIDv4") from exc
    if parsed.version != 4 or str(parsed) != raw_nonce:
        raise ValueError("parallel20 launch nonce must be canonical UUIDv4")
    value = {
        "schema": "hu_m31_t3_dataset_gcp_launch_claim_v2",
        "provider_plan_sha256": provider["provider_plan_sha256"],
        "wave_request_sha256": provider["wave_request_sha256"],
        "nonce_sha256": hashlib.sha256(raw_nonce.encode("ascii")).hexdigest(),
        "create_only": True,
    }
    payload = canonical_bytes(value)
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
    if (
        _PROVIDER_ID.fullmatch(generation) is None
        or transport.get_object_bytes(
            bucket=provider["bucket"],
            object_name=provider["claim_object"],
            generation=generation,
        )
        != payload
    ):
        raise DatasetProviderV2Error("parallel20 launch claim changed")
    return {
        "object_name": provider["claim_object"],
        "generation": generation,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "nonce_sha256": value["nonce_sha256"],
        "response_reconciled": reconciled,
    }


def _wait_operation(
    transport: ParallelDatasetCloudTransport,
    *,
    initial: Mapping[str, Any],
    operation_type: str,
    target_name: str,
    sleep: Callable[[float], None],
) -> None:
    quality_provider._wait_operation(  # type: ignore[arg-type]
        transport,
        initial=initial,
        operation_type=operation_type,
        target_name=target_name,
        sleep=sleep,
    )


def _delete_exact_selected(
    *,
    provider: Mapping[str, Any],
    stage_receipt: Mapping[str, Any],
    transport: ParallelDatasetCloudTransport,
    sleep: Callable[[float], None],
) -> list[dict[str, Any]]:
    workers = {row["shard_id"]: row for row in provider["workers"]}
    rows = []
    for selected in provider["selected_attempts"]:
        instance = transport.get_instance(instance_name=selected["instance_id"])
        if instance is None:
            rows.append(
                {
                    "shard_id": selected["shard_id"],
                    "instance_id": selected["instance_id"],
                    "instance_was_present": False,
                    "exact_owned_deleted": False,
                }
            )
            continue
        spec = _instance_spec(
            provider=provider,
            worker=workers[selected["shard_id"]],
            selected=selected,
            stage_receipt=stage_receipt,
        )
        provider_v1._validate_owned_instance(  # type: ignore[attr-defined]
            instance, expected_spec=spec
        )
        disk = transport.get_disk_optional(disk_name=selected["instance_id"])
        if disk is None:
            raise PermissionError("parallel20 boot disk vanished before cleanup")
        provider_v1._validate_owned_disk(disk, expected_spec=spec)  # type: ignore[attr-defined]
        request_id = _uuid_for(
            provider["provider_plan_sha256"],
            selected["shard_id"],
            selected["attempt_id"],
            "delete",
        )
        operation = transport.delete_instance(
            instance_name=selected["instance_id"], request_id=request_id
        )
        if operation is not None:
            _wait_operation(
                transport,
                initial=operation,
                operation_type="delete",
                target_name=selected["instance_id"],
                sleep=sleep,
            )
        rows.append(
            {
                "shard_id": selected["shard_id"],
                "instance_id": selected["instance_id"],
                "instance_was_present": True,
                "exact_owned_deleted": True,
            }
        )
    for attempt in range(ABSENCE_POLL_ATTEMPTS):
        remaining = [
            selected["instance_id"]
            for selected in provider["selected_attempts"]
            if transport.get_instance(instance_name=selected["instance_id"]) is not None
            or transport.get_disk_optional(disk_name=selected["instance_id"]) is not None
        ]
        if not remaining:
            break
        if attempt + 1 < ABSENCE_POLL_ATTEMPTS:
            sleep(ABSENCE_POLL_INTERVAL_SECONDS)
    else:
        raise TimeoutError("parallel20 VM/disk absence was not proven")
    return rows


def execute_wave_atomic(
    *,
    provider_plan: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    stage_receipt: Mapping[str, Any],
    iam_receipt: Mapping[str, Any],
    transport: ParallelDatasetCloudTransport,
    raw_nonce: str,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Launch one VM per shard; clean all exact owned compute on partial create."""

    preflight = validate_readonly_preflight(
        preflight_receipt, provider_plan=provider_plan
    )
    stage = validate_stage_receipt(stage_receipt, provider_plan=provider_plan)
    iam = validate_iam_receipt(iam_receipt, provider_plan=provider_plan)
    claim = _claim(provider_plan, transport=transport, raw_nonce=raw_nonce)
    workers = {row["shard_id"]: row for row in provider_plan["workers"]}
    rows: list[dict[str, Any]] = []
    failure: str | None = None
    try:
        for selected in provider_plan["selected_attempts"]:
            spec = _instance_spec(
                provider=provider_plan,
                worker=workers[selected["shard_id"]],
                selected=selected,
                stage_receipt=stage,
            )
            request_id = _uuid_for(
                provider_plan["provider_plan_sha256"],
                selected["shard_id"],
                selected["attempt_id"],
                "create",
            )
            if transport.get_instance(instance_name=selected["instance_id"]) is not None:
                raise FileExistsError("parallel20 selected instance pre-exists launch")
            operation = transport.create_instance(
                instance_spec=spec, request_id=request_id
            )
            _wait_operation(
                transport,
                initial=operation,
                operation_type="insert",
                target_name=selected["instance_id"],
                sleep=sleep,
            )
            instance = transport.get_instance(instance_name=selected["instance_id"])
            if instance is None:
                raise DatasetProviderV2Error("parallel20 created instance is absent")
            provider_id, disk_name, status = provider_v1._validate_owned_instance(  # type: ignore[attr-defined]
                instance, expected_spec=spec
            )
            disk = transport.get_disk_optional(disk_name=disk_name)
            if disk is None:
                raise DatasetProviderV2Error("parallel20 boot disk is absent")
            disk_id = provider_v1._validate_owned_disk(  # type: ignore[attr-defined]
                disk, expected_spec=spec
            )
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
                }
            )
    except Exception as exc:  # exact cleanup is part of the atomic boundary
        failure = f"{type(exc).__name__}: {exc}"
    cleanup = None
    iam_cleanup = None
    if failure is not None:
        deleted = _delete_exact_selected(
            provider=provider_plan,
            stage_receipt=stage,
            transport=transport,
            sleep=sleep,
        )
        iam_cleanup = remove_worker_iam(
            provider_plan=provider_plan,
            iam_receipt=iam,
            transport=transport,
        )
        cleanup = {
            "rows": deleted,
            "owned_vm_disk_absent": True,
            "worker_iam_removed": True,
            "wildcard_delete_used": False,
            "unrelated_resource_touched": False,
        }
    complete = failure is None and len(rows) == len(provider_plan["workers"])
    core = {
        "schema": LAUNCH_RECEIPT_SCHEMA,
        "status": (
            "parallel20_exact_wave_created"
            if complete
            else "partial_launch_cleaned_no_go"
        ),
        "provider_plan_sha256": provider_plan["provider_plan_sha256"],
        "wave_request_sha256": provider_plan["wave_request_sha256"],
        "preflight_receipt_sha256": preflight["receipt_sha256"],
        "stage_receipt_sha256": stage["receipt_sha256"],
        "iam_receipt_sha256": iam["receipt_sha256"],
        "claim": claim,
        "rows": rows,
        "selected_shard_count": len(provider_plan["workers"]),
        "created_instance_count": len(rows),
        "one_vm_per_shard": True,
        "at_most_twenty_c4": len(rows) <= bridge.MAX_CONCURRENT_VMS,
        "create_complete": complete,
        "failure": failure,
        "partial_cleanup": cleanup,
        "iam_cleanup_receipt": iam_cleanup,
        "partial_launch_cleanup_complete": failure is None or cleanup is not None,
        "unlisted_vm_created": 0,
        "cloud_mutated": True,
        "current_profile_changed": False,
    }
    result = {**core, "receipt_sha256": canonical_sha256(core)}
    return validate_launch_receipt(result, provider_plan=provider_plan)


def validate_launch_receipt(
    value: Mapping[str, Any], *, provider_plan: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = deepcopy(dict(value))
    rows = receipt.get("rows")
    complete = receipt.get("create_complete")
    failure = receipt.get("failure")
    partial = receipt.get("partial_cleanup")
    if (
        receipt.get("receipt_sha256") != _self_digest(receipt, "receipt_sha256")
        or receipt.get("schema") != LAUNCH_RECEIPT_SCHEMA
        or receipt.get("provider_plan_sha256")
        != provider_plan["provider_plan_sha256"]
        or receipt.get("wave_request_sha256")
        != provider_plan["wave_request_sha256"]
        or not isinstance(rows, list)
        or receipt.get("selected_shard_count") != len(provider_plan["workers"])
        or receipt.get("created_instance_count") != len(rows)
        or receipt.get("one_vm_per_shard") is not True
        or receipt.get("at_most_twenty_c4") is not True
        or len(rows) > bridge.MAX_CONCURRENT_VMS
        or not isinstance(complete, bool)
        or receipt.get("partial_launch_cleanup_complete") is not True
        or receipt.get("unlisted_vm_created") != 0
        or receipt.get("cloud_mutated") is not True
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("parallel20 launch receipt changed")
    if complete:
        if (
            receipt.get("status") != "parallel20_exact_wave_created"
            or len(rows) != len(provider_plan["workers"])
            or failure is not None
            or partial is not None
            or receipt.get("iam_cleanup_receipt") is not None
        ):
            raise ValueError("parallel20 complete launch receipt changed")
    else:
        if (
            receipt.get("status") != "partial_launch_cleaned_no_go"
            or not isinstance(failure, str)
            or not failure
            or not isinstance(partial, Mapping)
            or partial.get("owned_vm_disk_absent") is not True
            or partial.get("worker_iam_removed") is not True
            or partial.get("wildcard_delete_used") is not False
            or partial.get("unrelated_resource_touched") is not False
            or not isinstance(receipt.get("iam_cleanup_receipt"), Mapping)
        ):
            raise ValueError("parallel20 partial launch cleanup evidence changed")
    expected = provider_plan["selected_attempts"][: len(rows)]
    for row, selected in zip(rows, expected, strict=True):
        if (
            row.get("shard_id") != selected["shard_id"]
            or row.get("attempt_id") != selected["attempt_id"]
            or row.get("instance_id") != selected["instance_id"]
            or row.get("request_id")
            != _uuid_for(
                provider_plan["provider_plan_sha256"],
                selected["shard_id"],
                selected["attempt_id"],
                "create",
            )
        ):
            raise ValueError("parallel20 launch row changed")
    return receipt


def _read_remote(
    transport: ParallelDatasetCloudTransport,
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
        raise DatasetProviderV2Error("parallel20 remote object changed")
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
    transport: ParallelDatasetCloudTransport,
    *,
    bucket: str,
    pattern: str,
    maximum_sequence: int,
) -> tuple[int, dict[str, Any], bytes] | None:
    latest = None
    for sequence in range(maximum_sequence + 1):
        observed = _read_remote(
            transport, bucket=bucket, object_name=pattern % sequence
        )
        if observed is None:
            break
        record, payload = observed
        latest = (sequence, record, payload)
    return latest


def _attempt_observation(
    *,
    provider_plan: Mapping[str, Any],
    transport_plan: Mapping[str, Any],
    selected: Mapping[str, Any],
    transport: ParallelDatasetCloudTransport,
) -> dict[str, Any]:
    v1_plan = transport_v1.build_transport_plan(
        **bridge._v1_build_kwargs(transport_plan)  # type: ignore[attr-defined]
    )
    checkpoint_latest = _latest_sequence(
        transport,
        bucket=provider_plan["bucket"],
        pattern=selected["checkpoint_object_format"],
        maximum_sequence=bridge.SHARD_PAIR_COUNT,
    )
    heartbeat_latest = _latest_sequence(
        transport,
        bucket=provider_plan["bucket"],
        pattern=selected["heartbeat_object_format"],
        maximum_sequence=transport_v1.MAX_HEARTBEAT_SEQUENCE,
    )
    checkpoint_record = None
    completed = 0
    status = "failed"
    if checkpoint_latest is not None:
        sequence, record, payload = checkpoint_latest
        try:
            manifest = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DatasetProviderV2Error(
                "parallel20 checkpoint is not JSON"
            ) from exc
        if payload != canonical_bytes(manifest):
            raise DatasetProviderV2Error(
                "parallel20 checkpoint is not canonical"
            )
        manifest = transport_v1.validate_checkpoint_manifest(
            manifest,
            plan=v1_plan,
            shard_id=selected["shard_id"],
            attempt_id=selected["attempt_id"],
        )
        completed = sequence
        file_objects = []
        for file in manifest["files"]:
            observed = _read_remote(
                transport,
                bucket=provider_plan["bucket"],
                object_name=file["object_name"],
            )
            if observed is None:
                raise DatasetProviderV2Error(
                    "parallel20 checkpoint references a missing file"
                )
            file_record, _payload = observed
            if (
                file_record["sha256"] != file["sha256"]
                or file_record["bytes"] != file["bytes"]
            ):
                raise DatasetProviderV2Error(
                    "parallel20 checkpoint file bytes changed"
                )
            file_objects.append(file_record)
        checkpoint_record = {
            **record,
            "manifest": manifest,
            "file_objects": file_objects,
        }
        status = "ready" if manifest["complete"] else "checkpointed"
    heartbeat_record = None
    if heartbeat_latest is not None:
        _sequence, record, payload = heartbeat_latest
        try:
            heartbeat = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DatasetProviderV2Error(
                "parallel20 heartbeat is not JSON"
            ) from exc
        if payload != canonical_bytes(heartbeat):
            raise DatasetProviderV2Error(
                "parallel20 heartbeat is not canonical"
            )
        heartbeat = transport_v1.validate_heartbeat(
            heartbeat,
            plan=v1_plan,
            shard_id=selected["shard_id"],
            attempt_id=selected["attempt_id"],
        )
        heartbeat_record = {**record, "value": heartbeat}
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
    transport: ParallelDatasetCloudTransport,
) -> dict[str, Any]:
    checked = validate_provider_plan(
        provider_plan,
        transport_plan=transport_plan,
        ledger=ledger,
        resume=resume,
        wave_request=wave_request,
    )
    rows = []
    for selected in checked["selected_attempts"]:
        observed = _attempt_observation(
            provider_plan=checked,
            transport_plan=transport_plan,
            selected=selected,
            transport=transport,
        )
        rows.append(
            {
                "shard_id": selected["shard_id"],
                "attempt_id": selected["attempt_id"],
                "instance_present": transport.get_instance(
                    instance_name=selected["instance_id"]
                )
                is not None,
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
        "provider_plan_sha256": checked["provider_plan_sha256"],
        "rows": rows,
        "selected_shard_count": len(rows),
        "complete_count": sum(row["complete"] for row in rows),
        "read_only": True,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def build_lifecycle_after_cleanup(
    *,
    provider_plan: Mapping[str, Any],
    transport_plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    wave_request: Mapping[str, Any],
    cleanup_receipt: Mapping[str, Any],
    transport: ParallelDatasetCloudTransport,
) -> dict[str, Any]:
    validate_cleanup_receipt(cleanup_receipt, provider_plan=provider_plan)
    rows = [
        _attempt_observation(
            provider_plan=provider_plan,
            transport_plan=transport_plan,
            selected=selected,
            transport=transport,
        )
        for selected in provider_plan["selected_attempts"]
    ]
    return bridge.build_lifecycle_receipt(
        plan=transport_plan,
        ledger=ledger,
        resume=resume,
        wave_request=wave_request,
        attempt_rows=rows,
    )


def mirror_lifecycle_objects(
    *,
    transport_plan: Mapping[str, Any],
    lifecycle_receipt: Mapping[str, Any],
    bucket: str,
    output_root: str | Path,
    transport: ParallelDatasetCloudTransport,
) -> dict[str, Any]:
    lifecycle = bridge.validate_lifecycle_receipt(
        lifecycle_receipt, plan=transport_plan
    )
    root = Path(output_root).resolve()
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise ValueError("parallel20 object mirror root is unsafe")
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for row in lifecycle["attempt_rows"]:
        checkpoint = row["checkpoint"]
        if checkpoint is None:
            continue
        for object_record in checkpoint["file_objects"]:
            object_name = object_record["object"]
            payload = transport.get_object_bytes(
                bucket=bucket,
                object_name=object_name,
                generation=object_record["generation"],
            )
            if (
                payload is None
                or hashlib.sha256(payload).hexdigest()
                != object_record["sha256"]
                or len(payload) != object_record["bytes"]
            ):
                raise DatasetProviderV2Error(
                    "parallel20 mirror object changed"
                )
            destination = root.joinpath(*PurePosixPath(object_name).parts)
            if destination.exists() or destination.is_symlink():
                if (
                    destination.is_symlink()
                    or not destination.is_file()
                    or destination.read_bytes() != payload
                ):
                    raise FileExistsError(
                        "parallel20 mirror destination conflicts"
                    )
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(
                    f".{destination.name}.{os.getpid()}.tmp"
                )
                try:
                    with temporary.open("xb") as stream:
                        stream.write(payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.link(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
            records.append(
                {
                    "object": object_name,
                    "generation": object_record["generation"],
                    "sha256": object_record["sha256"],
                    "bytes": object_record["bytes"],
                }
            )
    core = {
        "schema": MIRROR_RECEIPT_SCHEMA,
        "lifecycle_receipt_sha256": lifecycle["receipt_sha256"],
        "object_root": str(root),
        "records": records,
        "record_count": len(records),
        "all_bytes_replayed": True,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def cleanup_wave(
    *,
    provider_plan: Mapping[str, Any],
    stage_receipt: Mapping[str, Any],
    iam_receipt: Mapping[str, Any],
    transport: ParallelDatasetCloudTransport,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    stage = validate_stage_receipt(stage_receipt, provider_plan=provider_plan)
    iam = validate_iam_receipt(iam_receipt, provider_plan=provider_plan)
    rows = _delete_exact_selected(
        provider=provider_plan,
        stage_receipt=stage,
        transport=transport,
        sleep=sleep,
    )
    iam_cleanup = remove_worker_iam(
        provider_plan=provider_plan,
        iam_receipt=iam,
        transport=transport,
    )
    core = {
        "schema": CLEANUP_RECEIPT_SCHEMA,
        "status": "parallel20_exact_owned_compute_and_iam_absent",
        "provider_plan_sha256": provider_plan["provider_plan_sha256"],
        "rows": rows,
        "selected_count": len(provider_plan["workers"]),
        "owned_vm_disk_absent": True,
        "worker_iam_removed": True,
        "iam_cleanup_receipt_sha256": iam_cleanup["receipt_sha256"],
        "partial_launch_cleanup_supported": True,
        "wildcard_delete_used": False,
        "unrelated_resource_touched": False,
        "current_profile_changed": False,
    }
    result = {**core, "receipt_sha256": canonical_sha256(core)}
    return validate_cleanup_receipt(result, provider_plan=provider_plan)


def validate_cleanup_receipt(
    value: Mapping[str, Any], *, provider_plan: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = deepcopy(dict(value))
    rows = receipt.get("rows")
    if (
        receipt.get("receipt_sha256") != _self_digest(receipt, "receipt_sha256")
        or receipt.get("schema") != CLEANUP_RECEIPT_SCHEMA
        or receipt.get("status")
        != "parallel20_exact_owned_compute_and_iam_absent"
        or receipt.get("provider_plan_sha256")
        != provider_plan["provider_plan_sha256"]
        or not isinstance(rows, list)
        or len(rows) != len(provider_plan["workers"])
        or receipt.get("selected_count") != len(rows)
        or receipt.get("owned_vm_disk_absent") is not True
        or receipt.get("worker_iam_removed") is not True
        or receipt.get("partial_launch_cleanup_supported") is not True
        or receipt.get("wildcard_delete_used") is not False
        or receipt.get("unrelated_resource_touched") is not False
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("parallel20 cleanup receipt changed")
    return receipt


__all__ = [
    "CLEANUP_RECEIPT_SCHEMA",
    "EXPECTED_WORKER_SERVICE_ACCOUNTS",
    "GcpParallelDatasetRestAdapter",
    "IAM_CLEANUP_SCHEMA",
    "IAM_RECEIPT_SCHEMA",
    "LAUNCH_RECEIPT_SCHEMA",
    "MIN_POST_LAUNCH_RESERVE_VCPUS",
    "MIRROR_RECEIPT_SCHEMA",
    "POLL_RECEIPT_SCHEMA",
    "PREFLIGHT_RECEIPT_SCHEMA",
    "PROVIDER_PLAN_SCHEMA",
    "STAGE_RECEIPT_SCHEMA",
    "build_provider_plan",
    "build_lifecycle_after_cleanup",
    "build_readonly_preflight",
    "canonical_bytes",
    "canonical_sha256",
    "cleanup_wave",
    "execute_wave_atomic",
    "install_worker_iam",
    "mirror_lifecycle_objects",
    "poll_wave",
    "remove_worker_iam",
    "stage_content",
    "validate_iam_receipt",
    "validate_cleanup_receipt",
    "validate_launch_receipt",
    "validate_provider_plan",
    "validate_readonly_preflight",
    "validate_stage_receipt",
    "worker_startup_script",
]
