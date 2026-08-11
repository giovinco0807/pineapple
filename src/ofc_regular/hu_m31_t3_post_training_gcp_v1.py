"""Production-safe Spot fanout for the two M3.1 post-training CPU grids.

This module intentionally has a narrower contract than the dataset transport:

* one ABR pair or one locked-promotion work item is one atomic cloud job;
* results are uploaded with ``ifGenerationMatch=0`` and a manifest last;
* an interrupted job is retried from zero under a distinct attempt prefix;
* at most eight ``c4-standard-16`` Spot VMs are launched in one wave;
* every mutable phase is guarded by an exact run name, phase sentinel, and
  controller token; production OAuth tokens must have at least 2700 seconds
  remaining and are never serialized;
* cleanup names every measured VM and boot disk, proves absence, and removes
  only the exact wave-scoped IAM conditions.

Plan construction, local smoke, status, and source replay are cloud-neutral.
Importing this module never contacts GCP or changes an AI profile.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import threading
import time
import urllib.parse
import urllib.request
import uuid
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import hu_m31_t3_abr_cli_v1 as abr_cli
from . import hu_m31_t3_abr_teacher_v1 as abr_teacher
from . import hu_m31_t3_dataset_gcp_provider_v1 as dataset_provider
from . import hu_m31_t3_locked_promotion_execution_v1 as promotion_execution
from . import hu_m31_t3_locked_promotion_provider_v1 as promotion_provider
from . import hu_m31_t3_post_training_runtime_v1 as runtime_bundle
from . import hu_m31_t3_step6d_fresh_quality_gcp_provider_v1 as quality_provider
from . import hu_m31_t3_step6d_locked_promotion_v1 as promotion
from . import hu_rl_c4_gcp_lifecycle as c4_gcp


CONTRACT_SCHEMA = "hu_m31_t3_post_training_workload_contract_v1"
SMOKE_SCHEMA = "hu_m31_t3_post_training_local_smoke_v1"
CLOUD_PLAN_SCHEMA = "hu_m31_t3_post_training_cloud_plan_v1"
WAVE_SCHEMA = "hu_m31_t3_post_training_wave_v1"
STAGE_SCHEMA = "hu_m31_t3_post_training_stage_v1"
IAM_SCHEMA = "hu_m31_t3_post_training_iam_v1"
LAUNCH_SCHEMA = "hu_m31_t3_post_training_launch_v1"
POLL_SCHEMA = "hu_m31_t3_post_training_poll_v1"
RESULT_MANIFEST_SCHEMA = "hu_m31_t3_post_training_result_manifest_v1"
LIFECYCLE_SCHEMA = "hu_m31_t3_post_training_lifecycle_v1"
RECEIVE_SCHEMA = "hu_m31_t3_post_training_receive_v1"
CONTROLLER_SCHEMA = "hu_m31_t3_post_training_controller_v1"

PROJECT = quality_provider.PROJECT
REGION = quality_provider.REGION
ZONE = quality_provider.ZONE
MACHINE_TYPE = "c4-standard-16"
VCPUS_PER_VM = 16
MAX_CONCURRENT_VMS = 8
MAX_ATTEMPTS = 2
MAX_RUN_SECONDS = 21_600
WATCHDOG_SECONDS = 21_300
IAM_TTL_SECONDS = 25_200
MIN_OAUTH_TTL_SECONDS = 2_700

CONTROLLER_TOKEN_ENV = "OFC_M31_POST_TRAINING_CONTROLLER_TOKEN"
PHASE_SENTINEL_ENV = "OFC_M31_POST_TRAINING_PHASE_SENTINEL"
OAUTH_TOKEN_ENV = "OFC_M31_POST_TRAINING_OAUTH_TOKEN"
OAUTH_EXPIRES_ENV = "OFC_M31_POST_TRAINING_OAUTH_EXPIRES_AT_UNIX"

_SHA = re.compile(r"^[0-9a-f]{64}$")
_RUN = re.compile(r"^[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?$")
_JOB = re.compile(r"^(?:abr-pair-[0-9]{4}|locked-work-[0-9]{4})$")
_PROVIDER_ID = re.compile(r"^[1-9][0-9]*$")
_SERVICE_ACCOUNT = re.compile(
    rf"^[a-z][a-z0-9-]{{4,28}}[a-z0-9]@{re.escape(PROJECT)}"
    r"\.iam\.gserviceaccount\.com$"
)
_IMAGE_LINK = re.compile(
    r"^https://www\.googleapis\.com/compute/v1/projects/"
    r"debian-cloud/global/images/[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?$"
)


class PostTrainingCloudError(RuntimeError):
    """A remote identity or lifecycle boundary changed."""


class CloudTransport(Protocol):
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


class GcpRestAdapter(quality_provider.GcpQualityRestAdapter):
    """Concrete REST transport. The OAuth bearer stays in memory only."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _self_digest(value: Mapping[str, Any], field: str) -> str:
    copied = deepcopy(dict(value))
    copied.pop(field, None)
    return canonical_sha256(copied)


def _regular_file(path: str | Path, label: str) -> Path:
    source = Path(path).resolve()
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return source


def _read_canonical(path: str | Path, label: str) -> dict[str, Any]:
    source = _regular_file(path, label)
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not a canonical JSON object")
    return value


def _write_once(path: str | Path, value: Mapping[str, Any]) -> Path:
    target = Path(path)
    raw = canonical_bytes(value)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if target.is_symlink() or not target.is_file() or target.read_bytes() != raw:
            raise FileExistsError(f"immutable artifact changed: {target}") from None
    return target.resolve()


def _safe_relative(value: str, label: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.as_posix() != value
    ):
        raise ValueError(f"{label} is unsafe")
    return value


def _source(
    *, kind: str, path: str | Path, relative_path: str
) -> dict[str, Any]:
    source = _regular_file(path, kind)
    relative = _safe_relative(relative_path, f"{kind} relative path")
    return {
        "kind": kind,
        "source_path": str(source),
        "relative_path": relative,
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


def _validate_tar(path: Path, top: str) -> None:
    with tarfile.open(path, "r:*") as archive:
        members = archive.getmembers()
        if not members:
            raise ValueError(f"{path.name} is empty")
        for member in members:
            pure = PurePosixPath(member.name)
            if (
                pure.is_absolute()
                or ".." in pure.parts
                or not pure.parts
                or pure.parts[0] != top
                or member.issym()
                or member.islnk()
            ):
                raise ValueError(f"{path.name} has an unsafe archive member")


def _validate_zip(path: Path, top: str) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if not names:
            raise ValueError(f"{path.name} is empty")
        for name in names:
            pure = PurePosixPath(name)
            if (
                pure.is_absolute()
                or ".." in pure.parts
                or not pure.parts
                or pure.parts[0] != top
            ):
                raise ValueError(f"{path.name} has an unsafe archive member")


def _tar_member_bytes(path: Path, member_name: str) -> bytes:
    with tarfile.open(path, "r:*") as archive:
        matches = [
            member
            for member in archive.getmembers()
            if member.name == member_name and member.isfile()
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{path.name} must contain exactly one {member_name}"
            )
        stream = archive.extractfile(matches[0])
        if stream is None:
            raise ValueError(f"{path.name} member cannot be read")
        return stream.read()


def _base_contract(
    *,
    run_name: str,
    bucket: str,
    workload: str,
    sources: Sequence[Mapping[str, Any]],
    jobs: Sequence[Mapping[str, Any]],
    scientific_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if _RUN.fullmatch(run_name) is None or not bucket:
        raise ValueError("post-training run name or bucket is invalid")
    source_rows = [deepcopy(dict(row)) for row in sources]
    job_rows = [deepcopy(dict(row)) for row in jobs]
    if (
        not source_rows
        or not job_rows
        or len({row["kind"] for row in source_rows}) != len(source_rows)
        or len({row["relative_path"] for row in source_rows}) != len(source_rows)
        or len({row["job_id"] for row in job_rows}) != len(job_rows)
        or any(_JOB.fullmatch(str(row["job_id"])) is None for row in job_rows)
    ):
        raise ValueError("post-training source or job grid is invalid")
    source_identity = canonical_sha256(
        [
            {
                key: row[key]
                for key in ("kind", "relative_path", "sha256", "bytes")
            }
            for row in source_rows
        ]
    )
    execution_identity = canonical_sha256(
        {
            "run_name": run_name,
            "workload": workload,
            "source_identity_sha256": source_identity,
            "jobs": job_rows,
            "scientific_identity": scientific_identity,
        }
    )
    core = {
        "schema": CONTRACT_SCHEMA,
        "status": "local_smoke_required_cloud_not_authorized",
        "run_name": run_name,
        "bucket": bucket,
        "workload": workload,
        "project": PROJECT,
        "region": REGION,
        "zone": ZONE,
        "machine_type": MACHINE_TYPE,
        "vcpus_per_vm": VCPUS_PER_VM,
        "max_concurrent_vms": MAX_CONCURRENT_VMS,
        "max_attempts": MAX_ATTEMPTS,
        "sources": source_rows,
        "source_identity_sha256": source_identity,
        "jobs": job_rows,
        "job_count": len(job_rows),
        "scientific_identity": deepcopy(dict(scientific_identity)),
        "execution_identity_sha256": execution_identity,
        "atomic_job_restart_only": True,
        "create_only_result_manifest_last": True,
        "hidden_opponent_discard_available_to_worker": False,
        "synthetic_abr_allowed": False,
        "teacher_values_are_realized_match_ev": False,
        "current_profile_changed": False,
        "cloud_launch_authorized": False,
    }
    return {**core, "contract_sha256": canonical_sha256(core)}


def build_abr_contract(
    *,
    run_name: str,
    bucket: str,
    mode: str,
    abr_plan_path: str | Path,
    runtime_bundle_manifest_path: str | Path,
    runtime_archive_path: str | Path,
    wheelhouse_archive_path: str | Path,
    accepted_library_path: str | Path,
    diagnostic_library_path: str | Path,
) -> dict[str, Any]:
    """Freeze the exact 50-pair pilot or exact 250-pair production grid."""

    if mode not in {"pilot50", "production250"}:
        raise ValueError("ABR mode must be pilot50 or production250")
    plan_path = _regular_file(abr_plan_path, "ABR plan")
    plan = abr_teacher.validate_plan(_read_canonical(plan_path, "ABR plan"))
    expected_pairs = 50 if mode == "pilot50" else 250
    accepted = _regular_file(accepted_library_path, "accepted ABR library")
    diagnostic = _regular_file(diagnostic_library_path, "diagnostic ABR library")
    if (
        plan["pair_count"] != expected_pairs
        or plan["accepted_library_sha256"] != sha256_file(accepted)
        or plan["diagnostic_library_sha256"] != sha256_file(diagnostic)
        or plan["synthetic_values_allowed"] is not False
        or plan["opponent_private_discards_used"] is not False
    ):
        raise ValueError("ABR plan or dual native-library pins changed")
    resolved_bundle = runtime_bundle.resolve_bundle(
        runtime_bundle_manifest_path,
        expected_diagnostic_library_sha256=plan["diagnostic_library_sha256"],
    )
    runtime = _regular_file(runtime_archive_path, "runtime archive")
    wheels = _regular_file(wheelhouse_archive_path, "wheelhouse archive")
    bundle_manifest = _regular_file(
        resolved_bundle["manifest_path"], "ABR runtime bundle manifest"
    )
    bundle_ready = _regular_file(
        resolved_bundle["ready_path"], "ABR runtime bundle READY"
    )
    wheelhouse_source_manifest = _regular_file(
        resolved_bundle["wheelhouse_source_manifest_path"],
        "ABR runtime wheelhouse source manifest",
    )
    if (
        runtime != _regular_file(
            resolved_bundle["runtime_archive_path"],
            "bundled runtime archive",
        )
        or wheels != _regular_file(
            resolved_bundle["wheelhouse_archive_path"],
            "bundled wheelhouse archive",
        )
        or accepted
        != _regular_file(
            resolved_bundle["accepted_library_path"],
            "bundled accepted ABR library",
        )
        or diagnostic
        != _regular_file(
            resolved_bundle["diagnostic_library_path"],
            "bundled diagnostic ABR library",
        )
    ):
        raise ValueError("ABR contract inputs are not one runtime bundle")
    _validate_tar(runtime, "runtime")
    _validate_zip(wheels, "wheelhouse")
    sources = [
        _source(
            kind="abr_plan",
            path=plan_path,
            relative_path="static/abr_plan/PLAN.json",
        ),
        _source(
            kind="runtime_archive",
            path=runtime,
            relative_path="static/runtime/runtime.tar.gz",
        ),
        _source(
            kind="abr_runtime_bundle_manifest",
            path=bundle_manifest,
            relative_path=(
                "static/runtime_bundle/ABR_RUNTIME_BUNDLE_MANIFEST.json"
            ),
        ),
        _source(
            kind="abr_runtime_bundle_ready",
            path=bundle_ready,
            relative_path=(
                "static/runtime_bundle/ABR_RUNTIME_BUNDLE_READY.json"
            ),
        ),
        _source(
            kind="abr_runtime_wheelhouse_source_manifest",
            path=wheelhouse_source_manifest,
            relative_path=(
                "static/runtime_bundle/wheelhouse_source_manifest.json"
            ),
        ),
        _source(
            kind="wheelhouse_archive",
            path=wheels,
            relative_path="static/wheelhouse/wheelhouse.zip",
        ),
        _source(
            kind="accepted_library",
            path=accepted,
            relative_path=f"static/accepted_library/{accepted.name}",
        ),
        _source(
            kind="diagnostic_library",
            path=diagnostic,
            relative_path=f"static/diagnostic_library/{diagnostic.name}",
        ),
    ]
    jobs = [
        {
            "ordinal": index,
            "job_id": f"abr-pair-{index:04d}",
            "pair_index": index,
            "output_relative_path": f"pairs/pair-{index:06d}.json",
        }
        for index in range(expected_pairs)
    ]
    return _base_contract(
        run_name=run_name,
        bucket=bucket,
        workload=f"abr_{mode}",
        sources=sources,
        jobs=jobs,
        scientific_identity={
            "abr_plan_sha256": canonical_sha256(plan),
            "pair_count": expected_pairs,
            "state_count": expected_pairs * 2,
            "accepted_library_sha256": plan["accepted_library_sha256"],
            "diagnostic_library_sha256": plan["diagnostic_library_sha256"],
            "abr_runtime_bundle_required": True,
            "abr_runtime_bundle_manifest_sha256": resolved_bundle["manifest"][
                "manifest_sha256"
            ],
            "abr_runtime_bundle_manifest_file_sha256": sha256_file(
                bundle_manifest
            ),
            "abr_runtime_bundle_ready_sha256": resolved_bundle["ready"][
                "ready_sha256"
            ],
            "abr_runtime_source_inventory_sha256": resolved_bundle["manifest"][
                "source_inventory_sha256"
            ],
            "abr_runtime_model_inventory_sha256": resolved_bundle["manifest"][
                "model_inventory_sha256"
            ],
            "abr_runtime_wheelhouse_entries_sha256": resolved_bundle[
                "manifest"
            ]["wheelhouse"]["entries_sha256"],
            "real_behavior_model_count": len(
                resolved_bundle["manifest"]["model_inventory"]
            ),
            "real_behavior_models_deserialized": True,
            "actor_observation_only": True,
            "synthetic_values_allowed": False,
        },
    )


def build_locked_promotion_contract(
    *,
    run_name: str,
    bucket: str,
    promotion_plan_path: str | Path,
    execution_plan_path: str | Path,
    runtime_archive_path: str | Path,
    wheelhouse_archive_path: str | Path,
    closure_package_path: str | Path,
    compatibility_threshold_lock_path: str | Path,
    policy_registry_path: str | Path,
    abr_bundle_archive_path: str | Path,
    expected_closure_sha256: str,
    expected_abr_bundle_archive_sha256: str,
    expected_abr_bundle_file_sha256: str,
    expected_abr_production_build_receipt_sha256: str,
) -> dict[str, Any]:
    """Freeze all 260 real population/ABR work items and all runtime pins."""

    plan_path = _regular_file(promotion_plan_path, "promotion plan")
    execution_path = _regular_file(execution_plan_path, "execution plan")
    plan = promotion.validate_locked_promotion_plan(
        _read_canonical(plan_path, "promotion plan")
    )
    execution = promotion_execution.validate_execution_plan(
        _read_canonical(execution_path, "execution plan"),
        promotion_plan=plan,
    )
    if (
        len(execution["work_items"]) != promotion_execution.EXPECTED_WORK_ITEMS
        or promotion_execution.EXPECTED_WORK_ITEMS != 260
        or promotion_execution.EXPECTED_ROWS != 13_000
        or _SHA.fullmatch(expected_closure_sha256) is None
        or _SHA.fullmatch(expected_abr_bundle_archive_sha256) is None
        or _SHA.fullmatch(expected_abr_bundle_file_sha256) is None
        or _SHA.fullmatch(expected_abr_production_build_receipt_sha256) is None
    ):
        raise ValueError("locked-promotion execution cardinality or pins changed")
    runtime = _regular_file(runtime_archive_path, "runtime archive")
    wheels = _regular_file(wheelhouse_archive_path, "wheelhouse archive")
    closure = _regular_file(closure_package_path, "runtime closure package")
    threshold = _regular_file(
        compatibility_threshold_lock_path, "compatibility threshold lock"
    )
    registry = _regular_file(policy_registry_path, "policy registry")
    abr_archive = _regular_file(abr_bundle_archive_path, "ABR bundle archive")
    if (
        sha256_file(closure) != expected_closure_sha256
        or sha256_file(abr_archive) != expected_abr_bundle_archive_sha256
    ):
        raise ValueError("runtime closure or ABR bundle archive changed")
    _validate_tar(runtime, "runtime")
    _validate_zip(wheels, "wheelhouse")
    _validate_tar(abr_archive, "abr_bundle")
    bundle_raw = _tar_member_bytes(abr_archive, "abr_bundle/bundle.json")
    receipt_raw = _tar_member_bytes(
        abr_archive,
        f"abr_bundle/{abr_cli.ABR_PRODUCTION_BUILD_RECEIPT_FILE}",
    )
    if (
        hashlib.sha256(bundle_raw).hexdigest()
        != expected_abr_bundle_file_sha256
        or hashlib.sha256(receipt_raw).hexdigest()
        != expected_abr_production_build_receipt_sha256
    ):
        raise ValueError(
            "ABR archive bundle.json or production receipt pin changed"
        )
    sources = [
        _source(
            kind="promotion_plan",
            path=plan_path,
            relative_path="static/promotion_plan/plan.json",
        ),
        _source(
            kind="execution_plan",
            path=execution_path,
            relative_path="static/execution_plan/plan.json",
        ),
        _source(
            kind="runtime_archive",
            path=runtime,
            relative_path="static/runtime/runtime.tar.gz",
        ),
        _source(
            kind="wheelhouse_archive",
            path=wheels,
            relative_path="static/wheelhouse/wheelhouse.zip",
        ),
        _source(
            kind="closure_package",
            path=closure,
            relative_path=f"static/closure_package/{closure.name}",
        ),
        _source(
            kind="compatibility_threshold_lock",
            path=threshold,
            relative_path=f"static/threshold_lock/{threshold.name}",
        ),
        _source(
            kind="policy_registry",
            path=registry,
            relative_path=f"static/policy_registry/{registry.name}",
        ),
        _source(
            kind="abr_bundle_archive",
            path=abr_archive,
            relative_path="static/abr_bundle/abr_bundle.tar.gz",
        ),
    ]
    jobs = [
        {
            "ordinal": int(item["ordinal"]),
            "job_id": f"locked-work-{int(item['ordinal']):04d}",
            "work_id": str(item["work_id"]),
            "output_filename": str(item["output_filename"]),
            "row_count": int(item["seed_index_stop_exclusive"])
            - int(item["seed_index_start"]),
        }
        for item in execution["work_items"]
    ]
    return _base_contract(
        run_name=run_name,
        bucket=bucket,
        workload="locked_promotion_260",
        sources=sources,
        jobs=jobs,
        scientific_identity={
            "promotion_plan_sha256": canonical_sha256(plan),
            "execution_plan_sha256": canonical_sha256(execution),
            "closure_package_sha256": expected_closure_sha256,
            "threshold_lock_file_sha256": sha256_file(threshold),
            "policy_registry_file_sha256": sha256_file(registry),
            "abr_bundle_archive_sha256": expected_abr_bundle_archive_sha256,
            "abr_bundle_file_sha256": expected_abr_bundle_file_sha256,
            "abr_production_build_receipt_file_sha256": (
                expected_abr_production_build_receipt_sha256
            ),
            "work_item_count": 260,
            "row_count": 13_000,
            "real_population_and_abr_only": True,
        },
    )


def validate_contract(
    value: Mapping[str, Any], *, replay_sources: bool
) -> dict[str, Any]:
    contract = deepcopy(dict(value))
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("contract_sha256")
        != _self_digest(contract, "contract_sha256")
        or contract.get("project") != PROJECT
        or contract.get("region") != REGION
        or contract.get("zone") != ZONE
        or contract.get("machine_type") != MACHINE_TYPE
        or contract.get("vcpus_per_vm") != VCPUS_PER_VM
        or contract.get("max_concurrent_vms") != MAX_CONCURRENT_VMS
        or contract.get("max_attempts") != MAX_ATTEMPTS
        or contract.get("cloud_launch_authorized") is not False
        or contract.get("current_profile_changed") is not False
        or contract.get("hidden_opponent_discard_available_to_worker") is not False
        or contract.get("synthetic_abr_allowed") is not False
    ):
        raise ValueError("post-training workload contract changed")
    sources = contract.get("sources")
    jobs = contract.get("jobs")
    if (
        not isinstance(sources, list)
        or not isinstance(jobs, list)
        or len(jobs) != contract.get("job_count")
        or len({row.get("kind") for row in sources}) != len(sources)
        or len({row.get("job_id") for row in jobs}) != len(jobs)
        or [row.get("ordinal") for row in jobs] != list(range(len(jobs)))
    ):
        raise ValueError("post-training source/job grid changed")
    expected_source_identity = canonical_sha256(
        [
            {
                key: row[key]
                for key in ("kind", "relative_path", "sha256", "bytes")
            }
            for row in sources
        ]
    )
    if contract.get("source_identity_sha256") != expected_source_identity:
        raise ValueError("post-training source identity changed")
    if replay_sources:
        for row in sources:
            path = _regular_file(row["source_path"], str(row["kind"]))
            if (
                _safe_relative(row["relative_path"], "source relative path")
                != row["relative_path"]
                or sha256_file(path) != row["sha256"]
                or path.stat().st_size != row["bytes"]
            ):
                raise ValueError("post-training immutable source changed")
    workload = contract.get("workload")
    if workload in {"abr_pilot50", "abr_production250"}:
        expected = 50 if workload == "abr_pilot50" else 250
        sources_by_kind = {
            str(row.get("kind")): row
            for row in sources
        }
        expected_source_kinds = {
            "abr_plan",
            "runtime_archive",
            "abr_runtime_bundle_manifest",
            "abr_runtime_bundle_ready",
            "abr_runtime_wheelhouse_source_manifest",
            "wheelhouse_archive",
            "accepted_library",
            "diagnostic_library",
        }
        science = contract.get("scientific_identity")
        if len(jobs) != expected or any(
            row != {
                "ordinal": index,
                "job_id": f"abr-pair-{index:04d}",
                "pair_index": index,
                "output_relative_path": f"pairs/pair-{index:06d}.json",
            }
            for index, row in enumerate(jobs)
        ) or (
            set(sources_by_kind) != expected_source_kinds
            or not isinstance(science, Mapping)
            or science.get("abr_runtime_bundle_required") is not True
            or science.get("real_behavior_model_count") != 11
            or science.get("real_behavior_models_deserialized") is not True
            or science.get("synthetic_values_allowed") is not False
            or science.get("actor_observation_only") is not True
            or _SHA.fullmatch(
                str(science.get("abr_runtime_bundle_manifest_sha256", ""))
            )
            is None
            or _SHA.fullmatch(
                str(
                    science.get(
                        "abr_runtime_bundle_manifest_file_sha256", ""
                    )
                )
            )
            is None
            or _SHA.fullmatch(
                str(science.get("abr_runtime_bundle_ready_sha256", ""))
            )
            is None
            or _SHA.fullmatch(
                str(science.get("abr_runtime_source_inventory_sha256", ""))
            )
            is None
            or _SHA.fullmatch(
                str(science.get("abr_runtime_model_inventory_sha256", ""))
            )
            is None
            or _SHA.fullmatch(
                str(
                    science.get(
                        "abr_runtime_wheelhouse_entries_sha256", ""
                    )
                )
            )
            is None
            or sources_by_kind["abr_runtime_bundle_manifest"]["sha256"]
            != science["abr_runtime_bundle_manifest_file_sha256"]
            or sources_by_kind["accepted_library"]["sha256"]
            != science.get("accepted_library_sha256")
            or sources_by_kind["diagnostic_library"]["sha256"]
            != science.get("diagnostic_library_sha256")
        ):
            raise ValueError("ABR atomic pair grid changed")
        if replay_sources:
            resolved = runtime_bundle.resolve_bundle(
                sources_by_kind["abr_runtime_bundle_manifest"]["source_path"],
                expected_diagnostic_library_sha256=str(
                    science["diagnostic_library_sha256"]
                ),
            )
            exact_paths = {
                "runtime_archive": resolved["runtime_archive_path"],
                "wheelhouse_archive": resolved["wheelhouse_archive_path"],
                "accepted_library": resolved["accepted_library_path"],
                "diagnostic_library": resolved["diagnostic_library_path"],
                "abr_runtime_bundle_manifest": resolved["manifest_path"],
                "abr_runtime_bundle_ready": resolved["ready_path"],
                "abr_runtime_wheelhouse_source_manifest": resolved[
                    "wheelhouse_source_manifest_path"
                ],
            }
            if any(
                _regular_file(
                    sources_by_kind[kind]["source_path"], kind
                )
                != _regular_file(path, f"resolved {kind}")
                for kind, path in exact_paths.items()
            ) or (
                resolved["manifest"]["manifest_sha256"]
                != science["abr_runtime_bundle_manifest_sha256"]
                or resolved["ready"]["ready_sha256"]
                != science["abr_runtime_bundle_ready_sha256"]
                or resolved["manifest"]["source_inventory_sha256"]
                != science["abr_runtime_source_inventory_sha256"]
                or resolved["manifest"]["model_inventory_sha256"]
                != science["abr_runtime_model_inventory_sha256"]
                or resolved["manifest"]["wheelhouse"]["entries_sha256"]
                != science["abr_runtime_wheelhouse_entries_sha256"]
            ):
                raise ValueError("ABR runtime bundle replay changed")
    elif workload == "locked_promotion_260":
        if (
            len(jobs) != 260
            or sum(int(row["row_count"]) for row in jobs) != 13_000
        ):
            raise ValueError("locked-promotion work grid changed")
    else:
        raise ValueError("unknown post-training workload")
    return contract


def _sources_by_kind(contract: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["kind"]): row for row in contract["sources"]}


def _extract_sources(staging_root: Path) -> None:
    _validate_tar(staging_root / "static/runtime/runtime.tar.gz", "runtime")
    _validate_zip(staging_root / "static/wheelhouse/wheelhouse.zip", "wheelhouse")
    with tarfile.open(staging_root / "static/runtime/runtime.tar.gz", "r:*") as archive:
        archive.extractall(staging_root)
    with zipfile.ZipFile(staging_root / "static/wheelhouse/wheelhouse.zip") as archive:
        archive.extractall(staging_root)
    abr_archive = staging_root / "static/abr_bundle/abr_bundle.tar.gz"
    if abr_archive.exists():
        _validate_tar(abr_archive, "abr_bundle")
        with tarfile.open(abr_archive, "r:*") as archive:
            archive.extractall(staging_root)


def stage_local_sources(
    contract: Mapping[str, Any], staging_root: str | Path
) -> Path:
    """Replay exact bytes into an empty local staging root and safe-extract."""

    checked = validate_contract(contract, replay_sources=True)
    root = Path(staging_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("local smoke staging root must be empty")
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise ValueError("local smoke staging root is unsafe")
    for row in checked["sources"]:
        target = root.joinpath(*PurePosixPath(row["relative_path"]).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with _regular_file(row["source_path"], row["kind"]).open("rb") as source:
            with target.open("xb") as output:
                shutil.copyfileobj(source, output)
                output.flush()
                os.fsync(output.fileno())
        if sha256_file(target) != row["sha256"]:
            raise ValueError("local source replay changed bytes")
    _extract_sources(root)
    return root


def _job(contract: Mapping[str, Any], job_id: str) -> dict[str, Any]:
    matches = [row for row in contract["jobs"] if row["job_id"] == job_id]
    if len(matches) != 1:
        raise ValueError("post-training job ID is unknown")
    return deepcopy(dict(matches[0]))


def run_worker_job(
    *,
    contract: Mapping[str, Any],
    job_id: str,
    staging_root: str | Path,
    output_root: str | Path,
    require_host_target: bool = True,
) -> dict[str, Any]:
    """Execute exactly one real ABR pair or locked work item."""

    checked = validate_contract(contract, replay_sources=False)
    job = _job(checked, job_id)
    staging = Path(staging_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink() or any(output.iterdir()):
        raise FileExistsError("worker output root must be a new empty directory")
    runtime = staging / "runtime"
    if not runtime.is_dir() or runtime.is_symlink():
        raise ValueError("worker runtime was not safely extracted")
    previous = Path.cwd()
    try:
        os.chdir(runtime)
        if checked["workload"].startswith("abr_"):
            run = output / "abr_run"
            (run / abr_teacher.PAIR_DIRECTORY).mkdir(parents=True)
            shutil.copyfile(staging / "static/abr_plan/PLAN.json", run / abr_teacher.PLAN_FILE)
            sources = _sources_by_kind(checked)
            result = abr_teacher.run_pairs(
                run_directory=run,
                accepted_library_path=staging.joinpath(
                    *PurePosixPath(sources["accepted_library"]["relative_path"]).parts
                ),
                diagnostic_library_path=staging.joinpath(
                    *PurePosixPath(sources["diagnostic_library"]["relative_path"]).parts
                ),
                max_new_pairs=1,
                pair_indices=[int(job["pair_index"])],
            )
            pair_path = abr_teacher.pair_evidence_path(run, int(job["pair_index"]))
            evidence = abr_teacher.validate_pair_evidence(
                _read_canonical(pair_path, "ABR pair evidence"),
                plan=abr_teacher.validate_plan(
                    _read_canonical(run / abr_teacher.PLAN_FILE, "ABR plan")
                ),
                pair_index=int(job["pair_index"]),
            )
            destination = output.joinpath(
                *PurePosixPath(job["output_relative_path"]).parts
            )
            destination.parent.mkdir(parents=True)
            _write_once(destination, evidence)
            files = [destination]
            scientific = {
                "new_pairs": result["new_pairs"],
                "synthetic_values_used": False,
                "opponent_private_discards_used": False,
            }
        else:
            sources = _sources_by_kind(checked)
            import torch

            receipt = promotion_provider.run_provider_work_item(
                plan_path=staging.joinpath(
                    *PurePosixPath(sources["promotion_plan"]["relative_path"]).parts
                ),
                closure_package_path=staging.joinpath(
                    *PurePosixPath(sources["closure_package"]["relative_path"]).parts
                ),
                expected_closure_package_sha256=checked["scientific_identity"][
                    "closure_package_sha256"
                ],
                extraction_root=output / "closure",
                source_replay_root=runtime,
                compatibility_threshold_lock_path=staging.joinpath(
                    *PurePosixPath(
                        sources["compatibility_threshold_lock"]["relative_path"]
                    ).parts
                ),
                policy_registry_path=staging.joinpath(
                    *PurePosixPath(sources["policy_registry"]["relative_path"]).parts
                ),
                abr_bundle_directory=staging / "abr_bundle",
                expected_abr_bundle_file_sha256=checked["scientific_identity"][
                    "abr_bundle_file_sha256"
                ],
                expected_abr_production_build_receipt_sha256=checked[
                    "scientific_identity"
                ]["abr_production_build_receipt_file_sha256"],
                execution_plan_path=staging.joinpath(
                    *PurePosixPath(sources["execution_plan"]["relative_path"]).parts
                ),
                work_id=str(job["work_id"]),
                shard_directory=output / "shards",
                torch=torch,
                require_host_target=require_host_target,
            )
            receipt_path = _write_once(output / "provider_receipt.json", receipt)
            shard_path = output / "shards" / str(job["output_filename"])
            if not shard_path.is_file() or shard_path.is_symlink():
                raise ValueError("locked-promotion worker did not publish its shard")
            files = [shard_path, receipt_path]
            scientific = {
                "row_count": job["row_count"],
                "real_population_and_abr_only": True,
            }
    finally:
        os.chdir(previous)
    records = [
        {
            "relative_path": path.relative_to(output).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in files
    ]
    core = {
        "schema": RESULT_MANIFEST_SCHEMA,
        "contract_sha256": checked["contract_sha256"],
        "job_id": job_id,
        "attempt_id": "local",
        "files": records,
        "file_count": len(records),
        "scientific": scientific,
        "manifest_published_after_files": True,
        "create_only": True,
        "hidden_opponent_discard_used": False,
        "synthetic_abr_used": False,
        "current_profile_changed": False,
    }
    return {**core, "manifest_sha256": canonical_sha256(core)}


def validate_result_manifest(
    value: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    job_id: str,
    attempt_id: str,
) -> dict[str, Any]:
    checked = validate_contract(contract, replay_sources=False)
    job = _job(checked, job_id)
    result = deepcopy(dict(value))
    files = result.get("files")
    if (
        result.get("schema") != RESULT_MANIFEST_SCHEMA
        or result.get("manifest_sha256")
        != _self_digest(result, "manifest_sha256")
        or result.get("contract_sha256") != checked["contract_sha256"]
        or result.get("job_id") != job_id
        or result.get("attempt_id") != attempt_id
        or not isinstance(files, list)
        or result.get("file_count") != len(files)
        or result.get("manifest_published_after_files") is not True
        or result.get("create_only") is not True
        or result.get("hidden_opponent_discard_used") is not False
        or result.get("synthetic_abr_used") is not False
        or result.get("current_profile_changed") is not False
    ):
        raise ValueError("post-training result manifest changed")
    expected_paths = (
        [job["output_relative_path"]]
        if checked["workload"].startswith("abr_")
        else [f"shards/{job['output_filename']}", "provider_receipt.json"]
    )
    if [row.get("relative_path") for row in files] != expected_paths:
        raise ValueError("post-training result file layout changed")
    for row in files:
        if (
            _safe_relative(row["relative_path"], "result path")
            != row["relative_path"]
            or _SHA.fullmatch(str(row.get("sha256"))) is None
            or not isinstance(row.get("bytes"), int)
            or isinstance(row.get("bytes"), bool)
            or row["bytes"] <= 0
        ):
            raise ValueError("post-training result file binding changed")
    return result


def build_local_smoke_receipt(
    *,
    contract: Mapping[str, Any],
    output_root: str | Path,
    manifest: Mapping[str, Any],
    filesystem_type: str,
    production_executor: bool,
) -> dict[str, Any]:
    """Validate job zero output; only a real ext4/xfs run can unlock Spot."""

    checked = validate_contract(contract, replay_sources=True)
    first = checked["jobs"][0]
    validated = validate_result_manifest(
        manifest,
        contract=checked,
        job_id=first["job_id"],
        attempt_id="local",
    )
    root = Path(output_root).resolve()
    for record in validated["files"]:
        path = root.joinpath(*PurePosixPath(record["relative_path"]).parts)
        if (
            path.is_symlink()
            or not path.is_file()
            or sha256_file(path) != record["sha256"]
            or path.stat().st_size != record["bytes"]
        ):
            raise ValueError("local smoke output bytes changed")
    job = _job(checked, first["job_id"])
    if checked["workload"].startswith("abr_"):
        abr_plan = abr_teacher.validate_plan(
            _read_canonical(
                _sources_by_kind(checked)["abr_plan"]["source_path"],
                "ABR plan",
            )
        )
        abr_teacher.validate_pair_evidence(
            _read_canonical(
                root.joinpath(
                    *PurePosixPath(first["output_relative_path"]).parts
                ),
                "local smoke ABR pair",
            ),
            plan=abr_plan,
            pair_index=int(job["pair_index"]),
        )
    else:
        sources = _sources_by_kind(checked)
        promotion_plan = promotion.validate_locked_promotion_plan(
            _read_canonical(
                sources["promotion_plan"]["source_path"], "promotion plan"
            )
        )
        execution_plan = promotion_execution.validate_execution_plan(
            _read_canonical(
                sources["execution_plan"]["source_path"], "execution plan"
            ),
            promotion_plan=promotion_plan,
        )
        item = next(
            item
            for item in execution_plan["work_items"]
            if item["work_id"] == job["work_id"]
        )
        promotion_execution.validate_work_item_shard(
            work_item=item,
            shard=_read_canonical(
                root / "shards" / str(job["output_filename"]),
                "local smoke promotion shard",
            ),
            promotion_plan=promotion_plan,
        )
    ext4 = filesystem_type in {"ext4", "xfs"}
    core = {
        "schema": SMOKE_SCHEMA,
        "status": (
            "pass_cloud_authorization_eligible"
            if production_executor and ext4
            else "diagnostic_only"
        ),
        "contract_sha256": checked["contract_sha256"],
        "job_id": first["job_id"],
        "manifest_sha256": validated["manifest_sha256"],
        "output_records": validated["files"],
        "filesystem_type": filesystem_type,
        "ext4_or_xfs_staging": ext4,
        "production_executor": production_executor,
        "one_real_atomic_job_completed": True,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return {**core, "smoke_sha256": canonical_sha256(core)}


def validate_local_smoke(
    value: Mapping[str, Any], *, contract: Mapping[str, Any]
) -> dict[str, Any]:
    checked = validate_contract(contract, replay_sources=True)
    smoke = deepcopy(dict(value))
    if (
        smoke.get("schema") != SMOKE_SCHEMA
        or smoke.get("smoke_sha256") != _self_digest(smoke, "smoke_sha256")
        or smoke.get("status") != "pass_cloud_authorization_eligible"
        or smoke.get("contract_sha256") != checked["contract_sha256"]
        or smoke.get("job_id") != checked["jobs"][0]["job_id"]
        or smoke.get("ext4_or_xfs_staging") is not True
        or smoke.get("production_executor") is not True
        or smoke.get("one_real_atomic_job_completed") is not True
        or smoke.get("cloud_mutated") is not False
        or smoke.get("current_profile_changed") is not False
    ):
        raise PermissionError("one-real-job ext4 local smoke has not passed")
    return smoke


def build_cloud_plan(
    *,
    contract_path: str | Path,
    smoke_receipt_path: str | Path,
    image_self_link: str,
    image_id: str,
    guest_os_features: Sequence[str],
    worker_service_accounts: Sequence[str],
) -> dict[str, Any]:
    """Authorize cloud fanout only after a source-replayed real local smoke."""

    contract_file = _regular_file(contract_path, "workload contract")
    contract = validate_contract(
        _read_canonical(contract_file, "workload contract"),
        replay_sources=True,
    )
    smoke_file = _regular_file(smoke_receipt_path, "local smoke receipt")
    smoke = validate_local_smoke(
        _read_canonical(smoke_file, "local smoke receipt"),
        contract=contract,
    )
    accounts = list(worker_service_accounts)
    features = sorted(set(guest_os_features))
    if (
        len(accounts) != MAX_CONCURRENT_VMS
        or len(set(accounts)) != len(accounts)
        or any(_SERVICE_ACCOUNT.fullmatch(value) is None for value in accounts)
        or not features
        or len(features) != len(guest_os_features)
        or _IMAGE_LINK.fullmatch(image_self_link) is None
        or _PROVIDER_ID.fullmatch(str(image_id)) is None
    ):
        raise ValueError("post-training provider identity is incomplete")
    static_sources = [
        {
            **deepcopy(dict(row)),
            "object_name": (
                f"m31-post/{contract['execution_identity_sha256']}/content/"
                f"{contract['source_identity_sha256'][:20]}/{row['relative_path']}"
            ),
        }
        for row in contract["sources"]
    ]
    contract_source = _source(
        kind="workload_contract",
        path=contract_file,
        relative_path="contract.json",
    )
    smoke_source = _source(
        kind="local_smoke_receipt",
        path=smoke_file,
        relative_path="smoke.json",
    )
    for row in (contract_source, smoke_source):
        row["object_name"] = (
            f"m31-post/{contract['execution_identity_sha256']}/content/"
            f"{contract['source_identity_sha256'][:20]}/{row['relative_path']}"
        )
    waves = [
        {
            "wave_index": index // MAX_CONCURRENT_VMS,
            "job_ids": [
                row["job_id"]
                for row in contract["jobs"][index : index + MAX_CONCURRENT_VMS]
            ],
        }
        for index in range(0, len(contract["jobs"]), MAX_CONCURRENT_VMS)
    ]
    core = {
        "schema": CLOUD_PLAN_SCHEMA,
        "status": "authorized_cloud_not_started",
        "contract": contract,
        "contract_file_sha256": sha256_file(contract_file),
        "smoke_receipt": smoke,
        "smoke_file_sha256": sha256_file(smoke_file),
        "project": PROJECT,
        "region": REGION,
        "zone": ZONE,
        "bucket": contract["bucket"],
        "image": {
            "self_link": image_self_link,
            "id": str(image_id),
            "guest_os_features": features,
        },
        "worker_service_accounts": accounts,
        "content_entries": [*static_sources, contract_source, smoke_source],
        "waves": waves,
        "wave_count": len(waves),
        "max_concurrent_vms": MAX_CONCURRENT_VMS,
        "machine_type": MACHINE_TYPE,
        "oauth_min_remaining_seconds": MIN_OAUTH_TTL_SECONDS,
        "one_wave_at_a_time": True,
        "exact_vm_disk_cleanup_required": True,
        "create_only_gcs_results": True,
        "cloud_launch_authorized": True,
        "current_profile_changed": False,
    }
    return {**core, "cloud_plan_sha256": canonical_sha256(core)}


def validate_cloud_plan(
    value: Mapping[str, Any], *, replay_sources: bool
) -> dict[str, Any]:
    plan = deepcopy(dict(value))
    if (
        plan.get("schema") != CLOUD_PLAN_SCHEMA
        or plan.get("cloud_plan_sha256")
        != _self_digest(plan, "cloud_plan_sha256")
        or plan.get("project") != PROJECT
        or plan.get("region") != REGION
        or plan.get("zone") != ZONE
        or plan.get("machine_type") != MACHINE_TYPE
        or plan.get("max_concurrent_vms") != MAX_CONCURRENT_VMS
        or plan.get("oauth_min_remaining_seconds") != MIN_OAUTH_TTL_SECONDS
        or plan.get("cloud_launch_authorized") is not True
        or plan.get("current_profile_changed") is not False
    ):
        raise ValueError("post-training cloud plan changed")
    contract = validate_contract(plan.get("contract", {}), replay_sources=replay_sources)
    validate_local_smoke(plan.get("smoke_receipt", {}), contract=contract)
    accounts = plan.get("worker_service_accounts")
    entries = plan.get("content_entries")
    image = plan.get("image")
    expected_waves = [
        {
            "wave_index": index // MAX_CONCURRENT_VMS,
            "job_ids": [
                row["job_id"]
                for row in contract["jobs"][index : index + MAX_CONCURRENT_VMS]
            ],
        }
        for index in range(0, len(contract["jobs"]), MAX_CONCURRENT_VMS)
    ]
    if (
        plan.get("bucket") != contract["bucket"]
        or plan.get("waves") != expected_waves
        or plan.get("wave_count") != len(expected_waves)
        or not isinstance(accounts, list)
        or len(accounts) != MAX_CONCURRENT_VMS
        or len(set(accounts)) != len(accounts)
        or any(_SERVICE_ACCOUNT.fullmatch(str(value)) is None for value in accounts)
        or not isinstance(image, Mapping)
        or _IMAGE_LINK.fullmatch(str(image.get("self_link", ""))) is None
        or _PROVIDER_ID.fullmatch(str(image.get("id", ""))) is None
        or not isinstance(image.get("guest_os_features"), list)
        or not image["guest_os_features"]
        or image["guest_os_features"]
        != sorted(set(image["guest_os_features"]))
        or not isinstance(entries, list)
        or len(entries) != len(contract["sources"]) + 2
        or len({row.get("kind") for row in entries}) != len(entries)
        or len({row.get("object_name") for row in entries}) != len(entries)
    ):
        raise ValueError("post-training cloud wave grid changed")
    by_kind = {row["kind"]: row for row in entries}
    prefix = (
        f"m31-post/{contract['execution_identity_sha256']}/content/"
        f"{contract['source_identity_sha256'][:20]}/"
    )
    for source in contract["sources"]:
        row = by_kind.get(source["kind"])
        if (
            not isinstance(row, Mapping)
            or {
                key: row.get(key)
                for key in (
                    "kind",
                    "source_path",
                    "relative_path",
                    "sha256",
                    "bytes",
                )
            }
            != source
            or row.get("object_name") != prefix + source["relative_path"]
        ):
            raise ValueError("post-training cloud content binding changed")
    contract_entry = by_kind.get("workload_contract", {})
    smoke_entry = by_kind.get("local_smoke_receipt", {})
    if (
        contract_entry.get("relative_path") != "contract.json"
        or contract_entry.get("sha256")
        != hashlib.sha256(canonical_bytes(contract)).hexdigest()
        or contract_entry.get("sha256") != plan.get("contract_file_sha256")
        or contract_entry.get("object_name") != prefix + "contract.json"
        or smoke_entry.get("relative_path") != "smoke.json"
        or smoke_entry.get("sha256")
        != hashlib.sha256(canonical_bytes(plan["smoke_receipt"])).hexdigest()
        or smoke_entry.get("sha256") != plan.get("smoke_file_sha256")
        or smoke_entry.get("object_name") != prefix + "smoke.json"
    ):
        raise ValueError("post-training contract/smoke content binding changed")
    if replay_sources:
        for row in plan["content_entries"]:
            path = _regular_file(row["source_path"], row["kind"])
            if sha256_file(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
                raise ValueError("post-training staged source changed")
    return plan


def _attempt_id(index: int) -> str:
    return f"a{index:02d}"


def _instance_name(plan: Mapping[str, Any], ordinal: int, attempt: int) -> str:
    return (
        f"m31pt-{plan['contract']['execution_identity_sha256'][:10]}-"
        f"j{ordinal:03d}-a{attempt:02d}"
    )


def build_wave(
    plan: Mapping[str, Any],
    *,
    accepted_lifecycles: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Select the earliest pending jobs and the next distinct attempts."""

    checked = validate_cloud_plan(plan, replay_sources=True)
    completed: set[str] = set()
    attempts: dict[str, set[str]] = {}
    for raw in accepted_lifecycles:
        lifecycle = validate_lifecycle(raw, cloud_plan=checked)
        for row in lifecycle["rows"]:
            attempts.setdefault(row["job_id"], set()).add(row["attempt_id"])
            if row["status"] == "complete":
                completed.add(row["job_id"])
    selected = []
    exhausted = []
    for job in checked["contract"]["jobs"]:
        job_id = job["job_id"]
        if job_id in completed:
            continue
        used = attempts.get(job_id, set())
        next_attempt = next(
            (
                index
                for index in range(MAX_ATTEMPTS)
                if _attempt_id(index) not in used
            ),
            None,
        )
        if next_attempt is None:
            exhausted.append(job_id)
            continue
        if len(selected) < MAX_CONCURRENT_VMS:
            selected.append(
                {
                    "job_id": job_id,
                    "ordinal": job["ordinal"],
                    "attempt_id": _attempt_id(next_attempt),
                    "instance_name": _instance_name(
                        checked, int(job["ordinal"]), next_attempt
                    ),
                    "result_prefix": (
                        f"m31-post/{checked['contract']['execution_identity_sha256']}/"
                        f"results/{job_id}/{_attempt_id(next_attempt)}"
                    ),
                }
            )
    if exhausted:
        raise PermissionError(f"post-training jobs exhausted retries: {exhausted}")
    if not selected:
        return {
            "status": "complete",
            "completed_job_count": len(completed),
            "job_count": checked["contract"]["job_count"],
            "cloud_mutated": False,
            "current_profile_changed": False,
        }
    wave_index = len(accepted_lifecycles)
    core = {
        "schema": WAVE_SCHEMA,
        "status": "planned_cloud_not_started",
        "cloud_plan_sha256": checked["cloud_plan_sha256"],
        "wave_index": wave_index,
        "selected": selected,
        "selected_count": len(selected),
        "completed_before_count": len(completed),
        "accepted_lifecycle_count": len(accepted_lifecycles),
        "max_concurrent_vms": MAX_CONCURRENT_VMS,
        "machine_type": MACHINE_TYPE,
        "one_wave_at_a_time": True,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return {**core, "wave_sha256": canonical_sha256(core)}


def validate_wave(
    value: Mapping[str, Any], *, cloud_plan: Mapping[str, Any]
) -> dict[str, Any]:
    plan = validate_cloud_plan(cloud_plan, replay_sources=True)
    wave = deepcopy(dict(value))
    selected = wave.get("selected")
    if (
        wave.get("schema") != WAVE_SCHEMA
        or wave.get("wave_sha256") != _self_digest(wave, "wave_sha256")
        or wave.get("cloud_plan_sha256") != plan["cloud_plan_sha256"]
        or not isinstance(selected, list)
        or not 1 <= len(selected) <= MAX_CONCURRENT_VMS
        or wave.get("selected_count") != len(selected)
        or len({row.get("job_id") for row in selected}) != len(selected)
        or len({row.get("instance_name") for row in selected}) != len(selected)
        or wave.get("current_profile_changed") is not False
    ):
        raise ValueError("post-training wave changed")
    return wave


def expected_phase_sentinel(
    plan: Mapping[str, Any], wave: Mapping[str, Any], phase: str
) -> str:
    checked = validate_cloud_plan(plan, replay_sources=False)
    frozen = validate_wave(wave, cloud_plan=checked)
    if phase not in {"execute", "poll", "cleanup", "cleanup-incomplete", "receive"}:
        raise ValueError("unknown post-training cloud phase")
    return (
        f"m31-post:{phase}:{checked['contract']['run_name']}:"
        f"{frozen['wave_sha256']}"
    )


def _stage(
    plan: Mapping[str, Any], transport: CloudTransport
) -> dict[str, Any]:
    records = []
    mutated = False
    for row in plan["content_entries"]:
        path = _regular_file(row["source_path"], row["kind"])
        payload = path.read_bytes()
        metadata = transport.get_object_metadata(
            bucket=plan["bucket"], object_name=row["object_name"]
        )
        created = metadata is None
        if metadata is None:
            metadata = transport.put_object_new(
                bucket=plan["bucket"],
                object_name=row["object_name"],
                payload=payload,
                content_type="application/octet-stream",
            )
            mutated = True
        generation = str(metadata.get("generation", ""))
        observed = transport.get_object_bytes(
            bucket=plan["bucket"],
            object_name=row["object_name"],
            generation=generation,
        )
        if (
            observed != payload
            or metadata.get("name") != row["object_name"]
            or _PROVIDER_ID.fullmatch(generation) is None
        ):
            raise PostTrainingCloudError("staged post-training content changed")
        records.append(
            {
                "kind": row["kind"],
                "relative_path": row["relative_path"],
                "object_name": row["object_name"],
                "generation": generation,
                "sha256": row["sha256"],
                "bytes": row["bytes"],
                "created": created,
            }
        )
    core = {
        "schema": STAGE_SCHEMA,
        "cloud_plan_sha256": plan["cloud_plan_sha256"],
        "records": records,
        "record_count": len(records),
        "create_only_or_identical_reuse": True,
        "cloud_mutated": mutated,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def _observe_stage(
    plan: Mapping[str, Any], transport: CloudTransport
) -> dict[str, Any]:
    """Reconstruct the exact stage receipt after a lost execute response."""

    records = []
    for row in plan["content_entries"]:
        metadata = transport.get_object_metadata(
            bucket=plan["bucket"], object_name=row["object_name"]
        )
        if metadata is None:
            raise PostTrainingCloudError("staged source is absent during recovery")
        generation = str(metadata.get("generation", ""))
        payload = transport.get_object_bytes(
            bucket=plan["bucket"],
            object_name=row["object_name"],
            generation=generation,
        )
        if (
            payload is None
            or hashlib.sha256(payload).hexdigest() != row["sha256"]
            or len(payload) != row["bytes"]
            or _PROVIDER_ID.fullmatch(generation) is None
        ):
            raise PostTrainingCloudError("staged source changed during recovery")
        records.append(
            {
                "kind": row["kind"],
                "relative_path": row["relative_path"],
                "object_name": row["object_name"],
                "generation": generation,
                "sha256": row["sha256"],
                "bytes": row["bytes"],
                "created": False,
            }
        )
    core = {
        "schema": STAGE_SCHEMA,
        "cloud_plan_sha256": plan["cloud_plan_sha256"],
        "records": records,
        "record_count": len(records),
        "create_only_or_identical_reuse": True,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def _policy_parts(value: Mapping[str, Any]) -> tuple[list[Any], str, int]:
    bindings = value.get("bindings", [])
    etag = value.get("etag")
    version = value.get("version", 1)
    if not isinstance(bindings, list) or not isinstance(etag, str) or not etag:
        raise PostTrainingCloudError("bucket IAM policy is incomplete")
    return deepcopy(bindings), etag, max(3, int(version))


def _iam_titles(plan: Mapping[str, Any], wave: Mapping[str, Any]) -> list[str]:
    prefix = plan["contract"]["execution_identity_sha256"][:12]
    return [
        f"ofc-pt-read-{prefix}-w{wave['wave_index']:03d}",
        *[
            f"ofc-pt-create-{prefix}-w{wave['wave_index']:03d}-s{index:02d}"
            for index in range(len(wave["selected"]))
        ],
    ]


def _iam_bindings(
    plan: Mapping[str, Any],
    wave: Mapping[str, Any],
    *,
    expires_at_utc: str,
) -> list[dict[str, Any]]:
    resource = f"projects/_/buckets/{plan['bucket']}/objects/"
    accounts = plan["worker_service_accounts"][: len(wave["selected"])]
    titles = _iam_titles(plan, wave)
    content_prefix = (
        f"m31-post/{plan['contract']['execution_identity_sha256']}/content/"
    )
    rows = [
        {
            "role": "roles/storage.objectViewer",
            "members": [f"serviceAccount:{email}" for email in accounts],
            "condition": {
                "title": titles[0],
                "description": "M3.1 post-training immutable input read",
                "expression": (
                    f"resource.name.startsWith('{resource}{content_prefix}') && "
                    f"request.time < timestamp('{expires_at_utc}')"
                ),
            },
        }
    ]
    for index, selected in enumerate(wave["selected"]):
        rows.append(
            {
                "role": "roles/storage.objectCreator",
                "members": [f"serviceAccount:{accounts[index]}"],
                "condition": {
                    "title": titles[index + 1],
                    "description": "M3.1 post-training create-only result",
                    "expression": (
                        f"resource.name.startsWith('{resource}"
                        f"{selected['result_prefix']}/') && "
                        f"request.time < timestamp('{expires_at_utc}')"
                    ),
                },
            }
        )
    return rows


def _install_iam(
    plan: Mapping[str, Any],
    wave: Mapping[str, Any],
    *,
    transport: CloudTransport,
    now_unix_seconds: int,
) -> dict[str, Any]:
    for email in plan["worker_service_accounts"][: len(wave["selected"])]:
        account = transport.get_service_account(email=email)
        if (
            account.get("email") != email
            or _PROVIDER_ID.fullmatch(str(account.get("uniqueId", ""))) is None
            or transport.test_service_account_act_as(email=email) is not True
        ):
            raise PermissionError("post-training worker service account changed")
    expires = dt.datetime.fromtimestamp(
        now_unix_seconds + IAM_TTL_SECONDS, tz=dt.timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    additions = _iam_bindings(plan, wave, expires_at_utc=expires)
    policy = dict(transport.get_bucket_iam_policy(bucket=plan["bucket"]))
    bindings, etag, version = _policy_parts(policy)
    titles = set(_iam_titles(plan, wave))
    if any(
        isinstance(row, Mapping)
        and isinstance(row.get("condition"), Mapping)
        and row["condition"].get("title") in titles
        for row in bindings
    ):
        raise FileExistsError("post-training wave IAM title already exists")
    desired = {
        **{key: value for key, value in policy.items() if key != "bindings"},
        "etag": etag,
        "version": version,
        "bindings": [*bindings, *additions],
    }
    transport.set_bucket_iam_policy(bucket=plan["bucket"], policy=desired)
    observed = dict(transport.get_bucket_iam_policy(bucket=plan["bucket"]))
    observed_bindings, _, _ = _policy_parts(observed)
    if any(row not in observed_bindings for row in additions):
        raise PostTrainingCloudError("post-training IAM readback changed")
    core = {
        "schema": IAM_SCHEMA,
        "cloud_plan_sha256": plan["cloud_plan_sha256"],
        "wave_sha256": wave["wave_sha256"],
        "expires_at_utc": expires,
        "condition_titles": _iam_titles(plan, wave),
        "bindings": additions,
        "cloud_mutated": True,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def worker_startup_script() -> str:
    """Small network-install-free bootstrap; all heavy logic is pinned Python."""

    return r"""#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=/var/lib/ofc-m31-post-training-v1
STAGING=$ROOT/staging
WORK=$ROOT/work
META=http://metadata.google.internal/computeMetadata/v1/instance/attributes
HEADER='Metadata-Flavor: Google'
mkdir -p "$STAGING" "$WORK"
chmod 0700 "$ROOT" "$STAGING" "$WORK"
exec >>/var/log/ofc-m31-post-training-v1.log 2>&1
meta() { curl -fsS -H "$HEADER" "$META/$1"; }
BUCKET="$(meta pt-bucket)"
JOB_ID="$(meta pt-job-id)"
ATTEMPT_ID="$(meta pt-attempt-id)"
PREFIX="$(meta pt-result-prefix)"
BINDINGS="$(meta pt-bindings-b64)"
WATCHDOG="$(meta pt-watchdog)"
( sleep "$WATCHDOG"; shutdown -h now ) &
WATCHDOG_PID=$!
trap 'kill "$WATCHDOG_PID" >/dev/null 2>&1 || true' EXIT
python3 - "$STAGING" "$BUCKET" "$BINDINGS" <<'PY'
import base64,hashlib,json,os,pathlib,sys,urllib.parse,urllib.request
root=pathlib.Path(sys.argv[1]).resolve(); bucket=sys.argv[2]
raw=base64.b64decode(sys.argv[3],validate=True); rows=json.loads(raw)
canon=lambda v: json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
if raw!=canon(rows) or not isinstance(rows,list): raise SystemExit("bad bindings")
token=json.load(urllib.request.urlopen(urllib.request.Request(
 "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
 headers={"Metadata-Flavor":"Google"})))["access_token"]
for row in rows:
 if set(row)!={"relative","object","generation","sha256","bytes"}: raise SystemExit("bad binding fields")
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
import pathlib,sys,tarfile,zipfile
root=pathlib.Path(sys.argv[1])
def untar(path,top):
 with tarfile.open(path,"r:*") as archive:
  for member in archive.getmembers():
   p=pathlib.PurePosixPath(member.name)
   if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0]!=top or member.issym() or member.islnk(): raise SystemExit("unsafe tar")
  archive.extractall(root)
def unzip(path,top):
 with zipfile.ZipFile(path) as archive:
  for name in archive.namelist():
   p=pathlib.PurePosixPath(name)
   if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0]!=top: raise SystemExit("unsafe zip")
  archive.extractall(root)
untar(root/"static/runtime/runtime.tar.gz","runtime")
unzip(root/"static/wheelhouse/wheelhouse.zip","wheelhouse")
abr=root/"static/abr_bundle/abr_bundle.tar.gz"
if abr.exists(): untar(abr,"abr_bundle")
PY
python3 -m venv --without-pip "$ROOT/venv"
PIP_WHEEL="$(find "$STAGING/wheelhouse" -type f -name 'pip-*.whl' | sort | head -n1)"
[[ -n "$PIP_WHEEL" ]]
PYTHONPATH="$PIP_WHEEL" "$ROOT/venv/bin/python" -m pip install --no-index --find-links "$STAGING/wheelhouse" "$STAGING"/wheelhouse/*.whl >/dev/null
export PYTHONPATH="$STAGING/runtime/src"
cd "$STAGING/runtime"
"$ROOT/venv/bin/python" -m ofc_regular.hu_m31_t3_post_training_gcp_v1 cloud-worker \
 --contract "$STAGING/contract.json" --job-id "$JOB_ID" --attempt-id "$ATTEMPT_ID" \
 --staging-root "$STAGING" --output-root "$WORK/output" --bucket "$BUCKET" --result-prefix "$PREFIX"
kill "$WATCHDOG_PID" >/dev/null 2>&1 || true
shutdown -h now
"""


def _content_bindings(
    plan: Mapping[str, Any], stage: Mapping[str, Any]
) -> list[dict[str, Any]]:
    records = {row["kind"]: row for row in stage["records"]}
    return [
        {
            "relative": entry["relative_path"],
            "object": entry["object_name"],
            "generation": records[entry["kind"]]["generation"],
            "sha256": entry["sha256"],
            "bytes": entry["bytes"],
        }
        for entry in plan["content_entries"]
    ]


def _instance_spec(
    plan: Mapping[str, Any],
    wave: Mapping[str, Any],
    selected: Mapping[str, Any],
    *,
    slot: int,
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    labels = {
        "ofc-owner": plan["contract"]["execution_identity_sha256"][:32],
        "ofc-plan": plan["cloud_plan_sha256"][:32],
        "ofc-wave": f"w{wave['wave_index']:03d}",
    }
    metadata = {
        "pt-bucket": plan["bucket"],
        "pt-job-id": selected["job_id"],
        "pt-attempt-id": selected["attempt_id"],
        "pt-result-prefix": selected["result_prefix"],
        "pt-watchdog": str(WATCHDOG_SECONDS),
        "pt-bindings-b64": base64.b64encode(
            canonical_bytes(_content_bindings(plan, stage))
        ).decode("ascii"),
        "startup-script": worker_startup_script(),
    }
    name = selected["instance_name"]
    return {
        "name": name,
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
                "interface": quality_provider.BOOT_DISK_INTERFACE,
                "deviceName": name,
                "initializeParams": {
                    "sourceImage": plan["image"]["self_link"],
                    "diskSizeGb": str(quality_provider.BOOT_DISK_SIZE_GB),
                    "diskName": name,
                    "labels": labels,
                    "diskType": (
                        f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
                        f"zones/{ZONE}/diskTypes/{quality_provider.BOOT_DISK_TYPE}"
                    ),
                },
            }
        ],
        "networkInterfaces": [
            {
                "network": (
                    f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
                    f"global/networks/{quality_provider.NETWORK}"
                ),
                "subnetwork": (
                    f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
                    f"regions/{REGION}/subnetworks/{quality_provider.SUBNETWORK}"
                ),
                "nicType": quality_provider.NIC_TYPE,
                "accessConfigs": [],
            }
        ],
        "serviceAccounts": [
            {
                "email": plan["worker_service_accounts"][slot],
                "scopes": [quality_provider.OAUTH_SCOPE],
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


def _uuid_for(*parts: str) -> str:
    raw = hashlib.sha256("\0".join(parts).encode("ascii")).hexdigest()
    return str(uuid.UUID(raw[:32], version=4))


def execute_wave(
    *,
    cloud_plan: Mapping[str, Any],
    wave: Mapping[str, Any],
    transport: CloudTransport,
    raw_nonce: str,
    now_unix_seconds: int,
    observed_at_utc: str,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Stage, install exact IAM, claim, and create only listed Spot VMs."""

    plan = validate_cloud_plan(cloud_plan, replay_sources=True)
    frozen = validate_wave(wave, cloud_plan=plan)
    try:
        parsed = uuid.UUID(raw_nonce)
    except (ValueError, AttributeError) as exc:
        raise ValueError("post-training launch nonce must be UUIDv4") from exc
    if parsed.version != 4 or str(parsed) != raw_nonce:
        raise ValueError("post-training launch nonce must be canonical UUIDv4")
    stage = _stage(plan, transport)
    iam = _install_iam(
        plan, frozen, transport=transport, now_unix_seconds=now_unix_seconds
    )
    quality_provider._validate_image(plan, transport)  # type: ignore[arg-type]
    quality_provider._validate_network(plan, transport)  # type: ignore[arg-type]
    quota_plan = {
        **plan,
        "workers": [{} for _ in frozen["selected"]],
    }
    quota = quality_provider.read_quota(  # type: ignore[arg-type]
        provider_plan=quota_plan,
        transport=transport,
        observed_at_utc=observed_at_utc,
    )
    claim_name = (
        f"m31-post/{plan['contract']['execution_identity_sha256']}/claims/"
        f"{frozen['wave_sha256']}.json"
    )
    claim_value = {
        "schema": "hu_m31_t3_post_training_claim_v1",
        "cloud_plan_sha256": plan["cloud_plan_sha256"],
        "wave_sha256": frozen["wave_sha256"],
        "nonce_sha256": hashlib.sha256(raw_nonce.encode("ascii")).hexdigest(),
        "create_only": True,
    }
    claim_raw = canonical_bytes(claim_value)
    if transport.get_object_metadata(bucket=plan["bucket"], object_name=claim_name):
        raise FileExistsError("post-training wave claim already exists")
    claim_meta = transport.put_object_new(
        bucket=plan["bucket"],
        object_name=claim_name,
        payload=claim_raw,
        content_type="application/json",
    )
    claim_generation = str(claim_meta.get("generation", ""))
    if (
        claim_meta.get("name") != claim_name
        or _PROVIDER_ID.fullmatch(claim_generation) is None
        or transport.get_object_bytes(
            bucket=plan["bucket"],
            object_name=claim_name,
            generation=claim_generation,
        )
        != claim_raw
    ):
        raise PostTrainingCloudError("post-training launch claim changed")
    rows = []
    for slot, selected in enumerate(frozen["selected"]):
        spec = _instance_spec(plan, frozen, selected, slot=slot, stage=stage)
        request_id = _uuid_for(
            plan["cloud_plan_sha256"],
            frozen["wave_sha256"],
            selected["job_id"],
            selected["attempt_id"],
            "create",
        )
        if transport.get_instance(instance_name=selected["instance_name"]) is not None:
            raise FileExistsError("post-training instance exists before launch")
        operation = transport.create_instance(
            instance_spec=spec, request_id=request_id
        )
        quality_provider._wait_operation(  # type: ignore[arg-type]
            transport,
            initial=operation,
            operation_type="insert",
            target_name=selected["instance_name"],
            sleep=sleep,
        )
        instance = transport.get_instance(instance_name=selected["instance_name"])
        if instance is None:
            raise PostTrainingCloudError("post-training instance is absent")
        provider_id, disk_name, status = dataset_provider._validate_owned_instance(
            instance, expected_spec=spec
        )
        disk = transport.get_disk_optional(disk_name=disk_name)
        if disk is None:
            raise PostTrainingCloudError("post-training boot disk is absent")
        disk_id = dataset_provider._validate_owned_disk(disk, expected_spec=spec)
        rows.append(
            {
                "job_id": selected["job_id"],
                "attempt_id": selected["attempt_id"],
                "instance_name": selected["instance_name"],
                "request_id": request_id,
                "spec_sha256": canonical_sha256(spec),
                "provider_instance_id": provider_id,
                "provider_boot_disk_id": disk_id,
                "observed_status": status,
            }
        )
    core = {
        "schema": LAUNCH_SCHEMA,
        "cloud_plan_sha256": plan["cloud_plan_sha256"],
        "wave_sha256": frozen["wave_sha256"],
        "stage": stage,
        "iam": iam,
        "quota": quota,
        "claim": {
            "object_name": claim_name,
            "generation": claim_generation,
            "sha256": hashlib.sha256(claim_raw).hexdigest(),
        },
        "rows": rows,
        "created_instance_count": len(rows),
        "at_most_eight_c4": len(rows) <= MAX_CONCURRENT_VMS,
        "cloud_mutated": True,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def _remote_result(
    plan: Mapping[str, Any],
    selected: Mapping[str, Any],
    transport: CloudTransport,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    manifest_name = f"{selected['result_prefix']}/manifest.json"
    metadata = transport.get_object_metadata(
        bucket=plan["bucket"], object_name=manifest_name
    )
    if metadata is None:
        return None
    generation = str(metadata.get("generation", ""))
    raw = transport.get_object_bytes(
        bucket=plan["bucket"], object_name=manifest_name, generation=generation
    )
    if raw is None:
        raise PostTrainingCloudError("post-training result manifest disappeared")
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PostTrainingCloudError("post-training result manifest is not JSON") from exc
    if raw != canonical_bytes(value):
        raise PostTrainingCloudError("post-training result manifest is not canonical")
    manifest = validate_result_manifest(
        value,
        contract=plan["contract"],
        job_id=selected["job_id"],
        attempt_id=selected["attempt_id"],
    )
    files = []
    for record in manifest["files"]:
        name = f"{selected['result_prefix']}/files/{record['relative_path']}"
        file_meta = transport.get_object_metadata(bucket=plan["bucket"], object_name=name)
        if file_meta is None:
            raise PostTrainingCloudError("result manifest references missing file")
        file_generation = str(file_meta.get("generation", ""))
        payload = transport.get_object_bytes(
            bucket=plan["bucket"], object_name=name, generation=file_generation
        )
        if (
            payload is None
            or hashlib.sha256(payload).hexdigest() != record["sha256"]
            or len(payload) != record["bytes"]
        ):
            raise PostTrainingCloudError("post-training result file changed")
        files.append(
            {
                **record,
                "object_name": name,
                "generation": file_generation,
            }
        )
    return manifest, files


def poll_wave(
    *,
    cloud_plan: Mapping[str, Any],
    wave: Mapping[str, Any],
    transport: CloudTransport,
) -> dict[str, Any]:
    plan = validate_cloud_plan(cloud_plan, replay_sources=True)
    frozen = validate_wave(wave, cloud_plan=plan)
    rows = []
    for selected in frozen["selected"]:
        remote = _remote_result(plan, selected, transport)
        rows.append(
            {
                "job_id": selected["job_id"],
                "attempt_id": selected["attempt_id"],
                "instance_present": transport.get_instance(
                    instance_name=selected["instance_name"]
                )
                is not None,
                "complete": remote is not None,
                "manifest_sha256": (
                    None if remote is None else remote[0]["manifest_sha256"]
                ),
            }
        )
    core = {
        "schema": POLL_SCHEMA,
        "cloud_plan_sha256": plan["cloud_plan_sha256"],
        "wave_sha256": frozen["wave_sha256"],
        "rows": rows,
        "complete_count": sum(row["complete"] for row in rows),
        "read_only": True,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def _remove_iam(
    plan: Mapping[str, Any],
    wave: Mapping[str, Any],
    iam: Mapping[str, Any],
    transport: CloudTransport,
) -> None:
    if (
        iam.get("schema") != IAM_SCHEMA
        or iam.get("receipt_sha256") != _self_digest(iam, "receipt_sha256")
        or iam.get("wave_sha256") != wave["wave_sha256"]
    ):
        raise ValueError("post-training IAM receipt changed")
    policy = dict(transport.get_bucket_iam_policy(bucket=plan["bucket"]))
    bindings, etag, version = _policy_parts(policy)
    exact = iam["bindings"]
    titles = set(iam["condition_titles"])
    matching = [
        row
        for row in bindings
        if isinstance(row, Mapping)
        and isinstance(row.get("condition"), Mapping)
        and row["condition"].get("title") in titles
    ]
    if matching != exact:
        raise PermissionError("post-training IAM changed before cleanup")
    unrelated = [row for row in bindings if row not in exact]
    transport.set_bucket_iam_policy(
        bucket=plan["bucket"],
        policy={
            **{key: value for key, value in policy.items() if key != "bindings"},
            "etag": etag,
            "version": version,
            "bindings": unrelated,
        },
    )
    observed, _, _ = _policy_parts(
        transport.get_bucket_iam_policy(bucket=plan["bucket"])
    )
    if observed != unrelated:
        raise PostTrainingCloudError("post-training IAM absence not proven")


def _remove_recoverable_iam(
    plan: Mapping[str, Any],
    wave: Mapping[str, Any],
    transport: CloudTransport,
) -> bool:
    """Remove a complete exact title set when the IAM receipt was lost."""

    policy = dict(transport.get_bucket_iam_policy(bucket=plan["bucket"]))
    bindings, etag, version = _policy_parts(policy)
    titles = set(_iam_titles(plan, wave))
    matching = [
        row
        for row in bindings
        if isinstance(row, Mapping)
        and isinstance(row.get("condition"), Mapping)
        and row["condition"].get("title") in titles
    ]
    if not matching:
        return False
    if (
        len(matching) != len(titles)
        or {
            row["condition"]["title"]
            for row in matching
            if isinstance(row.get("condition"), Mapping)
        }
        != titles
    ):
        raise PermissionError("recovery IAM title set is partial or duplicated")
    expiries = set()
    for row in matching:
        expression = str(row["condition"].get("expression", ""))
        match = re.search(r"timestamp\('([^']+)'\)", expression)
        if match is None:
            raise PermissionError("recovery IAM expiry is missing")
        expiries.add(match.group(1))
    if len(expiries) != 1 or matching != _iam_bindings(
        plan, wave, expires_at_utc=next(iter(expiries))
    ):
        raise PermissionError("recovery IAM binding differs from exact wave")
    unrelated = [row for row in bindings if row not in matching]
    transport.set_bucket_iam_policy(
        bucket=plan["bucket"],
        policy={
            **{key: value for key, value in policy.items() if key != "bindings"},
            "etag": etag,
            "version": version,
            "bindings": unrelated,
        },
    )
    observed, _, _ = _policy_parts(
        transport.get_bucket_iam_policy(bucket=plan["bucket"])
    )
    if observed != unrelated:
        raise PostTrainingCloudError("recovery IAM absence not proven")
    return True


def abort_wave(
    *,
    cloud_plan: Mapping[str, Any],
    wave: Mapping[str, Any],
    transport: CloudTransport,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Fail-closed recovery after execute mutated cloud but lost its receipt.

    Only exact planned names are inspected.  An instance is deleted only after
    its complete spec, provider ID, and boot-disk ID have been measured.
    """

    plan = validate_cloud_plan(cloud_plan, replay_sources=True)
    frozen = validate_wave(wave, cloud_plan=plan)
    stage = _observe_stage(plan, transport)
    rows = []
    for slot, selected in enumerate(frozen["selected"]):
        remote = _remote_result(plan, selected, transport)
        instance = transport.get_instance(instance_name=selected["instance_name"])
        if instance is not None:
            spec = _instance_spec(
                plan, frozen, selected, slot=slot, stage=stage
            )
            _provider_id, disk_name, _status = (
                dataset_provider._validate_owned_instance(
                    instance, expected_spec=spec
                )
            )
            disk = transport.get_disk_optional(disk_name=disk_name)
            if disk is None:
                raise PermissionError("abort cleanup boot disk is absent")
            dataset_provider._validate_owned_disk(disk, expected_spec=spec)
            operation = transport.delete_instance(
                instance_name=selected["instance_name"],
                request_id=_uuid_for(
                    plan["cloud_plan_sha256"],
                    frozen["wave_sha256"],
                    selected["job_id"],
                    selected["attempt_id"],
                    "abort-delete",
                ),
            )
            if operation is not None:
                quality_provider._wait_operation(  # type: ignore[arg-type]
                    transport,
                    initial=operation,
                    operation_type="delete",
                    target_name=selected["instance_name"],
                    sleep=sleep,
                )
        if (
            transport.get_instance(instance_name=selected["instance_name"])
            is not None
            or transport.get_disk_optional(
                disk_name=selected["instance_name"]
            )
            is not None
        ):
            raise TimeoutError("abort cleanup VM/disk absence was not proven")
        rows.append(
            {
                "job_id": selected["job_id"],
                "attempt_id": selected["attempt_id"],
                "instance_name": selected["instance_name"],
                "status": "complete" if remote is not None else "failed",
                "manifest": None if remote is None else remote[0],
                "files": [] if remote is None else remote[1],
                "owned_vm_disk_absent": True,
            }
        )
    _remove_recoverable_iam(plan, frozen, transport)
    core = {
        "schema": LIFECYCLE_SCHEMA,
        "cloud_plan_sha256": plan["cloud_plan_sha256"],
        "wave_sha256": frozen["wave_sha256"],
        "wave_index": frozen["wave_index"],
        "rows": rows,
        "selected_count": len(rows),
        "create_only_gcs": True,
        "exact_owned_cleanup": True,
        "owned_vm_disk_absent": True,
        "worker_iam_removed": True,
        "wildcard_delete_used": False,
        "unrelated_resource_touched": False,
        "gcs_evidence_deleted": False,
        "recovered_after_missing_launch_receipt": True,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def cleanup_wave(
    *,
    cloud_plan: Mapping[str, Any],
    wave: Mapping[str, Any],
    launch: Mapping[str, Any],
    transport: CloudTransport,
    allow_incomplete: bool,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    plan = validate_cloud_plan(cloud_plan, replay_sources=True)
    frozen = validate_wave(wave, cloud_plan=plan)
    if (
        launch.get("schema") != LAUNCH_SCHEMA
        or launch.get("receipt_sha256") != _self_digest(launch, "receipt_sha256")
        or launch.get("wave_sha256") != frozen["wave_sha256"]
    ):
        raise ValueError("post-training launch receipt changed")
    poll = poll_wave(cloud_plan=plan, wave=frozen, transport=transport)
    if poll["complete_count"] != len(frozen["selected"]) and not allow_incomplete:
        raise PermissionError("post-training wave has incomplete running jobs")
    launched = {row["job_id"]: row for row in launch["rows"]}
    rows = []
    for slot, selected in enumerate(frozen["selected"]):
        remote = _remote_result(plan, selected, transport)
        instance = transport.get_instance(instance_name=selected["instance_name"])
        if instance is not None:
            spec = _instance_spec(
                plan, frozen, selected, slot=slot, stage=launch["stage"]
            )
            provider_id, disk_name, _ = dataset_provider._validate_owned_instance(
                instance, expected_spec=spec
            )
            disk = transport.get_disk_optional(disk_name=disk_name)
            if disk is None:
                raise PermissionError("post-training disk absent before cleanup")
            disk_id = dataset_provider._validate_owned_disk(
                disk, expected_spec=spec
            )
            if (
                launched[selected["job_id"]]["provider_instance_id"] != provider_id
                or launched[selected["job_id"]]["provider_boot_disk_id"] != disk_id
            ):
                raise PermissionError("post-training owned identity changed")
            operation = transport.delete_instance(
                instance_name=selected["instance_name"],
                request_id=_uuid_for(
                    plan["cloud_plan_sha256"],
                    frozen["wave_sha256"],
                    selected["job_id"],
                    selected["attempt_id"],
                    "delete",
                ),
            )
            if operation is not None:
                quality_provider._wait_operation(  # type: ignore[arg-type]
                    transport,
                    initial=operation,
                    operation_type="delete",
                    target_name=selected["instance_name"],
                    sleep=sleep,
                )
        if (
            transport.get_instance(instance_name=selected["instance_name"]) is not None
            or transport.get_disk_optional(disk_name=selected["instance_name"]) is not None
        ):
            raise TimeoutError("post-training VM/disk absence was not proven")
        rows.append(
            {
                "job_id": selected["job_id"],
                "attempt_id": selected["attempt_id"],
                "instance_name": selected["instance_name"],
                "status": "complete" if remote is not None else "failed",
                "manifest": None if remote is None else remote[0],
                "files": [] if remote is None else remote[1],
                "owned_vm_disk_absent": True,
            }
        )
    _remove_iam(plan, frozen, launch["iam"], transport)
    core = {
        "schema": LIFECYCLE_SCHEMA,
        "cloud_plan_sha256": plan["cloud_plan_sha256"],
        "wave_sha256": frozen["wave_sha256"],
        "wave_index": frozen["wave_index"],
        "rows": rows,
        "selected_count": len(rows),
        "create_only_gcs": True,
        "exact_owned_cleanup": True,
        "owned_vm_disk_absent": True,
        "worker_iam_removed": True,
        "wildcard_delete_used": False,
        "unrelated_resource_touched": False,
        "gcs_evidence_deleted": False,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def validate_lifecycle(
    value: Mapping[str, Any], *, cloud_plan: Mapping[str, Any]
) -> dict[str, Any]:
    plan = validate_cloud_plan(cloud_plan, replay_sources=True)
    receipt = deepcopy(dict(value))
    rows = receipt.get("rows")
    if (
        receipt.get("schema") != LIFECYCLE_SCHEMA
        or receipt.get("receipt_sha256")
        != _self_digest(receipt, "receipt_sha256")
        or receipt.get("cloud_plan_sha256") != plan["cloud_plan_sha256"]
        or not isinstance(rows, list)
        or receipt.get("selected_count") != len(rows)
        or any(row.get("status") not in {"complete", "failed"} for row in rows)
        or receipt.get("exact_owned_cleanup") is not True
        or receipt.get("owned_vm_disk_absent") is not True
        or receipt.get("worker_iam_removed") is not True
        or receipt.get("wildcard_delete_used") is not False
        or receipt.get("unrelated_resource_touched") is not False
        or receipt.get("gcs_evidence_deleted") is not False
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("post-training lifecycle receipt changed")
    for row in rows:
        if row["status"] == "complete":
            validate_result_manifest(
                row["manifest"],
                contract=plan["contract"],
                job_id=row["job_id"],
                attempt_id=row["attempt_id"],
            )
    return receipt


def receive_lifecycle(
    *,
    cloud_plan: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
    transport: CloudTransport,
    output_root: str | Path,
) -> dict[str, Any]:
    """Download and scientifically validate exact files after cleanup."""

    plan = validate_cloud_plan(cloud_plan, replay_sources=True)
    checked = validate_lifecycle(lifecycle, cloud_plan=plan)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise ValueError("post-training receive root is unsafe")
    received = []
    for row in checked["rows"]:
        if row["status"] != "complete":
            continue
        job = _job(plan["contract"], row["job_id"])
        for record in row["files"]:
            payload = transport.get_object_bytes(
                bucket=plan["bucket"],
                object_name=record["object_name"],
                generation=record["generation"],
            )
            if (
                payload is None
                or hashlib.sha256(payload).hexdigest() != record["sha256"]
                or len(payload) != record["bytes"]
            ):
                raise PostTrainingCloudError("received post-training bytes changed")
            relative = record["relative_path"]
            if plan["contract"]["workload"].startswith("abr_"):
                destination = root.joinpath(*PurePosixPath(relative).parts)
            elif relative.startswith("shards/"):
                destination = root / "shards" / Path(relative).name
            else:
                destination = root / "receipts" / f"{row['job_id']}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                with destination.open("xb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            except FileExistsError:
                if (
                    destination.is_symlink()
                    or not destination.is_file()
                    or destination.read_bytes() != payload
                ):
                    raise FileExistsError(
                        f"immutable received result changed: {destination}"
                    ) from None
            if plan["contract"]["workload"].startswith("abr_"):
                abr_plan = abr_teacher.validate_plan(
                    _read_canonical(
                        _sources_by_kind(plan["contract"])["abr_plan"]["source_path"],
                        "ABR plan",
                    )
                )
                abr_teacher.validate_pair_evidence(
                    _read_canonical(destination, "received ABR pair"),
                    plan=abr_plan,
                    pair_index=int(job["pair_index"]),
                )
            elif relative.startswith("shards/"):
                promotion_plan = promotion.validate_locked_promotion_plan(
                    _read_canonical(
                        _sources_by_kind(plan["contract"])["promotion_plan"][
                            "source_path"
                        ],
                        "promotion plan",
                    )
                )
                execution_plan = promotion_execution.validate_execution_plan(
                    _read_canonical(
                        _sources_by_kind(plan["contract"])["execution_plan"][
                            "source_path"
                        ],
                        "execution plan",
                    ),
                    promotion_plan=promotion_plan,
                )
                item = next(
                    item
                    for item in execution_plan["work_items"]
                    if item["work_id"] == job["work_id"]
                )
                promotion_execution.validate_work_item_shard(
                    work_item=item,
                    shard=_read_canonical(destination, "received promotion shard"),
                    promotion_plan=promotion_plan,
                )
            received.append(
                {
                    "job_id": row["job_id"],
                    "relative_path": relative,
                    "local_path": str(destination.resolve()),
                    "sha256": record["sha256"],
                    "bytes": record["bytes"],
                }
            )
    core = {
        "schema": RECEIVE_SCHEMA,
        "cloud_plan_sha256": plan["cloud_plan_sha256"],
        "lifecycle_sha256": checked["receipt_sha256"],
        "received": received,
        "received_file_count": len(received),
        "source_replayed": True,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def _metadata_token() -> str:
    request = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/"
        "instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    return json.load(urllib.request.urlopen(request, timeout=30))["access_token"]


def _gcs_create_only(
    *, bucket: str, object_name: str, payload: bytes
) -> Mapping[str, Any]:
    url = (
        "https://storage.googleapis.com/upload/storage/v1/b/"
        + urllib.parse.quote(bucket, safe="")
        + "/o?uploadType=media&ifGenerationMatch=0&name="
        + urllib.parse.quote(object_name, safe="")
    )
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {_metadata_token()}",
            "Content-Type": "application/octet-stream",
        },
    )
    return json.load(urllib.request.urlopen(request, timeout=600))


def cloud_worker(
    *,
    contract_path: str | Path,
    job_id: str,
    attempt_id: str,
    staging_root: str | Path,
    output_root: str | Path,
    bucket: str,
    result_prefix: str,
) -> dict[str, Any]:
    """VM entrypoint: heartbeat, real work, files, then manifest last."""

    contract = validate_contract(
        _read_canonical(contract_path, "worker contract"), replay_sources=False
    )
    if bucket != contract["bucket"] or not re.fullmatch(r"a[0-9]{2}", attempt_id):
        raise ValueError("cloud-worker identity changed")
    heartbeat_stop = threading.Event()
    heartbeat_error: list[BaseException] = []

    def heartbeat() -> None:
        sequence = 0
        try:
            while not heartbeat_stop.is_set():
                value = {
                    "schema": "hu_m31_t3_post_training_heartbeat_v1",
                    "contract_sha256": contract["contract_sha256"],
                    "job_id": job_id,
                    "attempt_id": attempt_id,
                    "sequence": sequence,
                    "create_only": True,
                    "current_profile_changed": False,
                }
                _gcs_create_only(
                    bucket=bucket,
                    object_name=f"{result_prefix}/heartbeats/{sequence:06d}.json",
                    payload=canonical_bytes(value),
                )
                sequence += 1
                if heartbeat_stop.wait(60):
                    break
        except BaseException as exc:  # pragma: no cover - VM-only boundary
            heartbeat_error.append(exc)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        local_manifest = run_worker_job(
            contract=contract,
            job_id=job_id,
            staging_root=staging_root,
            output_root=output_root,
            require_host_target=True,
        )
    finally:
        heartbeat_stop.set()
        thread.join()
    if heartbeat_error:
        raise PostTrainingCloudError("post-training heartbeat failed") from heartbeat_error[0]
    manifest = {
        **{
            key: value
            for key, value in local_manifest.items()
            if key not in {"attempt_id", "manifest_sha256"}
        },
        "attempt_id": attempt_id,
    }
    manifest["manifest_sha256"] = _self_digest(manifest, "manifest_sha256")
    checked = validate_result_manifest(
        manifest,
        contract=contract,
        job_id=job_id,
        attempt_id=attempt_id,
    )
    output = Path(output_root)
    for record in checked["files"]:
        payload = output.joinpath(
            *PurePosixPath(record["relative_path"]).parts
        ).read_bytes()
        _gcs_create_only(
            bucket=bucket,
            object_name=f"{result_prefix}/files/{record['relative_path']}",
            payload=payload,
        )
    _gcs_create_only(
        bucket=bucket,
        object_name=f"{result_prefix}/manifest.json",
        payload=canonical_bytes(checked),
    )
    return checked


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("cloud-worker")
    worker.add_argument("--contract", required=True)
    worker.add_argument("--job-id", required=True)
    worker.add_argument("--attempt-id", required=True)
    worker.add_argument("--staging-root", required=True)
    worker.add_argument("--output-root", required=True)
    worker.add_argument("--bucket", required=True)
    worker.add_argument("--result-prefix", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "cloud-worker":
            result = cloud_worker(
                contract_path=args.contract,
                job_id=args.job_id,
                attempt_id=args.attempt_id,
                staging_root=args.staging_root,
                output_root=args.output_root,
                bucket=args.bucket,
                result_prefix=args.result_prefix,
            )
        else:  # pragma: no cover
            raise ValueError("unknown command")
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
