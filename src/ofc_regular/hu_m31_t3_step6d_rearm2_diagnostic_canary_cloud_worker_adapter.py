"""Backend-neutral transport adapter for the diagnostic worker package.

The older diagnostic VM preview remains bound to its local synthetic package.
This version binds the immutable native worker package and models the exact
runner output tree.  All remote state is caller-supplied in-memory data: there
is no object-store query/write, launcher, authorization, claim, or VM API.

Transport-only fixtures use deterministic file identities, never synthetic Q
values.  They are explicitly inadmissible as scientific, training, quality, or
promotion evidence.  A real receiver can later materialize the same tree and
pass it to ``runner.validate_completed_output``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence


class _LazyModule:
    """Defer development-only dependencies until their code path is used."""

    def __init__(self, relative_name: str) -> None:
        self._relative_name = relative_name
        self._module: Any | None = None

    def __getattr__(self, name: str) -> Any:
        if self._module is None:
            self._module = importlib.import_module(
                self._relative_name, package=__package__
            )
        return getattr(self._module, name)


worker = _LazyModule(
    ".hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_package"
)
plan = _LazyModule(".hu_m31_t3_step6d_rearm2_diagnostic_canary_plan")
legacy_vm = _LazyModule(
    ".hu_m31_t3_step6d_rearm2_diagnostic_canary_vm_adapter"
)
runner = _LazyModule(".run_hu_m31_t3_step6d_performance_v2")


ADAPTER_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_cloud_worker_adapter_v1"
)
PREVIEW_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_cloud_worker_preview_v1"
)
STAGE_TOKEN_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_cloud_worker_stage_token_v1"
)
REMOTE_MANIFEST_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_cloud_worker_remote_manifest_v1"
)
STAGE_IDENTITY_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_cloud_worker_stage_identity_v1"
)
MATERIALIZATION_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_runner_materialization_manifest_v1"
)
UPLOAD_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_runner_artifact_upload_v1"
)
HEARTBEAT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_runner_heartbeat_v1"
)
DONE_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_runner_done_envelope_v1"
)
RECEIVE_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_cloud_worker_receive_v1"
)
SNAPSHOT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_fake_remote_snapshot_v1"
)
SNAPSHOT_VALIDATION_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_fake_remote_validation_v1"
)
FIXTURE_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_transport_file_fixture_v1"
)

DEFAULT_PROJECT = "ofc-solver-485418"
DEFAULT_BUCKET = "pokerhu-ofc-solver-485418-training"
DEFAULT_ZONE = "asia-northeast1-b"
MAX_ATTEMPTS = 2
EXPECTED_PACKAGE_FILE_COUNT = 8
EXPECTED_SOURCE_ENTRY_COUNT = 60

_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$")
_SAFE_BUCKET = re.compile(r"^[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_KEYS = frozenset(
    {"source_role", "hand_index", "path", "sha256", "bytes"}
)
_FORBIDDEN_KEY_PARTS = (
    "opponent_private_discard",
    "opponent_hidden",
    "hidden_truth",
    "realized_deck_tail",
    "q_milli",
    "action_key",
)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} keys changed")


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA256.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise ValueError(f"{label} must be lowercase sha256")
    return value


def _reject_hidden_or_q(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).casefold()
            if any(part in key for part in _FORBIDDEN_KEY_PARTS):
                raise ValueError(f"forbidden transport field at {path}.{raw_key}")
            _reject_hidden_or_q(child, f"{path}.{raw_key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_hidden_or_q(child, f"{path}[{index}]")


def _stage(manifest: Mapping[str, Any], stage_id: str) -> dict[str, Any]:
    value = next(
        (row for row in manifest["stages"] if row["stage_id"] == stage_id),
        None,
    )
    if value is None:
        raise ValueError("unknown diagnostic worker stage")
    return dict(value)


def _package_files(package: Path) -> list[dict[str, Any]]:
    expected = (
        worker.MANIFEST_NAME,
        worker.READY_NAME,
        worker.SOURCE_NAME,
        worker.STARTUP_NAME,
        worker.VERIFIER_NAME,
        "jobs/candidate-shard-00.json",
        "jobs/candidate-shard-01.json",
        "jobs/reference-shard-01.json",
    )
    records = [
        {
            "path": relative,
            "bytes": (package / relative).stat().st_size,
            "sha256": sha256_file(package / relative),
        }
        for relative in expected
    ]
    if len(records) != EXPECTED_PACKAGE_FILE_COUNT:
        raise AssertionError("diagnostic worker package file count changed")
    return records


def _job_manifest_record(
    manifest: Mapping[str, Any], job_id: str
) -> dict[str, Any]:
    record = next(
        (row for row in manifest["job_manifests"] if row["job_id"] == job_id),
        None,
    )
    if record is None:
        raise ValueError("diagnostic worker job is absent")
    return dict(record)


def _root_record(
    manifest: Mapping[str, Any],
    *,
    source_role: str,
    hand_index: int,
) -> dict[str, Any]:
    relative = f"{worker.ROOT_PREFIX}/hand_{hand_index:03d}.json"
    entry = manifest["source_entries"].get(relative)
    if not isinstance(entry, Mapping) or entry.get("kind") != "root":
        raise ValueError("diagnostic worker root entry changed")
    return {
        "source_role": source_role,
        "hand_index": hand_index,
        "path": f"roots/hand_{hand_index:03d}.json",
        "sha256": entry["sha256"],
        "bytes": entry["bytes"],
    }


def _output_control_records(
    manifest: Mapping[str, Any],
    job_record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    contract_raw = canonical_bytes(manifest["run_contract"])
    return [
        {
            "path": "run_contract.json",
            "sha256": hashlib.sha256(contract_raw).hexdigest(),
            "bytes": len(contract_raw),
        },
        {
            "path": "shard_manifest.json",
            "sha256": job_record["sha256"],
            "bytes": job_record["bytes"],
        },
    ]


def _fixture_file(
    *,
    source_role: str,
    hand_index: int,
) -> dict[str, Any]:
    identity = {
        "schema": FIXTURE_SCHEMA,
        "source_role": source_role,
        "hand_index": hand_index,
        "path": f"hands/{source_role}/hand_{hand_index:03d}.json",
        "transport_fixture_only": True,
        "scientific_payload_present": False,
    }
    raw = canonical_bytes(identity)
    return {
        "source_role": source_role,
        "hand_index": hand_index,
        "path": identity["path"],
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _validate_artifact(
    value: Mapping[str, Any],
    *,
    source_role: str,
    hand_index: int,
    expected_path: str,
    label: str,
) -> dict[str, Any]:
    artifact = dict(value)
    _exact(artifact, set(_ARTIFACT_KEYS), label)
    if (
        artifact["source_role"] != source_role
        or type(artifact["hand_index"]) is not int
        or artifact["hand_index"] != hand_index
        or artifact["path"] != expected_path
        or _sha(artifact["sha256"], f"{label} sha256") != artifact["sha256"]
        or type(artifact["bytes"]) is not int
        or artifact["bytes"] <= 0
    ):
        raise ValueError(f"{label} identity changed")
    return artifact


def _validate_stage1_receive_bundle(
    *,
    package_manifest_sha256: str,
    preview: Mapping[str, Any] | None,
    receive: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if preview is None and receive is None:
        return None
    if not isinstance(preview, Mapping) or not isinstance(receive, Mapping):
        raise ValueError("stage1 prerequisite requires preview and receive")
    checked = validate_receive(receive, preview=preview)
    if (
        preview.get("stage_id") != plan.STAGE1_ID
        or preview.get("package_manifest_sha256") != package_manifest_sha256
        or checked["selected_job_ids"] != list(plan.STAGE1_JOB_IDS)
    ):
        raise ValueError("stage1 prerequisite escaped the worker package")
    return {
        "stage_id": plan.STAGE1_ID,
        "attempt_index": preview["attempt_index"],
        "preview_sha256": canonical_sha256(preview),
        "receive_sha256": canonical_sha256(checked),
        "selected_job_ids": checked["selected_job_ids"],
        "diagnostic_only": True,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
    }


def _resume_binding(
    *,
    package_dir: Path,
    stage_id: str,
    attempt_index: int,
    prior_preview: Mapping[str, Any] | None,
    prior_snapshot: Mapping[str, Any] | None,
    prerequisite_stage1_preview: Mapping[str, Any] | None,
    prerequisite_stage1_receive: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if type(attempt_index) is not int:
        raise ValueError("attempt_index must be an exact integer")
    if attempt_index == 0:
        if prior_preview is not None or prior_snapshot is not None:
            raise ValueError("attempt0 must not bind prior remote state")
        return None
    if attempt_index != 1:
        raise ValueError("diagnostic worker permits attempts 0 and 1 only")
    if not isinstance(prior_preview, Mapping) or not isinstance(
        prior_snapshot, Mapping
    ):
        raise ValueError("attempt1 requires exact attempt0 preview and snapshot")
    expected = build_preview(
        package_dir=package_dir,
        stage_id=stage_id,
        attempt_index=0,
        prerequisite_stage1_preview=prerequisite_stage1_preview,
        prerequisite_stage1_receive=prerequisite_stage1_receive,
        _skip_validation=True,
    )
    if dict(prior_preview) != expected:
        raise ValueError("attempt1 prior preview identity changed")
    validation = validate_snapshot(prior_preview, prior_snapshot)
    if (
        validation["receive_validated"]
        or validation["active_instance_names"] != []
        or all(
            row["state"] == "success_tree_ready"
            for row in validation["job_states"]
        )
    ):
        raise ValueError("attempt1 requires crashed/incomplete attempt0 state")
    return {
        "prior_attempt_index": 0,
        "prior_preview_sha256": canonical_sha256(prior_preview),
        "prior_snapshot_sha256": canonical_sha256(prior_snapshot),
        "prior_job_states": validation["job_states"],
        "active_instance_names": [],
        "vm_absence_proven_by_supplied_fixture": True,
        "remote_query_performed": False,
    }


def _build_stage_identity(
    *,
    manifest: Mapping[str, Any],
    package_manifest_sha256: str,
    package_files: Sequence[Mapping[str, Any]],
    stage: Mapping[str, Any],
    prerequisite: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the retry-invariant identity for immutable stage results."""

    jobs = [
        _job_manifest_record(manifest, job_id)
        for job_id in stage["selected_job_ids"]
    ]
    return {
        "schema": STAGE_IDENTITY_SCHEMA,
        "package_manifest_sha256": package_manifest_sha256,
        "package_tree_sha256": canonical_sha256(list(package_files)),
        "source_sha256": manifest["source_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "verifier_sha256": manifest["verifier_sha256"],
        "stage_id": stage["stage_id"],
        "run_name": stage["run_name"],
        "selected_job_ids": list(stage["selected_job_ids"]),
        "job_manifest_sha256s": [row["sha256"] for row in jobs],
        "run_contract_sha256": canonical_sha256(manifest["run_contract"]),
        "prerequisite_stage1_receive": prerequisite,
        "retry_invariant": True,
        "diagnostic_only": True,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
    }


def _tree_object(
    *,
    tree_prefix: str,
    path: str,
    identity_source: str,
    sha256: str | None,
    size: int | None,
) -> dict[str, Any]:
    if (
        not isinstance(path, str)
        or path.startswith("/")
        or "\\" in path
        or ".." in Path(path).parts
        or identity_source
        not in {"preview", "artifact_upload", "done_envelope"}
    ):
        raise ValueError("runner tree object path or source changed")
    if (sha256 is None) != (size is None):
        raise ValueError("runner tree object identity must be wholly known")
    if sha256 is not None:
        _sha(sha256, "runner tree object sha256")
        if type(size) is not int or size <= 0:
            raise ValueError("runner tree object bytes changed")
    return {
        "path": path,
        "uri": f"{tree_prefix}/{path}",
        "identity_source": identity_source,
        "sha256": sha256,
        "bytes": size,
    }


def _job(
    *,
    manifest: Mapping[str, Any],
    package_manifest_sha256: str,
    stage: Mapping[str, Any],
    record: Mapping[str, Any],
    result_prefix: str,
    stage_identity_sha256: str,
    attempt_index: int,
) -> dict[str, Any]:
    role = record["source_role"]
    job_id = record["job_id"]
    indices = list(record["work_hand_indices"])
    root_records = [
        _root_record(manifest, source_role=role, hand_index=index)
        for index in indices
    ]
    output_prefix = f"{result_prefix}/jobs/{job_id}"
    progress_prefix = f"{result_prefix}/progress/jobs/{job_id}"
    tree_prefix = f"{output_prefix}/tree"
    control_records = _output_control_records(manifest, record)
    tree_objects = [
        _tree_object(
            tree_prefix=tree_prefix,
            path=row["path"],
            identity_source="preview",
            sha256=row["sha256"],
            size=row["bytes"],
        )
        for row in control_records
    ]
    for index, root in zip(indices, root_records, strict=True):
        tree_objects.extend(
            (
                _tree_object(
                    tree_prefix=tree_prefix,
                    path=root["path"],
                    identity_source="preview",
                    sha256=root["sha256"],
                    size=root["bytes"],
                ),
                _tree_object(
                    tree_prefix=tree_prefix,
                    path=f"hands/{role}/hand_{index:03d}.json",
                    identity_source="artifact_upload",
                    sha256=None,
                    size=None,
                ),
            )
        )
    tree_objects.append(
        _tree_object(
            tree_prefix=tree_prefix,
            path="DONE.json",
            identity_source="done_envelope",
            sha256=None,
            size=None,
        )
    )
    return {
        "job_id": job_id,
        "stage_id": stage["stage_id"],
        "run_name": stage["run_name"],
        "source_role": role,
        "attempt_index": attempt_index,
        "attempt_id": f"{stage['run_name']}|{job_id}|attempt-{attempt_index}",
        "logical_job_id": f"{stage['run_name']}|{job_id}",
        "stage_identity_sha256": stage_identity_sha256,
        "package_manifest_sha256": package_manifest_sha256,
        "runner_job_manifest": {
            "path": record["path"],
            "sha256": record["sha256"],
            "bytes": record["bytes"],
        },
        "work_hand_indices": indices,
        "root_records": root_records,
        "output_control_records": control_records,
        "upload_uris": [
            f"{output_prefix}/uploads/hand_{index:03d}.json"
            for index in indices
        ],
        "heartbeat_uris": [
            f"{progress_prefix}/heartbeats/{sequence:06d}.json"
            for sequence in range(1, len(indices) + 1)
        ],
        "done_uri": f"{output_prefix}/DONE.envelope.json",
        "tree_prefix": tree_prefix,
        "tree_object_manifest": tree_objects,
        "tree_layout": {
            "run_contract_path": "run_contract.json",
            "shard_manifest_path": "shard_manifest.json",
            "root_paths": [
                f"roots/hand_{index:03d}.json" for index in indices
            ],
            "source_hand_paths": [
                f"hands/{role}/hand_{index:03d}.json" for index in indices
            ],
            "done_path": "DONE.json",
            "runner_validate_completed_output_compatible": True,
        },
        "transport_fixture_only": True,
        "scientific_payload_present": False,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
    }


def build_preview(
    *,
    package_dir: str | Path,
    stage_id: str,
    attempt_index: int = 0,
    prior_preview: Mapping[str, Any] | None = None,
    prior_snapshot: Mapping[str, Any] | None = None,
    prerequisite_stage1_preview: Mapping[str, Any] | None = None,
    prerequisite_stage1_receive: Mapping[str, Any] | None = None,
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
    zone: str = DEFAULT_ZONE,
    observed_spot_price: float | None = None,
    _skip_validation: bool = False,
) -> dict[str, Any]:
    """Build an in-memory transport preview for exact diagnostic jobs."""

    if type(attempt_index) is not int or attempt_index not in range(MAX_ATTEMPTS):
        raise ValueError("attempt_index must be exact integer zero or one")
    if (
        project != DEFAULT_PROJECT
        or bucket != DEFAULT_BUCKET
        or zone != DEFAULT_ZONE
        or _SAFE_ID.fullmatch(project) is None
        or _SAFE_BUCKET.fullmatch(bucket) is None
    ):
        raise ValueError("diagnostic worker target escaped fixed project/bucket/zone")
    package = Path(package_dir).resolve()
    manifest = worker.validate_package(package)
    if manifest["source_entry_count"] != EXPECTED_SOURCE_ENTRY_COUNT:
        raise ValueError("diagnostic worker source must have exactly sixty entries")
    stage = _stage(manifest, stage_id)
    expected_jobs = (
        list(plan.STAGE1_JOB_IDS)
        if stage_id == plan.STAGE1_ID
        else list(plan.STAGE2_JOB_IDS)
        if stage_id == plan.STAGE2_ID
        else None
    )
    if expected_jobs is None or stage["selected_job_ids"] != expected_jobs:
        raise ValueError("diagnostic worker escaped exact stage jobs")
    package_sha = sha256_file(package / worker.MANIFEST_NAME)
    prerequisite = _validate_stage1_receive_bundle(
        package_manifest_sha256=package_sha,
        preview=prerequisite_stage1_preview,
        receive=prerequisite_stage1_receive,
    )
    if stage_id == plan.STAGE1_ID and prerequisite is not None:
        raise ValueError("stage1 cannot bind a stage1 prerequisite")
    if stage_id == plan.STAGE2_ID and prerequisite is None:
        raise ValueError("stage2 requires validated worker stage1 receive")
    package_files = _package_files(package)
    stage_identity = _build_stage_identity(
        manifest=manifest,
        package_manifest_sha256=package_sha,
        package_files=package_files,
        stage=stage,
        prerequisite=prerequisite,
    )
    stage_identity_sha = canonical_sha256(stage_identity)
    resume = _resume_binding(
        package_dir=package,
        stage_id=stage_id,
        attempt_index=attempt_index,
        prior_preview=prior_preview,
        prior_snapshot=prior_snapshot,
        prerequisite_stage1_preview=prerequisite_stage1_preview,
        prerequisite_stage1_receive=prerequisite_stage1_receive,
    )
    cost = legacy_vm._cost_guard(len(expected_jobs), observed_spot_price)
    stage_prefix = (
        f"gs://{bucket}/hu-m31-r2diag-worker-v1/{stage['run_name']}/"
        f"{package_sha[:16]}"
    )
    result_prefix = f"{stage_prefix}/results"
    control_prefix = f"{stage_prefix}/control/attempt-{attempt_index}"
    jobs = [
        _job(
            manifest=manifest,
            package_manifest_sha256=package_sha,
            stage=stage,
            record=_job_manifest_record(manifest, job_id),
            result_prefix=result_prefix,
            stage_identity_sha256=stage_identity_sha,
            attempt_index=attempt_index,
        )
        for job_id in expected_jobs
    ]
    stage_token = {
        "schema": STAGE_TOKEN_SCHEMA,
        "status": "transport_preview_only_cloud_not_authorized",
        "package_manifest_sha256": package_sha,
        "package_tree_sha256": canonical_sha256(package_files),
        "source_sha256": manifest["source_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "verifier_sha256": manifest["verifier_sha256"],
        "stage_id": stage_id,
        "run_name": stage["run_name"],
        "selected_job_ids": expected_jobs,
        "attempt_index": attempt_index,
        "prerequisite_stage1_receive": prerequisite,
        "resume_binding": resume,
        "diagnostic_only": True,
        "cloud_launch_authorized": False,
        "gcloud_invocation_authorized": False,
        "claim_written": False,
        "authorization_written": False,
        "remote_write_performed": False,
    }
    control = [
        {
            "uri": f"{stage_prefix}/source/{record['path']}",
            "path": record["path"],
            "sha256": record["sha256"],
            "bytes": record["bytes"],
        }
        for record in package_files
    ]
    control.append(
        {
            "uri": f"{control_prefix}/stage_token.json",
            "path": "control/stage_token.json",
            "sha256": canonical_sha256(stage_token),
            "bytes": len(canonical_bytes(stage_token)),
        }
    )
    remote_manifest = {
        "schema": REMOTE_MANIFEST_SCHEMA,
        "prefix": stage_prefix,
        "result_prefix": result_prefix,
        "attempt_control_prefix": control_prefix,
        "stage_identity_sha256": stage_identity_sha,
        "stage_id": stage_id,
        "run_name": stage["run_name"],
        "attempt_index": attempt_index,
        "freshness": {
            "attempt_control_prefix_must_be_empty": True,
            "result_prefix_is_retry_invariant": True,
            "existing_result_objects_require_identical_readback": True,
            "unknown_object_is_fatal": True,
            "conditional_create_generation_match": 0,
            "performed_by_adapter": False,
        },
        "control_objects": control,
        "job_objects": [
            {
                "job_id": job["job_id"],
                "upload_uris": job["upload_uris"],
                "heartbeat_uris": job["heartbeat_uris"],
                "done_uri": job["done_uri"],
                "tree_prefix": job["tree_prefix"],
                "tree_object_manifest": job["tree_object_manifest"],
            }
            for job in jobs
        ],
        "receive_uri": f"{result_prefix}/received/{stage_id}.json",
    }
    preview = {
        "schema": PREVIEW_SCHEMA,
        "status": "backend_neutral_preview_cloud_not_authorized",
        "adapter_schema": ADAPTER_SCHEMA,
        "package_manifest_sha256": package_sha,
        "package_files": package_files,
        "package_file_count": len(package_files),
        "source_entry_count": manifest["source_entry_count"],
        "stage_id": stage_id,
        "run_name": stage["run_name"],
        "selected_job_ids": expected_jobs,
        "vm_count": len(expected_jobs),
        "attempt_index": attempt_index,
        "max_attempts": MAX_ATTEMPTS,
        "stage_identity": stage_identity,
        "stage_identity_sha256": stage_identity_sha,
        "stage_token": stage_token,
        "stage_token_sha256": canonical_sha256(stage_token),
        "remote_manifest": remote_manifest,
        "remote_manifest_sha256": canonical_sha256(remote_manifest),
        "jobs": jobs,
        "cost_guard": cost,
        "cost_guard_sha256": canonical_sha256(cost),
        "capabilities": {
            "cloud_executable": False,
            "launch_ready": False,
            "cloud_launch_authorized": False,
            "gcloud_invocation_authorized": False,
            "subprocess_invocation_authorized": False,
            "claim_write_authorized": False,
            "authorization_write_authorized": False,
            "object_write_authorized": False,
            "vm_create_authorized": False,
            "remote_query_performed": False,
            "production_all20_launcher_reused": False,
            "current_profile_changed": False,
            "performance_lock_evidence": False,
            "quality_evidence": False,
            "training_eligible": False,
            "promotion_evidence": False,
        },
    }
    if _skip_validation:
        return preview
    return validate_preview(
        preview,
        package_dir=package,
        prior_preview=prior_preview,
        prior_snapshot=prior_snapshot,
        prerequisite_stage1_preview=prerequisite_stage1_preview,
        prerequisite_stage1_receive=prerequisite_stage1_receive,
    )


def validate_preview(
    value: Mapping[str, Any],
    *,
    package_dir: str | Path,
    prior_preview: Mapping[str, Any] | None = None,
    prior_snapshot: Mapping[str, Any] | None = None,
    prerequisite_stage1_preview: Mapping[str, Any] | None = None,
    prerequisite_stage1_receive: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    preview = dict(value)
    _exact(
        preview,
        {
            "schema",
            "status",
            "adapter_schema",
            "package_manifest_sha256",
            "package_files",
            "package_file_count",
            "source_entry_count",
            "stage_id",
            "run_name",
            "selected_job_ids",
            "vm_count",
            "attempt_index",
            "max_attempts",
            "stage_identity",
            "stage_identity_sha256",
            "stage_token",
            "stage_token_sha256",
            "remote_manifest",
            "remote_manifest_sha256",
            "jobs",
            "cost_guard",
            "cost_guard_sha256",
            "capabilities",
        },
        "diagnostic worker preview",
    )
    expected = build_preview(
        package_dir=package_dir,
        stage_id=str(preview.get("stage_id")),
        attempt_index=preview.get("attempt_index"),
        prior_preview=prior_preview,
        prior_snapshot=prior_snapshot,
        prerequisite_stage1_preview=prerequisite_stage1_preview,
        prerequisite_stage1_receive=prerequisite_stage1_receive,
        project=DEFAULT_PROJECT,
        bucket=DEFAULT_BUCKET,
        zone=DEFAULT_ZONE,
        observed_spot_price=preview.get("cost_guard", {}).get(
            "observed_spot_price_usd_per_vm_hour"
        ),
        _skip_validation=True,
    )
    if preview != expected:
        raise ValueError("diagnostic worker preview changed")
    uris = [
        row["uri"] for row in preview["remote_manifest"]["control_objects"]
    ]
    for job in preview["jobs"]:
        uris.extend(job["upload_uris"])
        uris.extend(job["heartbeat_uris"])
        uris.append(job["done_uri"])
        uris.extend(row["uri"] for row in job["tree_object_manifest"])
    uris.append(preview["remote_manifest"]["receive_uri"])
    if len(uris) != len(set(uris)):
        raise ValueError("diagnostic worker remote URI collision")
    _reject_hidden_or_q(preview)
    return preview


def _tree_records_for_files(
    job: Mapping[str, Any], files: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_path = {row["path"]: row for row in job["tree_object_manifest"]}
    records: list[dict[str, Any]] = []
    for file_record in files:
        spec = by_path.get(file_record["path"])
        if spec is None:
            raise ValueError("runner file escaped enumerated tree objects")
        if spec["sha256"] is not None and (
            spec["sha256"] != file_record["sha256"]
            or spec["bytes"] != file_record["bytes"]
        ):
            raise ValueError("runner file changed a preview-bound tree object")
        records.append(
            {
                "path": file_record["path"],
                "uri": spec["uri"],
                "sha256": file_record["sha256"],
                "bytes": file_record["bytes"],
            }
        )
    return records


def build_artifact_upload(
    preview: Mapping[str, Any],
    *,
    job_id: str,
    sequence: int,
    root_file: Mapping[str, Any],
    source_hand_file: Mapping[str, Any],
    transport_fixture_only: bool,
) -> dict[str, Any]:
    job = next((row for row in preview["jobs"] if row["job_id"] == job_id), None)
    if (
        job is None
        or type(sequence) is not int
        or not 1 <= sequence <= len(job["work_hand_indices"])
        or type(transport_fixture_only) is not bool
    ):
        raise ValueError("runner artifact upload escaped exact job sequence")
    index = job["work_hand_indices"][sequence - 1]
    root = _validate_artifact(
        root_file,
        source_role=job["source_role"],
        hand_index=index,
        expected_path=f"roots/hand_{index:03d}.json",
        label="runner root artifact",
    )
    hand = _validate_artifact(
        source_hand_file,
        source_role=job["source_role"],
        hand_index=index,
        expected_path=f"hands/{job['source_role']}/hand_{index:03d}.json",
        label="runner source-hand artifact",
    )
    if root != job["root_records"][sequence - 1]:
        raise ValueError("runner root artifact does not match package root")
    if transport_fixture_only and hand != _fixture_file(
        source_role=job["source_role"], hand_index=index
    ):
        raise ValueError("transport-only source-hand fixture changed")
    tree_objects = _tree_records_for_files(job, [root, hand])
    value = {
        "schema": UPLOAD_SCHEMA,
        "stage_id": preview["stage_id"],
        "run_name": preview["run_name"],
        "stage_identity_sha256": preview["stage_identity_sha256"],
        "job_id": job_id,
        "source_role": job["source_role"],
        "logical_job_id": job["logical_job_id"],
        "runner_job_manifest_sha256": job["runner_job_manifest"]["sha256"],
        "sequence": sequence,
        "hand_index": index,
        "files": [root, hand],
        "files_sha256": canonical_sha256([root, hand]),
        "tree_object_records": tree_objects,
        "tree_object_records_sha256": canonical_sha256(tree_objects),
        "transport_fixture_only": transport_fixture_only,
        "scientific_payload_present": False,
        "runner_content_validated": False,
        "diagnostic_only": True,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
    }
    _reject_hidden_or_q(value)
    return value


def build_fixture_upload(
    preview: Mapping[str, Any], *, job_id: str, sequence: int
) -> dict[str, Any]:
    job = next((row for row in preview["jobs"] if row["job_id"] == job_id), None)
    if (
        job is None
        or type(sequence) is not int
        or not 1 <= sequence <= len(job["work_hand_indices"])
    ):
        raise ValueError("fixture upload escaped exact job")
    index = job["work_hand_indices"][sequence - 1]
    return build_artifact_upload(
        preview,
        job_id=job_id,
        sequence=sequence,
        root_file=job["root_records"][sequence - 1],
        source_hand_file=_fixture_file(
            source_role=job["source_role"], hand_index=index
        ),
        transport_fixture_only=True,
    )


def build_heartbeat(
    preview: Mapping[str, Any],
    *,
    job_id: str,
    uploads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    job = next((row for row in preview["jobs"] if row["job_id"] == job_id), None)
    if job is None or not uploads or len(uploads) > len(job["work_hand_indices"]):
        raise ValueError("runner heartbeat prefix changed")
    checked: list[dict[str, Any]] = []
    for sequence, upload in enumerate(uploads, 1):
        files = upload.get("files", [])
        expected = build_artifact_upload(
            preview,
            job_id=job_id,
            sequence=sequence,
            root_file=files[0] if len(files) == 2 else {},
            source_hand_file=files[1] if len(files) == 2 else {},
            transport_fixture_only=upload.get("transport_fixture_only"),
        )
        if dict(upload) != expected:
            raise ValueError("runner artifact upload changed before heartbeat")
        checked.append(expected)
    sequence = len(checked)
    return {
        "schema": HEARTBEAT_SCHEMA,
        "stage_id": preview["stage_id"],
        "run_name": preview["run_name"],
        "stage_identity_sha256": preview["stage_identity_sha256"],
        "job_id": job_id,
        "logical_job_id": job["logical_job_id"],
        "runner_job_manifest_sha256": job["runner_job_manifest"]["sha256"],
        "sequence": sequence,
        "completed_hand_indices": job["work_hand_indices"][:sequence],
        "pending_hand_indices": job["work_hand_indices"][sequence:],
        "latest_upload_sha256": canonical_sha256(checked[-1]),
        "transport_fixture_only": all(
            row["transport_fixture_only"] for row in checked
        ),
        "diagnostic_only": True,
    }


def _fixture_done_file(job: Mapping[str, Any], artifacts: Sequence[Any]) -> dict[str, Any]:
    identity = {
        "schema": FIXTURE_SCHEMA,
        "source_role": job["source_role"],
        "path": "DONE.json",
        "artifact_manifest_sha256": canonical_sha256(artifacts),
        "transport_fixture_only": True,
        "scientific_payload_present": False,
    }
    raw = canonical_bytes(identity)
    return {
        "path": "DONE.json",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def build_done(
    preview: Mapping[str, Any],
    *,
    job_id: str,
    uploads: Sequence[Mapping[str, Any]],
    heartbeats: Sequence[Mapping[str, Any]],
    runner_done_file: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    job = next((row for row in preview["jobs"] if row["job_id"] == job_id), None)
    if (
        job is None
        or len(uploads) != len(job["work_hand_indices"])
        or len(heartbeats) != len(uploads)
    ):
        raise ValueError("runner DONE requires every upload and heartbeat")
    checked_uploads: list[dict[str, Any]] = []
    checked_heartbeats: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for sequence, (upload, heartbeat) in enumerate(
        zip(uploads, heartbeats, strict=True), 1
    ):
        files = upload.get("files", [])
        checked_upload = build_artifact_upload(
            preview,
            job_id=job_id,
            sequence=sequence,
            root_file=files[0] if len(files) == 2 else {},
            source_hand_file=files[1] if len(files) == 2 else {},
            transport_fixture_only=upload.get("transport_fixture_only"),
        )
        if dict(upload) != checked_upload:
            raise ValueError("runner upload changed before DONE")
        checked_uploads.append(checked_upload)
        expected_heartbeat = build_heartbeat(
            preview, job_id=job_id, uploads=checked_uploads
        )
        if dict(heartbeat) != expected_heartbeat:
            raise ValueError("runner heartbeat changed before DONE")
        checked_heartbeats.append(expected_heartbeat)
        artifacts.extend(checked_upload["files"])
    if runner_done_file is None:
        done_file = _fixture_done_file(job, artifacts)
    else:
        done_file = dict(runner_done_file)
        _exact(done_file, {"path", "sha256", "bytes"}, "runner DONE file")
        if (
            done_file["path"] != "DONE.json"
            or _sha(done_file["sha256"], "runner DONE file sha256")
            != done_file["sha256"]
            or type(done_file["bytes"]) is not int
            or done_file["bytes"] <= 0
        ):
            raise ValueError("runner DONE file identity changed")
    tree_records = [
        *job["output_control_records"],
        *artifacts,
        done_file,
    ]
    paths = [row["path"] for row in tree_records]
    expected_paths = [
        "run_contract.json",
        "shard_manifest.json",
        *[
            path
            for index in job["work_hand_indices"]
            for path in (
                f"roots/hand_{index:03d}.json",
                f"hands/{job['source_role']}/hand_{index:03d}.json",
            )
        ],
        "DONE.json",
    ]
    if paths != expected_paths or len(paths) != len(set(paths)):
        raise ValueError("runner output tree layout changed")
    tree_object_records = _tree_records_for_files(job, tree_records)
    if [row["path"] for row in tree_object_records] != expected_paths:
        raise ValueError("runner tree object manifest order changed")
    fixture_only = all(
        upload["transport_fixture_only"] for upload in checked_uploads
    ) and runner_done_file is None
    value = {
        "schema": DONE_SCHEMA,
        "stage_id": preview["stage_id"],
        "run_name": preview["run_name"],
        "stage_identity_sha256": preview["stage_identity_sha256"],
        "job_id": job_id,
        "source_role": job["source_role"],
        "logical_job_id": job["logical_job_id"],
        "runner_job_manifest_sha256": job["runner_job_manifest"]["sha256"],
        "completed_hand_indices": job["work_hand_indices"],
        "upload_sha256s": [
            canonical_sha256(row) for row in checked_uploads
        ],
        "heartbeat_sha256s": [
            canonical_sha256(row) for row in checked_heartbeats
        ],
        "runner_artifact_manifest": artifacts,
        "runner_artifact_manifest_sha256": canonical_sha256(artifacts),
        "tree_records": tree_records,
        "tree_records_sha256": canonical_sha256(tree_records),
        "tree_object_records": tree_object_records,
        "tree_object_records_sha256": canonical_sha256(tree_object_records),
        "tree_layout_compatible_with_runner_validator": True,
        "runner_content_validation_deferred_until_real_receive": True,
        "done_published_last": True,
        "readback_required": True,
        "transport_fixture_only": fixture_only,
        "scientific_payload_present": False,
        "diagnostic_only": True,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
    }
    _reject_hidden_or_q(value)
    return value


def _validate_done(
    preview: Mapping[str, Any], job: Mapping[str, Any], value: Mapping[str, Any]
) -> dict[str, Any]:
    done = dict(value)
    artifacts = done.get("runner_artifact_manifest")
    if not isinstance(artifacts, list) or len(artifacts) != (
        len(job["work_hand_indices"]) * 2
    ):
        raise ValueError("runner DONE artifact manifest length changed")
    uploads = [
        build_artifact_upload(
            preview,
            job_id=job["job_id"],
            sequence=sequence,
            root_file=artifacts[(sequence - 1) * 2],
            source_hand_file=artifacts[(sequence - 1) * 2 + 1],
            transport_fixture_only=done.get("transport_fixture_only"),
        )
        for sequence in range(1, len(job["work_hand_indices"]) + 1)
    ]
    heartbeats = [
        build_heartbeat(
            preview, job_id=job["job_id"], uploads=uploads[:sequence]
        )
        for sequence in range(1, len(uploads) + 1)
    ]
    runner_done_file = (
        None
        if done.get("transport_fixture_only") is True
        else done.get("tree_records", [{}])[-1]
    )
    expected = build_done(
        preview,
        job_id=job["job_id"],
        uploads=uploads,
        heartbeats=heartbeats,
        runner_done_file=runner_done_file,
    )
    if done != expected:
        raise ValueError("runner DONE envelope changed")
    return done


def build_receive(
    preview: Mapping[str, Any],
    *,
    done_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if [row.get("job_id") for row in done_records] != preview["selected_job_ids"]:
        raise ValueError("worker receive requires exact ordered stage jobs")
    checked = [
        _validate_done(preview, job, done)
        for job, done in zip(preview["jobs"], done_records, strict=True)
    ]
    return {
        "schema": RECEIVE_SCHEMA,
        "status": "transport_tree_identities_received_content_validation_deferred",
        "package_manifest_sha256": preview["package_manifest_sha256"],
        "stage_identity_sha256": preview["stage_identity_sha256"],
        "stage_id": preview["stage_id"],
        "run_name": preview["run_name"],
        "selected_job_ids": preview["selected_job_ids"],
        "done_records": checked,
        "done_sha256s": [canonical_sha256(row) for row in checked],
        "tree_record_sha256s": [
            row["tree_records_sha256"] for row in checked
        ],
        "tree_object_record_sha256s": [
            row["tree_object_records_sha256"] for row in checked
        ],
        "prerequisite_stage1_receive": preview["stage_token"][
            "prerequisite_stage1_receive"
        ],
        "runner_content_validation_deferred": True,
        "transport_fixture_only": all(
            row["transport_fixture_only"] for row in checked
        ),
        "scientific_payload_present": False,
        "diagnostic_only": True,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "cloud_receive_performed": False,
        "remote_write_performed": False,
    }


def validate_receive(
    value: Mapping[str, Any], *, preview: Mapping[str, Any]
) -> dict[str, Any]:
    receive = dict(value)
    done_records = receive.get("done_records")
    if not isinstance(done_records, list):
        raise ValueError("worker receive DONE records are missing")
    expected = build_receive(preview, done_records=done_records)
    if receive != expected:
        raise ValueError("worker receive identity changed")
    _reject_hidden_or_q(receive)
    return receive


def build_materialization_manifest(
    preview: Mapping[str, Any],
    *,
    done_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Return exact remote objects needed to materialize one runner tree.

    This function performs no reads or writes.  A future receiver must download
    every URI, verify byte length and SHA-256, write only the listed relative
    paths beneath a fresh directory, and then call
    ``runner.validate_completed_output``.  Identity envelopes alone are never
    treated as file contents.
    """

    job_id = done_record.get("job_id")
    job = next((row for row in preview["jobs"] if row["job_id"] == job_id), None)
    if job is None:
        raise ValueError("materialization job escaped exact stage")
    done = _validate_done(preview, job, done_record)
    objects = list(done["tree_object_records"])
    if [row["uri"] for row in objects] != [
        row["uri"] for row in job["tree_object_manifest"]
    ]:
        raise ValueError("materialization tree URI order changed")
    if len({row["path"] for row in objects}) != len(objects) or len(
        {row["uri"] for row in objects}
    ) != len(objects):
        raise ValueError("materialization tree contains a collision")
    return {
        "schema": MATERIALIZATION_SCHEMA,
        "package_manifest_sha256": preview["package_manifest_sha256"],
        "stage_identity_sha256": preview["stage_identity_sha256"],
        "stage_id": preview["stage_id"],
        "run_name": preview["run_name"],
        "job_id": job_id,
        "source_role": job["source_role"],
        "runner_job_manifest_sha256": job["runner_job_manifest"]["sha256"],
        "objects": objects,
        "objects_sha256": canonical_sha256(objects),
        "fresh_destination_required": True,
        "reject_symlink_or_path_escape": True,
        "verify_bytes_and_sha256_before_write": True,
        "validate_completed_output_required": True,
        "content_is_not_embedded_in_envelope": True,
        "remote_read_performed": False,
        "local_write_performed": False,
        "runner_validation_performed": False,
        "diagnostic_only": True,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
    }


def empty_snapshot(preview: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": SNAPSHOT_SCHEMA,
        "preview_sha256": canonical_sha256(preview),
        "stage_id": preview["stage_id"],
        "run_name": preview["run_name"],
        "attempt_index": preview["attempt_index"],
        "objects": [],
        "active_instance_names": [],
        "cloud_query_performed": False,
        "cloud_write_performed": False,
        "diagnostic_fixture_only": True,
    }


def _record(uri: str, content: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "uri": uri,
        "generation": 1,
        "content_sha256": canonical_sha256(content),
        "content": dict(content),
    }


def fake_publish(
    preview: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a new in-memory snapshot with conditional-create semantics."""

    value = deepcopy(dict(snapshot))
    existing = {row["uri"]: row for row in value["objects"]}
    for raw in records:
        record = dict(raw)
        old = existing.get(record.get("uri"))
        if old is not None:
            if old != record:
                raise ValueError("fake remote different-content collision")
            continue
        value["objects"].append(record)
        existing[record["uri"]] = record
    validate_snapshot(preview, value)
    return value


def fixture_progress(
    preview: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    *,
    job_id: str,
    completed_count: int,
) -> dict[str, Any]:
    job = next((row for row in preview["jobs"] if row["job_id"] == job_id), None)
    if (
        job is None
        or type(completed_count) is not int
        or not 1 <= completed_count <= len(job["work_hand_indices"])
    ):
        raise ValueError("fixture progress count changed")
    uploads: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for sequence in range(1, completed_count + 1):
        upload = build_fixture_upload(
            preview, job_id=job_id, sequence=sequence
        )
        uploads.append(upload)
        heartbeat = build_heartbeat(
            preview, job_id=job_id, uploads=uploads
        )
        records.extend(
            (
                _record(job["upload_uris"][sequence - 1], upload),
                _record(job["heartbeat_uris"][sequence - 1], heartbeat),
            )
        )
    return fake_publish(preview, snapshot, records)


def fixture_done(
    preview: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    *,
    job_id: str,
) -> dict[str, Any]:
    job = next((row for row in preview["jobs"] if row["job_id"] == job_id), None)
    if job is None:
        raise ValueError("unknown fixture DONE job")
    by_uri = {row["uri"]: row["content"] for row in snapshot["objects"]}
    uploads = [by_uri.get(uri) for uri in job["upload_uris"]]
    heartbeats = [by_uri.get(uri) for uri in job["heartbeat_uris"]]
    if any(row is None for row in (*uploads, *heartbeats)):
        raise ValueError("fixture DONE cannot precede full artifact readback")
    done = build_done(
        preview,
        job_id=job_id,
        uploads=uploads,
        heartbeats=heartbeats,
    )
    return fake_publish(
        preview, snapshot, [_record(job["done_uri"], done)]
    )


def fixture_receive(
    preview: Mapping[str, Any], snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    by_uri = {row["uri"]: row["content"] for row in snapshot["objects"]}
    dones = [by_uri.get(job["done_uri"]) for job in preview["jobs"]]
    if any(done is None for done in dones):
        raise ValueError("fixture receive cannot precede every DONE readback")
    receive = build_receive(preview, done_records=dones)
    return fake_publish(
        preview,
        snapshot,
        [_record(preview["remote_manifest"]["receive_uri"], receive)],
    )


def validate_snapshot(
    preview: Mapping[str, Any], snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    value = dict(snapshot)
    _exact(
        value,
        {
            "schema",
            "preview_sha256",
            "stage_id",
            "run_name",
            "attempt_index",
            "objects",
            "active_instance_names",
            "cloud_query_performed",
            "cloud_write_performed",
            "diagnostic_fixture_only",
        },
        "diagnostic fake remote snapshot",
    )
    if (
        value["schema"] != SNAPSHOT_SCHEMA
        or value["preview_sha256"] != canonical_sha256(preview)
        or value["stage_id"] != preview["stage_id"]
        or value["run_name"] != preview["run_name"]
        or value["attempt_index"] != preview["attempt_index"]
        or value["active_instance_names"] != []
        or value["cloud_query_performed"] is not False
        or value["cloud_write_performed"] is not False
        or value["diagnostic_fixture_only"] is not True
        or not isinstance(value["objects"], list)
    ):
        raise ValueError("diagnostic fake remote boundary changed")
    allowed: set[str] = {preview["remote_manifest"]["receive_uri"]}
    for job in preview["jobs"]:
        allowed.update(job["upload_uris"])
        allowed.update(job["heartbeat_uris"])
        allowed.add(job["done_uri"])
    by_uri: dict[str, dict[str, Any]] = {}
    order: dict[str, int] = {}
    for position, raw in enumerate(value["objects"]):
        if not isinstance(raw, Mapping):
            raise ValueError("fake remote object must be an object")
        record = dict(raw)
        _exact(
            record,
            {"uri", "generation", "content_sha256", "content"},
            "fake remote object",
        )
        uri = record["uri"]
        if (
            uri not in allowed
            or uri in by_uri
            or type(record["generation"]) is not int
            or record["generation"] <= 0
            or record["content_sha256"] != canonical_sha256(record["content"])
            or not isinstance(record["content"], Mapping)
        ):
            raise ValueError("fake remote unknown object or collision")
        by_uri[uri] = record
        order[uri] = position
    states: list[dict[str, Any]] = []
    done_rows: list[dict[str, Any]] = []
    for job in preview["jobs"]:
        uploads: list[dict[str, Any]] = []
        heartbeats: list[dict[str, Any]] = []
        pending_heartbeat_sequence: int | None = None
        for sequence, (upload_uri, heartbeat_uri) in enumerate(
            zip(job["upload_uris"], job["heartbeat_uris"], strict=True), 1
        ):
            upload_record = by_uri.get(upload_uri)
            heartbeat_record = by_uri.get(heartbeat_uri)
            if upload_record is None and heartbeat_record is not None:
                raise ValueError("fake remote heartbeat has no artifact upload")
            if upload_record is None:
                if any(
                    uri in by_uri
                    for uri in (
                        job["upload_uris"][sequence:]
                        + job["heartbeat_uris"][sequence:]
                    )
                ):
                    raise ValueError("fake remote artifacts are not contiguous")
                break
            upload = upload_record["content"]
            files = upload.get("files", [])
            expected_upload = build_artifact_upload(
                preview,
                job_id=job["job_id"],
                sequence=sequence,
                root_file=files[0] if len(files) == 2 else {},
                source_hand_file=files[1] if len(files) == 2 else {},
                transport_fixture_only=upload.get("transport_fixture_only"),
            )
            if upload != expected_upload:
                raise ValueError("fake remote artifact content changed")
            uploads.append(expected_upload)
            if heartbeat_record is None:
                if any(
                    uri in by_uri
                    for uri in (
                        job["upload_uris"][sequence:]
                        + job["heartbeat_uris"][sequence:]
                        + [job["done_uri"]]
                    )
                ):
                    raise ValueError(
                        "fake remote pending heartbeat is not the trailing frontier"
                    )
                pending_heartbeat_sequence = sequence
                break
            expected_heartbeat = build_heartbeat(
                preview, job_id=job["job_id"], uploads=uploads
            )
            if heartbeat_record["content"] != expected_heartbeat:
                raise ValueError("fake remote heartbeat changed")
            if order[upload_uri] >= order[heartbeat_uri]:
                raise ValueError("fake remote heartbeat preceded upload")
            heartbeats.append(expected_heartbeat)
        done_record = by_uri.get(job["done_uri"])
        if done_record is not None:
            done = _validate_done(preview, job, done_record["content"])
            if (
                len(heartbeats) != len(job["work_hand_indices"])
                or pending_heartbeat_sequence is not None
            ):
                raise ValueError("fake remote DONE was published early")
            prior_positions = [
                order[uri]
                for uri in (*job["upload_uris"], *job["heartbeat_uris"])
            ]
            if order[job["done_uri"]] <= max(prior_positions):
                raise ValueError("fake remote DONE was not published last")
            done_rows.append(done)
            state = "success_tree_ready"
        else:
            state = "clean" if not uploads else "crashed_resumable"
        states.append(
            {
                "job_id": job["job_id"],
                "uploaded_sequences": len(uploads),
                "completed_sequences": len(heartbeats),
                "pending_heartbeat_sequence": pending_heartbeat_sequence,
                "state": state,
            }
        )
    receive_record = by_uri.get(preview["remote_manifest"]["receive_uri"])
    if receive_record is not None:
        expected_receive = build_receive(preview, done_records=done_rows)
        if receive_record["content"] != expected_receive:
            raise ValueError("fake remote receive changed or was early")
        if order[preview["remote_manifest"]["receive_uri"]] <= max(
            order[job["done_uri"]] for job in preview["jobs"]
        ):
            raise ValueError("fake remote receive preceded DONE readback")
    return {
        "schema": SNAPSHOT_VALIDATION_SCHEMA,
        "preview_sha256": canonical_sha256(preview),
        "attempt_index": preview["attempt_index"],
        "job_states": states,
        "receive_validated": receive_record is not None,
        "active_instance_names": [],
        "unknown_object_count": 0,
        "cloud_query_performed": False,
        "cloud_write_performed": False,
        "diagnostic_only": True,
    }


def audit(*, package_dir: str | Path) -> dict[str, Any]:
    manifest = worker.validate_package(package_dir)
    stage1 = build_preview(package_dir=package_dir, stage_id=plan.STAGE1_ID)
    if (
        manifest["source_entry_count"] != EXPECTED_SOURCE_ENTRY_COUNT
        or stage1["package_file_count"] != EXPECTED_PACKAGE_FILE_COUNT
    ):
        raise ValueError("diagnostic worker adapter package binding changed")
    return {
        "schema": ADAPTER_SCHEMA,
        "status": "backend_neutral_transport_ready_cloud_not_authorized",
        "package_manifest_sha256": stage1["package_manifest_sha256"],
        "package_file_count": EXPECTED_PACKAGE_FILE_COUNT,
        "source_entry_count": EXPECTED_SOURCE_ENTRY_COUNT,
        "stage_job_counts": [1, 2],
        "attempt_indices": [0, 1],
        "runner_tree_layout_bound": True,
        "synthetic_q_payload_permitted": False,
        "transport_fixture_scientific_evidence": False,
        "gcloud_callable": False,
        "remote_write_callable": False,
        "claim_or_authorization_writer_present": False,
        "vm_create_callable": False,
        "launch_ready": False,
        "cloud_executable": False,
        "current_profile_changed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument("--package-dir", type=Path, required=True)
    preview_parser = commands.add_parser("preview-stage1")
    preview_parser.add_argument("--package-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "audit":
        value = audit(package_dir=args.package_dir)
    else:
        value = build_preview(
            package_dir=args.package_dir, stage_id=plan.STAGE1_ID
        )
    print(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ADAPTER_SCHEMA",
    "PREVIEW_SCHEMA",
    "SNAPSHOT_SCHEMA",
    "audit",
    "build_artifact_upload",
    "build_done",
    "build_fixture_upload",
    "build_heartbeat",
    "build_materialization_manifest",
    "build_preview",
    "build_receive",
    "empty_snapshot",
    "fake_publish",
    "fixture_done",
    "fixture_progress",
    "fixture_receive",
    "main",
    "validate_preview",
    "validate_receive",
    "validate_snapshot",
]
