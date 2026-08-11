"""Fail-closed Spot packaging and receive lifecycle for Attempt08 development200.

Packaging is local-only and cannot open a root.  A separately emitted
development-open authorization from the bounded Attempt08 preflight is bound
into the package, then a second immutable launch authorization binds the
finished package.  Workers use one root per shard and publish operational DONE
last.  Receivers may address teacher content only after all 200 DONE records
are validated and both local and remote copies of the consumption claim exist.
Selection remains a one-shot, no-clobber command after the verified merge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .finalize_hu_m43_attempt08_preflight import (
    ATTEMPT08_DEVELOPMENT_OPEN_AUTHORIZATION_SCHEMA,
    ATTEMPT08_PREFLIGHT_FINALIZATION_SCHEMA,
    ATTEMPT08_PREFLIGHT_GO_STATUS,
    load_and_validate_development_open_authorization,
    validate_preflight_execution_evidence,
)
from .aggregate_hu_m43_attempt08_preflight import (
    ATTEMPT08_PREFLIGHT_AGGREGATE_SCHEMA,
    load_and_validate_preflight_aggregate,
    validate_preflight_aggregate_with_proofs,
)
from .hu_m43_attempt06_spot import (
    PINNED_MODEL_MANIFEST_SHA256,
    PINNED_NATIVE_MANIFEST_SHA256,
    PINNED_TEMPLATE_RUN,
    _copy_file,
    _copy_manifest_files,
    _copy_source_tree,
    _verify_pinned_runtime_closure,
    _write_deterministic_zip,
)
from .hu_m43_attempt08_contract import (
    AI_PROFILES_SHA256,
    ATTEMPT08_LAMBDA_MODEL_SHA256,
    M43_ATTEMPT08_PLAN_SHA256,
    M43_ATTEMPT08_PROFILES,
    enumerate_attempt08_seed_schedules,
    load_and_validate_attempt08_plan,
)
from .run_hu_m43_attempt08_development import (
    ATTEMPT08_CHECKPOINT_SCHEMA,
    ATTEMPT08_HEARTBEAT_SCHEMA,
    ATTEMPT08_SHARD_ROW_SCHEMA,
    ATTEMPT08_SHARD_SUMMARY_SCHEMA,
    load_attempt08_development_open_bindings,
)
from .run_hu_m43_attempt08_preflight import (
    ATTEMPT08_PREFLIGHT_SLOTS,
    DEFAULT_SOURCE_PATH as DEFAULT_PREFLIGHT_SOURCE_PATH,
)
from .hu_m43_attempt08_preflight_spot import (
    build_preflight_schedule,
    validate_preflight_receive_bundle,
)
from .hu_m43_attempt08_runtime_anchor import (
    ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
    ATTEMPT08_RUNTIME_SEMANTIC_FILES,
    validate_runtime_semantic_anchor,
)
from .hu_m43_attempt08_runtime_anchor_contract import (
    ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,
    ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
    ATTEMPT08_RUNTIME_SOURCE_FILE_COUNT,
)
from .hu_m43_attempt08_runtime_identity import (
    ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
    ATTEMPT08_GCP_IMAGE_ID,
    ATTEMPT08_GCP_IMAGE_NAME,
    ATTEMPT08_GCP_IMAGE_SELF_LINK,
)
from .select_hu_m43_attempt08_development import (
    _validate_row,
    _SELECTOR_WRITE_LIFECYCLE_TOKEN,
    select_attempt08_development,
    write_attempt08_development_decision,
)


PACKAGE_MANIFEST_SCHEMA = "hu_m43_attempt08_development_spot_package_v1"
SOURCE_CLOSURE_SCHEMA = "hu_m43_attempt08_development_source_closure_v1"
SHARD_SCHEMA = "hu_m43_attempt08_development_spot_shard_v1"
PACKAGE_RESULT_SCHEMA = "hu_m43_attempt08_development_package_result_v1"
LAUNCH_AUTHORIZATION_SCHEMA = (
    "hu_m43_attempt08_development_spot_launch_authorization_v1"
)
DONE_SCHEMA = "hu_m43_attempt08_development_spot_done_v1"
GLOBAL_CLAIM_SCHEMA = "hu_m43_attempt08_development_global_claim_v1"
ROOT_CLAIM_SCHEMA = "hu_m43_attempt08_development_root_claim_v1"
OUTPUT_CONSUMPTION_SCHEMA = (
    "hu_m43_attempt08_development_output_consumption_v1"
)
RECEIVE_AUDIT_SCHEMA = "hu_m43_attempt08_development_received_shard_v1"
RECEIVE_MERGE_SCHEMA = "hu_m43_attempt08_development_receive_merge_v1"
SELECTOR_CLAIM_SCHEMA = "hu_m43_attempt08_development_selector_claim_v1"
SELECTOR_EXECUTION_SCHEMA = "hu_m43_attempt08_development_selector_execution_v1"
SELECTOR_RECEIPT_SCHEMA = "hu_m43_attempt08_development_selector_receipt_v1"

EXPECTED_SHARDS = 200
ROOTS_PER_SHARD = 1
NATIVE_BATCH_THREADS = 4
MAX_WAVE_SHARDS = 25
RECOMMENDED_MACHINE_TYPE = "c4-highmem-4"
ROOT_REMATERIALIZATION_MODE = (
    "same_root_index_same_six_seeds_same_frozen_closure_after_matching_claim"
)
RUN_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,126}[a-z0-9]$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_PACKAGE_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "plan_sha256",
        "model_sha256",
        "ai_profiles_sha256",
        "development_open_authorization_sha256",
        "preflight_plan_sha256",
        "preflight_result_sha256",
        "preflight_proof_evidence_sha256",
        "preflight_execution_evidence_sha256",
        "preflight_aggregate_sha256",
        "preflight_finalization_sha256",
        "preflight_source_sha256",
        "preflight_proof_sha256",
        "runtime_semantic_anchor_sha256",
        "runtime_source_closure_sha256",
        "runtime_fingerprint_sha256",
        "runtime_requirements_sha256",
        "gcp_image_name",
        "gcp_image_id",
        "gcp_image_self_link",
        "schedule_sha256",
        "source_closure_sha256",
        "source_zip_sha256",
        "startup_sha256",
        "source_model_manifest_sha256",
        "source_native_manifest_sha256",
        "total_roots",
        "total_shards",
        "roots_per_shard",
        "root_profile_assignment",
        "batch_child_selectors",
        "native_batch_threads",
        "recommended_machine_type",
        "recommended_wave_shards",
        "fresh_root_opened",
        "teacher_executed",
        "gcloud_invoked",
        "instances_created",
        "selector_executed",
        "future_audit_authorized",
        "fit_performed",
        "threshold_selected",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
)
_LAUNCH_KEYS = frozenset(
    {
        "schema",
        "status",
        "spot_authorized",
        "run_name",
        "manifest_sha256",
        "plan_sha256",
        "model_sha256",
        "ai_profiles_sha256",
        "development_open_authorization_sha256",
        "preflight_plan_sha256",
        "preflight_result_sha256",
        "preflight_proof_evidence_sha256",
        "preflight_execution_evidence_sha256",
        "preflight_aggregate_sha256",
        "preflight_finalization_sha256",
        "preflight_source_sha256",
        "preflight_proof_sha256",
        "runtime_semantic_anchor_sha256",
        "runtime_source_closure_sha256",
        "runtime_fingerprint_sha256",
        "runtime_requirements_sha256",
        "gcp_image_name",
        "gcp_image_id",
        "gcp_image_self_link",
        "schedule_sha256",
        "source_closure_sha256",
        "source_zip_sha256",
        "startup_sha256",
        "source_model_manifest_sha256",
        "source_native_manifest_sha256",
        "total_roots",
        "total_shards",
        "roots_per_shard",
        "native_batch_threads",
        "max_wave_shards",
        "machine_type",
        "package_frozen_before_authorization",
        "development_started",
        "fresh_root_opened",
        "selector_executed",
        "future_audit_authorized",
        "fit_performed",
        "threshold_selected",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
)
_DONE_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "run_id",
        "shard",
        "root_index",
        "root_profile",
        "seeds",
        "output_prefix",
        "manifest_sha256",
        "launch_authorization_sha256",
        "development_open_authorization_sha256",
        "source_sha256",
        "startup_sha256",
        "schedule_sha256",
        "runtime_semantic_anchor_sha256",
        "runtime_source_closure_sha256",
        "runtime_fingerprint_sha256",
        "runtime_requirements_sha256",
        "gcp_image_name",
        "gcp_image_id",
        "gcp_image_self_link",
        "plan_sha256",
        "model_sha256",
        "ai_profiles_sha256",
        "source_closure_sha256",
        "source_model_manifest_sha256",
        "source_native_manifest_sha256",
        "preflight_aggregate_sha256",
        "preflight_finalization_sha256",
        "preflight_proof_sha256",
        "preflight_execution_evidence_sha256",
        "global_claim_sha256",
        "root_claim_sha256",
        "output_sha256",
        "checkpoint_sha256",
        "heartbeat_sha256",
        "generator_summary_sha256",
        "run_log_sha256",
        "boot_image_evidence_sha256",
        "time_report_sha256",
        "resume_commit_sha256",
        "config_sha256",
        "teacher_generator_elapsed_seconds",
        "process_elapsed_seconds",
        "peak_rss_bytes",
        "native_batch_threads",
        "teacher_values_are_realized_match_ev",
        "selector_executed",
        "future_audit_authorized",
        "fit_performed",
        "threshold_selected",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
)
_CONSUMPTION_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "manifest_sha256",
        "launch_authorization_sha256",
        "development_open_authorization_sha256",
        "schedule_sha256",
        "done_sha256",
        "expected_shards",
        "expected_roots",
        "all_done_markers_verified",
        "result_objects_addressed_when_claimed",
        "remote_claim_required_before_content_read",
        "selector_executed",
        "future_audit_authorized",
        "fit_performed",
        "threshold_selected",
        "current_profile_mutated",
        "runtime_policy_activated",
        "runtime_semantic_anchor_sha256",
        "preflight_execution_evidence_sha256",
        "runtime_source_closure_sha256",
        "runtime_fingerprint_sha256",
        "runtime_requirements_sha256",
        "gcp_image_name",
        "gcp_image_id",
        "gcp_image_self_link",
    }
)
_GLOBAL_CLAIM_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "manifest_sha256",
        "launch_authorization_sha256",
        "development_open_authorization_sha256",
        "source_sha256",
        "startup_sha256",
        "schedule_sha256",
        "plan_sha256",
        "model_sha256",
        "ai_profiles_sha256",
        "source_closure_sha256",
        "source_model_manifest_sha256",
        "source_native_manifest_sha256",
        "preflight_aggregate_sha256",
        "preflight_finalization_sha256",
        "preflight_proof_sha256",
        "runtime_semantic_anchor_sha256",
        "preflight_execution_evidence_sha256",
        "runtime_source_closure_sha256",
        "runtime_fingerprint_sha256",
        "runtime_requirements_sha256",
        "gcp_image_name",
        "gcp_image_id",
        "gcp_image_self_link",
        "fresh_root_opened_when_claimed",
        "selector_executed",
        "future_audit_authorized",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
)
_ROOT_CLAIM_KEYS = frozenset(
    set(_GLOBAL_CLAIM_KEYS)
    | {"shard", "root_index", "root_profile", "seeds", "output_prefix"}
)
_RECEIVE_AUDIT_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "shard",
        "root_index",
        "root_profile",
        "done_sha256",
        "output_sha256",
        "config_sha256",
        "local_consumption_claim_sha256",
        "remote_consumption_claim_sha256",
        "teacher_generator_elapsed_seconds",
        "process_elapsed_seconds",
        "peak_rss_bytes",
        "boot_image_evidence_sha256",
        "runtime_semantic_anchor_sha256",
        "preflight_execution_evidence_sha256",
        "runtime_source_closure_sha256",
        "runtime_fingerprint_sha256",
        "runtime_requirements_sha256",
        "gcp_image_name",
        "gcp_image_id",
        "gcp_image_self_link",
        "selector_executed",
        "future_audit_authorized",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
)
_RECEIVE_MERGE_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "manifest_sha256",
        "launch_authorization_sha256",
        "local_consumption_claim_sha256",
        "remote_consumption_claim_sha256",
        "schedule_sha256",
        "runtime_semantic_anchor_sha256",
        "preflight_execution_evidence_sha256",
        "runtime_source_closure_sha256",
        "runtime_fingerprint_sha256",
        "runtime_requirements_sha256",
        "gcp_image_name",
        "gcp_image_id",
        "gcp_image_self_link",
        "roots",
        "root_indices",
        "profiles",
        "received_audit_sha256",
        "merged_sha256",
        "selector_command_required",
        "selector_executed",
        "selector_must_execute_exactly_once",
        "future_audit_authorized",
        "fit_performed",
        "threshold_selected",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
)
_SELECTOR_CLAIM_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "manifest_sha256",
        "launch_authorization_sha256",
        "local_consumption_claim_sha256",
        "remote_consumption_claim_sha256",
        "merge_receipt_sha256",
        "merged_sha256",
        "plan_sha256",
        "preflight_plan_sha256",
        "development_open_authorization_sha256",
        "preflight_execution_evidence_sha256",
        "selector_source_sha256",
        "canonical_decision_path",
        "canonical_receipt_path",
        "gate_evaluation_count_before_claim",
        "selector_executed",
        "future_audit_authorized",
        "fit_performed",
        "threshold_selected",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable file: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"immutable file concurrently created: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(path, canonical_json_bytes(payload))


def _load_canonical_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
        raise ValueError(f"{label} is not a canonical mapping")
    return payload


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
        raise ValueError(f"{label} SHA-256 changed")


def _nonnegative_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _parse_gnu_time_report(path: Path) -> tuple[float, int]:
    try:
        text_value = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("Attempt08 GNU time report is unreadable") from exc
    elapsed_match = re.search(
        r"Elapsed \(wall clock\) time.*?:\s*([0-9:.]+)\s*$",
        text_value,
        flags=re.MULTILINE,
    )
    rss_match = re.search(
        r"Maximum resident set size \(kbytes\):\s*(\d+)\s*$",
        text_value,
        flags=re.MULTILINE,
    )
    if elapsed_match is None or rss_match is None:
        raise ValueError("Attempt08 GNU time report fields are missing")
    pieces = [float(value) for value in elapsed_match.group(1).split(":")]
    elapsed = sum(value * (60**index) for index, value in enumerate(reversed(pieces)))
    rss = int(rss_match.group(1)) * 1024
    if not math.isfinite(elapsed) or elapsed < 0.0 or rss < 0:
        raise ValueError("Attempt08 GNU time report metrics are invalid")
    return elapsed, rss


def _schedule_bytes(schedule: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in schedule)


def build_attempt08_spot_schedule(
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Return the exact canonical 200 one-root shard schedule."""

    schedules = enumerate_attempt08_seed_schedules(plan, population="development")
    expected_domains = {"hand", "rerank", "veto", "stress", "assessment", "child"}
    if set(schedules) != expected_domains:
        raise ValueError("Attempt08 schedule seed domains changed")
    rows = []
    for root_index in range(EXPECTED_SHARDS):
        rows.append(
            {
                "schema": SHARD_SCHEMA,
                "shard": root_index,
                "root_index": root_index,
                "roots": 1,
                "root_profile": M43_ATTEMPT08_PROFILES[
                    root_index % len(M43_ATTEMPT08_PROFILES)
                ],
                "seeds": {
                    domain: int(schedules[domain][root_index])
                    for domain in (
                        "hand",
                        "rerank",
                        "veto",
                        "stress",
                        "assessment",
                        "child",
                    )
                },
                "output_prefix": f"shard_{root_index:03d}",
                "baseline_profile": "stage18_p1",
                "continuation_profile": "stage9f_p2",
                "batch_child_selectors": True,
                "native_batch_threads": NATIVE_BATCH_THREADS,
                "machine_type": RECOMMENDED_MACHINE_TYPE,
            }
        )
    return tuple(rows)


def _closure_rows(root: Path) -> list[dict[str, Any]]:
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("Attempt08 package source may not contain symlinks")
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(
            (item for item in root.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(root).as_posix(),
        )
    ]


def _copy_tree_without_symlinks(source: Path, destination: Path, *, label: str) -> None:
    if not source.is_dir():
        raise ValueError(f"{label} directory is missing: {source}")
    if source.is_symlink() or any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError(f"{label} may not contain symlinks")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable directory: {destination}")
    shutil.copytree(source, destination)


_PREFLIGHT_PACKAGE_FILES = (
    "manifest.json",
    "shards_manifest.jsonl",
    "startup_hu_m43_attempt08_preflight.sh",
    "source.zip",
    "hu_joint_policy_m43_attempt08.json",
    "hu_joint_policy_m43_attempt08_preflight.json",
)
_PREFLIGHT_JOB_FILES = (
    "DONE.json",
    "proof.json",
    "checkpoint.json",
    "heartbeat.json",
    "summary.json",
    "run.log",
    "boot_image_evidence.json",
)


def _copy_preflight_execution_bundle(
    *,
    run_dir: Path,
    authorization_path: Path,
    local_evidence_path: Path,
    jobs_root: Path,
    destination: Path,
) -> dict[str, Path]:
    """Copy the exact package and five received jobs needed to reprove preflight."""

    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable directory: {destination}")
    if any(path.is_symlink() for path in (run_dir, authorization_path, local_evidence_path, jobs_root)):
        raise ValueError("Attempt08 preflight execution inputs may not be symlinks")
    run_destination = destination / "run"
    jobs_destination = destination / "jobs"
    run_destination.mkdir(parents=True)
    jobs_destination.mkdir()
    for name in _PREFLIGHT_PACKAGE_FILES:
        source = run_dir / name
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"Attempt08 preflight package file is missing: {name}")
        _copy_file(source, run_destination / name)
    _copy_tree_without_symlinks(
        run_dir / "package_src",
        run_destination / "package_src",
        label="Attempt08 preflight package source",
    )
    if not authorization_path.is_file() or not local_evidence_path.is_file():
        raise ValueError("Attempt08 preflight authorization or local evidence is missing")
    authorization_destination = run_destination / "execution_authorization.json"
    local_evidence_destination = destination / "local_evidence.json"
    _copy_file(authorization_path, authorization_destination)
    _copy_file(local_evidence_path, local_evidence_destination)

    schedule = build_preflight_schedule()
    expected_directories = {str(spec["output_prefix"]) for spec in schedule}
    actual_directories = {
        path.name for path in jobs_root.iterdir() if path.is_dir() and not path.is_symlink()
    }
    unexpected_entries = {
        path.name
        for path in jobs_root.iterdir()
        if path.name not in expected_directories
    }
    if actual_directories != expected_directories or unexpected_entries:
        raise ValueError("Attempt08 preflight jobs root must contain exact five jobs")
    for spec in schedule:
        name = str(spec["output_prefix"])
        source_directory = jobs_root / name
        if source_directory.is_symlink():
            raise ValueError("Attempt08 preflight job directory may not be a symlink")
        actual_files = {
            path.name
            for path in source_directory.iterdir()
            if path.is_file() and not path.is_symlink()
        }
        if actual_files != set(_PREFLIGHT_JOB_FILES) or any(
            path.is_symlink() or not path.is_file() for path in source_directory.iterdir()
        ):
            raise ValueError(f"Attempt08 preflight job file set changed: {name}")
        target_directory = jobs_destination / name
        target_directory.mkdir()
        for filename in _PREFLIGHT_JOB_FILES:
            _copy_file(source_directory / filename, target_directory / filename)
    return {
        "run_dir": run_destination,
        "authorization_path": authorization_destination,
        "local_evidence_path": local_evidence_destination,
        "jobs_root": jobs_destination,
    }


def _open_authorization_bindings(payload: Mapping[str, Any]) -> dict[str, str]:
    preflight_plan = payload.get("preflight_plan")
    preflight_result = payload.get("preflight_result")
    evidence = payload.get("evidence")
    if not all(
        isinstance(value, Mapping)
        for value in (preflight_plan, preflight_result, evidence)
    ):
        raise ValueError("Attempt08 development-open authorization closure changed")
    assert isinstance(preflight_plan, Mapping)
    assert isinstance(preflight_result, Mapping)
    assert isinstance(evidence, Mapping)
    bindings = {
        "preflight_plan_sha256": _require_sha256(
            preflight_plan.get("sha256"), "preflight plan"
        ),
        "preflight_result_sha256": _require_sha256(
            preflight_result.get("sha256"), "preflight result"
        ),
        "preflight_proof_evidence_sha256": _require_sha256(
            evidence.get("proof_evidence_sha256"), "preflight proof evidence"
        ),
        "preflight_execution_evidence_sha256": _require_sha256(
            evidence.get("preflight_execution_evidence_sha256"),
            "preflight execution evidence",
        ),
        "runtime_semantic_anchor_sha256": _require_sha256(
            evidence.get("runtime_semantic_anchor_sha256"),
            "runtime semantic anchor",
        ),
        "runtime_source_closure_sha256": _require_sha256(
            evidence.get("runtime_source_closure_sha256"),
            "runtime source closure",
        ),
        "runtime_fingerprint_sha256": _require_sha256(
            evidence.get("runtime_fingerprint_sha256"),
            "runtime fingerprint",
        ),
        "runtime_requirements_sha256": _require_sha256(
            evidence.get("runtime_requirements_sha256"),
            "runtime requirements",
        ),
        "gcp_image_name": str(evidence.get("gcp_image_name")),
        "gcp_image_id": str(evidence.get("gcp_image_id")),
    }
    expected_runtime = {
        "runtime_semantic_anchor_sha256": ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,
        "runtime_source_closure_sha256": ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
        "runtime_fingerprint_sha256": ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
        "runtime_requirements_sha256": ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
        "gcp_image_name": ATTEMPT08_GCP_IMAGE_NAME,
        "gcp_image_id": ATTEMPT08_GCP_IMAGE_ID,
    }
    if any(bindings.get(key) != value for key, value in expected_runtime.items()):
        raise ValueError("Attempt08 authorized external runtime changed")
    return bindings


def _validate_preflight_finalization(
    payload: Mapping[str, Any],
    *,
    aggregate_sha256: str,
    authorization_sha256: str,
    authorization: Mapping[str, Any],
) -> None:
    expected_top = {
        "schema",
        "milestone",
        "status",
        "decision",
        "inputs",
        "gate_digests",
        "authorization",
        "science_boundary",
        "next",
    }
    inputs = _mapping(payload.get("inputs"), "preflight finalization inputs")
    gates = _mapping(payload.get("gate_digests"), "preflight gate digests")
    auth = _mapping(payload.get("authorization"), "preflight authorization")
    science = _mapping(payload.get("science_boundary"), "preflight science boundary")
    evidence = _mapping(authorization.get("evidence"), "authorization evidence")
    if (
        set(payload) != expected_top
        or payload.get("schema") != ATTEMPT08_PREFLIGHT_FINALIZATION_SCHEMA
        or payload.get("milestone") != "M4.3-attempt08"
        or payload.get("status") != ATTEMPT08_PREFLIGHT_GO_STATUS
        or payload.get("decision")
        != "authorize_exact_development_roots_0_through_199_only"
        or dict(inputs)
        != {
            "attempt08_plan_sha256": M43_ATTEMPT08_PLAN_SHA256,
            "preflight_plan_sha256": authorization["preflight_plan"]["sha256"],
            "preflight_result_schema": ATTEMPT08_PREFLIGHT_AGGREGATE_SCHEMA,
            "preflight_result_sha256": aggregate_sha256,
            "proof_evidence_sha256": evidence["proof_evidence_sha256"],
            "runtime_semantic_anchor_sha256": (
                ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256
            ),
            "runtime_source_closure_sha256": (
                ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256
            ),
            "runtime_fingerprint_sha256": (
                ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256
            ),
            "runtime_requirements_sha256": ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
            "gcp_image_name": ATTEMPT08_GCP_IMAGE_NAME,
            "gcp_image_id": ATTEMPT08_GCP_IMAGE_ID,
            "preflight_execution_evidence_sha256": evidence[
                "preflight_execution_evidence_sha256"
            ],
        }
        or dict(gates)
        != {
            "proof_gates_sha256": evidence["proof_gates_sha256"],
            "operational_gates_sha256": evidence["operational_gates_sha256"],
        }
        or dict(auth)
        != {
            "emitted": True,
            "schema": ATTEMPT08_DEVELOPMENT_OPEN_AUTHORIZATION_SCHEMA,
            "sha256": authorization_sha256,
        }
        or dict(science)
        != {
            "future_audit_authorized": False,
            "fit_started": False,
            "threshold_selection_started": False,
            "runtime_policy_activated": False,
            "current_profile_changed": False,
            "full_replacement_enabled": False,
        }
        or payload.get("next")
        != "development_runner_and_spot_package_must_consume_exact_authorization_sha"
    ):
        raise ValueError("Attempt08 preflight finalization changed")


def package_attempt08_spot(
    *,
    repo_root: str | Path,
    run_dir: str | Path,
    run_name: str,
    plan_path: str | Path,
    preflight_plan_path: str | Path,
    preflight_aggregate_path: str | Path,
    preflight_execution_evidence_path: str | Path,
    preflight_spot_run_dir: str | Path,
    preflight_spot_launch_authorization_path: str | Path,
    preflight_spot_local_evidence_path: str | Path,
    preflight_spot_jobs_root: str | Path,
    preflight_finalization_path: str | Path,
    preflight_proof_paths: Mapping[str, str | Path],
    preflight_source_path: str | Path,
    development_open_authorization_path: str | Path,
    model_path: str | Path,
    startup_path: str | Path,
    resume_existing: bool = False,
) -> dict[str, Any]:
    """Build a frozen local package without invoking GCP or opening a root."""

    if RUN_NAME_RE.fullmatch(run_name) is None:
        raise ValueError("RunName is not a safe GCP identity")
    root = Path(repo_root).resolve()
    destination = Path(run_dir).resolve()
    expected_destination = (root / "outputs" / "gcp_runs" / run_name).resolve()
    if os.path.normcase(str(destination)) != os.path.normcase(
        str(expected_destination)
    ):
        raise ValueError("Attempt08 run directory must be outputs/gcp_runs/<run_name>")
    if destination.exists():
        if not resume_existing:
            raise FileExistsError(f"Attempt08 run directory exists: {destination}")
        manifest = validate_package(destination)
        return _package_result(destination, manifest, resumed=True)

    plan_file = Path(plan_path).resolve()
    preflight_plan_file = Path(preflight_plan_path).resolve()
    preflight_aggregate_file = Path(preflight_aggregate_path).resolve()
    preflight_execution_evidence_file = Path(
        preflight_execution_evidence_path
    ).resolve()
    preflight_spot_run = Path(preflight_spot_run_dir).resolve()
    preflight_spot_authorization = Path(
        preflight_spot_launch_authorization_path
    ).resolve()
    preflight_spot_local_evidence = Path(preflight_spot_local_evidence_path).resolve()
    preflight_spot_jobs = Path(preflight_spot_jobs_root).resolve()
    preflight_finalization_file = Path(preflight_finalization_path).resolve()
    preflight_source_file = Path(preflight_source_path).resolve()
    if set(preflight_proof_paths) != set(ATTEMPT08_PREFLIGHT_SLOTS):
        raise ValueError("Attempt08 package requires exact five preflight proofs")
    proof_files = {
        slot: Path(preflight_proof_paths[slot]).resolve()
        for slot in ATTEMPT08_PREFLIGHT_SLOTS
    }
    if len({os.path.normcase(str(path)) for path in proof_files.values()}) != 5:
        raise ValueError("Attempt08 preflight proof paths must be distinct")
    open_file = Path(development_open_authorization_path).resolve()
    model_file = Path(model_path).resolve()
    startup_file = Path(startup_path).resolve()
    expected_startup = (
        root / "scripts/startup_hu_m43_attempt08_development.sh"
    ).resolve()
    if os.path.normcase(str(startup_file)) != os.path.normcase(str(expected_startup)):
        raise ValueError("Attempt08 package requires the canonical development startup")
    plan = load_and_validate_attempt08_plan(plan_file)
    if sha256_file(plan_file) != M43_ATTEMPT08_PLAN_SHA256:
        raise ValueError("Attempt08 plan SHA-256 changed")
    if sha256_file(model_file) != ATTEMPT08_LAMBDA_MODEL_SHA256:
        raise ValueError("Attempt08 model SHA-256 changed")
    ai_profiles_file = root / "src/ofc_regular/ai_profiles.py"
    if sha256_file(ai_profiles_file) != AI_PROFILES_SHA256:
        raise ValueError("Attempt08 ai_profiles.py SHA-256 changed")
    open_authorization = load_and_validate_development_open_authorization(
        open_file,
        attempt08_plan_path=plan_file,
        preflight_plan_path=preflight_plan_file,
    )
    open_bindings = _open_authorization_bindings(open_authorization)
    aggregate = load_and_validate_preflight_aggregate(preflight_aggregate_file)
    preflight_execution_evidence = _load_canonical_mapping(
        preflight_execution_evidence_file,
        "Attempt08 preflight execution evidence",
    )
    validate_preflight_execution_evidence(preflight_execution_evidence)
    verified_execution_bundle = validate_preflight_receive_bundle(
        run_dir=preflight_spot_run,
        authorization_path=preflight_spot_authorization,
        local_evidence_path=preflight_spot_local_evidence,
        jobs_root=preflight_spot_jobs,
        execution_evidence_path=preflight_execution_evidence_file,
    )
    if verified_execution_bundle.get("execution_evidence") != preflight_execution_evidence:
        raise ValueError("Attempt08 preflight execution bundle changed after validation")
    validate_preflight_aggregate_with_proofs(
        aggregate,
        proof_paths=proof_files,
        source=preflight_source_file,
    )
    aggregate_sha256 = sha256_file(preflight_aggregate_file)
    execution_evidence_sha256 = sha256_file(preflight_execution_evidence_file)
    proof_sha256 = {slot: sha256_file(proof_files[slot]) for slot in proof_files}
    evidence = _mapping(open_authorization["evidence"], "authorization evidence")
    result = _mapping(open_authorization["preflight_result"], "authorization result")
    if (
        result.get("sha256") != aggregate_sha256
        or evidence.get("proof_file_sha256") != proof_sha256
        or aggregate.get("proof_file_sha256") != proof_sha256
        or aggregate.get("proof_evidence_sha256")
        != evidence.get("proof_evidence_sha256")
        or aggregate.get("contract", {}).get("runtime_semantic_anchor_sha256")
        != ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256
        or aggregate.get("contract", {}).get("runtime_source_closure_sha256")
        != ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256
        or aggregate.get("contract", {}).get("runtime_requirements_sha256")
        != ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256
        or aggregate.get("contract", {}).get("gcp_image_name")
        != ATTEMPT08_GCP_IMAGE_NAME
        or aggregate.get("contract", {}).get("gcp_image_id")
        != ATTEMPT08_GCP_IMAGE_ID
        or aggregate.get("runtime_fingerprint_sha256")
        != ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256
        or aggregate.get("spot_operational_evidence_sha256")
        != execution_evidence_sha256
        or open_bindings["preflight_execution_evidence_sha256"]
        != execution_evidence_sha256
        or open_bindings["runtime_semantic_anchor_sha256"]
        != ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256
    ):
        raise ValueError("Attempt08 authorization and proof chain differ")
    finalization = _load_canonical_mapping(
        preflight_finalization_file, "Attempt08 preflight finalization"
    )
    _validate_preflight_finalization(
        finalization,
        aggregate_sha256=aggregate_sha256,
        authorization_sha256=sha256_file(open_file),
        authorization=open_authorization,
    )
    schedule = build_attempt08_spot_schedule(plan)

    template_root = root / "outputs/gcp_runs" / PINNED_TEMPLATE_RUN / "package_src"
    model_manifest, native_manifest = _verify_pinned_runtime_closure(template_root)
    validate_runtime_semantic_anchor(
        repository_root=root,
        expected_source_closure_sha256=ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
        expected_anchor_sha256=ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,
        expected_source_file_count=ATTEMPT08_RUNTIME_SOURCE_FILE_COUNT,
        runtime_artifact_root=template_root,
        model_manifest_path=template_root / "source_model_manifest.json",
        native_manifest_path=template_root / "source_native_manifest.json",
        requirements_path=root / "configs/hu_m43_attempt08_runtime_requirements.txt",
    )
    staging = destination.with_name(destination.name + ".building")
    if staging.exists():
        raise FileExistsError(f"stale Attempt08 staging directory: {staging}")
    staging.mkdir(parents=True)
    try:
        package_root = staging / "package_src"
        package_root.mkdir()
        _copy_source_tree(root / "src/ofc_regular", package_root / "src/ofc_regular")
        _copy_manifest_files(template_root, package_root, model_manifest["models"])
        _copy_manifest_files(template_root, package_root, native_manifest["binaries"])
        _copy_file(
            template_root / "source_model_manifest.json",
            package_root / "source_model_manifest.json",
        )
        _copy_file(
            template_root / "source_native_manifest.json",
            package_root / "source_native_manifest.json",
        )
        _copy_file(model_file, package_root / "artifacts/lambda_rank_candidate.pkl")
        _copy_file(
            model_file,
            package_root
            / "outputs/hu_joint_policy/m43_attempt05_old_dev900_architecture_once"
            / "lambda_rank_candidate.pkl",
        )
        _copy_file(plan_file, package_root / "configs/hu_joint_policy_m43_attempt08.json")
        _copy_file(
            preflight_plan_file,
            package_root / "configs/hu_joint_policy_m43_attempt08_preflight.json",
        )
        _copy_file(
            open_file,
            package_root / "frozen/development_open_authorization.json",
        )
        _copy_file(
            preflight_aggregate_file,
            package_root / "frozen/preflight_aggregate.json",
        )
        _copy_file(
            preflight_execution_evidence_file,
            package_root / "frozen/preflight_execution_evidence.json",
        )
        copied_execution_bundle = _copy_preflight_execution_bundle(
            run_dir=preflight_spot_run,
            authorization_path=preflight_spot_authorization,
            local_evidence_path=preflight_spot_local_evidence,
            jobs_root=preflight_spot_jobs,
            destination=package_root / "frozen/preflight_execution_bundle",
        )
        copied_bundle_validation = validate_preflight_receive_bundle(
            **copied_execution_bundle,
            execution_evidence_path=(
                package_root / "frozen/preflight_execution_evidence.json"
            ),
            # The evidence command rows intentionally bind the original
            # absolute preflight manifest path.  The original bundle was
            # revalidated above before byte-exact copying, so the relocated
            # immutable copy must verify identities and hashes without
            # rewriting or reinterpreting those path-bound command rows.
            revalidate_local_commands=False,
        )
        if copied_bundle_validation.get("execution_evidence") != preflight_execution_evidence:
            raise ValueError("Attempt08 copied preflight execution bundle changed")
        _copy_file(
            preflight_finalization_file,
            package_root / "frozen/preflight_finalization.json",
        )
        _copy_file(preflight_source_file, package_root / "frozen/preflight_source.json")
        for slot, proof_file in proof_files.items():
            _copy_file(
                proof_file,
                package_root / "frozen/preflight_proofs" / f"{slot}.json",
            )
        _copy_file(ai_profiles_file, package_root / "frozen/ai_profiles.py")
        _copy_file(
            root / "configs/hu_m43_attempt08_runtime_requirements.txt",
            package_root / "configs/hu_m43_attempt08_runtime_requirements.txt",
        )
        _copy_file(
            root / "configs/hu_m43_attempt08_runtime_requirements.txt",
            package_root / "requirements-attempt08.txt",
        )
        for relative in ATTEMPT08_RUNTIME_SEMANTIC_FILES:
            _copy_file(root / relative, package_root / relative)
        anchored_startup = (
            package_root / "scripts/startup_hu_m43_attempt08_development.sh"
        )
        if (
            not anchored_startup.is_file()
            or anchored_startup.read_bytes() != startup_file.read_bytes()
        ):
            raise ValueError(
                "Attempt08 canonical startup differs from the anchored runtime copy"
            )
        validate_runtime_semantic_anchor(
            repository_root=package_root,
            expected_source_closure_sha256=ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
            expected_anchor_sha256=ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,
            expected_source_file_count=ATTEMPT08_RUNTIME_SOURCE_FILE_COUNT,
            runtime_artifact_root=package_root,
            model_manifest_path=package_root / "source_model_manifest.json",
            native_manifest_path=package_root / "source_native_manifest.json",
            requirements_path=(
                package_root / "configs/hu_m43_attempt08_runtime_requirements.txt"
            ),
        )
        _atomic_write(
            package_root / "shards_manifest.jsonl", _schedule_bytes(schedule)
        )
        closure = {
            "schema": SOURCE_CLOSURE_SCHEMA,
            "status": "closed_before_any_development_root",
            "run_name": run_name,
            "plan_sha256": M43_ATTEMPT08_PLAN_SHA256,
            "model_sha256": ATTEMPT08_LAMBDA_MODEL_SHA256,
            "ai_profiles_sha256": AI_PROFILES_SHA256,
            "development_open_authorization_sha256": sha256_file(open_file),
            **open_bindings,
            "preflight_aggregate_sha256": aggregate_sha256,
            "preflight_finalization_sha256": sha256_file(
                preflight_finalization_file
            ),
            "preflight_source_sha256": sha256_file(preflight_source_file),
            "preflight_proof_sha256": proof_sha256,
            "gcp_image_self_link": ATTEMPT08_GCP_IMAGE_SELF_LINK,
            "files": _closure_rows(package_root),
            "fresh_root_opened": False,
            "teacher_executed": False,
            "selector_executed": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        }
        _write_json(package_root / "source_closure_manifest.json", closure)

        schedule_path = staging / "shards_manifest.jsonl"
        closure_path = staging / "source_closure_manifest.json"
        source_zip = staging / "ofc_regular_hu_m43_attempt08_development_source.zip"
        startup_copy = staging / "startup_hu_m43_attempt08_development.sh"
        _copy_file(package_root / "shards_manifest.jsonl", schedule_path)
        _copy_file(package_root / "source_closure_manifest.json", closure_path)
        # Copy the already-anchored bytes so the outer startup cannot race a
        # second read of the repository file after semantic validation.
        _copy_file(anchored_startup, startup_copy)
        if startup_copy.read_bytes() != anchored_startup.read_bytes():
            raise AssertionError("Attempt08 outer and anchored startup copies differ")
        _copy_file(plan_file, staging / "hu_joint_policy_m43_attempt08.json")
        _copy_file(
            preflight_plan_file,
            staging / "hu_joint_policy_m43_attempt08_preflight.json",
        )
        _copy_file(open_file, staging / "development_open_authorization.json")
        _copy_file(preflight_aggregate_file, staging / "preflight_aggregate.json")
        _copy_file(
            preflight_execution_evidence_file,
            staging / "preflight_execution_evidence.json",
        )
        _copy_file(
            preflight_finalization_file, staging / "preflight_finalization.json"
        )
        _copy_file(preflight_source_file, staging / "preflight_source.json")
        for slot, proof_file in proof_files.items():
            _copy_file(proof_file, staging / "preflight_proofs" / f"{slot}.json")
        _copy_file(
            template_root / "source_model_manifest.json",
            staging / "source_model_manifest.json",
        )
        _copy_file(
            template_root / "source_native_manifest.json",
            staging / "source_native_manifest.json",
        )
        _write_deterministic_zip(package_root, source_zip)
        manifest = {
            "schema": PACKAGE_MANIFEST_SCHEMA,
            "status": "frozen_package_only_no_root_opened",
            "run_name": run_name,
            "plan_sha256": M43_ATTEMPT08_PLAN_SHA256,
            "model_sha256": ATTEMPT08_LAMBDA_MODEL_SHA256,
            "ai_profiles_sha256": AI_PROFILES_SHA256,
            "development_open_authorization_sha256": sha256_file(open_file),
            **open_bindings,
            "preflight_aggregate_sha256": aggregate_sha256,
            "preflight_finalization_sha256": sha256_file(
                preflight_finalization_file
            ),
            "preflight_source_sha256": sha256_file(preflight_source_file),
            "preflight_proof_sha256": proof_sha256,
            "gcp_image_self_link": ATTEMPT08_GCP_IMAGE_SELF_LINK,
            "schedule_sha256": sha256_file(schedule_path),
            "source_closure_sha256": sha256_file(closure_path),
            "source_zip_sha256": sha256_file(source_zip),
            "startup_sha256": sha256_file(startup_copy),
            "source_model_manifest_sha256": PINNED_MODEL_MANIFEST_SHA256,
            "source_native_manifest_sha256": PINNED_NATIVE_MANIFEST_SHA256,
            "total_roots": EXPECTED_SHARDS,
            "total_shards": EXPECTED_SHARDS,
            "roots_per_shard": ROOTS_PER_SHARD,
            "root_profile_assignment": "root_index_mod_5_in_frozen_profile_order",
            "batch_child_selectors": True,
            "native_batch_threads": NATIVE_BATCH_THREADS,
            "recommended_machine_type": RECOMMENDED_MACHINE_TYPE,
            "recommended_wave_shards": MAX_WAVE_SHARDS,
            "fresh_root_opened": False,
            "teacher_executed": False,
            "gcloud_invoked": False,
            "instances_created": False,
            "selector_executed": False,
            "future_audit_authorized": False,
            "fit_performed": False,
            "threshold_selected": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        }
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return _package_result(destination, manifest, resumed=False)


def _package_result(
    run_dir: Path, manifest: Mapping[str, Any], *, resumed: bool
) -> dict[str, Any]:
    return {
        "schema": PACKAGE_RESULT_SCHEMA,
        "status": "packaged_without_root_or_gcloud",
        "run_name": manifest["run_name"],
        "run_dir": str(run_dir),
        "manifest": str(run_dir / "manifest.json"),
        "manifest_sha256": sha256_file(run_dir / "manifest.json"),
        "total_shards": EXPECTED_SHARDS,
        "resumed_existing_package": resumed,
        "fresh_root_opened": False,
        "teacher_executed": False,
        "gcloud_invoked": False,
        "instances_created": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }


def _validate_source_closure(root: Path, manifest: Mapping[str, Any]) -> None:
    package_root = root / "package_src"
    closure_root = root / "source_closure_manifest.json"
    closure_in_package = package_root / "source_closure_manifest.json"
    _require_hash(
        closure_root,
        str(manifest["source_closure_sha256"]),
        "source closure manifest",
    )
    _require_hash(
        closure_in_package,
        str(manifest["source_closure_sha256"]),
        "packaged source closure manifest",
    )
    if closure_root.read_bytes() != closure_in_package.read_bytes():
        raise ValueError("Attempt08 source closure copies differ")
    closure = _load_canonical_mapping(closure_root, "Attempt08 source closure")
    expected_closure_keys = {
        "schema",
        "status",
        "run_name",
        "plan_sha256",
        "model_sha256",
        "ai_profiles_sha256",
        "development_open_authorization_sha256",
        "preflight_plan_sha256",
        "preflight_result_sha256",
        "preflight_proof_evidence_sha256",
        "preflight_execution_evidence_sha256",
        "preflight_aggregate_sha256",
        "preflight_finalization_sha256",
        "preflight_source_sha256",
        "preflight_proof_sha256",
        "runtime_semantic_anchor_sha256",
        "runtime_source_closure_sha256",
        "runtime_fingerprint_sha256",
        "runtime_requirements_sha256",
        "gcp_image_name",
        "gcp_image_id",
        "gcp_image_self_link",
        "files",
        "fresh_root_opened",
        "teacher_executed",
        "selector_executed",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
    if (
        set(closure) != expected_closure_keys
        or closure.get("schema") != SOURCE_CLOSURE_SCHEMA
        or closure.get("status") != "closed_before_any_development_root"
        or closure.get("run_name") != manifest["run_name"]
        or any(
            closure.get(field) != manifest[field]
            for field in (
                "plan_sha256",
                "model_sha256",
                "ai_profiles_sha256",
                "development_open_authorization_sha256",
                "preflight_plan_sha256",
                "preflight_result_sha256",
                "preflight_proof_evidence_sha256",
                "preflight_execution_evidence_sha256",
                "preflight_aggregate_sha256",
                "preflight_finalization_sha256",
                "preflight_source_sha256",
                "preflight_proof_sha256",
                "runtime_semantic_anchor_sha256",
                "runtime_source_closure_sha256",
                "runtime_fingerprint_sha256",
                "runtime_requirements_sha256",
                "gcp_image_name",
                "gcp_image_id",
                "gcp_image_self_link",
            )
        )
        or any(
            closure.get(field) is not False
            for field in (
                "fresh_root_opened",
                "teacher_executed",
                "selector_executed",
                "current_profile_mutated",
                "runtime_policy_activated",
            )
        )
    ):
        raise ValueError("Attempt08 source closure identity changed")
    files = closure.get("files")
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes, bytearray)):
        raise ValueError("Attempt08 source closure files changed")
    expected_rows: dict[str, Mapping[str, Any]] = {}
    for value in files:
        if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
            raise ValueError("Attempt08 source closure row changed")
        relative = value.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or "\\" in relative
            or ".." in Path(relative).parts
            or relative in expected_rows
        ):
            raise ValueError("Attempt08 source closure path changed")
        if type(value.get("bytes")) is not int or value["bytes"] < 0:
            raise ValueError("Attempt08 source closure byte count changed")
        _require_sha256(value.get("sha256"), "source closure file")
        expected_rows[relative] = value
    actual_files = {
        path.relative_to(package_root).as_posix(): path
        for path in package_root.rglob("*")
        if path.is_file()
    }
    if any(path.is_symlink() for path in package_root.rglob("*")):
        raise ValueError("Attempt08 packaged source contains symlinks")
    if set(actual_files) != set(expected_rows) | {"source_closure_manifest.json"}:
        raise ValueError("Attempt08 packaged source file set changed")
    for relative, row in expected_rows.items():
        path = actual_files[relative]
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError(f"Attempt08 packaged source changed: {relative}")

    archive = root / "ofc_regular_hu_m43_attempt08_development_source.zip"
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        expected_names = sorted(actual_files)
        if names != expected_names or len(names) != len(set(names)):
            raise ValueError("Attempt08 source archive member set changed")
        for name in names:
            info = handle.getinfo(name)
            if info.is_dir() or name.startswith("/") or ".." in Path(name).parts:
                raise ValueError("Attempt08 source archive path is unsafe")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Attempt08 source archive contains a symlink")
            if handle.read(name) != actual_files[name].read_bytes():
                raise ValueError(f"Attempt08 source archive changed: {name}")


def validate_package(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    manifest = _load_canonical_mapping(root / "manifest.json", "Attempt08 manifest")
    if (
        set(manifest) != _PACKAGE_KEYS
        or manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA
        or manifest.get("status") != "frozen_package_only_no_root_opened"
        or manifest.get("plan_sha256") != M43_ATTEMPT08_PLAN_SHA256
        or manifest.get("model_sha256") != ATTEMPT08_LAMBDA_MODEL_SHA256
        or manifest.get("ai_profiles_sha256") != AI_PROFILES_SHA256
        or manifest.get("runtime_semantic_anchor_sha256")
        != ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256
        or manifest.get("runtime_source_closure_sha256")
        != ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256
        or manifest.get("runtime_fingerprint_sha256")
        != ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256
        or manifest.get("runtime_requirements_sha256")
        != ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256
        or manifest.get("gcp_image_name") != ATTEMPT08_GCP_IMAGE_NAME
        or manifest.get("gcp_image_id") != ATTEMPT08_GCP_IMAGE_ID
        or manifest.get("gcp_image_self_link") != ATTEMPT08_GCP_IMAGE_SELF_LINK
        or manifest.get("total_roots") != EXPECTED_SHARDS
        or manifest.get("total_shards") != EXPECTED_SHARDS
        or manifest.get("roots_per_shard") != 1
        or manifest.get("native_batch_threads") != NATIVE_BATCH_THREADS
        or manifest.get("recommended_machine_type") != RECOMMENDED_MACHINE_TYPE
        or manifest.get("recommended_wave_shards") != MAX_WAVE_SHARDS
        or any(
            manifest.get(field) is not False
            for field in (
                "fresh_root_opened",
                "teacher_executed",
                "gcloud_invoked",
                "instances_created",
                "selector_executed",
                "future_audit_authorized",
                "fit_performed",
                "threshold_selected",
                "current_profile_mutated",
                "runtime_policy_activated",
            )
        )
    ):
        raise ValueError("Attempt08 package manifest changed")
    bindings = {
        "shards_manifest.jsonl": "schedule_sha256",
        "source_closure_manifest.json": "source_closure_sha256",
        "ofc_regular_hu_m43_attempt08_development_source.zip": "source_zip_sha256",
        "startup_hu_m43_attempt08_development.sh": "startup_sha256",
        "hu_joint_policy_m43_attempt08.json": "plan_sha256",
        "hu_joint_policy_m43_attempt08_preflight.json": "preflight_plan_sha256",
        "development_open_authorization.json": (
            "development_open_authorization_sha256"
        ),
        "preflight_aggregate.json": "preflight_aggregate_sha256",
        "preflight_execution_evidence.json": (
            "preflight_execution_evidence_sha256"
        ),
        "preflight_finalization.json": "preflight_finalization_sha256",
        "preflight_source.json": "preflight_source_sha256",
        "source_model_manifest.json": "source_model_manifest_sha256",
        "source_native_manifest.json": "source_native_manifest_sha256",
    }
    for name, field in bindings.items():
        _require_hash(root / name, str(manifest[field]), name)
    outer_startup = root / "startup_hu_m43_attempt08_development.sh"
    anchored_startup = (
        root / "package_src/scripts/startup_hu_m43_attempt08_development.sh"
    )
    if (
        not anchored_startup.is_file()
        or outer_startup.read_bytes() != anchored_startup.read_bytes()
    ):
        raise ValueError("Attempt08 outer and anchored startup copies differ")
    packaged_requirements = (
        root / "package_src/configs/hu_m43_attempt08_runtime_requirements.txt"
    )
    install_requirements = root / "package_src/requirements-attempt08.txt"
    _require_hash(
        packaged_requirements,
        ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
        "Attempt08 anchored runtime requirements",
    )
    _require_hash(
        install_requirements,
        ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
        "Attempt08 install runtime requirements",
    )
    if packaged_requirements.read_bytes() != install_requirements.read_bytes():
        raise ValueError("Attempt08 runtime requirement copies differ")
    proof_hashes = manifest.get("preflight_proof_sha256")
    if not isinstance(proof_hashes, Mapping) or set(proof_hashes) != set(
        ATTEMPT08_PREFLIGHT_SLOTS
    ):
        raise ValueError("Attempt08 package proof hash set changed")
    proof_paths = {
        slot: root / "preflight_proofs" / f"{slot}.json"
        for slot in ATTEMPT08_PREFLIGHT_SLOTS
    }
    for slot, path in proof_paths.items():
        _require_hash(path, str(proof_hashes[slot]), f"preflight proof {slot}")
    plan = load_and_validate_attempt08_plan(root / "hu_joint_policy_m43_attempt08.json")
    schedule = build_attempt08_spot_schedule(plan)
    if (root / "shards_manifest.jsonl").read_bytes() != _schedule_bytes(schedule):
        raise ValueError("Attempt08 shard schedule changed")
    open_authorization = load_and_validate_development_open_authorization(
        root / "development_open_authorization.json",
        attempt08_plan_path=root / "hu_joint_policy_m43_attempt08.json",
        preflight_plan_path=root / "hu_joint_policy_m43_attempt08_preflight.json",
    )
    aggregate = load_and_validate_preflight_aggregate(root / "preflight_aggregate.json")
    execution_evidence = _load_canonical_mapping(
        root / "preflight_execution_evidence.json",
        "Attempt08 preflight execution evidence",
    )
    validate_preflight_execution_evidence(execution_evidence)
    validate_preflight_aggregate_with_proofs(
        aggregate,
        proof_paths=proof_paths,
        source=root / "preflight_source.json",
    )
    if (
        aggregate.get("proof_file_sha256") != dict(proof_hashes)
        or aggregate.get("proof_evidence_sha256")
        != manifest["preflight_proof_evidence_sha256"]
        or aggregate.get("contract", {}).get("runtime_semantic_anchor_sha256")
        != manifest["runtime_semantic_anchor_sha256"]
        or aggregate.get("contract", {}).get("runtime_source_closure_sha256")
        != manifest["runtime_source_closure_sha256"]
        or aggregate.get("contract", {}).get("runtime_requirements_sha256")
        != manifest["runtime_requirements_sha256"]
        or aggregate.get("contract", {}).get("gcp_image_name")
        != manifest["gcp_image_name"]
        or aggregate.get("contract", {}).get("gcp_image_id")
        != manifest["gcp_image_id"]
        or aggregate.get("runtime_fingerprint_sha256")
        != manifest["runtime_fingerprint_sha256"]
        or aggregate.get("spot_operational_evidence_sha256")
        != manifest["preflight_execution_evidence_sha256"]
    ):
        raise ValueError("Attempt08 package aggregate closure changed")
    finalization = _load_canonical_mapping(
        root / "preflight_finalization.json", "Attempt08 preflight finalization"
    )
    _validate_preflight_finalization(
        finalization,
        aggregate_sha256=manifest["preflight_aggregate_sha256"],
        authorization_sha256=manifest["development_open_authorization_sha256"],
        authorization=open_authorization,
    )
    open_bindings = load_attempt08_development_open_bindings(
        root / "development_open_authorization.json",
        plan=root / "hu_joint_policy_m43_attempt08.json",
        preflight_plan=root / "hu_joint_policy_m43_attempt08_preflight.json",
    )
    if any(
        manifest.get(field) != open_bindings[field]
        for field in (
            "development_open_authorization_sha256",
            "preflight_plan_sha256",
            "preflight_result_sha256",
            "preflight_proof_evidence_sha256",
            "preflight_execution_evidence_sha256",
            "runtime_semantic_anchor_sha256",
            "runtime_source_closure_sha256",
            "runtime_fingerprint_sha256",
            "runtime_requirements_sha256",
            "gcp_image_name",
            "gcp_image_id",
        )
    ):
        raise ValueError("Attempt08 package authorization bindings changed")
    if manifest.get("runtime_semantic_anchor_sha256") != (
        ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256
    ):
        raise ValueError("Attempt08 package runtime anchor changed")
    if manifest.get("gcp_image_self_link") != ATTEMPT08_GCP_IMAGE_SELF_LINK:
        raise ValueError("Attempt08 package image self link changed")
    execution_bundle_root = root / "package_src/frozen/preflight_execution_bundle"
    execution_bundle_validation = validate_preflight_receive_bundle(
        run_dir=execution_bundle_root / "run",
        authorization_path=(
            execution_bundle_root / "run/execution_authorization.json"
        ),
        local_evidence_path=execution_bundle_root / "local_evidence.json",
        jobs_root=execution_bundle_root / "jobs",
        execution_evidence_path=(
            root / "package_src/frozen/preflight_execution_evidence.json"
        ),
        revalidate_local_commands=False,
    )
    if execution_bundle_validation.get("execution_evidence") != execution_evidence:
        raise ValueError("Attempt08 packaged preflight execution bundle changed")
    validate_runtime_semantic_anchor(
        repository_root=root / "package_src",
        expected_source_closure_sha256=ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
        expected_anchor_sha256=ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,
        expected_source_file_count=ATTEMPT08_RUNTIME_SOURCE_FILE_COUNT,
        runtime_artifact_root=root / "package_src",
        model_manifest_path=root / "package_src/source_model_manifest.json",
        native_manifest_path=root / "package_src/source_native_manifest.json",
        requirements_path=(
            root / "package_src/configs/hu_m43_attempt08_runtime_requirements.txt"
        ),
    )
    _validate_source_closure(root, manifest)
    return manifest


def authorize_attempt08_launch(
    *, run_dir: str | Path, output: str | Path
) -> dict[str, Any]:
    """Create a separate immutable launch authorization for one frozen package."""

    root = Path(run_dir)
    manifest = validate_package(root)
    payload = {
        "schema": LAUNCH_AUTHORIZATION_SCHEMA,
        "status": "authorized_after_attempt08_preflight_and_package_freeze",
        "spot_authorized": True,
        "run_name": manifest["run_name"],
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "plan_sha256": manifest["plan_sha256"],
        "model_sha256": manifest["model_sha256"],
        "ai_profiles_sha256": manifest["ai_profiles_sha256"],
        "development_open_authorization_sha256": manifest[
            "development_open_authorization_sha256"
        ],
        "preflight_plan_sha256": manifest["preflight_plan_sha256"],
        "preflight_result_sha256": manifest["preflight_result_sha256"],
        "preflight_proof_evidence_sha256": manifest[
            "preflight_proof_evidence_sha256"
        ],
        "preflight_execution_evidence_sha256": manifest[
            "preflight_execution_evidence_sha256"
        ],
        "preflight_aggregate_sha256": manifest["preflight_aggregate_sha256"],
        "preflight_finalization_sha256": manifest[
            "preflight_finalization_sha256"
        ],
        "preflight_source_sha256": manifest["preflight_source_sha256"],
        "preflight_proof_sha256": manifest["preflight_proof_sha256"],
        "runtime_semantic_anchor_sha256": manifest[
            "runtime_semantic_anchor_sha256"
        ],
        "runtime_source_closure_sha256": manifest[
            "runtime_source_closure_sha256"
        ],
        "runtime_fingerprint_sha256": manifest["runtime_fingerprint_sha256"],
        "runtime_requirements_sha256": manifest["runtime_requirements_sha256"],
        "gcp_image_name": manifest["gcp_image_name"],
        "gcp_image_id": manifest["gcp_image_id"],
        "gcp_image_self_link": manifest["gcp_image_self_link"],
        "schedule_sha256": manifest["schedule_sha256"],
        "source_closure_sha256": manifest["source_closure_sha256"],
        "source_zip_sha256": manifest["source_zip_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "source_model_manifest_sha256": manifest[
            "source_model_manifest_sha256"
        ],
        "source_native_manifest_sha256": manifest[
            "source_native_manifest_sha256"
        ],
        "total_roots": EXPECTED_SHARDS,
        "total_shards": EXPECTED_SHARDS,
        "roots_per_shard": 1,
        "native_batch_threads": NATIVE_BATCH_THREADS,
        "max_wave_shards": MAX_WAVE_SHARDS,
        "machine_type": RECOMMENDED_MACHINE_TYPE,
        "package_frozen_before_authorization": True,
        "development_started": False,
        "fresh_root_opened": False,
        "selector_executed": False,
        "future_audit_authorized": False,
        "fit_performed": False,
        "threshold_selected": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    _write_json(Path(output), payload)
    return payload


def validate_launch_authorization(
    authorization_path: str | Path, *, run_dir: str | Path
) -> dict[str, Any]:
    root = Path(run_dir)
    manifest = validate_package(root)
    authorization = _load_canonical_mapping(
        Path(authorization_path), "Attempt08 launch authorization"
    )
    if set(authorization) != _LAUNCH_KEYS:
        raise ValueError("Attempt08 launch authorization fields changed")
    expected = {
        "schema": LAUNCH_AUTHORIZATION_SCHEMA,
        "status": "authorized_after_attempt08_preflight_and_package_freeze",
        "spot_authorized": True,
        "run_name": manifest["run_name"],
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "plan_sha256": manifest["plan_sha256"],
        "model_sha256": manifest["model_sha256"],
        "ai_profiles_sha256": manifest["ai_profiles_sha256"],
        "development_open_authorization_sha256": manifest[
            "development_open_authorization_sha256"
        ],
        "preflight_plan_sha256": manifest["preflight_plan_sha256"],
        "preflight_result_sha256": manifest["preflight_result_sha256"],
        "preflight_proof_evidence_sha256": manifest[
            "preflight_proof_evidence_sha256"
        ],
        "preflight_execution_evidence_sha256": manifest[
            "preflight_execution_evidence_sha256"
        ],
        "preflight_aggregate_sha256": manifest["preflight_aggregate_sha256"],
        "preflight_finalization_sha256": manifest[
            "preflight_finalization_sha256"
        ],
        "preflight_source_sha256": manifest["preflight_source_sha256"],
        "preflight_proof_sha256": manifest["preflight_proof_sha256"],
        "runtime_semantic_anchor_sha256": manifest[
            "runtime_semantic_anchor_sha256"
        ],
        "runtime_source_closure_sha256": manifest[
            "runtime_source_closure_sha256"
        ],
        "runtime_fingerprint_sha256": manifest["runtime_fingerprint_sha256"],
        "runtime_requirements_sha256": manifest["runtime_requirements_sha256"],
        "gcp_image_name": manifest["gcp_image_name"],
        "gcp_image_id": manifest["gcp_image_id"],
        "gcp_image_self_link": manifest["gcp_image_self_link"],
        "schedule_sha256": manifest["schedule_sha256"],
        "source_closure_sha256": manifest["source_closure_sha256"],
        "source_zip_sha256": manifest["source_zip_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "source_model_manifest_sha256": manifest[
            "source_model_manifest_sha256"
        ],
        "source_native_manifest_sha256": manifest[
            "source_native_manifest_sha256"
        ],
        "total_roots": EXPECTED_SHARDS,
        "total_shards": EXPECTED_SHARDS,
        "roots_per_shard": 1,
        "native_batch_threads": NATIVE_BATCH_THREADS,
        "max_wave_shards": MAX_WAVE_SHARDS,
        "machine_type": RECOMMENDED_MACHINE_TYPE,
        "package_frozen_before_authorization": True,
        "development_started": False,
        "fresh_root_opened": False,
        "selector_executed": False,
        "future_audit_authorized": False,
        "fit_performed": False,
        "threshold_selected": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    if authorization != expected:
        raise ValueError("Attempt08 launch authorization changed")
    return authorization


def _claim_common(
    *, manifest: Mapping[str, Any], manifest_sha256: str, authorization_sha256: str
) -> dict[str, Any]:
    return {
        "run_name": manifest["run_name"],
        "manifest_sha256": manifest_sha256,
        "launch_authorization_sha256": authorization_sha256,
        "development_open_authorization_sha256": manifest[
            "development_open_authorization_sha256"
        ],
        "source_sha256": manifest["source_zip_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "schedule_sha256": manifest["schedule_sha256"],
        "plan_sha256": manifest["plan_sha256"],
        "model_sha256": manifest["model_sha256"],
        "ai_profiles_sha256": manifest["ai_profiles_sha256"],
        "source_closure_sha256": manifest["source_closure_sha256"],
        "source_model_manifest_sha256": manifest[
            "source_model_manifest_sha256"
        ],
        "source_native_manifest_sha256": manifest[
            "source_native_manifest_sha256"
        ],
        "preflight_aggregate_sha256": manifest["preflight_aggregate_sha256"],
        "preflight_finalization_sha256": manifest[
            "preflight_finalization_sha256"
        ],
        "preflight_proof_sha256": manifest["preflight_proof_sha256"],
        "runtime_semantic_anchor_sha256": manifest[
            "runtime_semantic_anchor_sha256"
        ],
        "preflight_execution_evidence_sha256": manifest[
            "preflight_execution_evidence_sha256"
        ],
        "runtime_source_closure_sha256": manifest[
            "runtime_source_closure_sha256"
        ],
        "runtime_fingerprint_sha256": manifest["runtime_fingerprint_sha256"],
        "runtime_requirements_sha256": manifest["runtime_requirements_sha256"],
        "gcp_image_name": manifest["gcp_image_name"],
        "gcp_image_id": manifest["gcp_image_id"],
        "gcp_image_self_link": manifest["gcp_image_self_link"],
        "fresh_root_opened_when_claimed": False,
        "selector_executed": False,
        "future_audit_authorized": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }


def build_global_claim(
    *, run_dir: str | Path, launch_authorization_path: str | Path
) -> dict[str, Any]:
    root = Path(run_dir)
    manifest = validate_package(root)
    validate_launch_authorization(launch_authorization_path, run_dir=root)
    payload = {
        "schema": GLOBAL_CLAIM_SCHEMA,
        "status": "claimed_before_any_attempt08_development_root",
        **_claim_common(
            manifest=manifest,
            manifest_sha256=sha256_file(root / "manifest.json"),
            authorization_sha256=sha256_file(launch_authorization_path),
        ),
    }
    if set(payload) != _GLOBAL_CLAIM_KEYS:
        raise AssertionError("Attempt08 global claim schema changed")
    return payload


def build_root_claim(
    *,
    run_dir: str | Path,
    launch_authorization_path: str | Path,
    shard: int,
) -> dict[str, Any]:
    if type(shard) is not int or not 0 <= shard < EXPECTED_SHARDS:
        raise ValueError("Attempt08 claim shard is outside 0..199")
    root = Path(run_dir)
    manifest = validate_package(root)
    validate_launch_authorization(launch_authorization_path, run_dir=root)
    schedule = build_attempt08_spot_schedule(
        load_and_validate_attempt08_plan(root / "hu_joint_policy_m43_attempt08.json")
    )
    spec = schedule[shard]
    payload = {
        "schema": ROOT_CLAIM_SCHEMA,
        "status": "claimed_before_attempt08_root_materialization",
        **_claim_common(
            manifest=manifest,
            manifest_sha256=sha256_file(root / "manifest.json"),
            authorization_sha256=sha256_file(launch_authorization_path),
        ),
        "shard": shard,
        "root_index": shard,
        "root_profile": spec["root_profile"],
        "seeds": spec["seeds"],
        "output_prefix": spec["output_prefix"],
    }
    if set(payload) != _ROOT_CLAIM_KEYS:
        raise AssertionError("Attempt08 root claim schema changed")
    return payload


def _validate_done(
    done: Mapping[str, Any],
    *,
    shard: int,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    launch_authorization_sha256: str,
    expected_spec: Mapping[str, Any],
) -> None:
    if set(done) != _DONE_KEYS:
        raise ValueError(f"Attempt08 DONE fields changed: {shard}")
    expected = {
        "schema": DONE_SCHEMA,
        "status": "complete",
        "run_name": manifest["run_name"],
        "run_id": f"{manifest['run_name']}:shard={shard}",
        "shard": shard,
        "root_index": shard,
        "root_profile": expected_spec["root_profile"],
        "seeds": expected_spec["seeds"],
        "output_prefix": expected_spec["output_prefix"],
        "manifest_sha256": manifest_sha256,
        "launch_authorization_sha256": launch_authorization_sha256,
        "development_open_authorization_sha256": manifest[
            "development_open_authorization_sha256"
        ],
        "source_sha256": manifest["source_zip_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "schedule_sha256": manifest["schedule_sha256"],
        "plan_sha256": manifest["plan_sha256"],
        "model_sha256": manifest["model_sha256"],
        "ai_profiles_sha256": manifest["ai_profiles_sha256"],
        "source_closure_sha256": manifest["source_closure_sha256"],
        "source_model_manifest_sha256": manifest[
            "source_model_manifest_sha256"
        ],
        "source_native_manifest_sha256": manifest[
            "source_native_manifest_sha256"
        ],
        "preflight_aggregate_sha256": manifest["preflight_aggregate_sha256"],
        "preflight_finalization_sha256": manifest[
            "preflight_finalization_sha256"
        ],
        "preflight_proof_sha256": manifest["preflight_proof_sha256"],
        "runtime_semantic_anchor_sha256": manifest[
            "runtime_semantic_anchor_sha256"
        ],
        "runtime_source_closure_sha256": manifest[
            "runtime_source_closure_sha256"
        ],
        "runtime_fingerprint_sha256": manifest["runtime_fingerprint_sha256"],
        "runtime_requirements_sha256": manifest["runtime_requirements_sha256"],
        "gcp_image_name": manifest["gcp_image_name"],
        "gcp_image_id": manifest["gcp_image_id"],
        "gcp_image_self_link": manifest["gcp_image_self_link"],
        "preflight_execution_evidence_sha256": manifest[
            "preflight_execution_evidence_sha256"
        ],
        "native_batch_threads": NATIVE_BATCH_THREADS,
        "teacher_values_are_realized_match_ev": False,
        "selector_executed": False,
        "future_audit_authorized": False,
        "fit_performed": False,
        "threshold_selected": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    if any(done.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Attempt08 DONE identity changed: {shard}")
    for field in (
        "global_claim_sha256",
        "root_claim_sha256",
        "output_sha256",
        "checkpoint_sha256",
        "heartbeat_sha256",
        "generator_summary_sha256",
        "run_log_sha256",
        "boot_image_evidence_sha256",
        "time_report_sha256",
        "resume_commit_sha256",
        "config_sha256",
    ):
        _require_sha256(done.get(field), f"DONE {shard}.{field}")
    _nonnegative_finite(
        done.get("teacher_generator_elapsed_seconds"),
        f"DONE {shard}.teacher elapsed",
    )
    _nonnegative_finite(done.get("process_elapsed_seconds"), f"DONE {shard}.elapsed")
    if type(done.get("peak_rss_bytes")) is not int or done["peak_rss_bytes"] < 0:
        raise ValueError(f"DONE {shard}.peak_rss_bytes is invalid")


def validate_done_file(
    path: str | Path,
    *,
    shard: int,
    run_dir: str | Path,
    launch_authorization_path: str | Path,
) -> dict[str, Any]:
    if type(shard) is not int or not 0 <= shard < EXPECTED_SHARDS:
        raise ValueError("Attempt08 DONE shard is outside 0..199")
    root = Path(run_dir)
    manifest = validate_package(root)
    validate_launch_authorization(launch_authorization_path, run_dir=root)
    schedule = build_attempt08_spot_schedule(
        load_and_validate_attempt08_plan(root / "hu_joint_policy_m43_attempt08.json")
    )
    done = _load_canonical_mapping(Path(path), f"Attempt08 DONE {shard}")
    _validate_done(
        done,
        shard=shard,
        manifest=manifest,
        manifest_sha256=sha256_file(root / "manifest.json"),
        launch_authorization_sha256=sha256_file(launch_authorization_path),
        expected_spec=schedule[shard],
    )
    return done


def claim_complete_output(
    *,
    done_root: str | Path,
    run_dir: str | Path,
    launch_authorization_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Read exactly 200 DONE files and create the pre-content consumption claim."""

    root = Path(run_dir)
    manifest = validate_package(root)
    validate_launch_authorization(launch_authorization_path, run_dir=root)
    manifest_sha = sha256_file(root / "manifest.json")
    authorization_sha = sha256_file(launch_authorization_path)
    validated = validate_complete_done_set(
        done_root=done_root,
        run_dir=root,
        launch_authorization_path=launch_authorization_path,
    )
    done_hashes = dict(validated["done_sha256"])
    payload = {
        "schema": OUTPUT_CONSUMPTION_SCHEMA,
        "status": "claimed_after_all_done_before_any_teacher_read",
        "run_name": manifest["run_name"],
        "manifest_sha256": manifest_sha,
        "launch_authorization_sha256": authorization_sha,
        "development_open_authorization_sha256": manifest[
            "development_open_authorization_sha256"
        ],
        "schedule_sha256": manifest["schedule_sha256"],
        "runtime_semantic_anchor_sha256": manifest[
            "runtime_semantic_anchor_sha256"
        ],
        "runtime_source_closure_sha256": manifest[
            "runtime_source_closure_sha256"
        ],
        "runtime_fingerprint_sha256": manifest["runtime_fingerprint_sha256"],
        "runtime_requirements_sha256": manifest["runtime_requirements_sha256"],
        "gcp_image_name": manifest["gcp_image_name"],
        "gcp_image_id": manifest["gcp_image_id"],
        "gcp_image_self_link": manifest["gcp_image_self_link"],
        "preflight_execution_evidence_sha256": manifest[
            "preflight_execution_evidence_sha256"
        ],
        "done_sha256": done_hashes,
        "expected_shards": EXPECTED_SHARDS,
        "expected_roots": EXPECTED_SHARDS,
        "all_done_markers_verified": True,
        "result_objects_addressed_when_claimed": False,
        "remote_claim_required_before_content_read": True,
        "selector_executed": False,
        "future_audit_authorized": False,
        "fit_performed": False,
        "threshold_selected": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    _write_json(Path(output), payload)
    return payload


def validate_complete_done_set(
    *,
    done_root: str | Path,
    run_dir: str | Path,
    launch_authorization_path: str | Path,
) -> dict[str, Any]:
    """Validate all operational DONE markers without reading result content."""

    root = Path(run_dir)
    manifest = validate_package(root)
    validate_launch_authorization(launch_authorization_path, run_dir=root)
    manifest_sha = sha256_file(root / "manifest.json")
    authorization_sha = sha256_file(launch_authorization_path)
    schedule = build_attempt08_spot_schedule(
        load_and_validate_attempt08_plan(root / "hu_joint_policy_m43_attempt08.json")
    )
    directory = Path(done_root)
    expected_done_names = {
        f"DONE-{shard:03d}.json" for shard in range(EXPECTED_SHARDS)
    }
    actual_done_names = {
        path.name for path in directory.glob("DONE-*.json") if path.is_file()
    }
    if actual_done_names != expected_done_names:
        raise ValueError("Attempt08 DONE directory must contain exact DONE-000..199")
    done_hashes: dict[str, str] = {}
    for shard, spec in enumerate(schedule):
        path = directory / f"DONE-{shard:03d}.json"
        done = _load_canonical_mapping(path, f"Attempt08 DONE {shard}")
        _validate_done(
            done,
            shard=shard,
            manifest=manifest,
            manifest_sha256=manifest_sha,
            launch_authorization_sha256=authorization_sha,
            expected_spec=spec,
        )
        done_hashes[f"{shard:03d}"] = sha256_file(path)
    return {
        "schema": "hu_m43_attempt08_development_done_set_validation_v1",
        "status": "all_200_operational_done_markers_verified",
        "run_name": manifest["run_name"],
        "manifest_sha256": manifest_sha,
        "launch_authorization_sha256": authorization_sha,
        "schedule_sha256": manifest["schedule_sha256"],
        "done_sha256": done_hashes,
        "verified_shards": EXPECTED_SHARDS,
        "selector_executed": False,
        "future_audit_authorized": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }


def validate_consumption_claim(
    path: str | Path,
    *,
    run_dir: str | Path,
    launch_authorization_path: str | Path,
) -> dict[str, Any]:
    claim = _load_canonical_mapping(Path(path), "Attempt08 consumption claim")
    manifest = validate_package(run_dir)
    validate_launch_authorization(launch_authorization_path, run_dir=run_dir)
    if (
        set(claim) != _CONSUMPTION_KEYS
        or claim.get("schema") != OUTPUT_CONSUMPTION_SCHEMA
        or claim.get("status")
        != "claimed_after_all_done_before_any_teacher_read"
        or claim.get("run_name") != manifest["run_name"]
        or claim.get("manifest_sha256")
        != sha256_file(Path(run_dir) / "manifest.json")
        or claim.get("launch_authorization_sha256")
        != sha256_file(launch_authorization_path)
        or claim.get("development_open_authorization_sha256")
        != manifest["development_open_authorization_sha256"]
        or claim.get("schedule_sha256") != manifest["schedule_sha256"]
        or claim.get("runtime_semantic_anchor_sha256")
        != manifest["runtime_semantic_anchor_sha256"]
        or claim.get("preflight_execution_evidence_sha256")
        != manifest["preflight_execution_evidence_sha256"]
        or any(
            claim.get(field) != manifest[field]
            for field in (
                "runtime_source_closure_sha256",
                "runtime_fingerprint_sha256",
                "runtime_requirements_sha256",
                "gcp_image_name",
                "gcp_image_id",
                "gcp_image_self_link",
            )
        )
        or claim.get("expected_shards") != EXPECTED_SHARDS
        or claim.get("expected_roots") != EXPECTED_SHARDS
        or claim.get("all_done_markers_verified") is not True
        or claim.get("result_objects_addressed_when_claimed") is not False
        or claim.get("remote_claim_required_before_content_read") is not True
        or any(
            claim.get(field) is not False
            for field in (
                "selector_executed",
                "future_audit_authorized",
                "fit_performed",
                "threshold_selected",
                "current_profile_mutated",
                "runtime_policy_activated",
            )
        )
    ):
        raise ValueError("Attempt08 consumption claim changed")
    done = claim.get("done_sha256")
    if not isinstance(done, Mapping) or set(done) != {
        f"{index:03d}" for index in range(EXPECTED_SHARDS)
    }:
        raise ValueError("Attempt08 consumption DONE set changed")
    for value in done.values():
        _require_sha256(value, "consumption DONE")
    return claim


def validate_local_remote_consumption_claims(
    *,
    local_path: str | Path,
    remote_path: str | Path,
    run_dir: str | Path,
    launch_authorization_path: str | Path,
) -> tuple[dict[str, Any], str]:
    """Require byte-identical local and downloaded remote claim copies."""

    local = Path(local_path)
    remote = Path(remote_path)
    claim = validate_consumption_claim(
        local,
        run_dir=run_dir,
        launch_authorization_path=launch_authorization_path,
    )
    validate_consumption_claim(
        remote,
        run_dir=run_dir,
        launch_authorization_path=launch_authorization_path,
    )
    if local.read_bytes() != remote.read_bytes():
        raise ValueError("Attempt08 local and remote consumption claims differ")
    digest = sha256_file(local)
    if sha256_file(remote) != digest:
        raise AssertionError("Attempt08 consumption claim bytes changed")
    return claim, digest


def _validate_received_output_boundary(
    *,
    done_root: str | Path,
    run_dir: str | Path,
    launch_authorization_path: str | Path,
    consumption_claim_path: str | Path,
    remote_consumption_claim_path: str | Path,
) -> dict[str, Any]:
    root = Path(run_dir)
    manifest = validate_package(root)
    claim, claim_sha256 = validate_local_remote_consumption_claims(
        local_path=consumption_claim_path,
        remote_path=remote_consumption_claim_path,
        run_dir=root,
        launch_authorization_path=launch_authorization_path,
    )
    done_validation = validate_complete_done_set(
        done_root=done_root,
        run_dir=root,
        launch_authorization_path=launch_authorization_path,
    )
    if claim.get("done_sha256") != done_validation.get("done_sha256"):
        raise ValueError("Attempt08 consumption claim does not bind exact DONE set")
    schedule = build_attempt08_spot_schedule(
        load_and_validate_attempt08_plan(root / "hu_joint_policy_m43_attempt08.json")
    )
    return {
        "root": root,
        "manifest": manifest,
        "claim": claim,
        "claim_sha256": claim_sha256,
        "schedule": schedule,
        "authorization_sha256": sha256_file(launch_authorization_path),
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "authorization_bindings": load_attempt08_development_open_bindings(
            root / "development_open_authorization.json",
            plan=root / "hu_joint_policy_m43_attempt08.json",
            preflight_plan=root / "hu_joint_policy_m43_attempt08_preflight.json",
        ),
    }


def _audit_received_shard_core(
    *,
    directory: str | Path,
    shard: int,
    launch_authorization_path: str | Path,
    consumption_claim_path: str | Path,
    remote_consumption_claim_path: str | Path,
    boundary: Mapping[str, Any],
) -> dict[str, Any]:
    if type(shard) is not int or not 0 <= shard < EXPECTED_SHARDS:
        raise ValueError("Attempt08 received shard is outside 0..199")
    root = Path(boundary["root"])
    manifest = _mapping(boundary["manifest"], "validated manifest")
    claim = _mapping(boundary["claim"], "validated claim")
    claim_sha256 = str(boundary["claim_sha256"])
    authorization_sha = str(boundary["authorization_sha256"])
    manifest_sha = str(boundary["manifest_sha256"])
    schedule = boundary["schedule"]
    if not isinstance(schedule, Sequence):
        raise AssertionError("Attempt08 internal schedule changed")
    spec = schedule[shard]
    received = Path(directory)
    done_path = received / "DONE.json"
    done = _load_canonical_mapping(done_path, "Attempt08 received DONE")
    if sha256_file(done_path) != claim["done_sha256"][f"{shard:03d}"]:
        raise ValueError("Attempt08 received DONE disagrees with consumption claim")
    _validate_done(
        done,
        shard=shard,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        launch_authorization_sha256=authorization_sha,
        expected_spec=spec,
    )
    artifact_fields = {
        "teacher.jsonl": "output_sha256",
        "checkpoint.json": "checkpoint_sha256",
        "heartbeat.json": "heartbeat_sha256",
        "generator_summary.json": "generator_summary_sha256",
        "run.log": "run_log_sha256",
        "boot_image_evidence.json": "boot_image_evidence_sha256",
        "time.txt": "time_report_sha256",
        "resume_commit.json": "resume_commit_sha256",
        "global_claim.json": "global_claim_sha256",
        "root_claim.json": "root_claim_sha256",
    }
    for name, field in artifact_fields.items():
        _require_hash(received / name, str(done[field]), f"received {name}")
    boot_image = _load_canonical_mapping(
        received / "boot_image_evidence.json", "Attempt08 boot image evidence"
    )
    if (
        set(boot_image)
        != {
            "schema",
            "run_name",
            "shard",
            "instance_name",
            "disk_name",
            "source_image",
            "source_image_id",
        }
        or boot_image.get("schema")
        != "hu_m43_attempt08_development_boot_image_evidence_v1"
        or boot_image.get("run_name") != manifest["run_name"]
        or boot_image.get("shard") != shard
        or not isinstance(boot_image.get("instance_name"), str)
        or not boot_image["instance_name"]
        or not isinstance(boot_image.get("disk_name"), str)
        or not boot_image["disk_name"]
        or not str(boot_image.get("source_image", "")).endswith(
            f"projects/debian-cloud/global/images/{manifest['gcp_image_name']}"
        )
        or str(boot_image.get("source_image_id")) != manifest["gcp_image_id"]
    ):
        raise ValueError("Attempt08 received boot image evidence changed")
    global_claim = _load_canonical_mapping(
        received / "global_claim.json", "Attempt08 global claim"
    )
    root_claim = _load_canonical_mapping(
        received / "root_claim.json", "Attempt08 root claim"
    )
    claim_common = _claim_common(
        manifest=manifest,
        manifest_sha256=manifest_sha,
        authorization_sha256=authorization_sha,
    )
    expected_global_claim = {
        "schema": GLOBAL_CLAIM_SCHEMA,
        "status": "claimed_before_any_attempt08_development_root",
        **claim_common,
    }
    expected_root_claim = {
        "schema": ROOT_CLAIM_SCHEMA,
        "status": "claimed_before_attempt08_root_materialization",
        **claim_common,
        "shard": shard,
        "root_index": shard,
        "root_profile": spec["root_profile"],
        "seeds": spec["seeds"],
        "output_prefix": spec["output_prefix"],
    }
    if global_claim != expected_global_claim or set(global_claim) != _GLOBAL_CLAIM_KEYS:
        raise ValueError("Attempt08 received global claim changed")
    if root_claim != expected_root_claim or set(root_claim) != _ROOT_CLAIM_KEYS:
        raise ValueError("Attempt08 received root claim changed")
    checkpoint = _load_canonical_mapping(
        received / "checkpoint.json", "Attempt08 checkpoint"
    )
    heartbeat = _load_canonical_mapping(
        received / "heartbeat.json", "Attempt08 heartbeat"
    )
    summary = _load_canonical_mapping(
        received / "generator_summary.json", "Attempt08 generator summary"
    )
    resume_commit = _load_canonical_mapping(
        received / "resume_commit.json", "Attempt08 resume commit"
    )
    time_elapsed_seconds, time_peak_rss_bytes = _parse_gnu_time_report(
        received / "time.txt"
    )
    authorization_bindings = _mapping(
        boundary["authorization_bindings"], "validated authorization bindings"
    )
    checkpoint_keys = {
        "schema",
        "config_sha256",
        "plan_sha256",
        "ai_profiles_sha256",
        "model_sha256",
        *authorization_bindings,
        "completed_roots",
        "target_roots",
        "root_index",
        "partial_sha256",
        "updated_unix_seconds",
        "generator_elapsed_seconds",
        "generator_peak_rss_bytes",
    }
    summary_keys = {
        "schema",
        "status",
        "root_index",
        "completed_roots",
        "target_roots",
        "config_sha256",
        "plan_sha256",
        "ai_profiles_sha256",
        "model_sha256",
        *authorization_bindings,
        "output_sha256",
        "elapsed_seconds",
        "generator_peak_rss_bytes",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
    expected_resume_commit = {
        "schema": "hu_m43_attempt08_development_resume_commit_v1",
        "status": "committed_completed_root_pair",
        "run_name": manifest["run_name"],
        "shard": shard,
        "manifest_sha256": manifest_sha,
        "launch_authorization_sha256": authorization_sha,
        "output_sha256": done["output_sha256"],
        "checkpoint_sha256": done["checkpoint_sha256"],
        "heartbeat_sha256": done["heartbeat_sha256"],
        "generator_summary_sha256": done["generator_summary_sha256"],
        "run_log_sha256": done["run_log_sha256"],
        "boot_image_evidence_sha256": done["boot_image_evidence_sha256"],
        "time_report_sha256": done["time_report_sha256"],
        "global_claim_sha256": done["global_claim_sha256"],
        "root_claim_sha256": done["root_claim_sha256"],
        "generator_elapsed_seconds": done["teacher_generator_elapsed_seconds"],
        "process_elapsed_seconds": done["process_elapsed_seconds"],
        "peak_rss_bytes": resume_commit.get("peak_rss_bytes"),
    }
    if (
        set(checkpoint) != checkpoint_keys
        or set(summary) != summary_keys
        or set(heartbeat) != summary_keys | {"summary_schema"}
        or checkpoint.get("schema") != ATTEMPT08_CHECKPOINT_SCHEMA
        or checkpoint.get("completed_roots") != 1
        or checkpoint.get("target_roots") != 1
        or checkpoint.get("root_index") != shard
        or checkpoint.get("config_sha256") != done["config_sha256"]
        or checkpoint.get("partial_sha256") != done["output_sha256"]
        or not isinstance(checkpoint.get("updated_unix_seconds"), (int, float))
        or isinstance(checkpoint.get("updated_unix_seconds"), bool)
        or not math.isfinite(float(checkpoint["updated_unix_seconds"]))
        or float(checkpoint["updated_unix_seconds"]) < 0.0
        or heartbeat.get("schema") != ATTEMPT08_HEARTBEAT_SCHEMA
        or heartbeat.get("status") != "complete"
        or heartbeat.get("summary_schema") != ATTEMPT08_SHARD_SUMMARY_SCHEMA
        or summary.get("schema") != ATTEMPT08_SHARD_SUMMARY_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("root_index") != shard
        or summary.get("completed_roots") != 1
        or summary.get("target_roots") != 1
        or summary.get("current_profile_mutated") is not False
        or summary.get("runtime_policy_activated") is not False
        or summary.get("config_sha256") != done["config_sha256"]
        or summary.get("output_sha256") != done["output_sha256"]
        or heartbeat
        != {
            **summary,
            "schema": ATTEMPT08_HEARTBEAT_SCHEMA,
            "summary_schema": ATTEMPT08_SHARD_SUMMARY_SCHEMA,
        }
        or any(
            checkpoint.get(field) != value
            or summary.get(field) != value
            or heartbeat.get(field) != value
            for field, value in {
                "plan_sha256": manifest["plan_sha256"],
                "ai_profiles_sha256": manifest["ai_profiles_sha256"],
                "model_sha256": manifest["model_sha256"],
                **authorization_bindings,
            }.items()
        )
        or checkpoint.get("generator_elapsed_seconds")
        != summary.get("elapsed_seconds")
        or checkpoint.get("generator_peak_rss_bytes")
        != summary.get("generator_peak_rss_bytes")
        or type(summary.get("generator_peak_rss_bytes")) is not int
        or summary["generator_peak_rss_bytes"] < 0
        or int(done["peak_rss_bytes"]) < summary["generator_peak_rss_bytes"]
        or set(resume_commit) != set(expected_resume_commit)
        or any(
            resume_commit.get(field) != value
            for field, value in expected_resume_commit.items()
        )
        or type(resume_commit.get("peak_rss_bytes")) is not int
        or resume_commit["peak_rss_bytes"] < 0
        or done["peak_rss_bytes"]
        != max(
            summary["generator_peak_rss_bytes"],
            time_peak_rss_bytes,
            resume_commit["peak_rss_bytes"],
        )
        or float(done["process_elapsed_seconds"]) != time_elapsed_seconds
        or float(summary.get("elapsed_seconds"))
        != float(done["teacher_generator_elapsed_seconds"])
    ):
        raise ValueError("Attempt08 received lifecycle metadata changed")
    raw = (received / "teacher.jsonl").read_bytes()
    if raw.count(b"\n") != 1 or not raw.endswith(b"\n"):
        raise ValueError("Attempt08 received shard must contain exactly one row")
    try:
        row = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Attempt08 received row is invalid") from exc
    if not isinstance(row, Mapping) or raw != canonical_json_bytes(row):
        raise ValueError("Attempt08 received row is not canonical")
    validated = _validate_row(
        row,
        root_index=shard,
        expected_seeds=spec["seeds"],
        authorization_bindings=authorization_bindings,
        expected_run_id=done["run_id"],
    )
    if validated["config_sha256"] != done["config_sha256"]:
        raise ValueError("Attempt08 DONE config disagrees with producer row")
    result = {
        "schema": RECEIVE_AUDIT_SCHEMA,
        "status": "verified_after_local_and_remote_consumption_claims",
        "run_name": manifest["run_name"],
        "shard": shard,
        "root_index": shard,
        "root_profile": spec["root_profile"],
        "done_sha256": sha256_file(done_path),
        "output_sha256": done["output_sha256"],
        "config_sha256": done["config_sha256"],
        "local_consumption_claim_sha256": claim_sha256,
        "remote_consumption_claim_sha256": sha256_file(
            remote_consumption_claim_path
        ),
        "teacher_generator_elapsed_seconds": done[
            "teacher_generator_elapsed_seconds"
        ],
        "process_elapsed_seconds": done["process_elapsed_seconds"],
        "peak_rss_bytes": done["peak_rss_bytes"],
        "boot_image_evidence_sha256": done["boot_image_evidence_sha256"],
        "runtime_semantic_anchor_sha256": manifest[
            "runtime_semantic_anchor_sha256"
        ],
        "preflight_execution_evidence_sha256": manifest[
            "preflight_execution_evidence_sha256"
        ],
        "runtime_source_closure_sha256": manifest[
            "runtime_source_closure_sha256"
        ],
        "runtime_fingerprint_sha256": manifest["runtime_fingerprint_sha256"],
        "runtime_requirements_sha256": manifest["runtime_requirements_sha256"],
        "gcp_image_name": manifest["gcp_image_name"],
        "gcp_image_id": manifest["gcp_image_id"],
        "gcp_image_self_link": manifest["gcp_image_self_link"],
        "selector_executed": False,
        "future_audit_authorized": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    if set(result) != _RECEIVE_AUDIT_KEYS:
        raise AssertionError("Attempt08 receive audit schema changed")
    return result


def audit_received_shard(
    *,
    directory: str | Path,
    shard: int,
    done_root: str | Path,
    run_dir: str | Path,
    launch_authorization_path: str | Path,
    consumption_claim_path: str | Path,
    remote_consumption_claim_path: str | Path,
) -> dict[str, Any]:
    """Reprove all 200 DONE markers, then validate one downloaded shard."""

    boundary = _validate_received_output_boundary(
        done_root=done_root,
        run_dir=run_dir,
        launch_authorization_path=launch_authorization_path,
        consumption_claim_path=consumption_claim_path,
        remote_consumption_claim_path=remote_consumption_claim_path,
    )
    return _audit_received_shard_core(
        directory=directory,
        shard=shard,
        launch_authorization_path=launch_authorization_path,
        consumption_claim_path=consumption_claim_path,
        remote_consumption_claim_path=remote_consumption_claim_path,
        boundary=boundary,
    )


def validate_completed_shard_bundle(
    *,
    directory: str | Path,
    shard: int,
    run_dir: str | Path,
    launch_authorization_path: str | Path,
) -> dict[str, Any]:
    """Reopen one completed producer bundle without claiming its result set.

    This is the worker-side retry validator.  It deliberately cannot replace
    the receiver's all-200 DONE boundary or consumption claims; it only proves
    that an already-published single-shard bundle is internally complete and
    byte-bound to the frozen package before a retry treats remote DONE as
    terminal.
    """

    if type(shard) is not int or not 0 <= shard < EXPECTED_SHARDS:
        raise ValueError("Attempt08 completed shard is outside 0..199")
    root = Path(run_dir)
    manifest = validate_package(root)
    validate_launch_authorization(launch_authorization_path, run_dir=root)
    manifest_sha = sha256_file(root / "manifest.json")
    authorization_sha = sha256_file(launch_authorization_path)
    schedule = build_attempt08_spot_schedule(
        load_and_validate_attempt08_plan(root / "hu_joint_policy_m43_attempt08.json")
    )
    received = Path(directory)
    done_path = received / "DONE.json"
    done = _load_canonical_mapping(done_path, "Attempt08 completed bundle DONE")
    _validate_done(
        done,
        shard=shard,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        launch_authorization_sha256=authorization_sha,
        expected_spec=schedule[shard],
    )
    done_sha = sha256_file(done_path)
    # The core only consumes this minimal synthetic boundary to bind the local
    # DONE bytes; no consumption claim is created or implied here.
    boundary = {
        "root": root,
        "manifest": manifest,
        "claim": {"done_sha256": {f"{shard:03d}": done_sha}},
        "claim_sha256": done_sha,
        "schedule": schedule,
        "authorization_sha256": authorization_sha,
        "manifest_sha256": manifest_sha,
        "authorization_bindings": load_attempt08_development_open_bindings(
            root / "development_open_authorization.json",
            plan=root / "hu_joint_policy_m43_attempt08.json",
            preflight_plan=root / "hu_joint_policy_m43_attempt08_preflight.json",
        ),
    }
    audit = _audit_received_shard_core(
        directory=received,
        shard=shard,
        launch_authorization_path=launch_authorization_path,
        consumption_claim_path=done_path,
        remote_consumption_claim_path=done_path,
        boundary=boundary,
    )
    return {
        "schema": "hu_m43_attempt08_development_completed_bundle_validation_v1",
        "status": "single_shard_bundle_fully_reopened_without_consumption_claim",
        "run_name": manifest["run_name"],
        "shard": shard,
        "done_sha256": done_sha,
        "output_sha256": audit["output_sha256"],
        "config_sha256": audit["config_sha256"],
        "boot_image_evidence_sha256": audit["boot_image_evidence_sha256"],
        "selector_executed": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }


def merge_received_shards(
    *,
    received_root: str | Path,
    audit_root: str | Path,
    done_root: str | Path,
    run_dir: str | Path,
    launch_authorization_path: str | Path,
    consumption_claim_path: str | Path,
    remote_consumption_claim_path: str | Path,
    output: str | Path,
    receipt: str | Path,
) -> dict[str, Any]:
    """Merge exact 0..199 rows after independent received-shard audits."""

    boundary = _validate_received_output_boundary(
        done_root=done_root,
        run_dir=run_dir,
        launch_authorization_path=launch_authorization_path,
        consumption_claim_path=consumption_claim_path,
        remote_consumption_claim_path=remote_consumption_claim_path,
    )
    manifest = _mapping(boundary["manifest"], "validated manifest")
    claim_sha256 = str(boundary["claim_sha256"])
    chunks: list[bytes] = []
    audit_hashes: list[str] = []
    profiles: Counter[str] = Counter()
    for shard in range(EXPECTED_SHARDS):
        audit_path = Path(audit_root) / f"audit-{shard:03d}.json"
        audit = _load_canonical_mapping(audit_path, f"Attempt08 audit {shard}")
        recomputed = _audit_received_shard_core(
            directory=Path(received_root) / f"shard_{shard:03d}",
            shard=shard,
            launch_authorization_path=launch_authorization_path,
            consumption_claim_path=consumption_claim_path,
            remote_consumption_claim_path=remote_consumption_claim_path,
            boundary=boundary,
        )
        if set(audit) != _RECEIVE_AUDIT_KEYS or audit != recomputed:
            raise ValueError(f"Attempt08 received audit changed: {shard}")
        teacher = Path(received_root) / f"shard_{shard:03d}" / "teacher.jsonl"
        _require_hash(teacher, str(audit["output_sha256"]), f"shard {shard} output")
        chunks.append(teacher.read_bytes())
        profiles[str(audit["root_profile"])] += 1
        audit_hashes.append(sha256_file(audit_path))
    if profiles != Counter({profile: 40 for profile in M43_ATTEMPT08_PROFILES}):
        raise ValueError("Attempt08 received profiles are not balanced 40/profile")
    merged = b"".join(chunks)
    output_path = Path(output)
    _atomic_write(output_path, merged)
    result = {
        "schema": RECEIVE_MERGE_SCHEMA,
        "status": "complete_without_selection",
        "run_name": manifest["run_name"],
        "manifest_sha256": sha256_file(Path(run_dir) / "manifest.json"),
        "launch_authorization_sha256": sha256_file(launch_authorization_path),
        "local_consumption_claim_sha256": claim_sha256,
        "remote_consumption_claim_sha256": sha256_file(
            remote_consumption_claim_path
        ),
        "schedule_sha256": manifest["schedule_sha256"],
        "runtime_semantic_anchor_sha256": manifest[
            "runtime_semantic_anchor_sha256"
        ],
        "preflight_execution_evidence_sha256": manifest[
            "preflight_execution_evidence_sha256"
        ],
        "runtime_source_closure_sha256": manifest[
            "runtime_source_closure_sha256"
        ],
        "runtime_fingerprint_sha256": manifest["runtime_fingerprint_sha256"],
        "runtime_requirements_sha256": manifest["runtime_requirements_sha256"],
        "gcp_image_name": manifest["gcp_image_name"],
        "gcp_image_id": manifest["gcp_image_id"],
        "gcp_image_self_link": manifest["gcp_image_self_link"],
        "roots": EXPECTED_SHARDS,
        "root_indices": "0..199",
        "profiles": dict(profiles),
        "received_audit_sha256": audit_hashes,
        "merged_sha256": sha256_file(output_path),
        "selector_command_required": (
            "python -B -m ofc_regular.select_hu_m43_attempt08_development"
        ),
        "selector_executed": False,
        "selector_must_execute_exactly_once": True,
        "future_audit_authorized": False,
        "fit_performed": False,
        "threshold_selected": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    if set(result) != _RECEIVE_MERGE_KEYS:
        raise AssertionError("Attempt08 receive merge schema changed")
    _write_json(Path(receipt), result)
    return result


def _selector_paths(run_dir: str | Path) -> dict[str, Path]:
    run = Path(run_dir).resolve()
    if len(run.parents) < 3:
        raise ValueError("Attempt08 run directory cannot identify repository root")
    repo = run.parents[2]
    base = (
        repo
        / "outputs/hu_joint_policy/m43_attempt08_development"
        / run.name
    )
    return {
        "merged": base / "merged/teacher.jsonl",
        "merge_receipt": base / "merged/merge_receipt.json",
        "claim": base / "selector/CLAIM.json",
        "remote_claim": base / "selector/REMOTE_CLAIM.json",
        "execution": base / "selector/EXECUTION_STARTED.json",
        "decision": base / "selector/decision.json",
        "receipt": base / "selector/decision_receipt.json",
    }


def validate_received_merge(
    *,
    run_dir: str | Path,
    launch_authorization_path: str | Path,
    consumption_claim_path: str | Path,
    remote_consumption_claim_path: str | Path,
) -> dict[str, Any]:
    """Recompute the canonical receive merge into scratch and compare bytes."""

    paths = _selector_paths(run_dir)
    base = paths["merged"].parents[1]
    with tempfile.TemporaryDirectory(prefix="attempt08-receive-revalidate-") as value:
        scratch = Path(value)
        recomputed_output = scratch / "teacher.jsonl"
        recomputed_receipt = scratch / "merge_receipt.json"
        result = merge_received_shards(
            received_root=base / "shards",
            audit_root=base / "audits",
            done_root=base / "done",
            run_dir=run_dir,
            launch_authorization_path=launch_authorization_path,
            consumption_claim_path=consumption_claim_path,
            remote_consumption_claim_path=remote_consumption_claim_path,
            output=recomputed_output,
            receipt=recomputed_receipt,
        )
        if (
            paths["merged"].read_bytes() != recomputed_output.read_bytes()
            or paths["merge_receipt"].read_bytes() != recomputed_receipt.read_bytes()
        ):
            raise ValueError("Attempt08 canonical received merge changed")
    return result


def _require_exact_path(actual: str | Path, expected: Path, label: str) -> Path:
    path = Path(actual).resolve()
    if os.path.normcase(str(path)) != os.path.normcase(str(expected.resolve())):
        raise ValueError(f"Attempt08 {label} must use the run-global canonical path")
    return path


def build_selector_claim(
    *,
    run_dir: str | Path,
    launch_authorization_path: str | Path,
    consumption_claim_path: str | Path,
    remote_consumption_claim_path: str | Path,
    merged_path: str | Path,
    merge_receipt_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Claim the sole run-global selector execution before evaluating gates."""

    paths = _selector_paths(run_dir)
    merged = _require_exact_path(merged_path, paths["merged"], "merged input")
    merge_receipt_file = _require_exact_path(
        merge_receipt_path, paths["merge_receipt"], "merge receipt"
    )
    output_path = _require_exact_path(output, paths["claim"], "selector claim")
    manifest = validate_package(run_dir)
    validate_launch_authorization(launch_authorization_path, run_dir=run_dir)
    _, consumption_sha = validate_local_remote_consumption_claims(
        local_path=consumption_claim_path,
        remote_path=remote_consumption_claim_path,
        run_dir=run_dir,
        launch_authorization_path=launch_authorization_path,
    )
    receipt = _load_canonical_mapping(
        merge_receipt_file, "Attempt08 merge receipt"
    )
    if (
        set(receipt) != _RECEIVE_MERGE_KEYS
        or receipt.get("schema") != RECEIVE_MERGE_SCHEMA
        or receipt.get("status") != "complete_without_selection"
        or receipt.get("run_name") != manifest["run_name"]
        or receipt.get("merged_sha256") != sha256_file(merged)
        or receipt.get("local_consumption_claim_sha256") != consumption_sha
        or receipt.get("remote_consumption_claim_sha256")
        != sha256_file(remote_consumption_claim_path)
        or receipt.get("selector_executed") is not False
        or receipt.get("selector_must_execute_exactly_once") is not True
    ):
        raise ValueError("Attempt08 merge receipt cannot open selector claim")
    for forbidden in (paths["execution"], paths["decision"], paths["receipt"]):
        if forbidden.exists():
            raise FileExistsError(
                "Attempt08 selector lifecycle already started or completed"
            )
    selector_source_sha256 = sha256_file(
        Path(__file__).with_name("select_hu_m43_attempt08_development.py")
    )
    payload = {
        "schema": SELECTOR_CLAIM_SCHEMA,
        "status": "claimed_before_single_frozen_gate_evaluation",
        "run_name": manifest["run_name"],
        "manifest_sha256": sha256_file(Path(run_dir) / "manifest.json"),
        "launch_authorization_sha256": sha256_file(launch_authorization_path),
        "local_consumption_claim_sha256": consumption_sha,
        "remote_consumption_claim_sha256": sha256_file(
            remote_consumption_claim_path
        ),
        "merge_receipt_sha256": sha256_file(merge_receipt_file),
        "merged_sha256": sha256_file(merged),
        "plan_sha256": manifest["plan_sha256"],
        "preflight_plan_sha256": manifest["preflight_plan_sha256"],
        "development_open_authorization_sha256": manifest[
            "development_open_authorization_sha256"
        ],
        "preflight_execution_evidence_sha256": manifest[
            "preflight_execution_evidence_sha256"
        ],
        "selector_source_sha256": selector_source_sha256,
        "canonical_decision_path": str(paths["decision"]),
        "canonical_receipt_path": str(paths["receipt"]),
        "gate_evaluation_count_before_claim": 0,
        "selector_executed": False,
        "future_audit_authorized": False,
        "fit_performed": False,
        "threshold_selected": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    if set(payload) != _SELECTOR_CLAIM_KEYS:
        raise AssertionError("Attempt08 selector claim schema changed")
    _write_json(output_path, payload)
    return payload


def execute_selector_once(
    *,
    run_dir: str | Path,
    selector_claim_path: str | Path,
    remote_selector_claim_path: str | Path,
    output: str | Path,
    receipt: str | Path,
) -> dict[str, Any]:
    """Consume the sole selector claim; any crash requires a separate recovery audit."""

    paths = _selector_paths(run_dir)
    local_claim_path = _require_exact_path(
        selector_claim_path, paths["claim"], "selector claim"
    )
    remote_claim_path = _require_exact_path(
        remote_selector_claim_path, paths["remote_claim"], "remote selector claim"
    )
    output_path = _require_exact_path(output, paths["decision"], "decision output")
    receipt_path = _require_exact_path(receipt, paths["receipt"], "decision receipt")
    claim = _load_canonical_mapping(local_claim_path, "Attempt08 selector claim")
    remote_claim = _load_canonical_mapping(
        remote_claim_path, "Attempt08 remote selector claim"
    )
    if (
        set(claim) != _SELECTOR_CLAIM_KEYS
        or claim != remote_claim
        or local_claim_path.read_bytes() != remote_claim_path.read_bytes()
        or claim.get("schema") != SELECTOR_CLAIM_SCHEMA
        or claim.get("status") != "claimed_before_single_frozen_gate_evaluation"
        or claim.get("run_name") != Path(run_dir).name
        or claim.get("selector_source_sha256")
        != sha256_file(Path(__file__).with_name("select_hu_m43_attempt08_development.py"))
        or claim.get("merged_sha256") != sha256_file(paths["merged"])
        or claim.get("merge_receipt_sha256") != sha256_file(paths["merge_receipt"])
        or claim.get("gate_evaluation_count_before_claim") != 0
        or claim.get("selector_executed") is not False
    ):
        raise ValueError("Attempt08 selector claim changed")
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError("Attempt08 selector decision already exists")
    execution = {
        "schema": SELECTOR_EXECUTION_SCHEMA,
        "status": "started_single_evaluation_no_automatic_recovery",
        "run_name": claim["run_name"],
        "selector_claim_sha256": sha256_file(local_claim_path),
        "remote_selector_claim_sha256": sha256_file(remote_claim_path),
        "merge_receipt_sha256": claim["merge_receipt_sha256"],
        "merged_sha256": claim["merged_sha256"],
        "selector_source_sha256": claim["selector_source_sha256"],
        "gate_evaluation_count_before_start": 0,
        "selector_executed": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    _write_json(paths["execution"], execution)
    run_root = Path(run_dir)
    report = select_attempt08_development(
        input_path=paths["merged"],
        plan_path=run_root / "hu_joint_policy_m43_attempt08.json",
        preflight_plan_path=run_root / "hu_joint_policy_m43_attempt08_preflight.json",
        development_open_authorization_path=(
            run_root / "development_open_authorization.json"
        ),
        run_name=claim["run_name"],
    )
    if report.get("decision_contract", {}).get("gate_evaluation_count") != 1:
        raise ValueError("Attempt08 selector did not evaluate frozen gates exactly once")
    write_attempt08_development_decision(
        output_path,
        report,
        _lifecycle_token=_SELECTOR_WRITE_LIFECYCLE_TOKEN,
    )
    result = {
        "schema": SELECTOR_RECEIPT_SCHEMA,
        "status": "single_frozen_gate_evaluation_complete",
        "run_name": claim["run_name"],
        "selector_claim_sha256": sha256_file(local_claim_path),
        "remote_selector_claim_sha256": sha256_file(remote_claim_path),
        "execution_marker_sha256": sha256_file(paths["execution"]),
        "merge_receipt_sha256": claim["merge_receipt_sha256"],
        "merged_sha256": claim["merged_sha256"],
        "selector_source_sha256": claim["selector_source_sha256"],
        "decision_sha256": sha256_file(output_path),
        "decision": report["decision"],
        "search_freeze_authorized": report["search_freeze_authorized"],
        "gate_evaluation_count": 1,
        "selector_executed": True,
        "future_audit_authorized": False,
        "fit_performed": False,
        "threshold_selected": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    _write_json(receipt_path, result)
    return result


def validate_selector_completion(*, run_dir: str | Path) -> dict[str, Any]:
    """Validate an already-finished one-shot selector without reevaluating it."""

    paths = _selector_paths(run_dir)
    root = Path(run_dir)
    manifest = validate_package(root)
    launch = root / "launch_authorization.json"
    validate_launch_authorization(launch, run_dir=root)
    claim = _load_canonical_mapping(paths["claim"], "Attempt08 selector claim")
    remote = _load_canonical_mapping(
        paths["remote_claim"], "Attempt08 remote selector claim"
    )
    execution = _load_canonical_mapping(
        paths["execution"], "Attempt08 selector execution"
    )
    receipt = _load_canonical_mapping(paths["receipt"], "Attempt08 selector receipt")
    try:
        decision_raw = paths["decision"].read_bytes()
        decision = json.loads(decision_raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Attempt08 selector decision is unreadable") from exc
    selector_source_sha256 = sha256_file(
        Path(__file__).with_name("select_hu_m43_attempt08_development.py")
    )
    if (
        not isinstance(decision, Mapping)
        or set(claim) != _SELECTOR_CLAIM_KEYS
        or claim != remote
        or paths["claim"].read_bytes() != paths["remote_claim"].read_bytes()
        or claim.get("schema") != SELECTOR_CLAIM_SCHEMA
        or claim.get("status") != "claimed_before_single_frozen_gate_evaluation"
        or claim.get("run_name") != manifest["run_name"]
        or claim.get("manifest_sha256") != sha256_file(root / "manifest.json")
        or claim.get("launch_authorization_sha256") != sha256_file(launch)
        or claim.get("merged_sha256") != sha256_file(paths["merged"])
        or claim.get("merge_receipt_sha256") != sha256_file(paths["merge_receipt"])
        or claim.get("selector_source_sha256") != selector_source_sha256
        or claim.get("canonical_decision_path") != str(paths["decision"])
        or claim.get("canonical_receipt_path") != str(paths["receipt"])
        or claim.get("gate_evaluation_count_before_claim") != 0
        or claim.get("selector_executed") is not False
    ):
        raise ValueError("Attempt08 completed selector claim changed")
    expected_execution = {
        "schema": SELECTOR_EXECUTION_SCHEMA,
        "status": "started_single_evaluation_no_automatic_recovery",
        "run_name": claim["run_name"],
        "selector_claim_sha256": sha256_file(paths["claim"]),
        "remote_selector_claim_sha256": sha256_file(paths["remote_claim"]),
        "merge_receipt_sha256": claim["merge_receipt_sha256"],
        "merged_sha256": claim["merged_sha256"],
        "selector_source_sha256": claim["selector_source_sha256"],
        "gate_evaluation_count_before_start": 0,
        "selector_executed": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    receipt_keys = {
        "schema",
        "status",
        "run_name",
        "selector_claim_sha256",
        "remote_selector_claim_sha256",
        "execution_marker_sha256",
        "merge_receipt_sha256",
        "merged_sha256",
        "selector_source_sha256",
        "decision_sha256",
        "decision",
        "search_freeze_authorized",
        "gate_evaluation_count",
        "selector_executed",
        "future_audit_authorized",
        "fit_performed",
        "threshold_selected",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
    decision_contract = decision.get("decision_contract")
    source = decision.get("source")
    science = decision.get("science_boundary")
    if (
        execution != expected_execution
        or set(receipt) != receipt_keys
        or receipt.get("schema") != SELECTOR_RECEIPT_SCHEMA
        or receipt.get("status") != "single_frozen_gate_evaluation_complete"
        or receipt.get("run_name") != claim["run_name"]
        or receipt.get("selector_claim_sha256") != sha256_file(paths["claim"])
        or receipt.get("remote_selector_claim_sha256")
        != sha256_file(paths["remote_claim"])
        or receipt.get("execution_marker_sha256") != sha256_file(paths["execution"])
        or receipt.get("merge_receipt_sha256") != claim["merge_receipt_sha256"]
        or receipt.get("merged_sha256") != claim["merged_sha256"]
        or receipt.get("selector_source_sha256") != selector_source_sha256
        or receipt.get("decision_sha256") != sha256_file(paths["decision"])
        or receipt.get("decision") != decision.get("decision")
        or receipt.get("search_freeze_authorized")
        != decision.get("search_freeze_authorized")
        or receipt.get("gate_evaluation_count") != 1
        or receipt.get("selector_executed") is not True
        or not isinstance(decision_contract, Mapping)
        or decision_contract.get("gate_evaluation_count") != 1
        or not isinstance(source, Mapping)
        or source.get("run_name") != claim["run_name"]
        or source.get("input_jsonl_sha256") != claim["merged_sha256"]
        or source.get("selector_source_sha256") != selector_source_sha256
        or not isinstance(science, Mapping)
        or science.get("current_profile_mutated") is not False
        or science.get("runtime_policy_activated") is not False
        or any(
            receipt.get(field) is not False
            for field in (
                "future_audit_authorized",
                "fit_performed",
                "threshold_selected",
                "current_profile_mutated",
                "runtime_policy_activated",
            )
        )
    ):
        raise ValueError("Attempt08 completed selector lifecycle changed")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    package = subparsers.add_parser("package")
    package.add_argument("--repo-root", required=True, type=Path)
    package.add_argument("--run-dir", required=True, type=Path)
    package.add_argument("--run-name", required=True)
    package.add_argument("--plan", required=True, type=Path)
    package.add_argument("--preflight-plan", required=True, type=Path)
    package.add_argument("--preflight-aggregate", required=True, type=Path)
    package.add_argument("--preflight-execution-evidence", required=True, type=Path)
    package.add_argument("--preflight-spot-run-dir", required=True, type=Path)
    package.add_argument(
        "--preflight-spot-launch-authorization", required=True, type=Path
    )
    package.add_argument("--preflight-spot-local-evidence", required=True, type=Path)
    package.add_argument("--preflight-spot-jobs-root", required=True, type=Path)
    package.add_argument("--preflight-finalization", required=True, type=Path)
    package.add_argument("--preflight-source", required=True, type=Path)
    for slot in ATTEMPT08_PREFLIGHT_SLOTS:
        package.add_argument(
            f"--{slot.replace('_', '-')}", required=True, type=Path
        )
    package.add_argument(
        "--development-open-authorization", required=True, type=Path
    )
    package.add_argument("--model", required=True, type=Path)
    package.add_argument("--startup", required=True, type=Path)
    package.add_argument("--resume-existing", action="store_true")
    schedule = subparsers.add_parser("schedule")
    schedule.add_argument("--plan", required=True, type=Path)
    schedule.add_argument("--output", required=True, type=Path)
    authorize = subparsers.add_parser("authorize-launch")
    authorize.add_argument("--run-dir", required=True, type=Path)
    authorize.add_argument("--output", required=True, type=Path)
    validate = subparsers.add_parser("validate-launch")
    validate.add_argument("--run-dir", required=True, type=Path)
    validate.add_argument("--authorization", required=True, type=Path)
    done_set = subparsers.add_parser("validate-done-set")
    done_set.add_argument("--done-root", required=True, type=Path)
    done_set.add_argument("--run-dir", required=True, type=Path)
    done_set.add_argument("--authorization", required=True, type=Path)
    done = subparsers.add_parser("validate-done")
    done.add_argument("--input", required=True, type=Path)
    done.add_argument("--shard", required=True, type=int)
    done.add_argument("--run-dir", required=True, type=Path)
    done.add_argument("--authorization", required=True, type=Path)
    completed = subparsers.add_parser("validate-completed-bundle")
    completed.add_argument("--directory", required=True, type=Path)
    completed.add_argument("--shard", required=True, type=int)
    completed.add_argument("--run-dir", required=True, type=Path)
    completed.add_argument("--authorization", required=True, type=Path)
    global_claim = subparsers.add_parser("global-claim")
    global_claim.add_argument("--run-dir", required=True, type=Path)
    global_claim.add_argument("--authorization", required=True, type=Path)
    global_claim.add_argument("--output", required=True, type=Path)
    root_claim = subparsers.add_parser("root-claim")
    root_claim.add_argument("--run-dir", required=True, type=Path)
    root_claim.add_argument("--authorization", required=True, type=Path)
    root_claim.add_argument("--shard", required=True, type=int)
    root_claim.add_argument("--output", required=True, type=Path)
    claim = subparsers.add_parser("claim")
    claim.add_argument("--done-root", required=True, type=Path)
    claim.add_argument("--run-dir", required=True, type=Path)
    claim.add_argument("--authorization", required=True, type=Path)
    claim.add_argument("--output", required=True, type=Path)
    claims = subparsers.add_parser("validate-claims")
    claims.add_argument("--run-dir", required=True, type=Path)
    claims.add_argument("--authorization", required=True, type=Path)
    claims.add_argument("--local-claim", required=True, type=Path)
    claims.add_argument("--remote-claim", required=True, type=Path)
    audit = subparsers.add_parser("audit-received")
    audit.add_argument("--directory", required=True, type=Path)
    audit.add_argument("--shard", required=True, type=int)
    audit.add_argument("--done-root", required=True, type=Path)
    audit.add_argument("--run-dir", required=True, type=Path)
    audit.add_argument("--authorization", required=True, type=Path)
    audit.add_argument("--local-claim", required=True, type=Path)
    audit.add_argument("--remote-claim", required=True, type=Path)
    audit.add_argument("--output", required=True, type=Path)
    merge = subparsers.add_parser("merge-received")
    merge.add_argument("--received-root", required=True, type=Path)
    merge.add_argument("--audit-root", required=True, type=Path)
    merge.add_argument("--done-root", required=True, type=Path)
    merge.add_argument("--run-dir", required=True, type=Path)
    merge.add_argument("--authorization", required=True, type=Path)
    merge.add_argument("--local-claim", required=True, type=Path)
    merge.add_argument("--remote-claim", required=True, type=Path)
    merge.add_argument("--output", required=True, type=Path)
    merge.add_argument("--receipt", required=True, type=Path)
    validate_merge = subparsers.add_parser("validate-merged-receive")
    validate_merge.add_argument("--run-dir", required=True, type=Path)
    validate_merge.add_argument("--authorization", required=True, type=Path)
    validate_merge.add_argument("--local-claim", required=True, type=Path)
    validate_merge.add_argument("--remote-claim", required=True, type=Path)
    selector_claim = subparsers.add_parser("selector-claim")
    selector_claim.add_argument("--run-dir", required=True, type=Path)
    selector_claim.add_argument("--authorization", required=True, type=Path)
    selector_claim.add_argument("--local-consumption-claim", required=True, type=Path)
    selector_claim.add_argument("--remote-consumption-claim", required=True, type=Path)
    selector_claim.add_argument("--merged", required=True, type=Path)
    selector_claim.add_argument("--merge-receipt", required=True, type=Path)
    selector_claim.add_argument("--output", required=True, type=Path)
    select_once = subparsers.add_parser("select-once")
    select_once.add_argument("--run-dir", required=True, type=Path)
    select_once.add_argument("--selector-claim", required=True, type=Path)
    select_once.add_argument("--remote-selector-claim", required=True, type=Path)
    select_once.add_argument("--output", required=True, type=Path)
    select_once.add_argument("--receipt", required=True, type=Path)
    validate_selector = subparsers.add_parser("validate-selector-completion")
    validate_selector.add_argument("--run-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "package":
        result = package_attempt08_spot(
            repo_root=args.repo_root,
            run_dir=args.run_dir,
            run_name=args.run_name,
            plan_path=args.plan,
            preflight_plan_path=args.preflight_plan,
            preflight_aggregate_path=args.preflight_aggregate,
            preflight_execution_evidence_path=args.preflight_execution_evidence,
            preflight_spot_run_dir=args.preflight_spot_run_dir,
            preflight_spot_launch_authorization_path=(
                args.preflight_spot_launch_authorization
            ),
            preflight_spot_local_evidence_path=args.preflight_spot_local_evidence,
            preflight_spot_jobs_root=args.preflight_spot_jobs_root,
            preflight_finalization_path=args.preflight_finalization,
            preflight_proof_paths={
                slot: getattr(args, slot) for slot in ATTEMPT08_PREFLIGHT_SLOTS
            },
            preflight_source_path=args.preflight_source,
            development_open_authorization_path=(
                args.development_open_authorization
            ),
            model_path=args.model,
            startup_path=args.startup,
            resume_existing=args.resume_existing,
        )
    elif args.command == "schedule":
        plan = load_and_validate_attempt08_plan(args.plan)
        _atomic_write(args.output, _schedule_bytes(build_attempt08_spot_schedule(plan)))
        result: Mapping[str, Any] = {
            "status": "complete",
            "shards": EXPECTED_SHARDS,
            "output": str(args.output),
        }
    elif args.command == "authorize-launch":
        result = authorize_attempt08_launch(run_dir=args.run_dir, output=args.output)
    elif args.command == "validate-launch":
        result = validate_launch_authorization(
            args.authorization, run_dir=args.run_dir
        )
    elif args.command == "validate-done-set":
        result = validate_complete_done_set(
            done_root=args.done_root,
            run_dir=args.run_dir,
            launch_authorization_path=args.authorization,
        )
    elif args.command == "validate-done":
        result = validate_done_file(
            args.input,
            shard=args.shard,
            run_dir=args.run_dir,
            launch_authorization_path=args.authorization,
        )
    elif args.command == "validate-completed-bundle":
        result = validate_completed_shard_bundle(
            directory=args.directory,
            shard=args.shard,
            run_dir=args.run_dir,
            launch_authorization_path=args.authorization,
        )
    elif args.command == "global-claim":
        result = build_global_claim(
            run_dir=args.run_dir,
            launch_authorization_path=args.authorization,
        )
        _write_json(args.output, result)
    elif args.command == "root-claim":
        result = build_root_claim(
            run_dir=args.run_dir,
            launch_authorization_path=args.authorization,
            shard=args.shard,
        )
        _write_json(args.output, result)
    elif args.command == "claim":
        result = claim_complete_output(
            done_root=args.done_root,
            run_dir=args.run_dir,
            launch_authorization_path=args.authorization,
            output=args.output,
        )
    elif args.command == "validate-claims":
        claim, digest = validate_local_remote_consumption_claims(
            local_path=args.local_claim,
            remote_path=args.remote_claim,
            run_dir=args.run_dir,
            launch_authorization_path=args.authorization,
        )
        result = {
            "schema": "hu_m43_attempt08_development_claim_pair_validation_v1",
            "status": "local_and_remote_consumption_claims_byte_identical",
            "run_name": claim["run_name"],
            "consumption_claim_sha256": digest,
            "selector_executed": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        }
    elif args.command == "audit-received":
        result = audit_received_shard(
            directory=args.directory,
            shard=args.shard,
            done_root=args.done_root,
            run_dir=args.run_dir,
            launch_authorization_path=args.authorization,
            consumption_claim_path=args.local_claim,
            remote_consumption_claim_path=args.remote_claim,
        )
        _write_json(args.output, result)
    elif args.command == "merge-received":
        result = merge_received_shards(
            received_root=args.received_root,
            audit_root=args.audit_root,
            done_root=args.done_root,
            run_dir=args.run_dir,
            launch_authorization_path=args.authorization,
            consumption_claim_path=args.local_claim,
            remote_consumption_claim_path=args.remote_claim,
            output=args.output,
            receipt=args.receipt,
        )
    elif args.command == "validate-merged-receive":
        result = validate_received_merge(
            run_dir=args.run_dir,
            launch_authorization_path=args.authorization,
            consumption_claim_path=args.local_claim,
            remote_consumption_claim_path=args.remote_claim,
        )
    elif args.command == "selector-claim":
        result = build_selector_claim(
            run_dir=args.run_dir,
            launch_authorization_path=args.authorization,
            consumption_claim_path=args.local_consumption_claim,
            remote_consumption_claim_path=args.remote_consumption_claim,
            merged_path=args.merged,
            merge_receipt_path=args.merge_receipt,
            output=args.output,
        )
    elif args.command == "select-once":
        result = execute_selector_once(
            run_dir=args.run_dir,
            selector_claim_path=args.selector_claim,
            remote_selector_claim_path=args.remote_selector_claim,
            output=args.output,
            receipt=args.receipt,
        )
    elif args.command == "validate-selector-completion":
        result = validate_selector_completion(run_dir=args.run_dir)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DONE_SCHEMA",
    "EXPECTED_SHARDS",
    "GLOBAL_CLAIM_SCHEMA",
    "LAUNCH_AUTHORIZATION_SCHEMA",
    "MAX_WAVE_SHARDS",
    "NATIVE_BATCH_THREADS",
    "OUTPUT_CONSUMPTION_SCHEMA",
    "PACKAGE_MANIFEST_SCHEMA",
    "RECOMMENDED_MACHINE_TYPE",
    "ROOT_CLAIM_SCHEMA",
    "audit_received_shard",
    "authorize_attempt08_launch",
    "build_attempt08_spot_schedule",
    "build_global_claim",
    "build_root_claim",
    "build_selector_claim",
    "claim_complete_output",
    "merge_received_shards",
    "execute_selector_once",
    "package_attempt08_spot",
    "validate_complete_done_set",
    "validate_consumption_claim",
    "validate_completed_shard_bundle",
    "validate_done_file",
    "validate_launch_authorization",
    "validate_local_remote_consumption_claims",
    "validate_received_merge",
    "validate_package",
    "validate_selector_completion",
]
