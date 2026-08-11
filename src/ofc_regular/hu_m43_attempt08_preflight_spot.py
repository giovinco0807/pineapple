"""Immutable five-job Spot lifecycle for the Attempt08 correctness preflight.

Packaging is inert.  A separate canonical launch authorization is required
before instances may be created.  Status reads only ``DONE.json``; receive
refuses to open any proof until all five DONE records are present and valid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from .aggregate_hu_m43_attempt08_preflight import aggregate_preflight_proofs
from .finalize_hu_m43_attempt08_preflight import (
    ATTEMPT08_PREFLIGHT_EXECUTION_EVIDENCE_SCHEMA,
    finalize_preflight,
    validate_preflight_execution_evidence,
)
from .hu_m43_attempt06_spot import (
    PINNED_MODEL_MANIFEST_SHA256,
    PINNED_NATIVE_MANIFEST_SHA256,
)
from .hu_m43_attempt07_preflight_spot import (
    ATTEMPT06_BASE_MANIFEST_SHA256,
    DEFAULT_BASE_RUN_DIR,
    _deterministic_zip,
    _is_lower_sha256,
    _load_mapping,
    _tree_digest,
    _tree_rows,
    _validate_base_package,
    _validate_source_zip,
    _write_atomic,
    _write_atomic_no_clobber,
    sha256_file,
)
from .hu_m43_attempt08_contract import (
    AI_PROFILES_SHA256,
    ATTEMPT07_CLOSEOUT_SHA256,
    ATTEMPT07_SELECTION_SHA256,
    ATTEMPT08_LAMBDA_MODEL_SHA256,
    M43_ATTEMPT08_PLAN_SHA256,
    load_and_validate_attempt08_plan,
    validate_attempt08_artifact_bindings,
)
from .hu_m43_attempt08_runtime_anchor import (
    ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
    ATTEMPT08_RUNTIME_SEMANTIC_FILES,
    runtime_semantic_anchor_payload,
    validate_runtime_semantic_anchor,
    validate_runtime_semantic_anchor_payload,
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
from .run_hu_m43_attempt08_preflight import (
    ATTEMPT08_PREFLIGHT_NATIVE_BATCH_THREADS,
    ATTEMPT08_PREFLIGHT_PLAN_SHA256,
    ATTEMPT08_PREFLIGHT_SLOTS,
    ATTEMPT08_PREFLIGHT_SOURCE_ROOTS,
    ATTEMPT07_PREFLIGHT_SOURCE_SHA256,
    DEFAULT_AI_PROFILES_PATH,
    DEFAULT_ATTEMPT08_PLAN_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREFLIGHT_PLAN_PATH,
    DEFAULT_SOURCE_PATH,
    canonical_json_bytes,
    load_preflight_plan,
)


PACKAGE_MANIFEST_SCHEMA = "hu_m43_attempt08_preflight_spot_package_manifest_v1"
PACKAGE_RESULT_SCHEMA = "hu_m43_attempt08_preflight_spot_package_result_v1"
OVERLAY_CLOSURE_SCHEMA = "hu_m43_attempt08_preflight_spot_source_closure_v1"
SCHEDULE_SCHEMA = "hu_m43_attempt08_preflight_spot_job_v1"
LOCAL_EVIDENCE_SCHEMA = "hu_m43_attempt08_preflight_spot_local_evidence_v1"
LAUNCH_AUTHORIZATION_SCHEMA = (
    "hu_m43_attempt08_preflight_spot_launch_authorization_v1"
)
DONE_SCHEMA = "hu_m43_attempt08_preflight_spot_done_v1"
CHECKPOINT_SCHEMA = "hu_m43_attempt08_preflight_spot_checkpoint_v1"
HEARTBEAT_SCHEMA = "hu_m43_attempt08_preflight_spot_heartbeat_v1"
SUMMARY_SCHEMA = "hu_m43_attempt08_preflight_spot_summary_v1"
RECEIVE_RECEIPT_SCHEMA = "hu_m43_attempt08_preflight_spot_receive_v1"

ATTEMPT08_PREFLIGHT_MACHINE_TYPE = "c4-highmem-4"
ATTEMPT08_PREFLIGHT_JOB_COUNT = 5
ATTEMPT08_PREFLIGHT_IMAGE_PROJECT = "debian-cloud"
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STARTUP_PATH = _REPO_ROOT / "scripts/startup_hu_m43_attempt08_preflight.sh"
DEFAULT_REQUIREMENTS_PATH = (
    _REPO_ROOT / "configs/hu_m43_attempt08_runtime_requirements.txt"
)
DEFAULT_CLOSEOUT_PATH = _REPO_ROOT / "configs/hu_joint_policy_m43_attempt07_closeout.json"
DEFAULT_SELECTION_PATH = _REPO_ROOT / (
    "outputs/hu_joint_policy/m43_attempt07_development/"
    "regular-hu-m43-attempt07-development100-20260714-150046/merged/"
    "development_arm_selection.json"
)

_MANIFEST_KEYS = frozenset(
    {
        "schema", "status", "run_name", "jobs", "source_roots", "machine_type",
        "native_batch_threads", "gcp_image_name", "gcp_image_id",
        "gcp_image_self_link", "base_attempt06_manifest_sha256",
        "base_attempt06_package_tree_sha256", "base_attempt06_source_zip_sha256",
        "package_tree_sha256", "overlay_closure_sha256", "source_zip_sha256",
        "source_zip_bytes", "startup_sha256", "schedule_sha256",
        "preflight_plan_sha256", "attempt08_plan_sha256", "source_merged_sha256",
        "model_sha256", "ai_profiles_sha256", "attempt07_closeout_sha256",
        "attempt07_selection_sha256", "runtime_semantic_anchor_sha256",
        "runtime_source_closure_sha256", "runtime_source_file_count",
        "runtime_requirements_sha256", "runtime_fingerprint_sha256",
        "source_model_manifest_sha256", "source_native_manifest_sha256",
        "new_root_generated", "teacher_executed", "gcloud_invoked",
        "instances_created", "policy_science_performed", "development200_authorized",
        "future_audit_authorized", "current_profile_resolved",
        "current_profile_mutated", "runtime_policy_activated",
    }
)
_AUTH_KEYS = frozenset(
    {
        "schema", "status", "run_name", "manifest_sha256", "schedule_sha256",
        "startup_sha256", "source_zip_sha256", "attempt08_plan_sha256",
        "preflight_plan_sha256", "source_merged_sha256", "model_sha256",
        "ai_profiles_sha256", "attempt07_closeout_sha256",
        "attempt07_selection_sha256", "runtime_semantic_anchor_sha256",
        "runtime_source_closure_sha256", "runtime_requirements_sha256",
        "runtime_fingerprint_sha256", "source_model_manifest_sha256",
        "source_native_manifest_sha256", "gcp_image_name", "gcp_image_id",
        "gcp_image_self_link", "machine_type", "native_batch_threads", "jobs",
        "source_roots", "local_evidence_sha256", "spot_authorized",
        "new_root_generation_allowed", "policy_science_allowed",
        "development200_allowed", "future_audit_allowed", "current_profile_resolved",
        "current_profile_mutated", "runtime_policy_activated",
    }
)
_DONE_KEYS = frozenset(
    {
        "schema", "status", "run_name", "job_index", "job_id", "slot",
        "source_root_index", "batch_child_selectors", "output_prefix", "machine_type",
        "native_batch_threads", "proof_sha256", "teacher_elapsed_seconds",
        "process_peak_rss_bytes", "checkpoint_sha256", "heartbeat_sha256",
        "summary_sha256", "run_log_sha256", "boot_image_evidence_sha256",
        "manifest_sha256", "authorization_sha256",
        "schedule_sha256", "attempt08_plan_sha256", "preflight_plan_sha256",
        "source_merged_sha256", "model_sha256", "ai_profiles_sha256",
        "attempt07_closeout_sha256", "attempt07_selection_sha256",
        "runtime_semantic_anchor_sha256", "runtime_source_closure_sha256",
        "runtime_requirements_sha256", "runtime_fingerprint_sha256",
        "source_model_manifest_sha256", "source_native_manifest_sha256",
        "gcp_image_name", "gcp_image_id", "teacher_action_or_value_details_exported",
        "new_root_generated", "policy_science_performed", "current_profile_resolved",
        "current_profile_mutated", "runtime_policy_activated",
    }
)
_LOCAL_GATES = (
    "correctness_smoke", "determinism_harness", "scalar_batch_parity_harness",
    "package_closure", "startup_shell_syntax",
)


def _is_safe_run_name(value: str) -> bool:
    return bool(value) and all(c in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in value)


def build_preflight_schedule() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "schema": SCHEDULE_SCHEMA,
            "job_index": index,
            "job_id": slot,
            "slot": slot,
            "source_root_index": root,
            "batch_child_selectors": batch,
            "machine_type": ATTEMPT08_PREFLIGHT_MACHINE_TYPE,
            "native_batch_threads": ATTEMPT08_PREFLIGHT_NATIVE_BATCH_THREADS,
            "output_prefix": f"job_{index:03d}_{slot}",
            "checkpoint_enabled": True,
            "heartbeat_enabled": True,
            "self_delete": True,
            "new_root_generation_allowed": False,
        }
        for index, (slot, (root, batch)) in enumerate(ATTEMPT08_PREFLIGHT_SLOTS.items())
    )


def validate_preflight_schedule(rows: Sequence[Mapping[str, Any]]) -> None:
    if tuple(dict(row) for row in rows) != build_preflight_schedule():
        raise ValueError("Attempt08 Spot preflight schedule changed")


def _schedule_bytes() -> bytes:
    return b"".join(canonical_json_bytes(row) for row in build_preflight_schedule())


def _load_canonical(path: Path, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise ValueError(f"Attempt08 {label} is not canonical JSON")
    return value


def _critical_manifest_values() -> dict[str, Any]:
    return {
        "attempt08_plan_sha256": M43_ATTEMPT08_PLAN_SHA256,
        "preflight_plan_sha256": ATTEMPT08_PREFLIGHT_PLAN_SHA256,
        "source_merged_sha256": ATTEMPT07_PREFLIGHT_SOURCE_SHA256,
        "model_sha256": ATTEMPT08_LAMBDA_MODEL_SHA256,
        "ai_profiles_sha256": AI_PROFILES_SHA256,
        "attempt07_closeout_sha256": ATTEMPT07_CLOSEOUT_SHA256,
        "attempt07_selection_sha256": ATTEMPT07_SELECTION_SHA256,
        "runtime_semantic_anchor_sha256": ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,
        "runtime_source_closure_sha256": ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
        "runtime_requirements_sha256": ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
        "runtime_fingerprint_sha256": ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
        "source_model_manifest_sha256": PINNED_MODEL_MANIFEST_SHA256,
        "source_native_manifest_sha256": PINNED_NATIVE_MANIFEST_SHA256,
        "gcp_image_name": ATTEMPT08_GCP_IMAGE_NAME,
        "gcp_image_id": ATTEMPT08_GCP_IMAGE_ID,
        "gcp_image_self_link": ATTEMPT08_GCP_IMAGE_SELF_LINK,
    }


def _validate_manifest_shape(value: Mapping[str, Any]) -> None:
    critical = _critical_manifest_values()
    if (
        set(value) != _MANIFEST_KEYS
        or value.get("schema") != PACKAGE_MANIFEST_SCHEMA
        or value.get("status") != "packaged_only_no_execution"
        or not isinstance(value.get("run_name"), str)
        or not _is_safe_run_name(value["run_name"])
        or value.get("jobs") != 5
        or value.get("source_roots") != list(ATTEMPT08_PREFLIGHT_SOURCE_ROOTS)
        or value.get("machine_type") != ATTEMPT08_PREFLIGHT_MACHINE_TYPE
        or value.get("native_batch_threads") != 4
        or value.get("runtime_source_file_count")
        != ATTEMPT08_RUNTIME_SOURCE_FILE_COUNT
        or any(value.get(key) != expected for key, expected in critical.items())
        or any(
            value.get(key) is not False
            for key in (
                "new_root_generated", "teacher_executed", "gcloud_invoked",
                "instances_created", "policy_science_performed",
                "development200_authorized", "future_audit_authorized",
                "current_profile_resolved", "current_profile_mutated",
                "runtime_policy_activated",
            )
        )
    ):
        raise ValueError("Attempt08 Spot package manifest changed")
    for key in (name for name in value if name.endswith("_sha256")):
        if not _is_lower_sha256(value[key]):
            raise ValueError(f"Attempt08 Spot manifest hash changed: {key}")


def validate_package_artifacts(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    manifest_path = root / "manifest.json"
    manifest = _load_canonical(manifest_path, "Spot manifest")
    _validate_manifest_shape(manifest)
    if (root / "shards_manifest.jsonl").read_bytes() != _schedule_bytes():
        raise ValueError("Attempt08 Spot schedule bytes changed")
    bindings = {
        "shards_manifest.jsonl": "schedule_sha256",
        "startup_hu_m43_attempt08_preflight.sh": "startup_sha256",
        "source.zip": "source_zip_sha256",
        "hu_joint_policy_m43_attempt08.json": "attempt08_plan_sha256",
        "hu_joint_policy_m43_attempt08_preflight.json": "preflight_plan_sha256",
    }
    for name, key in bindings.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != manifest[key]:
            raise ValueError(f"Attempt08 Spot package binding changed: {name}")
    package = root / "package_src"
    if (
        (root / "startup_hu_m43_attempt08_preflight.sh").read_bytes()
        != (package / "scripts/startup_hu_m43_attempt08_preflight.sh").read_bytes()
    ):
        raise ValueError("Attempt08 Spot startup differs from anchored package source")
    closure_path = package / "preflight_source_closure.json"
    closure = _load_canonical(closure_path, "Spot source closure")
    if (
        closure.get("schema") != OVERLAY_CLOSURE_SCHEMA
        or closure.get("runtime_semantic_anchor_sha256")
        != ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256
        or closure.get("runtime_source_closure_sha256")
        != ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256
        or closure.get("new_root_generated") is not False
        or closure.get("teacher_executed") is not False
        or sha256_file(closure_path) != manifest["overlay_closure_sha256"]
    ):
        raise ValueError("Attempt08 Spot source closure identity changed")
    expected_files = closure.get("files")
    if not isinstance(expected_files, list):
        raise ValueError("Attempt08 Spot source closure files changed")
    actual_rows = _tree_rows(package, exclude=frozenset({"preflight_source_closure.json"}))
    if expected_files != actual_rows or _tree_digest(_tree_rows(package)) != manifest[
        "package_tree_sha256"
    ]:
        raise ValueError("Attempt08 Spot source closure tree changed")
    anchor_path = package / "runtime_semantic_anchor.json"
    anchor = _load_canonical(anchor_path, "runtime anchor")
    validate_runtime_semantic_anchor_payload(
        anchor,
        expected_source_closure_sha256=ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
        expected_anchor_sha256=ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,
        expected_source_file_count=ATTEMPT08_RUNTIME_SOURCE_FILE_COUNT,
    )
    if sha256_file(anchor_path) != ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256:
        raise ValueError("Attempt08 Spot runtime anchor file changed")
    validate_runtime_semantic_anchor(
        repository_root=package,
        expected_source_closure_sha256=ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
        expected_anchor_sha256=ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,
        expected_source_file_count=ATTEMPT08_RUNTIME_SOURCE_FILE_COUNT,
        runtime_artifact_root=package,
        model_manifest_path=package / "source_model_manifest.json",
        native_manifest_path=package / "source_native_manifest.json",
        requirements_path=package / "configs/hu_m43_attempt08_runtime_requirements.txt",
    )
    package_rows = {row["path"]: row for row in _tree_rows(package)}
    _validate_source_zip(
        source_zip=root / "source.zip",
        expected_sha256=manifest["source_zip_sha256"],
        package_rows=package_rows,
    )
    return manifest


def package_preflight_run(
    *,
    repo_root: str | Path,
    run_dir: str | Path,
    run_name: str,
    base_run_dir: str | Path = DEFAULT_BASE_RUN_DIR,
    startup: str | Path = DEFAULT_STARTUP_PATH,
    resume_existing: bool = False,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    destination = Path(run_dir).resolve()
    if destination != (repo / "outputs" / "gcp_runs" / run_name).resolve():
        raise ValueError("Attempt08 Spot run_dir must be outputs/gcp_runs/<run_name>")
    if not _is_safe_run_name(run_name):
        raise ValueError("Attempt08 Spot run_name is unsafe")
    if destination.exists():
        if not resume_existing:
            raise FileExistsError(f"Attempt08 Spot package exists: {destination}")
        manifest = validate_package_artifacts(destination)
        if manifest["run_name"] != run_name:
            raise ValueError("Attempt08 resumed package run_name changed")
        return {"schema": PACKAGE_RESULT_SCHEMA, "status": "verified_existing_package",
                "run_name": run_name, "manifest_sha256": sha256_file(destination / "manifest.json"),
                "jobs": 5, "teacher_executed": False, "gcloud_invoked": False}

    base_manifest, base_manifest_sha, base_tree_sha = _validate_base_package(
        Path(base_run_dir).resolve()
    )
    plan = load_and_validate_attempt08_plan(DEFAULT_ATTEMPT08_PLAN_PATH)
    load_preflight_plan(DEFAULT_PREFLIGHT_PLAN_PATH)
    validate_attempt08_artifact_bindings(plan, repository_root=repo)
    fixed = {
        DEFAULT_ATTEMPT08_PLAN_PATH: M43_ATTEMPT08_PLAN_SHA256,
        DEFAULT_PREFLIGHT_PLAN_PATH: ATTEMPT08_PREFLIGHT_PLAN_SHA256,
        DEFAULT_SOURCE_PATH: ATTEMPT07_PREFLIGHT_SOURCE_SHA256,
        DEFAULT_MODEL_PATH: ATTEMPT08_LAMBDA_MODEL_SHA256,
        DEFAULT_AI_PROFILES_PATH: AI_PROFILES_SHA256,
        DEFAULT_CLOSEOUT_PATH: ATTEMPT07_CLOSEOUT_SHA256,
        DEFAULT_SELECTION_PATH: ATTEMPT07_SELECTION_SHA256,
        DEFAULT_REQUIREMENTS_PATH: ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
    }
    for path, expected in fixed.items():
        if not Path(path).is_file() or sha256_file(path) != expected:
            raise ValueError(f"Attempt08 Spot immutable input changed: {path}")
    startup_path = Path(startup).resolve()
    if (
        startup_path != DEFAULT_STARTUP_PATH.resolve()
        or not startup_path.is_file()
        or startup_path.is_symlink()
        or startup_path.stat().st_size < 1
    ):
        raise ValueError("Attempt08 Spot startup is missing")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + ".staging-" + uuid.uuid4().hex)
    try:
        package = staging / "package_src"
        shutil.copytree(Path(base_run_dir).resolve() / "package_src", package)
        # Replace the complete Python tree, not a hand-selected overlay.
        source_py_root = repo / "src" / "ofc_regular"
        target_py_root = package / "src" / "ofc_regular"
        shutil.rmtree(target_py_root)
        shutil.copytree(
            source_py_root,
            target_py_root,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        overlays = {
            "configs/hu_joint_policy_m43_attempt08.json": DEFAULT_ATTEMPT08_PLAN_PATH,
            "configs/hu_joint_policy_m43_attempt08_preflight.json": DEFAULT_PREFLIGHT_PLAN_PATH,
            "configs/hu_joint_policy_m43_attempt07_closeout.json": DEFAULT_CLOSEOUT_PATH,
            "configs/hu_m43_attempt08_runtime_requirements.txt": DEFAULT_REQUIREMENTS_PATH,
            "configs/fl_ev_regular_2k.json": repo / "configs/fl_ev_regular_2k.json",
            "preflight_source/teacher.jsonl": DEFAULT_SOURCE_PATH,
            str(DEFAULT_SELECTION_PATH.relative_to(repo)).replace("\\", "/"): DEFAULT_SELECTION_PATH,
            str(DEFAULT_MODEL_PATH.relative_to(repo)).replace("\\", "/"): DEFAULT_MODEL_PATH,
        }
        overlays.update(
            {relative: repo / relative for relative in ATTEMPT08_RUNTIME_SEMANTIC_FILES}
        )
        for relative, origin in overlays.items():
            target = package / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, target)
        schedule_bytes = _schedule_bytes()
        _write_atomic(package / "shards_manifest.jsonl", schedule_bytes)
        _write_atomic(staging / "shards_manifest.jsonl", schedule_bytes)
        anchor = runtime_semantic_anchor_payload(package)
        validate_runtime_semantic_anchor_payload(
            anchor,
            expected_source_closure_sha256=ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
            expected_anchor_sha256=ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,
            expected_source_file_count=ATTEMPT08_RUNTIME_SOURCE_FILE_COUNT,
        )
        _write_atomic(package / "runtime_semantic_anchor.json", canonical_json_bytes(anchor))
        validate_runtime_semantic_anchor(
            repository_root=package,
            expected_source_closure_sha256=ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
            expected_anchor_sha256=ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,
            expected_source_file_count=ATTEMPT08_RUNTIME_SOURCE_FILE_COUNT,
            runtime_artifact_root=package,
            model_manifest_path=package / "source_model_manifest.json",
            native_manifest_path=package / "source_native_manifest.json",
            requirements_path=package / "configs/hu_m43_attempt08_runtime_requirements.txt",
        )
        closure_rows = _tree_rows(package, exclude=frozenset({"preflight_source_closure.json"}))
        closure = {
            "schema": OVERLAY_CLOSURE_SCHEMA,
            "status": "full_python_tree_plus_pinned_runtime_no_execution",
            "base_attempt06_manifest_sha256": base_manifest_sha,
            "files": closure_rows,
            "file_count": len(closure_rows),
            **_critical_manifest_values(),
            "new_root_generated": False,
            "teacher_executed": False,
            "current_profile_mutated": False,
        }
        _write_atomic(package / "preflight_source_closure.json", canonical_json_bytes(closure))
        source_zip = staging / "source.zip"
        _deterministic_zip(package, source_zip)
        shutil.copy2(startup_path, staging / "startup_hu_m43_attempt08_preflight.sh")
        shutil.copy2(DEFAULT_ATTEMPT08_PLAN_PATH, staging / "hu_joint_policy_m43_attempt08.json")
        shutil.copy2(DEFAULT_PREFLIGHT_PLAN_PATH, staging / "hu_joint_policy_m43_attempt08_preflight.json")
        manifest = {
            "schema": PACKAGE_MANIFEST_SCHEMA, "status": "packaged_only_no_execution",
            "run_name": run_name, "jobs": 5,
            "source_roots": list(ATTEMPT08_PREFLIGHT_SOURCE_ROOTS),
            "machine_type": ATTEMPT08_PREFLIGHT_MACHINE_TYPE,
            "native_batch_threads": 4, "gcp_image_name": ATTEMPT08_GCP_IMAGE_NAME,
            "gcp_image_id": ATTEMPT08_GCP_IMAGE_ID,
            "gcp_image_self_link": ATTEMPT08_GCP_IMAGE_SELF_LINK,
            "base_attempt06_manifest_sha256": base_manifest_sha,
            "base_attempt06_package_tree_sha256": base_tree_sha,
            "base_attempt06_source_zip_sha256": base_manifest["source_zip_sha256"],
            "package_tree_sha256": _tree_digest(_tree_rows(package)),
            "overlay_closure_sha256": sha256_file(package / "preflight_source_closure.json"),
            "source_zip_sha256": sha256_file(source_zip), "source_zip_bytes": source_zip.stat().st_size,
            "startup_sha256": sha256_file(staging / "startup_hu_m43_attempt08_preflight.sh"),
            "schedule_sha256": sha256_file(staging / "shards_manifest.jsonl"),
            "runtime_source_file_count": ATTEMPT08_RUNTIME_SOURCE_FILE_COUNT,
            **_critical_manifest_values(),
            "new_root_generated": False, "teacher_executed": False,
            "gcloud_invoked": False, "instances_created": False,
            "policy_science_performed": False, "development200_authorized": False,
            "future_audit_authorized": False, "current_profile_resolved": False,
            "current_profile_mutated": False, "runtime_policy_activated": False,
        }
        _write_atomic(staging / "manifest.json", canonical_json_bytes(manifest))
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    validate_package_artifacts(destination)
    return {"schema": PACKAGE_RESULT_SCHEMA, "status": "packaged_without_execution",
            "run_name": run_name, "manifest_sha256": sha256_file(destination / "manifest.json"),
            "jobs": 5, "teacher_executed": False, "gcloud_invoked": False}


def create_local_evidence_receipt(**_: Any) -> dict[str, Any]:
    """Reject the old caller-asserted evidence API.

    Launch evidence must be produced by :func:`run_local_launch_gates`, which
    executes the producer-owned fixed commands and captures their outputs.
    """

    raise ValueError("Attempt08 caller-supplied local evidence is forbidden")


def _fixed_local_gate_commands(manifest_path: Path) -> dict[str, list[str]]:
    python = str(Path(sys.executable).resolve())
    mocked = (
        "tests/test_run_hu_m43_attempt08_preflight.py::"
        "test_mocked_runner_is_redacted_and_proves_batch_and_cross_mode_hashes"
    )
    bash = shutil.which("bash")
    if not bash:
        raise ValueError("Attempt08 startup syntax gate requires bash")
    return {
        "correctness_smoke": [
            python, "-m", "pytest",
            "tests/test_run_hu_m43_attempt08_preflight.py",
            "tests/test_aggregate_hu_m43_attempt08_preflight.py",
            "tests/test_hu_m43_attempt08_preflight_spot.py",
            "tests/test_finalize_hu_m43_attempt08_preflight.py",
            "-p", "no:cacheprovider", "-q",
        ],
        "determinism_harness": [
            python, "-m", "pytest", mocked, "-p", "no:cacheprovider", "-q"
        ],
        "scalar_batch_parity_harness": [
            python, "-m", "pytest", mocked, "-p", "no:cacheprovider", "-q"
        ],
        "package_closure": [
            python, "-c",
            (
                "from ofc_regular.hu_m43_attempt08_preflight_spot import "
                "validate_package_artifacts as v; v(r'"
                + str(manifest_path.parent)
                + "')"
            ),
        ],
        "startup_shell_syntax": [
            # The gates run with package_src as cwd.  A relative POSIX path is
            # portable across Debian bash and Windows' WSL bash.exe; an
            # absolute C:\\ path is interpreted by bash as escaped text.
            bash, "-n", "../startup_hu_m43_attempt08_preflight.sh"
        ],
    }


def run_local_launch_gates(
    *, manifest_path: str | Path, output: str | Path, timeout_seconds: int = 600
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).resolve()
    validate_package_artifacts(manifest_file.parent)
    commands = _fixed_local_gate_commands(manifest_file)
    results: dict[str, Any] = {}
    environment = dict(os.environ)
    package_root = manifest_file.parent / "package_src"
    environment["PYTHONPATH"] = str(package_root / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in _LOCAL_GATES:
        command = commands[name]
        completed = subprocess.run(
            command,
            cwd=package_root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        row = {
            "command": command,
            "command_sha256": hashlib.sha256(
                canonical_json_bytes(command)
            ).hexdigest(),
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
            "stderr": completed.stderr,
            "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
        }
        results[name] = row
        if completed.returncode != 0:
            raise ValueError(f"Attempt08 fixed local gate failed: {name}")
    payload = {
        "schema": LOCAL_EVIDENCE_SCHEMA,
        "status": "pass",
        "manifest_sha256": sha256_file(manifest_file),
        "results": results,
        "spot_result_observed": False,
        "current_profile_mutated": False,
    }
    _write_atomic_no_clobber(Path(output), canonical_json_bytes(payload))
    return payload


def _validate_local_evidence(
    path: Path, manifest_sha: str, manifest_path: Path
) -> None:
    value = _load_canonical(path, "local evidence")
    if (
        set(value) != {"schema", "status", "manifest_sha256", "results",
                       "spot_result_observed", "current_profile_mutated"}
        or value.get("schema") != LOCAL_EVIDENCE_SCHEMA or value.get("status") != "pass"
        or value.get("manifest_sha256") != manifest_sha
        or value.get("spot_result_observed") is not False
        or value.get("current_profile_mutated") is not False
    ):
        raise ValueError("Attempt08 local evidence changed")
    results = value.get("results")
    if not isinstance(results, dict) or set(results) != set(_LOCAL_GATES):
        raise ValueError("Attempt08 local evidence checks changed")
    commands = _fixed_local_gate_commands(manifest_path.resolve())
    for name, row in results.items():
        if (
            not isinstance(row, dict)
            or set(row) != {"command", "command_sha256", "exit_code", "stdout",
                            "stdout_sha256", "stderr", "stderr_sha256"}
            or row.get("command") != commands[name]
            or row.get("command_sha256")
            != hashlib.sha256(canonical_json_bytes(commands[name])).hexdigest()
            or row.get("exit_code") != 0
            or not isinstance(row.get("stdout"), str)
            or row.get("stdout_sha256")
            != hashlib.sha256(row["stdout"].encode("utf-8")).hexdigest()
            or not isinstance(row.get("stderr"), str)
            or row.get("stderr_sha256")
            != hashlib.sha256(row["stderr"].encode("utf-8")).hexdigest()
        ):
            raise ValueError("Attempt08 local evidence check changed")


def create_launch_authorization(
    *, manifest_path: str | Path, evidence_output: str | Path, output: str | Path
) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    manifest = validate_package_artifacts(manifest_file.parent)
    manifest_sha = sha256_file(manifest_file)
    run_local_launch_gates(manifest_path=manifest_file, output=evidence_output)
    local_evidence_path = Path(evidence_output)
    _validate_local_evidence(
        Path(local_evidence_path), manifest_sha, manifest_file
    )
    payload = {
        "schema": LAUNCH_AUTHORIZATION_SCHEMA, "status": "authorized_for_five_spot_jobs_only",
        "run_name": manifest["run_name"], "manifest_sha256": manifest_sha,
        **{key: manifest[key] for key in (
            "schedule_sha256", "startup_sha256", "source_zip_sha256",
            "attempt08_plan_sha256", "preflight_plan_sha256", "source_merged_sha256",
            "model_sha256", "ai_profiles_sha256", "attempt07_closeout_sha256",
            "attempt07_selection_sha256", "runtime_semantic_anchor_sha256",
            "runtime_source_closure_sha256", "runtime_requirements_sha256",
            "runtime_fingerprint_sha256", "source_model_manifest_sha256",
            "source_native_manifest_sha256", "gcp_image_name", "gcp_image_id",
            "gcp_image_self_link", "machine_type", "native_batch_threads", "jobs",
            "source_roots")},
        "local_evidence_sha256": sha256_file(local_evidence_path),
        "spot_authorized": True, "new_root_generation_allowed": False,
        "policy_science_allowed": False, "development200_allowed": False,
        "future_audit_allowed": False, "current_profile_resolved": False,
        "current_profile_mutated": False, "runtime_policy_activated": False,
    }
    _write_atomic_no_clobber(Path(output), canonical_json_bytes(payload))
    return payload


def validate_launch_authorization(
    path: str | Path, *, manifest: Mapping[str, Any], manifest_sha256: str,
    local_evidence_path: str | Path,
    revalidate_local_commands: bool = True,
) -> dict[str, Any]:
    value = _load_canonical(Path(path), "launch authorization")
    if revalidate_local_commands:
        _validate_local_evidence(
            Path(local_evidence_path), manifest_sha256,
            Path(path).resolve().parent / "manifest.json",
        )
    expected = create = {
        key: manifest[key] for key in (
            "schedule_sha256", "startup_sha256", "source_zip_sha256",
            "attempt08_plan_sha256", "preflight_plan_sha256", "source_merged_sha256",
            "model_sha256", "ai_profiles_sha256", "attempt07_closeout_sha256",
            "attempt07_selection_sha256", "runtime_semantic_anchor_sha256",
            "runtime_source_closure_sha256", "runtime_requirements_sha256",
            "runtime_fingerprint_sha256", "source_model_manifest_sha256",
            "source_native_manifest_sha256", "gcp_image_name", "gcp_image_id",
            "gcp_image_self_link", "machine_type", "native_batch_threads", "jobs",
            "source_roots")
    }
    if (
        set(value) != _AUTH_KEYS or value.get("schema") != LAUNCH_AUTHORIZATION_SCHEMA
        or value.get("status") != "authorized_for_five_spot_jobs_only"
        or value.get("run_name") != manifest["run_name"]
        or value.get("manifest_sha256") != manifest_sha256
        or any(value.get(key) != expected[key] for key in expected)
        or not _is_lower_sha256(value.get("local_evidence_sha256"))
        or value.get("local_evidence_sha256") != sha256_file(local_evidence_path)
        or value.get("spot_authorized") is not True
        or any(value.get(key) is not False for key in (
            "new_root_generation_allowed", "policy_science_allowed", "development200_allowed",
            "future_audit_allowed", "current_profile_resolved", "current_profile_mutated",
            "runtime_policy_activated"))
    ):
        raise ValueError("Attempt08 Spot launch authorization changed")
    return value


def validate_done_metadata(
    *, done_path: str | Path, run_dir: str | Path, authorization_path: str | Path,
    local_evidence_path: str | Path, job_index: int,
    revalidate_local_commands: bool = True,
) -> dict[str, Any]:
    if type(job_index) is not int or not 0 <= job_index < ATTEMPT08_PREFLIGHT_JOB_COUNT:
        raise ValueError("Attempt08 Spot DONE job_index changed")
    root = Path(run_dir)
    manifest = validate_package_artifacts(root)
    manifest_sha = sha256_file(root / "manifest.json")
    auth = validate_launch_authorization(
        authorization_path, manifest=manifest, manifest_sha256=manifest_sha,
        local_evidence_path=local_evidence_path,
        revalidate_local_commands=revalidate_local_commands,
    )
    spec = build_preflight_schedule()[job_index]
    value = _load_canonical(Path(done_path), "DONE")
    fixed = {
        "schema": DONE_SCHEMA, "status": "complete", "run_name": manifest["run_name"],
        "job_index": job_index, "job_id": spec["job_id"], "slot": spec["slot"],
        "source_root_index": spec["source_root_index"],
        "batch_child_selectors": spec["batch_child_selectors"],
        "output_prefix": spec["output_prefix"], "machine_type": ATTEMPT08_PREFLIGHT_MACHINE_TYPE,
        "native_batch_threads": 4, "manifest_sha256": manifest_sha,
        "authorization_sha256": sha256_file(authorization_path),
        **{key: auth[key] for key in (
            "schedule_sha256", "attempt08_plan_sha256", "preflight_plan_sha256",
            "source_merged_sha256", "model_sha256", "ai_profiles_sha256",
            "attempt07_closeout_sha256", "attempt07_selection_sha256",
            "runtime_semantic_anchor_sha256", "runtime_source_closure_sha256",
            "runtime_requirements_sha256", "runtime_fingerprint_sha256",
            "source_model_manifest_sha256", "source_native_manifest_sha256",
            "gcp_image_name", "gcp_image_id")},
        "teacher_action_or_value_details_exported": False, "new_root_generated": False,
        "policy_science_performed": False, "current_profile_resolved": False,
        "current_profile_mutated": False, "runtime_policy_activated": False,
    }
    if set(value) != _DONE_KEYS or any(value.get(k) != v for k, v in fixed.items()):
        raise ValueError("Attempt08 Spot DONE identity changed")
    if not _is_lower_sha256(value.get("proof_sha256")):
        raise ValueError("Attempt08 Spot DONE proof hash changed")
    for key in (
        "checkpoint_sha256", "heartbeat_sha256", "summary_sha256", "run_log_sha256",
        "boot_image_evidence_sha256",
    ):
        if not _is_lower_sha256(value.get(key)):
            raise ValueError(f"Attempt08 Spot DONE support hash changed: {key}")
    elapsed, rss = value.get("teacher_elapsed_seconds"), value.get("process_peak_rss_bytes")
    if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
            or not 0 < float(elapsed) <= 2400 or type(rss) is not int
            or not 0 < rss <= 30_064_771_072):
        raise ValueError("Attempt08 Spot DONE operational metrics changed")
    return value


def done_only_status(
    *, run_dir: str | Path, authorization_path: str | Path,
    local_evidence_path: str | Path, jobs_root: str | Path,
    revalidate_local_commands: bool = True,
) -> dict[str, Any]:
    done: list[int] = []
    missing: list[int] = []
    for spec in build_preflight_schedule():
        path = Path(jobs_root) / spec["output_prefix"] / "DONE.json"
        if not path.is_file():
            missing.append(spec["job_index"]); continue
        validate_done_metadata(done_path=path, run_dir=run_dir,
                               authorization_path=authorization_path,
                               local_evidence_path=local_evidence_path,
                               job_index=spec["job_index"],
                               revalidate_local_commands=revalidate_local_commands)
        done.append(spec["job_index"])
    return {"schema": "hu_m43_attempt08_preflight_spot_done_status_v1",
            "status": "all_five_done" if len(done) == 5 else "incomplete",
            "done_job_indices": done, "missing_job_indices": missing,
            "all_five_done": len(done) == 5, "proof_payloads_opened": False}


def validate_received_job_artifacts(
    *, directory: str | Path, done: Mapping[str, Any]
) -> Path:
    """Validate one completed job's proof and operational support artifacts."""

    root = Path(directory).resolve()
    proof = root / "proof.json"
    if not proof.is_file() or sha256_file(proof) != done.get("proof_sha256"):
        raise ValueError("Attempt08 received proof hash changed")
    support = {
        "checkpoint.json": ("checkpoint_sha256", CHECKPOINT_SCHEMA),
        "heartbeat.json": ("heartbeat_sha256", HEARTBEAT_SCHEMA),
        "summary.json": ("summary_sha256", SUMMARY_SCHEMA),
        "run.log": ("run_log_sha256", None),
        "boot_image_evidence.json": ("boot_image_evidence_sha256", None),
    }
    for name, (hash_key, schema) in support.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != done.get(hash_key):
            raise ValueError(f"Attempt08 received support artifact changed: {name}")
        if schema is None:
            continue
        payload = _load_canonical(path, f"received {name}")
        if (
            payload.get("schema") != schema
            or payload.get("run_name") != done.get("run_name")
            or payload.get("job_index") != done.get("job_index")
            or payload.get("slot") != done.get("slot")
        ):
            raise ValueError(f"Attempt08 received support identity changed: {name}")
        if name == "checkpoint.json" and (
            set(payload)
            != {
                "schema", "run_name", "job_index", "slot", "status",
                "proof_sha256", "resume_mode", "new_root_generated",
            }
            or payload.get("status") != "complete"
            or payload.get("proof_sha256") != done.get("proof_sha256")
            or payload.get("resume_mode") != "complete_no_recompute_needed"
            or payload.get("new_root_generated") is not False
        ):
            raise ValueError("Attempt08 received checkpoint semantics changed")
        if name == "heartbeat.json" and (
            set(payload)
            != {
                "schema", "run_name", "job_index", "slot", "status",
                "updated_unix_seconds", "deterministic_recompute_on_preemption",
                "proof_sha256", "process_alive", "new_root_generated",
                "current_profile_resolved",
            }
            or payload.get("status") != "complete"
            or payload.get("proof_sha256") != done.get("proof_sha256")
            or payload.get("process_alive") is not False
            or payload.get("deterministic_recompute_on_preemption") is not True
            or payload.get("new_root_generated") is not False
            or payload.get("current_profile_resolved") is not False
            or isinstance(payload.get("updated_unix_seconds"), bool)
            or not isinstance(payload.get("updated_unix_seconds"), (int, float))
            or not math.isfinite(float(payload["updated_unix_seconds"]))
            or float(payload["updated_unix_seconds"]) < 0.0
        ):
            raise ValueError("Attempt08 received heartbeat semantics changed")
        if name == "summary.json" and (
            set(payload)
            != {
                "schema", "run_name", "job_index", "slot", "proof_sha256",
                "teacher_elapsed_seconds", "process_peak_rss_bytes",
                "teacher_action_or_value_details_exported",
            }
            or payload.get("proof_sha256") != done.get("proof_sha256")
            or payload.get("teacher_elapsed_seconds")
            != done.get("teacher_elapsed_seconds")
            or payload.get("process_peak_rss_bytes")
            != done.get("process_peak_rss_bytes")
            or payload.get("teacher_action_or_value_details_exported") is not False
        ):
            raise ValueError("Attempt08 received summary semantics changed")
    boot = _load_canonical(root / "boot_image_evidence.json", "boot image evidence")
    if (
        set(boot)
        != {
            "schema", "run_name", "job_index", "instance_name", "disk_name",
            "source_image", "source_image_id",
        }
        or boot.get("schema") != "hu_m43_attempt08_boot_image_evidence_v1"
        or boot.get("run_name") != done.get("run_name")
        or boot.get("job_index") != done.get("job_index")
        or not str(boot.get("source_image", "")).endswith(
            "/projects/debian-cloud/global/images/debian-12-bookworm-v20260609"
        )
        or str(boot.get("source_image_id")) != ATTEMPT08_GCP_IMAGE_ID
    ):
        raise ValueError("Attempt08 boot image evidence changed")
    return proof


def validate_received_execution_bundle(
    *,
    run_dir: str | Path,
    authorization_path: str | Path,
    local_evidence_path: str | Path,
    jobs_root: str | Path,
    revalidate_local_commands: bool = True,
) -> dict[str, Any]:
    """Re-open the complete Spot bundle and derive its execution evidence.

    This is deliberately path-based: a caller-supplied attestation JSON is not
    sufficient to authorize development.  Every DONE, proof, support artifact,
    package binding, launch authorization, and boot-image record is revalidated.
    """

    status = done_only_status(
        run_dir=run_dir,
        authorization_path=authorization_path,
        local_evidence_path=local_evidence_path,
        jobs_root=jobs_root,
        revalidate_local_commands=revalidate_local_commands,
    )
    if status["all_five_done"] is not True:
        raise ValueError("Attempt08 execution bundle requires all five valid DONE records")
    root = Path(jobs_root).resolve()
    expected_prefixes = {
        spec["output_prefix"] for spec in build_preflight_schedule()
    }
    if (
        not root.is_dir()
        or any(path.is_symlink() for path in root.iterdir())
        or {path.name for path in root.iterdir() if path.is_dir()}
        != expected_prefixes
        or any(not path.is_dir() for path in root.iterdir())
    ):
        raise ValueError("Attempt08 execution bundle job directory set changed")
    expected_artifacts = {
        "DONE.json", "proof.json", "checkpoint.json", "heartbeat.json",
        "summary.json", "run.log", "boot_image_evidence.json",
    }
    for prefix in expected_prefixes:
        directory = root / prefix
        if (
            any(path.is_symlink() for path in directory.iterdir())
            or {path.name for path in directory.iterdir()} != expected_artifacts
            or any(not path.is_file() for path in directory.iterdir())
        ):
            raise ValueError(
                f"Attempt08 execution bundle artifacts changed: {prefix}"
            )
    proofs: dict[str, Path] = {}
    done_rows: dict[str, dict[str, Any]] = {}
    for spec in build_preflight_schedule():
        directory = (root / spec["output_prefix"]).resolve()
        if directory.parent != root:
            raise ValueError("Attempt08 execution bundle output path escaped jobs root")
        done = validate_done_metadata(
            done_path=directory / "DONE.json",
            run_dir=run_dir,
            authorization_path=authorization_path,
            local_evidence_path=local_evidence_path,
            job_index=spec["job_index"],
            revalidate_local_commands=revalidate_local_commands,
        )
        proof = validate_received_job_artifacts(directory=directory, done=done)
        if not proof.is_file() or sha256_file(proof) != done["proof_sha256"]:
            raise ValueError("Attempt08 received proof hash changed")
        support = {
            "checkpoint.json": ("checkpoint_sha256", CHECKPOINT_SCHEMA),
            "heartbeat.json": ("heartbeat_sha256", HEARTBEAT_SCHEMA),
            "summary.json": ("summary_sha256", SUMMARY_SCHEMA),
            "run.log": ("run_log_sha256", None),
            "boot_image_evidence.json": ("boot_image_evidence_sha256", None),
        }
        for name, (hash_key, schema) in support.items():
            path = directory / name
            if not path.is_file() or sha256_file(path) != done[hash_key]:
                raise ValueError(f"Attempt08 received support artifact changed: {name}")
            if schema is None:
                continue
            payload = _load_canonical(path, f"received {name}")
            if (
                payload.get("schema") != schema
                or payload.get("run_name") != done["run_name"]
                or payload.get("job_index") != done["job_index"]
                or payload.get("slot") != done["slot"]
            ):
                raise ValueError(f"Attempt08 received support identity changed: {name}")
            if name == "checkpoint.json" and (
                set(payload)
                != {
                    "schema", "run_name", "job_index", "slot", "status",
                    "proof_sha256", "resume_mode", "new_root_generated",
                }
                or payload.get("status") != "complete"
                or payload.get("proof_sha256") != done["proof_sha256"]
                or payload.get("resume_mode") != "complete_no_recompute_needed"
                or payload.get("new_root_generated") is not False
            ):
                raise ValueError("Attempt08 received checkpoint semantics changed")
            if name == "heartbeat.json" and (
                set(payload)
                != {
                    "schema", "run_name", "job_index", "slot", "status",
                    "updated_unix_seconds", "deterministic_recompute_on_preemption",
                    "proof_sha256", "process_alive", "new_root_generated",
                    "current_profile_resolved",
                }
                or payload.get("status") != "complete"
                or payload.get("proof_sha256") != done["proof_sha256"]
                or payload.get("process_alive") is not False
                or payload.get("deterministic_recompute_on_preemption") is not True
                or payload.get("new_root_generated") is not False
                or payload.get("current_profile_resolved") is not False
                or isinstance(payload.get("updated_unix_seconds"), bool)
                or not isinstance(payload.get("updated_unix_seconds"), (int, float))
                or not math.isfinite(float(payload["updated_unix_seconds"]))
                or float(payload["updated_unix_seconds"]) < 0.0
            ):
                raise ValueError("Attempt08 received heartbeat semantics changed")
            if name == "summary.json" and (
                set(payload)
                != {
                    "schema", "run_name", "job_index", "slot", "proof_sha256",
                    "teacher_elapsed_seconds", "process_peak_rss_bytes",
                    "teacher_action_or_value_details_exported",
                }
                or payload.get("proof_sha256") != done["proof_sha256"]
                or payload.get("teacher_elapsed_seconds")
                != done["teacher_elapsed_seconds"]
                or payload.get("process_peak_rss_bytes")
                != done["process_peak_rss_bytes"]
                or payload.get("teacher_action_or_value_details_exported") is not False
            ):
                raise ValueError("Attempt08 received summary semantics changed")
        boot = _load_canonical(
            directory / "boot_image_evidence.json", "boot image evidence"
        )
        if (
            set(boot)
            != {
                "schema", "run_name", "job_index", "instance_name", "disk_name",
                "source_image", "source_image_id",
            }
            or boot.get("schema") != "hu_m43_attempt08_boot_image_evidence_v1"
            or boot.get("run_name") != done["run_name"]
            or boot.get("job_index") != done["job_index"]
            or not str(boot.get("source_image", "")).endswith(
                "/projects/debian-cloud/global/images/debian-12-bookworm-v20260609"
            )
            or str(boot.get("source_image_id")) != ATTEMPT08_GCP_IMAGE_ID
        ):
            raise ValueError("Attempt08 boot image evidence changed")
        proofs[spec["slot"]] = proof
        done_rows[spec["slot"]] = done

    manifest = validate_package_artifacts(run_dir)
    evidence = {
        "schema": ATTEMPT08_PREFLIGHT_EXECUTION_EVIDENCE_SCHEMA,
        "status": "all_five_spot_jobs_received_and_validated",
        "run_name": manifest["run_name"],
        "manifest_sha256": sha256_file(Path(run_dir) / "manifest.json"),
        "launch_authorization_sha256": sha256_file(authorization_path),
        "local_evidence_sha256": sha256_file(local_evidence_path),
        "done_sha256": {
            spec["slot"]: sha256_file(root / spec["output_prefix"] / "DONE.json")
            for spec in build_preflight_schedule()
        },
        "support_sha256": {
            spec["slot"]: {
                "proof": sha256_file(root / spec["output_prefix"] / "proof.json"),
                "checkpoint": sha256_file(root / spec["output_prefix"] / "checkpoint.json"),
                "heartbeat": sha256_file(root / spec["output_prefix"] / "heartbeat.json"),
                "summary": sha256_file(root / spec["output_prefix"] / "summary.json"),
                "run_log": sha256_file(root / spec["output_prefix"] / "run.log"),
                "boot_image": sha256_file(
                    root / spec["output_prefix"] / "boot_image_evidence.json"
                ),
            }
            for spec in build_preflight_schedule()
        },
        "runtime": {
            "runtime_semantic_anchor_sha256": ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,
            "runtime_source_closure_sha256": ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
            "runtime_fingerprint_sha256": ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
            "runtime_requirements_sha256": ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
            "gcp_image_name": ATTEMPT08_GCP_IMAGE_NAME,
            "gcp_image_id": ATTEMPT08_GCP_IMAGE_ID,
        },
        "all_five_done_before_payloads_opened": True,
        "proof_payloads_opened_after_all_done": True,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    validate_preflight_execution_evidence(evidence)
    return {"execution_evidence": evidence, "proof_paths": proofs, "done_rows": done_rows}


def validate_preflight_receive_bundle(
    *,
    run_dir: str | Path,
    authorization_path: str | Path,
    local_evidence_path: str | Path,
    jobs_root: str | Path,
    execution_evidence_path: str | Path,
    revalidate_local_commands: bool = True,
) -> dict[str, Any]:
    """Recompute evidence from received bytes and require canonical byte equality."""

    result = validate_received_execution_bundle(
        run_dir=run_dir,
        authorization_path=authorization_path,
        local_evidence_path=local_evidence_path,
        jobs_root=jobs_root,
        revalidate_local_commands=revalidate_local_commands,
    )
    supplied = _load_canonical(
        Path(execution_evidence_path), "preflight execution evidence"
    )
    validate_preflight_execution_evidence(supplied)
    if canonical_json_bytes(result["execution_evidence"]) != Path(
        execution_evidence_path
    ).read_bytes():
        raise ValueError(
            "Attempt08 preflight execution evidence does not match received bundle"
        )
    return result


def receive_and_finalize(
    *, run_dir: str | Path, authorization_path: str | Path,
    local_evidence_path: str | Path, jobs_root: str | Path,
    output_dir: str | Path, source: str | Path,
) -> dict[str, Any]:
    run_root = Path(run_dir).resolve()
    expected_source = (
        run_root / "package_src" / "preflight_source" / "teacher.jsonl"
    ).resolve()
    supplied_source = Path(source).resolve()
    if os.path.normcase(str(supplied_source)) != os.path.normcase(
        str(expected_source)
    ):
        raise ValueError(
            "Attempt08 receive source must be the exact frozen packaged "
            "preflight source"
        )
    manifest = validate_package_artifacts(run_root)
    if (
        not supplied_source.is_file()
        or supplied_source.is_symlink()
        or sha256_file(supplied_source) != ATTEMPT07_PREFLIGHT_SOURCE_SHA256
        or manifest.get("source_merged_sha256")
        != ATTEMPT07_PREFLIGHT_SOURCE_SHA256
    ):
        raise ValueError("Attempt08 frozen packaged preflight source changed")
    status = done_only_status(run_dir=run_dir, authorization_path=authorization_path,
                              local_evidence_path=local_evidence_path,
                              jobs_root=jobs_root)
    if status["all_five_done"] is not True:
        raise ValueError("Attempt08 receive requires all five valid DONE records")
    # Enforce the exact-five directory/artifact set even when the eventual
    # correctness decision is No-Go and therefore emits no authorization.
    validate_received_execution_bundle(
        run_dir=run_dir,
        authorization_path=authorization_path,
        local_evidence_path=local_evidence_path,
        jobs_root=jobs_root,
    )
    root = Path(jobs_root)
    proofs: dict[str, Path] = {}
    done_rows: dict[str, dict[str, Any]] = {}
    for spec in build_preflight_schedule():
        directory = root / spec["output_prefix"]
        done = validate_done_metadata(done_path=directory / "DONE.json", run_dir=run_dir,
                                      authorization_path=authorization_path,
                                      local_evidence_path=local_evidence_path,
                                      job_index=spec["job_index"])
        proof = directory / "proof.json"
        if not proof.is_file() or sha256_file(proof) != done["proof_sha256"]:
            raise ValueError("Attempt08 received proof hash changed")
        support = {
            "checkpoint.json": ("checkpoint_sha256", CHECKPOINT_SCHEMA),
            "heartbeat.json": ("heartbeat_sha256", HEARTBEAT_SCHEMA),
            "summary.json": ("summary_sha256", SUMMARY_SCHEMA),
            "run.log": ("run_log_sha256", None),
            "boot_image_evidence.json": ("boot_image_evidence_sha256", None),
        }
        for name, (hash_key, schema) in support.items():
            path = directory / name
            if not path.is_file() or sha256_file(path) != done[hash_key]:
                raise ValueError(f"Attempt08 received support artifact changed: {name}")
            if schema is not None:
                payload = _load_canonical(path, f"received {name}")
                if (
                    payload.get("schema") != schema
                    or payload.get("run_name") != done["run_name"]
                    or payload.get("job_index") != done["job_index"]
                    or payload.get("slot") != done["slot"]
                ):
                    raise ValueError(f"Attempt08 received support identity changed: {name}")
                if name == "checkpoint.json" and (
                    set(payload)
                    != {"schema", "run_name", "job_index", "slot", "status",
                        "proof_sha256", "resume_mode", "new_root_generated"}
                    or payload.get("status") != "complete"
                    or payload.get("proof_sha256") != done["proof_sha256"]
                    or payload.get("resume_mode") != "complete_no_recompute_needed"
                    or payload.get("new_root_generated") is not False
                ):
                    raise ValueError("Attempt08 received checkpoint semantics changed")
                if name == "heartbeat.json" and (
                    set(payload)
                    != {"schema", "run_name", "job_index", "slot", "status",
                        "updated_unix_seconds", "deterministic_recompute_on_preemption",
                        "proof_sha256", "process_alive", "new_root_generated",
                        "current_profile_resolved"}
                    or payload.get("status") != "complete"
                    or payload.get("proof_sha256") != done["proof_sha256"]
                    or payload.get("process_alive") is not False
                    or payload.get("deterministic_recompute_on_preemption") is not True
                    or payload.get("new_root_generated") is not False
                    or payload.get("current_profile_resolved") is not False
                    or isinstance(payload.get("updated_unix_seconds"), bool)
                    or not isinstance(payload.get("updated_unix_seconds"), (int, float))
                    or not math.isfinite(float(payload["updated_unix_seconds"]))
                    or float(payload["updated_unix_seconds"]) < 0.0
                ):
                    raise ValueError("Attempt08 received heartbeat semantics changed")
                if name == "summary.json" and (
                    set(payload)
                    != {"schema", "run_name", "job_index", "slot", "proof_sha256",
                        "teacher_elapsed_seconds", "process_peak_rss_bytes",
                        "teacher_action_or_value_details_exported"}
                    or payload.get("proof_sha256") != done["proof_sha256"]
                    or payload.get("teacher_elapsed_seconds")
                    != done["teacher_elapsed_seconds"]
                    or payload.get("process_peak_rss_bytes")
                    != done["process_peak_rss_bytes"]
                    or payload.get("teacher_action_or_value_details_exported") is not False
                ):
                    raise ValueError("Attempt08 received summary semantics changed")
        boot = _load_canonical(
            directory / "boot_image_evidence.json", "boot image evidence"
        )
        if (
            set(boot)
            != {"schema", "run_name", "job_index", "instance_name", "disk_name",
                "source_image", "source_image_id"}
            or boot.get("schema") != "hu_m43_attempt08_boot_image_evidence_v1"
            or boot.get("run_name") != done["run_name"]
            or boot.get("job_index") != done["job_index"]
            or not str(boot.get("source_image", "")).endswith(
                "/projects/debian-cloud/global/images/debian-12-bookworm-v20260609"
            )
            or str(boot.get("source_image_id")) != ATTEMPT08_GCP_IMAGE_ID
        ):
            raise ValueError("Attempt08 boot image evidence changed")
        proofs[spec["slot"]] = proof
        done_rows[spec["slot"]] = done
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Attempt08 receive output exists: {destination}")
    staging = destination.with_name(destination.name + ".staging-" + uuid.uuid4().hex)
    staging.mkdir(parents=True)
    try:
        execution_evidence_path = staging / "preflight_execution_evidence.json"
        execution_evidence = {
            "schema": ATTEMPT08_PREFLIGHT_EXECUTION_EVIDENCE_SCHEMA,
            "status": "all_five_spot_jobs_received_and_validated",
            "run_name": validate_package_artifacts(run_dir)["run_name"],
            "manifest_sha256": sha256_file(Path(run_dir) / "manifest.json"),
            "launch_authorization_sha256": sha256_file(authorization_path),
            "local_evidence_sha256": sha256_file(local_evidence_path),
            "done_sha256": {
                spec["slot"]: sha256_file(
                    root / spec["output_prefix"] / "DONE.json"
                )
                for spec in build_preflight_schedule()
            },
            "support_sha256": {
                spec["slot"]: {
                    "proof": sha256_file(root / spec["output_prefix"] / "proof.json"),
                    "checkpoint": sha256_file(root / spec["output_prefix"] / "checkpoint.json"),
                    "heartbeat": sha256_file(root / spec["output_prefix"] / "heartbeat.json"),
                    "summary": sha256_file(root / spec["output_prefix"] / "summary.json"),
                    "run_log": sha256_file(root / spec["output_prefix"] / "run.log"),
                    "boot_image": sha256_file(
                        root / spec["output_prefix"] / "boot_image_evidence.json"
                    ),
                }
                for spec in build_preflight_schedule()
            },
            "runtime": {
                "runtime_semantic_anchor_sha256": ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,
                "runtime_source_closure_sha256": ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
                "runtime_fingerprint_sha256": ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
                "runtime_requirements_sha256": ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
                "gcp_image_name": ATTEMPT08_GCP_IMAGE_NAME,
                "gcp_image_id": ATTEMPT08_GCP_IMAGE_ID,
            },
            "all_five_done_before_payloads_opened": True,
            "proof_payloads_opened_after_all_done": True,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        }
        validate_preflight_execution_evidence(execution_evidence)
        _write_atomic(
            execution_evidence_path, canonical_json_bytes(execution_evidence)
        )
        execution_evidence_sha = sha256_file(execution_evidence_path)
        aggregate_path = staging / "aggregate.json"
        aggregate = aggregate_preflight_proofs(
            **proofs,
            output=aggregate_path,
            source=source,
            spot_operational_evidence_sha256=execution_evidence_sha,
        )
        finalization_path = staging / "finalization.json"
        development_auth = staging / "development_open_authorization.json"
        finalization = finalize_preflight(
            aggregate=aggregate_path, output=finalization_path,
            authorization_output=development_auth if aggregate["decision"] == "go" else None,
            proof_paths=proofs, source=source,
            execution_evidence=execution_evidence_path,
            spot_run_dir=run_dir,
            spot_authorization=authorization_path,
            spot_local_evidence=local_evidence_path,
            spot_jobs_root=jobs_root,
        )
        receipt = {
            "schema": RECEIVE_RECEIPT_SCHEMA, "status": "received_all_five_and_finalized",
            "manifest_sha256": sha256_file(Path(run_dir) / "manifest.json"),
            "launch_authorization_sha256": sha256_file(authorization_path),
            "done_sha256": {spec["slot"]: sha256_file(root / spec["output_prefix"] / "DONE.json")
                            for spec in build_preflight_schedule()},
            "proof_sha256": {slot: sha256_file(path) for slot, path in proofs.items()},
            "aggregate_sha256": sha256_file(aggregate_path),
            "preflight_execution_evidence_sha256": execution_evidence_sha,
            "finalization_sha256": sha256_file(finalization_path),
            "development_authorization_sha256": (
                sha256_file(development_auth) if development_auth.is_file() else None),
            "decision": aggregate["decision"], "all_five_done_before_proofs_opened": True,
            "current_profile_mutated": False, "runtime_policy_activated": False,
        }
        _write_atomic(staging / "receive_receipt.json", canonical_json_bytes(receipt))
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True); raise
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    package = sub.add_parser("package")
    package.add_argument("--repo-root", default=str(_REPO_ROOT)); package.add_argument("--run-dir", required=True)
    package.add_argument("--run-name", required=True); package.add_argument("--resume-existing", action="store_true")
    authorize = sub.add_parser("authorize")
    authorize.add_argument("--manifest", required=True); authorize.add_argument("--evidence-output", required=True)
    authorize.add_argument("--output", required=True)
    gates = sub.add_parser("gates")
    gates.add_argument("--manifest", required=True); gates.add_argument("--output", required=True)
    validate_launch = sub.add_parser("validate-launch")
    validate_launch.add_argument("--manifest", required=True)
    validate_launch.add_argument("--authorization", required=True)
    validate_launch.add_argument("--local-evidence", required=True)
    status = sub.add_parser("status")
    status.add_argument("--run-dir", required=True); status.add_argument("--authorization", required=True)
    status.add_argument("--local-evidence", required=True)
    status.add_argument("--jobs-root", required=True)
    receive = sub.add_parser("receive")
    receive.add_argument("--run-dir", required=True); receive.add_argument("--authorization", required=True)
    receive.add_argument("--local-evidence", required=True)
    receive.add_argument("--jobs-root", required=True); receive.add_argument("--output-dir", required=True)
    receive.add_argument("--source", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "package":
        result = package_preflight_run(repo_root=args.repo_root, run_dir=args.run_dir,
                                       run_name=args.run_name, resume_existing=args.resume_existing)
    elif args.command == "gates":
        result = run_local_launch_gates(manifest_path=args.manifest, output=args.output)
    elif args.command == "validate-launch":
        manifest_path = Path(args.manifest)
        manifest = validate_package_artifacts(manifest_path.parent)
        result = validate_launch_authorization(
            args.authorization,
            manifest=manifest,
            manifest_sha256=sha256_file(manifest_path),
            local_evidence_path=args.local_evidence,
        )
    elif args.command == "authorize":
        result = create_launch_authorization(manifest_path=args.manifest,
                                             evidence_output=args.evidence_output,
                                             output=args.output)
    elif args.command == "status":
        result = done_only_status(run_dir=args.run_dir, authorization_path=args.authorization,
                                  local_evidence_path=args.local_evidence,
                                  jobs_root=args.jobs_root)
    else:
        result = receive_and_finalize(run_dir=args.run_dir, authorization_path=args.authorization,
                                      local_evidence_path=args.local_evidence,
                                      jobs_root=args.jobs_root, output_dir=args.output_dir,
                                      source=args.source)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTEMPT08_PREFLIGHT_MACHINE_TYPE", "DONE_SCHEMA", "LAUNCH_AUTHORIZATION_SCHEMA",
    "PACKAGE_MANIFEST_SCHEMA", "build_preflight_schedule", "create_launch_authorization",
    "create_local_evidence_receipt", "done_only_status", "package_preflight_run",
    "receive_and_finalize", "run_local_launch_gates", "validate_done_metadata", "validate_launch_authorization",
    "validate_package_artifacts", "validate_preflight_receive_bundle",
    "validate_preflight_schedule", "validate_received_execution_bundle",
    "validate_received_job_artifacts",
]
