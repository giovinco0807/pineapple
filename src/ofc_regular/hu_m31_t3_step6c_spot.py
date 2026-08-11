"""Immutable bounded Spot lifecycle for the M3.1 Step 6c quality pilot.

This module can package and run only the frozen two-shard, 100-root pilot.  A
successful receive remains a diagnostic label-quality result: it does not
authorize production fanout, training, a named profile, or ``current``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hu_m31_t3_step6a_spot import (
    DEFAULT_BUCKET,
    DEFAULT_PROJECT,
    EXPECTED_IMAGE_ID,
    EXPECTED_IMAGE_NAME,
    EXPECTED_IMAGE_SELF_LINK,
    EXPECTED_MACHINE_TYPE,
    _MODEL_PATHS,
    _SAFE_RUN,
    _copy_file,
    _publish_once,
    _run,
    _subprocess_run,
    _zip_tree,
    canonical_bytes,
    sha256_file,
)
from .hu_infoset import ActorObservation
from .hu_m31_t3_runtime import HuM31T3RuntimeConfig, HuM31T3SearchSolver
from .hu_m31_t3_step6c_contract import (
    ACCEPTED_FEATURE_ENCODER_SHA256,
    ACCEPTED_NATIVE_LIBRARY_SHA256,
    CONFIRMATION_BUDGET,
    CONFIRMATION_HAND_INDICES,
    EXPECTED_STEP6C_CONTRACT_BYTE_SHA256,
    EXPECTED_STEP6C_CONTRACT_CANONICAL_SHA256,
    MAX_FIRST_P95_SECONDS,
    MAX_SECOND_P95_SECONDS,
    PILOT_HANDS_PER_SHARD,
    PILOT_HAND_INDICES,
    PILOT_ROOTS_PER_SHARD,
    PILOT_SHARD_COUNT,
    PRODUCTION_LABEL_BUDGET,
    STEP6C_BEHAVIOR_SCHEDULE_SCHEMA,
    STEP6C_RUN_ID,
    canonical_sha256,
    schedule_rows,
    train_seed_values,
)
from .run_hu_m31_t3_step6a_shard import portable_decision_sha256
from .run_hu_m31_t3_step6b_shard import EXPECTED_MODELS
from .run_hu_m31_t3_step6c_shard import (
    STEP6C_PARITY_SCHEMA,
    STEP6C_ROOT_TASK_SCHEMA,
    STEP6C_SEARCH_TASK_SCHEMA,
    STEP6C_SHARD_SCHEMA,
    STEP6C_SUMMARY_SCHEMA,
    _PARITY_GATE_KEYS,
    _validate_parity_report,
)
from .validate_hu_m31_t3_step6c_contract import validate_contract_file


STEP6C_PACKAGE_SCHEMA = "hu_m31_t3_step6c_spot_package_v1"
STEP6C_AUTHORIZATION_SCHEMA = "hu_m31_t3_step6c_launch_authorization_v1"
STEP6C_DRY_RUN_SCHEMA = "hu_m31_t3_step6c_local_dry_run_v1"
STEP6C_LAUNCH_SCHEMA = "hu_m31_t3_step6c_launch_v1"
STEP6C_STATUS_SCHEMA = "hu_m31_t3_step6c_cloud_status_v1"
STEP6C_DONE_SCHEMA = "hu_m31_t3_step6c_done_v1"
STEP6C_RECEIVE_SCHEMA = "hu_m31_t3_step6c_receive_v1"
STEP6C_PARITY_GOLDEN_SCHEMA = "hu_m31_t3_step6c_parity_golden_v1"

SOURCE_NAME = "ofc_regular_hu_m31_t3_step6c_source.zip"
SCHEDULE_NAME = "shards_manifest.jsonl"
STARTUP_NAME = "startup_hu_m31_t3_step6c.sh"
AUTHORIZED_SHARDS = tuple(range(PILOT_SHARD_COUNT))
MAX_LAUNCH_BATCH = 2
DEFAULT_ZONES = ("asia-northeast1-b", "asia-northeast1-c")
EXPECTED_WINDOWS_LIBRARY_SHA256 = (
    "d2d8f66ae56d978297a9228ca8b271622bb1b73cd2437237de8ef89915ea73e4"
)
EXPECTED_STEP5_CONTRACT_BYTE_SHA256 = (
    "5f9fab4d844f1a7411a99f9dada2fe289314d4dea578d0c0f6a6d14918263c93"
)
EXPECTED_STEP5_CONTRACT_CANONICAL_SHA256 = (
    "04c4298feaed78f327f7fb6601f70a6821c3a91dcd2996cc34becbc4bb38e0b4"
)
EXPECTED_STEP5_VALIDATION_SHA256 = (
    "055b900e06db174f69d3ecb25fd1466f9e812d272222529121a40b71dd003bcc"
)
EXPECTED_STEP6B_STATUS_SHA256 = (
    "76e327525bb9165732e36fe2f47dd70cf07752f93de7c68e2cdd9e59a99c16f8"
)
EXPECTED_STEP6B_VALIDATION_SHA256 = (
    "f95445cc71b402d97eb120cde2cfdbd1235ee57ca7c83b9227bcf5f9c3a238da"
)
EXPECTED_STEP6B_RECEIPT_SHA256 = (
    "63b289f3f436662ac56c74b1b16b58054a8e95e44362e9077179f6c2e5cf70cf"
)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ACCEPTED_STEP6B_ENGINE = Path(
    "outputs/gcp_runs/regular-hu-m31-step6b-canary-20260717-001/"
    "package_src/native/release/libofc_hu_m3_engine.so"
)
_ACCEPTED_STEP6B_FEATURE = Path(
    "outputs/gcp_runs/regular-hu-m31-step6b-canary-20260717-001/"
    "package_src/target/release/libofc_stage3_feature_encoder.so"
)
_CONFIG_PATHS = (
    "configs/hu_joint_policy_m31_t3_step5_contract.json",
    "configs/hu_joint_policy_m31_t3_step5_status.json",
    "configs/hu_joint_policy_m31_t3_step6a_status.json",
    "configs/hu_joint_policy_m31_t3_step6b_status.json",
    "configs/hu_joint_policy_m31_t3_step6c_contract.json",
    "configs/hu_m43_attempt08_runtime_requirements.txt",
)
_PACKAGE_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "total_shards",
        "authorized_shards",
        "paired_hands_per_shard",
        "roots_per_shard",
        "pilot_hand_indices",
        "confirmation_hand_indices",
        "step6c_run_id",
        "behavior_schedule_schema",
        "teacher_schedule_canonical_sha256",
        "source_name",
        "source_sha256",
        "source_bytes",
        "schedule_name",
        "schedule_sha256",
        "startup_name",
        "startup_sha256",
        "step5_contract_byte_sha256",
        "step5_contract_canonical_sha256",
        "step5_validation_sha256",
        "step6b_status_sha256",
        "step6b_validation_sha256",
        "step6b_receive_receipt_sha256",
        "step6c_contract_byte_sha256",
        "step6c_contract_canonical_sha256",
        "step6c_validation_sha256",
        "native_library",
        "feature_encoder_library",
        "models",
        "parity_golden_path",
        "parity_golden_sha256",
        "production_label_budget",
        "confirmation_budget",
        "source_entries",
        "source_entry_count",
        "machine_type",
        "image_name",
        "image_id",
        "checkpoint_unit",
        "heartbeat_interval_seconds",
        "resume_drill_required_each_shard",
        "quality_pilot_authorized",
        "pilot_rows_training_eligible",
        "production_fanout_authorized",
        "gcloud_invoked",
        "spot_vm_started",
        "current_profile_changed",
        "named_profile_added",
        "m31_complete",
    }
)
_PARITY_REPORT_KEYS = frozenset(
    {
        "schema",
        "status",
        "source_package_sha256",
        "manifest_sha256",
        "parity_golden_sha256",
        "native_library_sha256",
        "rows",
        "gates",
        "all_gates_passed",
        "current_profile_changed",
        "teacher_generation_started",
        "production_fanout_authorized",
    }
)
_PARITY_ROW_KEYS = frozenset(
    {
        "seat",
        "observation_fingerprint",
        "expected_portable_sha256",
        "observed_portable_sha256",
        "wall_seconds",
        "match",
    }
)
_RESUME_REPORT_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "shard",
        "hand_indices",
        "source_package_sha256",
        "manifest_sha256",
        "parity_report_sha256",
        "completed_tasks",
        "pending_tasks",
        "resumed_task_count",
        "quality_pilot_authorized",
        "training_eligible",
        "current_profile_changed",
        "production_fanout_authorized",
        "m31_complete",
    }
)
_DRY_RUN_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "manifest_sha256",
        "source_sha256",
        "schedule_sha256",
        "parity_report_sha256",
        "resume_report_sha256",
        "linux_production_budget_parity",
        "production_budget_speed_smoke",
        "speed_smoke_seconds_by_seat",
        "speed_smoke_limits_seconds",
        "resume_recovery",
        "gcloud_invoked",
        "training_eligible",
        "production_fanout_authorized",
        "current_profile_changed",
        "named_profile_added",
    }
)
_AUTHORIZATION_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "manifest_sha256",
        "source_sha256",
        "schedule_sha256",
        "startup_sha256",
        "step5_contract_canonical_sha256",
        "step6b_validation_sha256",
        "step6c_contract_canonical_sha256",
        "dry_run_receipt_sha256",
        "authorized_shards",
        "spot_authorized",
        "quality_pilot_only",
        "production_fanout_authorized",
        "training_eligible",
        "root_execution_started",
        "current_profile_changed",
        "named_profile_added",
        "authorized_unix_seconds",
    }
)
_DONE_KEYS = frozenset(
    {
        "schema",
        "status",
        "quality_status",
        "run_name",
        "shard",
        "source_sha256",
        "manifest_sha256",
        "schedule_sha256",
        "authorization_sha256",
        "native_library_sha256",
        "files",
        "resume_drill_passed",
        "training_eligible",
        "authorized_shards",
        "quality_pilot_only",
        "production_fanout_authorized",
        "current_profile_changed",
        "named_profile_added",
        "completed_unix_seconds",
    }
)
_SHARD_SUMMARY_GATE_KEYS = frozenset(
    {
        "exact_shard_hand_indices",
        "exactly_50_roots",
        "exactly_25_each_seat",
        "exact_confirmation_root_count",
        "unique_observation_fingerprints",
        "candidate_rng_unique",
        "evaluation_rng_unique",
        "confirmation_rng_unique",
        "candidate_evaluation_confirmation_rng_disjoint",
        "resume_drill_recovered_task",
        "first_p95_within_180_seconds",
        "second_p95_within_6_seconds",
        "peak_rss_within_1_gib",
        "geometry_diagnostic_only",
        "quality_thresholds_deferred_to_merged_validator",
        "no_training_current_profile_or_production_fanout",
    }
)


def _write_once(path: Path, value: Any, *, raw: bool = False) -> None:
    if path.exists():
        raise FileExistsError(f"immutable Step 6c artifact exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value if raw else canonical_bytes(value)
    if not isinstance(payload, bytes):
        raise TypeError("raw Step 6c artifact must be bytes")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_canonical(path: Path, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_bytes(dict(row)) for row in rows)


def _budget_dict(value: Any) -> dict[str, int]:
    if hasattr(value, "to_dict"):
        payload = dict(value.to_dict())
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        payload = {
            name: int(getattr(value, name))
            for name in (
                "candidate_samples",
                "evaluation_samples",
                "downstream_t3_samples",
                "downstream_t4_samples",
            )
        }
    payload.pop("label", None)
    return {name: int(payload[name]) for name in sorted(payload)}


def build_schedule(run_name: str) -> list[dict[str, Any]]:
    if not _SAFE_RUN.fullmatch(run_name):
        raise ValueError("Step 6c run name is not a safe bounded GCP identity")
    # The file is the frozen scientific 50-hand schedule.  The two execution
    # shards are represented by each row's contract-owned ``pilot_shard``.
    return list(schedule_rows(PILOT_HAND_INDICES))


def _canonical_file_sha(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"contract must be a JSON object: {path}")
    return canonical_sha256(value)


def _validate_prerequisites(
    *,
    root: Path,
    step5_validation: Path,
    step6b_validation: Path,
    step6b_received_dir: Path,
    step6c_contract: Path,
    step6c_validation: Path,
) -> dict[str, str]:
    paths = {
        "step5_contract": root / "configs/hu_joint_policy_m31_t3_step5_contract.json",
        "step5_validation": step5_validation,
        "step6b_status": root / "configs/hu_joint_policy_m31_t3_step6b_status.json",
        "step6b_validation": step6b_validation,
        "step6b_receive_receipt": step6b_received_dir / "receive_receipt.json",
        "step6c_contract": step6c_contract,
        "step6c_validation": step6c_validation,
    }
    if any(not path.is_file() for path in paths.values()):
        missing = [name for name, path in paths.items() if not path.is_file()]
        raise ValueError(f"Step 6c prerequisite is missing: {missing}")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if (
        hashes["step5_contract"] != EXPECTED_STEP5_CONTRACT_BYTE_SHA256
        or _canonical_file_sha(paths["step5_contract"])
        != EXPECTED_STEP5_CONTRACT_CANONICAL_SHA256
        or hashes["step5_validation"] != EXPECTED_STEP5_VALIDATION_SHA256
        or hashes["step6b_status"] != EXPECTED_STEP6B_STATUS_SHA256
        or hashes["step6b_validation"] != EXPECTED_STEP6B_VALIDATION_SHA256
        or hashes["step6b_receive_receipt"] != EXPECTED_STEP6B_RECEIPT_SHA256
    ):
        raise ValueError("Step 5/6b immutable prerequisite hash changed")
    step6b_status = json.loads(paths["step6b_status"].read_text(encoding="utf-8"))
    step6b = json.loads(paths["step6b_validation"].read_text(encoding="utf-8"))
    step6c = json.loads(paths["step6c_contract"].read_text(encoding="utf-8"))
    validation = json.loads(paths["step6c_validation"].read_text(encoding="utf-8"))
    expected_validation = validate_contract_file(
        paths["step6c_contract"], repo_root=root
    )
    contract_canonical = canonical_sha256(step6c)
    validation_contract_hash = validation.get(
        "contract_canonical_sha256", validation.get("contract_sha256")
    )
    if (
        step6b_status.get("status") != "step6b_infrastructure_canary_complete"
        or step6b_status.get("spot_production_fanout_authorized") is not False
        or step6b_status.get("current_profile_changed") is not False
        or step6b.get("status") != "pass"
        or step6b.get("all_gates_passed") is not True
        or step6b.get("production_fanout_authorized") is not False
        or hashes["step6c_contract"] != EXPECTED_STEP6C_CONTRACT_BYTE_SHA256
        or contract_canonical != EXPECTED_STEP6C_CONTRACT_CANONICAL_SHA256
        or validation.get("status") != "pass"
        or validation_contract_hash != contract_canonical
        or validation != expected_validation
    ):
        raise ValueError("Step 6c prerequisite semantic boundary changed")
    hashes["step5_contract_canonical"] = EXPECTED_STEP5_CONTRACT_CANONICAL_SHA256
    hashes["step6c_contract_canonical"] = contract_canonical
    return hashes


def _validate_parity_golden(path: Path, *, contract_sha256: str) -> dict[str, Any]:
    golden = _load_canonical(path, "Step 6c parity golden")
    rows = golden.get("rows")
    config = golden.get("runtime_config")
    expected_budget = _budget_dict(PRODUCTION_LABEL_BUDGET)
    if (
        golden.get("schema") != STEP6C_PARITY_GOLDEN_SCHEMA
        or golden.get("source_windows_library_sha256")
        != EXPECTED_WINDOWS_LIBRARY_SHA256
        or golden.get("step6c_contract_canonical_sha256") != contract_sha256
        or not isinstance(config, Mapping)
        or config.get("run_id") != STEP6C_RUN_ID
        or {name: config.get(name) for name in expected_budget} != expected_budget
        or not isinstance(rows, list)
        or len(rows) != 2
        or [row.get("seat") for row in rows if isinstance(row, Mapping)]
        != ["first", "second"]
        or any(
            not isinstance(row, Mapping)
            or not isinstance(row.get("observation"), Mapping)
            or not _is_sha256(row.get("observation_fingerprint"))
            or not _is_sha256(row.get("portable_decision_sha256"))
            for row in rows
        )
    ):
        raise ValueError("Step 6c production-budget parity golden changed")
    return golden


def build_parity_golden(
    *,
    output_path: str | Path,
    windows_library: str | Path,
    step6c_contract: str | Path,
    source_task: str | Path = (
        _REPO_ROOT
        / "outputs/hu_joint_policy/m31_t3_step4/local1000_v1/tasks/primary_hand_000.json"
    ),
) -> dict[str, Any]:
    """Create the explicit Windows 8/32/4 portable-parity input once."""

    library = Path(windows_library).resolve()
    contract = Path(step6c_contract).resolve()
    task_path = Path(source_task).resolve()
    if not library.is_file() or sha256_file(library) != EXPECTED_WINDOWS_LIBRARY_SHA256:
        raise ValueError("Step 6c parity requires the accepted Windows DLL")
    if (
        not contract.is_file()
        or sha256_file(contract) != EXPECTED_STEP6C_CONTRACT_BYTE_SHA256
        or _canonical_file_sha(contract) != EXPECTED_STEP6C_CONTRACT_CANONICAL_SHA256
    ):
        raise ValueError("Step 6c parity contract changed")
    task = json.loads(task_path.read_text(encoding="utf-8"))
    raw_rows = task.get("rows")
    if (
        task.get("all_gates_passed") is not True
        or not isinstance(raw_rows, list)
        or len(raw_rows) != 2
        or [row.get("seat") for row in raw_rows] != ["first", "second"]
    ):
        raise ValueError("Step 6c parity source task changed")
    seeds = train_seed_values(0)
    budget = _budget_dict(PRODUCTION_LABEL_BUDGET)
    solver = HuM31T3SearchSolver(
        HuM31T3RuntimeConfig(
            expected_library_sha256=EXPECTED_WINDOWS_LIBRARY_SHA256,
            library_path=library,
            run_id=STEP6C_RUN_ID,
            candidate_samples=budget["candidate_samples"],
            evaluation_samples=budget["evaluation_samples"],
            downstream_t3_samples=budget["downstream_t3_samples"],
            seed=seeds["child"],
            candidate_seed=seeds["candidate"],
            evaluation_seed=seeds["evaluation"],
        )
    )
    rows = []
    for raw in raw_rows:
        observation = ActorObservation.from_dict(raw["observation"])
        decision = solver.solve(observation).to_dict()
        rows.append(
            {
                "seat": observation.seat,
                "observation": observation.to_dict(),
                "observation_fingerprint": observation.fingerprint(),
                "portable_decision_sha256": portable_decision_sha256(decision),
            }
        )
    golden = {
        "schema": STEP6C_PARITY_GOLDEN_SCHEMA,
        "source_windows_library_sha256": EXPECTED_WINDOWS_LIBRARY_SHA256,
        "source_task_sha256": sha256_file(task_path),
        "step6c_contract_canonical_sha256": (EXPECTED_STEP6C_CONTRACT_CANONICAL_SHA256),
        "runtime_config": {
            "run_id": STEP6C_RUN_ID,
            **budget,
            "continuation_seed": seeds["child"],
            "candidate_seed": seeds["candidate"],
            "evaluation_seed": seeds["evaluation"],
        },
        "rows": rows,
        "current_profile_changed": False,
        "production_fanout_authorized": False,
        "training_eligible": False,
    }
    _write_once(Path(output_path).resolve(), golden)
    return golden


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_exact_parity_report(
    value: Mapping[str, Any],
    *,
    source_package_sha256: str,
    manifest_sha256: str,
    parity_golden_sha256: str,
    native_library_sha256: str,
) -> None:
    _validate_parity_report(
        value,
        source_package_sha256=source_package_sha256,
        manifest_sha256=manifest_sha256,
        parity_golden_sha256=parity_golden_sha256,
        native_library_sha256=native_library_sha256,
    )
    rows = value.get("rows")
    if (
        set(value) != _PARITY_REPORT_KEYS
        or not isinstance(rows, list)
        or any(
            not isinstance(row, Mapping) or set(row) != _PARITY_ROW_KEYS for row in rows
        )
    ):
        raise ValueError("Step 6c Linux parity exact schema changed")


def package_step6c(
    *,
    run_name: str,
    run_dir: str | Path,
    linux_library: str | Path,
    linux_feature_encoder: str | Path,
    step5_validation: str | Path,
    step6b_validation: str | Path,
    step6b_received_dir: str | Path,
    step6c_contract: str | Path,
    step6c_validation: str | Path,
    parity_golden: str | Path,
    repository_root: str | Path = _REPO_ROOT,
    startup: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    destination = Path(run_dir).resolve()
    if destination.exists():
        raise FileExistsError("Step 6c package run directory is immutable")
    if not _SAFE_RUN.fullmatch(run_name):
        raise ValueError("Step 6c run name is invalid")
    contract_path = Path(step6c_contract).resolve()
    validation_path = Path(step6c_validation).resolve()
    anchors = _validate_prerequisites(
        root=root,
        step5_validation=Path(step5_validation).resolve(),
        step6b_validation=Path(step6b_validation).resolve(),
        step6b_received_dir=Path(step6b_received_dir).resolve(),
        step6c_contract=contract_path,
        step6c_validation=validation_path,
    )
    golden_path = Path(parity_golden).resolve()
    _validate_parity_golden(
        golden_path, contract_sha256=anchors["step6c_contract_canonical"]
    )
    library = Path(linux_library).resolve()
    feature = Path(linux_feature_encoder).resolve()
    if (
        not library.is_file()
        or library != (root / _ACCEPTED_STEP6B_ENGINE).resolve()
        or library.name != "libofc_hu_m3_engine.so"
        or sha256_file(library) != ACCEPTED_NATIVE_LIBRARY_SHA256
        or not feature.is_file()
        or feature != (root / _ACCEPTED_STEP6B_FEATURE).resolve()
        or feature.name != "libofc_stage3_feature_encoder.so"
        or sha256_file(feature) != ACCEPTED_FEATURE_ENCODER_SHA256
    ):
        raise ValueError(
            "Step 6c requires the accepted Step 6b Linux binaries; current "
            "target/release binaries are not accepted"
        )
    startup_path = Path(startup or root / "scripts" / STARTUP_NAME).resolve()
    if not startup_path.is_file():
        raise ValueError("Step 6c startup script is missing")

    staging = destination.with_name(f".{destination.name}.{os.getpid()}.staging")
    if staging.exists():
        raise FileExistsError("Step 6c package staging already exists")
    package_root = staging / "package_src"
    package_root.mkdir(parents=True)
    try:
        entries: dict[str, dict[str, Any]] = {}
        for path in sorted((root / "src" / "ofc_regular").rglob("*.py")):
            if "__pycache__" in path.parts or path.is_symlink():
                continue
            relative = "src/" + path.relative_to(root / "src").as_posix()
            entries[relative] = _copy_file(path, package_root / relative)
        for relative in (*_CONFIG_PATHS, *_MODEL_PATHS):
            entries[relative] = _copy_file(root / relative, package_root / relative)
        if {
            relative: entries[relative]["sha256"] for relative in _MODEL_PATHS
        } != EXPECTED_MODELS:
            raise ValueError("Step 6c frozen model identity changed")
        native_relative = "native/release/libofc_hu_m3_engine.so"
        feature_relative = "target/release/libofc_stage3_feature_encoder.so"
        entries[native_relative] = _copy_file(library, package_root / native_relative)
        entries[feature_relative] = _copy_file(feature, package_root / feature_relative)
        evidence = {
            "artifacts/step5/contract_validation.json": Path(
                step5_validation
            ).resolve(),
            "artifacts/step6b/canary_validation.json": Path(
                step6b_validation
            ).resolve(),
            "artifacts/step6b/receive_receipt.json": Path(step6b_received_dir).resolve()
            / "receive_receipt.json",
            "artifacts/step6c/contract_validation.json": validation_path,
            "artifacts/step6c/parity_golden.json": golden_path,
        }
        for relative, source in evidence.items():
            entries[relative] = _copy_file(source, package_root / relative)
        source_path = staging / SOURCE_NAME
        _zip_tree(package_root, source_path)
        schedule_path = staging / SCHEDULE_NAME
        _write_once(schedule_path, _canonical_jsonl(build_schedule(run_name)), raw=True)
        shutil.copy2(startup_path, staging / STARTUP_NAME)
        manifest = {
            "schema": STEP6C_PACKAGE_SCHEMA,
            "status": "packaged_local_no_gcloud",
            "run_name": run_name,
            "total_shards": PILOT_SHARD_COUNT,
            "authorized_shards": list(AUTHORIZED_SHARDS),
            "paired_hands_per_shard": PILOT_HANDS_PER_SHARD,
            "roots_per_shard": PILOT_ROOTS_PER_SHARD,
            "pilot_hand_indices": list(PILOT_HAND_INDICES),
            "confirmation_hand_indices": list(CONFIRMATION_HAND_INDICES),
            "step6c_run_id": STEP6C_RUN_ID,
            "behavior_schedule_schema": STEP6C_BEHAVIOR_SCHEDULE_SCHEMA,
            "teacher_schedule_canonical_sha256": canonical_sha256(
                schedule_rows(PILOT_HAND_INDICES)
            ),
            "source_name": SOURCE_NAME,
            "source_sha256": sha256_file(source_path),
            "source_bytes": source_path.stat().st_size,
            "schedule_name": SCHEDULE_NAME,
            "schedule_sha256": sha256_file(schedule_path),
            "startup_name": STARTUP_NAME,
            "startup_sha256": sha256_file(staging / STARTUP_NAME),
            "step5_contract_byte_sha256": anchors["step5_contract"],
            "step5_contract_canonical_sha256": anchors["step5_contract_canonical"],
            "step5_validation_sha256": anchors["step5_validation"],
            "step6b_status_sha256": anchors["step6b_status"],
            "step6b_validation_sha256": anchors["step6b_validation"],
            "step6b_receive_receipt_sha256": anchors["step6b_receive_receipt"],
            "step6c_contract_byte_sha256": anchors["step6c_contract"],
            "step6c_contract_canonical_sha256": anchors["step6c_contract_canonical"],
            "step6c_validation_sha256": anchors["step6c_validation"],
            "native_library": {
                "path": native_relative,
                "sha256": entries[native_relative]["sha256"],
                "bytes": entries[native_relative]["bytes"],
                "engine_version": "ofc_hu_m3_engine/0.1.0",
            },
            "feature_encoder_library": {
                "path": feature_relative,
                "sha256": entries[feature_relative]["sha256"],
                "bytes": entries[feature_relative]["bytes"],
            },
            "models": EXPECTED_MODELS,
            "parity_golden_path": "artifacts/step6c/parity_golden.json",
            "parity_golden_sha256": entries["artifacts/step6c/parity_golden.json"][
                "sha256"
            ],
            "production_label_budget": _budget_dict(PRODUCTION_LABEL_BUDGET),
            "confirmation_budget": _budget_dict(CONFIRMATION_BUDGET),
            "source_entries": entries,
            "source_entry_count": len(entries),
            "machine_type": EXPECTED_MACHINE_TYPE,
            "image_name": EXPECTED_IMAGE_NAME,
            "image_id": EXPECTED_IMAGE_ID,
            "checkpoint_unit": "completed_paired_hand",
            "heartbeat_interval_seconds": 60,
            "resume_drill_required_each_shard": True,
            "quality_pilot_authorized": True,
            "pilot_rows_training_eligible": False,
            "production_fanout_authorized": False,
            "gcloud_invoked": False,
            "spot_vm_started": False,
            "current_profile_changed": False,
            "named_profile_added": False,
            "m31_complete": False,
        }
        _write_once(staging / "manifest.json", manifest)
        os.replace(staging, destination)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_package(run_dir: str | Path) -> dict[str, Any]:
    target = Path(run_dir).resolve()
    manifest = _load_canonical(target / "manifest.json", "Step 6c manifest")
    source = target / SOURCE_NAME
    schedule = target / SCHEDULE_NAME
    startup = target / STARTUP_NAME
    native = manifest.get("native_library")
    feature = manifest.get("feature_encoder_library")
    entries = manifest.get("source_entries")
    validation_entry = (
        entries.get("artifacts/step6c/contract_validation.json")
        if isinstance(entries, Mapping)
        else None
    )
    parity_entry = (
        entries.get("artifacts/step6c/parity_golden.json")
        if isinstance(entries, Mapping)
        else None
    )
    if (
        set(manifest) != _PACKAGE_MANIFEST_KEYS
        or manifest.get("schema") != STEP6C_PACKAGE_SCHEMA
        or manifest.get("status") != "packaged_local_no_gcloud"
        or manifest.get("source_name") != SOURCE_NAME
        or manifest.get("schedule_name") != SCHEDULE_NAME
        or manifest.get("startup_name") != STARTUP_NAME
        or manifest.get("total_shards") != PILOT_SHARD_COUNT
        or manifest.get("authorized_shards") != list(AUTHORIZED_SHARDS)
        or manifest.get("paired_hands_per_shard") != PILOT_HANDS_PER_SHARD
        or manifest.get("roots_per_shard") != PILOT_ROOTS_PER_SHARD
        or manifest.get("pilot_hand_indices") != list(PILOT_HAND_INDICES)
        or manifest.get("confirmation_hand_indices") != list(CONFIRMATION_HAND_INDICES)
        or manifest.get("step6c_run_id") != STEP6C_RUN_ID
        or manifest.get("behavior_schedule_schema") != STEP6C_BEHAVIOR_SCHEDULE_SCHEMA
        or manifest.get("teacher_schedule_canonical_sha256")
        != canonical_sha256(schedule_rows(PILOT_HAND_INDICES))
        or manifest.get("production_label_budget")
        != _budget_dict(PRODUCTION_LABEL_BUDGET)
        or manifest.get("confirmation_budget") != _budget_dict(CONFIRMATION_BUDGET)
        or manifest.get("source_sha256") != sha256_file(source)
        or manifest.get("source_bytes") != source.stat().st_size
        or manifest.get("schedule_sha256") != sha256_file(schedule)
        or manifest.get("startup_sha256") != sha256_file(startup)
        or manifest.get("step5_contract_byte_sha256")
        != EXPECTED_STEP5_CONTRACT_BYTE_SHA256
        or manifest.get("step5_contract_canonical_sha256")
        != EXPECTED_STEP5_CONTRACT_CANONICAL_SHA256
        or manifest.get("step5_validation_sha256") != EXPECTED_STEP5_VALIDATION_SHA256
        or manifest.get("step6b_status_sha256") != EXPECTED_STEP6B_STATUS_SHA256
        or manifest.get("step6b_validation_sha256") != EXPECTED_STEP6B_VALIDATION_SHA256
        or manifest.get("step6b_receive_receipt_sha256")
        != EXPECTED_STEP6B_RECEIPT_SHA256
        or manifest.get("step6c_contract_byte_sha256")
        != EXPECTED_STEP6C_CONTRACT_BYTE_SHA256
        or manifest.get("step6c_contract_canonical_sha256")
        != EXPECTED_STEP6C_CONTRACT_CANONICAL_SHA256
        or not isinstance(entries, Mapping)
        or not isinstance(validation_entry, Mapping)
        or manifest.get("step6c_validation_sha256") != validation_entry.get("sha256")
        or manifest.get("parity_golden_path") != "artifacts/step6c/parity_golden.json"
        or not isinstance(parity_entry, Mapping)
        or manifest.get("parity_golden_sha256") != parity_entry.get("sha256")
        or not isinstance(native, Mapping)
        or native.get("path") != "native/release/libofc_hu_m3_engine.so"
        or native.get("sha256") != ACCEPTED_NATIVE_LIBRARY_SHA256
        or native.get("engine_version") != "ofc_hu_m3_engine/0.1.0"
        or not isinstance(feature, Mapping)
        or feature.get("path") != "target/release/libofc_stage3_feature_encoder.so"
        or feature.get("sha256") != ACCEPTED_FEATURE_ENCODER_SHA256
        or manifest.get("models") != EXPECTED_MODELS
        or manifest.get("machine_type") != EXPECTED_MACHINE_TYPE
        or manifest.get("image_name") != EXPECTED_IMAGE_NAME
        or manifest.get("image_id") != EXPECTED_IMAGE_ID
        or manifest.get("checkpoint_unit") != "completed_paired_hand"
        or manifest.get("heartbeat_interval_seconds") != 60
        or manifest.get("resume_drill_required_each_shard") is not True
        or manifest.get("quality_pilot_authorized") is not True
        or manifest.get("pilot_rows_training_eligible") is not False
        or manifest.get("production_fanout_authorized") is not False
        or manifest.get("current_profile_changed") is not False
        or manifest.get("named_profile_added") is not False
        or manifest.get("gcloud_invoked") is not False
        or manifest.get("spot_vm_started") is not False
        or manifest.get("m31_complete") is not False
    ):
        raise ValueError("Step 6c package manifest changed")
    rows = [
        json.loads(line) for line in schedule.read_text(encoding="utf-8").splitlines()
    ]
    if rows != build_schedule(str(manifest["run_name"])):
        raise ValueError("Step 6c schedule changed")
    if not isinstance(entries, Mapping) or len(entries) != manifest.get(
        "source_entry_count"
    ):
        raise ValueError("Step 6c source entry manifest missing")
    with zipfile.ZipFile(source) as archive:
        names = tuple(sorted(archive.namelist()))
        if names != tuple(sorted(entries)) or any("\\" in name for name in names):
            raise ValueError("Step 6c zip entry set changed")
        for name, record in entries.items():
            data = archive.read(name)
            if (
                not isinstance(record, Mapping)
                or hashlib.sha256(data).hexdigest() != record.get("sha256")
                or len(data) != record.get("bytes")
            ):
                raise ValueError(f"Step 6c source content changed: {name}")
    return manifest


def record_local_dry_run(
    *, run_dir: str | Path, parity_report: str | Path, resume_report: str | Path
) -> dict[str, Any]:
    target = Path(run_dir).resolve()
    manifest = validate_package(target)
    parity_path = Path(parity_report).resolve()
    resume_path = Path(resume_report).resolve()
    parity = _load_canonical(parity_path, "Step 6c local parity")
    resume = _load_canonical(resume_path, "Step 6c local resume smoke")
    try:
        _validate_exact_parity_report(
            parity,
            source_package_sha256=str(manifest["source_sha256"]),
            manifest_sha256=sha256_file(target / "manifest.json"),
            parity_golden_sha256=str(manifest["parity_golden_sha256"]),
            native_library_sha256=ACCEPTED_NATIVE_LIBRARY_SHA256,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Step 6c local Linux dry-run failed") from error
    parity_rows = parity.get("rows")
    speed_by_seat = (
        {
            str(row.get("seat")): float(row.get("wall_seconds"))
            for row in parity_rows
            if isinstance(row, Mapping)
        }
        if isinstance(parity_rows, list)
        else {}
    )
    resume_completed = resume.get("completed_tasks")
    resume_pending = resume.get("pending_tasks")
    resume_shard = resume.get("shard")
    resume_resumed = resume.get("resumed_task_count")
    resume_shape_valid = (
        set(resume) == _RESUME_REPORT_KEYS
        and type(resume_shard) is int
        and resume_shard == 0
        and type(resume_completed) is int
        and type(resume_pending) is int
        and type(resume_resumed) is int
        and 2 <= resume_completed < PILOT_HANDS_PER_SHARD
        and resume_pending == PILOT_HANDS_PER_SHARD - resume_completed
        and 1 <= resume_resumed <= resume_completed
        and resume.get("status") == "interrupted_for_resume_drill"
    )
    if (
        parity.get("schema") != STEP6C_PARITY_SCHEMA
        or parity.get("all_gates_passed") is not True
        or parity.get("source_package_sha256") != manifest["source_sha256"]
        or parity.get("manifest_sha256") != sha256_file(target / "manifest.json")
        or parity.get("parity_golden_sha256") != manifest["parity_golden_sha256"]
        or parity.get("native_library_sha256") != ACCEPTED_NATIVE_LIBRARY_SHA256
        or set(speed_by_seat) != {"first", "second"}
        or not all(
            math.isfinite(value) and value >= 0.0 for value in speed_by_seat.values()
        )
        or speed_by_seat.get("first", math.inf) > MAX_FIRST_P95_SECONDS
        or speed_by_seat.get("second", math.inf) > MAX_SECOND_P95_SECONDS
        or resume.get("schema") != STEP6C_SUMMARY_SCHEMA
        or resume.get("run_name") != manifest["run_name"]
        or resume.get("hand_indices") != list(PILOT_HAND_INDICES[:25])
        or resume.get("source_package_sha256") != manifest["source_sha256"]
        or resume.get("manifest_sha256") != sha256_file(target / "manifest.json")
        or resume.get("parity_report_sha256") != sha256_file(parity_path)
        or not resume_shape_valid
        or resume.get("quality_pilot_authorized") is not True
        or resume.get("training_eligible") is not False
        or resume.get("current_profile_changed") is not False
        or resume.get("production_fanout_authorized") is not False
        or resume.get("m31_complete") is not False
        or parity.get("current_profile_changed") is not False
        or parity.get("production_fanout_authorized") is not False
    ):
        raise ValueError("Step 6c local Linux dry-run failed")
    receipt = {
        "schema": STEP6C_DRY_RUN_SCHEMA,
        "status": "pass",
        "run_name": manifest["run_name"],
        "manifest_sha256": sha256_file(target / "manifest.json"),
        "source_sha256": manifest["source_sha256"],
        "schedule_sha256": manifest["schedule_sha256"],
        "parity_report_sha256": sha256_file(parity_path),
        "resume_report_sha256": sha256_file(resume_path),
        "linux_production_budget_parity": True,
        "production_budget_speed_smoke": True,
        "speed_smoke_seconds_by_seat": speed_by_seat,
        "speed_smoke_limits_seconds": {
            "first": MAX_FIRST_P95_SECONDS,
            "second": MAX_SECOND_P95_SECONDS,
        },
        "resume_recovery": True,
        "gcloud_invoked": False,
        "training_eligible": False,
        "production_fanout_authorized": False,
        "current_profile_changed": False,
        "named_profile_added": False,
    }
    _write_once(target / "local_dry_run_receipt.json", receipt)
    return receipt


def _validated_dry_run_receipt(
    target: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    receipt_path = target / "local_dry_run_receipt.json"
    receipt = _load_canonical(receipt_path, "Step 6c dry-run receipt")
    speed = receipt.get("speed_smoke_seconds_by_seat")
    if (
        set(receipt) != _DRY_RUN_RECEIPT_KEYS
        or receipt.get("schema") != STEP6C_DRY_RUN_SCHEMA
        or receipt.get("status") != "pass"
        or receipt.get("run_name") != manifest["run_name"]
        or receipt.get("manifest_sha256") != sha256_file(target / "manifest.json")
        or receipt.get("source_sha256") != manifest["source_sha256"]
        or receipt.get("schedule_sha256") != manifest["schedule_sha256"]
        or not _is_sha256(receipt.get("parity_report_sha256"))
        or not _is_sha256(receipt.get("resume_report_sha256"))
        or receipt.get("linux_production_budget_parity") is not True
        or receipt.get("production_budget_speed_smoke") is not True
        or receipt.get("resume_recovery") is not True
        or receipt.get("gcloud_invoked") is not False
        or receipt.get("speed_smoke_limits_seconds")
        != {
            "first": MAX_FIRST_P95_SECONDS,
            "second": MAX_SECOND_P95_SECONDS,
        }
        or not isinstance(speed, Mapping)
        or set(speed) != {"first", "second"}
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            or float(value)
            > (MAX_FIRST_P95_SECONDS if seat == "first" else MAX_SECOND_P95_SECONDS)
            for seat, value in (speed.items() if isinstance(speed, Mapping) else ())
        )
        or receipt.get("training_eligible") is not False
        or receipt.get("production_fanout_authorized") is not False
        or receipt.get("current_profile_changed") is not False
        or receipt.get("named_profile_added") is not False
    ):
        raise ValueError("Step 6c dry-run receipt changed")
    return receipt


def authorize_launch(run_dir: str | Path) -> dict[str, Any]:
    target = Path(run_dir).resolve()
    manifest = validate_package(target)
    _validated_dry_run_receipt(target, manifest)
    authorization = {
        "schema": STEP6C_AUTHORIZATION_SCHEMA,
        "status": "authorized",
        "run_name": manifest["run_name"],
        "manifest_sha256": sha256_file(target / "manifest.json"),
        "source_sha256": manifest["source_sha256"],
        "schedule_sha256": manifest["schedule_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "step5_contract_canonical_sha256": manifest["step5_contract_canonical_sha256"],
        "step6b_validation_sha256": manifest["step6b_validation_sha256"],
        "step6c_contract_canonical_sha256": manifest[
            "step6c_contract_canonical_sha256"
        ],
        "dry_run_receipt_sha256": sha256_file(target / "local_dry_run_receipt.json"),
        "authorized_shards": list(AUTHORIZED_SHARDS),
        "spot_authorized": True,
        "quality_pilot_only": True,
        "production_fanout_authorized": False,
        "training_eligible": False,
        "root_execution_started": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "authorized_unix_seconds": time.time(),
    }
    _write_once(target / "launch_authorization.json", authorization)
    return authorization


def validate_launch(run_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    target = Path(run_dir).resolve()
    manifest = validate_package(target)
    _validated_dry_run_receipt(target, manifest)
    auth = _load_canonical(
        target / "launch_authorization.json", "Step 6c launch authorization"
    )
    timestamp = auth.get("authorized_unix_seconds")
    if (
        set(auth) != _AUTHORIZATION_KEYS
        or auth.get("schema") != STEP6C_AUTHORIZATION_SCHEMA
        or auth.get("status") != "authorized"
        or auth.get("run_name") != manifest["run_name"]
        or auth.get("manifest_sha256") != sha256_file(target / "manifest.json")
        or auth.get("source_sha256") != manifest["source_sha256"]
        or auth.get("schedule_sha256") != manifest["schedule_sha256"]
        or auth.get("startup_sha256") != manifest["startup_sha256"]
        or auth.get("step5_contract_canonical_sha256")
        != manifest["step5_contract_canonical_sha256"]
        or auth.get("step6b_validation_sha256") != manifest["step6b_validation_sha256"]
        or auth.get("step6c_contract_canonical_sha256")
        != manifest["step6c_contract_canonical_sha256"]
        or auth.get("dry_run_receipt_sha256")
        != sha256_file(target / "local_dry_run_receipt.json")
        or auth.get("authorized_shards") != list(AUTHORIZED_SHARDS)
        or auth.get("spot_authorized") is not True
        or auth.get("quality_pilot_only") is not True
        or auth.get("production_fanout_authorized") is not False
        or auth.get("training_eligible") is not False
        or auth.get("root_execution_started") is not False
        or auth.get("current_profile_changed") is not False
        or auth.get("named_profile_added") is not False
        or isinstance(timestamp, bool)
        or not isinstance(timestamp, (int, float))
        or not math.isfinite(float(timestamp))
    ):
        raise ValueError("Step 6c launch authorization changed")
    return manifest, auth


def _bounded_shards(shards: Sequence[int]) -> tuple[int, ...]:
    selected = tuple(shards)
    if (
        not selected
        or len(selected) > MAX_LAUNCH_BATCH
        or len(set(selected)) != len(selected)
        or any(
            isinstance(shard, bool) or shard not in AUTHORIZED_SHARDS
            for shard in selected
        )
    ):
        raise ValueError("Step 6c launch requires 1-2 unique shards from 0..1")
    return selected


def _parse_shards(value: str) -> tuple[int, ...]:
    try:
        return _bounded_shards(tuple(int(item) for item in value.split(",")))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _publish_package(
    *, target: Path, manifest: Mapping[str, Any], project: str, bucket: str
) -> str:
    prefix = f"gs://{bucket}/runs/{manifest['run_name']}"
    for source, uri in (
        (target / "manifest.json", f"{prefix}/manifest.json"),
        (target / SOURCE_NAME, f"{prefix}/source/{SOURCE_NAME}"),
        (target / SCHEDULE_NAME, f"{prefix}/source/{SCHEDULE_NAME}"),
        (
            target / "launch_authorization.json",
            f"{prefix}/source/launch_authorization.json",
        ),
    ):
        _publish_once(source, uri, project=project)
    return prefix


def _create_instance(
    *,
    target: Path,
    manifest: Mapping[str, Any],
    auth: Mapping[str, Any],
    prefix: str,
    shard: int,
    project: str,
    zone: str,
) -> dict[str, Any]:
    padded = f"{shard:03d}"
    done_uri = f"{prefix}/results/shard-{padded}/DONE.json"
    done = _subprocess_run(
        ["gcloud", "storage", "objects", "describe", done_uri, "--project", project],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    if done.returncode == 0:
        raise FileExistsError(f"Step 6c shard {shard} DONE already exists")
    instance = f"{manifest['run_name']}-s{padded}"
    existing = _subprocess_run(
        [
            "gcloud",
            "compute",
            "instances",
            "list",
            "--project",
            project,
            f"--filter=name={instance}",
            "--format=value(name)",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    if existing.returncode != 0:
        raise RuntimeError(f"failed to inspect Step 6c instance {instance}")
    if existing.stdout.strip():
        raise FileExistsError(f"Step 6c instance already exists: {instance}")
    metadata = ",".join(
        (
            f"PROJECT_ID={project}",
            f"BUCKET={prefix.split('/')[2]}",
            f"RUN_NAME={manifest['run_name']}",
            f"SHARD={shard}",
            f"SOURCE_URI={prefix}/source/{SOURCE_NAME}",
            f"SOURCE_SHA256={manifest['source_sha256']}",
            f"MANIFEST_SHA256={auth['manifest_sha256']}",
            f"SCHEDULE_SHA256={manifest['schedule_sha256']}",
            f"AUTHORIZATION_SHA256={sha256_file(target / 'launch_authorization.json')}",
            "SELF_DELETE=1",
        )
    )
    _run(
        [
            "gcloud",
            "compute",
            "instances",
            "create",
            instance,
            "--project",
            project,
            "--zone",
            zone,
            "--machine-type",
            EXPECTED_MACHINE_TYPE,
            "--provisioning-model=SPOT",
            "--instance-termination-action=DELETE",
            "--image-project=debian-cloud",
            f"--image={EXPECTED_IMAGE_NAME}",
            "--boot-disk-size=50GB",
            "--boot-disk-type=hyperdisk-balanced",
            "--scopes=https://www.googleapis.com/auth/cloud-platform",
            f"--metadata={metadata}",
            f"--metadata-from-file=startup-script={target / STARTUP_NAME}",
            "--quiet",
        ],
        timeout=600,
    )
    return {"shard": shard, "instance": instance, "zone": zone, "status": "created"}


def launch_shards(
    *,
    run_dir: str | Path,
    shards: Sequence[int],
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
    zones: Sequence[str] = DEFAULT_ZONES,
) -> dict[str, Any]:
    selected = _bounded_shards(shards)
    if not zones or any(zone not in DEFAULT_ZONES for zone in zones):
        raise ValueError("Step 6c launch zones must be asia-northeast1-b/c")
    target = Path(run_dir).resolve()
    manifest, auth = validate_launch(target)
    image = json.loads(
        _run(
            [
                "gcloud",
                "compute",
                "images",
                "describe",
                EXPECTED_IMAGE_NAME,
                "--project",
                "debian-cloud",
                "--format=json",
            ],
            timeout=120,
        ).stdout
    )
    if (
        str(image.get("id")) != EXPECTED_IMAGE_ID
        or image.get("selfLink") != EXPECTED_IMAGE_SELF_LINK
    ):
        raise ValueError("Step 6c immutable Debian image changed")
    prefix = _publish_package(
        target=target, manifest=manifest, project=project, bucket=bucket
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected)) as pool:
        futures = [
            pool.submit(
                _create_instance,
                target=target,
                manifest=manifest,
                auth=auth,
                prefix=prefix,
                shard=shard,
                project=project,
                zone=zones[shard % len(zones)],
            )
            for shard in selected
        ]
        created = [future.result() for future in futures]
    return {
        "schema": STEP6C_LAUNCH_SCHEMA,
        "status": "created",
        "run_name": manifest["run_name"],
        "created": sorted(created, key=lambda row: row["shard"]),
        "authorized_shards": list(AUTHORIZED_SHARDS),
        "quality_pilot_only": True,
        "production_fanout_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "named_profile_added": False,
    }


def cloud_status(
    *,
    run_dir: str | Path,
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    manifest, _auth = validate_launch(run_dir)
    prefix = f"gs://{bucket}/runs/{manifest['run_name']}"
    listing = _subprocess_run(
        ["gcloud", "storage", "ls", "--recursive", prefix, "--project", project],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    objects = listing.stdout.splitlines() if listing.returncode == 0 else []
    instances_result = _subprocess_run(
        [
            "gcloud",
            "compute",
            "instances",
            "list",
            "--project",
            project,
            f"--filter=name~'^{manifest['run_name']}-s00[01]$'",
            "--format=json",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    instances = (
        json.loads(instances_result.stdout)
        if instances_result.returncode == 0 and instances_result.stdout.strip()
        else []
    )
    by_name = {row.get("name"): row for row in instances}
    shards = {}
    for shard in AUTHORIZED_SHARDS:
        padded = f"{shard:03d}"
        result_prefix = f"/results/shard-{padded}/"
        progress_prefix = f"/progress/shard-{padded}/"
        vm = by_name.get(f"{manifest['run_name']}-s{padded}")
        shards[padded] = {
            "done": any(
                uri.endswith(f"/results/shard-{padded}/DONE.json") for uri in objects
            ),
            "heartbeat_present": any(
                uri.endswith(f"/progress/shard-{padded}/heartbeat.json")
                for uri in objects
            ),
            "progress_object_count": sum(progress_prefix in uri for uri in objects),
            "result_object_count": sum(result_prefix in uri for uri in objects),
            "instance": (
                None
                if vm is None
                else {
                    "name": vm.get("name"),
                    "status": vm.get("status"),
                    "zone": str(vm.get("zone", "")).rsplit("/", 1)[-1],
                }
            ),
        }
    return {
        "schema": STEP6C_STATUS_SCHEMA,
        "run_name": manifest["run_name"],
        "done_count": sum(row["done"] for row in shards.values()),
        "all_done": all(row["done"] for row in shards.values()),
        "shards": shards,
        "quality_pilot_only": True,
        "production_fanout_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
    }


def _safe_result_file(root: Path, relative: str) -> Path:
    pure = Path(relative)
    path = (root / pure).resolve()
    if pure.is_absolute() or ".." in pure.parts or not path.is_relative_to(root):
        raise ValueError("Step 6c DONE contains an unsafe file path")
    return path


def _scientific_result_relatives(shard: int) -> tuple[str, ...]:
    if shard not in AUTHORIZED_SHARDS:
        raise ValueError("Step 6c scientific result shard must be 0 or 1")
    first_hand = shard * PILOT_HANDS_PER_SHARD
    hand_indices = range(first_hand, first_hand + PILOT_HANDS_PER_SHARD)
    return (
        "parity.json",
        "summary.json",
        *(f"roots/hand_{index:03d}.json" for index in hand_indices),
        *(f"tasks/hand_{index:03d}.json" for index in hand_indices),
    )


def _validate_received_shard(
    *,
    shard_dir: Path,
    shard: int,
    manifest: Mapping[str, Any],
    auth_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    padded = f"{shard:03d}"
    done = _load_canonical(shard_dir / "DONE.json", f"Step 6c shard {shard} DONE")
    summary = _load_canonical(
        shard_dir / "summary.json", f"Step 6c shard {shard} summary"
    )
    parity_path = shard_dir / "parity.json"
    parity = _load_canonical(parity_path, f"Step 6c shard {shard} parity")
    _validate_exact_parity_report(
        parity,
        source_package_sha256=str(manifest["source_sha256"]),
        manifest_sha256=manifest_sha256,
        parity_golden_sha256=str(manifest["parity_golden_sha256"]),
        native_library_sha256=ACCEPTED_NATIVE_LIBRARY_SHA256,
    )
    gates = summary.get("gates")
    summary_gates_valid = (
        isinstance(gates, Mapping)
        and set(gates) == _SHARD_SUMMARY_GATE_KEYS
        and all(type(value) is bool for value in gates.values())
    )
    summary_gates_passed = summary_gates_valid and all(
        value is True for value in gates.values()
    )
    completed = done.get("completed_unix_seconds")
    if (
        set(done) != _DONE_KEYS
        or done.get("schema") != STEP6C_DONE_SCHEMA
        or done.get("status") != "complete"
        or done.get("quality_status") != summary.get("status")
        or done.get("run_name") != manifest["run_name"]
        or done.get("shard") != shard
        or done.get("source_sha256") != manifest["source_sha256"]
        or done.get("manifest_sha256") != manifest_sha256
        or done.get("schedule_sha256") != manifest["schedule_sha256"]
        or done.get("authorization_sha256") != auth_sha256
        or done.get("native_library_sha256") != ACCEPTED_NATIVE_LIBRARY_SHA256
        or done.get("resume_drill_passed") is not True
        or done.get("authorized_shards") != list(AUTHORIZED_SHARDS)
        or done.get("quality_pilot_only") is not True
        or done.get("training_eligible") is not False
        or done.get("production_fanout_authorized") is not False
        or done.get("current_profile_changed") is not False
        or done.get("named_profile_added") is not False
        or isinstance(completed, bool)
        or not isinstance(completed, (int, float))
        or not math.isfinite(float(completed))
        or float(completed) < 0.0
        or summary.get("schema") != STEP6C_SUMMARY_SCHEMA
        or not summary_gates_valid
        or summary.get("status") not in {"pass", "no_go"}
        or summary.get("status") != ("pass" if summary_gates_passed else "no_go")
        or summary.get("all_gates_passed") is not summary_gates_passed
        or summary.get("run_name") != manifest["run_name"]
        or summary.get("shard") != shard
        or summary.get("source_package_sha256") != manifest["source_sha256"]
        or summary.get("manifest_sha256") != manifest_sha256
        or summary.get("parity_report_sha256") != sha256_file(parity_path)
        or summary.get("resumed_task_count", 0) < 1
        or summary.get("teacher_value_status") != "diagnostic_not_match_EV"
        or summary.get("training_eligible") is not False
        or summary.get("production_fanout_authorized") is not False
        or summary.get("current_profile_changed") is not False
    ):
        raise ValueError(f"Step 6c received boundary failed: shard {padded}")
    files = done.get("files")
    if not isinstance(files, Mapping):
        raise ValueError(f"Step 6c DONE file manifest missing: shard {padded}")
    expected_files = set(_scientific_result_relatives(shard))
    if set(files) != expected_files:
        raise ValueError(f"Step 6c DONE scientific file set changed: shard {padded}")
    actual = {
        path.relative_to(shard_dir).as_posix()
        for path in shard_dir.rglob("*")
        if path.is_file()
    }
    if actual != expected_files | {"DONE.json"} or any(
        path.is_symlink() for path in shard_dir.rglob("*")
    ):
        raise ValueError(f"Step 6c received file set changed: shard {padded}")
    for relative, record in files.items():
        path = _safe_result_file(shard_dir, str(relative))
        if (
            not path.is_file()
            or not isinstance(record, Mapping)
            or sha256_file(path) != record.get("sha256")
            or path.stat().st_size != record.get("bytes")
        ):
            raise ValueError(f"Step 6c received file changed: {padded}/{relative}")
    root_files = list((shard_dir / "roots").glob("hand_*.json"))
    task_files = list((shard_dir / "tasks").glob("hand_*.json"))
    if (
        len(root_files) != PILOT_HANDS_PER_SHARD
        or len(task_files) != PILOT_HANDS_PER_SHARD
    ):
        raise ValueError(f"Step 6c received scientific task set failed: {padded}")
    return {
        "done_sha256": sha256_file(shard_dir / "DONE.json"),
        "summary_sha256": sha256_file(shard_dir / "summary.json"),
        "task_count": len(task_files),
        "root_task_count": len(root_files),
        "resumed_task_count": summary["resumed_task_count"],
        "shard_status": summary["status"],
    }


def receive_shards(
    *,
    run_dir: str | Path,
    output_dir: str | Path,
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    source = Path(run_dir).resolve()
    manifest, _auth = validate_launch(source)
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError("Step 6c receive destination is immutable")
    stage = destination.with_name(f".{destination.name}.{os.getpid()}.staging")
    shards_root = stage / "shards"
    shards_root.mkdir(parents=True)
    manifest_sha = sha256_file(source / "manifest.json")
    auth_sha = sha256_file(source / "launch_authorization.json")

    def download(shard: int) -> tuple[int, dict[str, Any]]:
        padded = f"{shard:03d}"
        shard_dir = shards_root / f"shard-{padded}"
        shard_dir.mkdir()
        prefix = f"gs://{bucket}/runs/{manifest['run_name']}/results/shard-{padded}"
        _run(
            [
                "gcloud",
                "storage",
                "rsync",
                "--recursive",
                prefix,
                str(shard_dir),
                "--project",
                project,
            ],
            timeout=7200,
        )
        return shard, _validate_received_shard(
            shard_dir=shard_dir,
            shard=shard,
            manifest=manifest,
            auth_sha256=auth_sha,
            manifest_sha256=manifest_sha,
        )

    try:
        _write_once(
            stage / "package_manifest.json",
            (source / "manifest.json").read_bytes(),
            raw=True,
        )
        _write_once(
            stage / "launch_authorization.json",
            (source / "launch_authorization.json").read_bytes(),
            raw=True,
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = dict(pool.map(download, AUTHORIZED_SHARDS))
        receipt = {
            "schema": STEP6C_RECEIVE_SCHEMA,
            "status": "pass",
            "run_name": manifest["run_name"],
            "source_sha256": manifest["source_sha256"],
            "manifest_sha256": manifest_sha,
            "schedule_sha256": manifest["schedule_sha256"],
            "authorization_sha256": auth_sha,
            "step5_contract_canonical_sha256": manifest[
                "step5_contract_canonical_sha256"
            ],
            "step6b_validation_sha256": manifest["step6b_validation_sha256"],
            "step6c_contract_canonical_sha256": manifest[
                "step6c_contract_canonical_sha256"
            ],
            "native_library_sha256": ACCEPTED_NATIVE_LIBRARY_SHA256,
            "feature_encoder_sha256": ACCEPTED_FEATURE_ENCODER_SHA256,
            "per_shard": {
                f"{shard:03d}": results[shard] for shard in AUTHORIZED_SHARDS
            },
            "all_shards_received": True,
            "quality_result_pending_validation": True,
            "training_eligible": False,
            "production_fanout_authorized": False,
            "current_profile_changed": False,
            "named_profile_added": False,
            "m31_complete": False,
        }
        _write_once(stage / "receive_receipt.json", receipt)
        os.replace(stage, destination)
        return receipt
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    parity = commands.add_parser("build-parity-golden")
    parity.add_argument("--output", type=Path, required=True)
    parity.add_argument("--windows-library", type=Path, required=True)
    parity.add_argument("--step6c-contract", type=Path, required=True)
    parity.add_argument("--source-task", type=Path)
    package = commands.add_parser("package")
    package.add_argument("--run-name", required=True)
    package.add_argument("--run-dir", type=Path, required=True)
    package.add_argument("--repository-root", type=Path, default=_REPO_ROOT)
    package.add_argument("--linux-library", type=Path, required=True)
    package.add_argument("--linux-feature-encoder", type=Path, required=True)
    package.add_argument("--step5-validation", type=Path, required=True)
    package.add_argument("--step6b-validation", type=Path, required=True)
    package.add_argument("--step6b-received-dir", type=Path, required=True)
    package.add_argument("--step6c-contract", type=Path, required=True)
    package.add_argument("--step6c-validation", type=Path, required=True)
    package.add_argument("--parity-golden", type=Path, required=True)
    package.add_argument("--startup", type=Path)
    validate = commands.add_parser("validate-package")
    validate.add_argument("--run-dir", type=Path, required=True)
    dry = commands.add_parser("record-dry-run")
    dry.add_argument("--run-dir", type=Path, required=True)
    dry.add_argument("--parity-report", type=Path, required=True)
    dry.add_argument("--resume-report", type=Path, required=True)
    authorize = commands.add_parser("authorize")
    authorize.add_argument("--run-dir", type=Path, required=True)
    launch = commands.add_parser("launch")
    launch.add_argument("--run-dir", type=Path, required=True)
    launch.add_argument("--shards", type=_parse_shards, required=True)
    launch.add_argument("--project", default=DEFAULT_PROJECT)
    launch.add_argument("--bucket", default=DEFAULT_BUCKET)
    launch.add_argument("--zones", default=",".join(DEFAULT_ZONES))
    status = commands.add_parser("status")
    status.add_argument("--run-dir", type=Path, required=True)
    status.add_argument("--project", default=DEFAULT_PROJECT)
    status.add_argument("--bucket", default=DEFAULT_BUCKET)
    receive = commands.add_parser("receive")
    receive.add_argument("--run-dir", type=Path, required=True)
    receive.add_argument("--output-dir", type=Path, required=True)
    receive.add_argument("--project", default=DEFAULT_PROJECT)
    receive.add_argument("--bucket", default=DEFAULT_BUCKET)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-parity-golden":
        kwargs = {
            "output_path": args.output,
            "windows_library": args.windows_library,
            "step6c_contract": args.step6c_contract,
        }
        if args.source_task is not None:
            kwargs["source_task"] = args.source_task
        result = build_parity_golden(**kwargs)
    elif args.command == "package":
        result = package_step6c(
            run_name=args.run_name,
            run_dir=args.run_dir,
            repository_root=args.repository_root,
            linux_library=args.linux_library,
            linux_feature_encoder=args.linux_feature_encoder,
            step5_validation=args.step5_validation,
            step6b_validation=args.step6b_validation,
            step6b_received_dir=args.step6b_received_dir,
            step6c_contract=args.step6c_contract,
            step6c_validation=args.step6c_validation,
            parity_golden=args.parity_golden,
            startup=args.startup,
        )
    elif args.command == "validate-package":
        result = validate_package(args.run_dir)
    elif args.command == "record-dry-run":
        result = record_local_dry_run(
            run_dir=args.run_dir,
            parity_report=args.parity_report,
            resume_report=args.resume_report,
        )
    elif args.command == "authorize":
        result = authorize_launch(args.run_dir)
    elif args.command == "launch":
        result = launch_shards(
            run_dir=args.run_dir,
            shards=args.shards,
            project=args.project,
            bucket=args.bucket,
            zones=tuple(item for item in args.zones.split(",") if item),
        )
    elif args.command == "status":
        result = cloud_status(
            run_dir=args.run_dir, project=args.project, bucket=args.bucket
        )
    else:
        result = receive_shards(
            run_dir=args.run_dir,
            output_dir=args.output_dir,
            project=args.project,
            bucket=args.bucket,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORIZED_SHARDS",
    "MAX_LAUNCH_BATCH",
    "STEP6C_AUTHORIZATION_SCHEMA",
    "STEP6C_DONE_SCHEMA",
    "STEP6C_PACKAGE_SCHEMA",
    "STEP6C_RECEIVE_SCHEMA",
    "build_schedule",
    "build_parity_golden",
    "cloud_status",
    "launch_shards",
    "package_step6c",
    "receive_shards",
    "validate_package",
]
