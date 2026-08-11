"""Validate the complete M3.1 Step 6c 100-root quality pilot.

Every root and task is reconstructed from ActorObservation data.  Shard
summaries are not averaged or trusted as labels.  The output is a quality-gate
receipt, never realized match EV, training authorization, profile activation,
or production fanout authorization.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ai_profiles import ModelBundle, ModelPaths, load_model_bundle
from .hu_infoset import ActorObservation
from .hu_m31_t3_behavior_roots import (
    M31_T3_BEHAVIOR_PROFILES,
    generate_behavior_t3_roots,
)
from .hu_m31_t3_step6c_contract import (
    ACCEPTED_FEATURE_ENCODER_SHA256,
    ACCEPTED_NATIVE_LIBRARY_SHA256,
    CONFIRMATION_HAND_INDICES,
    CONFIRMATION_REGRET_MAX,
    CONFIRMATION_REGRET_MEAN_MAX,
    CONFIRMATION_REGRET_P95_MAX,
    CONFIRMATION_REGRET_P99_MAX,
    CONFIRMATION_ROOT_COUNT,
    CONFIRMATION_ROOT_INDICES,
    EXPECTED_STEP6C_CONTRACT_BYTE_SHA256,
    EXPECTED_STEP6C_CONTRACT_CANONICAL_SHA256,
    MAX_FIRST_P95_SECONDS,
    MAX_PEAK_RSS_BYTES,
    MAX_SECOND_P95_SECONDS,
    PERCENTILE_METHOD,
    PILOT_HAND_INDICES,
    PILOT_HANDS_PER_PROFILE,
    PILOT_HANDS_PER_SHARD,
    PILOT_ROOT_COUNT,
    PILOT_ROOT_INDICES,
    PILOT_ROOTS_PER_SHARD,
    PILOT_ROOTS_PER_PROFILE,
    PILOT_SHARD_COUNT,
    STEP6C_BEHAVIOR_SCHEDULE_SCHEMA,
    STEP6C_RUN_ID,
    STEP5_CONTRACT_BYTE_SHA256,
    STEP5_CONTRACT_CANONICAL_SHA256,
    STEP5_VALIDATION_SHA256,
    STEP6B_STATUS_SHA256,
    STEP6B_VALIDATION_SHA256,
    behavior_profile_for_train_index,
    canonical_sha256,
    nearest_rank_percentile,
    schedule_rows,
    train_seed_values,
)
from .hu_m31_t3_step6c_spot import (
    EXPECTED_IMAGE_ID,
    EXPECTED_IMAGE_NAME,
    EXPECTED_MACHINE_TYPE,
    EXPECTED_STEP6B_RECEIPT_SHA256,
    SCHEDULE_NAME,
    SOURCE_NAME,
    STARTUP_NAME,
    STEP6C_AUTHORIZATION_SCHEMA,
    STEP6C_PACKAGE_SCHEMA,
    _PACKAGE_MANIFEST_KEYS,
)
from .run_hu_m31_t3_step6b_shard import EXPECTED_MODELS
from .run_hu_m31_t3_step6c_shard import (
    EXPECTED_NATIVE_LIBRARY_SHA256,
    STEP6C_ROOT_TASK_SCHEMA,
    STEP6C_SEARCH_TASK_SCHEMA,
    STEP6C_SHARD_GATE_KEYS,
    STEP6C_SUMMARY_SCHEMA,
    RowEvidence,
    ShardSpec,
    _CONFIRMATION_BUDGET,
    _PRIMARY_BUDGET,
    _read_json,
    _root_contract_digest,
    _sha256,
    _validate_root_task,
    _validate_parity_report,
    _validate_search_task,
    _write_once,
)


STEP6C_RECEIVE_SCHEMA = "hu_m31_t3_step6c_receive_v1"
STEP6C_QUALITY_SCHEMA = "hu_m31_t3_step6c_quality_validation_v1"


@dataclass(frozen=True)
class ValidatedShard:
    shard: int
    status: str
    run_name: str
    source_sha256: str
    manifest_sha256: str
    hand_indices: tuple[int, ...]
    profiles: tuple[str, ...]
    fingerprints: tuple[str, ...]
    seats: tuple[str, ...]
    evidence: tuple[RowEvidence, ...]
    peak_rss_bytes: int
    resumed_task_count: int
    summary_sha256: str


_RECEIVE_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "source_sha256",
        "manifest_sha256",
        "schedule_sha256",
        "authorization_sha256",
        "step5_contract_canonical_sha256",
        "step6b_validation_sha256",
        "step6c_contract_canonical_sha256",
        "native_library_sha256",
        "feature_encoder_sha256",
        "per_shard",
        "all_shards_received",
        "quality_result_pending_validation",
        "training_eligible",
        "production_fanout_authorized",
        "current_profile_changed",
        "named_profile_added",
        "m31_complete",
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
_PER_SHARD_RECEIPT_KEYS = frozenset(
    {
        "done_sha256",
        "summary_sha256",
        "task_count",
        "root_task_count",
        "resumed_task_count",
        "shard_status",
    }
)
_NATIVE_LIBRARY_KEYS = frozenset({"path", "sha256", "bytes", "engine_version"})
_FEATURE_LIBRARY_KEYS = frozenset({"path", "sha256", "bytes"})
_SOURCE_ENTRY_KEYS = frozenset({"sha256", "bytes"})
_NATIVE_LIBRARY_PATH = "native/release/libofc_hu_m3_engine.so"
_FEATURE_LIBRARY_PATH = "target/release/libofc_stage3_feature_encoder.so"
_PARITY_GOLDEN_PATH = "artifacts/step6c/parity_golden.json"


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _read_canonical_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not canonical JSON") from error
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _is_nonnegative_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _validate_source_entries(
    entries: Any,
    *,
    native: Mapping[str, Any],
    feature: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    if not isinstance(entries, Mapping) or not entries:
        raise ValueError("Step 6c package source-entry manifest changed")
    for relative, record in entries.items():
        parts = str(relative).split("/") if isinstance(relative, str) else []
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or "\\" in relative
            or not parts
            or any(part in {"", ".", ".."} for part in parts)
            or not isinstance(record, Mapping)
            or set(record) != _SOURCE_ENTRY_KEYS
            or not _is_sha256(record.get("sha256"))
            or not _is_nonnegative_int(record.get("bytes"))
        ):
            raise ValueError("Step 6c package source-entry manifest changed")
    expected_hashes = {
        "configs/hu_joint_policy_m31_t3_step5_contract.json": (
            STEP5_CONTRACT_BYTE_SHA256
        ),
        "configs/hu_joint_policy_m31_t3_step6b_status.json": STEP6B_STATUS_SHA256,
        "configs/hu_joint_policy_m31_t3_step6c_contract.json": (
            EXPECTED_STEP6C_CONTRACT_BYTE_SHA256
        ),
        "artifacts/step5/contract_validation.json": STEP5_VALIDATION_SHA256,
        "artifacts/step6b/canary_validation.json": STEP6B_VALIDATION_SHA256,
        "artifacts/step6b/receive_receipt.json": EXPECTED_STEP6B_RECEIPT_SHA256,
        "artifacts/step6c/contract_validation.json": manifest[
            "step6c_validation_sha256"
        ],
        _PARITY_GOLDEN_PATH: manifest["parity_golden_sha256"],
        _NATIVE_LIBRARY_PATH: native["sha256"],
        _FEATURE_LIBRARY_PATH: feature["sha256"],
        **EXPECTED_MODELS,
    }
    for relative, expected_sha256 in expected_hashes.items():
        record = entries.get(relative)
        if not isinstance(record, Mapping) or record.get("sha256") != expected_sha256:
            raise ValueError("Step 6c package source-entry anchor changed")
    if (
        entries[_NATIVE_LIBRARY_PATH].get("bytes") != native["bytes"]
        or entries[_FEATURE_LIBRARY_PATH].get("bytes") != feature["bytes"]
    ):
        raise ValueError("Step 6c package native source-entry size changed")


def _validate_package_manifest(
    manifest: Mapping[str, Any],
    *,
    run_name: str,
    source_sha256: str,
    schedule_sha256: str,
) -> None:
    native = manifest.get("native_library")
    feature = manifest.get("feature_encoder_library")
    entries = manifest.get("source_entries")
    if (
        set(manifest) != _PACKAGE_MANIFEST_KEYS
        or manifest.get("schema") != STEP6C_PACKAGE_SCHEMA
        or manifest.get("status") != "packaged_local_no_gcloud"
        or manifest.get("run_name") != run_name
        or manifest.get("source_name") != SOURCE_NAME
        or manifest.get("source_sha256") != source_sha256
        or not _is_nonnegative_int(manifest.get("source_bytes"))
        or manifest.get("source_bytes") == 0
        or manifest.get("schedule_name") != SCHEDULE_NAME
        or manifest.get("schedule_sha256") != schedule_sha256
        or manifest.get("startup_name") != STARTUP_NAME
        or not _is_sha256(manifest.get("startup_sha256"))
        or manifest.get("total_shards") != PILOT_SHARD_COUNT
        or manifest.get("authorized_shards") != list(range(PILOT_SHARD_COUNT))
        or manifest.get("paired_hands_per_shard") != PILOT_HANDS_PER_SHARD
        or manifest.get("roots_per_shard") != PILOT_ROOTS_PER_SHARD
        or manifest.get("pilot_hand_indices") != list(PILOT_HAND_INDICES)
        or manifest.get("confirmation_hand_indices") != list(CONFIRMATION_HAND_INDICES)
        or manifest.get("step6c_run_id") != STEP6C_RUN_ID
        or manifest.get("behavior_schedule_schema") != STEP6C_BEHAVIOR_SCHEDULE_SCHEMA
        or manifest.get("teacher_schedule_canonical_sha256")
        != canonical_sha256(schedule_rows(PILOT_HAND_INDICES))
        or manifest.get("step5_contract_byte_sha256") != STEP5_CONTRACT_BYTE_SHA256
        or manifest.get("step5_contract_canonical_sha256")
        != STEP5_CONTRACT_CANONICAL_SHA256
        or manifest.get("step5_validation_sha256") != STEP5_VALIDATION_SHA256
        or manifest.get("step6b_status_sha256") != STEP6B_STATUS_SHA256
        or manifest.get("step6b_validation_sha256") != STEP6B_VALIDATION_SHA256
        or manifest.get("step6b_receive_receipt_sha256")
        != EXPECTED_STEP6B_RECEIPT_SHA256
        or manifest.get("step6c_contract_byte_sha256")
        != EXPECTED_STEP6C_CONTRACT_BYTE_SHA256
        or manifest.get("step6c_contract_canonical_sha256")
        != EXPECTED_STEP6C_CONTRACT_CANONICAL_SHA256
        or not _is_sha256(manifest.get("step6c_validation_sha256"))
        or not isinstance(native, Mapping)
        or set(native) != _NATIVE_LIBRARY_KEYS
        or native.get("path") != _NATIVE_LIBRARY_PATH
        or native.get("sha256") != ACCEPTED_NATIVE_LIBRARY_SHA256
        or not _is_nonnegative_int(native.get("bytes"))
        or native.get("bytes") == 0
        or native.get("engine_version") != "ofc_hu_m3_engine/0.1.0"
        or not isinstance(feature, Mapping)
        or set(feature) != _FEATURE_LIBRARY_KEYS
        or feature.get("path") != _FEATURE_LIBRARY_PATH
        or feature.get("sha256") != ACCEPTED_FEATURE_ENCODER_SHA256
        or not _is_nonnegative_int(feature.get("bytes"))
        or feature.get("bytes") == 0
        or manifest.get("models") != EXPECTED_MODELS
        or manifest.get("parity_golden_path") != _PARITY_GOLDEN_PATH
        or not _is_sha256(manifest.get("parity_golden_sha256"))
        or manifest.get("production_label_budget") != _PRIMARY_BUDGET
        or manifest.get("confirmation_budget") != _CONFIRMATION_BUDGET
        or not _is_nonnegative_int(manifest.get("source_entry_count"))
        or not isinstance(entries, Mapping)
        or manifest.get("source_entry_count") != len(entries)
        or manifest.get("machine_type") != EXPECTED_MACHINE_TYPE
        or manifest.get("image_name") != EXPECTED_IMAGE_NAME
        or manifest.get("image_id") != EXPECTED_IMAGE_ID
        or manifest.get("checkpoint_unit") != "completed_paired_hand"
        or manifest.get("heartbeat_interval_seconds") != 60
        or manifest.get("resume_drill_required_each_shard") is not True
        or manifest.get("quality_pilot_authorized") is not True
        or manifest.get("pilot_rows_training_eligible") is not False
        or manifest.get("production_fanout_authorized") is not False
        or manifest.get("gcloud_invoked") is not False
        or manifest.get("spot_vm_started") is not False
        or manifest.get("current_profile_changed") is not False
        or manifest.get("named_profile_added") is not False
        or manifest.get("m31_complete") is not False
    ):
        raise ValueError("Step 6c package manifest provenance changed")
    _validate_source_entries(
        entries,
        native=native,
        feature=feature,
        manifest=manifest,
    )


def _validate_received_provenance(
    *, received_dir: Path, receipt_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if received_dir.is_symlink() or not received_dir.is_dir():
        raise ValueError("Step 6c received directory is missing or unsafe")
    receipt = _read_canonical_json(receipt_path, "Step 6c receive receipt")
    manifest_path = received_dir / "package_manifest.json"
    authorization_path = received_dir / "launch_authorization.json"
    manifest = _read_canonical_json(manifest_path, "Step 6c package manifest")
    authorization = _read_canonical_json(
        authorization_path, "Step 6c launch authorization"
    )
    per_shard = receipt.get("per_shard")
    run_name = receipt.get("run_name")
    source_sha256 = receipt.get("source_sha256")
    manifest_sha256 = receipt.get("manifest_sha256")
    schedule_sha256 = receipt.get("schedule_sha256")
    authorization_sha256 = receipt.get("authorization_sha256")
    if (
        set(receipt) != _RECEIVE_RECEIPT_KEYS
        or receipt.get("schema") != STEP6C_RECEIVE_SCHEMA
        or receipt.get("status") != "pass"
        or not isinstance(run_name, str)
        or not run_name
        or not _is_sha256(source_sha256)
        or not _is_sha256(manifest_sha256)
        or manifest_sha256 != _sha256(manifest_path)
        or not _is_sha256(schedule_sha256)
        or not _is_sha256(authorization_sha256)
        or authorization_sha256 != _sha256(authorization_path)
        or receipt.get("step5_contract_canonical_sha256")
        != STEP5_CONTRACT_CANONICAL_SHA256
        or receipt.get("step6b_validation_sha256") != STEP6B_VALIDATION_SHA256
        or receipt.get("step6c_contract_canonical_sha256")
        != EXPECTED_STEP6C_CONTRACT_CANONICAL_SHA256
        or receipt.get("native_library_sha256") != ACCEPTED_NATIVE_LIBRARY_SHA256
        or receipt.get("feature_encoder_sha256") != ACCEPTED_FEATURE_ENCODER_SHA256
        or receipt.get("all_shards_received") is not True
        or receipt.get("quality_result_pending_validation") is not True
        or receipt.get("training_eligible") is not False
        or receipt.get("production_fanout_authorized") is not False
        or receipt.get("current_profile_changed") is not False
        or receipt.get("named_profile_added") is not False
        or receipt.get("m31_complete") is not False
        or not isinstance(per_shard, Mapping)
        or set(per_shard) != {"000", "001"}
        or any(
            not isinstance(row, Mapping)
            or set(row) != _PER_SHARD_RECEIPT_KEYS
            or not _is_sha256(row.get("done_sha256"))
            or not _is_sha256(row.get("summary_sha256"))
            or row.get("task_count") != PILOT_HANDS_PER_SHARD
            or row.get("root_task_count") != PILOT_HANDS_PER_SHARD
            or not _is_nonnegative_int(row.get("resumed_task_count"))
            or row.get("shard_status") not in {"pass", "no_go"}
            for row in per_shard.values()
        )
    ):
        raise ValueError("Step 6c receive receipt boundary changed")

    _validate_package_manifest(
        manifest,
        run_name=run_name,
        source_sha256=source_sha256,
        schedule_sha256=schedule_sha256,
    )

    timestamp = authorization.get("authorized_unix_seconds")
    if (
        set(authorization) != _AUTHORIZATION_KEYS
        or authorization.get("schema") != STEP6C_AUTHORIZATION_SCHEMA
        or authorization.get("status") != "authorized"
        or authorization.get("run_name") != run_name
        or authorization.get("manifest_sha256") != manifest_sha256
        or authorization.get("source_sha256") != source_sha256
        or authorization.get("schedule_sha256") != schedule_sha256
        or authorization.get("startup_sha256") != manifest.get("startup_sha256")
        or authorization.get("step5_contract_canonical_sha256")
        != STEP5_CONTRACT_CANONICAL_SHA256
        or authorization.get("step6b_validation_sha256") != STEP6B_VALIDATION_SHA256
        or authorization.get("step6c_contract_canonical_sha256")
        != EXPECTED_STEP6C_CONTRACT_CANONICAL_SHA256
        or not _is_sha256(authorization.get("dry_run_receipt_sha256"))
        or authorization.get("authorized_shards") != list(range(PILOT_SHARD_COUNT))
        or authorization.get("spot_authorized") is not True
        or authorization.get("quality_pilot_only") is not True
        or authorization.get("production_fanout_authorized") is not False
        or authorization.get("training_eligible") is not False
        or authorization.get("root_execution_started") is not False
        or authorization.get("current_profile_changed") is not False
        or authorization.get("named_profile_added") is not False
        or isinstance(timestamp, bool)
        or not isinstance(timestamp, (int, float))
        or not math.isfinite(float(timestamp))
        or float(timestamp) < 0.0
    ):
        raise ValueError("Step 6c launch authorization provenance changed")
    return receipt, manifest, authorization


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{label} is outside the accepted range")
    return result


def _latency(values: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "mean_seconds": 0.0,
            "p50_seconds": 0.0,
            "p95_seconds": 0.0,
            "p99_seconds": 0.0,
            "max_seconds": 0.0,
        }
    return {
        "count": len(ordered),
        "mean_seconds": statistics.fmean(ordered),
        "p50_seconds": nearest_rank_percentile(ordered, 0.50),
        "p95_seconds": nearest_rank_percentile(ordered, 0.95),
        "p99_seconds": nearest_rank_percentile(ordered, 0.99),
        "max_seconds": max(ordered),
    }


def _regret_metrics(values: Sequence[float]) -> dict[str, float | int | str]:
    rows = [float(value) for value in values]
    if not rows:
        return {
            "count": 0,
            "mean": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "percentile_method": PERCENTILE_METHOD,
        }
    return {
        "count": len(rows),
        "mean": statistics.fmean(rows),
        "p95": nearest_rank_percentile(rows, 0.95),
        "p99": nearest_rank_percentile(rows, 0.99),
        "max": max(rows),
        "percentile_method": PERCENTILE_METHOD,
    }


def _validated_peak_rss(
    task_peak_rss_bytes: Sequence[int], summary: Mapping[str, Any]
) -> int:
    if not task_peak_rss_bytes:
        raise ValueError("Step 6c shard task RSS evidence is missing")
    recomputed = max(task_peak_rss_bytes)
    declared = summary.get("performance", {}).get("peak_process_rss_bytes")
    if (
        isinstance(declared, bool)
        or not isinstance(declared, int)
        or declared <= 0
        or declared != recomputed
    ):
        raise ValueError("Step 6c shard peak RSS summary changed")
    return recomputed


def _expected_indices(shard: int) -> tuple[int, ...]:
    start = shard * PILOT_HANDS_PER_SHARD
    return tuple(PILOT_HAND_INDICES[start : start + PILOT_HANDS_PER_SHARD])


def _validate_summary_envelope(
    summary: Mapping[str, Any],
    *,
    shard: int,
    hand_indices: tuple[int, ...],
) -> None:
    contract = summary.get("contract")
    integrity = summary.get("integrity")
    confirmation = summary.get("confirmation")
    performance = summary.get("performance")
    gates = summary.get("gates")
    gates_passed = (
        isinstance(gates, Mapping)
        and set(gates) == STEP6C_SHARD_GATE_KEYS
        and all(isinstance(value, bool) for value in gates.values())
        and all(value is True for value in gates.values())
    )
    if (
        summary.get("schema") != STEP6C_SUMMARY_SCHEMA
        or summary.get("status") not in {"pass", "no_go"}
        or summary.get("shard") != shard
        or summary.get("hand_indices") != list(hand_indices)
        or summary.get("native_library_sha256") != EXPECTED_NATIVE_LIBRARY_SHA256
        or summary.get("completed_tasks") != PILOT_HANDS_PER_SHARD
        or summary.get("pending_tasks") != 0
        or isinstance(summary.get("resumed_task_count"), bool)
        or not isinstance(summary.get("resumed_task_count"), int)
        or summary.get("resumed_task_count", 0) < 1
        or not isinstance(contract, Mapping)
        or contract.get("primary_budget") != _PRIMARY_BUDGET
        or contract.get("confirmation_budget") != _CONFIRMATION_BUDGET
        or contract.get("run_id") != STEP6C_RUN_ID
        or not isinstance(integrity, Mapping)
        or not isinstance(confirmation, Mapping)
        or confirmation.get("quality_thresholds_applied") is not False
        or not isinstance(performance, Mapping)
        or not isinstance(gates, Mapping)
        or set(gates) != STEP6C_SHARD_GATE_KEYS
        or any(not isinstance(value, bool) for value in gates.values())
        or summary.get("all_gates_passed") is not gates_passed
        or summary.get("status") != ("pass" if gates_passed else "no_go")
        or summary.get("teacher_value_status") != "diagnostic_not_match_EV"
        or summary.get("quality_pilot_authorized") is not True
        or summary.get("training_eligible") is not False
        or summary.get("named_profile_added") is not False
        or summary.get("current_profile_changed") is not False
        or summary.get("production_fanout_authorized") is not False
        or summary.get("m31_complete") is not False
    ):
        raise ValueError(f"Step 6c shard-{shard:03d} summary boundary changed")


def _validate_shard_parity(
    directory: Path,
    summary: Mapping[str, Any],
    *,
    source_package_sha256: str,
    manifest_sha256: str,
    parity_golden_sha256: str,
) -> None:
    parity_path = directory / "parity.json"
    parity = _read_canonical_json(
        parity_path, f"Step 6c shard-{int(summary['shard']):03d} parity"
    )
    _validate_parity_report(
        parity,
        source_package_sha256=source_package_sha256,
        manifest_sha256=manifest_sha256,
        parity_golden_sha256=parity_golden_sha256,
        native_library_sha256=EXPECTED_NATIVE_LIBRARY_SHA256,
    )
    if summary.get("parity_report_sha256") != _sha256(parity_path):
        raise ValueError("Step 6c shard summary parity binding changed")


def _validate_shard(
    directory: Path,
    shard: int,
    *,
    bundle: ModelBundle,
    expected_run_name: str,
    source_package_sha256: str,
    manifest_sha256: str,
    parity_golden_sha256: str,
) -> ValidatedShard:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"Step 6c shard directory is unsafe: {directory}")
    hand_indices = _expected_indices(shard)
    summary_path = directory / "summary.json"
    summary = _read_json(summary_path)
    _validate_summary_envelope(summary, shard=shard, hand_indices=hand_indices)
    manifest_sha = summary.get("manifest_sha256")
    source_sha = summary.get("source_package_sha256")
    run_name = summary.get("run_name")
    if (
        not _is_sha256(manifest_sha)
        or not _is_sha256(source_sha)
        or not isinstance(run_name, str)
        or not run_name
        or run_name != expected_run_name
        or source_sha != source_package_sha256
        or manifest_sha != manifest_sha256
    ):
        raise ValueError("Step 6c shard summary provenance changed")
    _validate_shard_parity(
        directory,
        summary,
        source_package_sha256=source_package_sha256,
        manifest_sha256=manifest_sha256,
        parity_golden_sha256=parity_golden_sha256,
    )
    spec = ShardSpec(run_name, shard, hand_indices)
    roots_dir = directory / "roots"
    tasks_dir = directory / "tasks"
    if roots_dir.is_symlink() or tasks_dir.is_symlink():
        raise ValueError("Step 6c received task tree contains a symlink")
    expected_names = {f"hand_{index:03d}.json" for index in hand_indices}
    if (
        not roots_dir.is_dir()
        or not tasks_dir.is_dir()
        or {path.name for path in roots_dir.iterdir()} != expected_names
        or {path.name for path in tasks_dir.iterdir()} != expected_names
        or any(path.is_symlink() or not path.is_file() for path in roots_dir.iterdir())
        or any(path.is_symlink() or not path.is_file() for path in tasks_dir.iterdir())
    ):
        raise ValueError("Step 6c received root/task file grid changed")

    profiles: list[str] = []
    fingerprints: list[str] = []
    seats: list[str] = []
    evidence: list[RowEvidence] = []
    task_hashes: list[dict[str, Any]] = []
    confirmation_regrets: list[float] = []
    task_peak_rss_bytes: list[int] = []
    for index in hand_indices:
        root_path = roots_dir / f"hand_{index:03d}.json"
        task_path = tasks_dir / f"hand_{index:03d}.json"
        root = _read_json(root_path)
        digest = _root_contract_digest(
            manifest_sha256=manifest_sha,
            spec=spec,
            global_hand_index=index,
        )
        seeds = train_seed_values(index)
        expected_observations = generate_behavior_t3_roots(
            hand_seed=seeds["hand"],
            behavior_seed=seeds["behavior"],
            profile=behavior_profile_for_train_index(index),
            bundle=bundle,
        )
        _validate_root_task(
            root,
            digest=digest,
            global_hand_index=index,
            expected_observations=expected_observations,
        )
        if root.get("schema") != STEP6C_ROOT_TASK_SCHEMA:
            raise ValueError("Step 6c root schema changed")
        task = _read_json(task_path)
        if task.get("schema") != STEP6C_SEARCH_TASK_SCHEMA:
            raise ValueError("Step 6c task schema changed")
        rows = _validate_search_task(
            task,
            root_task=root,
            library_sha256=EXPECTED_NATIVE_LIBRARY_SHA256,
        )
        task_peak_rss_bytes.append(int(task["memory"]["peak_rss_bytes"]))
        evidence.extend(rows)
        profiles.append(str(root["profile"]))
        for raw, row in zip(root["observations"], rows, strict=True):
            observation = ActorObservation.from_dict(raw["observation"])
            fingerprints.append(observation.fingerprint())
            seats.append(observation.seat)
            if row.confirmation_regret is not None:
                confirmation_regrets.append(row.confirmation_regret)
        task_hashes.append({"global_hand_index": index, "sha256": _sha256(task_path)})

    expected_confirmation_roots = 2 * sum(
        index in CONFIRMATION_HAND_INDICES for index in hand_indices
    )
    expected_candidate_count = len(evidence) * 8
    expected_evaluation_count = len(evidence) * 32
    expected_confirmation_count = expected_confirmation_roots * 128
    candidate = set().union(*(row.candidate_keys for row in evidence))
    evaluation = set().union(*(row.evaluation_keys for row in evidence))
    confirmation_keys = set().union(*(row.confirmation_keys for row in evidence))
    integrity = summary["integrity"]
    if (
        len(evidence) != 50
        or seats != [seat for _ in hand_indices for seat in ("first", "second")]
        or len(set(fingerprints)) != len(fingerprints)
        or len(candidate) != expected_candidate_count
        or len(evaluation) != expected_evaluation_count
        or len(confirmation_keys) != expected_confirmation_count
        or candidate & evaluation
        or candidate & confirmation_keys
        or evaluation & confirmation_keys
        or len(confirmation_regrets) != expected_confirmation_roots
        or integrity.get("candidate_rng_keys") != len(candidate)
        or integrity.get("evaluation_rng_keys") != len(evaluation)
        or integrity.get("confirmation_rng_keys") != len(confirmation_keys)
        or integrity.get("candidate_evaluation_overlap") != 0
        or integrity.get("candidate_confirmation_overlap") != 0
        or integrity.get("evaluation_confirmation_overlap") != 0
        or summary["confirmation"].get("root_count") != expected_confirmation_roots
        or summary["confirmation"].get("selected_regrets") != confirmation_regrets
        or summary.get("task_manifest") != task_hashes
    ):
        raise ValueError(f"Step 6c shard-{shard:03d} content summary changed")
    peak = _validated_peak_rss(task_peak_rss_bytes, summary)
    return ValidatedShard(
        shard=shard,
        status=str(summary["status"]),
        run_name=run_name,
        source_sha256=source_sha,
        manifest_sha256=manifest_sha,
        hand_indices=hand_indices,
        profiles=tuple(profiles),
        fingerprints=tuple(fingerprints),
        seats=tuple(seats),
        evidence=tuple(evidence),
        peak_rss_bytes=peak,
        resumed_task_count=int(summary["resumed_task_count"]),
        summary_sha256=_sha256(summary_path),
    )


def validate_quality(
    *, received_dir: Path, output_path: Path | None = None
) -> dict[str, Any]:
    received_dir = Path(received_dir)
    receipt_path = received_dir / "receive_receipt.json"
    receipt, manifest, _authorization = _validate_received_provenance(
        received_dir=received_dir,
        receipt_path=receipt_path,
    )
    per_shard = receipt.get("per_shard")
    assert isinstance(per_shard, Mapping)
    shards_root = received_dir / "shards"
    expected_shard_names = {f"shard-{shard:03d}" for shard in range(PILOT_SHARD_COUNT)}
    if (
        not shards_root.is_dir()
        or shards_root.is_symlink()
        or {path.name for path in shards_root.iterdir()} != expected_shard_names
    ):
        raise ValueError("Step 6c received shard grid changed")
    model_profiles = set(M31_T3_BEHAVIOR_PROFILES) - {"random_exact_final"}
    bundle = load_model_bundle(ModelPaths(), profiles=model_profiles)
    shards = tuple(
        _validate_shard(
            shards_root / f"shard-{shard:03d}",
            shard,
            bundle=bundle,
            expected_run_name=str(receipt["run_name"]),
            source_package_sha256=str(receipt["source_sha256"]),
            manifest_sha256=str(receipt["manifest_sha256"]),
            parity_golden_sha256=str(manifest["parity_golden_sha256"]),
        )
        for shard in range(PILOT_SHARD_COUNT)
    )
    for shard in shards:
        received = per_shard[f"{shard.shard:03d}"]
        if (
            not isinstance(received, Mapping)
            or shard.run_name != receipt["run_name"]
            or shard.source_sha256 != receipt["source_sha256"]
            or shard.manifest_sha256 != receipt["manifest_sha256"]
            or received.get("summary_sha256") != shard.summary_sha256
            or received.get("task_count") != PILOT_HANDS_PER_SHARD
            or received.get("root_task_count") != PILOT_HANDS_PER_SHARD
            or received.get("resumed_task_count") != shard.resumed_task_count
            or received.get("shard_status") != shard.status
        ):
            raise ValueError("Step 6c receive receipt per-shard binding changed")
    hands = [index for shard in shards for index in shard.hand_indices]
    profiles = [profile for shard in shards for profile in shard.profiles]
    fingerprints = [value for shard in shards for value in shard.fingerprints]
    seats = [seat for shard in shards for seat in shard.seats]
    evidence = [row for shard in shards for row in shard.evidence]
    candidate_sets = [row.candidate_keys for row in evidence]
    evaluation_sets = [row.evaluation_keys for row in evidence]
    confirmation_sets = [row.confirmation_keys for row in evidence]
    candidate = set().union(*candidate_sets)
    evaluation = set().union(*evaluation_sets)
    confirmation_keys = set().union(*confirmation_sets)
    regrets = [
        float(row.confirmation_regret)
        for row in evidence
        if row.confirmation_regret is not None
    ]
    confirmation_root_indices = [
        root_index
        for root_index, row in zip(PILOT_ROOT_INDICES, evidence, strict=True)
        if row.confirmation_regret is not None
    ]
    profile_hand_counts = {
        profile: profiles.count(profile) for profile in M31_T3_BEHAVIOR_PROFILES
    }
    profile_root_counts = {
        profile: count * 2 for profile, count in profile_hand_counts.items()
    }
    confirmation_profile_counts = {
        profile: sum(
            behavior_profile_for_train_index(index) == profile
            for index in CONFIRMATION_HAND_INDICES
        )
        for profile in M31_T3_BEHAVIOR_PROFILES
    }
    seat_counts = {seat: seats.count(seat) for seat in ("first", "second")}
    geometry_counts = {
        str(count): sum(row.legal_action_count == count for row in evidence)
        for count in sorted({row.legal_action_count for row in evidence})
    }
    primary_latency = {
        seat: _latency(
            [row.primary_wall_seconds for row in evidence if row.seat == seat]
        )
        for seat in ("first", "second")
    }
    confirmation_latency = {
        seat: _latency(
            [
                float(row.confirmation_wall_seconds)
                for row in evidence
                if row.seat == seat and row.confirmation_wall_seconds is not None
            ]
        )
        for seat in ("first", "second")
    }
    regret_metrics = _regret_metrics(regrets)
    regret_by_seat = {
        seat: _regret_metrics(
            [
                float(row.confirmation_regret)
                for row in evidence
                if row.seat == seat and row.confirmation_regret is not None
            ]
        )
        for seat in ("first", "second")
    }
    regret_by_profile = {
        profile: _regret_metrics(
            [
                float(row.confirmation_regret)
                for root_index, row in zip(PILOT_ROOT_INDICES, evidence, strict=True)
                if row.confirmation_regret is not None
                and behavior_profile_for_train_index(root_index // 2) == profile
            ]
        )
        for profile in M31_T3_BEHAVIOR_PROFILES
    }
    peak_rss = max(shard.peak_rss_bytes for shard in shards)
    gates = {
        "all_shard_summaries_pass": all(shard.status == "pass" for shard in shards),
        "exact_50_paired_hands": hands == list(PILOT_HAND_INDICES),
        "exact_100_root_grid": len(evidence) == PILOT_ROOT_COUNT,
        "exact_root_indices": list(PILOT_ROOT_INDICES) == list(range(100)),
        "unique_observation_fingerprints": len(set(fingerprints)) == PILOT_ROOT_COUNT,
        "exactly_50_each_seat": seat_counts == {"first": 50, "second": 50},
        "five_profiles_exactly_10_hands_each": all(
            count == PILOT_HANDS_PER_PROFILE for count in profile_hand_counts.values()
        ),
        "five_profiles_exactly_20_roots_each": all(
            count == PILOT_ROOTS_PER_PROFILE for count in profile_root_counts.values()
        ),
        "confirmation_one_paired_hand_each_profile": all(
            count == 1 for count in confirmation_profile_counts.values()
        ),
        "confirmation_exact_10_root_grid": confirmation_root_indices
        == list(CONFIRMATION_ROOT_INDICES),
        "all_800_candidate_rng_keys_unique": (
            sum(len(values) for values in candidate_sets) == len(candidate) == 800
        ),
        "all_3200_evaluation_rng_keys_unique": (
            sum(len(values) for values in evaluation_sets) == len(evaluation) == 3200
        ),
        "all_1280_confirmation_rng_keys_unique": (
            sum(len(values) for values in confirmation_sets)
            == len(confirmation_keys)
            == 1280
        ),
        "candidate_evaluation_confirmation_overlap_zero": not (
            candidate & evaluation
            or candidate & confirmation_keys
            or evaluation & confirmation_keys
        ),
        "every_shard_resumed_at_least_one_task": all(
            shard.resumed_task_count >= 1 for shard in shards
        ),
        "confirmation_regret_mean_at_most_0_75": regret_metrics["mean"]
        <= CONFIRMATION_REGRET_MEAN_MAX,
        "confirmation_regret_p95_at_most_3": regret_metrics["p95"]
        <= CONFIRMATION_REGRET_P95_MAX,
        "confirmation_regret_p99_at_most_6": regret_metrics["p99"]
        <= CONFIRMATION_REGRET_P99_MAX,
        "confirmation_regret_max_at_most_15": regret_metrics["max"]
        <= CONFIRMATION_REGRET_MAX,
        "first_primary_p95_within_180_seconds": primary_latency["first"]["p95_seconds"]
        <= MAX_FIRST_P95_SECONDS,
        "second_primary_p95_within_6_seconds": primary_latency["second"]["p95_seconds"]
        <= MAX_SECOND_P95_SECONDS,
        "peak_rss_within_1_gib": peak_rss <= MAX_PEAK_RSS_BYTES,
        "legal_action_geometry_diagnostic_only": True,
        "teacher_values_not_realized_match_ev": True,
        "no_training_profile_current_or_production_fanout": True,
    }
    passed = all(gates.values())
    result = {
        "schema": STEP6C_QUALITY_SCHEMA,
        "status": "pass" if passed else "no_go",
        "decision": (
            "production_label_quality_pilot_pass_separate_fanout_authorization_required"
            if passed
            else "production_label_quality_pilot_no_go_no_same_data_reselection"
        ),
        "received_dir": str(received_dir.resolve()),
        "receive_receipt_sha256": _sha256(receipt_path),
        "shards": [
            {
                "shard": shard.shard,
                "status": shard.status,
                "run_name": shard.run_name,
                "source_sha256": shard.source_sha256,
                "manifest_sha256": shard.manifest_sha256,
                "summary_sha256": shard.summary_sha256,
                "resumed_task_count": shard.resumed_task_count,
            }
            for shard in shards
        ],
        "integrity": {
            "paired_hands": len(hands),
            "roots": len(evidence),
            "seat_counts": seat_counts,
            "profile_hand_counts": profile_hand_counts,
            "profile_root_counts": profile_root_counts,
            "confirmation_profile_hand_counts": confirmation_profile_counts,
            "unique_fingerprints": len(set(fingerprints)),
            "candidate_rng_keys": len(candidate),
            "evaluation_rng_keys": len(evaluation),
            "confirmation_rng_keys": len(confirmation_keys),
            "candidate_evaluation_overlap": len(candidate & evaluation),
            "candidate_confirmation_overlap": len(candidate & confirmation_keys),
            "evaluation_confirmation_overlap": len(evaluation & confirmation_keys),
            "legal_action_geometry_counts": geometry_counts,
        },
        "confirmation_quality": {
            "root_indices": confirmation_root_indices,
            "selected_regrets": regrets,
            **regret_metrics,
            "by_seat_diagnostic": regret_by_seat,
            "by_profile_diagnostic": regret_by_profile,
            "top1_agreement_is_diagnostic_only": True,
        },
        "performance": {
            "primary_latency_by_seat": primary_latency,
            "confirmation_latency_by_seat": confirmation_latency,
            "peak_process_rss_bytes": peak_rss,
        },
        "gates": gates,
        "all_gates_passed": passed,
        "teacher_value_status": "diagnostic_not_match_EV",
        "teacher_values_are_realized_match_ev": False,
        "training_eligible": False,
        "named_profile_added": False,
        "current_profile_changed": False,
        "runtime_policy_activated": False,
        "production_fanout_authorized": False,
        "m31_complete": False,
    }
    if output_path is not None:
        _write_once(Path(output_path), result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--received-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = validate_quality(received_dir=args.received_dir, output_path=args.output)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "STEP6C_QUALITY_SCHEMA",
    "STEP6C_RECEIVE_SCHEMA",
    "ValidatedShard",
    "validate_quality",
]
