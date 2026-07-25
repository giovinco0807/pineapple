"""Source-isolated, parallel-safe Step 6d performance runner.

Version 1 intentionally executed the accepted reference and the candidate in
one process.  That was useful for a local paired comparison, but its run digest
also included the selected subset and therefore could not safely identify
independent Spot shards.  This v2 runner separates the immutable shared run
contract from each shard's role and work subset, and loads exactly one native
library in each worker process.

The module is performance-development infrastructure only.  It cannot train,
promote a profile, resolve ``current``, or consume opponent-private discards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import run_hu_m31_t3_step6d_performance as v1
from .hu_m31_t3_step6d_contract import (
    SEED_STRIDE,
    SEED_SCHEDULES,
    canonical_sha256 as contract_canonical_sha256,
    planned_seed_values,
    schedule_by_name,
)


RUN_CONTRACT_SCHEMA = "hu_m31_t3_step6d_performance_run_contract_v2"
SHARD_MANIFEST_SCHEMA = "hu_m31_t3_step6d_performance_shard_manifest_v2"
SOURCE_HAND_SCHEMA = "hu_m31_t3_step6d_performance_source_hand_v2"
DONE_SCHEMA = "hu_m31_t3_step6d_performance_source_shard_done_v2"

CANDIDATE01_VARIANT = "candidate01"
CANDIDATE02_VARIANT = "candidate02_compact_scorer"
CANDIDATE02_TAIL_V2_VARIANT = "candidate02_compact_scorer_tail_v2"
CANDIDATE02_PERFORMANCE_LOCK_VARIANT = "candidate02_compact_scorer_performance_lock"
CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT = (
    "candidate02_compact_scorer_performance_lock_recovery_v2"
)
CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_VARIANT = (
    "candidate02_compact_scorer_performance_lock_recovery_v3"
)
CANDIDATE02_PERFORMANCE_LOCK_V4_VARIANT = (
    "candidate02_compact_scorer_performance_lock_v4"
)
CANDIDATE02_RUN_CONTRACT_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_run_contract_v1"
)
CANDIDATE02_SOURCE_HAND_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_source_hand_v1"
)
CANDIDATE02_DONE_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_source_shard_done_v1"
)
CANDIDATE02_ROOT_SCHEMA = "hu_m31_t3_step6d_candidate02_performance_root_v1"
CANDIDATE02_SCHEDULE = "candidate02_performance_development"
CANDIDATE02_RUN_ID = "hu_m31_t3_step6d_candidate02_performance_v1"
CANDIDATE02_TAIL_V2_RUN_CONTRACT_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_run_contract_v2"
)
CANDIDATE02_TAIL_V2_SOURCE_HAND_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_source_hand_v2"
)
CANDIDATE02_TAIL_V2_DONE_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_source_shard_done_v2"
)
CANDIDATE02_TAIL_V2_RUN_ID = "hu_m31_t3_step6d_candidate02_performance_tail_v2"
CANDIDATE02_PERFORMANCE_LOCK_RUN_CONTRACT_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_lock_run_contract_v1"
)
CANDIDATE02_PERFORMANCE_LOCK_SOURCE_HAND_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_lock_source_hand_v1"
)
CANDIDATE02_PERFORMANCE_LOCK_DONE_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_lock_source_shard_done_v1"
)
CANDIDATE02_PERFORMANCE_LOCK_ROOT_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_lock_root_v1"
)
CANDIDATE02_PERFORMANCE_LOCK_SCHEDULE = "performance_lock"
CANDIDATE02_PERFORMANCE_LOCK_ROLE = "one_shot_performance_qualification_only"
CANDIDATE02_PERFORMANCE_LOCK_RUN_ID = "hu_m31_t3_step6d_candidate02_performance_lock_v1"
CANDIDATE02_PERFORMANCE_LOCK_SEED_SET_SHA256 = (
    "b5a37a8f96d2995b9020ef568a3737ba8f6203c55794e7a2a98055a92ef179ab"
)
CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_RUN_CONTRACT_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_lock_recovery_run_contract_v2"
)
CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SHARD_MANIFEST_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_lock_recovery_shard_manifest_v2"
)
CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SOURCE_HAND_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_lock_recovery_source_hand_v2"
)
CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_DONE_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_lock_recovery_source_shard_done_v2"
)
CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_ROOT_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_lock_recovery_root_v2"
)
CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SCHEDULE = "performance_lock_recovery_v2"
CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_ROLE = (
    "one_shot_infrastructure_recovery_performance_qualification_only"
)
CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_RUN_ID = (
    "hu_m31_t3_step6d_candidate02_performance_lock_recovery_v2"
)
CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SEED_SET_SHA256 = (
    "0df66657f263eb6de858706060620b4212e0e60173a0488bf0ddb3454c143aaa"
)
CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_NAMESPACE_BASES = {
    "hand": 700_108_071_901,
    "behavior": 701_108_071_901,
    "candidate": 702_108_071_901,
    "evaluation": 703_108_071_901,
    "child": 704_108_071_901,
    "confirmation": 705_108_071_901,
}
CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_RUN_CONTRACT_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_lock_recovery_run_contract_v3"
)
CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_SHARD_MANIFEST_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_lock_recovery_shard_manifest_v3"
)
CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_SOURCE_HAND_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_lock_recovery_source_hand_v3"
)
CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_DONE_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_lock_recovery_source_shard_done_v3"
)
CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_ROOT_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_lock_recovery_root_v3"
)
CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_SCHEDULE = "performance_lock_recovery_v3"
CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_ROLE = (
    "one_shot_second_infrastructure_recovery_performance_qualification_only"
)
CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_RUN_ID = (
    "hu_m31_t3_step6d_candidate02_performance_lock_recovery_v3"
)
CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_SEED_SET_SHA256 = (
    "e1e24394feb0de5ace140cdcd5f6dd87738059a88025789e69a1a991351927a8"
)
CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_NAMESPACE_BASES = {
    "hand": 710_108_071_901,
    "behavior": 711_108_071_901,
    "candidate": 712_108_071_901,
    "evaluation": 713_108_071_901,
    "child": 714_108_071_901,
    "confirmation": 715_108_071_901,
}
CANDIDATE02_PERFORMANCE_LOCK_V4_RUN_CONTRACT_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_lock_run_contract_v4"
)
CANDIDATE02_PERFORMANCE_LOCK_V4_SHARD_MANIFEST_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_lock_shard_manifest_v4"
)
CANDIDATE02_PERFORMANCE_LOCK_V4_SOURCE_HAND_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_lock_source_hand_v4"
)
CANDIDATE02_PERFORMANCE_LOCK_V4_DONE_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_lock_source_shard_done_v4"
)
CANDIDATE02_PERFORMANCE_LOCK_V4_ROOT_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_lock_root_v4"
)
CANDIDATE02_PERFORMANCE_LOCK_V4_SCHEDULE = "performance_lock_v4"
CANDIDATE02_PERFORMANCE_LOCK_V4_ROLE = (
    "run009_authorized_fresh_one_shot_performance_lock_only"
)
CANDIDATE02_PERFORMANCE_LOCK_V4_RUN_ID = (
    "hu_m31_t3_step6d_candidate02_performance_lock_v4"
)
CANDIDATE02_PERFORMANCE_LOCK_V4_SEED_SET_SHA256 = (
    "f63b2f0cb9212e9f16d9a05c79cdd6946fec54daccfd4a212f0db08c498a9847"
)
CANDIDATE02_PERFORMANCE_LOCK_V4_GATE_RECEIPT_FILE_SHA256 = (
    "664f86262436d41b62a8324cf1ba8df52e0f6af09f80f2071af825862b39e6b2"
)
CANDIDATE02_PERFORMANCE_LOCK_V4_GATE_RECEIPT_SHA256 = (
    "40bbad13cfc4a0680f7bf88027fbf173b848e86350cc5895c116708c6ad2c04d"
)
CANDIDATE02_PERFORMANCE_LOCK_V4_NAMESPACE_BASES = {
    "hand": 720_108_071_901,
    "behavior": 721_108_071_901,
    "candidate": 722_108_071_901,
    "evaluation": 723_108_071_901,
    "child": 724_108_071_901,
    "confirmation": 725_108_071_901,
}
CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_CANDIDATE_LIBRARY_SHA256 = (
    "4050e04b22d7943da9e1691402a2b34cf3d3781041a44557f35e2dfcf366d55d"
)
CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_REFERENCE_LIBRARY_SHA256 = (
    "3f831615d9e751b8e1f266f14dd2009903b3ac2a4094f7e47616e0aa372a01b0"
)
CANDIDATE02_TAIL_V2_SELECTION_MANIFEST_SHA256 = (
    "62cfbe2d95477ed7686a1b583997ee2e35ab23291fd338cfeac597e9a216fed1"
)
CANDIDATE02_PRIOR_PLANNED_SEED_MAX = 680_607_073_398
CANDIDATE02_SEED_STRIDE = 1_000_003
CANDIDATE02_NAMESPACE_BASES = {
    "hand": 690_108_071_901,
    "behavior": 691_108_071_901,
    "candidate": 692_108_071_901,
    "evaluation": 693_108_071_901,
    "child": 694_108_071_901,
    "confirmation": 695_108_071_901,
}

SOURCE_ROLES = ("candidate", "reference")
CONTRACT_HAND_INDICES = tuple(range(100))
TAIL_HAND_INDICES = (2, 6, 7, 9, 13, 20, 21, 29, 33, 50)
CANDIDATE02_TAIL_V2_TAIL_HAND_INDICES = (
    0,
    4,
    5,
    12,
    14,
    16,
    17,
    23,
    41,
    43,
)
CANDIDATE02_TAIL_V2_HEAVY_HAND_INDICES = (0, 5, 12, 16, 17, 23, 41, 43)
CANDIDATE02_TAIL_V2_RANDOM_HAND_INDICES = (4, 14)
CANDIDATE02_TAIL_V2_PRIOR_EXPOSED_HAND_INDICES = (
    2,
    6,
    7,
    9,
    13,
    20,
    21,
    29,
    33,
    50,
)
ALLOCATION = {"workers": 1, "rayon_threads_per_worker": 16}

_RUN_CONTRACT_KEYS = frozenset(
    {
        "schema",
        "step6d_run_id",
        "contract_canonical_sha256",
        "schedule",
        "contract_hand_indices",
        "tail_hand_indices",
        "budget",
        "allocation",
        "reference_library_sha256",
        "candidate_library_sha256",
        "execution_mode",
        "training_eligible",
        "quality_evidence",
        "promotion_evidence",
        "current_profile_changed",
    }
)
_CANDIDATE02_RUN_CONTRACT_KEYS = _RUN_CONTRACT_KEYS | frozenset(
    {"candidate_variant", "seed_contract"}
)
_CANDIDATE02_TAIL_V2_RUN_CONTRACT_KEYS = _CANDIDATE02_RUN_CONTRACT_KEYS | frozenset(
    {"selection_manifest_sha256"}
)
_CANDIDATE02_PERFORMANCE_LOCK_RUN_CONTRACT_KEYS = _CANDIDATE02_RUN_CONTRACT_KEYS
_CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_RUN_CONTRACT_KEYS = (
    _CANDIDATE02_RUN_CONTRACT_KEYS
)
_CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_RUN_CONTRACT_KEYS = (
    _CANDIDATE02_RUN_CONTRACT_KEYS
)
_CANDIDATE02_PERFORMANCE_LOCK_V4_RUN_CONTRACT_KEYS = (
    _CANDIDATE02_RUN_CONTRACT_KEYS
    | frozenset(
        {
            "authorizing_gate_receipt_file_sha256",
            "authorizing_gate_receipt_sha256",
        }
    )
)
_SHARD_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "run_contract",
        "run_contract_digest",
        "source_role",
        "work_hand_indices",
    }
)
_SOURCE_HAND_KEYS = frozenset(
    {
        "schema",
        "contract_canonical_sha256",
        "schedule",
        "run_contract_digest",
        "shard_manifest_sha256",
        "source_role",
        "hand_index",
        "root_indices",
        "schedule_row_sha256",
        "profile",
        "seeds",
        "budget",
        "root_artifact_sha256",
        "allocation",
        "reference_library_sha256",
        "candidate_library_sha256",
        "native_library_sha256",
        "engine_version",
        "queue_seconds",
        "worker_wall_seconds",
        "process_id",
        "memory",
        "rows",
        "teacher_value_status",
        "training_eligible",
        "quality_evidence",
        "promotion_evidence",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
        "cloud_started",
    }
)
_SOURCE_ROW_KEYS = frozenset(
    {
        "root_index",
        "seat",
        "observation_fingerprint",
        "observation_sha256",
        "geometry",
        "source_result",
        "portable_decision",
        "portable_decision_sha256",
        "wall_seconds",
    }
)
_DONE_KEYS = frozenset(
    {
        "schema",
        "status",
        "contract_canonical_sha256",
        "run_contract_digest",
        "shard_manifest_sha256",
        "source_role",
        "contract_hand_indices",
        "tail_hand_indices",
        "work_hand_indices",
        "completed_hand_indices",
        "budget",
        "allocation",
        "reference_library_sha256",
        "candidate_library_sha256",
        "native_library_sha256",
        "artifact_count",
        "artifact_manifest",
        "artifact_manifest_sha256",
        "training_eligible",
        "quality_evidence",
        "promotion_evidence",
        "current_profile_changed",
    }
)
_ARTIFACT_RECORD_KEYS = frozenset(
    {"source_role", "hand_index", "path", "sha256", "bytes"}
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


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.casefold()


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(f"{label} keys changed (missing={missing}, extra={extra})")


def _read_canonical(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"Step 6d v2 artifact is not canonical: {path}")
    return value


def _write_once(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Step 6d v2 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    try:
        # Publish with create-if-absent semantics.  ``os.replace`` would let a
        # racing writer replace an artifact after the initial existence check.
        os.link(temporary, path)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite Step 6d v2 artifact: {path}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _write_or_validate(path: Path, value: Mapping[str, Any], label: str) -> None:
    if path.exists():
        if _read_canonical(path) != dict(value):
            raise ValueError(f"existing {label} changed")
        return
    _write_once(path, value)


def _normalize_indices(values: Iterable[int], *, label: str) -> tuple[int, ...]:
    checked: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{label} must contain integers")
        if value not in CONTRACT_HAND_INDICES:
            raise ValueError(f"{label} must be within 0..99")
        checked.append(value)
    if not checked:
        raise ValueError(f"{label} must not be empty")
    if len(checked) != len(set(checked)):
        raise ValueError(f"{label} must be unique")
    if checked != sorted(checked):
        raise ValueError(f"{label} must be sorted")
    return tuple(checked)


def candidate02_seed_values(index: int) -> dict[str, int]:
    """Return the fresh candidate02 namespace seeds for one logical hand."""

    checked = _normalize_indices((index,), label="candidate02 hand index")[0]
    return {
        namespace: base + CANDIDATE02_SEED_STRIDE * checked
        for namespace, base in CANDIDATE02_NAMESPACE_BASES.items()
    }


def candidate02_seed_contract() -> dict[str, Any]:
    """Build and self-audit the globally fresh 100 hand x 6 namespace set."""

    rows = [
        seed
        for index in CONTRACT_HAND_INDICES
        for seed in candidate02_seed_values(index).values()
    ]
    if (
        len(rows) != 600
        or len(set(rows)) != 600
        or min(rows) <= CANDIDATE02_PRIOR_PLANNED_SEED_MAX
    ):
        raise RuntimeError("candidate02 fresh seed schedule is not globally disjoint")
    return {
        "schema": "hu_m31_t3_step6d_candidate02_disjoint_seed_schedule_v1",
        "prior_planned_seed_max": CANDIDATE02_PRIOR_PLANNED_SEED_MAX,
        "seed_stride": CANDIDATE02_SEED_STRIDE,
        "formula": "namespace_seed_base + seed_stride * hand_index",
        "namespace_bases": dict(CANDIDATE02_NAMESPACE_BASES),
        "hand_count": len(CONTRACT_HAND_INDICES),
        "namespace_count": len(CANDIDATE02_NAMESPACE_BASES),
        "seed_count": len(rows),
        "seed_min": min(rows),
        "seed_max": max(rows),
        "seed_set_sha256": canonical_sha256(sorted(rows)),
        "all_values_unique": True,
        "all_values_above_prior_planned_max": True,
        "training_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
    }


def _candidate02_performance_lock_schedule() -> Any:
    schedule = schedule_by_name(CANDIDATE02_PERFORMANCE_LOCK_SCHEDULE)
    expected_bases = {
        "hand": 490_108_071_901,
        "behavior": 491_108_071_901,
        "candidate": 492_108_071_901,
        "evaluation": 493_108_071_901,
        "child": 494_108_071_901,
        "confirmation": 495_108_071_901,
    }
    if (
        schedule.role != CANDIDATE02_PERFORMANCE_LOCK_ROLE
        or schedule.index_count != len(CONTRACT_HAND_INDICES)
        or schedule.unit != "paired_hand"
        or schedule.namespace_kind != "teacher"
        or dict(zip(schedule.namespace_keys, schedule.namespace_bases, strict=True))
        != expected_bases
        or schedule.training_eligible is not False
        or schedule.locked_before_content_read is not True
        or SEED_STRIDE != 1_000_003
    ):
        raise RuntimeError("candidate02 performance-lock schedule changed")
    return schedule


def candidate02_performance_lock_seed_values(index: int) -> dict[str, int]:
    """Return the preregistered one-shot performance-lock seeds for one hand."""

    checked = _normalize_indices(
        (index,), label="candidate02 performance-lock hand index"
    )[0]
    schedule = _candidate02_performance_lock_schedule()
    return {
        namespace: base + SEED_STRIDE * checked
        for namespace, base in zip(
            schedule.namespace_keys,
            schedule.namespace_bases,
            strict=True,
        )
    }


def candidate02_performance_lock_seed_contract() -> dict[str, Any]:
    """Self-audit the frozen lock schedule and its disjointness from development."""

    schedule = _candidate02_performance_lock_schedule()
    rows = [
        seed
        for index in CONTRACT_HAND_INDICES
        for seed in candidate02_performance_lock_seed_values(index).values()
    ]
    development = {
        seed
        for index in CONTRACT_HAND_INDICES
        for seed in candidate02_seed_values(index).values()
    }
    digest = contract_canonical_sha256(sorted(rows))
    if (
        len(rows) != 600
        or len(set(rows)) != 600
        or set(rows) & development
        or min(rows) != 490_108_071_901
        or max(rows) != 495_207_072_198
        or digest != CANDIDATE02_PERFORMANCE_LOCK_SEED_SET_SHA256
    ):
        raise RuntimeError("candidate02 performance-lock seed schedule changed")
    return {
        "schema": ("hu_m31_t3_step6d_candidate02_performance_lock_seed_schedule_v1"),
        "schedule": CANDIDATE02_PERFORMANCE_LOCK_SCHEDULE,
        "role": CANDIDATE02_PERFORMANCE_LOCK_ROLE,
        "seed_stride": SEED_STRIDE,
        "formula": "namespace_seed_base + seed_stride * hand_index",
        "namespace_bases": dict(
            zip(schedule.namespace_keys, schedule.namespace_bases, strict=True)
        ),
        "hand_count": len(CONTRACT_HAND_INDICES),
        "namespace_count": len(schedule.namespace_keys),
        "seed_count": len(rows),
        "seed_min": min(rows),
        "seed_max": max(rows),
        "seed_set_sha256": digest,
        "candidate02_development_overlap_count": 0,
        "all_values_unique": True,
        "locked_before_content_read": True,
        "training_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
    }


def candidate02_performance_lock_recovery_seed_values(
    index: int,
) -> dict[str, int]:
    """Return the fresh recovery-v2 namespace seeds for one logical hand."""

    checked = _normalize_indices(
        (index,), label="candidate02 performance-lock recovery hand index"
    )[0]
    return {
        namespace: base + SEED_STRIDE * checked
        for namespace, base in (
            CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_NAMESPACE_BASES.items()
        )
    }


def candidate02_performance_lock_recovery_seed_contract() -> dict[str, Any]:
    """Audit recovery-v2 against every frozen Step6d and Candidate02 seed."""

    rows = [
        seed
        for index in CONTRACT_HAND_INDICES
        for seed in candidate02_performance_lock_recovery_seed_values(index).values()
    ]
    recovery = set(rows)
    existing_step6d = set(planned_seed_values())
    candidate02_development = {
        seed
        for index in CONTRACT_HAND_INDICES
        for seed in candidate02_seed_values(index).values()
    }
    performance_lock_v1 = {
        seed
        for index in CONTRACT_HAND_INDICES
        for seed in candidate02_performance_lock_seed_values(index).values()
    }
    schedule_overlap_counts = {
        schedule.name: len(recovery & set(schedule.values()))
        for schedule in SEED_SCHEDULES
    }
    digest = contract_canonical_sha256(sorted(rows))
    if (
        len(rows) != 600
        or len(recovery) != 600
        or recovery & existing_step6d
        or recovery & candidate02_development
        or recovery & performance_lock_v1
        or any(schedule_overlap_counts.values())
        or min(rows) != 700_108_071_901
        or max(rows) != 705_207_072_198
        or digest != CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SEED_SET_SHA256
    ):
        raise RuntimeError(
            "candidate02 performance-lock recovery seed schedule changed"
        )
    return {
        "schema": (
            "hu_m31_t3_step6d_candidate02_" "performance_lock_recovery_seed_schedule_v2"
        ),
        "schedule": CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SCHEDULE,
        "role": CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_ROLE,
        "recovery_of_run_id": CANDIDATE02_PERFORMANCE_LOCK_RUN_ID,
        "seed_stride": SEED_STRIDE,
        "formula": "namespace_seed_base + seed_stride * hand_index",
        "namespace_bases": dict(CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_NAMESPACE_BASES),
        "hand_count": len(CONTRACT_HAND_INDICES),
        "namespace_count": len(CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_NAMESPACE_BASES),
        "seed_count": len(rows),
        "seed_min": min(rows),
        "seed_max": max(rows),
        "seed_set_sha256": digest,
        "existing_step6d_schedule_overlap_counts": schedule_overlap_counts,
        "existing_step6d_union_overlap_count": 0,
        "candidate02_development_overlap_count": 0,
        "performance_lock_v1_overlap_count": 0,
        "all_values_unique": True,
        "locked_before_content_read": True,
        "training_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
    }


def candidate02_performance_lock_recovery_v3_seed_values(
    index: int,
) -> dict[str, int]:
    """Return the fresh rearm2 recovery-v3 namespace seeds for one hand."""

    checked = _normalize_indices(
        (index,), label="candidate02 performance-lock recovery-v3 hand index"
    )[0]
    return {
        namespace: base + SEED_STRIDE * checked
        for namespace, base in (
            CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_NAMESPACE_BASES.items()
        )
    }


def candidate02_performance_lock_recovery_v3_seed_contract() -> dict[str, Any]:
    """Audit rearm2 against Step6d, development, v1 lock, and rearm1."""

    rows = [
        seed
        for index in CONTRACT_HAND_INDICES
        for seed in (
            candidate02_performance_lock_recovery_v3_seed_values(index).values()
        )
    ]
    recovery = set(rows)
    existing_step6d = set(planned_seed_values())
    candidate02_development = {
        seed
        for index in CONTRACT_HAND_INDICES
        for seed in candidate02_seed_values(index).values()
    }
    performance_lock_v1 = {
        seed
        for index in CONTRACT_HAND_INDICES
        for seed in candidate02_performance_lock_seed_values(index).values()
    }
    performance_lock_rearm1 = {
        seed
        for index in CONTRACT_HAND_INDICES
        for seed in candidate02_performance_lock_recovery_seed_values(index).values()
    }
    schedule_overlap_counts = {
        schedule.name: len(recovery & set(schedule.values()))
        for schedule in SEED_SCHEDULES
    }
    digest = contract_canonical_sha256(sorted(rows))
    if (
        len(rows) != 600
        or len(recovery) != 600
        or recovery & existing_step6d
        or recovery & candidate02_development
        or recovery & performance_lock_v1
        or recovery & performance_lock_rearm1
        or any(schedule_overlap_counts.values())
        or min(rows) != 710_108_071_901
        or max(rows) != 715_207_072_198
        or digest != CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_SEED_SET_SHA256
    ):
        raise RuntimeError(
            "candidate02 performance-lock recovery-v3 seed schedule changed"
        )
    return {
        "schema": (
            "hu_m31_t3_step6d_candidate02_"
            "performance_lock_recovery_seed_schedule_v3"
        ),
        "schedule": CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_SCHEDULE,
        "role": CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_ROLE,
        "recovery_of_run_id": CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_RUN_ID,
        "seed_stride": SEED_STRIDE,
        "formula": "namespace_seed_base + seed_stride * hand_index",
        "namespace_bases": dict(
            CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_NAMESPACE_BASES
        ),
        "hand_count": len(CONTRACT_HAND_INDICES),
        "namespace_count": len(
            CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_NAMESPACE_BASES
        ),
        "seed_count": len(rows),
        "seed_min": min(rows),
        "seed_max": max(rows),
        "seed_set_sha256": digest,
        "existing_step6d_schedule_overlap_counts": schedule_overlap_counts,
        "existing_step6d_union_overlap_count": 0,
        "candidate02_development_overlap_count": 0,
        "performance_lock_v1_overlap_count": 0,
        "performance_lock_rearm1_overlap_count": 0,
        "all_values_unique": True,
        "locked_before_content_read": True,
        "training_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
    }


def candidate02_performance_lock_v4_seed_values(index: int) -> dict[str, int]:
    """Return the fresh run009-authorized lock-v4 seeds for one paired hand."""

    checked = _normalize_indices(
        (index,), label="candidate02 performance-lock-v4 hand index"
    )[0]
    return {
        namespace: base + SEED_STRIDE * checked
        for namespace, base in CANDIDATE02_PERFORMANCE_LOCK_V4_NAMESPACE_BASES.items()
    }


def candidate02_performance_lock_v4_seed_contract() -> dict[str, Any]:
    """Audit lock-v4 against every existing Step6d and prior lock schedule."""

    rows = [
        seed
        for index in CONTRACT_HAND_INDICES
        for seed in candidate02_performance_lock_v4_seed_values(index).values()
    ]
    v4 = set(rows)
    existing_step6d = set(planned_seed_values())
    candidate02_development = {
        seed
        for index in CONTRACT_HAND_INDICES
        for seed in candidate02_seed_values(index).values()
    }
    performance_lock_v1 = {
        seed
        for index in CONTRACT_HAND_INDICES
        for seed in candidate02_performance_lock_seed_values(index).values()
    }
    performance_lock_rearm1 = {
        seed
        for index in CONTRACT_HAND_INDICES
        for seed in candidate02_performance_lock_recovery_seed_values(index).values()
    }
    performance_lock_rearm2 = {
        seed
        for index in CONTRACT_HAND_INDICES
        for seed in (
            candidate02_performance_lock_recovery_v3_seed_values(index).values()
        )
    }
    schedule_overlap_counts = {
        schedule.name: len(v4 & set(schedule.values())) for schedule in SEED_SCHEDULES
    }
    digest = contract_canonical_sha256(sorted(rows))
    if (
        len(rows) != 600
        or len(v4) != 600
        or v4 & existing_step6d
        or v4 & candidate02_development
        or v4 & performance_lock_v1
        or v4 & performance_lock_rearm1
        or v4 & performance_lock_rearm2
        or any(schedule_overlap_counts.values())
        or min(rows) != 720_108_071_901
        or max(rows) != 725_207_072_198
        or digest != CANDIDATE02_PERFORMANCE_LOCK_V4_SEED_SET_SHA256
    ):
        raise RuntimeError("candidate02 performance-lock-v4 seed schedule changed")
    return {
        "schema": (
            "hu_m31_t3_step6d_candidate02_performance_lock_seed_schedule_v4"
        ),
        "schedule": CANDIDATE02_PERFORMANCE_LOCK_V4_SCHEDULE,
        "role": CANDIDATE02_PERFORMANCE_LOCK_V4_ROLE,
        "seed_stride": SEED_STRIDE,
        "formula": "namespace_seed_base + seed_stride * hand_index",
        "namespace_bases": dict(CANDIDATE02_PERFORMANCE_LOCK_V4_NAMESPACE_BASES),
        "hand_count": len(CONTRACT_HAND_INDICES),
        "namespace_count": len(CANDIDATE02_PERFORMANCE_LOCK_V4_NAMESPACE_BASES),
        "seed_count": len(rows),
        "seed_min": min(rows),
        "seed_max": max(rows),
        "seed_set_sha256": digest,
        "existing_step6d_schedule_overlap_counts": schedule_overlap_counts,
        "existing_step6d_union_overlap_count": 0,
        "candidate02_development_overlap_count": 0,
        "performance_lock_v1_overlap_count": 0,
        "performance_lock_rearm1_overlap_count": 0,
        "performance_lock_rearm2_overlap_count": 0,
        "all_values_unique": True,
        "locked_before_content_read": True,
        "training_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
    }


def _candidate02_contract_anchor_sha256() -> str:
    return canonical_sha256(
        {
            "schema": "hu_m31_t3_step6d_candidate02_contract_anchor_v1",
            "parent_contract_canonical_sha256": (
                v1.EXPECTED_STEP6D_CONTRACT_CANONICAL_SHA256
            ),
            "candidate_variant": CANDIDATE02_VARIANT,
            "schedule": CANDIDATE02_SCHEDULE,
            "run_id": CANDIDATE02_RUN_ID,
            "contract_hand_indices": list(CONTRACT_HAND_INDICES),
            "tail_hand_indices": list(TAIL_HAND_INDICES),
            "budget": dict(v1.PERFORMANCE_BUDGET),
            "allocation": dict(ALLOCATION),
            "seed_contract": candidate02_seed_contract(),
        }
    )


def _candidate02_tail_v2_contract_anchor_sha256() -> str:
    return canonical_sha256(
        {
            "schema": "hu_m31_t3_step6d_candidate02_contract_anchor_v2",
            "parent_contract_anchor_sha256": _candidate02_contract_anchor_sha256(),
            "candidate_variant": CANDIDATE02_TAIL_V2_VARIANT,
            "schedule": CANDIDATE02_SCHEDULE,
            "run_id": CANDIDATE02_TAIL_V2_RUN_ID,
            "contract_hand_indices": list(CONTRACT_HAND_INDICES),
            "tail_hand_indices": list(CANDIDATE02_TAIL_V2_TAIL_HAND_INDICES),
            "budget": dict(v1.PERFORMANCE_BUDGET),
            "allocation": dict(ALLOCATION),
            "seed_contract": candidate02_seed_contract(),
            "selection_manifest_sha256": (
                CANDIDATE02_TAIL_V2_SELECTION_MANIFEST_SHA256
            ),
        }
    )


def _candidate02_performance_lock_contract_anchor_sha256() -> str:
    return canonical_sha256(
        {
            "schema": (
                "hu_m31_t3_step6d_candidate02_performance_lock_contract_anchor_v1"
            ),
            "parent_contract_anchor_sha256": _candidate02_contract_anchor_sha256(),
            "candidate_variant": CANDIDATE02_PERFORMANCE_LOCK_VARIANT,
            "schedule": CANDIDATE02_PERFORMANCE_LOCK_SCHEDULE,
            "role": CANDIDATE02_PERFORMANCE_LOCK_ROLE,
            "run_id": CANDIDATE02_PERFORMANCE_LOCK_RUN_ID,
            "contract_hand_indices": list(CONTRACT_HAND_INDICES),
            "tail_hand_indices": [],
            "budget": dict(v1.PERFORMANCE_BUDGET),
            "allocation": dict(ALLOCATION),
            "seed_contract": candidate02_performance_lock_seed_contract(),
        }
    )


def _candidate02_performance_lock_recovery_contract_anchor_sha256() -> str:
    return canonical_sha256(
        {
            "schema": (
                "hu_m31_t3_step6d_candidate02_"
                "performance_lock_recovery_contract_anchor_v2"
            ),
            "parent_performance_lock_contract_anchor_sha256": (
                _candidate02_performance_lock_contract_anchor_sha256()
            ),
            "candidate_variant": CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT,
            "schedule": CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SCHEDULE,
            "role": CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_ROLE,
            "recovery_of_run_id": CANDIDATE02_PERFORMANCE_LOCK_RUN_ID,
            "run_id": CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_RUN_ID,
            "contract_hand_indices": list(CONTRACT_HAND_INDICES),
            "tail_hand_indices": [],
            "budget": dict(v1.PERFORMANCE_BUDGET),
            "allocation": dict(ALLOCATION),
            "accepted_candidate_library_sha256": (
                CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_CANDIDATE_LIBRARY_SHA256
            ),
            "accepted_reference_library_sha256": (
                CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_REFERENCE_LIBRARY_SHA256
            ),
            "seed_contract": (candidate02_performance_lock_recovery_seed_contract()),
        }
    )


def _candidate02_performance_lock_recovery_v3_contract_anchor_sha256() -> str:
    return canonical_sha256(
        {
            "schema": (
                "hu_m31_t3_step6d_candidate02_"
                "performance_lock_recovery_contract_anchor_v3"
            ),
            "parent_recovery_contract_anchor_sha256": (
                _candidate02_performance_lock_recovery_contract_anchor_sha256()
            ),
            "candidate_variant": (
                CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_VARIANT
            ),
            "schedule": CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_SCHEDULE,
            "role": CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_ROLE,
            "recovery_of_run_id": CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_RUN_ID,
            "run_id": CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_RUN_ID,
            "contract_hand_indices": list(CONTRACT_HAND_INDICES),
            "tail_hand_indices": [],
            "budget": dict(v1.PERFORMANCE_BUDGET),
            "allocation": dict(ALLOCATION),
            "accepted_candidate_library_sha256": (
                CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_CANDIDATE_LIBRARY_SHA256
            ),
            "accepted_reference_library_sha256": (
                CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_REFERENCE_LIBRARY_SHA256
            ),
            "seed_contract": (
                candidate02_performance_lock_recovery_v3_seed_contract()
            ),
        }
    )


def _candidate02_performance_lock_v4_contract_anchor_sha256() -> str:
    return canonical_sha256(
        {
            "schema": (
                "hu_m31_t3_step6d_candidate02_performance_lock_contract_anchor_v4"
            ),
            "parent_recovery_v3_contract_anchor_sha256": (
                _candidate02_performance_lock_recovery_v3_contract_anchor_sha256()
            ),
            "candidate_variant": CANDIDATE02_PERFORMANCE_LOCK_V4_VARIANT,
            "schedule": CANDIDATE02_PERFORMANCE_LOCK_V4_SCHEDULE,
            "role": CANDIDATE02_PERFORMANCE_LOCK_V4_ROLE,
            "run_id": CANDIDATE02_PERFORMANCE_LOCK_V4_RUN_ID,
            "contract_hand_indices": list(CONTRACT_HAND_INDICES),
            "tail_hand_indices": [],
            "budget": dict(v1.PERFORMANCE_BUDGET),
            "allocation": dict(ALLOCATION),
            "accepted_candidate_library_sha256": (
                CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_CANDIDATE_LIBRARY_SHA256
            ),
            "accepted_reference_library_sha256": (
                CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_REFERENCE_LIBRARY_SHA256
            ),
            "authorizing_gate_receipt_file_sha256": (
                CANDIDATE02_PERFORMANCE_LOCK_V4_GATE_RECEIPT_FILE_SHA256
            ),
            "authorizing_gate_receipt_sha256": (
                CANDIDATE02_PERFORMANCE_LOCK_V4_GATE_RECEIPT_SHA256
            ),
            "seed_contract": candidate02_performance_lock_v4_seed_contract(),
        }
    )


def candidate02_schedule_row(index: int) -> dict[str, Any]:
    checked = _normalize_indices((index,), label="candidate02 hand index")[0]
    return {
        "schedule": CANDIDATE02_SCHEDULE,
        "hand_index": checked,
        "root_indices": [checked * 2, checked * 2 + 1],
        "profile": v1.behavior_profile_for_index(checked),
        "seeds": candidate02_seed_values(checked),
        "budget": dict(v1.PERFORMANCE_BUDGET),
        "training_eligible": False,
    }


def candidate02_performance_lock_schedule_row(index: int) -> dict[str, Any]:
    checked = _normalize_indices(
        (index,), label="candidate02 performance-lock hand index"
    )[0]
    return {
        "schedule": CANDIDATE02_PERFORMANCE_LOCK_SCHEDULE,
        "hand_index": checked,
        "root_indices": [checked * 2, checked * 2 + 1],
        "profile": v1.behavior_profile_for_index(checked),
        "seeds": candidate02_performance_lock_seed_values(checked),
        "budget": dict(v1.PERFORMANCE_BUDGET),
        "training_eligible": False,
    }


def candidate02_performance_lock_recovery_schedule_row(
    index: int,
) -> dict[str, Any]:
    checked = _normalize_indices(
        (index,), label="candidate02 performance-lock recovery hand index"
    )[0]
    return {
        "schedule": CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SCHEDULE,
        "hand_index": checked,
        "root_indices": [checked * 2, checked * 2 + 1],
        "profile": v1.behavior_profile_for_index(checked),
        "seeds": candidate02_performance_lock_recovery_seed_values(checked),
        "budget": dict(v1.PERFORMANCE_BUDGET),
        "training_eligible": False,
    }


def candidate02_performance_lock_recovery_v3_schedule_row(
    index: int,
) -> dict[str, Any]:
    checked = _normalize_indices(
        (index,), label="candidate02 performance-lock recovery-v3 hand index"
    )[0]
    return {
        "schedule": CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_SCHEDULE,
        "hand_index": checked,
        "root_indices": [checked * 2, checked * 2 + 1],
        "profile": v1.behavior_profile_for_index(checked),
        "seeds": candidate02_performance_lock_recovery_v3_seed_values(checked),
        "budget": dict(v1.PERFORMANCE_BUDGET),
        "training_eligible": False,
    }


def candidate02_performance_lock_v4_schedule_row(index: int) -> dict[str, Any]:
    checked = _normalize_indices(
        (index,), label="candidate02 performance-lock-v4 hand index"
    )[0]
    return {
        "schedule": CANDIDATE02_PERFORMANCE_LOCK_V4_SCHEDULE,
        "hand_index": checked,
        "root_indices": [checked * 2, checked * 2 + 1],
        "profile": v1.behavior_profile_for_index(checked),
        "seeds": candidate02_performance_lock_v4_seed_values(checked),
        "budget": dict(v1.PERFORMANCE_BUDGET),
        "training_eligible": False,
    }


def contract_variant(value: Mapping[str, Any]) -> str:
    schema = value.get("schema")
    if schema == RUN_CONTRACT_SCHEMA:
        return CANDIDATE01_VARIANT
    if schema == CANDIDATE02_RUN_CONTRACT_SCHEMA:
        return CANDIDATE02_VARIANT
    if schema == CANDIDATE02_TAIL_V2_RUN_CONTRACT_SCHEMA:
        return CANDIDATE02_TAIL_V2_VARIANT
    if schema == CANDIDATE02_PERFORMANCE_LOCK_RUN_CONTRACT_SCHEMA:
        return CANDIDATE02_PERFORMANCE_LOCK_VARIANT
    if schema == CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_RUN_CONTRACT_SCHEMA:
        return CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT
    if schema == CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_RUN_CONTRACT_SCHEMA:
        return CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_VARIANT
    if schema == CANDIDATE02_PERFORMANCE_LOCK_V4_RUN_CONTRACT_SCHEMA:
        return CANDIDATE02_PERFORMANCE_LOCK_V4_VARIANT
    raise ValueError("Step 6d v2 run contract schema is not recognized")


def _is_candidate02(contract: Mapping[str, Any]) -> bool:
    return contract_variant(contract) != CANDIDATE01_VARIANT


def _run_id(contract: Mapping[str, Any]) -> str:
    variant = contract_variant(contract)
    if variant == CANDIDATE01_VARIANT:
        return v1.STEP6D_RUN_ID
    if variant == CANDIDATE02_VARIANT:
        return CANDIDATE02_RUN_ID
    if variant == CANDIDATE02_TAIL_V2_VARIANT:
        return CANDIDATE02_TAIL_V2_RUN_ID
    if variant == CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT:
        return CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_RUN_ID
    if variant == CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_VARIANT:
        return CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_RUN_ID
    if variant == CANDIDATE02_PERFORMANCE_LOCK_V4_VARIANT:
        return CANDIDATE02_PERFORMANCE_LOCK_V4_RUN_ID
    return CANDIDATE02_PERFORMANCE_LOCK_RUN_ID


def _schedule_row(contract: Mapping[str, Any], index: int) -> dict[str, Any]:
    variant = contract_variant(contract)
    if variant == CANDIDATE01_VARIANT:
        return v1.performance_schedule_row(index)
    if variant == CANDIDATE02_PERFORMANCE_LOCK_VARIANT:
        return candidate02_performance_lock_schedule_row(index)
    if variant == CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT:
        return candidate02_performance_lock_recovery_schedule_row(index)
    if variant == CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_VARIANT:
        return candidate02_performance_lock_recovery_v3_schedule_row(index)
    if variant == CANDIDATE02_PERFORMANCE_LOCK_V4_VARIANT:
        return candidate02_performance_lock_v4_schedule_row(index)
    return candidate02_schedule_row(index)


def _source_hand_schema(contract: Mapping[str, Any]) -> str:
    variant = contract_variant(contract)
    if variant == CANDIDATE01_VARIANT:
        return SOURCE_HAND_SCHEMA
    if variant == CANDIDATE02_VARIANT:
        return CANDIDATE02_SOURCE_HAND_SCHEMA
    if variant == CANDIDATE02_TAIL_V2_VARIANT:
        return CANDIDATE02_TAIL_V2_SOURCE_HAND_SCHEMA
    if variant == CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT:
        return CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SOURCE_HAND_SCHEMA
    if variant == CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_VARIANT:
        return CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_SOURCE_HAND_SCHEMA
    if variant == CANDIDATE02_PERFORMANCE_LOCK_V4_VARIANT:
        return CANDIDATE02_PERFORMANCE_LOCK_V4_SOURCE_HAND_SCHEMA
    return CANDIDATE02_PERFORMANCE_LOCK_SOURCE_HAND_SCHEMA


def _done_schema(contract: Mapping[str, Any]) -> str:
    variant = contract_variant(contract)
    if variant == CANDIDATE01_VARIANT:
        return DONE_SCHEMA
    if variant == CANDIDATE02_VARIANT:
        return CANDIDATE02_DONE_SCHEMA
    if variant == CANDIDATE02_TAIL_V2_VARIANT:
        return CANDIDATE02_TAIL_V2_DONE_SCHEMA
    if variant == CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT:
        return CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_DONE_SCHEMA
    if variant == CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_VARIANT:
        return CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_DONE_SCHEMA
    if variant == CANDIDATE02_PERFORMANCE_LOCK_V4_VARIANT:
        return CANDIDATE02_PERFORMANCE_LOCK_V4_DONE_SCHEMA
    return CANDIDATE02_PERFORMANCE_LOCK_DONE_SCHEMA


def _shard_manifest_schema(contract: Mapping[str, Any]) -> str:
    variant = contract_variant(contract)
    if variant == CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT:
        return CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SHARD_MANIFEST_SCHEMA
    if variant == CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_VARIANT:
        return CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_SHARD_MANIFEST_SCHEMA
    if variant == CANDIDATE02_PERFORMANCE_LOCK_V4_VARIANT:
        return CANDIDATE02_PERFORMANCE_LOCK_V4_SHARD_MANIFEST_SCHEMA
    return SHARD_MANIFEST_SCHEMA


def build_run_contract(
    *,
    candidate_library_sha256: str,
    reference_library_sha256: str,
    workers: int = 1,
    rayon_threads_per_worker: int = 16,
    contract_hand_indices: Sequence[int] = CONTRACT_HAND_INDICES,
    variant: str = CANDIDATE01_VARIANT,
) -> dict[str, Any]:
    """Build the shared contract; source role and shard subset are excluded."""

    if tuple(contract_hand_indices) != CONTRACT_HAND_INDICES:
        raise ValueError("Step 6d v2 shared contract must bind all 100 hands")
    if {
        "workers": workers,
        "rayon_threads_per_worker": rayon_threads_per_worker,
    } != ALLOCATION:
        raise ValueError("Step 6d v2 allocation must be exactly 1 worker x 16 Rayon")
    if (
        not _is_sha256(candidate_library_sha256)
        or not _is_sha256(reference_library_sha256)
        or candidate_library_sha256 == reference_library_sha256
    ):
        raise ValueError("Step 6d v2 requires distinct lowercase binary SHA-256 values")
    if variant not in (
        CANDIDATE01_VARIANT,
        CANDIDATE02_VARIANT,
        CANDIDATE02_TAIL_V2_VARIANT,
        CANDIDATE02_PERFORMANCE_LOCK_VARIANT,
        CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT,
        CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_VARIANT,
        CANDIDATE02_PERFORMANCE_LOCK_V4_VARIANT,
    ):
        raise ValueError("Step 6d v2 contract variant is not recognized")
    candidate02 = variant != CANDIDATE01_VARIANT
    tail_v2 = variant == CANDIDATE02_TAIL_V2_VARIANT
    performance_lock = variant == CANDIDATE02_PERFORMANCE_LOCK_VARIANT
    performance_lock_recovery = variant == CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT
    performance_lock_recovery_v3 = (
        variant == CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_VARIANT
    )
    performance_lock_v4 = variant == CANDIDATE02_PERFORMANCE_LOCK_V4_VARIANT
    performance_lock_recovery_any = (
        performance_lock_recovery or performance_lock_recovery_v3
    )
    performance_lock_any = (
        performance_lock or performance_lock_recovery_any or performance_lock_v4
    )
    if (performance_lock_recovery_any or performance_lock_v4) and (
        candidate_library_sha256
        != CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_CANDIDATE_LIBRARY_SHA256
        or reference_library_sha256
        != CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_REFERENCE_LIBRARY_SHA256
    ):
        raise ValueError(
            "Candidate02 performance-lock recovery requires the accepted binary hashes"
        )
    value = {
        "schema": (
            CANDIDATE02_PERFORMANCE_LOCK_V4_RUN_CONTRACT_SCHEMA
            if performance_lock_v4
            else (
                CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_RUN_CONTRACT_SCHEMA
                if performance_lock_recovery_v3
                else (
                    CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_RUN_CONTRACT_SCHEMA
                    if performance_lock_recovery
                    else (
                        CANDIDATE02_PERFORMANCE_LOCK_RUN_CONTRACT_SCHEMA
                        if performance_lock
                        else (
                            CANDIDATE02_TAIL_V2_RUN_CONTRACT_SCHEMA
                            if tail_v2
                            else (
                                CANDIDATE02_RUN_CONTRACT_SCHEMA
                                if candidate02
                                else RUN_CONTRACT_SCHEMA
                            )
                        )
                    )
                )
            )
        ),
        "step6d_run_id": (
            CANDIDATE02_PERFORMANCE_LOCK_V4_RUN_ID
            if performance_lock_v4
            else (
                CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_RUN_ID
                if performance_lock_recovery_v3
                else (
                    CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_RUN_ID
                    if performance_lock_recovery
                    else (
                        CANDIDATE02_PERFORMANCE_LOCK_RUN_ID
                        if performance_lock
                        else (
                            CANDIDATE02_TAIL_V2_RUN_ID
                            if tail_v2
                            else (
                                CANDIDATE02_RUN_ID
                                if candidate02
                                else v1.STEP6D_RUN_ID
                            )
                        )
                    )
                )
            )
        ),
        "contract_canonical_sha256": (
            _candidate02_performance_lock_v4_contract_anchor_sha256()
            if performance_lock_v4
            else (
                _candidate02_performance_lock_recovery_v3_contract_anchor_sha256()
                if performance_lock_recovery_v3
                else (
                    _candidate02_performance_lock_recovery_contract_anchor_sha256()
                    if performance_lock_recovery
                    else (
                        _candidate02_performance_lock_contract_anchor_sha256()
                        if performance_lock
                        else (
                            _candidate02_tail_v2_contract_anchor_sha256()
                            if tail_v2
                            else (
                                _candidate02_contract_anchor_sha256()
                                if candidate02
                                else v1.EXPECTED_STEP6D_CONTRACT_CANONICAL_SHA256
                            )
                        )
                    )
                )
            )
        ),
        "schedule": (
            CANDIDATE02_PERFORMANCE_LOCK_V4_SCHEDULE
            if performance_lock_v4
            else (
                CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_SCHEDULE
                if performance_lock_recovery_v3
                else (
                    CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SCHEDULE
                    if performance_lock_recovery
                    else (
                        CANDIDATE02_PERFORMANCE_LOCK_SCHEDULE
                        if performance_lock
                        else (
                            CANDIDATE02_SCHEDULE
                            if candidate02
                            else v1.STEP6D_PERFORMANCE_SCHEDULE
                        )
                    )
                )
            )
        ),
        "contract_hand_indices": list(CONTRACT_HAND_INDICES),
        "tail_hand_indices": list(
            ()
            if performance_lock_any
            else CANDIDATE02_TAIL_V2_TAIL_HAND_INDICES if tail_v2 else TAIL_HAND_INDICES
        ),
        "budget": dict(v1.PERFORMANCE_BUDGET),
        "allocation": dict(ALLOCATION),
        "reference_library_sha256": reference_library_sha256,
        "candidate_library_sha256": candidate_library_sha256,
        "execution_mode": "one_source_one_process_scalar_1x16",
        "training_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }
    if candidate02:
        value["candidate_variant"] = variant
        value["seed_contract"] = (
            candidate02_performance_lock_v4_seed_contract()
            if performance_lock_v4
            else (
                candidate02_performance_lock_recovery_v3_seed_contract()
                if performance_lock_recovery_v3
                else (
                    candidate02_performance_lock_recovery_seed_contract()
                    if performance_lock_recovery
                    else (
                        candidate02_performance_lock_seed_contract()
                        if performance_lock
                        else candidate02_seed_contract()
                    )
                )
            )
        )
    if performance_lock_v4:
        value["authorizing_gate_receipt_file_sha256"] = (
            CANDIDATE02_PERFORMANCE_LOCK_V4_GATE_RECEIPT_FILE_SHA256
        )
        value["authorizing_gate_receipt_sha256"] = (
            CANDIDATE02_PERFORMANCE_LOCK_V4_GATE_RECEIPT_SHA256
        )
    if tail_v2:
        value["selection_manifest_sha256"] = (
            CANDIDATE02_TAIL_V2_SELECTION_MANIFEST_SHA256
        )
    return validate_run_contract(value)


def validate_run_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    variant = contract_variant(payload)
    expected_keys = {
        CANDIDATE01_VARIANT: _RUN_CONTRACT_KEYS,
        CANDIDATE02_VARIANT: _CANDIDATE02_RUN_CONTRACT_KEYS,
        CANDIDATE02_TAIL_V2_VARIANT: _CANDIDATE02_TAIL_V2_RUN_CONTRACT_KEYS,
        CANDIDATE02_PERFORMANCE_LOCK_VARIANT: (
            _CANDIDATE02_PERFORMANCE_LOCK_RUN_CONTRACT_KEYS
        ),
        CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT: (
            _CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_RUN_CONTRACT_KEYS
        ),
        CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_VARIANT: (
            _CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_RUN_CONTRACT_KEYS
        ),
        CANDIDATE02_PERFORMANCE_LOCK_V4_VARIANT: (
            _CANDIDATE02_PERFORMANCE_LOCK_V4_RUN_CONTRACT_KEYS
        ),
    }[variant]
    _require_exact_keys(payload, expected_keys, "Step 6d v2 run contract")
    indices = payload.get("contract_hand_indices")
    candidate02 = variant != CANDIDATE01_VARIANT
    tail_v2 = variant == CANDIDATE02_TAIL_V2_VARIANT
    performance_lock = variant == CANDIDATE02_PERFORMANCE_LOCK_VARIANT
    performance_lock_recovery = variant == CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT
    performance_lock_recovery_v3 = (
        variant == CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_VARIANT
    )
    performance_lock_v4 = variant == CANDIDATE02_PERFORMANCE_LOCK_V4_VARIANT
    performance_lock_recovery_any = (
        performance_lock_recovery or performance_lock_recovery_v3
    )
    performance_lock_any = (
        performance_lock or performance_lock_recovery_any or performance_lock_v4
    )
    expected_run_id = (
        CANDIDATE02_PERFORMANCE_LOCK_V4_RUN_ID
        if performance_lock_v4
        else (
            CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_RUN_ID
            if performance_lock_recovery_v3
            else (
                CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_RUN_ID
                if performance_lock_recovery
                else (
                    CANDIDATE02_PERFORMANCE_LOCK_RUN_ID
                    if performance_lock
                    else (
                        CANDIDATE02_TAIL_V2_RUN_ID
                        if tail_v2
                        else CANDIDATE02_RUN_ID if candidate02 else v1.STEP6D_RUN_ID
                    )
                )
            )
        )
    )
    expected_anchor = (
        _candidate02_performance_lock_v4_contract_anchor_sha256()
        if performance_lock_v4
        else (
            _candidate02_performance_lock_recovery_v3_contract_anchor_sha256()
            if performance_lock_recovery_v3
            else (
                _candidate02_performance_lock_recovery_contract_anchor_sha256()
                if performance_lock_recovery
                else (
                    _candidate02_performance_lock_contract_anchor_sha256()
                    if performance_lock
                    else (
                        _candidate02_tail_v2_contract_anchor_sha256()
                        if tail_v2
                        else (
                            _candidate02_contract_anchor_sha256()
                            if candidate02
                            else v1.EXPECTED_STEP6D_CONTRACT_CANONICAL_SHA256
                        )
                    )
                )
            )
        )
    )
    expected_schedule = (
        CANDIDATE02_PERFORMANCE_LOCK_V4_SCHEDULE
        if performance_lock_v4
        else (
            CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_SCHEDULE
            if performance_lock_recovery_v3
            else (
                CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SCHEDULE
                if performance_lock_recovery
                else (
                    CANDIDATE02_PERFORMANCE_LOCK_SCHEDULE
                    if performance_lock
                    else (
                        CANDIDATE02_SCHEDULE
                        if candidate02
                        else v1.STEP6D_PERFORMANCE_SCHEDULE
                    )
                )
            )
        )
    )
    expected_tail = (
        []
        if performance_lock_any
        else list(
            CANDIDATE02_TAIL_V2_TAIL_HAND_INDICES if tail_v2 else TAIL_HAND_INDICES
        )
    )
    expected_seed_contract = (
        candidate02_performance_lock_v4_seed_contract()
        if performance_lock_v4
        else (
            candidate02_performance_lock_recovery_v3_seed_contract()
            if performance_lock_recovery_v3
            else (
                candidate02_performance_lock_recovery_seed_contract()
                if performance_lock_recovery
                else (
                    candidate02_performance_lock_seed_contract()
                    if performance_lock
                    else candidate02_seed_contract()
                )
            )
        )
    )
    if (
        payload.get("step6d_run_id") != expected_run_id
        or payload.get("contract_canonical_sha256") != expected_anchor
        or payload.get("schedule") != expected_schedule
        or indices != list(CONTRACT_HAND_INDICES)
        or payload.get("tail_hand_indices") != expected_tail
        or payload.get("budget") != v1.PERFORMANCE_BUDGET
        or payload.get("allocation") != ALLOCATION
        or not _is_sha256(payload.get("reference_library_sha256"))
        or not _is_sha256(payload.get("candidate_library_sha256"))
        or payload.get("reference_library_sha256")
        == payload.get("candidate_library_sha256")
        or payload.get("execution_mode") != "one_source_one_process_scalar_1x16"
        or any(
            payload.get(field) is not False
            for field in (
                "training_eligible",
                "quality_evidence",
                "promotion_evidence",
                "current_profile_changed",
            )
        )
        or (
            candidate02
            and (
                payload.get("candidate_variant") != variant
                or payload.get("seed_contract") != expected_seed_contract
            )
        )
        or (
            tail_v2
            and payload.get("selection_manifest_sha256")
            != CANDIDATE02_TAIL_V2_SELECTION_MANIFEST_SHA256
        )
        or (
            (performance_lock_recovery_any or performance_lock_v4)
            and (
                payload.get("candidate_library_sha256")
                != CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_CANDIDATE_LIBRARY_SHA256
                or payload.get("reference_library_sha256")
                != CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_REFERENCE_LIBRARY_SHA256
            )
        )
        or (
            performance_lock_v4
            and (
                payload.get("authorizing_gate_receipt_file_sha256")
                != CANDIDATE02_PERFORMANCE_LOCK_V4_GATE_RECEIPT_FILE_SHA256
                or payload.get("authorizing_gate_receipt_sha256")
                != CANDIDATE02_PERFORMANCE_LOCK_V4_GATE_RECEIPT_SHA256
            )
        )
    ):
        raise ValueError("Step 6d v2 shared run contract changed")
    return payload


def build_shard_manifest(
    *,
    run_contract: Mapping[str, Any],
    source_role: str,
    work_hand_indices: Sequence[int],
) -> dict[str, Any]:
    contract = validate_run_contract(run_contract)
    work = _normalize_indices(work_hand_indices, label="work_hand_indices")
    if source_role not in SOURCE_ROLES:
        raise ValueError("Step 6d v2 source_role must be candidate or reference")
    if not set(work).issubset(contract["contract_hand_indices"]):
        raise ValueError("Step 6d v2 work lies outside the shared contract")
    if contract_variant(contract) == CANDIDATE02_TAIL_V2_VARIANT and not set(
        work
    ).issubset(contract["tail_hand_indices"]):
        raise ValueError("Candidate02 tail-v2 work lies outside the selected tail")
    return {
        "schema": _shard_manifest_schema(contract),
        "run_contract": contract,
        "run_contract_digest": canonical_sha256(contract),
        "source_role": source_role,
        "work_hand_indices": list(work),
    }


def validate_shard_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    _require_exact_keys(payload, _SHARD_MANIFEST_KEYS, "Step 6d v2 shard manifest")
    contract_raw = payload.get("run_contract")
    if not isinstance(contract_raw, Mapping):
        raise ValueError("Step 6d v2 shard run contract is missing")
    contract = validate_run_contract(contract_raw)
    role = payload.get("source_role")
    work_raw = payload.get("work_hand_indices")
    if not isinstance(work_raw, list):
        raise ValueError("Step 6d v2 work_hand_indices must be a list")
    work = _normalize_indices(work_raw, label="work_hand_indices")
    if contract_variant(contract) == CANDIDATE02_TAIL_V2_VARIANT and not set(
        work
    ).issubset(contract["tail_hand_indices"]):
        raise ValueError("Candidate02 tail-v2 work lies outside the selected tail")
    if (
        payload.get("schema") != _shard_manifest_schema(contract)
        or payload.get("run_contract_digest") != canonical_sha256(contract)
        or role not in SOURCE_ROLES
        or not set(work).issubset(contract["contract_hand_indices"])
    ):
        raise ValueError("Step 6d v2 shard manifest changed")
    payload["run_contract"] = contract
    return payload


def _validate_candidate02_root_artifact(
    value: Mapping[str, Any], *, expected_row: Mapping[str, Any]
) -> tuple[Any, Any]:
    if set(value) != v1._ROOT_KEYS or value.get("schema") != CANDIDATE02_ROOT_SCHEMA:
        raise ValueError("candidate02 performance root schema changed")
    observations = value.get("observations")
    if (
        value.get("contract_canonical_sha256") != _candidate02_contract_anchor_sha256()
        or value.get("schedule") != CANDIDATE02_SCHEDULE
        or value.get("schedule_row_sha256") != canonical_sha256(expected_row)
        or value.get("hand_index") != expected_row["hand_index"]
        or value.get("root_indices") != expected_row["root_indices"]
        or value.get("profile") != expected_row["profile"]
        or value.get("seeds") != expected_row["seeds"]
        or value.get("budget") != v1.PERFORMANCE_BUDGET
        or value.get("current_profile_resolved") is not False
        or value.get("opponent_private_discards_used") is not False
        or value.get("training_eligible") is not False
        or not isinstance(observations, list)
        or len(observations) != 2
    ):
        raise ValueError("candidate02 performance root provenance changed")
    parsed = []
    for offset, raw in enumerate(observations):
        if not isinstance(raw, Mapping) or set(raw) != v1._ROOT_OBSERVATION_KEYS:
            raise ValueError("candidate02 root observation schema changed")
        observation = v1.ActorObservation.from_dict(raw["observation"])
        expected_seat = "first" if offset == 0 else "second"
        if (
            raw.get("root_index") != expected_row["root_indices"][offset]
            or raw.get("seat") != expected_seat
            or observation.seat != expected_seat
            or observation.to_act_order != expected_seat
            or observation.street != "T3"
            or raw.get("observation_fingerprint") != observation.fingerprint()
        ):
            raise ValueError("candidate02 root observation changed")
        v1._reject_hidden(raw["observation"], "candidate02_root.observation")
        parsed.append(observation)
    return parsed[0], parsed[1]


def _validate_candidate02_performance_lock_root_artifact(
    value: Mapping[str, Any], *, expected_row: Mapping[str, Any]
) -> tuple[Any, Any]:
    if (
        set(value) != v1._ROOT_KEYS
        or value.get("schema") != CANDIDATE02_PERFORMANCE_LOCK_ROOT_SCHEMA
    ):
        raise ValueError("candidate02 performance-lock root schema changed")
    observations = value.get("observations")
    if (
        value.get("contract_canonical_sha256")
        != _candidate02_performance_lock_contract_anchor_sha256()
        or value.get("schedule") != CANDIDATE02_PERFORMANCE_LOCK_SCHEDULE
        or value.get("schedule_row_sha256") != canonical_sha256(expected_row)
        or value.get("hand_index") != expected_row["hand_index"]
        or value.get("root_indices") != expected_row["root_indices"]
        or value.get("profile") != expected_row["profile"]
        or value.get("seeds") != expected_row["seeds"]
        or value.get("budget") != v1.PERFORMANCE_BUDGET
        or value.get("current_profile_resolved") is not False
        or value.get("opponent_private_discards_used") is not False
        or value.get("training_eligible") is not False
        or not isinstance(observations, list)
        or len(observations) != 2
    ):
        raise ValueError("candidate02 performance-lock root provenance changed")
    parsed = []
    for offset, raw in enumerate(observations):
        if not isinstance(raw, Mapping) or set(raw) != v1._ROOT_OBSERVATION_KEYS:
            raise ValueError("candidate02 performance-lock root observation changed")
        observation = v1.ActorObservation.from_dict(raw["observation"])
        expected_seat = "first" if offset == 0 else "second"
        if (
            raw.get("root_index") != expected_row["root_indices"][offset]
            or raw.get("seat") != expected_seat
            or observation.seat != expected_seat
            or observation.to_act_order != expected_seat
            or observation.street != "T3"
            or raw.get("observation_fingerprint") != observation.fingerprint()
        ):
            raise ValueError("candidate02 performance-lock root observation changed")
        v1._reject_hidden(
            raw["observation"], "candidate02_performance_lock_root.observation"
        )
        parsed.append(observation)
    return parsed[0], parsed[1]


def _validate_candidate02_performance_lock_recovery_root_artifact(
    value: Mapping[str, Any],
    *,
    expected_row: Mapping[str, Any],
    recovery_v3: bool = False,
) -> tuple[Any, Any]:
    expected_schema = (
        CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_ROOT_SCHEMA
        if recovery_v3
        else CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_ROOT_SCHEMA
    )
    expected_anchor = (
        _candidate02_performance_lock_recovery_v3_contract_anchor_sha256()
        if recovery_v3
        else _candidate02_performance_lock_recovery_contract_anchor_sha256()
    )
    expected_schedule = (
        CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_SCHEDULE
        if recovery_v3
        else CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SCHEDULE
    )
    if (
        set(value) != v1._ROOT_KEYS
        or value.get("schema") != expected_schema
    ):
        raise ValueError("candidate02 performance-lock recovery root schema changed")
    observations = value.get("observations")
    if (
        value.get("contract_canonical_sha256")
        != expected_anchor
        or value.get("schedule") != expected_schedule
        or value.get("schedule_row_sha256") != canonical_sha256(expected_row)
        or value.get("hand_index") != expected_row["hand_index"]
        or value.get("root_indices") != expected_row["root_indices"]
        or value.get("profile") != expected_row["profile"]
        or value.get("seeds") != expected_row["seeds"]
        or value.get("budget") != v1.PERFORMANCE_BUDGET
        or value.get("current_profile_resolved") is not False
        or value.get("opponent_private_discards_used") is not False
        or value.get("training_eligible") is not False
        or not isinstance(observations, list)
        or len(observations) != 2
    ):
        raise ValueError(
            "candidate02 performance-lock recovery root provenance changed"
        )
    parsed = []
    for offset, raw in enumerate(observations):
        if not isinstance(raw, Mapping) or set(raw) != v1._ROOT_OBSERVATION_KEYS:
            raise ValueError(
                "candidate02 performance-lock recovery root observation changed"
            )
        observation = v1.ActorObservation.from_dict(raw["observation"])
        expected_seat = "first" if offset == 0 else "second"
        if (
            raw.get("root_index") != expected_row["root_indices"][offset]
            or raw.get("seat") != expected_seat
            or observation.seat != expected_seat
            or observation.to_act_order != expected_seat
            or observation.street != "T3"
            or raw.get("observation_fingerprint") != observation.fingerprint()
        ):
            raise ValueError(
                "candidate02 performance-lock recovery root observation changed"
            )
        v1._reject_hidden(
            raw["observation"],
            "candidate02_performance_lock_recovery_root.observation",
        )
        parsed.append(observation)
    return parsed[0], parsed[1]


def _validate_candidate02_performance_lock_v4_root_artifact(
    value: Mapping[str, Any], *, expected_row: Mapping[str, Any]
) -> tuple[Any, Any]:
    if (
        set(value) != v1._ROOT_KEYS
        or value.get("schema") != CANDIDATE02_PERFORMANCE_LOCK_V4_ROOT_SCHEMA
    ):
        raise ValueError("candidate02 performance-lock-v4 root schema changed")
    observations = value.get("observations")
    if (
        value.get("contract_canonical_sha256")
        != _candidate02_performance_lock_v4_contract_anchor_sha256()
        or value.get("schedule") != CANDIDATE02_PERFORMANCE_LOCK_V4_SCHEDULE
        or value.get("schedule_row_sha256") != canonical_sha256(expected_row)
        or value.get("hand_index") != expected_row["hand_index"]
        or value.get("root_indices") != expected_row["root_indices"]
        or value.get("profile") != expected_row["profile"]
        or value.get("seeds") != expected_row["seeds"]
        or value.get("budget") != v1.PERFORMANCE_BUDGET
        or value.get("current_profile_resolved") is not False
        or value.get("opponent_private_discards_used") is not False
        or value.get("training_eligible") is not False
        or not isinstance(observations, list)
        or len(observations) != 2
    ):
        raise ValueError("candidate02 performance-lock-v4 root provenance changed")
    parsed = []
    for offset, raw in enumerate(observations):
        if not isinstance(raw, Mapping) or set(raw) != v1._ROOT_OBSERVATION_KEYS:
            raise ValueError(
                "candidate02 performance-lock-v4 root observation changed"
            )
        observation = v1.ActorObservation.from_dict(raw["observation"])
        expected_seat = "first" if offset == 0 else "second"
        if (
            raw.get("root_index") != expected_row["root_indices"][offset]
            or raw.get("seat") != expected_seat
            or observation.seat != expected_seat
            or observation.to_act_order != expected_seat
            or observation.street != "T3"
            or raw.get("observation_fingerprint") != observation.fingerprint()
        ):
            raise ValueError(
                "candidate02 performance-lock-v4 root observation changed"
            )
        v1._reject_hidden(
            raw["observation"],
            "candidate02_performance_lock_v4_root.observation",
        )
        parsed.append(observation)
    return parsed[0], parsed[1]


def _validate_root_artifact(
    contract: Mapping[str, Any], value: Mapping[str, Any], *, index: int
) -> tuple[Any, Any]:
    expected_row = _schedule_row(contract, index)
    variant = contract_variant(contract)
    if variant == CANDIDATE01_VARIANT:
        return v1._validate_root_artifact(value, expected_row=expected_row)
    if variant == CANDIDATE02_PERFORMANCE_LOCK_VARIANT:
        return _validate_candidate02_performance_lock_root_artifact(
            value, expected_row=expected_row
        )
    if variant == CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT:
        return _validate_candidate02_performance_lock_recovery_root_artifact(
            value, expected_row=expected_row
        )
    if variant == CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_VARIANT:
        return _validate_candidate02_performance_lock_recovery_root_artifact(
            value, expected_row=expected_row, recovery_v3=True
        )
    if variant == CANDIDATE02_PERFORMANCE_LOCK_V4_VARIANT:
        return _validate_candidate02_performance_lock_v4_root_artifact(
            value, expected_row=expected_row
        )
    return _validate_candidate02_root_artifact(value, expected_row=expected_row)


def _materialize_candidate02_roots(
    *, repository_root: Path, output_dir: Path, indices: Sequence[int]
) -> list[dict[str, Any]]:
    root_dir = output_dir / "roots"
    root_dir.mkdir(parents=True, exist_ok=True)
    bundle = None
    roots: list[dict[str, Any]] = []
    model_profiles = set(v1.M31_T3_BEHAVIOR_PROFILES) - {"random_exact_final"}
    for index in indices:
        schedule_row = candidate02_schedule_row(index)
        path = root_dir / f"hand_{index:03d}.json"
        if path.exists():
            value = _read_canonical(path)
            _validate_candidate02_root_artifact(value, expected_row=schedule_row)
            roots.append(value)
            continue
        if bundle is None:
            bundle = v1.load_model_bundle(
                v1._absolute_model_paths(repository_root), profiles=model_profiles
            )
        observations = v1.generate_behavior_t3_roots(
            hand_seed=schedule_row["seeds"]["hand"],
            behavior_seed=schedule_row["seeds"]["behavior"],
            profile=schedule_row["profile"],
            bundle=bundle,
        )
        value = {
            "schema": CANDIDATE02_ROOT_SCHEMA,
            "contract_canonical_sha256": _candidate02_contract_anchor_sha256(),
            "schedule": CANDIDATE02_SCHEDULE,
            "schedule_row_sha256": canonical_sha256(schedule_row),
            "hand_index": index,
            "root_indices": schedule_row["root_indices"],
            "profile": schedule_row["profile"],
            "seeds": schedule_row["seeds"],
            "budget": dict(v1.PERFORMANCE_BUDGET),
            "observations": [
                {
                    "root_index": schedule_row["root_indices"][offset],
                    "seat": observation.seat,
                    "observation_fingerprint": observation.fingerprint(),
                    "observation": observation.to_dict(),
                }
                for offset, observation in enumerate(observations)
            ],
            "current_profile_resolved": False,
            "opponent_private_discards_used": False,
            "training_eligible": False,
        }
        _validate_candidate02_root_artifact(value, expected_row=schedule_row)
        _write_once(path, value)
        roots.append(value)
    return roots


def _materialize_candidate02_performance_lock_roots(
    *, repository_root: Path, output_dir: Path, indices: Sequence[int]
) -> list[dict[str, Any]]:
    root_dir = output_dir / "roots"
    root_dir.mkdir(parents=True, exist_ok=True)
    bundle = None
    roots: list[dict[str, Any]] = []
    model_profiles = set(v1.M31_T3_BEHAVIOR_PROFILES) - {"random_exact_final"}
    for index in indices:
        schedule_row = candidate02_performance_lock_schedule_row(index)
        path = root_dir / f"hand_{index:03d}.json"
        if path.exists():
            value = _read_canonical(path)
            _validate_candidate02_performance_lock_root_artifact(
                value, expected_row=schedule_row
            )
            roots.append(value)
            continue
        if bundle is None:
            bundle = v1.load_model_bundle(
                v1._absolute_model_paths(repository_root), profiles=model_profiles
            )
        observations = v1.generate_behavior_t3_roots(
            hand_seed=schedule_row["seeds"]["hand"],
            behavior_seed=schedule_row["seeds"]["behavior"],
            profile=schedule_row["profile"],
            bundle=bundle,
        )
        value = {
            "schema": CANDIDATE02_PERFORMANCE_LOCK_ROOT_SCHEMA,
            "contract_canonical_sha256": (
                _candidate02_performance_lock_contract_anchor_sha256()
            ),
            "schedule": CANDIDATE02_PERFORMANCE_LOCK_SCHEDULE,
            "schedule_row_sha256": canonical_sha256(schedule_row),
            "hand_index": index,
            "root_indices": schedule_row["root_indices"],
            "profile": schedule_row["profile"],
            "seeds": schedule_row["seeds"],
            "budget": dict(v1.PERFORMANCE_BUDGET),
            "observations": [
                {
                    "root_index": schedule_row["root_indices"][offset],
                    "seat": observation.seat,
                    "observation_fingerprint": observation.fingerprint(),
                    "observation": observation.to_dict(),
                }
                for offset, observation in enumerate(observations)
            ],
            "current_profile_resolved": False,
            "opponent_private_discards_used": False,
            "training_eligible": False,
        }
        _validate_candidate02_performance_lock_root_artifact(
            value, expected_row=schedule_row
        )
        _write_once(path, value)
        roots.append(value)
    return roots


def _materialize_candidate02_performance_lock_recovery_roots(
    *,
    repository_root: Path,
    output_dir: Path,
    indices: Sequence[int],
    recovery_v3: bool = False,
) -> list[dict[str, Any]]:
    root_dir = output_dir / "roots"
    root_dir.mkdir(parents=True, exist_ok=True)
    bundle = None
    roots: list[dict[str, Any]] = []
    model_profiles = set(v1.M31_T3_BEHAVIOR_PROFILES) - {"random_exact_final"}
    for index in indices:
        schedule_row = (
            candidate02_performance_lock_recovery_v3_schedule_row(index)
            if recovery_v3
            else candidate02_performance_lock_recovery_schedule_row(index)
        )
        path = root_dir / f"hand_{index:03d}.json"
        if path.exists():
            value = _read_canonical(path)
            _validate_candidate02_performance_lock_recovery_root_artifact(
                value, expected_row=schedule_row, recovery_v3=recovery_v3
            )
            roots.append(value)
            continue
        if bundle is None:
            bundle = v1.load_model_bundle(
                v1._absolute_model_paths(repository_root), profiles=model_profiles
            )
        observations = v1.generate_behavior_t3_roots(
            hand_seed=schedule_row["seeds"]["hand"],
            behavior_seed=schedule_row["seeds"]["behavior"],
            profile=schedule_row["profile"],
            bundle=bundle,
        )
        value = {
            "schema": (
                CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_ROOT_SCHEMA
                if recovery_v3
                else CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_ROOT_SCHEMA
            ),
            "contract_canonical_sha256": (
                _candidate02_performance_lock_recovery_v3_contract_anchor_sha256()
                if recovery_v3
                else _candidate02_performance_lock_recovery_contract_anchor_sha256()
            ),
            "schedule": (
                CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_SCHEDULE
                if recovery_v3
                else CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SCHEDULE
            ),
            "schedule_row_sha256": canonical_sha256(schedule_row),
            "hand_index": index,
            "root_indices": schedule_row["root_indices"],
            "profile": schedule_row["profile"],
            "seeds": schedule_row["seeds"],
            "budget": dict(v1.PERFORMANCE_BUDGET),
            "observations": [
                {
                    "root_index": schedule_row["root_indices"][offset],
                    "seat": observation.seat,
                    "observation_fingerprint": observation.fingerprint(),
                    "observation": observation.to_dict(),
                }
                for offset, observation in enumerate(observations)
            ],
            "current_profile_resolved": False,
            "opponent_private_discards_used": False,
            "training_eligible": False,
        }
        _validate_candidate02_performance_lock_recovery_root_artifact(
            value, expected_row=schedule_row, recovery_v3=recovery_v3
        )
        _write_once(path, value)
        roots.append(value)
    return roots


def _materialize_candidate02_performance_lock_v4_roots(
    *, repository_root: Path, output_dir: Path, indices: Sequence[int]
) -> list[dict[str, Any]]:
    root_dir = output_dir / "roots"
    root_dir.mkdir(parents=True, exist_ok=True)
    bundle = None
    roots: list[dict[str, Any]] = []
    model_profiles = set(v1.M31_T3_BEHAVIOR_PROFILES) - {"random_exact_final"}
    for index in indices:
        schedule_row = candidate02_performance_lock_v4_schedule_row(index)
        path = root_dir / f"hand_{index:03d}.json"
        if path.exists():
            value = _read_canonical(path)
            _validate_candidate02_performance_lock_v4_root_artifact(
                value, expected_row=schedule_row
            )
            roots.append(value)
            continue
        if bundle is None:
            bundle = v1.load_model_bundle(
                v1._absolute_model_paths(repository_root), profiles=model_profiles
            )
        observations = v1.generate_behavior_t3_roots(
            hand_seed=schedule_row["seeds"]["hand"],
            behavior_seed=schedule_row["seeds"]["behavior"],
            profile=schedule_row["profile"],
            bundle=bundle,
        )
        value = {
            "schema": CANDIDATE02_PERFORMANCE_LOCK_V4_ROOT_SCHEMA,
            "contract_canonical_sha256": (
                _candidate02_performance_lock_v4_contract_anchor_sha256()
            ),
            "schedule": CANDIDATE02_PERFORMANCE_LOCK_V4_SCHEDULE,
            "schedule_row_sha256": canonical_sha256(schedule_row),
            "hand_index": index,
            "root_indices": schedule_row["root_indices"],
            "profile": schedule_row["profile"],
            "seeds": schedule_row["seeds"],
            "budget": dict(v1.PERFORMANCE_BUDGET),
            "observations": [
                {
                    "root_index": schedule_row["root_indices"][offset],
                    "seat": observation.seat,
                    "observation_fingerprint": observation.fingerprint(),
                    "observation": observation.to_dict(),
                }
                for offset, observation in enumerate(observations)
            ],
            "current_profile_resolved": False,
            "opponent_private_discards_used": False,
            "training_eligible": False,
        }
        _validate_candidate02_performance_lock_v4_root_artifact(
            value, expected_row=schedule_row
        )
        _write_once(path, value)
        roots.append(value)
    return roots


def _materialize_roots(
    *,
    contract: Mapping[str, Any],
    repository_root: Path,
    output_dir: Path,
    indices: Sequence[int],
) -> list[dict[str, Any]]:
    variant = contract_variant(contract)
    if variant == CANDIDATE01_VARIANT:
        return v1._materialize_roots(
            repository_root=repository_root,
            output_dir=output_dir,
            indices=indices,
        )
    if variant == CANDIDATE02_PERFORMANCE_LOCK_VARIANT:
        return _materialize_candidate02_performance_lock_roots(
            repository_root=repository_root,
            output_dir=output_dir,
            indices=indices,
        )
    if variant == CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT:
        return _materialize_candidate02_performance_lock_recovery_roots(
            repository_root=repository_root,
            output_dir=output_dir,
            indices=indices,
        )
    if variant == CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_VARIANT:
        return _materialize_candidate02_performance_lock_recovery_roots(
            repository_root=repository_root,
            output_dir=output_dir,
            indices=indices,
            recovery_v3=True,
        )
    if variant == CANDIDATE02_PERFORMANCE_LOCK_V4_VARIANT:
        return _materialize_candidate02_performance_lock_v4_roots(
            repository_root=repository_root,
            output_dir=output_dir,
            indices=indices,
        )
    return _materialize_candidate02_roots(
        repository_root=repository_root,
        output_dir=output_dir,
        indices=indices,
    )


def _expected_library_sha256(contract: Mapping[str, Any], source_role: str) -> str:
    return str(contract[f"{source_role}_library_sha256"])


def _validate_binary(path: Path, expected_sha256: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(f"Step 6d v2 native library is missing: {resolved}")
    observed = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if observed != expected_sha256:
        raise ValueError(
            f"Step 6d v2 native library SHA mismatch: expected {expected_sha256}, got {observed}"
        )
    return resolved


def _memory_report(
    before: Mapping[str, Any], loaded: Mapping[str, Any]
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "before_solver_load": dict(before),
        "after_solver_load": dict(loaded),
        "after_hand": v1.process_memory_snapshot(),
    }
    peaks = [
        snapshot.get("peak_rss_bytes")
        for snapshot in value.values()
        if isinstance(snapshot, Mapping)
        and isinstance(snapshot.get("peak_rss_bytes"), int)
        and not isinstance(snapshot.get("peak_rss_bytes"), bool)
    ]
    value["peak_rss_bytes"] = max(peaks) if peaks else None
    return value


def _source_geometry(observation: Any, portable: Mapping[str, Any]) -> dict[str, int]:
    action_values = portable.get("action_values")
    if not isinstance(action_values, list) or not action_values:
        raise ValueError("Step 6d v2 portable action values are missing")
    hero_count = len(
        v1.generate_turn_actions(observation.hero_board, observation.dealt_cards)
    )
    if hero_count != len(action_values):
        raise ValueError("Step 6d v2 hero action/Q geometry changed")
    # At a first-seat root, opponent response cardinality depends only on the
    # public row capacities.  The observed three-card tuple is used solely as a
    # safe placeholder to enumerate that placement geometry.
    response_count = (
        len(
            v1.generate_turn_actions(
                observation.opponent_public_board, observation.dealt_cards
            )
        )
        if observation.seat == "first"
        else 0
    )
    rows = hero_count if observation.seat == "first" else 0
    columns = response_count if observation.seat == "first" else 0
    return {
        "hero_legal_action_count": hero_count,
        "opponent_response_legal_action_count": response_count,
        "first_seat_action_matrix_rows": rows,
        "first_seat_action_matrix_columns": columns,
        "first_seat_action_matrix_cells": rows * columns,
        "child_information_set_count": int(portable["child_information_set_count"]),
        "candidate_q_count": len(action_values),
        "evaluation_q_count": len(action_values),
    }


def _solver_for_contract(
    *,
    contract: Mapping[str, Any],
    library_path: Path,
    library_sha256: str,
    seeds: Mapping[str, int],
) -> Any:
    if contract_variant(contract) == CANDIDATE01_VARIANT:
        return v1._solver(
            library_path=library_path,
            library_sha256=library_sha256,
            seeds=seeds,
        )
    return v1.HuM31T3SearchSolver(
        v1.HuM31T3RuntimeConfig(
            expected_library_sha256=library_sha256,
            library_path=library_path,
            run_id=_run_id(contract),
            candidate_samples=8,
            evaluation_samples=32,
            downstream_t3_samples=4,
            seed=seeds["child"],
            candidate_seed=seeds["candidate"],
            evaluation_seed=seeds["evaluation"],
        )
    )


def _validate_candidate02_decision(
    decision: Mapping[str, Any],
    *,
    observation: Any,
    seeds: Mapping[str, int],
    library_sha256: str,
    run_id: str = CANDIDATE02_RUN_ID,
) -> None:
    if decision.get("run_id") != run_id:
        raise ValueError("candidate02 runtime decision run_id changed")
    # Reuse the frozen candidate01 structural/action/Q/RNG validator after
    # normalizing only the deliberately fresh run identifier.  The runtime
    # object has already validated its result certificates against the original
    # candidate02 decision.
    normalized = dict(decision)
    normalized["run_id"] = v1.STEP6D_RUN_ID
    v1._validate_decision(
        normalized,
        observation=observation,
        seeds=seeds,
        library_sha256=library_sha256,
    )


def _solve_source_for_contract(
    *,
    contract: Mapping[str, Any],
    source: str,
    solver: Any,
    observation: Any,
    seeds: Mapping[str, int],
) -> dict[str, Any]:
    if contract_variant(contract) == CANDIDATE01_VARIANT:
        return v1._solve_source(
            source=source,
            solver=solver,
            observation=observation,
            seeds=seeds,
        )
    started = time.perf_counter()
    decision = solver.solve(observation).to_dict()
    wall_seconds = time.perf_counter() - started
    _validate_candidate02_decision(
        decision,
        observation=observation,
        seeds=seeds,
        library_sha256=solver.library_sha256,
        run_id=_run_id(contract),
    )
    return {
        "source": source,
        "native_library_sha256": solver.library_sha256,
        "solve_wall_seconds": wall_seconds,
        "native_seconds": float(decision["native_latency_ms"]) / 1000.0,
        "validation_seconds": float(decision["validation_latency_ms"]) / 1000.0,
        "runtime_total_seconds": float(decision["total_latency_ms"]) / 1000.0,
        "rss_after": v1.process_memory_snapshot(),
        "decision": decision,
    }


def _validate_source_result_for_contract(
    value: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    source: str,
    observation: Any,
    seeds: Mapping[str, int],
    library_sha256: str,
) -> None:
    if contract_variant(contract) == CANDIDATE01_VARIANT:
        v1._validate_source_result(
            value,
            source=source,
            observation=observation,
            seeds=seeds,
            library_sha256=library_sha256,
        )
        return
    decision = value.get("decision")
    if not isinstance(decision, Mapping):
        raise ValueError("candidate02 source result decision is missing")
    _validate_candidate02_decision(
        decision,
        observation=observation,
        seeds=seeds,
        library_sha256=library_sha256,
        run_id=_run_id(contract),
    )
    normalized = dict(value)
    normalized_decision = dict(decision)
    normalized_decision["run_id"] = v1.STEP6D_RUN_ID
    normalized["decision"] = normalized_decision
    v1._validate_source_result(
        normalized,
        source=source,
        observation=observation,
        seeds=seeds,
        library_sha256=library_sha256,
    )


def _run_source_hand(
    *,
    root: Mapping[str, Any],
    run_contract: Mapping[str, Any],
    source_role: str,
    library_path: Path,
    library_sha256: str,
    run_contract_digest: str,
    shard_manifest_sha256: str,
    reference_library_sha256: str,
    candidate_library_sha256: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    contract = validate_run_contract(run_contract)
    hand_index = int(root["hand_index"])
    expected_row = _schedule_row(contract, hand_index)
    observations = _validate_root_artifact(contract, root, index=hand_index)
    before = v1.process_memory_snapshot()
    solver = _solver_for_contract(
        contract=contract,
        library_path=library_path,
        library_sha256=library_sha256,
        seeds=expected_row["seeds"],
    )
    loaded = v1.process_memory_snapshot()
    rows: list[dict[str, Any]] = []
    for offset, observation in enumerate(observations):
        result = _solve_source_for_contract(
            contract=contract,
            source=source_role,
            solver=solver,
            observation=observation,
            seeds=expected_row["seeds"],
        )
        portable = v1.portable_parity_payload(result["decision"])
        rows.append(
            {
                "root_index": expected_row["root_indices"][offset],
                "seat": observation.seat,
                "observation_fingerprint": observation.fingerprint(),
                "observation_sha256": canonical_sha256(observation.to_dict()),
                "geometry": _source_geometry(observation, portable),
                "source_result": result,
                "portable_decision": portable,
                "portable_decision_sha256": canonical_sha256(portable),
                "wall_seconds": result["solve_wall_seconds"],
            }
        )
    value = {
        "schema": _source_hand_schema(contract),
        "contract_canonical_sha256": contract["contract_canonical_sha256"],
        "schedule": contract["schedule"],
        "run_contract_digest": run_contract_digest,
        "shard_manifest_sha256": shard_manifest_sha256,
        "source_role": source_role,
        "hand_index": hand_index,
        "root_indices": expected_row["root_indices"],
        "schedule_row_sha256": canonical_sha256(expected_row),
        "profile": expected_row["profile"],
        "seeds": expected_row["seeds"],
        "budget": dict(contract["budget"]),
        "root_artifact_sha256": canonical_sha256(root),
        "allocation": dict(ALLOCATION),
        "reference_library_sha256": reference_library_sha256,
        "candidate_library_sha256": candidate_library_sha256,
        "native_library_sha256": library_sha256,
        "engine_version": solver.engine_version,
        "queue_seconds": 0.0,
        "worker_wall_seconds": time.perf_counter() - started,
        "process_id": os.getpid(),
        "memory": _memory_report(before, loaded),
        "rows": rows,
        "teacher_value_status": "diagnostic_not_match_EV",
        "training_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "cloud_started": False,
    }
    return _validate_source_hand(
        value,
        root=root,
        run_contract=contract,
        source_role=source_role,
        library_sha256=library_sha256,
        run_contract_digest=run_contract_digest,
        shard_manifest_sha256=shard_manifest_sha256,
        reference_library_sha256=reference_library_sha256,
        candidate_library_sha256=candidate_library_sha256,
    )


def _validate_source_hand(
    value: Mapping[str, Any],
    *,
    root: Mapping[str, Any],
    source_role: str,
    library_sha256: str,
    run_contract_digest: str,
    shard_manifest_sha256: str,
    reference_library_sha256: str,
    candidate_library_sha256: str,
    run_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(value)
    _require_exact_keys(payload, _SOURCE_HAND_KEYS, "Step 6d v2 source hand")
    contract = validate_run_contract(
        run_contract
        if run_contract is not None
        else build_run_contract(
            candidate_library_sha256=candidate_library_sha256,
            reference_library_sha256=reference_library_sha256,
        )
    )
    hand_index = int(root["hand_index"])
    expected_row = _schedule_row(contract, hand_index)
    observations = _validate_root_artifact(contract, root, index=hand_index)
    rows = payload.get("rows")
    if (
        payload.get("schema") != _source_hand_schema(contract)
        or payload.get("contract_canonical_sha256")
        != contract["contract_canonical_sha256"]
        or payload.get("schedule") != contract["schedule"]
        or payload.get("run_contract_digest") != run_contract_digest
        or run_contract_digest != canonical_sha256(contract)
        or payload.get("shard_manifest_sha256") != shard_manifest_sha256
        or payload.get("source_role") != source_role
        or payload.get("hand_index") != expected_row["hand_index"]
        or payload.get("root_indices") != expected_row["root_indices"]
        or payload.get("schedule_row_sha256") != canonical_sha256(expected_row)
        or payload.get("profile") != expected_row["profile"]
        or payload.get("seeds") != expected_row["seeds"]
        or payload.get("budget") != contract["budget"]
        or payload.get("root_artifact_sha256") != canonical_sha256(root)
        or payload.get("allocation") != ALLOCATION
        or payload.get("reference_library_sha256") != reference_library_sha256
        or payload.get("candidate_library_sha256") != candidate_library_sha256
        or reference_library_sha256 == candidate_library_sha256
        or payload.get("native_library_sha256") != library_sha256
        or not isinstance(payload.get("engine_version"), str)
        or not payload["engine_version"]
        or payload.get("teacher_value_status") != "diagnostic_not_match_EV"
        or not isinstance(rows, list)
        or len(rows) != 2
        or not isinstance(payload.get("memory"), Mapping)
        or any(
            payload.get(field) is not False
            for field in (
                "training_eligible",
                "quality_evidence",
                "promotion_evidence",
                "current_profile_changed",
                "named_profile_added",
                "runtime_policy_activated",
                "cloud_started",
            )
        )
    ):
        raise ValueError("Step 6d v2 source-hand provenance changed")
    v1._integer(payload.get("process_id"), "process id", minimum=1)
    v1._finite(payload.get("queue_seconds"), "queue seconds", minimum=0.0)
    v1._finite(payload.get("worker_wall_seconds"), "worker wall seconds", minimum=0.0)
    v1._validate_memory(payload["memory"])
    for offset, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError("Step 6d v2 source row must be an object")
        _require_exact_keys(raw, _SOURCE_ROW_KEYS, "Step 6d v2 source row")
        observation = observations[offset]
        result = raw.get("source_result")
        portable = raw.get("portable_decision")
        if (
            raw.get("root_index") != expected_row["root_indices"][offset]
            or raw.get("seat") != observation.seat
            or raw.get("observation_fingerprint") != observation.fingerprint()
            or raw.get("observation_sha256") != canonical_sha256(observation.to_dict())
            or not isinstance(result, Mapping)
            or not isinstance(portable, Mapping)
        ):
            raise ValueError("Step 6d v2 source row provenance changed")
        _validate_source_result_for_contract(
            result,
            contract=contract,
            source=source_role,
            observation=observation,
            seeds=expected_row["seeds"],
            library_sha256=library_sha256,
        )
        expected_portable = v1.portable_parity_payload(result["decision"])
        if (
            portable != expected_portable
            or raw.get("portable_decision_sha256")
            != canonical_sha256(expected_portable)
            or raw.get("geometry") != _source_geometry(observation, expected_portable)
            or raw.get("wall_seconds") != result.get("solve_wall_seconds")
        ):
            raise ValueError("Step 6d v2 source row portable/geometry evidence changed")
    v1._reject_hidden(payload, "source_hand")
    return payload


def _file_record(
    path: Path,
    *,
    relative_to: Path,
    source_role: str,
    hand_index: int,
) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "source_role": source_role,
        "hand_index": hand_index,
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _artifact_records(
    output_dir: Path, source_role: str, work_hand_indices: Sequence[int]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index in work_hand_indices:
        root_path = output_dir / "roots" / f"hand_{index:03d}.json"
        hand_path = output_dir / "hands" / source_role / f"hand_{index:03d}.json"
        if not root_path.is_file() or not hand_path.is_file():
            raise ValueError("Step 6d v2 shard artifact set is incomplete")
        records.append(
            _file_record(
                root_path,
                relative_to=output_dir,
                source_role=source_role,
                hand_index=index,
            )
        )
        records.append(
            _file_record(
                hand_path,
                relative_to=output_dir,
                source_role=source_role,
                hand_index=index,
            )
        )
    return records


def _build_done(
    *, output_dir: Path, shard_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    role = str(shard_manifest["source_role"])
    work = list(shard_manifest["work_hand_indices"])
    contract = shard_manifest["run_contract"]
    artifacts = _artifact_records(output_dir, role, work)
    native_sha256 = str(contract[f"{role}_library_sha256"])
    return {
        "schema": _done_schema(contract),
        "status": "complete_source_isolated_shard",
        "contract_canonical_sha256": contract["contract_canonical_sha256"],
        "run_contract_digest": shard_manifest["run_contract_digest"],
        "shard_manifest_sha256": canonical_sha256(shard_manifest),
        "source_role": role,
        "contract_hand_indices": contract["contract_hand_indices"],
        "tail_hand_indices": contract["tail_hand_indices"],
        "work_hand_indices": work,
        "completed_hand_indices": work,
        "budget": contract["budget"],
        "allocation": contract["allocation"],
        "reference_library_sha256": contract["reference_library_sha256"],
        "candidate_library_sha256": contract["candidate_library_sha256"],
        "native_library_sha256": native_sha256,
        "artifact_count": len(artifacts),
        "artifact_manifest": artifacts,
        "artifact_manifest_sha256": canonical_sha256(artifacts),
        "training_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }


def validate_done(
    value: Mapping[str, Any],
    *,
    output_dir: Path,
    shard_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(value)
    _require_exact_keys(payload, _DONE_KEYS, "Step 6d v2 DONE")
    artifacts = payload.get("artifact_manifest")
    if not isinstance(artifacts, list):
        raise ValueError("Step 6d v2 DONE artifact manifest is missing")
    for record in artifacts:
        if not isinstance(record, Mapping):
            raise ValueError("Step 6d v2 DONE artifact record changed")
        _require_exact_keys(record, _ARTIFACT_RECORD_KEYS, "Step 6d v2 artifact record")
    expected = _build_done(output_dir=output_dir, shard_manifest=shard_manifest)
    if payload != expected:
        raise ValueError("Step 6d v2 DONE or bound artifacts changed")
    return payload


def run_source_shard(
    *,
    repository_root: str | Path,
    output_dir: str | Path,
    shard_manifest: Mapping[str, Any],
    library_path: str | Path,
    stop_after_hands: int | None = None,
) -> dict[str, Any]:
    """Run/resume one source-isolated shard and publish ``DONE`` last."""

    root = Path(repository_root).resolve()
    destination = Path(output_dir).resolve()
    manifest = validate_shard_manifest(shard_manifest)
    contract = manifest["run_contract"]
    role = str(manifest["source_role"])
    work = tuple(manifest["work_hand_indices"])
    shard_manifest_sha256 = canonical_sha256(manifest)
    reference_sha256 = str(contract["reference_library_sha256"])
    candidate_sha256 = str(contract["candidate_library_sha256"])
    if stop_after_hands is not None:
        v1._integer(stop_after_hands, "stop_after_hands", minimum=1)
    v1.validate_performance_contract(
        root / "configs" / "hu_joint_policy_m31_t3_step6d_contract.json"
    )
    expected_sha = _expected_library_sha256(contract, role)
    native = _validate_binary(Path(library_path), expected_sha)
    os.environ["RAYON_NUM_THREADS"] = str(ALLOCATION["rayon_threads_per_worker"])
    os.environ["OFC_HU_M3_BATCH_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    destination.mkdir(parents=True, exist_ok=True)
    _write_or_validate(destination / "run_contract.json", contract, "run contract")
    _write_or_validate(destination / "shard_manifest.json", manifest, "shard manifest")

    roots = _materialize_roots(
        contract=contract,
        repository_root=root,
        output_dir=destination,
        indices=work,
    )
    root_by_index = {int(value["hand_index"]): value for value in roots}
    hand_dir = destination / "hands" / role
    hand_dir.mkdir(parents=True, exist_ok=True)
    existing: dict[int, dict[str, Any]] = {}
    missing: list[int] = []
    for index in work:
        path = hand_dir / f"hand_{index:03d}.json"
        if path.exists():
            existing[index] = _validate_source_hand(
                _read_canonical(path),
                root=root_by_index[index],
                run_contract=contract,
                source_role=role,
                library_sha256=expected_sha,
                run_contract_digest=manifest["run_contract_digest"],
                shard_manifest_sha256=shard_manifest_sha256,
                reference_library_sha256=reference_sha256,
                candidate_library_sha256=candidate_sha256,
            )
        else:
            missing.append(index)

    selected = missing[:stop_after_hands] if stop_after_hands else missing
    for index in selected:
        value = _run_source_hand(
            root=root_by_index[index],
            run_contract=contract,
            source_role=role,
            library_path=native,
            library_sha256=expected_sha,
            run_contract_digest=manifest["run_contract_digest"],
            shard_manifest_sha256=shard_manifest_sha256,
            reference_library_sha256=reference_sha256,
            candidate_library_sha256=candidate_sha256,
        )
        path = hand_dir / f"hand_{index:03d}.json"
        _write_once(path, value)
        existing[index] = value

    pending = [index for index in work if index not in existing]
    if pending:
        if (destination / "DONE.json").exists():
            raise ValueError("Step 6d v2 DONE exists for an incomplete shard")
        return {
            "schema": _done_schema(contract),
            "status": "interrupted_for_resume",
            "run_contract_digest": manifest["run_contract_digest"],
            "source_role": role,
            "work_hand_indices": list(work),
            "completed_hand_count": len(existing),
            "pending_hand_indices": pending,
            "training_eligible": False,
            "current_profile_changed": False,
        }

    done_path = destination / "DONE.json"
    if done_path.exists():
        return validate_done(
            _read_canonical(done_path),
            output_dir=destination,
            shard_manifest=manifest,
        )
    done = _build_done(output_dir=destination, shard_manifest=manifest)
    _write_once(done_path, done)
    return validate_done(done, output_dir=destination, shard_manifest=manifest)


def validate_completed_output(output_dir: str | Path) -> dict[str, Any]:
    """Independently revalidate a completed source directory for receive/merge."""

    destination = Path(output_dir).resolve()
    contract_path = destination / "run_contract.json"
    manifest_path = destination / "shard_manifest.json"
    done_path = destination / "DONE.json"
    if any(not path.is_file() for path in (contract_path, manifest_path, done_path)):
        raise ValueError(
            "Step 6d v2 completed output is missing contract/manifest/DONE"
        )
    contract = validate_run_contract(_read_canonical(contract_path))
    manifest = validate_shard_manifest(_read_canonical(manifest_path))
    if manifest["run_contract"] != contract:
        raise ValueError("Step 6d v2 completed output contract/manifest mismatch")
    role = str(manifest["source_role"])
    contract_digest = str(manifest["run_contract_digest"])
    manifest_digest = canonical_sha256(manifest)
    reference_sha256 = str(contract["reference_library_sha256"])
    candidate_sha256 = str(contract["candidate_library_sha256"])
    native_sha256 = str(contract[f"{role}_library_sha256"])
    for index in manifest["work_hand_indices"]:
        root_path = destination / "roots" / f"hand_{index:03d}.json"
        hand_path = destination / "hands" / role / f"hand_{index:03d}.json"
        if not root_path.is_file() or not hand_path.is_file():
            raise ValueError("Step 6d v2 completed output is missing a root/hand")
        root = v1._read_json(root_path)
        _validate_root_artifact(contract, root, index=index)
        _validate_source_hand(
            _read_canonical(hand_path),
            root=root,
            run_contract=contract,
            source_role=role,
            library_sha256=native_sha256,
            run_contract_digest=contract_digest,
            shard_manifest_sha256=manifest_digest,
            reference_library_sha256=reference_sha256,
            candidate_library_sha256=candidate_sha256,
        )
    return validate_done(
        _read_canonical(done_path),
        output_dir=destination,
        shard_manifest=manifest,
    )


def _parse_indices(value: str) -> tuple[int, ...]:
    try:
        items = tuple(int(token.strip()) for token in value.split(",") if token.strip())
    except ValueError as exc:
        raise ValueError(
            "--work-hand-indices must be comma-separated integers"
        ) from exc
    return _normalize_indices(items, label="work_hand_indices")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-manifest", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--stop-after-hands", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = _read_canonical(args.shard_manifest)
    report = run_source_shard(
        repository_root=args.repository_root,
        output_dir=args.output_dir,
        shard_manifest=manifest,
        library_path=args.library,
        stop_after_hands=args.stop_after_hands,
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ALLOCATION",
    "CANDIDATE01_VARIANT",
    "CANDIDATE02_DONE_SCHEMA",
    "CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_CANDIDATE_LIBRARY_SHA256",
    "CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_REFERENCE_LIBRARY_SHA256",
    "CANDIDATE02_PERFORMANCE_LOCK_DONE_SCHEMA",
    "CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_DONE_SCHEMA",
    "CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_NAMESPACE_BASES",
    "CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_ROLE",
    "CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_ROOT_SCHEMA",
    "CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_RUN_CONTRACT_SCHEMA",
    "CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_RUN_ID",
    "CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SCHEDULE",
    "CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SEED_SET_SHA256",
    "CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SHARD_MANIFEST_SCHEMA",
    "CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SOURCE_HAND_SCHEMA",
    "CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT",
    "CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_DONE_SCHEMA",
    "CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_NAMESPACE_BASES",
    "CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_ROLE",
    "CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_ROOT_SCHEMA",
    "CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_RUN_CONTRACT_SCHEMA",
    "CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_RUN_ID",
    "CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_SCHEDULE",
    "CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_SEED_SET_SHA256",
    "CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_SHARD_MANIFEST_SCHEMA",
    "CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_SOURCE_HAND_SCHEMA",
    "CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_VARIANT",
    "CANDIDATE02_PERFORMANCE_LOCK_V4_DONE_SCHEMA",
    "CANDIDATE02_PERFORMANCE_LOCK_V4_GATE_RECEIPT_FILE_SHA256",
    "CANDIDATE02_PERFORMANCE_LOCK_V4_GATE_RECEIPT_SHA256",
    "CANDIDATE02_PERFORMANCE_LOCK_V4_NAMESPACE_BASES",
    "CANDIDATE02_PERFORMANCE_LOCK_V4_ROLE",
    "CANDIDATE02_PERFORMANCE_LOCK_V4_ROOT_SCHEMA",
    "CANDIDATE02_PERFORMANCE_LOCK_V4_RUN_CONTRACT_SCHEMA",
    "CANDIDATE02_PERFORMANCE_LOCK_V4_RUN_ID",
    "CANDIDATE02_PERFORMANCE_LOCK_V4_SCHEDULE",
    "CANDIDATE02_PERFORMANCE_LOCK_V4_SEED_SET_SHA256",
    "CANDIDATE02_PERFORMANCE_LOCK_V4_SHARD_MANIFEST_SCHEMA",
    "CANDIDATE02_PERFORMANCE_LOCK_V4_SOURCE_HAND_SCHEMA",
    "CANDIDATE02_PERFORMANCE_LOCK_V4_VARIANT",
    "CANDIDATE02_PERFORMANCE_LOCK_ROOT_SCHEMA",
    "CANDIDATE02_PERFORMANCE_LOCK_RUN_CONTRACT_SCHEMA",
    "CANDIDATE02_PERFORMANCE_LOCK_RUN_ID",
    "CANDIDATE02_PERFORMANCE_LOCK_SCHEDULE",
    "CANDIDATE02_PERFORMANCE_LOCK_SEED_SET_SHA256",
    "CANDIDATE02_PERFORMANCE_LOCK_SOURCE_HAND_SCHEMA",
    "CANDIDATE02_PERFORMANCE_LOCK_VARIANT",
    "CANDIDATE02_RUN_CONTRACT_SCHEMA",
    "CANDIDATE02_SOURCE_HAND_SCHEMA",
    "CANDIDATE02_TAIL_V2_DONE_SCHEMA",
    "CANDIDATE02_TAIL_V2_HEAVY_HAND_INDICES",
    "CANDIDATE02_TAIL_V2_PRIOR_EXPOSED_HAND_INDICES",
    "CANDIDATE02_TAIL_V2_RANDOM_HAND_INDICES",
    "CANDIDATE02_TAIL_V2_RUN_CONTRACT_SCHEMA",
    "CANDIDATE02_TAIL_V2_RUN_ID",
    "CANDIDATE02_TAIL_V2_SELECTION_MANIFEST_SHA256",
    "CANDIDATE02_TAIL_V2_SOURCE_HAND_SCHEMA",
    "CANDIDATE02_TAIL_V2_TAIL_HAND_INDICES",
    "CANDIDATE02_TAIL_V2_VARIANT",
    "CANDIDATE02_VARIANT",
    "CONTRACT_HAND_INDICES",
    "DONE_SCHEMA",
    "RUN_CONTRACT_SCHEMA",
    "SHARD_MANIFEST_SCHEMA",
    "SOURCE_HAND_SCHEMA",
    "SOURCE_ROLES",
    "TAIL_HAND_INDICES",
    "build_run_contract",
    "build_shard_manifest",
    "candidate02_performance_lock_recovery_schedule_row",
    "candidate02_performance_lock_recovery_seed_contract",
    "candidate02_performance_lock_recovery_seed_values",
    "candidate02_performance_lock_recovery_v3_schedule_row",
    "candidate02_performance_lock_recovery_v3_seed_contract",
    "candidate02_performance_lock_recovery_v3_seed_values",
    "candidate02_performance_lock_v4_schedule_row",
    "candidate02_performance_lock_v4_seed_contract",
    "candidate02_performance_lock_v4_seed_values",
    "candidate02_performance_lock_schedule_row",
    "candidate02_performance_lock_seed_contract",
    "candidate02_performance_lock_seed_values",
    "canonical_bytes",
    "canonical_sha256",
    "main",
    "run_source_shard",
    "validate_done",
    "validate_completed_output",
    "validate_run_contract",
    "validate_shard_manifest",
]
