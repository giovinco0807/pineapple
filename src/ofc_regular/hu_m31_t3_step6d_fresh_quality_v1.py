"""Immutable M3.1 T3 fresh-quality plan, roots, seal, and package.

This module deliberately stops before cloud execution.  It freezes two
independent experiments:

* 50 fresh paired hands (100 roots) at the accepted 8/32/4/exact-T4 budget.
* 5 different paired hands (10 roots) at 8/32/4 plus an independent
  8/128/4/exact-T4 locked confirmation evaluation.

The old Step 6c search/decision validator remains the semantic oracle, but its
old root grid is not reused: Step 6c embedded confirmation inside its 50-hand
pilot.  Here the two root populations and every seed namespace are disjoint.

There are no cloud clients, training calls, profile resolution, or activation
side effects in this file.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .hu_infoset import ActorObservation
from .hu_m31_t3_behavior_roots import (
    M31_T3_BEHAVIOR_PROFILES,
    generate_behavior_t3_roots,
)
from .hu_m31_t3_step6d_performance_lock_v4_production_bridge import (
    QUALIFIED_DECISION,
    RECEIPT_SCHEMA as PERFORMANCE_RECEIPT_SCHEMA,
)
from . import hu_m31_t3_performance_lock_v4_portable_receipt_v1 as portable_receipt
from . import hu_m31_t3_step6c_contract as legacy_step6c
from . import hu_turn3_stage3_feature_rust as stage3_feature_rust
from . import run_hu_m31_t3_step6d_performance as step6d_v1
from . import run_hu_m31_t3_step6d_performance_v2 as step6d_v2


PLAN_SCHEMA = "hu_m31_t3_step6d_fresh_quality_plan_v1"
ROOT_SCHEMA = "hu_m31_t3_step6d_fresh_quality_root_v1"
MATERIALIZATION_SCHEMA = "hu_m31_t3_step6d_fresh_quality_materialization_v1"
ROOT_SEAL_SCHEMA = "hu_m31_t3_step6d_fresh_quality_root_seal_v1"
JOB_SCHEMA = "hu_m31_t3_step6d_fresh_quality_job_v1"
PACKAGE_SCHEMA = "hu_m31_t3_step6d_fresh_quality_package_v1"
PACKAGE_READY_SCHEMA = "hu_m31_t3_step6d_fresh_quality_package_ready_v1"
RUN_ID = "hu_m31_t3_step6d_fresh_quality_v1"
ENGINE_RUN_ID = "hu-m31-step6c-production-label-pilot-v1"

PRIMARY_PHASE = "primary"
CONFIRMATION_PHASE = "confirmation"
PHASES = (PRIMARY_PHASE, CONFIRMATION_PHASE)

PRIMARY_PAIR_INDICES = tuple(range(50))
CONFIRMATION_PAIR_INDICES = tuple(range(5))
PRIMARY_ROOT_INDICES = tuple(range(100))
CONFIRMATION_ROOT_INDICES = tuple(range(100, 110))

PRIMARY_BUDGET = {
    "candidate_samples": 8,
    "evaluation_samples": 32,
    "downstream_t3_samples": 4,
    "downstream_t4_samples": 0,
}
CONFIRMATION_BUDGET = {
    "candidate_samples": 8,
    "evaluation_samples": 128,
    "downstream_t3_samples": 4,
    "downstream_t4_samples": 0,
}

SEED_STRIDE = 1_000_003
PRIMARY_NAMESPACE_BASES = {
    "hand": 800_108_071_901,
    "behavior": 801_108_071_901,
    "candidate": 802_108_071_901,
    "evaluation": 803_108_071_901,
    "child": 804_108_071_901,
    "confirmation": 805_108_071_901,
}
CONFIRMATION_NAMESPACE_BASES = {
    "hand": 810_108_071_901,
    "behavior": 811_108_071_901,
    "candidate": 812_108_071_901,
    "evaluation": 813_108_071_901,
    "child": 814_108_071_901,
    "confirmation": 815_108_071_901,
}

ACCEPTED_CANDIDATE_LIBRARY_SHA256 = (
    step6d_v2.CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_CANDIDATE_LIBRARY_SHA256
)
CURRENT_PROFILE_REGISTRY_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)
ACCEPTED_FEATURE_ENCODER_SHA256 = (
    "82510e563ee290cd536e7aebe6bcf0efefac061af5488851f0a582bb4ef83411"
)
PERFORMANCE_V4_SEED_SET_SHA256 = (
    step6d_v2.CANDIDATE02_PERFORMANCE_LOCK_V4_SEED_SET_SHA256
)

_PLAN_KEYS = frozenset(
    {
        "schema",
        "run_id",
        "engine_run_id",
        "purpose",
        "authorizing_performance_receipt",
        "candidate_library_sha256",
        "feature_encoder_sha256",
        "current_profile_registry_sha256",
        "primary",
        "confirmation",
        "seed_contract",
        "schedule",
        "schedule_sha256",
        "execution_contract",
        "scientific_boundaries",
    }
)
_AUTH_KEYS = frozenset(
    {
        "schema",
        "status",
        "decision",
        "receipt_sha256",
        "performance_lock_qualified",
        "quality_pilot_authorized",
        "performance_lock_finalized",
        "one_shot_lock_consumed",
        "current_profile_changed",
    }
)
_PHASE_KEYS = frozenset(
    {
        "phase",
        "paired_hand_count",
        "root_count",
        "pair_indices",
        "root_indices",
        "hands_per_profile",
        "budget",
        "independent_locked_confirmation",
    }
)
_SEED_CONTRACT_KEYS = frozenset(
    {
        "schema",
        "seed_stride",
        "primary_namespace_bases",
        "confirmation_namespace_bases",
        "primary_seed_count",
        "confirmation_seed_count",
        "all_quality_seed_count",
        "all_quality_seeds_unique",
        "primary_confirmation_disjoint",
        "legacy_step6c_disjoint",
        "legacy_step6c_seed_set_sha256",
        "performance_v4_disjoint",
        "performance_v4_seed_set_sha256",
        "seed_set_sha256",
    }
)
_SCHEDULE_ROW_KEYS = frozenset(
    {
        "phase",
        "pair_index",
        "root_indices",
        "profile",
        "seeds",
        "primary_budget",
        "confirmation_budget",
    }
)
_ROOT_KEYS = frozenset(
    {
        "schema",
        "plan_sha256",
        "schedule_row_sha256",
        "phase",
        "pair_index",
        "root_indices",
        "profile",
        "seeds",
        "primary_budget",
        "confirmation_budget",
        "observations",
        "opponent_private_discards_used",
        "teacher_value_status",
        "training_eligible",
        "promotion_evidence",
        "current_profile_changed",
    }
)
_OBSERVATION_RECORD_KEYS = frozenset(
    {
        "root_index",
        "seat",
        "observation_fingerprint",
        "observation_sha256",
        "observation",
    }
)
_MATERIALIZATION_KEYS = frozenset(
    {
        "schema",
        "status",
        "plan_sha256",
        "root_directory",
        "root_count",
        "paired_hand_count",
        "phase_counts",
        "root_records",
        "root_record_aggregate_sha256",
        "observation_count",
        "unique_observation_fingerprint_count",
        "observation_fingerprint_aggregate_sha256",
        "hidden_information_field_count",
        "unknown_field_count",
        "missing_root_count",
        "current_profile_changed",
    }
)
_ROOT_RECORD_KEYS = frozenset(
    {"phase", "pair_index", "path", "sha256", "bytes", "root_indices"}
)
_SEAL_KEYS = frozenset(
    {
        "schema",
        "status",
        "plan_sha256",
        "materialization_sha256",
        "root_record_aggregate_sha256",
        "observation_fingerprint_aggregate_sha256",
        "root_count",
        "observation_count",
        "root_records",
        "quality_execution_authorized",
        "cloud_execution_started",
        "training_eligible",
        "current_profile_changed",
    }
)
_JOB_KEYS = frozenset(
    {
        "schema",
        "job_id",
        "phase",
        "pair_indices",
        "root_paths",
        "result_path",
        "plan_sha256",
        "root_seal_sha256",
        "candidate_library_sha256",
        "rayon_threads",
        "processes",
        "cloud_execution_started",
    }
)
_PACKAGE_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "status",
        "plan_sha256",
        "materialization_sha256",
        "root_seal_sha256",
        "job_count",
        "primary_job_count",
        "confirmation_job_count",
        "root_file_count",
        "entries",
        "entry_aggregate_sha256",
        "cloud_execution_started",
        "training_eligible",
        "current_profile_changed",
    }
)
_PACKAGE_ENTRY_KEYS = frozenset({"path", "sha256", "bytes"})
_PACKAGE_READY_KEYS = frozenset(
    {
        "schema",
        "status",
        "package_manifest_sha256",
        "package_file_sha256",
        "job_count",
        "cloud_execution_started",
        "training_eligible",
        "current_profile_changed",
    }
)


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        unknown = sorted(set(value) - expected)
        raise ValueError(
            f"{label} fields changed: missing={missing}, unknown={unknown}"
        )


def _read_canonical(path: str | Path, label: str) -> dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    raw = target.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _write_once(path: str | Path, value: Any) -> Path:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _phase_indices(phase: str) -> tuple[int, ...]:
    if phase == PRIMARY_PHASE:
        return PRIMARY_PAIR_INDICES
    if phase == CONFIRMATION_PHASE:
        return CONFIRMATION_PAIR_INDICES
    raise ValueError(f"unsupported fresh-quality phase: {phase!r}")


def _namespace_bases(phase: str) -> Mapping[str, int]:
    if phase == PRIMARY_PHASE:
        return PRIMARY_NAMESPACE_BASES
    if phase == CONFIRMATION_PHASE:
        return CONFIRMATION_NAMESPACE_BASES
    raise ValueError(f"unsupported fresh-quality phase: {phase!r}")


def seed_values(phase: str, pair_index: int) -> dict[str, int]:
    indices = _phase_indices(phase)
    if (
        isinstance(pair_index, bool)
        or not isinstance(pair_index, int)
        or pair_index not in indices
    ):
        raise ValueError(f"{phase} pair index is outside the frozen grid")
    return {
        namespace: base + SEED_STRIDE * pair_index
        for namespace, base in _namespace_bases(phase).items()
    }


def _root_indices(phase: str, pair_index: int) -> list[int]:
    if phase == PRIMARY_PHASE:
        return [pair_index * 2, pair_index * 2 + 1]
    if phase == CONFIRMATION_PHASE:
        return [100 + pair_index * 2, 101 + pair_index * 2]
    raise ValueError(f"unsupported fresh-quality phase: {phase!r}")


def schedule_row(phase: str, pair_index: int) -> dict[str, Any]:
    _ = seed_values(phase, pair_index)
    profile = M31_T3_BEHAVIOR_PROFILES[pair_index % len(M31_T3_BEHAVIOR_PROFILES)]
    return {
        "phase": phase,
        "pair_index": pair_index,
        "root_indices": _root_indices(phase, pair_index),
        "profile": profile,
        "seeds": seed_values(phase, pair_index),
        "primary_budget": dict(PRIMARY_BUDGET),
        "confirmation_budget": (
            dict(CONFIRMATION_BUDGET) if phase == CONFIRMATION_PHASE else None
        ),
    }


def schedule_rows() -> list[dict[str, Any]]:
    return [
        schedule_row(phase, index)
        for phase in PHASES
        for index in _phase_indices(phase)
    ]


def _seed_set(phase: str) -> set[int]:
    return {
        value
        for index in _phase_indices(phase)
        for value in seed_values(phase, index).values()
    }


def build_seed_contract() -> dict[str, Any]:
    primary = _seed_set(PRIMARY_PHASE)
    confirmation = _seed_set(CONFIRMATION_PHASE)
    performance = {
        value
        for index in range(100)
        for value in step6d_v2.candidate02_performance_lock_v4_seed_values(
            index
        ).values()
    }
    legacy = {
        value
        for index in legacy_step6c.PILOT_HAND_INDICES
        for value in legacy_step6c.train_seed_values(index).values()
    }
    all_quality = primary | confirmation
    value = {
        "schema": "hu_m31_t3_step6d_fresh_quality_seed_contract_v1",
        "seed_stride": SEED_STRIDE,
        "primary_namespace_bases": dict(PRIMARY_NAMESPACE_BASES),
        "confirmation_namespace_bases": dict(CONFIRMATION_NAMESPACE_BASES),
        "primary_seed_count": len(primary),
        "confirmation_seed_count": len(confirmation),
        "all_quality_seed_count": len(all_quality),
        "all_quality_seeds_unique": (
            len(primary) == 50 * 6
            and len(confirmation) == 5 * 6
            and len(all_quality) == 55 * 6
        ),
        "primary_confirmation_disjoint": not (primary & confirmation),
        "legacy_step6c_disjoint": not (all_quality & legacy),
        "legacy_step6c_seed_set_sha256": canonical_sha256(sorted(legacy)),
        "performance_v4_disjoint": not (all_quality & performance),
        "performance_v4_seed_set_sha256": PERFORMANCE_V4_SEED_SET_SHA256,
        "seed_set_sha256": canonical_sha256(sorted(all_quality)),
    }
    return validate_seed_contract(value)


def validate_seed_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    _exact_keys(payload, _SEED_CONTRACT_KEYS, "fresh-quality seed contract")
    # Avoid recursion: recompute the expected scalar fields directly.
    primary = _seed_set(PRIMARY_PHASE)
    confirmation = _seed_set(CONFIRMATION_PHASE)
    performance = {
        item
        for index in range(100)
        for item in step6d_v2.candidate02_performance_lock_v4_seed_values(
            index
        ).values()
    }
    legacy = {
        item
        for index in legacy_step6c.PILOT_HAND_INDICES
        for item in legacy_step6c.train_seed_values(index).values()
    }
    all_quality = primary | confirmation
    expected_value = {
        "schema": "hu_m31_t3_step6d_fresh_quality_seed_contract_v1",
        "seed_stride": SEED_STRIDE,
        "primary_namespace_bases": dict(PRIMARY_NAMESPACE_BASES),
        "confirmation_namespace_bases": dict(CONFIRMATION_NAMESPACE_BASES),
        "primary_seed_count": 300,
        "confirmation_seed_count": 30,
        "all_quality_seed_count": 330,
        "all_quality_seeds_unique": len(all_quality) == 330,
        "primary_confirmation_disjoint": not (primary & confirmation),
        "legacy_step6c_disjoint": not (all_quality & legacy),
        "legacy_step6c_seed_set_sha256": canonical_sha256(sorted(legacy)),
        "performance_v4_disjoint": not (all_quality & performance),
        "performance_v4_seed_set_sha256": PERFORMANCE_V4_SEED_SET_SHA256,
        "seed_set_sha256": canonical_sha256(sorted(all_quality)),
    }
    if payload != expected_value or not all(
        payload[field]
        for field in (
            "all_quality_seeds_unique",
            "primary_confirmation_disjoint",
            "legacy_step6c_disjoint",
            "performance_v4_disjoint",
        )
    ):
        raise ValueError(
            "fresh-quality seed namespaces are not the frozen disjoint set"
        )
    return payload


def _authorization_summary(receipt: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "schema": receipt.get("schema"),
        "status": receipt.get("status"),
        "decision": receipt.get("decision"),
        "receipt_sha256": receipt.get("receipt_sha256"),
        "performance_lock_qualified": receipt.get("performance_lock_qualified"),
        "quality_pilot_authorized": receipt.get("quality_pilot_authorized"),
        "performance_lock_finalized": receipt.get("performance_lock_finalized"),
        "one_shot_lock_consumed": receipt.get("one_shot_lock_consumed"),
        "current_profile_changed": receipt.get("current_profile_changed"),
    }
    _exact_keys(summary, _AUTH_KEYS, "performance authorization summary")
    if (
        summary["schema"] != PERFORMANCE_RECEIPT_SCHEMA
        or summary["status"] != "qualified"
        or summary["decision"] != QUALIFIED_DECISION
        or not _is_sha256(summary["receipt_sha256"])
        or any(
            summary[field] is not True
            for field in (
                "performance_lock_qualified",
                "quality_pilot_authorized",
                "performance_lock_finalized",
                "one_shot_lock_consumed",
            )
        )
        or summary["current_profile_changed"] is not False
    ):
        raise PermissionError("performance-lock v4 has not authorized fresh quality")
    return summary


def _load_performance_authorization(path: str | Path) -> dict[str, Any]:
    receipt, _audit = portable_receipt.load_preferred_or_pinned_receipt(
        Path(path),
        expected_profile_sha256=CURRENT_PROFILE_REGISTRY_SHA256,
    )
    return _authorization_summary(receipt)


def _phase_contract(phase: str) -> dict[str, Any]:
    indices = _phase_indices(phase)
    roots = [root for index in indices for root in _root_indices(phase, index)]
    return {
        "phase": phase,
        "paired_hand_count": len(indices),
        "root_count": len(roots),
        "pair_indices": list(indices),
        "root_indices": roots,
        "hands_per_profile": 10 if phase == PRIMARY_PHASE else 1,
        "budget": (
            dict(PRIMARY_BUDGET)
            if phase == PRIMARY_PHASE
            else dict(CONFIRMATION_BUDGET)
        ),
        "independent_locked_confirmation": phase == CONFIRMATION_PHASE,
    }


def build_fresh_quality_plan(*, performance_receipt_path: str | Path) -> dict[str, Any]:
    authorization = _load_performance_authorization(performance_receipt_path)
    rows = schedule_rows()
    plan = {
        "schema": PLAN_SCHEMA,
        "run_id": RUN_ID,
        "engine_run_id": ENGINE_RUN_ID,
        "purpose": "fresh_quality_only_not_match_ev_or_promotion",
        "authorizing_performance_receipt": authorization,
        "candidate_library_sha256": ACCEPTED_CANDIDATE_LIBRARY_SHA256,
        "feature_encoder_sha256": ACCEPTED_FEATURE_ENCODER_SHA256,
        "current_profile_registry_sha256": CURRENT_PROFILE_REGISTRY_SHA256,
        "primary": _phase_contract(PRIMARY_PHASE),
        "confirmation": _phase_contract(CONFIRMATION_PHASE),
        "seed_contract": build_seed_contract(),
        "schedule": rows,
        "schedule_sha256": canonical_sha256(rows),
        "execution_contract": {
            "processes_per_job": 1,
            "rayon_threads_per_process": 16,
            "candidate_and_evaluation_common_random_futures": True,
            "candidate_selection_and_locked_evaluation_independent": True,
            "primary_and_confirmation_root_populations_disjoint": True,
            "primary_jobs": 10,
            "confirmation_jobs": 5,
            "restart_safe_create_only_results": True,
        },
        "scientific_boundaries": {
            "opponent_private_discards_used": False,
            "realized_deck_tail_used": False,
            "teacher_values_are_realized_match_ev": False,
            "thresholds_reselected_on_holdout": False,
            "training_eligible": False,
            "production_fanout_authorized": False,
            "promotion_evidence": False,
            "current_profile_changed": False,
        },
    }
    return validate_fresh_quality_plan(plan)


def validate_fresh_quality_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    plan = deepcopy(dict(value))
    _exact_keys(plan, _PLAN_KEYS, "fresh-quality plan")
    primary = plan.get("primary")
    confirmation = plan.get("confirmation")
    authorization = plan.get("authorizing_performance_receipt")
    boundaries = plan.get("scientific_boundaries")
    execution = plan.get("execution_contract")
    if not all(
        isinstance(item, Mapping)
        for item in (primary, confirmation, authorization, boundaries, execution)
    ):
        raise ValueError("fresh-quality plan contains a missing object")
    _exact_keys(primary, _PHASE_KEYS, "fresh-quality primary contract")
    _exact_keys(confirmation, _PHASE_KEYS, "fresh-quality confirmation contract")
    auth = _authorization_summary(authorization)
    seed_contract = validate_seed_contract(
        plan.get("seed_contract")
        if isinstance(plan.get("seed_contract"), Mapping)
        else {}
    )
    rows = schedule_rows()
    if (
        plan["schema"] != PLAN_SCHEMA
        or plan["run_id"] != RUN_ID
        or plan["engine_run_id"] != ENGINE_RUN_ID
        or plan["purpose"] != "fresh_quality_only_not_match_ev_or_promotion"
        or authorization != auth
        or plan["candidate_library_sha256"] != ACCEPTED_CANDIDATE_LIBRARY_SHA256
        or plan["feature_encoder_sha256"] != ACCEPTED_FEATURE_ENCODER_SHA256
        or plan["current_profile_registry_sha256"] != CURRENT_PROFILE_REGISTRY_SHA256
        or primary != _phase_contract(PRIMARY_PHASE)
        or confirmation != _phase_contract(CONFIRMATION_PHASE)
        or plan["seed_contract"] != seed_contract
        or plan["schedule"] != rows
        or plan["schedule_sha256"] != canonical_sha256(rows)
        or execution
        != {
            "processes_per_job": 1,
            "rayon_threads_per_process": 16,
            "candidate_and_evaluation_common_random_futures": True,
            "candidate_selection_and_locked_evaluation_independent": True,
            "primary_and_confirmation_root_populations_disjoint": True,
            "primary_jobs": 10,
            "confirmation_jobs": 5,
            "restart_safe_create_only_results": True,
        }
        or boundaries
        != {
            "opponent_private_discards_used": False,
            "realized_deck_tail_used": False,
            "teacher_values_are_realized_match_ev": False,
            "thresholds_reselected_on_holdout": False,
            "training_eligible": False,
            "production_fanout_authorized": False,
            "promotion_evidence": False,
            "current_profile_changed": False,
        }
    ):
        raise ValueError("fresh-quality plan differs from the frozen experiment")
    return plan


def validate_plan_authorization(
    plan: Mapping[str, Any], *, performance_receipt_path: str | Path
) -> dict[str, Any]:
    validated = validate_fresh_quality_plan(plan)
    observed = _load_performance_authorization(performance_receipt_path)
    if observed != validated["authorizing_performance_receipt"]:
        raise PermissionError("fresh-quality plan authorization receipt changed")
    return validated


def write_fresh_quality_plan(
    *, performance_receipt_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    plan = build_fresh_quality_plan(performance_receipt_path=performance_receipt_path)
    _write_once(output_path, plan)
    stored = _read_canonical(output_path, "stored fresh-quality plan")
    if (
        validate_plan_authorization(
            stored, performance_receipt_path=performance_receipt_path
        )
        != plan
    ):
        raise ValueError("stored fresh-quality plan differs from source replay")
    return plan


def _validate_observation_record(
    value: Mapping[str, Any], *, expected_root_index: int, expected_seat: str
) -> ActorObservation:
    record = dict(value)
    _exact_keys(record, _OBSERVATION_RECORD_KEYS, "fresh-quality observation record")
    raw = record.get("observation")
    if not isinstance(raw, Mapping):
        raise ValueError("fresh-quality observation is missing")
    observation = ActorObservation.from_dict(raw)
    if (
        record["root_index"] != expected_root_index
        or record["seat"] != expected_seat
        or observation.seat != expected_seat
        or observation.street != "T3"
        or observation.fingerprint() != record["observation_fingerprint"]
        or canonical_sha256(observation.to_dict()) != record["observation_sha256"]
    ):
        raise ValueError("fresh-quality observation identity changed")
    step6d_v1._reject_hidden(record, "fresh_quality_observation")
    return observation


def validate_root(
    value: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    root = deepcopy(dict(value))
    validated_plan = validate_fresh_quality_plan(plan)
    _exact_keys(root, _ROOT_KEYS, "fresh-quality root")
    phase = root.get("phase")
    pair_index = root.get("pair_index")
    row = schedule_row(str(phase), int(pair_index))
    observations = root.get("observations")
    if (
        root["schema"] != ROOT_SCHEMA
        or root["plan_sha256"] != canonical_sha256(validated_plan)
        or root["schedule_row_sha256"] != canonical_sha256(row)
        or root["root_indices"] != row["root_indices"]
        or root["profile"] != row["profile"]
        or root["seeds"] != row["seeds"]
        or root["primary_budget"] != row["primary_budget"]
        or root["confirmation_budget"] != row["confirmation_budget"]
        or not isinstance(observations, list)
        or len(observations) != 2
        or root["opponent_private_discards_used"] is not False
        or root["teacher_value_status"] != "diagnostic_not_match_EV"
        or any(
            root[field] is not False
            for field in (
                "training_eligible",
                "promotion_evidence",
                "current_profile_changed",
            )
        )
    ):
        raise ValueError("fresh-quality root contract changed")
    _validate_observation_record(
        observations[0],
        expected_root_index=row["root_indices"][0],
        expected_seat="first",
    )
    _validate_observation_record(
        observations[1],
        expected_root_index=row["root_indices"][1],
        expected_seat="second",
    )
    step6d_v1._reject_hidden(root, "fresh_quality_root")
    return root


def _generate_observations(
    *, repository_root: Path, row: Mapping[str, Any], bundle: Any
) -> tuple[ActorObservation, ActorObservation]:
    del repository_root
    return generate_behavior_t3_roots(
        hand_seed=int(row["seeds"]["hand"]),
        behavior_seed=int(row["seeds"]["behavior"]),
        profile=str(row["profile"]),
        bundle=bundle,
    )


def _root_value(
    *, plan: Mapping[str, Any], row: Mapping[str, Any], observations: Sequence[Any]
) -> dict[str, Any]:
    records = []
    for root_index, expected_seat, observation in zip(
        row["root_indices"], ("first", "second"), observations, strict=True
    ):
        if (
            not isinstance(observation, ActorObservation)
            or observation.seat != expected_seat
        ):
            raise ValueError("root generator returned the wrong actor observation")
        payload = observation.to_dict()
        records.append(
            {
                "root_index": root_index,
                "seat": expected_seat,
                "observation_fingerprint": observation.fingerprint(),
                "observation_sha256": canonical_sha256(payload),
                "observation": payload,
            }
        )
    return {
        "schema": ROOT_SCHEMA,
        "plan_sha256": canonical_sha256(plan),
        "schedule_row_sha256": canonical_sha256(row),
        "phase": row["phase"],
        "pair_index": row["pair_index"],
        "root_indices": row["root_indices"],
        "profile": row["profile"],
        "seeds": row["seeds"],
        "primary_budget": row["primary_budget"],
        "confirmation_budget": row["confirmation_budget"],
        "observations": records,
        "opponent_private_discards_used": False,
        "teacher_value_status": "diagnostic_not_match_EV",
        "training_eligible": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }


def _root_relative_path(phase: str, pair_index: int) -> str:
    return f"roots/{phase}/pair_{pair_index:03d}.json"


def _root_records(
    *, root_directory: Path, plan: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    fingerprints: list[str] = []
    expected_paths = {
        _root_relative_path(row["phase"], row["pair_index"]) for row in schedule_rows()
    }
    actual_paths = {
        path.relative_to(root_directory).as_posix()
        for path in root_directory.rglob("*.json")
        if path.is_file() and not path.is_symlink()
    }
    if actual_paths != expected_paths:
        raise ValueError("fresh-quality materialized root grid has missing/extra files")
    for row in schedule_rows():
        relative = _root_relative_path(row["phase"], row["pair_index"])
        path = root_directory / relative
        root = validate_root(
            _read_canonical(path, f"fresh-quality root {relative}"), plan=plan
        )
        fingerprints.extend(
            str(item["observation_fingerprint"]) for item in root["observations"]
        )
        records.append(
            {
                "phase": row["phase"],
                "pair_index": row["pair_index"],
                "path": relative,
                "sha256": _file_sha256(path),
                "bytes": path.stat().st_size,
                "root_indices": row["root_indices"],
            }
        )
    return records, fingerprints


def build_materialization_receipt(
    *, plan: Mapping[str, Any], root_directory: str | Path
) -> dict[str, Any]:
    validated_plan = validate_fresh_quality_plan(plan)
    root_dir = Path(root_directory).resolve()
    if root_dir.is_symlink() or not root_dir.is_dir():
        raise ValueError("fresh-quality root directory is missing or unsafe")
    records, fingerprints = _root_records(root_directory=root_dir, plan=validated_plan)
    if len(fingerprints) != 110 or len(set(fingerprints)) != 110:
        raise ValueError("fresh-quality observations are missing or duplicated")
    value = {
        "schema": MATERIALIZATION_SCHEMA,
        "status": "exact_55_paired_110_roots_materialized",
        "plan_sha256": canonical_sha256(validated_plan),
        "root_directory": str(root_dir),
        "root_count": 110,
        "paired_hand_count": 55,
        "phase_counts": {
            "primary_paired_hands": 50,
            "primary_roots": 100,
            "confirmation_paired_hands": 5,
            "confirmation_roots": 10,
        },
        "root_records": records,
        "root_record_aggregate_sha256": canonical_sha256(records),
        "observation_count": 110,
        "unique_observation_fingerprint_count": 110,
        "observation_fingerprint_aggregate_sha256": canonical_sha256(fingerprints),
        "hidden_information_field_count": 0,
        "unknown_field_count": 0,
        "missing_root_count": 0,
        "current_profile_changed": False,
    }
    return validate_materialization_receipt(
        value, plan=validated_plan, replay_roots=True
    )


def validate_materialization_receipt(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    replay_roots: bool,
) -> dict[str, Any]:
    receipt = deepcopy(dict(value))
    validated_plan = validate_fresh_quality_plan(plan)
    _exact_keys(receipt, _MATERIALIZATION_KEYS, "fresh-quality materialization")
    records = receipt.get("root_records")
    if not isinstance(records, list) or len(records) != 55:
        raise ValueError("fresh-quality root record manifest changed")
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("fresh-quality root record is missing")
        _exact_keys(record, _ROOT_RECORD_KEYS, "fresh-quality root record")
    if (
        receipt["schema"] != MATERIALIZATION_SCHEMA
        or receipt["status"] != "exact_55_paired_110_roots_materialized"
        or receipt["plan_sha256"] != canonical_sha256(validated_plan)
        or receipt["root_count"] != 110
        or receipt["paired_hand_count"] != 55
        or receipt["phase_counts"]
        != {
            "primary_paired_hands": 50,
            "primary_roots": 100,
            "confirmation_paired_hands": 5,
            "confirmation_roots": 10,
        }
        or receipt["root_record_aggregate_sha256"] != canonical_sha256(records)
        or receipt["observation_count"] != 110
        or receipt["unique_observation_fingerprint_count"] != 110
        or any(
            receipt[field] != 0
            for field in (
                "hidden_information_field_count",
                "unknown_field_count",
                "missing_root_count",
            )
        )
        or receipt["current_profile_changed"] is not False
    ):
        raise ValueError("fresh-quality materialization receipt changed")
    if replay_roots is not True:
        raise PermissionError("fresh-quality receipt validation requires root replay")
    root_dir = Path(str(receipt["root_directory"]))
    if not root_dir.is_absolute() or root_dir.is_symlink() or not root_dir.is_dir():
        raise ValueError("fresh-quality materialization root path is unsafe")
    expected_records, fingerprints = _root_records(
        root_directory=root_dir.resolve(), plan=validated_plan
    )
    if (
        records != expected_records
        or receipt["observation_fingerprint_aggregate_sha256"]
        != canonical_sha256(fingerprints)
        or len(set(fingerprints)) != 110
    ):
        raise ValueError("fresh-quality materialization differs from root replay")
    return receipt


def build_root_seal(
    *, plan: Mapping[str, Any], materialization: Mapping[str, Any]
) -> dict[str, Any]:
    validated_plan = validate_fresh_quality_plan(plan)
    receipt = validate_materialization_receipt(
        materialization, plan=validated_plan, replay_roots=True
    )
    value = {
        "schema": ROOT_SEAL_SCHEMA,
        "status": "sealed_create_only_fresh_quality_roots",
        "plan_sha256": canonical_sha256(validated_plan),
        "materialization_sha256": canonical_sha256(receipt),
        "root_record_aggregate_sha256": receipt["root_record_aggregate_sha256"],
        "observation_fingerprint_aggregate_sha256": receipt[
            "observation_fingerprint_aggregate_sha256"
        ],
        "root_count": 110,
        "observation_count": 110,
        "root_records": deepcopy(receipt["root_records"]),
        "quality_execution_authorized": True,
        "cloud_execution_started": False,
        "training_eligible": False,
        "current_profile_changed": False,
    }
    return validate_root_seal(value, plan=validated_plan, materialization=receipt)


def validate_root_seal(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    seal = deepcopy(dict(value))
    validated_plan = validate_fresh_quality_plan(plan)
    receipt = validate_materialization_receipt(
        materialization, plan=validated_plan, replay_roots=True
    )
    _exact_keys(seal, _SEAL_KEYS, "fresh-quality root seal")
    expected = {
        "schema": ROOT_SEAL_SCHEMA,
        "status": "sealed_create_only_fresh_quality_roots",
        "plan_sha256": canonical_sha256(validated_plan),
        "materialization_sha256": canonical_sha256(receipt),
        "root_record_aggregate_sha256": receipt["root_record_aggregate_sha256"],
        "observation_fingerprint_aggregate_sha256": receipt[
            "observation_fingerprint_aggregate_sha256"
        ],
        "root_count": 110,
        "observation_count": 110,
        "root_records": deepcopy(receipt["root_records"]),
        "quality_execution_authorized": True,
        "cloud_execution_started": False,
        "training_eligible": False,
        "current_profile_changed": False,
    }
    if seal != expected:
        raise ValueError("fresh-quality root seal differs from source replay")
    return seal


def materialize_fresh_quality_roots(
    *,
    repository_root: str | Path,
    plan_path: str | Path,
    performance_receipt_path: str | Path,
    output_directory: str | Path,
    materialization_output: str | Path,
    seal_output: str | Path,
    feature_encoder_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repository = Path(repository_root).resolve()
    plan = validate_plan_authorization(
        _read_canonical(plan_path, "fresh-quality plan"),
        performance_receipt_path=performance_receipt_path,
    )
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError("fresh-quality root output is create-only")
    output.mkdir(parents=True)
    bundle = None
    model_profiles = set(M31_T3_BEHAVIOR_PROFILES) - {"random_exact_final"}
    feature_binding = (
        stage3_feature_rust.pinned_feature_encoder_library(
            feature_encoder_path,
            expected_sha256=ACCEPTED_FEATURE_ENCODER_SHA256,
        )
        if feature_encoder_path is not None
        else nullcontext()
    )
    try:
        with feature_binding:
            for row in schedule_rows():
                if bundle is None:
                    bundle = step6d_v1.load_model_bundle(
                        step6d_v1._absolute_model_paths(repository),
                        profiles=model_profiles,
                    )
                observations = _generate_observations(
                    repository_root=repository, row=row, bundle=bundle
                )
                root = validate_root(
                    _root_value(plan=plan, row=row, observations=observations),
                    plan=plan,
                )
                _write_once(
                    output / _root_relative_path(row["phase"], row["pair_index"]),
                    root,
                )
        materialization = build_materialization_receipt(
            plan=plan, root_directory=output
        )
        _write_once(materialization_output, materialization)
        seal = build_root_seal(plan=plan, materialization=materialization)
        _write_once(seal_output, seal)
        return materialization, seal
    except BaseException:
        # Never erase a partially generated scientific run.  The caller can
        # inspect it, but a retry must use a new output directory.
        raise


def build_job_descriptors(
    *, plan: Mapping[str, Any], seal: Mapping[str, Any], root_directory: str | Path
) -> list[dict[str, Any]]:
    validated_plan = validate_fresh_quality_plan(plan)
    root_dir = Path(root_directory).resolve()
    materialization_stub = {
        "schema": MATERIALIZATION_SCHEMA,
        "status": "exact_55_paired_110_roots_materialized",
        "plan_sha256": canonical_sha256(validated_plan),
        "root_directory": str(root_dir),
        "root_count": 110,
        "paired_hand_count": 55,
        "phase_counts": {
            "primary_paired_hands": 50,
            "primary_roots": 100,
            "confirmation_paired_hands": 5,
            "confirmation_roots": 10,
        },
        "root_records": deepcopy(seal.get("root_records")),
        "root_record_aggregate_sha256": seal.get("root_record_aggregate_sha256"),
        "observation_count": 110,
        "unique_observation_fingerprint_count": 110,
        "observation_fingerprint_aggregate_sha256": seal.get(
            "observation_fingerprint_aggregate_sha256"
        ),
        "hidden_information_field_count": 0,
        "unknown_field_count": 0,
        "missing_root_count": 0,
        "current_profile_changed": False,
    }
    validated_seal = validate_root_seal(
        seal, plan=validated_plan, materialization=materialization_stub
    )
    plan_sha = canonical_sha256(validated_plan)
    seal_sha = canonical_sha256(validated_seal)
    jobs: list[dict[str, Any]] = []
    for job_number in range(10):
        indices = list(range(job_number * 5, job_number * 5 + 5))
        jobs.append(
            {
                "schema": JOB_SCHEMA,
                "job_id": f"primary-{job_number:02d}",
                "phase": PRIMARY_PHASE,
                "pair_indices": indices,
                "root_paths": [
                    _root_relative_path(PRIMARY_PHASE, index) for index in indices
                ],
                "result_path": f"results/primary-{job_number:02d}.json",
                "plan_sha256": plan_sha,
                "root_seal_sha256": seal_sha,
                "candidate_library_sha256": ACCEPTED_CANDIDATE_LIBRARY_SHA256,
                "rayon_threads": 16,
                "processes": 1,
                "cloud_execution_started": False,
            }
        )
    for index in CONFIRMATION_PAIR_INDICES:
        jobs.append(
            {
                "schema": JOB_SCHEMA,
                "job_id": f"confirmation-{index:02d}",
                "phase": CONFIRMATION_PHASE,
                "pair_indices": [index],
                "root_paths": [_root_relative_path(CONFIRMATION_PHASE, index)],
                "result_path": f"results/confirmation-{index:02d}.json",
                "plan_sha256": plan_sha,
                "root_seal_sha256": seal_sha,
                "candidate_library_sha256": ACCEPTED_CANDIDATE_LIBRARY_SHA256,
                "rayon_threads": 16,
                "processes": 1,
                "cloud_execution_started": False,
            }
        )
    return [
        validate_job_descriptor(job, plan=validated_plan, seal=validated_seal)
        for job in jobs
    ]


def validate_job_descriptor(
    value: Mapping[str, Any], *, plan: Mapping[str, Any], seal: Mapping[str, Any]
) -> dict[str, Any]:
    job = deepcopy(dict(value))
    _exact_keys(job, _JOB_KEYS, "fresh-quality job")
    phase = job.get("phase")
    indices = job.get("pair_indices")
    if not isinstance(indices, list) or not indices:
        raise ValueError("fresh-quality job pair grid is missing")
    if phase == PRIMARY_PHASE:
        expected_id = f"primary-{indices[0] // 5:02d}"
        expected_indices = list(range((indices[0] // 5) * 5, (indices[0] // 5) * 5 + 5))
    elif phase == CONFIRMATION_PHASE:
        expected_id = f"confirmation-{indices[0]:02d}"
        expected_indices = [indices[0]]
    else:
        raise ValueError("fresh-quality job phase changed")
    expected_paths = [_root_relative_path(str(phase), index) for index in indices]
    if (
        indices != expected_indices
        or any(index not in _phase_indices(str(phase)) for index in indices)
        or job["job_id"] != expected_id
        or job["root_paths"] != expected_paths
        or job["result_path"] != f"results/{expected_id}.json"
        or job["plan_sha256"] != canonical_sha256(plan)
        or job["root_seal_sha256"] != canonical_sha256(seal)
        or job["candidate_library_sha256"] != ACCEPTED_CANDIDATE_LIBRARY_SHA256
        or job["rayon_threads"] != 16
        or job["processes"] != 1
        or job["cloud_execution_started"] is not False
    ):
        raise ValueError("fresh-quality job differs from the frozen shard grid")
    return job


def _zip_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError("fresh-quality package archive is create-only")
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(
            (item for item in source.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(source).as_posix(),
        ):
            relative = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


def validate_fresh_quality_package(
    *,
    package_directory: str | Path,
    performance_receipt_path: str | Path,
    archive_path: str | Path | None = None,
) -> dict[str, Any]:
    package = Path(package_directory).resolve()
    if package.is_symlink() or not package.is_dir():
        raise ValueError("fresh-quality package directory is missing or unsafe")
    manifest_path = package / "PACKAGE_MANIFEST.json"
    ready_path = package / "PACKAGE_READY.json"
    manifest = _read_canonical(manifest_path, "fresh-quality package manifest")
    ready = _read_canonical(ready_path, "fresh-quality package-ready receipt")
    _exact_keys(manifest, _PACKAGE_MANIFEST_KEYS, "fresh-quality package manifest")
    _exact_keys(ready, _PACKAGE_READY_KEYS, "fresh-quality package-ready receipt")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != 73:
        raise ValueError("fresh-quality package entry count changed")
    expected_entry_paths: set[str] = set()
    for record in entries:
        if not isinstance(record, Mapping):
            raise ValueError("fresh-quality package entry is missing")
        _exact_keys(record, _PACKAGE_ENTRY_KEYS, "fresh-quality package entry")
        relative = Path(str(record["path"]))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() in expected_entry_paths
        ):
            raise ValueError("fresh-quality package entry path is unsafe")
        path = package / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or _file_sha256(path) != record["sha256"]
            or path.stat().st_size != record["bytes"]
        ):
            raise ValueError("fresh-quality package entry hash/size changed")
        expected_entry_paths.add(relative.as_posix())
    actual_paths = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual_paths != expected_entry_paths | {
        "PACKAGE_MANIFEST.json",
        "PACKAGE_READY.json",
    }:
        raise ValueError("fresh-quality package has missing or extra files")
    plan = validate_plan_authorization(
        _read_canonical(package / "control" / "plan.json", "packaged plan"),
        performance_receipt_path=performance_receipt_path,
    )
    materialization = validate_materialization_receipt(
        _read_canonical(
            package / "control" / "materialization.json",
            "packaged materialization",
        ),
        plan=plan,
        replay_roots=True,
    )
    seal = validate_root_seal(
        _read_canonical(package / "control" / "root_seal.json", "packaged root seal"),
        plan=plan,
        materialization=materialization,
    )
    jobs = build_job_descriptors(
        plan=plan,
        seal=seal,
        root_directory=materialization["root_directory"],
    )
    for job in jobs:
        stored = _read_canonical(
            package / "jobs" / f"{job['job_id']}.json",
            f"packaged job {job['job_id']}",
        )
        if stored != job:
            raise ValueError("packaged fresh-quality job descriptor changed")
    record_by_path = {record["path"]: record for record in seal["root_records"]}
    for relative, record in record_by_path.items():
        packaged_relative = Path(relative)
        if packaged_relative.parts[0] != "roots":
            raise ValueError("fresh-quality root path left the frozen package tree")
        packaged_root = package.joinpath(*packaged_relative.parts)
        if (
            _file_sha256(packaged_root) != record["sha256"]
            or packaged_root.stat().st_size != record["bytes"]
        ):
            raise ValueError("packaged fresh-quality root differs from root seal")
    if (
        manifest["schema"] != PACKAGE_SCHEMA
        or manifest["status"] != "immutable_fresh_quality_inputs_ready"
        or manifest["plan_sha256"] != canonical_sha256(plan)
        or manifest["materialization_sha256"] != canonical_sha256(materialization)
        or manifest["root_seal_sha256"] != canonical_sha256(seal)
        or manifest["job_count"] != 15
        or manifest["primary_job_count"] != 10
        or manifest["confirmation_job_count"] != 5
        or manifest["root_file_count"] != 55
        or manifest["entry_aggregate_sha256"] != canonical_sha256(entries)
        or any(
            manifest[field] is not False
            for field in (
                "cloud_execution_started",
                "training_eligible",
                "current_profile_changed",
            )
        )
        or ready
        != {
            "schema": PACKAGE_READY_SCHEMA,
            "status": "ready_for_separate_execution_authorization",
            "package_manifest_sha256": canonical_sha256(manifest),
            "package_file_sha256": _file_sha256(manifest_path),
            "job_count": 15,
            "cloud_execution_started": False,
            "training_eligible": False,
            "current_profile_changed": False,
        }
    ):
        raise ValueError("fresh-quality package manifest/ready boundary changed")
    archive_record = None
    if archive_path is not None:
        archive = Path(archive_path).resolve()
        if archive.is_symlink() or not archive.is_file():
            raise ValueError("fresh-quality package archive is missing or unsafe")
        with zipfile.ZipFile(archive, "r") as zipped:
            names = zipped.namelist()
            if (
                names != sorted(actual_paths)
                or len(names) != len(set(names))
                or any(
                    zipped.getinfo(name).date_time != (1980, 1, 1, 0, 0, 0)
                    or zipped.read(name) != (package / Path(name)).read_bytes()
                    for name in names
                )
            ):
                raise ValueError("fresh-quality package ZIP differs from directory")
        archive_record = {
            "path": str(archive),
            "sha256": _file_sha256(archive),
            "bytes": archive.stat().st_size,
        }
    return {
        "manifest": manifest,
        "ready": ready,
        "plan_sha256": canonical_sha256(plan),
        "materialization_sha256": canonical_sha256(materialization),
        "root_seal_sha256": canonical_sha256(seal),
        "archive": archive_record,
    }


def create_fresh_quality_package(
    *,
    plan_path: str | Path,
    materialization_path: str | Path,
    seal_path: str | Path,
    performance_receipt_path: str | Path,
    output_directory: str | Path,
    archive_path: str | Path,
) -> dict[str, Any]:
    plan_file = Path(plan_path).resolve()
    plan = validate_plan_authorization(
        _read_canonical(plan_file, "fresh-quality plan"),
        performance_receipt_path=performance_receipt_path,
    )
    materialization = validate_materialization_receipt(
        _read_canonical(materialization_path, "fresh-quality materialization"),
        plan=plan,
        replay_roots=True,
    )
    seal = validate_root_seal(
        _read_canonical(seal_path, "fresh-quality root seal"),
        plan=plan,
        materialization=materialization,
    )
    root_dir = Path(str(materialization["root_directory"])).resolve()
    jobs = build_job_descriptors(plan=plan, seal=seal, root_directory=root_dir)
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError("fresh-quality package directory is create-only")
    destination.mkdir(parents=True)
    try:
        control = destination / "control"
        roots = destination / "roots"
        jobs_dir = destination / "jobs"
        control.mkdir()
        jobs_dir.mkdir()
        shutil.copytree(root_dir / "roots", roots)
        _write_once(control / "plan.json", plan)
        _write_once(control / "materialization.json", materialization)
        _write_once(control / "root_seal.json", seal)
        for job in jobs:
            _write_once(jobs_dir / f"{job['job_id']}.json", job)
        entries = []
        for path in sorted(
            (item for item in destination.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(destination).as_posix(),
        ):
            entries.append(
                {
                    "path": path.relative_to(destination).as_posix(),
                    "sha256": _file_sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
        manifest = {
            "schema": PACKAGE_SCHEMA,
            "status": "immutable_fresh_quality_inputs_ready",
            "plan_sha256": canonical_sha256(plan),
            "materialization_sha256": canonical_sha256(materialization),
            "root_seal_sha256": canonical_sha256(seal),
            "job_count": 15,
            "primary_job_count": 10,
            "confirmation_job_count": 5,
            "root_file_count": 55,
            "entries": entries,
            "entry_aggregate_sha256": canonical_sha256(entries),
            "cloud_execution_started": False,
            "training_eligible": False,
            "current_profile_changed": False,
        }
        _write_once(destination / "PACKAGE_MANIFEST.json", manifest)
        ready = {
            "schema": PACKAGE_READY_SCHEMA,
            "status": "ready_for_separate_execution_authorization",
            "package_manifest_sha256": canonical_sha256(manifest),
            "package_file_sha256": _file_sha256(destination / "PACKAGE_MANIFEST.json"),
            "job_count": 15,
            "cloud_execution_started": False,
            "training_eligible": False,
            "current_profile_changed": False,
        }
        _write_once(destination / "PACKAGE_READY.json", ready)
        archive = Path(archive_path).resolve()
        _zip_tree(destination, archive)
        validated = validate_fresh_quality_package(
            package_directory=destination,
            performance_receipt_path=performance_receipt_path,
            archive_path=archive,
        )
        return {
            "manifest": manifest,
            "ready": ready,
            "archive_path": str(archive),
            "archive_sha256": _file_sha256(archive),
            "archive_bytes": archive.stat().st_size,
            "validated": validated,
        }
    except BaseException:
        # Preserve partial packages for forensic inspection.
        raise


__all__ = [
    "ACCEPTED_CANDIDATE_LIBRARY_SHA256",
    "CONFIRMATION_BUDGET",
    "CONFIRMATION_NAMESPACE_BASES",
    "CONFIRMATION_PAIR_INDICES",
    "CONFIRMATION_PHASE",
    "CONFIRMATION_ROOT_INDICES",
    "CURRENT_PROFILE_REGISTRY_SHA256",
    "ENGINE_RUN_ID",
    "JOB_SCHEMA",
    "MATERIALIZATION_SCHEMA",
    "PACKAGE_SCHEMA",
    "PLAN_SCHEMA",
    "PRIMARY_BUDGET",
    "PRIMARY_NAMESPACE_BASES",
    "PRIMARY_PAIR_INDICES",
    "PRIMARY_PHASE",
    "PRIMARY_ROOT_INDICES",
    "ROOT_SCHEMA",
    "ROOT_SEAL_SCHEMA",
    "RUN_ID",
    "build_fresh_quality_plan",
    "build_job_descriptors",
    "build_materialization_receipt",
    "build_root_seal",
    "build_seed_contract",
    "canonical_bytes",
    "canonical_sha256",
    "create_fresh_quality_package",
    "materialize_fresh_quality_roots",
    "schedule_row",
    "schedule_rows",
    "seed_values",
    "validate_fresh_quality_plan",
    "validate_fresh_quality_package",
    "validate_job_descriptor",
    "validate_materialization_receipt",
    "validate_plan_authorization",
    "validate_root",
    "validate_root_seal",
    "validate_seed_contract",
    "write_fresh_quality_plan",
]
