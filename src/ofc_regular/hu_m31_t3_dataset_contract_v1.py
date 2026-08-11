"""Immutable, information-safe M3.1 T3 teacher-dataset contract.

This module freezes the 9,000 paired-hand dataset grid and the artifact
boundaries needed by a later local/Spot executor.  It deliberately performs no
search, training, cloud operation, profile lookup, or runtime activation.

The fixed split is:

* 6,000 train pairs,
* 1,000 safety-fit pairs,
* 1,000 threshold-lock pairs, and
* 1,000 diagnostic-holdout pairs.

Every split preregisters exactly ten percent of its pairs for independent
high-precision confirmation.  The seed ranges are the already-reserved Step 6d
training schedules; no new namespace is invented here.  Pair results persist
semantic ``ActionKey`` identities in canonical order and are independently
reconstructed from ``ActorObservation`` during validation.

The first 25 train pairs form the only initially open shard.  Every other shard
is marked as requiring a replayed smoke-gate receipt.  A dataset plan alone
never authorizes compute: the accepted fresh-quality gate remains an external
precondition for starting the smoke shard.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import hu_m31_t3_step6d_fresh_quality_v1 as fresh_quality
from . import run_hu_m31_t3_step6d_performance_v2 as performance_v2
from .action_key import (
    ACTION_KEY_SCHEMA,
    ActionKey,
    action_key,
    canonicalize_actions,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .action_space import generate_turn_actions
from .hu_infoset import ActorObservation
from .hu_m31_t3_behavior_roots import (
    M31_T3_BEHAVIOR_PROFILES,
    behavior_profile_for_index,
)
from .hu_m31_t3_step6d_contract import (
    EXPECTED_STEP6D_CONTRACT_CANONICAL_SHA256,
    PLANNED_SEED_SET_SHA256,
    SEED_STRIDE,
    STEP6D_RUN_ID,
    TEACHER_NAMESPACE_KEYS,
    schedule_by_name,
    validate_seed_schedule,
)


DATASET_PLAN_SCHEMA = "hu_m31_t3_teacher_dataset_plan_v1"
PAIR_CONTRACT_SCHEMA = "hu_m31_t3_teacher_pair_contract_v1"
PAIR_RESULT_SCHEMA = "hu_m31_t3_teacher_pair_result_v1"
SEAT_ROW_SCHEMA = "hu_m31_t3_teacher_seat_row_v1"
TEACHER_LABEL_SCHEMA = "hu_m31_t3_teacher_label_v1"
SHARD_DONE_SCHEMA = "hu_m31_t3_teacher_shard_done_v1"
SHARD_RESUME_SCHEMA = "hu_m31_t3_teacher_shard_resume_v1"
SMOKE_GATE_SCHEMA = "hu_m31_t3_teacher_smoke_gate_v1"
MERGE_SCHEMA = "hu_m31_t3_teacher_dataset_merge_v1"
RUN_ID = "hu_m31_t3_teacher_dataset_9000_paired_v1"

CURRENT_PROFILE_REGISTRY_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)
ACCEPTED_CANDIDATE_LIBRARY_SHA256 = (
    performance_v2.CANDIDATE02_PERFORMANCE_LOCK_ACCEPTED_CANDIDATE_LIBRARY_SHA256
)
ACCEPTED_FEATURE_ENCODER_SHA256 = fresh_quality.ACCEPTED_FEATURE_ENCODER_SHA256

SPLIT_IDS = (
    "train",
    "safety-fit",
    "threshold-lock",
    "diagnostic-holdout",
)
SPLIT_SPECS = (
    {
        "split": "train",
        "step6d_schedule": "train",
        "paired_hand_count": 6_000,
        "global_pair_offset": 0,
        "consumer": "core_model_fit_only",
        "core_model_fit_allowed": True,
        "risk_safe_fit_allowed": False,
        "threshold_selection_allowed": False,
        "diagnostic_only": False,
    },
    {
        "split": "safety-fit",
        "step6d_schedule": "safety_fit",
        "paired_hand_count": 1_000,
        "global_pair_offset": 6_000,
        "consumer": "risk_and_safe_heads_fit_only",
        "core_model_fit_allowed": False,
        "risk_safe_fit_allowed": True,
        "threshold_selection_allowed": False,
        "diagnostic_only": False,
    },
    {
        "split": "threshold-lock",
        "step6d_schedule": "threshold_lock",
        "paired_hand_count": 1_000,
        "global_pair_offset": 7_000,
        "consumer": "seat_calibration_and_threshold_lock_only",
        "core_model_fit_allowed": False,
        "risk_safe_fit_allowed": False,
        "threshold_selection_allowed": True,
        "diagnostic_only": False,
    },
    {
        "split": "diagnostic-holdout",
        "step6d_schedule": "diagnostic_teacher_holdout",
        "paired_hand_count": 1_000,
        "global_pair_offset": 8_000,
        "consumer": "diagnostic_only_after_model_and_threshold_freeze",
        "core_model_fit_allowed": False,
        "risk_safe_fit_allowed": False,
        "threshold_selection_allowed": False,
        "diagnostic_only": True,
    },
)

TOTAL_PAIRED_HANDS = 9_000
TOTAL_ROOTS = TOTAL_PAIRED_HANDS * 2
CONFIRMATION_MODULUS = 10
CONFIRMATION_RESIDUE = 0
TOTAL_CONFIRMATION_PAIRS = 900
SHARD_PAIR_COUNT = 25
SMOKE_SHARD_ID = "train-0000"
MAX_LEGAL_ACTIONS = 232

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

# Filled after the plan builder was implemented and independently replayed.
EXPECTED_DATASET_PLAN_SHA256 = (
    "519ccfb1b94f86ec0d4983f139d47ca6b4781a94315016f8a9db02690a5faed9"
)

_PAIR_RESULT_KEYS = frozenset(
    {
        "schema",
        "plan_sha256",
        "pair_contract_sha256",
        "shard_id",
        "split",
        "local_pair_index",
        "global_pair_index",
        "root_indices",
        "profile",
        "seeds",
        "confirmation_required",
        "rows",
        "opponent_private_discards_used",
        "realized_deck_tail_used",
        "teacher_values_are_realized_match_ev",
        "current_profile_changed",
    }
)
_SEAT_ROW_KEYS = frozenset(
    {
        "schema",
        "root_index",
        "seat",
        "observation_fingerprint",
        "observation_sha256",
        "observation",
        "action_key_schema",
        "legal_action_keys",
        "legal_action_set_digest",
        "legal_action_order_digest",
        "teacher",
    }
)
_TEACHER_KEYS = frozenset(
    {
        "schema",
        "primary_budget",
        "confirmation_budget",
        "baseline_action_key",
        "selected_action_key",
        "state_value",
        "confirmation_state_value",
        "teacher_value_status",
        "teacher_values_are_realized_match_ev",
        "action_targets",
    }
)
_TARGET_KEYS = frozenset(
    {
        "action_index",
        "action_key",
        "primary_q",
        "primary_delta",
        "primary_rank",
        "confirmation_q",
        "confirmation_delta",
        "confirmation_rank",
    }
)
_PAIR_RECORD_KEYS = frozenset(
    {
        "split",
        "local_pair_index",
        "global_pair_index",
        "shard_id",
        "path",
        "sha256",
        "bytes",
    }
)
_SHARD_RECORD_KEYS = frozenset(
    {
        "shard_id",
        "split",
        "done_sha256",
        "pair_record_aggregate_sha256",
        "pair_count",
    }
)
_SHARD_DONE_KEYS = frozenset(
    {
        "schema",
        "status",
        "plan_sha256",
        "shard_descriptor",
        "shard_descriptor_sha256",
        "pair_count",
        "root_count",
        "seat_counts",
        "confirmation_pair_count",
        "legal_action_count_min",
        "legal_action_count_max",
        "pair_records",
        "pair_record_aggregate_sha256",
        "hidden_information_field_count",
        "unknown_field_count",
        "action_key_mapping_mismatch_count",
        "missing_pair_count",
        "teacher_values_are_realized_match_ev",
        "current_profile_changed",
    }
)
_SMOKE_GATE_KEYS = frozenset(
    {
        "schema",
        "status",
        "decision",
        "plan_sha256",
        "smoke_shard_id",
        "smoke_shard_done_sha256",
        "metrics",
        "gates",
        "all_gates_passed",
        "full_9000_paired_fanout_authorized",
        "teacher_values_are_realized_match_ev",
        "current_profile_changed",
    }
)
_MERGE_KEYS = frozenset(
    {
        "schema",
        "status",
        "plan_sha256",
        "shard_records",
        "shard_record_aggregate_sha256",
        "pair_records",
        "pair_record_aggregate_sha256",
        "paired_hand_count",
        "root_count",
        "split_counts",
        "seat_counts",
        "confirmation_pair_count",
        "global_pair_index_digest",
        "hidden_information_field_count",
        "unknown_field_count",
        "action_key_mapping_mismatch_count",
        "missing_pair_count",
        "duplicate_pair_count",
        "dataset_ready_for_training",
        "teacher_values_are_realized_match_ev",
        "current_profile_changed",
    }
)

_FORBIDDEN_FIELDS = frozenset(
    {
        "opponent_private_discards",
        "true_opponent_private_discards",
        "opponent_discard",
        "true_dead_cards",
        "remaining_deck",
        "realized_deck_tail",
        "deck_tail",
        "draw_pile",
        "future_cards",
        "world_state",
        "replay_truth",
    }
)
_PAIR_FILE_PATTERN = re.compile(r"pair_(\d{6})\.json\Z")


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
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


def _exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{label} fields changed: "
            f"missing={sorted(expected-observed)}, unknown={sorted(observed-expected)}"
        )


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _reject_hidden(value: Any, path: str = "dataset") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in _FORBIDDEN_FIELDS:
                raise ValueError(f"forbidden hidden-information field at {path}.{key}")
            _reject_hidden(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_hidden(child, f"{path}[{index}]")


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
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite immutable artifact: {target}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _seed_values_from_bases(
    bases: Mapping[str, int], count: int
) -> set[int]:
    return {
        int(base) + SEED_STRIDE * index
        for base in bases.values()
        for index in range(count)
    }


def _schedule_seed_values(schedule_name: str) -> set[int]:
    return set(schedule_by_name(schedule_name).values())


def _build_seed_contract() -> dict[str, Any]:
    validate_seed_schedule()
    split_sets: dict[str, set[int]] = {}
    split_records: dict[str, Any] = {}
    for spec in SPLIT_SPECS:
        schedule = schedule_by_name(str(spec["step6d_schedule"]))
        count = int(spec["paired_hand_count"])
        if schedule.index_count != count or schedule.namespace_keys != TEACHER_NAMESPACE_KEYS:
            raise RuntimeError("Step 6d dataset reservation changed")
        values = set(schedule.values())
        if len(values) != count * len(TEACHER_NAMESPACE_KEYS):
            raise RuntimeError("Step 6d dataset seed uniqueness changed")
        split = str(spec["split"])
        split_sets[split] = values
        split_records[split] = {
            "step6d_schedule": schedule.name,
            "namespace_bases": dict(
                zip(schedule.namespace_keys, schedule.namespace_bases, strict=True)
            ),
            "seed_count": len(values),
            "seed_min": min(values),
            "seed_max": max(values),
            "seed_set_sha256": canonical_sha256(sorted(values)),
        }

    data_values: set[int] = set()
    for split in SPLIT_IDS:
        if data_values & split_sets[split]:
            raise RuntimeError("M3.1 dataset split seed sets overlap")
        data_values.update(split_sets[split])

    external_sets = {
        "step6d-performance-development": _schedule_seed_values(
            "performance_development"
        ),
        "step6d-performance-lock": _schedule_seed_values("performance_lock"),
        "step6d-quality-pilot": _schedule_seed_values("quality_pilot"),
        "candidate02-performance-development": _seed_values_from_bases(
            performance_v2.CANDIDATE02_NAMESPACE_BASES, 100
        ),
        "candidate02-performance-lock-rearm1": _seed_values_from_bases(
            performance_v2.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_NAMESPACE_BASES,
            100,
        ),
        "candidate02-performance-lock-rearm2": _seed_values_from_bases(
            performance_v2.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_NAMESPACE_BASES,
            100,
        ),
        "candidate02-performance-lock-v4": _seed_values_from_bases(
            performance_v2.CANDIDATE02_PERFORMANCE_LOCK_V4_NAMESPACE_BASES, 100
        ),
        "fresh-quality-primary": _seed_values_from_bases(
            fresh_quality.PRIMARY_NAMESPACE_BASES,
            len(fresh_quality.PRIMARY_PAIR_INDICES),
        ),
        "fresh-quality-confirmation": _seed_values_from_bases(
            fresh_quality.CONFIRMATION_NAMESPACE_BASES,
            len(fresh_quality.CONFIRMATION_PAIR_INDICES),
        ),
    }
    external_records = {
        name: {
            "seed_count": len(values),
            "seed_set_sha256": canonical_sha256(sorted(values)),
            "dataset_overlap_count": len(data_values & values),
        }
        for name, values in external_sets.items()
    }
    if any(row["dataset_overlap_count"] for row in external_records.values()):
        raise RuntimeError("M3.1 dataset overlaps a performance/quality namespace")
    if len(data_values) != TOTAL_PAIRED_HANDS * len(TEACHER_NAMESPACE_KEYS):
        raise RuntimeError("M3.1 dataset seed count changed")
    return {
        "schema": "hu_m31_t3_teacher_dataset_seed_contract_v1",
        "source_contract_schema": "hu_m31_t3_step6d_disjoint_seed_schedule_v1",
        "source_contract_run_id": STEP6D_RUN_ID,
        "source_contract_canonical_sha256": (
            EXPECTED_STEP6D_CONTRACT_CANONICAL_SHA256
        ),
        "source_planned_seed_set_sha256": PLANNED_SEED_SET_SHA256,
        "seed_stride": SEED_STRIDE,
        "formula": "namespace_base + seed_stride * local_pair_index",
        "namespace_keys": list(TEACHER_NAMESPACE_KEYS),
        "splits": split_records,
        "dataset_seed_count": len(data_values),
        "dataset_seed_min": min(data_values),
        "dataset_seed_max": max(data_values),
        "dataset_seed_set_sha256": canonical_sha256(sorted(data_values)),
        "all_dataset_seeds_unique": True,
        "split_seed_sets_disjoint": True,
        "external_namespace_audit": external_records,
        "all_external_overlap_counts_zero": True,
    }


def _split_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    count = int(spec["paired_hand_count"])
    confirmation_indices = [
        index
        for index in range(count)
        if index % CONFIRMATION_MODULUS == CONFIRMATION_RESIDUE
    ]
    if len(confirmation_indices) != count // CONFIRMATION_MODULUS:
        raise RuntimeError("M3.1 confirmation ratio changed")
    return {
        **dict(spec),
        "confirmation_selection": {
            "method": "local_pair_index_modulo_v1",
            "modulus": CONFIRMATION_MODULUS,
            "residue": CONFIRMATION_RESIDUE,
            "selected_pair_count": len(confirmation_indices),
            "selected_pair_index_sha256": canonical_sha256(confirmation_indices),
            "locked_before_label_read": True,
        },
    }


def _build_shards() -> list[dict[str, Any]]:
    shards: list[dict[str, Any]] = []
    for spec in SPLIT_SPECS:
        split = str(spec["split"])
        count = int(spec["paired_hand_count"])
        offset = int(spec["global_pair_offset"])
        if count % SHARD_PAIR_COUNT:
            raise RuntimeError("M3.1 split is not divisible by the shard size")
        for ordinal, start in enumerate(range(0, count, SHARD_PAIR_COUNT)):
            stop = start + SHARD_PAIR_COUNT
            local_indices = list(range(start, stop))
            global_indices = [offset + index for index in local_indices]
            shard_id = f"{split}-{ordinal:04d}"
            smoke = shard_id == SMOKE_SHARD_ID
            shards.append(
                {
                    "shard_id": shard_id,
                    "split": split,
                    "ordinal": ordinal,
                    "local_pair_start": start,
                    "local_pair_stop_exclusive": stop,
                    "local_pair_indices": local_indices,
                    "global_pair_indices": global_indices,
                    "paired_hand_count": len(local_indices),
                    "root_count": len(local_indices) * 2,
                    "confirmation_pair_count": sum(
                        index % CONFIRMATION_MODULUS == CONFIRMATION_RESIDUE
                        for index in local_indices
                    ),
                    "phase": (
                        "first_25_paired_correctness_smoke"
                        if smoke
                        else "full_fanout_after_smoke_gate"
                    ),
                    "requires_smoke_gate_receipt": not smoke,
                }
            )
    if (
        len(shards) != TOTAL_PAIRED_HANDS // SHARD_PAIR_COUNT
        or sum(row["paired_hand_count"] for row in shards) != TOTAL_PAIRED_HANDS
    ):
        raise RuntimeError("M3.1 shard grid changed")
    return shards


def _build_plan_value() -> dict[str, Any]:
    split_plans = [_split_plan(spec) for spec in SPLIT_SPECS]
    shards = _build_shards()
    return {
        "schema": DATASET_PLAN_SCHEMA,
        "run_id": RUN_ID,
        "purpose": (
            "fixed_information_safe_t3_search_teacher_dataset_for_StreetPolicyNetV1"
        ),
        "candidate_library_sha256": ACCEPTED_CANDIDATE_LIBRARY_SHA256,
        "feature_encoder_sha256": ACCEPTED_FEATURE_ENCODER_SHA256,
        "current_profile_registry_sha256": CURRENT_PROFILE_REGISTRY_SHA256,
        "paired_hand_count": TOTAL_PAIRED_HANDS,
        "root_count": TOTAL_ROOTS,
        "split_order": list(SPLIT_IDS),
        "splits": split_plans,
        "confirmation_contract": {
            "method": "local_pair_index_modulo_v1",
            "modulus": CONFIRMATION_MODULUS,
            "residue": CONFIRMATION_RESIDUE,
            "paired_hand_count": TOTAL_CONFIRMATION_PAIRS,
            "fraction": 0.1,
            "primary_budget": dict(PRIMARY_BUDGET),
            "confirmation_budget": dict(CONFIRMATION_BUDGET),
            "confirmation_may_replace_primary_action": False,
            "locked_before_label_read": True,
        },
        "seed_contract": _build_seed_contract(),
        "action_contract": {
            "schema": ACTION_KEY_SCHEMA,
            "maximum_legal_actions": MAX_LEGAL_ACTIONS,
            "persisted_order": "ascending_ActionKey.sort_key",
            "persistent_identity": "ActionKey_not_positional_generator_index",
            "runtime_resolution": "resolve_ActionKey_against_fresh_legal_actions",
            "illegal_action_mask_required": True,
            "duplicate_action_keys_allowed": False,
            "legal_set_digest_required": True,
            "ordered_mapping_digest_required": True,
        },
        "shard_contract": {
            "paired_hands_per_shard": SHARD_PAIR_COUNT,
            "shard_count": len(shards),
            "pair_artifact_mode": "one_canonical_write_once_file_per_paired_hand",
            "done_published_last": True,
            "resume_revalidates_every_existing_pair": True,
            "merge_requires_exact_grid_no_duplicates_no_gaps": True,
        },
        "shards": shards,
        "smoke_gate": {
            "shard_id": SMOKE_SHARD_ID,
            "paired_hand_count": SHARD_PAIR_COUNT,
            "root_count": SHARD_PAIR_COUNT * 2,
            "fresh_quality_gate_required_before_start": True,
            "fresh_quality_gate_schema": fresh_quality.PLAN_SCHEMA.replace(
                "plan", "gate"
            ),
            "full_fanout_before_smoke_pass_allowed": False,
            "source_replay_required": True,
        },
        "scientific_boundaries": {
            "opponent_private_discards_allowed": False,
            "realized_deck_tail_allowed": False,
            "teacher_values_are_realized_match_ev": False,
            "candidate_selection_and_evaluation_rng_must_be_independent": True,
            "confirmation_rng_must_be_independent": True,
            "threshold_lock_may_modify_model": False,
            "diagnostic_holdout_may_modify_model_or_threshold": False,
            "current_profile_changed": False,
            "named_profile_activation_authorized": False,
            "plan_alone_authorizes_compute": False,
            "full_replacement_authorized": False,
        },
    }


@lru_cache(maxsize=1)
def _expected_plan_bytes() -> bytes:
    return canonical_bytes(_build_plan_value())


def build_dataset_plan() -> dict[str, Any]:
    value = json.loads(_expected_plan_bytes().decode("ascii"))
    return validate_dataset_plan(value)


def validate_dataset_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    plan = dict(value)
    expected = json.loads(_expected_plan_bytes().decode("ascii"))
    if plan != expected:
        raise ValueError("M3.1 T3 dataset plan differs from the frozen replay")
    digest = canonical_sha256(plan)
    if digest != EXPECTED_DATASET_PLAN_SHA256:
        raise ValueError("M3.1 T3 dataset plan SHA-256 changed")
    return plan


def write_dataset_plan(path: str | Path) -> dict[str, Any]:
    plan = build_dataset_plan()
    _write_once(path, plan)
    stored = _read_canonical(path, "stored M3.1 dataset plan")
    if validate_dataset_plan(stored) != plan:
        raise ValueError("stored M3.1 dataset plan differs from source replay")
    return plan


def _split_descriptor(plan: Mapping[str, Any], split: str) -> dict[str, Any]:
    rows = [dict(row) for row in plan["splits"] if row["split"] == split]
    if len(rows) != 1:
        raise KeyError(f"unknown M3.1 dataset split: {split}")
    return rows[0]


def _shard_descriptor(plan: Mapping[str, Any], shard_id: str) -> dict[str, Any]:
    rows = [dict(row) for row in plan["shards"] if row["shard_id"] == shard_id]
    if len(rows) != 1:
        raise KeyError(f"unknown M3.1 dataset shard: {shard_id}")
    return rows[0]


def _pair_contract(
    plan: Mapping[str, Any], split: str, local_pair_index: int
) -> dict[str, Any]:
    index = _integer(local_pair_index, "local_pair_index")
    descriptor = _split_descriptor(plan, split)
    count = int(descriptor["paired_hand_count"])
    if index >= count:
        raise IndexError("local_pair_index is outside the frozen split")
    schedule = schedule_by_name(str(descriptor["step6d_schedule"]))
    seeds = {
        key: base + SEED_STRIDE * index
        for key, base in zip(
            schedule.namespace_keys, schedule.namespace_bases, strict=True
        )
    }
    global_index = int(descriptor["global_pair_offset"]) + index
    matching = [
        row
        for row in plan["shards"]
        if row["split"] == split and index in row["local_pair_indices"]
    ]
    if len(matching) != 1:
        raise RuntimeError("M3.1 pair does not map to exactly one shard")
    confirmation = index % CONFIRMATION_MODULUS == CONFIRMATION_RESIDUE
    return {
        "schema": PAIR_CONTRACT_SCHEMA,
        "plan_sha256": canonical_sha256(plan),
        "shard_id": matching[0]["shard_id"],
        "split": split,
        "step6d_schedule": schedule.name,
        "local_pair_index": index,
        "global_pair_index": global_index,
        "root_indices": [global_index * 2, global_index * 2 + 1],
        "profile": behavior_profile_for_index(global_index),
        "seeds": seeds,
        "confirmation_required": confirmation,
        "primary_budget": dict(PRIMARY_BUDGET),
        "confirmation_budget": (
            dict(CONFIRMATION_BUDGET) if confirmation else None
        ),
    }


def pair_contract(
    plan: Mapping[str, Any], split: str, local_pair_index: int
) -> dict[str, Any]:
    validated = validate_dataset_plan(plan)
    return _pair_contract(validated, split, local_pair_index)


def _legal_action_contract(
    observation: ActorObservation,
) -> tuple[list[str], str, str]:
    actions = canonicalize_actions(
        generate_turn_actions(observation.hero_board, observation.dealt_cards)
    )
    if not actions or len(actions) > MAX_LEGAL_ACTIONS:
        raise ValueError("T3 legal-action count is outside the frozen contract")
    tokens = [action_key(action).to_token() for action in actions]
    return (
        tokens,
        legal_action_set_digest(actions),
        ordered_action_mapping_digest(actions),
    )


def _validate_teacher(
    value: Mapping[str, Any],
    *,
    legal_action_keys: Sequence[str],
    confirmation_required: bool,
) -> dict[str, Any]:
    teacher = dict(value)
    _exact_keys(teacher, _TEACHER_KEYS, "M3.1 teacher label")
    if (
        teacher["schema"] != TEACHER_LABEL_SCHEMA
        or teacher["primary_budget"] != PRIMARY_BUDGET
        or teacher["confirmation_budget"]
        != (CONFIRMATION_BUDGET if confirmation_required else None)
        or teacher["teacher_value_status"]
        != "search_estimate_not_realized_match_ev"
        or teacher["teacher_values_are_realized_match_ev"] is not False
    ):
        raise ValueError("M3.1 teacher provenance changed")

    baseline = ActionKey.from_token(str(teacher["baseline_action_key"])).to_token()
    selected = ActionKey.from_token(str(teacher["selected_action_key"])).to_token()
    if baseline not in legal_action_keys or selected not in legal_action_keys:
        raise ValueError("M3.1 selected/baseline ActionKey is not legal")
    targets = teacher.get("action_targets")
    if not isinstance(targets, list) or len(targets) != len(legal_action_keys):
        raise ValueError("M3.1 teacher target count changed")

    normalized: list[dict[str, Any]] = []
    for index, (raw, expected_key) in enumerate(
        zip(targets, legal_action_keys, strict=True)
    ):
        if not isinstance(raw, Mapping):
            raise ValueError("M3.1 teacher target must be an object")
        target = dict(raw)
        _exact_keys(target, _TARGET_KEYS, "M3.1 action target")
        token = ActionKey.from_token(str(target["action_key"])).to_token()
        if target["action_index"] != index or token != expected_key:
            raise ValueError("M3.1 ActionKey/index mapping drifted")
        primary_q = _finite(target["primary_q"], "primary_q")
        primary_delta = _finite(target["primary_delta"], "primary_delta")
        primary_rank = _integer(target["primary_rank"], "primary_rank")
        confirmation_q = target["confirmation_q"]
        confirmation_delta = target["confirmation_delta"]
        confirmation_rank = target["confirmation_rank"]
        if confirmation_required:
            confirmation_q = _finite(confirmation_q, "confirmation_q")
            confirmation_delta = _finite(
                confirmation_delta, "confirmation_delta"
            )
            confirmation_rank = _integer(
                confirmation_rank, "confirmation_rank"
            )
        elif any(
            item is not None
            for item in (confirmation_q, confirmation_delta, confirmation_rank)
        ):
            raise ValueError("nonconfirmation pair contains confirmation targets")
        normalized.append(
            {
                "action_index": index,
                "action_key": token,
                "primary_q": primary_q,
                "primary_delta": primary_delta,
                "primary_rank": primary_rank,
                "confirmation_q": confirmation_q,
                "confirmation_delta": confirmation_delta,
                "confirmation_rank": confirmation_rank,
            }
        )

    baseline_index = legal_action_keys.index(baseline)
    primary_baseline = normalized[baseline_index]["primary_q"]
    primary_order = sorted(
        range(len(normalized)),
        key=lambda index: (
            -normalized[index]["primary_q"],
            ActionKey.from_token(normalized[index]["action_key"]).sort_key(),
        ),
    )
    primary_rank = {index: rank for rank, index in enumerate(primary_order)}
    for index, target in enumerate(normalized):
        if target["primary_rank"] != primary_rank[index] or not math.isclose(
            target["primary_delta"],
            target["primary_q"] - primary_baseline,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("M3.1 primary Q/delta/rank arithmetic changed")
    if selected != normalized[primary_order[0]]["action_key"]:
        raise ValueError("M3.1 selected ActionKey is not canonical primary argmax")
    state_value = _finite(teacher["state_value"], "state_value")
    if not math.isclose(
        state_value,
        normalized[primary_order[0]]["primary_q"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("M3.1 state value disagrees with primary argmax")

    if confirmation_required:
        confirmation_baseline = normalized[baseline_index]["confirmation_q"]
        confirmation_order = sorted(
            range(len(normalized)),
            key=lambda index: (
                -normalized[index]["confirmation_q"],
                ActionKey.from_token(normalized[index]["action_key"]).sort_key(),
            ),
        )
        ranks = {index: rank for rank, index in enumerate(confirmation_order)}
        for index, target in enumerate(normalized):
            if target["confirmation_rank"] != ranks[index] or not math.isclose(
                target["confirmation_delta"],
                target["confirmation_q"] - confirmation_baseline,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("M3.1 confirmation Q/delta/rank arithmetic changed")
        confirmation_state = _finite(
            teacher["confirmation_state_value"], "confirmation_state_value"
        )
        if not math.isclose(
            confirmation_state,
            normalized[confirmation_order[0]]["confirmation_q"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("M3.1 confirmation state value changed")
    elif teacher["confirmation_state_value"] is not None:
        raise ValueError("nonconfirmation pair contains confirmation state value")
    return teacher


def _validate_seat_row(
    value: Mapping[str, Any],
    *,
    expected_root_index: int,
    expected_seat: str,
    confirmation_required: bool,
) -> tuple[dict[str, Any], int]:
    row = dict(value)
    _exact_keys(row, _SEAT_ROW_KEYS, "M3.1 dataset seat row")
    observation_raw = row.get("observation")
    if not isinstance(observation_raw, Mapping):
        raise ValueError("M3.1 observation is missing")
    observation = ActorObservation.from_dict(observation_raw)
    if observation.to_dict() != dict(observation_raw):
        raise ValueError("M3.1 observation is not in canonical public schema")
    legal_keys, set_digest, order_digest = _legal_action_contract(observation)
    observed_keys = row.get("legal_action_keys")
    if (
        row["schema"] != SEAT_ROW_SCHEMA
        or row["root_index"] != expected_root_index
        or row["seat"] != expected_seat
        or observation.seat != expected_seat
        or observation.to_act_order != expected_seat
        or observation.street != "T3"
        or observation.fingerprint() != row["observation_fingerprint"]
        or canonical_sha256(observation.to_dict()) != row["observation_sha256"]
        or row["action_key_schema"] != ACTION_KEY_SCHEMA
        or observed_keys != legal_keys
        or row["legal_action_set_digest"] != set_digest
        or row["legal_action_order_digest"] != order_digest
    ):
        raise ValueError("M3.1 observation or ActionKey mapping changed")
    teacher = row.get("teacher")
    if not isinstance(teacher, Mapping):
        raise ValueError("M3.1 teacher label is missing")
    _validate_teacher(
        teacher,
        legal_action_keys=legal_keys,
        confirmation_required=confirmation_required,
    )
    _reject_hidden(row, "seat_row")
    return row, len(legal_keys)


def _validate_pair_result(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    expected_split: str | None = None,
    expected_local_pair_index: int | None = None,
) -> dict[str, Any]:
    pair = dict(value)
    _exact_keys(pair, _PAIR_RESULT_KEYS, "M3.1 paired result")
    split = str(pair.get("split"))
    local_index = _integer(pair.get("local_pair_index"), "local_pair_index")
    if expected_split is not None and split != expected_split:
        raise ValueError("M3.1 paired-result split changed")
    if (
        expected_local_pair_index is not None
        and local_index != expected_local_pair_index
    ):
        raise ValueError("M3.1 paired-result index changed")
    contract = _pair_contract(plan, split, local_index)
    rows = pair.get("rows")
    if (
        pair["schema"] != PAIR_RESULT_SCHEMA
        or pair["plan_sha256"] != canonical_sha256(plan)
        or pair["pair_contract_sha256"] != canonical_sha256(contract)
        or pair["shard_id"] != contract["shard_id"]
        or pair["global_pair_index"] != contract["global_pair_index"]
        or pair["root_indices"] != contract["root_indices"]
        or pair["profile"] != contract["profile"]
        or pair["seeds"] != contract["seeds"]
        or pair["confirmation_required"] != contract["confirmation_required"]
        or not isinstance(rows, list)
        or len(rows) != 2
        or pair["opponent_private_discards_used"] is not False
        or pair["realized_deck_tail_used"] is not False
        or pair["teacher_values_are_realized_match_ev"] is not False
        or pair["current_profile_changed"] is not False
    ):
        raise ValueError("M3.1 paired-result provenance changed")
    _validate_seat_row(
        rows[0],
        expected_root_index=contract["root_indices"][0],
        expected_seat="first",
        confirmation_required=contract["confirmation_required"],
    )
    _validate_seat_row(
        rows[1],
        expected_root_index=contract["root_indices"][1],
        expected_seat="second",
        confirmation_required=contract["confirmation_required"],
    )
    _reject_hidden(pair, "pair_result")
    return pair


def validate_pair_result(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    expected_split: str | None = None,
    expected_local_pair_index: int | None = None,
) -> dict[str, Any]:
    validated_plan = validate_dataset_plan(plan)
    return _validate_pair_result(
        value,
        plan=validated_plan,
        expected_split=expected_split,
        expected_local_pair_index=expected_local_pair_index,
    )


def pair_artifact_path(
    shard_directory: str | Path, local_pair_index: int
) -> Path:
    index = _integer(local_pair_index, "local_pair_index")
    return Path(shard_directory) / "pairs" / f"pair_{index:06d}.json"


def write_pair_result(
    *,
    plan: Mapping[str, Any],
    shard_id: str,
    shard_directory: str | Path,
    local_pair_index: int,
    value: Mapping[str, Any],
) -> Path:
    validated_plan = validate_dataset_plan(plan)
    shard = _shard_descriptor(validated_plan, shard_id)
    if local_pair_index not in shard["local_pair_indices"]:
        raise ValueError("pair index is outside the selected shard")
    pair = _validate_pair_result(
        value,
        plan=validated_plan,
        expected_split=shard["split"],
        expected_local_pair_index=local_pair_index,
    )
    if pair["shard_id"] != shard_id:
        raise ValueError("pair result belongs to a different shard")
    path = pair_artifact_path(shard_directory, local_pair_index)
    _write_once(path, pair)
    return path


def _pair_record(
    path: Path,
    pair: Mapping[str, Any],
    *,
    shard_directory: Path,
) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "split": pair["split"],
        "local_pair_index": pair["local_pair_index"],
        "global_pair_index": pair["global_pair_index"],
        "shard_id": pair["shard_id"],
        "path": path.relative_to(shard_directory).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _scan_shard(
    *,
    plan: Mapping[str, Any],
    shard: Mapping[str, Any],
    shard_directory: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    pair_dir = shard_directory / "pairs"
    expected = set(shard["local_pair_indices"])
    observed_paths: dict[int, Path] = {}
    if pair_dir.exists():
        if pair_dir.is_symlink() or not pair_dir.is_dir():
            raise ValueError("M3.1 pair directory is unsafe")
        for path in pair_dir.iterdir():
            if path.is_symlink() or not path.is_file():
                raise ValueError("M3.1 pair artifact is unsafe")
            match = _PAIR_FILE_PATTERN.fullmatch(path.name)
            if match is None:
                raise ValueError(f"unknown M3.1 pair artifact: {path.name}")
            index = int(match.group(1))
            if index not in expected or index in observed_paths:
                raise ValueError("M3.1 shard contains an extra/duplicate pair artifact")
            observed_paths[index] = path
    values: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for index in sorted(observed_paths):
        value = _read_canonical(observed_paths[index], "M3.1 pair artifact")
        pair = _validate_pair_result(
            value,
            plan=plan,
            expected_split=str(shard["split"]),
            expected_local_pair_index=index,
        )
        if pair["shard_id"] != shard["shard_id"]:
            raise ValueError("M3.1 pair artifact belongs to another shard")
        values.append(pair)
        records.append(
            _pair_record(
                observed_paths[index],
                pair,
                shard_directory=shard_directory,
            )
        )
    pending = sorted(expected - set(observed_paths))
    return values, records, pending


def _build_shard_done(
    *,
    plan: Mapping[str, Any],
    shard: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    action_counts = [
        len(row["legal_action_keys"])
        for pair in pairs
        for row in pair["rows"]
    ]
    if not action_counts:
        raise ValueError("M3.1 shard has no validated action rows")
    return {
        "schema": SHARD_DONE_SCHEMA,
        "status": "complete_immutable_shard",
        "plan_sha256": canonical_sha256(plan),
        "shard_descriptor": dict(shard),
        "shard_descriptor_sha256": canonical_sha256(shard),
        "pair_count": len(pairs),
        "root_count": len(pairs) * 2,
        "seat_counts": {"first": len(pairs), "second": len(pairs)},
        "confirmation_pair_count": sum(
            bool(pair["confirmation_required"]) for pair in pairs
        ),
        "legal_action_count_min": min(action_counts),
        "legal_action_count_max": max(action_counts),
        "pair_records": [dict(row) for row in records],
        "pair_record_aggregate_sha256": canonical_sha256(list(records)),
        "hidden_information_field_count": 0,
        "unknown_field_count": 0,
        "action_key_mapping_mismatch_count": 0,
        "missing_pair_count": 0,
        "teacher_values_are_realized_match_ev": False,
        "current_profile_changed": False,
    }


def _validate_done_value(
    value: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    done = dict(value)
    _exact_keys(done, _SHARD_DONE_KEYS, "M3.1 shard DONE")
    if done != dict(expected):
        raise ValueError("M3.1 shard DONE or a bound pair artifact changed")
    return done


def inspect_shard_resume(
    *,
    plan: Mapping[str, Any],
    shard_id: str,
    shard_directory: str | Path,
) -> dict[str, Any]:
    validated_plan = validate_dataset_plan(plan)
    shard = _shard_descriptor(validated_plan, shard_id)
    directory = Path(shard_directory)
    pairs, records, pending = _scan_shard(
        plan=validated_plan, shard=shard, shard_directory=directory
    )
    done_path = directory / "SHARD_DONE.json"
    done_present = done_path.exists()
    if done_present:
        if pending:
            raise ValueError("M3.1 SHARD_DONE exists for an incomplete shard")
        expected = _build_shard_done(
            plan=validated_plan, shard=shard, pairs=pairs, records=records
        )
        _validate_done_value(
            _read_canonical(done_path, "M3.1 SHARD_DONE"), expected=expected
        )
    return {
        "schema": SHARD_RESUME_SCHEMA,
        "shard_id": shard_id,
        "completed_pair_indices": [
            int(pair["local_pair_index"]) for pair in pairs
        ],
        "pending_pair_indices": pending,
        "completed_pair_count": len(pairs),
        "pending_pair_count": len(pending),
        "done_present": done_present,
        "safe_to_resume": not done_present and bool(pending),
        "already_complete": done_present and not pending,
    }


def finalize_shard(
    *,
    plan: Mapping[str, Any],
    shard_id: str,
    shard_directory: str | Path,
) -> dict[str, Any]:
    validated_plan = validate_dataset_plan(plan)
    shard = _shard_descriptor(validated_plan, shard_id)
    directory = Path(shard_directory)
    pairs, records, pending = _scan_shard(
        plan=validated_plan, shard=shard, shard_directory=directory
    )
    if pending:
        raise ValueError(
            f"M3.1 shard has pair gaps: {pending[:10]}"
        )
    done = _build_shard_done(
        plan=validated_plan, shard=shard, pairs=pairs, records=records
    )
    done_path = directory / "SHARD_DONE.json"
    if done_path.exists():
        return _validate_done_value(
            _read_canonical(done_path, "M3.1 SHARD_DONE"), expected=done
        )
    _write_once(done_path, done)
    return _validate_done_value(
        _read_canonical(done_path, "M3.1 SHARD_DONE"), expected=done
    )


def validate_completed_shard(
    *,
    plan: Mapping[str, Any],
    shard_id: str,
    shard_directory: str | Path,
) -> dict[str, Any]:
    validated_plan = validate_dataset_plan(plan)
    shard = _shard_descriptor(validated_plan, shard_id)
    directory = Path(shard_directory)
    pairs, records, pending = _scan_shard(
        plan=validated_plan, shard=shard, shard_directory=directory
    )
    if pending:
        raise ValueError("M3.1 completed shard has pair gaps")
    done_path = directory / "SHARD_DONE.json"
    if not done_path.is_file() or done_path.is_symlink():
        raise ValueError("M3.1 completed shard is missing safe SHARD_DONE")
    expected = _build_shard_done(
        plan=validated_plan, shard=shard, pairs=pairs, records=records
    )
    return _validate_done_value(
        _read_canonical(done_path, "M3.1 SHARD_DONE"), expected=expected
    )


def build_smoke_gate_receipt(
    *,
    plan: Mapping[str, Any],
    smoke_shard_directory: str | Path,
) -> dict[str, Any]:
    validated_plan = validate_dataset_plan(plan)
    done = validate_completed_shard(
        plan=validated_plan,
        shard_id=SMOKE_SHARD_ID,
        shard_directory=smoke_shard_directory,
    )
    metrics = {
        "paired_hand_count": done["pair_count"],
        "root_count": done["root_count"],
        "seat_counts": done["seat_counts"],
        "confirmation_pair_count": done["confirmation_pair_count"],
        "legal_action_count_min": done["legal_action_count_min"],
        "legal_action_count_max": done["legal_action_count_max"],
        "hidden_information_field_count": done[
            "hidden_information_field_count"
        ],
        "unknown_field_count": done["unknown_field_count"],
        "action_key_mapping_mismatch_count": done[
            "action_key_mapping_mismatch_count"
        ],
        "missing_pair_count": done["missing_pair_count"],
    }
    smoke_descriptor = _shard_descriptor(validated_plan, SMOKE_SHARD_ID)
    gates = {
        "exact_first_25_paired_50_roots": (
            metrics["paired_hand_count"] == SHARD_PAIR_COUNT
            and metrics["root_count"] == SHARD_PAIR_COUNT * 2
        ),
        "both_seats_exactly_25": metrics["seat_counts"]
        == {"first": SHARD_PAIR_COUNT, "second": SHARD_PAIR_COUNT},
        "confirmation_subset_matches_preregistered_smoke_count": (
            metrics["confirmation_pair_count"]
            == smoke_descriptor["confirmation_pair_count"]
        ),
        "hidden_unknown_action_mapping_missing_zero": all(
            metrics[field] == 0
            for field in (
                "hidden_information_field_count",
                "unknown_field_count",
                "action_key_mapping_mismatch_count",
                "missing_pair_count",
            )
        ),
        "legal_action_count_within_contract": (
            1
            <= metrics["legal_action_count_min"]
            <= metrics["legal_action_count_max"]
            <= MAX_LEGAL_ACTIONS
        ),
        "teacher_values_not_realized_match_ev": (
            done["teacher_values_are_realized_match_ev"] is False
        ),
        "current_profile_unchanged": done["current_profile_changed"] is False,
    }
    passed = all(gates.values())
    return {
        "schema": SMOKE_GATE_SCHEMA,
        "status": "pass" if passed else "no_go",
        "decision": (
            "open_remaining_8975_paired_fanout"
            if passed
            else "stop_before_full_fanout"
        ),
        "plan_sha256": canonical_sha256(validated_plan),
        "smoke_shard_id": SMOKE_SHARD_ID,
        "smoke_shard_done_sha256": canonical_sha256(done),
        "metrics": metrics,
        "gates": gates,
        "all_gates_passed": passed,
        "full_9000_paired_fanout_authorized": passed,
        "teacher_values_are_realized_match_ev": False,
        "current_profile_changed": False,
    }


def validate_smoke_gate_receipt(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    smoke_shard_directory: str | Path,
) -> dict[str, Any]:
    receipt = dict(value)
    _exact_keys(receipt, _SMOKE_GATE_KEYS, "M3.1 smoke gate")
    expected = build_smoke_gate_receipt(
        plan=plan, smoke_shard_directory=smoke_shard_directory
    )
    if receipt != expected:
        raise ValueError("M3.1 smoke gate differs from source replay")
    return receipt


def write_smoke_gate_receipt(
    *,
    plan: Mapping[str, Any],
    smoke_shard_directory: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    receipt = build_smoke_gate_receipt(
        plan=plan, smoke_shard_directory=smoke_shard_directory
    )
    if not receipt["all_gates_passed"]:
        raise ValueError("M3.1 smoke shard did not pass")
    _write_once(output_path, receipt)
    stored = _read_canonical(output_path, "stored M3.1 smoke gate")
    return validate_smoke_gate_receipt(
        stored, plan=plan, smoke_shard_directory=smoke_shard_directory
    )


def validate_shard_start_authorization(
    *,
    plan: Mapping[str, Any],
    shard_id: str,
    smoke_gate_receipt: Mapping[str, Any] | None = None,
    smoke_shard_directory: str | Path | None = None,
) -> dict[str, Any]:
    validated_plan = validate_dataset_plan(plan)
    shard = _shard_descriptor(validated_plan, shard_id)
    if shard_id == SMOKE_SHARD_ID:
        return {
            "shard_id": shard_id,
            "scope": "first_25_paired_smoke_only",
            "authorized_by_dataset_smoke_gate": False,
        }
    if smoke_gate_receipt is None or smoke_shard_directory is None:
        raise ValueError("full fanout requires a replayed M3.1 smoke-gate receipt")
    receipt = validate_smoke_gate_receipt(
        smoke_gate_receipt,
        plan=validated_plan,
        smoke_shard_directory=smoke_shard_directory,
    )
    if (
        not receipt["all_gates_passed"]
        or not receipt["full_9000_paired_fanout_authorized"]
        or not shard["requires_smoke_gate_receipt"]
    ):
        raise ValueError("M3.1 smoke gate does not authorize this shard")
    return {
        "shard_id": shard_id,
        "scope": "remaining_8975_paired_after_smoke",
        "authorized_by_dataset_smoke_gate": True,
        "smoke_gate_receipt_sha256": canonical_sha256(receipt),
    }


def _expected_pair_index() -> dict[int, tuple[str, int, str]]:
    plan = json.loads(_expected_plan_bytes().decode("ascii"))
    result: dict[int, tuple[str, int, str]] = {}
    for shard in plan["shards"]:
        for local_index, global_index in zip(
            shard["local_pair_indices"],
            shard["global_pair_indices"],
            strict=True,
        ):
            if global_index in result:
                raise RuntimeError("frozen M3.1 plan contains duplicate global pairs")
            result[global_index] = (
                shard["split"],
                local_index,
                shard["shard_id"],
            )
    return result


def validate_pair_record_index(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(records) != TOTAL_PAIRED_HANDS:
        raise ValueError("M3.1 merge pair-record count has gaps or extras")
    normalized: list[dict[str, Any]] = []
    global_indices: list[int] = []
    expected = _expected_pair_index()
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ValueError("M3.1 pair record must be an object")
        record = dict(raw)
        _exact_keys(record, _PAIR_RECORD_KEYS, "M3.1 pair record")
        global_index = _integer(record["global_pair_index"], "global_pair_index")
        local_index = _integer(record["local_pair_index"], "local_pair_index")
        expected_identity = expected.get(global_index)
        if expected_identity != (
            record["split"],
            local_index,
            record["shard_id"],
        ):
            raise ValueError("M3.1 pair record does not match the frozen grid")
        if record["path"] != f"pairs/pair_{local_index:06d}.json":
            raise ValueError("M3.1 pair record path changed")
        if not _is_sha256(record["sha256"]):
            raise ValueError("M3.1 pair record SHA-256 is invalid")
        _integer(record["bytes"], "pair record bytes", minimum=1)
        normalized.append(record)
        global_indices.append(global_index)
    if len(set(global_indices)) != len(global_indices):
        raise ValueError("M3.1 merge contains duplicate paired hands")
    expected_indices = set(range(TOTAL_PAIRED_HANDS))
    observed_indices = set(global_indices)
    if observed_indices != expected_indices:
        raise ValueError(
            "M3.1 merge paired-hand gap/extra: "
            f"missing={sorted(expected_indices-observed_indices)[:10]}, "
            f"extra={sorted(observed_indices-expected_indices)[:10]}"
        )
    if global_indices != sorted(global_indices):
        raise ValueError("M3.1 merge pair records are not in canonical global order")
    split_counts = {
        split: sum(record["split"] == split for record in normalized)
        for split in SPLIT_IDS
    }
    if split_counts != {
        str(spec["split"]): int(spec["paired_hand_count"]) for spec in SPLIT_SPECS
    }:
        raise ValueError("M3.1 merge split counts changed")
    return {
        "paired_hand_count": len(normalized),
        "split_counts": split_counts,
        "global_pair_index_digest": canonical_sha256(global_indices),
        "pair_record_aggregate_sha256": canonical_sha256(normalized),
    }


def build_merge_manifest(
    *,
    plan: Mapping[str, Any],
    shard_directories: Mapping[str, str | Path],
) -> dict[str, Any]:
    validated_plan = validate_dataset_plan(plan)
    expected_shards = {row["shard_id"] for row in validated_plan["shards"]}
    if set(shard_directories) != expected_shards:
        raise ValueError("M3.1 merge shard grid has missing/unknown shards")
    resolved = [Path(path).resolve() for path in shard_directories.values()]
    if len(set(resolved)) != len(resolved):
        raise ValueError("M3.1 merge reuses a shard directory")

    shard_records: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []
    confirmation_count = 0
    for shard in validated_plan["shards"]:
        shard_id = str(shard["shard_id"])
        directory = Path(shard_directories[shard_id])
        done = validate_completed_shard(
            plan=validated_plan,
            shard_id=shard_id,
            shard_directory=directory,
        )
        shard_records.append(
            {
                "shard_id": shard_id,
                "split": shard["split"],
                "done_sha256": canonical_sha256(done),
                "pair_record_aggregate_sha256": done[
                    "pair_record_aggregate_sha256"
                ],
                "pair_count": done["pair_count"],
            }
        )
        pair_records.extend(dict(row) for row in done["pair_records"])
        confirmation_count += int(done["confirmation_pair_count"])
    pair_records.sort(key=lambda row: int(row["global_pair_index"]))
    index = validate_pair_record_index(pair_records)
    return {
        "schema": MERGE_SCHEMA,
        "status": "complete_immutable_9000_paired_dataset_index",
        "plan_sha256": canonical_sha256(validated_plan),
        "shard_records": shard_records,
        "shard_record_aggregate_sha256": canonical_sha256(shard_records),
        "pair_records": pair_records,
        "pair_record_aggregate_sha256": index["pair_record_aggregate_sha256"],
        "paired_hand_count": index["paired_hand_count"],
        "root_count": TOTAL_ROOTS,
        "split_counts": index["split_counts"],
        "seat_counts": {"first": TOTAL_PAIRED_HANDS, "second": TOTAL_PAIRED_HANDS},
        "confirmation_pair_count": confirmation_count,
        "global_pair_index_digest": index["global_pair_index_digest"],
        "hidden_information_field_count": 0,
        "unknown_field_count": 0,
        "action_key_mapping_mismatch_count": 0,
        "missing_pair_count": 0,
        "duplicate_pair_count": 0,
        "dataset_ready_for_training": confirmation_count
        == TOTAL_CONFIRMATION_PAIRS,
        "teacher_values_are_realized_match_ev": False,
        "current_profile_changed": False,
    }


def validate_merge_manifest(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    shard_directories: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    merge = dict(value)
    _exact_keys(merge, _MERGE_KEYS, "M3.1 dataset merge")
    validated_plan = validate_dataset_plan(plan)
    records = merge.get("pair_records")
    if not isinstance(records, list):
        raise ValueError("M3.1 merge pair records are missing")
    shard_records = merge.get("shard_records")
    if not isinstance(shard_records, list):
        raise ValueError("M3.1 merge shard records are missing")
    expected_shards = {
        row["shard_id"]: row for row in validated_plan["shards"]
    }
    observed_shards: list[str] = []
    for raw in shard_records:
        if not isinstance(raw, Mapping):
            raise ValueError("M3.1 merge shard record must be an object")
        record = dict(raw)
        _exact_keys(record, _SHARD_RECORD_KEYS, "M3.1 merge shard record")
        shard_id = str(record["shard_id"])
        descriptor = expected_shards.get(shard_id)
        if (
            descriptor is None
            or record["split"] != descriptor["split"]
            or record["pair_count"] != descriptor["paired_hand_count"]
            or not _is_sha256(record["done_sha256"])
            or not _is_sha256(record["pair_record_aggregate_sha256"])
        ):
            raise ValueError("M3.1 merge shard record changed")
        observed_shards.append(shard_id)
    if (
        len(observed_shards) != len(expected_shards)
        or len(set(observed_shards)) != len(observed_shards)
        or set(observed_shards) != set(expected_shards)
    ):
        raise ValueError("M3.1 merge shard records have duplicates or gaps")
    index = validate_pair_record_index(records)
    if (
        merge["schema"] != MERGE_SCHEMA
        or merge["status"] != "complete_immutable_9000_paired_dataset_index"
        or merge["plan_sha256"] != canonical_sha256(validated_plan)
        or merge["pair_record_aggregate_sha256"]
        != index["pair_record_aggregate_sha256"]
        or merge["paired_hand_count"] != TOTAL_PAIRED_HANDS
        or merge["root_count"] != TOTAL_ROOTS
        or merge["split_counts"] != index["split_counts"]
        or merge["seat_counts"]
        != {"first": TOTAL_PAIRED_HANDS, "second": TOTAL_PAIRED_HANDS}
        or merge["confirmation_pair_count"] != TOTAL_CONFIRMATION_PAIRS
        or merge["global_pair_index_digest"]
        != index["global_pair_index_digest"]
        or merge["shard_record_aggregate_sha256"]
        != canonical_sha256(merge["shard_records"])
        or any(
            merge[field] != 0
            for field in (
                "hidden_information_field_count",
                "unknown_field_count",
                "action_key_mapping_mismatch_count",
                "missing_pair_count",
                "duplicate_pair_count",
            )
        )
        or merge["dataset_ready_for_training"] is not True
        or merge["teacher_values_are_realized_match_ev"] is not False
        or merge["current_profile_changed"] is not False
    ):
        raise ValueError("M3.1 dataset merge contract changed")
    if shard_directories is not None:
        expected = build_merge_manifest(
            plan=validated_plan, shard_directories=shard_directories
        )
        if merge != expected:
            raise ValueError("M3.1 merge differs from shard source replay")
    _reject_hidden(merge, "dataset_merge")
    return merge


def write_merge_manifest(
    *,
    plan: Mapping[str, Any],
    shard_directories: Mapping[str, str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    merge = build_merge_manifest(
        plan=plan, shard_directories=shard_directories
    )
    _write_once(output_path, merge)
    stored = _read_canonical(output_path, "stored M3.1 dataset merge")
    return validate_merge_manifest(
        stored, plan=plan, shard_directories=shard_directories
    )


__all__ = [
    "ACCEPTED_CANDIDATE_LIBRARY_SHA256",
    "ACCEPTED_FEATURE_ENCODER_SHA256",
    "CONFIRMATION_BUDGET",
    "CURRENT_PROFILE_REGISTRY_SHA256",
    "DATASET_PLAN_SCHEMA",
    "EXPECTED_DATASET_PLAN_SHA256",
    "MAX_LEGAL_ACTIONS",
    "MERGE_SCHEMA",
    "PAIR_RESULT_SCHEMA",
    "PRIMARY_BUDGET",
    "RUN_ID",
    "SHARD_PAIR_COUNT",
    "SMOKE_GATE_SCHEMA",
    "SMOKE_SHARD_ID",
    "SPLIT_IDS",
    "TOTAL_CONFIRMATION_PAIRS",
    "TOTAL_PAIRED_HANDS",
    "TOTAL_ROOTS",
    "build_dataset_plan",
    "build_merge_manifest",
    "build_smoke_gate_receipt",
    "canonical_bytes",
    "canonical_sha256",
    "finalize_shard",
    "inspect_shard_resume",
    "pair_artifact_path",
    "pair_contract",
    "validate_completed_shard",
    "validate_dataset_plan",
    "validate_merge_manifest",
    "validate_pair_record_index",
    "validate_pair_result",
    "validate_shard_start_authorization",
    "validate_smoke_gate_receipt",
    "write_dataset_plan",
    "write_merge_manifest",
    "write_pair_result",
    "write_smoke_gate_receipt",
]
