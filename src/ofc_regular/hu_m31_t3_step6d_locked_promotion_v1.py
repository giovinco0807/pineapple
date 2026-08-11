"""Immutable locked-population/ABR promotion foundation for M3.1 T3.

The module consumes realized paired hand counterfactuals only.  It freezes the
five-opponent population, one direct HU-score approximate response plus two
exploitative stress families, disjoint
locked seed namespaces, artifact bindings, merge semantics, and promotion
gates.  It deliberately contains no policy factory, cloud client, training
entry point, profile registration, or ``current`` resolution.

The generic statistical primitives are reused from ``evaluate_hu_m4_population``
because their hand-seed cluster unit, realized gain-per-fire ratio CI, and loss
tail definitions are street-independent.  T3 row validation and all scientific
authorization remain local to this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from .action_key import ACTION_KEY_SCHEMA, ActionKey
from . import evaluate_hu_m4_population as population_oracle
from . import run_hu_m31_t3_step6d_performance as step6d_v1


PLAN_SCHEMA = "hu_m31_t3_step6d_locked_promotion_plan_v1"
THRESHOLD_LOCK_SCHEMA = "hu_m31_t3_step6d_threshold_lock_v1"
ROW_SCHEMA = "hu_m31_t3_step6d_locked_promotion_hand_v1"
SHARD_SCHEMA = "hu_m31_t3_step6d_locked_promotion_shard_v1"
MERGE_SCHEMA = "hu_m31_t3_step6d_locked_promotion_merge_v1"
GATE_SCHEMA = "hu_m31_t3_step6d_locked_promotion_gate_v1"

LOCKED_POPULATION = "locked_population"
LOCKED_ABR = "locked_abr"
SEED_STRIDE = 1_000_003
POPULATION_SEED_COUNT = 1_000
ABR_SEED_COUNT = 500
SEATS = ("first", "second")

OPPONENT_DESCRIPTORS = (
    {
        "opponent_id": "stage19_p0",
        "role": "legacy_strong_opening_chain",
        "policy_source": "explicit_named_profile",
        "runtime_binding": "policy_registry_plus_evaluation_runtime_closure",
        "score_perspective": "candidate_hero_hu_score",
    },
    {
        "opponent_id": "stage9f_p2",
        "role": "selective_t2_override",
        "policy_source": "explicit_named_profile",
        "runtime_binding": "policy_registry_plus_evaluation_runtime_closure",
        "score_perspective": "candidate_hero_hu_score",
    },
    {
        "opponent_id": "stage7_m5_r10",
        "role": "frozen_t3_runtime_baseline",
        "policy_source": "explicit_named_profile",
        "runtime_binding": "policy_registry_plus_evaluation_runtime_closure",
        "score_perspective": "candidate_hero_hu_score",
    },
    {
        "opponent_id": "stage3_baseline",
        "role": "conservative_legacy_comparator",
        "policy_source": "explicit_named_profile",
        "runtime_binding": "policy_registry_plus_evaluation_runtime_closure",
        "score_perspective": "candidate_hero_hu_score",
    },
    {
        "opponent_id": "random_exact_final",
        "role": "off_policy_random_with_exact_final",
        "policy_source": "explicit_named_profile",
        "runtime_binding": "policy_registry_plus_evaluation_runtime_closure",
        "score_perspective": "candidate_hero_hu_score",
    },
)
ABR_DESCRIPTORS = (
    {
        "response_id": "greedy_search_response",
        "objective": (
            "direct_hu_score_approximate_best_response_"
            "not_exact_or_nashconv_bound"
        ),
        "family": "search_greedy",
        "runtime_binding": "evaluation_runtime_closure",
        "score_perspective": "candidate_hero_hu_score",
    },
    {
        "response_id": "foul_pressure_response",
        "objective": (
            "foul_and_scoop_exploitative_stress_"
            "not_direct_best_response"
        ),
        "family": "foul_pressure",
        "runtime_binding": "evaluation_runtime_closure",
        "score_perspective": "candidate_hero_hu_score",
    },
    {
        "response_id": "royalty_denial_response",
        "objective": (
            "royalty_and_fantasyland_denial_exploitative_stress_"
            "not_direct_best_response"
        ),
        "family": "royalty_denial",
        "runtime_binding": "evaluation_runtime_closure",
        "score_perspective": "candidate_hero_hu_score",
    },
)
POPULATION_NAMESPACE_BASES = {
    "hand": 640_108_071_901,
    "actor_policy": 642_108_071_901,
    "opponent_policy": 644_108_071_901,
    "evaluation": 646_108_071_901,
    "child": 648_108_071_901,
    "confirmation": 650_108_071_901,
}
ABR_NAMESPACE_BASES = {
    "hand": 670_108_071_901,
    "actor_policy": 672_108_071_901,
    "opponent_policy": 674_108_071_901,
    "evaluation": 676_108_071_901,
    "child": 678_108_071_901,
    "confirmation": 680_108_071_901,
}
PROMOTION_THRESHOLDS = {
    "invalid_counterfactuals_max": 0,
    "nonfire_cancellation_mismatches_max": 0,
    "nonfire_nonzero_deltas_max": 0,
    "nonfire_cancellation_unknown_max": 0,
    "valid_overrides_min": 300,
    "valid_overrides_each_seat_min": 100,
    "realized_gain_per_override_ci95_low_min_exclusive": 0.0,
    "paired_delta_ev_per_hand_ci95_low_min_exclusive": 0.0,
    "first_seat_delta_ev_per_hand_ci95_low_min_exclusive": 0.0,
    "second_seat_delta_ev_per_hand_ci95_low_min_exclusive": 0.0,
    "false_positive_override_rate_max": 0.30,
    "false_positive_override_rate_each_seat_max": 0.35,
    "override_loss_p95_max": 25.0,
    "override_loss_p99_max": 40.0,
    "override_loss_max": 50.0,
    "each_opponent_delta_ev_per_hand_min": -0.005,
    "each_opponent_delta_ev_per_hand_ci95_low_min": -0.02,
    "abr_candidate_minus_baseline_ev_per_hand_min": -0.005,
    "abr_candidate_minus_baseline_ci95_low_min": -0.02,
    "minimum_direct_response_or_stress_family_ev_per_hand_min": -0.01,
}
METRIC_ORACLE = (
    "evaluate_hu_m4_population:"
    "hand_seed_cluster_mean+cluster_ratio_influence+linear_loss_tail_v1"
)

_SHA = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,95}$")
_MODEL_RECORD_KEYS = frozenset(
    {"artifact_id", "filename", "sha256", "bytes"}
)
_THRESHOLD_KEYS = frozenset(
    {
        "schema",
        "status",
        "model_artifact_id",
        "model_sha256",
        "state_action_input_schema_sha256",
        "action_key_schema",
        "seat_thresholds",
        "seat_enabled",
        "source_training_threshold_lock_sha256",
        "source_checkpoint_bundle_identity_sha256",
        "locked_on_threshold_holdout_only",
        "model_change_requires_new_lock",
        "teacher_ev_lcb_is_runtime_gate",
        "top1_accuracy_is_promotion_gate",
        "current_profile_changed",
        "runtime_activated",
    }
)
_ARTIFACT_BINDING_KEYS = frozenset(
    {
        "model",
        "threshold_lock",
        "threshold_lock_content",
        "policy_registry",
        "evaluation_runtime_closure",
    }
)
_SEED_CONTRACT_KEYS = frozenset(
    {
        "seed_stride",
        "population_seed_count",
        "abr_seed_count",
        "population_namespace_bases",
        "abr_namespace_bases",
        "population_seed_set_sha256",
        "abr_seed_set_sha256",
        "combined_seed_set_sha256",
        "all_namespace_values_unique",
        "population_abr_disjoint",
        "paired_seat_swap_reuses_role_seeds",
        "candidate_baseline_common_randomness",
        "posthoc_extension_allowed",
    }
)
_PLAN_KEYS = frozenset(
    {
        "schema",
        "status",
        "plan_id",
        "artifact_binding",
        "opponents",
        "abr_families",
        "seed_contract",
        "evaluation_contract",
        "promotion_thresholds",
        "metric_oracle",
        "locked_before_content_read",
        "cloud_execution_started",
        "training_eligible",
        "named_profile_added",
        "current_profile_changed",
        "runtime_activated",
        "full_replacement_enabled",
    }
)
_ROW_KEYS = frozenset(
    {
        "schema",
        "schedule",
        "entity_id",
        "seed_index",
        "hand_seed",
        "actor_policy_seed",
        "opponent_policy_seed",
        "evaluation_seed",
        "child_seed",
        "confirmation_seed",
        "seat",
        "action_key_schema",
        "candidate_action_key",
        "baseline_action_key",
        "legal_action_mapping_sha256",
        "candidate_trajectory_sha256",
        "baseline_trajectory_sha256",
        "score_perspective",
        "candidate_score",
        "baseline_score",
        "delta",
        "override_log_valid",
        "override_fired",
        "nonfire_action_key_identical",
        "nonfire_trajectory_identical",
        "nonfire_cancellation_valid",
        "model_sha256",
        "threshold_lock_sha256",
        "policy_registry_sha256",
        "evaluation_runtime_closure_sha256",
        "counterfactual_basis",
        "teacher_values_used",
        "opponent_private_discards_used",
        "current_profile_resolved",
        "current_profile_changed",
    }
)
_SHARD_KEYS = frozenset(
    {
        "schema",
        "status",
        "shard_id",
        "plan_sha256",
        "model_sha256",
        "threshold_lock_sha256",
        "policy_registry_sha256",
        "evaluation_runtime_closure_sha256",
        "schedules",
        "entity_ids",
        "row_count",
        "rows",
        "row_aggregate_sha256",
        "realized_match_counterfactuals_only",
        "teacher_values_used",
        "promotion_decision_applied",
        "current_profile_changed",
    }
)
_SOURCE_RECORD_KEYS = frozenset({"path", "sha256", "bytes", "shard_id"})
_MERGE_KEYS = frozenset(
    {
        "schema",
        "status",
        "plan_sha256",
        "model_sha256",
        "threshold_lock_sha256",
        "policy_registry_sha256",
        "evaluation_runtime_closure_sha256",
        "source_shards",
        "source_shard_aggregate_sha256",
        "row_count",
        "coverage",
        "integrity",
        "population",
        "abr",
        "metric_oracle",
        "teacher_values_used",
        "promotion_decision_applied",
        "named_profile_added",
        "current_profile_changed",
        "runtime_activated",
    }
)
_GATE_KEYS = frozenset(
    {
        "schema",
        "status",
        "decision",
        "plan_sha256",
        "merge_sha256",
        "thresholds",
        "gates",
        "all_gates_passed",
        "scientific_promotion_passed",
        "separate_opt_in_profile_candidate_authorized",
        "teacher_values_used",
        "named_profile_added",
        "current_profile_changed",
        "runtime_activated",
        "full_replacement_enabled",
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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(f"{label} fields changed: missing={missing}, extra={extra}")


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _read_canonical(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path).resolve()
    if Path(path).is_symlink() or not source.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not a canonical object")
    return value


def _write_once(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path)
    raw = canonical_bytes(value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.read_bytes() != raw
        ):
            raise FileExistsError(
                f"immutable locked-promotion output changed: {destination}"
            ) from None
    return destination


def seed_values(schedule: str, index: int) -> dict[str, int]:
    if schedule == LOCKED_POPULATION:
        bases, count = POPULATION_NAMESPACE_BASES, POPULATION_SEED_COUNT
    elif schedule == LOCKED_ABR:
        bases, count = ABR_NAMESPACE_BASES, ABR_SEED_COUNT
    else:
        raise ValueError("locked-promotion schedule changed")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < count:
        raise ValueError("locked-promotion seed index is outside the frozen grid")
    return {name: base + SEED_STRIDE * index for name, base in bases.items()}


@lru_cache(maxsize=1)
def _seed_contract_bytes() -> bytes:
    population_values = [
        value
        for index in range(POPULATION_SEED_COUNT)
        for value in seed_values(LOCKED_POPULATION, index).values()
    ]
    abr_values = [
        value
        for index in range(ABR_SEED_COUNT)
        for value in seed_values(LOCKED_ABR, index).values()
    ]
    combined = population_values + abr_values
    value = {
        "seed_stride": SEED_STRIDE,
        "population_seed_count": POPULATION_SEED_COUNT,
        "abr_seed_count": ABR_SEED_COUNT,
        "population_namespace_bases": dict(POPULATION_NAMESPACE_BASES),
        "abr_namespace_bases": dict(ABR_NAMESPACE_BASES),
        "population_seed_set_sha256": canonical_sha256(population_values),
        "abr_seed_set_sha256": canonical_sha256(abr_values),
        "combined_seed_set_sha256": canonical_sha256(combined),
        "all_namespace_values_unique": len(combined) == len(set(combined)),
        "population_abr_disjoint": not (set(population_values) & set(abr_values)),
        "paired_seat_swap_reuses_role_seeds": True,
        "candidate_baseline_common_randomness": True,
        "posthoc_extension_allowed": False,
    }
    return canonical_bytes(value)


def _seed_contract() -> dict[str, Any]:
    # Return a fresh object so callers cannot mutate the process-wide cache.
    return json.loads(_seed_contract_bytes().decode("ascii"))


def _validate_threshold_lock(
    value: Mapping[str, Any], *, expected_model_sha256: str
) -> dict[str, Any]:
    lock = deepcopy(dict(value))
    _exact_keys(lock, _THRESHOLD_KEYS, "T3 threshold lock")
    thresholds = lock.get("seat_thresholds")
    if not isinstance(thresholds, Mapping) or set(thresholds) != set(SEATS):
        raise ValueError("T3 threshold lock seat calibration changed")
    enabled = lock.get("seat_enabled")
    if (
        not isinstance(enabled, Mapping)
        or set(enabled) != set(SEATS)
        or any(not isinstance(enabled[seat], bool) for seat in SEATS)
    ):
        raise ValueError("T3 threshold lock seat enablement changed")
    for seat in SEATS:
        threshold = _finite(thresholds[seat], f"{seat} threshold")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("T3 safe threshold is outside [0,1]")
    if (
        lock["schema"] != THRESHOLD_LOCK_SCHEMA
        or lock["status"] != "locked_no_holdout_reselection"
        or not isinstance(lock["model_artifact_id"], str)
        or _SAFE_ID.fullmatch(lock["model_artifact_id"]) is None
        or lock["model_sha256"] != expected_model_sha256
        or _SHA.fullmatch(str(lock["state_action_input_schema_sha256"])) is None
        or lock["action_key_schema"] != ACTION_KEY_SCHEMA
        or _SHA.fullmatch(
            str(lock["source_training_threshold_lock_sha256"])
        )
        is None
        or _SHA.fullmatch(
            str(lock["source_checkpoint_bundle_identity_sha256"])
        )
        is None
        or lock["locked_on_threshold_holdout_only"] is not True
        or lock["model_change_requires_new_lock"] is not True
        or lock["teacher_ev_lcb_is_runtime_gate"] is not False
        or lock["top1_accuracy_is_promotion_gate"] is not False
        or lock["current_profile_changed"] is not False
        or lock["runtime_activated"] is not False
    ):
        raise ValueError("T3 threshold lock boundary changed")
    return lock


def build_locked_promotion_plan(
    *,
    plan_id: str,
    model_artifact_id: str,
    model_path: str | Path,
    expected_model_sha256: str,
    threshold_lock_path: str | Path,
    expected_threshold_lock_sha256: str,
    policy_registry_path: str | Path,
    expected_policy_registry_sha256: str,
    evaluation_runtime_closure_path: str | Path,
    expected_evaluation_runtime_closure_sha256: str,
) -> dict[str, Any]:
    if _SAFE_ID.fullmatch(plan_id) is None or _SAFE_ID.fullmatch(model_artifact_id) is None:
        raise ValueError("locked-promotion plan/model id is unsafe")
    model = Path(model_path).resolve()
    threshold_path = Path(threshold_lock_path).resolve()
    policy_registry = Path(policy_registry_path).resolve()
    runtime_closure = Path(evaluation_runtime_closure_path).resolve()
    if (
        Path(model_path).is_symlink()
        or not model.is_file()
        or Path(threshold_lock_path).is_symlink()
        or not threshold_path.is_file()
        or Path(policy_registry_path).is_symlink()
        or not policy_registry.is_file()
        or Path(evaluation_runtime_closure_path).is_symlink()
        or not runtime_closure.is_file()
    ):
        raise ValueError("locked-promotion artifact is missing or unsafe")
    model_sha = sha256_file(model)
    threshold_sha = sha256_file(threshold_path)
    if (
        model_sha != _require_sha(expected_model_sha256, "trained model")
        or threshold_sha
        != _require_sha(expected_threshold_lock_sha256, "threshold lock")
        or sha256_file(policy_registry)
        != _require_sha(expected_policy_registry_sha256, "policy registry")
        or sha256_file(runtime_closure)
        != _require_sha(
            expected_evaluation_runtime_closure_sha256,
            "evaluation runtime closure",
        )
    ):
        raise ValueError("locked-promotion artifact hash changed")
    lock = _validate_threshold_lock(
        _read_canonical(threshold_path, "threshold lock"),
        expected_model_sha256=model_sha,
    )
    if (
        lock["model_artifact_id"] != model_artifact_id
        or lock["model_sha256"] != model_sha
    ):
        raise ValueError("threshold lock is not bound to the trained model")
    plan = {
        "schema": PLAN_SCHEMA,
        "status": "locked_before_population_and_abr_open",
        "plan_id": plan_id,
        "artifact_binding": {
            "model": {
                "artifact_id": model_artifact_id,
                "filename": model.name,
                "sha256": model_sha,
                "bytes": model.stat().st_size,
            },
            "threshold_lock": {
                "artifact_id": f"{model_artifact_id}.threshold_lock",
                "filename": threshold_path.name,
                "sha256": threshold_sha,
                "bytes": threshold_path.stat().st_size,
            },
            "threshold_lock_content": lock,
            "policy_registry": {
                "artifact_id": "locked-population-policy-registry",
                "filename": policy_registry.name,
                "sha256": sha256_file(policy_registry),
                "bytes": policy_registry.stat().st_size,
            },
            "evaluation_runtime_closure": {
                "artifact_id": "locked-population-evaluation-runtime-closure",
                "filename": runtime_closure.name,
                "sha256": sha256_file(runtime_closure),
                "bytes": runtime_closure.stat().st_size,
            },
        },
        "opponents": [dict(value) for value in OPPONENT_DESCRIPTORS],
        "abr_families": [
            {
                **value,
                "development_schedule": "abr_development",
                "locked_evaluation_schedule": LOCKED_ABR,
                "locked_seed_training_allowed": False,
                "opponent_private_discards_used": False,
            }
            for value in ABR_DESCRIPTORS
        ],
        "seed_contract": _seed_contract(),
        "evaluation_contract": {
            "baseline_profile": "stage7_m5_r10",
            "population_paired_seeds_per_opponent": POPULATION_SEED_COUNT,
            "abr_paired_seeds_per_response": ABR_SEED_COUNT,
            "paired_seat_swap": True,
            "confidence_interval_unit": (
                "hand_seed_cluster_after_opponent_or_response_average"
            ),
            "candidate_baseline_common_randomness": True,
            "policy_registry_sha256": sha256_file(policy_registry),
            "evaluation_runtime_closure_sha256": sha256_file(runtime_closure),
            "score_perspective": "candidate_hero_hu_score",
            "nonfire_complete_trajectory_identity_required": True,
            "realized_match_ev_only": True,
            "teacher_values_are_realized_match_ev": False,
            "holdout_threshold_reselection_allowed": False,
        },
        "promotion_thresholds": dict(PROMOTION_THRESHOLDS),
        "metric_oracle": METRIC_ORACLE,
        "locked_before_content_read": True,
        "cloud_execution_started": False,
        "training_eligible": False,
        "named_profile_added": False,
        "current_profile_changed": False,
        "runtime_activated": False,
        "full_replacement_enabled": False,
    }
    return validate_locked_promotion_plan(plan)


def validate_locked_promotion_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    plan = deepcopy(dict(value))
    _exact_keys(plan, _PLAN_KEYS, "locked-promotion plan")
    binding = plan.get("artifact_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("locked-promotion artifact binding is missing")
    _exact_keys(binding, _ARTIFACT_BINDING_KEYS, "artifact binding")
    for field in (
        "model",
        "threshold_lock",
        "policy_registry",
        "evaluation_runtime_closure",
    ):
        record = binding.get(field)
        if not isinstance(record, Mapping):
            raise ValueError(f"locked-promotion {field} record is missing")
        _exact_keys(record, _MODEL_RECORD_KEYS, f"{field} record")
        _require_sha(record["sha256"], field)
        if (
            _SAFE_ID.fullmatch(str(record["artifact_id"])) is None
            or not isinstance(record["filename"], str)
            or Path(record["filename"]).name != record["filename"]
            or isinstance(record["bytes"], bool)
            or not isinstance(record["bytes"], int)
            or record["bytes"] <= 0
        ):
            raise ValueError(f"locked-promotion {field} record changed")
    lock = _validate_threshold_lock(
        binding["threshold_lock_content"],
        expected_model_sha256=binding["model"]["sha256"],
    )
    lock_bytes = canonical_bytes(lock)
    seed_contract = plan.get("seed_contract")
    if not isinstance(seed_contract, Mapping):
        raise ValueError("locked-promotion seed contract is missing")
    _exact_keys(seed_contract, _SEED_CONTRACT_KEYS, "seed contract")
    expected_abr = [
        {
            **value,
            "development_schedule": "abr_development",
            "locked_evaluation_schedule": LOCKED_ABR,
            "locked_seed_training_allowed": False,
            "opponent_private_discards_used": False,
        }
        for value in ABR_DESCRIPTORS
    ]
    if (
        plan["schema"] != PLAN_SCHEMA
        or plan["status"] != "locked_before_population_and_abr_open"
        or _SAFE_ID.fullmatch(str(plan["plan_id"])) is None
        or lock != binding["threshold_lock_content"]
        or binding["model"]["artifact_id"] != lock["model_artifact_id"]
        or binding["threshold_lock"]["artifact_id"]
        != f"{lock['model_artifact_id']}.threshold_lock"
        or binding["threshold_lock"]["sha256"]
        != hashlib.sha256(lock_bytes).hexdigest()
        or binding["threshold_lock"]["bytes"] != len(lock_bytes)
        or plan["opponents"] != [dict(value) for value in OPPONENT_DESCRIPTORS]
        or plan["abr_families"] != expected_abr
        or seed_contract != _seed_contract()
        or seed_contract["all_namespace_values_unique"] is not True
        or seed_contract["population_abr_disjoint"] is not True
        or plan["evaluation_contract"]
        != {
            "baseline_profile": "stage7_m5_r10",
            "population_paired_seeds_per_opponent": POPULATION_SEED_COUNT,
            "abr_paired_seeds_per_response": ABR_SEED_COUNT,
            "paired_seat_swap": True,
            "confidence_interval_unit": (
                "hand_seed_cluster_after_opponent_or_response_average"
            ),
            "candidate_baseline_common_randomness": True,
            "policy_registry_sha256": binding["policy_registry"]["sha256"],
            "evaluation_runtime_closure_sha256": binding[
                "evaluation_runtime_closure"
            ]["sha256"],
            "score_perspective": "candidate_hero_hu_score",
            "nonfire_complete_trajectory_identity_required": True,
            "realized_match_ev_only": True,
            "teacher_values_are_realized_match_ev": False,
            "holdout_threshold_reselection_allowed": False,
        }
        or plan["promotion_thresholds"] != PROMOTION_THRESHOLDS
        or plan["metric_oracle"] != METRIC_ORACLE
        or plan["locked_before_content_read"] is not True
        or any(
            plan[field] is not False
            for field in (
                "cloud_execution_started",
                "training_eligible",
                "named_profile_added",
                "current_profile_changed",
                "runtime_activated",
                "full_replacement_enabled",
            )
        )
    ):
        raise ValueError("locked-promotion plan boundary changed")
    return plan


def validate_artifact_files(
    plan: Mapping[str, Any],
    *,
    model_path: str | Path,
    threshold_lock_path: str | Path,
    policy_registry_path: str | Path,
    evaluation_runtime_closure_path: str | Path,
) -> None:
    validated = validate_locked_promotion_plan(plan)
    model = Path(model_path).resolve()
    threshold = Path(threshold_lock_path).resolve()
    policy_registry = Path(policy_registry_path).resolve()
    runtime_closure = Path(evaluation_runtime_closure_path).resolve()
    if (
        Path(model_path).is_symlink()
        or not model.is_file()
        or Path(threshold_lock_path).is_symlink()
        or not threshold.is_file()
        or Path(policy_registry_path).is_symlink()
        or not policy_registry.is_file()
        or Path(evaluation_runtime_closure_path).is_symlink()
        or not runtime_closure.is_file()
        or sha256_file(model)
        != validated["artifact_binding"]["model"]["sha256"]
        or model.stat().st_size
        != validated["artifact_binding"]["model"]["bytes"]
        or sha256_file(threshold)
        != validated["artifact_binding"]["threshold_lock"]["sha256"]
        or threshold.stat().st_size
        != validated["artifact_binding"]["threshold_lock"]["bytes"]
        or _read_canonical(threshold, "threshold lock")
        != validated["artifact_binding"]["threshold_lock_content"]
        or sha256_file(policy_registry)
        != validated["artifact_binding"]["policy_registry"]["sha256"]
        or policy_registry.stat().st_size
        != validated["artifact_binding"]["policy_registry"]["bytes"]
        or sha256_file(runtime_closure)
        != validated["artifact_binding"]["evaluation_runtime_closure"]["sha256"]
        or runtime_closure.stat().st_size
        != validated["artifact_binding"]["evaluation_runtime_closure"]["bytes"]
    ):
        raise ValueError("locked-promotion artifact replay changed")


def write_locked_promotion_plan(
    *, plan: Mapping[str, Any], output_path: str | Path
) -> dict[str, Any]:
    validated = validate_locked_promotion_plan(plan)
    _write_once(output_path, validated)
    return validated


def _allowed_entities(schedule: str) -> tuple[str, ...]:
    if schedule == LOCKED_POPULATION:
        return tuple(row["opponent_id"] for row in OPPONENT_DESCRIPTORS)
    if schedule == LOCKED_ABR:
        return tuple(row["response_id"] for row in ABR_DESCRIPTORS)
    raise ValueError("locked-promotion row schedule changed")


def _row_key(row: Mapping[str, Any]) -> tuple[int, str, int, int]:
    schedule_order = 0 if row["schedule"] == LOCKED_POPULATION else 1
    seat_order = 0 if row["seat"] == "first" else 1
    return (
        schedule_order,
        str(row["entity_id"]),
        int(row["seed_index"]),
        seat_order,
    )


def _validate_hand_row_against_plan(
    value: Mapping[str, Any], *, validated_plan: Mapping[str, Any]
) -> dict[str, Any]:
    row = deepcopy(dict(value))
    _exact_keys(row, _ROW_KEYS, "locked-promotion hand row")
    schedule = row.get("schedule")
    index = row.get("seed_index")
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("locked-promotion row seed index changed")
    seeds = seed_values(str(schedule), index)
    candidate = _finite(row.get("candidate_score"), "candidate score")
    baseline = _finite(row.get("baseline_score"), "baseline score")
    delta = _finite(row.get("delta"), "realized delta")
    if row.get("action_key_schema") != ACTION_KEY_SCHEMA:
        raise ValueError("locked-promotion ActionKey schema changed")
    candidate_action = ActionKey.from_token(str(row.get("candidate_action_key")))
    baseline_action = ActionKey.from_token(str(row.get("baseline_action_key")))
    _require_sha(row.get("legal_action_mapping_sha256"), "legal action mapping")
    candidate_trajectory = _require_sha(
        row.get("candidate_trajectory_sha256"), "candidate trajectory"
    )
    baseline_trajectory = _require_sha(
        row.get("baseline_trajectory_sha256"), "baseline trajectory"
    )
    valid = row.get("override_log_valid")
    fired = row.get("override_fired")
    if not isinstance(valid, bool):
        raise ValueError("locked-promotion override validity changed")
    if valid and not isinstance(fired, bool):
        raise ValueError("valid locked-promotion row lacks fire state")
    if not valid and fired is not None:
        raise ValueError("invalid locked-promotion row claims a fire state")
    nonfire_fields = (
        "nonfire_action_key_identical",
        "nonfire_trajectory_identical",
        "nonfire_cancellation_valid",
    )
    if valid and fired is False:
        if any(not isinstance(row[field], bool) for field in nonfire_fields):
            raise ValueError("locked-promotion nonfire evidence is missing")
        expected_action_identity = candidate_action == baseline_action
        expected_trajectory_identity = (
            candidate_trajectory == baseline_trajectory
        )
        expected_cancellation = (
            expected_action_identity
            and expected_trajectory_identity
            and math.isclose(delta, 0.0, rel_tol=0.0, abs_tol=1e-12)
        )
        if (
            row["nonfire_action_key_identical"] is not expected_action_identity
            or row["nonfire_trajectory_identical"]
            is not expected_trajectory_identity
            or row["nonfire_cancellation_valid"] is not expected_cancellation
        ):
            raise ValueError(
                "locked-promotion nonfire evidence differs from digest replay"
            )
    elif valid and fired is True:
        if candidate_action == baseline_action:
            raise ValueError(
                "locked-promotion fired override did not change ActionKey"
            )
    elif any(row[field] is not None for field in nonfire_fields):
        raise ValueError("locked-promotion fired/invalid row has nonfire evidence")
    if (
        row["schema"] != ROW_SCHEMA
        or schedule not in (LOCKED_POPULATION, LOCKED_ABR)
        or row["entity_id"] not in _allowed_entities(str(schedule))
        or row["seat"] not in SEATS
        or row["hand_seed"] != seeds["hand"]
        or row["actor_policy_seed"] != seeds["actor_policy"]
        or row["opponent_policy_seed"] != seeds["opponent_policy"]
        or row["evaluation_seed"] != seeds["evaluation"]
        or row["child_seed"] != seeds["child"]
        or row["confirmation_seed"] != seeds["confirmation"]
        or not math.isclose(delta, candidate - baseline, rel_tol=0.0, abs_tol=1e-12)
        or row["model_sha256"]
        != validated_plan["artifact_binding"]["model"]["sha256"]
        or row["threshold_lock_sha256"]
        != validated_plan["artifact_binding"]["threshold_lock"]["sha256"]
        or row["policy_registry_sha256"]
        != validated_plan["artifact_binding"]["policy_registry"]["sha256"]
        or row["evaluation_runtime_closure_sha256"]
        != validated_plan["artifact_binding"]["evaluation_runtime_closure"][
            "sha256"
        ]
        or row["score_perspective"] != "candidate_hero_hu_score"
        or row["counterfactual_basis"]
        != "same_hand_role_policy_seeds_physical_seat_v1"
        or row["teacher_values_used"] is not False
        or row["opponent_private_discards_used"] is not False
        or row["current_profile_resolved"] is not False
        or row["current_profile_changed"] is not False
    ):
        raise ValueError("locked-promotion hand row boundary changed")
    step6d_v1._reject_hidden(row, "locked_promotion_hand")
    return row


def validate_hand_row(
    value: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    return _validate_hand_row_against_plan(
        value, validated_plan=validate_locked_promotion_plan(plan)
    )


def build_evaluation_shard(
    *,
    plan: Mapping[str, Any],
    shard_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validated_plan = validate_locked_promotion_plan(plan)
    if _SAFE_ID.fullmatch(shard_id) is None or not rows:
        raise ValueError("locked-promotion shard id/rows changed")
    validated_rows = [
        _validate_hand_row_against_plan(row, validated_plan=validated_plan)
        for row in rows
    ]
    if validated_rows != sorted(validated_rows, key=_row_key):
        raise ValueError("locked-promotion shard rows are not canonical")
    keys = [_row_key(row) for row in validated_rows]
    if len(keys) != len(set(keys)):
        raise ValueError("locked-promotion shard duplicates a hand row")
    return {
        "schema": SHARD_SCHEMA,
        "status": "complete_realized_counterfactual_shard",
        "shard_id": shard_id,
        "plan_sha256": canonical_sha256(validated_plan),
        "model_sha256": validated_plan["artifact_binding"]["model"]["sha256"],
        "threshold_lock_sha256": validated_plan["artifact_binding"][
            "threshold_lock"
        ]["sha256"],
        "policy_registry_sha256": validated_plan["artifact_binding"][
            "policy_registry"
        ]["sha256"],
        "evaluation_runtime_closure_sha256": validated_plan[
            "artifact_binding"
        ]["evaluation_runtime_closure"]["sha256"],
        "schedules": sorted({row["schedule"] for row in validated_rows}),
        "entity_ids": sorted({row["entity_id"] for row in validated_rows}),
        "row_count": len(validated_rows),
        "rows": validated_rows,
        "row_aggregate_sha256": canonical_sha256(validated_rows),
        "realized_match_counterfactuals_only": True,
        "teacher_values_used": False,
        "promotion_decision_applied": False,
        "current_profile_changed": False,
    }


def validate_evaluation_shard(
    value: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    validated_plan = validate_locked_promotion_plan(plan)
    shard = deepcopy(dict(value))
    _exact_keys(shard, _SHARD_KEYS, "locked-promotion shard")
    rows = shard.get("rows")
    if not isinstance(rows, list):
        raise ValueError("locked-promotion shard rows are missing")
    expected = build_evaluation_shard(
        plan=validated_plan,
        shard_id=str(shard.get("shard_id")),
        rows=rows,
    )
    if shard != expected:
        raise ValueError("locked-promotion shard differs from row replay")
    return shard


def write_evaluation_shard(
    *,
    plan: Mapping[str, Any],
    shard_id: str,
    rows: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> dict[str, Any]:
    shard = build_evaluation_shard(plan=plan, shard_id=shard_id, rows=rows)
    _write_once(output_path, shard)
    return shard


def _pair_values(
    rows: Sequence[Mapping[str, Any]], *, score_key: str
) -> dict[tuple[str, int], float]:
    grouped: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["entity_id"]), int(row["seed_index"])), {}
        )[str(row["seat"])] = row
    result: dict[tuple[str, int], float] = {}
    for key, by_seat in grouped.items():
        if set(by_seat) != set(SEATS):
            raise ValueError("locked-promotion paired seat swap is incomplete")
        result[key] = (
            float(by_seat["first"][score_key])
            + float(by_seat["second"][score_key])
        ) / 2.0
    return result


def _cluster_average(
    values: Mapping[tuple[str, int], float]
) -> dict[str, Any]:
    by_index: dict[int, list[float]] = {}
    for (_entity, index), value in values.items():
        by_index.setdefault(index, []).append(float(value))
    return population_oracle._mean_ci95(
        [sum(group) / len(group) for _index, group in sorted(by_index.items())]
    )


def _valid_counterfactual(row: Mapping[str, Any]) -> bool:
    if row["override_log_valid"] is not True:
        return False
    return (
        row["override_fired"] is True
        or (
            row["nonfire_action_key_identical"] is True
            and row["nonfire_trajectory_identical"] is True
            and row["nonfire_cancellation_valid"] is True
        )
    )


def _population_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if _valid_counterfactual(row)]
    fired = [row for row in valid if row["override_fired"] is True]
    valid_by_seat = {
        seat: [row for row in valid if row["seat"] == seat] for seat in SEATS
    }
    all_by_seat = {
        seat: [row for row in rows if row["seat"] == seat] for seat in SEATS
    }
    # EV remains a realized paired match metric even when an audit row is
    # invalid.  Integrity gates then fail closed without making the immutable
    # merge itself impossible to construct.
    paired_delta = _pair_values(rows, score_key="delta")
    opponent_metrics = {}
    for descriptor in OPPONENT_DESCRIPTORS:
        opponent_id = descriptor["opponent_id"]
        subset = {
            key: value
            for key, value in paired_delta.items()
            if key[0] == opponent_id
        }
        opponent_metrics[opponent_id] = {
            "paired_delta_ev_per_hand": population_oracle._mean_ci95(
                [subset[key] for key in sorted(subset)]
            )
        }
    transformed = [{**row, "seed": row["hand_seed"]} for row in valid]
    fired_delta = [float(row["delta"]) for row in fired]
    first_delta = [
        {**row, "seed": row["hand_seed"]} for row in all_by_seat["first"]
    ]
    second_delta = [
        {**row, "seed": row["hand_seed"]} for row in all_by_seat["second"]
    ]
    nonfires = [
        row
        for row in rows
        if row["override_log_valid"] is True and row["override_fired"] is False
    ]
    invalid = [row for row in rows if not _valid_counterfactual(row)]

    def false_positive_rate(items: Sequence[Mapping[str, Any]]) -> float | None:
        fired_items = [row for row in items if row["override_fired"] is True]
        if not fired_items:
            return None
        return sum(float(row["delta"]) <= 0.0 for row in fired_items) / len(
            fired_items
        )

    return {
        "paired_seat_swap_delta_ev_per_hand": _cluster_average(paired_delta),
        "by_seat_delta_ev_per_hand": {
            "first": population_oracle._cluster_mean_ci95(
                first_delta, "delta"
            ),
            "second": population_oracle._cluster_mean_ci95(
                second_delta, "delta"
            ),
        },
        "realized_gain_per_override": population_oracle._cluster_ratio_ci95(
            transformed
        ),
        "valid_overrides": len(fired),
        "valid_overrides_by_seat": {
            seat: sum(
                row["override_fired"] is True for row in valid_by_seat[seat]
            )
            for seat in SEATS
        },
        "false_positive_override_rate": false_positive_rate(valid),
        "false_positive_override_rate_by_seat": {
            seat: false_positive_rate(valid_by_seat[seat]) for seat in SEATS
        },
        "override_loss_tail": population_oracle._loss_tail(fired_delta),
        "by_opponent": opponent_metrics,
        "invalid_counterfactuals": len(invalid),
        "nonfires": len(nonfires),
        "nonfire_cancellation_mismatches": sum(
            row["nonfire_action_key_identical"] is not True
            or row["nonfire_trajectory_identical"] is not True
            or row["nonfire_cancellation_valid"] is not True
            for row in nonfires
        ),
        "nonfire_nonzero_deltas": sum(
            float(row["delta"]) != 0.0 for row in nonfires
        ),
        "nonfire_cancellation_unknown": sum(
            row["nonfire_action_key_identical"] is not True
            or row["nonfire_trajectory_identical"] is not True
            or row["nonfire_cancellation_valid"] is not True
            for row in nonfires
        )
        + sum(row["override_log_valid"] is False for row in rows),
    }


def _abr_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    paired_delta = _pair_values(rows, score_key="delta")
    paired_candidate = _pair_values(rows, score_key="candidate_score")
    nonfires = [
        row
        for row in rows
        if row["override_log_valid"] is True and row["override_fired"] is False
    ]
    by_response = {}
    for descriptor in ABR_DESCRIPTORS:
        response_id = descriptor["response_id"]
        deltas = [
            value for (entity, _index), value in paired_delta.items()
            if entity == response_id
        ]
        candidate = [
            value for (entity, _index), value in paired_candidate.items()
            if entity == response_id
        ]
        by_response[response_id] = {
            "scientific_role": (
                "direct_hu_score_approximate_best_response"
                if response_id == "greedy_search_response"
                else "exploitative_stress_family_not_direct_best_response"
            ),
            "candidate_minus_baseline_ev_per_hand": (
                population_oracle._mean_ci95(deltas)
            ),
            "candidate_ev_per_hand": population_oracle._mean_ci95(candidate),
        }
    aggregate_delta = _cluster_average(paired_delta)
    worst_response = min(
        (
            float(summary["candidate_ev_per_hand"]["mean"])
            for summary in by_response.values()
        ),
        default=float("-inf"),
    )
    direct_approximate_response = float(
        by_response["greedy_search_response"]["candidate_ev_per_hand"]["mean"]
    )
    stress_family_floor = min(
        float(
            by_response[response_id]["candidate_ev_per_hand"]["mean"]
        )
        for response_id in (
            "foul_pressure_response",
            "royalty_denial_response",
        )
    )
    return {
        "candidate_minus_baseline_ev_per_hand": aggregate_delta,
        "by_response": by_response,
        "direct_hu_approximate_best_response_ev_per_hand": (
            direct_approximate_response
        ),
        "exploitative_stress_family_floor_ev_per_hand": stress_family_floor,
        "worst_response_ev_per_hand": worst_response,
        "worst_response_interpretation": (
            "minimum_across_one_direct_approximate_response_and_two_stress_"
            "families_not_nashconv_or_exploitability_upper_bound"
        ),
        "invalid_counterfactuals": sum(
            not _valid_counterfactual(row) for row in rows
        ),
        "nonfire_cancellation_mismatches": sum(
            row["nonfire_action_key_identical"] is not True
            or row["nonfire_trajectory_identical"] is not True
            or row["nonfire_cancellation_valid"] is not True
            for row in nonfires
        ),
        "nonfire_nonzero_deltas": sum(
            float(row["delta"]) != 0.0 for row in nonfires
        ),
        "nonfire_cancellation_unknown": sum(
            row["nonfire_action_key_identical"] is not True
            or row["nonfire_trajectory_identical"] is not True
            or row["nonfire_cancellation_valid"] is not True
            for row in nonfires
        )
        + sum(row["override_log_valid"] is False for row in rows),
    }


def _expected_coverage() -> set[tuple[int, str, int, int]]:
    values: set[tuple[int, str, int, int]] = set()
    for schedule, entities, count in (
        (
            LOCKED_POPULATION,
            _allowed_entities(LOCKED_POPULATION),
            POPULATION_SEED_COUNT,
        ),
        (LOCKED_ABR, _allowed_entities(LOCKED_ABR), ABR_SEED_COUNT),
    ):
        schedule_order = 0 if schedule == LOCKED_POPULATION else 1
        for entity in entities:
            for index in range(count):
                for seat_order, _seat in enumerate(SEATS):
                    values.add((schedule_order, entity, index, seat_order))
    return values


def _merge_value(
    *,
    plan: Mapping[str, Any],
    shards: Sequence[Mapping[str, Any]],
    source_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [row for shard in shards for row in shard["rows"]]
    keys = [_row_key(row) for row in rows]
    expected = _expected_coverage()
    if len(keys) != len(set(keys)):
        raise ValueError("locked-promotion merge duplicates a hand row")
    missing = expected - set(keys)
    extra = set(keys) - expected
    if missing or extra:
        raise ValueError(
            f"locked-promotion coverage changed: missing={len(missing)}, "
            f"extra={len(extra)}"
        )
    population_rows = [
        row for row in rows if row["schedule"] == LOCKED_POPULATION
    ]
    abr_rows = [row for row in rows if row["schedule"] == LOCKED_ABR]
    population = _population_metrics(population_rows)
    abr = _abr_metrics(abr_rows)
    return {
        "schema": MERGE_SCHEMA,
        "status": "complete_locked_population_and_abr_merge",
        "plan_sha256": canonical_sha256(plan),
        "model_sha256": plan["artifact_binding"]["model"]["sha256"],
        "threshold_lock_sha256": plan["artifact_binding"]["threshold_lock"][
            "sha256"
        ],
        "policy_registry_sha256": plan["artifact_binding"]["policy_registry"][
            "sha256"
        ],
        "evaluation_runtime_closure_sha256": plan["artifact_binding"][
            "evaluation_runtime_closure"
        ]["sha256"],
        "source_shards": [dict(record) for record in source_records],
        "source_shard_aggregate_sha256": canonical_sha256(source_records),
        "row_count": len(rows),
        "coverage": {
            "population_opponents": len(OPPONENT_DESCRIPTORS),
            "population_paired_seeds_per_opponent": POPULATION_SEED_COUNT,
            "population_rows": len(population_rows),
            "abr_families": len(ABR_DESCRIPTORS),
            "abr_paired_seeds_per_response": ABR_SEED_COUNT,
            "abr_rows": len(abr_rows),
            "first_rows": sum(row["seat"] == "first" for row in rows),
            "second_rows": sum(row["seat"] == "second" for row in rows),
            "missing_rows": 0,
            "duplicate_rows": 0,
        },
        "integrity": {
            "population_abr_seed_overlap": len(
                {
                    row["hand_seed"] for row in population_rows
                }
                & {row["hand_seed"] for row in abr_rows}
            ),
            "opponent_private_discard_rows": sum(
                row["opponent_private_discards_used"] is not False for row in rows
            ),
            "current_profile_resolution_rows": sum(
                row["current_profile_resolved"] is not False for row in rows
            ),
        },
        "population": population,
        "abr": abr,
        "metric_oracle": METRIC_ORACLE,
        "teacher_values_used": False,
        "promotion_decision_applied": False,
        "named_profile_added": False,
        "current_profile_changed": False,
        "runtime_activated": False,
    }


def build_locked_promotion_merge(
    *,
    plan: Mapping[str, Any],
    shard_paths: Sequence[str | Path],
) -> dict[str, Any]:
    validated_plan = validate_locked_promotion_plan(plan)
    if not shard_paths:
        raise ValueError("locked-promotion merge has no source shards")
    records = []
    shards = []
    seen_paths: set[str] = set()
    for raw_path in shard_paths:
        path = Path(raw_path).resolve()
        if (
            Path(raw_path).is_symlink()
            or not path.is_file()
            or str(path) in seen_paths
        ):
            raise ValueError("locked-promotion shard path is missing/unsafe/duplicate")
        shard = validate_evaluation_shard(
            _read_canonical(path, "locked-promotion shard"),
            plan=validated_plan,
        )
        seen_paths.add(str(path))
        records.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "shard_id": shard["shard_id"],
            }
        )
        shards.append(shard)
    if len({record["shard_id"] for record in records}) != len(records):
        raise ValueError("locked-promotion shard ids are duplicated")
    records.sort(key=lambda record: record["shard_id"])
    shards.sort(key=lambda shard: shard["shard_id"])
    return _merge_value(plan=validated_plan, shards=shards, source_records=records)


def validate_locked_promotion_merge(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    replay_sources: bool,
) -> dict[str, Any]:
    if replay_sources is not True:
        raise PermissionError("locked-promotion merge requires source replay")
    validated_plan = validate_locked_promotion_plan(plan)
    merge = deepcopy(dict(value))
    _exact_keys(merge, _MERGE_KEYS, "locked-promotion merge")
    sources = merge.get("source_shards")
    if not isinstance(sources, list) or not sources:
        raise ValueError("locked-promotion merge source records are missing")
    paths = []
    for record in sources:
        if not isinstance(record, Mapping):
            raise ValueError("locked-promotion source record is missing")
        _exact_keys(record, _SOURCE_RECORD_KEYS, "source shard record")
        path = Path(str(record["path"]))
        if (
            not path.is_absolute()
            or path.is_symlink()
            or not path.is_file()
            or sha256_file(path) != _require_sha(record["sha256"], "source shard")
            or path.stat().st_size != record["bytes"]
        ):
            raise ValueError("locked-promotion source shard changed")
        paths.append(path)
    expected = build_locked_promotion_merge(
        plan=validated_plan, shard_paths=paths
    )
    if merge != expected:
        raise ValueError("locked-promotion merge differs from source replay")
    return merge


def write_locked_promotion_merge(
    *,
    plan: Mapping[str, Any],
    shard_paths: Sequence[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    merge = build_locked_promotion_merge(plan=plan, shard_paths=shard_paths)
    _write_once(output_path, merge)
    return merge


def _number(value: Any, label: str) -> float:
    result = _finite(value, label)
    return result


def build_locked_promotion_gate(
    *,
    plan: Mapping[str, Any],
    merge: Mapping[str, Any],
    replay_sources: bool,
) -> dict[str, Any]:
    validated_plan = validate_locked_promotion_plan(plan)
    validated = validate_locked_promotion_merge(
        merge, plan=validated_plan, replay_sources=replay_sources
    )
    pop = validated["population"]
    abr = validated["abr"]
    thresholds = validated_plan["promotion_thresholds"]
    gain = pop["realized_gain_per_override"]
    overall = pop["paired_seat_swap_delta_ev_per_hand"]
    by_seat = pop["by_seat_delta_ev_per_hand"]
    loss = pop["override_loss_tail"]
    opponent_pass = all(
        _number(summary["paired_delta_ev_per_hand"]["mean"], "opponent mean")
        >= thresholds["each_opponent_delta_ev_per_hand_min"]
        and _number(
            summary["paired_delta_ev_per_hand"]["ci95_low"],
            "opponent CI lower",
        )
        >= thresholds["each_opponent_delta_ev_per_hand_ci95_low_min"]
        for summary in pop["by_opponent"].values()
    )
    abr_delta = abr["candidate_minus_baseline_ev_per_hand"]
    gates = {
        "complete_exact_population_and_abr_coverage": (
            validated["row_count"]
            == len(OPPONENT_DESCRIPTORS) * POPULATION_SEED_COUNT * 2
            + len(ABR_DESCRIPTORS) * ABR_SEED_COUNT * 2
            and validated["coverage"]["missing_rows"] == 0
            and validated["coverage"]["duplicate_rows"] == 0
        ),
        "hidden_current_seed_integrity_zero": (
            validated["integrity"]["population_abr_seed_overlap"] == 0
            and validated["integrity"]["opponent_private_discard_rows"] == 0
            and validated["integrity"]["current_profile_resolution_rows"] == 0
        ),
        "invalid_counterfactuals_zero": (
            pop["invalid_counterfactuals"]
            <= thresholds["invalid_counterfactuals_max"]
            and abr["invalid_counterfactuals"]
            <= thresholds["invalid_counterfactuals_max"]
        ),
        "nonfire_complete_cancellation_zero_mismatch": (
            pop["nonfire_cancellation_mismatches"]
            <= thresholds["nonfire_cancellation_mismatches_max"]
            and abr["nonfire_cancellation_mismatches"]
            <= thresholds["nonfire_cancellation_mismatches_max"]
            and pop["nonfire_nonzero_deltas"]
            <= thresholds["nonfire_nonzero_deltas_max"]
            and abr["nonfire_nonzero_deltas"]
            <= thresholds["nonfire_nonzero_deltas_max"]
            and pop["nonfire_cancellation_unknown"]
            <= thresholds["nonfire_cancellation_unknown_max"]
            and abr["nonfire_cancellation_unknown"]
            <= thresholds["nonfire_cancellation_unknown_max"]
        ),
        "valid_overrides_at_least_300_and_100_each_seat": (
            pop["valid_overrides"] >= thresholds["valid_overrides_min"]
            and all(
                pop["valid_overrides_by_seat"][seat]
                >= thresholds["valid_overrides_each_seat_min"]
                for seat in SEATS
            )
        ),
        "realized_gain_per_override_ci95_lower_positive": (
            gain["ci95_low"]
            is not None
            and _number(gain["ci95_low"], "gain per override CI lower")
            > thresholds[
                "realized_gain_per_override_ci95_low_min_exclusive"
            ]
        ),
        "overall_and_each_seat_ev_ci95_lower_positive": (
            overall["ci95_low"] is not None
            and _number(overall["ci95_low"], "overall EV CI lower")
            > thresholds["paired_delta_ev_per_hand_ci95_low_min_exclusive"]
            and _number(by_seat["first"]["ci95_low"], "first EV CI lower")
            > thresholds[
                "first_seat_delta_ev_per_hand_ci95_low_min_exclusive"
            ]
            and _number(by_seat["second"]["ci95_low"], "second EV CI lower")
            > thresholds[
                "second_seat_delta_ev_per_hand_ci95_low_min_exclusive"
            ]
        ),
        "false_positive_rates_within_limits": (
            pop["false_positive_override_rate"] is not None
            and pop["false_positive_override_rate"]
            <= thresholds["false_positive_override_rate_max"]
            and all(
                pop["false_positive_override_rate_by_seat"][seat] is not None
                and pop["false_positive_override_rate_by_seat"][seat]
                <= thresholds["false_positive_override_rate_each_seat_max"]
                for seat in SEATS
            )
        ),
        "override_loss_tail_within_25_40_50": (
            loss["p95"] is not None
            and loss["p99"] is not None
            and loss["max"] is not None
            and loss["p95"] <= thresholds["override_loss_p95_max"]
            and loss["p99"] <= thresholds["override_loss_p99_max"]
            and loss["max"] <= thresholds["override_loss_max"]
        ),
        "each_opponent_mean_and_ci_floor": opponent_pass,
        "direct_approximate_response_and_stress_family_floors": (
            _number(abr_delta["mean"], "ABR delta mean")
            >= thresholds["abr_candidate_minus_baseline_ev_per_hand_min"]
            and _number(abr_delta["ci95_low"], "ABR delta CI lower")
            >= thresholds["abr_candidate_minus_baseline_ci95_low_min"]
            and _number(
                abr["direct_hu_approximate_best_response_ev_per_hand"],
                "direct HU approximate response",
            )
            >= thresholds[
                "minimum_direct_response_or_stress_family_ev_per_hand_min"
            ]
            and _number(
                abr["exploitative_stress_family_floor_ev_per_hand"],
                "exploitative stress family floor",
            )
            >= thresholds[
                "minimum_direct_response_or_stress_family_ev_per_hand_min"
            ]
        ),
        "artifact_binding_immutable_and_no_activation": (
            validated["model_sha256"]
            == validated_plan["artifact_binding"]["model"]["sha256"]
            and validated["threshold_lock_sha256"]
            == validated_plan["artifact_binding"]["threshold_lock"]["sha256"]
            and validated["policy_registry_sha256"]
            == validated_plan["artifact_binding"]["policy_registry"]["sha256"]
            and validated["evaluation_runtime_closure_sha256"]
            == validated_plan["artifact_binding"]["evaluation_runtime_closure"][
                "sha256"
            ]
            and validated["named_profile_added"] is False
            and validated["current_profile_changed"] is False
            and validated["runtime_activated"] is False
        ),
    }
    passed = all(gates.values())
    return {
        "schema": GATE_SCHEMA,
        "status": "pass" if passed else "no_go",
        "decision": (
            "locked_population_and_abr_pass_separate_opt_in_profile_candidate_only"
            if passed
            else "locked_promotion_no_go_no_holdout_reselection_or_extension"
        ),
        "plan_sha256": canonical_sha256(validated_plan),
        "merge_sha256": canonical_sha256(validated),
        "thresholds": deepcopy(thresholds),
        "gates": gates,
        "all_gates_passed": passed,
        "scientific_promotion_passed": passed,
        "separate_opt_in_profile_candidate_authorized": passed,
        "teacher_values_used": False,
        "named_profile_added": False,
        "current_profile_changed": False,
        "runtime_activated": False,
        "full_replacement_enabled": False,
    }


def validate_locked_promotion_gate(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    merge: Mapping[str, Any],
    replay_sources: bool,
) -> dict[str, Any]:
    gate_value = deepcopy(dict(value))
    _exact_keys(gate_value, _GATE_KEYS, "locked-promotion gate")
    expected = build_locked_promotion_gate(
        plan=plan, merge=merge, replay_sources=replay_sources
    )
    if gate_value != expected:
        raise ValueError("locked-promotion gate differs from source replay")
    return gate_value


def write_locked_promotion_gate(
    *,
    plan: Mapping[str, Any],
    merge: Mapping[str, Any],
    replay_sources: bool,
    output_path: str | Path,
) -> dict[str, Any]:
    gate_value = build_locked_promotion_gate(
        plan=plan, merge=merge, replay_sources=replay_sources
    )
    _write_once(output_path, gate_value)
    return gate_value


__all__ = [
    "ABR_DESCRIPTORS",
    "ABR_NAMESPACE_BASES",
    "ABR_SEED_COUNT",
    "GATE_SCHEMA",
    "LOCKED_ABR",
    "LOCKED_POPULATION",
    "MERGE_SCHEMA",
    "METRIC_ORACLE",
    "OPPONENT_DESCRIPTORS",
    "PLAN_SCHEMA",
    "POPULATION_NAMESPACE_BASES",
    "POPULATION_SEED_COUNT",
    "PROMOTION_THRESHOLDS",
    "ROW_SCHEMA",
    "SEATS",
    "SEED_STRIDE",
    "SHARD_SCHEMA",
    "THRESHOLD_LOCK_SCHEMA",
    "build_evaluation_shard",
    "build_locked_promotion_gate",
    "build_locked_promotion_merge",
    "build_locked_promotion_plan",
    "canonical_bytes",
    "canonical_sha256",
    "seed_values",
    "sha256_file",
    "validate_artifact_files",
    "validate_evaluation_shard",
    "validate_hand_row",
    "validate_locked_promotion_gate",
    "validate_locked_promotion_merge",
    "validate_locked_promotion_plan",
    "write_evaluation_shard",
    "write_locked_promotion_gate",
    "write_locked_promotion_merge",
    "write_locked_promotion_plan",
]
