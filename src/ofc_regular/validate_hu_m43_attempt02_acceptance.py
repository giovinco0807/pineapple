"""Attempt02 v4 population preflight and final M4 acceptance.

The module has two deliberately separate operations.  ``preflight`` validates
the frozen v4 model, the one-shot inherited-holdout receipt, the canonical
global marker, and the fresh population seed grid before a Spot package may be
created.  ``accept`` recomputes every played-hand metric from the merged JSONL
and applies the predeclared gates without selecting a new threshold.

No teacher, calibration, or locked-holdout JSONL is accepted by either CLI.
Only immutable hashes and lifecycle receipts cross the cloud boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evaluate_hu_m4_population import (
    M4_OPPONENT_PROFILES,
    M4_POPULATION_EVALUATION_SCHEMA,
    summarize_hu_m4_population_records,
)
from .hu_m43_attempt02_contract import M43_ATTEMPT02_DATA_CONTRACT_SCHEMA
from .hu_m43_attempt02_lifecycle import (
    M43_ATTEMPT02_FREEZE_SCHEMA,
    M43_ATTEMPT02_LOCKED_RECEIPT_SCHEMA,
    M43_ATTEMPT02_MARKER_SCHEMA,
    file_sha256,
    validate_attempt02_freeze,
    validate_attempt02_locked_receipt,
)
from .hu_m43_joint_model_loader import load_hu_m43_joint_action_model
from .hu_m43_joint_model_v4 import (
    HU_M43_V4_ACTION_SCORE_MODE,
    HU_M43_V4_MODEL_SCHEMA,
    HU_M43_V4_TRAINING_MANIFEST_SCHEMA,
    HuM43JointModelV4,
)
from .merge_hu_m4_population_shards import M4_POPULATION_MERGE_SCHEMA


ATTEMPT02_POPULATION_PLAN_SCHEMA = "hu_m43_population_acceptance_plan_v1"
ATTEMPT02_POPULATION_PREFLIGHT_SCHEMA = (
    "hu_m43_attempt02_population_launch_preflight_v1"
)
ATTEMPT02_POPULATION_MANIFEST_SCHEMA = (
    "hu_m43_attempt02_population_spot_manifest_v1"
)
ATTEMPT02_POPULATION_RECEIPT_SCHEMA = (
    "hu_m43_attempt02_population_spot_receipt_v1"
)
ATTEMPT02_ACCEPTANCE_CONFIG_SCHEMA = (
    "hu_m43_attempt02_population_acceptance_config_v1"
)
ATTEMPT02_ACCEPTANCE_STATUS_SCHEMA = (
    "hu_m43_attempt02_population_acceptance_status_v1"
)

EXPECTED_OPPONENTS = tuple(M4_OPPONENT_PROFILES)
EXPECTED_POPULATION_SEED = 6_106_071_901
EXPECTED_POPULATION_SEED_STRIDE = 1_000_003
EXPECTED_TEACHER_SCHEDULES = {
    ("M4", "correctness_smoke", 2_026_071_401, 1_000_003, 1),
    ("M4", "train_a", 3_026_072_001, 1_000_003, 32),
    ("M4", "train_b", 4_026_072_001, 1_000_003, 32),
    ("M4", "calibration", 5_026_072_001, 1_000_003, 24),
    ("M4", "locked_holdout", 6_026_072_001, 1_000_003, 24),
    ("M4.1", "correctness_smoke", 9_026_072_001, 1_000_003, 1),
    ("M4.1", "train_a", 10_026_072_001, 1_000_003, 25),
    ("M4.1", "train_b", 20_026_072_001, 1_000_003, 25),
    ("M4.1", "calibration", 30_026_072_001, 1_000_003, 30),
    ("M4.1", "locked_holdout", 40_026_072_001, 1_000_003, 20),
    ("M4.2", "correctness_smoke", 50_026_072_001, 1_000_003, 1),
    ("M4.2", "train", 1_706_071_901, 1_000_003, 20),
    ("M4.2", "calibration", 1_806_071_901, 1_000_003, 10),
    ("M4.2", "locked_holdout", 1_906_071_901, 1_000_003, 10),
    ("M4.3-attempt01", "train", 2_306_071_901, 1_000_003, 100),
    ("M4.3-attempt01", "calibration", 2_506_071_901, 1_000_003, 60),
    ("M4.3-attempt01", "locked_holdout", 2_706_071_901, 1_000_003, 40),
    ("M4.3-attempt02", "train", 7_106_071_901, 1_000_003, 200),
    ("M4.3-attempt02", "calibration", 7_506_071_901, 1_000_003, 100),
}
EXPECTED_PRIOR_POPULATION_SCHEDULES = {
    ("M4", 7_026_072_001, 1_000_003, 2),
    ("M4.1", 50_026_072_001, 1_000_003, 2),
    ("M4.2", 2_126_071_901, 1_000_003, 2),
}
FIXED_GATES: dict[str, float | int] = {
    "valid_overrides_min": 300,
    "realized_gain_per_override_ci95_low_exclusive": 0.0,
    "paired_seat_swap_delta_ci95_low_exclusive": 0.0,
    "second_seat_delta_ci95_low_exclusive": 0.0,
    "first_seat_delta_exact": 0.0,
    "false_positive_override_rate_max": 0.30,
    "override_loss_p95_max": 25.0,
    "override_loss_p99_max": 40.0,
    "override_loss_max": 50.0,
    "each_opponent_delta_ev_per_hand_min": -0.005,
    "each_opponent_delta_ci95_low_min": -0.02,
    "nonfire_cancellation_mismatches_max": 0,
}


def _load_json(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"population record {line_number} is not an object")
            rows.append(value)
    return rows


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a list")
    return value


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _sha256_text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 digest")
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return normalized


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _same_number(left: Any, right: Any) -> bool:
    left_number = _number(left)
    right_number = _number(right)
    return (
        left_number is not None
        and right_number is not None
        and math.isclose(left_number, right_number, rel_tol=0.0, abs_tol=1e-12)
    )


def _self_digest(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return hashlib.sha256(_canonical(unsigned).encode("ascii")).hexdigest()


def _seed_schedule(seed: int, stride: int, count: int) -> set[int]:
    return {seed + index * stride for index in range(count)}


def load_and_validate_attempt02_population_plan(
    source: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    plan = (
        _load_json(source, "Attempt02 population plan")
        if not isinstance(source, Mapping)
        else dict(source)
    )
    if plan.get("schema") != ATTEMPT02_POPULATION_PLAN_SCHEMA:
        raise ValueError("Attempt02 population plan schema mismatch")
    if plan.get("status") != "frozen_before_population_evaluation":
        raise ValueError("Attempt02 population plan is not frozen")
    if plan.get("policy_attempt") != "attempt02_v4":
        raise ValueError("population plan is not bound to Attempt02 v4")
    if plan.get("fixed_baseline_profile") != "stage18_p1":
        raise ValueError("population baseline profile changed")
    if plan.get("opponents") != list(EXPECTED_OPPONENTS):
        raise ValueError("population opponent order changed")
    if plan.get("paired_seat_swap") is not True:
        raise ValueError("population seat-swap requirement changed")

    training_plan = _mapping(
        plan.get("attempt02_training_plan"), "attempt02_training_plan"
    )
    _sha256_text(training_plan.get("file_sha256"), "Attempt02 training plan SHA")
    runtime = _mapping(plan.get("runtime_contract"), "runtime_contract")
    if dict(runtime) != {
        "model_schema": HU_M43_V4_MODEL_SCHEMA,
        "action_score_mode": HU_M43_V4_ACTION_SCORE_MODE,
        "runtime_teacher_inputs": False,
        "runtime_lcb_gate": False,
    }:
        raise ValueError("Attempt02 v4 runtime contract changed")

    paired = _integer(
        plan.get("paired_seeds_per_opponent"),
        "paired_seeds_per_opponent",
        minimum=1,
    )
    seed = _integer(plan.get("seed"), "seed", minimum=1)
    stride = _integer(plan.get("seed_stride"), "seed_stride", minimum=1)
    shards = _integer(plan.get("shards"), "shards", minimum=1)
    per_shard = _integer(
        plan.get("paired_seeds_per_shard"), "paired_seeds_per_shard", minimum=1
    )
    if (paired, shards, per_shard) != (1000, 20, 50):
        raise ValueError("Attempt02 final population power/shard plan changed")
    if (seed, stride) != (
        EXPECTED_POPULATION_SEED,
        EXPECTED_POPULATION_SEED_STRIDE,
    ):
        raise ValueError("Attempt02 final population seed schedule changed")
    if shards * per_shard != paired:
        raise ValueError("population shards do not cover the frozen seed grid")
    if plan.get("candidate_records") != paired * len(EXPECTED_OPPONENTS) * 2:
        raise ValueError("population candidate record count changed")
    if plan.get("baseline_records") != paired * len(EXPECTED_OPPONENTS) * 2:
        raise ValueError("population baseline record count changed")
    if plan.get("second_seat_override_opportunities") != paired * len(
        EXPECTED_OPPONENTS
    ):
        raise ValueError("population second-seat opportunity count changed")
    if plan.get("terminal_trace_hands") != paired * len(EXPECTED_OPPONENTS) * 4:
        raise ValueError("population terminal trace count changed")
    if plan.get("minimum_valid_overrides") != 300:
        raise ValueError("population 300-valid-override gate changed")
    if not _same_number(plan.get("minimum_population_fire_rate_needed"), 0.075):
        raise ValueError("population minimum fire-rate sizing changed")

    sizing = _mapping(plan.get("sizing_rule"), "sizing_rule")
    if dict(sizing) != {
        "formula": "paired_seeds_required = ceil(minimum_valid_overrides / (opponents * independent_fire_rate_lower_bound))",
        "fixed_before_realized_population_labels": True,
        "optional_stopping_or_posthoc_extension_allowed": False,
        "insufficient_overrides_decision": "complete_no_go",
    }:
        raise ValueError("population sizing/optional-stopping contract changed")
    evaluation_contract = _mapping(
        plan.get("evaluation_contract"), "evaluation_contract"
    )
    if dict(evaluation_contract) != {
        "candidate_and_baseline_same_hand_seed": True,
        "candidate_and_baseline_same_physical_seat": True,
        "candidate_and_baseline_same_opponent_policy_seed": True,
        "first_and_second_seat_metrics": True,
        "nonfire_full_trajectory_digest_cancellation": True,
        "invalid_counterfactuals_excluded": True,
        "false_positive_override_definition": "realized_delta_le_zero",
        "confidence_interval_unit": "hand_seed_cluster_after_opponent_average",
        "teacher_values_are_realized_match_ev": False,
        "threshold_reselection_allowed": False,
    }:
        raise ValueError("population counterfactual/evaluation contract changed")
    spot_execution = _mapping(plan.get("spot_execution"), "spot_execution")
    if dict(spot_execution) != {
        "small_shards": True,
        "checkpoint_unit": "completed_shard",
        "heartbeat_required": True,
        "resume_missing_shards_only": True,
        "done_commit_last": True,
        "merge_recomputes_metrics_from_records": True,
    }:
        raise ValueError("population Spot execution contract changed")
    declared_gates = _mapping(plan.get("fixed_acceptance_gates"), "fixed gates")
    if set(declared_gates) != set(FIXED_GATES) or any(
        not _same_number(declared_gates.get(key), expected)
        for key, expected in FIXED_GATES.items()
    ):
        raise ValueError("Attempt02 population acceptance gates changed")
    guards = _mapping(plan.get("activation_guards"), "activation_guards")
    if set(guards) != {
        "current_profile_changed",
        "runtime_policy_activated",
        "full_replacement_enabled",
        "threshold_changed_after_lock",
    } or any(value is not False for value in guards.values()):
        raise ValueError("population activation guards changed")

    planned = _seed_schedule(seed, stride, paired)
    freshness = _mapping(plan.get("freshness"), "freshness")
    if (
        freshness.get("exclude_all_m4_m41_m42_m43_teacher_hand_seeds") is not True
        or freshness.get("exclude_prior_population_smoke_seeds") is not True
        or freshness.get("planned_overlap_count_at_freeze") != 0
        or freshness.get("acceptance_validator_rechecks_teacher_seed_overlap")
        is not True
    ):
        raise ValueError("population freshness declarations changed")
    teacher_schedules = _sequence(
        freshness.get("excluded_teacher_schedules"), "excluded_teacher_schedules"
    )
    observed_schedules: set[tuple[str, str, int, int, int]] = set()
    teacher_checked = 0
    for index, raw in enumerate(teacher_schedules):
        schedule = _mapping(raw, f"teacher schedule {index}")
        identity = (
            str(schedule.get("milestone")),
            str(schedule.get("split")),
            _integer(schedule.get("seed"), f"teacher schedule {index}.seed", minimum=1),
            _integer(
                schedule.get("seed_stride"),
                f"teacher schedule {index}.seed_stride",
                minimum=1,
            ),
            _integer(schedule.get("roots"), f"teacher schedule {index}.roots", minimum=1),
        )
        if identity in observed_schedules:
            raise ValueError("duplicate teacher seed schedule")
        observed_schedules.add(identity)
        excluded = _seed_schedule(identity[2], identity[3], identity[4])
        teacher_checked += len(excluded)
        if planned & excluded:
            raise ValueError("population/teacher seed overlap")
    if observed_schedules != EXPECTED_TEACHER_SCHEDULES:
        missing = sorted(EXPECTED_TEACHER_SCHEDULES - observed_schedules)
        extra = sorted(observed_schedules - EXPECTED_TEACHER_SCHEDULES)
        raise ValueError(
            "M4 through Attempt02 teacher seed exclusions changed: "
            f"missing={missing}, extra={extra}"
        )

    prior_checked = 0
    observed_prior: set[tuple[str, int, int, int]] = set()
    prior_schedules = _sequence(
        freshness.get("excluded_population_schedules"),
        "excluded_population_schedules",
    )
    if not prior_schedules:
        raise ValueError("prior population schedules are missing")
    for index, raw in enumerate(prior_schedules):
        schedule = _mapping(raw, f"prior population schedule {index}")
        identity = (
            str(schedule.get("milestone")),
            _integer(schedule.get("seed"), f"prior schedule {index}.seed", minimum=1),
            _integer(
                schedule.get("seed_stride"),
                f"prior schedule {index}.seed_stride",
                minimum=1,
            ),
            _integer(
                schedule.get("paired_seeds"),
                f"prior schedule {index}.paired_seeds",
                minimum=1,
            ),
        )
        if identity in observed_prior:
            raise ValueError("duplicate prior population seed schedule")
        observed_prior.add(identity)
        prior = _seed_schedule(identity[1], identity[2], identity[3])
        prior_checked += len(prior)
        if planned & prior:
            raise ValueError("population/prior schedule seed overlap")
    if observed_prior != EXPECTED_PRIOR_POPULATION_SCHEDULES:
        missing = sorted(EXPECTED_PRIOR_POPULATION_SCHEDULES - observed_prior)
        extra = sorted(observed_prior - EXPECTED_PRIOR_POPULATION_SCHEDULES)
        raise ValueError(
            "prior population seed exclusions changed: "
            f"missing={missing}, extra={extra}"
        )
    plan["_freshness_counts"] = {
        "population": len(planned),
        "teacher_schedule": teacher_checked,
        "prior_population": prior_checked,
    }
    return plan


def build_attempt02_population_launch_preflight(
    *,
    model_path: str | Path,
    training_manifest_path: str | Path,
    data_contract_path: str | Path,
    freeze_manifest_path: str | Path,
    locked_receipt_path: str | Path,
    consumption_marker_path: str | Path,
    population_plan_path: str | Path,
) -> dict[str, Any]:
    paths = {
        "model": Path(model_path).resolve(),
        "training": Path(training_manifest_path).resolve(),
        "contract": Path(data_contract_path).resolve(),
        "freeze": Path(freeze_manifest_path).resolve(),
        "receipt": Path(locked_receipt_path).resolve(),
        "marker": Path(consumption_marker_path).resolve(),
        "plan": Path(population_plan_path).resolve(),
    }
    if len(set(paths.values())) != len(paths) or not all(
        path.is_file() for path in paths.values()
    ):
        raise ValueError("Attempt02 population lifecycle paths are missing or aliased")
    training = _load_json(paths["training"], "training manifest")
    contract = _load_json(paths["contract"], "data contract")
    freeze = _load_json(paths["freeze"], "freeze manifest")
    receipt = _load_json(paths["receipt"], "locked receipt")
    marker = _load_json(paths["marker"], "canonical consumption marker")
    plan = load_and_validate_attempt02_population_plan(paths["plan"])

    if contract.get("schema") != M43_ATTEMPT02_DATA_CONTRACT_SCHEMA:
        raise ValueError("Attempt02 data contract schema mismatch")
    if training.get("schema") != HU_M43_V4_TRAINING_MANIFEST_SCHEMA:
        raise ValueError("Attempt02 training manifest is not v4")
    if freeze.get("schema") != M43_ATTEMPT02_FREEZE_SCHEMA:
        raise ValueError("Attempt02 freeze schema mismatch")
    if receipt.get("schema") != M43_ATTEMPT02_LOCKED_RECEIPT_SCHEMA:
        raise ValueError("Attempt02 locked receipt schema mismatch")
    if marker.get("schema") != M43_ATTEMPT02_MARKER_SCHEMA:
        raise ValueError("Attempt02 canonical marker schema mismatch")

    validate_attempt02_freeze(freeze, contract=contract)
    validate_attempt02_locked_receipt(
        receipt, freeze=freeze, contract=contract, marker=marker
    )
    expected_marker = Path(
        str(freeze.get("canonical_consumption_marker_resolved_path", ""))
    ).resolve()
    if paths["marker"] != expected_marker:
        raise ValueError("consumption marker is not the canonical global marker path")
    expected_population_plan = Path(
        str(freeze.get("population_plan_resolved_path", ""))
    ).resolve()
    if paths["plan"] != expected_population_plan:
        raise ValueError("population plan is not the path frozen before locked access")

    hashes = {name: file_sha256(path) for name, path in paths.items()}
    expected_hashes = {
        "model": freeze.get("model_sha256"),
        "training": freeze.get("training_manifest_sha256"),
        "contract": freeze.get("data_contract_file_sha256"),
        "freeze": receipt.get("freeze_file_sha256"),
        "marker": receipt.get("consumption_marker_file_sha256"),
        "plan": freeze.get("population_plan_file_sha256"),
    }
    for name, expected in expected_hashes.items():
        if hashes[name] != expected:
            raise ValueError(f"Attempt02 {name} bytes disagree with lifecycle")
    if freeze.get("plan_sha256") != plan["attempt02_training_plan"]["file_sha256"]:
        raise ValueError("population plan is bound to a different Attempt02 plan")
    if receipt.get("requires_fresh_population_acceptance") is not True:
        raise ValueError("locked receipt bypasses fresh population acceptance")
    if receipt.get("minimum_population_valid_overrides") != 300:
        raise ValueError("locked receipt changed the 300-override minimum")
    if any(
        receipt.get(field) is not False
        for field in (
            "threshold_search_performed",
            "threshold_reselection_performed",
            "model_selection_performed",
            "feature_selection_performed",
            "current_profile_resolved",
            "current_profile_changed",
            "runtime_policy_activated",
            "policy_promoted",
        )
    ):
        raise ValueError("locked receipt contains an unsafe lifecycle flag")

    model = load_hu_m43_joint_action_model(
        paths["model"],
        expected_sha256=hashes["model"],
        freeze_manifest=freeze,
        training_manifest_path=paths["training"],
    )
    if not isinstance(model, HuM43JointModelV4):
        raise TypeError("Attempt02 population requires the v4 joint model")
    if model.action_score_mode != HU_M43_V4_ACTION_SCORE_MODE:
        raise ValueError("Attempt02 v4 action-score mode changed")
    if training.get("runtime_teacher_inputs") is not False:
        raise ValueError("Attempt02 runtime may not consume teacher inputs")

    counts = plan["_freshness_counts"]
    return {
        "schema": ATTEMPT02_POPULATION_PREFLIGHT_SCHEMA,
        "status": "pass",
        "model_sha256": hashes["model"],
        "training_manifest_sha256": hashes["training"],
        "data_contract_file_sha256": hashes["contract"],
        "freeze_manifest_file_sha256": hashes["freeze"],
        "locked_receipt_file_sha256": hashes["receipt"],
        "consumption_marker_file_sha256": hashes["marker"],
        "population_plan_file_sha256": hashes["plan"],
        "freeze_sha256": freeze["freeze_sha256"],
        "locked_receipt_sha256": receipt["receipt_sha256"],
        "consumption_marker_canonical_sha256": marker["marker_sha256"],
        "model_schema": model.schema,
        "model_id": model.model_id,
        "action_score_mode": model.action_score_mode,
        "frozen_threshold": float(freeze["frozen_threshold"]),
        "population_hand_seeds_checked": counts["population"],
        "teacher_schedule_seeds_checked": counts["teacher_schedule"],
        "prior_population_seeds_checked": counts["prior_population"],
        "teacher_overlap_count": 0,
        "prior_population_overlap_count": 0,
        "canonical_global_marker_verified": True,
        "teacher_calibration_locked_content_packaged": False,
        "current_profile_mutated": False,
        "no_runtime_activation": True,
    }


def _summary_for_comparison(summary: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(summary)
    normalized.pop("runtime_config", None)
    normalized.pop("shard_merge", None)
    normalized.pop("records", None)
    normalized["records_output"] = None
    normalized["elapsed_seconds"] = None
    return normalized


def _path(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _gate(name: str, passed: bool, observed: Any, requirement: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "requirement": requirement,
    }


def evaluate_attempt02_population_gates(
    evaluation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    population = _mapping(evaluation.get("population"), "evaluation.population")
    all_seats = _mapping(population.get("all_seats"), "population.all_seats")
    by_seat = _mapping(population.get("by_seat"), "population.by_seat")
    first = _mapping(by_seat.get("first"), "population.by_seat.first")
    second = _mapping(by_seat.get("second"), "population.by_seat.second")
    paired = _mapping(
        population.get("paired_seat_swap"), "population.paired_seat_swap"
    )
    gates: list[dict[str, Any]] = []
    invalid = _integer(
        evaluation.get("invalid_counterfactuals"), "invalid_counterfactuals"
    )
    gates.append(_gate("invalid_counterfactuals", invalid == 0, invalid, "== 0"))
    cancellation = {
        "top_level": evaluation.get("nonfire_cancellation_mismatches"),
        "mismatches": all_seats.get("nonfire_cancellation_mismatches"),
        "nonzero": all_seats.get("nonfire_nonzero_deltas"),
        "unknown": all_seats.get("nonfire_cancellation_unknown"),
    }
    gates.append(
        _gate(
            "nonfire_counterfactual_cancellation",
            all(value == 0 for value in cancellation.values()),
            cancellation,
            "all mismatch/nonzero/unknown counts == 0",
        )
    )
    overrides = all_seats.get("overrides")
    gates.append(
        _gate(
            "minimum_valid_overrides",
            isinstance(overrides, int) and not isinstance(overrides, bool) and overrides >= 300,
            overrides,
            ">= 300",
        )
    )
    gain_low = _number(_path(all_seats, "realized_gain_per_override", "ci95_low"))
    gates.append(_gate("realized_gain_per_override", gain_low is not None and gain_low > 0, gain_low, "CI95 low > 0"))
    paired_low = _number(_path(paired, "delta_ev_per_hand", "ci95_low"))
    gates.append(_gate("paired_seat_swap_ev", paired_low is not None and paired_low > 0, paired_low, "CI95 low > 0"))
    second_low = _number(_path(second, "delta_ev_per_hand", "ci95_low"))
    gates.append(_gate("second_seat_ev", second_low is not None and second_low > 0, second_low, "CI95 low > 0"))
    first_delta = _mapping(first.get("delta_ev_per_hand"), "first delta")
    first_exact = first.get("overrides") == 0 and all(
        _same_number(first_delta.get(key), 0.0)
        for key in ("mean", "ci95_low", "ci95_high")
    )
    gates.append(_gate("first_seat_exact_cancellation", first_exact, {"delta": dict(first_delta), "overrides": first.get("overrides")}, "exactly zero with no first-seat overrides"))
    false_positive = _number(all_seats.get("false_positive_override_rate"))
    false_positive_definition = all_seats.get("false_positive_definition")
    gates.append(
        _gate(
            "false_positive_override_rate",
            false_positive_definition == "realized_override_delta_le_zero"
            and false_positive is not None
            and false_positive <= 0.30,
            {
                "rate": false_positive,
                "definition": false_positive_definition,
            },
            "realized delta <= 0 rate <= 0.30",
        )
    )
    tail = _mapping(all_seats.get("override_loss_tail"), "override loss tail")
    tail_values = {name: _number(tail.get(name)) for name in ("p95", "p99", "max")}
    tail_pass = (
        tail_values["p95"] is not None and tail_values["p95"] <= 25.0
        and tail_values["p99"] is not None and tail_values["p99"] <= 40.0
        and tail_values["max"] is not None and tail_values["max"] <= 50.0
    )
    gates.append(_gate("override_loss_tail", tail_pass, tail_values, "p95 <= 25, p99 <= 40, max <= 50"))
    opponent_rows = _mapping(evaluation.get("by_opponent"), "by_opponent")
    opponent_observed: dict[str, Any] = {}
    opponent_pass = set(opponent_rows) == set(EXPECTED_OPPONENTS)
    for opponent in EXPECTED_OPPONENTS:
        delta = _mapping(_path(opponent_rows, opponent, "seat_swap", "delta_ev_per_hand"), f"{opponent} delta")
        mean = _number(delta.get("mean"))
        low = _number(delta.get("ci95_low"))
        opponent_observed[opponent] = {"mean": mean, "ci95_low": low}
        opponent_pass = bool(
            opponent_pass
            and mean is not None and mean >= -0.005
            and low is not None and low >= -0.02
        )
    gates.append(_gate("each_opponent_robustness", opponent_pass, opponent_observed, "each mean >= -0.005 and CI95 low >= -0.02"))
    return gates


def validate_attempt02_population_acceptance(
    *,
    evaluation: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    population_plan: Mapping[str, Any],
    merge_manifest: Mapping[str, Any],
    spot_receipt: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    lifecycle_preflight: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = load_and_validate_attempt02_population_plan(population_plan)
    plan.pop("_freshness_counts", None)
    plan_sha = source_hashes.get("population_plan")
    records_sha = source_hashes.get("records")
    evaluation_sha = source_hashes.get("evaluation")
    merge_sha = source_hashes.get("merge_manifest")
    run_sha = source_hashes.get("run_manifest")
    for name, value in (
        ("population_plan", plan_sha),
        ("records", records_sha),
        ("evaluation", evaluation_sha),
        ("merge_manifest", merge_sha),
        ("run_manifest", run_sha),
    ):
        _sha256_text(value, f"{name} SHA")

    recomputed = summarize_hu_m4_population_records(
        records,
        opponents=EXPECTED_OPPONENTS,
        paired_seeds=int(plan["paired_seeds_per_opponent"]),
        seed=int(plan["seed"]),
        seed_stride=int(plan["seed_stride"]),
    )
    content_match = _canonical(_summary_for_comparison(evaluation)) == _canonical(
        _summary_for_comparison(recomputed)
    )
    runtime = _mapping(evaluation.get("runtime_config"), "runtime_config")
    runtime_match = (
        runtime.get("candidate_model_sha256") == lifecycle_preflight.get("model_sha256")
        and runtime.get("safety_model_sha256") == lifecycle_preflight.get("model_sha256")
        and runtime.get("model_schema") == HU_M43_V4_MODEL_SCHEMA
        and runtime.get("model_id") == lifecycle_preflight.get("model_id")
        and runtime.get("action_score_mode") == HU_M43_V4_ACTION_SCORE_MODE
        and runtime.get("runtime_binding_verified") is True
        and runtime.get("freeze_manifest_sha256") == lifecycle_preflight.get("freeze_manifest_file_sha256")
        and runtime.get("training_manifest_sha256") == lifecycle_preflight.get("training_manifest_sha256")
        and runtime.get("population_plan_sha256") == plan_sha
        and runtime.get("sharded_evaluation") is True
        and runtime.get("shard_count") == 20
        and runtime.get("final_metrics_recomputed_from_merged_records") is True
        and runtime.get("current_profile_used") is False
        and runtime.get("promotion_artifact_contract") is True
        and runtime.get("diagnostic_legacy") is False
        and runtime.get("safety_enabled") is True
        and _same_number(
            runtime.get("safety_threshold"),
            lifecycle_preflight.get("frozen_threshold"),
        )
        and all(row.get("runtime_binding_verified") is True for row in records)
    )
    lifecycle_match = (
        lifecycle_preflight.get("schema") == ATTEMPT02_POPULATION_PREFLIGHT_SCHEMA
        and lifecycle_preflight.get("status") == "pass"
        and lifecycle_preflight.get("teacher_overlap_count") == 0
        and lifecycle_preflight.get("prior_population_overlap_count") == 0
        and lifecycle_preflight.get("canonical_global_marker_verified") is True
        and lifecycle_preflight.get("teacher_calibration_locked_content_packaged") is False
        and lifecycle_preflight.get("current_profile_mutated") is False
        and lifecycle_preflight.get("no_runtime_activation") is True
    )
    merge_shards_raw = merge_manifest.get("shards")
    receipt_shards_raw = spot_receipt.get("shards")
    shard_provenance_match = (
        isinstance(merge_shards_raw, Sequence)
        and not isinstance(merge_shards_raw, (str, bytes))
        and isinstance(receipt_shards_raw, Sequence)
        and not isinstance(receipt_shards_raw, (str, bytes))
        and len(merge_shards_raw) == 20
        and len(receipt_shards_raw) == 20
    )
    if shard_provenance_match:
        merge_by_offset: dict[int, Mapping[str, Any]] = {}
        receipt_by_shard: dict[int, Mapping[str, Any]] = {}
        for raw in merge_shards_raw:
            if not isinstance(raw, Mapping):
                shard_provenance_match = False
                break
            offset = raw.get("offset")
            if isinstance(offset, bool) or not isinstance(offset, int) or offset in merge_by_offset:
                shard_provenance_match = False
                break
            merge_by_offset[offset] = raw
        for raw in receipt_shards_raw:
            if not isinstance(raw, Mapping):
                shard_provenance_match = False
                break
            shard = raw.get("shard")
            if isinstance(shard, bool) or not isinstance(shard, int) or shard in receipt_by_shard:
                shard_provenance_match = False
                break
            receipt_by_shard[shard] = raw
        if shard_provenance_match:
            for shard in range(20):
                offset = shard * 50
                merged = merge_by_offset.get(offset)
                received = receipt_by_shard.get(shard)
                expected_seed = int(plan["seed"]) + offset * int(plan["seed_stride"])
                if (
                    merged is None
                    or received is None
                    or merged.get("seed") != expected_seed
                    or merged.get("paired_seeds") != 50
                    or merged.get("records") != 400
                    or not _is_sha256(merged.get("evaluation_sha256"))
                    or not _is_sha256(merged.get("records_sha256"))
                    or merged.get("evaluation_sha256") != received.get("evaluation_sha256")
                    or merged.get("records_sha256") != received.get("records_sha256")
                    or not _is_sha256(received.get("done_sha256"))
                ):
                    shard_provenance_match = False
                    break
    merge_match = (
        merge_manifest.get("schema") == M4_POPULATION_MERGE_SCHEMA
        and merge_manifest.get("status") == "complete_content_verified"
        and merge_manifest.get("population_plan_sha256") == plan_sha
        and merge_manifest.get("merged_records_sha256") == records_sha
        and merge_manifest.get("evaluation_sha256") == evaluation_sha
        and merge_manifest.get("paired_seeds_per_opponent") == 1000
        and merge_manifest.get("opponents") == list(EXPECTED_OPPONENTS)
        and merge_manifest.get("merged_records") == 8000
        and merge_manifest.get("current_profile_used") is False
        and merge_manifest.get("metrics_recomputed_from_merged_seed_clusters")
        is True
        and shard_provenance_match
    )
    manifest_runtime = _mapping(run_manifest.get("runtime"), "run runtime")
    manifest_preflight = _mapping(
        run_manifest.get("launch_preflight"), "run launch_preflight"
    )
    source_boundary = _mapping(
        run_manifest.get("source_boundary"), "run source_boundary"
    )
    manifest_plan = _mapping(
        run_manifest.get("population_plan"), "run population_plan"
    )
    manifest_shards = _mapping(run_manifest.get("shards"), "run shards")
    manifest_checkpoint = _mapping(
        run_manifest.get("checkpoint"), "run checkpoint"
    )
    manifest_compute = _mapping(run_manifest.get("compute"), "run compute")
    run_match = (
        run_manifest.get("schema") == ATTEMPT02_POPULATION_MANIFEST_SCHEMA
        and manifest_plan.get("sha256") == plan_sha
        and manifest_plan.get("paired_seeds") == 1000
        and manifest_plan.get("seed") == int(plan["seed"])
        and manifest_plan.get("seed_stride") == int(plan["seed_stride"])
        and manifest_plan.get("shards") == 20
        and manifest_plan.get("paired_seeds_per_shard") == 50
        and manifest_shards.get("count") == 20
        and _is_sha256(manifest_shards.get("sha256"))
        and dict(manifest_checkpoint)
        == {
            "unit": "completed_shard",
            "retry": "deterministic_full_shard",
            "resume_missing_shards_only": True,
            "done_commit_last": True,
        }
        and manifest_compute.get("provisioning_model") == "SPOT"
        and manifest_compute.get("instance_termination_action") == "DELETE"
        and plan_sha == lifecycle_preflight.get("population_plan_file_sha256")
        and manifest_runtime.get("model_sha256") == lifecycle_preflight.get("model_sha256")
        and manifest_runtime.get("training_manifest_sha256") == lifecycle_preflight.get("training_manifest_sha256")
        and manifest_runtime.get("data_contract_sha256") == lifecycle_preflight.get("data_contract_file_sha256")
        and manifest_runtime.get("freeze_manifest_sha256") == lifecycle_preflight.get("freeze_manifest_file_sha256")
        and manifest_runtime.get("locked_receipt_sha256") == lifecycle_preflight.get("locked_receipt_file_sha256")
        and manifest_runtime.get("consumption_marker_sha256") == lifecycle_preflight.get("consumption_marker_file_sha256")
        and manifest_runtime.get("model_schema") == HU_M43_V4_MODEL_SCHEMA
        and manifest_runtime.get("model_id") == lifecycle_preflight.get("model_id")
        and manifest_runtime.get("action_score_mode") == HU_M43_V4_ACTION_SCORE_MODE
        and manifest_runtime.get("runtime_teacher_inputs") is False
        and manifest_runtime.get("current_profile_used") is False
        and _same_number(
            manifest_runtime.get("frozen_threshold"),
            lifecycle_preflight.get("frozen_threshold"),
        )
        and dict(manifest_preflight) == dict(lifecycle_preflight)
        and dict(source_boundary)
        == {
            "teacher_jsonl_packaged": False,
            "calibration_jsonl_packaged": False,
            "locked_jsonl_packaged": False,
            "current_profile_artifact_packaged": False,
            "lifecycle_hashes_and_receipts_only": True,
        }
        and run_manifest.get("no_runtime_activation") is True
        and run_manifest.get("current_profile_mutated") is False
    )
    receipt_match = (
        spot_receipt.get("schema") == ATTEMPT02_POPULATION_RECEIPT_SCHEMA
        and spot_receipt.get("status") == "verified_and_merged"
        and spot_receipt.get("run_manifest_sha256") == run_sha
        and spot_receipt.get("population_plan_sha256") == plan_sha
        and spot_receipt.get("model_sha256") == lifecycle_preflight.get("model_sha256")
        and spot_receipt.get("model_schema") == HU_M43_V4_MODEL_SCHEMA
        and spot_receipt.get("action_score_mode") == HU_M43_V4_ACTION_SCORE_MODE
        and spot_receipt.get("evaluation_sha256") == evaluation_sha
        and spot_receipt.get("records_sha256") == records_sha
        and spot_receipt.get("merge_manifest_sha256") == merge_sha
        and spot_receipt.get("valid_overrides")
        == recomputed["population"]["all_seats"]["overrides"]
        and spot_receipt.get("paired_seeds_per_opponent") == 1000
        and spot_receipt.get("invalid_counterfactuals")
        == recomputed["invalid_counterfactuals"]
        and spot_receipt.get("nonfire_cancellation_mismatches")
        == recomputed["nonfire_cancellation_mismatches"]
        and shard_provenance_match
        and spot_receipt.get("teacher_calibration_locked_content_received") is False
        and spot_receipt.get("current_profile_mutated") is False
        and spot_receipt.get("no_runtime_activation") is True
    )

    gates = [
        _gate("lifecycle_hash_chain", lifecycle_match and run_match and receipt_match and merge_match, {
            "lifecycle": lifecycle_match,
            "run_manifest": run_match,
            "spot_receipt": receipt_match,
            "merge_manifest": merge_match,
        }, "all Attempt02 lifecycle and byte-hash bindings pass"),
        _gate("population_content_recomputed", content_match, content_match, "stored summary equals JSONL recomputation"),
        _gate("v4_runtime_binding", runtime_match, dict(runtime), "bound Attempt02 v4 model/freeze/training and current unused"),
        *evaluate_attempt02_population_gates(recomputed),
    ]
    passed = sum(gate["passed"] is True for gate in gates)
    decision = "complete_go" if passed == len(gates) else "complete_no_go"
    config = {
        "schema": ATTEMPT02_ACCEPTANCE_CONFIG_SCHEMA,
        "status": "frozen_evaluated_once",
        "policy_attempt": "attempt02_v4",
        "population_plan_sha256": plan_sha,
        "model_sha256": lifecycle_preflight.get("model_sha256"),
        "model_schema": HU_M43_V4_MODEL_SCHEMA,
        "action_score_mode": HU_M43_V4_ACTION_SCORE_MODE,
        "paired_seeds_per_opponent": 1000,
        "opponents": list(EXPECTED_OPPONENTS),
        "fixed_gates": dict(FIXED_GATES),
        "threshold_reselection_allowed": False,
        "teacher_values_used_as_realized_ev": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    status = {
        "schema": ATTEMPT02_ACCEPTANCE_STATUS_SCHEMA,
        "decision": decision,
        "gates_passed": passed,
        "gates_total": len(gates),
        "gates": gates,
        "valid_overrides": recomputed["population"]["all_seats"]["overrides"],
        "source_hashes": dict(source_hashes),
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    return config, status


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _preflight_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("preflight")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--data-contract", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--locked-receipt", type=Path, required=True)
    parser.add_argument("--consumption-marker", type=Path, required=True)
    parser.add_argument("--population-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)


def _accept_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("accept")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--data-contract", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--locked-receipt", type=Path, required=True)
    parser.add_argument("--consumption-marker", type=Path, required=True)
    parser.add_argument("--population-plan", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--merge-manifest", type=Path, required=True)
    parser.add_argument("--spot-receipt", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--config-output", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _preflight_parser(subparsers)
    _accept_parser(subparsers)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    preflight = build_attempt02_population_launch_preflight(
        model_path=args.model,
        training_manifest_path=args.training_manifest,
        data_contract_path=args.data_contract,
        freeze_manifest_path=args.freeze_manifest,
        locked_receipt_path=args.locked_receipt,
        consumption_marker_path=args.consumption_marker,
        population_plan_path=args.population_plan,
    )
    if args.command == "preflight":
        if args.output is not None:
            _write_json(args.output, preflight)
        print(json.dumps(preflight, ensure_ascii=True, sort_keys=True))
        return 0

    paths = {
        "population_plan": args.population_plan,
        "evaluation": args.evaluation,
        "records": args.records,
        "merge_manifest": args.merge_manifest,
        "spot_receipt": args.spot_receipt,
        "run_manifest": args.run_manifest,
    }
    source_hashes = {name: file_sha256(path) for name, path in paths.items()}
    config, status = validate_attempt02_population_acceptance(
        evaluation=_load_json(args.evaluation, "population evaluation"),
        records=_load_jsonl(args.records),
        population_plan=_load_json(args.population_plan, "population plan"),
        merge_manifest=_load_json(args.merge_manifest, "merge manifest"),
        spot_receipt=_load_json(args.spot_receipt, "Spot receipt"),
        run_manifest=_load_json(args.run_manifest, "run manifest"),
        lifecycle_preflight=preflight,
        source_hashes=source_hashes,
    )
    _write_json(args.config_output, config)
    _write_json(args.status_output, status)
    print(json.dumps(status, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ATTEMPT02_ACCEPTANCE_CONFIG_SCHEMA",
    "ATTEMPT02_ACCEPTANCE_STATUS_SCHEMA",
    "ATTEMPT02_POPULATION_MANIFEST_SCHEMA",
    "ATTEMPT02_POPULATION_PLAN_SCHEMA",
    "ATTEMPT02_POPULATION_PREFLIGHT_SCHEMA",
    "ATTEMPT02_POPULATION_RECEIPT_SCHEMA",
    "FIXED_GATES",
    "build_attempt02_population_launch_preflight",
    "evaluate_attempt02_population_gates",
    "load_and_validate_attempt02_population_plan",
    "validate_attempt02_population_acceptance",
]
