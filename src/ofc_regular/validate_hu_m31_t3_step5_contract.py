"""Validate the frozen M3.1 T3 Step 5 science and seed contract.

The validator is deliberately fail-closed.  It does not generate teacher rows,
open a holdout, start a VM, train a model, resolve ``current``, or activate a
policy.  Passing this contract authorizes only the first immutable Linux Spot
infrastructure canary shard; production fanout remains forbidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


STEP5_CONTRACT_SCHEMA = "hu_joint_policy_m31_t3_step5_contract_v1"
STEP5_VALIDATION_SCHEMA = "hu_joint_policy_m31_t3_step5_validation_v1"
EXPECTED_CANONICAL_CONTRACT_SHA256 = (
    "04c4298feaed78f327f7fb6601f70a6821c3a91dcd2996cc34becbc4bb38e0b4"
)
EXPECTED_SEED_STRIDE = 1_000_003
EXPECTED_HISTORICAL_CONFIG_SEED_MAX = 256_108_071_901
EXPECTED_PLANNED_SEED_MIN = 300_108_071_901

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema",
        "status_date",
        "status",
        "decision",
        "scope",
        "prerequisites",
        "baseline_invariants",
        "information_set",
        "action_contract",
        "teacher_search",
        "behavior_population",
        "seed_contract",
        "data_quality_gates",
        "model_candidate_gates",
        "promotion_evaluation",
        "exploitability_proxy",
        "spot_execution",
        "activation_guards",
        "forbidden",
        "next_step",
    }
)
_PREREQUISITE_PATHS = {
    "step4_status": "configs/hu_joint_policy_m31_t3_step4_status.json",
    "step4_summary": ("outputs/hu_joint_policy/m31_t3_step4/local1000_v1/summary.json"),
    "step4_audit": "docs/hu_joint_policy_m31_t3_step4_completion_audit.md",
    "policy_registry": "src/ofc_regular/ai_profiles.py",
    "native_library": "target/m31_opt_build/release/ofc_hu_m3_engine.dll",
}
_TEACHER_SCHEDULE_NAMES = (
    "infrastructure_canary",
    "train",
    "safety_fit",
    "threshold_lock",
    "diagnostic_teacher_holdout",
)
_POPULATION_SCHEDULE_NAMES = (
    "development_population",
    "locked_population",
    "locked_abr_probe",
)
_TEACHER_NAMESPACE_KEYS = frozenset(
    {
        "hand_seed_base",
        "behavior_policy_seed_base",
        "candidate_seed_base",
        "evaluation_seed_base",
        "child_policy_seed_base",
        "confirmation_seed_base",
    }
)
_POPULATION_NAMESPACE_KEYS = frozenset(
    {
        "hand_seed_base",
        "actor_policy_seed_base",
        "opponent_policy_seed_base",
        "evaluation_seed_base",
        "child_policy_seed_base",
        "confirmation_seed_base",
    }
)
_BEHAVIOR_PROFILES = (
    "stage19_p0",
    "stage9f_p2",
    "stage7_m5_r10",
    "stage3_baseline",
    "random_exact_final",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def canonical_contract_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be an array")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _hash_string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 string")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError(f"{label} must be a SHA-256 string")
    return normalized


def _require_exact_keys(
    value: Mapping[str, Any], expected: Iterable[str], label: str
) -> None:
    expected_set = set(expected)
    actual_set = set(value)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        unknown = sorted(actual_set - expected_set)
        raise ValueError(f"{label} keys changed: missing={missing}, unknown={unknown}")


def _integer_leaves(value: Any) -> Iterable[int]:
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        yield value
        return
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _integer_leaves(child)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            yield from _integer_leaves(child)


def _seed_like_integers(value: Any) -> Iterable[int]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if "seed" in str(key).lower():
                yield from _integer_leaves(child)
            yield from _seed_like_integers(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            yield from _seed_like_integers(child)


def scan_historical_config_seed_max(
    repo_root: Path,
    *,
    excluded_names: Iterable[str] = (
        "hu_joint_policy_m31_t3_step5_contract.json",
        "hu_joint_policy_m31_t3_step5_status.json",
    ),
) -> tuple[int, int]:
    excluded = set(excluded_names)
    values: list[int] = []
    file_count = 0
    for path in sorted((repo_root / "configs").glob("*.json")):
        if path.name in excluded:
            continue
        payload = _load_json(path)
        file_count += 1
        values.extend(_seed_like_integers(payload))
    if not values:
        raise ValueError("historical config seed scan found no seed values")
    return max(values), file_count


def _planned_seed_values(
    seed_contract: Mapping[str, Any],
) -> tuple[list[int], list[dict[str, Any]]]:
    stride = _integer(seed_contract.get("seed_stride"), "seed stride")
    schedules = _sequence(seed_contract.get("schedules"), "seed schedules")
    all_values: list[int] = []
    rows: list[dict[str, Any]] = []
    names: list[str] = []
    for raw_schedule in schedules:
        schedule = _mapping(raw_schedule, "seed schedule")
        name = schedule.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("seed schedule name must be non-empty")
        names.append(name)
        count = _integer(schedule.get("index_count"), f"{name} index count")
        if count <= 0:
            raise ValueError(f"{name} index count must be positive")
        bases = _mapping(schedule.get("namespace_bases"), f"{name} namespaces")
        expected_namespace_keys = (
            _TEACHER_NAMESPACE_KEYS
            if name in _TEACHER_SCHEDULE_NAMES
            else _POPULATION_NAMESPACE_KEYS
        )
        _require_exact_keys(bases, expected_namespace_keys, f"{name} namespaces")
        schedule_values: list[int] = []
        for namespace, raw_base in bases.items():
            base = _integer(raw_base, f"{name}.{namespace}")
            values = [base + stride * index for index in range(count)]
            schedule_values.extend(values)
            all_values.extend(values)
        rows.append(
            {
                "name": name,
                "index_count": count,
                "namespace_count": len(bases),
                "planned_seed_count": len(schedule_values),
                "min_seed": min(schedule_values),
                "max_seed": max(schedule_values),
            }
        )
    expected_names = _TEACHER_SCHEDULE_NAMES + _POPULATION_SCHEDULE_NAMES
    if tuple(names) != expected_names:
        raise ValueError("seed schedule order or names changed")
    return all_values, rows


def _validate_prerequisites(
    contract: Mapping[str, Any], repo_root: Path
) -> dict[str, Any]:
    prerequisites = _mapping(contract.get("prerequisites"), "prerequisites")
    _require_exact_keys(prerequisites, _PREREQUISITE_PATHS, "prerequisites")
    observed: dict[str, Any] = {}
    for name, expected_relative_path in _PREREQUISITE_PATHS.items():
        entry = _mapping(prerequisites.get(name), f"prerequisite {name}")
        relative_path = entry.get("path")
        if relative_path != expected_relative_path:
            raise ValueError(f"prerequisite {name} path changed")
        expected_hash = _hash_string(entry.get("sha256"), f"{name} sha256")
        path = repo_root / expected_relative_path
        if not path.is_file():
            raise ValueError(f"missing prerequisite: {path}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"prerequisite {name} SHA-256 mismatch")
        observed[name] = {
            "path": expected_relative_path,
            "sha256": actual_hash,
        }

    step4_status = _load_json(repo_root / _PREREQUISITE_PATHS["step4_status"])
    if (
        step4_status.get("status") != "step4_local1000_complete"
        or step4_status.get("m31_complete") is not False
        or step4_status.get("current_profile_changed") is not False
        or step4_status.get("spot_vm_started") is not False
        or step4_status.get("spot_vm_authorized") is not False
        or step4_status.get("spot_prerequisite_local1000_passed") is not True
    ):
        raise ValueError("Step 4 status does not authorize Step 5 contract freezing")
    summary = _load_json(repo_root / _PREREQUISITE_PATHS["step4_summary"])
    integrity = _mapping(summary.get("integrity"), "Step 4 integrity")
    if (
        summary.get("status") != "pass"
        or summary.get("all_gates_passed") is not True
        or integrity.get("fingerprint_count") != 1000
        or integrity.get("unique_fingerprint_count") != 1000
        or integrity.get("step3_fingerprint_overlap_count") != 0
        or integrity.get("step3_hand_seed_overlap_count") != 0
        or integrity.get("candidate_evaluation_rng_overlap_count") != 0
    ):
        raise ValueError("Step 4 summary integrity boundary changed")
    native = _mapping(prerequisites.get("native_library"), "native library")
    if native.get("engine_version") != "ofc_hu_m3_engine/0.1.0":
        raise ValueError("native engine version changed")
    return observed


def _validate_semantics(contract: Mapping[str, Any]) -> dict[str, Any]:
    baseline = _mapping(contract.get("baseline_invariants"), "baseline invariants")
    if (
        tuple(_sequence(baseline.get("fixed_profiles"), "fixed profiles"))
        != ("stage19_p0", "stage18_p1", "stage9f_p2", "stage7_m5_r10")
        or baseline.get("candidate_baseline") != "stage7_m5_r10"
        or baseline.get("current_profile_resolution_allowed") is not False
        or baseline.get("legacy_t3_weights_as_label_source_allowed") is not False
    ):
        raise ValueError("baseline or current-profile invariant changed")

    information = _mapping(contract.get("information_set"), "information set")
    for forbidden in (
        "opponent_private_discards",
        "realized_remaining_deck",
        "world_state_or_replay_truth",
    ):
        if information.get(forbidden) is not False:
            raise ValueError(f"forbidden information enabled: {forbidden}")
    if information.get("unknown_fields_rejected") is not True:
        raise ValueError("unknown information-set fields must fail closed")

    teacher = _mapping(contract.get("teacher_search"), "teacher search")
    if _mapping(teacher.get("infrastructure_canary_budget"), "canary budget") != {
        "candidate_samples": 4,
        "evaluation_samples": 8,
        "downstream_t3_samples": 2,
        "downstream_t4_samples": 0,
    }:
        raise ValueError("infrastructure canary budget changed")
    if _mapping(teacher.get("production_label_budget"), "label budget") != {
        "candidate_samples": 8,
        "evaluation_samples": 32,
        "downstream_t3_samples": 4,
        "downstream_t4_samples": 0,
    }:
        raise ValueError("production label budget changed")
    confirmation = _mapping(
        teacher.get("independent_confirmation"), "independent confirmation"
    )
    if (
        _number(confirmation.get("fraction_of_each_teacher_split"), "fraction") != 0.1
        or confirmation.get("evaluation_samples") != 128
        or confirmation.get("separate_rng_namespace") is not True
        or confirmation.get("used_as_runtime_gate") is not False
        or teacher.get("teacher_values_are_realized_match_ev") is not False
    ):
        raise ValueError("independent teacher confirmation contract changed")

    population = _mapping(contract.get("behavior_population"), "behavior population")
    raw_profiles = _sequence(population.get("profiles"), "behavior profiles")
    profiles = tuple(
        _mapping(row, "behavior profile").get("profile") for row in raw_profiles
    )
    weights = [
        _number(_mapping(row, "behavior profile").get("weight"), "profile weight")
        for row in raw_profiles
    ]
    if (
        profiles != _BEHAVIOR_PROFILES
        or any(not math.isclose(weight, 0.2) for weight in weights)
        or not math.isclose(sum(weights), 1.0)
        or population.get("current_profile_allowed") is not False
    ):
        raise ValueError("behavior population changed")

    promotion = _mapping(contract.get("promotion_evaluation"), "promotion evaluation")
    if (
        promotion.get("paired_seeds_per_opponent") != 1000
        or tuple(_sequence(promotion.get("opponents"), "promotion opponents"))
        != _BEHAVIOR_PROFILES
        or promotion.get("valid_overrides_min") != 300
        or promotion.get("valid_overrides_each_seat_min") != 100
        or promotion.get("threshold_reselection_allowed") is not False
        or promotion.get("posthoc_seed_extension_allowed") is not False
        or promotion.get("teacher_values_reported_as_match_ev") is not False
    ):
        raise ValueError("promotion evaluation contract changed")
    for metric in (
        "realized_gain_per_override_ci95_low_min_exclusive",
        "paired_delta_ev_per_hand_ci95_low_min_exclusive",
        "first_seat_delta_ev_per_hand_ci95_low_min_exclusive",
        "second_seat_delta_ev_per_hand_ci95_low_min_exclusive",
    ):
        if _number(promotion.get(metric), metric) != 0.0:
            raise ValueError(f"positive-CI promotion gate changed: {metric}")

    spot = _mapping(contract.get("spot_execution"), "Spot execution")
    if (
        spot.get("initial_authorization") != "shard_0_only"
        or spot.get("canary_rows_eligible_for_training") is not False
        or spot.get("production_quality_pilot_required_before_fanout") is not True
        or spot.get("production_fanout_authorized") is not False
        or spot.get("shard_0_preemption_resume_drill_required") is not True
    ):
        raise ValueError("Spot authorization boundary changed")
    guards = _mapping(contract.get("activation_guards"), "activation guards")
    if any(value is not False for value in guards.values()):
        raise ValueError("an activation guard is enabled")
    return {
        "behavior_profiles": list(profiles),
        "promotion_opponents": list(_BEHAVIOR_PROFILES),
        "canary_budget": dict(teacher["infrastructure_canary_budget"]),
        "production_label_budget": dict(teacher["production_label_budget"]),
    }


def validate_contract_payload(
    contract: Mapping[str, Any],
    *,
    repo_root: Path,
    verify_prerequisites: bool = True,
    verify_historical_scan: bool = True,
) -> dict[str, Any]:
    _require_exact_keys(contract, _TOP_LEVEL_KEYS, "Step 5 contract")
    if contract.get("schema") != STEP5_CONTRACT_SCHEMA:
        raise ValueError("Step 5 contract schema changed")
    canonical_sha = canonical_contract_sha256(contract)
    if canonical_sha != EXPECTED_CANONICAL_CONTRACT_SHA256:
        raise ValueError("frozen Step 5 canonical contract SHA-256 changed")
    if (
        contract.get("status") != "frozen_before_spot_canary_or_teacher_generation"
        or contract.get("scope")
        != "m31_t3_step5_contract_only_no_model_training_evaluation_or_activation"
    ):
        raise ValueError("Step 5 status or scope changed")

    semantic_observed = _validate_semantics(contract)
    seed_contract = _mapping(contract.get("seed_contract"), "seed contract")
    if (
        seed_contract.get("seed_stride") != EXPECTED_SEED_STRIDE
        or seed_contract.get("historical_config_seed_max_at_freeze")
        != EXPECTED_HISTORICAL_CONFIG_SEED_MAX
        or seed_contract.get("planned_seed_min") != EXPECTED_PLANNED_SEED_MIN
        or seed_contract.get("alternate_seed_after_results_allowed") is not False
        or seed_contract.get("optional_stopping_or_posthoc_extension_allowed")
        is not False
    ):
        raise ValueError("seed freeze constants changed")
    planned_values, schedule_rows = _planned_seed_values(seed_contract)
    unique_values = set(planned_values)
    if len(unique_values) != len(planned_values):
        raise ValueError("planned seed namespaces overlap")
    planned_min = min(unique_values)
    planned_max = max(unique_values)
    if planned_min != EXPECTED_PLANNED_SEED_MIN:
        raise ValueError("planned minimum seed changed")

    historical_max = EXPECTED_HISTORICAL_CONFIG_SEED_MAX
    historical_file_count: int | None = None
    if verify_historical_scan:
        historical_max, historical_file_count = scan_historical_config_seed_max(
            repo_root
        )
        if historical_max != EXPECTED_HISTORICAL_CONFIG_SEED_MAX:
            raise ValueError(
                "historical config seed maximum changed; re-audit freshness before use"
            )
    if historical_max >= planned_min:
        raise ValueError("planned seeds are not above the frozen historical boundary")

    prerequisites: dict[str, Any] | None = None
    if verify_prerequisites:
        prerequisites = _validate_prerequisites(contract, repo_root)

    sorted_seed_bytes = json.dumps(
        sorted(unique_values), separators=(",", ":")
    ).encode()
    seed_digest = hashlib.sha256(sorted_seed_bytes).hexdigest()
    teacher_schedules = {
        row["name"]: row
        for row in schedule_rows
        if row["name"] in _TEACHER_SCHEDULE_NAMES
    }
    for name in _TEACHER_SCHEDULE_NAMES:
        schedule = next(
            _mapping(row, f"{name} schedule")
            for row in _sequence(seed_contract["schedules"], "seed schedules")
            if _mapping(row, "seed schedule").get("name") == name
        )
        if schedule.get("root_count") != 2 * schedule.get("index_count"):
            raise ValueError(f"{name} is not exactly balanced two-seat root generation")
        if schedule.get("index_count") % len(_BEHAVIOR_PROFILES) != 0:
            raise ValueError(f"{name} cannot have exact equal behavior quotas")

    return {
        "schema": STEP5_VALIDATION_SCHEMA,
        "status": "pass",
        "decision": "spot_shard_0_canary_authorized_production_fanout_no_go",
        "contract_canonical_sha256": canonical_sha,
        "checks": {
            "top_level_schema_and_keys_frozen": True,
            "numeric_and_science_contract_hash_frozen": True,
            "information_boundary_hidden_discard_safe": True,
            "all_legal_action_mapping_required": True,
            "teacher_candidate_evaluation_confirmation_rng_separate": True,
            "teacher_values_not_match_ev_or_runtime_gate": True,
            "behavior_population_balanced_and_current_forbidden": True,
            "planned_seed_values_globally_unique": True,
            "planned_seeds_above_historical_config_boundary": True,
            "teacher_splits_exactly_seat_balanced": True,
            "threshold_and_locked_splits_not_training_data": True,
            "paired_both_seat_realized_promotion_gates_frozen": True,
            "exploitability_proxy_frozen_without_nash_claim": True,
            "spot_authorization_shard_0_only": True,
            "production_fanout_forbidden_before_quality_pilot": True,
            "activation_guards_all_false": True,
            "prerequisites_hash_bound": verify_prerequisites,
            "historical_config_scan_performed": verify_historical_scan,
        },
        "seed_audit": {
            "seed_stride": EXPECTED_SEED_STRIDE,
            "historical_config_seed_max": historical_max,
            "historical_config_file_count": historical_file_count,
            "planned_seed_count": len(planned_values),
            "unique_planned_seed_count": len(unique_values),
            "planned_seed_min": planned_min,
            "planned_seed_max": planned_max,
            "planned_seed_set_sha256": seed_digest,
            "schedules": schedule_rows,
        },
        "teacher_roots": {
            name: teacher_schedules[name]["index_count"] * 2
            for name in _TEACHER_SCHEDULE_NAMES
        },
        "observed": semantic_observed,
        "prerequisites": prerequisites,
        "spot_canary_authorized": True,
        "spot_vm_started": False,
        "production_fanout_authorized": False,
        "current_profile_changed": False,
        "m31_complete": False,
    }


def validate_contract_file(
    contract_path: Path,
    *,
    repo_root: Path,
    verify_prerequisites: bool = True,
    verify_historical_scan: bool = True,
) -> dict[str, Any]:
    payload = _load_json(contract_path)
    result = validate_contract_payload(
        payload,
        repo_root=repo_root,
        verify_prerequisites=verify_prerequisites,
        verify_historical_scan=verify_historical_scan,
    )
    result["contract_path"] = str(contract_path)
    result["contract_byte_sha256"] = _sha256(contract_path)
    result["validator_source_sha256"] = _sha256(Path(__file__).resolve())
    return result


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/hu_joint_policy_m31_t3_step5_contract.json"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    contract_path = (
        args.contract if args.contract.is_absolute() else repo_root / args.contract
    ).resolve()
    result = validate_contract_file(contract_path, repo_root=repo_root)
    _write_json_atomic(args.output, result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
