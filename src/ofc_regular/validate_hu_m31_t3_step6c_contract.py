"""Fail-closed validation for the M3.1 T3 Step 6c pilot contract.

Passing this validator freezes only a 100-root production-label quality-pilot
contract.  It does not launch Spot, generate rows, authorize teacher fanout or
training, resolve ``current``, add a profile, or activate a policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hu_m31_t3_behavior_roots import M31_T3_BEHAVIOR_PROFILES
from .hu_m31_t3_step6c_contract import (
    ACCEPTED_FEATURE_ENCODER_SHA256,
    ACCEPTED_NATIVE_LIBRARY_SHA256,
    CONFIRMATION_BUDGET,
    CONFIRMATION_FRACTION,
    CONFIRMATION_HAND_INDICES,
    CONFIRMATION_REGRET_MAX,
    CONFIRMATION_REGRET_MEAN_MAX,
    CONFIRMATION_REGRET_P95_MAX,
    CONFIRMATION_REGRET_P99_MAX,
    CONFIRMATION_ROOT_INDICES,
    EXPECTED_STEP6C_CONTRACT_BYTE_SHA256,
    EXPECTED_STEP6C_CONTRACT_CANONICAL_SHA256,
    MAX_FIRST_P95_SECONDS,
    MAX_PEAK_RSS_BYTES,
    MAX_SECOND_P95_SECONDS,
    PERCENTILE_METHOD,
    PILOT_HAND_COUNT,
    PILOT_HAND_INDICES,
    PILOT_HANDS_PER_PROFILE,
    PILOT_HANDS_PER_SHARD,
    PILOT_ROOT_COUNT,
    PILOT_ROOT_INDICES,
    PILOT_ROOTS_PER_PROFILE,
    PILOT_ROOTS_PER_SHARD,
    PILOT_SHARD_COUNT,
    POLICY_REGISTRY_SHA256,
    PRODUCTION_LABEL_BUDGET,
    SEED_STRIDE,
    STEP5_CONTRACT_BYTE_SHA256,
    STEP5_CONTRACT_CANONICAL_SHA256,
    STEP5_VALIDATION_SHA256,
    STEP6B_STATUS_SHA256,
    STEP6B_VALIDATION_SHA256,
    STEP6C_BEHAVIOR_SCHEDULE_SCHEMA,
    STEP6C_CONTRACT_SCHEMA,
    STEP6C_RUN_ID,
    STEP6C_SPLIT,
    STEP6C_VALIDATION_SCHEMA,
    TRAIN_BEHAVIOR_SEED_BASE,
    TRAIN_CANDIDATE_SEED_BASE,
    TRAIN_CHILD_SEED_BASE,
    TRAIN_CONFIRMATION_SEED_BASE,
    TRAIN_EVALUATION_SEED_BASE,
    TRAIN_HAND_COUNT,
    TRAIN_HAND_SEED_BASE,
    TRAIN_HANDS_PER_PROFILE,
    TRAIN_ROOT_COUNT,
    canonical_bytes,
    canonical_sha256,
    nearest_rank_percentile,
    profile_counts,
    schedule_rows,
    seed_set,
    validate_frozen_schedule,
)


_TOP_LEVEL_KEYS = frozenset(
    {
        "schema",
        "status_date",
        "status",
        "decision",
        "scope",
        "anchors",
        "immutable_boundaries",
        "train_split",
        "behavior_schedule",
        "seed_contract",
        "search",
        "confirmation_subset",
        "data_quality_gates",
        "operational_gates",
        "activation_guards",
        "forbidden",
        "next_step",
    }
)
_ANCHOR_SPECS = {
    "step5_contract": (
        "configs/hu_joint_policy_m31_t3_step5_contract.json",
        STEP5_CONTRACT_BYTE_SHA256,
    ),
    "step5_validation": (
        "outputs/hu_joint_policy/m31_t3_step5/contract_validation_v2.json",
        STEP5_VALIDATION_SHA256,
    ),
    "step6b_status": (
        "configs/hu_joint_policy_m31_t3_step6b_status.json",
        STEP6B_STATUS_SHA256,
    ),
    "step6b_merged_validation": (
        "outputs/hu_joint_policy/m31_t3_step6b/canary500_validation_v1.json",
        STEP6B_VALIDATION_SHA256,
    ),
    "policy_registry": ("src/ofc_regular/ai_profiles.py", POLICY_REGISTRY_SHA256),
    "accepted_linux_native_library": (
        "outputs/gcp_runs/regular-hu-m31-step6b-canary-20260717-001/"
        "package_src/native/release/libofc_hu_m3_engine.so",
        ACCEPTED_NATIVE_LIBRARY_SHA256,
    ),
    "accepted_linux_feature_encoder": (
        "outputs/gcp_runs/regular-hu-m31-step6b-canary-20260717-001/"
        "package_src/target/release/libofc_stage3_feature_encoder.so",
        ACCEPTED_FEATURE_ENCODER_SHA256,
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        raise ValueError(
            f"{label} fields changed: missing={missing}, unknown={unknown}"
        )


def _validate_anchor_payloads(
    anchors: Mapping[str, Any], repo_root: Path
) -> dict[str, Any]:
    if set(anchors) != set(_ANCHOR_SPECS):
        raise ValueError("Step 6c anchor set changed")
    observed: dict[str, Any] = {}
    for name, (relative_path, expected_sha) in _ANCHOR_SPECS.items():
        entry = _mapping(anchors.get(name), f"anchor {name}")
        if entry.get("path") != relative_path:
            raise ValueError(f"Step 6c anchor path changed: {name}")
        declared_sha = entry.get("byte_sha256", entry.get("sha256"))
        if declared_sha != expected_sha:
            raise ValueError(f"Step 6c anchor declared hash changed: {name}")
        path = repo_root / relative_path
        if not path.is_file() or _sha256(path) != expected_sha:
            raise ValueError(f"Step 6c anchor file/hash mismatch: {name}")
        observed[name] = {"path": relative_path, "sha256": expected_sha}

    step5_entry = _mapping(anchors["step5_contract"], "Step 5 anchor")
    if step5_entry.get("canonical_sha256") != STEP5_CONTRACT_CANONICAL_SHA256:
        raise ValueError("Step 5 canonical anchor changed")
    step5 = _load_json(repo_root / _ANCHOR_SPECS["step5_contract"][0])
    if canonical_sha256(step5) != STEP5_CONTRACT_CANONICAL_SHA256:
        raise ValueError("live Step 5 canonical contract changed")

    native = _mapping(anchors["accepted_linux_native_library"], "native anchor")
    if native.get("engine_version") != "ofc_hu_m3_engine/0.1.0":
        raise ValueError("accepted native engine version changed")

    step6b_status = _load_json(repo_root / _ANCHOR_SPECS["step6b_status"][0])
    if (
        step6b_status.get("status") != "step6b_infrastructure_canary_complete"
        or step6b_status.get("spot_production_fanout_authorized") is not False
        or step6b_status.get("canary_rows_eligible_for_training") is not False
        or step6b_status.get("current_profile_changed") is not False
    ):
        raise ValueError("Step 6b status no longer preserves the No-Go boundary")
    step6b = _load_json(repo_root / _ANCHOR_SPECS["step6b_merged_validation"][0])
    if (
        step6b.get("schema") != "hu_m31_t3_step6b_canary_validation_v1"
        or step6b.get("all_gates_passed") is not True
        or step6b.get("training_eligible") is not False
        or step6b.get("production_fanout_authorized") is not False
        or step6b.get("current_profile_changed") is not False
    ):
        raise ValueError("Step 6b merged validation is not an accepted No-Go anchor")
    return observed


def _validate_semantics(contract: Mapping[str, Any]) -> dict[str, Any]:
    immutable = _mapping(contract.get("immutable_boundaries"), "immutable boundaries")
    if any(
        immutable.get(name) is not False
        for name in (
            "step5_contract_edited",
            "infrastructure_canary_rows_training_eligible",
            "pilot_rows_training_authorized",
            "production_teacher_fanout_authorized",
            "current_profile_resolution_allowed",
            "named_profile_addition_allowed",
            "runtime_activation_allowed",
            "full_replacement_allowed",
            "teacher_values_are_realized_match_ev",
        )
    ):
        raise ValueError("Step 6c immutable No-Go boundary changed")

    split = _mapping(contract.get("train_split"), "train split")
    pilot_hands = _mapping(split.get("pilot_hand_indices"), "pilot hand range")
    pilot_roots = _mapping(split.get("pilot_root_indices"), "pilot root range")
    if (
        split.get("name") != STEP6C_SPLIT
        or split.get("full_paired_hands") != TRAIN_HAND_COUNT
        or split.get("full_roots") != TRAIN_ROOT_COUNT
        or split.get("full_hand_index_start") != 0
        or split.get("full_hand_index_stop_exclusive") != TRAIN_HAND_COUNT
        or dict(pilot_hands)
        != {"start": 0, "stop_exclusive": PILOT_HAND_COUNT, "count": PILOT_HAND_COUNT}
        or dict(pilot_roots)
        != {"start": 0, "stop_exclusive": PILOT_ROOT_COUNT, "count": PILOT_ROOT_COUNT}
        or split.get("pilot_shards") != PILOT_SHARD_COUNT
        or split.get("paired_hands_per_shard") != PILOT_HANDS_PER_SHARD
        or split.get("roots_per_shard") != PILOT_ROOTS_PER_SHARD
        or split.get("first_roots") != 50
        or split.get("second_roots") != 50
        or split.get("pilot_rows_quarantined_until_separate_training_authorization")
        is not True
    ):
        raise ValueError("Step 6c train/pilot split changed")

    schedule = _mapping(contract.get("behavior_schedule"), "behavior schedule")
    if (
        schedule.get("schema") != STEP6C_BEHAVIOR_SCHEDULE_SCHEMA
        or schedule.get("resolution")
        != "new_versioned_production_schedule_without_mutating_step5_or_canary"
        or schedule.get("split") != STEP6C_SPLIT
        or schedule.get("block_hands") != 5
        or tuple(_sequence(schedule.get("profile_order"), "profile order"))
        != M31_T3_BEHAVIOR_PROFILES
        or schedule.get("content_dependent") is not False
        or schedule.get("full_hands_per_profile") != TRAIN_HANDS_PER_PROFILE
        or schedule.get("pilot_hands_per_profile") != PILOT_HANDS_PER_PROFILE
        or schedule.get("pilot_roots_per_profile") != PILOT_ROOTS_PER_PROFILE
        or schedule.get("pilot_shard_hands_per_profile") != 5
        or schedule.get("full_schedule_sha256") != canonical_sha256(schedule_rows())
        or schedule.get("pilot_schedule_sha256")
        != canonical_sha256(schedule_rows(PILOT_HAND_INDICES))
    ):
        raise ValueError("Step 6c behavior schedule changed")

    seeds = _mapping(contract.get("seed_contract"), "seed contract")
    expected_bases = {
        "hand": TRAIN_HAND_SEED_BASE,
        "behavior": TRAIN_BEHAVIOR_SEED_BASE,
        "candidate": TRAIN_CANDIDATE_SEED_BASE,
        "evaluation": TRAIN_EVALUATION_SEED_BASE,
        "child": TRAIN_CHILD_SEED_BASE,
        "confirmation": TRAIN_CONFIRMATION_SEED_BASE,
    }
    pilot_seed_values = seed_set(PILOT_HAND_INDICES)
    if (
        seeds.get("seed_stride") != SEED_STRIDE
        or seeds.get("namespace_bases") != expected_bases
        or seeds.get("pilot_namespace_seed_values") != PILOT_HAND_COUNT * 6
        or seeds.get("pilot_seed_min") != min(pilot_seed_values)
        or seeds.get("pilot_seed_max") != max(pilot_seed_values)
        or seeds.get("pilot_seed_values_unique") is not True
        or seeds.get("overlap_with_infrastructure_canary_namespace") is not False
        or seeds.get("alternate_seed_after_results_allowed") is not False
        or seeds.get("posthoc_extension_allowed") is not False
    ):
        raise ValueError("Step 6c seed contract changed")

    search = _mapping(contract.get("search"), "search contract")
    if (
        search.get("run_id") != STEP6C_RUN_ID
        or search.get("primary_budget") != PRODUCTION_LABEL_BUDGET.to_dict()
        or search.get("confirmation_budget") != CONFIRMATION_BUDGET.to_dict()
        or search.get("confirmation_candidate_seed_source") != "candidate"
        or search.get("confirmation_evaluation_seed_source") != "confirmation"
        or search.get("confirmation_continuation_seed_source") != "child"
        or search.get("confirmation_reuses_primary_run_id") is not True
        or search.get("confirmation_candidate_rng_keys_equal_primary_subset")
        is not True
        or search.get("confirmation_selected_action_source")
        != "primary_candidate_selected_action_key"
        or search.get("confirmation_action_join_key") != "regular_ofc_action_key_v1"
        or search.get("confirmation_may_replace_primary_selection") is not False
        or search.get("common_random_futures_within_each_action_comparison") is not True
        or search.get("candidate_evaluation_and_confirmation_evaluation_rng_disjoint")
        is not True
        or search.get("all_legal_actions_required") is not True
        or search.get("original_index_and_order_mapping_required") is not True
        or search.get("downstream_t4") != "exact_native_enumeration"
        or search.get("teacher_value_status") != "diagnostic_not_match_EV"
    ):
        raise ValueError("Step 6c search/confirmation semantics changed")

    confirmation = _mapping(contract.get("confirmation_subset"), "confirmation")
    if (
        confirmation.get("selection_unit") != "paired_hand"
        or confirmation.get("selected_before_game_content") is not True
        or tuple(confirmation.get("paired_hand_indices", ()))
        != CONFIRMATION_HAND_INDICES
        or tuple(confirmation.get("root_indices", ())) != CONFIRMATION_ROOT_INDICES
        or confirmation.get("paired_hands") != len(CONFIRMATION_HAND_INDICES)
        or confirmation.get("roots") != len(CONFIRMATION_ROOT_INDICES)
        or confirmation.get("fraction_of_pilot_roots") != CONFIRMATION_FRACTION
        or confirmation.get("first_roots") != 5
        or confirmation.get("second_roots") != 5
        or confirmation.get("roots_per_profile") != 2
        or confirmation.get("evaluation_samples")
        != CONFIRMATION_BUDGET.evaluation_samples
        or confirmation.get("all_legal_actions") is not True
        or confirmation.get("expected_primary_candidate_rng_keys") != 800
        or confirmation.get("expected_primary_evaluation_rng_keys") != 3200
        or confirmation.get("expected_confirmation_evaluation_rng_keys") != 1280
        or confirmation.get("primary_candidate_evaluation_overlap_max") != 0
        or confirmation.get("primary_candidate_confirmation_evaluation_overlap_max")
        != 0
        or confirmation.get("primary_evaluation_confirmation_evaluation_overlap_max")
        != 0
        or confirmation.get("percentile_method") != PERCENTILE_METHOD
        or confirmation.get("ten_root_p95_and_p99_equal_max") is not True
        or confirmation.get("selected_regret_mean_max") != CONFIRMATION_REGRET_MEAN_MAX
        or confirmation.get("selected_regret_p95_max") != CONFIRMATION_REGRET_P95_MAX
        or confirmation.get("selected_regret_p99_max") != CONFIRMATION_REGRET_P99_MAX
        or confirmation.get("selected_regret_max") != CONFIRMATION_REGRET_MAX
        or confirmation.get("top1_agreement_is_diagnostic_only") is not True
    ):
        raise ValueError("Step 6c confirmation subset or gates changed")

    operational = _mapping(contract.get("operational_gates"), "operational gates")
    if (
        operational.get("first_seat_p95_seconds_max") != MAX_FIRST_P95_SECONDS
        or operational.get("second_seat_p95_seconds_max") != MAX_SECOND_P95_SECONDS
        or operational.get("peak_process_rss_bytes_max") != MAX_PEAK_RSS_BYTES
        or operational.get(
            "production_budget_speed_smoke_required_before_spot_authorization"
        )
        is not True
        or operational.get("separate_launch_authorization_required") is not True
        or operational.get("spot_quality_pilot_authorized") is not False
    ):
        raise ValueError("Step 6c operational authorization boundary changed")

    guards = _mapping(contract.get("activation_guards"), "activation guards")
    if any(value is not False for value in guards.values()):
        raise ValueError("Step 6c activation or training guard enabled")

    quality = _mapping(contract.get("data_quality_gates"), "data quality gates")
    zero_gates = (
        "forbidden_hidden_truth_rows_max",
        "unknown_field_rows_max",
        "invalid_action_mapping_rows_max",
        "nonfinite_action_value_rows_max",
        "duplicate_observation_fingerprints_max",
        "primary_candidate_evaluation_rng_overlap_max",
        "primary_candidate_confirmation_evaluation_rng_overlap_max",
        "primary_evaluation_confirmation_evaluation_rng_overlap_max",
        "ambiguous_rows_dropped_max",
    )
    if (
        any(quality.get(name) != 0 for name in zero_gates)
        or quality.get("complete_legal_action_value_fraction_min") != 1.0
        or quality.get("exact_t4_decision_fraction_min") != 1.0
        or quality.get("first_second_root_ratio_exact") != 1.0
        or quality.get("confirmation_fraction_exact") != CONFIRMATION_FRACTION
        or quality.get("action_geometry_distribution_is_pilot_diagnostic_only")
        is not True
        or quality.get("failed_gate_decision")
        != "complete_no_go_no_reseed_no_extension_no_same_data_threshold_reselection"
    ):
        raise ValueError("Step 6c data-quality gates changed")

    schedule_evidence = validate_frozen_schedule()
    confirmation_profile_counts = profile_counts(CONFIRMATION_HAND_INDICES)
    if any(count != 1 for count in confirmation_profile_counts.values()):
        raise ValueError("confirmation subset is not profile balanced")
    example = tuple(float(value) for value in range(10))
    if nearest_rank_percentile(example, 0.95) != max(
        example
    ) or nearest_rank_percentile(example, 0.99) != max(example):
        raise ValueError("nearest-rank ten-root tail semantics changed")
    return {
        "schedule": dict(schedule_evidence),
        "confirmation_profile_counts": confirmation_profile_counts,
        "nearest_rank_ten_root_p95_equals_max": True,
        "nearest_rank_ten_root_p99_equals_max": True,
    }


def validate_contract_payload(
    contract: Mapping[str, Any], *, repo_root: Path, verify_anchors: bool = True
) -> dict[str, Any]:
    _require_exact_keys(contract, _TOP_LEVEL_KEYS, "Step 6c contract")
    if contract.get("schema") != STEP6C_CONTRACT_SCHEMA:
        raise ValueError("Step 6c contract schema changed")
    observed_canonical = canonical_sha256(contract)
    if observed_canonical != EXPECTED_STEP6C_CONTRACT_CANONICAL_SHA256:
        raise ValueError("Step 6c canonical contract SHA-256 changed")
    if (
        contract.get("status") != "frozen_before_production_label_quality_pilot"
        or contract.get("scope")
        != "m31_t3_step6c_contract_only_no_search_cloud_launch_training_fanout_or_activation"
    ):
        raise ValueError("Step 6c status/scope changed")
    semantics = _validate_semantics(contract)
    anchors = (
        _validate_anchor_payloads(
            _mapping(contract.get("anchors"), "anchors"), repo_root
        )
        if verify_anchors
        else None
    )
    return {
        "schema": STEP6C_VALIDATION_SCHEMA,
        "status": "pass",
        "decision": "quality_pilot_contract_frozen_spot_launch_requires_separate_authorization",
        "scope": "contract_only_no_search_cloud_training_fanout_or_activation",
        "contract_canonical_sha256": observed_canonical,
        "contract_byte_sha256": EXPECTED_STEP6C_CONTRACT_BYTE_SHA256,
        "checks": {
            "step5_contract_preserved": True,
            "step6b_500_root_canary_anchored": verify_anchors,
            "accepted_linux_binaries_anchored": verify_anchors,
            "full_train_equal_quota_schedule_frozen": True,
            "pilot_two_shard_schedule_frozen": True,
            "six_seed_namespaces_disjoint": True,
            "confirmation_subset_content_independent_and_balanced": True,
            "primary_and_confirmation_budgets_frozen": True,
            "nearest_rank_percentiles_frozen_before_results": True,
            "all_quality_thresholds_frozen": True,
            "infrastructure_rows_training_forbidden": True,
            "fanout_training_current_profile_activation_forbidden": True,
        },
        "schedule": semantics["schedule"],
        "confirmation": {
            "hand_indices": list(CONFIRMATION_HAND_INDICES),
            "root_indices": list(CONFIRMATION_ROOT_INDICES),
            "profile_counts": semantics["confirmation_profile_counts"],
            "primary_budget": PRODUCTION_LABEL_BUDGET.to_dict(),
            "confirmation_budget": CONFIRMATION_BUDGET.to_dict(),
            "percentile_method": PERCENTILE_METHOD,
            "ten_root_p95_and_p99_equal_max": True,
        },
        "anchors": anchors,
        "spot_quality_pilot_authorized": False,
        "production_teacher_fanout_authorized": False,
        "training_authorized": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "m31_complete": False,
    }


def validate_contract_file(
    contract_path: Path, *, repo_root: Path, verify_anchors: bool = True
) -> dict[str, Any]:
    if _sha256(contract_path) != EXPECTED_STEP6C_CONTRACT_BYTE_SHA256:
        raise ValueError("Step 6c contract byte SHA-256 changed")
    return validate_contract_payload(
        _load_json(contract_path), repo_root=repo_root, verify_anchors=verify_anchors
    )


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Step 6c validation: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(canonical_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/hu_joint_policy_m31_t3_step6c_contract.json"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = validate_contract_file(
        args.contract.resolve(), repo_root=args.repo_root.resolve()
    )
    if args.output is not None:
        _write_once(args.output.resolve(), report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "main",
    "validate_contract_file",
    "validate_contract_payload",
]
