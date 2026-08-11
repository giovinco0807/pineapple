"""Apply the frozen Attempt13 Development200 gates exactly once.

The aggregation and write-once selector mechanics are inherited from the
validated Attempt12 implementation.  This module scopes every plan, teacher,
seed, schema, gate, and authorization identity to Attempt13.
"""

from __future__ import annotations

import argparse
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from . import select_hu_m43_attempt09_development as _engine
from . import select_hu_m43_attempt12_development as _base
from .hu_m43_attempt12_contract import AI_PROFILES_SHA256
from .hu_m43_attempt13_contract import (
    M43_ATTEMPT13_PLAN_SHA256,
    M43_ATTEMPT13_PROFILES,
    enumerate_attempt13_seed_schedules,
    load_and_validate_attempt13_plan,
    validate_attempt13_plan,
)
from .hu_m43_attempt13_teacher import (
    Attempt13TeacherConfig,
    validate_attempt13_teacher_output,
)
from .run_hu_m43_attempt13 import (
    ATTEMPT13_AUTHORIZATION_SCHEMA,
    ATTEMPT13_BASELINE_PROFILE,
    ATTEMPT13_CONTINUATION_PROFILE,
    ATTEMPT13_LAMBDA_MODEL_SHA256,
    ATTEMPT13_ROOT_CONTRACT_SCHEMA,
    ATTEMPT13_ROW_SCHEMA,
    _attempt13_bindings as _runner_bindings,
)


ATTEMPT13_DEVELOPMENT_DECISION_SCHEMA = "hu_m43_attempt13_development_go_no_go_v1"
ATTEMPT13_DEVELOPMENT_RECEIPT_SCHEMA = (
    "hu_m43_attempt13_development_selector_receipt_v1"
)
ATTEMPT13_TEACHER_VALIDATION_SCHEMA = "hu_m43_attempt13_teacher_validation_v1"

_LOCK = threading.RLock()
_ATTEMPT13_GATE_RELATIVE = "artifacts/attempt13/preceding_gate.json"
_CANDIDATE_NONBASELINE_COUNT = object()
_ASSESSMENT_SOURCE = (
    "disjoint_E512_locked_final_nonbaseline_output_vs_explicit_baseline"
)
_INTEGRITY_NAMES = (
    "action_mapping",
    "rng_domain",
    "hidden_information",
    "risk_reserve_contract",
    "retained_order",
    "phase_filter",
    "pooled_phase",
    "locked_action_change",
    "extreme_tail_statistic",
)
_ATTEMPT13_DEVELOPMENT_GATES = {
    "fires_total_min": 40,
    "fires_each_profile_min": 3,
    "mean_delta_per_state_strictly_greater_than": 0,
    "mean_delta_per_fire_strictly_greater_than": 0,
    "false_positive_rate_per_fire_max": 0.4,
    "override_loss_p95_max": 25,
    "override_loss_p99_max": 40,
    "override_loss_max": 50,
    "action_mapping_violation_count_max": 0,
    "rng_domain_violation_count_max": 0,
    "hidden_information_violation_count_max": 0,
    "risk_reserve_contract_violation_count_max": 0,
    "retained_order_violation_count_max": 0,
    "phase_filter_violation_count_max": 0,
    "pooled_phase_violation_count_max": 0,
    "locked_action_change_violation_count_max": 0,
    "extreme_tail_statistic_violation_count_max": 0,
    "nonfire_exact_baseline_action_fallback_required": True,
    "teacher_values_reported_as_realized_match_ev": False,
    "raw_minimum_search_gate_allowed": False,
}
_ATTEMPT13_OPEN_PHASE_ORDER = (
    "rerank_r128",
    "veto_v256",
    "stress_x1024",
    "confirmation_c1024",
    "evaluation_e512",
)
_ATTEMPT13_PHASE_MAPPING_FIELDS = (
    "rng_key_digests",
    "belief_digests",
    "phase_child_information_set_counts",
)
_ATTEMPT12_AGGREGATE_ROWS = _base._aggregate_attempt12_rows


def _expected_search_config(
    config: Attempt13TeacherConfig, seeds: Mapping[str, int]
) -> dict[str, Any]:
    return {
        "all_legal_candidates": config.all_legal_candidates,
        "candidate_nonbaseline_count": _CANDIDATE_NONBASELINE_COUNT,
        "baseline_added_exactly_once": True,
        "rerank_samples": config.rerank_samples,
        "rerank_head_max": config.rerank_head_max,
        "shortlist_max": config.shortlist_max,
        "veto_samples": config.veto_samples,
        "stress_samples": config.stress_samples,
        "confirmation_samples": config.confirmation_samples,
        "evaluation_samples": config.evaluation_samples,
        "hand_seed": seeds["hand"],
        "rerank_seed": seeds["rerank"],
        "veto_seed": seeds["veto"],
        "stress_seed": seeds["stress"],
        "confirmation_seed": seeds["confirmation"],
        "evaluation_seed": seeds["evaluation"],
        "child_policy_seed": seeds["child"],
        "run_id": config.run_id,
        "batch_child_selectors": True,
    }


def _expected_seed_domain_provenance(seeds: Mapping[str, int]) -> dict[str, Any]:
    return {
        "domain_order": [
            "hand_external",
            "rerank_r128",
            "veto_v256",
            "stress_x1024",
            "confirmation_c1024",
            "evaluation_e512",
            "child_policy",
        ],
        "hand_external": seeds["hand"],
        "rerank_r128": seeds["rerank"],
        "veto_v256": seeds["veto"],
        "stress_x1024": seeds["stress"],
        "confirmation_c1024": seeds["confirmation"],
        "evaluation_e512": seeds["evaluation"],
        "child_policy": seeds["child"],
        "all_seven_base_seeds_pairwise_distinct": True,
        "hand_sampled_inside_teacher": False,
    }


def _normalize_attempt13_phase_order(
    row: Mapping[str, Any], *, root: int
) -> Mapping[str, Any]:
    """Restore canonical JSON phase mappings to the frozen logical order."""

    teacher = row.get("teacher")
    if not isinstance(teacher, Mapping):
        return row
    rng = teacher.get("rng_key_digests")
    if not isinstance(rng, Mapping):
        return row

    logical_order = [
        phase for phase in _ATTEMPT13_OPEN_PHASE_ORDER if phase in rng
    ]
    accepted_orders = (logical_order, sorted(logical_order))
    normalized_teacher = dict(teacher)
    for field in _ATTEMPT13_PHASE_MAPPING_FIELDS:
        phase_mapping = teacher.get(field)
        if not isinstance(phase_mapping, Mapping):
            # Preserve the inherited strict type/schema failure for malformed
            # rows; this shim only reconciles the two valid key orders.
            return row
        actual_order = list(phase_mapping)
        if actual_order not in accepted_orders:
            if field == "rng_key_digests":
                raise ValueError(
                    f"Attempt13 opened RNG phase order changed at root {root}"
                )
            raise ValueError(
                "Attempt13 opened phase provenance changed "
                f"at root {root}:{field}"
            )
        normalized_teacher[field] = {
            phase: phase_mapping[phase] for phase in logical_order
        }

    normalized_row = dict(row)
    normalized_row["teacher"] = normalized_teacher
    return normalized_row


def _aggregate_attempt13_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    source_input_sha256: str,
    source_plan_sha256: str,
    authorization_sha256: str,
    source_package_sha256: str,
    run_name: str,
) -> dict[str, Any]:
    normalized_rows = [
        _normalize_attempt13_phase_order(
            row, root=_engine.ROOT_INDEX_FIRST + offset
        )
        if isinstance(row, Mapping)
        else row
        for offset, row in enumerate(rows)
    ]
    return _ATTEMPT12_AGGREGATE_ROWS(
        normalized_rows,
        plan=plan,
        source_input_sha256=source_input_sha256,
        source_plan_sha256=source_plan_sha256,
        authorization_sha256=authorization_sha256,
        source_package_sha256=source_package_sha256,
        run_name=run_name,
    )


def _engine_bindings() -> dict[str, Any]:
    return {
        "AI_PROFILES_SHA256": AI_PROFILES_SHA256,
        "ATTEMPT09_LAMBDA_MODEL_SHA256": ATTEMPT13_LAMBDA_MODEL_SHA256,
        "M43_ATTEMPT09_PLAN_SHA256": M43_ATTEMPT13_PLAN_SHA256,
        "M43_ATTEMPT09_PROFILES": M43_ATTEMPT13_PROFILES,
        "enumerate_attempt09_seed_schedules": enumerate_attempt13_seed_schedules,
        "load_and_validate_attempt09_plan": load_and_validate_attempt13_plan,
        "validate_attempt09_plan": validate_attempt13_plan,
        "Attempt09TeacherConfig": Attempt13TeacherConfig,
        "validate_attempt09_teacher_output": validate_attempt13_teacher_output,
        "ATTEMPT09_BASELINE_PROFILE": ATTEMPT13_BASELINE_PROFILE,
        "ATTEMPT09_CONTINUATION_PROFILE": ATTEMPT13_CONTINUATION_PROFILE,
        "ATTEMPT09_ROW_SCHEMA": ATTEMPT13_ROW_SCHEMA,
        "ATTEMPT09_DEVELOPMENT_DECISION_SCHEMA": ATTEMPT13_DEVELOPMENT_DECISION_SCHEMA,
        "ATTEMPT09_DEVELOPMENT_RECEIPT_SCHEMA": ATTEMPT13_DEVELOPMENT_RECEIPT_SCHEMA,
        "ROOT_CONTRACT_SCHEMA": ATTEMPT13_ROOT_CONTRACT_SCHEMA,
        "TEACHER_VALIDATION_SCHEMA": ATTEMPT13_TEACHER_VALIDATION_SCHEMA,
        "NO_GO_STATUS": "no_go_close_attempt13_development",
        "EVALUATION_SAMPLE_COUNT": 512,
        "SELECTOR_SOURCE_PATH": Path(__file__),
        "_GATES": _ATTEMPT13_DEVELOPMENT_GATES,
        "_exact": _base._attempt12_exact,
        "_expected_search_config": _expected_search_config,
        "_expected_seed_domain_provenance": _expected_seed_domain_provenance,
        "_authorization": _base._attempt12_authorization,
    }


def _module_bindings() -> dict[str, Any]:
    return {
        "AI_PROFILES_SHA256": AI_PROFILES_SHA256,
        "ATTEMPT12_LAMBDA_MODEL_SHA256": ATTEMPT13_LAMBDA_MODEL_SHA256,
        "M43_ATTEMPT12_PLAN_SHA256": M43_ATTEMPT13_PLAN_SHA256,
        "M43_ATTEMPT12_PROFILES": M43_ATTEMPT13_PROFILES,
        "enumerate_attempt12_seed_schedules": enumerate_attempt13_seed_schedules,
        "load_and_validate_attempt12_plan": load_and_validate_attempt13_plan,
        "validate_attempt12_plan": validate_attempt13_plan,
        "Attempt12TeacherConfig": Attempt13TeacherConfig,
        "validate_attempt12_teacher_output": validate_attempt13_teacher_output,
        "ATTEMPT12_BASELINE_PROFILE": ATTEMPT13_BASELINE_PROFILE,
        "ATTEMPT12_CONTINUATION_PROFILE": ATTEMPT13_CONTINUATION_PROFILE,
        "ATTEMPT12_ROW_SCHEMA": ATTEMPT13_ROW_SCHEMA,
        "ATTEMPT12_AUTHORIZATION_SCHEMA": ATTEMPT13_AUTHORIZATION_SCHEMA,
        "ATTEMPT12_ROOT_CONTRACT_SCHEMA": ATTEMPT13_ROOT_CONTRACT_SCHEMA,
        "ATTEMPT12_DEVELOPMENT_DECISION_SCHEMA": ATTEMPT13_DEVELOPMENT_DECISION_SCHEMA,
        "ATTEMPT12_DEVELOPMENT_RECEIPT_SCHEMA": ATTEMPT13_DEVELOPMENT_RECEIPT_SCHEMA,
        "ATTEMPT12_TEACHER_VALIDATION_SCHEMA": ATTEMPT13_TEACHER_VALIDATION_SCHEMA,
        "_ATTEMPT12_GATE_RELATIVE": _ATTEMPT13_GATE_RELATIVE,
        "_CANDIDATE_NONBASELINE_COUNT": _CANDIDATE_NONBASELINE_COUNT,
        "_ASSESSMENT_SOURCE": _ASSESSMENT_SOURCE,
        "_INTEGRITY_NAMES": _INTEGRITY_NAMES,
        "_ATTEMPT12_DEVELOPMENT_GATES": _ATTEMPT13_DEVELOPMENT_GATES,
        "_expected_search_config": _expected_search_config,
        "_expected_seed_domain_provenance": _expected_seed_domain_provenance,
        "_BINDINGS": _engine_bindings(),
        "_runner_bindings": _runner_bindings,
        "_aggregate_attempt12_rows": _aggregate_attempt13_rows,
    }


@contextmanager
def _attempt13_selector_bindings() -> Iterator[None]:
    """Bind Attempt13 into the one-shot selector without leaking globals."""

    bindings = _module_bindings()
    with _LOCK:
        prior = {name: getattr(_base, name) for name in bindings}
        try:
            for name, value in bindings.items():
                setattr(_base, name, value)
            with _base._attempt12_selector_bindings():
                yield
        finally:
            for name, value in prior.items():
                setattr(_base, name, value)


def aggregate_attempt13_development_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    source_input_sha256: str,
    source_plan_sha256: str,
    authorization_sha256: str,
    source_package_sha256: str,
    run_name: str,
) -> dict[str, Any]:
    with _attempt13_selector_bindings():
        return _base._aggregate_attempt12_rows(
            rows,
            plan=plan,
            source_input_sha256=source_input_sha256,
            source_plan_sha256=source_plan_sha256,
            authorization_sha256=authorization_sha256,
            source_package_sha256=source_package_sha256,
            run_name=run_name,
        )


def select_attempt13_development(
    *,
    input_path: str | Path,
    plan_path: str | Path,
    authorization_path: str | Path,
    source_package_sha256: str,
    run_name: str,
) -> dict[str, Any]:
    with _attempt13_selector_bindings():
        return _engine.select_attempt09_development(
            input_path=input_path,
            plan_path=plan_path,
            authorization_path=authorization_path,
            source_package_sha256=source_package_sha256,
            run_name=run_name,
        )


def execute_attempt13_development_selector(
    *,
    input_path: str | Path,
    plan_path: str | Path,
    authorization_path: str | Path,
    source_package_sha256: str,
    run_name: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    with _attempt13_selector_bindings():
        return _engine.execute_attempt09_development_selector(
            input_path=input_path,
            plan_path=plan_path,
            authorization_path=authorization_path,
            source_package_sha256=source_package_sha256,
            run_name=run_name,
            output_dir=output_dir,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--source-package-sha256", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = execute_attempt13_development_selector(
        input_path=args.input,
        plan_path=args.plan,
        authorization_path=args.authorization,
        source_package_sha256=args.source_package_sha256,
        run_name=args.run_name,
        output_dir=args.output_dir,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTEMPT13_DEVELOPMENT_DECISION_SCHEMA",
    "ATTEMPT13_DEVELOPMENT_RECEIPT_SCHEMA",
    "aggregate_attempt13_development_rows",
    "execute_attempt13_development_selector",
    "select_attempt13_development",
]
