"""Apply the frozen Attempt12 Development200 gates exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from . import select_hu_m43_attempt09_development as _base
from .hu_m43_attempt12_contract import (
    AI_PROFILES_SHA256,
    ATTEMPT12_LAMBDA_MODEL_SHA256,
    M43_ATTEMPT12_PLAN_SHA256,
    M43_ATTEMPT12_PROFILES,
    enumerate_attempt12_seed_schedules,
    load_and_validate_attempt12_plan,
    validate_attempt12_plan,
)
from .hu_m43_attempt12_teacher import Attempt12TeacherConfig, validate_attempt12_teacher_output
from .run_hu_m43_attempt12 import (
    ATTEMPT12_AUTHORIZATION_SCHEMA,
    ATTEMPT12_BASELINE_PROFILE,
    ATTEMPT12_CONTINUATION_PROFILE,
    ATTEMPT12_ROOT_CONTRACT_SCHEMA,
    ATTEMPT12_ROW_SCHEMA,
    _attempt12_bindings as _runner_bindings,
)


ATTEMPT12_DEVELOPMENT_DECISION_SCHEMA = "hu_m43_attempt12_development_go_no_go_v1"
ATTEMPT12_DEVELOPMENT_RECEIPT_SCHEMA = (
    "hu_m43_attempt12_development_selector_receipt_v1"
)
ATTEMPT12_TEACHER_VALIDATION_SCHEMA = "hu_m43_attempt12_teacher_validation_v1"
_BIND_LOCK = threading.RLock()
_ATTEMPT12_GATE_RELATIVE = "artifacts/attempt12/preceding_gate.json"
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
)
_ATTEMPT12_DEVELOPMENT_GATES = {
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
    "nonfire_exact_baseline_action_fallback_required": True,
    "nonfire_complete_trajectory_acceptance_deferred": True,
    "teacher_values_reported_as_realized_match_ev": False,
    "raw_minimum_search_gate_allowed": False,
}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _attempt12_authorization(
    path: str | Path | None,
    *,
    mode: str,
    source_package_sha256: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate authorization without resolving its gate against process CWD."""

    if mode == "preflight":
        if path is not None:
            raise ValueError("Attempt12 preflight must not consume later authorization")
        return None, None
    if path is None:
        raise ValueError(f"Attempt12 {mode} requires explicit authorization")
    if mode not in {"development", "future_audit"}:
        raise ValueError(f"Attempt12 authorization mode is unsupported: {mode}")
    target = Path(path).resolve()
    raw = target.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Attempt12 execution authorization is invalid JSON") from exc
    canonical = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    if not isinstance(payload, dict) or raw != canonical:
        raise ValueError("Attempt12 execution authorization is not canonical JSON")
    expected_keys = {
        "schema",
        "status",
        "mode",
        "plan_sha256",
        "source_package_sha256",
        "preceding_gate_artifact",
        "preceding_gate_sha256",
        "root_index_first",
        "root_index_last",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
    expected_range = (0, 199) if mode == "development" else (200, 249)
    if (
        set(payload) != expected_keys
        or payload.get("schema")
        != ATTEMPT12_AUTHORIZATION_SCHEMA
        or payload.get("status") != "authorized"
        or payload.get("mode") != mode
        or payload.get("plan_sha256") != M43_ATTEMPT12_PLAN_SHA256
        or payload.get("root_index_first") != expected_range[0]
        or payload.get("root_index_last") != expected_range[1]
        or payload.get("current_profile_mutated") is not False
        or payload.get("runtime_policy_activated") is not False
    ):
        raise ValueError("Attempt12 authorization boundary changed")
    package_hash = payload.get("source_package_sha256")
    if not _is_sha256(package_hash) or source_package_sha256 != package_hash:
        raise ValueError("Attempt12 source package authorization changed")
    declared_gate = payload.get("preceding_gate_artifact")
    gate_hash = payload.get("preceding_gate_sha256")
    if declared_gate != _ATTEMPT12_GATE_RELATIVE or not _is_sha256(gate_hash):
        raise ValueError("Attempt12 preceding gate declaration changed")

    # On a VM, execution_authorization.json and artifacts/ are siblings.  From
    # a repository-root selector invocation, the authorization stays in the
    # run directory while the immutable gate is under run_dir/package_src/.
    # Resolve only those two package-owned locations; never process CWD.
    candidates = (
        target.parent / _ATTEMPT12_GATE_RELATIVE,
        target.parent / "package_src" / _ATTEMPT12_GATE_RELATIVE,
    )
    if not any(
        candidate.is_file()
        and not candidate.is_symlink()
        and hashlib.sha256(candidate.read_bytes()).hexdigest() == gate_hash
        for candidate in candidates
    ):
        raise ValueError("Attempt12 preceding gate artifact is missing or changed")
    return payload, hashlib.sha256(raw).hexdigest()


def _expected_search_config(
    config: Attempt12TeacherConfig, seeds: Mapping[str, int]
) -> dict[str, Any]:
    return {
        "all_legal_candidates": config.all_legal_candidates,
        # The strict Attempt12 teacher validator independently reconstructs the
        # legal action set and verifies this exact count.  The generic selector
        # callback does not receive the observation, so this precheck accepts
        # only the legal 0..26 range before that stronger validation runs.
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


def _attempt12_exact(actual: Any, expected: Any) -> bool:
    if expected is _CANDIDATE_NONBASELINE_COUNT:
        return type(actual) is int and 0 <= actual <= 26
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(
                _attempt12_exact(actual[key], value)
                for key, value in expected.items()
            )
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _attempt12_exact(left, right)
                for left, right in zip(actual, expected, strict=True)
            )
        )
    return type(actual) is type(expected) and actual == expected


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


_BINDINGS: dict[str, Any] = {
    "AI_PROFILES_SHA256": AI_PROFILES_SHA256,
    "ATTEMPT09_LAMBDA_MODEL_SHA256": ATTEMPT12_LAMBDA_MODEL_SHA256,
    "M43_ATTEMPT09_PLAN_SHA256": M43_ATTEMPT12_PLAN_SHA256,
    "M43_ATTEMPT09_PROFILES": M43_ATTEMPT12_PROFILES,
    "enumerate_attempt09_seed_schedules": enumerate_attempt12_seed_schedules,
    "load_and_validate_attempt09_plan": load_and_validate_attempt12_plan,
    "validate_attempt09_plan": validate_attempt12_plan,
    "Attempt09TeacherConfig": Attempt12TeacherConfig,
    "validate_attempt09_teacher_output": validate_attempt12_teacher_output,
    "ATTEMPT09_BASELINE_PROFILE": ATTEMPT12_BASELINE_PROFILE,
    "ATTEMPT09_CONTINUATION_PROFILE": ATTEMPT12_CONTINUATION_PROFILE,
    "ATTEMPT09_ROW_SCHEMA": ATTEMPT12_ROW_SCHEMA,
    "ATTEMPT09_DEVELOPMENT_DECISION_SCHEMA": ATTEMPT12_DEVELOPMENT_DECISION_SCHEMA,
    "ATTEMPT09_DEVELOPMENT_RECEIPT_SCHEMA": ATTEMPT12_DEVELOPMENT_RECEIPT_SCHEMA,
    "ROOT_CONTRACT_SCHEMA": ATTEMPT12_ROOT_CONTRACT_SCHEMA,
    "TEACHER_VALIDATION_SCHEMA": ATTEMPT12_TEACHER_VALIDATION_SCHEMA,
    "NO_GO_STATUS": "no_go_close_attempt12_development",
    "EVALUATION_SAMPLE_COUNT": 512,
    "SELECTOR_SOURCE_PATH": Path(__file__),
    "_GATES": _ATTEMPT12_DEVELOPMENT_GATES,
    "_exact": _attempt12_exact,
    "_expected_search_config": _expected_search_config,
    "_expected_seed_domain_provenance": _expected_seed_domain_provenance,
    "_authorization": _attempt12_authorization,
}


@contextmanager
def _attempt12_selector_bindings() -> Iterator[None]:
    with _BIND_LOCK, _runner_bindings():
        bindings = {
            **_BINDINGS,
            "aggregate_attempt09_development_rows": _aggregate_attempt12_rows,
        }
        prior = {name: getattr(_base, name) for name in bindings}
        try:
            for name, value in bindings.items():
                setattr(_base, name, value)
            yield
        finally:
            for name, value in prior.items():
                setattr(_base, name, value)


def _aggregate_attempt12_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    source_input_sha256: str,
    source_plan_sha256: str,
    authorization_sha256: str,
    source_package_sha256: str,
    run_name: str,
) -> dict[str, Any]:
    """Aggregate Development200 or Audit50 using only locked E512 labels."""

    validate_attempt12_plan(plan)
    declared = _base._mapping(
        plan.get(_base.GATE_SECTION), _base.GATE_SECTION
    )
    if any(
        declared.get(key) != value
        or type(declared.get(key)) is not type(value)
        for key, value in _base._GATES.items()
    ):
        raise ValueError("Attempt12 frozen search-quality gates changed")
    if (
        declared.get("assessment_source") != _ASSESSMENT_SOURCE
        or declared.get("quantile_method") != "numpy_linear"
        or declared.get("all_gates_required") is not True
    ):
        raise ValueError("Attempt12 E512 gate semantics changed")
    if (
        source_plan_sha256 != M43_ATTEMPT12_PLAN_SHA256
        or not all(
            _base._sha(value)
            for value in (
                source_input_sha256,
                authorization_sha256,
                source_package_sha256,
            )
        )
    ):
        raise ValueError("Attempt12 selector source hashes changed")
    if not _base._SAFE_RUN.fullmatch(run_name) or len(rows) != _base._ROOTS:
        raise ValueError("Attempt12 selector run identity or root count changed")

    schedules = enumerate_attempt12_seed_schedules(
        plan, population=_base.POPULATION
    )
    if tuple(schedules) != _base._DOMAINS:
        raise ValueError("Attempt12 seed domain order changed")
    validated: list[dict[str, Any]] = []
    for offset, raw_row in enumerate(rows):
        root = _base.ROOT_INDEX_FIRST + offset
        seeds = {name: int(schedules[name][offset]) for name in _base._DOMAINS}
        validated.append(
            _base._validate_row(
                _base._mapping(raw_row, f"row[{root}]"),
                root,
                seeds,
                run_name=run_name,
                authorization_sha256=authorization_sha256,
                source_package_sha256=source_package_sha256,
            )
        )

    profiles = _base.Counter(str(row["profile"]) for row in validated)
    expected_profiles = _base.Counter(
        {
            profile: _base.ROOTS_PER_PROFILE
            for profile in M43_ATTEMPT12_PROFILES
        }
    )
    if profiles != expected_profiles:
        raise ValueError(f"Attempt12 {_base.POPULATION} profile balance changed")
    if (
        len({row["hand_seed"] for row in validated}) != _base._ROOTS
        or len({row["observation_fingerprint"] for row in validated})
        != _base._ROOTS
    ):
        raise ValueError("Attempt12 selector roots repeat")
    all_rng = [digest for row in validated for digest in row["rng_digests"]]
    all_beliefs = [
        digest for row in validated for digest in row["belief_digests"]
    ]
    if len(all_rng) != len(set(all_rng)):
        raise ValueError("Attempt12 RNG domains overlap across roots")
    if len(all_beliefs) != len(set(all_beliefs)):
        raise ValueError("Attempt12 belief digests overlap across roots")

    overall = _base._metrics(validated)
    by_profile = {
        profile: _base._metrics(
            [row for row in validated if row["profile"] == profile]
        )
        for profile in M43_ATTEMPT12_PROFILES
    }
    fires_by_profile = {
        profile: by_profile[profile]["fires"]
        for profile in M43_ATTEMPT12_PROFILES
    }
    tails = overall["maximum_per_fired_root_loss"]
    gates = [
        _base._gate(
            "fires_total",
            overall["fires"] >= int(declared["fires_total_min"]),
            overall["fires"],
            f">= {declared['fires_total_min']}",
        ),
        _base._gate(
            "fires_each_profile",
            all(
                value >= int(declared["fires_each_profile_min"])
                for value in fires_by_profile.values()
            ),
            fires_by_profile,
            f"each >= {declared['fires_each_profile_min']}",
        ),
        _base._gate(
            "mean_delta_per_state",
            overall["mean_delta_per_state"] > 0,
            overall["mean_delta_per_state"],
            "> 0",
        ),
        _base._gate(
            "mean_delta_per_fire",
            overall["mean_delta_per_fire"] is not None
            and overall["mean_delta_per_fire"] > 0,
            overall["mean_delta_per_fire"],
            "> 0",
        ),
        _base._gate(
            "false_positive_rate_per_fire",
            overall["false_positive_rate_per_fire"] is not None
            and overall["false_positive_rate_per_fire"]
            <= float(declared["false_positive_rate_per_fire_max"]),
            overall["false_positive_rate_per_fire"],
            f"<= {declared['false_positive_rate_per_fire_max']}",
        ),
    ]
    for label, key, declared_key in (
        ("p95", "p95", "override_loss_p95_max"),
        ("p99", "p99", "override_loss_p99_max"),
        ("max", "max", "override_loss_max"),
    ):
        limit = float(declared[declared_key])
        observed = tails[key]
        gates.append(
            _base._gate(
                f"maximum_per_fired_root_{label}_loss",
                observed is not None and observed <= limit,
                observed,
                f"<= {limit:g}",
            )
        )
    for name in _INTEGRITY_NAMES:
        gates.append(_base._gate(f"{name}_violation_count", True, 0, "= 0"))
    gates.append(
        _base._gate(
            "nonfire_exact_baseline_action_fallback", True, True, "required"
        )
    )
    passed = all(gate["passed"] for gate in gates)
    identities = [
        {
            key: row[key]
            for key in (
                "root_index",
                "profile",
                "hand_seed",
                "observation_fingerprint",
                "baseline_action_key",
                "config_sha256",
            )
        }
        for row in validated
    ]
    integrity = {
        f"{name}_violation_count": 0 for name in _INTEGRITY_NAMES
    }
    integrity["nonfire_exact_baseline_action_fallback_verified"] = True
    return {
        "schema": _base.ATTEMPT09_DEVELOPMENT_DECISION_SCHEMA,
        "status": _base.GO_STATUS if passed else _base.NO_GO_STATUS,
        "decision": "go" if passed else "no_go",
        "search_freeze_authorized": passed,
        "selected_arm": None,
        "selected_threshold": None,
        "source": {
            "input_jsonl_sha256": source_input_sha256,
            "plan_sha256": source_plan_sha256,
            "authorization_sha256": authorization_sha256,
            "source_package_sha256": source_package_sha256,
            "run_name": run_name,
            "selector_source_sha256": hashlib.sha256(
                Path(_base.SELECTOR_SOURCE_PATH).read_bytes()
            ).hexdigest(),
            "root_identity_sha256": _base._canonical_sha256(
                {"roots": identities}
            ),
        },
        _base.POPULATION_REPORT_KEY: {
            "roots": _base._ROOTS,
            "profile_counts": dict(profiles),
        },
        "metrics": {
            "overall": overall,
            "by_profile": by_profile,
            "fired_root_diagnostics": [
                {
                    "root_index": row["root_index"],
                    "profile": row["profile"],
                    "selected_action_key": row["selected_action_key"],
                    "e512_paired_delta_mean": row["mean"],
                    "e512_per_root_loss": row["loss"],
                    "false_positive": row["mean"] <= 0,
                }
                for row in validated
                if row["fired"]
            ],
        },
        "gates": gates,
        "decision_contract": {
            "single_frozen_search_architecture": True,
            "arm_selection_performed": False,
            "threshold_selection_performed": False,
            "gate_evaluation_count": 1,
            "all_gates_required": True,
        },
        "integrity": integrity,
        "science_boundary": {
            "assessment_source": _ASSESSMENT_SOURCE,
            "teacher_values_are_realized_match_ev": False,
            "future_audit_authorized": False,
            "fit_performed": False,
            "threshold_selected": False,
            "runtime_policy_activated": False,
            "current_profile_mutated": False,
            "full_replacement_enabled": False,
        },
    }


def aggregate_attempt12_development_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    source_input_sha256: str,
    source_plan_sha256: str,
    authorization_sha256: str,
    source_package_sha256: str,
    run_name: str,
) -> dict[str, Any]:
    with _attempt12_selector_bindings():
        return _aggregate_attempt12_rows(
            rows,
            plan=plan,
            source_input_sha256=source_input_sha256,
            source_plan_sha256=source_plan_sha256,
            authorization_sha256=authorization_sha256,
            source_package_sha256=source_package_sha256,
            run_name=run_name,
        )


def select_attempt12_development(
    *,
    input_path: str | Path,
    plan_path: str | Path,
    authorization_path: str | Path,
    source_package_sha256: str,
    run_name: str,
) -> dict[str, Any]:
    with _attempt12_selector_bindings():
        return _base.select_attempt09_development(
            input_path=input_path,
            plan_path=plan_path,
            authorization_path=authorization_path,
            source_package_sha256=source_package_sha256,
            run_name=run_name,
        )


def execute_attempt12_development_selector(
    *,
    input_path: str | Path,
    plan_path: str | Path,
    authorization_path: str | Path,
    source_package_sha256: str,
    run_name: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    with _attempt12_selector_bindings():
        return _base.execute_attempt09_development_selector(
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
    receipt = execute_attempt12_development_selector(
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
    "ATTEMPT12_DEVELOPMENT_DECISION_SCHEMA",
    "ATTEMPT12_DEVELOPMENT_RECEIPT_SCHEMA",
    "aggregate_attempt12_development_rows",
    "execute_attempt12_development_selector",
    "select_attempt12_development",
]
