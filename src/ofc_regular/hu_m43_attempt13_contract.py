"""Frozen pre-preflight contract for the M4.3 Attempt13 search teacher.

Attempt12 remains an immutable No-Go.  Its Development200 rows are consumed
architecture/training diagnostics only; they are never Attempt13 evaluation
rows.  Attempt13 owns wholly new preflight, development, audit, and reserved
population seed namespaces and authorizes none of them merely by this plan.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import hu_m43_attempt12_contract as _attempt12


M43_ATTEMPT13_PLAN_SCHEMA = "hu_m43_attempt13_plan_v1"
M43_ATTEMPT13_PLAN_SHA256 = (
    "9b860be1de05570840e0fd7b1e4a3e3c2f52e5785be4a882d530b5f9baf5c6c2"
)
M43_ATTEMPT13_CLOSEOUT_SCHEMA = "hu_m43_attempt12_closeout_v1"
M43_ATTEMPT13_CLOSEOUT_SHA256 = (
    "f8b46789698e799233d0902e9925d11d62caa054084a3c255354fa0b44f167a3"
)
M43_ATTEMPT13_SEED_STRIDE = 1_000_003
M43_ATTEMPT13_PROFILES = (
    "stage19_p0",
    "stage9f_p2",
    "stage7_m5_r10",
    "stage3_baseline",
    "random_exact_final",
)

ATTEMPT13_CANDIDATE_MAX = 26
ATTEMPT13_RERANK_SAMPLES = 128
ATTEMPT13_SHORTLIST_MAX = 8
ATTEMPT13_VETO_SAMPLES = 256
ATTEMPT13_STRESS_SAMPLES = 1024
ATTEMPT13_CONFIRMATION_SAMPLES = 1024
ATTEMPT13_POOLED_SAMPLES = 2048
ATTEMPT13_EVALUATION_SAMPLES = 512
ATTEMPT13_VETO_MIN_MEAN = 0.0
ATTEMPT13_VETO_NORMALIZED_RISK_MAX = 1.05
ATTEMPT13_POOLED_MIN_MEAN = 0.0
ATTEMPT13_POOLED_NORMALIZED_RISK_MAX = 1.0
ATTEMPT13_POOLED_Q001_MIN = -50.0
ATTEMPT13_POOLED_ES01_MIN = -40.0

ATTEMPT12_PLAN_SHA256 = _attempt12.M43_ATTEMPT12_PLAN_SHA256
ATTEMPT12_DECISION_SHA256 = (
    "c829cf35d53eb24bc9201564f789ee4598c8cb91b3ac5311a41643af14549ac5"
)
ATTEMPT12_DECISION_RECEIPT_SHA256 = (
    "abfec493b3b57e2a756bbe3e39f663ba90b80d60a81fc99975308faee6611eee"
)
ATTEMPT12_MERGED_TEACHER_SHA256 = (
    "504dfad760f1d5bf139fa94a16f92e66330ba8113b4ce8db77df398eb8c8c305"
)
ATTEMPT12_RECEIVE_RECEIPT_SHA256 = (
    "4f630882f0e4867ee24afe4f16203696607996c243e349a438e94c287ddfc819"
)
ATTEMPT13_LAMBDA_MODEL_SHA256 = _attempt12.ATTEMPT12_LAMBDA_MODEL_SHA256
AI_PROFILES_SHA256 = _attempt12.AI_PROFILES_SHA256

ATTEMPT13_SEED_BASES = {
    "hand": 230_108_071_901,
    "rerank": 231_108_071_901,
    "veto": 232_108_071_901,
    "stress": 233_108_071_901,
    "confirmation": 234_108_071_901,
    "evaluation": 235_108_071_901,
    "child": 236_108_071_901,
}
ATTEMPT13_PREFLIGHT_SEED_BASES = {
    "hand": 240_108_071_901,
    "rerank": 241_108_071_901,
    "veto": 242_108_071_901,
    "stress": 243_108_071_901,
    "confirmation": 244_108_071_901,
    "evaluation": 245_108_071_901,
    "child": 246_108_071_901,
}
ATTEMPT13_POPULATION_SEED_BASES = {
    "hand": 250_108_071_901,
    "rerank": 251_108_071_901,
    "veto": 252_108_071_901,
    "stress": 253_108_071_901,
    "confirmation": 254_108_071_901,
    "evaluation": 255_108_071_901,
    "child": 256_108_071_901,
}

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL_PLAN_PATH = (
    _REPOSITORY_ROOT / "configs" / "hu_joint_policy_m43_attempt13.json"
)
_CANONICAL_CLOSEOUT_PATH = (
    _REPOSITORY_ROOT / "configs" / "hu_joint_policy_m43_attempt12_closeout.json"
)
_ATTEMPT12_POPULATION_SEED = 220_108_071_901


def load_and_validate_attempt13_plan(path: str | Path) -> dict[str, Any]:
    """Load the byte-bound Attempt13 plan and fail closed on any drift."""

    target = Path(path)
    if _file_sha256(target) != M43_ATTEMPT13_PLAN_SHA256:
        raise ValueError("Attempt13 plan SHA-256 changed")
    plan = _load_json_mapping(target, "Attempt13 plan")
    validate_attempt13_plan(plan)
    return plan


def load_and_validate_attempt12_closeout(path: str | Path) -> dict[str, Any]:
    """Load the immutable Attempt12 No-Go/reclassification receipt."""

    target = Path(path)
    if _file_sha256(target) != M43_ATTEMPT13_CLOSEOUT_SHA256:
        raise ValueError("Attempt12 closeout SHA-256 changed")
    closeout = _load_json_mapping(target, "Attempt12 closeout")
    validate_attempt12_closeout(closeout)
    return closeout


def validate_attempt13_plan(plan: Mapping[str, Any]) -> None:
    """Validate the canonical mapping and independently derived invariants."""

    _require_exact(plan, _load_canonical_plan(), "plan")
    _validate_frozen_semantics(plan)
    _validate_profile_balance(plan)
    _validate_search_contract(plan)
    _validate_go_no_go(plan)
    _validate_compute_accounting(plan)
    validate_attempt13_seed_freshness(plan)


def validate_attempt12_closeout(closeout: Mapping[str, Any]) -> None:
    """Prove Attempt12 is closed and only diagnostic/training reuse is allowed."""

    _require_exact(
        closeout.get("schema"), M43_ATTEMPT13_CLOSEOUT_SCHEMA, "closeout.schema"
    )
    _require_exact(
        closeout.get("status"),
        "consumed_architecture_diagnostic_only",
        "closeout.status",
    )
    plan = _mapping(closeout.get("attempt12_plan"), "attempt12_plan")
    _require_exact(plan.get("sha256"), ATTEMPT12_PLAN_SHA256, "attempt12 plan SHA")
    artifacts = _mapping(closeout.get("immutable_artifacts"), "immutable_artifacts")
    expected_hashes = {
        "decision": ATTEMPT12_DECISION_SHA256,
        "decision_receipt": ATTEMPT12_DECISION_RECEIPT_SHA256,
        "merged_teacher": ATTEMPT12_MERGED_TEACHER_SHA256,
        "receive_receipt": ATTEMPT12_RECEIVE_RECEIPT_SHA256,
    }
    for name, expected in expected_hashes.items():
        item = _mapping(artifacts.get(name), f"immutable_artifacts.{name}")
        _require_exact(item.get("sha256"), expected, f"{name} SHA")
    result = _mapping(closeout.get("frozen_result"), "frozen_result")
    _require_exact(result.get("decision"), "no_go", "frozen decision")
    _require_exact(result.get("fires_total"), 35, "frozen fires")
    _require_exact(
        result.get("failed_gates"),
        [
            "fires_total",
            "fires_each_profile",
            "maximum_per_fired_root_max_loss",
        ],
        "failed gates",
    )
    science = _mapping(closeout.get("science_boundary"), "science_boundary")
    for name in (
        "fresh_claim",
        "fresh_generalization_evidence",
        "teacher_values_are_realized_match_ev",
        "attempt12_future_audit_authorized",
        "attempt12_fit_authorized",
        "attempt12_runtime_authorized",
        "attempt12_population_authorized",
        "attempt12_current_profile_mutated",
        "attempt12_full_replacement_enabled",
        "attempt12_seed_reuse_allowed",
    ):
        if science.get(name) is not False:
            raise ValueError(f"Attempt12 closeout {name} must remain false")
    allowed = science.get("allowed_future_use")
    if not isinstance(allowed, list) or "attempt13_training_only" not in allowed:
        raise ValueError("Attempt12 closeout training-only reuse changed")
    forbidden = science.get("forbidden_future_use")
    if not isinstance(forbidden, list) or "attempt13_development_or_audit_evaluation" not in forbidden:
        raise ValueError("Attempt12 closeout evaluation prohibition changed")


def validate_attempt13_artifact_bindings(
    plan: Mapping[str, Any], *, repository_root: str | Path
) -> None:
    """Validate every frozen source artifact without mutating Attempt12."""

    validate_attempt13_plan(plan)
    root = Path(repository_root).resolve()
    boundary = _mapping(plan.get("attempt12_boundary"), "attempt12_boundary")
    search = _mapping(plan.get("search_protocol"), "search_protocol")
    baseline = _mapping(plan.get("baseline_hash_audit"), "baseline_hash_audit")
    bindings = (
        (boundary["closeout_path"], boundary["closeout_sha256"], "Attempt12 closeout"),
        (
            "configs/hu_joint_policy_m43_attempt12.json",
            ATTEMPT12_PLAN_SHA256,
            "Attempt12 plan",
        ),
        (
            search["candidate_generator_artifact_path"],
            search["candidate_generator_artifact_sha256"],
            "Lambda model",
        ),
        (
            baseline["policy_registry_path"],
            baseline["policy_registry_expected_sha256"],
            "AI profile registry",
        ),
    )
    for relative, expected, label in bindings:
        target = _resolve_inside(root, str(relative), label)
        if _file_sha256(target) != expected:
            raise ValueError(f"{label} SHA-256 changed")
    closeout = load_and_validate_attempt12_closeout(root / boundary["closeout_path"])
    artifacts = _mapping(closeout.get("immutable_artifacts"), "immutable_artifacts")
    for name, item_value in artifacts.items():
        item = _mapping(item_value, f"immutable_artifacts.{name}")
        target = _resolve_inside(root, str(item.get("path")), f"Attempt12 {name}")
        if _file_sha256(target) != item.get("sha256"):
            raise ValueError(f"Attempt12 {name} SHA-256 changed")


def enumerate_attempt13_seed_schedules(
    plan: Mapping[str, Any], *, population: str
) -> dict[str, tuple[int, ...]]:
    """Derive seven schedules for one frozen Attempt13 population."""

    if population == "preflight":
        contract = _mapping(plan.get("preflight_seed_contract"), population)
        indices = contract.get("source_root_indices")
        if not isinstance(indices, list) or any(type(item) is not int for item in indices):
            raise ValueError("Attempt13 preflight indices changed")
    elif population == "population_reserved":
        contract = _mapping(plan.get("population_seed_reservation"), population)
        count = _strict_int(contract.get("paired_seed_count"), "paired_seed_count")
        _require_exact(count, 1000, "population seed count")
        indices = list(range(count))
    elif population in {"development", "future_audit"}:
        contract = _mapping(plan.get("seed_contract"), "seed_contract")
        section = _mapping(plan.get(f"{population}_population"), population)
        first = _strict_int(section.get("root_index_first"), f"{population}.first")
        last = _strict_int(section.get("root_index_last"), f"{population}.last")
        roots = _strict_int(section.get("roots"), f"{population}.roots")
        if last - first + 1 != roots:
            raise ValueError(f"Attempt13 {population} root range changed")
        indices = list(range(first, last + 1))
    else:
        raise ValueError(
            "population must be development, future_audit, preflight, or population_reserved"
        )
    stride = _strict_int(contract.get("seed_stride"), f"{population}.seed_stride")
    return _enumerate_from_contract(contract, stride=stride, indices=indices)


def enumerate_attempt13_prior_seed_schedules() -> dict[str, tuple[int, ...]]:
    """Materialize every prior namespace relevant to Attempt13 freshness."""

    schedules = {
        f"prior.{name}": values
        for name, values in _attempt12.enumerate_known_seed_schedules().items()
    }
    attempt12_plan = _attempt12.load_and_validate_attempt12_plan(
        _REPOSITORY_ROOT / "configs" / "hu_joint_policy_m43_attempt12.json"
    )
    for population in ("development", "future_audit", "preflight"):
        for domain, values in _attempt12.enumerate_attempt12_seed_schedules(
            attempt12_plan, population=population
        ).items():
            schedules[f"Attempt12.{population}.{domain}"] = values
    schedules["Attempt12.population_reserved"] = tuple(
        _ATTEMPT12_POPULATION_SEED + M43_ATTEMPT13_SEED_STRIDE * index
        for index in range(1000)
    )
    return schedules


def validate_attempt13_seed_freshness(plan: Mapping[str, Any]) -> None:
    """Prove all 28 new schedules are unique, disjoint, and prior-fresh."""

    expected_counts = {
        "development": 200,
        "future_audit": 50,
        "preflight": 5,
        "population_reserved": 1000,
    }
    declared: dict[str, tuple[int, ...]] = {}
    for population, expected in expected_counts.items():
        schedules = enumerate_attempt13_seed_schedules(plan, population=population)
        if set(schedules) != set(ATTEMPT13_SEED_BASES):
            raise ValueError(f"Attempt13 {population} seed domains changed")
        for domain, values in schedules.items():
            if len(values) != expected or len(set(values)) != expected:
                raise ValueError(f"Attempt13 {population}.{domain} is not unique")
            declared[f"{population}.{domain}"] = values
    _reject_pairwise_overlap(declared, label="Attempt13 declared seed schedules")
    prior = enumerate_attempt13_prior_seed_schedules()
    for new_name, new_values in declared.items():
        new_set = set(new_values)
        for old_name, old_values in prior.items():
            if new_set.intersection(old_values):
                raise ValueError(f"Attempt13 {new_name} overlaps {old_name}")


def _validate_frozen_semantics(plan: Mapping[str, Any]) -> None:
    _require_exact(plan.get("schema"), M43_ATTEMPT13_PLAN_SCHEMA, "plan.schema")
    _require_exact(plan.get("milestone"), "M4.3-attempt13", "plan.milestone")
    _require_exact(plan.get("status"), "frozen_pre_preflight", "plan.status")
    _require_exact(plan.get("profiles"), list(M43_ATTEMPT13_PROFILES), "profiles")
    scope = _mapping(plan.get("scope"), "scope")
    if scope.get("public_information_set_only") is not True:
        raise ValueError("Attempt13 must use the public information set only")
    for name in (
        "opponent_private_discard_input_allowed",
        "opponent_profile_runtime_feature_allowed",
        "teacher_values_are_realized_match_ev",
        "teacher_ev_or_lcb_runtime_gate_allowed",
        "current_profile_mutated",
        "runtime_policy_activated",
        "full_replacement_enabled",
    ):
        if scope.get(name) is not False:
            raise ValueError(f"Attempt13 scope.{name} must remain false")
    boundary = _mapping(plan.get("attempt12_boundary"), "attempt12_boundary")
    expected_boundary = {
        "closeout_sha256": M43_ATTEMPT13_CLOSEOUT_SHA256,
        "source_plan_sha256": ATTEMPT12_PLAN_SHA256,
        "decision_sha256": ATTEMPT12_DECISION_SHA256,
        "decision_receipt_sha256": ATTEMPT12_DECISION_RECEIPT_SHA256,
        "merged_teacher_sha256": ATTEMPT12_MERGED_TEACHER_SHA256,
        "receive_receipt_sha256": ATTEMPT12_RECEIVE_RECEIPT_SHA256,
        "closeout_status": "consumed_architecture_diagnostic_only",
        "decision": "no_go",
        "fresh_claim": False,
        "attempt12_future_audit_authorized": False,
        "attempt12_fit_authorized": False,
        "attempt12_runtime_authorized": False,
        "attempt12_threshold_reselection_allowed": False,
    }
    for name, expected in expected_boundary.items():
        _require_exact(boundary.get(name), expected, f"attempt12_boundary.{name}")
    guards = _mapping(plan.get("activation_guards"), "activation_guards")
    if not guards or any(value is not False for value in guards.values()):
        raise ValueError("all Attempt13 activation guards must remain false")
    population = _mapping(plan.get("population_seed_reservation"), "population")
    if population.get("authorized") is not False or population.get("content_opened") is not False:
        raise ValueError("Attempt13 population reservation must remain closed")


def _validate_search_contract(plan: Mapping[str, Any]) -> None:
    search = _mapping(plan.get("search_protocol"), "search_protocol")
    _require_exact(
        search.get("candidate_generator_artifact_sha256"),
        ATTEMPT13_LAMBDA_MODEL_SHA256,
        "candidate artifact SHA",
    )
    _require_exact(search.get("learned_nonbaseline_max"), 26, "candidate max")
    rerank = _mapping(search.get("rerank"), "rerank")
    _require_exact(rerank.get("symbol"), "R128", "rerank.symbol")
    _require_exact(rerank.get("samples"), 128, "rerank.samples")
    shortlist = _mapping(search.get("shortlist"), "shortlist")
    _require_exact(shortlist.get("symbol"), "K=min(8,n)", "shortlist.symbol")
    _require_exact(shortlist.get("nonbaseline_actions_max"), 8, "shortlist.max")
    veto = _mapping(search.get("veto"), "veto")
    _require_exact(veto.get("symbol"), "V256", "veto.symbol")
    _require_exact(veto.get("samples"), 256, "veto.samples")
    veto_gate = _mapping(veto.get("eligibility"), "veto.eligibility")
    _require_exact(
        veto_gate.get("paired_delta_mean_strictly_greater_than"), 0, "V mean"
    )
    _require_exact(
        veto_gate.get("normalized_p95_p99_tail_risk_max"), 1.05, "V risk"
    )
    _require_exact(veto_gate.get("equivalent_p05_min"), -26.25, "V p05")
    _require_exact(veto_gate.get("equivalent_p01_min"), -42, "V p01")
    _require_exact(
        veto.get("raw_minimum_use"),
        "diagnostic_only_never_filter_select_or_gate",
        "V raw minimum",
    )
    stress = _mapping(search.get("stress"), "stress")
    confirmation = _mapping(search.get("confirmation"), "confirmation")
    for phase, symbol in ((stress, "X1024"), (confirmation, "C1024")):
        _require_exact(phase.get("symbol"), symbol, f"{symbol}.symbol")
        _require_exact(phase.get("samples"), 1024, f"{symbol}.samples")
        if phase.get("filtering_allowed") is not False:
            raise ValueError(f"{symbol} may not filter")
    pooled = _mapping(search.get("pooled_decision"), "pooled_decision")
    _require_exact(pooled.get("symbol"), "P2048", "pooled.symbol")
    _require_exact(pooled.get("samples_per_action"), 2048, "pooled.samples")
    eligibility = _mapping(pooled.get("eligibility"), "pooled.eligibility")
    expected = {
        "paired_delta_mean_strictly_greater_than": 0,
        "normalized_p95_p99_tail_risk_max": 1,
        "q001_min": -50,
        "es01_definition": "mean_lowest_ceil_0.01_times_2048_samples_equal_21",
        "es01_min": -40,
    }
    for name, value in expected.items():
        _require_exact(eligibility.get(name), value, f"pooled.{name}")
    _require_exact(
        pooled.get("raw_minimum_use"),
        "diagnostic_only_never_filter_select_or_gate",
        "pooled raw minimum",
    )
    evaluation = _mapping(search.get("evaluation"), "evaluation")
    _require_exact(evaluation.get("symbol"), "E512", "evaluation.symbol")
    _require_exact(evaluation.get("samples"), 512, "evaluation.samples")
    if (
        evaluation.get("diagnostics_only") is not True
        or evaluation.get("decision_frozen_before_namespace_open") is not True
        or evaluation.get("may_rerank_veto_confirm_promote_or_change_root_output")
        is not False
    ):
        raise ValueError("Attempt13 E512 decision isolation changed")
    _require_exact(
        search.get("seed_domains"),
        ["hand", "rerank", "veto", "stress", "confirmation", "evaluation", "child"],
        "seed domains",
    )
    if (
        search.get("rng_domains_pairwise_disjoint") is not True
        or search.get("candidate_selection_and_evaluation_rng_independent") is not True
        or search.get("hidden_information_input_allowed") is not False
    ):
        raise ValueError("Attempt13 RNG or information boundary changed")


def _validate_go_no_go(plan: Mapping[str, Any]) -> None:
    development = _mapping(plan.get("development_go_no_go"), "development gates")
    expected = {
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
        "extreme_tail_statistic_violation_count_max": 0,
        "locked_action_change_violation_count_max": 0,
    }
    for name, value in expected.items():
        _require_exact(development.get(name), value, f"development_go_no_go.{name}")
    if (
        development.get("all_gates_required") is not True
        or development.get("nonfire_exact_baseline_action_fallback_required")
        is not True
        or development.get("teacher_values_reported_as_realized_match_ev") is not False
        or development.get("raw_minimum_search_gate_allowed") is not False
    ):
        raise ValueError("Attempt13 development gate boundary changed")
    audit = _mapping(plan.get("future_audit_go_no_go"), "audit gates")
    _require_exact(audit.get("fires_total_min"), 10, "audit fires")
    _require_exact(audit.get("fires_each_profile_min"), 1, "audit profile fires")


def _validate_profile_balance(plan: Mapping[str, Any]) -> None:
    profiles = tuple(plan.get("profiles", ()))
    for section, expected in (
        ("development_population", 40),
        ("future_audit_population", 10),
    ):
        spec = _mapping(plan.get(section), section)
        first = _strict_int(spec.get("root_index_first"), f"{section}.first")
        last = _strict_int(spec.get("root_index_last"), f"{section}.last")
        assigned = [profiles[index % len(profiles)] for index in range(first, last + 1)]
        if any(assigned.count(profile) != expected for profile in profiles):
            raise ValueError(f"Attempt13 {section} is not profile-balanced")


def _validate_compute_accounting(plan: Mapping[str, Any]) -> None:
    cost = _mapping(plan.get("single_arm_cost"), "single_arm_cost")
    expected_max = (
        (ATTEMPT13_CANDIDATE_MAX + 1) * ATTEMPT13_RERANK_SAMPLES
        + (ATTEMPT13_SHORTLIST_MAX + 1) * ATTEMPT13_VETO_SAMPLES
        + 2
        * (ATTEMPT13_SHORTLIST_MAX + 1)
        * ATTEMPT13_STRESS_SAMPLES
        + 2 * ATTEMPT13_EVALUATION_SAMPLES
    )
    _require_exact(
        cost.get("maximum_full_fire_action_futures_per_root"),
        expected_max,
        "maximum compute",
    )


def _load_canonical_plan() -> dict[str, Any]:
    if _file_sha256(_CANONICAL_PLAN_PATH) != M43_ATTEMPT13_PLAN_SHA256:
        raise ValueError("canonical Attempt13 plan SHA-256 changed")
    return _load_json_mapping(_CANONICAL_PLAN_PATH, "canonical Attempt13 plan")


def _enumerate_from_contract(
    contract: Mapping[str, Any], *, stride: int, indices: Sequence[int]
) -> dict[str, tuple[int, ...]]:
    key_by_domain = {
        "hand": "hand_seed_base",
        "rerank": "rerank_seed_base",
        "veto": "veto_seed_base",
        "stress": "stress_seed_base",
        "confirmation": "confirmation_seed_base",
        "evaluation": "evaluation_seed_base",
        "child": "child_policy_seed_base",
    }
    return {
        domain: tuple(
            _strict_int(contract.get(key), key) + stride * index for index in indices
        )
        for domain, key in key_by_domain.items()
    }


def _reject_pairwise_overlap(
    schedules: Mapping[str, Sequence[int]], *, label: str
) -> None:
    names = tuple(schedules)
    for index, left_name in enumerate(names):
        left = set(schedules[left_name])
        for right_name in names[index + 1 :]:
            if left.intersection(schedules[right_name]):
                raise ValueError(f"{label} overlap: {left_name} vs {right_name}")


def _resolve_inside(root: Path, relative: str, label: str) -> Path:
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes repository root") from exc
    if not target.is_file():
        raise ValueError(f"{label} is missing: {relative}")
    return target


def _load_json_mapping(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"artifact is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_exact(actual: Any, expected: Any, path: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ValueError(f"{path} changed")
        for key, value in expected.items():
            _require_exact(actual[key], value, f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{path} changed")
        for index, (actual_value, expected_value) in enumerate(zip(actual, expected)):
            _require_exact(actual_value, expected_value, f"{path}[{index}]")
        return
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{path} changed")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _strict_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


__all__ = [
    "AI_PROFILES_SHA256",
    "ATTEMPT12_DECISION_RECEIPT_SHA256",
    "ATTEMPT12_DECISION_SHA256",
    "ATTEMPT12_MERGED_TEACHER_SHA256",
    "ATTEMPT12_PLAN_SHA256",
    "ATTEMPT12_RECEIVE_RECEIPT_SHA256",
    "ATTEMPT13_CANDIDATE_MAX",
    "ATTEMPT13_CONFIRMATION_SAMPLES",
    "ATTEMPT13_EVALUATION_SAMPLES",
    "ATTEMPT13_LAMBDA_MODEL_SHA256",
    "ATTEMPT13_POOLED_ES01_MIN",
    "ATTEMPT13_POOLED_MIN_MEAN",
    "ATTEMPT13_POOLED_NORMALIZED_RISK_MAX",
    "ATTEMPT13_POOLED_Q001_MIN",
    "ATTEMPT13_POOLED_SAMPLES",
    "ATTEMPT13_POPULATION_SEED_BASES",
    "ATTEMPT13_PREFLIGHT_SEED_BASES",
    "ATTEMPT13_RERANK_SAMPLES",
    "ATTEMPT13_SEED_BASES",
    "ATTEMPT13_SHORTLIST_MAX",
    "ATTEMPT13_STRESS_SAMPLES",
    "ATTEMPT13_VETO_MIN_MEAN",
    "ATTEMPT13_VETO_NORMALIZED_RISK_MAX",
    "ATTEMPT13_VETO_SAMPLES",
    "M43_ATTEMPT13_CLOSEOUT_SCHEMA",
    "M43_ATTEMPT13_CLOSEOUT_SHA256",
    "M43_ATTEMPT13_PLAN_SCHEMA",
    "M43_ATTEMPT13_PLAN_SHA256",
    "M43_ATTEMPT13_PROFILES",
    "M43_ATTEMPT13_SEED_STRIDE",
    "enumerate_attempt13_prior_seed_schedules",
    "enumerate_attempt13_seed_schedules",
    "load_and_validate_attempt12_closeout",
    "load_and_validate_attempt13_plan",
    "validate_attempt12_closeout",
    "validate_attempt13_artifact_bindings",
    "validate_attempt13_plan",
    "validate_attempt13_seed_freshness",
]
