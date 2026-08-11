"""Frozen pre-preflight contract for the M4.3 Attempt11 search arm.

Attempt11 is a development-only, profile-blind T1-second search contract.  It
does not authorize preflight or development generation, the reserved audit,
fitting, runtime use, or any change to ``current``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


M43_ATTEMPT11_PLAN_SCHEMA = "hu_m43_attempt11_plan_v1"
M43_ATTEMPT11_PLAN_SHA256 = (
    "1896f311eec31a48eb19de13ffebdbdabda9629015f48e25291b8dcb2f96c26c"
)
M43_ATTEMPT11_SEED_STRIDE = 1_000_003
M43_ATTEMPT11_PROFILES = (
    "stage19_p0",
    "stage9f_p2",
    "stage7_m5_r10",
    "stage3_baseline",
    "random_exact_final",
)

ATTEMPT11_CANDIDATE_MAX = 12
ATTEMPT11_RERANK_SAMPLES = 128
ATTEMPT11_SHORTLIST_MAX = 8
ATTEMPT11_VETO_SAMPLES = 256
ATTEMPT11_STRESS_SAMPLES = 1024
ATTEMPT11_CONFIRMATION_SAMPLES = 512
ATTEMPT11_EVALUATION_SAMPLES = 256
ATTEMPT11_VETO_MIN_MEAN = 0.0
ATTEMPT11_VETO_MIN_P05 = -25.0
ATTEMPT11_VETO_MIN_P01 = -40.0
ATTEMPT11_VETO_MIN_VALUE = -50.0
ATTEMPT11_STRICT_MIN_P05 = -22.0
ATTEMPT11_STRICT_MIN_P01 = -36.0
ATTEMPT11_STRICT_MIN_VALUE = -45.0

ATTEMPT10_PLAN_SHA256 = (
    "206e62a31bc7ec1ad3a2d1ca3c303e82d0957e4b010adbeee43ddff7b492986e"
)
ATTEMPT11_LAMBDA_MODEL_SHA256 = (
    "e7fa728308c8973e2e202baf79309e30b9b6f58c37431d8ea55850390abc28c3"
)
AI_PROFILES_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL_PLAN_PATH = (
    _REPOSITORY_ROOT / "configs" / "hu_joint_policy_m43_attempt11.json"
)

_SEED_BASES = {
    "hand": 140_108_071_901,
    "rerank": 141_108_071_901,
    "veto": 142_108_071_901,
    "stress": 143_108_071_901,
    "confirmation": 144_108_071_901,
    "evaluation": 145_108_071_901,
    "child": 146_108_071_901,
}
_PREFLIGHT_SEED_BASES = {
    "hand": 150_108_071_901,
    "rerank": 151_108_071_901,
    "veto": 152_108_071_901,
    "stress": 153_108_071_901,
    "confirmation": 154_108_071_901,
    "evaluation": 155_108_071_901,
    "child": 156_108_071_901,
}
_KNOWN_SEED_GROUPS: tuple[tuple[str, Mapping[str, int], range], ...] = (
    (
        "Attempt06",
        {
            "hand": 17_306_071_901,
            "candidate": 23_306_071_901,
            "evaluation": 24_306_071_901,
            "child": 25_306_071_901,
        },
        range(50),
    ),
    (
        "Attempt07",
        {
            "hand": 60_106_071_901,
            "screen": 61_106_071_901,
            "rerank": 62_106_071_901,
            "veto": 63_106_071_901,
            "assessment": 64_106_071_901,
            "child": 65_106_071_901,
        },
        range(150),
    ),
    (
        "Attempt07 preflight",
        {
            "screen": 71_106_071_901,
            "rerank": 72_106_071_901,
            "veto": 73_106_071_901,
            "assessment": 74_106_071_901,
            "child": 75_106_071_901,
        },
        range(3),
    ),
    (
        "Attempt08",
        {
            "hand": 80_108_071_901,
            "rerank": 81_108_071_901,
            "veto": 82_108_071_901,
            "stress": 83_108_071_901,
            "assessment": 84_108_071_901,
            "child": 85_108_071_901,
        },
        range(250),
    ),
    (
        "Attempt08 preflight",
        {
            "hand": 90_108_071_901,
            "rerank": 91_108_071_901,
            "veto": 92_108_071_901,
            "stress": 93_108_071_901,
            "assessment": 94_108_071_901,
            "child": 95_108_071_901,
        },
        range(3),
    ),
    (
        "Attempt09",
        {
            "hand": 100_108_071_901,
            "rerank": 101_108_071_901,
            "veto": 102_108_071_901,
            "stress": 103_108_071_901,
            "confirmation": 104_108_071_901,
            "evaluation": 105_108_071_901,
            "child": 106_108_071_901,
        },
        range(250),
    ),
    (
        "Attempt09 preflight",
        {
            "hand": 110_108_071_901,
            "rerank": 111_108_071_901,
            "veto": 112_108_071_901,
            "stress": 113_108_071_901,
            "confirmation": 114_108_071_901,
            "evaluation": 115_108_071_901,
            "child": 116_108_071_901,
        },
        range(3),
    ),
    (
        "Attempt10",
        {
            "hand": 120_108_071_901,
            "rerank": 121_108_071_901,
            "veto": 122_108_071_901,
            "stress": 123_108_071_901,
            "confirmation": 124_108_071_901,
            "evaluation": 125_108_071_901,
            "child": 126_108_071_901,
        },
        range(250),
    ),
    (
        "Attempt10 preflight",
        {
            "hand": 130_108_071_901,
            "rerank": 131_108_071_901,
            "veto": 132_108_071_901,
            "stress": 133_108_071_901,
            "confirmation": 134_108_071_901,
            "evaluation": 135_108_071_901,
            "child": 136_108_071_901,
        },
        range(3),
    ),
)


def load_and_validate_attempt11_plan(path: str | Path) -> dict[str, Any]:
    """Load a byte-identical plan and fail closed on any drift."""

    target = Path(path)
    if hashlib.sha256(target.read_bytes()).hexdigest() != M43_ATTEMPT11_PLAN_SHA256:
        raise ValueError("Attempt11 plan SHA-256 changed")
    plan = json.loads(target.read_text(encoding="utf-8-sig"))
    if not isinstance(plan, dict):
        raise ValueError("Attempt11 plan must be a mapping")
    validate_attempt11_plan(plan)
    return plan


def validate_attempt11_plan(plan: Mapping[str, Any]) -> None:
    """Validate the byte-bound plan plus its independently derived invariants."""

    canonical = _load_canonical_plan()
    _require_exact(plan, canonical, "plan")
    _validate_frozen_semantics(plan)
    _validate_profile_balance(plan)
    _validate_compute_accounting(plan)
    validate_attempt11_seed_freshness(plan)


def validate_attempt11_artifact_bindings(
    plan: Mapping[str, Any], *, repository_root: str | Path
) -> None:
    """Validate the preserved Attempt10 plan and unchanged frozen dependencies."""

    validate_attempt11_plan(plan)
    root = Path(repository_root).resolve()
    search = _mapping(plan.get("search_protocol"), "search_protocol")
    baseline = _mapping(plan.get("baseline_hash_audit"), "baseline_hash_audit")
    bindings = (
        (
            "configs/hu_joint_policy_m43_attempt10.json",
            ATTEMPT10_PLAN_SHA256,
            "preserved Attempt10 plan",
        ),
        (
            search["candidate_generator_artifact_path"],
            search["candidate_generator_artifact_sha256"],
            "Attempt11 Lambda model",
        ),
        (
            baseline["policy_registry_path"],
            baseline["policy_registry_expected_sha256"],
            "AI profile registry",
        ),
    )
    resolved: dict[str, Path] = {}
    for relative_path, expected_sha256, label in bindings:
        target = (root / str(relative_path)).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{label} path escapes repository root") from exc
        if not target.is_file():
            raise ValueError(f"{label} artifact is missing: {relative_path}")
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError(f"{label} SHA-256 changed")
        resolved[label] = target

    attempt10_payload = json.loads(
        resolved["preserved Attempt10 plan"].read_text(encoding="utf-8-sig")
    )
    if (
        attempt10_payload.get("schema") != "hu_m43_attempt10_plan_v1"
        or attempt10_payload.get("activation_guards", {}).get(
            "runtime_policy_activated"
        )
        is not False
        or attempt10_payload.get("scope", {}).get("current_profile_mutated")
        is not False
    ):
        raise ValueError("Attempt10 preserved boundary changed")


def enumerate_attempt11_seed_schedules(
    plan: Mapping[str, Any], *, population: str
) -> dict[str, tuple[int, ...]]:
    """Derive all seven schedules for development, audit, or preflight."""

    if population == "preflight":
        return enumerate_attempt11_preflight_seed_schedules(plan)

    seeds = _mapping(plan.get("seed_contract"), "seed_contract")
    stride = _strict_int(seeds.get("seed_stride"), "seed_contract.seed_stride")
    if population == "development":
        spec = _mapping(plan.get("development_population"), population)
    elif population == "future_audit":
        spec = _mapping(plan.get("future_audit_population"), population)
    else:
        raise ValueError(
            "population must be development, future_audit, or preflight"
        )
    first = _strict_int(spec.get("root_index_first"), f"{population}.first")
    last = _strict_int(spec.get("root_index_last"), f"{population}.last")
    roots = _strict_int(spec.get("roots"), f"{population}.roots")
    if last - first + 1 != roots:
        raise ValueError(f"{population} root-index range does not match roots")
    return _enumerate_from_contract(
        seeds,
        stride=stride,
        indices=range(first, last + 1),
    )


def enumerate_attempt11_preflight_seed_schedules(
    plan: Mapping[str, Any],
) -> dict[str, tuple[int, ...]]:
    """Derive the seven unique preflight source-root schedules."""

    seeds = _mapping(plan.get("preflight_seed_contract"), "preflight_seed_contract")
    stride = _strict_int(
        seeds.get("seed_stride"), "preflight_seed_contract.seed_stride"
    )
    indices = seeds.get("source_root_indices")
    if not isinstance(indices, list) or any(type(value) is not int for value in indices):
        raise ValueError("preflight source_root_indices changed")
    return _enumerate_from_contract(seeds, stride=stride, indices=indices)


def enumerate_known_seed_schedules() -> dict[str, tuple[int, ...]]:
    """Materialize every known Attempt06 through Attempt10 seed namespace."""

    schedules: dict[str, tuple[int, ...]] = {}
    for group, bases, indices in _KNOWN_SEED_GROUPS:
        for domain, base in bases.items():
            schedules[f"{group}.{domain}"] = tuple(
                base + M43_ATTEMPT11_SEED_STRIDE * index for index in indices
            )
    return schedules


def validate_attempt11_seed_freshness(plan: Mapping[str, Any]) -> None:
    """Prove new development/audit/preflight schedules are mutually fresh."""

    declared: dict[str, tuple[int, ...]] = {}
    for population, expected_count in (("development", 200), ("future_audit", 50)):
        for domain, values in enumerate_attempt11_seed_schedules(
            plan, population=population
        ).items():
            if len(values) != expected_count or len(set(values)) != expected_count:
                raise ValueError(f"Attempt11 {population}.{domain} is not unique")
            declared[f"{population}.{domain}"] = values
    for domain, values in enumerate_attempt11_preflight_seed_schedules(plan).items():
        if len(values) != 5 or len(set(values)) != 5:
            raise ValueError(f"Attempt11 preflight.{domain} is not unique")
        declared[f"preflight.{domain}"] = values
    _reject_pairwise_overlap(declared, label="Attempt11 declared seed schedules")

    known = enumerate_known_seed_schedules()
    for new_name, new_values in declared.items():
        new_set = set(new_values)
        for old_name, old_values in known.items():
            if new_set.intersection(old_values):
                raise ValueError(f"Attempt11 {new_name} overlaps {old_name}")


def _load_canonical_plan() -> dict[str, Any]:
    if not _CANONICAL_PLAN_PATH.is_file():
        raise ValueError("canonical Attempt11 plan is missing")
    raw = _CANONICAL_PLAN_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != M43_ATTEMPT11_PLAN_SHA256:
        raise ValueError("canonical Attempt11 plan SHA-256 changed")
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("canonical Attempt11 plan must be a mapping")
    return value


def _validate_frozen_semantics(plan: Mapping[str, Any]) -> None:
    _require_exact(plan.get("schema"), M43_ATTEMPT11_PLAN_SCHEMA, "plan.schema")
    _require_exact(plan.get("milestone"), "M4.3-attempt11", "plan.milestone")
    _require_exact(plan.get("status_date"), "2026-07-15", "plan.status_date")
    _require_exact(plan.get("status"), "frozen_pre_preflight", "plan.status")
    _require_exact(plan.get("profiles"), list(M43_ATTEMPT11_PROFILES), "profiles")
    preflight = _mapping(plan.get("preflight_population"), "preflight_population")
    _require_exact(preflight.get("source_root_indices"), [0, 1, 2, 3, 4], "preflight roots")
    _require_exact(
        preflight.get("execution_slots"),
        [
            "root0_batch_a",
            "root0_batch_b",
            "root0_scalar",
            "root1_batch",
            "root2_batch",
            "root3_batch",
            "root4_batch",
        ],
        "preflight slots",
    )
    if (
        preflight.get("all_five_profiles_covered") is not True
        or preflight.get("root4_variable_candidate_count_explicitly_verified")
        is not True
    ):
        raise ValueError("Attempt11 seven-shard full-profile preflight changed")

    boundary = _mapping(plan.get("attempt10_boundary"), "attempt10_boundary")
    _require_exact(
        boundary.get("status"),
        "development_label_generation_failed_before_labels",
        "attempt10_boundary.status",
    )
    _require_exact(
        boundary.get("failed_root_indices"), [4, 24, 79], "failed roots"
    )
    _require_exact(
        boundary.get("failed_profile"), "random_exact_final", "failed profile"
    )
    if boundary.get("all_attempt10_files_and_artifacts_preserved") is not True:
        raise ValueError("Attempt10 artifacts must remain preserved")

    scope = _mapping(plan.get("scope"), "scope")
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
            raise ValueError(f"scope.{name} must remain false")
    if scope.get("public_information_set_only") is not True:
        raise ValueError("Attempt11 must remain public-information-set only")

    search = _mapping(plan.get("search_protocol"), "search_protocol")
    _require_exact(
        search.get("candidate_generator_artifact_sha256"),
        ATTEMPT11_LAMBDA_MODEL_SHA256,
        "candidate artifact sha",
    )
    _require_exact(search.get("learned_nonbaseline_max"), 12, "candidate max")
    _require_exact(
        search.get("actual_nonbaseline_count_range_inclusive"),
        [0, 12],
        "candidate count range",
    )
    if (
        search.get("candidate_padding_allowed") is not False
        or search.get("candidate_duplication_allowed") is not False
        or search.get("baseline_added_exactly_once") is not True
    ):
        raise ValueError("Attempt11 candidate uniqueness/baseline contract changed")
    rerank = _mapping(search.get("rerank"), "rerank")
    _require_exact(rerank.get("symbol"), "R128", "rerank.symbol")
    _require_exact(rerank.get("samples"), 128, "rerank.samples")
    _require_exact(
        rerank.get("action_scope"),
        "n_unique_ranked_nonbaseline_actions_plus_explicit_baseline_where_n_is_0_to_12",
        "rerank.action_scope",
    )
    shortlist = _mapping(search.get("shortlist"), "shortlist")
    _require_exact(shortlist.get("symbol"), "K=min(8,n)", "shortlist.symbol")
    _require_exact(shortlist.get("nonbaseline_actions"), "min(8,n)", "shortlist.size")
    _require_exact(shortlist.get("nonbaseline_actions_max"), 8, "shortlist.max")
    _require_exact(
        shortlist.get("rerank_head_actions"), "min(4,n)", "shortlist.head"
    )
    _require_exact(
        shortlist.get("selection"),
        "first_min(4,n)_in_R_order_plus_unique_lambda_predicted_risk_fill_from_remaining_R_positions",
        "shortlist.selection",
    )
    reserve = _mapping(shortlist.get("risk_reserve"), "shortlist.risk_reserve")
    _require_exact(
        reserve.get("candidate_R_positions"),
        "remaining_R_positions_after_first_min(4,n)",
        "shortlist.risk_reserve.positions",
    )
    _require_exact(
        reserve.get("count"),
        "min(8,n)-min(4,n)",
        "shortlist.risk_reserve.count",
    )
    _require_exact(
        reserve.get("selection"),
        "minimum_lambda_risk_score_then_ActionKey_without_replacement_until_K_is_full",
        "shortlist.risk_reserve.selection",
    )
    _require_exact(
        shortlist.get("final_nonbaseline_order"),
        "original_R_order",
        "shortlist.order",
    )

    veto = _mapping(search.get("veto"), "veto")
    _require_exact(veto.get("symbol"), "V256", "veto.symbol")
    _require_exact(veto.get("samples"), 256, "veto.samples")
    _require_exact(
        veto.get("eligibility"),
        {
            "paired_delta_mean_strictly_greater_than": 0.0,
            "paired_delta_p05_min": -25.0,
            "paired_delta_p01_min": -40.0,
            "paired_delta_min_min": -50.0,
        },
        "veto.eligibility",
    )
    _require_exact(
        veto.get("retention"),
        "all_passing_nonbaseline_actions_in_original_R_order",
        "veto.retention",
    )
    if veto.get("may_select_final_action") is not False:
        raise ValueError("V256 may not select the final action")

    for name, symbol, samples, scope_name in (
        (
            "stress",
            "X1024",
            1024,
            "all_V256_retained_nonbaseline_actions_plus_explicit_baseline",
        ),
        (
            "confirmation",
            "C512",
            512,
            "all_X1024_retained_nonbaseline_actions_plus_explicit_baseline",
        ),
    ):
        phase = _mapping(search.get(name), name)
        _require_exact(phase.get("symbol"), symbol, f"{name}.symbol")
        _require_exact(phase.get("samples"), samples, f"{name}.samples")
        _require_exact(phase.get("action_scope"), scope_name, f"{name}.scope")
        _require_exact(
            phase.get("eligibility"),
            {
                "paired_delta_mean_strictly_greater_than": 0.0,
                "paired_delta_p05_min": -22.0,
                "paired_delta_p01_min": -36.0,
                "paired_delta_min_min": -45.0,
            },
            f"{name}.eligibility",
        )
        _require_exact(
            phase.get("retention"),
            "all_passing_nonbaseline_actions_in_original_R_order",
            f"{name}.retention",
        )
        if name == "stress" and phase.get("may_rerank_candidates") is not False:
            raise ValueError(f"{symbol} may not rerank candidates")
    confirmation = _mapping(search.get("confirmation"), "confirmation")
    _require_exact(
        confirmation.get("selection"),
        "minimum_normalized_tail_risk_then_paired_delta_mean_desc_then_original_R_position_then_ActionKey",
        "confirmation.selection",
    )
    _require_exact(
        confirmation.get("selection_risk_score"),
        "max(max(0,-paired_delta_p05)/22.0,max(0,-paired_delta_p01)/36.0,max(0,-paired_delta_min)/45.0)",
        "confirmation.selection_risk_score",
    )
    if confirmation.get("may_rerank_candidates") is not True:
        raise ValueError("C512 must apply the frozen final selection ordering")
    if confirmation.get("selection_is_final_only_after_strict_eligibility") is not True:
        raise ValueError("C512 selection must follow strict eligibility")

    evaluation = _mapping(search.get("evaluation"), "evaluation")
    _require_exact(evaluation.get("symbol"), "E256", "evaluation.symbol")
    _require_exact(evaluation.get("samples"), 256, "evaluation.samples")
    if evaluation.get("diagnostics_only") is not True:
        raise ValueError("E256 must remain diagnostics-only")
    if evaluation.get("may_rerank_veto_confirm_promote_or_change_root_output") is not False:
        raise ValueError("E256 may not alter the frozen root decision")
    _require_exact(
        search.get("rng_domains"),
        ["rerank", "veto", "stress", "confirmation", "evaluation"],
        "rng_domains",
    )
    if search.get("candidate_selection_and_evaluation_rng_independent") is not True:
        raise ValueError("selection and E256 RNG must remain independent")

    _require_exact(
        plan.get("future_audit_go_no_go"),
        {
            "all_gates_required": True,
            "assessment_source": "disjoint_E256_locked_final_nonbaseline_output_vs_explicit_baseline",
            "fires_total_min": 10,
            "fires_each_profile_min": 1,
            "mean_delta_per_state_strictly_greater_than": 0.0,
            "mean_delta_per_fire_strictly_greater_than": 0.0,
            "false_positive_rate_per_fire_max": 0.4,
            "override_loss_p95_max": 25.0,
            "override_loss_p99_max": 40.0,
            "override_loss_max": 50.0,
            "quantile_method": "numpy_linear",
            "nonfire_exact_baseline_action_fallback_required": True,
            "teacher_values_reported_as_realized_match_ev": False,
            "realized_population_acceptance_still_required": True,
        },
        "future_audit_go_no_go",
    )

    guards = _mapping(plan.get("activation_guards"), "activation_guards")
    if not guards or any(value is not False for value in guards.values()):
        raise ValueError("all Attempt11 activation guards must remain false")


def _validate_profile_balance(plan: Mapping[str, Any]) -> None:
    profiles = tuple(plan.get("profiles", ()))
    for section, expected in (("development_population", 40), ("future_audit_population", 10)):
        spec = _mapping(plan.get(section), section)
        first = _strict_int(spec.get("root_index_first"), f"{section}.first")
        last = _strict_int(spec.get("root_index_last"), f"{section}.last")
        assigned = [profiles[index % len(profiles)] for index in range(first, last + 1)]
        if any(assigned.count(profile) != expected for profile in profiles):
            raise ValueError(f"Attempt11 {section} is not profile-balanced")


def _validate_compute_accounting(plan: Mapping[str, Any]) -> None:
    search = _mapping(plan.get("search_protocol"), "search_protocol")
    r = _strict_int(_mapping(search.get("rerank"), "rerank").get("samples"), "R")
    v = _strict_int(_mapping(search.get("veto"), "veto").get("samples"), "V")
    x = _strict_int(_mapping(search.get("stress"), "stress").get("samples"), "X")
    c = _strict_int(
        _mapping(search.get("confirmation"), "confirmation").get("samples"), "C"
    )
    e = _strict_int(
        _mapping(search.get("evaluation"), "evaluation").get("samples"), "E"
    )
    _require_exact(
        plan.get("single_arm_cost"),
        {
            "rerank_action_futures_formula": "(n+1)*128",
            "rerank_action_futures_min": r,
            "rerank_action_futures_max": 13 * r,
            "veto_action_futures_formula": "(min(8,n)+1)*256",
            "veto_action_futures_min": v,
            "veto_action_futures_max": 9 * v,
            "fixed_prefilter_action_futures_formula": (
                "(n+1)*128+(min(8,n)+1)*256"
            ),
            "fixed_prefilter_action_futures_min": r + v,
            "fixed_prefilter_action_futures_max": 13 * r + 9 * v,
            "stress_action_futures_formula": "(V_retained_nonbaseline_count+1)*1024",
            "stress_action_futures_min_when_opened": 2 * x,
            "stress_action_futures_max_when_opened": 9 * x,
            "confirmation_action_futures_formula": "(X_retained_nonbaseline_count+1)*512",
            "confirmation_action_futures_min_when_opened": 2 * c,
            "confirmation_action_futures_max_when_opened": 9 * c,
            "evaluation_action_futures_per_final_fire": 2 * e,
            "maximum_full_fire_action_futures_per_root": 13 * r
            + 9 * v
            + 9 * x
            + 9 * c
            + 2 * e,
        },
        "single_arm_cost",
    )


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
    "ATTEMPT10_PLAN_SHA256",
    "ATTEMPT11_CANDIDATE_MAX",
    "ATTEMPT11_CONFIRMATION_SAMPLES",
    "ATTEMPT11_EVALUATION_SAMPLES",
    "ATTEMPT11_LAMBDA_MODEL_SHA256",
    "ATTEMPT11_RERANK_SAMPLES",
    "ATTEMPT11_SHORTLIST_MAX",
    "ATTEMPT11_STRESS_SAMPLES",
    "ATTEMPT11_STRICT_MIN_P01",
    "ATTEMPT11_STRICT_MIN_P05",
    "ATTEMPT11_STRICT_MIN_VALUE",
    "ATTEMPT11_VETO_MIN_MEAN",
    "ATTEMPT11_VETO_MIN_P01",
    "ATTEMPT11_VETO_MIN_P05",
    "ATTEMPT11_VETO_MIN_VALUE",
    "ATTEMPT11_VETO_SAMPLES",
    "M43_ATTEMPT11_PLAN_SCHEMA",
    "M43_ATTEMPT11_PLAN_SHA256",
    "M43_ATTEMPT11_PROFILES",
    "M43_ATTEMPT11_SEED_STRIDE",
    "enumerate_attempt11_preflight_seed_schedules",
    "enumerate_attempt11_seed_schedules",
    "enumerate_known_seed_schedules",
    "load_and_validate_attempt11_plan",
    "validate_attempt11_artifact_bindings",
    "validate_attempt11_plan",
    "validate_attempt11_seed_freshness",
]
