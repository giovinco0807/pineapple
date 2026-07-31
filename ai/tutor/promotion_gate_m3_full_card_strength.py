"""Fail-closed M3 full-card T3/T4 strength-promotion gate.

This module validates *independent holdout* evidence.  It intentionally does
not run the solver and it never treats the six-stratum execution smoke as
promotion evidence.  The validator is pure: a JSON-compatible evidence object
and a locked JSON-compatible configuration produce a JSON-compatible result.

The evidence contract is deliberately raw.  Each root/seed row carries the
root action distributions, action payoff estimates, information-set coverage,
audits, checkpoint binding, and runtime.  Policy payoff, reference payoff,
best payoff, EV regret, policy drift, percentiles, seat-swap noninferiority,
and runtime aggregates are re-derived here; published aggregates are accepted
only when they exactly match those re-derived values.

This is a practical-strength gate.  Full-card exact exploitability is not
computed by this workflow and every relevant manifest/row must explicitly say
``exact_exploitability_computed == false``.

The behavior contract distinguishes exogenous T1/T2 trace calibration from
the endogenous T3-BB action likelihood needed to build a BTN ``t3_second``
posterior.  Promotion requires that fifth route to be bound to a converged
candidate-policy/posterior fixed point; the non-promoting smoke's frozen T3-BB
ranking prior is intentionally ineligible.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections import defaultdict
from fractions import Fraction
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "ofc_m3_full_card_strength_gate_config/v1"
EVIDENCE_SCHEMA = "ofc_m3_full_card_strength_evidence/v1"
RESULT_SCHEMA = "ofc_m3_full_card_strength_gate_result/v1"
GATE_ID = "promotion_gate_m3_full_card_strength"
SCOPE = "full_card_t3_t4_root_disjoint_strength_promotion"
EVIDENCE_KIND = "independent_locked_holdout"
PASS_STATUS = "m3_full_card_strength_ready"
FAIL_STATUS = "m3_full_card_strength_blocked"

BEHAVIOR_SCHEMA = "ofc_calibrated_behavior_manifest/v2"
BEHAVIOR_MODEL_TYPE = (
    "t1_t2_temperature_calibrated_plus_t3_bb_endogenous_fixed_point_v2"
)
T3_BB_LIKELIHOOD_SCHEMA = "ofc_m3_t3_bb_endogenous_likelihood_binding/v1"
T3_BB_LIKELIHOOD_METHOD = "candidate_policy_posterior_fixed_point_v1"
SOURCE_SCHEMA = "ofc_m3_full_card_source_manifest/v1"
SOLVER_SCHEMA = "ofc_m3_full_card_solver_manifest/v1"
ROOT_MANIFEST_SCHEMA = "ofc_m3_full_card_holdout_roots/v1"
ROOT_COMMITMENT_SCHEMA = "ofc_m3_full_card_holdout_root_commitment/v1"
ROOT_IDENTITY_COMMITMENT_SCHEMA = "ofc_root_identity_commitment/v1"
ROOT_PARTITION_SCHEMA = "ofc_root_commitment_partition/v1"
CHECKPOINT_SCHEMA = "ofc_m3_full_card_checkpoint_manifest/v1"

RULESET = "standard_ofc_pineapple_hu_joker2"
POSITION_CONTRACT_VERSION = "bb_first_v1"
SOLVER_METHOD = "full_card_dynamic_external_sampling_mccfr_plus_v1"
SOLVER_ADAPTER = "full_card_generative_t3_t4_v1"
INFORMATION_MODEL = "public_only_no_opponent_private_cards"

REQUIRED_STRATA = (
    "bb_joker0",
    "bb_joker1",
    "bb_joker2",
    "btn_joker0",
    "btn_joker1",
    "btn_joker2",
)
REQUIRED_EXCLUDED_PARTITIONS = ("training", "calibration", "smoke")
REQUIRED_AUDITS = (
    "legal_action_failures",
    "hidden_information_leak_failures",
    "opponent_private_card_policy_input_failures",
    "hidden_discard_policy_key_failures",
)
CALIBRATION_ROLE_KEYS = ("t1_bb", "t1_btn", "t2_bb", "t2_btn")
CALIBRATION_BINDING_KEYS = (
    "checkpoint_sha256",
    "model_sha256",
    "row_extractor_sha256",
    "adapter_source_sha256",
)
CALIBRATED_BEHAVIOR_KEYS = frozenset(
    {
        "schema",
        "model_type",
        "calibrated",
        "calibration_method",
        "calibration_dataset_kind",
        "root_split",
        "promotion_eligible",
        "role_temperatures",
        "calibration_artifact_sha256",
        "calibration_gate_config_sha256",
        "calibration_gate_result_sha256",
        "role_model_bindings",
        "t3_bb_likelihood_binding",
        "training_root_partition_sha256",
        "calibration_root_partition_sha256",
    }
)
T3_BB_LIKELIHOOD_KEYS = frozenset(
    {
        "schema",
        "route",
        "consumer_root_actor",
        "consumer_phase",
        "method",
        "promotion_eligible",
        "fixed_point_converged",
        "candidate_policy_artifact_sha256",
        "candidate_policy_checkpoint_sha256",
        "fixed_point_evidence_sha256",
        "fixed_point_gate_result_sha256",
        "fixed_point_config_sha256",
        "range_builder_source_sha256",
        "solver_manifest_sha256",
        "binding_sha256",
    }
)
T3_FULL_CARD_RANGE_SOURCE_PATH = "ai/tutor/t3_hu_full_card_range.py"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FLOAT_ABS_TOL = 1e-12


def canonical_json(value: Any) -> str:
    """Return the finite canonical JSON representation used by this gate."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 of :func:`canonical_json`."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def root_commitment_sha256(root: Mapping[str, Any]) -> str:
    """Commit every root field except the commitment itself with domain separation."""

    content = dict(root)
    content.pop("root_commitment_sha256", None)
    return canonical_sha256({"schema": ROOT_COMMITMENT_SCHEMA, "root": content})


def root_identity_commitment_sha256(root_id: str) -> str:
    """Commit a split-independent root ID for overlap checks.

    Calibration, training, smoke, and strength artifacts must use this same
    domain.  The full root commitment above binds this identity commitment to
    the actor/Joker/public-observation record, while overlap checks use the
    identity commitment so changing turn- or actor-specific fields cannot hide
    reuse of the same source root.
    """

    if not isinstance(root_id, str) or not root_id:
        raise ValueError("root_id must be a non-empty string")
    return canonical_sha256(
        {"schema": ROOT_IDENTITY_COMMITMENT_SCHEMA, "root_id": root_id}
    )


def self_hash(value: Mapping[str, Any], field: str) -> str:
    """Compute a canonical self-hash after removing ``field``.

    Artifact producers may use this helper to create evidence whose hashing is
    byte-for-byte identical to the verifier.
    """

    content = dict(value)
    content.pop(field, None)
    return canonical_sha256(content)


def _validate_t3_bb_likelihood_binding(
    value: Any,
    *,
    solver_manifest_sha256: str | None,
    range_builder_source_sha256: str | None,
    label: str,
    failures: list[str],
) -> str | None:
    """Validate the likelihood route needed by a BTN ``t3_second`` root.

    T1/T2 observations are exogenous history and may be temperature calibrated
    from locked traces.  The immediately preceding BB-T3 action is different:
    in an equilibrium solve its likelihood must come from the candidate policy
    whose posterior/policy fixed point has converged.  Keeping this as a
    separate binding prevents a four-role calibration artifact from silently
    standing in for the fifth route used by the physical range builder.
    """

    if not isinstance(value, Mapping):
        failures.append(f"{label}: object required")
        return None
    computed = _validate_self_hash(
        value, field="binding_sha256", label=label, failures=failures
    )
    if set(value) != T3_BB_LIKELIHOOD_KEYS:
        missing = sorted(T3_BB_LIKELIHOOD_KEYS - set(value))
        extra = sorted(set(value) - T3_BB_LIKELIHOOD_KEYS)
        failures.append(
            f"{label}: exact versioned fields required; "
            f"missing={missing}, extra={extra}"
        )
    expected = {
        "schema": T3_BB_LIKELIHOOD_SCHEMA,
        "route": "t3_bb",
        "consumer_root_actor": "btn",
        "consumer_phase": "t3_second",
        "method": T3_BB_LIKELIHOOD_METHOD,
        "promotion_eligible": True,
        "fixed_point_converged": True,
    }
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            failures.append(f"{label}.{field}: must equal {wanted!r}")
    for field in (
        "candidate_policy_artifact_sha256",
        "candidate_policy_checkpoint_sha256",
        "fixed_point_evidence_sha256",
        "fixed_point_gate_result_sha256",
        "fixed_point_config_sha256",
        "range_builder_source_sha256",
        "solver_manifest_sha256",
    ):
        if not _valid_sha256(value.get(field)):
            failures.append(f"{label}.{field}: lowercase SHA256 required")
    if (
        solver_manifest_sha256 is not None
        and value.get("solver_manifest_sha256") != solver_manifest_sha256
    ):
        failures.append(f"{label}.solver_manifest_sha256: solver binding mismatch")
    if (
        range_builder_source_sha256 is not None
        and value.get("range_builder_source_sha256")
        != range_builder_source_sha256
    ):
        failures.append(
            f"{label}.range_builder_source_sha256: source manifest binding mismatch"
        )
    return computed


def verify_t3_bb_likelihood_binding(
    value: Any,
    *,
    solver_manifest_sha256: str | None = None,
    range_builder_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Fail closed on a standalone endogenous T3-BB likelihood binding."""

    failures: list[str] = []
    _validate_t3_bb_likelihood_binding(
        value,
        solver_manifest_sha256=solver_manifest_sha256,
        range_builder_source_sha256=range_builder_source_sha256,
        label="t3_bb_likelihood_binding",
        failures=failures,
    )
    if failures:
        raise ValueError("; ".join(failures))
    return dict(value)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _canonical_positive_fraction(value: Any) -> Fraction | None:
    """Parse one reduced ``numerator/denominator`` calibration temperature."""

    if not isinstance(value, str) or "/" not in value:
        return None
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    if parsed <= 0 or value != f"{parsed.numerator}/{parsed.denominator}":
        return None
    return parsed


def _expected_identity(stratum: str) -> tuple[str, int]:
    actor, joker_text = stratum.split("_joker", 1)
    return actor, int(joker_text)


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("nearest-rank quantile requires at least one value")
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _distribution_tv(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    if set(left) != set(right):
        return 1.0
    return 0.5 * math.fsum(
        abs(float(left[action]) - float(right[action])) for action in sorted(left)
    )


def _weighted_payoff(
    distribution: Mapping[str, Any], action_payoffs: Mapping[str, Any]
) -> float:
    return math.fsum(
        float(distribution[action]) * float(action_payoffs[action])
        for action in sorted(action_payoffs)
    )


def derive_strength_metrics(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Re-derive all published strength metrics from raw evidence rows.

    This helper assumes structurally valid evidence.  The public validator
    performs the fail-closed structural checks before calling it.  It is also
    exposed so the artifact producer can publish the exact expected summary
    without maintaining a second aggregation implementation.
    """

    rows = list(evidence["raw_run_rows"])
    roots = list(evidence["root_manifest"]["roots"])
    partitions = evidence["excluded_root_partitions"]

    roots_by_stratum: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for root in roots:
        roots_by_stratum[str(root["stratum"])].append(root)

    rows_by_stratum: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    rows_by_root: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_stratum[str(row["stratum"])].append(row)
        rows_by_root[str(row["root_id"])].append(row)

    holdout_commitments = {
        str(root["root_identity_commitment_sha256"]) for root in roots
    }
    excluded_sets = {
        name: set(partitions[name]["root_commitments"])
        for name in REQUIRED_EXCLUDED_PARTITIONS
    }
    excluded_union = set().union(*excluded_sets.values())
    excluded_pair_overlap = sum(
        len(excluded_sets[left] & excluded_sets[right])
        for left, right in itertools.combinations(REQUIRED_EXCLUDED_PARTITIONS, 2)
    )

    regrets = [float(row["ev_regret_estimate"]) for row in rows]
    runtimes = [float(row["runtime_ms"]) for row in rows]
    coverage = [float(row["encountered_infoset_coverage"]) for row in rows]
    audit_totals = {
        field: sum(int(row["audits"][field]) for row in rows)
        for field in REQUIRED_AUDITS
    }

    tv_by_root: dict[str, list[float]] = {}
    all_tvs: list[float] = []
    for root_id, root_rows in rows_by_root.items():
        tvs = [
            _distribution_tv(left["root_action_distribution"], right["root_action_distribution"])
            for left, right in itertools.combinations(root_rows, 2)
        ]
        tv_by_root[root_id] = tvs
        all_tvs.extend(tvs)

    pair_groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        pair_groups[(str(row["seat_swap_pair_id"]), int(row["solver_seed"]))].append(row)
    single_seat_deltas: list[float] = []
    pair_mean_deltas: list[float] = []
    for group in pair_groups.values():
        deltas = [
            float(row["policy_payoff_estimate"])
            - float(row["reference_payoff_estimate"])
            for row in group
        ]
        single_seat_deltas.extend(deltas)
        pair_mean_deltas.append(math.fsum(deltas) / len(deltas))

    strata_metrics: dict[str, Any] = {}
    for name in REQUIRED_STRATA:
        stratum_rows = rows_by_stratum[name]
        stratum_root_ids = [str(root["root_id"]) for root in roots_by_stratum[name]]
        stratum_regrets = [float(row["ev_regret_estimate"]) for row in stratum_rows]
        stratum_tvs = [
            tv for root_id in stratum_root_ids for tv in tv_by_root.get(root_id, ())
        ]
        seeds = {int(row["solver_seed"]) for row in stratum_rows}
        seed_counts = [
            len({int(row["solver_seed"]) for row in rows_by_root[root_id]})
            for root_id in stratum_root_ids
        ]
        strata_metrics[name] = {
            "root_count": len(stratum_root_ids),
            "raw_run_row_count": len(stratum_rows),
            "independent_seed_count": len(seeds),
            "min_independent_seeds_per_root": min(seed_counts),
            "min_encountered_infoset_coverage": min(
                float(row["encountered_infoset_coverage"])
                for row in stratum_rows
            ),
            "policy_tv_comparison_count": len(stratum_tvs),
            "max_policy_tv_drift": max(stratum_tvs or [0.0]),
            "mean_ev_regret_score": math.fsum(stratum_regrets) / len(stratum_regrets),
            "p95_ev_regret_score": _nearest_rank(stratum_regrets, 0.95),
            "p99_ev_regret_score": _nearest_rank(stratum_regrets, 0.99),
        }

    return {
        "required_strata": list(REQUIRED_STRATA),
        "strata": strata_metrics,
        "global": {
            "holdout_root_count": len(roots),
            "raw_run_row_count": len(rows),
            "checkpoint_count": len(evidence["checkpoint_manifests"]),
            "holdout_excluded_root_overlap_count": len(
                holdout_commitments & excluded_union
            ),
            "excluded_partition_pairwise_overlap_count": excluded_pair_overlap,
            "legal_action_failures": audit_totals["legal_action_failures"],
            "hidden_information_leak_failures": audit_totals[
                "hidden_information_leak_failures"
            ],
            "opponent_private_card_policy_input_failures": audit_totals[
                "opponent_private_card_policy_input_failures"
            ],
            "hidden_discard_policy_key_failures": audit_totals[
                "hidden_discard_policy_key_failures"
            ],
            "min_encountered_infoset_coverage": min(coverage),
            "policy_tv_comparison_count": len(all_tvs),
            "max_policy_tv_drift": max(all_tvs or [0.0]),
            "mean_ev_regret_score": math.fsum(regrets) / len(regrets),
            "p95_ev_regret_score": _nearest_rank(regrets, 0.95),
            "p99_ev_regret_score": _nearest_rank(regrets, 0.99),
            "paired_seat_swap_comparison_count": len(pair_groups),
            "paired_seat_swap_min_single_seat_delta_score": min(
                single_seat_deltas
            ),
            "paired_seat_swap_min_pair_mean_delta_score": min(pair_mean_deltas),
            "runtime_ms_p95": _nearest_rank(runtimes, 0.95),
            "runtime_ms_max": max(runtimes),
            "runtime_ms_total": math.fsum(runtimes),
            "exact_exploitability_computed": False,
        },
    }


def _validate_self_hash(
    value: Any, *, field: str, label: str, failures: list[str]
) -> str | None:
    if not isinstance(value, Mapping):
        failures.append(f"{label}: must be an object")
        return None
    claimed = value.get(field)
    try:
        computed = self_hash(value, field)
    except (TypeError, ValueError, OverflowError):
        failures.append(f"{label}: must be finite canonical JSON")
        return None
    if claimed != computed:
        failures.append(f"{label}.{field}: canonical self-hash mismatch")
    return computed


def _config_snapshot(
    config: Any, failures: list[str]
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(config, Mapping):
        failures.append("config: must be an object")
        return None, None
    try:
        digest = canonical_sha256(config)
    except (TypeError, ValueError, OverflowError):
        failures.append("config: must be finite canonical JSON")
        return None, None

    if config.get("schema") != CONFIG_SCHEMA:
        failures.append(f"config.schema: must equal {CONFIG_SCHEMA!r}")
    if config.get("gate_id") != GATE_ID:
        failures.append(f"config.gate_id: must equal {GATE_ID!r}")

    for field in (
        "approved_calibrated_behavior_sha256",
        "approved_solver_manifest_sha256",
        "approved_source_manifest_sha256",
        "approved_holdout_root_manifest_sha256",
    ):
        if not _valid_sha256(config.get(field)):
            failures.append(f"config.{field}: lowercase SHA256 required")

    approved_partitions = config.get("approved_excluded_root_partition_sha256")
    if not isinstance(approved_partitions, Mapping):
        failures.append(
            "config.approved_excluded_root_partition_sha256: object required"
        )
        approved_partitions = {}
    if set(approved_partitions) != set(REQUIRED_EXCLUDED_PARTITIONS):
        failures.append(
            "config.approved_excluded_root_partition_sha256: exact "
            "training/calibration/smoke set required"
        )
    for name in REQUIRED_EXCLUDED_PARTITIONS:
        if not _valid_sha256(approved_partitions.get(name)):
            failures.append(
                "config.approved_excluded_root_partition_sha256."
                f"{name}: lowercase SHA256 required"
            )

    raw_thresholds = config.get("thresholds")
    required_thresholds = {
        "min_independent_seeds_per_stratum",
        "min_roots_per_stratum",
        "min_encountered_infoset_coverage",
        "max_policy_tv_drift",
        "max_mean_ev_regret_score",
        "max_p95_ev_regret_score",
        "max_p99_ev_regret_score",
        "paired_seat_swap_noninferiority_margin_score",
        "max_runtime_ms_p95",
        "max_runtime_ms_max",
        "max_runtime_ms_total",
    }
    if not isinstance(raw_thresholds, Mapping):
        failures.append("config.thresholds: object required")
        raw_thresholds = {}
    if set(raw_thresholds) != required_thresholds:
        failures.append(
            "config.thresholds: exact versioned threshold field set required"
        )

    seeds = raw_thresholds.get("min_independent_seeds_per_stratum")
    roots = raw_thresholds.get("min_roots_per_stratum")
    if not _is_int(seeds) or seeds < 2:
        failures.append(
            "config.thresholds.min_independent_seeds_per_stratum: integer >= 2 required"
        )
    if not _is_int(roots) or roots < 1:
        failures.append("config.thresholds.min_roots_per_stratum: integer >= 1 required")

    numeric_fields = required_thresholds - {
        "min_independent_seeds_per_stratum",
        "min_roots_per_stratum",
    }
    for field in sorted(numeric_fields):
        value = raw_thresholds.get(field)
        if not _finite(value) or float(value) < 0:
            failures.append(f"config.thresholds.{field}: finite nonnegative value required")
    coverage = raw_thresholds.get("min_encountered_infoset_coverage")
    tv = raw_thresholds.get("max_policy_tv_drift")
    if _finite(coverage) and float(coverage) > 1:
        failures.append(
            "config.thresholds.min_encountered_infoset_coverage: must be <= 1"
        )
    if _finite(tv) and float(tv) > 1:
        failures.append("config.thresholds.max_policy_tv_drift: must be <= 1")
    mean = raw_thresholds.get("max_mean_ev_regret_score")
    p95 = raw_thresholds.get("max_p95_ev_regret_score")
    p99 = raw_thresholds.get("max_p99_ev_regret_score")
    if all(_finite(value) for value in (mean, p95, p99)) and not (
        float(mean) <= float(p95) <= float(p99)
    ):
        failures.append("config.thresholds: regret limits must satisfy mean <= p95 <= p99")
    rt_p95 = raw_thresholds.get("max_runtime_ms_p95")
    rt_max = raw_thresholds.get("max_runtime_ms_max")
    rt_total = raw_thresholds.get("max_runtime_ms_total")
    if all(_finite(value) for value in (rt_p95, rt_max, rt_total)) and not (
        float(rt_p95) <= float(rt_max) <= float(rt_total)
    ):
        failures.append(
            "config.thresholds: runtime limits must satisfy p95 <= max <= total"
        )

    if failures:
        return None, digest
    return (
        {
            "approved_calibrated_behavior_sha256": config[
                "approved_calibrated_behavior_sha256"
            ],
            "approved_solver_manifest_sha256": config[
                "approved_solver_manifest_sha256"
            ],
            "approved_source_manifest_sha256": config[
                "approved_source_manifest_sha256"
            ],
            "approved_holdout_root_manifest_sha256": config[
                "approved_holdout_root_manifest_sha256"
            ],
            "approved_excluded_root_partition_sha256": dict(approved_partitions),
            "thresholds": dict(raw_thresholds),
        },
        digest,
    )


def _validate_partition(
    name: str,
    value: Any,
    *,
    approved_sha256: str | None,
    failures: list[str],
) -> set[str]:
    label = f"excluded_root_partitions.{name}"
    computed = _validate_self_hash(
        value, field="manifest_sha256", label=label, failures=failures
    )
    if not isinstance(value, Mapping):
        return set()
    if value.get("schema") != ROOT_PARTITION_SCHEMA:
        failures.append(f"{label}.schema: unsupported")
    if value.get("purpose") != name:
        failures.append(f"{label}.purpose: must equal {name!r}")
    commitments = value.get("root_commitments")
    if not isinstance(commitments, list) or not commitments:
        failures.append(f"{label}.root_commitments: non-empty array required")
        return set()
    if any(not _valid_sha256(item) for item in commitments):
        failures.append(f"{label}.root_commitments: every entry must be SHA256")
    if len(commitments) != len(set(commitments)):
        failures.append(f"{label}.root_commitments: duplicates forbidden")
    if commitments != sorted(commitments):
        failures.append(f"{label}.root_commitments: canonical sorted order required")
    if computed != approved_sha256:
        failures.append(f"{label}.manifest_sha256: not approved by locked config")
    return {item for item in commitments if _valid_sha256(item)}


def _validate_behavior(
    value: Any,
    *,
    claimed_sha256: Any,
    approved_sha256: str | None,
    partition_hashes: Mapping[str, Any],
    solver_manifest_sha256: str | None = None,
    range_builder_source_sha256: str | None = None,
    failures: list[str],
) -> str | None:
    label = "calibrated_behavior_manifest"
    if not isinstance(value, Mapping):
        failures.append(f"{label}: object required")
        return None
    try:
        computed = canonical_sha256(value)
    except (TypeError, ValueError, OverflowError):
        failures.append(f"{label}: finite canonical JSON required")
        return None
    if claimed_sha256 != computed:
        failures.append("calibrated_behavior_sha256: manifest hash mismatch")
    if computed != approved_sha256:
        failures.append("calibrated_behavior_sha256: not approved by locked config")
    if set(value) != CALIBRATED_BEHAVIOR_KEYS:
        missing = sorted(CALIBRATED_BEHAVIOR_KEYS - set(value))
        extra = sorted(set(value) - CALIBRATED_BEHAVIOR_KEYS)
        failures.append(
            f"{label}: exact versioned fields required; missing={missing}, extra={extra}"
        )
    expected = {
        "schema": BEHAVIOR_SCHEMA,
        "model_type": BEHAVIOR_MODEL_TYPE,
        "calibrated": True,
        "calibration_method": "t1_t2_observed_action_nll_temperature",
        "calibration_dataset_kind": "observed_full_trace_actions",
        "root_split": "root_disjoint_train_calibration_test",
        "promotion_eligible": True,
    }
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            failures.append(f"{label}.{field}: must equal {wanted!r}")
    for field in (
        "calibration_artifact_sha256",
        "calibration_gate_config_sha256",
        "calibration_gate_result_sha256",
    ):
        if not _valid_sha256(value.get(field)):
            failures.append(f"{label}.{field}: lowercase SHA256 required")

    role_temperatures = value.get("role_temperatures")
    if not isinstance(role_temperatures, Mapping) or set(role_temperatures) != set(
        CALIBRATION_ROLE_KEYS
    ):
        failures.append(
            f"{label}.role_temperatures: exact t1_bb/t1_btn/t2_bb/t2_btn set required"
        )
    else:
        for role in CALIBRATION_ROLE_KEYS:
            parsed = _canonical_positive_fraction(role_temperatures.get(role))
            if parsed is None:
                failures.append(
                    f"{label}.role_temperatures.{role}: reduced positive rational required"
                )
            elif not Fraction(1, 20) <= parsed <= Fraction(20, 1):
                failures.append(
                    f"{label}.role_temperatures.{role}: must be within [1/20,20/1]"
                )

    role_bindings = value.get("role_model_bindings")
    if not isinstance(role_bindings, Mapping) or set(role_bindings) != set(
        CALIBRATION_ROLE_KEYS
    ):
        failures.append(
            f"{label}.role_model_bindings: exact t1_bb/t1_btn/t2_bb/t2_btn set required"
        )
    else:
        for role in CALIBRATION_ROLE_KEYS:
            binding = role_bindings.get(role)
            binding_label = f"{label}.role_model_bindings.{role}"
            if not isinstance(binding, Mapping) or set(binding) != set(
                CALIBRATION_BINDING_KEYS
            ):
                failures.append(
                    f"{binding_label}: exact checkpoint/model/extractor/adapter hashes required"
                )
                continue
            for field in CALIBRATION_BINDING_KEYS:
                if not _valid_sha256(binding.get(field)):
                    failures.append(
                        f"{binding_label}.{field}: lowercase SHA256 required"
                    )
    _validate_t3_bb_likelihood_binding(
        value.get("t3_bb_likelihood_binding"),
        solver_manifest_sha256=solver_manifest_sha256,
        range_builder_source_sha256=range_builder_source_sha256,
        label=f"{label}.t3_bb_likelihood_binding",
        failures=failures,
    )
    for name in ("training", "calibration"):
        field = f"{name}_root_partition_sha256"
        if value.get(field) != partition_hashes.get(name):
            failures.append(f"{label}.{field}: excluded partition hash mismatch")
    return computed


def _validate_source(
    value: Any,
    *,
    claimed_sha256: Any,
    approved_sha256: str | None,
    failures: list[str],
) -> str | None:
    label = "source_manifest"
    if not isinstance(value, Mapping):
        failures.append(f"{label}: object required")
        return None
    try:
        computed = canonical_sha256(value)
    except (TypeError, ValueError, OverflowError):
        failures.append(f"{label}: finite canonical JSON required")
        return None
    if claimed_sha256 != computed:
        failures.append("source_manifest_sha256: manifest hash mismatch")
    if computed != approved_sha256:
        failures.append("source_manifest_sha256: not approved by locked config")
    if value.get("schema") != SOURCE_SCHEMA:
        failures.append(f"{label}.schema: unsupported")
    files = value.get("files")
    if not isinstance(files, list) or not files:
        failures.append(f"{label}.files: non-empty array required")
        return computed
    paths: list[str] = []
    for index, item in enumerate(files):
        item_label = f"{label}.files[{index}]"
        if not isinstance(item, Mapping):
            failures.append(f"{item_label}: object required")
            continue
        if set(item) != {"path", "sha256"}:
            failures.append(f"{item_label}: exact path/sha256 fields required")
        path = item.get("path")
        if not isinstance(path, str) or not path or path.startswith(("/", "\\")):
            failures.append(f"{item_label}.path: non-empty repository-relative path required")
        else:
            paths.append(path)
        if not _valid_sha256(item.get("sha256")):
            failures.append(f"{item_label}.sha256: lowercase SHA256 required")
    if len(paths) != len(set(paths)):
        failures.append(f"{label}.files: duplicate paths forbidden")
    if paths != sorted(paths):
        failures.append(f"{label}.files: canonical path order required")
    return computed


def _required_source_file_sha256(
    value: Any, *, path: str, failures: list[str]
) -> str | None:
    """Return one required source hash from an already validated manifest."""

    files = value.get("files") if isinstance(value, Mapping) else None
    matches = (
        [
            item
            for item in files
            if isinstance(item, Mapping) and item.get("path") == path
        ]
        if isinstance(files, list)
        else []
    )
    if len(matches) != 1:
        failures.append(f"source_manifest.files: exactly one {path!r} entry required")
        return None
    digest = matches[0].get("sha256")
    if not _valid_sha256(digest):
        failures.append(
            f"source_manifest.files[{path!r}].sha256: lowercase SHA256 required"
        )
        return None
    return str(digest)


def _validate_solver(
    value: Any,
    *,
    claimed_sha256: Any,
    approved_sha256: str | None,
    source_sha256: str | None,
    failures: list[str],
) -> str | None:
    label = "solver_manifest"
    if not isinstance(value, Mapping):
        failures.append(f"{label}: object required")
        return None
    try:
        computed = canonical_sha256(value)
    except (TypeError, ValueError, OverflowError):
        failures.append(f"{label}: finite canonical JSON required")
        return None
    if claimed_sha256 != computed:
        failures.append("solver_manifest_sha256: manifest hash mismatch")
    if computed != approved_sha256:
        failures.append("solver_manifest_sha256: not approved by locked config")
    expected = {
        "schema": SOLVER_SCHEMA,
        "method": SOLVER_METHOD,
        "adapter": SOLVER_ADAPTER,
        "ruleset": RULESET,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "turns": [3, 4],
        "actors": ["bb", "btn"],
        "physical_joker_ids": ["X1", "X2"],
        "information_model": INFORMATION_MODEL,
        "strategy_fusion": False,
        "full_card": True,
        "hu_exact": False,
        "exact_exploitability_computed": False,
    }
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            failures.append(f"{label}.{field}: must equal {wanted!r}")
    if value.get("source_manifest_sha256") != source_sha256:
        failures.append(f"{label}.source_manifest_sha256: source binding mismatch")
    return computed


def _validate_roots(
    value: Any,
    *,
    approved_sha256: str | None,
    partition_hashes: Mapping[str, Any],
    min_roots_per_stratum: int,
    failures: list[str],
) -> tuple[dict[str, Mapping[str, Any]], str | None]:
    label = "root_manifest"
    computed = _validate_self_hash(
        value, field="root_manifest_sha256", label=label, failures=failures
    )
    if not isinstance(value, Mapping):
        return {}, computed
    expected = {
        "schema": ROOT_MANIFEST_SCHEMA,
        "purpose": "independent_promotion_holdout",
        "locked_before_evaluation": True,
        "ruleset": RULESET,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "physical_joker_ids": ["X1", "X2"],
    }
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            failures.append(f"{label}.{field}: must equal {wanted!r}")
    if computed != approved_sha256:
        failures.append(f"{label}.root_manifest_sha256: not approved by locked config")
    if value.get("excluded_root_partition_sha256") != dict(partition_hashes):
        failures.append(
            f"{label}.excluded_root_partition_sha256: partition binding mismatch"
        )
    roots = value.get("roots")
    if not isinstance(roots, list) or not roots:
        failures.append(f"{label}.roots: non-empty array required")
        return {}, computed

    by_id: dict[str, Mapping[str, Any]] = {}
    commitments: set[str] = set()
    observations: set[str] = set()
    counts = {name: 0 for name in REQUIRED_STRATA}
    pairs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, root in enumerate(roots):
        item_label = f"{label}.roots[{index}]"
        if not isinstance(root, Mapping):
            failures.append(f"{item_label}: object required")
            continue
        root_id = root.get("root_id")
        if not isinstance(root_id, str) or not root_id:
            failures.append(f"{item_label}.root_id: non-empty string required")
            continue
        if root_id in by_id:
            failures.append(f"{item_label}.root_id: duplicate {root_id!r}")
        by_id[root_id] = root
        stratum = root.get("stratum")
        if stratum not in REQUIRED_STRATA:
            failures.append(f"{item_label}.stratum: unsupported")
            continue
        actor, joker_count = _expected_identity(str(stratum))
        if root.get("actor") != actor:
            failures.append(f"{item_label}.actor: must equal {actor!r}")
        if root.get("visible_joker_count") != joker_count:
            failures.append(
                f"{item_label}.visible_joker_count: must equal {joker_count}"
            )
        observation = root.get("observation_digest")
        if not _valid_sha256(observation):
            failures.append(f"{item_label}.observation_digest: invalid SHA256")
        elif observation in observations:
            failures.append(f"{item_label}.observation_digest: duplicate")
        else:
            observations.add(observation)
        pair_id = root.get("seat_swap_pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            failures.append(f"{item_label}.seat_swap_pair_id: non-empty string required")
        else:
            pairs[pair_id].append(root)
        commitment = root.get("root_commitment_sha256")
        identity_commitment = root.get("root_identity_commitment_sha256")
        try:
            expected_identity_commitment = root_identity_commitment_sha256(root_id)
        except (TypeError, ValueError):
            expected_identity_commitment = None
        if identity_commitment != expected_identity_commitment:
            failures.append(
                f"{item_label}.root_identity_commitment_sha256: root_id content mismatch"
            )
        try:
            expected_commitment = root_commitment_sha256(root)
        except (TypeError, ValueError, OverflowError):
            expected_commitment = None
        if commitment != expected_commitment or not _valid_sha256(commitment):
            failures.append(f"{item_label}.root_commitment_sha256: content mismatch")
        elif commitment in commitments:
            failures.append(f"{item_label}.root_commitment_sha256: duplicate")
        else:
            commitments.add(commitment)
        counts[str(stratum)] += 1

    for name, count in counts.items():
        if count < min_roots_per_stratum:
            failures.append(
                f"{label}: {name} needs >= {min_roots_per_stratum} locked roots, got {count}"
            )
    for pair_id, pair_roots in pairs.items():
        actors = {root.get("actor") for root in pair_roots}
        jokers = {root.get("visible_joker_count") for root in pair_roots}
        if len(pair_roots) != 2 or actors != {"bb", "btn"} or len(jokers) != 1:
            failures.append(
                f"{label}: seat_swap_pair_id {pair_id!r} must contain exactly "
                "one BB and one BTN root in the same Joker stratum"
            )
    return by_id, computed


def _validate_distribution(
    value: Any, *, label: str, failures: list[str]
) -> dict[str, float] | None:
    if not isinstance(value, Mapping) or not value:
        failures.append(f"{label}: non-empty action-probability object required")
        return None
    result: dict[str, float] = {}
    for action, probability in value.items():
        if not isinstance(action, str) or not action:
            failures.append(f"{label}: action IDs must be non-empty strings")
            continue
        if not _finite(probability) or float(probability) < 0:
            failures.append(f"{label}.{action}: finite nonnegative probability required")
            continue
        result[action] = float(probability)
    if not result or not math.isclose(
        math.fsum(result.values()), 1.0, rel_tol=0.0, abs_tol=_FLOAT_ABS_TOL
    ):
        failures.append(f"{label}: probabilities must sum to one")
    return result


def _validate_rows(
    rows_value: Any,
    *,
    roots_by_id: Mapping[str, Mapping[str, Any]],
    root_manifest_sha256: str | None,
    behavior_sha256: str | None,
    source_sha256: str | None,
    solver_sha256: str | None,
    checkpoint_manifests: Any,
    min_seeds: int,
    failures: list[str],
) -> list[Mapping[str, Any]]:
    if not isinstance(rows_value, list) or not rows_value:
        failures.append("raw_run_rows: non-empty array required")
        return []
    if not isinstance(checkpoint_manifests, Mapping) or not checkpoint_manifests:
        failures.append("checkpoint_manifests: non-empty object required")
        checkpoint_manifests = {}

    checkpoint_by_key: dict[tuple[str, int], str] = {}
    checkpoint_manifest_hashes: dict[str, str] = {}
    for checkpoint_sha, manifest in checkpoint_manifests.items():
        label = f"checkpoint_manifests.{checkpoint_sha}"
        if not _valid_sha256(checkpoint_sha):
            failures.append(f"{label}: key must be checkpoint SHA256")
        manifest_hash = _validate_self_hash(
            manifest,
            field="checkpoint_manifest_sha256",
            label=label,
            failures=failures,
        )
        if not isinstance(manifest, Mapping):
            continue
        expected = {
            "schema": CHECKPOINT_SCHEMA,
            "checkpoint_sha256": checkpoint_sha,
            "completed": True,
            "solver_manifest_sha256": solver_sha256,
            "source_manifest_sha256": source_sha256,
            "calibrated_behavior_sha256": behavior_sha256,
            "root_manifest_sha256": root_manifest_sha256,
            "exact_exploitability_computed": False,
        }
        for field, wanted in expected.items():
            if manifest.get(field) != wanted:
                failures.append(f"{label}.{field}: must equal {wanted!r}")
        stratum = manifest.get("stratum")
        seed = manifest.get("solver_seed")
        iterations = manifest.get("iterations")
        if stratum not in REQUIRED_STRATA:
            failures.append(f"{label}.stratum: unsupported")
        if not _is_int(seed) or seed < 0:
            failures.append(f"{label}.solver_seed: nonnegative integer required")
        if not _is_int(iterations) or iterations <= 0:
            failures.append(f"{label}.iterations: positive integer required")
        if stratum in REQUIRED_STRATA and _is_int(seed):
            key = (str(stratum), int(seed))
            if key in checkpoint_by_key:
                failures.append(f"{label}: duplicate checkpoint for {key!r}")
            checkpoint_by_key[key] = str(checkpoint_sha)
        if isinstance(checkpoint_sha, str) and isinstance(manifest_hash, str):
            checkpoint_manifest_hashes[checkpoint_sha] = manifest_hash

    valid_rows: list[Mapping[str, Any]] = []
    run_ids: set[str] = set()
    row_keys: set[tuple[str, int]] = set()
    referenced_checkpoints: set[str] = set()
    seeds_by_root: dict[str, set[int]] = defaultdict(set)
    payoff_seeds_by_root: dict[str, set[int]] = defaultdict(set)
    seed_sets_by_stratum_root: dict[str, dict[str, set[int]]] = defaultdict(dict)
    rows_by_pair_seed: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)

    for index, row in enumerate(rows_value):
        label = f"raw_run_rows[{index}]"
        if not isinstance(row, Mapping):
            failures.append(f"{label}: object required")
            continue
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            failures.append(f"{label}.run_id: non-empty string required")
        elif run_id in run_ids:
            failures.append(f"{label}.run_id: duplicate")
        else:
            run_ids.add(run_id)
        root_id = row.get("root_id")
        root = roots_by_id.get(root_id) if isinstance(root_id, str) else None
        if root is None:
            failures.append(f"{label}.root_id: not in locked root manifest")
            continue
        for field in (
            "root_commitment_sha256",
            "root_identity_commitment_sha256",
            "stratum",
            "actor",
            "visible_joker_count",
            "seat_swap_pair_id",
        ):
            if row.get(field) != root.get(field):
                failures.append(f"{label}.{field}: locked root binding mismatch")
        stratum = str(root["stratum"])
        solver_seed = row.get("solver_seed")
        payoff_seed = row.get("payoff_sample_seed")
        if not _is_int(solver_seed) or solver_seed < 0:
            failures.append(f"{label}.solver_seed: nonnegative integer required")
            continue
        if not _is_int(payoff_seed) or payoff_seed < 0:
            failures.append(f"{label}.payoff_sample_seed: nonnegative integer required")
            continue
        if solver_seed == payoff_seed:
            failures.append(f"{label}: solver/payoff seeds must use independent values")
        row_key = (str(root_id), int(solver_seed))
        if row_key in row_keys:
            failures.append(f"{label}: duplicate root/solver_seed row")
        row_keys.add(row_key)
        seeds_by_root[str(root_id)].add(int(solver_seed))
        payoff_seeds_by_root[str(root_id)].add(int(payoff_seed))
        rows_by_pair_seed[(str(root["seat_swap_pair_id"]), int(solver_seed))].append(row)

        expected_hashes = {
            "calibrated_behavior_sha256": behavior_sha256,
            # The paired noninferiority baseline is the approved calibrated
            # behavior policy, never an unbound row-local reference.
            "reference_policy_sha256": behavior_sha256,
            "source_manifest_sha256": source_sha256,
            "solver_manifest_sha256": solver_sha256,
            "root_manifest_sha256": root_manifest_sha256,
        }
        for field, wanted in expected_hashes.items():
            if row.get(field) != wanted:
                failures.append(f"{label}.{field}: provenance binding mismatch")
        checkpoint_sha = row.get("checkpoint_sha256")
        if checkpoint_sha != checkpoint_by_key.get((stratum, int(solver_seed))):
            failures.append(f"{label}.checkpoint_sha256: stratum/seed binding mismatch")
        elif isinstance(checkpoint_sha, str):
            referenced_checkpoints.add(checkpoint_sha)
        if row.get("checkpoint_manifest_sha256") != checkpoint_manifest_hashes.get(
            str(checkpoint_sha)
        ):
            failures.append(f"{label}.checkpoint_manifest_sha256: manifest mismatch")
        if row.get("exact_exploitability_computed") is not False:
            failures.append(f"{label}.exact_exploitability_computed: must be false")

        audits = row.get("audits")
        if not isinstance(audits, Mapping) or set(audits) != set(REQUIRED_AUDITS):
            failures.append(f"{label}.audits: exact zero-failure audit field set required")
        else:
            for field in REQUIRED_AUDITS:
                if not _is_int(audits.get(field)) or audits.get(field) != 0:
                    failures.append(f"{label}.audits.{field}: must equal integer zero")

        eligible = row.get("eligible_infoset_digests")
        encountered = row.get("encountered_infoset_digests")
        if not isinstance(eligible, list) or not eligible:
            failures.append(f"{label}.eligible_infoset_digests: non-empty array required")
            eligible = []
        if not isinstance(encountered, list) or not encountered:
            failures.append(f"{label}.encountered_infoset_digests: non-empty array required")
            encountered = []
        for field, values in (
            ("eligible_infoset_digests", eligible),
            ("encountered_infoset_digests", encountered),
        ):
            if any(not _valid_sha256(value) for value in values):
                failures.append(f"{label}.{field}: every entry must be SHA256")
            if len(values) != len(set(values)):
                failures.append(f"{label}.{field}: duplicates forbidden")
            if values != sorted(values):
                failures.append(f"{label}.{field}: canonical sorted order required")
        if not set(encountered).issubset(set(eligible)):
            failures.append(f"{label}: encountered infosets must be eligible")
        expected_eligible_count = len(eligible)
        expected_encountered_count = len(encountered)
        expected_coverage = (
            expected_encountered_count / expected_eligible_count
            if expected_eligible_count
            else 0.0
        )
        if row.get("eligible_infoset_count") != expected_eligible_count:
            failures.append(f"{label}.eligible_infoset_count: raw-derived mismatch")
        if row.get("encountered_infoset_count") != expected_encountered_count:
            failures.append(f"{label}.encountered_infoset_count: raw-derived mismatch")
        published_coverage = row.get("encountered_infoset_coverage")
        if not _finite(published_coverage) or not math.isclose(
            float(published_coverage),
            expected_coverage,
            rel_tol=0.0,
            abs_tol=_FLOAT_ABS_TOL,
        ):
            failures.append(f"{label}.encountered_infoset_coverage: raw-derived mismatch")

        policy = _validate_distribution(
            row.get("root_action_distribution"),
            label=f"{label}.root_action_distribution",
            failures=failures,
        )
        reference = _validate_distribution(
            row.get("reference_action_distribution"),
            label=f"{label}.reference_action_distribution",
            failures=failures,
        )
        action_payoffs = row.get("action_payoff_estimates")
        if not isinstance(action_payoffs, Mapping) or not action_payoffs:
            failures.append(f"{label}.action_payoff_estimates: non-empty object required")
            action_payoffs = {}
        elif any(
            not isinstance(action, str) or not action or not _finite(value)
            for action, value in action_payoffs.items()
        ):
            failures.append(
                f"{label}.action_payoff_estimates: valid action IDs and finite values required"
            )
        if policy is not None and reference is not None:
            if set(policy) != set(reference) or set(policy) != set(action_payoffs):
                failures.append(f"{label}: policy/reference/payoff action sets must match")
            else:
                policy_payoff = _weighted_payoff(policy, action_payoffs)
                reference_payoff = _weighted_payoff(reference, action_payoffs)
                best_payoff = max(float(value) for value in action_payoffs.values())
                regret = max(0.0, best_payoff - policy_payoff)
                expected_values = {
                    "policy_payoff_estimate": policy_payoff,
                    "reference_payoff_estimate": reference_payoff,
                    "best_action_payoff_estimate": best_payoff,
                    "ev_regret_estimate": regret,
                }
                for field, wanted in expected_values.items():
                    actual = row.get(field)
                    if not _finite(actual) or not math.isclose(
                        float(actual), wanted, rel_tol=0.0, abs_tol=_FLOAT_ABS_TOL
                    ):
                        failures.append(f"{label}.{field}: raw-derived mismatch")
        runtime = row.get("runtime_ms")
        if not _finite(runtime) or float(runtime) <= 0:
            failures.append(f"{label}.runtime_ms: finite positive value required")
        valid_rows.append(row)

    if set(checkpoint_manifests) != referenced_checkpoints:
        failures.append("checkpoint_manifests: exact referenced checkpoint set required")
    for root_id in roots_by_id:
        seeds = seeds_by_root.get(root_id, set())
        payoff_seeds = payoff_seeds_by_root.get(root_id, set())
        if len(seeds) < min_seeds:
            failures.append(
                f"raw_run_rows: root {root_id!r} needs >= {min_seeds} independent solver seeds"
            )
        if len(payoff_seeds) != len(seeds):
            failures.append(
                f"raw_run_rows: root {root_id!r} must use distinct payoff seeds per solver seed"
            )
        stratum = str(roots_by_id[root_id]["stratum"])
        seed_sets_by_stratum_root[stratum][root_id] = seeds
    for stratum, root_sets in seed_sets_by_stratum_root.items():
        unique_sets = {tuple(sorted(values)) for values in root_sets.values()}
        if len(unique_sets) != 1:
            failures.append(f"raw_run_rows: {stratum} roots must share one solver-seed set")
    for key, pair_rows in rows_by_pair_seed.items():
        actors = {row.get("actor") for row in pair_rows}
        payoff_seeds = {row.get("payoff_sample_seed") for row in pair_rows}
        jokers = {row.get("visible_joker_count") for row in pair_rows}
        if (
            len(pair_rows) != 2
            or actors != {"bb", "btn"}
            or len(payoff_seeds) != 1
            or len(jokers) != 1
        ):
            failures.append(
                f"raw_run_rows: paired seat-swap {key!r} requires BB/BTN rows "
                "with the same Joker layer and common payoff seed"
            )
    return valid_rows


def _compare_thresholds(
    metrics: Mapping[str, Any], config: Mapping[str, Any], failures: list[str]
) -> None:
    thresholds = config["thresholds"]
    for name in REQUIRED_STRATA:
        values = metrics["strata"][name]
        checks = (
            (
                "root_count",
                float(values["root_count"])
                >= float(thresholds["min_roots_per_stratum"]),
            ),
            (
                "min_independent_seeds_per_root",
                float(values["min_independent_seeds_per_root"])
                >= float(thresholds["min_independent_seeds_per_stratum"]),
            ),
            (
                "min_encountered_infoset_coverage",
                float(values["min_encountered_infoset_coverage"])
                >= float(thresholds["min_encountered_infoset_coverage"]),
            ),
            (
                "max_policy_tv_drift",
                float(values["max_policy_tv_drift"])
                <= float(thresholds["max_policy_tv_drift"]),
            ),
            (
                "mean_ev_regret_score",
                float(values["mean_ev_regret_score"])
                <= float(thresholds["max_mean_ev_regret_score"]),
            ),
            (
                "p95_ev_regret_score",
                float(values["p95_ev_regret_score"])
                <= float(thresholds["max_p95_ev_regret_score"]),
            ),
            (
                "p99_ev_regret_score",
                float(values["p99_ev_regret_score"])
                <= float(thresholds["max_p99_ev_regret_score"]),
            ),
        )
        for field, passed in checks:
            if not passed:
                failures.append(f"derived_metrics.strata.{name}.{field}: threshold failed")

    global_values = metrics["global"]
    for field in (
        "holdout_excluded_root_overlap_count",
        "excluded_partition_pairwise_overlap_count",
        *REQUIRED_AUDITS,
    ):
        if global_values[field] != 0:
            failures.append(f"derived_metrics.global.{field}: must equal zero")
    global_checks = (
        (
            "min_encountered_infoset_coverage",
            float(global_values["min_encountered_infoset_coverage"])
            >= float(thresholds["min_encountered_infoset_coverage"]),
        ),
        (
            "max_policy_tv_drift",
            float(global_values["max_policy_tv_drift"])
            <= float(thresholds["max_policy_tv_drift"]),
        ),
        (
            "mean_ev_regret_score",
            float(global_values["mean_ev_regret_score"])
            <= float(thresholds["max_mean_ev_regret_score"]),
        ),
        (
            "p95_ev_regret_score",
            float(global_values["p95_ev_regret_score"])
            <= float(thresholds["max_p95_ev_regret_score"]),
        ),
        (
            "p99_ev_regret_score",
            float(global_values["p99_ev_regret_score"])
            <= float(thresholds["max_p99_ev_regret_score"]),
        ),
        (
            "paired_seat_swap_min_single_seat_delta_score",
            float(global_values["paired_seat_swap_min_single_seat_delta_score"])
            >= -float(thresholds["paired_seat_swap_noninferiority_margin_score"]),
        ),
        (
            "paired_seat_swap_min_pair_mean_delta_score",
            float(global_values["paired_seat_swap_min_pair_mean_delta_score"])
            >= -float(thresholds["paired_seat_swap_noninferiority_margin_score"]),
        ),
        (
            "runtime_ms_p95",
            float(global_values["runtime_ms_p95"])
            <= float(thresholds["max_runtime_ms_p95"]),
        ),
        (
            "runtime_ms_max",
            float(global_values["runtime_ms_max"])
            <= float(thresholds["max_runtime_ms_max"]),
        ),
        (
            "runtime_ms_total",
            float(global_values["runtime_ms_total"])
            <= float(thresholds["max_runtime_ms_total"]),
        ),
    )
    for field, passed in global_checks:
        if not passed:
            failures.append(f"derived_metrics.global.{field}: threshold failed")


def validate_promotion_evidence_m3_full_card_strength(
    evidence: Any, *, config: Any
) -> dict[str, Any]:
    """Validate a locked six-stratum, root-disjoint M3 strength artifact."""

    failures: list[str] = []
    normalized_config, config_sha256 = _config_snapshot(config, failures)
    try:
        evidence_sha256 = canonical_sha256(evidence)
    except (TypeError, ValueError, OverflowError):
        evidence_sha256 = None
        failures.append("evidence: must be finite canonical JSON")

    if not isinstance(evidence, Mapping):
        failures.append("evidence: must be an object")
        evidence = {}
    schema_text = str(evidence.get("schema", ""))
    scope_text = str(evidence.get("scope", ""))
    kind_text = str(evidence.get("evidence_kind", ""))
    if "smoke" in schema_text.lower() or "smoke" in scope_text.lower() or "smoke" in kind_text.lower():
        failures.append("evidence: execution smoke artifacts are ineligible for promotion")
    if evidence.get("schema") != EVIDENCE_SCHEMA:
        failures.append(f"evidence.schema: must equal {EVIDENCE_SCHEMA!r}")
    if evidence.get("gate_id") != GATE_ID:
        failures.append(f"evidence.gate_id: must equal {GATE_ID!r}")
    if evidence.get("scope") != SCOPE:
        failures.append(f"evidence.scope: must equal {SCOPE!r}")
    if evidence.get("evidence_kind") != EVIDENCE_KIND:
        failures.append(f"evidence.evidence_kind: must equal {EVIDENCE_KIND!r}")
    if evidence.get("exact_exploitability_computed") is not False:
        failures.append("evidence.exact_exploitability_computed: must be false")
    if evidence.get("gate_config_sha256") != config_sha256:
        failures.append("evidence.gate_config_sha256: locked config hash mismatch")
    _validate_self_hash(
        evidence,
        field="artifact_sha256",
        label="evidence",
        failures=failures,
    )

    approved = normalized_config or {}
    raw_partitions = evidence.get("excluded_root_partitions")
    if not isinstance(raw_partitions, Mapping):
        failures.append("excluded_root_partitions: object required")
        raw_partitions = {}
    if set(raw_partitions) != set(REQUIRED_EXCLUDED_PARTITIONS):
        failures.append(
            "excluded_root_partitions: exact training/calibration/smoke set required"
        )
    approved_partitions = approved.get(
        "approved_excluded_root_partition_sha256", {}
    )
    partition_sets: dict[str, set[str]] = {}
    partition_hashes: dict[str, Any] = {}
    for name in REQUIRED_EXCLUDED_PARTITIONS:
        value = raw_partitions.get(name)
        partition_sets[name] = _validate_partition(
            name,
            value,
            approved_sha256=approved_partitions.get(name),
            failures=failures,
        )
        partition_hashes[name] = (
            value.get("manifest_sha256") if isinstance(value, Mapping) else None
        )

    source_sha256 = _validate_source(
        evidence.get("source_manifest"),
        claimed_sha256=evidence.get("source_manifest_sha256"),
        approved_sha256=approved.get("approved_source_manifest_sha256"),
        failures=failures,
    )
    solver_sha256 = _validate_solver(
        evidence.get("solver_manifest"),
        claimed_sha256=evidence.get("solver_manifest_sha256"),
        approved_sha256=approved.get("approved_solver_manifest_sha256"),
        source_sha256=source_sha256,
        failures=failures,
    )
    range_builder_source_sha256 = _required_source_file_sha256(
        evidence.get("source_manifest"),
        path=T3_FULL_CARD_RANGE_SOURCE_PATH,
        failures=failures,
    )
    behavior_sha256 = _validate_behavior(
        evidence.get("calibrated_behavior_manifest"),
        claimed_sha256=evidence.get("calibrated_behavior_sha256"),
        approved_sha256=approved.get("approved_calibrated_behavior_sha256"),
        partition_hashes=partition_hashes,
        solver_manifest_sha256=solver_sha256,
        range_builder_source_sha256=range_builder_source_sha256,
        failures=failures,
    )
    min_roots = int(approved.get("thresholds", {}).get("min_roots_per_stratum", 1))
    roots_by_id, root_manifest_sha256 = _validate_roots(
        evidence.get("root_manifest"),
        approved_sha256=approved.get("approved_holdout_root_manifest_sha256"),
        partition_hashes=partition_hashes,
        min_roots_per_stratum=min_roots,
        failures=failures,
    )
    min_seeds = int(
        approved.get("thresholds", {}).get(
            "min_independent_seeds_per_stratum", 2
        )
    )
    valid_rows = _validate_rows(
        evidence.get("raw_run_rows"),
        roots_by_id=roots_by_id,
        root_manifest_sha256=root_manifest_sha256,
        behavior_sha256=behavior_sha256,
        source_sha256=source_sha256,
        solver_sha256=solver_sha256,
        checkpoint_manifests=evidence.get("checkpoint_manifests"),
        min_seeds=min_seeds,
        failures=failures,
    )

    # Root overlap is checked independently of the published aggregate.
    holdout = {
        str(root.get("root_identity_commitment_sha256"))
        for root in roots_by_id.values()
    }
    for name, commitments in partition_sets.items():
        overlap = holdout & commitments
        if overlap:
            failures.append(
                f"root_manifest: {len(overlap)} holdout roots overlap {name} partition"
            )
    for left, right in itertools.combinations(REQUIRED_EXCLUDED_PARTITIONS, 2):
        overlap = partition_sets[left] & partition_sets[right]
        if overlap:
            failures.append(
                f"excluded_root_partitions: {left}/{right} overlap count {len(overlap)}"
            )

    derived_metrics: dict[str, Any] = {}
    if roots_by_id and valid_rows and all(
        isinstance(evidence.get(field), Mapping)
        for field in ("root_manifest", "checkpoint_manifests")
    ):
        try:
            derived_metrics = derive_strength_metrics(evidence)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            failures.append(
                f"derived_metrics: cannot derive from raw rows ({type(exc).__name__}: {exc})"
            )
        else:
            if evidence.get("published_summary") != derived_metrics:
                failures.append(
                    "published_summary: must exactly equal raw-derived strength metrics"
                )
            if normalized_config is not None:
                _compare_thresholds(derived_metrics, normalized_config, failures)

    passed = normalized_config is not None and not failures
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "gate_id": GATE_ID,
        "passed": passed,
        "status": PASS_STATUS if passed else FAIL_STATUS,
        "m3_full_card_strength_promoted": passed,
        "full_card_policy_promoted": passed,
        "exact_exploitability_computed": False,
        "gate_config_sha256": config_sha256,
        "evidence_sha256": evidence_sha256,
        "derived_metrics": derived_metrics,
        "failures": failures,
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


validate_m3_full_card_strength_evidence = (
    validate_promotion_evidence_m3_full_card_strength
)


__all__ = [
    "BEHAVIOR_SCHEMA",
    "BEHAVIOR_MODEL_TYPE",
    "CALIBRATION_BINDING_KEYS",
    "CALIBRATION_ROLE_KEYS",
    "CHECKPOINT_SCHEMA",
    "CONFIG_SCHEMA",
    "EVIDENCE_KIND",
    "EVIDENCE_SCHEMA",
    "GATE_ID",
    "INFORMATION_MODEL",
    "POSITION_CONTRACT_VERSION",
    "REQUIRED_EXCLUDED_PARTITIONS",
    "REQUIRED_STRATA",
    "RESULT_SCHEMA",
    "ROOT_MANIFEST_SCHEMA",
    "ROOT_IDENTITY_COMMITMENT_SCHEMA",
    "ROOT_PARTITION_SCHEMA",
    "RULESET",
    "SCOPE",
    "SOLVER_ADAPTER",
    "SOLVER_METHOD",
    "SOLVER_SCHEMA",
    "SOURCE_SCHEMA",
    "T3_BB_LIKELIHOOD_KEYS",
    "T3_BB_LIKELIHOOD_METHOD",
    "T3_BB_LIKELIHOOD_SCHEMA",
    "T3_FULL_CARD_RANGE_SOURCE_PATH",
    "canonical_json",
    "canonical_sha256",
    "derive_strength_metrics",
    "root_commitment_sha256",
    "root_identity_commitment_sha256",
    "self_hash",
    "verify_t3_bb_likelihood_binding",
    "validate_m3_full_card_strength_evidence",
    "validate_promotion_evidence_m3_full_card_strength",
]
