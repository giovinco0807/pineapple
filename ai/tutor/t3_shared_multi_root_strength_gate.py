"""Independent-holdout gate for shared-infoset multi-root T3 strategies.

The historical M3 strength gate correctly re-derives practical-strength
statistics from locked holdout rows, but its checkpoint contract predates the
shared-infoset chance-super-root solver.  A ``strategy_fusion == false`` flag
alone cannot prove that one public policy table was shared by every private
root, nor does it bind a holdout row to the serialized candidate policy.

This module adds that narrow missing boundary without importing the mutable
multi-root solver implementation.  ``build_shared_multi_root_strategy_artifact``
adapts any result object exposing the public result attributes into a strict,
JSON-compatible artifact.  The verifier then checks the serialized public
strategy, exact root prior, shared-table topology, physical-range provenance,
and sampling audit from content rather than from a Python type identity.

The promotion gate locks exactly six Joker-layer/role artifacts, requires the
production T3-BB fixed-point likelihood binding, and fresh-derives practical
strength from a frozen candidate on independent roots and payoff samples.
Candidate training roots, holdout roots, calibration/smoke roots, training
seeds, evaluation seeds, and payoff seeds are all checked for forbidden reuse.

This is an algorithm-validation gate only.  A tabular profile must already
contain the exact evaluated information set, so these rows cannot constitute
a genuinely unseen production holdout.  It never computes or claims exact
full-card exploitability and it never authorizes production promotion, even
when every locked validation threshold passes.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import re
from collections import defaultdict
from fractions import Fraction
from typing import Any, Mapping, Sequence

from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import Board
from ai.tutor.exact_late import action_key
from ai.tutor.promotion_gate_m3_full_card_strength import (
    POSITION_CONTRACT_VERSION,
    REQUIRED_AUDITS,
    REQUIRED_EXCLUDED_PARTITIONS,
    REQUIRED_STRATA,
    ROOT_MANIFEST_SCHEMA,
    ROOT_PARTITION_SCHEMA,
    RULESET,
    canonical_json,
    canonical_sha256,
    root_commitment_sha256,
    root_identity_commitment_sha256,
    self_hash,
    verify_t3_bb_likelihood_binding,
)
from ai.tutor.t3_bb_fixed_point_gate_v2 import reconstruct_infoset_key


STRATEGY_ARTIFACT_SCHEMA = "ofc_t3_shared_multi_root_strategy/v2"
CANDIDATE_BUNDLE_SCHEMA = "ofc_t3_shared_multi_root_candidate_bundle/v2"
EVALUATOR_MANIFEST_SCHEMA = "ofc_t3_shared_multi_root_holdout_evaluator/v2"
CONFIG_SCHEMA = "ofc_t3_shared_multi_root_strength_gate_config/v2"
EVIDENCE_SCHEMA = "ofc_t3_shared_multi_root_strength_evidence/v2"
RESULT_SCHEMA = "ofc_t3_shared_multi_root_strength_gate_result/v2"
GATE_ID = "promotion_gate_t3_shared_multi_root_strength"
SCOPE = "shared_infoset_multi_root_t3_algorithm_validation_only"
EVIDENCE_KIND = "locked_frozen_candidate_algorithm_validation"
ALGORITHM_VALIDATION_ONLY_STATUS = (
    "t3_shared_multi_root_algorithm_validation_only_nonpromoting"
)
ALGORITHM_VALIDATION_ONLY_FAILURE = (
    "tabular exact-infoset evaluation is algorithm-validation only; "
    "production promotion is unsupported"
)
FAIL_STATUS = "t3_shared_multi_root_strength_blocked"

MULTI_ROOT_SOLVER_METHOD = "full_card_shared_infoset_multi_root_mccfr_plus_v1"
SUPER_ROOT_SAMPLING_CONTRACT = "exact_fraction_chance_super_root_v1"
POLICY_IDENTITY_CONTRACT = "infoset_key_plus_lexical_action_key_v1"
PUBLIC_STRATEGY_SCHEMA = "ofc_full_card_public_strategy/v1"
HOLDOUT_EVALUATOR_METHOD = "frozen_public_policy_independent_action_payoff_v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FLOAT_TOL = 1e-12

_STRATEGY_ARTIFACT_KEYS = frozenset(
    {
        "schema",
        "artifact_kind",
        "stratum",
        "actor",
        "visible_joker_count",
        "solver_method",
        "policy_identity_contract",
        "source_manifest_sha256",
        "range_builder_source_sha256",
        "t3_bb_likelihood_binding_sha256",
        "algorithm_validation_only",
        "root_scope_binding_contract",
        "global_policy_claim",
        "unseen_root_generalization_claim",
        "promotion_eligible",
        "exact_exploitability_computed",
        "strategy_profile",
        "average_strategy_sha256",
        "root_prior_manifest",
        "root_prior_manifest_sha256",
        "training_roots",
        "producer_contract",
        "sampling_audit",
        "artifact_sha256",
    }
)
_PRODUCER_CONTRACT_KEYS = frozenset(
    {
        "method",
        "chance_super_root",
        "super_root_sampling_contract",
        "root_prior_normalized_exact",
        "root_count",
        "traverser_schedule",
        "alternating_updates",
        "regret_matching_plus",
        "regret_clip_scope",
        "linear_averaging",
        "root_prior_sampled_once_per_traversal",
        "conditional_posterior_sampled_once_per_traversal",
        "root_probability_multiplied_after_sampling",
        "posterior_probability_multiplied_after_sampling",
        "chance_probability_multiplied_after_sampling",
        "joint_particle_weight_used_after_sampling",
        "policy_identity_contract",
        "policy_table_shared_across_all_roots",
        "table_key_type",
        "table_key_contains_root_id",
        "table_key_contains_private_type_id",
        "table_key_contains_particle_commitment",
        "table_key_contains_remaining_cards",
        "strategy_serialization_contains_root_id",
        "strategy_serialization_contains_hidden_particle",
        "strategy_fusion",
        "independent_per_root_solve",
        "shared_across_roots_infoset_count",
        "compatible_full_card_adapters",
        "full_card",
        "full_card_policy_promoted",
        "promotion_eligible",
        "runtime_integrated",
        "hu_exact",
        "exact_exploitability_computed",
        "iterations",
        "traversals",
        "seed",
        "max_infosets",
        "position_contract_version",
    }
)
_SAMPLING_AUDIT_KEYS = frozenset(
    {
        "traversals",
        "traversals_by_actor",
        "super_root_samples",
        "conditional_root_posterior_samples",
        "root_samples_by_opaque_id",
        "distinct_root_adapters_sampled",
        "infosets_created",
    }
)
_BUNDLE_KEYS = frozenset(
    {
        "schema",
        "artifact_kind",
        "ruleset",
        "position_contract_version",
        "physical_joker_ids",
        "locked_before_holdout",
        "holdout_used_for_candidate_selection",
        "algorithm_validation_only",
        "production_promotion_supported",
        "unseen_root_strength_supported",
        "promotion_eligible",
        "exact_exploitability_computed",
        "t3_bb_likelihood_binding",
        "candidate_scope_manifest",
        "candidate_scope_manifest_sha256",
        "strategies",
        "candidate_bundle_sha256",
    }
)
_ROOT_KEYS = frozenset(
    {
        "root_id",
        "stratum",
        "actor",
        "visible_joker_count",
        "observation_digest",
        "seat_swap_pair_id",
        "root_identity_commitment_sha256",
        "root_commitment_sha256",
    }
)
_ROOT_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "purpose",
        "locked_before_evaluation",
        "ruleset",
        "position_contract_version",
        "physical_joker_ids",
        "excluded_root_partition_sha256",
        "roots",
        "root_manifest_sha256",
    }
)
_PARTITION_KEYS = frozenset(
    {"schema", "purpose", "root_commitments", "manifest_sha256"}
)
_EVALUATOR_KEYS = frozenset(
    {
        "schema",
        "method",
        "ruleset",
        "position_contract_version",
        "candidate_bundle_sha256",
        "root_manifest_sha256",
        "reference_policy_sha256",
        "source_manifest_sha256",
        "candidate_policy_frozen",
        "strategy_updates_during_evaluation",
        "holdout_used_for_candidate_selection",
        "action_payoff_samples_independent_of_training",
        "exact_exploitability_computed",
        "manifest_sha256",
    }
)
_THRESHOLD_KEYS = frozenset(
    {
        "min_independent_evaluation_seeds_per_root",
        "min_roots_per_stratum",
        "min_action_payoff_samples_per_action",
        "min_encountered_infoset_coverage",
        "max_action_payoff_standard_error",
        "max_reference_action_payoff_standard_error",
        "max_policy_reference_delta_standard_error",
        "max_mean_ev_regret_score",
        "max_p95_ev_regret_score",
        "max_p99_ev_regret_score",
        "paired_seat_swap_noninferiority_margin_score",
        "max_runtime_ms_p95",
        "max_runtime_ms_max",
        "max_runtime_ms_total",
    }
)
_CONFIG_KEYS = frozenset(
    {
        "schema",
        "gate_id",
        "approved_candidate_bundle_sha256",
        "approved_holdout_root_manifest_sha256",
        "approved_evaluator_manifest_sha256",
        "approved_excluded_root_partition_sha256",
        "thresholds",
        "config_sha256",
    }
)
_ROW_KEYS = frozenset(
    {
        "run_id",
        "root_id",
        "root_commitment_sha256",
        "root_identity_commitment_sha256",
        "stratum",
        "actor",
        "visible_joker_count",
        "seat_swap_pair_id",
        "evaluation_seed",
        "payoff_sample_seed",
        "candidate_bundle_sha256",
        "candidate_strategy_artifact_sha256",
        "candidate_strategy_sha256",
        "policy_infoset_digest",
        "range_content_sha256",
        "range_build_sha256",
        "policy_action_distribution",
        "reference_policy_sha256",
        "reference_action_distribution",
        "action_payoff_estimates",
        "action_payoff_sample_counts",
        "action_payoff_aggregates",
        "action_payoff_standard_errors",
        "max_action_payoff_standard_error",
        "reference_action_payoff_estimates",
        "reference_action_payoff_sample_counts",
        "reference_action_payoff_aggregates",
        "reference_action_payoff_standard_errors",
        "max_reference_action_payoff_standard_error",
        "policy_payoff_estimate",
        "candidate_continuation_uniform_root_payoff_estimate",
        "reference_payoff_estimate",
        "policy_reference_paired_aggregate",
        "policy_reference_delta_estimate",
        "policy_reference_delta_standard_error",
        "best_action_payoff_estimate",
        "ev_regret_estimate",
        "eligible_infoset_digests",
        "encountered_infoset_digests",
        "eligible_infoset_count",
        "encountered_infoset_count",
        "encountered_infoset_coverage",
        "audits",
        "runtime_ms",
        "exact_exploitability_computed",
        "row_sha256",
    }
)
_EVIDENCE_KEYS = frozenset(
    {
        "schema",
        "gate_id",
        "scope",
        "evidence_kind",
        "production_promotion_claim",
        "exact_exploitability_computed",
        "gate_config_sha256",
        "candidate_bundle",
        "candidate_bundle_sha256",
        "root_manifest",
        "excluded_root_partitions",
        "evaluator_manifest",
        "evaluator_manifest_sha256",
        "raw_holdout_rows",
        "published_summary",
        "artifact_sha256",
    }
)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _require_sha256(value: Any, *, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label}: lowercase SHA256 required")
    return str(value)


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label}: object required")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    if set(value) != set(expected):
        raise ValueError(
            f"{label}: exact fields required; "
            f"missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _require_int(value: Any, *, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label}: integer required")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label}: must be >= {minimum}")
    return value


def _require_finite(value: Any, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label}: finite number required")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{label}: {qualifier} number required")
    return result


def _json_copy(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _expected_identity(stratum: str) -> tuple[str, int]:
    if stratum not in REQUIRED_STRATA:
        raise ValueError(f"unsupported stratum {stratum!r}")
    actor, joker = stratum.split("_joker", 1)
    return actor, int(joker)


def _fraction_text(value: Any, *, label: str) -> Fraction:
    if not isinstance(value, str):
        raise TypeError(f"{label}: reduced positive rational string required")
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{label}: reduced positive rational required") from exc
    if parsed <= 0 or value != f"{parsed.numerator}/{parsed.denominator}":
        raise ValueError(f"{label}: reduced positive rational required")
    return parsed


def _parse_canonical_json(value: str, *, label: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise TypeError(f"{label}: canonical JSON string required")
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"{label}: valid JSON required") from exc
    if not isinstance(parsed, dict) or canonical_json(parsed) != value:
        raise ValueError(f"{label}: canonical JSON object required")
    return parsed


def _legal_action_ids(key: Any) -> tuple[str, ...]:
    rows = key.board_bb if key.actor == "bb" else key.board_btn
    board = Board(top=list(rows[0]), middle=list(rows[1]), bottom=list(rows[2]))
    return tuple(
        sorted(
            action_key(action)
            for action in get_turn_actions(list(key.current_draw), board)
        )
    )


def verify_public_strategy_profile(
    value: Any,
) -> tuple[dict[str, Any], dict[str, dict[str, float]], dict[str, Any]]:
    """Verify a serialized public-only profile and return rows by digest."""

    raw = _require_mapping(value, label="strategy_profile")
    _require_exact_keys(
        raw,
        frozenset({"schema", "policy_identity_contract", "records"}),
        label="strategy_profile",
    )
    if raw.get("schema") != PUBLIC_STRATEGY_SCHEMA:
        raise ValueError("strategy_profile.schema: mismatch")
    if raw.get("policy_identity_contract") != POLICY_IDENTITY_CONTRACT:
        raise ValueError("strategy_profile.policy_identity_contract: mismatch")
    records = raw.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("strategy_profile.records: non-empty list required")

    distributions: dict[str, dict[str, float]] = {}
    infosets: dict[str, Any] = {}
    order: list[str] = []
    for index, record_value in enumerate(records):
        label = f"strategy_profile.records[{index}]"
        record = _require_mapping(record_value, label=label)
        _require_exact_keys(
            record,
            frozenset({"infoset_digest", "infoset", "actions"}),
            label=label,
        )
        digest = _require_sha256(
            record.get("infoset_digest"), label=f"{label}.infoset_digest"
        )
        if digest in distributions:
            raise ValueError(f"{label}.infoset_digest: duplicate")
        key = reconstruct_infoset_key(record.get("infoset"), label=f"{label}.infoset")
        if key.digest() != digest:
            raise ValueError(f"{label}.infoset_digest: content mismatch")
        actions = record.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError(f"{label}.actions: non-empty list required")
        row: dict[str, float] = {}
        action_order: list[str] = []
        for action_index, action_value in enumerate(actions):
            action_label = f"{label}.actions[{action_index}]"
            action = _require_mapping(action_value, label=action_label)
            _require_exact_keys(
                action,
                frozenset({"action_id", "probability"}),
                label=action_label,
            )
            action_id = action.get("action_id")
            if not isinstance(action_id, str) or not action_id:
                raise ValueError(f"{action_label}.action_id: non-empty string required")
            if action_id in row:
                raise ValueError(f"{action_label}.action_id: duplicate")
            probability = _require_finite(
                action.get("probability"), label=f"{action_label}.probability"
            )
            if probability < 0.0:
                raise ValueError(f"{action_label}.probability: nonnegative required")
            row[action_id] = probability
            action_order.append(action_id)
        if action_order != sorted(action_order):
            raise ValueError(f"{label}.actions: canonical action order required")
        if tuple(action_order) != _legal_action_ids(key):
            raise ValueError(f"{label}.actions: exact legal action set required")
        if not math.isclose(
            math.fsum(row.values()), 1.0, rel_tol=0.0, abs_tol=_FLOAT_TOL
        ):
            raise ValueError(f"{label}.actions: probabilities must sum to one")
        distributions[digest] = row
        infosets[digest] = key
        order.append(digest)
    if order != sorted(order):
        raise ValueError("strategy_profile.records: canonical digest order required")
    return copy.deepcopy(dict(raw)), distributions, infosets


def _verify_root_prior(value: Any) -> tuple[dict[str, Any], tuple[str, ...]]:
    raw = _require_mapping(value, label="root_prior_manifest")
    _require_exact_keys(
        raw,
        frozenset({"schema", "sampling_contract", "roots"}),
        label="root_prior_manifest",
    )
    if raw.get("schema") != "ofc_multi_root_exact_prior/v1":
        raise ValueError("root_prior_manifest.schema: mismatch")
    if raw.get("sampling_contract") != SUPER_ROOT_SAMPLING_CONTRACT:
        raise ValueError("root_prior_manifest.sampling_contract: mismatch")
    roots = raw.get("roots")
    if not isinstance(roots, list) or len(roots) < 2:
        raise ValueError("root_prior_manifest.roots: at least two roots required")
    opaque_ids: list[str] = []
    masses: list[Fraction] = []
    for index, root_value in enumerate(roots):
        label = f"root_prior_manifest.roots[{index}]"
        root = _require_mapping(root_value, label=label)
        _require_exact_keys(
            root,
            frozenset(
                {
                    "root_id_sha256",
                    "prior_mass_exact",
                    "observation_sha256",
                    "conditional_particle_count",
                    "range_content_sha256",
                    "range_build_sha256",
                }
            ),
            label=label,
        )
        opaque_ids.append(
            _require_sha256(root.get("root_id_sha256"), label=f"{label}.root_id_sha256")
        )
        masses.append(
            _fraction_text(root.get("prior_mass_exact"), label=f"{label}.prior_mass_exact")
        )
        _require_sha256(
            root.get("observation_sha256"), label=f"{label}.observation_sha256"
        )
        _require_int(
            root.get("conditional_particle_count"),
            label=f"{label}.conditional_particle_count",
            minimum=1,
        )
        _require_sha256(
            root.get("range_content_sha256"),
            label=f"{label}.range_content_sha256",
        )
        _require_sha256(
            root.get("range_build_sha256"), label=f"{label}.range_build_sha256"
        )
    if len(set(opaque_ids)) != len(opaque_ids):
        raise ValueError("root_prior_manifest.roots: duplicate opaque root ID")
    if sum(masses, Fraction(0, 1)) != 1:
        raise ValueError("root_prior_manifest.roots: exact prior must sum to one")
    return copy.deepcopy(dict(raw)), tuple(opaque_ids)


def build_shared_multi_root_strategy_artifact(
    result: Any,
    *,
    stratum: str,
    training_root_identity_by_opaque_id: Mapping[str, str],
    source_manifest_sha256: str,
    range_builder_source_sha256: str,
    t3_bb_likelihood_binding_sha256: str,
) -> dict[str, Any]:
    """Adapt the public multi-root result surface into a strict artifact.

    No concrete result class is imported.  This keeps the persisted contract
    stable while the experimental solver implementation evolves.
    """

    actor, joker = _expected_identity(stratum)
    source_sha = _require_sha256(source_manifest_sha256, label="source_manifest_sha256")
    range_sha = _require_sha256(
        range_builder_source_sha256, label="range_builder_source_sha256"
    )
    binding_sha = _require_sha256(
        t3_bb_likelihood_binding_sha256,
        label="t3_bb_likelihood_binding_sha256",
    )
    metadata = _require_mapping(getattr(result, "metadata", None), label="result.metadata")
    sampling = _require_mapping(
        getattr(result, "sampling_stats", None), label="result.sampling_stats"
    )
    strategy_text = getattr(result, "average_strategy_json", None)
    strategy_profile = _parse_canonical_json(
        strategy_text, label="result.average_strategy_json"
    )
    verify_public_strategy_profile(strategy_profile)
    strategy_sha = hashlib.sha256(strategy_text.encode("utf-8")).hexdigest()
    if getattr(result, "average_strategy_sha256", None) != strategy_sha:
        raise ValueError("result.average_strategy_sha256: content mismatch")
    if metadata.get("average_strategy_sha256") != strategy_sha:
        raise ValueError("result.metadata.average_strategy_sha256: content mismatch")

    root_prior, opaque_ids = _verify_root_prior(metadata.get("root_prior_manifest"))
    prior_sha = canonical_sha256(root_prior)
    if metadata.get("root_prior_manifest_sha256") != prior_sha:
        raise ValueError("result.metadata.root_prior_manifest_sha256: content mismatch")
    supplied_training = _require_mapping(
        training_root_identity_by_opaque_id,
        label="training_root_identity_by_opaque_id",
    )
    if set(supplied_training) != set(opaque_ids):
        raise ValueError(
            "training_root_identity_by_opaque_id: exact root-prior opaque ID set required"
        )
    prior_by_id = {
        root["root_id_sha256"]: root for root in root_prior["roots"]
    }
    training_roots = []
    for opaque_id in opaque_ids:
        prior_root = prior_by_id[opaque_id]
        training_roots.append(
            {
                "root_id_sha256": opaque_id,
                "root_identity_commitment_sha256": _require_sha256(
                    supplied_training[opaque_id],
                    label=(
                        "training_root_identity_by_opaque_id."
                        f"{opaque_id}"
                    ),
                ),
                "observation_digest": prior_root["observation_sha256"],
                "range_content_sha256": prior_root["range_content_sha256"],
                "range_build_sha256": prior_root["range_build_sha256"],
            }
        )

    producer_contract = {
        field: _json_copy(metadata.get(field)) for field in _PRODUCER_CONTRACT_KEYS
    }
    sampling_audit = {
        field: _json_copy(sampling.get(field)) for field in _SAMPLING_AUDIT_KEYS
    }
    artifact: dict[str, Any] = {
        "schema": STRATEGY_ARTIFACT_SCHEMA,
        "artifact_kind": "shared_infoset_chance_super_root_t3_strategy",
        "stratum": stratum,
        "actor": actor,
        "visible_joker_count": joker,
        "solver_method": MULTI_ROOT_SOLVER_METHOD,
        "policy_identity_contract": POLICY_IDENTITY_CONTRACT,
        "source_manifest_sha256": source_sha,
        "range_builder_source_sha256": range_sha,
        "t3_bb_likelihood_binding_sha256": binding_sha,
        "algorithm_validation_only": True,
        "root_scope_binding_contract": (
            "training_prior_observation_and_range_commitments_v1"
        ),
        "global_policy_claim": False,
        "unseen_root_generalization_claim": False,
        "promotion_eligible": False,
        "exact_exploitability_computed": False,
        "strategy_profile": strategy_profile,
        "average_strategy_sha256": strategy_sha,
        "root_prior_manifest": root_prior,
        "root_prior_manifest_sha256": prior_sha,
        "training_roots": training_roots,
        "producer_contract": producer_contract,
        "sampling_audit": sampling_audit,
    }
    artifact["artifact_sha256"] = self_hash(artifact, "artifact_sha256")
    return verify_shared_multi_root_strategy_artifact(artifact)


def verify_shared_multi_root_strategy_artifact(value: Any) -> dict[str, Any]:
    """Fail closed unless an artifact proves one shared public policy table."""

    raw = _require_mapping(value, label="strategy_artifact")
    _require_exact_keys(raw, _STRATEGY_ARTIFACT_KEYS, label="strategy_artifact")
    if raw.get("schema") != STRATEGY_ARTIFACT_SCHEMA:
        raise ValueError("strategy_artifact.schema: mismatch")
    if raw.get("artifact_kind") != "shared_infoset_chance_super_root_t3_strategy":
        raise ValueError("strategy_artifact.artifact_kind: mismatch")
    actor, joker = _expected_identity(str(raw.get("stratum")))
    if raw.get("actor") != actor or raw.get("visible_joker_count") != joker:
        raise ValueError("strategy_artifact: Joker-layer/role identity mismatch")
    if raw.get("solver_method") != MULTI_ROOT_SOLVER_METHOD:
        raise ValueError("strategy_artifact.solver_method: mismatch")
    if raw.get("policy_identity_contract") != POLICY_IDENTITY_CONTRACT:
        raise ValueError("strategy_artifact.policy_identity_contract: mismatch")
    scope_expected = {
        "algorithm_validation_only": True,
        "root_scope_binding_contract": (
            "training_prior_observation_and_range_commitments_v1"
        ),
        "global_policy_claim": False,
        "unseen_root_generalization_claim": False,
    }
    for field, wanted in scope_expected.items():
        if raw.get(field) != wanted:
            raise ValueError(f"strategy_artifact.{field}: mismatch")
    for field in (
        "source_manifest_sha256",
        "range_builder_source_sha256",
        "t3_bb_likelihood_binding_sha256",
        "average_strategy_sha256",
        "root_prior_manifest_sha256",
        "artifact_sha256",
    ):
        _require_sha256(raw.get(field), label=f"strategy_artifact.{field}")
    if raw.get("promotion_eligible") is not False:
        raise ValueError("strategy_artifact.promotion_eligible: must be false")
    if raw.get("exact_exploitability_computed") is not False:
        raise ValueError(
            "strategy_artifact.exact_exploitability_computed: must be false"
        )
    if raw.get("artifact_sha256") != self_hash(raw, "artifact_sha256"):
        raise ValueError("strategy_artifact.artifact_sha256: self-hash mismatch")

    profile, _distributions, infosets = verify_public_strategy_profile(
        raw.get("strategy_profile")
    )
    if canonical_sha256(profile) != raw.get("average_strategy_sha256"):
        raise ValueError("strategy_artifact.average_strategy_sha256: profile mismatch")
    required_phase = "t3_first" if actor == "bb" else "t3_second"
    if not any(
        key.actor == actor and key.turn == 3 and key.phase == required_phase
        for key in infosets.values()
    ):
        raise ValueError(
            "strategy_artifact.strategy_profile: missing own T3 role infoset"
        )

    prior, opaque_ids = _verify_root_prior(raw.get("root_prior_manifest"))
    if canonical_sha256(prior) != raw.get("root_prior_manifest_sha256"):
        raise ValueError("strategy_artifact.root_prior_manifest_sha256: mismatch")
    training_roots = raw.get("training_roots")
    if not isinstance(training_roots, list) or len(training_roots) != len(opaque_ids):
        raise ValueError("strategy_artifact.training_roots: exact prior coverage required")
    identities: list[str] = []
    prior_roots = prior["roots"]
    for index, item_value in enumerate(training_roots):
        label = f"strategy_artifact.training_roots[{index}]"
        item = _require_mapping(item_value, label=label)
        _require_exact_keys(
            item,
            frozenset(
                {
                    "root_id_sha256",
                    "root_identity_commitment_sha256",
                    "observation_digest",
                    "range_content_sha256",
                    "range_build_sha256",
                }
            ),
            label=label,
        )
        if item.get("root_id_sha256") != opaque_ids[index]:
            raise ValueError(f"{label}.root_id_sha256: prior order mismatch")
        prior_root = prior_roots[index]
        content_bindings = {
            "observation_digest": prior_root["observation_sha256"],
            "range_content_sha256": prior_root["range_content_sha256"],
            "range_build_sha256": prior_root["range_build_sha256"],
        }
        for field, wanted in content_bindings.items():
            actual = _require_sha256(item.get(field), label=f"{label}.{field}")
            if actual != wanted:
                raise ValueError(f"{label}.{field}: root-prior content mismatch")
        identities.append(
            _require_sha256(
                item.get("root_identity_commitment_sha256"),
                label=f"{label}.root_identity_commitment_sha256",
            )
        )
    if len(set(identities)) != len(identities):
        raise ValueError("strategy_artifact.training_roots: duplicate identity")

    producer = _require_mapping(
        raw.get("producer_contract"), label="strategy_artifact.producer_contract"
    )
    _require_exact_keys(
        producer, _PRODUCER_CONTRACT_KEYS, label="strategy_artifact.producer_contract"
    )
    expected = {
        "method": MULTI_ROOT_SOLVER_METHOD,
        "chance_super_root": True,
        "super_root_sampling_contract": SUPER_ROOT_SAMPLING_CONTRACT,
        "root_prior_normalized_exact": True,
        "root_count": len(opaque_ids),
        "traverser_schedule": "bb_then_btn_each_iteration",
        "alternating_updates": True,
        "regret_matching_plus": True,
        "regret_clip_scope": "once_per_infoset_after_traversal",
        "root_prior_sampled_once_per_traversal": True,
        "conditional_posterior_sampled_once_per_traversal": True,
        "root_probability_multiplied_after_sampling": False,
        "posterior_probability_multiplied_after_sampling": False,
        "chance_probability_multiplied_after_sampling": False,
        "joint_particle_weight_used_after_sampling": False,
        "policy_identity_contract": POLICY_IDENTITY_CONTRACT,
        "policy_table_shared_across_all_roots": True,
        "table_key_type": "InfoSetKey",
        "table_key_contains_root_id": False,
        "table_key_contains_private_type_id": False,
        "table_key_contains_particle_commitment": False,
        "table_key_contains_remaining_cards": False,
        "strategy_serialization_contains_root_id": False,
        "strategy_serialization_contains_hidden_particle": False,
        "strategy_fusion": False,
        "independent_per_root_solve": False,
        "compatible_full_card_adapters": True,
        "full_card": True,
        "full_card_policy_promoted": False,
        "promotion_eligible": False,
        "runtime_integrated": False,
        "hu_exact": False,
        "exact_exploitability_computed": False,
        "position_contract_version": POSITION_CONTRACT_VERSION,
    }
    for field, wanted in expected.items():
        if producer.get(field) != wanted:
            raise ValueError(f"strategy_artifact.producer_contract.{field}: mismatch")
    if not isinstance(producer.get("linear_averaging"), bool):
        raise TypeError("strategy_artifact.producer_contract.linear_averaging: bool required")
    iterations = _require_int(
        producer.get("iterations"),
        label="strategy_artifact.producer_contract.iterations",
        minimum=1,
    )
    traversals = _require_int(
        producer.get("traversals"),
        label="strategy_artifact.producer_contract.traversals",
        minimum=1,
    )
    _require_int(
        producer.get("seed"), label="strategy_artifact.producer_contract.seed"
    )
    _require_int(
        producer.get("max_infosets"),
        label="strategy_artifact.producer_contract.max_infosets",
        minimum=1,
    )
    shared_count = _require_int(
        producer.get("shared_across_roots_infoset_count"),
        label=(
            "strategy_artifact.producer_contract."
            "shared_across_roots_infoset_count"
        ),
        minimum=1,
    )
    if traversals != 2 * iterations:
        raise ValueError("strategy_artifact.producer_contract.traversals: mismatch")
    if shared_count > len(profile["records"]):
        raise ValueError(
            "strategy_artifact.producer_contract.shared_across_roots_infoset_count: "
            "exceeds strategy size"
        )

    sampling = _require_mapping(
        raw.get("sampling_audit"), label="strategy_artifact.sampling_audit"
    )
    _require_exact_keys(
        sampling, _SAMPLING_AUDIT_KEYS, label="strategy_artifact.sampling_audit"
    )
    for field in (
        "traversals",
        "super_root_samples",
        "conditional_root_posterior_samples",
    ):
        if sampling.get(field) != traversals:
            raise ValueError(f"strategy_artifact.sampling_audit.{field}: mismatch")
    if sampling.get("traversals_by_actor") != {
        "bb": iterations,
        "btn": iterations,
    }:
        raise ValueError(
            "strategy_artifact.sampling_audit.traversals_by_actor: mismatch"
        )
    root_counts = _require_mapping(
        sampling.get("root_samples_by_opaque_id"),
        label="strategy_artifact.sampling_audit.root_samples_by_opaque_id",
    )
    if set(root_counts) != set(opaque_ids):
        raise ValueError(
            "strategy_artifact.sampling_audit.root_samples_by_opaque_id: "
            "exact prior root set required"
        )
    counts = [
        _require_int(
            root_counts[root_id],
            label=(
                "strategy_artifact.sampling_audit.root_samples_by_opaque_id."
                f"{root_id}"
            ),
            minimum=0,
        )
        for root_id in opaque_ids
    ]
    if sum(counts) != traversals or any(count <= 0 for count in counts):
        raise ValueError(
            "strategy_artifact.sampling_audit: every root must be sampled and "
            "counts must sum to traversals"
        )
    if sampling.get("distinct_root_adapters_sampled") != len(opaque_ids):
        raise ValueError(
            "strategy_artifact.sampling_audit.distinct_root_adapters_sampled: mismatch"
        )
    if sampling.get("infosets_created") != len(profile["records"]):
        raise ValueError("strategy_artifact.sampling_audit.infosets_created: mismatch")
    return copy.deepcopy(dict(raw))


def _candidate_scope_manifest(
    strategies: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    source_hashes = {
        artifact["source_manifest_sha256"] for artifact in strategies.values()
    }
    if len(source_hashes) != 1:
        raise ValueError("candidate scope requires one solver source manifest")
    return {
        "schema": "ofc_t3_tabular_candidate_root_scope/v1",
        "policy_scope": "training_root_scoped_encountered_infosets_only",
        "global_policy_claim": False,
        "unseen_root_generalization_claim": False,
        "source_manifest_sha256": next(iter(source_hashes)),
        "strategies": {
            stratum: {
                "artifact_sha256": strategies[stratum]["artifact_sha256"],
                "root_prior_manifest_sha256": strategies[stratum][
                    "root_prior_manifest_sha256"
                ],
                "training_roots": copy.deepcopy(
                    strategies[stratum]["training_roots"]
                ),
            }
            for stratum in REQUIRED_STRATA
        },
    }


def build_shared_multi_root_candidate_bundle(
    strategies: Mapping[str, Mapping[str, Any]],
    *,
    t3_bb_likelihood_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Lock the exact six role/Joker strategies before holdout evaluation."""

    verified_binding = verify_t3_bb_likelihood_binding(t3_bb_likelihood_binding)
    if set(strategies) != set(REQUIRED_STRATA):
        raise ValueError("strategies: exact six Joker-layer/role strata required")
    verified_strategies = {
        stratum: verify_shared_multi_root_strategy_artifact(strategies[stratum])
        for stratum in REQUIRED_STRATA
    }
    scope_manifest = _candidate_scope_manifest(verified_strategies)
    bundle: dict[str, Any] = {
        "schema": CANDIDATE_BUNDLE_SCHEMA,
        "artifact_kind": "six_stratum_shared_infoset_multi_root_t3_candidate",
        "ruleset": RULESET,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "physical_joker_ids": ["X1", "X2"],
        "locked_before_holdout": True,
        "holdout_used_for_candidate_selection": False,
        "algorithm_validation_only": True,
        "production_promotion_supported": False,
        "unseen_root_strength_supported": False,
        "promotion_eligible": False,
        "exact_exploitability_computed": False,
        "t3_bb_likelihood_binding": verified_binding,
        "candidate_scope_manifest": scope_manifest,
        "candidate_scope_manifest_sha256": canonical_sha256(scope_manifest),
        "strategies": verified_strategies,
    }
    bundle["candidate_bundle_sha256"] = self_hash(
        bundle, "candidate_bundle_sha256"
    )
    return verify_shared_multi_root_candidate_bundle(bundle)


def verify_shared_multi_root_candidate_bundle(value: Any) -> dict[str, Any]:
    raw = _require_mapping(value, label="candidate_bundle")
    _require_exact_keys(raw, _BUNDLE_KEYS, label="candidate_bundle")
    expected = {
        "schema": CANDIDATE_BUNDLE_SCHEMA,
        "artifact_kind": "six_stratum_shared_infoset_multi_root_t3_candidate",
        "ruleset": RULESET,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "physical_joker_ids": ["X1", "X2"],
        "locked_before_holdout": True,
        "holdout_used_for_candidate_selection": False,
        "algorithm_validation_only": True,
        "production_promotion_supported": False,
        "unseen_root_strength_supported": False,
        "promotion_eligible": False,
        "exact_exploitability_computed": False,
    }
    for field, wanted in expected.items():
        if raw.get(field) != wanted:
            raise ValueError(f"candidate_bundle.{field}: mismatch")
    if raw.get("candidate_bundle_sha256") != self_hash(
        raw, "candidate_bundle_sha256"
    ):
        raise ValueError("candidate_bundle.candidate_bundle_sha256: self-hash mismatch")
    _require_sha256(
        raw.get("candidate_scope_manifest_sha256"),
        label="candidate_bundle.candidate_scope_manifest_sha256",
    )
    binding = verify_t3_bb_likelihood_binding(raw.get("t3_bb_likelihood_binding"))
    binding_sha = binding["binding_sha256"]
    range_source_sha = binding["range_builder_source_sha256"]
    strategies = _require_mapping(raw.get("strategies"), label="candidate_bundle.strategies")
    if set(strategies) != set(REQUIRED_STRATA):
        raise ValueError("candidate_bundle.strategies: exact six strata required")

    verified: dict[str, dict[str, Any]] = {}
    training_identities: set[str] = set()
    source_hashes: set[str] = set()
    artifact_hashes: set[str] = set()
    for stratum in REQUIRED_STRATA:
        artifact = verify_shared_multi_root_strategy_artifact(strategies[stratum])
        if artifact["stratum"] != stratum:
            raise ValueError(
                f"candidate_bundle.strategies.{stratum}: stratum binding mismatch"
            )
        if artifact["t3_bb_likelihood_binding_sha256"] != binding_sha:
            raise ValueError(
                f"candidate_bundle.strategies.{stratum}: fixed-point binding mismatch"
            )
        if artifact["range_builder_source_sha256"] != range_source_sha:
            raise ValueError(
                f"candidate_bundle.strategies.{stratum}: range source mismatch"
            )
        artifact_sha = artifact["artifact_sha256"]
        if artifact_sha in artifact_hashes:
            raise ValueError("candidate_bundle.strategies: duplicate artifact identity")
        artifact_hashes.add(artifact_sha)
        source_hashes.add(artifact["source_manifest_sha256"])
        identities = {
            row["root_identity_commitment_sha256"]
            for row in artifact["training_roots"]
        }
        overlap = training_identities & identities
        if overlap:
            raise ValueError(
                "candidate_bundle.strategies: training roots reused across strata"
            )
        training_identities.update(identities)
        verified[stratum] = artifact
    if len(source_hashes) != 1:
        raise ValueError("candidate_bundle.strategies: source manifest mismatch")
    expected_scope = _candidate_scope_manifest(verified)
    if raw.get("candidate_scope_manifest") != expected_scope:
        raise ValueError("candidate_bundle.candidate_scope_manifest: content mismatch")
    if raw.get("candidate_scope_manifest_sha256") != canonical_sha256(
        expected_scope
    ):
        raise ValueError(
            "candidate_bundle.candidate_scope_manifest_sha256: content mismatch"
        )
    result = copy.deepcopy(dict(raw))
    result["strategies"] = verified
    return result


def build_excluded_root_partition(
    purpose: str, root_identity_commitments: Sequence[str]
) -> dict[str, Any]:
    if purpose not in REQUIRED_EXCLUDED_PARTITIONS:
        raise ValueError("unsupported excluded partition purpose")
    commitments = sorted(
        _require_sha256(value, label=f"{purpose} root identity")
        for value in root_identity_commitments
    )
    if not commitments or len(commitments) != len(set(commitments)):
        raise ValueError("excluded partition requires unique non-empty commitments")
    partition: dict[str, Any] = {
        "schema": ROOT_PARTITION_SCHEMA,
        "purpose": purpose,
        "root_commitments": commitments,
    }
    partition["manifest_sha256"] = self_hash(partition, "manifest_sha256")
    _verify_partition(purpose, partition)
    return partition


def _verify_partition(
    purpose: str, value: Any
) -> tuple[dict[str, Any], set[str]]:
    raw = _require_mapping(value, label=f"excluded_root_partitions.{purpose}")
    _require_exact_keys(
        raw, _PARTITION_KEYS, label=f"excluded_root_partitions.{purpose}"
    )
    if raw.get("schema") != ROOT_PARTITION_SCHEMA or raw.get("purpose") != purpose:
        raise ValueError(f"excluded_root_partitions.{purpose}: schema/purpose mismatch")
    if raw.get("manifest_sha256") != self_hash(raw, "manifest_sha256"):
        raise ValueError(
            f"excluded_root_partitions.{purpose}.manifest_sha256: self-hash mismatch"
        )
    commitments = raw.get("root_commitments")
    if not isinstance(commitments, list) or not commitments:
        raise ValueError(
            f"excluded_root_partitions.{purpose}.root_commitments: non-empty list required"
        )
    normalized = [
        _require_sha256(item, label=f"excluded_root_partitions.{purpose} root")
        for item in commitments
    ]
    if normalized != sorted(normalized) or len(normalized) != len(set(normalized)):
        raise ValueError(
            f"excluded_root_partitions.{purpose}.root_commitments: "
            "canonical unique order required"
        )
    return copy.deepcopy(dict(raw)), set(normalized)


def build_holdout_root_record(
    root_id: str,
    *,
    stratum: str,
    observation_digest: str,
    seat_swap_pair_id: str,
) -> dict[str, Any]:
    if not isinstance(root_id, str) or not root_id.strip() or root_id != root_id.strip():
        raise ValueError("root_id: non-empty trimmed string required")
    actor, joker = _expected_identity(stratum)
    if not isinstance(seat_swap_pair_id, str) or not seat_swap_pair_id.strip():
        raise ValueError("seat_swap_pair_id: non-empty string required")
    root: dict[str, Any] = {
        "root_id": root_id,
        "stratum": stratum,
        "actor": actor,
        "visible_joker_count": joker,
        "observation_digest": _require_sha256(
            observation_digest, label="observation_digest"
        ),
        "seat_swap_pair_id": seat_swap_pair_id,
        "root_identity_commitment_sha256": root_identity_commitment_sha256(root_id),
    }
    root["root_commitment_sha256"] = root_commitment_sha256(root)
    return root


def build_holdout_root_manifest(
    roots: Sequence[Mapping[str, Any]],
    *,
    excluded_root_partition_sha256: Mapping[str, str],
) -> dict[str, Any]:
    partition_hashes = _require_mapping(
        excluded_root_partition_sha256,
        label="excluded_root_partition_sha256",
    )
    if set(partition_hashes) != set(REQUIRED_EXCLUDED_PARTITIONS):
        raise ValueError("excluded_root_partition_sha256: exact partition set required")
    ordered = sorted(
        (_json_copy(root) for root in roots),
        key=lambda root: (root.get("stratum", ""), root.get("root_id", "")),
    )
    manifest: dict[str, Any] = {
        "schema": ROOT_MANIFEST_SCHEMA,
        "purpose": "independent_promotion_holdout",
        "locked_before_evaluation": True,
        "ruleset": RULESET,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "physical_joker_ids": ["X1", "X2"],
        "excluded_root_partition_sha256": {
            purpose: _require_sha256(
                partition_hashes[purpose],
                label=f"excluded_root_partition_sha256.{purpose}",
            )
            for purpose in REQUIRED_EXCLUDED_PARTITIONS
        },
        "roots": ordered,
    }
    manifest["root_manifest_sha256"] = self_hash(
        manifest, "root_manifest_sha256"
    )
    _verify_root_manifest(manifest, min_roots_per_stratum=1)
    return manifest


def _verify_root_manifest(
    value: Any, *, min_roots_per_stratum: int
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, int]]:
    raw = _require_mapping(value, label="root_manifest")
    _require_exact_keys(raw, _ROOT_MANIFEST_KEYS, label="root_manifest")
    expected = {
        "schema": ROOT_MANIFEST_SCHEMA,
        "purpose": "independent_promotion_holdout",
        "locked_before_evaluation": True,
        "ruleset": RULESET,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "physical_joker_ids": ["X1", "X2"],
    }
    for field, wanted in expected.items():
        if raw.get(field) != wanted:
            raise ValueError(f"root_manifest.{field}: mismatch")
    if raw.get("root_manifest_sha256") != self_hash(
        raw, "root_manifest_sha256"
    ):
        raise ValueError("root_manifest.root_manifest_sha256: self-hash mismatch")
    partitions = _require_mapping(
        raw.get("excluded_root_partition_sha256"),
        label="root_manifest.excluded_root_partition_sha256",
    )
    if set(partitions) != set(REQUIRED_EXCLUDED_PARTITIONS):
        raise ValueError(
            "root_manifest.excluded_root_partition_sha256: exact partition set required"
        )
    for purpose in REQUIRED_EXCLUDED_PARTITIONS:
        _require_sha256(
            partitions[purpose],
            label=f"root_manifest.excluded_root_partition_sha256.{purpose}",
        )
    roots = raw.get("roots")
    if not isinstance(roots, list) or not roots:
        raise ValueError("root_manifest.roots: non-empty list required")
    by_id: dict[str, dict[str, Any]] = {}
    counts = {stratum: 0 for stratum in REQUIRED_STRATA}
    observations: set[str] = set()
    identities: set[str] = set()
    full_commitments: set[str] = set()
    pairs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    order: list[tuple[str, str]] = []
    for index, root_value in enumerate(roots):
        label = f"root_manifest.roots[{index}]"
        root = _require_mapping(root_value, label=label)
        _require_exact_keys(root, _ROOT_KEYS, label=label)
        root_id = root.get("root_id")
        if not isinstance(root_id, str) or not root_id.strip() or root_id != root_id.strip():
            raise ValueError(f"{label}.root_id: non-empty trimmed string required")
        if root_id in by_id:
            raise ValueError(f"{label}.root_id: duplicate")
        stratum = str(root.get("stratum"))
        actor, joker = _expected_identity(stratum)
        if root.get("actor") != actor or root.get("visible_joker_count") != joker:
            raise ValueError(f"{label}: Joker-layer/role identity mismatch")
        observation = _require_sha256(
            root.get("observation_digest"), label=f"{label}.observation_digest"
        )
        if observation in observations:
            raise ValueError(f"{label}.observation_digest: duplicate")
        observations.add(observation)
        pair_id = root.get("seat_swap_pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"{label}.seat_swap_pair_id: non-empty string required")
        identity = root_identity_commitment_sha256(root_id)
        if root.get("root_identity_commitment_sha256") != identity:
            raise ValueError(f"{label}.root_identity_commitment_sha256: mismatch")
        if identity in identities:
            raise ValueError(f"{label}.root_identity_commitment_sha256: duplicate")
        identities.add(identity)
        commitment = root_commitment_sha256(root)
        if root.get("root_commitment_sha256") != commitment:
            raise ValueError(f"{label}.root_commitment_sha256: mismatch")
        if commitment in full_commitments:
            raise ValueError(f"{label}.root_commitment_sha256: duplicate")
        full_commitments.add(commitment)
        by_id[root_id] = copy.deepcopy(dict(root))
        counts[stratum] += 1
        pairs[str(pair_id)].append(root)
        order.append((stratum, root_id))
    if order != sorted(order):
        raise ValueError("root_manifest.roots: canonical stratum/root order required")
    for stratum, count in counts.items():
        if count < min_roots_per_stratum:
            raise ValueError(
                f"root_manifest: {stratum} needs >= {min_roots_per_stratum} roots"
            )
    for pair_id, pair_roots in pairs.items():
        actors = {root["actor"] for root in pair_roots}
        jokers = {root["visible_joker_count"] for root in pair_roots}
        if len(pair_roots) != 2 or actors != {"bb", "btn"} or len(jokers) != 1:
            raise ValueError(
                f"root_manifest: pair {pair_id!r} requires one BB and one BTN "
                "root in one Joker layer"
            )
    return copy.deepcopy(dict(raw)), by_id, counts


def build_holdout_evaluator_manifest(
    *,
    candidate_bundle_sha256: str,
    root_manifest_sha256: str,
    reference_policy_sha256: str,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": EVALUATOR_MANIFEST_SCHEMA,
        "method": HOLDOUT_EVALUATOR_METHOD,
        "ruleset": RULESET,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "candidate_bundle_sha256": _require_sha256(
            candidate_bundle_sha256, label="candidate_bundle_sha256"
        ),
        "root_manifest_sha256": _require_sha256(
            root_manifest_sha256, label="root_manifest_sha256"
        ),
        "reference_policy_sha256": _require_sha256(
            reference_policy_sha256, label="reference_policy_sha256"
        ),
        "source_manifest_sha256": _require_sha256(
            source_manifest_sha256, label="source_manifest_sha256"
        ),
        "candidate_policy_frozen": True,
        "strategy_updates_during_evaluation": False,
        "holdout_used_for_candidate_selection": False,
        "action_payoff_samples_independent_of_training": True,
        "exact_exploitability_computed": False,
    }
    manifest["manifest_sha256"] = self_hash(manifest, "manifest_sha256")
    return verify_holdout_evaluator_manifest(manifest)


def verify_holdout_evaluator_manifest(value: Any) -> dict[str, Any]:
    raw = _require_mapping(value, label="evaluator_manifest")
    _require_exact_keys(raw, _EVALUATOR_KEYS, label="evaluator_manifest")
    expected = {
        "schema": EVALUATOR_MANIFEST_SCHEMA,
        "method": HOLDOUT_EVALUATOR_METHOD,
        "ruleset": RULESET,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "candidate_policy_frozen": True,
        "strategy_updates_during_evaluation": False,
        "holdout_used_for_candidate_selection": False,
        "action_payoff_samples_independent_of_training": True,
        "exact_exploitability_computed": False,
    }
    for field, wanted in expected.items():
        if raw.get(field) != wanted:
            raise ValueError(f"evaluator_manifest.{field}: mismatch")
    for field in (
        "candidate_bundle_sha256",
        "root_manifest_sha256",
        "reference_policy_sha256",
        "source_manifest_sha256",
        "manifest_sha256",
    ):
        _require_sha256(raw.get(field), label=f"evaluator_manifest.{field}")
    if raw.get("manifest_sha256") != self_hash(raw, "manifest_sha256"):
        raise ValueError("evaluator_manifest.manifest_sha256: self-hash mismatch")
    return copy.deepcopy(dict(raw))


def _validate_thresholds(value: Any) -> dict[str, Any]:
    raw = _require_mapping(value, label="config.thresholds")
    _require_exact_keys(raw, _THRESHOLD_KEYS, label="config.thresholds")
    normalized = copy.deepcopy(dict(raw))
    _require_int(
        raw.get("min_independent_evaluation_seeds_per_root"),
        label="config.thresholds.min_independent_evaluation_seeds_per_root",
        minimum=2,
    )
    _require_int(
        raw.get("min_roots_per_stratum"),
        label="config.thresholds.min_roots_per_stratum",
        minimum=1,
    )
    _require_int(
        raw.get("min_action_payoff_samples_per_action"),
        label="config.thresholds.min_action_payoff_samples_per_action",
        minimum=2,
    )
    numeric = _THRESHOLD_KEYS - {
        "min_independent_evaluation_seeds_per_root",
        "min_roots_per_stratum",
        "min_action_payoff_samples_per_action",
    }
    for field in numeric:
        number = _require_finite(raw.get(field), label=f"config.thresholds.{field}")
        if number < 0.0:
            raise ValueError(f"config.thresholds.{field}: nonnegative required")
    coverage = float(raw["min_encountered_infoset_coverage"])
    if coverage > 1.0:
        raise ValueError("config.thresholds.min_encountered_infoset_coverage: <= 1 required")
    regrets = (
        float(raw["max_mean_ev_regret_score"]),
        float(raw["max_p95_ev_regret_score"]),
        float(raw["max_p99_ev_regret_score"]),
    )
    if regrets != tuple(sorted(regrets)):
        raise ValueError("config.thresholds: regret limits must be ordered")
    runtimes = (
        float(raw["max_runtime_ms_p95"]),
        float(raw["max_runtime_ms_max"]),
        float(raw["max_runtime_ms_total"]),
    )
    if runtimes != tuple(sorted(runtimes)):
        raise ValueError("config.thresholds: runtime limits must be ordered")
    return normalized


def build_locked_shared_multi_root_strength_config(
    *,
    approved_candidate_bundle_sha256: str,
    approved_holdout_root_manifest_sha256: str,
    approved_evaluator_manifest_sha256: str,
    approved_excluded_root_partition_sha256: Mapping[str, str],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    partition_hashes = _require_mapping(
        approved_excluded_root_partition_sha256,
        label="approved_excluded_root_partition_sha256",
    )
    if set(partition_hashes) != set(REQUIRED_EXCLUDED_PARTITIONS):
        raise ValueError("approved excluded partitions: exact set required")
    config: dict[str, Any] = {
        "schema": CONFIG_SCHEMA,
        "gate_id": GATE_ID,
        "approved_candidate_bundle_sha256": _require_sha256(
            approved_candidate_bundle_sha256,
            label="approved_candidate_bundle_sha256",
        ),
        "approved_holdout_root_manifest_sha256": _require_sha256(
            approved_holdout_root_manifest_sha256,
            label="approved_holdout_root_manifest_sha256",
        ),
        "approved_evaluator_manifest_sha256": _require_sha256(
            approved_evaluator_manifest_sha256,
            label="approved_evaluator_manifest_sha256",
        ),
        "approved_excluded_root_partition_sha256": {
            purpose: _require_sha256(
                partition_hashes[purpose],
                label=f"approved_excluded_root_partition_sha256.{purpose}",
            )
            for purpose in REQUIRED_EXCLUDED_PARTITIONS
        },
        "thresholds": _validate_thresholds(thresholds),
    }
    config["config_sha256"] = self_hash(config, "config_sha256")
    return verify_locked_shared_multi_root_strength_config(config)


def verify_locked_shared_multi_root_strength_config(value: Any) -> dict[str, Any]:
    raw = _require_mapping(value, label="config")
    _require_exact_keys(raw, _CONFIG_KEYS, label="config")
    if raw.get("schema") != CONFIG_SCHEMA or raw.get("gate_id") != GATE_ID:
        raise ValueError("config: schema/gate mismatch")
    for field in (
        "approved_candidate_bundle_sha256",
        "approved_holdout_root_manifest_sha256",
        "approved_evaluator_manifest_sha256",
        "config_sha256",
    ):
        _require_sha256(raw.get(field), label=f"config.{field}")
    partitions = _require_mapping(
        raw.get("approved_excluded_root_partition_sha256"),
        label="config.approved_excluded_root_partition_sha256",
    )
    if set(partitions) != set(REQUIRED_EXCLUDED_PARTITIONS):
        raise ValueError("config approved partitions: exact set required")
    for purpose in REQUIRED_EXCLUDED_PARTITIONS:
        _require_sha256(
            partitions[purpose],
            label=f"config.approved_excluded_root_partition_sha256.{purpose}",
        )
    _validate_thresholds(raw.get("thresholds"))
    if raw.get("config_sha256") != self_hash(raw, "config_sha256"):
        raise ValueError("config.config_sha256: self-hash mismatch")
    return copy.deepcopy(dict(raw))


def _distribution(value: Any, *, label: str) -> dict[str, float]:
    raw = _require_mapping(value, label=label)
    if not raw:
        raise ValueError(f"{label}: non-empty distribution required")
    result: dict[str, float] = {}
    for action_id, probability_value in raw.items():
        if not isinstance(action_id, str) or not action_id:
            raise ValueError(f"{label}: non-empty action IDs required")
        probability = _require_finite(
            probability_value, label=f"{label}.{action_id}"
        )
        if probability < 0.0:
            raise ValueError(f"{label}.{action_id}: nonnegative required")
        result[action_id] = probability
    if not math.isclose(
        math.fsum(result.values()), 1.0, rel_tol=0.0, abs_tol=_FLOAT_TOL
    ):
        raise ValueError(f"{label}: probabilities must sum to one")
    return result


def _weighted_payoff(
    distribution: Mapping[str, float], payoffs: Mapping[str, float]
) -> float:
    return math.fsum(
        distribution[action_id] * payoffs[action_id]
        for action_id in sorted(payoffs)
    )


def _aggregate_mean_standard_error(
    *, count: int, total: float, total_squares: float, label: str
) -> tuple[float, float]:
    if count < 2:
        raise ValueError(f"{label}.count: at least two samples required")
    minimum_squares = total * total / count
    tolerance = _FLOAT_TOL * max(1.0, abs(total_squares), abs(minimum_squares))
    if total_squares + tolerance < minimum_squares:
        raise ValueError(f"{label}.sum_squares: impossible raw moments")
    centered = max(0.0, total_squares - minimum_squares)
    sample_variance = centered / (count - 1)
    return total / count, math.sqrt(sample_variance / count)


def _verify_action_moment_family(
    row: Mapping[str, Any],
    *,
    label: str,
    prefix: str,
    action_ids: set[str],
    min_samples: int,
) -> tuple[dict[str, float], dict[str, int], dict[str, float]]:
    estimates_field = f"{prefix}action_payoff_estimates"
    counts_field = f"{prefix}action_payoff_sample_counts"
    aggregates_field = f"{prefix}action_payoff_aggregates"
    errors_field = f"{prefix}action_payoff_standard_errors"
    max_error_field = f"max_{prefix}action_payoff_standard_error"

    estimate_values = _require_mapping(
        row.get(estimates_field), label=f"{label}.{estimates_field}"
    )
    estimates = {
        action_id: _require_finite(
            value, label=f"{label}.{estimates_field}.{action_id}"
        )
        for action_id, value in estimate_values.items()
    }
    count_values = _require_mapping(
        row.get(counts_field), label=f"{label}.{counts_field}"
    )
    counts = {
        action_id: _require_int(
            value,
            label=f"{label}.{counts_field}.{action_id}",
            minimum=min_samples,
        )
        for action_id, value in count_values.items()
    }
    aggregate_values = _require_mapping(
        row.get(aggregates_field), label=f"{label}.{aggregates_field}"
    )
    error_values = _require_mapping(
        row.get(errors_field), label=f"{label}.{errors_field}"
    )
    if (
        set(estimates) != action_ids
        or set(counts) != action_ids
        or set(aggregate_values) != action_ids
        or set(error_values) != action_ids
    ):
        raise ValueError(f"{label}.{prefix}action_payoff: exact action support required")

    derived_errors: dict[str, float] = {}
    for action_id in sorted(action_ids):
        aggregate_label = f"{label}.{aggregates_field}.{action_id}"
        aggregate = _require_mapping(
            aggregate_values[action_id], label=aggregate_label
        )
        _require_exact_keys(
            aggregate,
            frozenset({"count", "sum", "sum_squares"}),
            label=aggregate_label,
        )
        count = _require_int(
            aggregate.get("count"),
            label=f"{aggregate_label}.count",
            minimum=min_samples,
        )
        total = _require_finite(
            aggregate.get("sum"), label=f"{aggregate_label}.sum"
        )
        total_squares = _require_finite(
            aggregate.get("sum_squares"),
            label=f"{aggregate_label}.sum_squares",
        )
        if total_squares < 0.0:
            raise ValueError(f"{aggregate_label}.sum_squares: nonnegative required")
        if count != counts[action_id]:
            raise ValueError(f"{aggregate_label}.count: sample count mismatch")
        mean, standard_error = _aggregate_mean_standard_error(
            count=count,
            total=total,
            total_squares=total_squares,
            label=aggregate_label,
        )
        if not math.isclose(
            mean, estimates[action_id], rel_tol=0.0, abs_tol=_FLOAT_TOL
        ):
            raise ValueError(f"{aggregate_label}: mean/payoff mismatch")
        published_error = _require_finite(
            error_values[action_id],
            label=f"{label}.{errors_field}.{action_id}",
        )
        if published_error < 0.0 or not math.isclose(
            published_error,
            standard_error,
            rel_tol=0.0,
            abs_tol=_FLOAT_TOL,
        ):
            raise ValueError(
                f"{label}.{errors_field}.{action_id}: raw-derived mismatch"
            )
        derived_errors[action_id] = standard_error

    derived_max = max(derived_errors.values())
    published_max = _require_finite(
        row.get(max_error_field), label=f"{label}.{max_error_field}"
    )
    if published_max < 0.0 or not math.isclose(
        published_max, derived_max, rel_tol=0.0, abs_tol=_FLOAT_TOL
    ):
        raise ValueError(f"{label}.{max_error_field}: raw-derived mismatch")
    return estimates, counts, derived_errors


def _candidate_state(
    bundle: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, dict[str, float]]],
    set[str],
    set[int],
]:
    artifacts: dict[str, dict[str, Any]] = {}
    profiles: dict[str, dict[str, dict[str, float]]] = {}
    training_roots: set[str] = set()
    training_seeds: set[int] = set()
    for stratum in REQUIRED_STRATA:
        artifact = bundle["strategies"][stratum]
        artifacts[stratum] = artifact
        _profile, distributions, _infosets = verify_public_strategy_profile(
            artifact["strategy_profile"]
        )
        profiles[stratum] = distributions
        training_roots.update(
            row["root_identity_commitment_sha256"]
            for row in artifact["training_roots"]
        )
        training_seeds.add(int(artifact["producer_contract"]["seed"]))
    return artifacts, profiles, training_roots, training_seeds


def _candidate_training_content(
    bundle: Mapping[str, Any],
) -> tuple[set[str], set[str], set[str]]:
    observations: set[str] = set()
    range_contents: set[str] = set()
    range_builds: set[str] = set()
    for stratum in REQUIRED_STRATA:
        for root in bundle["strategies"][stratum]["training_roots"]:
            observations.add(str(root["observation_digest"]))
            range_contents.add(str(root["range_content_sha256"]))
            range_builds.add(str(root["range_build_sha256"]))
    return observations, range_contents, range_builds


def _verify_holdout_rows(
    rows_value: Any,
    *,
    roots_by_id: Mapping[str, Mapping[str, Any]],
    bundle: Mapping[str, Any],
    evaluator: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], set[int], set[int]]:
    if not isinstance(rows_value, list) or not rows_value:
        raise ValueError("raw_holdout_rows: non-empty list required")
    artifacts, profiles, _training_roots, training_seeds = _candidate_state(bundle)
    (
        training_observations,
        training_range_contents,
        training_range_builds,
    ) = _candidate_training_content(bundle)
    bundle_sha = bundle["candidate_bundle_sha256"]
    reference_sha = evaluator["reference_policy_sha256"]
    min_samples = int(thresholds["min_action_payoff_samples_per_action"])
    min_seeds = int(thresholds["min_independent_evaluation_seeds_per_root"])

    rows: list[dict[str, Any]] = []
    run_ids: set[str] = set()
    row_keys: set[tuple[str, int]] = set()
    evaluation_seeds_by_root: dict[str, set[int]] = defaultdict(set)
    evaluation_seeds_by_stratum: dict[str, dict[str, set[int]]] = defaultdict(dict)
    pair_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    payoff_seed_owner: dict[int, tuple[str, int]] = {}
    range_binding_by_root: dict[str, tuple[str, str]] = {}
    evaluation_seeds: set[int] = set()
    payoff_seeds: set[int] = set()

    for index, row_value in enumerate(rows_value):
        label = f"raw_holdout_rows[{index}]"
        row = _require_mapping(row_value, label=label)
        _require_exact_keys(row, _ROW_KEYS, label=label)
        if row.get("row_sha256") != self_hash(row, "row_sha256"):
            raise ValueError(f"{label}.row_sha256: self-hash mismatch")
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"{label}.run_id: non-empty string required")
        if run_id in run_ids:
            raise ValueError(f"{label}.run_id: duplicate")
        run_ids.add(run_id)

        root_id = row.get("root_id")
        root = roots_by_id.get(str(root_id))
        if root is None:
            raise ValueError(f"{label}.root_id: outside locked holdout")
        for field in (
            "root_commitment_sha256",
            "root_identity_commitment_sha256",
            "stratum",
            "actor",
            "visible_joker_count",
            "seat_swap_pair_id",
        ):
            if row.get(field) != root.get(field):
                raise ValueError(f"{label}.{field}: locked root binding mismatch")
        stratum = str(root["stratum"])
        artifact = artifacts[stratum]
        profile = profiles[stratum]
        if row.get("candidate_bundle_sha256") != bundle_sha:
            raise ValueError(f"{label}.candidate_bundle_sha256: mismatch")
        if row.get("candidate_strategy_artifact_sha256") != artifact[
            "artifact_sha256"
        ]:
            raise ValueError(
                f"{label}.candidate_strategy_artifact_sha256: mismatch"
            )
        if row.get("candidate_strategy_sha256") != artifact[
            "average_strategy_sha256"
        ]:
            raise ValueError(f"{label}.candidate_strategy_sha256: mismatch")
        infoset_digest = row.get("policy_infoset_digest")
        if infoset_digest != root["observation_digest"]:
            raise ValueError(f"{label}.policy_infoset_digest: root mismatch")
        if infoset_digest in training_observations:
            raise ValueError(
                f"{label}.policy_infoset_digest: candidate training observation "
                "relabelled as holdout"
            )
        if infoset_digest not in profile:
            raise ValueError(
                f"{label}.policy_infoset_digest: candidate has no exact policy row"
            )
        policy = _distribution(
            row.get("policy_action_distribution"),
            label=f"{label}.policy_action_distribution",
        )
        candidate_policy = profile[str(infoset_digest)]
        if set(policy) != set(candidate_policy) or any(
            not math.isclose(
                policy[action_id],
                candidate_policy[action_id],
                rel_tol=0.0,
                abs_tol=_FLOAT_TOL,
            )
            for action_id in policy
        ):
            raise ValueError(
                f"{label}.policy_action_distribution: candidate content mismatch"
            )
        if row.get("reference_policy_sha256") != reference_sha:
            raise ValueError(f"{label}.reference_policy_sha256: mismatch")
        reference = _distribution(
            row.get("reference_action_distribution"),
            label=f"{label}.reference_action_distribution",
        )
        if set(policy) != set(reference):
            raise ValueError(f"{label}: candidate/reference action sets must match")
        action_ids = set(policy)
        range_content_sha = _require_sha256(
            row.get("range_content_sha256"),
            label=f"{label}.range_content_sha256",
        )
        range_build_sha = _require_sha256(
            row.get("range_build_sha256"),
            label=f"{label}.range_build_sha256",
        )
        if range_content_sha in training_range_contents:
            raise ValueError(
                f"{label}.range_content_sha256: candidate training range "
                "relabelled as holdout"
            )
        if range_build_sha in training_range_builds:
            raise ValueError(
                f"{label}.range_build_sha256: candidate training range build "
                "relabelled as holdout"
            )
        range_binding = (range_content_sha, range_build_sha)
        previous_range_binding = range_binding_by_root.get(str(root_id))
        if previous_range_binding is not None and previous_range_binding != range_binding:
            raise ValueError(f"{label}: holdout root range binding changed across seeds")
        range_binding_by_root[str(root_id)] = range_binding

        payoffs, samples, _candidate_errors = _verify_action_moment_family(
            row,
            label=label,
            prefix="",
            action_ids=action_ids,
            min_samples=min_samples,
        )
        (
            reference_payoffs,
            reference_samples,
            _reference_errors,
        ) = _verify_action_moment_family(
            row,
            label=label,
            prefix="reference_",
            action_ids=action_ids,
            min_samples=min_samples,
        )
        candidate_sample_counts = set(samples.values())
        reference_sample_counts = set(reference_samples.values())
        if (
            len(candidate_sample_counts) != 1
            or len(reference_sample_counts) != 1
            or candidate_sample_counts != reference_sample_counts
        ):
            raise ValueError(
                f"{label}: candidate/reference action moments require one common "
                "paired sample count"
            )
        policy_payoff = _weighted_payoff(policy, payoffs)
        candidate_uniform_payoff = _weighted_payoff(reference, payoffs)
        reference_payoff = _weighted_payoff(reference, reference_payoffs)
        best_payoff = max(payoffs.values())
        regret = max(0.0, best_payoff - policy_payoff)
        derived = {
            "policy_payoff_estimate": policy_payoff,
            "candidate_continuation_uniform_root_payoff_estimate": (
                candidate_uniform_payoff
            ),
            "reference_payoff_estimate": reference_payoff,
            "best_action_payoff_estimate": best_payoff,
            "ev_regret_estimate": regret,
        }
        for field, wanted in derived.items():
            actual = _require_finite(row.get(field), label=f"{label}.{field}")
            if not math.isclose(
                actual, wanted, rel_tol=0.0, abs_tol=_FLOAT_TOL
            ):
                raise ValueError(f"{label}.{field}: raw-derived mismatch")

        paired_label = f"{label}.policy_reference_paired_aggregate"
        paired = _require_mapping(
            row.get("policy_reference_paired_aggregate"), label=paired_label
        )
        _require_exact_keys(
            paired,
            frozenset({"count", "sum", "sum_squares"}),
            label=paired_label,
        )
        paired_count = _require_int(
            paired.get("count"), label=f"{paired_label}.count", minimum=min_samples
        )
        common_count = next(iter(candidate_sample_counts))
        if paired_count != common_count:
            raise ValueError(f"{paired_label}.count: action sample count mismatch")
        paired_sum = _require_finite(
            paired.get("sum"), label=f"{paired_label}.sum"
        )
        paired_squares = _require_finite(
            paired.get("sum_squares"), label=f"{paired_label}.sum_squares"
        )
        if paired_squares < 0.0:
            raise ValueError(f"{paired_label}.sum_squares: nonnegative required")
        paired_mean, paired_standard_error = _aggregate_mean_standard_error(
            count=paired_count,
            total=paired_sum,
            total_squares=paired_squares,
            label=paired_label,
        )
        expected_delta = policy_payoff - reference_payoff
        published_delta = _require_finite(
            row.get("policy_reference_delta_estimate"),
            label=f"{label}.policy_reference_delta_estimate",
        )
        if not math.isclose(
            paired_mean, expected_delta, rel_tol=0.0, abs_tol=_FLOAT_TOL
        ) or not math.isclose(
            published_delta, expected_delta, rel_tol=0.0, abs_tol=_FLOAT_TOL
        ):
            raise ValueError(
                f"{label}.policy_reference_delta_estimate: paired raw-derived mismatch"
            )
        published_delta_error = _require_finite(
            row.get("policy_reference_delta_standard_error"),
            label=f"{label}.policy_reference_delta_standard_error",
        )
        if published_delta_error < 0.0 or not math.isclose(
            published_delta_error,
            paired_standard_error,
            rel_tol=0.0,
            abs_tol=_FLOAT_TOL,
        ):
            raise ValueError(
                f"{label}.policy_reference_delta_standard_error: raw-derived mismatch"
            )

        eligible = row.get("eligible_infoset_digests")
        encountered = row.get("encountered_infoset_digests")
        if not isinstance(eligible, list) or not eligible:
            raise ValueError(f"{label}.eligible_infoset_digests: non-empty list required")
        if not isinstance(encountered, list) or not encountered:
            raise ValueError(
                f"{label}.encountered_infoset_digests: non-empty list required"
            )
        for field, values in (
            ("eligible_infoset_digests", eligible),
            ("encountered_infoset_digests", encountered),
        ):
            if any(not _is_sha256(value) for value in values):
                raise ValueError(f"{label}.{field}: SHA256 entries required")
            if values != sorted(values) or len(values) != len(set(values)):
                raise ValueError(f"{label}.{field}: canonical unique order required")
        if not set(encountered).issubset(set(eligible)):
            raise ValueError(f"{label}: encountered infosets must be eligible")
        if infoset_digest not in encountered:
            raise ValueError(f"{label}: evaluated policy infoset must be encountered")
        eligible_count = len(eligible)
        encountered_count = len(encountered)
        coverage = encountered_count / eligible_count
        if row.get("eligible_infoset_count") != eligible_count:
            raise ValueError(f"{label}.eligible_infoset_count: raw-derived mismatch")
        if row.get("encountered_infoset_count") != encountered_count:
            raise ValueError(f"{label}.encountered_infoset_count: raw-derived mismatch")
        actual_coverage = _require_finite(
            row.get("encountered_infoset_coverage"),
            label=f"{label}.encountered_infoset_coverage",
        )
        if not math.isclose(
            actual_coverage, coverage, rel_tol=0.0, abs_tol=_FLOAT_TOL
        ):
            raise ValueError(
                f"{label}.encountered_infoset_coverage: raw-derived mismatch"
            )
        audits = _require_mapping(row.get("audits"), label=f"{label}.audits")
        if set(audits) != set(REQUIRED_AUDITS):
            raise ValueError(f"{label}.audits: exact audit field set required")
        for field in REQUIRED_AUDITS:
            if audits.get(field) != 0 or isinstance(audits.get(field), bool):
                raise ValueError(f"{label}.audits.{field}: integer zero required")
        _require_finite(row.get("runtime_ms"), label=f"{label}.runtime_ms", positive=True)
        if row.get("exact_exploitability_computed") is not False:
            raise ValueError(f"{label}.exact_exploitability_computed: must be false")

        evaluation_seed = _require_int(
            row.get("evaluation_seed"), label=f"{label}.evaluation_seed", minimum=0
        )
        payoff_seed = _require_int(
            row.get("payoff_sample_seed"),
            label=f"{label}.payoff_sample_seed",
            minimum=0,
        )
        if evaluation_seed == payoff_seed:
            raise ValueError(f"{label}: evaluation/payoff seeds must be independent")
        if evaluation_seed in training_seeds or payoff_seed in training_seeds:
            raise ValueError(f"{label}: candidate training seed reused by holdout")
        row_key = (str(root_id), evaluation_seed)
        if row_key in row_keys:
            raise ValueError(f"{label}: duplicate root/evaluation_seed row")
        row_keys.add(row_key)
        evaluation_seeds.add(evaluation_seed)
        payoff_seeds.add(payoff_seed)
        evaluation_seeds_by_root[str(root_id)].add(evaluation_seed)
        pair_key = (str(root["seat_swap_pair_id"]), evaluation_seed)
        owner = payoff_seed_owner.get(payoff_seed)
        if owner is not None and owner != pair_key:
            raise ValueError(f"{label}.payoff_sample_seed: reused across holdout pairs")
        payoff_seed_owner[payoff_seed] = pair_key
        normalized_row = copy.deepcopy(dict(row))
        pair_groups[pair_key].append(normalized_row)
        rows.append(normalized_row)

    if evaluation_seeds & payoff_seeds:
        raise ValueError("raw_holdout_rows: evaluation/payoff seed sets overlap")
    if training_seeds & (evaluation_seeds | payoff_seeds):
        raise ValueError("raw_holdout_rows: candidate training/holdout seed sets overlap")
    for root_id in roots_by_id:
        seeds = evaluation_seeds_by_root.get(root_id, set())
        if len(seeds) < min_seeds:
            raise ValueError(
                f"raw_holdout_rows: root {root_id!r} needs >= {min_seeds} "
                "independent evaluation seeds"
            )
        stratum = str(roots_by_id[root_id]["stratum"])
        evaluation_seeds_by_stratum[stratum][root_id] = seeds
    for stratum, root_sets in evaluation_seeds_by_stratum.items():
        if len({tuple(sorted(values)) for values in root_sets.values()}) != 1:
            raise ValueError(
                f"raw_holdout_rows: {stratum} roots require one evaluation-seed set"
            )
    expected_keys = {
        (root_id, seed)
        for root_id, seeds in evaluation_seeds_by_root.items()
        for seed in seeds
    }
    if row_keys != expected_keys or set(evaluation_seeds_by_root) != set(roots_by_id):
        raise ValueError("raw_holdout_rows: incomplete locked root/seed coverage")
    for pair_key, pair_rows in pair_groups.items():
        actors = {row["actor"] for row in pair_rows}
        jokers = {row["visible_joker_count"] for row in pair_rows}
        payoff = {row["payoff_sample_seed"] for row in pair_rows}
        if len(pair_rows) != 2 or actors != {"bb", "btn"} or len(jokers) != 1 or len(
            payoff
        ) != 1:
            raise ValueError(
                f"raw_holdout_rows: pair {pair_key!r} requires BB/BTN rows in "
                "one Joker layer with one common payoff seed"
            )
    return rows, evaluation_seeds, payoff_seeds


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def derive_shared_multi_root_strength_metrics(
    evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Re-derive all practical-strength metrics from raw holdout rows."""

    rows = list(evidence["raw_holdout_rows"])
    roots = list(evidence["root_manifest"]["roots"])
    bundle = evidence["candidate_bundle"]
    partitions = evidence["excluded_root_partitions"]
    by_stratum: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    roots_by_stratum: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_root: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for root in roots:
        roots_by_stratum[str(root["stratum"])].append(root)
    for row in rows:
        by_stratum[str(row["stratum"])].append(row)
        by_root[str(row["root_id"])].append(row)
    training_roots = {
        root["root_identity_commitment_sha256"]
        for artifact in bundle["strategies"].values()
        for root in artifact["training_roots"]
    }
    holdout_roots = {
        root["root_identity_commitment_sha256"] for root in roots
    }
    (
        training_observations,
        training_range_contents,
        training_range_builds,
    ) = _candidate_training_content(bundle)
    holdout_observations = {root["observation_digest"] for root in roots}
    holdout_range_contents = {row["range_content_sha256"] for row in rows}
    holdout_range_builds = {row["range_build_sha256"] for row in rows}
    partition_sets = {
        purpose: set(partitions[purpose]["root_commitments"])
        for purpose in REQUIRED_EXCLUDED_PARTITIONS
    }
    excluded_union = set().union(*partition_sets.values())
    pair_groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        pair_groups[(str(row["seat_swap_pair_id"]), int(row["evaluation_seed"]))].append(
            row
        )
    deltas = [float(row["policy_reference_delta_estimate"]) for row in rows]
    pair_deltas = [
        math.fsum(
            float(row["policy_reference_delta_estimate"])
            for row in group
        )
        / len(group)
        for group in pair_groups.values()
    ]
    regrets = [float(row["ev_regret_estimate"]) for row in rows]
    runtimes = [float(row["runtime_ms"]) for row in rows]
    coverages = [float(row["encountered_infoset_coverage"]) for row in rows]
    sample_counts = [
        int(count)
        for row in rows
        for family in (
            row["action_payoff_sample_counts"],
            row["reference_action_payoff_sample_counts"],
        )
        for count in family.values()
    ]
    standard_errors = [
        float(row["max_action_payoff_standard_error"])
        for row in rows
    ]
    reference_standard_errors = [
        float(row["max_reference_action_payoff_standard_error"])
        for row in rows
    ]
    delta_standard_errors = [
        float(row["policy_reference_delta_standard_error"])
        for row in rows
    ]
    audits = {
        field: sum(int(row["audits"][field]) for row in rows)
        for field in REQUIRED_AUDITS
    }
    strata: dict[str, Any] = {}
    for stratum in REQUIRED_STRATA:
        stratum_rows = by_stratum[stratum]
        stratum_regrets = [float(row["ev_regret_estimate"]) for row in stratum_rows]
        stratum_roots = roots_by_stratum[stratum]
        seeds_by_root = [
            len({int(row["evaluation_seed"]) for row in by_root[root["root_id"]]})
            for root in stratum_roots
        ]
        stratum_samples = [
            int(count)
            for row in stratum_rows
            for family in (
                row["action_payoff_sample_counts"],
                row["reference_action_payoff_sample_counts"],
            )
            for count in family.values()
        ]
        strata[stratum] = {
            "root_count": len(stratum_roots),
            "raw_holdout_row_count": len(stratum_rows),
            "min_independent_evaluation_seeds_per_root": min(seeds_by_root),
            "min_action_payoff_samples_per_action": min(stratum_samples),
            "max_action_payoff_standard_error": max(
                float(row["max_action_payoff_standard_error"])
                for row in stratum_rows
            ),
            "max_reference_action_payoff_standard_error": max(
                float(row["max_reference_action_payoff_standard_error"])
                for row in stratum_rows
            ),
            "max_policy_reference_delta_standard_error": max(
                float(row["policy_reference_delta_standard_error"])
                for row in stratum_rows
            ),
            "min_encountered_infoset_coverage": min(
                float(row["encountered_infoset_coverage"])
                for row in stratum_rows
            ),
            "mean_ev_regret_score": math.fsum(stratum_regrets)
            / len(stratum_regrets),
            "p95_ev_regret_score": _nearest_rank(stratum_regrets, 0.95),
            "p99_ev_regret_score": _nearest_rank(stratum_regrets, 0.99),
        }
    return {
        "required_strata": list(REQUIRED_STRATA),
        "strata": strata,
        "global": {
            "candidate_training_root_count": len(training_roots),
            "holdout_root_count": len(holdout_roots),
            "raw_holdout_row_count": len(rows),
            "candidate_holdout_root_overlap_count": len(
                training_roots & holdout_roots
            ),
            "candidate_holdout_observation_overlap_count": len(
                training_observations & holdout_observations
            ),
            "candidate_holdout_range_content_overlap_count": len(
                training_range_contents & holdout_range_contents
            ),
            "candidate_holdout_range_build_overlap_count": len(
                training_range_builds & holdout_range_builds
            ),
            "holdout_excluded_root_overlap_count": len(
                holdout_roots & excluded_union
            ),
            "candidate_training_roots_missing_from_training_partition_count": len(
                training_roots - partition_sets["training"]
            ),
            "excluded_partition_pairwise_overlap_count": sum(
                len(partition_sets[left] & partition_sets[right])
                for left, right in itertools.combinations(
                    REQUIRED_EXCLUDED_PARTITIONS, 2
                )
            ),
            **audits,
            "min_action_payoff_samples_per_action": min(sample_counts),
            "max_action_payoff_standard_error": max(standard_errors),
            "max_reference_action_payoff_standard_error": max(
                reference_standard_errors
            ),
            "max_policy_reference_delta_standard_error": max(
                delta_standard_errors
            ),
            "min_encountered_infoset_coverage": min(coverages),
            "mean_ev_regret_score": math.fsum(regrets) / len(regrets),
            "p95_ev_regret_score": _nearest_rank(regrets, 0.95),
            "p99_ev_regret_score": _nearest_rank(regrets, 0.99),
            "paired_seat_swap_comparison_count": len(pair_groups),
            "paired_seat_swap_min_single_seat_delta_score": min(deltas),
            "paired_seat_swap_min_pair_mean_delta_score": min(pair_deltas),
            "runtime_ms_p95": _nearest_rank(runtimes, 0.95),
            "runtime_ms_max": max(runtimes),
            "runtime_ms_total": math.fsum(runtimes),
            "exact_exploitability_computed": False,
        },
    }


def _threshold_failures(
    metrics: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> list[str]:
    failures: list[str] = []
    for stratum in REQUIRED_STRATA:
        values = metrics["strata"][stratum]
        checks = {
            "root_count": values["root_count"]
            >= thresholds["min_roots_per_stratum"],
            "min_independent_evaluation_seeds_per_root": values[
                "min_independent_evaluation_seeds_per_root"
            ]
            >= thresholds["min_independent_evaluation_seeds_per_root"],
            "min_action_payoff_samples_per_action": values[
                "min_action_payoff_samples_per_action"
            ]
            >= thresholds["min_action_payoff_samples_per_action"],
            "max_action_payoff_standard_error": values[
                "max_action_payoff_standard_error"
            ]
            <= thresholds["max_action_payoff_standard_error"],
            "max_reference_action_payoff_standard_error": values[
                "max_reference_action_payoff_standard_error"
            ]
            <= thresholds["max_reference_action_payoff_standard_error"],
            "max_policy_reference_delta_standard_error": values[
                "max_policy_reference_delta_standard_error"
            ]
            <= thresholds["max_policy_reference_delta_standard_error"],
            "min_encountered_infoset_coverage": values[
                "min_encountered_infoset_coverage"
            ]
            >= thresholds["min_encountered_infoset_coverage"],
            "mean_ev_regret_score": values["mean_ev_regret_score"]
            <= thresholds["max_mean_ev_regret_score"],
            "p95_ev_regret_score": values["p95_ev_regret_score"]
            <= thresholds["max_p95_ev_regret_score"],
            "p99_ev_regret_score": values["p99_ev_regret_score"]
            <= thresholds["max_p99_ev_regret_score"],
        }
        failures.extend(
            f"derived_metrics.strata.{stratum}.{field}: threshold failed"
            for field, passed in checks.items()
            if not passed
        )
    global_values = metrics["global"]
    for field in (
        "candidate_holdout_root_overlap_count",
        "candidate_holdout_observation_overlap_count",
        "candidate_holdout_range_content_overlap_count",
        "candidate_holdout_range_build_overlap_count",
        "holdout_excluded_root_overlap_count",
        "candidate_training_roots_missing_from_training_partition_count",
        "excluded_partition_pairwise_overlap_count",
        *REQUIRED_AUDITS,
    ):
        if global_values[field] != 0:
            failures.append(f"derived_metrics.global.{field}: must equal zero")
    global_checks = {
        "min_action_payoff_samples_per_action": global_values[
            "min_action_payoff_samples_per_action"
        ]
        >= thresholds["min_action_payoff_samples_per_action"],
        "max_action_payoff_standard_error": global_values[
            "max_action_payoff_standard_error"
        ]
        <= thresholds["max_action_payoff_standard_error"],
        "max_reference_action_payoff_standard_error": global_values[
            "max_reference_action_payoff_standard_error"
        ]
        <= thresholds["max_reference_action_payoff_standard_error"],
        "max_policy_reference_delta_standard_error": global_values[
            "max_policy_reference_delta_standard_error"
        ]
        <= thresholds["max_policy_reference_delta_standard_error"],
        "min_encountered_infoset_coverage": global_values[
            "min_encountered_infoset_coverage"
        ]
        >= thresholds["min_encountered_infoset_coverage"],
        "mean_ev_regret_score": global_values["mean_ev_regret_score"]
        <= thresholds["max_mean_ev_regret_score"],
        "p95_ev_regret_score": global_values["p95_ev_regret_score"]
        <= thresholds["max_p95_ev_regret_score"],
        "p99_ev_regret_score": global_values["p99_ev_regret_score"]
        <= thresholds["max_p99_ev_regret_score"],
        "paired_seat_swap_min_single_seat_delta_score": global_values[
            "paired_seat_swap_min_single_seat_delta_score"
        ]
        >= -thresholds["paired_seat_swap_noninferiority_margin_score"],
        "paired_seat_swap_min_pair_mean_delta_score": global_values[
            "paired_seat_swap_min_pair_mean_delta_score"
        ]
        >= -thresholds["paired_seat_swap_noninferiority_margin_score"],
        "runtime_ms_p95": global_values["runtime_ms_p95"]
        <= thresholds["max_runtime_ms_p95"],
        "runtime_ms_max": global_values["runtime_ms_max"]
        <= thresholds["max_runtime_ms_max"],
        "runtime_ms_total": global_values["runtime_ms_total"]
        <= thresholds["max_runtime_ms_total"],
    }
    failures.extend(
        f"derived_metrics.global.{field}: threshold failed"
        for field, passed in global_checks.items()
        if not passed
    )
    return failures


def build_shared_multi_root_strength_evidence(
    *,
    config: Mapping[str, Any],
    candidate_bundle: Mapping[str, Any],
    root_manifest: Mapping[str, Any],
    excluded_root_partitions: Mapping[str, Mapping[str, Any]],
    evaluator_manifest: Mapping[str, Any],
    raw_holdout_rows: Sequence[Mapping[str, Any]],
    production_promotion_claim: bool = False,
) -> dict[str, Any]:
    """Build nonpromoting algorithm-validation evidence."""

    if not isinstance(production_promotion_claim, bool):
        raise TypeError("production_promotion_claim: bool required")
    locked = verify_locked_shared_multi_root_strength_config(config)
    bundle = verify_shared_multi_root_candidate_bundle(candidate_bundle)
    roots, _by_id, _counts = _verify_root_manifest(
        root_manifest,
        min_roots_per_stratum=int(locked["thresholds"]["min_roots_per_stratum"]),
    )
    evaluator = verify_holdout_evaluator_manifest(evaluator_manifest)
    partitions = {
        purpose: _verify_partition(purpose, excluded_root_partitions[purpose])[0]
        for purpose in REQUIRED_EXCLUDED_PARTITIONS
    }
    rows = [copy.deepcopy(dict(row)) for row in raw_holdout_rows]
    evidence: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "gate_id": GATE_ID,
        "scope": SCOPE,
        "evidence_kind": EVIDENCE_KIND,
        "production_promotion_claim": production_promotion_claim,
        "exact_exploitability_computed": False,
        "gate_config_sha256": locked["config_sha256"],
        "candidate_bundle": bundle,
        "candidate_bundle_sha256": bundle["candidate_bundle_sha256"],
        "root_manifest": roots,
        "excluded_root_partitions": partitions,
        "evaluator_manifest": evaluator,
        "evaluator_manifest_sha256": evaluator["manifest_sha256"],
        "raw_holdout_rows": rows,
        "published_summary": {},
    }
    evidence["published_summary"] = derive_shared_multi_root_strength_metrics(
        evidence
    )
    evidence["artifact_sha256"] = self_hash(evidence, "artifact_sha256")
    return evidence


def _failure_result(
    *,
    failures: Sequence[str],
    config_sha256: str | None,
    evidence_sha256: str | None,
    candidate_bundle_sha256: str | None,
    derived_metrics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metrics = copy.deepcopy(dict(derived_metrics or {}))
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "gate_id": GATE_ID,
        "scope": SCOPE,
        "passed": False,
        "status": FAIL_STATUS,
        "promotion_eligible": False,
        "full_card_policy_promoted": False,
        "shared_multi_root_strength_promoted": False,
        "independent_holdout_replayed": bool(metrics),
        "strategic_strength_evaluated": bool(metrics),
        "exact_exploitability_computed": False,
        "config_sha256": config_sha256,
        "evidence_sha256": evidence_sha256,
        "candidate_bundle_sha256": candidate_bundle_sha256,
        "derived_metrics": metrics,
        "failures": list(failures),
    }
    result["result_sha256"] = self_hash(result, "result_sha256")
    return result


def validate_shared_multi_root_strength_evidence(
    evidence: Any, *, config: Any
) -> dict[str, Any]:
    """Fresh-replay the candidate, split, raw rows, and strength thresholds."""

    failures: list[str] = []
    try:
        locked = verify_locked_shared_multi_root_strength_config(config)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        locked = None
        failures.append(f"config: {exc}")
    config_sha = locked["config_sha256"] if locked is not None else None
    try:
        evidence_sha = canonical_sha256(evidence)
    except (TypeError, ValueError, OverflowError):
        evidence_sha = None
        failures.append("evidence: finite canonical JSON required")
    if not isinstance(evidence, Mapping):
        failures.append("evidence: object required")
        return _failure_result(
            failures=failures,
            config_sha256=config_sha,
            evidence_sha256=evidence_sha,
            candidate_bundle_sha256=None,
            derived_metrics=None,
        )
    raw = evidence
    try:
        _require_exact_keys(raw, _EVIDENCE_KEYS, label="evidence")
    except ValueError as exc:
        failures.append(str(exc))
    expected = {
        "schema": EVIDENCE_SCHEMA,
        "gate_id": GATE_ID,
        "scope": SCOPE,
        "evidence_kind": EVIDENCE_KIND,
        "exact_exploitability_computed": False,
    }
    for field, wanted in expected.items():
        if raw.get(field) != wanted:
            failures.append(f"evidence.{field}: must equal {wanted!r}")
    claim = raw.get("production_promotion_claim")
    if not isinstance(claim, bool):
        failures.append("evidence.production_promotion_claim: bool required")
    try:
        expected_artifact_sha = self_hash(raw, "artifact_sha256")
    except (TypeError, ValueError, OverflowError):
        expected_artifact_sha = None
        failures.append("evidence.artifact_sha256: finite canonical content required")
    if raw.get("artifact_sha256") != expected_artifact_sha:
        failures.append("evidence.artifact_sha256: self-hash mismatch")
    if raw.get("gate_config_sha256") != config_sha:
        failures.append("evidence.gate_config_sha256: locked config mismatch")

    bundle: dict[str, Any] | None = None
    bundle_sha = raw.get("candidate_bundle_sha256")
    try:
        bundle = verify_shared_multi_root_candidate_bundle(
            raw.get("candidate_bundle")
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        failures.append(f"candidate_bundle: {exc}")
    else:
        if bundle_sha != bundle["candidate_bundle_sha256"]:
            failures.append("evidence.candidate_bundle_sha256: content mismatch")
        if locked is not None and bundle_sha != locked[
            "approved_candidate_bundle_sha256"
        ]:
            failures.append("candidate_bundle: not approved by locked config")

    partitions: dict[str, dict[str, Any]] = {}
    partition_sets: dict[str, set[str]] = {}
    partition_values = raw.get("excluded_root_partitions")
    if not isinstance(partition_values, Mapping) or set(partition_values) != set(
        REQUIRED_EXCLUDED_PARTITIONS
    ):
        failures.append("excluded_root_partitions: exact partition set required")
        partition_values = {}
    for purpose in REQUIRED_EXCLUDED_PARTITIONS:
        try:
            partition, identities = _verify_partition(
                purpose, partition_values.get(purpose)
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            failures.append(f"excluded_root_partitions.{purpose}: {exc}")
            continue
        partitions[purpose] = partition
        partition_sets[purpose] = identities
        if locked is not None and partition["manifest_sha256"] != locked[
            "approved_excluded_root_partition_sha256"
        ][purpose]:
            failures.append(
                f"excluded_root_partitions.{purpose}: not approved by locked config"
            )
    for left, right in itertools.combinations(REQUIRED_EXCLUDED_PARTITIONS, 2):
        if partition_sets.get(left, set()) & partition_sets.get(right, set()):
            failures.append(
                f"excluded_root_partitions: {left}/{right} root overlap"
            )

    root_manifest: dict[str, Any] | None = None
    roots_by_id: dict[str, dict[str, Any]] = {}
    try:
        root_manifest, roots_by_id, _counts = _verify_root_manifest(
            raw.get("root_manifest"),
            min_roots_per_stratum=(
                int(locked["thresholds"]["min_roots_per_stratum"])
                if locked is not None
                else 1
            ),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        failures.append(f"root_manifest: {exc}")
    else:
        if locked is not None and root_manifest["root_manifest_sha256"] != locked[
            "approved_holdout_root_manifest_sha256"
        ]:
            failures.append("root_manifest: not approved by locked config")
        expected_partition_hashes = {
            purpose: partitions[purpose]["manifest_sha256"]
            for purpose in REQUIRED_EXCLUDED_PARTITIONS
            if purpose in partitions
        }
        if len(expected_partition_hashes) == len(REQUIRED_EXCLUDED_PARTITIONS) and root_manifest[
            "excluded_root_partition_sha256"
        ] != expected_partition_hashes:
            failures.append("root_manifest: excluded partition binding mismatch")

    evaluator: dict[str, Any] | None = None
    try:
        evaluator = verify_holdout_evaluator_manifest(
            raw.get("evaluator_manifest")
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        failures.append(f"evaluator_manifest: {exc}")
    else:
        if raw.get("evaluator_manifest_sha256") != evaluator["manifest_sha256"]:
            failures.append("evidence.evaluator_manifest_sha256: content mismatch")
        if locked is not None and evaluator["manifest_sha256"] != locked[
            "approved_evaluator_manifest_sha256"
        ]:
            failures.append("evaluator_manifest: not approved by locked config")
        if bundle is not None and evaluator["candidate_bundle_sha256"] != bundle[
            "candidate_bundle_sha256"
        ]:
            failures.append("evaluator_manifest: candidate bundle binding mismatch")
        if root_manifest is not None and evaluator["root_manifest_sha256"] != root_manifest[
            "root_manifest_sha256"
        ]:
            failures.append("evaluator_manifest: root manifest binding mismatch")

    training_roots: set[str] = set()
    training_seeds: set[int] = set()
    training_observations: set[str] = set()
    if bundle is not None:
        _artifacts, _profiles, training_roots, training_seeds = _candidate_state(
            bundle
        )
        training_observations, _range_contents, _range_builds = (
            _candidate_training_content(bundle)
        )
        if "training" in partition_sets and not training_roots.issubset(
            partition_sets["training"]
        ):
            failures.append(
                "candidate_bundle: training roots missing from training partition"
            )
    holdout_roots = {
        root["root_identity_commitment_sha256"] for root in roots_by_id.values()
    }
    overlap = training_roots & holdout_roots
    if overlap:
        failures.append(
            f"root_manifest: {len(overlap)} holdout roots reused by candidate training"
        )
    holdout_observations = {
        root["observation_digest"] for root in roots_by_id.values()
    }
    observation_overlap = training_observations & holdout_observations
    if observation_overlap:
        failures.append(
            "root_manifest: "
            f"{len(observation_overlap)} candidate training observations "
            "relabelled as holdout"
        )
    for purpose, identities in partition_sets.items():
        overlap = holdout_roots & identities
        if overlap:
            failures.append(
                f"root_manifest: {len(overlap)} holdout roots overlap {purpose} partition"
            )

    rows: list[dict[str, Any]] = []
    rows_valid = False
    if bundle is not None and evaluator is not None and roots_by_id and locked is not None:
        try:
            rows, evaluation_seeds, payoff_seeds = _verify_holdout_rows(
                raw.get("raw_holdout_rows"),
                roots_by_id=roots_by_id,
                bundle=bundle,
                evaluator=evaluator,
                thresholds=locked["thresholds"],
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            failures.append(f"raw_holdout_rows: {exc}")
        else:
            rows_valid = True
            if training_seeds & (evaluation_seeds | payoff_seeds):
                failures.append("seed partitions: candidate/holdout overlap")
    elif not isinstance(raw.get("raw_holdout_rows"), list) or not raw.get(
        "raw_holdout_rows"
    ):
        failures.append("raw_holdout_rows: non-empty list required")

    derived_metrics: dict[str, Any] = {}
    if rows_valid and bundle is not None and root_manifest is not None and len(
        partitions
    ) == len(REQUIRED_EXCLUDED_PARTITIONS):
        normalized_for_metrics = dict(raw)
        normalized_for_metrics["candidate_bundle"] = bundle
        normalized_for_metrics["root_manifest"] = root_manifest
        normalized_for_metrics["excluded_root_partitions"] = partitions
        normalized_for_metrics["raw_holdout_rows"] = rows
        try:
            derived_metrics = derive_shared_multi_root_strength_metrics(
                normalized_for_metrics
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            failures.append(f"derived_metrics: cannot derive ({exc})")
        else:
            if raw.get("published_summary") != derived_metrics:
                failures.append(
                    "published_summary: must exactly equal raw-derived metrics"
                )
            if locked is not None:
                failures.extend(
                    _threshold_failures(derived_metrics, locked["thresholds"])
                )
    if claim is False:
        failures.append("production promotion claim is false")
    failures.append(ALGORITHM_VALIDATION_ONLY_FAILURE)
    allowed_nonpromotion_failures = {ALGORITHM_VALIDATION_ONLY_FAILURE}
    if claim is False:
        allowed_nonpromotion_failures.add("production promotion claim is false")
    algorithm_validation_only = bool(derived_metrics) and set(failures).issubset(
        allowed_nonpromotion_failures
    )
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "gate_id": GATE_ID,
        "scope": SCOPE,
        "passed": False,
        "status": (
            ALGORITHM_VALIDATION_ONLY_STATUS
            if algorithm_validation_only
            else FAIL_STATUS
        ),
        "promotion_eligible": False,
        "full_card_policy_promoted": False,
        "shared_multi_root_strength_promoted": False,
        "independent_holdout_replayed": bool(derived_metrics),
        "strategic_strength_evaluated": bool(derived_metrics),
        "exact_exploitability_computed": False,
        "config_sha256": config_sha,
        "evidence_sha256": evidence_sha,
        "candidate_bundle_sha256": bundle_sha if _is_sha256(bundle_sha) else None,
        "derived_metrics": derived_metrics,
        "failures": failures,
    }
    result["result_sha256"] = self_hash(result, "result_sha256")
    return result


validate_t3_shared_multi_root_strength_evidence = (
    validate_shared_multi_root_strength_evidence
)


__all__ = [
    "ALGORITHM_VALIDATION_ONLY_FAILURE",
    "ALGORITHM_VALIDATION_ONLY_STATUS",
    "CANDIDATE_BUNDLE_SCHEMA",
    "CONFIG_SCHEMA",
    "EVALUATOR_MANIFEST_SCHEMA",
    "EVIDENCE_KIND",
    "EVIDENCE_SCHEMA",
    "GATE_ID",
    "HOLDOUT_EVALUATOR_METHOD",
    "MULTI_ROOT_SOLVER_METHOD",
    "RESULT_SCHEMA",
    "SCOPE",
    "STRATEGY_ARTIFACT_SCHEMA",
    "build_excluded_root_partition",
    "build_holdout_evaluator_manifest",
    "build_holdout_root_manifest",
    "build_holdout_root_record",
    "build_locked_shared_multi_root_strength_config",
    "build_shared_multi_root_candidate_bundle",
    "build_shared_multi_root_strategy_artifact",
    "build_shared_multi_root_strength_evidence",
    "derive_shared_multi_root_strength_metrics",
    "validate_shared_multi_root_strength_evidence",
    "validate_t3_shared_multi_root_strength_evidence",
    "verify_holdout_evaluator_manifest",
    "verify_locked_shared_multi_root_strength_config",
    "verify_public_strategy_profile",
    "verify_shared_multi_root_candidate_bundle",
    "verify_shared_multi_root_strategy_artifact",
]
