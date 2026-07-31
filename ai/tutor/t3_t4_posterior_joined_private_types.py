"""Production boundary for posterior-joined T3/T4 public-root mixtures.

The public-only generator emits an unbiased *proposal* over compatible actor
private types.  It deliberately does not emit a posterior.  This module is the
only compiler boundary that may authorize those proposals as an MCCFR prior:

* the exact proposal batch and every private-type commitment are replayed;
* a repository-owned promoted, no-fallback behavior runtime is required;
* the actor's observed public-action likelihood is freshly evaluated for each
  proposal;
* the opponent conditional :class:`FullCardRange` is freshly built and
  independently verified for the same proposal and behavior runtime; and
* the actor-type likelihoods are normalized with exact ``Fraction`` math.

The older explicit finite-support compiler remains useful algorithm-validation
wiring.  Its result is embedded here, but it remains explicitly unauthorized
for production on its own.  Only the content-bound wrapper returned by
``compile_posterior_joined_public_root_mixture`` carries posterior-join
authorization.  Strategic-strength promotion and serving-default changes are
still outside this module.

Current verified behavior coverage closes T3 BB and T3 BTN roots.  The
repository does not yet have verified T3-BTN/T4 likelihood routes needed for
T4 public beliefs, so both T4 phases fail closed at this boundary.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import InitVar, dataclass, field
from fractions import Fraction
from functools import wraps
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

import ai.tutor.calibrated_behavior_runtime as _calibrated_runtime_module
import ai.tutor.t3_bb_fixed_point_runtime as _fixed_point_runtime_module
import ai.tutor.t3_hu_full_card_range as _range_module
import ai.tutor.t3_hu_multi_root_mccfr as _multi_root_module
import ai.tutor.t3_t4_counterfactual_private_types as _proposal_module
import ai.tutor.t3_t4_public_root_mixture as _mixture_module
from ai.tutor.calibrated_behavior_runtime import (
    VerifiedCalibratedHuT1T2BehaviorDispatch,
)
from ai.tutor.t3_bb_fixed_point_runtime import (
    VerifiedM3BehaviorLikelihoodDispatch,
)
from ai.tutor.t3_hu_full_card_range import FullCardRange, FrozenBehaviorModel
from ai.tutor.t3_t4_counterfactual_private_types import (
    CounterfactualPrivateTypeProposal,
    PrivacySafeCounterfactualPrivateTypeProposalBatch,
)
from ai.tutor.t3_t4_public_root_mixture import (
    CompiledPublicRootMixture,
    ExplicitHypotheticalPrivateType,
    PublicRootContext,
)


POSTERIOR_JOIN_SCHEMA = "ofc_t3_t4_posterior_joined_public_root_mixture/v1"
POSTERIOR_JOIN_AUTHORIZATION_CLASS = (
    "proposal_origin_promoted_behavior_conditional_range_posterior_join_v1"
)
POSTERIOR_JOIN_EPSILON = Fraction(1, 1000)
POSTERIOR_JOIN_MAX_PARTICLES = 2048
POSTERIOR_JOIN_RANGE_SEED_DERIVATION = (
    "sha256_public_proposal_batch_and_private_type_commitment_low63_v1"
)

_CONSTRUCTION_TOKEN = object()
_PHASE_BEHAVIOR_REQUIREMENT = MappingProxyType(
    {
        "t3_first": "verified_promoted_calibrated_t1_t2_no_fallback",
        "t3_second": "verified_promoted_t1_t2_plus_fixed_point_t3_bb_no_fallback",
        "t4_first": "unavailable_requires_promoted_t3_btn_route",
        "t4_second": "unavailable_requires_promoted_t3_btn_and_t4_bb_routes",
    }
)
_SOURCE_PATHS = MappingProxyType(
    {
        "posterior_join_compiler": str(Path(__file__).resolve()),
        "proposal_generator": str(Path(_proposal_module.__file__).resolve()),
        "explicit_mixture_compiler": str(Path(_mixture_module.__file__).resolve()),
        "full_card_range": str(Path(_range_module.__file__).resolve()),
        "multi_root_runtime_graph": str(Path(_multi_root_module.__file__).resolve()),
        "calibrated_behavior_runtime": str(Path(
            _calibrated_runtime_module.__file__
        ).resolve()),
        "t3_bb_fixed_point_runtime": str(Path(
            _fixed_point_runtime_module.__file__
        ).resolve()),
    }
)


class PosteriorJoinError(ValueError):
    """Proposal origin, behavior provenance, or posterior replay is invalid."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _live_source_sha256s() -> dict[str, str]:
    return {
        name: _file_sha256(path) for name, path in sorted(_SOURCE_PATHS.items())
    }


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PosteriorJoinError(f"{label} must be a lowercase SHA256")
    return value


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _validated_promoted_behavior(
    context: PublicRootContext,
    behavior_model: FrozenBehaviorModel,
) -> tuple[str, str, dict[str, Any]]:
    """Accept only canonical promoted runtimes with phase-complete routes."""

    model_type = type(behavior_model)
    if context.phase == "t3_first":
        allowed = (
            VerifiedCalibratedHuT1T2BehaviorDispatch,
            VerifiedM3BehaviorLikelihoodDispatch,
        )
    elif context.phase == "t3_second":
        allowed = (VerifiedM3BehaviorLikelihoodDispatch,)
    else:
        requirement = _PHASE_BEHAVIOR_REQUIREMENT[context.phase]
        raise PosteriorJoinError(
            f"{context.phase} production posterior join is unavailable: {requirement}"
        )
    if model_type not in allowed:
        raise TypeError(
            "production posterior join requires the exact repository-owned "
            f"promoted behavior runtime for {context.phase}"
        )

    model_id, model_sha256, model_manifest = (
        _CANONICAL_VALIDATE_BEHAVIOR_IDENTITY(behavior_model)
    )
    snapshot = json.loads(_canonical_json(dict(model_manifest)))
    if snapshot.get("promotion_eligible") is not True:
        raise PosteriorJoinError("behavior runtime is not promotion eligible")
    if snapshot.get("no_fallback") is not True:
        raise PosteriorJoinError("behavior runtime must forbid fallback")
    if snapshot.get("position_contract_version") != context.contract_version:
        raise PosteriorJoinError("behavior runtime position contract mismatch")
    if model_id != context.bindings.behavior_model_id:
        raise PosteriorJoinError("behavior runtime ID does not match public context")
    if model_sha256 != context.bindings.behavior_model_sha256:
        raise PosteriorJoinError("behavior runtime hash does not match public context")
    if context.phase == "t3_second" and snapshot.get(
        "m3_t3_root_behavior_complete"
    ) is not True:
        raise PosteriorJoinError(
            "T3 second posterior requires the verified fixed-point T3-BB route"
        )
    return model_id, model_sha256, snapshot


def _actor_history_likelihood(
    proposal: CounterfactualPrivateTypeProposal,
    behavior_model: FrozenBehaviorModel,
) -> tuple[Fraction, tuple[Mapping[str, Any], ...]]:
    observation = proposal.observation
    actor_turns = tuple(
        turn
        for turn, actor, _placements in observation.public_action_history
        if actor == observation.actor and turn > 0
    )
    recall_turns = tuple(
        turn for turn, _card in observation.own_recall.discards_by_turn
    )
    if recall_turns != actor_turns:
        raise PosteriorJoinError(
            "proposal actor recall does not cover the observed actor history exactly"
        )
    discard_by_turn = dict(observation.own_recall.discards_by_turn)
    likelihood, audits = _CANONICAL_BEHAVIOR_LIKELIHOOD(
        observation,
        opponent=observation.actor,
        discard_by_turn=discard_by_turn,
        behavior_model=behavior_model,
        behavior_cache={},
        epsilon=POSTERIOR_JOIN_EPSILON,
    )
    if likelihood <= 0:
        raise PosteriorJoinError(
            "promoted behavior assigns zero likelihood to a proposed actor history"
        )
    if len(audits) != len(actor_turns):
        raise PosteriorJoinError(
            "actor behavior audit does not cover every observed actor action"
        )
    if any(audit.get("used_fallback") is not False for audit in audits):
        raise PosteriorJoinError("actor history likelihood used a fallback route")
    if any(audit.get("source") not in ("model", "table") for audit in audits):
        raise PosteriorJoinError(
            "actor history likelihood did not come from a promoted model/table route"
        )
    return likelihood, audits


def _conditional_range_seed(
    proposal_batch: PrivacySafeCounterfactualPrivateTypeProposalBatch,
    proposal: CounterfactualPrivateTypeProposal,
) -> int:
    payload = {
        "schema": "ofc_t3_t4_posterior_join_range_seed/v1",
        "proposal_batch_audit_manifest_sha256": (
            proposal_batch.audit_manifest_sha256
        ),
        "privacy_safe_traversal_derivation_commitment_sha256": (
            proposal_batch.traversal_derivation_commitment_sha256
        ),
        "private_type_commitment": proposal.private_type_commitment,
        "derivation": POSTERIOR_JOIN_RANGE_SEED_DERIVATION,
    }
    # Keep the value inside a conventional signed-positive 63-bit seed range.
    return int(_canonical_sha256(payload)[:16], 16) & ((1 << 63) - 1)


def _fresh_conditional_range(
    proposal_batch: PrivacySafeCounterfactualPrivateTypeProposalBatch,
    proposal: CounterfactualPrivateTypeProposal,
    behavior_model: FrozenBehaviorModel,
) -> FullCardRange:
    seed = _conditional_range_seed(proposal_batch, proposal)
    result = _CANONICAL_BUILD_RANGE(
        proposal.observation,
        behavior_model,
        epsilon=POSTERIOR_JOIN_EPSILON,
        max_particles=POSTERIOR_JOIN_MAX_PARTICLES,
        seed=seed,
    )
    verification = _CANONICAL_VERIFY_RANGE(proposal.observation, result)
    if verification.get("verified") is not True:
        raise PosteriorJoinError("conditional FullCard range did not verify")
    if not isinstance(result.evidence_normalizer, Fraction) or (
        result.evidence_normalizer <= 0
    ):
        raise PosteriorJoinError(
            "conditional range requires a positive exact evidence normalizer"
        )
    build_manifest = result.metadata.get("build_manifest")
    model_manifest = result.metadata.get("behavior_model_manifest")
    if not isinstance(build_manifest, Mapping) or not isinstance(
        model_manifest, Mapping
    ):
        raise PosteriorJoinError("conditional range provenance is incomplete")
    if build_manifest.get("behavior_uniform_fallback_count") != 0:
        raise PosteriorJoinError("conditional range used fallback behavior")
    if any(
        row.get("used_fallback") is not False
        or row.get("source") not in ("model", "table")
        for row in build_manifest.get("behavior_query_audit", ())
        if isinstance(row, Mapping)
    ):
        raise PosteriorJoinError(
            "conditional range contains an unpromoted behavior query route"
        )
    if model_manifest.get("promotion_eligible") is not True:
        raise PosteriorJoinError(
            "conditional range is not bound to a promoted behavior runtime"
        )
    if model_manifest.get("no_fallback") is not True:
        raise PosteriorJoinError(
            "conditional range behavior manifest does not forbid fallback"
        )
    if result.behavior_model_sha256 != behavior_model.model_sha256:
        raise PosteriorJoinError("conditional range behavior hash mismatch")
    return result


def _support_row(
    proposal: CounterfactualPrivateTypeProposal,
    *,
    proposal_batch: PrivacySafeCounterfactualPrivateTypeProposalBatch,
    likelihood: Fraction,
    joint_unnormalized_weight: Fraction,
    prior_mass: Fraction,
    actor_audits: tuple[Mapping[str, Any], ...],
    root_range: FullCardRange,
) -> dict[str, Any]:
    build_manifest = root_range.metadata["build_manifest"]
    return {
        "hypothetical_private_type_commitment": proposal.private_type_commitment,
        "proposal_sample_ordinal": proposal.sample_ordinal,
        "proposal_origin_verified": True,
        "actor_history_likelihood_exact": _fraction_text(likelihood),
        "opponent_public_evidence_normalizer_exact": _fraction_text(
            root_range.evidence_normalizer
        ),
        "joint_unnormalized_actor_type_weight_exact": _fraction_text(
            joint_unnormalized_weight
        ),
        "posterior_prior_mass_exact": _fraction_text(prior_mass),
        "actor_history_behavior_query_count": len(actor_audits),
        "actor_history_uniform_fallback_count": 0,
        "conditional_range_content_sha256": root_range.range_content_sha256,
        "conditional_range_build_sha256": root_range.range_build_sha256,
        "conditional_range_behavior_model_sha256": (
            root_range.behavior_model_sha256
        ),
        "conditional_range_particle_count": root_range.particle_count,
        "conditional_range_posterior_scope": build_manifest["posterior_scope"],
        "conditional_range_evidence_scope": build_manifest[
            "opponent_public_evidence_scope"
        ],
        "conditional_range_evidence_sampled_assignment_count": build_manifest[
            "sampled_assignment_count"
        ],
        "conditional_range_evidence_total_assignment_count": build_manifest[
            "candidate_assignment_count"
        ],
        "conditional_range_hidden_assignment_exhaustive": build_manifest[
            "exhaustive_hidden_discard_enumeration"
        ],
        "conditional_range_uniform_fallback_count": 0,
        "range_seed_sha256": hashlib.sha256(
            str(_conditional_range_seed(proposal_batch, proposal)).encode("utf-8")
        ).hexdigest(),
    }


def _audit_manifest(
    context: PublicRootContext,
    proposal_batch: PrivacySafeCounterfactualPrivateTypeProposalBatch,
    *,
    behavior_model_id: str,
    behavior_model_sha256: str,
    behavior_manifest: Mapping[str, Any],
    rows: tuple[Mapping[str, Any], ...],
    compiled: CompiledPublicRootMixture,
    runtime_semantic_binding_sha256: str,
) -> dict[str, Any]:
    sampled_exhaustive = (
        proposal_batch.start_ordinal == 0
        and proposal_batch.sample_count == proposal_batch.universe_size
    )
    conditional_ranges_exhaustive = all(
        row["conditional_range_hidden_assignment_exhaustive"] for row in rows
    )
    source_sha256s = _live_source_sha256s()
    return {
        "schema": POSTERIOR_JOIN_SCHEMA,
        "authorization_class": POSTERIOR_JOIN_AUTHORIZATION_CLASS,
        "public_context": context.to_canonical_dict(),
        "public_context_digest": context.digest(),
        "proposal_origin": {
            "proposal_batch_schema": proposal_batch.audit_manifest["schema"],
            "proposal_batch_audit_manifest_sha256": (
                proposal_batch.audit_manifest_sha256
            ),
            "privacy_safe_traversal_derivation_commitment_sha256": (
                proposal_batch.traversal_derivation_commitment_sha256
            ),
            "repository_owned_schedule_id": proposal_batch.schedule_id,
            "proposal_universe_sha256": proposal_batch.audit_manifest[
                "universe_sha256"
            ],
            "proposal_support_size": proposal_batch.sample_count,
            "proposal_support_exhaustive": sampled_exhaustive,
            "proposal_coverage_fraction_exact": proposal_batch.audit_manifest[
                "privacy_safe_traversal"
            ]["coverage_fraction_exact"],
            "commitment_order_sha256": proposal_batch.audit_manifest[
                "privacy_safe_traversal"
            ][
                "sample_commitment_order_sha256"
            ],
            "origin_commitments_freshly_verified": True,
            "privacy_safe_origin_verified": True,
            "caller_supplied_entropy_consumed": False,
            "generic_seed_plan_consumed": False,
            "generic_batch_conversion_consumed": False,
            "actual_actor_private_cards_consumed": False,
            "hidden_derived_seed_postselection_allowed": False,
        },
        "behavior_provenance": {
            "behavior_model_id": behavior_model_id,
            "behavior_model_sha256": behavior_model_sha256,
            "behavior_manifest_sha256": _canonical_sha256(dict(behavior_manifest)),
            "behavior_model_type": behavior_manifest.get("model_type"),
            "promotion_eligible": True,
            "no_fallback": True,
            "phase_requirement": _PHASE_BEHAVIOR_REQUIREMENT[context.phase],
            "actor_history_likelihood_freshly_recomputed": True,
        },
        "conditional_range_contract": {
            "builder": "build_history_weighted_full_card_range",
            "epsilon_exact": _fraction_text(POSTERIOR_JOIN_EPSILON),
            "max_particles": POSTERIOR_JOIN_MAX_PARTICLES,
            "range_seed_derivation": POSTERIOR_JOIN_RANGE_SEED_DERIVATION,
            "caller_supplied_range_seed_parameter_exists": False,
            "freshly_built_and_verified_for_every_proposal": True,
            "all_queries_no_fallback": True,
            "all_ranges_hidden_assignment_exhaustive": (
                conditional_ranges_exhaustive
            ),
        },
        "posterior_join": {
            "proposal_measure": (
                "uniform_without_replacement_public_compatible_actor_private_types"
            ),
            "actor_history_behavior_likelihood_applied": True,
            "opponent_public_evidence_marginal_likelihood_applied": True,
            "opponent_conditional_range_applied": True,
            "normalization_arithmetic": "exact_fraction",
            "posterior_probability_mass_exact": "1/1",
            "support_rows": [dict(row) for row in rows],
            "support_rows_sha256": _canonical_sha256([dict(row) for row in rows]),
            "sampled_actor_type_support": not sampled_exhaustive,
            "opponent_evidence_estimators_all_exact": (
                conditional_ranges_exhaustive
            ),
            "full_joint_posterior_exact": (
                sampled_exhaustive and conditional_ranges_exhaustive
            ),
        },
        "compiled_mixture": {
            "support_sha256": compiled.support_sha256,
            "audit_manifest_sha256": compiled.audit_manifest_sha256,
            "embedded_compiler_authorization_class": (
                compiled.support_authorization_class
            ),
            "embedded_compiler_authorized_as_production_mccfr_prior": False,
        },
        "content_bindings": {
            "source_sha256s": source_sha256s,
            "source_set_sha256": _canonical_sha256(source_sha256s),
            "runtime_semantic_binding_sha256": (
                runtime_semantic_binding_sha256
            ),
            "rules_sha256": context.bindings.rules_sha256,
        },
        "proposal_origin_commitments_verified": True,
        "promoted_behavior_likelihood_verified": True,
        "conditional_range_posterior_verified": True,
        "posterior_join_normalization_verified": True,
        "authorized_as_production_mccfr_prior": True,
        "production_sampling_ready": False,
        "promotion_eligible": False,
        "strategic_strength_evaluated": False,
        "serving_default_changed": False,
        "remaining_before_policy_promotion": [
            "pass_locked_online_solver_strength_and_uncertainty_gates",
            "distill_and_pass_root_family_disjoint_holdout_and_ood_gates",
            "complete_fixed_point_readback_after_upstream_policy_updates",
        ],
    }


@dataclass(frozen=True, slots=True)
class PosteriorJoinedPublicRootMixture:
    """Verified posterior join plus an algorithm-only compiled inner mixture."""

    context_digest: str
    proposal_batch: PrivacySafeCounterfactualPrivateTypeProposalBatch = field(
        repr=False
    )
    behavior_model: FrozenBehaviorModel = field(repr=False)
    actor_history_likelihoods: tuple[tuple[str, Fraction], ...] = field(
        repr=False
    )
    compiled_mixture: CompiledPublicRootMixture
    audit_manifest_json: str = field(repr=False)
    audit_manifest_sha256: str
    _construction_token: InitVar[object | None] = None

    def __post_init__(self, _construction_token: object | None) -> None:
        if type(self) is not PosteriorJoinedPublicRootMixture:
            raise TypeError("posterior-joined mixture subclasses are forbidden")
        if _construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError(
                "use compile_posterior_joined_public_root_mixture"
            )
        _require_sha256(self.context_digest, label="context_digest")
        if type(self.proposal_batch) is not (
            PrivacySafeCounterfactualPrivateTypeProposalBatch
        ):
            raise TypeError(
                "proposal_batch must be the exact privacy-safe verified batch type"
            )
        if type(self.compiled_mixture) is not CompiledPublicRootMixture:
            raise TypeError("compiled_mixture must be exact CompiledPublicRootMixture")
        if not isinstance(self.actor_history_likelihoods, tuple) or not (
            self.actor_history_likelihoods
        ):
            raise TypeError("actor_history_likelihoods must be a non-empty tuple")
        prior_key: str | None = None
        for commitment, likelihood in self.actor_history_likelihoods:
            _require_sha256(commitment, label="actor likelihood commitment")
            if prior_key is not None and commitment <= prior_key:
                raise PosteriorJoinError(
                    "actor likelihood commitments must be unique canonical order"
                )
            prior_key = commitment
            if not isinstance(likelihood, Fraction) or likelihood <= 0:
                raise TypeError("actor likelihoods must be positive Fractions")
        if not isinstance(self.audit_manifest_json, str):
            raise TypeError("audit_manifest_json must be text")
        try:
            manifest = json.loads(self.audit_manifest_json)
        except json.JSONDecodeError as exc:
            raise PosteriorJoinError("posterior join manifest is not JSON") from exc
        if not isinstance(manifest, dict) or _canonical_json(manifest) != (
            self.audit_manifest_json
        ):
            raise PosteriorJoinError(
                "posterior join manifest must be a canonical JSON object"
            )
        _require_sha256(
            self.audit_manifest_sha256,
            label="posterior join audit manifest SHA256",
        )
        if hashlib.sha256(self.audit_manifest_json.encode("utf-8")).hexdigest() != (
            self.audit_manifest_sha256
        ):
            raise PosteriorJoinError("posterior join audit manifest hash mismatch")

    @property
    def audit_manifest(self) -> Mapping[str, Any]:
        return MappingProxyType(json.loads(self.audit_manifest_json))

    @property
    def entries(self):
        return self.compiled_mixture.entries

    @property
    def support_sha256(self) -> str:
        return self.compiled_mixture.support_sha256

    @property
    def support_authorization_class(self) -> str:
        return POSTERIOR_JOIN_AUTHORIZATION_CLASS

    @property
    def authorized_as_production_mccfr_prior(self) -> bool:
        return True

    @property
    def production_sampling_ready(self) -> bool:
        return False

    @property
    def promotion_eligible(self) -> bool:
        return False


def _build_joined(
    context: PublicRootContext,
    proposal_batch: PrivacySafeCounterfactualPrivateTypeProposalBatch,
    behavior_model: FrozenBehaviorModel,
) -> PosteriorJoinedPublicRootMixture:
    if type(context) is not PublicRootContext:
        raise TypeError("context must be exact PublicRootContext")
    if type(proposal_batch) is not PrivacySafeCounterfactualPrivateTypeProposalBatch:
        raise TypeError(
            "production compiler requires the exact repository-derived privacy-safe "
            "proposal batch; caller-seeded and ExplicitHypotheticalPrivateType "
            "support are algorithm-only"
        )
    if proposal_batch.sample_count < 2:
        raise PosteriorJoinError(
            "posterior-joined public mixture requires at least two proposals"
        )
    _CANONICAL_VERIFY_PROPOSALS(context, proposal_batch)
    model_id, model_sha256, model_manifest = _validated_promoted_behavior(
        context, behavior_model
    )

    prepared: list[
        tuple[
            CounterfactualPrivateTypeProposal,
            Fraction,
            tuple[Mapping[str, Any], ...],
            FullCardRange,
        ]
    ] = []
    for proposal in proposal_batch.proposals:
        if _CANONICAL_PRIVATE_TYPE_COMMITMENT(
            context.digest(), proposal.observation
        ) != proposal.private_type_commitment:
            raise PosteriorJoinError("proposal-origin private commitment mismatch")
        likelihood, actor_audits = _actor_history_likelihood(
            proposal, behavior_model
        )
        root_range = _fresh_conditional_range(
            proposal_batch, proposal, behavior_model
        )
        prepared.append((proposal, likelihood, actor_audits, root_range))

    total_joint_weight = sum(
        (row[1] * row[3].evidence_normalizer for row in prepared),
        Fraction(0, 1),
    )
    if total_joint_weight <= 0:
        raise PosteriorJoinError("posterior join has zero total likelihood")

    support: list[ExplicitHypotheticalPrivateType] = []
    rows: list[dict[str, Any]] = []
    likelihood_rows: list[tuple[str, Fraction]] = []
    for proposal, likelihood, actor_audits, root_range in prepared:
        joint_weight = likelihood * root_range.evidence_normalizer
        prior_mass = joint_weight / total_joint_weight
        support.append(
            ExplicitHypotheticalPrivateType(
                type_id=(
                    "posterior-joined-proposal-"
                    f"{proposal.private_type_commitment}"
                ),
                hypothetical_own_recall=proposal.observation.own_recall,
                hypothetical_current_draw=proposal.observation.current_draw,
                prior_mass=prior_mass,
                root_range=root_range,
            )
        )
        rows.append(
            _support_row(
                proposal,
                proposal_batch=proposal_batch,
                likelihood=likelihood,
                joint_unnormalized_weight=joint_weight,
                prior_mass=prior_mass,
                actor_audits=actor_audits,
                root_range=root_range,
            )
        )
        likelihood_rows.append((proposal.private_type_commitment, likelihood))
    if sum((item.prior_mass for item in support), Fraction(0, 1)) != 1:
        raise AssertionError("posterior join exact mass is not one")

    compiled = _CANONICAL_COMPILE_EXPLICIT(context, tuple(support))
    explicit_verification = _CANONICAL_VERIFY_EXPLICIT(context, compiled)
    if explicit_verification.get("authorized_as_production_mccfr_prior") is not False:
        raise PosteriorJoinError("inner explicit compiler overclaimed production")

    ordered_rows = tuple(
        sorted(rows, key=lambda row: row["hypothetical_private_type_commitment"])
    )
    ordered_likelihoods = tuple(sorted(likelihood_rows))
    manifest = _audit_manifest(
        context,
        proposal_batch,
        behavior_model_id=model_id,
        behavior_model_sha256=model_sha256,
        behavior_manifest=model_manifest,
        rows=ordered_rows,
        compiled=compiled,
        runtime_semantic_binding_sha256=(
            _CANONICAL_RUNTIME_SEMANTIC_BINDING_SHA256
        ),
    )
    manifest_json = _canonical_json(manifest)
    return PosteriorJoinedPublicRootMixture(
        context_digest=context.digest(),
        proposal_batch=proposal_batch,
        behavior_model=behavior_model,
        actor_history_likelihoods=ordered_likelihoods,
        compiled_mixture=compiled,
        audit_manifest_json=manifest_json,
        audit_manifest_sha256=hashlib.sha256(
            manifest_json.encode("utf-8")
        ).hexdigest(),
        _construction_token=_CONSTRUCTION_TOKEN,
    )


def _compiled_semantic_snapshot(
    compiled: CompiledPublicRootMixture,
) -> tuple[Any, ...]:
    return (
        compiled.context_digest,
        compiled.type_id_sha256s,
        compiled.private_type_commitments,
        compiled.support_sha256,
        compiled.audit_manifest_json,
        compiled.audit_manifest_sha256,
        tuple(
            (
                entry.root_id,
                entry.prior_mass,
                entry.adapter.observation,
                entry.adapter.root_range,
            )
            for entry in compiled.entries
        ),
    )


def compile_posterior_joined_public_root_mixture(
    context: PublicRootContext,
    proposal_batch: PrivacySafeCounterfactualPrivateTypeProposalBatch,
    behavior_model: FrozenBehaviorModel,
) -> PosteriorJoinedPublicRootMixture:
    """Compile only after a fresh proposal/behavior/range posterior join."""

    _require_runtime_integrity()
    result = _build_joined(context, proposal_batch, behavior_model)
    verify_posterior_joined_public_root_mixture(context, result)
    return result


def verify_posterior_joined_public_root_mixture(
    context: PublicRootContext,
    result: PosteriorJoinedPublicRootMixture,
) -> Mapping[str, Any]:
    """Freshly replay every proposal, behavior likelihood, range, and mass."""

    _require_runtime_integrity()
    if type(context) is not PublicRootContext:
        raise TypeError("context must be exact PublicRootContext")
    if type(result) is not PosteriorJoinedPublicRootMixture:
        raise TypeError("result must be exact PosteriorJoinedPublicRootMixture")
    if result.context_digest != context.digest():
        raise PosteriorJoinError("posterior join public context mismatch")
    expected = _build_joined(
        context,
        result.proposal_batch,
        result.behavior_model,
    )
    if result.actor_history_likelihoods != expected.actor_history_likelihoods:
        raise PosteriorJoinError("actor history likelihood replay mismatch")
    if _compiled_semantic_snapshot(
        result.compiled_mixture
    ) != _compiled_semantic_snapshot(expected.compiled_mixture):
        raise PosteriorJoinError("posterior joined compiled mixture replay mismatch")
    if result.audit_manifest_json != expected.audit_manifest_json:
        raise PosteriorJoinError("posterior join audit manifest content mismatch")
    if result.audit_manifest_sha256 != expected.audit_manifest_sha256:
        raise PosteriorJoinError("posterior join audit manifest hash mismatch")
    return MappingProxyType(
        {
            "verified": True,
            "public_context_digest": context.digest(),
            "proposal_batch_audit_manifest_sha256": (
                result.proposal_batch.audit_manifest_sha256
            ),
            "support_sha256": result.support_sha256,
            "audit_manifest_sha256": result.audit_manifest_sha256,
            "support_authorization_class": POSTERIOR_JOIN_AUTHORIZATION_CLASS,
            "proposal_origin_commitments_verified": True,
            "promoted_behavior_likelihood_verified": True,
            "conditional_range_posterior_verified": True,
            "posterior_join_normalization_verified": True,
            "authorized_as_production_mccfr_prior": True,
            "production_sampling_ready": False,
            "promotion_eligible": False,
        }
    )


_CANONICAL_VALIDATE_BEHAVIOR_IDENTITY = _range_module._validated_behavior_identity
_CANONICAL_BEHAVIOR_LIKELIHOOD = _range_module._behavior_likelihood
_CANONICAL_BUILD_RANGE = _range_module.build_history_weighted_full_card_range
_CANONICAL_VERIFY_RANGE = _range_module.verify_full_card_range
_CANONICAL_VERIFY_PROPOSALS = (
    _proposal_module.verify_privacy_safe_counterfactual_private_type_proposals
)
_CANONICAL_PRIVATE_TYPE_COMMITMENT = _mixture_module._private_type_commitment
_CANONICAL_COMPILE_EXPLICIT = _mixture_module.compile_explicit_public_root_mixture
_CANONICAL_VERIFY_EXPLICIT = _mixture_module.verify_compiled_public_root_mixture

_CANONICAL_MODULE_ALIASES = (
    ("_calibrated_runtime_module", _calibrated_runtime_module),
    ("_fixed_point_runtime_module", _fixed_point_runtime_module),
    ("_range_module", _range_module),
    ("_multi_root_module", _multi_root_module),
    ("_proposal_module", _proposal_module),
    ("_mixture_module", _mixture_module),
)
_CANONICAL_EXTERNALS = (
    (
        "range behavior identity",
        _range_module,
        "_validated_behavior_identity",
        _CANONICAL_VALIDATE_BEHAVIOR_IDENTITY,
    ),
    (
        "range actor behavior likelihood",
        _range_module,
        "_behavior_likelihood",
        _CANONICAL_BEHAVIOR_LIKELIHOOD,
    ),
    (
        "range builder",
        _range_module,
        "build_history_weighted_full_card_range",
        _CANONICAL_BUILD_RANGE,
    ),
    (
        "range verifier",
        _range_module,
        "verify_full_card_range",
        _CANONICAL_VERIFY_RANGE,
    ),
    (
        "proposal verifier",
        _proposal_module,
        "verify_privacy_safe_counterfactual_private_type_proposals",
        _CANONICAL_VERIFY_PROPOSALS,
    ),
    (
        "private commitment",
        _mixture_module,
        "_private_type_commitment",
        _CANONICAL_PRIVATE_TYPE_COMMITMENT,
    ),
    (
        "explicit compiler",
        _mixture_module,
        "compile_explicit_public_root_mixture",
        _CANONICAL_COMPILE_EXPLICIT,
    ),
    (
        "explicit verifier",
        _mixture_module,
        "verify_compiled_public_root_mixture",
        _CANONICAL_VERIFY_EXPLICIT,
    ),
    (
        "calibrated behavior class",
        _calibrated_runtime_module,
        "VerifiedCalibratedHuT1T2BehaviorDispatch",
        VerifiedCalibratedHuT1T2BehaviorDispatch,
    ),
    (
        "M3 behavior class",
        _fixed_point_runtime_module,
        "VerifiedM3BehaviorLikelihoodDispatch",
        VerifiedM3BehaviorLikelihoodDispatch,
    ),
    (
        "range legal action generator",
        _range_module,
        "get_turn_actions",
        _range_module.get_turn_actions,
    ),
    (
        "range action key",
        _range_module,
        "action_key",
        _range_module.action_key,
    ),
    (
        "range epsilon smoothing",
        _range_module,
        "epsilon_smoothed_likelihood",
        _range_module.epsilon_smoothed_likelihood,
    ),
    (
        "range behavior distribution verifier",
        _range_module,
        "_validated_behavior_distribution",
        _range_module._validated_behavior_distribution,
    ),
    (
        "range assignment selector",
        _range_module,
        "_select_assignments",
        _range_module._select_assignments,
    ),
    (
        "range physical partition verifier",
        _range_module,
        "_validate_full_card_partition",
        _range_module._validate_full_card_partition,
    ),
    (
        "range opponent-turn derivation",
        _range_module,
        "_opponent_turns",
        _range_module._opponent_turns,
    ),
    (
        "range recall reconstruction",
        _range_module,
        "_recall_from_discards",
        _range_module._recall_from_discards,
    ),
    (
        "range particle commitment",
        _range_module,
        "_particle_commitment",
        _range_module._particle_commitment,
    ),
    (
        "explicit observation builder",
        _mixture_module,
        "_observation_for_type",
        _mixture_module._observation_for_type,
    ),
    (
        "explicit support row",
        _mixture_module,
        "_support_row",
        _mixture_module._support_row,
    ),
    (
        "explicit audit manifest",
        _mixture_module,
        "_audit_manifest",
        _mixture_module._audit_manifest,
    ),
    (
        "calibrated behavior action dispatch",
        VerifiedCalibratedHuT1T2BehaviorDispatch,
        "action_distribution",
        VerifiedCalibratedHuT1T2BehaviorDispatch.action_distribution,
    ),
    (
        "M3 behavior action dispatch",
        VerifiedM3BehaviorLikelihoodDispatch,
        "action_distribution",
        VerifiedM3BehaviorLikelihoodDispatch.action_distribution,
    ),
)
_CANONICAL_EXTERNAL_DATA = (
    (
        "canonical range deck",
        _range_module,
        "ALL_CARDS",
        _range_module.ALL_CARDS,
        tuple(_range_module.ALL_CARDS),
    ),
    (
        "range expected undealt counts",
        _range_module,
        "EXPECTED_UNDEALT_BY_PHASE",
        _range_module.EXPECTED_UNDEALT_BY_PHASE,
        tuple(sorted(_range_module.EXPECTED_UNDEALT_BY_PHASE.items())),
    ),
    (
        "range behavior sources",
        _range_module,
        "BEHAVIOR_DISTRIBUTION_SOURCES",
        _range_module.BEHAVIOR_DISTRIBUTION_SOURCES,
        tuple(sorted(_range_module.BEHAVIOR_DISTRIBUTION_SOURCES)),
    ),
    (
        "range row order",
        _range_module,
        "ROWS",
        _range_module.ROWS,
        tuple(_range_module.ROWS),
    ),
)
_CANONICAL_EXTERNAL_SCALARS = (
    ("range model", _range_module, "RANGE_MODEL", _range_module.RANGE_MODEL),
    ("range sampler", _range_module, "SAMPLER", _range_module.SAMPLER),
    ("range content schema", _range_module, "CONTENT_SCHEMA", _range_module.CONTENT_SCHEMA),
    ("range build schema", _range_module, "BUILD_SCHEMA", _range_module.BUILD_SCHEMA),
    (
        "explicit authorization class",
        _mixture_module,
        "EXPLICIT_SUPPORT_AUTHORIZATION_CLASS",
        _mixture_module.EXPLICIT_SUPPORT_AUTHORIZATION_CLASS,
    ),
)
_CANONICAL_EXTERNAL_CLASSES = (
    ("range Action", _range_module, "Action", _range_module.Action),
    (
        "range BehaviorInfoSet",
        _range_module,
        "BehaviorInfoSet",
        _range_module.BehaviorInfoSet,
    ),
    ("range InfoSetKey", _range_module, "InfoSetKey", _range_module.InfoSetKey),
    (
        "range JointParticle",
        _range_module,
        "JointParticle",
        _range_module.JointParticle,
    ),
    (
        "range PrivateRecall",
        _range_module,
        "PrivateRecall",
        _range_module.PrivateRecall,
    ),
    ("range FullCardRange", _range_module, "FullCardRange", _range_module.FullCardRange),
    (
        "explicit compiled mixture class",
        _mixture_module,
        "CompiledPublicRootMixture",
        _mixture_module.CompiledPublicRootMixture,
    ),
    (
        "explicit private type class",
        _mixture_module,
        "ExplicitHypotheticalPrivateType",
        _mixture_module.ExplicitHypotheticalPrivateType,
    ),
    (
        "explicit adapter class",
        _mixture_module,
        "FullCardGenerativeAdapter",
        _mixture_module.FullCardGenerativeAdapter,
    ),
    (
        "explicit chance entry class",
        _mixture_module,
        "MultiRootChanceEntry",
        _mixture_module.MultiRootChanceEntry,
    ),
)
_CANONICAL_TRANSITIVE_MODULE_FUNCTIONS = tuple(
    (
        f"{module.__name__}.{name}",
        module,
        name,
        value,
    )
    for module in (_range_module, _mixture_module)
    for name, value in sorted(vars(module).items())
    if inspect.isfunction(value) and value.__module__ == module.__name__
)
_CANONICAL_DATA_ALIASES = (
    ("POSTERIOR_JOIN_SCHEMA", POSTERIOR_JOIN_SCHEMA),
    ("POSTERIOR_JOIN_AUTHORIZATION_CLASS", POSTERIOR_JOIN_AUTHORIZATION_CLASS),
    ("POSTERIOR_JOIN_EPSILON", POSTERIOR_JOIN_EPSILON),
    ("POSTERIOR_JOIN_MAX_PARTICLES", POSTERIOR_JOIN_MAX_PARTICLES),
    (
        "POSTERIOR_JOIN_RANGE_SEED_DERIVATION",
        POSTERIOR_JOIN_RANGE_SEED_DERIVATION,
    ),
    ("_PHASE_BEHAVIOR_REQUIREMENT", _PHASE_BEHAVIOR_REQUIREMENT),
    ("_SOURCE_PATHS", _SOURCE_PATHS),
)


def _live_runtime_semantic_binding() -> dict[str, Any]:
    return _multi_root_module._runtime_semantic_graph(
        {
            "audit_manifest": _audit_manifest,
            "conditional_range_seed": _conditional_range_seed,
            "support_row": _support_row,
        },
        data_roots={name: value for name, value in _CANONICAL_DATA_ALIASES},
    )


_CANONICAL_LOCAL_HELPERS = (
    ("_canonical_json", _canonical_json),
    ("_canonical_sha256", _canonical_sha256),
    ("_file_sha256", _file_sha256),
    ("_live_source_sha256s", _live_source_sha256s),
    ("_require_sha256", _require_sha256),
    ("_fraction_text", _fraction_text),
    ("_validated_promoted_behavior", _validated_promoted_behavior),
    ("_actor_history_likelihood", _actor_history_likelihood),
    ("_conditional_range_seed", _conditional_range_seed),
    ("_fresh_conditional_range", _fresh_conditional_range),
    ("_support_row", _support_row),
    ("_audit_manifest", _audit_manifest),
    ("_build_joined", _build_joined),
    ("_compiled_semantic_snapshot", _compiled_semantic_snapshot),
    ("_live_runtime_semantic_binding", _live_runtime_semantic_binding),
)
_CANONICAL_SOURCE_SHA256S = MappingProxyType(_live_source_sha256s())
_CANONICAL_RUNTIME_SEMANTIC_BINDING_SHA256 = _live_runtime_semantic_binding()[
    "binding_sha256"
]


def _require_runtime_integrity() -> str:
    for name, expected in _CANONICAL_MODULE_ALIASES:
        if globals().get(name) is not expected:
            raise PosteriorJoinError(f"posterior join module alias drifted: {name}")
    for label, module, attribute, expected in _CANONICAL_EXTERNALS:
        if getattr(module, attribute) is not expected:
            raise PosteriorJoinError(
                f"posterior join canonical runtime drifted: {label}"
            )
    for label, module, attribute, expected_object, expected_value in (
        _CANONICAL_EXTERNAL_DATA
    ):
        actual = getattr(module, attribute)
        if actual is not expected_object:
            raise PosteriorJoinError(
                f"posterior join canonical runtime data drifted: {label}"
            )
        if isinstance(actual, Mapping):
            live_value = tuple(sorted(actual.items()))
        else:
            live_value = tuple(sorted(actual)) if isinstance(
                actual, (set, frozenset)
            ) else tuple(actual)
        if live_value != expected_value:
            raise PosteriorJoinError(
                f"posterior join canonical runtime data content drifted: {label}"
            )
    for label, module, attribute, expected in _CANONICAL_EXTERNAL_SCALARS:
        actual = getattr(module, attribute)
        if type(actual) is not type(expected) or actual != expected:
            raise PosteriorJoinError(
                f"posterior join canonical runtime scalar drifted: {label}"
            )
    for label, module, attribute, expected in _CANONICAL_EXTERNAL_CLASSES:
        if getattr(module, attribute) is not expected:
            raise PosteriorJoinError(
                f"posterior join canonical runtime class drifted: {label}"
            )
    for label, module, attribute, expected in (
        _CANONICAL_TRANSITIVE_MODULE_FUNCTIONS
    ):
        if getattr(module, attribute) is not expected:
            raise PosteriorJoinError(
                f"posterior join transitive runtime function drifted: {label}"
            )
    for name, expected in _CANONICAL_LOCAL_HELPERS:
        if globals().get(name) is not expected:
            raise PosteriorJoinError(f"posterior join helper drifted: {name}")
    for name, expected in _CANONICAL_DATA_ALIASES:
        if globals().get(name) is not expected:
            raise PosteriorJoinError(f"posterior join data alias drifted: {name}")
    live_graph = _live_runtime_semantic_binding()
    if live_graph.get("binding_sha256") != (
        _CANONICAL_RUNTIME_SEMANTIC_BINDING_SHA256
    ):
        raise PosteriorJoinError(
            "posterior join transitive runtime semantic binding drifted"
        )
    live_sources = _live_source_sha256s()
    if live_sources != dict(_CANONICAL_SOURCE_SHA256S):
        changed = sorted(
            name
            for name in set(live_sources) | set(_CANONICAL_SOURCE_SHA256S)
            if live_sources.get(name) != _CANONICAL_SOURCE_SHA256S.get(name)
        )
        raise PosteriorJoinError(f"posterior join source binding drifted: {changed}")
    return _CANONICAL_RUNTIME_SEMANTIC_BINDING_SHA256


_COMPILE_IMPL = compile_posterior_joined_public_root_mixture
_VERIFY_IMPL = verify_posterior_joined_public_root_mixture
_CANONICAL_LOCAL_HELPERS = _CANONICAL_LOCAL_HELPERS + (
    ("_require_runtime_integrity", _require_runtime_integrity),
    ("_COMPILE_IMPL", _COMPILE_IMPL),
    ("_VERIFY_IMPL", _VERIFY_IMPL),
)


def _guard_entrypoint(function: Callable[..., Any]) -> Callable[..., Any]:
    module_aliases = _CANONICAL_MODULE_ALIASES
    externals = _CANONICAL_EXTERNALS
    external_data = _CANONICAL_EXTERNAL_DATA
    external_scalars = _CANONICAL_EXTERNAL_SCALARS
    external_classes = _CANONICAL_EXTERNAL_CLASSES
    transitive_functions = _CANONICAL_TRANSITIVE_MODULE_FUNCTIONS
    helpers = _CANONICAL_LOCAL_HELPERS
    data_aliases = _CANONICAL_DATA_ALIASES
    source_hashes = _CANONICAL_SOURCE_SHA256S
    runtime_sha256 = _CANONICAL_RUNTIME_SEMANTIC_BINDING_SHA256
    runtime_guard = _require_runtime_integrity

    @wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        if globals().get("_CANONICAL_MODULE_ALIASES") is not module_aliases:
            raise PosteriorJoinError("posterior join module identity set drifted")
        if globals().get("_CANONICAL_EXTERNALS") is not externals:
            raise PosteriorJoinError("posterior join external identity set drifted")
        if globals().get("_CANONICAL_EXTERNAL_DATA") is not external_data:
            raise PosteriorJoinError(
                "posterior join external data identity set drifted"
            )
        if globals().get("_CANONICAL_EXTERNAL_SCALARS") is not external_scalars:
            raise PosteriorJoinError(
                "posterior join external scalar identity set drifted"
            )
        if globals().get("_CANONICAL_EXTERNAL_CLASSES") is not external_classes:
            raise PosteriorJoinError(
                "posterior join external class identity set drifted"
            )
        if globals().get(
            "_CANONICAL_TRANSITIVE_MODULE_FUNCTIONS"
        ) is not transitive_functions:
            raise PosteriorJoinError(
                "posterior join transitive function identity set drifted"
            )
        if globals().get("_CANONICAL_LOCAL_HELPERS") is not helpers:
            raise PosteriorJoinError("posterior join helper identity set drifted")
        if globals().get("_CANONICAL_DATA_ALIASES") is not data_aliases:
            raise PosteriorJoinError("posterior join data identity set drifted")
        if globals().get("_CANONICAL_SOURCE_SHA256S") is not source_hashes:
            raise PosteriorJoinError("posterior join source snapshot drifted")
        if globals().get(
            "_CANONICAL_RUNTIME_SEMANTIC_BINDING_SHA256"
        ) != runtime_sha256:
            raise PosteriorJoinError("posterior join runtime snapshot drifted")
        runtime_guard()
        return function(*args, **kwargs)

    return guarded


compile_posterior_joined_public_root_mixture = _guard_entrypoint(_COMPILE_IMPL)
verify_posterior_joined_public_root_mixture = _guard_entrypoint(_VERIFY_IMPL)


__all__ = [
    "POSTERIOR_JOIN_AUTHORIZATION_CLASS",
    "POSTERIOR_JOIN_EPSILON",
    "POSTERIOR_JOIN_MAX_PARTICLES",
    "POSTERIOR_JOIN_RANGE_SEED_DERIVATION",
    "POSTERIOR_JOIN_SCHEMA",
    "PosteriorJoinError",
    "PosteriorJoinedPublicRootMixture",
    "compile_posterior_joined_public_root_mixture",
    "verify_posterior_joined_public_root_mixture",
]
