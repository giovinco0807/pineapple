from __future__ import annotations

import inspect
import json
from dataclasses import fields
from fractions import Fraction
from types import MappingProxyType, MethodType

import pytest

import ai.tutor.t3_t4_posterior_joined_private_types as join_module
from ai.tutor.t3_hu_full_card_range import (
    BehaviorDistribution,
    UniformLegalBehaviorModel,
)
from ai.tutor.t3_t4_counterfactual_private_types import (
    build_counterfactual_private_type_proposals,
    build_privacy_safe_counterfactual_private_type_proposals,
    build_preregistered_public_seed_plan,
    build_public_seed_reveal_commitment,
)
from ai.tutor.t3_t4_posterior_joined_private_types import (
    POSTERIOR_JOIN_AUTHORIZATION_CLASS,
    POSTERIOR_JOIN_SCHEMA,
    PosteriorJoinError,
    PosteriorJoinedPublicRootMixture,
    compile_posterior_joined_public_root_mixture,
    verify_posterior_joined_public_root_mixture,
)
from ai.tutor.t3_t4_public_root_mixture import (
    EXPLICIT_SUPPORT_AUTHORIZATION_CLASS,
    PublicRootBindings,
    PublicRootContext,
)
from test_calibrated_behavior_runtime import _build_runtime
from test_t3_t4_public_root_mixture import _phase_public_payload


def _context(phase: str, model) -> PublicRootContext:
    return PublicRootContext.from_public_mapping(
        _phase_public_payload(phase),
        bindings=PublicRootBindings.capture(
            behavior_model_id=model.model_id,
            behavior_model_sha256=model.model_sha256,
        ),
    )


def _proposal_batch(
    context: PublicRootContext,
    *,
    label: str,
    seed: int,
    sample_count: int = 2,
):
    namespace = f"posterior-join-test/{label}"
    nonce = f"external-seed-reveal/{label}/{seed}"
    reveal = build_public_seed_reveal_commitment(
        context,
        seed_namespace=namespace,
        seed=seed,
        seed_nonce=nonce,
    )
    plan = build_preregistered_public_seed_plan(
        context,
        seed_namespace=namespace,
        seed_reveal_commitment_sha256=reveal,
        sample_count=sample_count,
        trusted_registry_id="test-public-seed-registry-v1",
        registration_record_id=f"posterior-join-test-record/{label}",
    )
    return build_counterfactual_private_type_proposals(
        context,
        sample_count=sample_count,
        seed_plan=plan,
        approved_seed_plan_sha256=plan.plan_sha256,
        seed=seed,
        seed_nonce=nonce,
    )


def _privacy_safe_batch(context: PublicRootContext):
    return build_privacy_safe_counterfactual_private_type_proposals(context)


def test_production_boundary_rejects_explicit_support_and_unpromoted_model():
    model = UniformLegalBehaviorModel()
    context = _context("t3_first", model)
    batch = _proposal_batch(context, label="negative", seed=101)

    assert set(
        inspect.signature(compile_posterior_joined_public_root_mixture).parameters
    ) == {"context", "proposal_batch", "behavior_model"}
    with pytest.raises(TypeError, match="privacy-safe proposal batch"):
        compile_posterior_joined_public_root_mixture(
            context,
            tuple(batch.proposals),  # type: ignore[arg-type]
            model,
        )
    with pytest.raises(TypeError, match="privacy-safe proposal batch"):
        compile_posterior_joined_public_root_mixture(context, batch, model)


def test_t4_join_fails_closed_until_promoted_t3_btn_and_t4_routes_exist():
    model = UniformLegalBehaviorModel()
    for phase in ("t4_first", "t4_second"):
        context = _context(phase, model)
        batch = _privacy_safe_batch(context)
        with pytest.raises(PosteriorJoinError, match="unavailable_requires"):
            compile_posterior_joined_public_root_mixture(context, batch, model)


def test_promoted_t3_first_join_recomputes_likelihood_ranges_and_exact_prior(
    monkeypatch,
):
    (*_unused, model, _calls, _evaluators) = _build_runtime(monkeypatch)
    context = _context("t3_first", model)
    batch = _privacy_safe_batch(context)

    result = compile_posterior_joined_public_root_mixture(
        context, batch, model
    )
    manifest = result.audit_manifest

    assert type(result) is PosteriorJoinedPublicRootMixture
    assert result.authorized_as_production_mccfr_prior is True
    assert result.production_sampling_ready is False
    assert result.promotion_eligible is False
    assert result.support_authorization_class == POSTERIOR_JOIN_AUTHORIZATION_CLASS
    assert manifest["schema"] == POSTERIOR_JOIN_SCHEMA
    assert manifest["proposal_origin_commitments_verified"] is True
    assert manifest["promoted_behavior_likelihood_verified"] is True
    assert manifest["conditional_range_posterior_verified"] is True
    assert manifest["posterior_join_normalization_verified"] is True
    assert manifest["authorized_as_production_mccfr_prior"] is True
    assert manifest["promotion_eligible"] is False
    assert manifest["serving_default_changed"] is False
    assert manifest["compiled_mixture"][
        "embedded_compiler_authorization_class"
    ] == EXPLICIT_SUPPORT_AUTHORIZATION_CLASS
    assert manifest["compiled_mixture"][
        "embedded_compiler_authorized_as_production_mccfr_prior"
    ] is False
    assert manifest["proposal_origin"]["origin_commitments_freshly_verified"] is True
    assert manifest["proposal_origin"]["privacy_safe_origin_verified"] is True
    assert manifest["proposal_origin"]["caller_supplied_entropy_consumed"] is False
    assert manifest["proposal_origin"]["generic_seed_plan_consumed"] is False
    assert manifest["proposal_origin"]["actual_actor_private_cards_consumed"] is False
    assert manifest["conditional_range_contract"][
        "caller_supplied_range_seed_parameter_exists"
    ] is False
    assert manifest["conditional_range_contract"]["all_queries_no_fallback"] is True
    assert manifest["posterior_join"]["posterior_probability_mass_exact"] == "1/1"
    assert len(manifest["posterior_join"]["support_rows"]) == 2
    assert all(
        row["proposal_origin_verified"] is True
        and row["actor_history_uniform_fallback_count"] == 0
        and row["conditional_range_uniform_fallback_count"] == 0
        for row in manifest["posterior_join"]["support_rows"]
    )
    assert sum((entry.prior_mass for entry in result.entries)) == 1
    assert result.compiled_mixture.authorized_as_production_mccfr_prior is False


def test_join_runtime_substitution_and_manual_forgery_fail_closed(
    monkeypatch,
):
    model = UniformLegalBehaviorModel()
    context = _context("t3_first", model)
    batch = _privacy_safe_batch(context)

    for name in ("_select_assignments", "_board"):
        with monkeypatch.context() as patch:
            patch.setattr(join_module._range_module, name, lambda *a, **k: ())
            with pytest.raises(
                PosteriorJoinError,
                match="canonical runtime drifted|transitive runtime function drifted",
            ):
                compile_posterior_joined_public_root_mixture(context, batch, model)

    forged = object.__new__(PosteriorJoinedPublicRootMixture)
    for item in fields(PosteriorJoinedPublicRootMixture):
        if item.name == "context_digest":
            value = context.digest()
        elif item.name == "proposal_batch":
            value = batch
        elif item.name == "behavior_model":
            value = model
        elif item.name == "actor_history_likelihoods":
            value = ((batch.proposals[0].private_type_commitment, 1),)
        elif item.name == "compiled_mixture":
            value = object()
        elif item.name == "audit_manifest_json":
            value = json.dumps({"promotion_eligible": True})
        else:
            value = "0" * 64
        object.__setattr__(forged, item.name, value)
    with pytest.raises((TypeError, PosteriorJoinError)):
        verify_posterior_joined_public_root_mixture(context, forged)


def test_outer_posterior_multiplies_opponent_public_evidence_normalizer(
    monkeypatch,
):
    (*_unused, model, _calls, _evaluators) = _build_runtime(monkeypatch)

    def skewed_distribution(_child, information):
        legal = information.legal_action_ids
        observed_placements = {
            ("bb", 1): {("7d", "middle"), ("8h", "middle")},
            ("bb", 2): {("9s", "middle"), ("Tc", "middle")},
            ("btn", 1): {("9d", "middle"), ("Jh", "middle")},
            ("btn", 2): {("Kc", "middle"), ("Qs", "middle")},
        }[(information.actor, information.turn)]
        observed = next(
            action_id
            for action_id in legal
            if {
                tuple(item)
                for item in json.loads(action_id)["placements"]
            }
            == observed_placements
        )
        probability = (
            Fraction(3, 4)
            if "Ts" in information.current_draw
            else Fraction(1, 4)
        )
        remainder = (1 - probability) / (len(legal) - 1)
        probabilities = {
            action_id: probability if action_id == observed else remainder
            for action_id in legal
        }
        return BehaviorDistribution(
            information_digest=information.digest(),
            probabilities=MappingProxyType(probabilities),
            source="model",
            used_fallback=False,
        )

    for child in model._routes.values():
        child.action_distribution = MethodType(skewed_distribution, child)

    context = _context("t3_first", model)
    result = compile_posterior_joined_public_root_mixture(
        context, _privacy_safe_batch(context), model
    )
    rows = result.audit_manifest["posterior_join"]["support_rows"]
    actor_likelihoods = [Fraction(row["actor_history_likelihood_exact"]) for row in rows]
    evidence = [
        Fraction(row["opponent_public_evidence_normalizer_exact"])
        for row in rows
    ]
    joint = [
        Fraction(row["joint_unnormalized_actor_type_weight_exact"])
        for row in rows
    ]
    priors = [Fraction(row["posterior_prior_mass_exact"]) for row in rows]

    assert len(set(evidence)) == 2
    assert joint == [
        actor_likelihood * normalizer
        for actor_likelihood, normalizer in zip(actor_likelihoods, evidence)
    ]
    assert priors == [weight / sum(joint) for weight in joint]
    actor_only = [weight / sum(actor_likelihoods) for weight in actor_likelihoods]
    assert priors != actor_only
    assert [entry.prior_mass for entry in result.entries] == priors
    assert all(
        row["conditional_range_evidence_sampled_assignment_count"] > 0
        and row["conditional_range_evidence_total_assignment_count"]
        >= row["conditional_range_evidence_sampled_assignment_count"]
        for row in rows
    )
