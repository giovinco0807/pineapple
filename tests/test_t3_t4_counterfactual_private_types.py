import hashlib
import inspect
import json
import math
from collections import Counter
from dataclasses import fields, replace

import pytest

import ai.tutor.t3_t4_counterfactual_private_types as proposal_module
import ai.tutor.t3_t4_public_root_mixture as mixture_module
from ai.engine.encoding import ALL_CARDS
from ai.tutor.t3_hu_full_card_range import UniformLegalBehaviorModel
from ai.tutor.t3_hu_public_cfr import InfoSetKey
from ai.tutor.t3_t4_counterfactual_private_types import (
    COUNTERFACTUAL_PRIVATE_TYPE_BATCH_SCHEMA,
    PROPOSAL_SEMANTICS,
    CounterfactualPrivateTypeError,
    PrivacySafeCounterfactualPrivateTypeProposalBatch,
    PreregisteredPublicSeedPlan,
    build_counterfactual_private_type_proposals,
    build_privacy_safe_counterfactual_private_type_proposals,
    build_preregistered_public_seed_plan,
    build_public_seed_reveal_commitment,
    verify_counterfactual_private_type_proposals,
    verify_privacy_safe_counterfactual_private_type_proposals,
    verify_preregistered_public_seed_plan,
)
from ai.tutor.t3_t4_public_root_mixture import (
    PublicRootBindings,
    PublicRootContext,
)
from test_t3_t4_public_root_mixture import _phase_public_payload


EXPECTED_UNIVERSE_SIZES = {
    "t3_first": 7_539_840,
    "t3_second": 5_565_120,
    "t4_first": 108_743_040,
    "t4_second": 71_253_000,
}


@pytest.fixture(scope="module")
def public_contexts():
    model = UniformLegalBehaviorModel()
    bindings = PublicRootBindings.capture(
        behavior_model_id=model.model_id,
        behavior_model_sha256=model.model_sha256,
    )
    return {
        phase: PublicRootContext.from_public_mapping(
            _phase_public_payload(phase), bindings=bindings
        )
        for phase in EXPECTED_UNIVERSE_SIZES
    }


def _partition_cards(proposal) -> list[str]:
    observation = proposal.observation
    witness = proposal.physical_completion_witness
    cards = [
        card
        for _turn, _actor, placements in observation.public_action_history
        for card, _row in placements
    ]
    cards.extend(observation.current_draw)
    cards.extend(witness.undealt_cards)
    for recall in (witness.bb_recall, witness.btn_recall):
        cards.extend(card for _turn, card in recall.discards_by_turn)
    return cards


def _trusted_seed(
    context,
    label: str,
    seed: int,
    *,
    sample_count: int,
    start_ordinal: int = 0,
) -> dict:
    namespace = f"public-seed-namespace/{label}"
    nonce = f"external-randomness-reveal/{label}/{seed}"
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
        start_ordinal=start_ordinal,
        trusted_registry_id="test-public-seed-registry-v1",
        registration_record_id=f"test-registration/{label}",
    )
    assert verify_preregistered_public_seed_plan(
        context,
        plan,
        approved_seed_plan_sha256=plan.plan_sha256,
    )["verified"] is True
    return {
        "seed_plan": plan,
        "approved_seed_plan_sha256": plan.plan_sha256,
        "seed": seed,
        "seed_nonce": nonce,
    }


@pytest.mark.parametrize("phase", tuple(EXPECTED_UNIVERSE_SIZES))
def test_all_four_public_cuts_define_exact_complete_universes_and_physical_types(
    public_contexts, phase
):
    context = public_contexts[phase]
    result = build_counterfactual_private_type_proposals(
        context,
        sample_count=32,
        **_trusted_seed(
            context, f"all-cuts-{phase}", 20260713, sample_count=32
        ),
    )
    verification = verify_counterfactual_private_type_proposals(context, result)

    assert verification["verified"] is True
    assert result.universe_size == EXPECTED_UNIVERSE_SIZES[phase]
    assert verification["universe_size"] == EXPECTED_UNIVERSE_SIZES[phase]
    assert len(result.proposals) == len(
        {item.private_type_commitment for item in result.proposals}
    ) == 32
    assert len({item.universe_rank for item in result.proposals}) == 32

    for proposal in result.proposals:
        observation = proposal.observation
        witness = proposal.physical_completion_witness
        rebuilt = InfoSetKey.for_particle(
            witness,
            contract_version=context.contract_version,
            actor=context.actor,
            turn=context.turn,
            phase=context.phase,
            board_bb=context.board_bb,
            board_btn=context.board_btn,
            public_action_history=context.public_action_history,
            current_draw=observation.current_draw,
            fantasy_state=context.fantasy_state,
        )
        assert rebuilt == observation
        assert proposal.private_type_commitment == (
            mixture_module._private_type_commitment(
                context.digest(), observation
            )
        )
        physical = _partition_cards(proposal)
        assert len(physical) == 54
        assert Counter(physical) == Counter(ALL_CARDS)
        assert physical.count("X1") == physical.count("X2") == 1


def test_audit_binds_public_context_coverage_sources_and_nonposterior_boundary(
    public_contexts,
):
    context = public_contexts["t3_first"]
    result = build_counterfactual_private_type_proposals(
        context,
        sample_count=64,
        **_trusted_seed(context, "audit", 17, sample_count=64),
    )
    manifest = result.audit_manifest
    traversal = manifest["traversal"]
    posterior = manifest["posterior_boundary"]
    bindings = manifest["content_bindings"]

    assert manifest["schema"] == COUNTERFACTUAL_PRIVATE_TYPE_BATCH_SCHEMA
    assert manifest["artifact_kind"] == (
        "unweighted_actor_private_type_proposal_support_only"
    )
    assert manifest["public_context"] == context.to_canonical_dict()
    assert manifest["public_context_digest"] == context.digest()
    assert manifest["universe"][
        "complete_publicly_compatible_universe_defined"
    ] is True
    assert manifest["universe"]["universe_size"] == 7_539_840
    assert traversal["proposal_semantics"] == PROPOSAL_SEMANTICS
    assert traversal["without_replacement"] is True
    assert traversal["full_cycle_bijective"] is True
    assert traversal["seed_material_serialized"] is False
    assert traversal["external_seed_material_required_for_replay"] is True
    assert traversal["sampled_support_exhaustive"] is False
    assert traversal["coverage_fraction_exact"] == "1/117810"
    assert traversal["marginal_inclusion_probability_exact"] == "1/117810"
    assert traversal["sample_commitments_unique"] is True
    assert traversal["raw_universe_ranks_serialized"] is False
    assert traversal["raw_private_cards_serialized"] is False
    proof = manifest["universe"]["closed_form_physical_completion_proof"]
    assert proof["actor_prior_discard_count"] == 2
    assert proof["actor_current_draw_count"] == 3
    assert proof["opponent_completed_discard_count"] == 2
    assert proof["guaranteed_undealt_count"] == 29
    assert proof["phase_expected_undealt_count"] == 29
    assert proof["partition_card_count_equation"] == "18+2+3+2+29=54"

    assert posterior == {
        "uniform_proposal_is_posterior": False,
        "actor_history_behavior_likelihood_applied": False,
        "opponent_conditional_range_applied": False,
        "calibrated_behavior_artifact_bound": False,
        "conditional_range_artifacts_bound": False,
        "posterior_normalization_applied": False,
        "authorized_as_mccfr_prior": False,
        "required_downstream_join": (
            "fresh_verified_promoted_behavior_likelihood_and_conditional_range"
        ),
    }
    assert manifest["promotion_eligible"] is False
    assert manifest["production_sampling_ready"] is False
    assert result.promotion_eligible is False
    assert result.production_sampling_ready is False

    assert bindings["rules_sha256"] == context.bindings.rules_sha256
    assert bindings["expected_behavior_model_id"] == (
        context.bindings.behavior_model_id
    )
    assert bindings["expected_behavior_model_sha256"] == (
        context.bindings.behavior_model_sha256
    )
    for manifest_name, context_name in (
        ("public_root_mixture_source_sha256", "public_root_mixture_compiler"),
        ("public_infoset_source_sha256", "public_infoset"),
        ("action_space_source_sha256", "action_space"),
        ("deck_encoding_source_sha256", "deck_encoding"),
        ("turn_order_source_sha256", "turn_order"),
        ("action_semantics_source_sha256", "terminal_scoring"),
    ):
        assert bindings[manifest_name] == context.bindings.source_sha256s[
            context_name
        ]

    assert hashlib.sha256(result.audit_manifest_json.encode()).hexdigest() == (
        result.audit_manifest_sha256
    )
    private_field_names = {
        "own_recall",
        "current_draw",
        "universe_rank",
        "discards_by_turn",
        "dealt_by_turn",
    }
    assert not private_field_names.intersection(
        json.loads(result.audit_manifest_json)["traversal"]
    )


def test_traversal_is_deterministic_unique_and_slice_stable(public_contexts):
    context = public_contexts["t4_second"]
    kwargs = _trusted_seed(context, "shards", -91, sample_count=24)
    whole = build_counterfactual_private_type_proposals(
        context, sample_count=24, **kwargs
    )
    replay = build_counterfactual_private_type_proposals(
        context, sample_count=24, **kwargs
    )
    middle = build_counterfactual_private_type_proposals(
        context,
        start_ordinal=7,
        sample_count=9,
        **_trusted_seed(
            context,
            "shards",
            -91,
            sample_count=9,
            start_ordinal=7,
        ),
    )

    assert replay == whole
    assert middle.proposals == whole.proposals[7:16]
    assert tuple(item.sample_ordinal for item in middle) == tuple(range(7, 16))
    assert len({item.universe_rank for item in whole}) == 24

    other_seed = build_counterfactual_private_type_proposals(
        context,
        sample_count=24,
        **_trusted_seed(
            context, "shards-other-seed", -90, sample_count=24
        ),
    )
    assert tuple(item.private_type_commitment for item in other_seed) != tuple(
        item.private_type_commitment for item in whole
    )


def test_rank_decoder_is_bijective_on_a_small_complete_universe():
    pool = ("a", "b", "c", "d", "e", "f")
    discard_count = 2
    universe_size = math.perm(len(pool), discard_count) * math.comb(
        len(pool) - discard_count, 3
    )
    decoded = [
        proposal_module._decode_universe_rank(pool, discard_count, rank)
        for rank in range(universe_size)
    ]
    assert universe_size == 120
    assert len(set(decoded)) == universe_size
    assert all(
        len(discards) == 2
        and len(set(discards)) == 2
        and len(draw) == 3
        and set(discards).isdisjoint(draw)
        for discards, draw in decoded
    )


def test_complete_universe_preserves_both_physical_joker_identities(
    public_contexts,
):
    context = public_contexts["t3_first"]
    pool = proposal_module._available_actor_private_pool(context)
    total = proposal_module._universe_size(len(pool), 2)
    first_discards, first_draw = proposal_module._decode_universe_rank(pool, 2, 0)
    last_discards, last_draw = proposal_module._decode_universe_rank(
        pool, 2, total - 1
    )
    covered = set((*first_discards, *first_draw, *last_discards, *last_draw))

    assert {"X1", "X2"}.issubset(pool)
    assert {"X1", "X2"}.issubset(covered)
    universe = proposal_module._universe_manifest(context)
    assert universe["deck_manifest"]["physical_joker_ids"] == ["X1", "X2"]
    assert universe["deck_manifest"]["joker_physical_identity_collapsed"] is False


def test_public_api_and_result_types_have_no_actual_hand_input(public_contexts):
    build_parameters = set(
        inspect.signature(build_counterfactual_private_type_proposals).parameters
    )
    verify_parameters = set(
        inspect.signature(verify_counterfactual_private_type_proposals).parameters
    )
    forbidden = {
        "actual_actor_private_cards",
        "actual_private_cards",
        "actor_private_cards",
        "actual_hand",
        "own_recall",
        "current_draw",
    }
    assert build_parameters == {
        "context",
        "sample_count",
        "seed_plan",
        "approved_seed_plan_sha256",
        "seed",
        "seed_nonce",
        "start_ordinal",
    }
    assert verify_parameters == {"context", "result"}
    assert not build_parameters.intersection(forbidden)
    assert not verify_parameters.intersection(forbidden)
    assert {item.name for item in fields(PublicRootContext)} == {
        "contract_version",
        "actor",
        "turn",
        "phase",
        "board_bb",
        "board_btn",
        "public_action_history",
        "fantasy_state",
        "bindings",
    }

    payload = _phase_public_payload("t3_first")
    payload["actual_actor_private_cards"] = ["Ad", "Kd", "Qd"]
    with pytest.raises(ValueError, match="leaked forbidden private fields"):
        PublicRootContext.from_public_mapping(
            payload, bindings=public_contexts["t3_first"].bindings
        )


def test_bounds_and_tamper_fail_closed(public_contexts):
    context = public_contexts["t3_first"]
    with pytest.raises(ValueError, match="positive integer"):
        build_counterfactual_private_type_proposals(
            context,
            sample_count=0,
            **_trusted_seed(context, "bounds-zero", 1, sample_count=1),
        )
    with pytest.raises(ValueError, match="exceeds one"):
        build_counterfactual_private_type_proposals(
            context,
            start_ordinal=EXPECTED_UNIVERSE_SIZES["t3_first"] - 1,
            sample_count=2,
            **_trusted_seed(context, "bounds-overflow", 1, sample_count=1),
        )

    result = build_counterfactual_private_type_proposals(
        context,
        sample_count=8,
        **_trusted_seed(context, "tamper", 2, sample_count=8),
    )
    swapped = replace(
        result,
        proposals=(result.proposals[1], result.proposals[0], *result.proposals[2:]),
    )
    with pytest.raises(CounterfactualPrivateTypeError, match="contents"):
        verify_counterfactual_private_type_proposals(context, swapped)

    tampered_json = result.audit_manifest_json.replace(
        '"promotion_eligible":false', '"promotion_eligible":true'
    )
    with pytest.raises(CounterfactualPrivateTypeError, match="SHA256 mismatch"):
        replace(result, audit_manifest_json=tampered_json)


def test_runtime_helper_and_deck_constant_substitution_fail_closed(
    public_contexts, monkeypatch
):
    context = public_contexts["t3_first"]
    seed_material = _trusted_seed(
        context, "runtime-substitution", 81, sample_count=4
    )

    substitutions = (
        (
            "_traversal_parameters",
            lambda *_args, **_kwargs: (0, 1, seed_material[
                "seed_plan"
            ].manifest["seed_reveal_commitment_sha256"]),
        ),
        ("_CANONICAL_DECK", tuple(card for card in ALL_CARDS if card not in ("X1", "X2"))),
        ("_PHYSICAL_JOKERS", ()),
        ("_unrank_combination", lambda values, count, rank: tuple(values[:count])),
    )
    for name, replacement in substitutions:
        with monkeypatch.context() as patch:
            patch.setattr(proposal_module, name, replacement)
            with pytest.raises(
                CounterfactualPrivateTypeError,
                match="runtime helper drifted|runtime data alias drifted",
            ):
                build_counterfactual_private_type_proposals(
                    context, sample_count=4, **seed_material
                )

    with monkeypatch.context() as patch:
        patch.setattr(
            proposal_module._encoding_module,
            "ALL_CARDS",
            [card for card in ALL_CARDS if card not in ("X1", "X2")],
        )
        with pytest.raises(
            CounterfactualPrivateTypeError, match="runtime data drifted"
        ):
            build_counterfactual_private_type_proposals(
                context, sample_count=4, **seed_material
            )


def test_seed_plan_requires_external_approval_and_exact_reveal(public_contexts):
    context = public_contexts["t3_first"]
    approved = _trusted_seed(
        context, "approved-seed", 91, sample_count=4
    )

    with pytest.raises(
        CounterfactualPrivateTypeError, match="externally approved"
    ):
        build_counterfactual_private_type_proposals(
            context,
            sample_count=4,
            **{
                **approved,
                "approved_seed_plan_sha256": "0" * 64,
            },
        )
    with pytest.raises(
        CounterfactualPrivateTypeError, match="raw seed reveal"
    ):
        build_counterfactual_private_type_proposals(
            context,
            sample_count=4,
            **{**approved, "seed": 92},
        )
    with pytest.raises(
        CounterfactualPrivateTypeError, match="raw seed reveal"
    ):
        build_counterfactual_private_type_proposals(
            context,
            sample_count=4,
            **{**approved, "seed_nonce": approved["seed_nonce"] + "-changed"},
        )
    with pytest.raises(
        CounterfactualPrivateTypeError, match="not externally pre-registered"
    ):
        build_counterfactual_private_type_proposals(
            context,
            sample_count=3,
            **approved,
        )
    with pytest.raises(
        CounterfactualPrivateTypeError, match="not externally pre-registered"
    ):
        build_counterfactual_private_type_proposals(
            context,
            start_ordinal=1,
            sample_count=4,
            **approved,
        )

    postselected = _trusted_seed(
        context, "hidden-derived-postselection", 91, sample_count=4
    )
    with pytest.raises(
        CounterfactualPrivateTypeError, match="externally approved"
    ):
        build_counterfactual_private_type_proposals(
            context,
            sample_count=4,
            **{
                **postselected,
                # The trusted external value remains the pre-registered plan.
                "approved_seed_plan_sha256": approved[
                    "approved_seed_plan_sha256"
                ],
            },
        )

    plan = approved["seed_plan"]
    assert type(plan) is PreregisteredPublicSeedPlan
    assert plan.manifest["hidden_state_inputs_forbidden"] is True
    assert plan.manifest["actual_hand_inputs_forbidden"] is True
    assert plan.manifest["raw_seed_serialized"] is False
    assert plan.manifest["start_ordinal"] == 0
    assert plan.manifest["sample_count"] == 4
    assert plan.manifest["slice_preregistered_before_seed_reveal"] is True
    assert plan.manifest["namespace_source"] == (
        "external_public_literal_preregistered_before_private_observation"
    )
    assert "seed" not in inspect.signature(
        build_preregistered_public_seed_plan
    ).parameters
    assert "seed_nonce" not in inspect.signature(
        build_preregistered_public_seed_plan
    ).parameters


def test_manual_zero_batch_and_subclass_overrides_are_rejected(public_contexts):
    context = public_contexts["t3_first"]
    result = build_counterfactual_private_type_proposals(
        context,
        sample_count=4,
        **_trusted_seed(context, "exact-types", 101, sample_count=4),
    )

    forged = object.__new__(type(result))
    for item in fields(type(result)):
        object.__setattr__(forged, item.name, getattr(result, item.name))
    object.__setattr__(forged, "proposals", ())
    object.__setattr__(forged, "sample_count", 0)
    with pytest.raises(
        CounterfactualPrivateTypeError, match="sample_count must be positive"
    ):
        verify_counterfactual_private_type_proposals(context, forged)

    class EvilBatch(type(result)):
        @property
        def promotion_eligible(self):
            return True

    evil = object.__new__(EvilBatch)
    for item in fields(type(result)):
        object.__setattr__(evil, item.name, getattr(result, item.name))
    with pytest.raises(TypeError, match="exact Counterfactual"):
        verify_counterfactual_private_type_proposals(context, evil)


def test_privacy_safe_batch_is_context_only_deterministic_and_not_launderable(
    public_contexts,
):
    context = public_contexts["t3_first"]
    first = build_privacy_safe_counterfactual_private_type_proposals(context)
    replay = build_privacy_safe_counterfactual_private_type_proposals(context)
    verification = verify_privacy_safe_counterfactual_private_type_proposals(
        context, first
    )

    assert first == replay
    assert verification["privacy_safe_origin_verified"] is True
    assert first.eligible_as_posterior_join_origin is True
    assert first.authorized_as_mccfr_prior is False
    assert first.production_sampling_ready is False
    assert set(
        inspect.signature(
            build_privacy_safe_counterfactual_private_type_proposals
        ).parameters
    ) == {"context"}
    forbidden = {
        "seed",
        "seed_nonce",
        "seed_plan",
        "approved_seed_plan_sha256",
        "seed_namespace",
        "trusted_registry_id",
        "registration_record_id",
        "actual_hand",
        "actual_actor_private_cards",
    }
    assert not forbidden.intersection(
        item.name for item in fields(type(first))
    )
    traversal = first.audit_manifest["privacy_safe_traversal"]
    assert traversal["caller_supplied_entropy_parameter_exists"] is False
    assert traversal["caller_supplied_namespace_parameter_exists"] is False
    assert traversal["caller_supplied_slice_parameter_exists"] is False
    assert traversal["caller_supplied_registry_parameter_exists"] is False
    assert traversal["generic_seed_plan_consumed"] is False
    assert traversal["generic_batch_conversion_consumed"] is False

    constructor_values = {
        item.name: getattr(first, item.name)
        for item in fields(type(first))
    }
    with pytest.raises(TypeError, match="use build_privacy_safe"):
        PrivacySafeCounterfactualPrivateTypeProposalBatch(**constructor_values)

    generic = build_counterfactual_private_type_proposals(
        context,
        sample_count=2,
        **_trusted_seed(
            context,
            "hidden-derived-generic-cannot-launder",
            999,
            sample_count=2,
        ),
    )
    forged = object.__new__(PrivacySafeCounterfactualPrivateTypeProposalBatch)
    for item in fields(type(first)):
        value = generic.proposals if item.name == "proposals" else getattr(
            first, item.name
        )
        object.__setattr__(forged, item.name, value)
    with pytest.raises(
        CounterfactualPrivateTypeError, match="deterministic public replay"
    ):
        verify_privacy_safe_counterfactual_private_type_proposals(
            context, forged
        )
