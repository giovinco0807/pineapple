import hashlib
import inspect
import math
import random
from collections import Counter
from dataclasses import fields, replace
from fractions import Fraction

import pytest

import ai.tutor.t3_hu_full_card_mccfr as full_mccfr_module
import ai.tutor.t3_hu_full_card_range as range_module
import ai.tutor.t3_hu_multi_root_mccfr as multi_root_module
from ai.engine.encoding import ALL_CARDS
from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor.t3_hu_full_card_mccfr import FullCardGenerativeAdapter
from ai.tutor.t3_hu_full_card_range import (
    UniformLegalBehaviorModel,
    build_history_weighted_full_card_range,
    verify_full_card_range,
)
from ai.tutor.t3_hu_multi_root_mccfr import MultiRootChanceEntry
from ai.tutor.t3_hu_public_cfr import InfoSetKey, PrivateRecall
from ai.tutor.t3_hu_public_tree import PublicTreeTerminalState
from ai.tutor.t3_t4_public_root_mixture import (
    EXPLICIT_SUPPORT_AUTHORIZATION_CLASS,
    EXPLICIT_SUPPORT_SCOPE,
    MIXTURE_EXACTNESS,
    ExplicitHypotheticalPrivateType,
    PublicRootBindings,
    PublicRootContext,
    PublicRootMixtureError,
    compile_explicit_public_root_mixture,
    verify_compiled_public_root_mixture,
)
from ai.tutor.t3_t4_solve_root_teacher_adapter import (
    SolveRootTeacherAdapterError,
    build_prelabel_solve_root_teacher_plan,
    build_solve_root_teacher_evidence,
    verify_evidence_bound_teacher_bundle,
    write_evidence_bound_teacher_bundle,
    write_prelabel_solve_root_teacher_plan,
)


BB_BOARD = (
    ("4h", "2c", "3d"),
    ("9s", "7c", "8h", "7d", "Tc"),
    ("Qc",),
)
BTN_BOARD = (
    ("8c", "5c", "6d"),
    ("Kc", "9c", "Jh", "9d", "Qs"),
    ("As",),
)
PUBLIC_HISTORY = (
    (
        0,
        "bb",
        (
            ("4h", "top"),
            ("2c", "top"),
            ("3d", "top"),
            ("7c", "middle"),
            ("Qc", "bottom"),
        ),
    ),
    (
        0,
        "btn",
        (
            ("8c", "top"),
            ("5c", "top"),
            ("6d", "top"),
            ("9c", "middle"),
            ("As", "bottom"),
        ),
    ),
    (1, "bb", (("7d", "middle"), ("8h", "middle"))),
    (1, "btn", (("9d", "middle"), ("Jh", "middle"))),
    (2, "bb", (("9s", "middle"), ("Tc", "middle"))),
    (2, "btn", (("Qs", "middle"), ("Kc", "middle"))),
)
BB_T3_PLACEMENTS = (("Qd", "bottom"), ("Kh", "bottom"))
BTN_T3_PLACEMENTS = (("2h", "bottom"), ("3h", "bottom"))
BB_BOARD_11 = (BB_BOARD[0], BB_BOARD[1], ("Qc", "Qd", "Kh"))
BTN_BOARD_11 = (BTN_BOARD[0], BTN_BOARD[1], ("As", "2h", "3h"))
PUBLIC_HISTORY_AFTER_BB_T3 = PUBLIC_HISTORY + ((3, "bb", BB_T3_PLACEMENTS),)
PUBLIC_HISTORY_AFTER_T3 = PUBLIC_HISTORY_AFTER_BB_T3 + (
    (3, "btn", BTN_T3_PLACEMENTS),
)
BB_T4_PLACEMENTS = (("Ad", "bottom"), ("Kd", "bottom"))
BB_BOARD_13 = (
    BB_BOARD_11[0],
    BB_BOARD_11[1],
    ("Qc", "Qd", "Kh", "Ad", "Kd"),
)
PUBLIC_HISTORY_AFTER_BB_T4 = PUBLIC_HISTORY_AFTER_T3 + (
    (4, "bb", BB_T4_PLACEMENTS),
)


def _recall(actor: str, first_discard: str, second_discard: str) -> PrivateRecall:
    placements = {
        "bb": (("7d", "8h"), ("9s", "Tc")),
        "btn": (("9d", "Jh"), ("Qs", "Kc")),
    }[actor]
    return PrivateRecall(
        dealt_by_turn=(
            (1, (*placements[0], first_discard)),
            (2, (*placements[1], second_discard)),
        ),
        discards_by_turn=((1, first_discard), (2, second_discard)),
    )


def _append_recall(
    recall: PrivateRecall,
    *,
    turn: int,
    placements: tuple[tuple[str, str], ...],
    discard: str,
) -> PrivateRecall:
    placed = tuple(card for card, _row in placements)
    return PrivateRecall(
        dealt_by_turn=recall.dealt_by_turn + ((turn, (*placed, discard)),),
        discards_by_turn=recall.discards_by_turn + ((turn, discard),),
    )


def _phase_public_payload(phase: str) -> dict:
    if phase == "t3_first":
        return {
            "contract_version": POSITION_CONTRACT_VERSION,
            "actor": "bb",
            "turn": 3,
            "phase": phase,
            "board_bb": BB_BOARD,
            "board_btn": BTN_BOARD,
            "public_action_history": PUBLIC_HISTORY,
            "fantasy_state": None,
        }
    if phase == "t3_second":
        return {
            "contract_version": POSITION_CONTRACT_VERSION,
            "actor": "btn",
            "turn": 3,
            "phase": phase,
            "board_bb": BB_BOARD_11,
            "board_btn": BTN_BOARD,
            "public_action_history": PUBLIC_HISTORY_AFTER_BB_T3,
            "fantasy_state": None,
        }
    if phase == "t4_first":
        return {
            "contract_version": POSITION_CONTRACT_VERSION,
            "actor": "bb",
            "turn": 4,
            "phase": phase,
            "board_bb": BB_BOARD_11,
            "board_btn": BTN_BOARD_11,
            "public_action_history": PUBLIC_HISTORY_AFTER_T3,
            "fantasy_state": None,
        }
    if phase == "t4_second":
        return {
            "contract_version": POSITION_CONTRACT_VERSION,
            "actor": "btn",
            "turn": 4,
            "phase": phase,
            "board_bb": BB_BOARD_13,
            "board_btn": BTN_BOARD_11,
            "public_action_history": PUBLIC_HISTORY_AFTER_BB_T4,
            "fantasy_state": None,
        }
    raise ValueError(f"unsupported fixture phase: {phase}")


def _own_recall(phase: str) -> PrivateRecall:
    if phase == "t3_first":
        return _recall("bb", "6c", "6s")
    if phase == "t3_second":
        return _recall("btn", "4s", "7h")
    if phase == "t4_first":
        return _append_recall(
            _recall("bb", "6c", "6s"),
            turn=3,
            placements=BB_T3_PLACEMENTS,
            discard="5d",
        )
    if phase == "t4_second":
        return _append_recall(
            _recall("btn", "4s", "7h"),
            turn=3,
            placements=BTN_T3_PLACEMENTS,
            discard="4d",
        )
    raise ValueError(f"unsupported fixture phase: {phase}")


def _current_draw(phase: str, joker: str) -> tuple[str, str, str]:
    if phase == "t3_first":
        return ("Ad", "Kd", joker)
    if phase == "t4_second":
        return ("5s", "8d", joker)
    return ("Ad", "Jd", joker)


def _observation(
    context: PublicRootContext,
    recall: PrivateRecall,
    draw: tuple[str, ...],
) -> InfoSetKey:
    return InfoSetKey(
        contract_version=context.contract_version,
        actor=context.actor,
        turn=context.turn,
        phase=context.phase,
        board_bb=context.board_bb,
        board_btn=context.board_btn,
        public_action_history=context.public_action_history,
        own_recall=recall,
        current_draw=draw,
        fantasy_state=context.fantasy_state,
    )


def _one_natural_assignment(
    pool, count, *, max_particles, seed, observation_digest
):
    del max_particles, seed, observation_digest
    assignment = tuple(card for card in pool if card not in ("X1", "X2"))[:count]
    assert len(assignment) == count
    return (assignment,), math.perm(len(pool), count), False


def _build_case(phase: str):
    model = UniformLegalBehaviorModel()
    bindings = PublicRootBindings.capture(
        behavior_model_id=model.model_id,
        behavior_model_sha256=model.model_sha256,
    )
    context = PublicRootContext.from_public_mapping(
        _phase_public_payload(phase),
        bindings=bindings,
    )
    recall = _own_recall(phase)
    support = []
    for index, (joker, mass) in enumerate(
        (("X1", Fraction(1, 3)), ("X2", Fraction(2, 3)))
    ):
        draw = _current_draw(phase, joker)
        observation = _observation(context, recall, draw)
        root_range = build_history_weighted_full_card_range(
            observation,
            model,
            epsilon=0,
            max_particles=1,
            seed=100 + index,
        )
        support.append(
            ExplicitHypotheticalPrivateType(
                type_id=f"opaque-type-{index}",
                hypothetical_own_recall=recall,
                hypothetical_current_draw=draw,
                prior_mass=mass,
                root_range=root_range,
            )
        )
    result = compile_explicit_public_root_mixture(context, tuple(support))
    return context, tuple(support), result


@pytest.fixture(scope="module")
def compiled_cases():
    patcher = pytest.MonkeyPatch()
    patcher.setattr(range_module, "_select_assignments", _one_natural_assignment)
    try:
        yield {
            phase: _build_case(phase)
            for phase in ("t3_first", "t3_second", "t4_first", "t4_second")
        }
    finally:
        patcher.undo()


def _physical_partition_cards(observation: InfoSetKey, particle) -> list[str]:
    cards = [
        card
        for _turn, _actor, placements in observation.public_action_history
        for card, _row in placements
    ]
    cards.extend(observation.current_draw)
    cards.extend(particle.undealt_cards)
    for recall in (particle.bb_recall, particle.btn_recall):
        cards.extend(card for _turn, card in recall.discards_by_turn)
    return cards


@pytest.mark.parametrize(
    ("phase", "actor", "next_phase"),
    (
        ("t3_first", "bb", "t3_second"),
        ("t3_second", "btn", "t4_first"),
        ("t4_first", "bb", "t4_second"),
    ),
)
def test_real_full_card_bb_btn_and_t4_first_vertical_slice(
    compiled_cases, phase, actor, next_phase
):
    context, _support, result = compiled_cases[phase]

    verification = verify_compiled_public_root_mixture(context, result)
    assert verification["verified"] is True
    assert verification["support_size"] == 2
    assert verification["prior_probability_mass_exact"] == "1/1"
    assert verification["production_sampling_ready"] is False
    assert verification["support_authorization_class"] == (
        EXPLICIT_SUPPORT_AUTHORIZATION_CLASS
    )
    assert verification["authorized_as_production_mccfr_prior"] is False
    assert result.support_authorization_class == EXPLICIT_SUPPORT_AUTHORIZATION_CLASS
    assert result.authorized_as_production_mccfr_prior is False
    assert sum((entry.prior_mass for entry in result.entries), Fraction()) == 1
    assert all(isinstance(entry, MultiRootChanceEntry) for entry in result.entries)
    assert multi_root_module._validated_entries(result.entries) == result.entries

    seen_jokers = set()
    for entry in result.entries:
        assert type(entry.adapter) is FullCardGenerativeAdapter
        assert entry.adapter.observation.phase == phase
        assert entry.adapter.observation.actor == actor
        seen_jokers.update(entry.adapter.observation.current_draw)
        verify_full_card_range(entry.adapter.observation, entry.adapter.root_range)
        sample = entry.adapter.sample_root_for_traversal(random.Random(20260713))
        actions = entry.adapter.legal_actions(sample.state)
        assert actions
        pending = entry.adapter.apply_action_id(sample.state, actions[0][0])
        transition = entry.adapter.sample_next_draw(
            pending,
            random.Random(20260714),
        )
        assert transition.next_phase == next_phase
        assert transition.state.infoset_key.phase == next_phase
        physical = _physical_partition_cards(
            entry.adapter.observation,
            entry.adapter.root_range.particles[0],
        )
        assert len(physical) == 54
        assert Counter(physical) == Counter(ALL_CARDS)
        assert physical.count("X1") == physical.count("X2") == 1
    assert {"X1", "X2"}.issubset(seen_jokers)
    assert len(set(result.private_type_commitments)) == 2


def test_real_full_card_t4_second_compiles_preserves_jokers_and_terminates(
    compiled_cases,
):
    context, _support, result = compiled_cases["t4_second"]
    assert verify_compiled_public_root_mixture(context, result)["verified"] is True
    seen_jokers = set()
    for entry in result.entries:
        observation = entry.adapter.observation
        assert observation.actor == "btn"
        assert observation.phase == "t4_second"
        seen_jokers.update(observation.current_draw)
        sample = entry.adapter.sample_root_for_traversal(random.Random(20260715))
        actions = entry.adapter.legal_actions(sample.state)
        terminal = entry.adapter.apply_action_id(sample.state, actions[0][0])
        assert isinstance(terminal, PublicTreeTerminalState)
        assert math.isfinite(entry.adapter.terminal_utility_bb(terminal))
        physical = _physical_partition_cards(
            observation,
            entry.adapter.root_range.particles[0],
        )
        assert Counter(physical) == Counter(ALL_CARDS)
        assert physical.count("X1") == physical.count("X2") == 1
    assert {"X1", "X2"}.issubset(seen_jokers)


def test_t4_btn_exact_teacher_evidence_is_seedless_and_ingestion_is_bound(
    compiled_cases,
    tmp_path,
):
    context, _support, mixture = compiled_cases["t4_second"]
    selected = mixture.entries[0]
    plan_args = {
        "root_id_sha256": selected.root_id_sha256,
        "full_deal_commitment_sha256": hashlib.sha256(
            b"exact-full-deal"
        ).hexdigest(),
        "public_root_family_commitment_sha256": hashlib.sha256(
            b"exact-root-family"
        ).hexdigest(),
    }
    plan = build_prelabel_solve_root_teacher_plan(
        context,
        mixture,
        **plan_args,
    )
    assert plan.manifest["label_branch"] == "t4_btn_exact"
    assert plan.manifest["sampled_plan"] is None
    with pytest.raises(
        SolveRootTeacherAdapterError,
        match="must not declare sampled provenance",
    ):
        build_prelabel_solve_root_teacher_plan(
            context,
            mixture,
            **plan_args,
            solver_seed_index=0,
        )

    plan_path = write_prelabel_solve_root_teacher_plan(
        tmp_path / "exact-plan.json",
        context,
        mixture,
        plan,
    )
    evidence = build_solve_root_teacher_evidence(
        context,
        mixture,
        prelabel_plan_path=plan_path,
        output_dir=tmp_path / "exact-evidence",
    )
    assert evidence.manifest["branch"] == "t4_btn_exact"
    assert evidence.manifest["checkpoint_evidence"] is None
    assert evidence.manifest["payoff_evidence"] is None
    assert evidence.manifest["online_resolve_binding"] is None
    exact = evidence.manifest["exact_evidence"]
    assert exact["resolver_manifest_sha256"] == evidence.teacher_row["solver"][
        "resolver_manifest_sha256"
    ]
    assert exact["resolver_result_sha256"] == evidence.teacher_row["solver"][
        "resolver_result_sha256"
    ]
    assert exact["chance_sampling_used"] is False
    assert exact["policy_sampling_used"] is False
    assert exact["solver_seed_used"] is False
    assert exact["payoff_seed_used"] is False
    assert exact["checkpoint_used"] is False
    assert "seed_plan" not in evidence.teacher_row["solver"]
    assert "solver_checkpoint_sha256" not in evidence.teacher_row["bindings"]

    bundle = write_evidence_bound_teacher_bundle(
        tmp_path / "bound-teacher",
        [evidence],
        shard_size=1,
    )
    verified = verify_evidence_bound_teacher_bundle(
        bundle.root,
        [evidence],
        expected_manifest_sha256=bundle.manifest["manifest_sha256"],
    )
    assert verified.manifest["row_count"] == 1
    assert verified.manifest["bare_teacher_row_ingestion_allowed"] is False
    with pytest.raises(TypeError, match="bare teacher rows are forbidden"):
        write_evidence_bound_teacher_bundle(
            tmp_path / "bare-row-teacher",
            [dict(evidence.teacher_row)],
            shard_size=1,
        )


def test_audit_is_deterministic_private_payload_free_and_fail_closed(compiled_cases):
    context, support, result = compiled_cases["t3_first"]
    replay = compile_explicit_public_root_mixture(context, tuple(reversed(support)))
    manifest = result.audit_manifest

    assert replay.audit_manifest_json == result.audit_manifest_json
    assert replay.audit_manifest_sha256 == result.audit_manifest_sha256
    assert hashlib.sha256(result.audit_manifest_json.encode()).hexdigest() == (
        result.audit_manifest_sha256
    )
    assert manifest["support_scope"] == EXPLICIT_SUPPORT_SCOPE
    assert manifest["mixture_exactness"] == MIXTURE_EXACTNESS
    assert manifest["full_deck_actor_private_type_enumeration_exhaustive"] is False
    assert (
        manifest[
            "independently_verified_conditional_hidden_assignment_enumeration_exhaustive_for_all_types"
        ]
        is False
    )
    assert manifest["promotion_eligible"] is False
    assert manifest["serving_default_changed"] is False
    assert manifest["production_sampling_ready"] is False
    assert manifest["support_authorization"] == {
        "authorization_class": EXPLICIT_SUPPORT_AUTHORIZATION_CLASS,
        "proposal_origin_commitments_verified": False,
        "promoted_behavior_likelihood_verified": False,
        "conditional_range_posterior_verified": False,
        "posterior_join_normalization_verified": False,
        "authorized_as_production_mccfr_prior": False,
        "required_production_api": (
            "compile_posterior_joined_public_root_mixture"
        ),
    }
    assert manifest["remaining_before_production_sampling"]
    assert manifest["physical_joker_ids"] == ["X1", "X2"]
    assert manifest["joker_physical_identity_collapsed"] is False
    hidden_audit = manifest["hidden_information_audit"]
    assert hidden_audit["public_context_contains_actor_private_fields"] is False
    assert hidden_audit["actual_actor_private_cards_parameter_exists"] is False
    assert hidden_audit["actual_actor_type_selector_consumed"] is False
    assert hidden_audit["declared_support_origin_independently_verified"] is False
    assert hidden_audit["hypothetical_support_raw_cards_serialized_in_audit"] is False
    assert hidden_audit["singleton_actual_hand_conditioning_rejected"] is True
    assert "own_recall" not in manifest["public_context"]
    assert "current_draw" not in manifest["public_context"]
    assert all(item.type_id not in result.audit_manifest_json for item in support)
    for row in manifest["support_rows"]:
        assert set(row) == {
            "root_id_sha256",
            "type_id_sha256",
            "hypothetical_private_type_commitment",
            "prior_mass_exact",
            "observation_digest",
            "range_content_sha256",
            "range_build_sha256",
            "behavior_model_sha256",
            "canonical_full_card_adapter_binding_sha256",
            "producer_claimed_hidden_assignment_enumeration_exhaustive",
            "independently_verified_hidden_assignment_enumeration_exhaustive",
        }
        assert "own_recall" not in row
        assert "current_draw" not in row
        assert row["observation_digest"]
        assert row["range_content_sha256"]
        assert row["range_build_sha256"]
        assert row["behavior_model_sha256"] == context.bindings.behavior_model_sha256
        assert row["canonical_full_card_adapter_binding_sha256"]
        assert (
            row[
                "independently_verified_hidden_assignment_enumeration_exhaustive"
            ]
            is False
        )


def test_public_context_and_compiler_api_have_no_actual_private_input():
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
    assert set(inspect.signature(compile_explicit_public_root_mixture).parameters) == {
        "context",
        "hypothetical_support",
    }


def test_context_parser_rejects_private_or_unknown_leakage_fields(compiled_cases):
    context, _support, _result = compiled_cases["t3_first"]
    payload = _phase_public_payload("t3_first")
    payload["actual_actor_private_cards"] = ["Ad", "Kd", "Qd"]
    with pytest.raises(PublicRootMixtureError, match="leaked forbidden private fields"):
        PublicRootContext.from_public_mapping(payload, bindings=context.bindings)

    payload = _phase_public_payload("t3_first")
    payload["opaque_hidden_hint"] = "not allowed"
    with pytest.raises(PublicRootMixtureError, match="schema mismatch"):
        PublicRootContext.from_public_mapping(payload, bindings=context.bindings)

    private_text = "private-note:X1-Ad-Kd"
    payload = _phase_public_payload("t3_first")
    payload["fantasy_state"] = private_text
    with pytest.raises(PublicRootMixtureError, match="requires fantasy_state=None"):
        PublicRootContext.from_public_mapping(payload, bindings=context.bindings)
    assert private_text not in compiled_cases["t3_first"][2].audit_manifest_json


def test_overlap_and_actual_hand_singleton_fail_closed(compiled_cases):
    context, support, _result = compiled_cases["t3_first"]
    overlapped = replace(
        support[0],
        hypothetical_current_draw=("4h", "Ad", "Kd"),
    )
    with pytest.raises(ValueError, match="overlaps public"):
        compile_explicit_public_root_mixture(context, (overlapped, support[1]))

    with pytest.raises(PublicRootMixtureError, match="singleton actual-hand"):
        compile_explicit_public_root_mixture(context, (support[0],))


def test_exact_fraction_prior_and_duplicate_support_are_enforced(compiled_cases):
    context, support, _result = compiled_cases["t3_first"]
    with pytest.raises(TypeError, match="fractions.Fraction"):
        replace(support[0], prior_mass=0.5)
    with pytest.raises(PublicRootMixtureError, match="sum exactly to one"):
        compile_explicit_public_root_mixture(
            context,
            (
                replace(support[0], prior_mass=Fraction(1, 2)),
                replace(support[1], prior_mass=Fraction(1, 3)),
            ),
        )
    with pytest.raises(PublicRootMixtureError, match="type IDs must be unique"):
        compile_explicit_public_root_mixture(
            context,
            (support[0], replace(support[1], type_id=support[0].type_id)),
        )
    with pytest.raises(PublicRootMixtureError, match="duplicate actor-private types"):
        compile_explicit_public_root_mixture(
            context,
            (
                replace(support[0], prior_mass=Fraction(1, 2)),
                replace(
                    support[0],
                    type_id="opaque-duplicate",
                    prior_mass=Fraction(1, 2),
                ),
            ),
        )


@pytest.mark.parametrize("binding_kind", ("rules", "source"))
def test_live_rule_and_source_hash_bindings_fail_closed(
    compiled_cases, binding_kind
):
    context, support, _result = compiled_cases["t3_first"]
    if binding_kind == "rules":
        bad_bindings = replace(context.bindings, rules_sha256="0" * 64)
        match = "rules hash"
    else:
        sources = dict(context.bindings.source_sha256s)
        first = sorted(sources)[0]
        sources[first] = "0" * 64
        bad_bindings = replace(context.bindings, source_sha256s=sources)
        match = "source hash binding"
    bad_context = replace(context, bindings=bad_bindings)
    with pytest.raises(PublicRootMixtureError, match=match):
        compile_explicit_public_root_mixture(bad_context, support)


def test_live_full_card_semantics_and_range_verifier_identity_fail_closed(
    compiled_cases, monkeypatch
):
    context, support, result = compiled_cases["t3_first"]

    with monkeypatch.context() as patch:
        patch.setattr(
            full_mccfr_module,
            "terminal_metrics",
            lambda _bb, _btn: {"score": 0.0},
        )
        with pytest.raises(ValueError, match="drifted|overridden methods"):
            compile_explicit_public_root_mixture(context, support)
        with pytest.raises(ValueError, match="drifted|overridden methods"):
            verify_compiled_public_root_mixture(context, result)


def test_attestor_lambda_replay_and_solver_sampler_drift_fail_closed(
    compiled_cases, monkeypatch
):
    context, support, result = compiled_cases["t3_first"]
    source_attestor = multi_root_module._multi_root_source_binding
    adapter_attestor = multi_root_module._adapter_checkpoint_binding
    sampler = multi_root_module.sample_exact_fraction_index

    with monkeypatch.context() as patch:
        patch.setattr(
            multi_root_module,
            "_multi_root_source_binding",
            lambda: source_attestor(),
        )
        with pytest.raises(
            PublicRootMixtureError,
            match="source-binding attestor was overridden",
        ):
            compile_explicit_public_root_mixture(context, support)
        with pytest.raises(
            PublicRootMixtureError,
            match="source-binding attestor was overridden",
        ):
            verify_compiled_public_root_mixture(context, result)

    with monkeypatch.context() as patch:
        patch.setattr(
            multi_root_module,
            "_adapter_checkpoint_binding",
            lambda entry: adapter_attestor(entry),
        )
        with pytest.raises(
            PublicRootMixtureError,
            match="adapter-binding attestor was overridden",
        ):
            compile_explicit_public_root_mixture(context, support)
        with pytest.raises(
            PublicRootMixtureError,
            match="adapter-binding attestor was overridden",
        ):
            verify_compiled_public_root_mixture(context, result)

    with monkeypatch.context() as patch:
        patch.setattr(
            multi_root_module,
            "sample_exact_fraction_index",
            lambda weights, rng: sampler(weights, rng),
        )
        with pytest.raises(ValueError, match="runtime globals were overridden"):
            compile_explicit_public_root_mixture(context, support)
        with pytest.raises(ValueError, match="runtime globals were overridden"):
            verify_compiled_public_root_mixture(context, result)

    with monkeypatch.context() as patch:
        patch.setattr(
            full_mccfr_module,
            "get_turn_actions",
            lambda _draw, _board: [],
        )
        with pytest.raises(ValueError, match="drifted|overridden methods"):
            compile_explicit_public_root_mixture(context, support)
        with pytest.raises(ValueError, match="drifted|overridden methods"):
            verify_compiled_public_root_mixture(context, result)

    with monkeypatch.context() as patch:
        patch.setattr(
            range_module,
            "verify_full_card_range",
            lambda _observation, _root_range: {"verified": True},
        )
        with pytest.raises(
            PublicRootMixtureError,
            match="verify_full_card_range binding was overridden",
        ):
            compile_explicit_public_root_mixture(context, support)
        with pytest.raises(
            PublicRootMixtureError,
            match="verify_full_card_range binding was overridden",
        ):
            verify_compiled_public_root_mixture(context, result)


def test_audit_binds_canonical_solver_runtime_and_transitive_sources(compiled_cases):
    context, _support, result = compiled_cases["t3_first"]
    bindings = result.audit_manifest["public_context"]["bindings"]
    assert bindings["canonical_solver_source_binding_sha256"] == (
        context.bindings.canonical_solver_source_binding_sha256
    )
    assert bindings["canonical_solver_runtime_semantic_binding_sha256"] == (
        context.bindings.canonical_solver_runtime_semantic_binding_sha256
    )
    assert {
        "game_engine",
        "scoring",
        "rollout_evaluator",
        "fantasyland_ev",
    }.issubset(bindings["source_sha256s"])
    assert (
        result.audit_manifest[
            "observation_range_behavior_source_rules_and_live_semantics_bound"
        ]
        is True
    )


def test_range_binding_and_committed_audit_tamper_fail_closed(compiled_cases):
    context, support, result = compiled_cases["t3_first"]
    with pytest.raises(ValueError, match="observation digest"):
        compile_explicit_public_root_mixture(
            context,
            (replace(support[0], root_range=support[1].root_range), support[1]),
        )

    tampered = replace(
        result,
        audit_manifest_json=result.audit_manifest_json.replace(
            '"promotion_eligible":false', '"promotion_eligible":true'
        ),
    )
    with pytest.raises(PublicRootMixtureError, match="manifest content mismatch"):
        verify_compiled_public_root_mixture(context, tampered)
