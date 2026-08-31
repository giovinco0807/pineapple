import json
import math
from collections import Counter
from dataclasses import replace
from fractions import Fraction

import pytest

import ai.tutor.t3_hu_full_card_range as full_card_range_module
from ai.engine.encoding import ALL_CARDS
from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor.t3_hu_full_card_range import (
    BehaviorDistribution,
    FrozenBehaviorTable,
    UniformLegalBehaviorModel,
    build_history_weighted_full_card_range,
    verify_full_card_range,
)
from ai.tutor.t3_hu_public_cfr import InfoSetKey, JointParticle, PrivateRecall


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
    (0, "bb", (("4h", "top"), ("2c", "top"), ("3d", "top"), ("7c", "middle"), ("Qc", "bottom"))),
    (0, "btn", (("8c", "top"), ("5c", "top"), ("6d", "top"), ("9c", "middle"), ("As", "bottom"))),
    (1, "bb", (("7d", "middle"), ("8h", "middle"))),
    (1, "btn", (("9d", "middle"), ("Jh", "middle"))),
    (2, "bb", (("9s", "middle"), ("Tc", "middle"))),
    (2, "btn", (("Qs", "middle"), ("Kc", "middle"))),
)
CURRENT_DRAW = ("Ad", "Kd", "Qd")
BB_T3_PLACEMENTS = (("Qd", "bottom"), ("Kh", "bottom"))
BB_BOARD_11 = (BB_BOARD[0], BB_BOARD[1], ("Qc", "Qd", "Kh"))
PUBLIC_HISTORY_AFTER_BB_T3 = PUBLIC_HISTORY + ((3, "bb", BB_T3_PLACEMENTS),)
BTN_T3_PLACEMENTS = (("2h", "bottom"), ("3h", "bottom"))
BTN_BOARD_11 = (BTN_BOARD[0], BTN_BOARD[1], ("As", "2h", "3h"))
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


def _observation() -> InfoSetKey:
    particle = JointParticle(
        bb_recall=_recall("bb", "6c", "6s"),
        btn_recall=_recall("btn", "4s", "7h"),
        undealt_cards=("X1", "X2", "Ac"),
    )
    return InfoSetKey.for_particle(
        particle,
        contract_version=POSITION_CONTRACT_VERSION,
        actor="bb",
        turn=3,
        phase="t3_first",
        board_bb=BB_BOARD,
        board_btn=BTN_BOARD,
        public_action_history=PUBLIC_HISTORY,
        current_draw=CURRENT_DRAW,
    )


def _append_recall(
    recall: PrivateRecall,
    *,
    turn: int,
    placements: tuple[tuple[str, str], ...],
    discard: str,
) -> PrivateRecall:
    placed_cards = tuple(card for card, _row in placements)
    return PrivateRecall(
        dealt_by_turn=recall.dealt_by_turn
        + ((turn, (*placed_cards, discard)),),
        discards_by_turn=recall.discards_by_turn + ((turn, discard),),
    )


def _later_phase_observation(phase: str) -> InfoSetKey:
    bb_t2 = _recall("bb", "6c", "6s")
    btn_t2 = _recall("btn", "4s", "7h")
    bb_t3 = _append_recall(
        bb_t2,
        turn=3,
        placements=BB_T3_PLACEMENTS,
        discard="5d",
    )
    btn_t3 = _append_recall(
        btn_t2,
        turn=3,
        placements=BTN_T3_PLACEMENTS,
        discard="4d",
    )

    if phase == "t3_second":
        particle = JointParticle(
            bb_recall=bb_t3,
            btn_recall=btn_t2,
            undealt_cards=("X1", "X2", "Ac"),
        )
        args = {
            "actor": "btn",
            "turn": 3,
            "board_bb": BB_BOARD_11,
            "board_btn": BTN_BOARD,
            "public_action_history": PUBLIC_HISTORY_AFTER_BB_T3,
            "current_draw": ("Ad", "Kd", "Jd"),
        }
    elif phase == "t4_first":
        particle = JointParticle(
            bb_recall=bb_t3,
            btn_recall=btn_t3,
            undealt_cards=("X1", "X2", "Ac"),
        )
        args = {
            "actor": "bb",
            "turn": 4,
            "board_bb": BB_BOARD_11,
            "board_btn": BTN_BOARD_11,
            "public_action_history": PUBLIC_HISTORY_AFTER_T3,
            "current_draw": ("Ad", "Kd", "Jd"),
        }
    elif phase == "t4_second":
        bb_t4 = _append_recall(
            bb_t3,
            turn=4,
            placements=BB_T4_PLACEMENTS,
            discard="Jd",
        )
        particle = JointParticle(
            bb_recall=bb_t4,
            btn_recall=btn_t3,
            undealt_cards=("X1", "X2", "Ac"),
        )
        args = {
            "actor": "btn",
            "turn": 4,
            "board_bb": BB_BOARD_13,
            "board_btn": BTN_BOARD_11,
            "public_action_history": PUBLIC_HISTORY_AFTER_BB_T4,
            "current_draw": ("5s", "8d", "9h"),
        }
    else:
        raise ValueError(f"unsupported later phase fixture: {phase}")

    return InfoSetKey.for_particle(
        particle,
        contract_version=POSITION_CONTRACT_VERSION,
        phase=phase,
        **args,
    )


def _rebuild(observation: InfoSetKey, particle: JointParticle) -> InfoSetKey:
    return InfoSetKey.for_particle(
        particle,
        contract_version=observation.contract_version,
        actor=observation.actor,
        turn=observation.turn,
        phase=observation.phase,
        board_bb=observation.board_bb,
        board_btn=observation.board_btn,
        public_action_history=observation.public_action_history,
        current_draw=observation.current_draw,
    )


def _physical_card_atoms(value) -> set[str]:
    if isinstance(value, str):
        return {value} if value in ALL_CARDS else set()
    if isinstance(value, dict):
        return set().union(
            *(_physical_card_atoms(item) for item in value.values()), set()
        )
    if isinstance(value, (list, tuple)):
        return set().union(*(_physical_card_atoms(item) for item in value), set())
    return set()


def test_uniform_full_support_is_exact_930_world_range_with_full_deck_partitions():
    observation = _observation()
    result = build_history_weighted_full_card_range(
        observation,
        UniformLegalBehaviorModel(),
        epsilon=0,
        max_particles=None,
    )

    assert result.particle_count == 930  # 31P2 ordered hidden discards
    assert result.effective_sample_size == 930
    assert result.metadata["candidate_assignment_count"] == 930
    assert result.metadata["exhaustive_hidden_discard_enumeration"] is True
    assert result.metadata["full_card_physical_remainder"] is True
    assert all(particle.weight == Fraction(1, 930) for particle in result.particles)
    public = [
        card
        for _turn, _actor, placements in observation.public_action_history
        for card, _row in placements
    ]
    for particle in result.particles:
        assert len(particle.undealt_cards) == 29
        assert _rebuild(observation, particle) == observation
        physical = [*public, *observation.current_draw, *particle.undealt_cards]
        for recall in (particle.bb_recall, particle.btn_recall):
            physical.extend(card for _turn, card in recall.discards_by_turn)
        assert len(physical) == 54
        assert Counter(physical) == Counter(ALL_CARDS)
        assert physical.count("X1") == physical.count("X2") == 1


@pytest.mark.parametrize(
    ("phase", "expected_undealt", "expected_hidden_turns"),
    [
        ("t3_second", 26, [1, 2, 3]),
        ("t4_first", 23, [1, 2, 3]),
        ("t4_second", 20, [1, 2, 3, 4]),
    ],
)
def test_later_phase_ranges_partition_54_cards_and_keep_jokers_distinct(
    monkeypatch, phase, expected_undealt, expected_hidden_turns
):
    observation = _later_phase_observation(phase)
    hidden_count = len(expected_hidden_turns)
    natural_tail = ("2d", "3c") if hidden_count == 3 else ("2d", "3c", "4c")

    def select_joker_worlds(
        pool, count, *, max_particles, seed, observation_digest
    ):
        assert count == hidden_count
        assert {"X1", "X2", *natural_tail}.issubset(pool)
        return (
            (("X1", *natural_tail), ("X2", *natural_tail)),
            math.perm(len(pool), count),
            False,
        )

    monkeypatch.setattr(
        full_card_range_module, "_select_assignments", select_joker_worlds
    )
    result = build_history_weighted_full_card_range(
        observation,
        UniformLegalBehaviorModel(),
        epsilon=0,
        max_particles=2,
        seed=17,
    )

    assert result.particle_count == 2
    assert result.metadata["opponent_hidden_turns"] == expected_hidden_turns
    assert result.particle_commitments[0] != result.particle_commitments[1]
    assert all(particle.weight == Fraction(1, 2) for particle in result.particles)
    public = [
        card
        for _turn, _actor, placements in observation.public_action_history
        for card, _row in placements
    ]
    hidden_joker_worlds = set()
    for particle in result.particles:
        assert len(particle.undealt_cards) == expected_undealt
        assert _rebuild(observation, particle) == observation
        physical = [*public, *observation.current_draw, *particle.undealt_cards]
        for recall in (particle.bb_recall, particle.btn_recall):
            physical.extend(card for _turn, card in recall.discards_by_turn)
        assert len(physical) == 54
        assert Counter(physical) == Counter(ALL_CARDS)
        assert physical.count("X1") == physical.count("X2") == 1

        opponent_recall = (
            particle.btn_recall if observation.actor == "bb" else particle.bb_recall
        )
        hidden_jokers = tuple(
            card
            for _turn, card in opponent_recall.discards_by_turn
            if card in ("X1", "X2")
        )
        hidden_joker_worlds.add(hidden_jokers)

    assert hidden_joker_worlds == {("X1",), ("X2",)}


def test_exhaustive_range_content_is_seed_independent_but_build_hash_binds_seed():
    observation = _observation()
    model = UniformLegalBehaviorModel()
    seed_17 = build_history_weighted_full_card_range(
        observation, model, epsilon=0, max_particles=None, seed=17
    )
    seed_18 = build_history_weighted_full_card_range(
        observation, model, epsilon=0, max_particles=None, seed=18
    )

    assert seed_17.particle_commitments == seed_18.particle_commitments
    assert tuple(particle.weight for particle in seed_17.particles) == tuple(
        particle.weight for particle in seed_18.particles
    )
    assert seed_17.effective_sample_size == seed_18.effective_sample_size == 930
    assert seed_17.range_content_sha256 == seed_18.range_content_sha256
    assert seed_17.range_build_sha256 != seed_18.range_build_sha256


def test_hash_priority_sample_is_reproducible_and_seed_changes_support_not_policy_key():
    observation = _observation()
    model = UniformLegalBehaviorModel()
    first = build_history_weighted_full_card_range(
        observation, model, epsilon=0, max_particles=16, seed=17
    )
    replay = build_history_weighted_full_card_range(
        observation, model, epsilon=0, max_particles=16, seed=17
    )
    other = build_history_weighted_full_card_range(
        observation, model, epsilon=0, max_particles=16, seed=18
    )

    assert first.particle_commitments == replay.particle_commitments
    assert first.range_sha256 == replay.range_sha256
    assert first.particle_commitments != other.particle_commitments
    assert first.range_sha256 != other.range_sha256
    assert first.observation_digest == other.observation_digest == observation.digest()
    assert all(_rebuild(observation, particle) == observation for particle in other.particles)


def test_independent_range_verifier_rederives_physics_hashes_ess_and_behavior_audit():
    observation = _observation()
    result = build_history_weighted_full_card_range(
        observation,
        UniformLegalBehaviorModel(),
        epsilon=0,
        max_particles=16,
        seed=17,
    )

    verified = verify_full_card_range(observation, result)

    assert verified["verified"] is True
    assert verified["particle_count"] == 16
    assert verified["behavior_uniform_fallback_count"] == 0
    assert verified["behavior_query_count"] == sum(
        entry["evaluation_count"]
        for entry in result.metadata["build_manifest"]["behavior_query_audit"]
    )
    assert verified["range_content_sha256"] == result.range_content_sha256
    assert verified["range_build_sha256"] == result.range_build_sha256


def test_independent_range_verifier_rejects_tampered_particle_weight():
    observation = _observation()
    result = build_history_weighted_full_card_range(
        observation,
        UniformLegalBehaviorModel(),
        epsilon=0,
        max_particles=4,
        seed=17,
    )
    first = result.particles[0]
    tampered_first = JointParticle(
        bb_recall=first.bb_recall,
        btn_recall=first.btn_recall,
        undealt_cards=first.undealt_cards,
        weight=first.weight + Fraction(1, 100),
    )
    tampered = replace(
        result,
        particles=(tampered_first, *result.particles[1:]),
    )

    with pytest.raises(ValueError, match="sum exactly to one"):
        verify_full_card_range(observation, tampered)


def _observed_action_id(information) -> str:
    placements = next(
        placements
        for turn, actor, placements in PUBLIC_HISTORY_AFTER_BB_T4
        if turn == information.turn and actor == information.actor
    )
    placed_cards = {card for card, _row in placements}
    discard_cards = set(information.current_draw) - placed_cards
    assert len(discard_cards) == 1
    discard = next(iter(discard_cards))
    matching = []
    for action_id in information.legal_action_ids:
        payload = json.loads(action_id)
        action_placements = {
            (card, row) for card, row in payload["placements"]
        }
        if payload["discard"] == discard and action_placements == set(placements):
            matching.append(action_id)
    assert len(matching) == 1
    return matching[0]


def _distribution_with_target_probability(
    information, target_action: str, probability: Fraction
) -> dict[str, Fraction]:
    assert target_action in information.legal_action_ids
    assert 0 <= probability <= 1
    other_actions = [
        action_id
        for action_id in information.legal_action_ids
        if action_id != target_action
    ]
    if not other_actions:
        assert probability == 1
        return {target_action: Fraction(1)}
    remainder = (1 - probability) / len(other_actions)
    return {
        action_id: probability if action_id == target_action else remainder
        for action_id in information.legal_action_ids
    }


def _uniform_distribution(information) -> dict[str, Fraction]:
    probability = Fraction(1, information.legal_action_count)
    return {action_id: probability for action_id in information.legal_action_ids}


class _ExactTestBehavior:
    model_id = "exact_test_behavior"

    @property
    def model_manifest(self):
        return {
            "schema": "ofc_frozen_behavior_model/v1",
            "model_id": self.model_id,
            "model_type": type(self).__name__,
            "position_contract_version": POSITION_CONTRACT_VERSION,
        }

    @property
    def model_sha256(self):
        return full_card_range_module._canonical_sha256(self.model_manifest)

    def distribution_source(self, information):
        return "test_exact_distribution"

    def action_distribution(self, information):
        return BehaviorDistribution(
            information_digest=information.digest(),
            probabilities=self._action_probabilities(information),
            source="model",
            used_fallback=False,
        )


class _JokerDiscardBehavior(_ExactTestBehavior):
    model_id = "test_joker_discard_behavior"

    def _action_probabilities(self, information):
        target_action = _observed_action_id(information)
        discard = json.loads(target_action)["discard"]
        probability = Fraction(3, 4) if discard == "X1" else Fraction(1, 4)
        return _distribution_with_target_probability(
            information, target_action, probability
        )


def test_sequential_behavior_likelihood_changes_posterior_and_exact_ess():
    observation = _observation()
    result = build_history_weighted_full_card_range(
        observation,
        _JokerDiscardBehavior(),
        epsilon=0,
        max_particles=None,
    )

    x1_weights = []
    natural_weights = []
    for particle in result.particles:
        discards = dict(particle.btn_recall.discards_by_turn)
        if "X1" in discards.values():
            x1_weights.append(particle.weight)
        else:
            natural_weights.append(particle.weight)
    assert x1_weights and natural_weights
    assert x1_weights[0] / natural_weights[0] == 3
    assert result.effective_sample_size < result.particle_count
    assert sum((particle.weight for particle in result.particles), Fraction(0, 1)) == 1
    assert result.metadata["history_weighted"] is True
    assert result.metadata["policy_key_contains_hidden_assignment"] is False


class _TwoWorldBehavior(_ExactTestBehavior):
    model_id = "test_two_world_behavior"

    def _action_probabilities(self, information):
        assert information.legal_action_count == 12
        target_action = _observed_action_id(information)
        discard = json.loads(target_action)["discard"]
        if information.turn == 1:
            probability = {
                "X1": Fraction(3, 4),
                "X2": Fraction(1, 4),
            }[discard]
        elif information.turn == 2:
            probability = {"X1": Fraction(1), "X2": Fraction(1, 2)}[discard]
        else:
            raise AssertionError(f"unexpected behavior turn: {information.turn}")
        return _distribution_with_target_probability(
            information, target_action, probability
        )


@pytest.mark.parametrize(
    ("epsilon", "expected_weights", "expected_ess", "expected_evidence"),
    [
        (
            Fraction(0),
            {("X1", "X2"): Fraction(3, 5), ("X2", "X1"): Fraction(2, 5)},
            Fraction(25, 13),
            Fraction(5, 16),
        ),
        (
            Fraction(1, 10),
            {
                ("X1", "X2"): Fraction(2255, 3781),
                ("X2", "X1"): Fraction(1526, 3781),
            },
            Fraction(14295961, 7413701),
            Fraction(3781, 14400),
        ),
    ],
)
def test_two_world_sequential_bayes_and_ess_are_exact(
    monkeypatch, epsilon, expected_weights, expected_ess, expected_evidence
):
    def select_two_worlds(
        pool, count, *, max_particles, seed, observation_digest
    ):
        assert count == 2
        assert {"X1", "X2"}.issubset(pool)
        return (("X1", "X2"), ("X2", "X1")), 930, False

    monkeypatch.setattr(
        full_card_range_module, "_select_assignments", select_two_worlds
    )
    result = build_history_weighted_full_card_range(
        _observation(),
        _TwoWorldBehavior(),
        epsilon=epsilon,
        max_particles=2,
        seed=17,
    )
    actual_weights = {
        tuple(card for _turn, card in particle.btn_recall.discards_by_turn): particle.weight
        for particle in result.particles
    }

    assert actual_weights == expected_weights
    assert result.effective_sample_size == expected_ess
    assert result.evidence_normalizer == expected_evidence
    assert result.metadata[
        "opponent_public_evidence_normalizer_exact"
    ] == f"{expected_evidence.numerator}/{expected_evidence.denominator}"
    assert result.metadata["build_manifest"][
        "opponent_public_evidence_scope"
    ] == "deterministic_sample_mean_hidden_assignment_marginal_likelihood"
    assert result.metadata["build_manifest"]["sampled_assignment_count"] == 2
    assert result.metadata["build_manifest"]["candidate_assignment_count"] == 930
    assert sum(actual_weights.values(), Fraction(0)) == 1


class _ZeroOnX1Behavior(_ExactTestBehavior):
    model_id = "test_zero_on_x1"

    def _action_probabilities(self, information):
        target_action = _observed_action_id(information)
        discard = json.loads(target_action)["discard"]
        probability = Fraction(0) if discard == "X1" else Fraction(1, 2)
        return _distribution_with_target_probability(
            information, target_action, probability
        )


def test_epsilon_keeps_legal_off_path_x1_worlds_reachable():
    observation = _observation()
    without_smoothing = build_history_weighted_full_card_range(
        observation,
        _ZeroOnX1Behavior(),
        epsilon=0,
        max_particles=None,
    )
    smoothed = build_history_weighted_full_card_range(
        observation,
        _ZeroOnX1Behavior(),
        epsilon=Fraction(1, 10),
        max_particles=None,
    )

    assert all(
        "X1" not in dict(particle.btn_recall.discards_by_turn).values()
        for particle in without_smoothing.particles
    )
    assert any(
        "X1" in dict(particle.btn_recall.discards_by_turn).values()
        for particle in smoothed.particles
    )
    assert all(particle.weight > 0 for particle in smoothed.particles)


class _InvalidFloatBehavior(_ExactTestBehavior):
    model_id = "invalid_float"

    def _action_probabilities(self, information):
        probability = 1.0 / information.legal_action_count
        return {
            action_id: probability for action_id in information.legal_action_ids
        }


def test_behavior_model_must_return_exact_probabilities():
    with pytest.raises(TypeError, match="fractions.Fraction"):
        build_history_weighted_full_card_range(
            _observation(),
            _InvalidFloatBehavior(),
            max_particles=1,
        )


class _CaptureBehavior(_ExactTestBehavior):
    model_id = "capture_behavior"

    def __init__(self):
        self.informations = []

    def _action_probabilities(self, information):
        self.informations.append(information)
        return _uniform_distribution(information)


class _HistoryCutoffSpy(_ExactTestBehavior):
    model_id = "history_cutoff_spy"

    def __init__(self):
        self.informations = []

    def _action_probabilities(self, information):
        self.informations.append(information)
        return _uniform_distribution(information)


@pytest.mark.parametrize(
    ("phase", "expected_decision_count"),
    [("t3_first", 2), ("t3_second", 3)],
)
def test_behavior_queries_stop_at_the_historical_action_and_exclude_future_state(
    phase, expected_decision_count
):
    observation = _observation() if phase == "t3_first" else _later_phase_observation(phase)
    opponent = "btn" if observation.actor == "bb" else "bb"
    opponent_indices = [
        index
        for index, (turn, actor, _placements) in enumerate(
            observation.public_action_history
        )
        if actor == opponent and turn > 0
    ]
    spy = _HistoryCutoffSpy()
    result = build_history_weighted_full_card_range(
        observation,
        spy,
        epsilon=0,
        max_particles=1,
        seed=17,
    )

    assert len(spy.informations) == expected_decision_count == len(opponent_indices)
    particle = result.particles[0]
    for information, history_index in zip(spy.informations, opponent_indices):
        turn, actor, _placements = observation.public_action_history[history_index]
        assert information.actor == actor == opponent
        assert information.turn == turn
        assert information.public_action_history == observation.public_action_history[:history_index]

        payload = information.to_canonical_dict()
        assert not {
            "undealt_cards",
            "remaining_cards",
            "remaining_deck",
            "particle_id",
            "world_id",
            "seed",
            "rng_seed",
        }.intersection(payload)
        query_cards = _physical_card_atoms(payload)
        assert query_cards.isdisjoint(observation.current_draw)
        assert query_cards.isdisjoint(particle.undealt_cards)

        future_public_cards = {
            card
            for _future_turn, _future_actor, placements in observation.public_action_history[
                history_index + 1 :
            ]
            for card, _row in placements
        }
        assert query_cards.isdisjoint(future_public_cards)


def test_frozen_behavior_table_rejects_incomplete_action_coverage():
    capture = _CaptureBehavior()
    build_history_weighted_full_card_range(
        _observation(), capture, epsilon=0, max_particles=1, seed=17
    )
    information = capture.informations[0]
    missing_row_table = FrozenBehaviorTable({})
    assert missing_row_table.uniform_fallback is False
    with pytest.raises(KeyError, match="has no row"):
        missing_row_table.action_distribution(information)

    missing_action = information.legal_action_ids[0]
    retained = [
        action_id
        for action_id in information.legal_action_ids
        if action_id != missing_action
    ]
    table = FrozenBehaviorTable(
        {
            information.digest(): {
                action_id: Fraction(1, len(retained)) for action_id in retained
            }
        }
    )

    with pytest.raises(ValueError, match="action coverage mismatch"):
        table.action_distribution(information)


def test_explicit_uniform_fallback_is_atomic_and_fully_audited():
    model = FrozenBehaviorTable({}, uniform_fallback=True)
    result = build_history_weighted_full_card_range(
        _observation(),
        model,
        epsilon=0,
        max_particles=1,
        seed=17,
    )

    assert result.metadata["behavior_query_count"] == 2
    assert result.metadata["behavior_model_evaluation_count"] == 2
    assert result.metadata["behavior_distribution_source_counts"] == {
        "uniform_fallback": 2
    }
    assert result.metadata["behavior_uniform_fallback_count"] == 2
    assert result.metadata["behavior_uniform_fallback_unique_count"] == 2
    assert result.metadata["behavior_model_hit_rate_exact"] == "0/1"
    audits = result.metadata["behavior_query_audit"]
    assert len(audits) == 2
    assert sum(audit["evaluation_count"] for audit in audits) == 2
    assert all(audit["source"] == "uniform_fallback" for audit in audits)
    assert all(audit["used_fallback"] is True for audit in audits)


class _MetadataBehavior(_ExactTestBehavior):
    def __init__(self, model_id, model_sha256):
        self.model_id = model_id
        self._model_sha256 = model_sha256

    @property
    def model_sha256(self):
        return self._model_sha256

    def _action_probabilities(self, information):
        return _uniform_distribution(information)


@pytest.mark.parametrize(
    ("model_id", "model_sha256"),
    [
        ("", "f" * 64),
        ("valid_model", "too-short"),
        ("valid_model", "z" * 64),
    ],
)
def test_behavior_model_identity_and_hash_are_validated(model_id, model_sha256):
    with pytest.raises(ValueError):
        build_history_weighted_full_card_range(
            _observation(),
            _MetadataBehavior(model_id, model_sha256),
            epsilon=0,
            max_particles=1,
            seed=17,
        )


class _ManifestHashMismatchBehavior(_ExactTestBehavior):
    model_id = "manifest_hash_mismatch"

    @property
    def model_sha256(self):
        return "0" * 64

    def _action_probabilities(self, information):
        return _uniform_distribution(information)


def test_behavior_model_manifest_must_match_declared_hash():
    with pytest.raises(ValueError, match="canonical model_manifest"):
        build_history_weighted_full_card_range(
            _observation(),
            _ManifestHashMismatchBehavior(),
            epsilon=0,
            max_particles=1,
            seed=17,
        )


class _IncompleteDistributionBehavior(_ExactTestBehavior):
    model_id = "incomplete_distribution"

    def _action_probabilities(self, information):
        retained = information.legal_action_ids[1:]
        probability = Fraction(1, len(retained))
        return {action_id: probability for action_id in retained}


def test_behavior_model_distribution_requires_every_legal_action():
    with pytest.raises(ValueError, match="action coverage mismatch"):
        build_history_weighted_full_card_range(
            _observation(),
            _IncompleteDistributionBehavior(),
            epsilon=0,
            max_particles=1,
            seed=17,
        )
