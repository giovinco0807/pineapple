from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from ofc_regular.action_space import generate_turn_actions
from ofc_regular.cards import ALL_CARDS
from ofc_regular.decision_trace import (
    attach_replay_truth,
    capture_decision_log_positions,
)
from ofc_regular.hu_infoset import (
    ActorObservation,
    CardFreeMetadata,
    InformationSetError,
    ReplayTruth,
    ScoringContext,
    WorldState,
    card_free_metadata,
    policy_feature_sample_from_record,
    replay_truth_from_record,
)
from ofc_regular.hu_turn3_model import hu_policy_sample, sample_to_matrix
from ofc_regular.policy import RegularAiPolicy
from ofc_regular.state import Board


def _boards():
    hero = Board.from_rows(
        top=("2h", "3h"),
        middle=("4h", "5h"),
        bottom=("6h", "7h", "8h"),
    )
    opponent = Board.from_rows(
        top=("2d", "3d"),
        middle=("4d", "5d"),
        bottom=("6d", "7d", "8d"),
    )
    return hero, opponent


def _observation(opponent_discard: str = "9d") -> ActorObservation:
    hero, opponent = _boards()
    world = WorldState(
        boards=(hero, opponent),
        private_discards=(("9h",), (opponent_discard,)),
        street="T2",
        next_player=0,
    )
    return world.observe(0, ("Th", "Jh", "Qh"))


def _board_with_count(cards):
    cards = tuple(cards)
    return Board.from_rows(
        top=cards[: min(3, len(cards))],
        middle=cards[3 : min(8, len(cards))],
        bottom=cards[8:],
    )


def test_observation_is_invariant_to_opponent_private_discard_identity():
    first = _observation("9d")
    second = _observation("Td")

    assert first == second
    assert first.fingerprint() == second.fingerprint()
    assert not hasattr(first, "opponent_private_discards")
    assert "opponent_private_discards" not in first.to_dict()
    assert "true_dead_cards" not in first.to_dict()


def test_observation_exposes_only_known_cards_and_public_discard_count():
    observation = _observation()

    assert observation.legacy_dead_cards() == (
        *observation.opponent_public_board.all_cards(),
        "9h",
    )
    assert set(observation.known_unavailable_cards()) == {
        *observation.hero_board.all_cards(),
        *observation.opponent_public_board.all_cards(),
        *observation.dealt_cards,
        "9h",
    }
    assert observation.opponent_discard_count == 1
    assert observation.scoring == ScoringContext()


@pytest.mark.parametrize(
    "street,order,hero_count,opponent_count,deal_count,discard_count",
    [
        ("T0", "first", 0, 0, 5, 0),
        ("T0", "second", 0, 5, 5, 0),
        ("T1", "first", 5, 5, 3, 0),
        ("T1", "second", 5, 7, 3, 0),
        ("T2", "first", 7, 7, 3, 1),
        ("T2", "second", 7, 9, 3, 1),
        ("T3", "first", 9, 9, 3, 2),
        ("T3", "second", 9, 11, 3, 2),
        ("T4", "first", 11, 11, 3, 3),
        ("T4", "second", 11, 13, 3, 3),
    ],
)
def test_actor_observation_accepts_only_regular_street_geometry(
    street, order, hero_count, opponent_count, deal_count, discard_count
):
    cursor = 0
    hero_cards = ALL_CARDS[cursor : cursor + hero_count]
    cursor += hero_count
    opponent_cards = ALL_CARDS[cursor : cursor + opponent_count]
    cursor += opponent_count
    dealt = ALL_CARDS[cursor : cursor + deal_count]
    cursor += deal_count
    discards = ALL_CARDS[cursor : cursor + discard_count]
    observation = ActorObservation(
        hero_board=_board_with_count(hero_cards),
        opponent_public_board=_board_with_count(opponent_cards),
        dealt_cards=dealt,
        hero_private_discards=discards,
        seat=order,
        street=street,
        to_act_order=order,
    )
    assert len(observation.fingerprint()) == 64

    if order == "first":
        with pytest.raises(InformationSetError, match="seat must match"):
            ActorObservation(
                hero_board=observation.hero_board,
                opponent_public_board=observation.opponent_public_board,
                dealt_cards=observation.dealt_cards,
                hero_private_discards=observation.hero_private_discards,
                seat="second",
                street=street,
                to_act_order=order,
            )


def test_world_state_rejects_overlapping_private_cards():
    hero, opponent = _boards()

    with pytest.raises(ValueError, match="duplicate card"):
        WorldState(
            boards=(hero, opponent),
            private_discards=(("2h",), ()),
            street="T2",
            next_player=0,
        )


def test_safe_feature_adapter_matches_legacy_runtime_tensor():
    observation = _observation()
    actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    legacy = hu_policy_sample(
        observation.hero_board,
        observation.dealt_cards,
        actions,
        opponent_board=observation.opponent_public_board,
        dead_cards=observation.legacy_dead_cards(),
        seat=observation.seat,
        to_act_order=observation.to_act_order,
    )
    record = {
        **legacy,
        "turn": "T2",
        "visible_dead_cards": list(observation.legacy_dead_cards()),
        "hero_private_discards": list(observation.hero_private_discards),
        "opponent_private_discards": ["9d"],
        "dead_cards": ["9h", "9d"],
        "policy_observation": observation.to_dict(),
    }

    expected, expected_targets = sample_to_matrix(legacy)
    actual, actual_targets = sample_to_matrix(policy_feature_sample_from_record(record))

    assert np.array_equal(actual, expected)
    assert np.array_equal(actual_targets, expected_targets)


def test_safe_feature_adapter_ignores_replay_truth_but_rejects_visible_contamination():
    observation = _observation()
    actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    base = hu_policy_sample(
        observation.hero_board,
        observation.dealt_cards,
        actions,
        opponent_board=observation.opponent_public_board,
        dead_cards=observation.legacy_dead_cards(),
        seat=observation.seat,
        to_act_order=observation.to_act_order,
    )
    base.update(
        {
            "turn": "T2",
            "visible_dead_cards": list(observation.legacy_dead_cards()),
            "hero_private_discards": list(observation.hero_private_discards),
        }
    )
    first = deepcopy(base)
    first.update({"dead_cards": ["9h", "9d"], "opponent_private_discards": ["9d"]})
    second = deepcopy(base)
    second.update({"dead_cards": ["9h", "Td"], "opponent_private_discards": ["Td"]})

    first_features, _ = sample_to_matrix(policy_feature_sample_from_record(first))
    second_features, _ = sample_to_matrix(policy_feature_sample_from_record(second))
    assert np.array_equal(first_features, second_features)

    contaminated = deepcopy(first)
    contaminated["visible_dead_cards"].append("9d")
    with pytest.raises(InformationSetError, match="visible_dead_cards disagree"):
        policy_feature_sample_from_record(contaminated)


def test_regular_policy_observation_adapter_passes_only_actor_visible_cards():
    class RecordingPolicy(RegularAiPolicy):
        def __post_init__(self):
            super().__post_init__()
            self.received = None

        def choose_action(self, board, dealt_cards, **kwargs):
            self.received = (board, tuple(dealt_cards), kwargs)
            return generate_turn_actions(board, dealt_cards)[0]

    observation = _observation()
    policy = RecordingPolicy(seat="first")

    policy.choose_action_observation(observation, hand_id=7, decision_seed=11)

    board, dealt, kwargs = policy.received
    assert board == observation.hero_board
    assert dealt == observation.dealt_cards
    assert kwargs["dead_cards"] == observation.legacy_dead_cards()
    assert "9d" not in kwargs["dead_cards"]
    assert kwargs["opponent_board"] == observation.opponent_public_board
    assert kwargs["street"] == "T2"


def test_card_free_metadata_rejects_hidden_fields_on_all_mutation_paths():
    metadata = card_free_metadata({"runtime_profile": "stage19_p0", "nested": {"ok": 1}})

    assert isinstance(metadata, CardFreeMetadata)
    assert isinstance(metadata["nested"], CardFreeMetadata)
    with pytest.raises(InformationSetError, match="true_dead_cards"):
        metadata.update({"true_dead_cards": ["2c"]})
    with pytest.raises(InformationSetError, match="opponent_private_discards"):
        metadata["opponent_private_discards"] = ["3c"]
    with pytest.raises(InformationSetError, match="replay_world"):
        metadata["nested"].update({"replay_world": {}})
    with pytest.raises(InformationSetError, match="deck_tail"):
        card_free_metadata({"nested": [{"deck_tail": ["4c"]}]})
    with pytest.raises(InformationSetError, match="card-bearing object"):
        card_free_metadata(
            {"nested": [ReplayTruth(("2c",), ("2c",), ("2c",), ())]}
        )


def test_replay_truth_requires_visible_hero_cards_and_checks_explicit_empty_alias():
    with pytest.raises(ValueError, match="hero private discards must be actor-visible"):
        ReplayTruth(("2c",), (), ("2c",), ())

    truth = ReplayTruth(
        true_dead_cards=("2c", "3d"),
        visible_dead_cards=("2c",),
        hero_private_discards=("2c",),
        opponent_private_discards=("3d",),
    )
    record = {
        "replay_truth": truth.to_dict(),
        "opponent_private_discards": [],
    }
    with pytest.raises(
        InformationSetError, match="opponent_private_discards disagrees"
    ):
        replay_truth_from_record(record)


def test_policy_context_assignment_and_subclass_contexts_are_always_guarded():
    policy = RegularAiPolicy(decision_context={"runtime_profile": "safe"})

    with pytest.raises(InformationSetError, match="future_cards"):
        policy.decision_context = {"future_cards": ["2c"]}
    with pytest.raises(InformationSetError, match="world"):
        policy.hu_turn2_context = {"world": object()}
    with pytest.raises(InformationSetError, match="remaining_deck"):
        policy.topk_context = {"remaining_deck": ["3c"]}

    policy.hu_turn2_context = {"runtime_status": "safe"}
    with pytest.raises(InformationSetError, match="private_discards"):
        policy.hu_turn2_context.update({"private_discards": ["4c"]})


def test_replay_truth_matches_world_but_is_not_part_of_observation():
    hero, opponent = _boards()
    world = WorldState(
        boards=(hero, opponent),
        private_discards=(("9h",), ("9d",)),
        street="T2",
        next_player=0,
    )
    observation = world.observe(0, ("Th", "Jh", "Qh"))

    truth = ReplayTruth.from_world(world, actor=0, observation=observation)

    assert truth.hero_private_discards == ("9h",)
    assert truth.opponent_private_discards == ("9d",)
    assert "9d" not in observation.known_unavailable_cards()
    assert '"9d"' not in json.dumps(observation.to_dict(), sort_keys=True)


def test_replay_attachment_preserves_actor_visible_fields_and_uses_true_prefixes():
    observation = _observation()
    visible = list(observation.legacy_dead_cards())
    policy = SimpleNamespace(hu_turn1_decision_log=[])
    positions = capture_decision_log_positions(policy)
    record = {
        "dead_cards": list(visible),
        "visible_dead_cards": list(visible),
        "hero_private_discards": ["9h"],
        "visibility_model": "actor_observation_v1",
        "replay_ready": False,
    }
    policy.hu_turn1_decision_log.append(record)
    truth = ReplayTruth(
        true_dead_cards=("9h", "9d"),
        visible_dead_cards=tuple(visible),
        hero_private_discards=("9h",),
        opponent_private_discards=("9d",),
    )

    assert attach_replay_truth(positions, truth) == 1

    assert record["dead_cards"] == visible
    assert record["visible_dead_cards"] == visible
    assert record["hero_private_discards"] == ["9h"]
    assert "opponent_private_discards" not in record
    assert record["visibility_model"] == "actor_observation_v1"
    assert record["true_dead_cards"] == ["9h", "9d"]
    assert record["true_hero_private_discards"] == ["9h"]
    assert record["true_opponent_private_discards"] == ["9d"]
    assert replay_truth_from_record(record) == truth


def test_replay_truth_compatibility_accepts_only_unambiguous_legacy_rows():
    visible = list(_observation().legacy_dead_cards())
    legacy = {
        "visibility_model": "hidden_discard",
        "dead_cards": ["9h", "9d"],
        "true_dead_cards": ["9h", "9d"],
        "visible_dead_cards": visible,
        "hero_private_discards": ["9h"],
        "opponent_private_discards": ["9d"],
    }

    truth = replay_truth_from_record(legacy)
    assert truth.visible_dead_cards == tuple(visible)
    assert truth.opponent_private_discards == ("9d",)

    ambiguous = dict(legacy)
    ambiguous.pop("true_dead_cards")
    ambiguous["dead_cards"] = visible
    with pytest.raises(InformationSetError, match="uniquely identify"):
        replay_truth_from_record(ambiguous)

    contradictory = dict(legacy)
    contradictory["visible_dead_cards"] = [*visible, "9d"]
    with pytest.raises(InformationSetError, match="invalid replay truth"):
        replay_truth_from_record(contradictory)
