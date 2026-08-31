import pytest

from ai.cfr.game_state import NodeType, OFCState
from ai.engine.turn_order import (
    FIRST_POSITION,
    SECOND_POSITION,
    action_order,
    expected_decision_board_counts,
    normalize_position,
    position_for_seat,
    validate_decision_board_counts,
)


def test_non_button_acts_first_and_button_acts_second():
    assert action_order(0) == (1, 0)
    assert action_order(1) == (0, 1)
    assert position_for_seat(1, 0) == FIRST_POSITION
    assert position_for_seat(0, 0) == SECOND_POSITION


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("bb", "bb"),
        ("first", "bb"),
        ("先行", "bb"),
        ("btn", "btn"),
        ("second", "btn"),
        ("後攻", "btn"),
    ],
)
def test_position_aliases_do_not_confuse_button_with_first(value, expected):
    assert normalize_position(value) == expected


def test_position_and_is_btn_must_agree():
    assert normalize_position(is_btn=False) == "bb"
    assert normalize_position(is_btn=True) == "btn"
    assert normalize_position(is_btn="false") == "bb"
    assert normalize_position(is_btn="true") == "btn"
    with pytest.raises(ValueError, match="contradictory HU position"):
        normalize_position("first", is_btn=True)
    with pytest.raises(ValueError, match="is_btn must be"):
        normalize_position(is_btn="not-a-boolean")


def test_expected_public_board_counts_cover_every_turn_and_position():
    assert [expected_decision_board_counts(turn, "bb") for turn in range(5)] == [
        (0, 0),
        (5, 5),
        (7, 7),
        (9, 9),
        (11, 11),
    ]
    assert [expected_decision_board_counts(turn, "btn") for turn in range(5)] == [
        (0, 5),
        (5, 7),
        (7, 9),
        (9, 11),
        (11, 13),
    ]


def test_t3_contract_rejects_position_board_mismatch():
    validate_decision_board_counts(3, "bb", 9, 9)
    validate_decision_board_counts(3, "btn", 9, 11)
    with pytest.raises(ValueError, match="requires hero/opponent board counts 9/11"):
        validate_decision_board_counts(3, "btn", 9, 9)


def test_cfr_uses_the_same_non_button_then_button_order():
    state = OFCState(
        hands=(
            ["2h", "3h", "4h", "5h", "6h"],
            ["2d", "3d", "4d", "5d", "6d"],
        ),
        btn=0,
    )
    assert state.node_type == NodeType.PLAYER_1

    state.placed = (False, True)
    assert state.node_type == NodeType.PLAYER_0
