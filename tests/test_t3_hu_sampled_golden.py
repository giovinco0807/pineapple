from fractions import Fraction

import pytest

from ai.engine.encoding import Board
from ai.tutor.t3_hu_reference import (
    BeliefWorld,
    PlayerView,
    ReducedT3HUSolver,
    values_by_discard,
    world_values_by_discard,
)


BTN9 = Board(
    top=["5c", "6d", "8c"],
    middle=["9c", "9d", "Jh", "Qs", "Kc"],
    bottom=["As"],
)
BB9 = Board(
    top=["2c", "3d", "4h"],
    middle=["7c", "7d", "8h", "9s", "Tc"],
    bottom=["Qc"],
)
BB11 = Board(
    top=["2c", "3d", "4h"],
    middle=["7c", "7d", "8h", "9s", "Tc"],
    bottom=["Qc", "Qd", "Kh"],
)
KNOWN_SELF_DISCARDS = ["6c", "6s"]


def test_btn_t3_public_belief_averages_before_selecting_root_action():
    view = PlayerView.from_boards(
        actor="btn",
        board_self=BTN9,
        board_opponent=BB11,
        dealt_cards=["2s", "Jd", "Ts"],
        known_discards_self=KNOWN_SELF_DISCARDS,
    )
    worlds = [
        BeliefWorld(
            weight=Fraction(1, 2),
            opponent_private_discards=("4s", "7h", "Th"),
            live=("X1", "X2", "Ac", "Kd", "Ks", "Jc", "Td"),
        ),
        BeliefWorld(
            weight=Fraction(1, 2),
            opponent_private_discards=("4s", "7h", "X1"),
            live=("Th", "X2", "Ac", "Kd", "Ks", "Jc", "Td"),
        ),
    ]

    result = ReducedT3HUSolver().evaluate_public_belief(view, worlds)
    assert result.evaluation_scope == "hu_reduced_root_grouped_pimc"
    assert result.root_strategy_fusion_free is True
    assert result.future_strategy_fusion_free is False
    assert result.equilibrium_approx is False
    expected = values_by_discard(result)
    by_world = world_values_by_discard(result)

    assert expected == pytest.approx(
        {
            "2s": float(Fraction(13, 35)),
            "Jd": float(Fraction(27, 70)),
            "Ts": float(Fraction(-8, 7)),
        }
    )
    assert result.best_action.discard == "Jd"
    assert result.best_value == pytest.approx(float(Fraction(27, 70)))
    assert max(by_world[0], key=by_world[0].get) == "Jd"
    assert max(by_world[1], key=by_world[1].get) == "2s"
    wrong_e_max = sum(max(values.values()) for values in by_world) / 2
    assert wrong_e_max == pytest.approx(float(Fraction(13, 28)))
    assert wrong_e_max != pytest.approx(result.best_value)
    assert result.hu_exact is False
    assert result.inner_t4_exact is True


def test_bb_t3_public_belief_averages_before_selecting_root_action():
    view = PlayerView.from_boards(
        actor="bb",
        board_self=BB9,
        board_opponent=BTN9,
        dealt_cards=["Ad", "Ts", "Qh"],
        known_discards_self=KNOWN_SELF_DISCARDS,
    )
    worlds = [
        BeliefWorld(
            weight=Fraction(1, 2),
            opponent_private_discards=("4s", "Th"),
            live=("X1", "X2", "Ac", "Kd", "Ks", "Jc", "Td", "5s", "Qd"),
        ),
        BeliefWorld(
            weight=Fraction(1, 2),
            opponent_private_discards=("4s", "X1"),
            live=("Th", "X2", "Ac", "Kd", "Ks", "Jc", "Td", "5s", "Qd"),
        ),
    ]

    result = ReducedT3HUSolver().evaluate_public_belief(view, worlds)
    expected = values_by_discard(result)
    by_world = world_values_by_discard(result)

    assert expected == pytest.approx(
        {
            "Ad": float(Fraction(-6247, 1680)),
            "Ts": float(Fraction(-3641, 840)),
            "Qh": float(Fraction(-19651, 3360)),
        }
    )
    assert result.best_action.discard == "Ad"
    assert result.best_value == pytest.approx(float(Fraction(-6247, 1680)))
    assert max(by_world[0], key=by_world[0].get) == "Ts"
    assert max(by_world[1], key=by_world[1].get) == "Ad"
    wrong_e_max = sum(max(values.values()) for values in by_world) / 2
    assert wrong_e_max == pytest.approx(float(Fraction(-1999, 560)))
    assert wrong_e_max != pytest.approx(result.best_value)


def test_reduced_world_rejects_redealing_any_private_discard():
    view = PlayerView.from_boards(
        actor="btn",
        board_self=BTN9,
        board_opponent=BB11,
        dealt_cards=["2s", "Jd", "Ts"],
        known_discards_self=KNOWN_SELF_DISCARDS,
    )
    invalid = BeliefWorld(
        live=("6c", "X2", "Ac", "Kd", "Ks", "Jc", "Td"),
        opponent_private_discards=("4s", "7h", "Th"),
    )

    with pytest.raises(ValueError, match="redeals"):
        ReducedT3HUSolver().world_action_values(view, invalid)


def test_two_jokers_remain_distinct_in_reduced_world():
    assert "X1" != "X2"
    world = BeliefWorld(
        live=("X1", "X2", "Ac", "Kd", "Ks", "Jc", "Td"),
        opponent_private_discards=("4s", "7h", "Th"),
    )
    assert len(world.live) == len(set(world.live))
