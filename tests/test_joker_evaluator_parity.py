"""Cross-engine parity: ported joker evaluator vs the main-track reference.

Acceptance-gate pilot for the accepted engine-unification decision
(``ai/docs/engine_unification_decision_20260728.md``, gate 1/2): the canonical
Joker board evaluation ported into this repository must agree with
``ai/engine/game_engine.py`` on busted flags, royalties, Fantasyland chain
entry, and pairwise line results, across Joker 0/1/2 strata and the known
edge cases (wheel straight, double joker, royal flush).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

from ofc_regular.joker_evaluator import (
    evaluate_board_with_joker_constraint,
    evaluate_hand_joker,
)
from ofc_regular.joker_rules import ALL_CARDS_54, JOKER_CHAIN_RULES

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from ai.engine import game_engine as reference  # noqa: E402


def _random_board(rng: random.Random, joker_count: int) -> list[list[str]]:
    naturals = [card for card in ALL_CARDS_54 if card not in ("X1", "X2")]
    rng.shuffle(naturals)
    cards = naturals[: 13 - joker_count] + list(("X1", "X2")[:joker_count])
    rng.shuffle(cards)
    return [cards[:3], cards[3:8], cards[8:13]]


def _random_board_pair(
    rng: random.Random, joker_split: tuple[int, int]
) -> tuple[list[list[str]], list[list[str]]]:
    naturals = [card for card in ALL_CARDS_54 if card not in ("X1", "X2")]
    rng.shuffle(naturals)
    jokers = ["X1", "X2"]
    boards = []
    for count in joker_split:
        cards = [naturals.pop() for _ in range(13 - count)]
        cards.extend(jokers.pop() for _ in range(count))
        rng.shuffle(cards)
        boards.append([cards[:3], cards[3:8], cards[8:13]])
    return boards[0], boards[1]


def _sign(a, b) -> int:
    return (a > b) - (a < b)


def _assert_board_parity(board: list[list[str]]) -> None:
    ported = evaluate_board_with_joker_constraint(*board)
    ref = reference.evaluate_board_with_joker_constraint(
        list(board[0]), list(board[1]), list(board[2])
    )
    context = f"board={board}"
    assert ported["busted"] == ref["busted"], context
    assert ported["royalties"] == ref["royalties"], context
    assert ported["fl_entry"] == bool(ref["fl_entry"]), context
    assert ported["fl_card_count"] == ref["fl_card_count"], context


@pytest.mark.parametrize("joker_count", (0, 1, 2))
def test_random_board_parity_by_joker_stratum(joker_count):
    rng = random.Random(20260728 + joker_count)
    boards = 120 if joker_count == 0 else (140 if joker_count == 1 else 60)
    for _ in range(boards):
        _assert_board_parity(_random_board(rng, joker_count))


@pytest.mark.parametrize("joker_split", ((0, 0), (1, 0), (1, 1), (2, 0), (0, 2)))
def test_pairwise_line_results_match_reference(joker_split):
    rng = random.Random(773 + sum(joker_split) * 10 + joker_split[0])
    for _ in range(32):
        board_a, board_b = _random_board_pair(rng, joker_split)
        ported_a = evaluate_board_with_joker_constraint(*board_a)
        ported_b = evaluate_board_with_joker_constraint(*board_b)
        ref_a = reference.evaluate_board_with_joker_constraint(
            list(board_a[0]), list(board_a[1]), list(board_a[2])
        )
        ref_b = reference.evaluate_board_with_joker_constraint(
            list(board_b[0]), list(board_b[1]), list(board_b[2])
        )
        for row in ("top", "middle", "bottom"):
            ported_sign = _sign(ported_a["values"][row], ported_b["values"][row])
            ref_sign = _sign(ref_a["values"][row], ref_b["values"][row])
            assert ported_sign == ref_sign, (
                f"row={row} split={joker_split} a={board_a} b={board_b}"
            )


def test_wheel_straight_is_five_high_in_both_engines():
    wheel = ["Ah", "2c", "3d", "4s", "5h"]
    six_high = ["2d", "3h", "4c", "5s", "6d"]
    ported_wheel = evaluate_hand_joker(wheel, 5)
    ported_six = evaluate_hand_joker(six_high, 5)
    assert ported_wheel < ported_six
    ref_wheel = reference.evaluate_hand(wheel, 5)
    ref_six = reference.evaluate_hand(six_high, 5)
    assert ref_wheel < ref_six
    # Joker completing the wheel keeps five-high semantics.
    joker_wheel_board = [
        ["2s", "3c", "4h"],
        ["Ad", "2c", "3d", "4s", "X1"],
        ["6c", "7c", "8c", "9c", "Tc"],
    ]
    _assert_board_parity(joker_wheel_board)


def test_fl_chain_counts_match_reference_schedule():
    assert JOKER_CHAIN_RULES.fl_entry_cards == {
        "qq": 14,
        "kk": 15,
        "aa": 16,
        "trips": 17,
    }
    cases = (
        (["Qs", "Qh", "2d"], 14),
        (["Ks", "Kh", "2d"], 15),
        (["As", "Ah", "2d"], 16),
        (["7s", "7h", "7d"], 17),
        (["Js", "Jh", "2d"], 0),
    )
    strong_tail = [["Ad", "Kd", "Qd", "Jd", "Td"], ["As", "Ks", "Qs", "Js", "Ts"]]
    for top, expected_count in cases:
        board = [top, *strong_tail]
        ported = evaluate_board_with_joker_constraint(*board)
        ref = reference.evaluate_board_with_joker_constraint(
            list(top), list(strong_tail[0]), list(strong_tail[1])
        )
        assert ported["fl_card_count"] == ref["fl_card_count"] == expected_count


def test_double_joker_top_is_constrained_below_middle():
    board = [
        ["X1", "X2", "2d"],
        ["3c", "3d", "4h", "5s", "6c"],  # pair of threes
        ["7c", "7d", "7h", "2s", "9c"],  # trips
    ]
    ported = evaluate_board_with_joker_constraint(*board)
    assert ported["busted"] is False
    _assert_board_parity(board)
