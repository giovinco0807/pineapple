"""Top (3-card) vs middle (5-card) ordering: the foul verdict under scrutiny.

The frozen rule (rules_contract_audit Q6) is that 3-card hands are compared on
the same category scale as 5-card hands: a 3-card trips is a trips (beats two
pair, loses to a straight, rank-compared against a 5-card trips), pairs
compare by pair rank then kickers in order, high cards by ranks in order, and
missing cards compare as zero (so equal prefixes favour the longer hand).

Two layers of checking:
- targeted boards whose verdict is derived by hand from the rule above, and
- a bulk invariant: on any non-busted board the constrained values must
  satisfy top <= middle <= bottom, and on any natural board the busted flag
  must equal the raw ordering check exactly.
"""
import random

import pytest

from ai.engine.encoding import ALL_CARDS
from ai.engine.game_engine import evaluate_board_with_joker_constraint, evaluate_hand

ROYAL_SPADES = ["As", "Ks", "Qs", "Js", "Ts"]


def verdict(top, mid, bottom=None):
    board = evaluate_board_with_joker_constraint(top, mid, bottom or ROYAL_SPADES)
    return bool(board["busted"])


# (label, top, mid, expected_busted) -- bottom is a spade royal unless given.
TARGETED = [
    ("trips top beats two pair mid -> foul",
     ["2c", "2d", "2h"], ["Kc", "Kd", "Qc", "Qd", "4h"], True),
    ("deuce trips top vs treys trips mid -> ok",
     ["2c", "2d", "2h"], ["3c", "3d", "3h", "4d", "5c"], False),
    ("jack trips top vs ten trips mid -> foul",
     ["Jc", "Jd", "Jh"], ["Tc", "Td", "Th", "4d", "5c"], True),
    ("queen trips top vs six-high straight mid -> ok",
     ["Qc", "Qd", "Qh"], ["2h", "3d", "4c", "5h", "6d"], False),
    ("QQ+A top vs QQ+K kickers mid -> foul (kicker ace wins)",
     ["Qc", "Qd", "Ah"], ["Qh", "Qs", "Kc", "3d", "2c"],
     True, ["5h", "6h", "7h", "8h", "9h"]),
    ("QQ+A top vs QQ+A54 mid -> ok (mid has more kickers)",
     ["Qc", "Qd", "Ah"], ["Qh", "Qs", "Ac", "5d", "4c"],
     False, ["5h", "6h", "7h", "8h", "9h"]),
    ("AKQ high top vs AQJ92 mid -> foul (second card K > Q)",
     ["Ah", "Kh", "Qh"], ["Ac", "Qc", "Jd", "9d", "2c"], True),
    ("AKQ high top vs AKJ92 mid -> foul (third card Q > J)",
     ["Ah", "Kh", "Qh"], ["Ac", "Kd", "Jd", "9d", "2c"], True),
    ("AKQ high top vs AKQ92 mid -> ok (missing cards count as zero)",
     ["Ah", "Kh", "Qh"], ["Ac", "Kd", "Qc", "9d", "2c"], False),
    ("66 pair top vs ace-high mid -> foul (pair beats high card)",
     ["6c", "6d", "2h"], ["Ac", "Kd", "Jd", "9d", "3c"], True),
    ("KK top vs AA mid -> ok",
     ["Kc", "Kd", "2h"], ["Ac", "Ad", "Jd", "9d", "3c"], False),
    ("AA top vs KK mid -> foul",
     ["Ac", "Ad", "2h"], ["Kc", "Kd", "Jd", "9d", "3c"], True),
]


@pytest.mark.parametrize(
    "case", TARGETED, ids=[case[0] for case in TARGETED]
)
def test_top_mid_verdicts(case):
    label, top, mid, expected = case[0], case[1], case[2], case[3]
    bottom = case[4] if len(case) > 4 else ROYAL_SPADES
    used = set(top) | set(mid)
    assert len(used) == 8, f"card overlap in fixture: {label}"
    assert not (used & set(bottom)), f"fixture uses bottom cards: {label}"
    assert verdict(top, mid, bottom) is expected


def test_joker_top_drops_below_mid_pair():
    # Top [Qh X1 2c]: unconstrained max is a queen pair, above the jack-pair
    # middle; the joker must fall to keep top <= mid, not foul the board.
    top, mid = ["Qh", "X1", "2c"], ["Jc", "Jd", "8h", "5d", "3c"]
    board = evaluate_board_with_joker_constraint(top, mid, ROYAL_SPADES)
    assert not board["busted"]
    assert evaluate_hand(board["top"], 3) <= evaluate_hand(board["middle"], 5)


def test_double_joker_top_survives_weak_pair_mid():
    # [X1 X2 Ah] can be aces trips, far above a deuce-pair middle; the only
    # legal use is dropping both jokers to a high card below the pair.
    top, mid = ["X1", "X2", "Ah"], ["2c", "2d", "8h", "5d", "3c"]
    board = evaluate_board_with_joker_constraint(top, mid, ROYAL_SPADES)
    assert not board["busted"]
    assert evaluate_hand(board["top"], 3) <= evaluate_hand(board["middle"], 5)


def test_bulk_invariants_on_random_boards():
    rng = random.Random(20260731)
    natural_checked = joker_checked = 0
    for _ in range(5000):
        deck = list(ALL_CARDS)
        rng.shuffle(deck)
        top, mid, bottom = deck[:3], deck[3:8], deck[8:13]
        board = evaluate_board_with_joker_constraint(top, mid, bottom)
        has_joker = any(card in ("X1", "X2") for card in top + mid + bottom)
        if not board["busted"]:
            top_value = evaluate_hand(board["top"], 3)
            mid_value = evaluate_hand(board["middle"], 5)
            bottom_value = evaluate_hand(board["bottom"], 5)
            assert top_value <= mid_value <= bottom_value, (top, mid, bottom)
        if not has_joker:
            raw_busted = (
                evaluate_hand(top, 3) > evaluate_hand(mid, 5)
                or evaluate_hand(mid, 5) > evaluate_hand(bottom, 5)
            )
            assert board["busted"] == raw_busted, (top, mid, bottom)
            natural_checked += 1
        else:
            joker_checked += 1
    assert natural_checked > 1000 and joker_checked > 1000
