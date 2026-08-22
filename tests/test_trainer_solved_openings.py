"""Which T0 openings the AI answers from the table instead of the model.

The table holds measured openings, but only the `settled` rows are ones whose
first-to-second gap excludes zero.  An `unresolved` row's pick is the argmax of
noisy means and carries the upward bias that always comes with a maximum, so
the model answers those.  These tests pin that split, the suit-isomorphism that
lets one row answer up to twenty-four deals, and the geometry gate -- the table
indexes an empty T0 first seat and nothing else.

The table is built here rather than read from D:/, so the tests say what the
code does and not what one machine happens to hold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trainer import engine_eval  # noqa: E402

EMPTY: dict[str, list[str]] = {"top": [], "middle": [], "bottom": []}

# Two rows in canonical suiting, one of each status. The placements use exactly
# the five cards of their key, which is what the caller's legality check needs.
TABLE = {
    "schema": "hu_t0_solved_openings_v1",
    "openings": {
        "Ac Qc 7d 3h 3s": {
            "canonical_hand": ["Ac", "Qc", "7d", "3h", "3s"],
            "best": {"top": ["Qc"], "middle": ["Ac"], "bottom": ["7d", "3h", "3s"]},
            "runner_up": {"top": ["Qc"], "middle": ["Ac", "7d"], "bottom": ["3h", "3s"]},
            "best_ev": 1.471,
            "gap_to_second": 0.2833,
            "gap_ci95": [0.1483, 0.4183],
            "status": "settled",
            "particles_behind_gap": 46080,
        },
        "Ac Ad Ah 8s 4c": {
            "canonical_hand": ["Ac", "Ad", "Ah", "8s", "4c"],
            "best": {"top": ["Ac", "Ad", "Ah"], "middle": [], "bottom": ["8s", "4c"]},
            "runner_up": {"top": ["Ac", "Ad"], "middle": ["Ah"], "bottom": ["8s", "4c"]},
            "best_ev": 6.0,
            "gap_to_second": 0.433,
            "gap_ci95": [-0.029, 0.895],
            "status": "unresolved",
            "particles_behind_gap": 1024,
        },
    },
}

SETTLED = ["Ac", "Qc", "7d", "3h", "3s"]
UNRESOLVED = ["Ac", "Ad", "Ah", "8s", "4c"]


@pytest.fixture(autouse=True)
def _table(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve the fixture table, and never the one on disk."""
    monkeypatch.setattr(engine_eval, "_SOLVED_OPENINGS", TABLE, raising=False)
    monkeypatch.setattr(engine_eval, "_SOLVED_OPENINGS_LOADED", True, raising=False)


def solved(**over):
    call = {
        "hero_board": dict(EMPTY),
        "opp_board": dict(EMPTY),
        "dealt": list(SETTLED),
        "dead": [],
        "turn": 0,
        "position": "first",
    }
    call.update(over)
    return engine_eval._solved_opening(**call)


def relabel(hand, mapping):
    return [f"{card[0]}{mapping[card[1]]}" for card in hand]


def rows_of(answer):
    placed: dict[str, list[str]] = {"top": [], "middle": [], "bottom": []}
    for card, row in answer["action"]["placements"]:
        placed[row].append(card)
    return {row: sorted(cards) for row, cards in placed.items()}


def test_settled_opening_comes_from_the_table():
    answer = solved()
    assert answer is not None
    assert answer["evaluator"].startswith("table:t0_solved_openings")
    assert answer["solved_opening"]["status"] == "settled"
    assert rows_of(answer) == {
        "top": ["Qc"], "middle": ["Ac"], "bottom": ["3h", "3s", "7d"],
    }


def test_unresolved_opening_falls_through_to_the_model():
    """The gap does not exclude zero, so the table's pick is not established."""
    assert solved(dealt=list(UNRESOLVED)) is None


def test_stored_action_places_every_dealt_card_and_no_other():
    answer = solved()
    placed = sorted(card for card, _ in answer["action"]["placements"])
    assert placed == sorted(SETTLED)
    assert answer["action"]["discard"] is None


@pytest.mark.parametrize("mapping", [
    {"c": "c", "d": "d", "h": "h", "s": "s"},
    {"c": "h", "d": "s", "h": "c", "s": "d"},
    {"c": "s", "d": "h", "h": "d", "s": "c"},
    {"c": "d", "d": "c", "h": "s", "s": "h"},
])
def test_suit_isomorphic_deals_get_the_same_shape_in_their_own_suits(mapping):
    """One row answers the deal under any relabelling of the four suits."""
    hand = relabel(SETTLED, mapping)
    answer = solved(dealt=hand)
    assert answer is not None, f"{hand} missed the table"

    placed = sorted(card for card, _ in answer["action"]["placements"])
    assert placed == sorted(hand), "the answer must use the querent's own cards"

    base = rows_of(solved())
    assert rows_of(answer) == {
        row: sorted(relabel(cards, mapping)) for row, cards in base.items()
    }


def test_unresolved_stays_unresolved_under_relabelling():
    hand = relabel(UNRESOLVED, {"c": "h", "d": "s", "h": "c", "s": "d"})
    assert solved(dealt=hand) is None


@pytest.mark.parametrize("label,over", [
    ("second seat", {"position": "second"}),
    ("a later street", {"turn": 1}),
    ("opponent has placed", {"opp_board": {**EMPTY, "bottom": ["2c", "3c"]}}),
    ("hero has placed", {"hero_board": {**EMPTY, "top": ["2c"]}}),
    ("a discard exists", {"dead": ["2c"]}),
    ("three cards dealt", {"dealt": SETTLED[:3]}),
    ("no cards dealt", {"dealt": None}),
])
def test_table_answers_only_an_empty_t0_first_seat(label, over):
    """Everything else is a position the table does not index."""
    assert solved(**over) is None, label


def test_hand_outside_the_table_falls_through():
    assert solved(dealt=["2c", "3d", "4h", "5s", "7c"]) is None


def test_a_missing_table_is_not_an_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(engine_eval, "_SOLVED_OPENINGS", None, raising=False)
    assert solved() is None
