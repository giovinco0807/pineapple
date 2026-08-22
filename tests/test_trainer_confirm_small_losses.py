"""A penalty the model cannot resolve is re-checked before it is shown.

Measured at T3 (`docs/t3_tie_misses_20260815.md`, 166 roots): the model charges
a penalty on pairs the search puts together at 5.2 % of first-seat and 21.3 %
of second-seat roots, every such claim under 0.30.  The model is a single
forward pass and cannot express a tie -- distinct boards give distinct reals --
so the fix belongs where the grade is produced, not in the model.

The engine is not needed: `grade_decision` takes its evaluator as an argument,
so a stub can stand in for both instruments and the routing is what is pinned.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trainer import handlog


HERO = {"top": ["Kc", "Ah"], "middle": ["5c", "9c", "6c"], "bottom": ["7d", "8d", "Jd", "Kd"]}
OPP = {"top": ["Ac", "Td"], "middle": ["Ts", "Js", "6s", "4s"],
       "bottom": ["Kh", "Qh", "Jh", "2h", "3h"]}
# The played move and the one the model prefers: whichever of Th/7s fills the
# last top slot, with 4c to the middle.  The cards make them equal.
PLAYED = [["Th", "top"], ["4c", "middle"]]
OTHER = [["7s", "top"], ["4c", "middle"]]


def _ctx(turn=3):
    return {
        "seat": "hero", "kind": "street", "turn": turn, "position": "second",
        "hero_board": HERO, "opp_board": OPP,
        "dealt": ["Th", "4c", "7s"], "dead": ["Ks", "3s"],
        "action": {"placements": PLAYED, "discard": "7s"},
    }


def _candidate(placements, score):
    board = {r: list(HERO[r]) for r in ("top", "middle", "bottom")}
    for card, row in placements:
        board[row].append(card)
    return {
        "action": {"placements": placements, "discard": None},
        "board": board,
        "metrics": {"ev": score, "rank_score": score, "ranked_by": "ev"},
    }


def _stub(model_gap, search_gap, seen=None):
    """An evaluator that answers differently depending on who is asking."""
    def evaluate(**kwargs):
        if seen is not None:
            seen.append((kwargs["method"], kwargs["precision"]))
        gap = search_gap if kwargs["method"] == "teacher" else model_gap
        # `OTHER` on top, `PLAYED` a gap below it.
        return {"candidates": [_candidate(OTHER, 0.0),
                               _candidate(PLAYED, -gap)]}
    return evaluate


def test_a_small_model_penalty_is_overturned_by_the_search():
    seen = []
    graded = handlog.grade_decision(
        _ctx(), precision="fast", evaluate=_stub(0.186, 0.0, seen), method="model")

    assert [m for m, _ in seen] == ["model", "teacher"]
    assert seen[1][1] == handlog.CONFIRM_PRECISION
    assert graded["model_ev_loss"] == 0.186
    assert graded["ev_loss"] == 0.0
    assert graded["overturned"] is True
    assert graded["confirmed_by"] == "teacher"
    assert graded["tied_with_best"] is True


def test_a_small_penalty_the_search_agrees_with_is_kept():
    graded = handlog.grade_decision(
        _ctx(), precision="fast", evaluate=_stub(0.10, 0.25), method="model")

    assert graded["model_ev_loss"] == 0.10
    assert graded["ev_loss"] == 0.25
    assert graded["overturned"] is False


def test_a_large_penalty_is_not_re_checked():
    """Above the threshold the model is taken at its word -- the confirmation
    exists for differences it cannot resolve, not for every grade."""
    seen = []
    graded = handlog.grade_decision(
        _ctx(), precision="fast", evaluate=_stub(2.5, 0.0, seen), method="model")

    assert [m for m, _ in seen] == ["model"]
    assert graded["ev_loss"] == 2.5
    assert "model_ev_loss" not in graded


def test_a_zero_loss_is_not_re_checked():
    """Zero is not a small penalty: there is nothing left to overturn.

    (The stub always lists the played move second, so this is the tied case
    rather than the outright-best one -- which is the case that would otherwise
    be re-checked for no reason.)"""
    seen = []
    graded = handlog.grade_decision(
        _ctx(), precision="fast", evaluate=_stub(0.0, 0.0, seen), method="model")

    assert [m for m, _ in seen] == ["model"]
    assert graded["ev_loss"] == 0.0
    assert graded["tied_with_best"] is True
    assert "model_ev_loss" not in graded


def test_only_the_measured_street_is_confirmed():
    """T0 is minutes a decision and T1/T2 have no measured rate, so neither is
    re-checked until one exists."""
    for turn in (0, 1, 2, 4):
        seen = []
        handlog.grade_decision(
            _ctx(turn), precision="fast", evaluate=_stub(0.1, 0.0, seen),
            method="model")
        assert [m for m, _ in seen] == ["model"], f"T{turn} was re-checked"


def test_the_teacher_grader_does_not_confirm_itself():
    seen = []
    handlog.grade_decision(
        _ctx(), precision="deep", evaluate=_stub(0.1, 0.0, seen), method="teacher")
    assert [m for m, _ in seen] == ["teacher"]


def test_the_threshold_covers_every_spurious_penalty_measured():
    """The pilot's worst spurious claim was 0.272 at the second seat."""
    assert handlog.CONFIRM_BELOW > 0.272
