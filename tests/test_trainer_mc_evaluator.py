"""CRN pins for the trainer's MC fallback (``trainer.evaluator``).

The defect these tests pin (found by the 2026-08-12 3-max M1 adversarial
review): futures' cards were pre-drawn, but the hero and opponent placements
consumed one shared rng inside the candidate loop, so a candidate's EV
depended on which candidates ``generate_actions`` happened to put before it.
The corrected scheme mirrors ``ofc_regular.three_max.mc``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest

from trainer import evaluator as trainer_evaluator
from trainer.evaluator import evaluate_position_mc

# T3: hero has 9 cards placed, 3 dealt, one pineapple street left to sample.
T3_POSITION = dict(
    hero_board={
        "top": ["Qh", "Qd"],
        "middle": ["8s", "8d", "8c", "2c"],
        "bottom": ["9s", "9d", "9c"],
    },
    opp_board={
        "top": ["Jh", "Jc"],
        "middle": ["5s", "5d", "6c", "6d"],
        "bottom": ["Ts", "Td", "Th"],
    },
    dealt=["As", "Kd", "7h"],
    turn=3,
)

# T4 with a complete opponent: nothing is left to sample, EV must be exact.
T4_POSITION = dict(
    hero_board={
        "top": ["Qh", "Qd", "2h"],
        "middle": ["Ks", "Kd", "3c", "4c", "5c"],
        "bottom": ["As", "Ad", "6c"],
    },
    opp_board={
        "top": ["Jh", "Jd", "2s"],
        "middle": ["Ts", "Td", "3d", "4d", "5d"],
        "bottom": ["Ah", "Ac", "6d", "7s", "8s"],
    },
    dealt=["7d", "8h", "9h"],
    turn=4,
)


def _metrics_by_key(result):
    return {c["key"]: c["metrics"] for c in result["candidates"]}


def test_candidate_ev_does_not_depend_on_evaluation_order(monkeypatch):
    """Reversing the action order must not move any candidate's EV."""
    forward = evaluate_position_mc(**T3_POSITION, sims_override=16)

    original = trainer_evaluator.generate_actions
    monkeypatch.setattr(
        trainer_evaluator,
        "generate_actions",
        lambda board, dealt: list(reversed(original(board, dealt))),
    )
    backward = evaluate_position_mc(**T3_POSITION, sims_override=16)

    assert _metrics_by_key(forward) == _metrics_by_key(backward)


def test_ranking_is_deterministic():
    first = evaluate_position_mc(**T3_POSITION, sims_override=16)
    again = evaluate_position_mc(**T3_POSITION, sims_override=16)
    assert _metrics_by_key(first) == _metrics_by_key(again)
    assert [c["key"] for c in first["candidates"]] == [
        c["key"] for c in again["candidates"]
    ]


def test_last_street_with_complete_opponent_is_exact():
    """No hero street and no opponent fill remain: sims must not matter."""
    one = evaluate_position_mc(**T4_POSITION, sims_override=1)
    many = evaluate_position_mc(**T4_POSITION, sims_override=50)
    assert _metrics_by_key(one).keys() == _metrics_by_key(many).keys()
    for key, metrics in _metrics_by_key(one).items():
        other = _metrics_by_key(many)[key]
        assert metrics["ev"] == pytest.approx(other["ev"])
        assert metrics["bust_rate"] == other["bust_rate"]


def test_overloaded_dead_list_is_rejected_not_silently_truncated():
    """T3 needs 3 hero + 4 opponent-fill cards; leave only 6 in the deck."""
    from ofc_regular.cards import ALL_CARDS

    used = set(T3_POSITION["dealt"])
    for board in (T3_POSITION["hero_board"], T3_POSITION["opp_board"]):
        for row in board.values():
            used.update(row)
    dead = [c for c in ALL_CARDS if c not in used][:25]
    with pytest.raises(ValueError, match="賄えません"):
        evaluate_position_mc(**T3_POSITION, dead=dead, sims_override=2)
