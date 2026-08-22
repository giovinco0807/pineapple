"""Pins for the trainer's model-ranking grader (``method="model"``).

The trainer graded with the joint-exact teacher search until 2026-08-13. What
replaced it is not a cheaper teacher, it is a different instrument: the street's
learned model scoring every legal action in one forward pass. The property that
motivated the switch is the one pinned here -- the teacher's ranking is
seed-dominated (``docs/trainer_ranking_quality_20260808.md``: five seeds, five
different best openings at T0, mean per-action spread 25.87 points), while the
model's is a function of the position alone.

These tests need the engine DLL and its pinned weights; they skip without them.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import pytest

from trainer import engine_eval
from trainer.evaluator import evaluate_position

pytestmark = pytest.mark.skipif(
    not engine_eval.available(), reason="m3 engine DLL or pinned weights unavailable"
)

# T2 first seat: hero has 7 placed and one discard behind them, opponent 7.
T2_POSITION = dict(
    hero_board={"top": ["3d"], "middle": ["Kc", "7s"], "bottom": ["Ah", "Ad", "2c", "9s"]},
    opp_board={"top": ["4s"], "middle": ["9h", "8c"], "bottom": ["Qs", "Qd", "5h", "6d"]},
    dealt=["Ac", "5s", "5d"],
    dead=["2d"],
    turn=2,
    position="first",
)


def _model(**overrides):
    return evaluate_position(**{**T2_POSITION, "precision": "fast", "method": "model", **overrides})


def test_model_ranking_is_deterministic():
    """Same position in, same ranking out -- no seed anywhere in the path."""
    runs = [_model() for _ in range(3)]
    keys = [[c["key"] for c in r["candidates"]] for r in runs]
    scores = [[c["metrics"]["rank_score"] for c in r["candidates"]] for r in runs]
    assert keys[0] == keys[1] == keys[2]
    assert scores[0] == scores[1] == scores[2]


def test_model_ranking_is_sorted_by_the_quantity_it_reports():
    """Rank and EV gap must come from one number.

    The teacher path needs a per-street table for this (``RANK_FIELD_BY_TURN``)
    because it emits two different measurements and sorts by whichever one that
    street sorted by. The model emits one, and the order is that one.
    """
    result = _model()
    scores = [c["metrics"]["rank_score"] for c in result["candidates"]]
    assert scores == sorted(scores, reverse=True)
    assert all(c["metrics"]["ranked_by"] == "model" for c in result["candidates"])
    assert all(c["metrics"]["ev"] == c["metrics"]["rank_score"] for c in result["candidates"])


def test_model_covers_every_legal_action_with_a_complete_board():
    """Decoding the engine's action keys must reproduce whole boards.

    Each candidate is the hero's existing board plus the action's placements --
    a decode that dropped or duplicated a card would still sort, so the shape is
    checked rather than assumed.
    """
    result = _model()
    teacher = evaluate_position(**{**T2_POSITION, "precision": "fast", "method": "teacher"})
    assert len(result["candidates"]) == len(teacher["candidates"])
    assert {c["key"] for c in result["candidates"]} == {c["key"] for c in teacher["candidates"]}

    placed = sum(len(v) for v in T2_POSITION["hero_board"].values())
    for candidate in result["candidates"]:
        cards = [c for row in ("top", "middle", "bottom") for c in candidate["board"][row]]
        assert len(cards) == placed + 2, candidate
        assert len(set(cards)) == len(cards), candidate
        assert candidate["action"]["discard"] in T2_POSITION["dealt"]


def test_t0_second_seat_is_graded_by_the_model():
    """T0 is the one street whose legal set comes from the opening generator.

    The engine's `model_scores` shared arm handed the second seat TURN actions
    -- two of three onto an empty board -- which the feature encoder rejects, so
    every T0-second grade silently became a Monte-Carlo one on a different
    scale. Found by review on 2026-08-13 and fixed in search.rs; pinned here
    because the failure was invisible from the outside.
    """
    result = evaluate_position(
        hero_board={"top": [], "middle": [], "bottom": []},
        opp_board={"top": ["4s"], "middle": ["9h", "8c"], "bottom": ["Qs", "Qd"]},
        dealt=["8s", "4c", "2s", "Ts", "5c"],
        dead=[],
        turn=0,
        position="second",
        precision="fast",
        method="model",
    )
    assert result["evaluator"] == "rust:model(T0)"
    # 232 is the whole legal opening fan: 3^5 assignments less the 11 that
    # overfill the three-slot top row.
    assert len(result["candidates"]) == 232
    assert all(len(c["action"]["placements"]) == 5 for c in result["candidates"])
    assert all(c["action"]["discard"] is None for c in result["candidates"])


def test_t4_falls_through_to_exact_enumeration():
    """T4 has no learned ranking; the engine says so and the street keeps its own."""
    result = evaluate_position(
        hero_board={
            "top": ["3d", "Kc"],
            "middle": ["7s", "Ah", "Ad", "2c"],
            "bottom": ["9s", "Qs", "Qd", "5h", "6d"],
        },
        opp_board={
            "top": ["4s", "9h"],
            "middle": ["8c", "Jd", "7h", "Th"],
            "bottom": ["2h", "3h", "5c", "6c", "8d"],
        },
        dealt=["Ac", "5s", "5d"],
        dead=["2d", "4d", "6h"],
        turn=4,
        position="first",
        precision="fast",
        method="model",
    )
    assert result["evaluator"] == "rust:t4_exact"
