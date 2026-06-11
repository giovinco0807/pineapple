import random

from ofc_regular.early_teacher_data import (
    build_early_sample,
    evaluate_bootstrap_actions,
    sample_early_state,
)
from ofc_regular.state import Board
from ofc_regular.turn3_model import train_ridge_model


def _model():
    sample = {
        "sample_id": 0,
        "rule_set": "regular",
        "phase": "turn2_7card",
        "board": {
            "top": ["Qh"],
            "middle": ["Kh", "Kd", "6c"],
            "bottom": ["9c", "9d", "9s"],
        },
        "dealt": ["Qs", "Ah", "7d"],
        "best_action": 0,
        "score_gap": 2.0,
        "actions": [
            {
                "placements": [["Qs", "top"], ["Ah", "top"]],
                "discards": ["7d"],
                "score": 3.0,
                "future_count": 4,
                "next_board": {
                    "top": ["Qh", "Qs", "Ah"],
                    "middle": ["Kh", "Kd", "6c"],
                    "bottom": ["9c", "9d", "9s"],
                },
            },
            {
                "placements": [["Qs", "bottom"], ["Ah", "top"]],
                "discards": ["7d"],
                "score": 1.0,
                "future_count": 4,
                "next_board": {
                    "top": ["Qh", "Ah"],
                    "middle": ["Kh", "Kd", "6c"],
                    "bottom": ["9c", "9d", "9s", "Qs"],
                },
            },
        ],
    }
    return train_ridge_model([sample, sample], l2=1.0)


def test_sample_early_state_shapes():
    opening_board, opening_dealt, opening_remaining = sample_early_state(
        random.Random(1),
        phase="opening",
    )
    assert opening_board.card_count() == 0
    assert len(opening_dealt) == 5
    assert len(opening_remaining) == 47

    turn1_board, turn1_dealt, turn1_remaining = sample_early_state(
        random.Random(2),
        phase="turn1",
    )
    assert turn1_board.card_count() == 5
    assert len(turn1_dealt) == 3
    assert len(turn1_remaining) == 44


def test_evaluate_bootstrap_actions_for_turn1():
    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd"],
        bottom=["9c", "9d"],
    )
    ranked = evaluate_bootstrap_actions(
        board,
        ["Qs", "Ah", "7d"],
        downstream_model=_model(),
        remaining_cards=["2h", "3h", "4h", "5h", "6h", "7h"],
        future_samples=2,
        rng=random.Random(3),
    )
    assert ranked
    assert ranked[0]["score"] >= ranked[-1]["score"]
    assert ranked[0]["future_count"] == 2


def test_evaluate_bootstrap_actions_batch_size_is_equivalent():
    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd"],
        bottom=["9c", "9d"],
    )
    kwargs = {
        "downstream_model": _model(),
        "remaining_cards": ["2h", "3h", "4h", "5h", "6h", "7h"],
        "future_samples": 2,
    }

    one_by_one = evaluate_bootstrap_actions(
        board,
        ["Qs", "Ah", "7d"],
        action_batch_size=1,
        rng=random.Random(3),
        **kwargs,
    )
    batched = evaluate_bootstrap_actions(
        board,
        ["Qs", "Ah", "7d"],
        action_batch_size=0,
        rng=random.Random(3),
        **kwargs,
    )

    assert [
        (action["placements"], action["discards"], action["score"])
        for action in one_by_one
    ] == [
        (action["placements"], action["discards"], action["score"])
        for action in batched
    ]


def test_build_opening_sample_sorts_actions():
    sample = build_early_sample(
        rng=random.Random(4),
        sample_id=1,
        phase="opening",
        downstream_model=_model(),
        future_samples=2,
    )
    assert sample["sample_id"] == 1
    assert sample["phase"] == "opening_0card"
    scores = [action["score"] for action in sample["actions"]]
    assert scores == sorted(scores, reverse=True)
