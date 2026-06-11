import random

from ofc_regular.state import Board
from ofc_regular.t2_teacher_data import build_t2_sample, evaluate_t2_actions, sample_t2_state
from ofc_regular.turn3_model import train_ridge_model


def _model():
    sample = {
        "sample_id": 0,
        "rule_set": "regular",
        "phase": "turn3_9card",
        "fl_ev": 12.196164,
        "board": {
            "top": ["Qh"],
            "middle": ["Kh", "Kd", "6c", "8s"],
            "bottom": ["9c", "9d", "9s", "Kc"],
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
                    "middle": ["Kh", "Kd", "6c", "8s"],
                    "bottom": ["9c", "9d", "9s", "Kc"],
                },
            },
            {
                "placements": [["Qs", "bottom"], ["Ah", "top"]],
                "discards": ["7d"],
                "score": 1.0,
                "future_count": 4,
                "next_board": {
                    "top": ["Qh", "Ah"],
                    "middle": ["Kh", "Kd", "6c", "8s"],
                    "bottom": ["9c", "9d", "9s", "Kc", "Qs"],
                },
            },
        ],
    }
    return train_ridge_model([sample, sample], l2=1.0)


def test_sample_t2_state_has_board_deal_and_remaining_cards():
    board, dealt, remaining = sample_t2_state(random.Random(5))
    assert board.card_count() == 7
    assert len(dealt) == 3
    assert len(remaining) == 42
    assert len(set(board.all_cards()) | set(dealt) | set(remaining)) == 52


def test_evaluate_t2_actions_uses_turn3_model_scores():
    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c"],
        bottom=["9c", "9d", "9s"],
    )
    ranked = evaluate_t2_actions(
        board,
        ["Qs", "Ah", "7d"],
        model=_model(),
        remaining_cards=["2h", "3h", "4h", "5h", "6h", "7h"],
        future_samples=2,
        rng=random.Random(1),
    )
    assert ranked
    assert ranked[0]["score"] >= ranked[-1]["score"]
    assert ranked[0]["future_count"] == 2


def test_evaluate_t2_actions_batch_size_is_equivalent():
    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c"],
        bottom=["9c", "9d", "9s"],
    )
    kwargs = {
        "model": _model(),
        "remaining_cards": ["2h", "3h", "4h", "5h", "6h", "7h"],
        "future_samples": 2,
    }

    one_by_one = evaluate_t2_actions(
        board,
        ["Qs", "Ah", "7d"],
        action_batch_size=1,
        rng=random.Random(1),
        **kwargs,
    )
    batched = evaluate_t2_actions(
        board,
        ["Qs", "Ah", "7d"],
        action_batch_size=0,
        rng=random.Random(1),
        **kwargs,
    )

    assert [
        (action["placements"], action["discards"], action["score"])
        for action in one_by_one
    ] == [
        (action["placements"], action["discards"], action["score"])
        for action in batched
    ]


def test_build_t2_sample_sorts_actions():
    sample = build_t2_sample(
        rng=random.Random(8),
        sample_id=2,
        model=_model(),
        future_samples=2,
    )
    assert sample["sample_id"] == 2
    assert sample["phase"] == "turn2_7card"
    scores = [action["score"] for action in sample["actions"]]
    assert scores == sorted(scores, reverse=True)
