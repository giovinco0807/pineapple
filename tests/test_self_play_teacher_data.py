import random

import numpy as np

from ofc_regular.self_play_teacher_data import build_self_play_sample, remaining_for_teacher
from ofc_regular.state import Board


class ConstantModel:
    def predict_matrix(self, features):
        return np.zeros(features.shape[0], dtype=np.float64)


def test_remaining_for_teacher_removes_visible_opponent_cards():
    board = Board.from_rows(top=["Qh"])
    opponent = Board.from_rows(middle=["Ah"], bottom=["2c"])

    remaining = remaining_for_teacher(board, ["Kd", "Ks", "3h"], opponent)

    assert "Qh" not in remaining
    assert "Kd" not in remaining
    assert "Ah" not in remaining
    assert len(remaining) == 46


def test_remaining_for_teacher_removes_dead_cards():
    board = Board.from_rows(top=["Qh"])
    opponent = Board.from_rows(middle=["Ah"], bottom=["2c"])

    remaining = remaining_for_teacher(board, ["Kd", "Ks", "3h"], opponent, ["7d", "8d"])

    assert "7d" not in remaining
    assert "8d" not in remaining
    assert len(remaining) == 44


def test_build_self_play_opening_sample_keeps_existing_schema():
    sample = build_self_play_sample(
        sample_id=0,
        phase="opening",
        board=Board.from_rows(),
        dealt_cards=["Qh", "Qs", "Ah", "2c", "3d"],
        opponent_board=Board.from_rows(),
        downstream_model=ConstantModel(),
        future_samples=1,
        action_batch_size=0,
        rng=__import__("random").Random(1),
    )

    assert sample is not None
    assert sample["phase"] == "opening_0card"
    assert sample["best_action"] == 0
    assert sample["actions"]
    assert "opponent_board" not in sample


def test_build_self_play_turn3_sample_uses_exact_final_teacher_schema():
    sample = build_self_play_sample(
        sample_id=0,
        phase="turn3",
        board=Board.from_rows(
            top=["Qh"],
            middle=["Kh", "Kd", "6c", "8s"],
            bottom=["9c", "9d", "9s", "Kc"],
        ),
        dealt_cards=["Qs", "Ah", "7d"],
        opponent_board=Board.from_rows(top=["2h"], middle=["3h"], bottom=["4h"]),
        downstream_model=None,
        future_samples=2,
        action_batch_size=0,
        rng=random.Random(1),
    )

    assert sample is not None
    assert sample["phase"] == "turn3_9card"
    assert sample["best_action"] == 0
    assert sample["actions"]
    assert all(action["future_count"] == 2 for action in sample["actions"])
    assert all("non_bust_future_count" in action for action in sample["actions"])
    assert "opponent_board" not in sample
