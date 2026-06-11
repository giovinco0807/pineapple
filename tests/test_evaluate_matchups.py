import numpy as np

from ofc_regular.evaluate_matchups import classify_board, summarize_scores
from ofc_regular.state import Board


def test_classify_board_marks_fl_chase_bust():
    board = Board.from_rows(
        top=["Qh", "Qs", "2c"],
        middle=["Ah", "Kd", "7c", "5s", "3d"],
        bottom=["9h", "8h", "6h", "4h", "2h"],
    )

    assert classify_board(board) == "FL狙いバースト"


def test_classify_board_marks_fl_success():
    board = Board.from_rows(
        top=["Qh", "Qs", "2c"],
        middle=["Ah", "Ad", "7c", "5s", "3d"],
        bottom=["9h", "9d", "9s", "4h", "4d"],
    )

    assert classify_board(board) == "FL成功"


def test_summarize_scores_ci_is_finite():
    summary = summarize_scores([1.0, -1.0, 3.0, 5.0])

    assert summary["avg_score_per_hand_for_a"] == 2.0
    assert np.isfinite(summary["std_error"])
    assert summary["ci95_low"] < summary["ci95_high"]
