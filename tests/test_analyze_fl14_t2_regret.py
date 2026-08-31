import pytest

from ai.tutor.analyze_fl14_t2_regret import (
    action_parts,
    destination,
    matching_rows,
    rank_band,
    summary,
)


def test_action_parts_separates_three_rows_and_discard():
    rows, discard = action_parts("As|2c,2d|3h,4h|Kd")
    assert rows == ["As", "2c,2d", "3h,4h"]
    assert discard == "Kd"


def test_summary_charges_groups_by_total_regret():
    rows = [
        {"kind": "a", "regret": 0.0},
        {"kind": "a", "regret": 2.0},
        {"kind": "b", "regret": 1.0},
    ]
    grouped = {row["group"]: row for row in summary(rows, lambda row: row["kind"])}
    assert grouped["a"]["mean_regret"] == 1.0
    assert grouped["a"]["regret_share"] == pytest.approx(2 / 3)
    assert grouped["b"]["regret_share"] == pytest.approx(1 / 3)


def test_rank_band():
    assert rank_band("2c") == "low_2_6"
    assert rank_band("Th") == "mid_7_T"
    assert rank_band("As") == "high_J_A"
    assert rank_band("X1") == "joker"


def test_pair_match_and_destination():
    board = ["As", "2c,2d", "3h,4h"]
    assert matching_rows("Ah", board) == [0]
    assert matching_rows("2s", board) == [1]
    assert matching_rows("Kd", board) == []
    assert destination("Ah", ["As,Ah", "2c,2d", "3h,4h"]) == "top"
    assert destination("Kd", ["As,Ah", "2c,2d", "3h,4h"]) == "discard"
