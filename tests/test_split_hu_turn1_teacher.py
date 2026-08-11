import pytest

from ofc_regular.split_hu_turn1_teacher import holdout_score, split_rows, state_identity


def _row(index: int) -> dict:
    return {
        "hand_seed": index,
        "seat": "first",
        "board": {"top": ["As"], "middle": [], "bottom": []},
        "opponent_board": {"top": ["Kh"], "middle": [], "bottom": []},
        "dealt": ["2c", "3d", "4h"],
        "visible_dead_cards": [],
    }


def test_split_rows_is_deterministic_and_complete():
    rows = [_row(index) for index in range(200)]

    first = split_rows(rows, holdout_fraction=0.2, seed=17)
    second = split_rows(rows, holdout_fraction=0.2, seed=17)

    assert [row["hand_seed"] for row in first[0]] == [row["hand_seed"] for row in second[0]]
    assert [row["hand_seed"] for row in first[1]] == [row["hand_seed"] for row in second[1]]
    assert len(first[0]) + len(first[1]) == 200
    assert 20 <= len(first[1]) <= 60


def test_identical_state_has_identical_split_score():
    row = _row(1)
    copy = dict(row)

    assert state_identity(row) == state_identity(copy)
    assert holdout_score(row, seed=9) == holdout_score(copy, seed=9)


def test_split_rejects_invalid_fraction():
    with pytest.raises(ValueError, match="between 0 and 1"):
        split_rows([_row(1), _row(2)], holdout_fraction=1.0, seed=1)
