from ofc_regular.estimate_hu_fl_ev_direct import score_fl_vs_normal
from ofc_regular.state import Board


def test_current_fl_top_qq_entry_does_not_award_next_fl_bonus():
    hero = Board.from_rows(
        top=("Qc", "Qd", "2c"),
        middle=("3h", "4d", "5c", "6s", "7h"),
        bottom=("8h", "9h", "Th", "Jh", "Kh"),
    )
    opponent = Board.from_rows(
        top=("2h", "3d", "4c"),
        middle=("5d", "5s", "6c", "7d", "8c"),
        bottom=("6h", "6d", "7c", "8d", "9c"),
    )

    score_zero, *_ = score_fl_vs_normal(hero, opponent, next_fl_ev=0.0)
    score_large, *_ = score_fl_vs_normal(hero, opponent, next_fl_ev=100.0)

    assert score_large == score_zero


def test_current_fl_stay_awards_next_fl_bonus():
    hero = Board.from_rows(
        top=("Qc", "Qd", "Qs"),
        middle=("3h", "4d", "5c", "6s", "7h"),
        bottom=("8h", "9h", "Th", "Jh", "Kh"),
    )
    opponent = Board.from_rows(
        top=("2h", "3d", "4c"),
        middle=("5d", "5s", "6c", "7d", "8c"),
        bottom=("6h", "6d", "7c", "8d", "9c"),
    )

    score_zero, *_ = score_fl_vs_normal(hero, opponent, next_fl_ev=0.0)
    score_large, *_ = score_fl_vs_normal(hero, opponent, next_fl_ev=100.0)

    assert score_large == score_zero + 100.0


def test_normal_opponent_fl_entry_subtracts_next_fl_bonus():
    hero = Board.from_rows(
        top=("Ac", "Kd", "2c"),
        middle=("3h", "4d", "5c", "6s", "7h"),
        bottom=("8h", "9h", "Th", "Jh", "Kh"),
    )
    opponent = Board.from_rows(
        top=("Qc", "Qd", "4c"),
        middle=("3d", "4s", "5d", "6c", "7c"),
        bottom=("8s", "9s", "Ts", "Js", "As"),
    )

    score_zero, *_ = score_fl_vs_normal(hero, opponent, next_fl_ev=0.0)
    score_large, *_ = score_fl_vs_normal(hero, opponent, next_fl_ev=100.0)

    assert score_large == score_zero - 100.0
