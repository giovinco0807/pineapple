from pathlib import Path

import pytest

from ai.engine.encoding import ALL_CARDS, Board
from ai.tutor.exact_late import action_key, default_rust_t3_exact_solver_path, evaluate_late_position


def test_t3_evaluate_late_position_uses_rust_exact_when_available():
    solver = default_rust_t3_exact_solver_path()
    if not Path(solver).exists():
        pytest.skip(f"Rust T3 exact solver is not built: {solver}")

    board = Board(
        top=["Qd"],
        middle=["2c", "Kc", "3s", "Ah"],
        bottom=["8h", "Jc", "8s", "Js"],
    )
    opponent = Board(
        top=["Qc", "X2", "Ad"],
        middle=["2d", "7d", "6d"],
        bottom=["8c", "9h", "8d"],
    )

    result = evaluate_late_position(
        board,
        ["Td", "3h", "6c"],
        3,
        opponent_board=opponent,
        top_n=27,
        prefer_rust=True,
        rust_timeout_s=2.0,
    )

    assert result["source"] == "rust_exact"
    assert result["exact_scope"] == "t3_self_board_all_t4_draws_best_t4"
    assert result["hu_exact"] is False
    assert result["legal_actions"] == 21
    assert result["candidate_count"] == 21
    assert len(result["candidates"]) == 21
    assert result["chosen_action"] == result["best"]["action"]
    assert result["elapsed_ms"] < 100.0
    assert result["best"]["metrics"]["samples"] > 0
    assert "bust_rate" in result["best"]["metrics"]
    assert "fl_rate" in result["best"]["metrics"]


def test_t3_evaluate_late_position_can_disable_rust_fallback(tmp_path):
    board = Board(
        top=["Qd"],
        middle=["2c", "Kc", "3s", "Ah"],
        bottom=["8h", "Jc", "8s", "Js"],
    )

    with pytest.raises(RuntimeError, match="rust_exact_failed:FileNotFoundError"):
        evaluate_late_position(
            board,
            ["Td", "3h", "6c"],
            3,
            top_n=1,
            prefer_rust=True,
            rust_solver_path=tmp_path / "missing_solver.exe",
            rust_timeout_s=0.25,
            fallback_on_rust_error=False,
        )


def test_t4_bb_rust_exact_matches_python_opponent_response():
    solver = default_rust_t3_exact_solver_path()
    if not Path(solver).exists():
        pytest.skip(f"Rust late exact solver is not built: {solver}")

    board = Board(
        top=["Qh", "Qs"],
        middle=["Kh", "Ks", "9d", "8c", "7h"],
        bottom=["Ah", "Ad", "Ac", "5s"],
    )
    opponent = Board(
        top=["2c", "3c"],
        middle=["6h", "6d", "7s", "8s", "Tc"],
        bottom=["Jh", "Jd", "Qc", "Qd"],
    )
    dealt = ["X1", "4d", "9s"]
    reply_draw = ["X2", "4c", "Kc"]
    live = set(board.all_cards()) | set(opponent.all_cards()) | set(dealt) | set(reply_draw)
    exclude = [card for card in ALL_CARDS if card not in live]

    python_result = evaluate_late_position(
        board,
        dealt,
        4,
        opponent_board=opponent,
        exclude=exclude,
        top_n=50,
        prefer_rust=False,
    )
    rust_result = evaluate_late_position(
        board,
        dealt,
        4,
        opponent_board=opponent,
        exclude=exclude,
        top_n=50,
        prefer_rust=True,
        rust_timeout_s=2.0,
        fallback_on_rust_error=False,
    )

    assert rust_result["source"] == "rust_exact"
    assert rust_result["exact_scope"] == "t4_all_opponent_draws_best_response_given_exclude"
    assert rust_result["hu_exact"] is False
    assert rust_result["legal_actions"] == python_result["legal_actions"]
    assert rust_result["candidate_count"] == python_result["candidate_count"]
    assert rust_result["chosen_action"] == python_result["chosen_action"]
    assert rust_result["best"]["metrics"]["source"] == "exact_hu_response"
    assert rust_result["best"]["metrics"]["samples"] == 1
    assert rust_result["best"]["metrics"]["enumerated_draws"] == 1
    assert rust_result["best"]["metrics"]["remaining_deck_size"] == 3
    assert rust_result["best"]["metrics"]["opponent_response"] is True
    assert rust_result["best"]["metrics"]["score"] == pytest.approx(
        python_result["best"]["metrics"]["score"]
    )
    assert rust_result["best"]["metrics"]["raw_score"] == pytest.approx(
        python_result["best"]["metrics"]["raw_score"]
    )
    python_by_action = {
        action_key(candidate["action"]): candidate["metrics"]
        for candidate in python_result["candidates"]
    }
    for candidate in rust_result["candidates"]:
        expected = python_by_action[action_key(candidate["action"])]
        assert candidate["metrics"]["score"] == pytest.approx(expected["score"])
        assert candidate["metrics"]["raw_score"] == pytest.approx(expected["raw_score"])
        assert candidate["metrics"]["royalty"] == pytest.approx(expected["royalty"])
        assert candidate["metrics"]["bust_rate"] == pytest.approx(expected["bust_rate"])
        assert candidate["metrics"]["fl_rate"] == pytest.approx(expected["fl_rate"])
