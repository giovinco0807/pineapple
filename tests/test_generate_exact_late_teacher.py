import pytest

from ai.engine.action_space import Action, get_turn_actions
from ai.engine.encoding import ALL_CARDS
from ai.tutor.exact_late import (
    Board,
    apply_action,
    exact_candidate_metrics,
    exact_sample_size,
    evaluate_late_position,
    terminal_metrics,
)
from ai.tutor.generate_exact_late_teacher import evaluate_target


def test_evaluate_t4_target_scores_all_terminal_actions_exactly():
    target = {
        "turn": 4,
        "board": {
            "top": ["Ad", "Kh"],
            "middle": ["2h", "2s", "4c", "4s", "5s"],
            "bottom": ["3c", "3s", "6c", "6h"],
        },
        "opponent_board": {},
        "dealt": ["Jc", "Qd", "7h"],
        "known_discards": [],
        "exclude": [],
        "is_btn": True,
    }

    record = evaluate_target((1, target))
    board = Board(
        top=target["board"]["top"],
        middle=target["board"]["middle"],
        bottom=target["board"]["bottom"],
    )

    assert record is not None
    assert record["turn"] == 4
    assert record["eval_mode"] == "exact"
    assert record["exact_scope"] == "t4_terminal_actions"
    assert record["n_candidates"] == len(get_turn_actions(target["dealt"], board))
    assert record["total_exact_samples"] == record["n_candidates"]
    assert all(candidate["exact"]["samples"] == 1 for candidate in record["candidates"])
    scores = [candidate["target_score"] for candidate in record["candidates"]]
    assert scores == sorted(scores, reverse=True)


def test_t4_btn_terminal_comparison_is_hu_exact():
    board = Board(
        top=["Ad", "Kh"],
        middle=["2h", "2s", "4c", "4s", "5s"],
        bottom=["3c", "3s", "6c", "6h"],
    )
    opponent = Board(
        top=["2d", "3d", "4d"],
        middle=["5h", "5d", "7s", "8s", "Tc"],
        bottom=["9h", "9d", "Th", "Jh", "Qs"],
    )

    result = evaluate_late_position(
        board,
        ["Jc", "Qd", "7h"],
        4,
        opponent_board=opponent,
        prefer_rust=False,
        top_n=50,
    )

    assert result["exact_scope"] == "t4_terminal_vs_complete_opponent"
    assert result["hu_exact"] is True
    assert all(candidate["metrics"]["samples"] == 1 for candidate in result["candidates"])


def test_evaluate_t3_target_enumerates_every_t4_draw_for_each_action():
    board_cards = ["Ad", "2h", "2s", "4c", "4s", "3c", "3s", "6c", "6h"]
    dealt = ["Jc", "Qd", "7h"]
    kept_next_draw = {"8d", "9d", "Td"}
    used = set(board_cards) | set(dealt) | kept_next_draw
    exclude = [card for card in ALL_CARDS if card not in used]
    target = {
        "turn": 3,
        "board": {
            "top": ["Ad"],
            "middle": ["2h", "2s", "4c", "4s"],
            "bottom": ["3c", "3s", "6c", "6h"],
        },
        "opponent_board": {},
        "dealt": dealt,
        "known_discards": [],
        "exclude": exclude,
        "is_btn": False,
    }

    record = evaluate_target((1, target))
    board = Board(
        top=target["board"]["top"],
        middle=target["board"]["middle"],
        bottom=target["board"]["bottom"],
    )

    assert record is not None
    assert record["turn"] == 3
    assert record["exact_scope"] == "t3_self_board_all_t4_draws_best_t4"
    assert record["n_candidates"] == len(get_turn_actions(dealt, board))
    assert all(candidate["exact"]["source"] == "exact" for candidate in record["candidates"])
    assert all(candidate["exact"]["remaining_deck_size"] == 3 for candidate in record["candidates"])
    assert all(candidate["exact"]["samples"] == 1 for candidate in record["candidates"])
    assert record["total_exact_samples"] == record["n_candidates"]
    scores = [candidate["target_score"] for candidate in record["candidates"]]
    assert scores == sorted(scores, reverse=True)


def test_t4_bb_candidate_enumerates_btn_best_joker_response():
    board = Board(
        top=["2c", "3c"],
        middle=["6h", "6d", "7s", "8s", "Tc"],
        bottom=["Jh", "Jd", "Qc", "Qd"],
    )
    opponent = Board(
        top=["Qh", "Qs"],
        middle=["Kh", "Ks", "9d", "8c", "7h"],
        bottom=["Ah", "Ad", "Ac", "5s"],
    )
    dealt = ["4c", "Kc", "5d"]
    reply_draw = ["X1", "4d", "2s", "9s"]
    action = Action(
        placements=[("4c", "top"), ("Kc", "bottom")],
        discard="5d",
    )
    live = set(board.all_cards()) | set(opponent.all_cards()) | set(dealt) | set(reply_draw)
    exclude = [card for card in ALL_CARDS if card not in live]

    final_board = apply_action(board, action)
    hero_terminal = terminal_metrics(final_board)
    metrics = exact_candidate_metrics(
        board,
        dealt,
        action,
        4,
        opponent_board=opponent,
        exclude=exclude,
    )

    assert metrics["source"] == "exact_hu_response"
    assert metrics["opponent_response"] is True
    assert metrics["remaining_deck_size"] == 4
    assert metrics["enumerated_draws"] == metrics["samples"] == 4
    # Across all four draws BTN's best scores are 23, 23, 23, and 13.
    # This independently fixes draw enumeration, averaging, best response,
    # Joker placement, and removal of Hero's discarded 5d from the live deck.
    assert metrics["score"] == pytest.approx(-20.5)
    assert metrics["raw_score"] == pytest.approx(-20.5)
    assert metrics["royalty"] == hero_terminal["royalty"]
    assert metrics["bust_rate"] == float(hero_terminal["bust"])
    assert exact_sample_size(4, 4, opponent_response=True) == 4

    record = evaluate_target(
        (
            1,
            {
                "turn": 4,
                "board": {"top": board.top, "middle": board.middle, "bottom": board.bottom},
                "opponent_board": {
                    "top": opponent.top,
                    "middle": opponent.middle,
                    "bottom": opponent.bottom,
                },
                "dealt": dealt,
                "exclude": exclude,
                "is_btn": False,
            },
        )
    )
    assert record is not None
    assert record["exact_scope"] == "t4_all_opponent_draws_best_response_given_exclude"
    assert record["hu_exact"] is False
    assert record["total_exact_samples"] == record["n_candidates"] * 4
    assert all(candidate["exact"]["source"] == "exact_hu_response" for candidate in record["candidates"])


def test_t4_bb_response_preserves_two_unique_jokers_and_canonical_downgrade():
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
    action = Action(
        placements=[("X1", "top"), ("4d", "bottom")],
        discard="9s",
    )
    live = set(board.all_cards()) | set(opponent.all_cards()) | set(dealt) | set(reply_draw)
    exclude = [card for card in ALL_CARDS if card not in live]

    metrics = exact_candidate_metrics(
        board,
        dealt,
        action,
        4,
        opponent_board=opponent,
        exclude=exclude,
    )

    assert metrics["score"] == pytest.approx(2.0)
    assert metrics["raw_score"] == pytest.approx(2.0)
    assert metrics["royalty"] == pytest.approx(7.0)
    assert metrics["bust_rate"] == 0.0
    assert metrics["fl_rate"] == 1.0
    assert metrics["fl_type_rates"] == {"qq": 1.0, "kk": 0.0, "aa": 0.0, "trips": 0.0}
    assert metrics["samples"] == 1


def test_non_late_turn_target_is_skipped():
    assert evaluate_target((1, {"turn": 2})) is None


def test_t4_rejects_ambiguous_partial_opponent_board():
    board = Board(
        top=["Ad", "Kh"],
        middle=["2h", "2s", "4c", "4s", "5s"],
        bottom=["3c", "3s", "6c", "6h"],
    )
    dealt = ["Jc", "Qd", "7h"]
    action = get_turn_actions(dealt, board)[0]

    with pytest.raises(ValueError, match="T4 opponent board must be empty"):
        exact_candidate_metrics(
            board,
            dealt,
            action,
            4,
            opponent_board=Board(top=["X1"]),
        )


def test_t3_requires_nine_board_cards():
    with pytest.raises(ValueError, match="requires 9 board cards"):
        evaluate_target((1, {"turn": 3, "board": {"top": ["Ad"]}, "dealt": ["2c", "3c", "4c"]}))
