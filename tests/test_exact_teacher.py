from ofc_regular.action_space import generate_actions, generate_initial_actions, generate_turn_actions
from ofc_regular.state import Board
from ofc_regular.teacher import evaluate_turn_actions, evaluate_two_turn_actions, terminal_score


def test_generate_initial_actions_place_all_five_cards():
    board = Board.from_rows()
    actions = generate_initial_actions(board, ["Ah", "Kh", "Qh", "Jh", "Th"])
    assert actions
    for action in actions:
        assert len(action.placements) == 5
        assert action.discards == ()
        assert board.place(action.placements).card_count() == 5


def test_generate_final_turn_actions_fit_open_slots():
    board = Board.from_rows(
        top=["Qh", "2d"],
        middle=["Kh", "Kd", "6c", "8s", "Th"],
        bottom=["9c", "9d", "9s", "Kc"],
    )
    actions = generate_turn_actions(board, ["Qs", "Ah", "7d"])
    assert actions
    for action in actions:
        next_board = board.place(action.placements)
        assert next_board.is_complete()
        assert len(action.discards) == 1
        assert action.discards[0] in {"Qs", "Ah", "7d"}


def test_generate_one_slot_final_turn_keeps_two_discards():
    board = Board.from_rows(
        top=["Qh", "2d"],
        middle=["Kh", "Kd", "6c", "8s", "Th"],
        bottom=["9c", "9d", "9s", "Kc", "Ah"],
    )
    actions = generate_actions(board, ["Qs", "Ac", "7d"])
    assert actions
    for action in actions:
        assert len(action.placements) == 1
        assert len(action.discards) == 2
        assert board.place(action.placements).is_complete()


def test_exact_final_turn_prefers_qq_fl_when_board_stays_valid():
    board = Board.from_rows(
        top=["Qh", "2d"],
        middle=["Kh", "Kd", "6c", "8s", "Th"],
        bottom=["9c", "9d", "9s", "Kc"],
    )
    ranked = evaluate_turn_actions(board, ["Qs", "Ah", "7d"], fl_ev={14: 8.0})
    assert ranked
    best = ranked[0]
    assert ("Qs", "top") in best.action.placements
    assert best.board_score.fl_entry.qualifies
    assert best.board_score.fl_entry.entry_type == "qq"
    assert best.board_score.fl_entry.card_count == 14


def test_terminal_heads_up_score_includes_lines_royalties_and_fl_ev():
    hero = Board.from_rows(
        top=["Qh", "Qs", "2d"],
        middle=["Kh", "Kd", "6c", "8s", "Th"],
        bottom=["9c", "9d", "9s", "Kc", "Ah"],
    )
    opp = Board.from_rows(
        top=["Jh", "Js", "3d"],
        middle=["2h", "3h", "4c", "5d", "7s"],
        bottom=["Ac", "Ad", "4h", "4s", "8c"],
    )
    score, board_score = terminal_score(hero, opp, fl_ev={14: 8.0})
    assert not board_score.busted
    assert board_score.fl_entry.entry_type == "qq"
    assert score == 21.0


def test_terminal_heads_up_subtracts_opponent_fl_when_hero_busts():
    hero = Board.from_rows(
        top=["Ah", "As", "Kd"],
        middle=["2h", "3h", "4c", "5d", "7s"],
        bottom=["4d", "4s", "8c", "9c", "Td"],
    )
    opp = Board.from_rows(
        top=["Qh", "Qs", "2d"],
        middle=["Kh", "Kc", "6c", "8s", "Jh"],
        bottom=["9h", "9s", "9d", "Tc", "Jc"],
    )
    score, board_score = terminal_score(hero, opp, fl_ev={14: 8.0})
    assert board_score.busted
    assert score == -21.0


def test_terminal_heads_up_rejects_overlapping_cards():
    hero = Board.from_rows(
        top=["Qh", "Qs", "2d"],
        middle=["Kh", "Kd", "6c", "8s", "Th"],
        bottom=["9c", "9d", "9s", "Kc", "Ah"],
    )
    opp = Board.from_rows(
        top=["Jh", "Js", "Qh"],
        middle=["2h", "3h", "4c", "5d", "7s"],
        bottom=["Ac", "Ad", "4h", "4s", "8c"],
    )
    try:
        terminal_score(hero, opp, fl_ev={14: 8.0})
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("expected overlapping HU boards to be rejected")


def test_two_turn_expectimax_averages_best_final_turn_scores():
    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c", "8s"],
        bottom=["9c", "9d", "9s", "Kc"],
    )
    ranked = evaluate_two_turn_actions(
        board,
        ["Qs", "Ah", "7d"],
        fl_ev={14: 8.0},
        future_deals=[
            ("Qc", "2c", "3c"),
            ("Ad", "2c", "3c"),
        ],
    )
    assert ranked
    assert ranked[0].future_count == 2
    assert ranked[0].non_bust_future_count > 0
    assert ranked[0].score >= ranked[-1].score
    assert any(("Qs", "top") in item.action.placements for item in ranked)


def test_two_turn_expectimax_can_sample_future_deals():
    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c", "8s"],
        bottom=["9c", "9d", "9s", "Kc"],
    )
    ranked = evaluate_two_turn_actions(
        board,
        ["Qs", "Ah", "7d"],
        dead_cards=["2h", "3h"],
        fl_ev={14: 8.0},
        max_future_deals=3,
        seed=1,
    )
    assert ranked
    assert all(item.future_count == 3 for item in ranked)
    assert all(item.non_bust_future_count <= item.future_count for item in ranked)


def test_two_turn_expectimax_rejects_dead_card_overlap():
    board = Board.from_rows(
        top=["Qh"],
        middle=["Kh", "Kd", "6c", "8s"],
        bottom=["9c", "9d", "9s", "Kc"],
    )
    try:
        evaluate_two_turn_actions(
            board,
            ["Qs", "Ah", "7d"],
            dead_cards=["Qh"],
            fl_ev={14: 8.0},
        )
    except ValueError as exc:
        assert "duplicate card" in str(exc)
    else:
        raise AssertionError("expected overlapping dead cards to be rejected")
