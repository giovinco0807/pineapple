"""Golden tests for the two-Joker, progressive-Fantasyland ruleset.

The important rule is board-level, not row-local: a Joker takes the strongest
substitution that keeps ``Top <= Middle <= Bottom``.  Every terminal consumer
(the headless engine, backend scoring, rollout scoring, and Rust teachers) must
agree with the cases documented here.
"""

from types import SimpleNamespace

from ai.engine.encoding import Board
from ai.engine.action_space import Action
from ai.cfr.ofc_cfr import _is_valid_t0_action
from ai.engine.game_engine import (
    GameEngine,
    evaluate_board_with_joker_constraint,
    evaluate_hand,
    get_bottom_royalty,
    hand_category,
)
from ai.engine.scoring import calculate_scores, check_fl_stay_from_cards
from ai.mcts.rollout_evaluator import RolloutEvaluator


def _qq_joker_board() -> Board:
    """Raw Top is QQQ, but it must downgrade to QQ below Middle KK."""
    return Board(
        top=["Qh", "Qs", "X1"],
        middle=["Kh", "Ks", "9d", "8c", "7h"],
        bottom=["Ah", "Ad", "Ac", "5s", "4d"],
    )


def _weak_legal_opponent() -> Board:
    return Board(
        top=["2c", "3c", "4c"],
        middle=["6h", "6d", "7s", "8s", "Tc"],
        bottom=["Jh", "Jd", "Qc", "Qd", "Kc"],
    )


def _headless_result(hero: Board, opponent: Board):
    hand = SimpleNamespace(boards=[hero, opponent])
    return GameEngine.compute_result(hand)


def _backend_result(
    hero: Board,
    opponent: Board,
    *,
    hero_is_fl: bool = False,
    hero_fl_cards: int = 0,
):
    game = SimpleNamespace(
        boards=[hero.to_dict(), opponent.to_dict()],
        is_fantasyland=[hero_is_fl, False],
        fl_card_count=[hero_fl_cards, 0],
        chips=[200, 200],
        btn=0,
    )
    return calculate_scores(game)


def test_qq_joker_downgrade_has_one_golden_terminal_result_across_python_paths():
    """QQX must become the best legal QQ, not an illegal QQQ.

    Against the fixture opponent Hero scoops (+6) and receives the QQ Top
    royalty (+7), so the exact raw score is +13.  QQ also enters 14-card FL.
    """
    hero = _qq_joker_board()
    opponent = _weak_legal_opponent()

    evaluated = evaluate_board_with_joker_constraint(
        hero.top, hero.middle, hero.bottom
    )
    assert evaluated["busted"] is False
    assert evaluated["values"]["top"] == evaluate_hand(
        ["Qh", "Qs", "As"], 3
    )
    assert evaluated["royalties"] == {
        "top": 7,
        "middle": 0,
        "bottom": 0,
        "total": 7,
    }
    assert (evaluated["fl_entry"], evaluated["fl_card_count"]) == (True, 14)

    headless = _headless_result(hero, opponent)
    backend = _backend_result(hero, opponent)

    assert headless.busted == [False, False]
    assert headless.royalties[0]["total"] == 7
    assert (headless.fl_entry[0], headless.fl_card_count[0]) == (True, 14)
    assert headless.raw_score == [13, -13]

    assert backend["busted"] == headless.busted
    assert backend["royalties"] == headless.royalties
    assert backend["fl_entry"] == headless.fl_entry
    assert backend["fl_card_count"] == headless.fl_card_count
    assert backend["raw_score"] == headless.raw_score

    # Rollouts are teachers too; they must not independently maximize QQQ and
    # call the same legal board a foul.
    assert RolloutEvaluator.compute_score_raw(hero, opponent) == 13.0


def test_middle_joker_downgrades_to_the_strongest_pair_below_bottom():
    top = ["2h", "3d", "4s"]
    middle = ["Qh", "Qd", "X1", "8c", "7c"]
    bottom = ["Kh", "Kd", "9s", "6s", "5d"]

    evaluated = evaluate_board_with_joker_constraint(top, middle, bottom)

    assert evaluated["busted"] is False
    assert evaluated["values"]["middle"] == evaluate_hand(
        ["Qh", "Qd", "As", "8c", "7c"], 5
    )
    assert evaluated["royalties"]["middle"] == 0


def test_two_jokers_choose_quads_instead_of_a_lower_full_house():
    """Shared Python/Rust regression case for greedy Joker allocation."""
    cards = ["As", "2s", "2h", "X1", "X2"]

    value = evaluate_hand(cards, 5)

    assert hand_category(value) == 7
    assert get_bottom_royalty(cards) == 10


def test_joker_flush_uses_the_strongest_available_kicker():
    """Joker must become Ah, not a zero-valued placeholder in a flush."""
    joker_value = evaluate_hand(["Kh", "Qh", "9h", "3h", "X1"], 5)
    natural_value = evaluate_hand(["Ah", "Kh", "Qh", "9h", "3h"], 5)

    assert joker_value == natural_value

    two_joker_quads = evaluate_hand(["Qs", "Qc", "Qd", "X1", "X2"], 5)
    natural_quads = evaluate_hand(["Qs", "Qc", "Qd", "Qh", "As"], 5)
    assert two_joker_quads == natural_quads


def test_top_trips_are_compared_by_rank_against_middle_trips():
    """Top AAA is stronger than Middle 222 and therefore still fouls."""
    evaluated = evaluate_board_with_joker_constraint(
        ["Ah", "Ad", "Ac"],
        ["2h", "2d", "2c", "Ks", "Qd"],
        ["3h", "4d", "5c", "6s", "7h"],
    )

    assert evaluated["busted"] is True
    assert evaluated["royalties"]["total"] == 0
    assert (evaluated["fl_entry"], evaluated["fl_card_count"]) == (False, 0)


def test_fl_stay_uses_the_constrained_top_value_and_preserves_card_count():
    hero = _qq_joker_board()

    # Raw QQX looks like trips, but canonical evaluation downgrades it to QQ;
    # it must not satisfy the Top-trips stay condition.
    assert check_fl_stay_from_cards(
        hero.top,
        hero.bottom,
        16,
        middle_cards=hero.middle,
    ) == (False, 0)

    backend = _backend_result(
        hero,
        _weak_legal_opponent(),
        hero_is_fl=True,
        hero_fl_cards=16,
    )
    assert (backend["fl_entry"][0], backend["fl_card_count"][0]) == (False, 0)

    # A Joker-assisted Bottom quads is an unconstrained legitimate stay and
    # keeps the exact progressive FL card count from the previous hand.
    assert check_fl_stay_from_cards(
        ["2h", "3d", "4s"],
        ["Ah", "Ad", "Ac", "X1", "Kh"],
        15,
        middle_cards=["6h", "6d", "7s", "8s", "Tc"],
    ) == (True, 15)


def test_python_cfr_does_not_remove_legal_ace_or_joker_rows():
    """Rank-based placement preferences must not masquerade as rules."""
    assert _is_valid_t0_action(Action([("As", "bottom")]))
    assert _is_valid_t0_action(Action([("X1", "middle")]))
    assert _is_valid_t0_action(
        Action([("X1", "middle"), ("X2", "middle")])
    )
