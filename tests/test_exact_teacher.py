import json
from pathlib import Path

from ofc_regular.action_space import generate_actions, generate_initial_actions, generate_turn_actions
from ofc_regular.hu_infoset import (
    FALLBACK_FL_EV,
    FL_EV_CONFIG_PATH,
    FL_EV_CONFIG_RELPATH,
    ScoringContext,
)
from ofc_regular.state import Board
from ofc_regular.teacher import DEFAULT_FL_EV, evaluate_turn_actions, evaluate_two_turn_actions, load_fl_ev, terminal_score

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ADOPTED_FL_EV_14 = 9.6
# The supersession chain, oldest first. Every one of these files stays on disk
# unmodified, because a corpus labelled under a constant can only be validated
# against the file that records it.
_SUPERSEDED_CONFIG = "configs/fl_ev_regular_2k.json"
_SUPERSEDED_FL_EV_14 = 10.227020614683454
_SUPERSEDED_V3_CONFIG = "configs/fl_ev_regular_v3_direct2.json"
_SUPERSEDED_V3_FL_EV_14 = 9.109


def test_default_fl_ev_uses_direct_hu_calibration_config():
    assert DEFAULT_FL_EV[14] == _ADOPTED_FL_EV_14
    assert load_fl_ev()[14] == _ADOPTED_FL_EV_14


def test_default_fl_ev_config_is_the_v4_selfplay_measurement_with_provenance():
    """The adopted constant is one file, and that file says where it came from.

    Pinned rather than merely loaded: a value this deep in the scoring path can
    be changed by editing one number, and the run, interval and predecessor are
    what make such an edit reviewable afterwards.
    """
    assert FL_EV_CONFIG_RELPATH == "configs/fl_ev_regular_v4_selfplay.json"
    assert FL_EV_CONFIG_PATH == _REPO_ROOT / FL_EV_CONFIG_RELPATH

    payload = json.loads(FL_EV_CONFIG_PATH.read_text(encoding="utf-8"))
    assert payload["fl_ev"] == {"14": _ADOPTED_FL_EV_14}
    assert payload["rule_set"] == "regular"
    assert payload["deck_cards"] == 52
    assert payload["include_jokers"] is False

    provenance = payload["provenance"]
    assert provenance["method"] == (
        "self_play_entry_episode_fixed_point_v4 (M6 run B; adaptive Fantasyland "
        "actually played on both sides)"
    )
    assert provenance["run"] == "~/ofc-ladder2/runB"
    assert provenance["date"] == "2026-08-06"
    assert provenance["adopted_fl_ev_14"] == _ADOPTED_FL_EV_14
    assert provenance["hands"] == 150000
    assert provenance["seed_range"] == [998500000, 998507499]

    # The fixed point is arithmetic on two measured numbers, so it is recomputed
    # here rather than trusted: a provenance block that does not reproduce its
    # own adopted value is the exact failure the June estimator shipped.
    inputs = provenance["fixed_point_inputs"]
    v0 = inputs["V0_entry_episode_value"]
    x0 = inputs["x0_constant_engine_was_told"]
    slope = inputs["slope"]
    recomputed = (v0 - slope * x0) / (1.0 - slope)
    assert round(recomputed, 3) == round(provenance["measured_fixed_point_14"], 3)
    assert round(recomputed, 1) == _ADOPTED_FL_EV_14
    assert x0 == _SUPERSEDED_V3_FL_EV_14

    assert provenance["predecessor_file"] == _SUPERSEDED_V3_CONFIG
    assert provenance["predecessor_fl_ev_14"] == _SUPERSEDED_V3_FL_EV_14
    # The whole chain is carried forward, June's defect included.
    chain = {entry["file"]: entry for entry in provenance["predecessor_chain"]}
    assert set(chain) == {_SUPERSEDED_CONFIG, _SUPERSEDED_V3_CONFIG}
    assert chain[_SUPERSEDED_CONFIG]["fl_ev_14"] == _SUPERSEDED_FL_EV_14
    assert chain[_SUPERSEDED_CONFIG]["defect"]
    assert chain[_SUPERSEDED_V3_CONFIG]["fl_ev_14"] == _SUPERSEDED_V3_FL_EV_14
    assert chain[_SUPERSEDED_V3_CONFIG]["defect"]


def test_superseded_configs_survive_untouched_and_are_no_longer_read():
    """History stays on disk; only the reader moved.

    Both superseded constants are checked, not just the most recent one: the
    older corpora are the ones most likely to be revalidated by someone who was
    not here, and they can only be revalidated while their config exists.
    """
    for relpath, value in (
        (_SUPERSEDED_CONFIG, _SUPERSEDED_FL_EV_14),
        (_SUPERSEDED_V3_CONFIG, _SUPERSEDED_V3_FL_EV_14),
    ):
        superseded = json.loads(
            (_REPO_ROOT / relpath).read_text(encoding="utf-8")
        )
        assert superseded["fl_ev"] == {"14": value}
        assert load_fl_ev()[14] != value
        assert load_fl_ev(_REPO_ROOT / relpath)[14] == value


def test_observation_default_and_fallback_cannot_drift_from_the_config():
    """One source of truth: no consumer keeps its own copy of the constant.

    The fallback is asserted equal to the config rather than to any earlier
    value, so a tree shipped without ``configs/`` degrades to today's number
    instead of silently reviving a superseded one.
    """
    assert ScoringContext().fl_ev == tuple(sorted(load_fl_ev().items()))
    assert FALLBACK_FL_EV == {14: _ADOPTED_FL_EV_14}
    assert dict(ScoringContext().fl_ev) == DEFAULT_FL_EV


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
