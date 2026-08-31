import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.encoding import Board
from ai.engine.game_engine import GameEngine, Hand
from ai.training.rl_session_env import HeuristicAgent, SessionState, play_session


def test_session_starts_at_200_and_conserves_points():
    state = SessionState.new(starting_chips=200, finish_gap=40)
    assert state.chips == [200, 200]
    assert sum(state.chips) == 400

    delta = state.apply_capped_score([12, -12])
    assert delta == [12, -12]
    assert state.chips == [212, 188]
    assert sum(state.chips) == 400


def test_capped_score_never_goes_negative():
    state = SessionState(starting_chips=200, chips=[395, 5], btn=0)
    delta = state.apply_capped_score([20, -20])
    assert delta == [5, -5]
    assert state.chips == [400, 0]
    assert sum(state.chips) == 400

    delta = state.apply_capped_score([-30, 30])
    assert delta == [-30, 30]
    assert state.chips == [370, 30]
    assert sum(state.chips) == 400


def test_session_end_requires_no_fl_and_gap():
    state = SessionState(starting_chips=200, chips=[220, 180], btn=0)
    assert state.should_end_session()

    state.is_fl = [True, False]
    assert not state.should_end_session()

    state.is_fl = [False, False]
    state.chips = [219, 181]
    assert not state.should_end_session()


def test_button_alternates():
    state = SessionState(starting_chips=200, chips=[200, 200], btn=0)
    state.advance_button()
    assert state.btn == 1
    assert state.hand_count == 1
    state.advance_button()
    assert state.btn == 0
    assert state.hand_count == 2


def test_play_session_adds_session_return_and_legal_masks():
    agents = [HeuristicAgent(top_k=3, temperature=0.0), HeuristicAgent(top_k=3, temperature=0.0)]
    records, summary = play_session(
        agents,
        session_id=7,
        max_hands=1,
        fl_samples=10,
        rng_seed=123,
    )

    assert summary["hands"] == 1
    assert sum(summary["final_chips"]) == 400
    assert records
    for rec in records:
        assert rec["session_id"] == 7
        assert rec["session_return"] is not None
        assert rec["legal_mask"]
        assert rec["chosen_action_index"] in rec["candidate_actions"]
        if rec["turn"] > 0:
            assert len(rec["legal_mask"]) == 27


def test_fl_entry_and_stay_state_helpers():
    hand = Hand(deck=[], btn=0)
    hand.boards = [
        Board(
            top=["Qh", "Qs", "2d"],
            middle=["Kh", "Ks", "Kd", "7c", "9h"],
            bottom=["Ah", "Ad", "Ac", "As", "2h"],
        ),
        Board(
            top=["2h", "3s", "4d"],
            middle=["5h", "6s", "7d", "8c", "9s"],
            bottom=["Th", "Ts", "Jd", "Qc", "Ks"],
        ),
    ]
    result = GameEngine.compute_result(hand)
    assert result.fl_entry[0]
    assert result.fl_card_count[0] == 14
