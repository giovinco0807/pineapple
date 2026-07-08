"""Regression tests for the hidden-discard information model.

Standard Pineapple OFC hides each player's discards from the opponent.
These tests play full hands through the real play loops with probe
policies that record every ``dead_cards`` argument they receive, then
assert no policy ever saw a card the opponent discarded.
"""

from __future__ import annotations

from ofc_regular.action_space import generate_actions, generate_turn_actions
from ofc_regular.play_ai import play_hand
from ofc_regular.visibility import HuDiscardTracker


class ProbePolicy:
    """Deterministic policy that records the information it is shown."""

    def __init__(self) -> None:
        self.seen_dead: list[tuple[str, ...]] = []
        self.own_discards: list[str] = []

    def choose_action(self, board, dealt_cards, *, dead_cards=(), opponent_board=None, **_kwargs):
        self.seen_dead.append(tuple(dead_cards))
        dealt = tuple(dealt_cards)
        if board.card_count() == 0:
            action = generate_actions(board, dealt)[0]
        else:
            action = generate_turn_actions(board, dealt)[0]
        self.own_discards.extend(action.discards)
        return action


def test_play_hand_never_reveals_opponent_discards():
    policy_p0 = ProbePolicy()
    policy_p1 = ProbePolicy()
    result = play_hand(seed=123, policy_p0=policy_p0, policy_p1=policy_p1)

    assert result.board_p0.card_count() == 13
    assert result.board_p1.card_count() == 13
    # Pineapple: 4 turns x 1 discard each.
    assert len(policy_p0.own_discards) == 4
    assert len(policy_p1.own_discards) == 4

    for hero, villain, hero_board, villain_board in (
        (policy_p0, policy_p1, result.board_p0, result.board_p1),
        (policy_p1, policy_p0, result.board_p1, result.board_p0),
    ):
        villain_discards = set(villain.own_discards)
        for seen in hero.seen_dead:
            leaked = set(seen) & villain_discards
            assert not leaked, f"policy saw opponent discards: {sorted(leaked)}"
        # The final decision must still include all of the hero's own prior
        # discards (3 by the last turn) and only cards from the opponent board
        # or the hero's own discards.
        final_seen = set(hero.seen_dead[-1])
        prior_own = set(hero.own_discards[:3])
        assert prior_own <= final_seen
        assert final_seen <= set(villain_board.all_cards()) | prior_own


def test_discard_tracker_separates_players():
    tracker = HuDiscardTracker()
    tracker.record(0, ["As"])
    tracker.record(1, ["Kd", "Qh"])
    assert tracker.own_discards(0) == ("As",)
    assert tracker.own_discards(1) == ("Kd", "Qh")
    assert set(tracker.all_discards()) == {"As", "Kd", "Qh"}
    clone = tracker.clone()
    clone.record(0, ["2c"])
    assert tracker.own_discards(0) == ("As",)
    assert clone.own_discards(0) == ("As", "2c")
