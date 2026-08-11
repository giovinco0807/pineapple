from __future__ import annotations

from dataclasses import replace

import pytest

from ofc_webapp.domain import (
    ActionSubmission,
    FinalScore,
    HandStatus,
    IllegalAction,
    InvalidTransition,
    MatchStatus,
    apply_action,
    build_observation,
    continue_match,
    create_match,
    legal_actions,
    settle_hand,
    start_hand,
)
from ofc_regular.hu_infoset import ActorObservation


def _submission(action) -> ActionSubmission:
    return ActionSubmission(
        placements=action.placements,
        discards=action.discards,
    )


def _autoplay_to_score(hand):
    decisions = []
    while hand.status == HandStatus.PLAYING:
        turn = hand.current_turn
        assert turn is not None
        if turn.street == "FL":
            cards = turn.dealt_cards
            submission = ActionSubmission(
                placements=tuple(
                    [(card, "top") for card in cards[:3]]
                    + [(card, "middle") for card in cards[3:8]]
                    + [(card, "bottom") for card in cards[8:13]]
                ),
                discards=(cards[13],),
            )
        else:
            submission = _submission(legal_actions(hand)[0])
        transition = apply_action(hand, submission, actor=turn.actor)
        decisions.append(transition.decision)
        hand = transition.hand
    return hand, decisions


def test_normal_hand_progresses_through_t0_to_t4_and_alternates_positions():
    match = create_match(seed=7129, match_id="match-1")
    active_match, hand = start_hand(
        match, hand_id="hand-0", started_at="2026-07-29T00:00:00+00:00"
    )

    assert active_match.status == MatchStatus.IN_HAND
    assert [turn.street for turn in hand.turns] == [
        "T0",
        "T0",
        "T1",
        "T1",
        "T2",
        "T2",
        "T3",
        "T3",
        "T4",
        "T4",
    ]
    assert len(hand.deck_order) == 52
    assert all("X" not in card for card in hand.deck_order)

    finished, decisions = _autoplay_to_score(hand)
    assert finished.status == HandStatus.AWAITING_SCORE
    assert len(decisions) == 10
    assert [decision.sequence_no for decision in decisions] == list(range(10))
    assert all(board.is_complete() for board in finished.boards)
    assert tuple(len(cards) for cards in finished.private_discards) == (4, 4)

    settled_match, settled_hand = settle_hand(
        active_match,
        finished,
        FinalScore(
            hu_score=7,
            breakdown={"source": "fake-score-port", "fouls": [False, False]},
        ),
        ended_at="2026-07-29T00:01:00+00:00",
    )
    assert settled_hand.result is not None
    assert settled_hand.result.breakdown["source"] == "fake-score-port"
    assert settled_match.status == MatchStatus.AWAITING_CONTINUE
    first = hand.turns[0].actor
    expected_stacks = (207, 193) if first == "human" else (193, 207)
    assert settled_match.stacks == expected_stacks

    ready = continue_match(settled_match, should_continue=True)
    next_active, next_hand = start_hand(ready, hand_id="hand-1")
    assert next_hand.positions == (
        ("second", "first")
        if hand.positions == ("first", "second")
        else ("first", "second")
    )
    assert next_active.current_hand_id == "hand-1"


def test_table_stakes_caps_transfer_and_zero_forces_match_end():
    match = replace(
        create_match(seed=4, match_id="match-cap"),
        first_hand_first="human",
        stacks=(399, 1),
    )
    active, hand = start_hand(match, hand_id="hand-cap")
    hand, _ = _autoplay_to_score(hand)
    settled, scored_hand = settle_hand(
        active,
        hand,
        FinalScore(
            hu_score=40,
            breakdown={"engine": True},
            first_next_fantasyland=True,
        ),
    )

    assert scored_hand.result is not None
    assert scored_hand.result.raw_score == 40
    assert scored_hand.result.capped_score == 1
    assert settled.stacks == (400, 0)
    assert settled.status == MatchStatus.COMPLETED
    assert settled.pending_fantasyland == (False, False)
    with pytest.raises(InvalidTransition):
        start_hand(settled)


def test_no_fl_hand_requires_explicit_continue_or_finish():
    match = create_match(seed=10, match_id="match-continue")
    active, hand = start_hand(match, hand_id="hand-continue")
    hand, _ = _autoplay_to_score(hand)
    awaiting, _ = settle_hand(
        active, hand, FinalScore(hu_score=0, breakdown={})
    )
    assert awaiting.status == MatchStatus.AWAITING_CONTINUE
    assert continue_match(awaiting, should_continue=False).status == MatchStatus.COMPLETED


def test_fl_entry_is_auto_continue_and_always_deals_14_next_hand():
    match = replace(
        create_match(seed=11, match_id="match-fl"),
        first_hand_first="human",
    )
    active, hand = start_hand(match, hand_id="qualifying-hand")
    hand, _ = _autoplay_to_score(hand)
    ready, _ = settle_hand(
        active,
        hand,
        FinalScore(
            hu_score=3,
            breakdown={"engine": True},
            first_next_fantasyland=True,
        ),
    )
    assert ready.status == MatchStatus.READY
    assert ready.pending_fantasyland == (True, False)

    _, fl_hand = start_hand(ready, hand_id="fl-hand")
    human_fl = next(
        turn
        for turn in fl_hand.turns
        if turn.actor == "human" and turn.street == "FL"
    )
    assert len(human_fl.dealt_cards) == 14
    assert all(
        turn.actor != "human" or turn.street == "FL"
        for turn in fl_hand.turns
    )


@pytest.mark.parametrize(
    ("fantasyland", "turn_count", "discard_counts"),
    [
        ((True, False), 6, (1, 4)),
        ((False, True), 6, (4, 1)),
        ((True, True), 2, (1, 1)),
    ],
)
def test_mixed_and_double_fl_hands_complete_without_revealing_extra_turns(
    fantasyland, turn_count, discard_counts
):
    match = replace(
        create_match(seed=19, match_id=f"match-mixed-{fantasyland}"),
        pending_fantasyland=fantasyland,
    )
    _, hand = start_hand(match, hand_id=f"hand-mixed-{fantasyland}")
    assert len(hand.turns) == turn_count
    completed, decisions = _autoplay_to_score(hand)

    assert completed.status == HandStatus.AWAITING_SCORE
    assert len(decisions) == turn_count
    assert all(board.is_complete() for board in completed.boards)
    assert tuple(len(cards) for cards in completed.private_discards) == discard_counts
    for player, in_fl in zip(("human", "ai"), fantasyland, strict=True):
        streets = [
            turn.street for turn in completed.turns if turn.actor == player
        ]
        assert streets == (["FL"] if in_fl else ["T0", "T1", "T2", "T3", "T4"])


def test_normal_observation_uses_existing_actor_observation():
    match = replace(
        create_match(seed=25, match_id="match-observation"),
        first_hand_first="human",
    )
    _, hand = start_hand(match, hand_id="hand-observation")
    observation = build_observation(hand, actor="human")

    assert isinstance(observation, ActorObservation)
    assert observation.dealt_cards == hand.current_turn.dealt_cards
    assert observation.opponent_public_board.card_count() == 0
    assert observation.scoring.fantasyland_cards == 14
