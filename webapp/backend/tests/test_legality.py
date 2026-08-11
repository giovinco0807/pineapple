from __future__ import annotations

from dataclasses import replace

import pytest

from ofc_webapp.domain import (
    ActionSubmission,
    IllegalAction,
    apply_action,
    create_match,
    legal_actions,
    start_hand,
)


def _human_first_hand():
    match = replace(
        create_match(seed=101, match_id="legality-match"),
        first_hand_first="human",
    )
    return start_hand(match, hand_id="legality-hand")[1]


@pytest.mark.parametrize(
    "submission_factory",
    [
        lambda cards: ActionSubmission(
            placements=tuple((card, "top") for card in cards),
        ),
        lambda cards: ActionSubmission(
            placements=(
                (cards[0], "top"),
                (cards[0], "middle"),
                (cards[2], "bottom"),
                (cards[3], "bottom"),
                (cards[4], "bottom"),
            ),
        ),
        lambda cards: ActionSubmission(
            placements=(
                (cards[0], "top"),
                (cards[1], "top"),
                (cards[2], "middle"),
                (cards[3], "middle"),
                (cards[4], "bottom"),
            ),
            discards=(cards[4],),
        ),
    ],
)
def test_invalid_opening_actions_are_rejected(submission_factory):
    hand = _human_first_hand()
    cards = hand.current_turn.dealt_cards
    with pytest.raises(IllegalAction):
        apply_action(hand, submission_factory(cards), actor="human")


def test_t1_requires_exactly_two_placements_and_one_discard():
    hand = _human_first_hand()
    for _ in range(2):
        action = legal_actions(hand)[0]
        hand = apply_action(
            hand,
            ActionSubmission(action.placements, action.discards),
            actor=hand.current_turn.actor,
        ).hand
    assert hand.current_turn.street == "T1"
    cards = hand.current_turn.dealt_cards
    missing_discard = ActionSubmission(
        placements=((cards[0], "top"), (cards[1], "middle")),
        discards=(),
    )
    with pytest.raises(IllegalAction):
        apply_action(hand, missing_discard, actor=hand.current_turn.actor)


def test_action_submission_order_does_not_change_legality():
    hand = _human_first_hand()
    action = legal_actions(hand)[0]
    reversed_submission = ActionSubmission(
        placements=tuple(reversed(action.placements)),
        discards=tuple(reversed(action.discards)),
    )
    transition = apply_action(hand, reversed_submission, actor="human")
    assert transition.hand.current_turn_index == 1


def test_fl_requires_13_placements_one_discard_and_full_rows():
    match = replace(
        create_match(seed=333, match_id="fl-legality-match"),
        first_hand_first="human",
        pending_fantasyland=(True, False),
    )
    _, hand = start_hand(match, hand_id="fl-legality-hand")
    cards = hand.current_turn.dealt_cards

    wrong_discard_count = ActionSubmission(
        placements=tuple(
            [(card, "top") for card in cards[:3]]
            + [(card, "middle") for card in cards[3:8]]
            + [(card, "bottom") for card in cards[8:13]]
        ),
        discards=(),
    )
    with pytest.raises(IllegalAction):
        apply_action(hand, wrong_discard_count, actor="human")

    overflow = ActionSubmission(
        placements=tuple((card, "top") for card in cards[:13]),
        discards=(cards[13],),
    )
    with pytest.raises(IllegalAction):
        apply_action(hand, overflow, actor="human")

    duplicate = ActionSubmission(
        placements=tuple(
            [(cards[0], "top"), (cards[0], "top"), (cards[2], "top")]
            + [(card, "middle") for card in cards[3:8]]
            + [(card, "bottom") for card in cards[8:13]]
        ),
        discards=(cards[13],),
    )
    with pytest.raises(IllegalAction):
        apply_action(hand, duplicate, actor="human")

    valid = ActionSubmission(
        placements=tuple(
            [(card, "top") for card in cards[:3]]
            + [(card, "middle") for card in cards[3:8]]
            + [(card, "bottom") for card in cards[8:13]]
        ),
        discards=(cards[13],),
    )
    transition = apply_action(hand, valid, actor="human")
    assert transition.hand.board_for("human").is_complete()
    assert transition.hand.discards_for("human") == (cards[13],)

