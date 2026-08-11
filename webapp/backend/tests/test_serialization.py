from __future__ import annotations

from dataclasses import replace
import json

from ofc_webapp.domain import (
    ActionSubmission,
    HiddenFantasylandObservation,
    apply_action,
    build_observation,
    create_match,
    start_hand,
)
from ofc_webapp.serialization import serialize_hand, serialize_match


def _fl_submission(cards):
    return ActionSubmission(
        placements=tuple(
            [(card, "top") for card in cards[:3]]
            + [(card, "middle") for card in cards[3:8]]
            + [(card, "bottom") for card in cards[8:13]]
        ),
        discards=(cards[13],),
    )


def test_client_hand_view_hides_deck_ai_hand_discards_and_in_progress_fl_board():
    match = replace(
        create_match(seed=404, match_id="safe-match"),
        first_hand_first="ai",
        pending_fantasyland=(False, True),
    )
    active, hand = start_hand(match, hand_id="safe-hand")
    ai_turn = hand.current_turn
    assert ai_turn.actor == "ai"
    assert ai_turn.street == "FL"
    ai_private_cards = set(ai_turn.dealt_cards)

    hand = apply_action(
        hand, _fl_submission(ai_turn.dealt_cards), actor="ai"
    ).hand
    assert hand.current_turn.actor == "human"
    payload = serialize_hand(hand, viewer="human")
    encoded = json.dumps(payload, sort_keys=True)

    assert payload["boards"]["ai"] == {"top": [], "middle": [], "bottom": []}
    assert payload["dealt_cards"] == list(hand.current_turn.dealt_cards)
    assert payload["private_discards"] == []
    assert payload["opponent_discard_count"] == 0
    assert "deck_order" not in payload
    assert "turns" not in payload
    assert all(card not in encoded for card in ai_private_cards)

    observation = build_observation(hand, actor="human")
    assert isinstance(observation, HiddenFantasylandObservation)
    observation_json = json.dumps(observation.to_dict(), sort_keys=True)
    assert all(card not in observation_json for card in ai_private_cards)

    match_payload = serialize_match(active)
    assert "seed" not in match_payload


def test_only_current_viewers_deal_is_returned():
    match = replace(
        create_match(seed=405, match_id="turn-safe-match"),
        first_hand_first="ai",
    )
    _, hand = start_hand(match, hand_id="turn-safe-hand")
    ai_cards = hand.current_turn.dealt_cards

    human_view = serialize_hand(hand, viewer="human")
    ai_view = serialize_hand(hand, viewer="ai")
    assert human_view["dealt_cards"] == []
    assert ai_view["dealt_cards"] == list(ai_cards)
    assert human_view["ai_pending"] is True

