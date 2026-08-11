from __future__ import annotations

from copy import deepcopy

import pytest

from ofc_regular.action_key import ActionKey
from ofc_regular.cards import ALL_CARDS
from ofc_regular.hu_rl_contract import HuRlActorViewV1, PublicPlacement
from ofc_regular.hu_rl_parity_fixture import (
    GOLDEN_HU_RL_PARITY_FIXTURE_SHA256,
    HU_RL_PARITY_FIXTURE_SCHEMA,
    HuRlParityFixtureError,
    HuRlParityFixtureV1,
    build_golden_hu_rl_parity_fixture_v1,
)
from ofc_regular.hu_rl_reference import HuRlReferenceEnv


def test_golden_fixture_is_deterministic_versioned_and_round_trips() -> None:
    first = build_golden_hu_rl_parity_fixture_v1()
    second = build_golden_hu_rl_parity_fixture_v1()

    assert first.to_dict()["schema"] == HU_RL_PARITY_FIXTURE_SCHEMA
    assert first.canonical_json() == second.canonical_json()
    assert first.digest() == second.digest()
    assert first.digest() == GOLDEN_HU_RL_PARITY_FIXTURE_SHA256
    assert len(first.digest()) == 64

    restored = HuRlParityFixtureV1.from_dict(first.to_dict())
    assert restored.to_dict() == first.to_dict()
    assert restored.digest() == first.digest()
    assert "oracle_only=<redacted>" in repr(restored)


def test_fixture_contains_exact_ten_decisions_and_full_action_mappings() -> None:
    payload = build_golden_hu_rl_parity_fixture_v1().to_dict()
    decisions = payload["decisions"]

    assert len(decisions) == 10
    for ordinal, decision in enumerate(decisions):
        assert decision["ordinal"] == ordinal
        assert decision["actor"] == ordinal % 2
        assert decision["street"] == f"T{ordinal // 2}"
        view = HuRlActorViewV1.from_dict(decision["actor_view"])
        assert view.digest() == decision["actor_view_digest"]

        mapping = decision["legal_action_mapping"]
        tokens = mapping["ordered_action_keys"]
        assert mapping["action_count"] == len(tokens)
        assert tokens == [
            key.to_token() for key in view.legal_action_mapping.action_keys
        ]
        assert mapping["action_set_digest"] == view.legal_action_mapping.action_set_digest
        assert mapping["action_order_digest"] == view.legal_action_mapping.action_order_digest
        selected_index = decision["selected_legal_index"]
        assert selected_index == len(tokens) // 2
        assert decision["selected_legal_action_key"] == tokens[selected_index]

        event = PublicPlacement.from_dict(decision["step"]["public_event"])
        selected = ActionKey.from_token(decision["selected_legal_action_key"])
        assert event.placement_masks == selected.masks[:3]
        assert event.discard_count == selected.discard_mask.bit_count()
        assert decision["step"]["done"] is (ordinal == 9)
        if ordinal < 9:
            assert decision["step"]["rewards"] == [0.0, 0.0]

    terminal = payload["terminal"]
    assert [sum(len(cards) for cards in board.values()) for board in terminal["boards"]] == [13, 13]
    assert terminal["rewards"][0] == -terminal["rewards"][1]
    assert decisions[-1]["step"]["rewards"] == terminal["rewards"]


def test_explicit_deck_and_hidden_world_fields_are_absent_outside_oracle_only() -> None:
    payload = build_golden_hu_rl_parity_fixture_v1().to_dict()
    assert payload["oracle_only"] == {
        "schema": "regular_ofc_hu_rl_parity_oracle_only_v1",
        "explicit_deck": [*ALL_CARDS[5:], *ALL_CARDS[:5]],
    }

    public_copy = deepcopy(payload)
    public_copy.pop("oracle_only")
    forbidden = {
        "explicit_deck",
        "deck",
        "deck_tail",
        "remaining_deck",
        "world_state",
        "private_discards",
        "opponent_private_discard",
        "opponent_private_discards",
        "true_dead_cards",
        "audit_truth",
        "replay_truth",
    }
    assert forbidden.isdisjoint(_all_mapping_keys(public_copy))

    for decision in payload["decisions"]:
        event_keys = set(decision["step"]["public_event"])
        assert "discard_mask" not in event_keys
        assert "opponent_legal_digest" not in event_keys


def test_fixture_selected_keys_replay_views_steps_and_terminal_exactly() -> None:
    fixture = build_golden_hu_rl_parity_fixture_v1()
    payload = fixture.to_dict()
    env = HuRlReferenceEnv(payload["oracle_only"]["explicit_deck"])

    for decision in payload["decisions"]:
        assert env.observe().to_dict() == decision["actor_view"]
        selected = ActionKey.from_token(decision["selected_legal_action_key"])
        step = env.step(selected)
        assert step.public_placement.to_dict() == decision["step"]["public_event"]
        assert step.done == decision["step"]["done"]
        assert list(step.rewards) == decision["step"]["rewards"]

    assert [
        {
            row: sorted(getattr(board, row), key=ALL_CARDS.index)
            for row in ("top", "middle", "bottom")
        }
        for board in env.boards
    ] == payload["terminal"]["boards"]
    assert env.terminal_rewards() == payload["terminal"]["rewards"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("actor_view_digest", "actor_view digest mismatch"),
        ("action_order_digest", "action order digest mismatch"),
        ("selected_key", "selected legal ActionKey disagrees"),
        ("step_reward", "nonterminal decision rewards must be zero"),
        ("terminal_board", "invalid terminal boards|terminal boards failed exact replay"),
        ("oracle_deck", "pinned oracle deck"),
    ),
)
def test_tampered_fixture_is_rejected_fail_closed(mutation: str, message: str) -> None:
    payload = build_golden_hu_rl_parity_fixture_v1().to_dict()
    if mutation == "actor_view_digest":
        payload["decisions"][0]["actor_view_digest"] = "0" * 64
    elif mutation == "action_order_digest":
        payload["decisions"][0]["legal_action_mapping"]["action_order_digest"] = "0" * 64
    elif mutation == "selected_key":
        payload["decisions"][0]["selected_legal_action_key"] = payload["decisions"][0][
            "legal_action_mapping"
        ]["ordered_action_keys"][0]
    elif mutation == "step_reward":
        payload["decisions"][0]["step"]["rewards"] = [1.0, -1.0]
    elif mutation == "terminal_board":
        payload["terminal"]["boards"][0]["top"][0] = payload["terminal"]["boards"][1]["top"][0]
    elif mutation == "oracle_deck":
        payload["oracle_only"]["explicit_deck"][0], payload["oracle_only"]["explicit_deck"][1] = (
            payload["oracle_only"]["explicit_deck"][1],
            payload["oracle_only"]["explicit_deck"][0],
        )

    with pytest.raises(HuRlParityFixtureError, match=message):
        HuRlParityFixtureV1.from_dict(payload)


def test_unknown_fixture_and_nested_fields_are_rejected() -> None:
    payload = build_golden_hu_rl_parity_fixture_v1().to_dict()
    payload["world_state"] = {}
    with pytest.raises(HuRlParityFixtureError, match="unknown fields: world_state"):
        HuRlParityFixtureV1.from_dict(payload)

    payload = build_golden_hu_rl_parity_fixture_v1().to_dict()
    payload["decisions"][3]["step"]["audit_truth"] = {}
    with pytest.raises(HuRlParityFixtureError, match="unknown fields: audit_truth"):
        HuRlParityFixtureV1.from_dict(payload)


def _all_mapping_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        keys.update(str(key) for key in value)
        for nested in value.values():
            keys.update(_all_mapping_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.update(_all_mapping_keys(nested))
    return keys
