from __future__ import annotations

from dataclasses import replace

import pytest

from ofc_regular.action_key import ActionKey
from ofc_regular.cards import ALL_CARDS
from ofc_regular.hu_rl_contract import PublicPlacement
from ofc_regular.hu_rl_reference import (
    HuRlReferenceEnv,
    HuRlReferenceError,
    HuRlReferenceSnapshot,
)
from ofc_regular.state import Board
from ofc_regular.teacher import DEFAULT_FL_EV, terminal_score


def _choose_middle_key(env: HuRlReferenceEnv) -> ActionKey:
    keys = env.legal_actions()
    return keys[len(keys) // 2]


def _play_to_terminal(env: HuRlReferenceEnv) -> tuple[list[ActionKey], list[float]]:
    chosen: list[ActionKey] = []
    while not env.done:
        selected = _choose_middle_key(env)
        chosen.append(selected)
        result = env.step(selected)
        assert result.done == env.done
        expected = (0.0, 0.0) if not env.done else tuple(env.terminal_rewards())
        assert result.rewards == expected
    return chosen, env.terminal_rewards()


def test_full_hand_has_ten_decisions_conserves_cards_and_is_zero_sum() -> None:
    env = HuRlReferenceEnv(ALL_CARDS)
    selected: list[ActionKey] = []

    for decision in range(10):
        view = env.observe()
        expected_street = f"T{decision // 2}"
        expected_seat = "first" if decision % 2 == 0 else "second"
        assert view.observation.street == expected_street
        assert view.observation.seat == expected_seat
        assert view.observation.to_act_order == expected_seat
        assert len(view.public_history) == decision
        assert env.legal_actions() == env.legal_mapping().action_keys

        key = _choose_middle_key(env)
        selected.append(key)
        result = env.step(key)
        assert type(result.public_placement) is PublicPlacement
        assert result.actor == decision % 2
        assert result.street == expected_street

    assert env.done
    assert env.decision_count == 10
    assert tuple(board.card_count() for board in env.boards) == (13, 13)
    assert len(env.public_history) == 10
    assert sum(event.placement_mask.bit_count() for event in env.public_history) == 26
    assert sum(event.discard_count for event in env.public_history) == 8

    placed = {card for board in env.boards for card in board.all_cards()}
    discarded = {card for key in selected for card in key.cards("discards")}
    assert len(placed) == 26
    assert len(discarded) == 8
    assert placed.isdisjoint(discarded)
    assert placed | discarded == set(ALL_CARDS[:34])

    rewards = env.terminal_rewards()
    expected, _ = terminal_score(env.boards[0], env.boards[1], fl_ev=DEFAULT_FL_EV)
    assert rewards == [expected, -expected]
    assert rewards[0] + rewards[1] == 0.0
    with pytest.raises(HuRlReferenceError, match="already terminal"):
        env.observe()


def test_illegal_action_key_and_wrong_type_fail_without_fallback_or_mutation() -> None:
    env = HuRlReferenceEnv(ALL_CARDS)
    before = env.snapshot()

    with pytest.raises(HuRlReferenceError, match="not legal"):
        env.step(ActionKey())
    assert env.snapshot() == before

    with pytest.raises(TypeError, match="requires an ActionKey"):
        env.step(env.legal_actions()[0].to_token())  # type: ignore[arg-type]
    assert env.snapshot() == before


def test_snapshot_restore_replays_identical_views_actions_and_rewards() -> None:
    env = HuRlReferenceEnv(tuple(reversed(ALL_CARDS)))
    for _ in range(4):
        env.step(_choose_middle_key(env))

    checkpoint = env.snapshot()
    assert type(checkpoint) is HuRlReferenceSnapshot
    assert "hidden_state=<redacted>" in repr(checkpoint)
    expected_view_digest = env.observe().digest()
    expected_mapping = env.legal_mapping()
    expected_keys, expected_rewards = _play_to_terminal(env)

    env.restore(checkpoint)
    assert env.observe().digest() == expected_view_digest
    assert env.legal_mapping() == expected_mapping
    replayed_keys, replayed_rewards = _play_to_terminal(env)

    assert replayed_keys == expected_keys
    assert replayed_rewards == expected_rewards


def test_restore_rejects_cross_actor_private_discard_swap_and_rolls_back() -> None:
    env = HuRlReferenceEnv(ALL_CARDS)
    for _ in range(4):
        env.step(_choose_middle_key(env))

    checkpoint = env.snapshot()
    assert checkpoint._private_discards[0] != checkpoint._private_discards[1]
    swapped = replace(
        checkpoint,
        _private_discards=(
            checkpoint._private_discards[1],
            checkpoint._private_discards[0],
        ),
    )
    with pytest.raises(HuRlReferenceError, match="per-decision deals"):
        env.restore(swapped)
    assert env.snapshot() == checkpoint

    extra_lane = replace(
        checkpoint,
        _private_discards=(
            checkpoint._private_discards[0],
            checkpoint._private_discards[1],
            (),
        ),
    )
    with pytest.raises(HuRlReferenceError, match="exactly two private-discard"):
        env.restore(extra_lane)
    assert env.snapshot() == checkpoint

    event = checkpoint._public_history[0]
    masks = list(event.placement_masks)
    source = next(index for index, mask in enumerate(masks) if mask)
    target = (source + 1) % len(masks)
    moved_bit = masks[source] & -masks[source]
    masks[source] ^= moved_bit
    masks[target] |= moved_bit
    wrong_row_event = replace(
        event,
        top_placement_mask=masks[0],
        middle_placement_mask=masks[1],
        bottom_placement_mask=masks[2],
    )
    wrong_row = replace(
        checkpoint,
        _public_history=(wrong_row_event, *checkpoint._public_history[1:]),
    )
    with pytest.raises(HuRlReferenceError, match="board row"):
        env.restore(wrong_row)
    assert env.snapshot() == checkpoint

    invalid_board = replace(
        checkpoint,
        _boards=(
            Board(
                top=(*checkpoint._boards[0].top, "As", "Ks", "Qs", "Js"),
                middle=checkpoint._boards[0].middle,
                bottom=checkpoint._boards[0].bottom,
            ),
            checkpoint._boards[1],
        ),
    )
    with pytest.raises(HuRlReferenceError, match="invalid board state"):
        env.restore(invalid_board)
    assert env.snapshot() == checkpoint


def test_actor_view_hides_opponent_discard_and_deck_tail() -> None:
    env = HuRlReferenceEnv(ALL_CARDS)
    env.step(_choose_middle_key(env))  # T0 first
    env.step(_choose_middle_key(env))  # T0 second
    first_t1 = _choose_middle_key(env)
    opponent_private_discard = first_t1.cards("discards")[0]
    env.step(first_t1)  # T1 first; T1 second is now the actor.

    view = env.observe()
    payload = view.to_dict()
    encoded = view.canonical_json()
    assert opponent_private_discard not in encoded
    assert all(type(event) is PublicPlacement for event in view.public_history)
    assert set(payload) == {
        "schema",
        "observation",
        "public_history",
        "legal_action_mapping",
    }
    assert all(
        set(event) == {
            "schema",
            "street",
            "acting_seat",
            "top_placement_mask",
            "middle_placement_mask",
            "bottom_placement_mask",
            "discard_count",
        }
        for event in payload["public_history"]
    )
    forbidden = {
        "deck",
        "deck_tail",
        "remaining_deck",
        "world_state",
        "opponent_private_discard",
        "opponent_private_discards",
        "discard_mask",
    }
    assert forbidden.isdisjoint(_all_mapping_keys(payload))

    # The same card is visible when its owner acts again, but only through that
    # actor's own private-discard field.
    env.step(_choose_middle_key(env))
    next_first_view = env.observe()
    assert opponent_private_discard in next_first_view.observation.hero_private_discards


def test_action_keys_and_actor_views_ignore_card_order_within_each_deal() -> None:
    original = tuple(ALL_CARDS)
    permuted = list(original)
    starts = (0, 5, 10, 13, 16, 19, 22, 25, 28, 31)
    sizes = (5, 5, 3, 3, 3, 3, 3, 3, 3, 3)
    for start, size in zip(starts, sizes):
        permuted[start : start + size] = reversed(permuted[start : start + size])

    left = HuRlReferenceEnv(original)
    right = HuRlReferenceEnv(tuple(permuted))
    while not left.done:
        assert left.observe().digest() == right.observe().digest()
        assert left.legal_mapping() == right.legal_mapping()
        selected = _choose_middle_key(left)
        assert selected in right.legal_actions()
        left.step(selected)
        right.step(selected)

    assert left.terminal_rewards() == right.terminal_rewards()
    for left_board, right_board in zip(left.boards, right.boards):
        assert set(left_board.top) == set(right_board.top)
        assert set(left_board.middle) == set(right_board.middle)
        assert set(left_board.bottom) == set(right_board.bottom)


def test_reset_reuses_or_replaces_only_complete_explicit_decks() -> None:
    env = HuRlReferenceEnv(ALL_CARDS)
    env.step(_choose_middle_key(env))
    reset_view = env.reset()
    assert env.decision_count == 0
    assert reset_view == env.observe()

    reversed_view = env.reset(tuple(reversed(ALL_CARDS)))
    assert reversed_view.observation.dealt_cards == tuple(reversed(ALL_CARDS))[:5]

    with pytest.raises(HuRlReferenceError, match="exactly 52"):
        env.reset(ALL_CARDS[:-1])
    with pytest.raises(HuRlReferenceError, match="duplicate"):
        env.reset((*ALL_CARDS[:-1], ALL_CARDS[0]))
    with pytest.raises(TypeError, match="card sequence"):
        HuRlReferenceEnv("not-a-deck")  # type: ignore[arg-type]


def test_fl_bootstrap_is_current_normal_hand_contract() -> None:
    env = HuRlReferenceEnv(ALL_CARDS)
    view = env.observe()
    assert view.observation.street == "T0"
    assert view.observation.scoring.fantasyland_cards == 14
    assert dict(view.observation.scoring.fl_ev) == DEFAULT_FL_EV
    assert DEFAULT_FL_EV[14] == 9.6


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
