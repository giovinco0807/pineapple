"""Showing a 3-max decision to the frozen heads-up engine.

The bridge is only sound if every 3-max decision maps onto a heads-up sub-game
with IDENTICAL card counts -- the heads-up geometry table is fail-closed, so a
mapping that is off by a card does not silently degrade, it refuses.  These
tests check the mapping against the shipped heads-up table rather than against
a copy of it.
"""

from __future__ import annotations

import random

import pytest

from ofc_regular.action_space import generate_actions
from ofc_regular.cards import create_deck
from ofc_regular.hu_infoset import _REGULAR_DECISION_GEOMETRY
from ofc_regular.three_max import ACT_ORDER, SEAT_BB, SEAT_BTN, SEAT_SB, WorldState3
from ofc_regular.three_max.hu_bridge import (
    _SHOWN_OPPONENT,
    available,
    decide,
    heads_up_view,
    hu_policy,
)
from ofc_regular.three_max.mc import mc_policy

requires_engine = pytest.mark.skipif(
    not available(), reason="heads-up m3 engine or its pinned weights are unavailable"
)


def _world_before_btn_t3(seed: int) -> WorldState3:
    policy = mc_policy(sims=2)
    world = WorldState3.new_hand(create_deck(shuffle=True, rng=random.Random(seed)))
    while world.current_slot().decision_index < 11:
        world = world.apply(policy(world.observe(), seed * 17 + world.decision_index))
    return world


def test_every_seat_maps_to_a_heads_up_act_order():
    assert set(_SHOWN_OPPONENT) == set(ACT_ORDER)
    # The opener has nobody behind it, so it is shown as heads-up first; the
    # other two have just watched somebody act, so they are shown as second.
    assert _SHOWN_OPPONENT[SEAT_SB][1] == "first"
    assert _SHOWN_OPPONENT[SEAT_BB][1] == "second"
    assert _SHOWN_OPPONENT[SEAT_BTN][1] == "second"


def test_the_shown_opponent_is_the_most_recent_actor():
    world = WorldState3.new_hand(create_deck(shuffle=False))
    view = heads_up_view(world.observe())
    assert world.current_slot().seat == SEAT_SB
    # The SB opens, so the shown opponent is the player who answers next.
    assert view["shown_opponent_seat"] == SEAT_BB
    assert view["hidden_opponent_seat"] == SEAT_BTN


def test_card_counts_match_the_shipped_heads_up_table_at_every_decision():
    """The load-bearing check: a wrong mapping would be refused, not degraded."""
    for seed in (5, 61, 233):
        world = WorldState3.new_hand(create_deck(shuffle=True, rng=random.Random(seed)))
        policy = mc_policy(sims=2)
        seen = 0
        while world.current_slot().decision_index < 11:
            observation = world.observe()
            view = heads_up_view(observation)
            actual = (
                sum(len(cards) for cards in view["hero_board"].values()),
                sum(len(cards) for cards in view["opp_board"].values()),
                len(view["dealt"]),
                len(view["dead"]),
            )
            assert actual == _REGULAR_DECISION_GEOMETRY[
                (observation.street, view["position"])
            ], f"{observation.street}/{observation.seat} does not fit the HU table"
            seen += 1
            world = world.apply(
                policy(observation, seed * 17 + world.decision_index)
            )
        assert seen == 11


def test_the_hidden_opponent_is_never_shown():
    """One opponent is invisible to the model; that is the known approximation."""
    world = _world_before_btn_t3(seed=5)
    observation = world.observe()
    view = heads_up_view(observation)
    hidden_seat = view["hidden_opponent_seat"]
    hidden_cards = set(world.boards[hidden_seat].all_cards())
    shown = set()
    for cards in view["hero_board"].values():
        shown.update(cards)
    for cards in view["opp_board"].values():
        shown.update(cards)
    shown.update(view["dealt"])
    shown.update(view["dead"])
    assert hidden_cards and not (hidden_cards & shown)


@requires_engine
def test_the_engine_returns_a_legal_3_max_action():
    world = WorldState3.new_hand(create_deck(shuffle=True, rng=random.Random(5)))
    for _ in range(4):
        observation = world.observe()
        action = decide(observation)
        assert action in generate_actions(
            observation.hero_board, observation.dealt_cards
        )
        assert sorted(
            [card for card, _row in action.placements] + list(action.discards)
        ) == sorted(observation.dealt_cards)
        world = world.apply(action)


@requires_engine
def test_hu_policy_plays_every_pre_t3_decision():
    world = WorldState3.new_hand(create_deck(shuffle=True, rng=random.Random(61)))
    policy = hu_policy()
    while world.current_slot().decision_index < 11:
        world = world.apply(policy(world.observe(), 0))
    slot = world.current_slot()
    assert (slot.street, slot.seat) == ("T3", SEAT_BTN)
    for seat in ACT_ORDER:
        assert world.boards[seat].card_count() in (9, 11)


@requires_engine
def test_hu_roots_reach_fantasyland_far_more_often_than_referee_roots():
    """The reason the root generator changed, asserted rather than assumed.

    Measured over 180 finished boards: the Monte-Carlo referee enters
    Fantasyland on 1.1% of them and the heads-up models on 29.4%, against a
    heads-up production measurement of 24.5%.  A T3 corpus built on referee
    roots barely contains the race the street is partly deciding.
    """
    from ofc_regular.three_max import play_hand
    from ofc_regular.three_max.mc import _board_terminal_for_test as terminal

    def entry_rate(policy, hands: int) -> float:
        entries = 0
        boards = 0
        for seed in range(9_700_000, 9_700_000 + hands):
            result = play_hand(
                seed=seed, policies={seat: policy for seat in ACT_ORDER}
            )
            for seat in ACT_ORDER:
                entries += terminal(result.world.boards[seat]).fl_entry
                boards += 1
        return entries / boards

    assert entry_rate(hu_policy(), 8) > entry_rate(mc_policy(sims=4), 8)
