"""M0 acceptance tests for the 3-max core.

Every invariant here is traceable to docs/three_max_rules_contract_20260812.md.
"""

from __future__ import annotations

import json
import random

import pytest

from ofc_regular.action_space import Action
from ofc_regular.cards import ALL_CARDS, create_deck
from ofc_regular.state import Board
from ofc_regular.teacher import terminal_score
from ofc_regular.three_max import (
    ACT_ORDER,
    CARDS_DEALT_PER_HAND,
    DEAL_SCHEDULE,
    DECISIONS_PER_HAND,
    SEAT_BTN,
    SEAT_SB,
    SEAT_BB,
    SETTLEMENT_ORDER,
    STREETS,
    WorldState3,
    act_order_of,
    decision_seed,
    fl_ev_hand_total,
    geometry_for,
    load_fl_ev_3max,
    opponent_seats,
    pair_scores,
    placement_split,
    play_hand,
    play_session,
    player_of_seat,
    rotate_button,
    seat_of_player,
    settle_hand,
    uniform_random_policy,
)
from ofc_regular.three_max.scoring import DEFAULT_FL_EV_PER_PAIR

# Three disjoint complete boards: left and right are legal, the button fouls
# (trips on top over a high-card middle).
LEFT_BOARD = Board.from_rows(
    top=["2h", "3d", "4c"],
    middle=["5h", "6d", "7c", "8s", "Th"],
    bottom=["9h", "9d", "2s", "3s", "4s"],
)
RIGHT_BOARD = Board.from_rows(
    top=["5c", "7d", "8h"],
    middle=["9c", "Td", "Jh", "2c", "3c"],
    bottom=["4d", "4h", "6c", "6h", "8c"],
)
BUTTON_BOARD_FOULED = Board.from_rows(
    top=["Ah", "As", "Ad"],
    middle=["2d", "5d", "7h", "9s", "Js"],
    bottom=["Ks", "Qs", "3h", "6s", "Tc"],
)
FIXTURE_BOARDS = {
    SEAT_SB: LEFT_BOARD,
    SEAT_BB: RIGHT_BOARD,
    SEAT_BTN: BUTTON_BOARD_FOULED,
}

# A second fixture where one seat scoops both opponents AND enters Fantasyland.
# FIXTURE_BOARDS alone scores zero royalty and no FL entry everywhere, which
# makes every assertion on it invariant to the fl_ev table -- and the per-pair
# vs hand-total hazard invisible.
FL_BOARDS = {
    SEAT_SB: Board.from_rows(
        top=["Qh", "Qd", "2h"],
        middle=["Ks", "Kd", "3c", "4c", "5c"],
        bottom=["As", "Ad", "6c", "7d", "8h"],
    ),
    SEAT_BB: Board.from_rows(
        top=["2s", "3d", "4h"],
        middle=["5s", "6d", "7h", "8c", "Tc"],
        bottom=["9s", "9c", "Js", "Qs", "Kc"],
    ),
    SEAT_BTN: Board.from_rows(
        top=["3h", "5h", "6s"],
        middle=["7s", "8s", "Th", "Jh", "2c"],
        bottom=["Ac", "9h", "Td", "4d", "2d"],
    ),
}


def _seat_policies(policy):
    return {seat: policy for seat in ACT_ORDER}


# --- R1: deal schedule and deck conservation ---------------------------------


def test_deal_schedule_covers_fifteen_decisions_and_51_cards():
    assert len(DEAL_SCHEDULE) == DECISIONS_PER_HAND == 15
    assert [slot.decision_index for slot in DEAL_SCHEDULE] == list(range(15))
    assert sum(slot.size for slot in DEAL_SCHEDULE) == CARDS_DEALT_PER_HAND == 51

    offset = 0
    for slot in DEAL_SCHEDULE:
        assert slot.offset == offset
        offset += slot.size
        assert slot.size == (5 if slot.street == "T0" else 3)


def test_act_order_is_left_right_button_on_every_street():
    """R2: the button's left neighbour opens and the button closes."""
    assert ACT_ORDER == (SEAT_SB, SEAT_BB, SEAT_BTN)
    for street in STREETS:
        seats = [slot.seat for slot in DEAL_SCHEDULE if slot.street == street]
        assert seats == [SEAT_SB, SEAT_BB, SEAT_BTN]


def test_played_hand_conserves_the_deck():
    result = play_hand(seed=7, policies=_seat_policies(uniform_random_policy))
    world = result.world

    held: list[str] = []
    for seat in ACT_ORDER:
        board = world.boards[seat]
        assert board.is_complete()
        held.extend(board.all_cards())
        held.extend(world.private_discards[seat])
        assert len(world.private_discards[seat]) == 4

    assert len(held) == CARDS_DEALT_PER_HAND
    assert len(set(held)) == CARDS_DEALT_PER_HAND
    assert len(world.undealt_cards) == 1
    assert set(held) | set(world.undealt_cards) == set(ALL_CARDS)


# --- R2/R3: seating, rotation, act-relative opponents -------------------------


def test_seat_and_button_rotation_round_trip():
    for button in range(3):
        seats = {player: seat_of_player(player, button) for player in range(3)}
        assert set(seats.values()) == set(ACT_ORDER)
        assert seats[(button + 1) % 3] == SEAT_SB
        assert seats[(button + 2) % 3] == SEAT_BB
        assert seats[button] == SEAT_BTN
        for player, seat in seats.items():
            assert player_of_seat(seat, button) == player

    assert [rotate_button(0), rotate_button(1), rotate_button(2)] == [1, 2, 0]


def test_opponents_are_ordered_by_who_acts_next():
    assert opponent_seats(SEAT_SB) == (SEAT_BB, SEAT_BTN)
    assert opponent_seats(SEAT_BB) == (SEAT_BTN, SEAT_SB)
    assert opponent_seats(SEAT_BTN) == (SEAT_SB, SEAT_BB)


def test_session_rotates_the_button_clockwise():
    hands = play_session(
        base_seed=101,
        policies_by_player={p: uniform_random_policy for p in range(3)},
        hands=4,
        button_player=0,
    )
    assert [hand.button_player for hand in hands] == [0, 1, 2, 0]


# --- Decision geometry over a real hand --------------------------------------


def test_every_decision_matches_the_geometry_table():
    world = WorldState3.new_hand(create_deck(shuffle=True, rng=random.Random(3)))
    seen: list[tuple[str, int]] = []
    while not world.is_terminal:
        slot = world.current_slot()
        observation = world.observe()  # __post_init__ enforces the geometry
        expected = geometry_for(slot.street, slot.act_order)
        assert (
            observation.hero_board.card_count(),
            tuple(board.card_count() for board in observation.opponent_boards),
            len(observation.dealt_cards),
            len(observation.hero_private_discards),
        ) == expected
        assert observation.opponent_seats == opponent_seats(slot.seat)
        seen.append((slot.street, slot.act_order))
        world = world.apply(uniform_random_policy(observation, 12345 + slot.decision_index))
    assert len(seen) == 15


def test_unknown_card_counts_match_the_design_arithmetic():
    """The 3-max deck is 51/52 dealt, so late streets see a *smaller* unknown set."""
    expected = {
        ("T1", 0): 34,
        ("T3", 0): 20,
        ("T4", 0): 13,
        ("T4", 2): 9,
    }
    world = WorldState3.new_hand(create_deck(shuffle=True, rng=random.Random(11)))
    measured: dict[tuple[str, int], int] = {}
    while not world.is_terminal:
        slot = world.current_slot()
        observation = world.observe()
        measured[(slot.street, slot.act_order)] = observation.unknown_card_count()
        assert len(observation.unknown_cards()) == observation.unknown_card_count()
        world = world.apply(uniform_random_policy(observation, 999 + slot.decision_index))

    for key, count in expected.items():
        assert measured[key] == count, f"{key}: expected {count}, got {measured[key]}"


def test_observation_never_exposes_opponent_private_discards():
    world = WorldState3.new_hand(create_deck(shuffle=True, rng=random.Random(5)))
    while not world.is_terminal:
        slot = world.current_slot()
        observation = world.observe()
        visible = set(observation.visible_cards())
        for seat in ACT_ORDER:
            if seat == slot.seat:
                continue
            hidden = set(world.private_discards[seat])
            assert not (visible & hidden), f"{seat} discards leaked at {slot.street}"
        world = world.apply(uniform_random_policy(observation, 4242 + slot.decision_index))


# --- R4: settlement -----------------------------------------------------------


def test_pair_scores_equal_the_heads_up_kernel():
    """The 3-max pair term is the HU score, not a re-derivation of it."""
    scores = pair_scores(FIXTURE_BOARDS)
    for first, second in SETTLEMENT_ORDER:
        expected, _ = terminal_score(
            FIXTURE_BOARDS[first],
            FIXTURE_BOARDS[second],
            dict(DEFAULT_FL_EV_PER_PAIR),
        )
        assert scores[(first, second)] == pytest.approx(expected)


def test_infinite_stack_settlement_is_zero_sum_and_a_plain_pair_sum():
    settlement = settle_hand(FIXTURE_BOARDS)
    assert settlement.final_stacks is None
    assert sum(settlement.raw_totals.values()) == pytest.approx(0.0)

    # left loses the scoop to right, both collect the fouled button.
    assert settlement.raw_totals[SEAT_SB] == pytest.approx(0.0)
    assert settlement.raw_totals[SEAT_BB] == pytest.approx(12.0)
    assert settlement.raw_totals[SEAT_BTN] == pytest.approx(-12.0)
    assert settlement.transferred_totals == settlement.raw_totals
    assert not settlement.any_capped


def test_settlement_order_is_irrelevant_without_stacks():
    scores = pair_scores(FIXTURE_BOARDS)
    totals = {seat: 0.0 for seat in ACT_ORDER}
    for (first, second), value in sorted(scores.items(), reverse=True):
        totals[first] += value
        totals[second] -= value
    settlement = settle_hand(FIXTURE_BOARDS)
    for seat in ACT_ORDER:
        assert totals[seat] == pytest.approx(settlement.raw_totals[seat])


def test_finite_stacks_cap_later_pairs_in_contract_order():
    """The owner's worked example: a button emptied by left pays right nothing."""
    settlement = settle_hand(
        FIXTURE_BOARDS, stacks={SEAT_SB: 100.0, SEAT_BB: 100.0, SEAT_BTN: 4.0}
    )
    by_pair = {pair.seats: pair for pair in settlement.pairs}

    assert [pair.seats for pair in settlement.pairs] == list(SETTLEMENT_ORDER)
    assert by_pair[(SEAT_SB, SEAT_BB)].transferred == pytest.approx(-6.0)
    assert by_pair[(SEAT_SB, SEAT_BTN)].raw == pytest.approx(6.0)
    assert by_pair[(SEAT_SB, SEAT_BTN)].transferred == pytest.approx(4.0)
    assert by_pair[(SEAT_BB, SEAT_BTN)].raw == pytest.approx(6.0)
    assert by_pair[(SEAT_BB, SEAT_BTN)].transferred == pytest.approx(0.0)

    final = settlement.final_stacks
    assert final is not None
    assert final[SEAT_SB] == pytest.approx(98.0)
    assert final[SEAT_BB] == pytest.approx(106.0)
    assert final[SEAT_BTN] == pytest.approx(0.0)
    assert sum(final.values()) == pytest.approx(204.0)
    assert all(value >= 0 for value in final.values())
    assert settlement.any_capped


def test_fantasyland_entry_is_worth_the_per_pair_value_in_each_pair():
    """The 9.6 is per pair; feeding the 19.2 hand total here would double it."""
    settlement = settle_hand(FL_BOARDS, fl_ev_per_pair={14: 9.6})
    scores = pair_scores(FL_BOARDS, {14: 9.6})

    # left scoops both opponents (6) and carries 7 royalty, so each pair is
    # 6 + 7 + 9.6 = 22.6 -- the Fantasyland term enters once per pair.
    assert scores[(SEAT_SB, SEAT_BB)] == pytest.approx(22.6)
    assert scores[(SEAT_SB, SEAT_BTN)] == pytest.approx(22.6)
    assert scores[(SEAT_BB, SEAT_BTN)] == pytest.approx(-1.0)
    assert settlement.raw_totals[SEAT_SB] == pytest.approx(45.2)
    assert sum(settlement.raw_totals.values()) == pytest.approx(0.0)

    # The hand-level worth of that entry is 2 x 9.6, visible as the gap
    # between an fl_ev of 0 and the pinned constant.
    without_fl = settle_hand(FL_BOARDS, fl_ev_per_pair={14: 0.0})
    assert without_fl.raw_totals[SEAT_SB] == pytest.approx(26.0)
    assert settlement.raw_totals[SEAT_SB] - without_fl.raw_totals[
        SEAT_SB
    ] == pytest.approx(fl_ev_hand_total({14: 9.6})[14])

    # The 2x error the config guards against is loud here.
    doubled = settle_hand(FL_BOARDS, fl_ev_per_pair={14: 19.2})
    assert doubled.raw_totals[SEAT_SB] == pytest.approx(64.4)


def test_capping_only_changes_the_finite_stack_view():
    settlement = settle_hand(
        FIXTURE_BOARDS, stacks={SEAT_SB: 100.0, SEAT_BB: 100.0, SEAT_BTN: 4.0}
    )
    infinite = settle_hand(FIXTURE_BOARDS)
    assert settlement.raw_totals == infinite.raw_totals
    assert settlement.transferred_totals != infinite.transferred_totals


def test_a_short_first_seat_caps_what_it_pays_the_second():
    """The other capping branch: the pair's *first* seat is the one who is short."""
    settlement = settle_hand(
        FIXTURE_BOARDS, stacks={SEAT_SB: 2.0, SEAT_BB: 100.0, SEAT_BTN: 100.0}
    )
    by_pair = {pair.seats: pair for pair in settlement.pairs}

    assert by_pair[(SEAT_SB, SEAT_BB)].raw == pytest.approx(-6.0)
    assert by_pair[(SEAT_SB, SEAT_BB)].transferred == pytest.approx(-2.0)

    final = settlement.final_stacks
    assert final is not None
    # left is emptied by right, then refilled by the fouled button.
    assert final[SEAT_SB] == pytest.approx(6.0)
    assert final[SEAT_BB] == pytest.approx(108.0)
    assert final[SEAT_BTN] == pytest.approx(88.0)
    assert sum(final.values()) == pytest.approx(202.0)


def test_capped_settlement_still_conserves_chips():
    settlement = settle_hand(
        FIXTURE_BOARDS, stacks={SEAT_SB: 100.0, SEAT_BB: 100.0, SEAT_BTN: 4.0}
    )
    assert settlement.any_capped
    assert sum(settlement.transferred_totals.values()) == pytest.approx(0.0)


def test_session_conserves_chips_across_hands():
    hands = play_session(
        base_seed=555,
        policies_by_player={p: uniform_random_policy for p in range(3)},
        hands=6,
        button_player=0,
        starting_stacks={0: 200.0, 1: 200.0, 2: 200.0},
    )
    for hand in hands:
        assert hand.stacks_after is not None
        assert sum(hand.stacks_after.values()) == pytest.approx(600.0)
        assert all(value >= 0 for value in hand.stacks_after.values())


def test_session_gives_each_player_its_own_seat_and_its_own_chips():
    """Identical policies and equal stacks cannot tell a rotation bug from a fix."""
    seen: list[tuple[int, str]] = []

    def recording(player: int):
        def policy(observation, seed):
            seen.append((player, observation.seat))
            return uniform_random_policy(observation, seed)

        return policy

    starting = {0: 100.0, 1: 200.0, 2: 300.0}
    hands = play_session(
        base_seed=880,
        policies_by_player={p: recording(p) for p in range(3)},
        hands=5,
        button_player=0,
        starting_stacks=starting,
    )

    before = dict(starting)
    for hand in hands:
        # Each policy was invoked under exactly the seat the button says it holds.
        for player in range(3):
            expected_seat = seat_of_player(player, hand.button_player)
            assert hand.seats[player] == expected_seat
        hand_calls = seen[: 3 * 5]
        del seen[: 3 * 5]
        for player, seat in hand_calls:
            assert seat == hand.seats[player]

        # Chips land on the player who held that seat, not on the seat index.
        assert hand.stacks_after is not None
        for player in range(3):
            moved = hand.result.settlement.transferred_totals[hand.seats[player]]
            assert hand.stacks_after[player] == pytest.approx(before[player] + moved)
        before = dict(hand.stacks_after)


# --- R6: the Fantasyland constant is per-pair --------------------------------


def test_bootstrap_fl_ev_is_per_pair_with_a_matching_hand_total():
    per_pair = load_fl_ev_3max()
    assert per_pair[14] == pytest.approx(9.6)
    assert fl_ev_hand_total(per_pair)[14] == pytest.approx(19.2)


def test_loader_rejects_a_hand_total_that_contradicts_the_per_pair_value(tmp_path):
    """19.2 fed to the per-pair kernel would double every Fantasyland term."""
    bad = tmp_path / "fl_ev_bad.json"
    bad.write_text(
        json.dumps(
            {
                "fl_ev": {"14": 19.2},
                "fl_ev_units": "per_pair",
                "opponents_per_player": 2,
                "fl_ev_hand_total": {"14": 19.2},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="contradicts"):
        load_fl_ev_3max(bad)


def test_loader_rejects_hand_total_units(tmp_path):
    bad = tmp_path / "fl_ev_units.json"
    bad.write_text(
        json.dumps({"fl_ev": {"14": 19.2}, "fl_ev_units": "hand_total"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="per-pair"):
        load_fl_ev_3max(bad)


# --- Determinism and fail-closed state ---------------------------------------


def test_same_seed_reproduces_the_same_hand():
    first = play_hand(seed=2026, policies=_seat_policies(uniform_random_policy))
    second = play_hand(seed=2026, policies=_seat_policies(uniform_random_policy))
    assert first.world.boards == second.world.boards
    assert first.world.private_discards == second.world.private_discards
    assert first.settlement.raw_totals == second.settlement.raw_totals


def test_decision_seeds_are_distinct_per_seat_street_and_decision():
    world = WorldState3.new_hand(create_deck(shuffle=True, rng=random.Random(19)))
    seeds = []
    while not world.is_terminal:
        slot = world.current_slot()
        seeds.append(
            decision_seed(
                base_seed=1,
                decision_index=slot.decision_index,
                seat=slot.seat,
                street=slot.street,
            )
        )
        world = world.apply(uniform_random_policy(world.observe(), 1))
    assert len(set(seeds)) == 15

    base = dict(base_seed=1, decision_index=0, seat=SEAT_SB, street="T0")
    assert decision_seed(**base) != decision_seed(**{**base, "seat": SEAT_BB})
    assert decision_seed(**base) != decision_seed(**{**base, "street": "T1"})
    assert decision_seed(**base) != decision_seed(**{**base, "decision_index": 3})
    assert decision_seed(**base) != decision_seed(**{**base, "base_seed": 2})
    assert decision_seed(**base) == decision_seed(**base)


def test_play_hand_passes_the_fl_ev_table_through_to_settlement():
    """Contract R6's sensitivity probe runs entirely through this argument.

    Seed 947 is one where uniform-random play happens to leave a seat holding
    a Fantasyland entry, which is what makes the table observable at all.
    """
    policies = _seat_policies(uniform_random_policy)
    without = play_hand(seed=947, policies=policies, fl_ev_per_pair={14: 0.0})
    with_fl = play_hand(seed=947, policies=policies, fl_ev_per_pair={14: 100.0})

    assert without.world.boards == with_fl.world.boards
    assert dict(without.settlement.raw_totals) != dict(with_fl.settlement.raw_totals)
    assert without.settlement.raw_totals[SEAT_BB] == pytest.approx(26.0)
    assert with_fl.settlement.raw_totals[SEAT_BB] == pytest.approx(226.0)


def test_apply_enforces_the_place_and_discard_split():
    """R1: T0 places five and discards none; every later street places two of three."""
    world = WorldState3.new_hand(create_deck(shuffle=True, rng=random.Random(23)))
    for _ in range(3):  # finish T0 legally
        world = world.apply(uniform_random_policy(world.observe(), 1))

    dealt = world.dealt_cards()
    all_three = Action(
        placements=tuple((card, "bottom") for card in dealt), discards=()
    )
    with pytest.raises(ValueError, match="must place 2 and discard 1"):
        world.apply(all_three)

    one_only = Action(placements=((dealt[0], "bottom"),), discards=dealt[1:])
    with pytest.raises(ValueError, match="must place 2 and discard 1"):
        world.apply(one_only)


def test_apply_rejects_a_discard_on_the_opening_street():
    world = WorldState3.new_hand(create_deck(shuffle=True, rng=random.Random(29)))
    dealt = world.dealt_cards()
    with pytest.raises(ValueError, match="must place 5 and discard 0"):
        world.apply(
            Action(
                placements=tuple((card, "bottom") for card in dealt[:4]),
                discards=(dealt[4],),
            )
        )


def test_world_state_is_hashable_and_immutable():
    world = WorldState3.new_hand(create_deck(shuffle=False))
    assert hash(world) == hash(WorldState3.new_hand(create_deck(shuffle=False)))
    assert {world}  # usable as a memo key

    with pytest.raises(TypeError):
        world.boards[SEAT_SB] = LEFT_BOARD  # type: ignore[index]


def test_world_rejects_a_board_with_an_over_capacity_row():
    deck = create_deck(shuffle=False)
    with pytest.raises(ValueError, match="exceeds capacity"):
        WorldState3(
            deck=tuple(deck),
            boards={
                SEAT_SB: Board(top=tuple(deck[0:5])),
                SEAT_BB: Board(top=tuple(deck[5:8]), middle=tuple(deck[8:10])),
                SEAT_BTN: Board(top=tuple(deck[10:13]), middle=tuple(deck[13:15])),
            },
            private_discards={seat: () for seat in ACT_ORDER},
            decision_index=3,
        )


def test_apply_rejects_an_action_that_does_not_use_the_dealt_cards():
    world = WorldState3.new_hand(create_deck(shuffle=False))
    observation = world.observe()
    legal = uniform_random_policy(observation, 1)
    foreign = tuple(
        (card, row)
        for card, row in zip(
            [c for c in ALL_CARDS if c not in observation.dealt_cards][:5],
            [row for _card, row in legal.placements],
        )
    )
    with pytest.raises(ValueError, match="must use exactly the dealt"):
        world.apply(Action(placements=foreign))


def test_world_rejects_a_short_deck():
    with pytest.raises(ValueError, match="52-card deck"):
        WorldState3.new_hand(ALL_CARDS[:40])


def test_settlement_rejects_incomplete_and_overlapping_boards():
    partial = dict(FIXTURE_BOARDS)
    partial[SEAT_SB] = Board.from_rows(top=["2h"])
    with pytest.raises(ValueError, match="not complete"):
        settle_hand(partial)

    overlapping = dict(FIXTURE_BOARDS)
    overlapping[SEAT_BTN] = LEFT_BOARD
    with pytest.raises(ValueError, match="overlap"):
        settle_hand(overlapping)


def test_settlement_rejects_missing_and_unknown_seats():
    with pytest.raises(ValueError, match="missing 3-max"):
        settle_hand({SEAT_SB: LEFT_BOARD, SEAT_BB: RIGHT_BOARD})
    with pytest.raises(ValueError, match="unknown 3-max"):
        settle_hand({**FIXTURE_BOARDS, "dealer": LEFT_BOARD})


def test_stack_validation_is_fail_closed():
    with pytest.raises(ValueError, match="missing stacks"):
        settle_hand(FIXTURE_BOARDS, stacks={SEAT_SB: 10.0, SEAT_BB: 10.0})
    with pytest.raises(ValueError, match="unknown 3-max"):
        settle_hand(
            FIXTURE_BOARDS,
            stacks={SEAT_SB: 10.0, SEAT_BB: 10.0, SEAT_BTN: 10.0, "x": 1.0},
        )
    with pytest.raises(ValueError, match="non-negative"):
        settle_hand(
            FIXTURE_BOARDS,
            stacks={SEAT_SB: -1.0, SEAT_BB: 10.0, SEAT_BTN: 10.0},
        )


def test_seating_helpers_are_fail_closed():
    with pytest.raises(ValueError, match="unknown 3-max seat"):
        act_order_of("dealer")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no 3-max decision geometry"):
        geometry_for("T5", 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown 3-max street"):
        placement_split("T9")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="out of range"):
        seat_of_player(3, 0)
    with pytest.raises(TypeError, match="must be an integer"):
        rotate_button(True)  # type: ignore[arg-type]


def test_play_helpers_reject_incomplete_player_sets():
    with pytest.raises(ValueError, match="missing 3-max policies for seats"):
        play_hand(seed=1, policies={SEAT_SB: uniform_random_policy})
    with pytest.raises(ValueError, match="missing 3-max policies for players"):
        play_session(
            base_seed=1,
            policies_by_player={0: uniform_random_policy},
            hands=1,
        )
    with pytest.raises(ValueError, match="hands must be non-negative"):
        play_session(
            base_seed=1,
            policies_by_player={p: uniform_random_policy for p in range(3)},
            hands=-1,
        )
    with pytest.raises(ValueError, match="starting_stacks must cover"):
        play_session(
            base_seed=1,
            policies_by_player={p: uniform_random_policy for p in range(3)},
            hands=1,
            starting_stacks={0: 100.0},
        )


def test_placement_split_matches_the_geometry_table():
    """The split and the geometry table must agree; they are separate constants."""
    for street_index, street in enumerate(STREETS):
        place_count, discard_count = placement_split(street)
        hero_now, _opponents, dealt, discards_now = geometry_for(street, 0)
        assert place_count + discard_count == dealt
        assert discards_now == sum(
            placement_split(earlier)[1] for earlier in STREETS[:street_index]
        )
        assert hero_now == sum(
            placement_split(earlier)[0] for earlier in STREETS[:street_index]
        )
