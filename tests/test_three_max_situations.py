"""The six 3-max Fantasyland situations, and the reductions they permit.

The load-bearing claim is that a Fantasyland opponent shows nothing, so the
hero's unseen-card arithmetic in ``NNF`` is heads-up normal play exactly and in
``NFF`` is heads-up vs-Fantasyland play exactly.  These tests compute both
sides independently and compare.
"""

from __future__ import annotations

import pytest

from ofc_regular.three_max.seating import (
    ACT_ORDER,
    OPENING_DEAL_SIZE,
    SEAT_BTN,
    SEAT_SB,
    SEAT_BB,
    STREETS,
    TURN_DEAL_SIZE,
)
from ofc_regular.three_max.situations import (
    DECK_SIZE,
    FANTASYLAND_CARDS,
    NORMAL_CARDS_PER_HAND,
    TableConfig,
    all_configs,
    deal_plan,
    hero_unseen_count,
    reduction_of,
    situation_summary,
    street_decision_count,
    street_geometry,
)

ALL_NORMAL = TableConfig.all_normal()
ONE_FL = TableConfig.of(SEAT_BTN)
TWO_FL = TableConfig.of(SEAT_BB, SEAT_BTN)
ALL_FL = TableConfig.of(*ACT_ORDER)


def _heads_up_unseen(street_index: int, *, opponent_visible: bool) -> int:
    """Unseen count for a heads-up hero acting SECOND on this street.

    Computed from heads-up rules alone, with no reference to the 3-max code:
    the hero holds its board plus discards plus the current deal, and sees the
    opponent's board only if the opponent is playing a normal hand.
    """
    hero_cards = 0 if street_index == 0 else 5 + 2 * (street_index - 1)
    hero_discards = 0 if street_index == 0 else street_index - 1
    dealt = OPENING_DEAL_SIZE if street_index == 0 else TURN_DEAL_SIZE
    # The opponent acted first this street, so it is ahead by that street's
    # placement count: five on the opening street, two afterwards.
    ahead = OPENING_DEAL_SIZE if street_index == 0 else 2
    opponent = (hero_cards + ahead) if opponent_visible else 0
    return DECK_SIZE - (hero_cards + hero_discards + dealt + opponent)


# --- The six situations ------------------------------------------------------


def test_every_configuration_fits_in_one_deck():
    configs = list(all_configs())
    assert len(configs) == 8  # every subset of the three seats
    for config in configs:
        assert config.cards_dealt <= DECK_SIZE
        assert config.undealt_cards >= 0
        assert config.cards_dealt == (
            len(config.normal_seats) * NORMAL_CARDS_PER_HAND
            + len(config.fantasyland_seats) * FANTASYLAND_CARDS
        )

    # Eight masks give eight hero-relative labels (which opponent is in
    # Fantasyland matters, because opponent slots are ordered by act order),
    # but they collapse onto five decision families.
    assert {config.label(SEAT_SB) for config in configs} == {
        "NNN", "NNF", "NFN", "NFF", "FNN", "FNF", "FFN", "FFF",
    }
    assert {reduction_of(config, SEAT_SB) for config in configs} == {
        "three_max_normal",
        "heads_up_normal_shaped",
        "heads_up_vs_fl_shaped",
        "fantasyland_best_response",
        "fantasyland_blind",
    }


def test_card_budgets_match_the_contract():
    assert ALL_NORMAL.cards_dealt == 51 and ALL_NORMAL.undealt_cards == 1
    assert ONE_FL.cards_dealt == 48 and ONE_FL.undealt_cards == 4
    assert TWO_FL.cards_dealt == 45 and TWO_FL.undealt_cards == 7
    assert ALL_FL.cards_dealt == 42 and ALL_FL.undealt_cards == 10


def test_labels_are_hero_relative():
    assert ALL_NORMAL.label(SEAT_SB) == "NNN"
    # button in Fantasyland: left sees (right normal, button FL)
    assert ONE_FL.label(SEAT_SB) == "NNF"
    assert ONE_FL.label(SEAT_BB) == "NFN"
    assert ONE_FL.label(SEAT_BTN) == "FNN"
    assert TWO_FL.label(SEAT_SB) == "NFF"
    assert ALL_FL.label(SEAT_BTN) == "FFF"


def test_reductions_route_each_hero_to_the_right_family():
    assert reduction_of(ALL_NORMAL, SEAT_SB) == "three_max_normal"
    assert reduction_of(ONE_FL, SEAT_SB) == "heads_up_normal_shaped"
    assert reduction_of(ONE_FL, SEAT_BTN) == "fantasyland_best_response"
    assert reduction_of(TWO_FL, SEAT_SB) == "heads_up_vs_fl_shaped"
    assert reduction_of(TWO_FL, SEAT_BB) == "fantasyland_best_response"
    assert reduction_of(ALL_FL, SEAT_SB) == "fantasyland_blind"


def test_street_decision_counts_shrink_with_each_fantasyland_seat():
    assert street_decision_count(ALL_NORMAL) == 15
    assert street_decision_count(ONE_FL) == 10
    assert street_decision_count(TWO_FL) == 5
    assert street_decision_count(ALL_FL) == 0


# --- The reduction, in numbers ----------------------------------------------


def test_one_fantasyland_opponent_gives_the_hero_heads_up_normal_geometry():
    """NNF: the hero sees exactly one board, so the unseen counts are heads-up."""
    hero_act_index = ONE_FL.act_index_of(SEAT_BB)  # acts last among the normals
    for street_index, street in enumerate(STREETS):
        assert hero_unseen_count(ONE_FL, street, hero_act_index) == _heads_up_unseen(
            street_index, opponent_visible=True
        )


def test_nnf_and_nff_match_the_shipped_heads_up_geometry_table():
    """The strongest form of the claim: compare against the real HU table.

    ``_REGULAR_DECISION_GEOMETRY`` is the frozen heads-up contract, so agreeing
    with it (rather than with a formula written in this test file) is what
    licenses reusing the heads-up machinery for these two situations.
    """
    from ofc_regular.hu_infoset import _REGULAR_DECISION_GEOMETRY

    def hu_unseen(street: str, order: str, *, opponent_visible: bool = True) -> int:
        hero, opponent, dealt, discards = _REGULAR_DECISION_GEOMETRY[(street, order)]
        return DECK_SIZE - (
            hero + discards + dealt + (opponent if opponent_visible else 0)
        )

    for act_index, order in ((0, "first"), (1, "second")):
        for street in STREETS:
            assert hero_unseen_count(ONE_FL, street, act_index) == hu_unseen(
                street, order
            ), f"NNF diverges from heads-up normal at {street}/{order}"

    for street in STREETS:
        assert hero_unseen_count(TWO_FL, street, 0) == hu_unseen(
            street, "first", opponent_visible=False
        ), f"NFF diverges from heads-up vs-Fantasyland at {street}"


def test_two_fantasyland_opponents_give_the_hero_vs_fl_geometry():
    """NFF: no visible opponent at all, exactly as in heads-up vs-Fantasyland."""
    for street_index, street in enumerate(STREETS):
        assert hero_unseen_count(TWO_FL, street, 0) == _heads_up_unseen(
            street_index, opponent_visible=False
        )


def test_the_all_normal_table_is_the_only_one_that_is_not_heads_up_shaped():
    """NNN is the genuinely new problem: the hero sees two boards at once."""
    _hero, opponents, _dealt, _discards = street_geometry(ALL_NORMAL, "T3", 2)
    assert len([count for count in opponents if count > 0]) == 2
    assert hero_unseen_count(ALL_NORMAL, "T3", 2) == 16
    assert hero_unseen_count(ALL_NORMAL, "T3", 2) != _heads_up_unseen(
        3, opponent_visible=True
    )


def test_fantasyland_opponents_contribute_no_visible_cards():
    for act_index in range(len(ONE_FL.normal_seats)):
        _hero, opponents, _dealt, _discards = street_geometry(ONE_FL, "T2", act_index)
        assert 0 in opponents  # the Fantasyland seat


def test_all_normal_geometry_agrees_with_the_frozen_m0_table():
    """The situation module must reproduce the M0 table it generalises."""
    from ofc_regular.three_max.seating import DECISION_GEOMETRY

    for (street, act_order), expected in DECISION_GEOMETRY.items():
        hero, opponents, dealt, discards = street_geometry(
            ALL_NORMAL, street, act_order
        )
        assert (hero, opponents, dealt, discards) == expected


# --- The deal plan -----------------------------------------------------------


def test_fantasyland_hands_are_dealt_before_any_street():
    fantasyland, slots = deal_plan(TWO_FL)
    assert [deal.seat for deal in fantasyland] == [SEAT_BB, SEAT_BTN]
    assert [deal.offset for deal in fantasyland] == [0, 14]
    assert slots[0].offset == 28  # the lone normal seat opens after both FL hands
    assert slots[0].seat == SEAT_SB


def test_deal_plans_are_contiguous_and_exact_for_every_configuration():
    for config in all_configs():
        fantasyland, slots = deal_plan(config)
        offset = 0
        for deal in fantasyland:
            assert deal.offset == offset
            offset += deal.size
        for index, slot in enumerate(slots):
            assert slot.decision_index == index
            assert slot.offset == offset
            offset += slot.size
            assert slot.size == (5 if slot.street == "T0" else 3)
        assert offset == config.cards_dealt <= DECK_SIZE
        assert len(slots) == street_decision_count(config)


def test_summary_routes_a_fantasyland_hero_away_from_street_play():
    summary = situation_summary(ONE_FL, SEAT_BTN)
    assert summary["hero_acts"] is False
    assert summary["hero_act_index"] is None
    assert summary["reduction"] == "fantasyland_best_response"

    playing = situation_summary(ONE_FL, SEAT_SB)
    assert playing["hero_acts"] is True
    assert playing["hero_act_index"] == 0
    assert playing["street_decisions"] == 10


def test_act_index_refuses_a_fantasyland_seat():
    with pytest.raises(ValueError, match="plays no streets"):
        ONE_FL.act_index_of(SEAT_BTN)
    with pytest.raises(ValueError, match="out of range"):
        street_geometry(TWO_FL, "T1", 1)
    with pytest.raises(ValueError, match="unknown 3-max seats"):
        TableConfig.of("dealer")  # type: ignore[arg-type]
