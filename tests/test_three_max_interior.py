"""The middle seat's T3 label, and the interior node it has to fill.

Two things are being pinned here.  The generic round induction must reproduce
the hand-rolled one it generalises, bit for bit, on the case they overlap --
otherwise the BTN corpus and the BB corpus are quietly on different rules.  And
the middle-seat evaluator's stratified draws must partition the unseen set the
way a real hand does, because the interior policy is shown a deck and a wrong
deck is an invisible error: every board stays legal, every number stays
plausible, and the label is simply wrong.
"""

from __future__ import annotations

import random

import pytest

from ofc_regular.action_space import generate_actions
from ofc_regular.cards import ALL_CARDS, create_deck
from ofc_regular.three_max import WorldState3
from ofc_regular.three_max.exact import (
    _completing_terminals,
    _induced_hero_value,
    _resolve_t4_round,
    resolve_round_indices,
    round_hero_value,
)
from ofc_regular.three_max.interior import (
    MIDDLE_HERO_INDEX,
    STRATUM_CARDS,
    T4_ROUND_CARDS,
    _interior_observation,
    evaluate_t3_middle,
    exact_interior,
    mc_interior,
    random_interior,
    sample_middle_draws,
)
from ofc_regular.three_max.mc import (
    _board_terminal_for_test as terminal_of,
    _score_pair,
    mc_policy,
)

FL_EV_14 = 9.6
FL = {14: FL_EV_14}


def _root_at(seat: str, seed: int) -> WorldState3:
    policy = mc_policy(sims=2)
    world = WorldState3.new_hand(create_deck(shuffle=True, rng=random.Random(seed)))
    while True:
        slot = world.current_slot()
        if slot.street == "T3" and slot.seat == seat:
            return world
        world = world.apply(policy(world.observe(), seed * 31 + slot.decision_index))


def _round_options(seed: int, hero_seat: str):
    """Three completion lists in act order, from a real T3 position."""
    observation = _root_at(hero_seat, seed).observe()
    rng = random.Random(seed ^ 0xC0FFEE)
    unknown = list(observation.unknown_cards())
    cards = rng.sample(unknown, 9)
    return observation, cards


# --- The generic induction reproduces the one it generalises -----------------


def test_score_pair_is_antisymmetric():
    """The induction builds one matrix per unordered pair and negates it."""
    rng = random.Random(4)
    seen = 0
    for seed in (5, 19, 44):
        observation = _root_at("btn", seed).observe()
        unknown = list(observation.unknown_cards())
        pool = [
            terminal_of(
                observation.opponent_boards[index].place(
                    generate_actions(
                        observation.opponent_boards[index], tuple(rng.sample(unknown, 3))
                    )[0].placements
                )
            )
            for index in range(2)
            for _ in range(3)
        ]
        for left in pool:
            for right in pool:
                assert _score_pair(left, right, FL_EV_14) == pytest.approx(
                    -_score_pair(right, left, FL_EV_14)
                )
                seen += 1
    assert seen > 100


def test_generic_induction_matches_the_closing_seat_kernel():
    compared = 0
    for seed in (5, 19, 44, 77, 101, 136):
        observation = _root_at("btn", seed).observe()
        rng = random.Random(seed)
        unknown = list(observation.unknown_cards())
        for _ in range(4):
            cards = rng.sample(unknown, 9)
            first = _completing_terminals(observation.opponent_boards[0], cards[0:3])
            second = _completing_terminals(observation.opponent_boards[1], cards[3:6])
            for action in generate_actions(
                observation.hero_board, observation.dealt_cards
            )[:5]:
                placed = observation.hero_board.place(action.placements)
                hero = _completing_terminals(placed, cards[6:9])
                assert round_hero_value(
                    (first, second, hero), 2, FL_EV_14
                ) == _induced_hero_value(first, second, hero, FL_EV_14)
                compared += 1
    assert compared >= 100


def test_generic_induction_matches_the_recursive_solver_for_every_seat():
    """``_resolve_t4_round`` walks boards; this walks precomputed terminals."""
    compared = 0
    for seed in (12, 63, 128):
        observation = _root_at("bb", seed).observe()
        btn_board, sb_board = observation.opponent_boards
        rng = random.Random(seed)
        unknown = list(observation.unknown_cards())
        for _ in range(3):
            filled = btn_board.place(
                generate_actions(btn_board, tuple(rng.sample(unknown, 3)))[0].placements
            )
            cards = rng.sample(
                [card for card in unknown if card not in filled.all_cards()], 9
            )
            for action in generate_actions(
                observation.hero_board, observation.dealt_cards
            )[:4]:
                hero_after = observation.hero_board.place(action.placements)
                actors = [
                    (sb_board, cards[0:3]),
                    (hero_after, cards[3:6]),
                    (filled, cards[6:9]),
                ]
                chosen = _resolve_t4_round(actors, (), FL_EV_14)
                expected = sum(
                    _score_pair(chosen[MIDDLE_HERO_INDEX], chosen[index], FL_EV_14)
                    for index in (0, 2)
                )
                options = [
                    _completing_terminals(board, dealt) for board, dealt in actors
                ]
                assert round_hero_value(
                    options, MIDDLE_HERO_INDEX, FL_EV_14
                ) == pytest.approx(expected)
                picks = resolve_round_indices(options, FL_EV_14)
                assert len(picks) == 3
                assert all(0 <= pick < len(opts) for pick, opts in zip(picks, options))
                compared += 1
    assert compared >= 20


# --- Stratified draws must partition a real deck ----------------------------


def test_middle_root_geometry_is_the_contract_one():
    observation = _root_at("bb", 21).observe()
    btn_board, sb_board = observation.opponent_boards
    assert (btn_board.card_count(), sb_board.card_count()) == (9, 11)
    assert observation.unknown_card_count() == 18
    assert observation.unknown_card_count() == STRATUM_CARDS + T4_ROUND_CARDS + 1


def test_a_stratum_assigns_every_card_at_most_one_role():
    observation = _root_at("bb", 33).observe()
    unseen = observation.unknown_cards()
    draws = sample_middle_draws(unseen, strata=5, per_stratum=4, seed=9)
    assert len(draws) == 5
    for draw in draws:
        head = (*draw.btn_discards, *draw.sb_discards, *draw.btn_dealt)
        assert len(head) == STRATUM_CARDS
        assert len(set(head)) == STRATUM_CARDS
        assert set(head) <= set(unseen)
        for t4 in draw.t4_draws:
            assert len(t4) == T4_ROUND_CARDS
            assert len(set(t4)) == T4_ROUND_CARDS
            assert set(t4) <= set(unseen)
            # A card the stratum already spent cannot be dealt again.
            assert not set(t4) & set(head)


def test_sample_middle_draws_refuses_a_deck_that_cannot_cover_the_hand():
    with pytest.raises(ValueError):
        sample_middle_draws(ALL_CARDS[:10], strata=2, per_stratum=2, seed=1)


def test_the_interior_view_is_a_legal_button_observation():
    observation = _root_at("bb", 47).observe()
    btn_board, sb_board = observation.opponent_boards
    draw = sample_middle_draws(
        observation.unknown_cards(), strata=1, per_stratum=1, seed=3
    )[0]
    action = generate_actions(observation.hero_board, observation.dealt_cards)[0]
    interior = _interior_observation(
        btn_board=btn_board,
        sb_board=sb_board,
        hero_board_after=observation.hero_board.place(action.placements),
        draw=draw,
    )
    assert interior.seat == "btn"
    assert interior.street == "T3"
    # Act-relative: the SB answers first at T4, the hero second.
    assert interior.opponent_seats == ("sb", "bb")
    assert [board.card_count() for board in interior.opponent_boards] == [11, 11]
    # The BTN sees 36 cards, so 16 are unknown to it -- the NNN geometry.
    assert interior.unknown_card_count() == 16
    # Everything the T4 round will deal is still unknown to the interior actor.
    assert set(draw.t4_draws[0]) <= set(interior.unknown_cards())


# --- The middle-seat label ---------------------------------------------------


def test_evaluate_t3_middle_rejects_the_seats_it_is_not_for():
    with pytest.raises(ValueError):
        observation = _root_at("btn", 8).observe()
        evaluate_t3_middle(
            observation,
            interior_policy=random_interior(),
            draws=sample_middle_draws(
                observation.unknown_cards(), strata=1, per_stratum=1, seed=0
            ),
        )


def test_the_label_is_deterministic_and_covers_every_legal_action():
    observation = _root_at("bb", 55).observe()
    draws = sample_middle_draws(
        observation.unknown_cards(), strata=3, per_stratum=3, seed=2
    )
    policy = mc_interior(sims=2)
    first = evaluate_t3_middle(
        observation, interior_policy=policy, draws=draws, seed=7, fl_ev_per_pair=FL
    )
    again = evaluate_t3_middle(
        observation, interior_policy=policy, draws=draws, seed=7, fl_ev_per_pair=FL
    )
    legal = generate_actions(observation.hero_board, observation.dealt_cards)
    assert len(first) == len(legal)
    assert {candidate.action for candidate in first} == set(legal)
    assert [candidate.ev for candidate in first] == [
        candidate.ev for candidate in again
    ]
    assert all(candidate.samples == 9 for candidate in first)
    assert first == sorted(first, key=lambda item: -item.ev)


def test_two_interior_policies_that_agree_produce_the_same_label_to_the_bit():
    """Common random numbers: only the interior CHOICE may move the label."""
    observation = _root_at("bb", 66).observe()
    draws = sample_middle_draws(
        observation.unknown_cards(), strata=3, per_stratum=3, seed=5
    )
    recorded: list = []

    def recording(obs, seed):
        action = mc_interior(sims=2)(obs, seed)
        recorded.append(action)
        return action

    def replaying(obs, seed):
        return recorded.pop(0)

    live = evaluate_t3_middle(
        observation, interior_policy=recording, draws=draws, seed=4, fl_ev_per_pair=FL
    )
    replayed = evaluate_t3_middle(
        observation, interior_policy=replaying, draws=draws, seed=4, fl_ev_per_pair=FL
    )
    assert not recorded
    assert [candidate.ev for candidate in live] == [
        candidate.ev for candidate in replayed
    ]


def test_the_teacher_labels_the_middle_seat_and_says_who_stood_inside():
    """A BB corpus is defined by its interior policy, so the config carries it."""
    from ofc_regular.three_max.teacher import TeacherConfig, label_root, sample_root

    config = TeacherConfig(
        seat="bb", strata=2, per_stratum=2, interior="mc", interior_sims=2,
        root_policy="mc",
    )
    record = label_root(
        sample_root(9001, config=config).observe(), seed=9001, config=config
    )
    assert record["seat"] == "bb"
    assert record["street"] == "T3"
    assert record["samples"] == 4
    assert record["ev_holdout"] is not None
    assert len(record["actions"]) == len(
        generate_actions(
            sample_root(9001, config=config).observe().hero_board,
            sample_root(9001, config=config).observe().dealt_cards,
        )
    )
    # Two corpora that differ only in who played the interior node are
    # different datasets and must not share a fingerprint.
    other = TeacherConfig(
        seat="bb", strata=2, per_stratum=2, interior="exact", interior_samples=4,
        root_policy="mc",
    )
    assert config.fingerprint() != other.fingerprint()
    assert config.fingerprint() != TeacherConfig().fingerprint()


def test_the_encoder_refuses_the_middle_seat_instead_of_guessing():
    """The 225/261 encoder is a (T3, BTN) encoder; it must say so out loud.

    At (T3, BB) the BTN opponent still has four open slots, and the outlook
    blocks only know how to fill two.  Both completion helpers now refuse it --
    the one that did not used to fill the first two slots and hand an
    incomplete board to the evaluator.  A BB model needs a wider outlook first;
    until it exists this must fail, not encode something plausible.
    """
    from ofc_regular.three_max.features import encode_record_action
    from ofc_regular.three_max.teacher import TeacherConfig, label_root, sample_root

    config = TeacherConfig(
        seat="bb", strata=1, per_stratum=1, interior="mc", interior_sims=2,
        root_policy="mc",
    )
    record = label_root(
        sample_root(9002, config=config).observe(), seed=9002, config=config
    )
    with pytest.raises(ValueError, match="at most two open slots"):
        encode_record_action(record, 0)


def test_the_middle_teacher_refuses_a_model_it_was_not_given():
    from ofc_regular.three_max.teacher import TeacherConfig

    with pytest.raises(ValueError):
        TeacherConfig(seat="bb", interior="model", interior_model="")


def test_a_worse_interior_policy_moves_the_label():
    """The measurement must be able to see an interior node being played badly.

    Without this the probe cannot distinguish "the interior policy does not
    matter" from "the harness is not wired to the interior policy at all",
    which is the failure mode this track has already shipped twice.
    """
    moved = 0
    for seed in (13, 29, 58, 91):
        observation = _root_at("bb", seed).observe()
        draws = sample_middle_draws(
            observation.unknown_cards(), strata=4, per_stratum=4, seed=seed
        )
        good = evaluate_t3_middle(
            observation,
            interior_policy=exact_interior(samples=4),
            draws=draws,
            seed=seed,
            fl_ev_per_pair=FL,
        )
        bad = evaluate_t3_middle(
            observation,
            interior_policy=random_interior(),
            draws=draws,
            seed=seed,
            fl_ev_per_pair=FL,
        )
        by_action = {candidate.action: candidate.ev for candidate in good}
        moved += any(
            by_action[candidate.action] != candidate.ev for candidate in bad
        )
    assert moved == 4
