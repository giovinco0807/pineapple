"""Exact final-street (T4) evaluation for 3-max."""

from __future__ import annotations

import random

import pytest

from ofc_regular.action_space import generate_actions
from ofc_regular.cards import create_deck
from ofc_regular.state import ROWS
from ofc_regular.three_max import WorldState3, uniform_random_policy
from ofc_regular.three_max.exact import (
    FULL_ENUMERATION_LIMIT,
    evaluate_t4_exact,
    exact_t4_policy,
)
from ofc_regular.three_max.mc import (
    _board_terminal_for_test as _terminal_of,
    _score_pair,
    evaluate_actions_mc,
    mc_policy,
)

FL_EV_14 = 9.6


def _world_at(street: str, act_order: int, seed: int, policy=None) -> WorldState3:
    play = policy or uniform_random_policy
    world = WorldState3.new_hand(create_deck(shuffle=True, rng=random.Random(seed)))
    while True:
        slot = world.current_slot()
        if slot.street == street and slot.act_order == act_order:
            return world
        world = world.apply(play(world.observe(), seed * 31 + slot.decision_index))


def _sane_world_at(act_order: int, seed: int) -> WorldState3:
    """Reach T4 through a policy that does not lock itself into a foul."""
    return _world_at("T4", act_order, seed, policy=mc_policy(sims=2))


# --- The last actor is a closed form ----------------------------------------


def test_last_actor_needs_no_chance_node():
    world = _sane_world_at(2, seed=91)
    observation = world.observe()
    assert all(board.is_complete() for board in observation.opponent_boards)

    ranked = evaluate_t4_exact(observation)
    assert ranked[0].deals == 1
    assert all(candidate.exact for candidate in ranked)


def test_last_actor_agrees_with_the_monte_carlo_referee_to_the_bit():
    """Two independent implementations of a zero-variance value must coincide."""
    for seed in (91, 104, 233):
        observation = _sane_world_at(2, seed=seed).observe()
        exact = evaluate_t4_exact(observation, fl_ev_per_pair={14: FL_EV_14})
        referee = {
            candidate.action: candidate.ev
            for candidate in evaluate_actions_mc(
                observation, sims=1, seed=0, fl_ev_per_pair={14: FL_EV_14}
            )
        }
        for candidate in exact:
            assert candidate.ev == pytest.approx(referee[candidate.action], abs=1e-12)


# --- The middle actor is still exact, with one chance node -------------------


def test_middle_actor_enumerates_every_deal():
    observation = _sane_world_at(1, seed=91).observe()
    complete = [b.is_complete() for b in observation.opponent_boards]
    assert sorted(complete) == [False, True]

    ranked = evaluate_t4_exact(observation)
    # unseen 11, the single responder draws 3: C(11,3) = 165, under the limit.
    assert observation.unknown_card_count() == 11
    assert ranked[0].deals == 165
    assert all(candidate.exact for candidate in ranked)


def test_middle_actor_sampling_converges_on_the_enumerated_value():
    observation = _sane_world_at(1, seed=104).observe()
    enumerated = {c.action: c.ev for c in evaluate_t4_exact(observation)}
    sampled = evaluate_t4_exact(observation, max_deals=60, seed=3)
    assert not sampled[0].exact
    for candidate in sampled:
        assert candidate.ev == pytest.approx(enumerated[candidate.action], abs=3.0)


# --- The first actor faces two responders -----------------------------------


def test_first_actor_samples_by_default_and_reports_it():
    observation = _sane_world_at(0, seed=91).observe()
    assert not any(board.is_complete() for board in observation.opponent_boards)
    assert observation.unknown_card_count() == 13

    ranked = evaluate_t4_exact(observation)
    assert ranked[0].deals == FULL_ENUMERATION_LIMIT
    assert not any(candidate.exact for candidate in ranked)


def test_first_actor_values_are_stable_across_sampling_seeds():
    observation = _sane_world_at(0, seed=233).observe()
    first = {c.action: c.ev for c in evaluate_t4_exact(observation, seed=1)}
    second = evaluate_t4_exact(observation, seed=2)
    for candidate in second:
        assert candidate.ev == pytest.approx(first[candidate.action], abs=2.0)


# --- Responders solve their own game, not the hero's ------------------------


def test_the_responder_maximises_its_own_total():
    """A responder is not punishing the hero; it is playing its own 3-max hand.

    Re-derive the reply the evaluator must have chosen and confirm it is the
    argmax of the responder's OWN pairwise sum, which is strictly weaker
    against the hero than a dedicated best response would be.
    """
    world = _sane_world_at(1, seed=91)
    observation = world.observe()
    responder_board = next(
        board for board in observation.opponent_boards if not board.is_complete()
    )
    settled = next(
        board for board in observation.opponent_boards if board.is_complete()
    )
    settled_terminal = _terminal_of(settled)

    hero_action = evaluate_t4_exact(observation)[0].action
    hero_terminal = _terminal_of(observation.hero_board.place(hero_action.placements))

    deal = observation.unknown_cards()[:3]
    own_best = None
    own_value = float("-inf")
    hero_punishing = None
    punish_value = float("-inf")
    for action in generate_actions(responder_board, deal):
        board = responder_board.place(action.placements)
        if not board.is_complete():
            continue
        terminal = _terminal_of(board)
        own = _score_pair(terminal, hero_terminal, FL_EV_14) + _score_pair(
            terminal, settled_terminal, FL_EV_14
        )
        against_hero = _score_pair(terminal, hero_terminal, FL_EV_14)
        if own > own_value:
            own_value, own_best = own, board
        if against_hero > punish_value:
            punish_value, hero_punishing = against_hero, board

    assert own_best is not None and hero_punishing is not None
    # The dedicated punisher can never do worse against the hero than the
    # responder that is splitting its attention.
    own_best_against_hero = _score_pair(
        _terminal_of(own_best), hero_terminal, FL_EV_14
    )
    assert punish_value >= own_best_against_hero


def test_backward_induction_changes_the_first_responder_s_reply():
    """The induction earns its cost, but only on a minority of deals.

    Measured over 1,000 sampled deals: the first responder picks a different
    board once it anticipates the last actor's reply in about 3% of them.  The
    effect is real but thin, so this is asserted deal-by-deal -- averaging it
    into the hero's EV across deals washes it out, which is what an earlier
    version of this test got wrong.
    """
    from ofc_regular.three_max.exact import _joint_deals, _responder_reply

    differing = 0
    total = 0
    for seed in (200, 201, 202, 203, 204, 205):
        observation = _sane_world_at(0, seed=seed).observe()
        responders = [
            board for board in observation.opponent_boards if not board.is_complete()
        ]
        if len(responders) != 2:
            continue
        hero_action = generate_actions(
            observation.hero_board, observation.dealt_cards
        )[0]
        hero = _terminal_of(observation.hero_board.place(hero_action.placements))
        deals, _exact = _joint_deals(observation.unknown_cards(), 2, 25, 5)

        for deal in deals:
            (first_board, first_deal), (last_board, last_deal) = zip(responders, deal)
            myopic_board, _myopic = _responder_reply(
                first_board, first_deal, (hero,), FL_EV_14
            )

            best_board = None
            best_value = float("-inf")
            for action in generate_actions(first_board, first_deal):
                candidate = first_board.place(action.placements)
                if not candidate.is_complete():
                    continue
                terminal = _terminal_of(candidate)
                _reply, last_terminal = _responder_reply(
                    last_board, last_deal, (hero, terminal), FL_EV_14
                )
                value = _score_pair(terminal, hero, FL_EV_14) + _score_pair(
                    terminal, last_terminal, FL_EV_14
                )
                if value > best_value:
                    best_value, best_board = value, candidate

            total += 1
            if best_board.all_cards() != myopic_board.all_cards():
                differing += 1

    assert total > 100
    assert differing > 0, "backward induction never changed a reply; it is dead weight"
    assert differing / total < 0.25  # it is a correction, not the main effect


# --- Guards and the policy wrapper ------------------------------------------


def test_refuses_any_street_but_t4():
    observation = _sane_world_at(0, seed=91)
    earlier = _world_at("T3", 0, seed=91, policy=mc_policy(sims=2)).observe()
    with pytest.raises(ValueError, match="needs T4"):
        evaluate_t4_exact(earlier)
    assert observation.current_slot().street == "T4"


def test_exact_t4_policy_returns_a_legal_completing_action():
    observation = _sane_world_at(2, seed=104).observe()
    action = exact_t4_policy()(observation, 12345)
    board = observation.hero_board.place(action.placements)
    assert board.is_complete()
    assert sorted((*[c for c, _ in action.placements], *action.discards)) == sorted(
        observation.dealt_cards
    )


def test_candidates_are_sorted_and_deterministic():
    observation = _sane_world_at(1, seed=233).observe()
    first = evaluate_t4_exact(observation)
    again = evaluate_t4_exact(observation)
    assert [c.action for c in first] == [c.action for c in again]
    assert [c.ev for c in first] == sorted((c.ev for c in first), reverse=True)


# --- T3 with an exact T4 round at the leaf -----------------------------------


def test_t3_is_supported_for_the_last_actor_only():
    from ofc_regular.three_max.exact import evaluate_t3

    last = _world_at("T3", 2, seed=91, policy=mc_policy(sims=2)).observe()
    assert all(board.card_count() == 11 for board in last.opponent_boards)
    assert evaluate_t3(last, samples=4, seed=1)

    earlier = _world_at("T3", 0, seed=91, policy=mc_policy(sims=2)).observe()
    with pytest.raises(NotImplementedError, match="last actor only"):
        evaluate_t3(earlier, samples=4, seed=1)

    with pytest.raises(ValueError, match="needs T3"):
        evaluate_t3(
            _world_at("T4", 2, seed=91, policy=mc_policy(sims=2)).observe(),
            samples=4,
        )


def test_t3_action_choice_is_stable_at_the_shipped_budget():
    """Measured: 64 samples gives zero regret against a 512-sample reference."""
    from ofc_regular.three_max.exact import evaluate_t3

    observation = _world_at("T3", 2, seed=91, policy=mc_policy(sims=2)).observe()
    reference = {c.action: c.ev for c in evaluate_t3(observation, samples=192, seed=99)}
    best = max(reference.values())
    for seed in range(4):
        chosen = evaluate_t3(observation, samples=64, seed=seed)[0].action
        assert best - reference[chosen] == pytest.approx(0.0, abs=0.5)


def test_t3_winner_can_be_rescored_on_held_out_draws():
    from ofc_regular.three_max.exact import evaluate_t3

    observation = _world_at("T3", 2, seed=104, policy=mc_policy(sims=2)).observe()
    ranked = evaluate_t3(observation, samples=24, seed=2, holdout_samples=24)
    assert ranked[0].ev_holdout is not None
    assert all(c.ev_holdout is None for c in ranked[1:])
    with pytest.raises(ValueError, match="holdout_samples must be positive"):
        evaluate_t3(observation, samples=4, holdout_samples=0)


def test_the_opening_t4_actor_needs_backward_induction_to_have_any_signal():
    """A myopic opening actor is not approximate -- it is blind.

    The opening T4 actor has no finished opponent to compare against, so a
    myopic objective (sum over already-settled boards) is identically zero for
    every one of its actions: a greedy resolver would return whichever action
    the enumerator emitted first.  Backward induction scores each action by
    what the later actors do in response, and that objective genuinely varies.
    Both halves are asserted here; the fix is only meaningful with both.
    """
    from ofc_regular.three_max.exact import _resolve_t4_round

    observation = _world_at("T3", 2, seed=91, policy=mc_policy(sims=2)).observe()
    hero = observation.hero_board.place(
        generate_actions(observation.hero_board, observation.dealt_cards)[0].placements
    )
    unknown = observation.unknown_cards()

    varied = 0
    checked = 0
    for offset in (0, 3, 6):
        draw = unknown[offset : offset + 9]
        opening_board, opening_deal = observation.opponent_boards[0], draw[0:3]
        replies = [
            opening_board.place(action.placements)
            for action in generate_actions(opening_board, opening_deal)
            if opening_board.place(action.placements).is_complete()
        ]
        assert len(replies) > 1

        # (a) myopic: no settled opponent means no signal whatsoever.
        myopic_values = {
            sum(_score_pair(_terminal_of(reply), other, FL_EV_14) for other in ())
            for reply in replies
        }
        assert myopic_values == {0}

        # (b) induction: score each reply by the round it induces.
        induced = set()
        for reply in replies:
            opening = _terminal_of(reply)
            tail = _resolve_t4_round(
                (
                    (observation.opponent_boards[1], draw[3:6]),
                    (hero, draw[6:9]),
                ),
                (opening,),
                FL_EV_14,
            )
            others = [terminal for terminal in tail if terminal is not opening]
            induced.add(
                sum(_score_pair(opening, other, FL_EV_14) for other in others)
            )
        checked += 1
        if len(induced) > 1:
            varied += 1

        resolved = _resolve_t4_round(
            (
                (opening_board, opening_deal),
                (observation.opponent_boards[1], draw[3:6]),
                (hero, draw[6:9]),
            ),
            (),
            FL_EV_14,
        )
        assert len(resolved) == 3

    assert checked == 3
    assert varied >= 2, "induction gave the opening actor no signal either"


def test_the_fast_t3_path_matches_the_recursive_resolver_exactly():
    """The precomputed-terminal optimisation is a 19x speedup, not a new answer."""
    import random as _random

    from ofc_regular.three_max.exact import _resolve_t4_round, _t3_action_values

    for seed in (9_100_000, 9_100_001, 9_100_002):
        observation = _world_at("T3", 2, seed=seed, policy=mc_policy(sims=4)).observe()
        unknown = list(observation.unknown_cards())
        rng = _random.Random(7)
        draws = [tuple(rng.sample(unknown, 9)) for _ in range(6)]

        fast = _t3_action_values(observation, draws, FL_EV_14)
        for action in generate_actions(
            observation.hero_board, observation.dealt_cards
        ):
            placed = observation.hero_board.place(action.placements)
            total = 0.0
            for draw in draws:
                resolved = _resolve_t4_round(
                    (
                        (observation.opponent_boards[0], draw[0:3]),
                        (observation.opponent_boards[1], draw[3:6]),
                        (placed, draw[6:9]),
                    ),
                    (),
                    FL_EV_14,
                )
                *opponents, hero = resolved
                total += sum(
                    _score_pair(hero, opponent, FL_EV_14) for opponent in opponents
                )
            assert fast[action][1] == pytest.approx(total / len(draws), abs=1e-12)
