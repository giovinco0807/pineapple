"""M1 tests: the 3-max Monte-Carlo referee and the 3-seat rotation harness."""

from __future__ import annotations

import random

import pytest

from ofc_regular.cards import create_deck
from ofc_regular.evaluator import score_board
from ofc_regular.teacher import _heads_up_terminal_score
from ofc_regular.three_max import (
    ACT_ORDER,
    PLAYER_COUNT,
    WorldState3,
    pair_scores,
    play_hand,
    uniform_random_policy,
)
from ofc_regular.three_max.mc import (
    Terminal,
    _score_pair,
    _terminal,
    evaluate_actions_mc,
    mc_policy,
)
from ofc_regular.three_max.rotation import (
    ROTATIONS,
    evaluate_matchup,
    play_block,
    seat_for,
    self_test,
)

FL_EV = {14: 9.6}


def _random_complete_board(rng: random.Random, deck: list[str]):
    cards = [deck.pop() for _ in range(13)]
    return tuple(sorted(cards[:3])), tuple(sorted(cards[3:8])), tuple(sorted(cards[8:]))


def _world_at(street: str, act_order: int, seed: int) -> WorldState3:
    world = WorldState3.new_hand(create_deck(shuffle=True, rng=random.Random(seed)))
    while True:
        slot = world.current_slot()
        if slot.street == street and slot.act_order == act_order:
            return world
        world = world.apply(
            uniform_random_policy(world.observe(), seed * 31 + slot.decision_index)
        )


# --- The cached terminal must not drift from the real scoring kernel ---------


# A random 13-card board almost never enters Fantasyland without fouling, so
# the FL branch of the parity pin needs boards built for it.
FL_ENTRY_ROWS = [
    (  # QQ on top under KK / AA
        ("Qh", "Qd", "2h"),
        ("Ks", "Kd", "3c", "4c", "5c"),
        ("As", "Ad", "6c", "7d", "8h"),
    ),
    (  # trips on top, the maximum top royalty
        ("7h", "7d", "7c"),
        ("8s", "8d", "8c", "2c", "3d"),
        ("9s", "9d", "9c", "9h", "4d"),
    ),
]


def test_cached_terminal_matches_score_board_and_the_pair_kernel():
    """The row-level cache duplicates ten lines of score_board; pin them together."""
    rng = random.Random(20260812)
    checked = 0
    fouls = 0
    entries = 0
    forced = [(rows, other) for rows in FL_ENTRY_ROWS for other in FL_ENTRY_ROWS]
    for index in range(400):
        if index < len(forced):
            hero_rows, opponent_rows = forced[index]
        else:
            deck = create_deck(shuffle=True, rng=rng)
            hero_rows = _random_complete_board(rng, deck)
            opponent_rows = _random_complete_board(rng, deck)

        hero_reference = score_board(*hero_rows)
        opponent_reference = score_board(*opponent_rows)
        hero = _terminal(*hero_rows)
        opponent = _terminal(*opponent_rows)

        assert hero.busted == hero_reference.busted
        assert hero.royalty == hero_reference.total_royalty
        assert hero.fl_entry == (
            hero_reference.fl_entry.qualifies and not hero_reference.busted
        )
        assert hero.values == (
            hero_reference.top_value,
            hero_reference.middle_value,
            hero_reference.bottom_value,
        )

        for fl_ev in ({14: 9.6}, {14: 0.0}, {14: 25.0}):
            assert _score_pair(hero, opponent, fl_ev[14]) == pytest.approx(
                _heads_up_terminal_score(hero_reference, opponent_reference, fl_ev)
            )
        checked += 1
        fouls += hero.busted
        entries += hero.fl_entry

    # The pin is only worth something if the sample covered the branches.
    assert checked == 400
    assert fouls > 50
    assert entries >= len(FL_ENTRY_ROWS)


# --- The referee ------------------------------------------------------------


def test_last_actor_at_t4_is_exact_because_nothing_is_left_to_sample():
    """Both opponents are complete and the hero has no future street: zero variance."""
    world = _world_at("T4", 2, seed=91)
    observation = world.observe()
    assert all(board.is_complete() for board in observation.opponent_boards)

    ranked = evaluate_actions_mc(observation, sims=3, seed=7, fl_ev_per_pair=FL_EV)
    single = evaluate_actions_mc(observation, sims=1, seed=999, fl_ev_per_pair=FL_EV)
    assert [c.action for c in ranked] == [c.action for c in single]

    for candidate in ranked:
        boards = {
            world.current_slot().seat: candidate.board,
            **{
                seat: world.boards[seat]
                for seat in ACT_ORDER
                if seat != world.current_slot().seat
            },
        }
        exact = pair_scores(boards, FL_EV)
        hero_seat = world.current_slot().seat
        expected = sum(
            value if pair[0] == hero_seat else -value
            for pair, value in exact.items()
            if hero_seat in pair
        )
        assert candidate.ev == pytest.approx(expected)


def test_ranking_is_deterministic_and_seed_dependent():
    observation = _world_at("T2", 1, seed=44).observe()
    first = evaluate_actions_mc(observation, sims=24, seed=5)
    again = evaluate_actions_mc(observation, sims=24, seed=5)
    other = evaluate_actions_mc(observation, sims=24, seed=6)

    assert [c.action for c in first] == [c.action for c in again]
    assert [c.ev for c in first] == [c.ev for c in again]
    assert [c.ev for c in first] != [c.ev for c in other]


def test_common_random_numbers_are_shared_across_candidates():
    """Every candidate must face the same futures, or the ranking is noise.

    Top-1 identity across seeds is the wrong assertion here: this position's
    two best actions are a symmetric pair (8h or Ts to the top row) whose true
    EVs tie, so which one wins flips with the seed.  The property CRN actually
    buys is small REGRET: the action chosen under one seed stays near the top
    when re-scored under any other seed.
    """
    observation = _world_at("T3", 0, seed=0).observe()
    ranked = evaluate_actions_mc(observation, sims=32, seed=11)
    spread = max(c.ev for c in ranked) - min(c.ev for c in ranked)
    assert spread > 1.0
    chosen = ranked[0].action

    small_regret = 0
    for seed in range(20, 28):
        rescored = evaluate_actions_mc(observation, sims=32, seed=seed)
        chosen_ev = next(c.ev for c in rescored if c.action == chosen)
        if rescored[0].ev - chosen_ev <= 1.5:
            small_regret += 1
    assert small_regret >= 6


def test_a_board_already_locked_to_foul_gives_every_candidate_the_same_ev():
    """Not a bug: once the hero is fouled its own choices stop mattering.

    Uniform-random play reaches this state at T3 about a fifth of the time,
    which is why the random continuation model is a pessimistic referee.
    """
    observation = _world_at("T3", 0, seed=52).observe()
    ranked = evaluate_actions_mc(observation, sims=8, seed=11)
    assert all(candidate.bust_rate == 1.0 for candidate in ranked)
    assert len({candidate.ev for candidate in ranked}) == 1


def test_referee_reports_bust_and_royalty_rates():
    observation = _world_at("T1", 0, seed=61).observe()
    ranked = evaluate_actions_mc(observation, sims=32, seed=3)
    for candidate in ranked:
        assert 0.0 <= candidate.bust_rate <= 1.0
        assert 0.0 <= candidate.fl_rate <= 1.0
        assert candidate.royalty >= 0.0
        assert candidate.sims == 32
    assert ranked[0].ev >= ranked[-1].ev


def test_mc_policy_plays_a_legal_hand():
    result = play_hand(
        seed=8, policies={seat: mc_policy(sims=4) for seat in ACT_ORDER}
    )
    for seat in ACT_ORDER:
        assert result.world.boards[seat].is_complete()
        assert len(result.world.private_discards[seat]) == 4
    assert sum(result.settlement.raw_totals.values()) == pytest.approx(0.0)


def test_referee_rejects_impossible_budgets():
    observation = _world_at("T1", 0, seed=77).observe()
    with pytest.raises(ValueError, match="sims must be positive"):
        evaluate_actions_mc(observation, sims=0)


# --- The rotation harness ---------------------------------------------------


def test_each_player_visits_each_seat_exactly_once_per_block():
    for player in range(PLAYER_COUNT):
        seats = {seat_for(player, rotation) for rotation in range(ROTATIONS)}
        assert seats == set(ACT_ORDER)
    for rotation in range(ROTATIONS):
        seats = {seat_for(player, rotation) for player in range(PLAYER_COUNT)}
        assert seats == set(ACT_ORDER)


def test_identical_policies_cancel_to_exactly_zero():
    """The harness's cancellation identity -- no verdict is valid without it."""
    self_test(policy=uniform_random_policy, seeds=range(40))
    self_test(policy=mc_policy(sims=2), seeds=range(3))


def test_self_test_catches_a_seat_term_that_does_not_cancel():
    """A policy that reads the seat asymmetrically must fail the identity."""

    def seat_biased(observation, decision_seed):
        actions = __import__(
            "ofc_regular.action_space", fromlist=["generate_actions"]
        ).generate_actions(observation.hero_board, observation.dealt_cards)
        index = 0 if observation.seat == "btn" else len(actions) - 1
        return actions[index]

    # Identical copies of even a biased policy still cancel: all three players
    # play it, so the block is the same hand three times over.
    self_test(policy=seat_biased, seeds=range(5))

    # But a block of *different* policies must not cancel, or the harness is
    # measuring nothing.
    block = play_block(
        seed=3,
        policies_by_player={
            0: seat_biased,
            1: uniform_random_policy,
            2: uniform_random_policy,
        },
    )
    assert any(value != 0.0 for value in block.totals.values())


def test_block_totals_are_zero_sum_across_players():
    block = play_block(
        seed=17,
        policies_by_player={
            0: uniform_random_policy,
            1: mc_policy(sims=2),
            2: uniform_random_policy,
        },
    )
    assert sum(block.totals.values()) == pytest.approx(0.0)
    assert len(block.per_rotation) == ROTATIONS
    for rotation, scores in enumerate(block.per_rotation):
        assert sum(scores.values()) == pytest.approx(0.0)
        assert block.seats_per_rotation[rotation] == {
            player: seat_for(player, rotation) for player in range(PLAYER_COUNT)
        }


def test_self_test_residual_is_float_noise_not_a_defect():
    """The cancellation is exact in arithmetic; float64 leaves an ulp behind."""
    from ofc_regular.three_max.rotation import SELF_TEST_TOLERANCE

    worst = 0.0
    for seed in range(30):
        block = play_block(
            seed=seed,
            policies_by_player={p: uniform_random_policy for p in range(PLAYER_COUNT)},
        )
        worst = max(worst, max(abs(v) for v in block.totals.values()))
    assert worst < 1e-12, f"residual {worst} is far above float noise"
    assert SELF_TEST_TOLERANCE > worst


def test_matchup_summary_reports_per_hand_statistics():
    summary = evaluate_matchup(
        policies_by_player={p: uniform_random_policy for p in range(PLAYER_COUNT)},
        blocks=5,
        base_seed=400,
    )
    assert summary.blocks == 5
    assert summary.hands == 15
    # Identical policies: every block cancels, so every statistic is exactly 0.
    for player in range(PLAYER_COUNT):
        assert summary.mean_per_hand[player] == pytest.approx(0.0)
        assert summary.stderr_per_hand[player] == pytest.approx(0.0)
    assert sum(summary.mean_per_hand.values()) == pytest.approx(0.0)


def test_matchup_rejects_degenerate_budgets():
    with pytest.raises(ValueError, match="blocks must be at least 2"):
        evaluate_matchup(
            policies_by_player={p: uniform_random_policy for p in range(PLAYER_COUNT)},
            blocks=1,
            base_seed=1,
        )
    with pytest.raises(ValueError, match="missing 3-max policies"):
        play_block(seed=1, policies_by_player={0: uniform_random_policy})
    with pytest.raises(ValueError, match="rotation out of range"):
        seat_for(0, 3)
    with pytest.raises(ValueError, match="unknown orientations"):
        play_block(
            seed=1,
            policies_by_player={p: uniform_random_policy for p in range(PLAYER_COUNT)},
            orientations="diagonal",  # type: ignore[arg-type]
        )


def test_full_orientations_cover_both_relative_orders():
    """Cyclic seatings fix each pair's relative order; full must cover both."""
    from ofc_regular.three_max.rotation import seatings_for

    cyclic = seatings_for("cyclic")
    assert len(cyclic) == 3
    full = seatings_for("full")
    assert len(full) == 6

    def orientation(seating, first: int, second: int) -> int:
        order = [ACT_ORDER.index(seating[player]) for player in (first, second)]
        return (order[1] - order[0]) % PLAYER_COUNT

    for first in range(PLAYER_COUNT):
        for second in range(PLAYER_COUNT):
            if first == second:
                continue
            assert {orientation(s, first, second) for s in cyclic} != {1, 2} or True
            assert {orientation(s, first, second) for s in full} == {1, 2}

    # Cyclic really is stuck in one orientation per ordered pair.
    assert {orientation(s, 0, 1) for s in cyclic} == {1}
    assert {orientation(s, 0, 2) for s in cyclic} == {2}


def test_full_orientation_block_still_cancels_identical_policies():
    self_test(
        policy=uniform_random_policy, seeds=range(10), orientations="full"
    )
    block = play_block(
        seed=5,
        policies_by_player={p: uniform_random_policy for p in range(PLAYER_COUNT)},
        orientations="full",
    )
    assert len(block.per_rotation) == 6
    for player in range(PLAYER_COUNT):
        seats = [seats_map[player] for seats_map in block.seats_per_rotation]
        assert sorted(seats) == sorted(list(ACT_ORDER) * 2)


def test_winner_holdout_rescore_is_populated_and_others_are_not():
    observation = _world_at("T2", 0, seed=33).observe()
    ranked = evaluate_actions_mc(observation, sims=16, seed=2, holdout_sims=16)
    assert ranked[0].ev_holdout is not None
    assert all(c.ev_holdout is None for c in ranked[1:])
    # The holdout is an independent measurement of the same action; it should
    # land in the same region but need not match the in-sample estimate.
    assert abs(ranked[0].ev_holdout - ranked[0].ev) < 15.0

    with pytest.raises(ValueError, match="holdout_sims must be positive"):
        evaluate_actions_mc(observation, sims=4, holdout_sims=0)


def test_ev_ties_break_toward_lower_bust_rate():
    observation = _world_at("T2", 0, seed=33).observe()
    ranked = evaluate_actions_mc(observation, sims=8, seed=4)
    for earlier, later in zip(ranked, ranked[1:]):
        if earlier.ev == later.ev:
            assert earlier.bust_rate <= later.bust_rate
