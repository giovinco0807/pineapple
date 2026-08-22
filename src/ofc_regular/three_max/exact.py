"""Exact final-street evaluation for 3-max regular OFC.

This is the leaf the whole cascade will stand on, and 3-max makes it cheaper
than heads-up rather than dearer: 51 of 52 cards are dealt, so by T4 the hero's
unseen set has collapsed to 13, 11, or 9 cards depending on where it sits.

Three cases, by how many opponents still have a decision left:

``act_index 2`` (last)
    Both opponents are complete.  Nothing is left to sample, so the value of
    every hero action is a closed form: place, score both pairs, done.

``act_index 1``
    One opponent already acted this street; one still has to.  The unseen set
    is 11 cards and the responder draws 3, so all C(11,3) = 165 deals are
    enumerated by default -- still exact, just with a chance node in front.

``act_index 0`` (first)
    Both opponents still have to act, in their own order.  C(13,3) x C(10,3) =
    34,320 joint deals, so this one samples by default.

Responders are resolved by backward induction, not myopically: the last actor
best-responds to everything already on the table, and the actor before it picks
the reply that is best once that response lands.  Each responder maximises ITS
own 3-max total -- the sum over its two opponents (contract R4/R7) -- which is
what makes this a game solve rather than a hero-centric one.  Note the
consequence: a responder is not trying to punish the hero specifically, so it
hurts the hero less than a dedicated best response would.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Sequence

from ..action_space import Action, generate_actions
from ..state import ROWS, Board
from .mc import Terminal, _key, _score_pair, _terminal
from .scoring import DEFAULT_FL_EV_PER_PAIR
from .world import ThreeMaxObservation

# Enumerate the chance node when it is at most this wide, otherwise sample.
FULL_ENUMERATION_LIMIT = 400

# Independent stream for a winner's holdout re-score.
_HOLDOUT_SEED_SALT = 0x9E3779B97F4A7C15


@dataclass(frozen=True)
class ExactCandidate:
    action: Action
    board: Board
    ev: float
    deals: int
    exact: bool


def _board_terminal(board: Board) -> Terminal:
    return _terminal(*_key({row: getattr(board, row) for row in ROWS}))


def _responder_reply(
    board: Board,
    dealt: Sequence[str],
    others: Sequence[Terminal],
    fl_ev_14: float,
) -> tuple[Board, Terminal]:
    """The reply maximising this actor's own total against ``others``."""
    best_board: Board | None = None
    best_terminal: Terminal | None = None
    best_value = float("-inf")
    for action in generate_actions(board, dealt):
        candidate = board.place(action.placements)
        if not candidate.is_complete():
            continue
        terminal = _board_terminal(candidate)
        value = sum(_score_pair(terminal, other, fl_ev_14) for other in others)
        if value > best_value:
            best_value, best_board, best_terminal = value, candidate, terminal
    if best_board is None or best_terminal is None:
        raise ValueError("responder has no completing action")
    return best_board, best_terminal


def _resolve_responders(
    hero: Terminal,
    responders: Sequence[tuple[Board, Sequence[str]]],
    settled: Sequence[Terminal],
    fl_ev_14: float,
) -> tuple[Terminal, ...]:
    """Play out the actors after the hero, by backward induction."""
    if not responders:
        return tuple(settled)

    if len(responders) == 1:
        board, dealt = responders[0]
        _final, terminal = _responder_reply(
            board, dealt, (hero, *settled), fl_ev_14
        )
        return (*settled, terminal)

    (first_board, first_dealt), (second_board, second_dealt) = responders
    best_terminals: tuple[Terminal, Terminal] | None = None
    best_value = float("-inf")
    for action in generate_actions(first_board, first_dealt):
        candidate = first_board.place(action.placements)
        if not candidate.is_complete():
            continue
        first_terminal = _board_terminal(candidate)
        # The last actor sees this reply and best-responds to it.
        _last_board, last_terminal = _responder_reply(
            second_board, second_dealt, (hero, first_terminal), fl_ev_14
        )
        value = _score_pair(first_terminal, hero, fl_ev_14) + _score_pair(
            first_terminal, last_terminal, fl_ev_14
        )
        if value > best_value:
            best_value = value
            best_terminals = (first_terminal, last_terminal)
    if best_terminals is None:
        raise ValueError("responder has no completing action")
    return (*settled, *best_terminals)


def _joint_deals(
    unknown: Sequence[str], responder_count: int, max_deals: int | None, seed: int
) -> tuple[tuple[tuple[str, ...], ...], bool]:
    """Deals for each responder, enumerated when cheap enough else sampled."""
    if responder_count == 0:
        return ((),), True

    if responder_count == 1:
        options = [(cards,) for cards in combinations(unknown, 3)]
    else:
        options = []
        for first in combinations(unknown, 3):
            remaining = [card for card in unknown if card not in first]
            for second in combinations(remaining, 3):
                options.append((first, second))

    limit = max_deals if max_deals is not None else FULL_ENUMERATION_LIMIT
    if len(options) <= limit:
        return tuple(options), True
    rng = random.Random(seed)
    return tuple(rng.sample(options, limit)), False


def evaluate_t4_exact(
    observation: ThreeMaxObservation,
    *,
    fl_ev_per_pair: Mapping[int, float] | None = None,
    max_deals: int | None = None,
    seed: int = 0,
) -> list[ExactCandidate]:
    """Rank every legal T4 action of ``observation``.

    ``exact`` is True on every candidate when the chance node was enumerated
    rather than sampled; with no responders left it is exact unconditionally.
    """
    if observation.street != "T4":
        raise ValueError(f"final-street evaluation needs T4, got {observation.street}")

    table = dict(fl_ev_per_pair or DEFAULT_FL_EV_PER_PAIR)
    fl_ev_14 = float(table.get(14, 0.0))

    settled: list[Terminal] = []
    responders: list[Board] = []
    for board in observation.opponent_boards:
        if board.is_complete():
            settled.append(_board_terminal(board))
        else:
            responders.append(board)

    deals, exact = _joint_deals(
        observation.unknown_cards(), len(responders), max_deals, seed
    )

    candidates: list[ExactCandidate] = []
    for action in generate_actions(observation.hero_board, observation.dealt_cards):
        placed = observation.hero_board.place(action.placements)
        if not placed.is_complete():
            continue
        hero = _board_terminal(placed)

        total = 0.0
        for deal in deals:
            opponents = _resolve_responders(
                hero,
                tuple(zip(responders, deal)),
                settled,
                fl_ev_14,
            )
            total += sum(
                _score_pair(hero, opponent, fl_ev_14) for opponent in opponents
            )
        candidates.append(
            ExactCandidate(
                action=action,
                board=placed,
                ev=total / len(deals),
                deals=len(deals),
                exact=exact,
            )
        )

    if not candidates:
        raise ValueError("no completing action at T4")
    candidates.sort(
        key=lambda item: (
            -item.ev,
            _key({row: getattr(item.board, row) for row in ROWS}),
        )
    )
    return candidates


@dataclass(frozen=True)
class T3Candidate:
    action: Action
    board: Board
    ev: float
    samples: int
    leaf: str
    # Set on the winner only, when ``holdout_samples`` is given: its ``ev`` is
    # the max of estimates sharing one draw set and so is biased upward by
    # selection.  A label that CONSUMES the value must read this instead.
    ev_holdout: float | None = None


def _resolve_t4_round(
    actors: Sequence[tuple[Board, Sequence[str]]],
    settled: Sequence[Terminal],
    fl_ev_14: float,
) -> tuple[Terminal, ...]:
    """Play out a whole T4 round in act order by backward induction.

    Every actor maximises ITS OWN 3-max total, and each anticipates the actors
    that follow it.  Resolving greedily instead would be worse than an
    approximation here: the first actor has no finished opponent to compare
    against yet, so a myopic objective is identically zero for all of its
    actions and it would pick whatever the enumerator happened to emit first.
    """
    if not actors:
        return tuple(settled)

    (board, dealt), rest = actors[0], actors[1:]
    best: tuple[Terminal, ...] | None = None
    best_value = float("-inf")
    for action in generate_actions(board, dealt):
        candidate = board.place(action.placements)
        if not candidate.is_complete():
            continue
        terminal = _board_terminal(candidate)
        tail = _resolve_t4_round(rest, (*settled, terminal), fl_ev_14)
        others = [t for t in tail if t is not terminal]
        value = sum(_score_pair(terminal, other, fl_ev_14) for other in others)
        if value > best_value:
            best_value, best = value, tail
    if best is None:
        raise ValueError("T4 actor has no completing action")
    return best


def _completing_terminals(board: Board, dealt: Sequence[str]) -> tuple[Terminal, ...]:
    """Every completion of ``board`` from ``dealt``, scored once."""
    return tuple(
        _board_terminal(board.place(action.placements))
        for action in generate_actions(board, dealt)
        if board.place(action.placements).is_complete()
    )


def _t3_action_values(
    observation: ThreeMaxObservation,
    draws: Sequence[tuple[str, ...]],
    fl_ev_14: float,
    score_opponents: int = 2,
) -> dict[Action, tuple[Board, float]]:
    """Value every hero T3 action against a shared draw set.

    An opponent's T4 replies depend only on its own board and its own deal, and
    neither changes with the hero's T3 choice, so their terminals are scored
    once per draw and reused across all the hero's candidates.  What is left
    inside the loop is arithmetic on cached terminals, which is where the
    backward induction stops being the expensive part.
    """
    hero_actions = [
        (action, observation.hero_board.place(action.placements))
        for action in generate_actions(
            observation.hero_board, observation.dealt_cards
        )
    ]

    per_draw_opponents = [
        (
            _completing_terminals(observation.opponent_boards[0], draw[0:3]),
            _completing_terminals(observation.opponent_boards[1], draw[3:6]),
        )
        for draw in draws
    ]

    values: dict[Action, tuple[Board, float]] = {}
    for action, placed in hero_actions:
        total = 0.0
        for draw, (first_options, second_options) in zip(draws, per_draw_opponents):
            hero_options = _completing_terminals(placed, draw[6:9])
            total += _induced_hero_value(
                first_options, second_options, hero_options, fl_ev_14,
                score_opponents,
            )
        values[action] = (placed, total / len(draws))
    return values


def resolve_round_indices(
    options: Sequence[Sequence[Terminal]], fl_ev_14: float
) -> tuple[int, int, int]:
    """Indices each actor picks in a T4 round, by backward induction.

    ``options`` is in ACT order and says nothing about where the hero sits, so
    unlike :func:`_induced_hero_value` -- which is written for the seat that
    closes the street -- this serves any of the three seats.  That is what a
    middle-seat label needs: at (T3, BB) the hero still has the BTN acting
    after it at T4, so its own reply is not the last word.

    Each actor maximises its own 3-max total against the other two (R7).  Ties
    go to the first option in enumeration order, matching the hand-rolled
    version this is pinned against.

    The three pairwise score matrices are built once and the induction then
    runs on floats: with six completions a side that is 108 hand comparisons
    instead of the 648 the naive triple loop repeats.
    """
    first, second, third = options
    # _score_pair is antisymmetric, so one matrix per unordered pair is enough.
    a = [[_score_pair(x, y, fl_ev_14) for y in second] for x in first]
    b = [[_score_pair(x, z, fl_ev_14) for z in third] for x in first]
    c = [[_score_pair(y, z, fl_ev_14) for z in third] for y in second]

    best_first_value = float("-inf")
    chosen = (0, 0, 0)
    for i, row_a in enumerate(a):
        row_b = b[i]
        best_second_value = float("-inf")
        pick_second = 0
        pick_third = 0
        for j, row_c in enumerate(c):
            # The last actor's total is -(b[i][k] + c[j][k]); maximising it is
            # minimising that sum.
            best_third = float("inf")
            k_star = 0
            for k, c_jk in enumerate(row_c):
                total = row_b[k] + c_jk
                if total < best_third:
                    best_third, k_star = total, k
            second_value = -row_a[j] + row_c[k_star]
            if second_value > best_second_value:
                best_second_value = second_value
                pick_second, pick_third = j, k_star
        first_value = row_a[pick_second] + row_b[pick_third]
        if first_value > best_first_value:
            best_first_value = first_value
            chosen = (i, pick_second, pick_third)
    return chosen


def round_hero_value(
    options: Sequence[Sequence[Terminal]], hero_index: int, fl_ev_14: float
) -> float:
    """The hero's score once a T4 round in ``options`` resolves."""
    picks = resolve_round_indices(options, fl_ev_14)
    hero = options[hero_index][picks[hero_index]]
    return sum(
        _score_pair(hero, options[index][picks[index]], fl_ev_14)
        for index in range(len(options))
        if index != hero_index
    )


def _induced_hero_value(
    first_options: Sequence[Terminal],
    second_options: Sequence[Terminal],
    hero_options: Sequence[Terminal],
    fl_ev_14: float,
    score_opponents: int = 2,
) -> float:
    """Hero's score after a T4 round resolved by backward induction.

    Act order is first, second, hero (the hero is the BTN and closes), and each
    actor maximises its own total against the other two.
    """
    best_first_value = float("-inf")
    hero_score = 0.0
    for first in first_options:
        best_second_value = float("-inf")
        chosen_second: Terminal | None = None
        chosen_hero: Terminal | None = None
        for second in second_options:
            best_hero_value = float("-inf")
            hero_pick: Terminal | None = None
            for hero in hero_options:
                value = _score_pair(hero, first, fl_ev_14) + _score_pair(
                    hero, second, fl_ev_14
                )
                if value > best_hero_value:
                    best_hero_value, hero_pick = value, hero
            assert hero_pick is not None
            second_value = _score_pair(second, first, fl_ev_14) + _score_pair(
                second, hero_pick, fl_ev_14
            )
            if second_value > best_second_value:
                best_second_value = second_value
                chosen_second, chosen_hero = second, hero_pick
        assert chosen_second is not None and chosen_hero is not None
        first_value = _score_pair(first, chosen_second, fl_ev_14) + _score_pair(
            first, chosen_hero, fl_ev_14
        )
        if first_value > best_first_value:
            best_first_value = first_value
            hero_score = _score_pair(chosen_hero, first, fl_ev_14)
            if score_opponents > 1:
                hero_score += _score_pair(chosen_hero, chosen_second, fl_ev_14)
    return hero_score


def evaluate_t3(
    observation: ThreeMaxObservation,
    *,
    samples: int = 64,
    seed: int = 0,
    fl_ev_per_pair: Mapping[int, float] | None = None,
    holdout_samples: int | None = None,
    score_opponents: int = 2,
) -> list[T3Candidate]:
    """Rank T3 actions for the LAST actor, with an exact T4 round at the leaf.

    Only the last actor is supported: by then both opponents have finished
    their own T3, so the whole continuation is one T4 round and the tree has no
    interior opponent decisions.  This is the same place heads-up started (T3
    second seat, 996K exact labels) and for the same reason.

    The two earlier act orders still have opponent T3 decisions in the middle
    of the tree.  Those need an interior-node policy, which is exactly what a
    distilled model is for -- see the roadmap note in the module docstring.
    """
    if observation.street != "T3":
        raise ValueError(f"T3 evaluation needs T3, got {observation.street}")
    if not all(board.card_count() == 11 for board in observation.opponent_boards):
        raise NotImplementedError(
            "T3 is implemented for the last actor only; earlier act orders leave "
            "opponent T3 decisions inside the tree and need an interior policy"
        )

    table = dict(fl_ev_per_pair or DEFAULT_FL_EV_PER_PAIR)
    fl_ev_14 = float(table.get(14, 0.0))
    unknown = list(observation.unknown_cards())
    if len(unknown) < 9:
        raise ValueError(f"T3 continuation needs 9 cards, only {len(unknown)} unseen")

    # The hero acts last at T4 too, so the round runs opponents first (in their
    # own act order) and the hero's reply closes it as a pure best response.
    # ``opponent_seats`` orders the observation's boards act-relative, so for
    # the last actor slot 0 is the opening seat and slot 1 the one after it.
    rng = random.Random(seed)
    draws = [tuple(rng.sample(unknown, 9)) for _ in range(samples)]
    values = _t3_action_values(observation, draws, fl_ev_14, score_opponents)

    candidates = [
        T3Candidate(
            action=action,
            board=board,
            ev=value,
            samples=samples,
            leaf="exact_t4",
        )
        for action, (board, value) in values.items()
    ]
    candidates.sort(
        key=lambda item: (
            -item.ev,
            _key({row: getattr(item.board, row) for row in ROWS}),
        )
    )

    if holdout_samples is not None:
        if holdout_samples <= 0:
            raise ValueError("holdout_samples must be positive")
        holdout_rng = random.Random(seed ^ _HOLDOUT_SEED_SALT)
        holdout_draws = [
            tuple(holdout_rng.sample(unknown, 9)) for _ in range(holdout_samples)
        ]
        winner = candidates[0]
        _board, rescored = _t3_action_values(
            observation, holdout_draws, fl_ev_14
        )[winner.action]
        candidates[0] = T3Candidate(
            action=winner.action,
            board=winner.board,
            ev=winner.ev,
            samples=winner.samples,
            leaf=winner.leaf,
            ev_holdout=rescored,
        )
    return candidates


def exact_t4_policy(
    *,
    fl_ev_per_pair: Mapping[int, float] | None = None,
    max_deals: int | None = None,
):
    """A Policy3 for T4 only; raises on any other street."""

    def policy(observation: ThreeMaxObservation, decision_seed: int) -> Action:
        ranked = evaluate_t4_exact(
            observation,
            fl_ev_per_pair=fl_ev_per_pair,
            max_deals=max_deals,
            seed=decision_seed,
        )
        return ranked[0].action

    policy.__name__ = "exact_t4_policy"
    return policy
