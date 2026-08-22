"""Monte-Carlo referee for 3-max regular OFC.

Ranks every legal action of one decision by common-random-number rollouts.
Every candidate is scored against the SAME sampled continuations, which is what
removes almost all of the between-candidate variance; the two opponents'
completions are drawn once per future and shared by every candidate, because
under common random numbers they do not depend on the hero's choice.

Continuation model (the information-set-correct approximation the heads-up
trainer also uses): the hero keeps playing pineapple -- three cards a street,
two placed at random-legal slots, one discarded -- and each opponent's board is
filled from the unseen deck.  Opponents' own discards are not simulated; they
simply stay in the unseen pool.

This is a referee and a baseline opponent, not a solver.  Its job in M1 is to
give the 3-max track a scoring yardstick and a policy ladder whose rungs differ
only in compute, so later work has something honest to be measured against.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Sequence

from ..action_space import Action, generate_actions
from ..evaluator import (
    evaluate_3_card,
    evaluate_5_card,
    get_bottom_royalty,
    get_middle_royalty,
    get_top_royalty,
)
from ..rules import check_fl_entry
from ..state import ROW_CAPACITY, ROWS, Board
from .scoring import DEFAULT_FL_EV_PER_PAIR
from .seating import STREETS
from .world import ThreeMaxObservation

# Budget follows the fan width: T0 chooses among 232 candidates and gets the
# most rollouts; T4's last actor is exact at one rollout (nothing is left to
# sample) so spending there is waste.
DEFAULT_SIMS = {"T0": 128, "T1": 96, "T2": 64, "T3": 48, "T4": 32}
FOUL_PENALTY = 6
SCOOP_BONUS = 3


@dataclass(frozen=True)
class McCandidate:
    action: Action
    board: Board
    ev: float
    bust_rate: float
    fl_rate: float
    royalty: float
    sims: int
    # EV of this candidate re-scored on futures the ranking never saw.  Only
    # the ranking's winner gets one (when ``holdout_sims`` is set): the winner
    # is the argmax of noisy estimates, so its in-sample ``ev`` is biased
    # upward by selection; ``ev_holdout`` is the unbiased read.
    ev_holdout: float | None = None


# Rollout scoring is dominated by hand evaluation, and rows repeat across
# candidates far more often than whole boards do, so the caches sit at the row
# level.  Composing a board score from them duplicates ten lines of
# ``evaluator.score_board``; ``test_three_max_mc.py`` pins the composition
# against the real kernel over random boards so the two cannot drift.
@lru_cache(maxsize=400_000)
def _top_row(cards: tuple[str, ...]) -> tuple[tuple[int, tuple[int, ...]], int, bool]:
    value = evaluate_3_card(cards)
    return value, get_top_royalty(cards), check_fl_entry(cards).qualifies


@lru_cache(maxsize=400_000)
def _middle_row(cards: tuple[str, ...]) -> tuple[tuple[int, tuple[int, ...]], int]:
    return evaluate_5_card(cards), get_middle_royalty(cards)


@lru_cache(maxsize=400_000)
def _bottom_row(cards: tuple[str, ...]) -> tuple[tuple[int, tuple[int, ...]], int]:
    return evaluate_5_card(cards), get_bottom_royalty(cards)


@dataclass(frozen=True)
class Terminal:
    """A complete board reduced to everything pairwise scoring needs."""

    busted: bool
    values: tuple[tuple[int, tuple[int, ...]], ...]
    royalty: int
    fl_entry: bool


def _terminal(
    top: tuple[str, ...], middle: tuple[str, ...], bottom: tuple[str, ...]
) -> Terminal:
    top_value, top_royalty, fl_entry = _top_row(top)
    middle_value, middle_royalty = _middle_row(middle)
    bottom_value, bottom_royalty = _bottom_row(bottom)
    if top_value > middle_value or middle_value > bottom_value:
        return Terminal(True, (top_value, middle_value, bottom_value), 0, False)
    return Terminal(
        False,
        (top_value, middle_value, bottom_value),
        top_royalty + middle_royalty + bottom_royalty,
        fl_entry,
    )


def _score_pair(hero: Terminal, opponent: Terminal, fl_ev_14: float) -> float:
    """One pair term, mirroring ``teacher._heads_up_terminal_score`` exactly."""
    hero_fl = fl_ev_14 if hero.fl_entry else 0.0
    opponent_fl = fl_ev_14 if opponent.fl_entry else 0.0
    if hero.busted and opponent.busted:
        return 0.0
    if hero.busted:
        return float(-FOUL_PENALTY - opponent.royalty - opponent_fl)
    if opponent.busted:
        return float(FOUL_PENALTY + hero.royalty + hero_fl)

    lines = 0
    for hero_value, opponent_value in zip(hero.values, opponent.values):
        if hero_value > opponent_value:
            lines += 1
        elif hero_value < opponent_value:
            lines -= 1

    score = float(lines)
    if lines == 3:
        score += SCOOP_BONUS
    elif lines == -3:
        score -= SCOOP_BONUS
    return score + hero.royalty - opponent.royalty + hero_fl - opponent_fl


def _board_terminal_for_test(board) -> Terminal:
    """Score a complete Board. Exported for tests and for exact.py."""
    return _terminal(*_key({row: getattr(board, row) for row in ROWS}))


def _key(rows: Mapping[str, Sequence[str]]) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(sorted(rows[row])) for row in ROWS)


def _fill(rows: dict[str, list[str]], cards: Sequence[str], rng: random.Random) -> None:
    slots = [row for row in ROWS for _ in range(ROW_CAPACITY[row] - len(rows[row]))]
    rng.shuffle(slots)
    for card, row in zip(cards, slots):
        rows[row].append(card)


def _continue_pineapple(
    rows: dict[str, list[str]],
    future: Sequence[str],
    keep_plan: Sequence[tuple[int, int]],
    row_rng: random.Random,
) -> None:
    """Play the hero's remaining streets under complete common random numbers.

    Two candidate boards differ in shape, and any stream consumption that
    depends on shape desynchronises the candidates' continuations (measured:
    cross-shape candidate pairs kept different future cards in half their
    futures under the old ``rng.sample``+``rng.choice`` interleaving).  So the
    two random inputs are decoupled from shape entirely:

    - which 2 of the 3 dealt cards to keep is ``keep_plan``, drawn once per
      future before any candidate is scored;
    - each placement consumes exactly one ``row_rng.random()`` -- a fixed-width
      draw, unlike ``choice``'s rejection sampling -- so every candidate reads
      the same stream at the same offsets no matter how many rows are open.
    """
    for street_index, (first, second) in enumerate(keep_plan):
        dealt = future[street_index * 3 : street_index * 3 + 3]
        for keep_index in (first, second):
            draw = row_rng.random()
            open_rows = [row for row in ROWS if len(rows[row]) < ROW_CAPACITY[row]]
            if not open_rows:
                return
            rows[open_rows[min(int(draw * len(open_rows)), len(open_rows) - 1)]].append(
                dealt[keep_index]
            )


@dataclass(frozen=True)
class _Future:
    """One common-random continuation, fully drawn before candidates exist."""

    hero_cards: tuple[str, ...]
    keep_plan: tuple[tuple[int, int], ...]
    row_seed: int
    opponent_terminals: tuple[Terminal, ...]


def _draw_futures(
    observation: ThreeMaxObservation, rollouts: int, seed: int
) -> list[_Future]:
    deck = list(observation.unknown_cards())
    streets_left = len(STREETS) - 1 - STREETS.index(observation.street)
    hero_need = 3 * streets_left
    opponent_needs = [13 - board.card_count() for board in observation.opponent_boards]
    draw = hero_need + sum(opponent_needs)
    if draw > len(deck):
        raise ValueError(
            f"unseen deck of {len(deck)} cannot cover a {draw}-card continuation"
        )

    rng = random.Random(seed)
    futures: list[_Future] = []
    for _ in range(rollouts):
        sample = rng.sample(deck, draw)
        cursor = hero_need
        opponent_terminals = []
        for board, need in zip(observation.opponent_boards, opponent_needs):
            rows = {row: list(getattr(board, row)) for row in ROWS}
            _fill(rows, sample[cursor : cursor + need], rng)
            cursor += need
            opponent_terminals.append(_terminal(*_key(rows)))
        futures.append(
            _Future(
                hero_cards=tuple(sample[:hero_need]),
                keep_plan=tuple(
                    tuple(sorted(rng.sample(range(3), 2)))
                    for _ in range(streets_left)
                ),
                row_seed=rng.randrange(1 << 62),
                opponent_terminals=tuple(opponent_terminals),
            )
        )
    return futures


def _score_action(
    observation: ThreeMaxObservation,
    action: Action,
    futures: Sequence[_Future],
    fl_ev_14: float,
) -> McCandidate:
    placed = observation.hero_board.place(action.placements)
    base = {row: list(getattr(placed, row)) for row in ROWS}

    total = 0.0
    busts = 0
    entries = 0
    royalty = 0.0
    for future in futures:
        rows = {row: list(base[row]) for row in ROWS}
        _continue_pineapple(
            rows, future.hero_cards, future.keep_plan, random.Random(future.row_seed)
        )
        hero_term = _terminal(*_key(rows))
        total += sum(
            _score_pair(hero_term, opponent, fl_ev_14)
            for opponent in future.opponent_terminals
        )
        busts += hero_term.busted
        if not hero_term.busted:
            entries += hero_term.fl_entry
            royalty += hero_term.royalty

    rollouts = len(futures)
    return McCandidate(
        action=action,
        board=placed,
        ev=total / rollouts,
        bust_rate=busts / rollouts,
        fl_rate=entries / rollouts,
        royalty=royalty / rollouts,
        sims=rollouts,
    )


# Independent stream for the winner's holdout re-score; any fixed odd constant
# that decorrelates the two seeds works.
_HOLDOUT_SEED_SALT = 0x9E3779B97F4A7C15


def evaluate_actions_mc(
    observation: ThreeMaxObservation,
    *,
    sims: int | None = None,
    seed: int = 0,
    fl_ev_per_pair: Mapping[int, float] | None = None,
    holdout_sims: int | None = None,
) -> list[McCandidate]:
    """Rank every legal action of ``observation`` by Monte-Carlo EV.

    EV is the hero's 3-max score: the sum over both opponents of the pairwise
    heads-up terminal score (contract R4, infinite stacks).

    The top candidate's ``ev`` is the max of noisy estimates and therefore
    selection-biased upward (measured about +1.9/hand at T0's 232-wide fan
    with small budgets).  Ranking is unaffected -- every candidate shares the
    same futures -- but anything that CONSUMES the winner's value (labels, the
    M2 fl_ev fixed point) must pass ``holdout_sims`` and read ``ev_holdout``,
    which re-scores the already-chosen winner on futures the ranking never
    saw.

    Ties on ev are broken toward lower bust rate, then higher royalty, then a
    canonical board key for determinism.
    """
    actions = generate_actions(observation.hero_board, observation.dealt_cards)
    if not actions:
        raise ValueError("no legal action available")

    table = dict(fl_ev_per_pair or DEFAULT_FL_EV_PER_PAIR)
    fl_ev_14 = float(table.get(14, 0.0))
    rollouts = sims if sims is not None else DEFAULT_SIMS[observation.street]
    if rollouts <= 0:
        raise ValueError("sims must be positive")

    futures = _draw_futures(observation, rollouts, seed)
    candidates = [
        _score_action(observation, action, futures, fl_ev_14) for action in actions
    ]
    candidates.sort(
        key=lambda item: (
            -item.ev,
            item.bust_rate,
            -item.royalty,
            _key({row: getattr(item.board, row) for row in ROWS}),
        )
    )

    if holdout_sims is not None:
        if holdout_sims <= 0:
            raise ValueError("holdout_sims must be positive")
        holdout = _draw_futures(
            observation, holdout_sims, seed ^ _HOLDOUT_SEED_SALT
        )
        winner = _score_action(observation, candidates[0].action, holdout, fl_ev_14)
        candidates[0] = McCandidate(
            action=candidates[0].action,
            board=candidates[0].board,
            ev=candidates[0].ev,
            bust_rate=candidates[0].bust_rate,
            fl_rate=candidates[0].fl_rate,
            royalty=candidates[0].royalty,
            sims=candidates[0].sims,
            ev_holdout=winner.ev,
        )
    return candidates


def mc_policy(
    *,
    sims: int | None = None,
    fl_ev_per_pair: Mapping[int, float] | None = None,
):
    """A Policy3 that plays the Monte-Carlo referee's top-ranked action."""

    def policy(observation: ThreeMaxObservation, decision_seed: int) -> Action:
        ranked = evaluate_actions_mc(
            observation,
            sims=sims,
            seed=decision_seed,
            fl_ev_per_pair=fl_ev_per_pair,
        )
        return ranked[0].action

    policy.__name__ = f"mc_policy(sims={sims if sims is not None else 'default'})"
    return policy
