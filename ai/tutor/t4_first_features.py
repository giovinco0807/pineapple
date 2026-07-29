"""Feature encoding for a learned T4 first-seat evaluator (Joker ruleset).

Design ported from the regular track's `rust/hu_m3_engine/src/t4_features.rs`,
whose header records the lessons this encoder is built to respect:

- Feeding raw cards reached a held-out correlation of 0.038.  Separating states
  is not the same as carrying structure a model can generalise over.
- The residual errors were not hard positions but positions whose deciding
  facts were absent from the input: an opponent already committed to fouling, a
  completed opponent flush described as a high card, and a kicker that decided
  a line.  All three are exactly computable, so they are computed here rather
  than learned.
- The opponent block is independent of which action the hero takes, so it is
  produced once per node and shared across every legal action.  That sharing is
  what keeps the encoder affordable relative to an exact solve.

Joker-specific additions, since the regular encoder is a 52-card design:

- Joker counts are carried explicitly for hero board, opponent board, and the
  unknown pool.  A wild card in the unseen pool changes the opponent outlook
  far more than any single natural card, and that fact is not recoverable from
  rank histograms.
- Fantasyland is the chain schedule, so the top row carries its EV from
  `ai/config/fl_ev.json` rather than a flat entry flag.

Target for the learned model is the exact uniform-deal EV produced by
`ai/tutor/t4_bb_exact_resolver.py` / the `t4_first_exact` Rust crate.

At T4 first seat the hero board is COMPLETE once an action is applied, so the
hero block is exact terminal fact, not an estimate.  Only the opponent block
is uncertain.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Iterable, Sequence

from ai.engine.encoding import ALL_CARDS
from ai.engine.game_engine import (
    check_fl_entry,
    evaluate_board_with_joker_constraint,
    evaluate_hand,
    get_bottom_royalty,
    get_middle_royalty,
    get_top_royalty,
    hand_category,
    is_joker,
)
from ai.mcts.rollout_evaluator import RolloutEvaluator

FEATURE_SCHEMA = "ofc_t4_first_features/v1"

ROW_NAMES = ("top", "middle", "bottom")
ROW_CAPACITY = (3, 5, 5)
CATEGORIES = 9  # high card .. straight flush
_B = 15
_MAX_ROYALTY = 25.0
_MAX_FL_EV = 63.5

# Block sizes.  Kept explicit so a drifted encoder is loud rather than silent.
# bust/royalty/FL (8) + 3 rows x spread (11) + joker count (1)
HERO_SIZE = 42
# 3 rows x (category histogram 9 + royalty + FL + room) + 3 suits + 2 joker
# counts + the joint completion block (8).
OPPONENT_SIZE = 49
# Self value the opponent assigns to a fouled board, matching the scorer.
FOUL_SELF_VALUE = -6.0
JOINT_SIZE_ADDED = 8
# locked-foul facts (5) + per-row comparison (3) + wins/scoop/bust/rooms (4)
JOINT_SIZE = 12
CONTEXT_SIZE = 6
FEATURE_SIZE = HERO_SIZE + OPPONENT_SIZE + JOINT_SIZE + CONTEXT_SIZE  # 101


def _row_royalty(row_index: int, cards: Sequence[str]) -> int:
    if row_index == 0:
        return get_top_royalty(list(cards))
    if row_index == 1:
        return get_middle_royalty(list(cards))
    return get_bottom_royalty(list(cards))


def _value_and_category(cards: Sequence[str], capacity: int) -> tuple[int, int]:
    """Encoded hand value and category for a complete row."""
    value = evaluate_hand(list(cards), capacity)
    return value, hand_category(value)


def partial_category(cards: Sequence[str], capacity: int) -> int:
    """Category of an incomplete row, by rank multiplicity plus jokers.

    An incomplete row cannot hold a straight or flush yet, so ranking it by
    multiplicity is exact rather than approximate, and it is a sound lower
    bound on what the row will finish as: every card already placed counts
    toward the final hand.  Jokers are counted as wild toward the best group.
    """
    if not cards:
        return 0
    if len(cards) == capacity:
        return _value_and_category(cards, capacity)[1]
    jokers = sum(1 for card in cards if is_joker(card))
    counts = Counter(card[0] for card in cards if not is_joker(card))
    best = (max(counts.values()) if counts else 0) + jokers
    pairs = sum(1 for count in counts.values() if count >= 2)
    if best >= 4:
        return 7
    if best == 3 and pairs >= 2:
        return 6
    if best >= 3:
        return 3
    if pairs >= 2:
        return 2
    if best == 2:
        return 1
    return 0


def _spread(value: int, category: int) -> list[float]:
    """Category one-hot plus the leading rank tiebreaks.

    The rank terms are what make a kicker that decides a line visible; the
    regular-track encoder found their absence in the residual errors.
    """
    out = [0.0] * (CATEGORIES + 2)
    out[min(category, CATEGORIES - 1)] = 1.0
    out[CATEGORIES] = ((value // (_B**4)) % _B) / 14.0
    out[CATEGORIES + 1] = ((value // (_B**3)) % _B) / 14.0
    return out


def unknown_pool(
    hero_board: Sequence[Sequence[str]],
    opponent_board: Sequence[Sequence[str]],
    dead: Iterable[str],
) -> list[str]:
    """Cards neither player can see, from the hero's information set."""
    seen = set()
    for rows in (hero_board, opponent_board):
        for row in rows:
            seen.update(row)
    seen.update(dead)
    return [card for card in ALL_CARDS if card not in seen]


def hero_block(hero_board: Sequence[Sequence[str]]) -> list[float]:
    """Exact terminal facts of the completed hero board."""
    evaluation = evaluate_board_with_joker_constraint(
        list(hero_board[0]), list(hero_board[1]), list(hero_board[2])
    )
    busted = bool(evaluation["busted"])
    rows = (evaluation["top"], evaluation["middle"], evaluation["bottom"])
    royalties = [0 if busted else _row_royalty(index, rows[index]) for index in range(3)]
    fl_qualified, fl_count = (False, 0)
    if not busted:
        fl_qualified, fl_count = check_fl_entry(list(rows[0]))
    fl_ev = float(RolloutEvaluator.FL_EV.get(fl_count, 0)) if fl_qualified else 0.0

    out: list[float] = [
        1.0 if busted else 0.0,
        sum(royalties) / _MAX_ROYALTY,
        royalties[0] / _MAX_ROYALTY,
        royalties[1] / _MAX_ROYALTY,
        royalties[2] / _MAX_ROYALTY,
        1.0 if fl_qualified else 0.0,
        fl_count / 17.0,
        fl_ev / _MAX_FL_EV,
    ]
    for index in range(3):
        value, category = _value_and_category(rows[index], ROW_CAPACITY[index])
        out.extend(_spread(value, category))
    out.append(
        sum(1 for row in hero_board for card in row if is_joker(card)) / 2.0
    )
    assert len(out) == HERO_SIZE, len(out)
    return out


def _top_fl_ev(top_cards: Sequence[str]) -> float:
    qualified, count = check_fl_entry(list(top_cards))
    return float(RolloutEvaluator.FL_EV.get(count, 0)) if qualified else 0.0


def _constrained_facts(rows: Sequence[Sequence[str]]) -> tuple[bool, float, float]:
    evaluation = evaluate_board_with_joker_constraint(
        list(rows[0]), list(rows[1]), list(rows[2])
    )
    if evaluation["busted"]:
        return True, 0.0, 0.0
    royalty = sum(
        _row_royalty(index, evaluation[name])
        for index, name in enumerate(("top", "middle", "bottom"))
    )
    return False, float(royalty), _top_fl_ev(evaluation["top"])


def opponent_joint_block(
    opponent_board: Sequence[Sequence[str]],
    pool: Sequence[str],
) -> list[float]:
    """Exact joint outlook of the opponent's two-card completion.

    The per-row histograms above treat each row independently, but the
    opponent places only two cards in total and chooses which row each goes
    to.  The diagnosis in
    ``ai/reports/t4_first_node_error_diagnosis_20260729/`` showed that this
    omission is the dominant deficiency: the opponent's foul probability alone
    correlates -0.470 with the model's within-root common error, and the
    current per-row features explain 0.9% of that error against 41.5% once
    these joint facts are present.  Fantasyland is also folded in here
    conditional on surviving, which fixes the structural error of adding a
    row-wise FL expectation independently of the foul.

    For every draw the opponent keeps two of three cards and takes the legal
    final maximizing its own royalty + FL EV, with a foul worth -6.  Line wins
    against the hero are deliberately excluded: they are action-dependent,
    while this block must stay shared across the node's actions.
    """
    rooms = [ROW_CAPACITY[index] - len(opponent_board[index]) for index in range(3)]
    open_rows = [index for index in range(3) if rooms[index] > 0]
    if sum(rooms) != 2 or not open_rows:
        raise ValueError("opponent must have exactly two open slots at T4 first")
    board_has_joker = any(is_joker(card) for row in opponent_board for card in row)

    fixed_value: dict[int, int] = {}
    fixed_royalty: dict[int, float] = {}
    for index in range(3):
        if rooms[index] == 0:
            fixed_value[index] = evaluate_hand(
                list(opponent_board[index]), ROW_CAPACITY[index]
            )
            fixed_royalty[index] = float(_row_royalty(index, opponent_board[index]))
    fixed_top_fl = _top_fl_ev(opponent_board[0]) if rooms[0] == 0 else None

    # Per-row completion tables: the enumeration below is then lookups.
    one_card: dict[int, dict[str, tuple[int, float, float]]] = {}
    two_card: dict[int, dict[tuple[str, str], tuple[int, float, float]]] = {}
    for index in open_rows:
        base = list(opponent_board[index])
        if rooms[index] == 1:
            one_card[index] = {}
            for card in pool:
                filled = base + [card]
                one_card[index][card] = (
                    evaluate_hand(filled, ROW_CAPACITY[index]),
                    float(_row_royalty(index, filled)),
                    _top_fl_ev(filled) if index == 0 else 0.0,
                )
        else:
            two_card[index] = {}
            for pair in combinations(pool, 2):
                key = tuple(sorted(pair))
                filled = base + list(key)
                two_card[index][key] = (
                    evaluate_hand(filled, ROW_CAPACITY[index]),
                    float(_row_royalty(index, filled)),
                    _top_fl_ev(filled) if index == 0 else 0.0,
                )

    def finals_for(kept: tuple[str, ...]):
        results = []
        base_values = [0, 0, 0]
        base_royalty = [0.0, 0.0, 0.0]
        for index in range(3):
            if rooms[index] == 0:
                base_values[index] = fixed_value[index]
                base_royalty[index] = fixed_royalty[index]
        if len(open_rows) == 2:
            first, second = open_rows
            for a, b in ((kept[0], kept[1]), (kept[1], kept[0])):
                values = list(base_values)
                royalties = list(base_royalty)
                fl = fixed_top_fl if fixed_top_fl is not None else 0.0
                value_a, royalty_a, fl_a = one_card[first][a]
                value_b, royalty_b, fl_b = one_card[second][b]
                values[first], royalties[first] = value_a, royalty_a
                values[second], royalties[second] = value_b, royalty_b
                if first == 0:
                    fl = fl_a
                elif second == 0:
                    fl = fl_b
                results.append((values, sum(royalties), fl, (a, b)))
        else:
            row = open_rows[0]
            values = list(base_values)
            royalties = list(base_royalty)
            fl = fixed_top_fl if fixed_top_fl is not None else 0.0
            key = tuple(sorted(kept))
            value, royalty, fl_row = two_card[row][key]
            values[row], royalties[row] = value, royalty
            if row == 0:
                fl = fl_row
            results.append((values, sum(royalties), fl, key))
        return results

    def self_value(values, royalty, fl, placed) -> tuple[float, float, float]:
        """Return (value, royalty, FL) that are always mutually consistent.

        The royalty and Fantasyland parts must come from the same evaluation
        as the value.  Reading them off the unconstrained per-row tables while
        the value came from the constrained board overstates both whenever a
        joker was pushed down, which is the divergence the Rust parity check
        caught.
        """
        if values[0] <= values[1] <= values[2]:
            return royalty + fl, royalty, fl
        # A raw ordering violation is only rescuable through the canonical
        # joker constraint, so fall back to it just when a joker is present.
        if not (board_has_joker or any(is_joker(card) for card in placed)):
            return FOUL_SELF_VALUE, 0.0, 0.0
        rows_final = [list(opponent_board[index]) for index in range(3)]
        if len(open_rows) == 2:
            rows_final[open_rows[0]].append(placed[0])
            rows_final[open_rows[1]].append(placed[1])
        else:
            rows_final[open_rows[0]].extend(placed)
        busted, constrained_royalty, constrained_fl = _constrained_facts(rows_final)
        if busted:
            return FOUL_SELF_VALUE, 0.0, 0.0
        return constrained_royalty + constrained_fl, constrained_royalty, constrained_fl

    best_values: list[float] = []
    fouls = 0
    survive_royalty = 0.0
    survive_fl = 0.0
    survivors = 0
    for draw in combinations(pool, 3):
        best = None
        best_parts = (0.0, 0.0)
        for kept in combinations(draw, 2):
            for values, royalty, fl, placed in finals_for(kept):
                value, used_royalty, used_fl = self_value(values, royalty, fl, placed)
                if best is None or value > best:
                    best = value
                    best_parts = (used_royalty, used_fl)
        best_values.append(float(best))
        if best == FOUL_SELF_VALUE:
            fouls += 1
        else:
            survivors += 1
            survive_royalty += best_parts[0]
            survive_fl += best_parts[1]

    count = len(best_values)
    if count == 0:
        raise ValueError("opponent joint enumeration is empty")
    mean = sum(best_values) / count
    variance = sum((value - mean) ** 2 for value in best_values) / count
    survive_denominator = max(survivors, 1)
    return [
        fouls / count,
        mean / _MAX_ROYALTY,
        (survive_royalty / survive_denominator) / _MAX_ROYALTY,
        (survive_fl / survive_denominator) / _MAX_FL_EV,
        (variance**0.5) / _MAX_ROYALTY,
        sum(1 for value in best_values if value >= 6.0) / count,
        sum(1 for value in best_values if value >= 15.0) / count,
        (
            sum(value for value in best_values if value > FOUL_SELF_VALUE)
            / survive_denominator
        )
        / _MAX_ROYALTY,
    ]


def opponent_block(
    opponent_board: Sequence[Sequence[str]],
    pool: Sequence[str],
    joint_block: Sequence[float] | None = None,
) -> tuple[list[float], list[int]]:
    """Outlook over the opponent's remaining draw, action-independent.

    Per-row completion histograms are exact in isolation: the row capacities
    are fixed, so every open slot must be filled and no card can move once
    placed.  What they cannot express is the joint condition, which the joint
    block below supplies from exact facts.
    """
    rooms = [ROW_CAPACITY[index] - len(opponent_board[index]) for index in range(3)]
    out: list[float] = []
    categories_now: list[int] = []

    for index in range(3):
        cards = list(opponent_board[index])
        room = rooms[index]
        categories_now.append(partial_category(cards, ROW_CAPACITY[index]))
        histogram = [0.0] * CATEGORIES
        royalty_total = 0.0
        fl_total = 0.0
        samples = 0
        if room == 0:
            value, category = _value_and_category(cards, ROW_CAPACITY[index])
            histogram[min(category, CATEGORIES - 1)] = 1.0
            royalty_total = float(_row_royalty(index, cards))
            if index == 0:
                qualified, count = check_fl_entry(cards)
                fl_total = (
                    float(RolloutEvaluator.FL_EV.get(count, 0)) if qualified else 0.0
                )
            samples = 1
        else:
            for extra in combinations(pool, room):
                filled = cards + list(extra)
                value, category = _value_and_category(filled, ROW_CAPACITY[index])
                histogram[min(category, CATEGORIES - 1)] += 1.0
                royalty_total += float(_row_royalty(index, filled))
                if index == 0:
                    qualified, count = check_fl_entry(filled)
                    if qualified:
                        fl_total += float(RolloutEvaluator.FL_EV.get(count, 0))
                samples += 1
        denominator = max(samples, 1)
        out.extend(bin_count / denominator for bin_count in histogram)
        out.append(royalty_total / denominator / _MAX_ROYALTY)
        out.append(fl_total / denominator / _MAX_FL_EV)
        out.append(room / 5.0)

    for index in range(3):
        suits = Counter(
            card[1] for card in opponent_board[index] if not is_joker(card)
        )
        out.append((max(suits.values()) if suits else 0) / 5.0)
    out.append(
        sum(1 for row in opponent_board for card in row if is_joker(card)) / 2.0
    )
    out.append(sum(1 for card in pool if is_joker(card)) / 2.0)
    # The Rust solver already enumerates the opponent's pair terminals, so it
    # can hand this block over for free; recompute only when it did not.
    if joint_block is None:
        joint_block = opponent_joint_block(opponent_board, pool)
    if len(joint_block) != JOINT_SIZE_ADDED:
        raise ValueError("supplied joint block has the wrong width")
    out.extend(float(value) for value in joint_block)
    assert len(out) == OPPONENT_SIZE, len(out)
    return out, categories_now


def joint_block(
    hero_board: Sequence[Sequence[str]],
    opponent_board: Sequence[Sequence[str]],
    opponent_categories: Sequence[int],
) -> list[float]:
    """Exactly computable facts the per-row histograms cannot express."""
    rooms = [ROW_CAPACITY[index] - len(opponent_board[index]) for index in range(3)]
    hero_eval = evaluate_board_with_joker_constraint(
        list(hero_board[0]), list(hero_board[1]), list(hero_board[2])
    )
    hero_rows = (hero_eval["top"], hero_eval["middle"], hero_eval["bottom"])
    hero_values = [
        evaluate_hand(list(hero_rows[index]), ROW_CAPACITY[index]) for index in range(3)
    ]

    # A full row can no longer move, so a row below it that already wins fixes
    # a foul regardless of the remaining deal.
    locked_middle = rooms[2] == 0 and opponent_categories[1] > opponent_categories[2]
    locked_top = rooms[1] == 0 and opponent_categories[0] > opponent_categories[1]
    out: list[float] = [
        1.0 if locked_middle else 0.0,
        1.0 if locked_top else 0.0,
        1.0 if (locked_middle or locked_top) else 0.0,
        (opponent_categories[1] - opponent_categories[2]) / 8.0,
        (opponent_categories[0] - opponent_categories[1]) / 8.0,
    ]

    # Per-row comparison against the opponent's currently-made hand.  This is a
    # lower bound on the opponent, so it states which lines the hero already
    # beats outright.
    wins = 0
    for index in range(3):
        opponent_partial = list(opponent_board[index])
        if rooms[index] == 0:
            opponent_value = evaluate_hand(opponent_partial, ROW_CAPACITY[index])
            sign = (hero_values[index] > opponent_value) - (
                hero_values[index] < opponent_value
            )
        else:
            hero_category = hand_category(hero_values[index])
            sign = (hero_category > opponent_categories[index]) - (
                hero_category < opponent_categories[index]
            )
        out.append(float(sign))
        wins += 1 if sign > 0 else 0
    out.append(wins / 3.0)
    out.append(1.0 if wins == 3 else 0.0)
    out.append(1.0 if bool(hero_eval["busted"]) else 0.0)
    out.append(sum(rooms) / 5.0)
    assert len(out) == JOINT_SIZE, len(out)
    return out


def context_block(pool: Sequence[str], dead_count: int) -> list[float]:
    ranks = Counter(card[0] for card in pool if not is_joker(card))
    high = sum(count for rank, count in ranks.items() if rank in ("A", "K", "Q"))
    return [
        len(pool) / 54.0,
        dead_count / 6.0,
        high / max(len(pool), 1),
        sum(1 for card in pool if is_joker(card)) / 2.0,
        (max(ranks.values()) if ranks else 0) / 4.0,
        len(ranks) / 13.0,
    ]


class NodeCache:
    """Action-independent part of one T4 first-seat node.

    The unknown pool is the same for every legal action: two of the three
    drawn cards land on the hero board and the third is discarded, so all
    three leave the pool either way.  The opponent block and the context
    block therefore depend only on the node, and computing them once is what
    keeps the encoder cheap relative to an exact solve.
    """

    __slots__ = ("opponent_board", "pool", "_opponent", "_categories", "_context")

    def __init__(
        self,
        opponent_board: Sequence[Sequence[str]],
        pool: Sequence[str],
        dead_count: int,
        joint_block: Sequence[float] | None = None,
    ) -> None:
        self.opponent_board = tuple(tuple(row) for row in opponent_board)
        self.pool = tuple(pool)
        self._opponent, self._categories = opponent_block(
            self.opponent_board, self.pool, joint_block
        )
        self._context = context_block(self.pool, dead_count)

    @classmethod
    def for_root(
        cls,
        hero_board_11: Sequence[Sequence[str]],
        opponent_board: Sequence[Sequence[str]],
        draw: Sequence[str],
        prior_dead: Sequence[str],
        joint_block: Sequence[float] | None = None,
    ) -> "NodeCache":
        pool = unknown_pool(
            hero_board_11,
            opponent_board,
            list(prior_dead) + list(draw),
        )
        return cls(opponent_board, pool, len(prior_dead) + 1, joint_block)


def encode_action(
    hero_final_board: Sequence[Sequence[str]],
    cache: NodeCache,
) -> list[float]:
    """Feature vector for one legal action, reusing the shared node block."""
    if sum(len(row) for row in hero_final_board) != 13:
        raise ValueError("hero board must be complete (13 cards) at T4 first seat")
    hero = hero_block(hero_final_board)
    joint = joint_block(hero_final_board, cache.opponent_board, cache._categories)
    vector = hero + list(cache._opponent) + joint + list(cache._context)
    if len(vector) != FEATURE_SIZE:
        raise AssertionError(f"feature size drifted: {len(vector)} != {FEATURE_SIZE}")
    return vector


def encode(
    hero_final_board: Sequence[Sequence[str]],
    opponent_board: Sequence[Sequence[str]],
    dead: Sequence[str],
) -> list[float]:
    """Full feature vector for one (hero action applied, opponent, deck) state.

    ``hero_final_board`` must already be complete (13 cards): at T4 first seat
    the hero's board is fixed once the action is chosen, which is what makes
    the hero block exact.
    """
    if sum(len(row) for row in hero_final_board) != 13:
        raise ValueError("hero board must be complete (13 cards) at T4 first seat")
    if sum(len(row) for row in opponent_board) != 11:
        raise ValueError("opponent board must hold 11 cards at T4 first seat")
    pool = unknown_pool(hero_final_board, opponent_board, dead)
    hero = hero_block(hero_final_board)
    opponent, categories = opponent_block(opponent_board, pool)
    joint = joint_block(hero_final_board, opponent_board, categories)
    context = context_block(pool, len(dead))
    vector = hero + opponent + joint + context
    if len(vector) != FEATURE_SIZE:
        raise AssertionError(f"feature size drifted: {len(vector)} != {FEATURE_SIZE}")
    return vector


__all__ = [
    "CATEGORIES",
    "FEATURE_SCHEMA",
    "FEATURE_SIZE",
    "HERO_SIZE",
    "JOINT_SIZE",
    "OPPONENT_SIZE",
    "CONTEXT_SIZE",
    "encode",
    "hero_block",
    "joint_block",
    "opponent_block",
    "context_block",
    "partial_category",
    "unknown_pool",
]
