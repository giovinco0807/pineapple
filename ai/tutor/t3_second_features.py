"""Feature encoding for a learned T3 second-seat evaluator.

At T3 second seat BTN holds 9 cards with four open slots and, unlike T4, its
board is NOT complete after the action: two slots remain for T4.  So the acting
seat's block cannot be exact terminal fact the way the T4 hero block was; it is
a partial-hand description plus the exactly-computable facts about what the row
can still become.

The design reuses what the T4 diagnosis established, since the same structural
omissions apply one street earlier:

- The dominant missing fact at T4 was the opponent's JOINT completion, because
  the opponent places two cards total and chooses which row each goes to.  The
  same is true here for BB, which is already at 11 cards with two open slots --
  so BB's joint block is computed with the same enumeration.
- Per-row completion histograms are exact in isolation but overstate what can
  be achieved simultaneously; the joint block is what corrects that.
- Deck-composition detail and joker-specific terms were measured and refuted at
  T4, so they are not carried here either beyond the joker counts.

The acting seat has FOUR open slots after this street's two placements, so its
own outlook needs a reachability description rather than a finished value.
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

from ai.engine.game_engine import check_fl_entry, evaluate_hand, hand_category, is_joker
from ai.tutor.t4_first_features import (
    CATEGORIES,
    ROW_CAPACITY,
    _MAX_FL_EV,
    _MAX_ROYALTY,
    _row_royalty,
    _spread,
    context_block,
    opponent_block,
    partial_category,
    unknown_pool,
)

FEATURE_SCHEMA = "ofc_t3_second_features/v1"

# 3 rows x (spread 11 + royalty + room) + slack/lock (4) + suit + FL (2)
# + joker count + total room
ACTOR_SIZE = 48
# BB's block reuses the T4 opponent encoder (per-row histograms + joint).
OPPONENT_SIZE = 49
COMPARISON_SIZE = 10
CONTEXT_SIZE = 6
FEATURE_SIZE = ACTOR_SIZE + OPPONENT_SIZE + COMPARISON_SIZE + CONTEXT_SIZE  # 113


def actor_block(rows: Sequence[Sequence[str]], pool: Sequence[str]) -> list[float]:
    """The acting seat's own board after placing two, with two slots left.

    Not terminal fact: two cards are still to come, so each row carries its
    made value, its room, and the royalty/Fantasyland it would pay if it
    finished as it stands.  Those last terms are a sound lower bound, since
    every card already placed counts toward the final hand.
    """
    rooms = [ROW_CAPACITY[index] - len(rows[index]) for index in range(3)]
    out: list[float] = []
    categories: list[int] = []
    for index in range(3):
        cards = list(rows[index])
        capacity = ROW_CAPACITY[index]
        category = partial_category(cards, capacity)
        categories.append(category)
        if len(cards) == capacity:
            value = evaluate_hand(cards, capacity)
            out.extend(_spread(value, hand_category(value)))
            out.append(float(_row_royalty(index, cards)) / _MAX_ROYALTY)
        else:
            spread = [0.0] * (CATEGORIES + 2)
            spread[min(category, CATEGORIES - 1)] = 1.0
            out.extend(spread)
            out.append(0.0)
        out.append(rooms[index] / 5.0)

    # Ordering slack: a violation that survives to the end is a foul, and with
    # rows still open the slack is what says how much room there is to fix it.
    out.append((categories[1] - categories[2]) / 8.0)
    out.append((categories[0] - categories[1]) / 8.0)
    out.append(1.0 if categories[0] > categories[1] else 0.0)
    out.append(1.0 if categories[1] > categories[2] else 0.0)
    suits = Counter(card[1] for card in rows[2] if not is_joker(card))
    out.append((max(suits.values()) if suits else 0) / 5.0)
    fl_qualified, fl_count = (
        check_fl_entry(list(rows[0])) if len(rows[0]) == 3 else (False, 0)
    )
    out.append(1.0 if fl_qualified else 0.0)
    out.append(fl_count / 17.0)
    out.append(sum(1 for row in rows for card in row if is_joker(card)) / 2.0)
    out.append(sum(rooms) / 5.0)
    assert len(out) == ACTOR_SIZE, len(out)
    return out, categories


def comparison_block(
    actor_categories: Sequence[int],
    opponent_categories: Sequence[int],
    opponent_rows: Sequence[Sequence[str]],
) -> list[float]:
    """Row-by-row standing against BB's made hand, a lower bound on BB."""
    out: list[float] = []
    wins = 0
    for index in range(3):
        sign = (actor_categories[index] > opponent_categories[index]) - (
            actor_categories[index] < opponent_categories[index]
        )
        out.append(float(sign))
        wins += 1 if sign > 0 else 0
        out.append((actor_categories[index] - opponent_categories[index]) / 8.0)
    out.append(wins / 3.0)
    out.append(1.0 if wins == 3 else 0.0)
    out.append(
        sum(ROW_CAPACITY[index] - len(opponent_rows[index]) for index in range(3)) / 5.0
    )
    out.append(1.0 if wins == 0 else 0.0)
    assert len(out) == COMPARISON_SIZE, len(out)
    return out


class T3NodeCache:
    """BB's block and the context block, shared across BTN's actions.

    BB's board and the pool do not depend on which action BTN takes -- BTN's
    two placed cards and its discard all leave the pool either way -- so this
    is computed once per root.
    """

    __slots__ = ("bb_rows", "pool", "opponent", "categories", "context")

    def __init__(
        self,
        bb_rows: Sequence[Sequence[str]],
        btn_rows_9: Sequence[Sequence[str]],
        btn_dead_2: Sequence[str],
        draw: Sequence[str],
        joint_block: Sequence[float] | None = None,
    ) -> None:
        self.bb_rows = tuple(tuple(row) for row in bb_rows)
        self.pool = tuple(
            unknown_pool(bb_rows, btn_rows_9, list(btn_dead_2) + list(draw))
        )
        self.opponent, self.categories = opponent_block(
            self.bb_rows, self.pool, joint_block
        )
        self.context = context_block(self.pool, len(btn_dead_2) + 1)


def encode_action(
    btn_rows_11: Sequence[Sequence[str]],
    cache: T3NodeCache,
) -> list[float]:
    if sum(len(row) for row in btn_rows_11) != 11:
        raise ValueError("BTN board must hold 11 cards after a T3 placement")
    actor, actor_categories = actor_block(btn_rows_11, cache.pool)
    comparison = comparison_block(actor_categories, cache.categories, cache.bb_rows)
    vector = actor + list(cache.opponent) + comparison + list(cache.context)
    if len(vector) != FEATURE_SIZE:
        raise AssertionError(f"feature size drifted: {len(vector)} != {FEATURE_SIZE}")
    return vector


__all__ = [
    "ACTOR_SIZE",
    "COMPARISON_SIZE",
    "CONTEXT_SIZE",
    "FEATURE_SCHEMA",
    "FEATURE_SIZE",
    "OPPONENT_SIZE",
    "T3NodeCache",
    "actor_block",
    "comparison_block",
    "encode_action",
]
