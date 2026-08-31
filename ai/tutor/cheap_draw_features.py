"""Deterministic draw descriptors -- the Python twin of `cheap_draw_block`.

Sixteen dims appended to the 96-dim ranker vector to make width 112.  Every
one is a count over the board and the unseen pool: no sampling, no
completions, so a candidate costs what a 96-dim candidate costs.  That is the
whole point.  The 110-dim encoder reached its accuracy through a
400-completion joint block costing ~50 ms a candidate, which the
no-thinking-time-at-serve rule forbids (owner, 2026-08-30); most of what that
block summarised is reachable by counting.

Every definition here mirrors `evaluator.rs::cheap_draw_block` line for line.
The two are compared on raw vectors over every audited board, because a drift
between them means serving reads a vector the model never trained on and
nothing downstream would say so.

Dim order, which is part of the interface:

     0-1  middle/bottom flush progress
     2-3  middle/bottom flush liveness
     4-5  middle/bottom straight-window occupancy
     6-7  middle/bottom straight-flush-window occupancy
       8  bottom royal-window occupancy
    9-11  jokers placed in top/middle/bottom
      12  top's highest placed rank
      13  top's progress toward a QQ+ pair
      14  top's pair liveness
      15  jokers anywhere on the board
"""
from __future__ import annotations

CHEAP_DRAW_SIZE = 16
CHEAP_DRAW_V2_SIZE = 24
RANKS = "23456789TJQKA"
SUITS = "shdc"

# The five-rank runs a straight can occupy, wheel first.
STRAIGHT_WINDOWS = [(14, 2, 3, 4, 5)] + [tuple(range(low, low + 5)) for low in range(2, 11)]


def parse(card: str):
    """(rank, suit) for a natural card, or None for a joker."""
    if card in ("X1", "X2"):
        return None
    return RANKS.index(card[0]) + 2, SUITS.index(card[1])


def _naturals(row):
    return [c for c in (parse(card) for card in row) if c is not None]


def _jokers(row):
    return sum(1 for card in row if card in ("X1", "X2"))


def window_occupancy(row, jokers: int, single_suit: bool, windows) -> float:
    """How full a row is toward its best five-rank window, as a fraction of 5.

    A window scores zero unless every placed natural lies inside it with a
    distinct rank: a row holding an outside card, or a pair, cannot become
    that straight at all.  Jokers count toward every window, having no rank to
    conflict.  `single_suit` additionally requires one suit, which turns the
    same walk into straight-flush (and over one window, royal) progress.
    """
    naturals = _naturals(row)
    if single_suit and naturals and any(s != naturals[0][1] for _r, s in naturals):
        return 0.0
    best = 0.0
    for window in windows:
        filled = set()
        reachable = True
        for rank, _suit in naturals:
            if rank not in window or rank in filled:
                reachable = False
                break
            filled.add(rank)
        if reachable:
            best = max(best, (len(filled) + jokers) / 5.0)
    return min(best, 1.0)


def best_suit(row):
    """(count, suit) of the suit a row is closest to a flush in.

    Ties break on the lowest suit index -- arbitrary, but it must be written
    down and obeyed on both sides, because the liveness dim reads the deck
    through whichever suit this returns.
    """
    counts = [0, 0, 0, 0]
    for _rank, suit in _naturals(row):
        counts[suit] += 1
    chosen = None
    for suit in range(4):
        if counts[suit] > 0 and (chosen is None or counts[suit] > counts[chosen]):
            chosen = suit
    return (0 if chosen is None else counts[chosen]), chosen


def cheap_draw_block(rows, pool) -> list[float]:
    unseen_suit = [0, 0, 0, 0]
    unseen_rank = [0] * 15
    for card in pool:
        parsed = parse(card)
        if parsed is not None:
            rank, suit = parsed
            unseen_suit[suit] += 1
            unseen_rank[rank] += 1
    jokers = [_jokers(rows[r]) for r in range(3)]

    out: list[float] = []
    # 0-3 flush progress and liveness for middle and bottom.  The top row
    # holds three cards and cannot make a flush, so it is not asked.
    suited = [None, best_suit(rows[1]), best_suit(rows[2])]
    for row in (1, 2):
        out.append((suited[row][0] + jokers[row]) / 5.0)
    for row in (1, 2):
        suit = suited[row][1]
        # No natural in the row, so no suit is established yet.
        out.append(0.0 if suit is None else min(unseen_suit[suit] / 8.0, 1.0))
    # 4-7 straight and straight-flush windows.
    for row in (1, 2):
        out.append(window_occupancy(rows[row], jokers[row], False, STRAIGHT_WINDOWS))
    for row in (1, 2):
        out.append(window_occupancy(rows[row], jokers[row], True, STRAIGHT_WINDOWS))
    # 8 the royal run, bottom only: the one window worth its own dim.
    out.append(window_occupancy(rows[2], jokers[2], True, STRAIGHT_WINDOWS[9:10]))
    # 9-11 jokers per row.
    for row in range(3):
        out.append(jokers[row] / 2.0)
    # 12-14 the top row, which is what Fantasyland entry is decided on.
    top = _naturals(rows[0])
    top_max = max((rank for rank, _suit in top), default=None)
    out.append(0.0 if top_max is None else top_max / 14.0)
    queens_up = max((sum(1 for rank, _s in top if rank == k) for k in (12, 13, 14)), default=0)
    out.append(min(queens_up + jokers[0], 2) / 2.0)
    out.append(0.0 if top_max is None else unseen_rank[top_max] / 3.0)
    # 15 jokers anywhere on the board.
    out.append(sum(jokers) / 2.0)

    if len(out) != CHEAP_DRAW_SIZE:
        raise AssertionError(f"cheap draw block is {len(out)} dims, not {CHEAP_DRAW_SIZE}")
    return out


# ---------------------------------------------------------------------------
# v2: the same idea without the buckets.
#
# v1 reached 96-grade serving cost but quantised every dim to multiples of
# 1/5, 1/2, 1/3 and 1/8, so distinct openings collided: on the audit holdout
# 14 of 97 roots had the referee's best placement carrying a vector identical
# to a strictly worse one, which no training can separate.  The sampled
# 110-dim block collided on none -- not because it sampled, but because it
# emitted continuous high-entropy values.
#
# So v2 keeps the determinism and drops the buckets: raw sums of ranks and
# squared ranks per row (which nearly determine which cards went where), a
# suit signature, and raw out-counts for flushes, straights and pairs.  On the
# same holdout it collides on zero roots and zero candidates.
#
# Mirrored in `evaluator.rs::cheap_draw_block_v2`.  Arithmetic stays in
# Python floats (doubles) and is narrowed to f32 only when stored, which is
# what the Rust side does too -- the divisors are not exactly representable,
# so computing in f32 would round differently and break parity.
# ---------------------------------------------------------------------------

def _census(pool):
    unseen_rank = [0] * 15
    unseen_suit = [0, 0, 0, 0]
    for card in pool:
        p = parse(card)
        if p is not None:
            unseen_rank[p[0]] += 1
            unseen_suit[p[1]] += 1
    return unseen_rank, unseen_suit


def _best_window(row, jokers, unseen_rank):
    """(fill fraction, outs filling the gaps) for the row's best straight window.

    Ties on fill break toward the window with more outs, then on window order,
    so the pair is a function of the board and not of iteration luck.
    """
    naturals = _naturals(row)
    best = (-1.0, -1.0)
    for window in STRAIGHT_WINDOWS:
        filled, ok = set(), True
        for rank, _suit in naturals:
            if rank not in window or rank in filled:
                ok = False
                break
            filled.add(rank)
        if not ok:
            continue
        outs = sum(unseen_rank[k] for k in window if k not in filled)
        cand = (min((len(filled) + jokers) / 5.0, 1.0), outs / 20.0)
        if cand > best:
            best = cand
    return best if best[0] >= 0 else (0.0, 0.0)


def cheap_draw_block_v2(rows, pool):
    unseen_rank, unseen_suit = _census(pool)
    jokers = [_jokers(rows[r]) for r in range(3)]
    out = []
    # --- rank/suit content per row (12) -------------------------------------
    # Raw sums, not buckets: two openings that differ only by which card went
    # where differ here, which is exactly the separation v1 lacked.
    for r in range(3):
        nat = _naturals(rows[r])
        ranks = [rk for rk, _s in nat]
        out.append(sum(ranks) / 70.0)
        out.append(sum(k * k for k in ranks) / 980.0)
        out.append((max(ranks) / 14.0) if ranks else 0.0)
        out.append(sum(s + 1 for _r, s in nat) / 20.0)
    # --- flush outs, middle and bottom (4) ----------------------------------
    for r in (1, 2):
        count, suit = best_suit(rows[r])
        outs = unseen_suit[suit] if suit is not None else 0
        out.append(outs / 13.0)
        out.append((count + jokers[r]) / 5.0 * outs / 13.0)
    # --- straight fill and its outs, middle and bottom (4) ------------------
    for r in (1, 2):
        fill, outs = _best_window(rows[r], jokers[r], unseen_rank)
        out.append(fill)
        out.append(outs)
    # --- pairing outs per row (3) -------------------------------------------
    for r in range(3):
        out.append(sum(unseen_rank[rk] for rk, _s in _naturals(rows[r])) / 9.0)
    # --- top row Fantasyland material (1) -----------------------------------
    top = _naturals(rows[0])
    qq = sum(unseen_rank[k] for k in (12, 13, 14) if any(rk == k for rk, _s in top))
    out.append(qq / 9.0)
    return out
