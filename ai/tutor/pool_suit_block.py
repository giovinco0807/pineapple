"""Pool suit composition block (7 dims) -- the fact the worst hands missed.

Reading the T2 v2 gate's worst-regret roots showed a repeated shape: the
model breaks four-card suited draws, especially in the middle row.  The
encoder carries each row's own suit concentration but not the other half of
a flush draw's value -- how many cards of that suit the unseen pool still
holds.  Both halves are exactly computable, so they get encoded rather than
left for the net to guess.

Dims: pool count per suit (4, /13) + per-row outs of the row's dominant
suit (3, /13).
"""
from __future__ import annotations

SUITS = "shdc"


def pool_suit_block(rows, pool) -> list[float]:
    pool_counts = {suit: 0 for suit in SUITS}
    for card in pool:
        if card not in ("X1", "X2"):
            pool_counts[card[1]] += 1
    out = [pool_counts[suit] / 13.0 for suit in SUITS]
    for row in rows:
        row_counts = {suit: 0 for suit in SUITS}
        for card in row:
            if card not in ("X1", "X2"):
                row_counts[card[1]] += 1
        if any(row_counts.values()):
            dominant = max(SUITS, key=lambda suit: row_counts[suit])
            out.append(pool_counts[dominant] / 13.0)
        else:
            out.append(0.0)
    return out
