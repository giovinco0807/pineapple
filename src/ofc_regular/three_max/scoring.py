"""3-max settlement: pairwise sum for the AI, ordered capping for the product.

Contract R4.  The terminal score of a 3-max hand is the sum of three
independent heads-up comparisons, so the pairwise kernel is reused verbatim
from :mod:`ofc_regular.teacher` rather than re-derived here -- a second copy of
those rules is exactly how the two tracks would silently drift apart.

Two settlement modes exist and they are deliberately different objects:

``pair_scores`` / ``settle_hand(stacks=None)``
    Infinite stacks.  This is what the AI optimises: ``score(hero) = sum over
    the two opponents``, order-independent and zero-sum.

``settle_hand(stacks=...)``
    Finite stacks, settled left-vs-right, then left-vs-button, then
    right-vs-button.  A player emptied by an earlier pair cannot pay a later
    one, so the order is load-bearing: a seat's realised total stops matching
    its raw pairwise sum.  Chips are still conserved -- every transfer is
    applied symmetrically, so ``transferred_totals`` sums to zero and the
    stacks sum to what they started at -- what capping changes is the split,
    not the total.  This is a product rule for the game implementation; it must
    never enter a teacher label, a feature, or a gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..evaluator import BoardScore, score_board
from ..state import Board
from ..teacher import _heads_up_terminal_score as _pair_kernel
from .seating import ACT_ORDER, SETTLEMENT_ORDER, Seat3

FL_EV_3MAX_CONFIG_RELPATH = "configs/fl_ev_3max_v0_bootstrap.json"
FL_EV_3MAX_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "fl_ev_3max_v0_bootstrap.json"
)
# Per-pair, not per-hand.  See the config's provenance.units_warning.
FALLBACK_FL_EV_PER_PAIR = {14: 9.6}
OPPONENTS_PER_PLAYER = 2


def load_fl_ev_3max(path: str | Path | None = None) -> dict[int, float]:
    """Read the pinned 3-max PER-PAIR Fantasyland EV table.

    The file also records the hand total.  A file whose total is not
    ``opponents_per_player`` times the per-pair value is rejected: that
    mismatch is the 2x error that would otherwise ride silently into every
    label in the cascade.
    """
    if path is None:
        config_path = FL_EV_3MAX_CONFIG_PATH
        if not config_path.exists():
            return dict(FALLBACK_FL_EV_PER_PAIR)
    else:
        config_path = Path(path)

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    raw = payload.get("fl_ev", {})
    if not raw:
        raise ValueError(f"{config_path} has no fl_ev table")
    units = payload.get("fl_ev_units")
    if units != "per_pair":
        raise ValueError(
            f"{config_path} declares fl_ev_units={units!r}; the 3-max scoring "
            "kernel consumes per-pair values only"
        )
    per_pair = {int(cards): float(value) for cards, value in raw.items()}

    opponents = int(payload.get("opponents_per_player", OPPONENTS_PER_PLAYER))
    if opponents != OPPONENTS_PER_PLAYER:
        raise ValueError(
            f"{config_path} is for {opponents} opponents, not {OPPONENTS_PER_PLAYER}"
        )
    totals = payload.get("fl_ev_hand_total")
    if totals is not None:
        declared = {int(cards): float(value) for cards, value in totals.items()}
        if set(declared) != set(per_pair):
            raise ValueError(f"{config_path} fl_ev_hand_total covers different card counts")
        for cards, value in declared.items():
            expected = per_pair[cards] * opponents
            if abs(value - expected) > 1e-9:
                raise ValueError(
                    f"{config_path} fl_ev_hand_total[{cards}]={value} contradicts "
                    f"{opponents} x per-pair {per_pair[cards]} = {expected}"
                )
    return per_pair


DEFAULT_FL_EV_PER_PAIR = load_fl_ev_3max()


def fl_ev_hand_total(fl_ev_per_pair: Mapping[int, float]) -> dict[int, float]:
    """Convert per-pair Fantasyland EV to the hand-level value of entering."""
    return {cards: value * OPPONENTS_PER_PLAYER for cards, value in fl_ev_per_pair.items()}


@dataclass(frozen=True)
class PairSettlement:
    """One settled pair, always reported from ``seats[0]``'s perspective."""

    seats: tuple[Seat3, Seat3]
    raw: float
    transferred: float

    @property
    def capped(self) -> bool:
        return abs(self.transferred - self.raw) > 1e-12


@dataclass(frozen=True)
class HandSettlement:
    pairs: tuple[PairSettlement, ...]
    raw_totals: Mapping[Seat3, float]
    transferred_totals: Mapping[Seat3, float]
    final_stacks: Mapping[Seat3, float] | None

    @property
    def any_capped(self) -> bool:
        return any(pair.capped for pair in self.pairs)


def pair_scores(
    boards: Mapping[Seat3, Board],
    fl_ev_per_pair: Mapping[int, float] | None = None,
) -> dict[tuple[Seat3, Seat3], float]:
    """Raw heads-up score of every settlement pair, from the first seat's view."""
    scores = _board_scores(boards)
    table = dict(fl_ev_per_pair or DEFAULT_FL_EV_PER_PAIR)
    return {
        pair: _pair_kernel(scores[pair[0]], scores[pair[1]], table)
        for pair in SETTLEMENT_ORDER
    }


def settle_hand(
    boards: Mapping[Seat3, Board],
    *,
    fl_ev_per_pair: Mapping[int, float] | None = None,
    stacks: Mapping[Seat3, float] | None = None,
) -> HandSettlement:
    """Settle a finished 3-max hand.

    With ``stacks=None`` (the AI's assumption, contract R4) the result is the
    plain pairwise sum: order-independent and zero-sum.  With stacks supplied
    the pairs settle in :data:`SETTLEMENT_ORDER` and each transfer is capped by
    what the losing side still has.
    """
    raw = pair_scores(boards, fl_ev_per_pair)
    raw_totals: dict[Seat3, float] = {seat: 0.0 for seat in ACT_ORDER}
    for (first, second), value in raw.items():
        raw_totals[first] += value
        raw_totals[second] -= value

    if stacks is None:
        pairs = tuple(
            PairSettlement(seats=pair, raw=raw[pair], transferred=raw[pair])
            for pair in SETTLEMENT_ORDER
        )
        return HandSettlement(
            pairs=pairs,
            raw_totals=raw_totals,
            transferred_totals=dict(raw_totals),
            final_stacks=None,
        )

    remaining = _validated_stacks(stacks)
    transferred_totals: dict[Seat3, float] = {seat: 0.0 for seat in ACT_ORDER}
    settled: list[PairSettlement] = []
    for pair in SETTLEMENT_ORDER:
        first, second = pair
        value = raw[pair]
        if value > 0:
            transfer = min(value, remaining[second])
        elif value < 0:
            transfer = -min(-value, remaining[first])
        else:
            transfer = 0.0
        remaining[first] += transfer
        remaining[second] -= transfer
        transferred_totals[first] += transfer
        transferred_totals[second] -= transfer
        settled.append(PairSettlement(seats=pair, raw=value, transferred=transfer))

    return HandSettlement(
        pairs=tuple(settled),
        raw_totals=raw_totals,
        transferred_totals=transferred_totals,
        final_stacks=remaining,
    )


def _board_scores(boards: Mapping[Seat3, Board]) -> dict[Seat3, BoardScore]:
    missing = set(ACT_ORDER) - set(boards)
    if missing:
        raise ValueError(f"missing 3-max boards for seats: {sorted(missing)}")
    unknown = set(boards) - set(ACT_ORDER)
    if unknown:
        raise ValueError(f"unknown 3-max seats: {sorted(unknown)}")

    seen: set[str] = set()
    for seat in ACT_ORDER:
        board = boards[seat]
        if not board.is_complete():
            raise ValueError(f"seat {seat} board is not complete")
        cards = set(board.all_cards())
        overlap = seen & cards
        if overlap:
            raise ValueError(f"boards overlap on {sorted(overlap)}")
        seen |= cards

    return {
        seat: score_board(boards[seat].top, boards[seat].middle, boards[seat].bottom)
        for seat in ACT_ORDER
    }


def _validated_stacks(stacks: Mapping[Seat3, float]) -> dict[Seat3, float]:
    missing = set(ACT_ORDER) - set(stacks)
    if missing:
        raise ValueError(f"missing stacks for seats: {sorted(missing)}")
    unknown = set(stacks) - set(ACT_ORDER)
    if unknown:
        raise ValueError(f"unknown 3-max seats in stacks: {sorted(unknown)}")
    remaining = {seat: float(stacks[seat]) for seat in ACT_ORDER}
    negative = [seat for seat, value in remaining.items() if value < 0]
    if negative:
        raise ValueError(f"stacks must be non-negative: {sorted(negative)}")
    return remaining
