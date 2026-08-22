"""3-seat rotation harness: the 3-max replacement for heads-up AB/BA pairing.

Heads-up gates are built on mirrored duplicate deals -- the same deck played
both ways round, ``paired_score = (ab - ba) / 2`` -- which cancels the seat
advantage exactly.  Three seats need a rotation instead of a swap: one deck is
played three times, with each policy occupying each seat exactly once.  Summed
over the block, every policy has paid and collected the same positional rent,
so the seat term cancels by construction rather than by averaging.

The identity that makes this trustworthy: if the three policies are identical,
every block sums to exactly 0.0 for every player.  ``self_test`` asserts it.
That is the 3-max analogue of the heads-up harness's NEW-vs-NEW check, and no
verdict from this harness should be believed until it passes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import permutations
from typing import Literal, Mapping, Sequence

from .play import Policy3, play_hand
from .seating import ACT_ORDER, PLAYER_COUNT, Seat3

ROTATIONS = PLAYER_COUNT

Orientations = Literal["cyclic", "full"]

# The 3 cyclic seatings cancel the additive seat term: every player visits
# every seat once.  What they do NOT cancel is relative order -- under cyclic
# rotation player 1 sits immediately after player 0 in every seating, so the
# A-vs-B and A-vs-C comparisons of a three-way matchup come from disjoint
# orientation halves.  "full" plays all 6 permutations, which covers both
# orientations of every ordered pair and cancels relative order too.  Cyclic is
# fine (and half the cost) whenever the two non-hero policies are identical;
# a genuine A-vs-B-vs-C gate should use full.
_SEATINGS: dict[Orientations, tuple[tuple[Seat3, ...], ...]] = {
    "cyclic": tuple(
        tuple(ACT_ORDER[(player + rotation) % PLAYER_COUNT] for player in range(PLAYER_COUNT))
        for rotation in range(ROTATIONS)
    ),
    "full": tuple(permutations(ACT_ORDER)),
}


def seatings_for(orientations: Orientations) -> tuple[tuple[Seat3, ...], ...]:
    """Seat assignments (indexed by player) for one block."""
    if orientations not in _SEATINGS:
        raise ValueError(f"unknown orientations mode: {orientations!r}")
    return _SEATINGS[orientations]


@dataclass(frozen=True)
class BlockResult:
    """One deck played through all three rotations."""

    seed: int
    totals: Mapping[int, float]
    per_rotation: tuple[Mapping[int, float], ...]
    seats_per_rotation: tuple[Mapping[int, Seat3], ...]


@dataclass(frozen=True)
class MatchupSummary:
    blocks: int
    hands: int
    mean_per_hand: Mapping[int, float]
    stderr_per_hand: Mapping[int, float]
    ci95_per_hand: Mapping[int, tuple[float, float]]
    mean_per_block: Mapping[int, float]

    def describe(self, labels: Mapping[int, str] | None = None) -> str:
        names = dict(labels or {})
        lines = [f"blocks={self.blocks} hands={self.hands} (3 rotations per deck)"]
        for player in range(PLAYER_COUNT):
            low, high = self.ci95_per_hand[player]
            name = names.get(player, f"p{player}")
            lines.append(
                f"  {name:<22} {self.mean_per_hand[player]:+8.3f} / hand   "
                f"95% CI [{low:+.3f}, {high:+.3f}]"
            )
        return "\n".join(lines)


def seat_for(player: int, rotation: int) -> Seat3:
    """Seat of ``player`` in ``rotation``; each player visits each seat once."""
    if not 0 <= rotation < ROTATIONS:
        raise ValueError(f"rotation out of range: {rotation}")
    if not 0 <= player < PLAYER_COUNT:
        raise ValueError(f"player out of range: {player}")
    return ACT_ORDER[(player + rotation) % PLAYER_COUNT]


def play_block(
    *,
    seed: int,
    policies_by_player: Mapping[int, Policy3],
    fl_ev_per_pair: Mapping[int, float] | None = None,
    orientations: Orientations = "cyclic",
) -> BlockResult:
    """Play one deck through every seating of the chosen orientation set.

    The same ``seed`` drives every seating, so the deck and every decision
    seed are identical across them; only who sits where changes.
    """
    missing = set(range(PLAYER_COUNT)) - set(policies_by_player)
    if missing:
        raise ValueError(f"missing 3-max policies for players: {sorted(missing)}")

    totals = {player: 0.0 for player in range(PLAYER_COUNT)}
    per_rotation: list[Mapping[int, float]] = []
    seats_per_rotation: list[Mapping[int, Seat3]] = []
    for seating in seatings_for(orientations):
        seats = {player: seating[player] for player in range(PLAYER_COUNT)}
        result = play_hand(
            seed=seed,
            policies={seats[player]: policies_by_player[player] for player in seats},
            fl_ev_per_pair=fl_ev_per_pair,
        )
        scores = {
            player: result.settlement.raw_totals[seats[player]]
            for player in range(PLAYER_COUNT)
        }
        for player, value in scores.items():
            totals[player] += value
        per_rotation.append(scores)
        seats_per_rotation.append(seats)

    return BlockResult(
        seed=seed,
        totals=totals,
        per_rotation=tuple(per_rotation),
        seats_per_rotation=tuple(seats_per_rotation),
    )


def evaluate_matchup(
    *,
    policies_by_player: Mapping[int, Policy3],
    blocks: int,
    base_seed: int,
    seed_stride: int = 1,
    fl_ev_per_pair: Mapping[int, float] | None = None,
    orientations: Orientations = "cyclic",
    progress_every: int = 0,
) -> MatchupSummary:
    """Play ``blocks`` decks, each through every seating of the orientation set.

    The block is the independent unit, so the reported standard error is over
    blocks; ``mean_per_hand`` divides by the hands inside one.  The 1.96
    quantile is the normal approximation -- block totals are heavy-tailed, so
    treat a CI that barely clears zero with suspicion and add blocks.
    """
    if blocks < 2:
        raise ValueError("blocks must be at least 2 (one block has no variance estimate)")
    if seed_stride <= 0:
        raise ValueError("seed_stride must be positive")

    hands_per_block = len(seatings_for(orientations))
    observations: dict[int, list[float]] = {
        player: [] for player in range(PLAYER_COUNT)
    }
    for index in range(blocks):
        block = play_block(
            seed=base_seed + index * seed_stride,
            policies_by_player=policies_by_player,
            fl_ev_per_pair=fl_ev_per_pair,
            orientations=orientations,
        )
        for player, value in block.totals.items():
            observations[player].append(value)
        if progress_every and (index + 1) % progress_every == 0:
            print(f"  block {index + 1}/{blocks}", flush=True)

    mean_block: dict[int, float] = {}
    mean_hand: dict[int, float] = {}
    stderr_hand: dict[int, float] = {}
    ci_hand: dict[int, tuple[float, float]] = {}
    for player, values in observations.items():
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        stderr = math.sqrt(variance / len(values))
        mean_block[player] = mean
        mean_hand[player] = mean / hands_per_block
        stderr_hand[player] = stderr / hands_per_block
        ci_hand[player] = (
            mean_hand[player] - 1.96 * stderr_hand[player],
            mean_hand[player] + 1.96 * stderr_hand[player],
        )

    return MatchupSummary(
        blocks=blocks,
        hands=blocks * hands_per_block,
        mean_per_hand=mean_hand,
        stderr_per_hand=stderr_hand,
        ci95_per_hand=ci_hand,
        mean_per_block=mean_block,
    )


SELF_TEST_TOLERANCE = 1e-9


def self_test(
    *,
    policy: Policy3,
    seeds: Sequence[int],
    fl_ev_per_pair: Mapping[int, float] | None = None,
    orientations: Orientations = "cyclic",
    tolerance: float = SELF_TEST_TOLERANCE,
) -> None:
    """Assert the harness's identities on identical policies.

    With three copies of one policy every seating of a block is the SAME hand
    (decision seeds are keyed by seat, never by player), which gives two
    checkable identities:

    1. every player's block total cancels to zero; and
    2. the score of a given SEAT is identical across the block's seatings
       (attribution: totals were mapped seat->player correctly).

    The first alone is weak -- it is implied by settlement zero-sum plus the
    seat-keyed seeds.  The second is what catches a harness that shuffles
    scores between players, which the first can never see, and it IS checked
    bit-exactly because it compares one seat's score against itself.

    The cancellation is exact in arithmetic but not in float64: a block total
    is ``(s1+s2) + (s3-s1) + (-s2-s3)`` over pair scores carrying a non-dyadic
    Fantasyland constant, so each partial sum rounds and a residual of order
    1e-15 survives.  ``tolerance`` sits six orders above that residual and nine
    below any real defect -- a mis-attributed score moves a block total by
    whole points, never by an ulp.
    """
    for seed in seeds:
        block = play_block(
            seed=seed,
            policies_by_player={player: policy for player in range(PLAYER_COUNT)},
            fl_ev_per_pair=fl_ev_per_pair,
            orientations=orientations,
        )
        for player, value in block.totals.items():
            if abs(value) > tolerance:
                raise AssertionError(
                    f"rotation self-test failed at seed {seed}: player {player} "
                    f"scored {value!r} against copies of itself (must cancel to "
                    f"within {tolerance})"
                )

        seat_scores: dict[Seat3, float] = {}
        for rotation, scores in enumerate(block.per_rotation):
            seats = block.seats_per_rotation[rotation]
            for player, value in scores.items():
                seat = seats[player]
                if seat not in seat_scores:
                    seat_scores[seat] = value
                elif seat_scores[seat] != value:
                    raise AssertionError(
                        f"rotation self-test failed at seed {seed}: seat {seat} "
                        f"scored {seat_scores[seat]!r} in one seating and {value!r} "
                        "in another with identical policies -- block scores are "
                        "being attributed to the wrong player"
                    )
