"""T0 hands folded down to the states the game can actually tell apart.

At T0 hero holds five cards and places all five -- no draw, no discard, no
board yet -- so the whole state is those five cards.  Two symmetries leave
the game unchanged.  The four suits carry no meaning of their own, only
which cards share one; and the two jokers are the same card twice, since
`is_joker` in `ai.engine.game_engine` counts jokers and never asks which.
Folding both away turns C(54,5) = 3,162,510 deals into 152,646 classes, so
one label covers 20.7 deals on average and the whole T0 state space is
small enough to cover a real fraction of.  That is why T0 gets this
treatment and the later streets do not.

# The key

`canonical` returns the deal in its class that sorts first by deck index,
as a tuple of card strings.  A key is therefore itself a legal hand, which
is what makes it readable in the JSONL label files it is headed for, and
makes `representative` little more than a cast.

# Which weighting to sample with

`sample_canonical` will not guess, because the two answers differ by up to
24x on a single class and the choice is not a detail.

Uniform over classes buys the most state space per label: every class is
one decision hero has to get right, and a labelling run that spends its
budget in proportion to how often a class comes up would leave the rare
tails of the space untouched.

Uniform over deals reproduces the rate at which each class reaches the
table.  A gate that charges regret per decision faced in play needs this
one -- scored class-uniformly, a class hero sees twice as often as another
would contribute the same regret as it, and the number the gate reports
would not be the number hero loses.
"""
from __future__ import annotations

import hashlib
import heapq
import itertools
import math
from collections import Counter
from typing import Iterable, Iterator, List, Tuple

from ai.engine.encoding import ALL_CARDS, CARD_TO_IDX, RANK_VALUES, RANKS, SUITS

HAND_SIZE = 5
JOKERS = ("X1", "X2")
TOTAL_DEALS = 3_162_510  # C(54, 5)
TOTAL_CANONICAL = 152_646
WEIGHTINGS = ("class", "deal")

CanonicalKey = Tuple[str, ...]

_JOKER_SET = frozenset(JOKERS)
_RANK_COUNT = len(RANKS)  # 13
_SUIT_COUNT = len(SUITS)  # 4

# A hand's natural cards are exactly a rank set per suit, and a suit
# relabelling is exactly a rearrangement of those four sets.  Every
# invariant below reads off that one fact.
_DECK_INDEX = [
    [CARD_TO_IDX[RANKS[rank] + SUITS[suit]] for rank in range(_RANK_COUNT)]
    for suit in range(_SUIT_COUNT)
]
_MASKS_BY_POPCOUNT = {
    size: sorted(sum(1 << rank for rank in combo)
                 for combo in itertools.combinations(range(_RANK_COUNT), size))
    for size in range(HAND_SIZE + 1)
}
_SUIT_BLOCK: List[dict] = [{} for _ in range(_SUIT_COUNT)]
_MASK_ORDER: dict = {}

# `_mask_order` below is only the least arrangement because the deck runs
# suit-major and contiguous, so a suit's cards never interleave with
# another's.  Reorder ALL_CARDS and the derivation goes with it.
assert all(_DECK_INDEX[suit][rank] == suit * _RANK_COUNT + rank
           for suit in range(_SUIT_COUNT) for rank in range(_RANK_COUNT)), \
    "t0_canonical assumes ALL_CARDS is suit-major over 13 contiguous ranks"


def _mask_order(mask: int) -> Tuple[int, ...]:
    """Sort key placing `mask` in the suit that yields the least hand.

    Swapping the rank sets on two adjacent suits changes the hand's deck
    indices only within those two suits, and the swap that wins compares
    the two rank sequences ascending -- except that a sequence which is a
    proper prefix of the other must come second, because the longer one
    keeps scoring low indices in the earlier suit while the shorter has
    already moved on to the next.  Padding with a value above every rank
    says exactly that, and sorting on it beats a bubble of those swaps.
    """
    order = _MASK_ORDER.get(mask)
    if order is None:
        ranks = tuple(rank for rank in range(_RANK_COUNT) if mask >> rank & 1)
        order = ranks + (_RANK_COUNT,) * (HAND_SIZE - len(ranks))
        _MASK_ORDER[mask] = order
    return order


def _block(suit: int, mask: int) -> Tuple[int, ...]:
    """Deck indices of `mask`'s ranks worn in `suit`, memoised.

    Four calls on each of the 152,646 classes, against 2,380 reachable
    masks per suit, so the bit scan is worth doing once.
    """
    cache = _SUIT_BLOCK[suit]
    block = cache.get(mask)
    if block is None:
        row = _DECK_INDEX[suit]
        block = tuple(row[rank] for rank in range(_RANK_COUNT) if mask >> rank & 1)
        cache[mask] = block
    return block


def _suit_masks(hand: Iterable[str]) -> Tuple[Tuple[int, ...], int]:
    cards = tuple(hand)
    if len(cards) != HAND_SIZE:
        raise ValueError(f"a T0 hand is {HAND_SIZE} cards, got {len(cards)}: {cards}")
    if len(set(cards)) != len(cards):
        raise ValueError(f"repeated card in {cards}")
    masks = [0] * _SUIT_COUNT
    jokers = 0
    for card in cards:
        if card in _JOKER_SET:
            jokers += 1
            continue
        if card not in CARD_TO_IDX:
            raise ValueError(f"not a card: {card!r}")
        masks[SUITS.index(card[1])] |= 1 << RANK_VALUES[card[0]]
    return tuple(masks), jokers


def _class_of(masks: Tuple[int, ...], jokers: int) -> Tuple[CanonicalKey, int]:
    """The class's key and its multiplicity, without walking the orbit.

    The enumeration asks for both on every one of the 152,646 classes, so
    neither is allowed to cost a pass over the 48 relabellings.
    """
    indices = []
    for suit, mask in enumerate(sorted(masks, key=_mask_order)):
        indices.extend(_block(suit, mask))
    key = tuple(ALL_CARDS[index] for index in indices) + JOKERS[:jokers]
    # Orbit-stabiliser over the 48 relabellings.  Distinct rearrangements of
    # the four rank sets give distinct hands, so the suit part is 4! knocked
    # down by the repeats; the joker swap doubles that only when hero holds
    # exactly one joker, since with none or both it leaves the hand alone.
    arrangements = math.factorial(_SUIT_COUNT)
    for repeats in Counter(masks).values():
        arrangements //= math.factorial(repeats)
    return key, arrangements * (2 if jokers == 1 else 1)


def canonical(hand: Iterable[str]) -> CanonicalKey:
    """The canonical form of a 5-card T0 hand.

    Invariant under all 24 suit relabellings and under swapping X1 with X2.
    """
    masks, jokers = _suit_masks(hand)
    return _class_of(masks, jokers)[0]


def multiplicity(key: Iterable[str]) -> int:
    """How many of the 3,162,510 deals share `key`'s class.

    Exact, and takes any hand rather than only a canonical one, since it is
    a property of the class and not of the representative.
    """
    masks, jokers = _suit_masks(key)
    return _class_of(masks, jokers)[1]


def representative(key: Iterable[str]) -> List[str]:
    """One concrete hand from `key`'s class, always the same one."""
    return list(canonical(key))


def _popcount_profiles(total: int, slots: int = _SUIT_COUNT,
                       cap: int = HAND_SIZE) -> Iterator[Tuple[int, ...]]:
    """Non-increasing ways to split `total` cards across `slots` suits."""
    if slots == 1:
        if total <= cap:
            yield (total,)
        return
    for part in range(min(total, cap), -1, -1):
        if part * slots < total:  # the remaining slots cannot reach `total`
            break
        for rest in _popcount_profiles(total - part, slots - 1, part):
            yield (part,) + rest


def _iter_classes() -> Iterator[Tuple[CanonicalKey, int]]:
    """Every class exactly once, with its multiplicity.

    Walks unordered four-tuples of rank sets, which are the classes, rather
    than canonicalising 3,162,510 deals and discarding the 95% that repeat
    one already seen.  Grouping by popcount is what lets it skip the
    seen-set: selections differing inside a group differ as multisets, and
    two profiles differ in their popcounts, so no class is reachable twice.
    """
    for jokers in range(len(JOKERS) + 1):
        for profile in _popcount_profiles(HAND_SIZE - jokers):
            groups = [(popcount, sum(1 for _ in group))
                      for popcount, group in itertools.groupby(profile)]
            pools = [
                list(itertools.combinations_with_replacement(
                    _MASKS_BY_POPCOUNT[popcount], width))
                for popcount, width in groups
            ]
            for choice in itertools.product(*pools):
                masks = tuple(mask for chunk in choice for mask in chunk)
                yield _class_of(masks, jokers)


def enumerate_canonical() -> Iterator[CanonicalKey]:
    """All 152,646 T0 classes, in a fixed order."""
    for key, _ in _iter_classes():
        yield key


def _score(seed: int, key: CanonicalKey, weight: float) -> float:
    """Efraimidis-Spirakis draw for `key`, smallest scores win.

    Hashed from the key rather than drawn from a stream in enumeration
    order, which buys two things a labelling run cares about: the sample
    survives any future change to how `_iter_classes` walks the space, and
    raising `count` extends the sample instead of redrawing it, so labels
    already paid for stay in it.
    """
    digest = hashlib.blake2b(f"{seed}:{''.join(key)}".encode(), digest_size=8).digest()
    uniform = (int.from_bytes(digest, "big") + 0.5) / 2 ** 64
    return -math.log(uniform) / weight


def sample_canonical(count: int, seed: int, *, weighting: str) -> List[List[str]]:
    """`count` distinct classes as concrete hands, reproducible in `seed`.

    `weighting` is mandatory; see the module docstring for which one a job
    wants.  "class" draws uniformly over the 152,646 classes, "deal"
    uniformly over the 3,162,510 deals.

    Returned in selection order, so any prefix is itself a valid sample of
    that size.
    """
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}, got {weighting!r}")
    if not 0 <= count <= TOTAL_CANONICAL:
        raise ValueError(f"count must be within 0..{TOTAL_CANONICAL}, got {count}")
    by_deal = weighting == "deal"
    scored = (
        (_score(seed, key, float(deals) if by_deal else 1.0), key)
        for key, deals in _iter_classes()
    )
    return [list(key) for _, key in heapq.nsmallest(count, scored)]
