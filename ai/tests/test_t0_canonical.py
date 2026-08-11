"""Checks on the T0 canonical form.

The counts are verified two ways: `t0_canonical` walks unordered rank-set
tuples, and the tests count orbits with Burnside over the five cycle types
of S4.  Agreeing on 152,646 from opposite directions is worth more than
either matching a number written down in advance, so the constants are
asserted against the Burnside count as well as against themselves.

The symmetry claims are checked against brute force over the 48 group
elements, which is cheap enough per hand to be the reference.
"""
import itertools
import math
import random

import pytest

from ai.engine.encoding import ALL_CARDS, CARD_TO_IDX, SUITS
from ai.tutor.t0_canonical import (
    HAND_SIZE, JOKERS, TOTAL_CANONICAL, TOTAL_DEALS, canonical,
    enumerate_canonical, multiplicity, representative, sample_canonical,
)

NATURALS = [card for card in ALL_CARDS if card not in JOKERS]
SUIT_PERMS = list(itertools.permutations(SUITS))


def _relabel(hand, perm, swap):
    """`hand` under a suit relabelling and optionally the joker swap."""
    out = []
    for card in hand:
        if card in JOKERS:
            out.append(JOKERS[1 - JOKERS.index(card)] if swap else card)
        else:
            out.append(card[0] + perm[SUITS.index(card[1])])
    return out


def _orbit(hand):
    return {frozenset(_relabel(hand, perm, swap))
            for perm in SUIT_PERMS for swap in (False, True)}


def _sample_hands(seed, count, jokers):
    rng = random.Random(seed)
    hands = []
    for n in range(count):
        held = rng.sample(NATURALS, HAND_SIZE - jokers)
        if jokers == 1:
            held.append(JOKERS[n % 2])  # both namings must canonicalise alike
        elif jokers == 2:
            held.extend(JOKERS)
        rng.shuffle(held)
        hands.append(held)
    return hands


def _mixed_sample(seed, per_bucket):
    return [hand
            for jokers in (0, 1, 2)
            for hand in _sample_hands(seed + jokers, per_bucket, jokers)]


def _burnside_orbits(size):
    """Orbits of S4 on `size`-subsets of the 52 naturals.

    A suit permutation acts on the 52 cards as its own cycle structure
    repeated once per rank, so a subset it fixes is a union of those
    cycles, and counting those unions is a subset-sum over cycle lengths.
    """
    conjugacy = [(1, [1, 1, 1, 1]), (6, [2, 1, 1]), (3, [2, 2]), (8, [3, 1]), (6, [4])]
    fixed_total = 0
    for count, cycle_type in conjugacy:
        lengths = [length for length in cycle_type for _ in range(13)]
        ways = [0] * (size + 1)
        ways[0] = 1
        for length in lengths:
            for target in range(size, length - 1, -1):
                ways[target] += ways[target - length]
        fixed_total += count * ways[size]
    orbits, remainder = divmod(fixed_total, 24)
    assert remainder == 0
    return orbits


@pytest.fixture(scope="module")
def classes():
    keys = list(enumerate_canonical())
    return keys, [multiplicity(key) for key in keys]


def test_class_count_matches_burnside(classes):
    keys, _ = classes
    # One class per suit-orbit of the naturals at each joker count; the two
    # jokers being interchangeable is what makes the 4-card case one class
    # and not two.
    expected = _burnside_orbits(5) + _burnside_orbits(4) + _burnside_orbits(3)
    assert expected == TOTAL_CANONICAL == 152_646
    assert len(keys) == expected
    assert len(set(keys)) == expected


def test_multiplicities_cover_every_deal(classes):
    _, deals = classes
    assert sum(deals) == TOTAL_DEALS == math.comb(54, HAND_SIZE) == 3_162_510
    assert min(deals) >= 1


def test_enumeration_is_deterministic():
    first = list(itertools.islice(enumerate_canonical(), 5000))
    second = list(itertools.islice(enumerate_canonical(), 5000))
    assert first == second


@pytest.mark.parametrize("jokers", [0, 1, 2])
def test_canonical_is_invariant_under_the_group(jokers):
    for hand in _sample_hands(11 + jokers, 40, jokers):
        key = canonical(hand)
        images = {canonical(_relabel(hand, perm, swap))
                  for perm in SUIT_PERMS for swap in (False, True)}
        assert images == {key}, hand


def test_key_equality_is_exactly_the_group_orbit():
    hands = _mixed_sample(101, 30)
    orbits = [_orbit(hand) for hand in hands]
    keys = [canonical(hand) for hand in hands]
    for i, j in itertools.combinations(range(len(hands)), 2):
        related = frozenset(hands[j]) in orbits[i]
        assert (keys[i] == keys[j]) == related, (hands[i], hands[j])


def test_multiplicity_equals_brute_force_orbit_size():
    for hand in _mixed_sample(202, 25):
        assert multiplicity(hand) == len(_orbit(hand))


def test_representative_round_trips(classes):
    keys, _ = classes
    rng = random.Random(303)
    for key in rng.sample(keys, 3000):
        hand = representative(key)
        assert sorted(hand) == sorted(key)
        assert canonical(hand) == key
    for hand in _mixed_sample(404, 25):
        assert representative(hand) == list(canonical(hand))


def test_canonical_is_the_least_deal_in_its_class():
    # The module picks the representative by sorting rank sets rather than
    # searching the orbit; this is the claim that shortcut has to earn.
    for hand in _mixed_sample(606, 25):
        ordered = [sorted(member, key=CARD_TO_IDX.__getitem__) for member in _orbit(hand)]
        least = min(ordered, key=lambda cards: [CARD_TO_IDX[card] for card in cards])
        assert list(canonical(hand)) == least


def test_representative_lands_in_the_class():
    for hand in _mixed_sample(505, 25):
        assert frozenset(representative(hand)) in _orbit(hand)


@pytest.mark.parametrize("weighting", ["class", "deal"])
def test_sample_is_distinct_reproducible_and_prefix_stable(weighting):
    drawn = sample_canonical(400, seed=7, weighting=weighting)
    assert len(drawn) == 400
    keys = [canonical(hand) for hand in drawn]
    assert len(set(keys)) == 400
    assert keys == [tuple(hand) for hand in drawn]  # samples come out canonical
    assert sample_canonical(400, seed=7, weighting=weighting) == drawn
    assert sample_canonical(150, seed=7, weighting=weighting) == drawn[:150]
    assert sample_canonical(400, seed=8, weighting=weighting) != drawn


def test_deal_weighting_favours_the_common_classes():
    by_class = sample_canonical(3000, seed=13, weighting="class")
    by_deal = sample_canonical(3000, seed=13, weighting="deal")
    assert by_class != by_deal
    mean_class = sum(multiplicity(h) for h in by_class) / len(by_class)
    mean_deal = sum(multiplicity(h) for h in by_deal) / len(by_deal)
    # Class-uniform reproduces the population mean of 20.7; deal-uniform is
    # size-biased and must land above it.
    assert 19.5 < mean_class < 22.0
    assert mean_deal > mean_class + 2.0


def test_sample_rejects_an_unstated_weighting():
    with pytest.raises(TypeError):
        sample_canonical(10, 1)  # weighting is keyword-only and has no default
    with pytest.raises(ValueError):
        sample_canonical(10, 1, weighting="uniform")
    with pytest.raises(ValueError):
        sample_canonical(TOTAL_CANONICAL + 1, 1, weighting="class")


@pytest.mark.parametrize("hand", [
    ["Ah", "Kh", "Qh", "Jh"],
    ["Ah", "Kh", "Qh", "Jh", "Th", "9h"],
    ["Ah", "Ah", "Qh", "Jh", "Th"],
    ["Ah", "Kh", "Qh", "Jh", "JK"],
    ["Ah", "Kh", "Qh", "Jh", "1z"],
])
def test_canonical_rejects_a_hand_that_is_not_a_deal(hand):
    with pytest.raises(ValueError):
        canonical(hand)
