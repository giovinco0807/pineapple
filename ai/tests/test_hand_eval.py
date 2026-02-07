"""
OFC Pineapple - Hand Evaluation Unit Tests

Tests edge cases in evaluate_hand (full house with jokers, straights,
straight flushes) and royalty calculations.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.game_engine import (
    evaluate_hand, get_top_royalty, get_middle_royalty,
    get_bottom_royalty, check_fl_entry
)


def test_full_house():
    """Full House (7000-7999) edge cases with jokers."""
    # Natural full house: AAA KK
    val = evaluate_hand(["Ah", "Ad", "Ac", "Kh", "Kd"], 5)
    assert 7000 <= val < 8000, f"AAA KK should be FH, got {val}"

    # Two pair + joker → FH: AA KK + X1
    val = evaluate_hand(["Ah", "Ad", "Kh", "Kd", "X1"], 5)
    assert 7000 <= val < 8000, f"AA KK X1 should be FH, got {val}"

    # Trips + 2 jokers → Quads (joker makes 4th): AAA + X1 X2
    val = evaluate_hand(["Ah", "Ad", "Ac", "X1", "X2"], 5)
    assert 8000 <= val < 9000, f"AAA X1 X2 should be quads, got {val}"

    # Pair + 3 distinct ranks + joker → Trips, NOT FH
    val = evaluate_hand(["Ah", "Ad", "Kh", "Qs", "X1"], 5)
    assert val < 7000, f"AA K Q X1 should be trips not FH, got {val}"

    # 4 distinct ranks + joker → Pair, NOT FH
    val = evaluate_hand(["Ah", "Kh", "Qs", "7d", "X1"], 5)
    assert val < 7000, f"A K Q 7 X1 should be pair not FH, got {val}"

    print("  ✅ Full house tests passed")


def test_straight_flush():
    """Straight flush (9000+) with jokers."""
    # Natural royal flush
    val = evaluate_hand(["Ah", "Kh", "Qh", "Jh", "Th"], 5)
    assert val >= 9000, f"Royal flush should be 9000+, got {val}"
    assert val >= 9014, f"Royal flush A-high should be 9014+, got {val}"

    # Flush + straight with joker
    val = evaluate_hand(["Ah", "Kh", "Qh", "Jh", "X1"], 5)
    assert val >= 9000, f"A K Q J + joker (all hearts) should be SF, got {val}"

    # Non-SF: different suits
    val = evaluate_hand(["Ah", "Ks", "Qh", "Jh", "Th"], 5)
    assert val < 9000, f"Mixed suit straight should not be SF, got {val}"

    print("  ✅ Straight flush tests passed")


def test_four_of_a_kind():
    """Four of a kind (8000-8999)."""
    val = evaluate_hand(["Ah", "Ad", "Ac", "As", "Kh"], 5)
    assert 8000 <= val < 9000, f"AAAA K should be quads, got {val}"

    # Trips + joker = quads
    val = evaluate_hand(["Ah", "Ad", "Ac", "X1", "Kh"], 5)
    assert 8000 <= val < 9000, f"AAA X1 K should be quads, got {val}"

    print("  ✅ Four of a kind tests passed")


def test_flush_and_straight():
    """Flush (6000) and Straight (5000)."""
    # Flush
    val = evaluate_hand(["Ah", "9h", "7h", "4h", "2h"], 5)
    assert 6000 <= val < 7000, f"Five hearts should be flush, got {val}"

    # Straight
    val = evaluate_hand(["Ah", "2s", "3h", "4d", "5c"], 5)
    assert 5000 <= val < 6000, f"A-5 straight (wheel) should be straight, got {val}"

    val = evaluate_hand(["Th", "Js", "Qh", "Kd", "Ac"], 5)
    assert 5000 <= val < 6000, f"T-A straight should be straight, got {val}"

    print("  ✅ Flush and straight tests passed")


def test_top_royalty():
    """Top row royalty calculations."""
    # No royalty for low pairs
    assert get_top_royalty(["2h", "3s", "4d"]) == 0
    assert get_top_royalty(["5h", "5s", "4d"]) == 0  # 55 < 66

    # Pair royalties: 66=1, 77=2, ..., AA=9
    assert get_top_royalty(["6h", "6s", "4d"]) == 1
    assert get_top_royalty(["Ah", "As", "4d"]) == 9

    # Trips: 222=10, ..., AAA=22
    assert get_top_royalty(["2h", "2s", "2d"]) == 10
    assert get_top_royalty(["Ah", "As", "Ad"]) == 22

    # QQ = 7 (FL entry threshold)
    assert get_top_royalty(["Qh", "Qs", "4d"]) == 7

    print("  ✅ Top royalty tests passed")


def test_middle_royalty():
    """Middle row royalty calculations."""
    assert get_middle_royalty(["2h", "3s", "4d", "7c", "9h"]) == 0   # high card
    assert get_middle_royalty(["Ah", "Ad", "Ac", "Kh", "Kd"]) == 12  # FH
    assert get_middle_royalty(["Ah", "Ad", "Ac", "As", "Kh"]) == 20  # Quads
    assert get_middle_royalty(["Ah", "Kh", "Qh", "Jh", "Th"]) == 50 # RF

    print("  ✅ Middle royalty tests passed")


def test_bottom_royalty():
    """Bottom row royalty calculations."""
    assert get_bottom_royalty(["Ah", "Ad", "Ac", "Kh", "Kd"]) == 6   # FH
    assert get_bottom_royalty(["Ah", "Ad", "Ac", "As", "Kh"]) == 10  # Quads
    assert get_bottom_royalty(["Ah", "Kh", "Qh", "Jh", "Th"]) == 25 # RF

    print("  ✅ Bottom royalty tests passed")


def test_fl_entry():
    """Fantasyland entry conditions."""
    # QQ+ enters FL
    fl, cards = check_fl_entry(["Qh", "Qs", "4d"])
    assert fl == True and cards == 14, f"QQ should enter FL, got {fl}, {cards}"

    fl, cards = check_fl_entry(["Kh", "Ks", "4d"])
    assert fl == True and cards == 15, f"KK should enter FL, got {fl}, {cards}"

    fl, cards = check_fl_entry(["Ah", "As", "4d"])
    assert fl == True and cards == 16, f"AA should enter FL, got {fl}, {cards}"

    # Trips enter FL with 17 cards
    fl, cards = check_fl_entry(["2h", "2s", "2d"])
    assert fl == True and cards == 17, f"Trips should enter FL 17 cards, got {fl}, {cards}"

    # JJ and below: no FL
    fl, _ = check_fl_entry(["Jh", "Js", "4d"])
    assert fl == False, f"JJ should NOT enter FL"

    fl, _ = check_fl_entry(["Th", "Ts", "4d"])
    assert fl == False, f"TT should NOT enter FL"

    # No pair: no FL
    fl, _ = check_fl_entry(["Ah", "Ks", "4d"])
    assert fl == False, f"A K 4 should NOT enter FL"

    print("  ✅ Fantasyland entry tests passed")


if __name__ == "__main__":
    print("Running hand evaluation tests...\n")
    test_full_house()
    test_straight_flush()
    test_four_of_a_kind()
    test_flush_and_straight()
    test_top_royalty()
    test_middle_royalty()
    test_bottom_royalty()
    test_fl_entry()
    print("\n✅ All tests passed!")
