"""
Automated tests for FL Solver

Tests known hands against expected minimum scores.
"""
import sys
sys.path.insert(0, '.')

from game.card import Deck
from ai.fl_solver_v2 import solve_fantasyland, cards_to_str
import random
import time

# Test cases: (seed, jokers, expected_min_score, expected_fl_stay, description)
TEST_CASES = [
    # Seed 27394: Royal Flush possible with joker
    (27394, True, 30, True, "Royal Flush + TT top"),
    
    # Seed 24606: QQ on top better than JJ
    (24606, False, 7, False, "QQ top (7pts) vs JJ (6pts)"),
    
    # Seed 10163: KKKK quads with jokers, but Royal Flush better
    (10163, True, 31, True, "Royal Flush or Quads FL stay"),
    
    # Seed 72561: Full House bottom + TT top
    (72561, False, 11, False, "Full House + TT pair"),
    
    # Random seeds for regression
    (42, True, 0, None, "Random hand 1"),
    (123, True, 0, None, "Random hand 2"),
    (456, True, 0, None, "Random hand 3"),
    (789, False, 0, None, "Random hand 4"),
    (1000, True, 0, None, "Random hand 5"),
]


def run_test(seed: int, include_jokers: bool, min_score: int, 
             expected_stay: bool | None, description: str) -> tuple[bool, str]:
    """Run a single test case. Returns (passed, message)."""
    random.seed(seed)
    deck = Deck(include_jokers=include_jokers)
    deck.shuffle()
    hand = deck.deal(14)
    
    start = time.time()
    solutions = solve_fantasyland(hand, max_solutions=3)
    elapsed = time.time() - start
    
    if not solutions:
        return False, f"No solutions found for seed {seed}"
    
    best = solutions[0]
    
    # Check minimum score
    if best.royalties < min_score:
        return False, (f"Score {best.royalties} < expected {min_score}\n"
                      f"  Hand: {cards_to_str(hand)}\n"
                      f"  Top: {cards_to_str(best.top)}\n"
                      f"  Mid: {cards_to_str(best.middle)}\n"
                      f"  Bot: {cards_to_str(best.bottom)}")
    
    # Check FL stay if expected
    if expected_stay is not None and best.can_stay != expected_stay:
        return False, f"FL stay={best.can_stay}, expected={expected_stay}"
    
    # Check not bust
    if best.is_bust:
        return False, "Solution is bust!"
    
    return True, f"✓ {best.royalties}pts, FL={best.can_stay}, {elapsed:.1f}s"


def run_all_tests():
    """Run all test cases."""
    print("=" * 60)
    print("FL Solver Automated Tests")
    print("=" * 60)
    print()
    
    passed = 0
    failed = 0
    
    for seed, jokers, min_score, expected_stay, desc in TEST_CASES:
        print(f"Testing: {desc} (seed={seed})")
        
        try:
            success, message = run_test(seed, jokers, min_score, expected_stay, desc)
            
            if success:
                print(f"  PASS: {message}")
                passed += 1
            else:
                print(f"  FAIL: {message}")
                failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1
        
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


def run_random_tests(count: int = 10, include_jokers: bool = True):
    """Run random tests to check for crashes and basic validity."""
    print(f"\nRunning {count} random tests...")
    
    crashes = 0
    busts = 0
    no_solutions = 0
    total_time = 0
    
    for i in range(count):
        seed = random.randint(0, 99999)
        random.seed(seed)
        deck = Deck(include_jokers=include_jokers)
        deck.shuffle()
        hand = deck.deal(14)
        
        try:
            start = time.time()
            solutions = solve_fantasyland(hand, max_solutions=1)
            elapsed = time.time() - start
            total_time += elapsed
            
            if not solutions:
                no_solutions += 1
                print(f"  {i+1}. seed={seed}: No solutions!")
            elif solutions[0].is_bust:
                busts += 1
                print(f"  {i+1}. seed={seed}: Bust!")
            else:
                best = solutions[0]
                print(f"  {i+1}. seed={seed}: {best.royalties}pts, FL={best.can_stay}, {elapsed:.1f}s")
        except Exception as e:
            crashes += 1
            print(f"  {i+1}. seed={seed}: CRASH - {e}")
    
    print()
    print(f"Random test results:")
    print(f"  Crashes: {crashes}")
    print(f"  No solutions: {no_solutions}")
    print(f"  Busts: {busts}")
    print(f"  Avg time: {total_time/count:.2f}s")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--random', type=int, default=0, help='Number of random tests')
    parser.add_argument('--jokers', action='store_true', help='Include jokers in random tests')
    args = parser.parse_args()
    
    # Run known test cases
    all_passed = run_all_tests()
    
    # Run random tests if requested
    if args.random > 0:
        run_random_tests(args.random, args.jokers)
    
    sys.exit(0 if all_passed else 1)
