"""
Compare exhaustive solver vs heuristic solver

Runs both on 10 random hands and compares results.
"""
import sys
sys.path.insert(0, '.')

from game.card import Deck
from ai.fl_solver import solve_fantasyland_exhaustive
from ai.fl_solver_v2 import solve_fantasyland as solve_heuristic, cards_to_str
import random
import time


def compare_solvers(num_hands: int = 10, include_jokers: bool = True):
    """Compare exhaustive vs heuristic solver."""
    print("=" * 70)
    print("Comparing Exhaustive vs Heuristic Solver")
    print("=" * 70)
    print()
    
    results = []
    
    for i in range(num_hands):
        seed = random.randint(0, 99999)
        random.seed(seed)
        deck = Deck(include_jokers=include_jokers)
        deck.shuffle()
        hand = deck.deal(14)
        
        print(f"Hand {i+1} (seed={seed})")
        print(f"  Cards: {cards_to_str(hand)}")
        
        # Run heuristic solver
        h_start = time.time()
        h_solutions = solve_heuristic(hand, max_solutions=1)
        h_time = time.time() - h_start
        
        if h_solutions:
            h_best = h_solutions[0]
            h_score = h_best.royalties
            h_stay = h_best.can_stay
        else:
            h_score = -1
            h_stay = False
        
        # Run exhaustive solver (limit to avoid timeout)
        e_start = time.time()
        e_solutions = solve_fantasyland_exhaustive(hand, max_solutions=1)
        e_time = time.time() - e_start
        
        if e_solutions:
            e_best = e_solutions[0]
            e_score = e_best.royalties
            e_stay = e_best.can_stay
        else:
            e_score = -1
            e_stay = False
        
        # Compare
        match = h_score >= e_score  # Heuristic should be at least as good
        status = "✓ MATCH" if match else "✗ MISS"
        
        if h_score > e_score:
            status = "✓ BETTER"  # Heuristic found better (unlikely but possible)
        
        print(f"  Heuristic: {h_score}pts, FL={h_stay}, {h_time:.1f}s")
        print(f"  Exhaustive: {e_score}pts, FL={e_stay}, {e_time:.1f}s")
        print(f"  {status}")
        
        if not match:
            print(f"  [!] Heuristic missed: {e_score - h_score} points")
            print(f"      Exhaustive solution:")
            print(f"        Top:    {cards_to_str(e_best.top)}")
            print(f"        Middle: {cards_to_str(e_best.middle)}")
            print(f"        Bottom: {cards_to_str(e_best.bottom)}")
        
        print()
        
        results.append({
            'seed': seed,
            'h_score': h_score,
            'e_score': e_score,
            'match': match,
            'h_time': h_time,
            'e_time': e_time,
        })
    
    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    
    matches = sum(1 for r in results if r['match'])
    total_h_time = sum(r['h_time'] for r in results)
    total_e_time = sum(r['e_time'] for r in results)
    
    print(f"Matches: {matches}/{num_hands}")
    print(f"Avg Heuristic time: {total_h_time/num_hands:.2f}s")
    print(f"Avg Exhaustive time: {total_e_time/num_hands:.2f}s")
    print(f"Speedup: {total_e_time/total_h_time:.1f}x")
    
    if matches < num_hands:
        print()
        print("Failed cases:")
        for r in results:
            if not r['match']:
                print(f"  seed={r['seed']}: H={r['h_score']} vs E={r['e_score']}")
    
    return matches == num_hands


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--hands', type=int, default=10, help='Number of hands')
    parser.add_argument('--jokers', action='store_true', default=True, help='Include jokers')
    parser.add_argument('--no-jokers', action='store_true', help='No jokers')
    args = parser.parse_args()
    
    jokers = not args.no_jokers
    
    success = compare_solvers(args.hands, jokers)
    sys.exit(0 if success else 1)
