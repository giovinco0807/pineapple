"""
Python wrapper for Rust FL Solver.

Communicates with the Rust solver via subprocess JSON stdin/stdout.
"""

import subprocess
import json
import os
from typing import List, Optional, Dict, Any
from pathlib import Path


# Default path to Rust solver
RUST_SOLVER_PATH = Path(__file__).parent / "rust_solver" / "target" / "release" / "fl_solver.exe"


class RustFLSolver:
    """Wrapper for Rust FL solver."""
    
    def __init__(self, solver_path: Optional[str] = None):
        self.solver_path = solver_path or str(RUST_SOLVER_PATH)
        if not os.path.exists(self.solver_path):
            raise FileNotFoundError(f"Rust solver not found: {self.solver_path}")
        
        # Start persistent subprocess
        self.process = subprocess.Popen(
            [self.solver_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
    
    def __del__(self):
        if hasattr(self, 'process') and self.process:
            self.process.terminate()
    
    def solve(self, cards: List) -> Optional[Dict[str, Any]]:
        """
        Solve Fantasyland placement.
        
        Args:
            cards: List of Card objects or tuples (rank, suit)
        
        Returns:
            Dictionary with placement result or None
        """
        # Convert cards to JSON format
        card_list = []
        for c in cards:
            if hasattr(c, 'rank_value'):
                # Card object
                rank = 0 if c.is_joker else c.rank_value
                suit = 4 if c.is_joker else {'spades': 0, 'hearts': 1, 'diamonds': 2, 'clubs': 3}.get(c.suit, 0)
            else:
                # Tuple (rank, suit)
                rank, suit = c
            card_list.append({"rank": rank, "suit": suit})
        
        request = {"cards": card_list}
        
        # Send request
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        
        # Read response
        response_line = self.process.stdout.readline()
        if not response_line:
            return None
        
        response = json.loads(response_line)
        
        if response.get("success") and response.get("placement"):
            return response["placement"]
        return None


def solve_fantasyland_rust(cards: List) -> Optional[Dict[str, Any]]:
    """
    Convenience function to solve a single Fantasyland hand.
    
    Creates a new solver instance for each call.
    For batch processing, use RustFLSolver class directly.
    """
    solver = RustFLSolver()
    result = solver.solve(cards)
    del solver
    return result


if __name__ == "__main__":
    import time
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    from fl_solver import deal_fantasyland_hand, solve_fantasyland_exhaustive
    
    print("Rust FL Solver Test")
    print("=" * 50)
    
    solver = RustFLSolver()
    
    for i in range(3):
        print(f"\n--- Test {i+1} ---")
        hand = deal_fantasyland_hand(14, include_jokers=False)
        print(f"Hand: {' '.join(str(c) for c in hand[:5])} ...")
        
        # Rust
        start = time.time()
        result_rust = solver.solve(hand)
        t_rust = time.time() - start
        
        # Python exhaustive
        start = time.time()
        result_py = solve_fantasyland_exhaustive(hand, max_solutions=1)
        t_py = time.time() - start
        
        rust_score = result_rust["score"] if result_rust else 0
        py_score = result_py[0].score if result_py else 0
        
        print(f"Rust:   {t_rust:.3f}s, score: {rust_score}")
        print(f"Python: {t_py:.2f}s, score: {py_score}")
        print(f"Speedup: {t_py/t_rust:.0f}x")
        print(f"Match: {'YES' if abs(rust_score - py_score) < 0.01 else 'NO'}")
    
    # Test 17 cards with jokers
    print("\n" + "=" * 50)
    print("17 Cards + 2 Jokers Test")
    print("=" * 50)
    
    hand = deal_fantasyland_hand(17, include_jokers=True)
    jokers = sum(1 for c in hand if c.is_joker)
    print(f"Hand: {len(hand)} cards, {jokers} jokers")
    
    start = time.time()
    result = solver.solve(hand)
    elapsed = time.time() - start
    
    if result:
        print(f"Time: {elapsed:.2f}s")
        print(f"Score: {result['score']}")
        print(f"FL Stay: {result['can_stay']}")
        print(f"Royalties: Top={result['top_royalty']}, Mid={result['middle_royalty']}, Bot={result['bottom_royalty']}")
