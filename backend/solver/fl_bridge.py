"""
FL Solver Bridge - Calls Rust solver for Fantasyland
"""

import subprocess
import json
from pathlib import Path
from typing import Optional


FL_SOLVER_PATH = Path(__file__).parent.parent.parent / "ai" / "rust_solver" / "target" / "release" / "fl_solver.exe"


def solve_fantasyland(cards: list[str]) -> Optional[dict]:
    """
    Call Rust FL solver for optimal placement.
    
    Args:
        cards: List of cards (14-17), e.g., ["Ah", "Kd", "X1", ...]
    
    Returns:
        {
            "top": ["Ah", "Ad", "Ac"],
            "middle": ["7s", "8s", "9s", "Ts", "Js"],
            "bottom": ["Kh", "Kd", "Kc", "Ks", "2h"],
            "discarded": ["3c", "4d"],
            "score": 47.0,
            "fl_stay": True,
            "royalties": {"top": 22, "middle": 30, "bottom": 10}
        }
    """
    if not FL_SOLVER_PATH.exists():
        raise FileNotFoundError(f"FL Solver not found at {FL_SOLVER_PATH}")
    
    # Convert cards to solver format
    solver_cards = []
    for card in cards:
        if card in ["X1", "X2"]:
            solver_cards.append({"rank": 0, "suit": 4})  # Joker
        else:
            rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                       '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
            suit_map = {'h': 0, 'd': 1, 'c': 2, 's': 3}
            solver_cards.append({
                "rank": rank_map.get(card[0], 0),
                "suit": suit_map.get(card[1], 0)
            })
    
    request = json.dumps({"cards": solver_cards, "version": 2})
    
    try:
        result = subprocess.run(
            [str(FL_SOLVER_PATH)],
            input=request,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        for line in result.stdout.split('\n'):
            if line.startswith('{'):
                response = json.loads(line)
                if response.get("success") and response.get("placement"):
                    placement = response["placement"]
                    
                    # Track original joker names for restoration
                    joker_names = [c for c in cards if c in ("X1", "X2")]
                    joker_idx = 0
                    
                    # Convert back to card strings
                    def cards_to_str(cards_list):
                        nonlocal joker_idx
                        result = []
                        rank_rev = {2: '2', 3: '3', 4: '4', 5: '5', 6: '6',
                                   7: '7', 8: '8', 9: '9', 10: 'T', 11: 'J',
                                   12: 'Q', 13: 'K', 14: 'A'}
                        suit_rev = {0: 'h', 1: 'd', 2: 'c', 3: 's'}
                        for c in cards_list:
                            if c.get("rank", 0) == 0:
                                # Use original joker name (X1/X2)
                                if joker_idx < len(joker_names):
                                    result.append(joker_names[joker_idx])
                                    joker_idx += 1
                                else:
                                    result.append("X1")
                            else:
                                r = rank_rev.get(c["rank"], '?')
                                s = suit_rev.get(c["suit"], '?')
                                result.append(f"{r}{s}")
                        return result
                    
                    return {
                        "top": cards_to_str(placement["top"]),
                        "middle": cards_to_str(placement["middle"]),
                        "bottom": cards_to_str(placement["bottom"]),
                        "score": placement.get("score", 0),
                        "fl_stay": placement.get("can_stay", False),
                        "royalties": placement.get("royalties", 0)
                    }
        
        return None
        
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        print(f"FL Solver error: {e}")
        return None


if __name__ == "__main__":
    # Test
    test_cards = ["Ah", "Ad", "Ac", "Kh", "Kd", "Ks", "Qh", "Qd",
                  "7s", "8s", "9s", "Ts", "Js", "2c"]
    result = solve_fantasyland(test_cards)
    print(json.dumps(result, indent=2))
