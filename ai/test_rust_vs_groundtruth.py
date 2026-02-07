"""Test Rust solver against ground truth data."""
import json
import subprocess
from pathlib import Path

def rank_to_int(rank: str) -> int:
    """Convert rank string to integer (2-14, 0 for joker)."""
    rank_map = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
        'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14, 'Joker': 0
    }
    return rank_map.get(rank, 0)

def suit_to_int(suit: str) -> int:
    """Convert suit string to integer."""
    suit_map = {'s': 0, 'h': 1, 'd': 2, 'c': 3, 'joker': 4}
    return suit_map.get(suit, 0)

def convert_hand_for_rust(hand: list) -> list:
    """Convert hand cards to Rust format."""
    cards = []
    for card in hand:
        if card.get('is_joker'):
            cards.append({'rank': 0, 'suit': 4})
        else:
            cards.append({
                'rank': rank_to_int(card['rank']),
                'suit': suit_to_int(card['suit'])
            })
    return cards

def solve_with_rust(hand: list) -> dict:
    """Solve hand using Rust solver."""
    rust_cards = convert_hand_for_rust(hand)
    request = json.dumps({'cards': rust_cards})
    
    exe_path = Path(__file__).parent / 'rust_solver' / 'target' / 'release' / 'fl_solver.exe'
    
    result = subprocess.run(
        [str(exe_path)],
        input=request,
        capture_output=True,
        text=True,
        timeout=120
    )
    
    for line in result.stdout.strip().split('\n'):
        if line.startswith('{'):
            return json.loads(line)
    return None

def main():
    data_file = Path(__file__).parent / 'data' / 'fl_joker0_combined.jsonl'
    
    print("Testing Rust solver vs ground truth...")
    print("=" * 60)
    
    matches = 0
    total = 0
    score_diffs = []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 20:  # Test first 20 samples
                break
            
            sample = json.loads(line)
            hand = sample['hand']
            expected_reward = sample['reward']
            expected_can_stay = sample.get('can_stay', False)
            
            try:
                result = solve_with_rust(hand)
                if result and result.get('success'):
                    rust_score = result['placement']['score']
                    rust_stay = result['placement']['can_stay']
                    
                    # Compare
                    score_match = abs(rust_score - expected_reward) < 0.1
                    stay_match = rust_stay == expected_can_stay
                    is_match = score_match and stay_match
                    
                    if is_match:
                        matches += 1
                        print(f"[{i+1:2d}] ✓ Score: {rust_score:.0f} (expected: {expected_reward})")
                    else:
                        score_diffs.append(rust_score - expected_reward)
                        print(f"[{i+1:2d}] ✗ Rust: {rust_score:.0f}, Expected: {expected_reward}, Stay: {rust_stay} vs {expected_can_stay}")
                else:
                    print(f"[{i+1:2d}] ✗ Rust solver failed")
            except Exception as e:
                print(f"[{i+1:2d}] ✗ Error: {e}")
            
            total += 1
    
    print("=" * 60)
    print(f"Match Rate: {matches}/{total} ({100*matches/total:.1f}%)")
    if score_diffs:
        print(f"Avg Score Diff: {sum(score_diffs)/len(score_diffs):.1f}")

if __name__ == '__main__':
    main()
