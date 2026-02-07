"""Verify if mismatches are due to bust issues in GT data."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from game.card import Card
from game.board import Board, Row
from game.joker_optimizer import hand_strength_3, hand_strength_5, compare_3_vs_5

data_file = Path(__file__).parent / 'data' / 'fl_joker0_combined.jsonl'
exe = Path(__file__).parent / 'rust_solver' / 'target' / 'release' / 'fl_solver.exe'

RANK_MAP = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}
SUIT_MAP = {'s':0,'h':1,'d':2,'c':3}

def convert_card(c):
    if c.get('is_joker'):
        return {'rank': 0, 'suit': 4}
    return {'rank': RANK_MAP.get(c['rank'], 0), 'suit': SUIT_MAP.get(c['suit'], 0)}

def check_gt_bust(sample):
    """Check if GT solution is a bust."""
    sol = sample['solution']
    top = [Card(c['rank'], c['suit']) for c in sol['top']]
    mid = [Card(c['rank'], c['suit']) for c in sol['middle']]
    bot = [Card(c['rank'], c['suit']) for c in sol['bottom']]
    
    board = Board()
    for c in top: board.place_card(Row.TOP, c)
    for c in mid: board.place_card(Row.MIDDLE, c)
    for c in bot: board.place_card(Row.BOTTOM, c)
    
    return board.is_bust()

# Find mismatches and check if GT is bust
mismatches = []
gt_bust_count = 0
rust_better_count = 0

with open(data_file, 'r', encoding='utf-8-sig') as f:
    for i, line in enumerate(f):
        if i >= 1000:
            break
        sample = json.loads(line)
        expected = sample['reward']
        cards = [convert_card(c) for c in sample['hand']]
        
        result = subprocess.run([str(exe)], input=json.dumps({'cards': cards}), 
                                capture_output=True, text=True, timeout=120)
        for l in result.stdout.split('\n'):
            if l.startswith('{'):
                p = json.loads(l)['placement']
                rust = p['score']
                if abs(rust - expected) >= 0.1:
                    gt_is_bust = check_gt_bust(sample)
                    if gt_is_bust:
                        gt_bust_count += 1
                    elif rust > expected:
                        rust_better_count += 1
                    mismatches.append({
                        'sample': i+1,
                        'rust': rust,
                        'gt': expected,
                        'diff': rust - expected,
                        'gt_is_bust': gt_is_bust
                    })
                break

print("=" * 60)
print("MISMATCH ANALYSIS (Joker-0, 1000 samples)")
print("=" * 60)
print(f"Total mismatches: {len(mismatches)}")
print(f"  - GT is BUST: {gt_bust_count}")
print(f"  - Rust found better solution: {rust_better_count}")
print(f"  - Other: {len(mismatches) - gt_bust_count - rust_better_count}")
print()
print("First 10 mismatches:")
for m in mismatches[:10]:
    status = "GT_BUST" if m['gt_is_bust'] else ("RUST_BETTER" if m['diff'] > 0 else "RUST_WORSE")
    print(f"  #{m['sample']:4d}: Rust={m['rust']:.0f} GT={m['gt']:.0f} diff={m['diff']:+.0f} -> {status}")
