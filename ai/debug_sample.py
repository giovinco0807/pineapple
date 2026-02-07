"""Debug specific sample comparison."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from game.card import Card
from game.royalty import get_top_royalty, get_middle_royalty, get_bottom_royalty, check_fantasyland_stay

RANK_MAP = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}
SUIT_MAP = {'s':0,'h':1,'d':2,'c':3}
RANK_REV = {v:k for k,v in RANK_MAP.items()}
SUIT_REV = {0:'s',1:'h',2:'d',3:'c'}

def cards_from_json(cards_json):
    result = []
    for c in cards_json:
        rank = c['rank'] if isinstance(c['rank'], str) else RANK_REV.get(c['rank'], '?')
        suit = c['suit'] if isinstance(c['suit'], str) else SUIT_REV.get(c['suit'], '?')
        result.append(Card(rank, suit))
    return result

def main():
    data_file = Path(__file__).parent / 'data' / 'fl_joker0_combined.jsonl'
    exe = Path(__file__).parent / 'rust_solver' / 'target' / 'release' / 'fl_solver.exe'
    
    with open(data_file, 'r', encoding='utf-8-sig') as f:
        for i, line in enumerate(f):
            if i != 1:  # Sample 2 (0-indexed)
                continue
            
            sample = json.loads(line)
            hand = sample['hand']
            expected = sample['reward']
            sol = sample['solution']
            
            print(f"=== Sample {i+1} ===")
            print(f"Expected: reward={expected}, royalties={sample['royalties']}, stay={sample.get('can_stay', False)}")
            print()
            
            # Python solution
            top_cards = cards_from_json(sol['top'])
            mid_cards = cards_from_json(sol['middle'])
            bot_cards = cards_from_json(sol['bottom'])
            
            py_top_roy = get_top_royalty(top_cards)
            py_mid_roy = get_middle_royalty(mid_cards)
            py_bot_roy = get_bottom_royalty(bot_cards)
            py_stay = check_fantasyland_stay(top_cards, bot_cards)
            py_score = py_top_roy + py_mid_roy + py_bot_roy + (50 if py_stay else 0)
            
            print(f"Python solution:")
            print(f"  Top: {[str(c) for c in top_cards]} -> roy={py_top_roy}")
            print(f"  Mid: {[str(c) for c in mid_cards]} -> roy={py_mid_roy}")
            print(f"  Bot: {[str(c) for c in bot_cards]} -> roy={py_bot_roy}")
            print(f"  Stay={py_stay}, Total={py_score}")
            print()
            
            # Rust solution
            cards = [{'rank': RANK_MAP.get(c['rank'], 0), 'suit': SUIT_MAP.get(c['suit'], 0)} for c in hand]
            req = json.dumps({'cards': cards})
            result = subprocess.run([str(exe)], input=req, capture_output=True, text=True, timeout=60)
            
            for l in result.stdout.strip().split('\n'):
                if l.startswith('{'):
                    resp = json.loads(l)
                    p = resp['placement']
                    print(f"Rust solution:")
                    print(f"  Score={p['score']}, Royalties={p['total_royalty']}")
                    print(f"  Top Roy={p['top_royalty']}, Mid Roy={p['middle_royalty']}, Bot Roy={p['bottom_royalty']}")
                    print(f"  Stay={p['can_stay']}")
                    break

if __name__ == '__main__':
    main()
