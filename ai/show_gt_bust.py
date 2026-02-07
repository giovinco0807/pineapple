"""Show all GT_BUST mismatches for user review."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from game.card import Card
from game.board import Board, Row

data_file = Path(__file__).parent / 'data' / 'fl_joker0_combined.jsonl'
exe = Path(__file__).parent / 'rust_solver' / 'target' / 'release' / 'fl_solver.exe'

RANK_MAP = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}
SUIT_MAP = {'s':0,'h':1,'d':2,'c':3}
RANK_REV = {v:k for k,v in RANK_MAP.items()}
SUIT_REV = {0:'s',1:'h',2:'d',3:'c'}

def convert_card(c):
    if c.get('is_joker'):
        return {'rank': 0, 'suit': 4}
    return {'rank': RANK_MAP.get(c['rank'], 0), 'suit': SUIT_MAP.get(c['suit'], 0)}

def fmt(cards):
    return ' '.join([c['rank']+c['suit'] for c in cards])

def fmt_rust(cards):
    return ' '.join([RANK_REV[c['rank']]+SUIT_REV[c['suit']] for c in cards])

def check_gt_bust(sample):
    sol = sample['solution']
    top = [Card(c['rank'], c['suit']) for c in sol['top']]
    mid = [Card(c['rank'], c['suit']) for c in sol['middle']]
    bot = [Card(c['rank'], c['suit']) for c in sol['bottom']]
    board = Board()
    for c in top: board.place_card(Row.TOP, c)
    for c in mid: board.place_card(Row.MIDDLE, c)
    for c in bot: board.place_card(Row.BOTTOM, c)
    return board.is_bust()

count = 0
with open(data_file, 'r', encoding='utf-8-sig') as f:
    for i, line in enumerate(f):
        if i >= 200:
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
                    count += 1
                    print(f"=== Mismatch #{count}: Sample {i+1} ===")
                    print(f"Hand: {fmt(sample['hand'])}")
                    print()
                    sol = sample['solution']
                    print(f"[GT] score={expected}")
                    print(f"  Top: {fmt(sol['top'])}")
                    print(f"  Mid: {fmt(sol['middle'])}")
                    print(f"  Bot: {fmt(sol['bottom'])}")
                    print(f"  Python is_bust: {check_gt_bust(sample)}")
                    print()
                    print(f"[Rust] score={rust:.0f}")
                    print(f"  Top: {fmt_rust(p['top'])}")
                    print(f"  Mid: {fmt_rust(p['middle'])}")
                    print(f"  Bot: {fmt_rust(p['bottom'])}")
                    print()
                break
