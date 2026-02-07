"""Show mismatched samples for joker-1 data."""
import json
import subprocess
from pathlib import Path

data_file = Path(__file__).parent / 'data' / 'fl_joker1.jsonl'
exe = Path(__file__).parent / 'rust_solver' / 'target' / 'release' / 'fl_solver.exe'

RANK_MAP = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}
SUIT_MAP = {'s':0,'h':1,'d':2,'c':3}
RANK_REV = {v:k for k,v in RANK_MAP.items()}
SUIT_REV = {0:'s',1:'h',2:'d',3:'c',4:'J'}

def convert_card(c):
    if c.get('is_joker'):
        return {'rank': 0, 'suit': 4}
    return {'rank': RANK_MAP.get(c['rank'], 0), 'suit': SUIT_MAP.get(c['suit'], 0)}

def fmt(cards):
    result = []
    for c in cards:
        if c.get('is_joker'):
            result.append('JK')
        else:
            result.append(c['rank'] + c['suit'])
    return ' '.join(result)

def fmt_rust(cards):
    result = []
    for c in cards:
        if c['rank'] == 0:
            result.append('JK')
        else:
            result.append(RANK_REV.get(c['rank'],'?') + SUIT_REV.get(c['suit'],'?'))
    return ' '.join(result)

with open(data_file, 'r', encoding='utf-8-sig') as f:
    for i, line in enumerate(f):
        if i >= 30:
            break
        sample = json.loads(line)
        expected = sample['reward']
        cards = [convert_card(c) for c in sample['hand']]
        result = subprocess.run([str(exe)], input=json.dumps({'cards': cards}), capture_output=True, text=True, timeout=120)
        for l in result.stdout.split('\n'):
            if l.startswith('{'):
                p = json.loads(l)['placement']
                rust = p['score']
                if abs(rust - expected) >= 0.1:
                    print(f"=== Sample {i+1} ===")
                    print(f"Hand: {fmt(sample['hand'])}")
                    print()
                    print(f"[GT] score={expected} roy={sample['royalties']} stay={sample['can_stay']}")
                    print(f"  Top: {fmt(sample['solution']['top'])}")
                    print(f"  Mid: {fmt(sample['solution']['middle'])}")
                    print(f"  Bot: {fmt(sample['solution']['bottom'])}")
                    print()
                    print(f"[Rust] score={rust:.0f}")
                    print(f"  Top: {fmt_rust(p['top'])} (roy={p['top_royalty']})")
                    print(f"  Mid: {fmt_rust(p['middle'])} (roy={p['middle_royalty']})")
                    print(f"  Bot: {fmt_rust(p['bottom'])} (roy={p['bottom_royalty']})")
                    print(f"  stay={p['can_stay']}")
                    print()
                break
