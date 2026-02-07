"""Test Rust solver with joker-1 data."""
import json
import subprocess
from pathlib import Path

data_file = Path(__file__).parent / 'data' / 'fl_joker1.jsonl'
exe = Path(__file__).parent / 'rust_solver' / 'target' / 'release' / 'fl_solver.exe'

RANK_MAP = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}
SUIT_MAP = {'s':0,'h':1,'d':2,'c':3}

def convert_card(c):
    if c.get('is_joker'):
        return {'rank': 0, 'suit': 4}
    return {'rank': RANK_MAP.get(c['rank'], 0), 'suit': SUIT_MAP.get(c['suit'], 0)}

matches = 0
total = 0

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
                ok = 'Y' if abs(rust - expected) < 0.1 else 'N'
                if ok == 'Y':
                    matches += 1
                print(f"{i+1:2}. {ok} R={rust:.0f} E={expected}")
                total += 1
                break

print()
print(f"JOKER-1 ACCURACY: {matches}/{total} ({100*matches/total:.0f}%)")
