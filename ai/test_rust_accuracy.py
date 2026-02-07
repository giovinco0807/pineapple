"""Test Rust solver accuracy against ground truth."""
import json
import subprocess
from pathlib import Path

RANK_MAP = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}
SUIT_MAP = {'s':0,'h':1,'d':2,'c':3}

data_file = Path(__file__).parent / 'data' / 'fl_joker0_combined.jsonl'
exe = Path(__file__).parent / 'rust_solver' / 'target' / 'release' / 'fl_solver.exe'

matches = 0
total = 0

with open(data_file, 'r', encoding='utf-8-sig') as f:
    for i, line in enumerate(f):
        if i >= 50:
            break
        
        sample = json.loads(line)
        hand = sample['hand']
        expected = sample['reward']
        
        cards = [{'rank': RANK_MAP.get(c['rank'], 0), 'suit': SUIT_MAP.get(c['suit'], 0)} for c in hand]
        req = json.dumps({'cards': cards})
        
        result = subprocess.run([str(exe)], input=req, capture_output=True, text=True, timeout=120)
        
        for l in result.stdout.strip().split('\n'):
            if l.startswith('{'):
                resp = json.loads(l)
                rust_score = resp['placement']['score']
                is_match = abs(rust_score - expected) < 0.1
                if is_match:
                    matches += 1
                status = 'Y' if is_match else 'N'
                diff = rust_score - expected
                print(f"{i+1:3}. {status} R={rust_score:5.0f} E={expected:3} diff={diff:+5.0f}")
                break
        
        total += 1

print()
print(f"Match: {matches}/{total} ({100*matches/total:.1f}%)")
