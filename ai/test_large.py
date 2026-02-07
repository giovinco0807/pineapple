"""Large-scale accuracy test for Rust solver."""
import json
import subprocess
from pathlib import Path
import time

def test_dataset(data_file, exe, num_samples):
    RANK_MAP = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}
    SUIT_MAP = {'s':0,'h':1,'d':2,'c':3}

    def convert_card(c):
        if c.get('is_joker'):
            return {'rank': 0, 'suit': 4}
        return {'rank': RANK_MAP.get(c['rank'], 0), 'suit': SUIT_MAP.get(c['suit'], 0)}

    matches = 0
    total = 0
    mismatches = []

    with open(data_file, 'r', encoding='utf-8-sig') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            sample = json.loads(line)
            expected = sample['reward']
            cards = [convert_card(c) for c in sample['hand']]
            
            try:
                result = subprocess.run([str(exe)], input=json.dumps({'cards': cards}), 
                                        capture_output=True, text=True, timeout=120)
                for l in result.stdout.split('\n'):
                    if l.startswith('{'):
                        p = json.loads(l)['placement']
                        rust = p['score']
                        if abs(rust - expected) < 0.1:
                            matches += 1
                        else:
                            mismatches.append((i+1, rust, expected))
                        total += 1
                        break
            except Exception as e:
                print(f"Error at sample {i+1}: {e}")
            
            if (i+1) % 100 == 0:
                print(f"  {i+1}/{num_samples}... ({100*matches/total:.1f}%)")

    return matches, total, mismatches

if __name__ == '__main__':
    exe = Path(__file__).parent / 'rust_solver' / 'target' / 'release' / 'fl_solver.exe'
    
    # Test joker-0
    print("=" * 50)
    print("JOKER-0 TEST (1000 samples)")
    print("=" * 50)
    j0_file = Path(__file__).parent / 'data' / 'fl_joker0_combined.jsonl'
    start = time.time()
    m0, t0, mis0 = test_dataset(j0_file, exe, 1000)
    print(f"\nJoker-0: {m0}/{t0} ({100*m0/t0:.1f}%) in {time.time()-start:.1f}s")
    if mis0:
        print(f"First 5 mismatches: {mis0[:5]}")
    
    print()
    
    # Test joker-1 (all 3000)
    print("=" * 50)
    print("JOKER-1 TEST (1000 samples)")
    print("=" * 50)
    j1_file = Path(__file__).parent / 'data' / 'fl_joker1.jsonl'
    start = time.time()
    m1, t1, mis1 = test_dataset(j1_file, exe, 1000)
    print(f"\nJoker-1: {m1}/{t1} ({100*m1/t1:.1f}%) in {time.time()-start:.1f}s")
    if mis1:
        print(f"First 5 mismatches: {mis1[:5]}")
