"""Benchmark v1 vs v2 solver accuracy and speed."""
import subprocess
import json
import time
import random

RUST_SOLVER = 'ai/rust_solver/target/release/fl_solver.exe'

# Generate random hands
random.seed(42)
def make_deck():
    deck = []
    for suit in range(4):
        for rank in range(2, 15):
            deck.append({'rank': rank, 'suit': suit})
    return deck

def deal_hand(num_cards=14):
    deck = make_deck()
    random.shuffle(deck)
    return deck[:num_cards]

def solve_hand(cards, version=1):
    request = json.dumps({'cards': cards, 'version': version})
    start = time.time()
    result = subprocess.run(
        [RUST_SOLVER],
        input=request,
        capture_output=True,
        text=True,
        timeout=120
    )
    elapsed = time.time() - start
    
    for line in result.stdout.split('\n'):
        if line.startswith('{'):
            response = json.loads(line)
            return response, elapsed
    return None, elapsed

# Test with 20 random hands
NUM_TESTS = 20
print(f"Testing {NUM_TESTS} hands, comparing v1 (exhaustive) vs v2 (role-based)...")
print()

v1_times = []
v2_times = []
matches = 0
mismatches = []

for i in range(NUM_TESTS):
    hand = deal_hand(14)
    
    r1, t1 = solve_hand(hand, version=1)
    r2, t2 = solve_hand(hand, version=2)
    
    v1_times.append(t1)
    v2_times.append(t2)
    
    if r1 and r2 and r1['success'] and r2['success']:
        p1 = r1['placement']
        p2 = r2['placement']
        
        # Compare scores
        if abs(p1['score'] - p2['score']) < 0.01:
            matches += 1
            status = "✓"
        else:
            mismatches.append((i, p1['score'], p2['score']))
            status = f"✗ (v1={p1['score']}, v2={p2['score']})"
    else:
        status = "ERROR"
        if r2 is None or not r2.get('success'):
            mismatches.append((i, 'v2 failed', ''))
    
    print(f"Hand {i+1}: v1={t1:.3f}s, v2={t2:.3f}s, {status}")

print()
print("=" * 50)
print(f"Accuracy: {matches}/{NUM_TESTS} ({100*matches/NUM_TESTS:.1f}%)")
print(f"V1 avg time: {sum(v1_times)/len(v1_times):.3f}s")
print(f"V2 avg time: {sum(v2_times)/len(v2_times):.3f}s")
print(f"Speedup: {sum(v1_times)/sum(v2_times):.2f}x")

if mismatches:
    print(f"\nMismatches:")
    for m in mismatches[:5]:
        print(f"  Hand {m[0]}: v1={m[1]}, v2={m[2]}")
