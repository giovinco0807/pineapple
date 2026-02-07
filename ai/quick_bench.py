"""Quick benchmark - fewer tests for speed."""
import subprocess
import json
import time
import random

RUST_SOLVER = 'ai/rust_solver/target/release/fl_solver.exe'

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
            if response.get('success') and response.get('placement'):
                return response['placement']['score'], elapsed
    return None, elapsed

# Test with 10 random hands
NUM_TESTS = 10
print(f"Testing {NUM_TESTS} hands...")

v1_times = []
v2_times = []
matches = 0

for i in range(NUM_TESTS):
    hand = deal_hand(14)
    
    s1, t1 = solve_hand(hand, version=1)
    s2, t2 = solve_hand(hand, version=2)
    
    v1_times.append(t1)
    v2_times.append(t2)
    
    if s1 is not None and s2 is not None:
        if abs(s1 - s2) < 0.01:
            matches += 1
            print(f"Hand {i+1}: v1={t1:.3f}s v2={t2:.3f}s score={s1:.1f} OK")
        else:
            print(f"Hand {i+1}: v1={t1:.3f}s v2={t2:.3f}s v1_score={s1:.1f} v2_score={s2:.1f} MISMATCH")
    else:
        print(f"Hand {i+1}: ERROR (s1={s1}, s2={s2})")

print()
print(f"Accuracy: {matches}/{NUM_TESTS}")
print(f"V1 avg: {sum(v1_times)/len(v1_times):.3f}s")
print(f"V2 avg: {sum(v2_times)/len(v2_times):.3f}s")
if sum(v2_times) > 0:
    print(f"Speedup: {sum(v1_times)/sum(v2_times):.2f}x")
