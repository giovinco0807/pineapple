"""Debug - verify hand and solution."""
import json

data = [json.loads(l) for l in open('ai/data/fl_rust_14cards_random.jsonl')]

d = data[0]
print("=== Sample 0 ===")
print("Hand:")
for c in d['hand']:
    print(f"  {c}")
print("\nSolution Top:")
for c in d['solution']['top']:
    print(f"  {c}")
print("\nSolution Middle:")
for c in d['solution']['middle']:
    print(f"  {c}")
print("\nSolution Bottom:")
for c in d['solution']['bottom']:
    print(f"  {c}")
