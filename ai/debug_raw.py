"""Debug - check raw Rust output."""
import subprocess
import json

# Generate a test hand
from pathlib import Path
import sys
sys.path.insert(0, str(Path('.').resolve()))
from game.card import Deck

SUIT_MAP = {'s': 0, 'h': 1, 'd': 2, 'c': 3}

deck = Deck(include_jokers=True)
deck.shuffle()

hand_cards = []
while len(hand_cards) < 14:
    dealt = deck.deal()
    if dealt is None:
        break
    card = dealt[0] if isinstance(dealt, list) else dealt
    if card.is_joker:
        hand_cards.append({'rank': 0, 'suit': 4})
    else:
        hand_cards.append({
            'rank': card.rank_value, 
            'suit': SUIT_MAP.get(card.suit[0].lower(), 0),
        })

print("Sending to Rust:")
print(json.dumps({'cards': hand_cards}, indent=2))
print()

result = subprocess.run(
    ['ai/rust_solver/target/release/fl_solver.exe'],
    input=json.dumps({'cards': hand_cards}),
    capture_output=True,
    text=True
)

print("Raw Rust output:")
print(result.stdout[:1000])
print()

# Parse and check
for line in result.stdout.split('\n'):
    if line.startswith('{'):
        response = json.loads(line)
        if response.get('success') and response.get('placement'):
            p = response['placement']
            print("Top cards (raw):")
            for c in p['top']:
                print(f"  rank={c['rank']} (type={type(c['rank']).__name__}), suit={c['suit']}")
