"""Debug - check what ranks are returned."""
import subprocess
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path('.').resolve()))
from game.card import Deck

SUIT_MAP = {'s': 0, 'h': 1, 'd': 2, 'c': 3}
RANK_REV = {2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'T', 11:'J', 12:'Q', 13:'K', 14:'A'}

deck = Deck(include_jokers=False)  # No jokers for simple test
deck.shuffle()

hand_cards = []
while len(hand_cards) < 14:
    dealt = deck.deal()
    if dealt is None:
        break
    card = dealt[0] if isinstance(dealt, list) else dealt
    hand_cards.append({
        'rank': card.rank_value, 
        'suit': SUIT_MAP.get(card.suit[0].lower(), 0),
    })

print("Input ranks:", [c['rank'] for c in hand_cards])

result = subprocess.run(
    ['ai/rust_solver/target/release/fl_solver.exe'],
    input=json.dumps({'cards': hand_cards}),
    capture_output=True,
    text=True
)

for line in result.stdout.split('\n'):
    if line.startswith('{'):
        response = json.loads(line)
        if response.get('success') and response.get('placement'):
            p = response['placement']
            all_ranks = []
            for cards in [p['top'], p['middle'], p['bottom']]:
                for c in cards:
                    all_ranks.append(c['rank'])
            print("Output ranks:", all_ranks)
            
            # Check for invalid ranks
            invalid = [r for r in all_ranks if r not in RANK_REV and r != 0]
            if invalid:
                print("INVALID RANKS:", invalid)
            else:
                print("All ranks valid!")
            
            # Map and display
            print("\nTop:", [RANK_REV.get(c['rank'], f"?{c['rank']}") for c in p['top']])
            print("Mid:", [RANK_REV.get(c['rank'], f"?{c['rank']}") for c in p['middle']])
            print("Bot:", [RANK_REV.get(c['rank'], f"?{c['rank']}") for c in p['bottom']])
