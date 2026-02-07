"""Debug script to verify Rust solver output."""
import subprocess
import json

# Test with a simple hand - Royal Flush possible in spades
cards = [
    {'rank': 14, 'suit': 0},  # As
    {'rank': 13, 'suit': 0},  # Ks
    {'rank': 12, 'suit': 0},  # Qs
    {'rank': 11, 'suit': 0},  # Js
    {'rank': 10, 'suit': 0},  # Ts
    {'rank': 9, 'suit': 0},   # 9s
    {'rank': 8, 'suit': 0},   # 8s
    {'rank': 7, 'suit': 0},   # 7s
    {'rank': 6, 'suit': 0},   # 6s
    {'rank': 5, 'suit': 0},   # 5s
    {'rank': 4, 'suit': 1},   # 4h
    {'rank': 3, 'suit': 1},   # 3h
    {'rank': 2, 'suit': 1},   # 2h
    {'rank': 14, 'suit': 1},  # Ah
]

RANK_REV = {2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'T', 11:'J', 12:'Q', 13:'K', 14:'A'}
SUIT_REV = {0:'s', 1:'h', 2:'d', 3:'c'}

def format_card(c):
    return f"{RANK_REV.get(c['rank'], str(c['rank']))}{SUIT_REV.get(c['suit'], str(c['suit']))}"

def format_hand(cards):
    return ' '.join(format_card(c) for c in cards)

print("Input cards:", format_hand(cards))
print()

result = subprocess.run(
    ['ai/rust_solver/target/release/fl_solver.exe'],
    input=json.dumps({'cards': cards}),
    capture_output=True,
    text=True
)

print("Stderr:", result.stderr)
print()

for line in result.stdout.split('\n'):
    if line.startswith('{'):
        response = json.loads(line)
        print("Success:", response['success'])
        if response['success']:
            p = response['placement']
            print("Top:", format_hand(p['top']))
            print("Middle:", format_hand(p['middle']))
            print("Bottom:", format_hand(p['bottom']))
            print("Score:", p['score'])
            print("Can Stay:", p['can_stay'])
            print("Top Royalty:", p['top_royalty'])
            print("Mid Royalty:", p['middle_royalty'])
            print("Bot Royalty:", p['bottom_royalty'])
