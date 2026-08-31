"""Verify Rust vs Python encoding match.

Outputs encoding for a known board state.
"""
import sys, json
sys.path.insert(0, '.')
from ai.engine.encoding import encode_state, ALL_CARDS, CARD_TO_IDX, STATE_DIM, Board, Observation

hero = Board(
    top=['Ah', 'Kh'],
    middle=['Qs', 'Js', 'Ts'],
    bottom=['9c', '8c', '7c', '6c', '5c'],
)
opp = Board(
    top=['2h'],
    middle=['3s', '4s'],
    bottom=['5s', '6s', '7s'],
)

obs = Observation(
    board_self=hero,
    board_opponent=opp,
    dealt_cards=['Ad', 'Kd', 'Qd'],
    known_discards_self=['2c'],
    turn=2,
    is_btn=True,
    chips_self=200,
    chips_opponent=200,
)

state = encode_state(obs)

# Print card indices
test_cards = ['Ah','Kh','Qs','Js','Ts','9c','8c','7c','6c','5c',
              '2h','3s','4s','5s','6s','7s','Ad','Kd','Qd','2c']
print("Card indices:")
for c in test_cards:
    print(f"  {c} = {CARD_TO_IDX[c]}")

# Print non-zero elements
print(f"\nNon-zero elements ({sum(1 for v in state if v != 0.0)} total):")
for i, v in enumerate(state):
    if v != 0.0:
        print(f"  [{i:3d}] = {v:.6f}")

# Save
with open('ai/verify_encoding_state.json', 'w') as f:
    json.dump([float(v) for v in state], f)
print(f"\nFull state saved to ai/verify_encoding_state.json")
