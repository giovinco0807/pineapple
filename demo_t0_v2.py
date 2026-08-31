import sys
import random
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ai.engine.encoding import Board, Observation, encode_state, ALL_CARDS
from ai.engine.action_space import get_initial_actions, MAX_ACTIONS

def action_to_desc(action) -> str:
    by_pos = {"top": [], "middle": [], "bottom": []}
    for card, pos in action.placements:
        by_pos[pos].append(card)
    
    top_str = " ".join(sorted(by_pos["top"]))
    mid_str = " ".join(sorted(by_pos["middle"]))
    bot_str = " ".join(sorted(by_pos["bottom"]))
    
    return f"Top [{top_str:^8}] | Middle [{mid_str:^12}] | Bottom [{bot_str:^12}]"

def main():
    model_path = "ai/models/t0_ranking_v2/policy.onnx"
    try:
        sess = ort.InferenceSession(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    for hand_idx in range(3):
        # Generate random 5 cards
        deck = list(ALL_CARDS)
        random.shuffle(deck)
        hand = deck[:5]
        print(f"==========================================")
        print(f" Hand #{hand_idx+1}: {' '.join(hand)}")
        print(f"==========================================")

        # Encode state
        obs = Observation(
            board_self=Board(),
            board_opponent=Board(),
            dealt_cards=hand,
            known_discards_self=[],
            turn=0,
            is_btn=True,
            is_fl=False,
            opp_is_fl=False,
        )
        state = encode_state(obs).reshape(1, -1)

        # Get valid mask and actions
        actions = get_initial_actions(hand, Board())
        mask = np.zeros(MAX_ACTIONS, dtype=bool)
        for i in range(min(len(actions), MAX_ACTIONS)):
            mask[i] = True

        # Run inference
        probs = sess.run(None, {
            "state": state.astype(np.float32),
            "valid_mask": mask.reshape(1, -1),
        })[0][0]

        # Sort
        valid_indices = np.where(mask)[0]
        valid_probs = probs[valid_indices]
        sorted_order = np.argsort(-valid_probs)

        for i in range(min(5, len(valid_indices))):
            idx = valid_indices[sorted_order[i]]
            prob = valid_probs[sorted_order[i]]
            desc = action_to_desc(actions[idx])
            print(f"{i+1:>2}位: {prob*100:>5.1f}% | {desc}")
        print()

if __name__ == "__main__":
    main()
