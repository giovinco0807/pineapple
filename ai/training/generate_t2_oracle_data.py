import sys
import json
import random
import argparse
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai.engine.encoding import Board, Observation, ALL_CARDS, encode_state
from ai.engine.action_space import get_turn_actions, create_action_mask
from ai.training.train_t3_oracle_v2 import T3PolicyValueNet

def load_t3_oracle(path, device='cpu'):
    ck = torch.load(path, map_location=device, weights_only=False)
    state_dim = ck.get('state_dim', 522)
    n_actions = ck.get('n_actions', 27)
    hidden = ck.get('hidden', 1024)
    n_blocks = ck.get('n_blocks', 4)
    
    model = T3PolicyValueNet(
        state_dim=state_dim, 
        n_actions=n_actions, 
        hidden=hidden, 
        n_blocks=n_blocks
    )
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    model.to(device)
    return model

def apply_placements(board: Board, placements) -> Board:
    b = Board(
        top=list(board.top),
        middle=list(board.middle),
        bottom=list(board.bottom)
    )
    for card, pos in placements:
        getattr(b, pos).append(card)
    return b

def evaluate_t2_actions_with_t3_oracle(board: Board, deal, discards, model, device, n_samples=30):
    """
    Evaluate all valid T2 actions by sampling N T3 deals and asking the T3 Oracle for the expected EV.
    """
    actions = get_turn_actions(deal, board)
    if not actions:
        return [], []
    
    used_cards = set(board.top + board.middle + board.bottom + deal + discards)
    rem_deck = [c for c in ALL_CARDS if c not in used_cards]
    
    # Fast path if only 1 action
    if len(actions) == 1:
        return actions, [0.0]
        
    t3_states = []
    t3_indices = [] # Maps (action_idx) -> list of state indices in the batch
    
    for ai, action in enumerate(actions):
        next_board = apply_placements(board, action.placements)
        action_disc = discards + ([action.discard] if action.discard else [])
        
        action_indices = []
        for _ in range(n_samples):
            # Sample 3 cards for T3
            t3_deal = random.sample(rem_deck, 3)
            
            obs = Observation(
                board_self=next_board,
                board_opponent=Board(), # Ignore opp for now
                dealt_cards=t3_deal,
                known_discards_self=action_disc,
                turn=3,
                is_btn=True
            )
            state_vec = encode_state(obs)
            # Truncate to the model's expected state_dim (e.g. 520) in case the encoder added new features
            state_vec = state_vec[:model.input_proj[0].in_features]
            t3_states.append(state_vec)
            action_indices.append(len(t3_states) - 1)
            
        t3_indices.append(action_indices)
        
    # Batch evaluate all T3 states
    batch_states = torch.FloatTensor(np.array(t3_states)).to(device)
    
    # We just need the Value head
    with torch.no_grad():
        # model returns (logits, value)
        # We pass masks=None because we just want the scalar value.
        _, values = model(batch_states, masks=None)
        values = values.cpu().numpy()
        
    action_evs = []
    for ai, action in enumerate(actions):
        ev = np.mean(values[t3_indices[ai]])
        action_evs.append(ev)
        
    return actions, action_evs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--t3-model', required=True, help="Path to t3_policyvalue_v2_best.pt")
    parser.add_argument('--n-samples', type=int, default=30, help="T3 deals to sample per T2 action")
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    print(f"Loading T3 Oracle from {args.t3_model} on {device}...")
    model = load_t3_oracle(args.t3_model, device)
    
    print("Testing with a dummy T2 state...")
    # Dummy T2 state
    b = Board(top=[], middle=['Ah', 'Kh'], bottom=['2s', '3s', '4s'])
    deal = ['5s', '6s', 'Qs']
    discards = ['2h']
    
    t0 = time.time()
    actions, evs = evaluate_t2_actions_with_t3_oracle(b, deal, discards, model, device, n_samples=args.n_samples)
    elapsed = time.time() - t0
    
    print(f"Evaluated {len(actions)} actions with {args.n_samples} samples/action in {elapsed:.3f}s")
    for a, ev in zip(actions, evs):
        print(f"  {a} -> EV: {ev:.3f}")

if __name__ == '__main__':
    main()
