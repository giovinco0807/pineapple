import os
import sys
import json
import torch
import torch.nn.functional as F
import random
import itertools
import numpy as np
import argparse
from tqdm import tqdm

# Ensure we can import from ai/
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ai.train_t0_placement import T0PlacementNet, CARD_DIM, NUM_ROWS, MAX_CARDS, encode_card, compute_hand_features

# Deck definition
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['spades', 'hearts', 'diamonds', 'clubs']
DECK = [{'rank': r, 'suit': s} for r in RANKS for s in SUITS]

def card_to_idx(card):
    rank_map = {r: i for i, r in enumerate(RANKS)}
    suit_map = {s: i for i, s in enumerate(SUITS)}
    return rank_map[card['rank']] * 4 + suit_map[card['suit']]

def idx_to_card(idx):
    return DECK[idx]

def get_deck_indices():
    # CardIdx is 0..51
    return list(range(52))

def generate_valid_t0_placements():
    valid = []
    for p in itertools.product([0, 1, 2], repeat=5):
        if p.count(0) <= 3:
            valid.append(p)
    return torch.tensor(valid, dtype=torch.long)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-hands', type=int, default=50000)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--output', type=str, default='ai/data/t0_top50_50k.jsonl')
    parser.add_argument('--model', type=str, default='ai/models/t0_placement_net_v4.pt')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    d_model = config.get('d_model', 128)
    num_layers = config.get('num_layers', 4)
    dropout = config.get('dropout', 0.2)
    
    model = T0PlacementNet(
        d_model=d_model, nhead=4, num_layers=num_layers,
        dim_ff=d_model * 2, dropout=dropout
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    V_tensor = generate_valid_t0_placements().to(device)
    num_valid = V_tensor.shape[0]
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    deck_indices = get_deck_indices()
    
    with open(args.output, 'w') as f:
        for i in tqdm(range(0, args.n_hands, args.batch_size)):
            actual_batch = min(args.batch_size, args.n_hands - i)
            hands_idx = [random.sample(deck_indices, 5) for _ in range(actual_batch)]
            
            # Map back to dicts for encoding
            hands_dicts = []
            for h in hands_idx:
                hand_dict = []
                for idx in h:
                    rank = idx // 4
                    suit = idx % 4
                    hand_dict.append({'rank': RANKS[rank], 'suit': SUITS[suit]})
                hands_dicts.append(hand_dict)
            
            # Prepare input tensor
            X = np.zeros((actual_batch, 5, CARD_DIM), dtype=np.float32)
            for b in range(actual_batch):
                for j in range(5):
                    X[b, j, :18] = encode_card(hands_dicts[b][j])
                extras = compute_hand_features(hands_dicts[b])
                X[b, :, 18:] = extras
            
            X_tensor = torch.tensor(X, device=device)
            
            with torch.no_grad():
                logits, _ = model(X_tensor)
                log_p = F.log_softmax(logits, dim=-1)
                
                scores = torch.zeros(actual_batch, num_valid, device=device)
                b_idx = torch.arange(actual_batch, device=device).view(-1, 1)
                for c_idx in range(5):
                    v_idx = V_tensor[:, c_idx].view(1, -1)
                    scores += log_p[b_idx, c_idx, v_idx]
                
                top_scores, top_indices = torch.topk(scores, 50, dim=1)
                top_indices = top_indices.cpu().numpy()
            
            for b in range(actual_batch):
                top50 = []
                for k in range(50):
                    idx = top_indices[b, k]
                    placement = V_tensor[idx].cpu().numpy().tolist()
                    top50.append(placement)
                
                record = {
                    "t0_hand": hands_idx[b],
                    "top50": top50
                }
                f.write(json.dumps(record) + '\n')

if __name__ == '__main__':
    main()
