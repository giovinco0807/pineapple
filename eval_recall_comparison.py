#!/usr/bin/env python3
"""Evaluate recall@K for the augmented T0 model.

Loads the trained model, scores all placements for each hand,
and measures how often the CFR-best placement appears in the model's top-K.
"""

import json
import torch
import numpy as np
import sys
sys.path.insert(0, 'ai')
from train_t0_placement import T0PlacementNet, encode_card


def card_key(c):
    return f"{c['rank']}_{c['suit']}"


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load augmented model
    ckpt_aug = torch.load('ai/models/t0_placement_net_v3_aug.pt', weights_only=False)
    model_aug = T0PlacementNet(d_model=128, nhead=4, num_layers=4, dim_ff=256)
    model_aug.load_state_dict(ckpt_aug['model_state_dict'])
    model_aug.eval().to(device)
    
    # Load original model for comparison
    ckpt_orig = torch.load('ai/models/t0_placement_net_v3.pt', weights_only=False)
    model_orig = T0PlacementNet(d_model=128, nhead=4, num_layers=4, dim_ff=256)
    model_orig.load_state_dict(ckpt_orig['model_state_dict'])
    model_orig.eval().to(device)
    
    # Load test data (original non-augmented)
    data_path = r'c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\t0_training_v3.jsonl'
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))
    
    ks = [1, 3, 5, 10, 20, 50, 100]
    
    def evaluate_model(model, model_name):
        recall = {k: 0 for k in ks}
        total = 0
        
        for sample in samples:
            hand = sample['hand']
            if len(hand) != 5:
                continue
            placements = sample.get('placements', [])
            if not placements:
                continue
            
            feats = np.stack([encode_card(c) for c in hand])
            feats_t = torch.from_numpy(feats).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits, _ = model(feats_t)
                probs = torch.softmax(logits, dim=-1)[0]  # (5, 3)
            
            hand_keys = [card_key(c) for c in hand]
            scored = []
            
            for p in placements:
                card_row = {}
                for row_name, row_idx in [('top', 0), ('mid', 1), ('bot', 2)]:
                    for card in p.get(row_name, []):
                        card_row[card_key(card)] = row_idx
                
                score = 0.0
                for i, key in enumerate(hand_keys):
                    if key in card_row:
                        score += torch.log(probs[i, card_row[key]] + 1e-8).item()
                scored.append((score, p.get('ev', 0)))
            
            scored.sort(key=lambda x: x[0], reverse=True)
            best_ev = max(p.get('ev', -999) for p in placements)
            
            for rank, (sc, ev) in enumerate(scored, 1):
                if abs(ev - best_ev) < 0.001:
                    for k in recall:
                        if rank <= k:
                            recall[k] += 1
                    break
            
            total += 1
        
        print(f"\n{'='*50}")
        print(f"{model_name} (total={total})")
        print(f"{'='*50}")
        print(f"{'K':>6} | {'Recall':>8} | {'Count':>10}")
        print("-" * 30)
        for k in ks:
            pct = recall[k] / total * 100 if total > 0 else 0
            print(f"{k:>6} | {pct:>7.1f}% | {recall[k]:>5}/{total}")
    
    evaluate_model(model_orig, "v3 Original (1,133 samples)")
    evaluate_model(model_aug, "v3 + Suit Augmentation (25,324 samples)")


if __name__ == '__main__':
    main()
