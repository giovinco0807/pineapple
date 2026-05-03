import torch
import json
import numpy as np
from itertools import product
from ai.models.t1_network import T1PlacementNet, encode_card_str
from ai.train_t1_placement import parse_board

def eval_topk(model_path, data_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading {model_path} on {device}...")
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    
    d_model = config.get('d_model', 128)
    num_layers = config.get('num_layers', 4)
    dropout = config.get('dropout', 0.2)
    
    model = T1PlacementNet(d_model=d_model, nhead=4, num_layers=num_layers, dim_ff=d_model*2, dropout=dropout)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))
                
    print(f"Loaded {len(samples)} samples from {data_path}")

    # Map target strings to class indices
    target_map = {'Top': 0, 'Mid': 1, 'Bot': 2, 'Bottom': 2, 'Middle': 1, 'Discard': 3}
    
    top1 = 0
    top3 = 0
    top5 = 0
    top10 = 0
    total = 0

    for sample in samples:
        top_row, mid_row, bot_row = parse_board(sample['board'])
        hand = sample['hand'].split()
        
        # Get ground truth best placement
        best_p = sample['placements'][0]
        gt_target = [-1, -1, -1]
        
        d_card = best_p['d']
        p_parts = best_p['p'].split(', ') if best_p['p'] else []
        
        for i, c in enumerate(hand):
            if c == d_card:
                gt_target[i] = 3 # Discard
            else:
                for part in p_parts:
                    if part.startswith(c):
                        _, t_str = part.split('→')
                        gt_target[i] = target_map[t_str]
                        break
                        
        if -1 in gt_target:
            continue # Malformed ground truth
            
        gt_config = tuple(gt_target)
        
        # Predict
        all_cards = []
        for c in top_row: all_cards.append((c, 1))
        for c in mid_row: all_cards.append((c, 2))
        for c in bot_row: all_cards.append((c, 3))
        for c in hand: all_cards.append((c, 0))
        
        features = np.stack([encode_card_str(c, r) for c, r in all_cards])
        features = torch.from_numpy(features).unsqueeze(0).to(device)
        
        with torch.no_grad():
            row_logits, _ = model(features)
            
        logits = row_logits[0].cpu().numpy()
        
        scored_configs = []
        for config in product(range(4), repeat=3):
            if config.count(3) != 1:
                continue
                
            added = {0: 0, 1: 0, 2: 0}
            for p in config:
                if p != 3:
                    added[p] += 1
            
            if len(top_row) + added[0] > 3: continue
            if len(mid_row) + added[1] > 5: continue
            if len(bot_row) + added[2] > 5: continue
            
            score = logits[0, config[0]] + logits[1, config[1]] + logits[2, config[2]]
            scored_configs.append((score, config))
            
        scored_configs.sort(key=lambda x: x[0], reverse=True)
        
        rank = -1
        for i, (s, c) in enumerate(scored_configs):
            if c == gt_config:
                rank = i
                break
                
        if rank != -1:
            if rank < 1: top1 += 1
            if rank < 3: top3 += 1
            if rank < 5: top5 += 1
            if rank < 10: top10 += 1
        total += 1
        
    print(f"\n--- Evaluation Results ---")
    print(f"Total Evaluated : {total}")
    print(f"Top-1 Accuracy  : {top1/total*100:.2f}%")
    print(f"Top-3 Accuracy  : {top3/total*100:.2f}%")
    print(f"Top-5 Accuracy  : {top5/total*100:.2f}%")
    print(f"Top-10 Accuracy : {top10/total*100:.2f}%")

if __name__ == '__main__':
    eval_topk('ai/models/t1_placement_net_v2.pt', 'ai/data/t1_local_test.jsonl')
