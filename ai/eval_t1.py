import torch
import json
import numpy as np
from ai.models.t1_network import T1PlacementNet, CARD_DIM, NUM_CLASSES, encode_card_str
from ai.train_t1_placement import T1PlacementDataset, parse_board
from torch.utils.data import DataLoader

def eval_model(model_path, data_path, num_examples=5):
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

    print(f"Loaded epoch {checkpoint.get('epoch', '?')} (Val Hand Acc: {checkpoint.get('val_hand_acc', 0):.4f})")

    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))
    
    print(f"Loaded {len(samples)} samples from {data_path}")
    
    # Process dataset
    ds = T1PlacementDataset(samples, top_k=1)
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    
    total_card, total_hand, total_samples = 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            features = batch['features'].to(device)
            hard_labels = batch['hard_labels'].to(device)
            
            logits, _ = model(features)
            pred = logits.argmax(dim=-1)
            
            card_c = (pred == hard_labels).float()
            total_card += card_c.sum().item()
            total_hand += card_c.prod(dim=1).sum().item()
            total_samples += features.size(0)
    
    print(f"\nEvaluation on {data_path}:")
    print(f"Card Accuracy: {total_card / (total_samples * 3):.4f}")
    print(f"Hand Accuracy: {total_hand / total_samples:.4f}")

    print("\n--- Sample Predictions ---")
    
    # Show some examples
    target_map_rev = {0: 'Top', 1: 'Mid', 2: 'Bot', 3: 'Discard'}
    
    for i in range(min(num_examples, len(samples))):
        s = samples[i]
        # create features for single item
        processed = ds._process(s, top_k=1)
        if not processed: continue
        feat = torch.from_numpy(processed['features']).unsqueeze(0).to(device)
        hard_l = processed['hard_labels']
        
        logits, ev_pred = model(feat)
        pred = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
        
        hand = s['hand'].split()
        board = s['board']
        best_p = s['placements'][0]
        
        print(f"\nExample {i+1}:")
        print(f"Board: {board}")
        print(f"Hand:  {s['hand']}")
        
        pred_strs = []
        gt_strs = []
        for j, c in enumerate(hand):
            p_str = target_map_rev[pred[j]]
            g_str = target_map_rev[hard_l[j]]
            pred_strs.append(f"{c}->{p_str}")
            gt_strs.append(f"{c}->{g_str}")
            
        print(f"Predicted : {', '.join(pred_strs)}")
        print(f"Ground T  : {', '.join(gt_strs)}")
        print(f"Predicted EV: {ev_pred.item():.2f} | Ground Truth EV: {best_p.get('ev', 0):.2f}")

if __name__ == '__main__':
    eval_model('ai/models/t1_placement_net_v1.pt', 'ai/data/t1_mc_test.jsonl')
