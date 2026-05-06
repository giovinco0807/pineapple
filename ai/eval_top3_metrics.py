#!/usr/bin/env python3
import os
import sys
import glob
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ai.models.t3_policy_value import T3PolicyValueNet

class T3Dataset(Dataset):
    def __init__(self, states, masks, evs):
        self.states = torch.from_numpy(states).float()
        self.masks = torch.from_numpy(masks).bool()
        self.evs = torch.from_numpy(evs).float()
        
    def __len__(self):
        return len(self.states)
        
    def __getitem__(self, idx):
        return {
            'state': self.states[idx],
            'mask': self.masks[idx],
            'evs': self.evs[idx]
        }

@torch.no_grad()
def evaluate_top_k(model, loader, device):
    model.eval()
    total = 0
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    
    for batch in loader:
        states = batch['state'].to(device)
        masks = batch['mask'].to(device)
        target_evs = batch['evs'].to(device)
        
        pred_q = model(states)
        valid_mask = masks & (target_evs > -1e8)
        
        if not valid_mask.any():
            continue
            
        pred_q_masked = pred_q.clone()
        pred_q_masked[~valid_mask] = -float('inf')
        
        target_evs_masked = target_evs.clone()
        target_evs_masked[~valid_mask] = -float('inf')
        
        true_best_idx = target_evs_masked.argmax(dim=1) # (B,)
        
        # Top-5 predictions
        # k=5, but some states might have less than 5 valid actions
        # we can safely get top 5, invalid actions will have -inf score and won't match true_best_idx
        _, pred_top5_idx = torch.topk(pred_q_masked, k=5, dim=1) # (B, 5)
        
        has_valid = valid_mask.any(dim=1)
        
        # Filter to only valid rows
        true_best = true_best_idx[has_valid]
        pred_top5 = pred_top5_idx[has_valid]
        
        total += has_valid.sum().item()
        
        # Check Top-1
        top1_correct += (pred_top5[:, 0] == true_best).sum().item()
        
        # Check Top-3
        # True if true_best is in the first 3 columns
        matches_top3 = (pred_top5[:, :3] == true_best.unsqueeze(1)).any(dim=1)
        top3_correct += matches_top3.sum().item()
        
        # Check Top-5
        matches_top5 = (pred_top5 == true_best.unsqueeze(1)).any(dim=1)
        top5_correct += matches_top5.sum().item()
        
    return top1_correct, top3_correct, top5_correct, total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T3PolicyValueNet().to(device)
    model_path = "ai/models/t3_policy.pt"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    print(f"Loading data chunks...")
    state_files = sorted(glob.glob("data/t3_dataset/states_chunk_*.npy"))
    mask_files = sorted(glob.glob("data/t3_dataset/action_masks_chunk_*.npy"))
    ev_files = sorted(glob.glob("data/t3_dataset/action_evs_chunk_*.npy"))
    
    # Just load chunk 4 for a quick robust validation set (100k states)
    print("Loading Validation Set (Chunk 4)...")
    states = np.load(state_files[-1])
    masks = np.load(mask_files[-1])
    evs = np.load(ev_files[-1])
    
    val_ds = T3Dataset(states, masks, evs)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False)
    
    print(f"Evaluating {len(val_ds)} validation states...\n")
    top1, top3, top5, total = evaluate_top_k(model, val_loader, device)
    
    print("==================================================")
    print("             TOP-K ACCURACY METRICS               ")
    print("==================================================")
    print(f"Total Evaluated : {total}")
    print(f"Top-1 Accuracy  : {top1/total * 100:.2f}%")
    print(f"Top-3 Accuracy  : {top3/total * 100:.2f}%")
    print(f"Top-5 Accuracy  : {top5/total * 100:.2f}%")
    print("==================================================")

if __name__ == "__main__":
    main()
