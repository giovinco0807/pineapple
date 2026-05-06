#!/usr/bin/env python3
import os
import sys
import glob
from pathlib import Path
import numpy as np
import torch
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
def evaluate_regret(model, loader, device, limit=1000):
    model.eval()
    total = 0
    total_regret = 0.0
    
    for batch in loader:
        if total >= limit:
            break
            
        states = batch['state'].to(device)
        masks = batch['mask'].to(device)
        target_evs = batch['evs'].to(device)
        
        pred_q = model(states)
        valid_mask = masks & (target_evs > -1e8)
        
        has_valid = valid_mask.any(dim=1)
        
        pred_q_masked = pred_q.clone()
        pred_q_masked[~valid_mask] = -float('inf')
        
        # Calculate NN's best choice (Top-1 index)
        pred_best_idx = pred_q_masked.argmax(dim=1) # (B,)
        
        for i in range(len(states)):
            if total >= limit:
                break
            if not has_valid[i]:
                continue
                
            # Ground truth array for this state
            true_evs = target_evs[i]
            val_mask = valid_mask[i]
            
            # Absolute best EV (Ground Truth)
            gt_best_ev = true_evs[val_mask].max().item()
            
            # Actual EV of the action the NN chose
            chosen_action_idx = pred_best_idx[i].item()
            nn_actual_ev = true_evs[chosen_action_idx].item()
            
            regret = gt_best_ev - nn_actual_ev
            
            if regret > 0:
                total_regret += regret
            
            total += 1
            
    return total_regret, total

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
    
    print("Loading Validation Set (Chunk 4)...")
    states = np.load(state_files[-1])
    masks = np.load(mask_files[-1])
    evs = np.load(ev_files[-1])
    
    val_ds = T3Dataset(states, masks, evs)
    # Batch size 128 to easily slice exactly 1000
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    
    print(f"Evaluating exactly 1000 states for EV Regret...\n")
    total_regret, total = evaluate_regret(model, val_loader, device, limit=1000)
    
    print("==================================================")
    print("                EV REGRET SUMMARY                 ")
    print("==================================================")
    print(f"Total Evaluated   : {total}")
    print(f"Average EV Regret : {total_regret / total:.4f} points")
    print("==================================================")

if __name__ == "__main__":
    main()
