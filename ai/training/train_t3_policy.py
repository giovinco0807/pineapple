import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class T3PolicyDataset(Dataset):
    def __init__(self, data_dir="data/t3_dataset"):
        self.data_dir = Path(data_dir)
        
        print(f"Loading dataset from {self.data_dir}...")
        self.states = np.load(self.data_dir / "states.npy")
        self.action_evs = np.load(self.data_dir / "action_evs.npy")
        self.action_masks = np.load(self.data_dir / "action_masks.npy")
        
        self.states = torch.tensor(self.states, dtype=torch.float32)
        self.action_evs = torch.tensor(self.action_evs, dtype=torch.float32)
        self.action_masks = torch.tensor(self.action_masks, dtype=torch.bool)
        
    def __len__(self):
        return len(self.states)
        
    def __getitem__(self, idx):
        return self.states[idx], self.action_evs[idx], self.action_masks[idx]

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        return self.act(out + residual)

class T3PolicyMLP(nn.Module):
    def __init__(self, input_dim=490, hidden_dim=1024, output_dim=250, num_blocks=3):
        super().__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        return self.output_layer(x)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = T3PolicyDataset()
    print(f"Loaded {len(dataset)} states.")
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)
    
    model = T3PolicyMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    epochs = 100
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_mae = 0.0
        top1_correct = 0
        total_samples = 0
        valid_train_samples = 0
        
        for batch_states, batch_evs, batch_masks in train_loader:
            batch_states = batch_states.to(device)
            batch_evs = batch_evs.to(device)
            batch_masks = batch_masks.to(device)
            
            optimizer.zero_grad()
            pred_evs = model(batch_states)
            
            # Compute loss only on valid actions
            valid_preds = pred_evs[batch_masks]
            valid_targets = batch_evs[batch_masks]
            
            # Filter out -1e9 defaults (actions valid in Python but pruned/unevaluated by Rust)
            evaluated_mask = valid_targets > -1e8
            valid_preds = valid_preds[evaluated_mask]
            valid_targets = valid_targets[evaluated_mask]
            
            if len(valid_preds) > 0:
                loss = F.mse_loss(valid_preds, valid_targets)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients from huge EV spikes
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item() * batch_states.size(0)
                total_mae += F.l1_loss(valid_preds, valid_targets).item() * batch_states.size(0)
            total_samples += batch_states.size(0)
            
            # Top-1 accuracy (Did it predict the action with the highest True EV?)
            for i in range(batch_states.size(0)):
                mask_i = batch_masks[i]
                if not mask_i.any(): continue
                
                valid_pred_i = pred_evs[i][mask_i]
                valid_targ_i = batch_evs[i][mask_i]
                
                evaluated_mask = valid_targ_i > -1e8
                if not evaluated_mask.any(): continue
                
                valid_pred_i = valid_pred_i[evaluated_mask]
                valid_targ_i = valid_targ_i[evaluated_mask]
                
                pred_best_idx = torch.argmax(valid_pred_i)
                targ_best_idx = torch.argmax(valid_targ_i)
                
                if pred_best_idx == targ_best_idx:
                    top1_correct += 1
                valid_train_samples += 1
                
        avg_train_loss = total_loss / total_samples
        avg_train_mae = total_mae / total_samples
        train_acc = top1_correct / max(1, valid_train_samples)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_top1_correct = 0
        valid_val_samples = 0
        
        with torch.no_grad():
            for batch_states, batch_evs, batch_masks in val_loader:
                batch_states = batch_states.to(device)
                batch_evs = batch_evs.to(device)
                batch_masks = batch_masks.to(device)
                
                pred_evs = model(batch_states)
                
                valid_preds = pred_evs[batch_masks]
                valid_targets = batch_evs[batch_masks]
                
                evaluated_mask = valid_targets > -1e8
                valid_preds = valid_preds[evaluated_mask]
                valid_targets = valid_targets[evaluated_mask]
                
                if len(valid_preds) > 0:
                    loss = F.mse_loss(valid_preds, valid_targets)
                    val_loss += loss.item() * batch_states.size(0)
                    val_mae += F.l1_loss(valid_preds, valid_targets).item() * batch_states.size(0)
                
                for i in range(batch_states.size(0)):
                    mask_i = batch_masks[i]
                    if not mask_i.any(): continue
                    
                    valid_pred_i = pred_evs[i][mask_i]
                    valid_targ_i = batch_evs[i][mask_i]
                    
                    evaluated_mask = valid_targ_i > -1e8
                    if not evaluated_mask.any(): continue
                    
                    valid_pred_i = valid_pred_i[evaluated_mask]
                    valid_targ_i = valid_targ_i[evaluated_mask]
                    
                    if torch.argmax(valid_pred_i) == torch.argmax(valid_targ_i):
                        val_top1_correct += 1
                    valid_val_samples += 1
                        
        avg_val_loss = val_loss / val_size
        avg_val_mae = val_mae / val_size
        val_acc = val_top1_correct / max(1, valid_val_samples)
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1:03d} | "
              f"Train Loss (MSE): {avg_train_loss:.2f}, MAE: {avg_train_mae:.2f}, Acc: {train_acc:.2%} | "
              f"Val Loss: {avg_val_loss:.2f}, MAE: {avg_val_mae:.2f}, Acc: {val_acc:.2%}")
              
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "data/t3_dataset/t3_policy_model.pt")
            
    print("Training complete! Best model saved to data/t3_dataset/t3_policy_model.pt")

if __name__ == "__main__":
    train()
