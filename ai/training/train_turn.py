"""
Training script for Turn Value model (T1-T4).
"""

import argparse
import json
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from turn_dataset import TurnValueDataset, collate_fn
from model import TurnValueModel


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_l1 = 0
    
    for batch in loader:
        board = batch['board'].to(device)
        rows = batch['rows'].to(device)
        ev_target = batch['ev'].to(device).unsqueeze(1)  # [batch, 1]
        
        optimizer.zero_grad()
        ev_pred = model(board, rows)
        
        loss = criterion(ev_pred, ev_target)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_l1 += torch.nn.functional.l1_loss(ev_pred, ev_target).item()
    
    n_batches = len(loader)
    return total_loss / n_batches, total_l1 / n_batches


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_l1 = 0
    
    with torch.no_grad():
        for batch in loader:
            board = batch['board'].to(device)
            rows = batch['rows'].to(device)
            ev_target = batch['ev'].to(device).unsqueeze(1)
            
            ev_pred = model(board, rows)
            loss = criterion(ev_pred, ev_target)
            
            total_loss += loss.item()
            total_l1 += torch.nn.functional.l1_loss(ev_pred, ev_target).item()
            
    n_batches = len(loader)
    return total_loss / n_batches, total_l1 / n_batches


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset
    print(f"Loading data from {args.data}")
    dataset = TurnValueDataset(args.data)
    
    if len(dataset) == 0:
        print("Dataset is empty. Exiting.")
        return
        
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Model
    model = TurnValueModel(
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout
    )
    print(f"Model: hidden={args.hidden_dim}, heads={args.n_heads}, layers={args.n_layers}")
    
    model = model.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    
    # Optimizer & Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_l1 = float('inf')
    history = []
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 70)
    
    for epoch in range(args.epochs):
        train_loss, train_l1 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_l1 = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_l1': train_l1,
            'val_loss': val_loss,
            'val_l1': val_l1,
            'lr': scheduler.get_last_lr()[0]
        })
        
        # Print progress
        print(f"Epoch {epoch+1:3d} | "
              f"Train MSE: {train_loss:.4f}, L1: {train_l1:.4f} | "
              f"Val MSE: {val_loss:.4f}, L1: {val_l1:.4f}")
        
        # Save best model based on MAE (L1 loss)
        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_l1': val_l1,
                'args': vars(args)
            }, args.output)
            print(f"  → Saved best model (MAE: {val_l1:.4f})")
    
    print("-" * 70)
    print(f"Training complete! Best validation MAE: {best_val_l1:.4f}")
    
    # Save history
    history_path = args.output.replace('.pt', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Turn Value model")
    
    # Data
    parser.add_argument('--data', type=str, required=True, help='Path to JSONL data file')
    parser.add_argument('--output', type=str, default='turn_model.pt', help='Output model path')
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    
    args = parser.parse_args()
    main(args)
