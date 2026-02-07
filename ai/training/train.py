"""
Training script for Fantasyland model.
"""

import argparse
import json
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import FantasylandDataset, collate_fn
from model import FantasylandModel, FantasylandModelLarge


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        hand = batch['hand'].to(device)
        labels = batch['labels'].to(device)
        n_cards = batch['n_cards'].to(device)
        
        optimizer.zero_grad()
        logits = model(hand, n_cards)
        
        # Flatten for loss calculation
        loss = criterion(logits.view(-1, 4), labels.view(-1))
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # Accuracy calculation (ignore padding)
        preds = logits.argmax(dim=-1)
        mask = labels != -1
        correct += ((preds == labels) & mask).sum().item()
        total += mask.sum().item()
    
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # Per-class accuracy
    class_correct = [0, 0, 0, 0]  # top, mid, bot, discard
    class_total = [0, 0, 0, 0]
    
    with torch.no_grad():
        for batch in loader:
            hand = batch['hand'].to(device)
            labels = batch['labels'].to(device)
            n_cards = batch['n_cards'].to(device)
            
            logits = model(hand, n_cards)
            loss = criterion(logits.view(-1, 4), labels.view(-1))
            
            total_loss += loss.item()
            
            preds = logits.argmax(dim=-1)
            mask = labels != -1
            correct += ((preds == labels) & mask).sum().item()
            total += mask.sum().item()
            
            # Per-class stats
            for c in range(4):
                class_mask = (labels == c)
                class_correct[c] += ((preds == labels) & class_mask).sum().item()
                class_total[c] += class_mask.sum().item()
    
    class_acc = []
    for c in range(4):
        if class_total[c] > 0:
            class_acc.append(class_correct[c] / class_total[c])
        else:
            class_acc.append(0.0)
    
    return total_loss / len(loader), correct / total, class_acc


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset
    print(f"Loading data from {args.data}")
    dataset = FantasylandDataset(args.data)
    
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
    if args.large:
        model = FantasylandModelLarge()
        print("Using large model")
    else:
        model = FantasylandModel(
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
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    # Training loop
    best_val_acc = 0
    history = []
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 70)
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, class_acc = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'class_acc': class_acc,
            'lr': scheduler.get_last_lr()[0]
        })
        
        # Print progress
        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2%} | "
              f"T/M/B/D: {class_acc[0]:.0%}/{class_acc[1]:.0%}/{class_acc[2]:.0%}/{class_acc[3]:.0%}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': vars(args)
            }, args.output)
            print(f"  → Saved best model (acc: {val_acc:.2%})")
    
    print("-" * 70)
    print(f"Training complete! Best validation accuracy: {best_val_acc:.2%}")
    
    # Save history
    history_path = args.output.replace('.pt', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fantasyland model")
    
    # Data
    parser.add_argument('--data', type=str, required=True, help='Path to JSONL data file')
    parser.add_argument('--output', type=str, default='best_model.pt', help='Output model path')
    
    # Model
    parser.add_argument('--large', action='store_true', help='Use large model')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    
    args = parser.parse_args()
    main(args)
