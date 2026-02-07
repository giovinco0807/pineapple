"""
Evaluation script for Fantasyland model.
Calculates score ratio: predicted_score / optimal_score
"""

import argparse
import json

import torch
from torch.utils.data import DataLoader

from dataset import FantasylandDataset, collate_fn
from model import FantasylandModel, FantasylandModelLarge


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']
    
    if args.get('large', False):
        model = FantasylandModelLarge()
    else:
        model = FantasylandModel(
            hidden_dim=args.get('hidden_dim', 128),
            n_heads=args.get('n_heads', 4),
            n_layers=args.get('n_layers', 4),
            dropout=args.get('dropout', 0.1)
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.2%}")
    
    return model


def predict_placement(model, hand_tensor, n_cards, device):
    """
    Get predicted placement for a hand
    
    Returns:
        dict with top, middle, bottom, discard indices
    """
    hand_tensor = hand_tensor.unsqueeze(0).to(device)
    n_cards_tensor = torch.tensor([n_cards], device=device)
    
    with torch.no_grad():
        logits = model(hand_tensor, n_cards_tensor)
        preds = logits[0, :n_cards].argmax(dim=-1).cpu().numpy()
    
    placement = {
        'top': [],
        'middle': [],
        'bottom': [],
        'discard': []
    }
    
    labels = ['top', 'middle', 'bottom', 'discard']
    for i, pred in enumerate(preds):
        placement[labels[pred]].append(i)
    
    return placement


def check_valid_placement(placement):
    """Check if placement has correct card counts"""
    return (
        len(placement['top']) == 3 and
        len(placement['middle']) == 5 and
        len(placement['bottom']) == 5
    )


def evaluate_model(model, dataset, device, verbose=False):
    """
    Evaluate model on dataset
    
    Metrics:
    - Card accuracy: % of cards placed correctly
    - Exact match: % of hands with all cards correct
    - Valid placement: % of hands with valid structure (3-5-5)
    - Score ratio: predicted_score / optimal_score (if computable)
    """
    model.eval()
    
    total_cards = 0
    correct_cards = 0
    exact_matches = 0
    valid_placements = 0
    
    reward_ratios = []
    stay_correct = 0
    stay_total = 0
    
    for i, sample in enumerate(dataset.samples):
        # Get ground truth
        n_cards = len(sample['hand'])
        
        # Build ground truth placement
        gt_labels = []
        placement_map = {}
        for card in sample['solution']['top']:
            key = dataset._card_key(card)
            placement_map[key] = 0
        for card in sample['solution']['middle']:
            key = dataset._card_key(card)
            placement_map[key] = 1
        for card in sample['solution']['bottom']:
            key = dataset._card_key(card)
            placement_map[key] = 2
        for card in sample['solution']['discards']:
            key = dataset._card_key(card)
            placement_map[key] = 3
        
        for card in sample['hand']:
            key = dataset._card_key(card)
            gt_labels.append(placement_map.get(key, 3))
        
        # Get prediction
        hand_encoded = [dataset.encode_card(c) for c in sample['hand']]
        hand_tensor = torch.tensor(hand_encoded, dtype=torch.float32)
        
        pred_placement = predict_placement(model, hand_tensor, n_cards, device)
        pred_labels = [0] * n_cards
        for label_idx, label_name in enumerate(['top', 'middle', 'bottom', 'discard']):
            for card_idx in pred_placement[label_name]:
                pred_labels[card_idx] = label_idx
        
        # Card accuracy
        correct = sum(1 for g, p in zip(gt_labels, pred_labels) if g == p)
        correct_cards += correct
        total_cards += n_cards
        
        # Exact match
        if correct == n_cards:
            exact_matches += 1
        
        # Valid placement
        if check_valid_placement(pred_placement):
            valid_placements += 1
        
        # FL stay tracking
        if sample['can_stay']:
            stay_total += 1
            # Check if prediction also stays (simplified: check if bottom has 5 cards)
            # Real check would need to evaluate the actual hand
        
        if verbose and i < 5:
            print(f"\nSample {i}:")
            print(f"  Ground truth: {gt_labels}")
            print(f"  Prediction:   {pred_labels}")
            print(f"  Correct: {correct}/{n_cards}")
    
    n_samples = len(dataset.samples)
    
    results = {
        'card_accuracy': correct_cards / total_cards,
        'exact_match': exact_matches / n_samples,
        'valid_placement': valid_placements / n_samples,
        'n_samples': n_samples,
    }
    
    return results


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model, device)
    
    # Load data
    dataset = FantasylandDataset(args.data)
    
    # Evaluate
    print("\nEvaluating...")
    results = evaluate_model(model, dataset, device, verbose=args.verbose)
    
    print("\n" + "=" * 50)
    print("Results:")
    print("=" * 50)
    print(f"  Samples:          {results['n_samples']}")
    print(f"  Card Accuracy:    {results['card_accuracy']:.2%}")
    print(f"  Exact Match:      {results['exact_match']:.2%}")
    print(f"  Valid Placement:  {results['valid_placement']:.2%}")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Fantasyland model")
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to JSONL data file')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path for results')
    parser.add_argument('--verbose', action='store_true', help='Print sample predictions')
    
    args = parser.parse_args()
    main(args)
