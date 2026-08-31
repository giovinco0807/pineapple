"""
T3 Oracle Diagnostic — Detailed validation metrics

Evaluates a saved checkpoint with extended metrics:
  - val_regret (mean, median, p90)
  - top1, top3 hit rates
  - value_mse
  - regret distribution analysis
  - EV calibration check
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_model_and_data(model_path, data_dir, device='cuda'):
    """Load model checkpoint and validation data."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get model config from checkpoint
    state_dim = ckpt.get('state_dim', 520)
    n_actions = ckpt.get('n_actions', 27)
    hidden = ckpt.get('hidden', 1024)
    n_blocks = ckpt.get('n_blocks', 4)
    
    print(f"Model config: state_dim={state_dim}, n_actions={n_actions}, "
          f"hidden={hidden}, n_blocks={n_blocks}")
    
    # Import model class
    import sys
    sys.path.insert(0, '.')
    from ai.training.train_t3_oracle_v2 import T3PolicyValueNet
    
    model = T3PolicyValueNet(
        state_dim=state_dim, n_actions=n_actions,
        hidden=hidden, n_blocks=n_blocks
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Load data
    data_dir = Path(data_dir)
    states = torch.from_numpy(
        np.load(data_dir / 'states.npy', mmap_mode='r')[:].astype(np.float32)
    ).to(device)
    evs = torch.from_numpy(
        np.load(data_dir / 'action_evs.npy', mmap_mode='r')[:, :n_actions].astype(np.float32)
    ).to(device)
    masks = torch.from_numpy(
        np.load(data_dir / 'valid_masks.npy', mmap_mode='r')[:, :n_actions]
    ).to(device)
    
    evs_for_best = evs.clone()
    evs_for_best[~masks] = -1e9
    bests = evs_for_best.max(dim=-1).values
    
    # Use same split as training (seed=42, 10% val)
    N = len(states)
    gen = torch.Generator()
    gen.manual_seed(42)
    perm = torch.randperm(N, generator=gen)
    n_val = max(1000, int(N * 0.1))
    val_idx = perm[N - n_val:]
    
    print(f"Validation set: {len(val_idx):,} samples")
    
    return model, states, evs, masks, bests, val_idx


def detailed_eval(model, states, evs, masks, bests, val_idx, batch_size=4096):
    """Run detailed evaluation on validation set."""
    all_regrets = []
    all_top1 = []
    all_top3 = []
    all_value_errors = []
    all_pred_evs = []
    all_best_evs = []
    
    with torch.no_grad():
        for i in range(0, len(val_idx), batch_size):
            idx = val_idx[i:i + batch_size]
            s = states[idx]
            e = evs[idx]
            m = masks[idx]
            b = bests[idx]
            
            logits, value = model(s, m)
            
            has_ev = (e > -1e8) & m
            
            # Top-1
            pred_best = logits.masked_fill(~has_ev, -1e9).argmax(dim=-1)
            true_best = e.masked_fill(~has_ev, -1e9).argmax(dim=-1)
            valid = has_ev.any(dim=-1)
            
            top1_correct = (pred_best[valid] == true_best[valid])
            all_top1.extend(top1_correct.cpu().numpy().tolist())
            
            # Top-3
            _, pred_top3 = logits.masked_fill(~has_ev, -1e9).topk(
                min(3, logits.shape[-1]), dim=-1)
            top3_hit = (pred_top3[valid] == true_best[valid].unsqueeze(1)).any(dim=-1)
            all_top3.extend(top3_hit.cpu().numpy().tolist())
            
            # Regret
            pred_ev = e.gather(1, pred_best.unsqueeze(1)).squeeze(1)
            best_ev = e.masked_fill(~has_ev, -1e9).max(dim=-1).values
            regret = (best_ev - pred_ev)[valid]
            all_regrets.extend(regret.cpu().numpy().tolist())
            
            # Value head error
            value_err = (value[valid] - b[valid]).abs()
            all_value_errors.extend(value_err.cpu().numpy().tolist())
            
            # For calibration
            all_pred_evs.extend(pred_ev[valid].cpu().numpy().tolist())
            all_best_evs.extend(best_ev[valid].cpu().numpy().tolist())
    
    regrets = np.array(all_regrets)
    top1 = np.array(all_top1)
    top3 = np.array(all_top3)
    val_errs = np.array(all_value_errors)
    pred_evs = np.array(all_pred_evs)
    best_evs = np.array(all_best_evs)
    
    results = {
        'val_regret_mean': float(regrets.mean()),
        'val_regret_median': float(np.median(regrets)),
        'val_regret_p90': float(np.percentile(regrets, 90)),
        'val_regret_p99': float(np.percentile(regrets, 99)),
        'val_regret_std': float(regrets.std()),
        'val_top1': float(top1.mean()),
        'val_top3': float(top3.mean()),
        'value_mae': float(val_errs.mean()),
        'value_mse': float((val_errs ** 2).mean()),
        'zero_regret_pct': float((regrets < 0.01).mean()),
        'low_regret_pct': float((regrets < 1.0).mean()),
        'best_ev_mean': float(best_evs.mean()),
        'best_ev_std': float(best_evs.std()),
        'pred_ev_mean': float(pred_evs.mean()),
        'n_samples': len(regrets),
    }
    
    # Normalized regret (regret / best_ev_std)
    results['normalized_regret'] = results['val_regret_mean'] / max(results['best_ev_std'], 0.01)
    
    return results, regrets


def print_results(results):
    print(f"\n{'='*60}")
    print(f"  T3 Oracle Detailed Diagnostics")
    print(f"{'='*60}")
    print(f"  Samples evaluated: {results['n_samples']:,}")
    print()
    print(f"  --- Regret ---")
    print(f"  Mean:   {results['val_regret_mean']:.4f}")
    print(f"  Median: {results['val_regret_median']:.4f}")
    print(f"  P90:    {results['val_regret_p90']:.4f}")
    print(f"  P99:    {results['val_regret_p99']:.4f}")
    print(f"  Std:    {results['val_regret_std']:.4f}")
    print(f"  Zero regret (<0.01): {results['zero_regret_pct']:.1%}")
    print(f"  Low regret  (<1.0):  {results['low_regret_pct']:.1%}")
    print()
    print(f"  --- Accuracy ---")
    print(f"  Top-1: {results['val_top1']:.1%}")
    print(f"  Top-3: {results['val_top3']:.1%}")
    print()
    print(f"  --- Value Head ---")
    print(f"  MAE:  {results['value_mae']:.3f}")
    print(f"  MSE:  {results['value_mse']:.3f}")
    print()
    print(f"  --- EV Context ---")
    print(f"  Best EV: mean={results['best_ev_mean']:.2f}, std={results['best_ev_std']:.2f}")
    print(f"  Pred EV: mean={results['pred_ev_mean']:.2f}")
    print(f"  Normalized regret (regret/best_ev_std): {results['normalized_regret']:.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ai/data/t3_oracle/t3_policyvalue_v2_best.pt')
    parser.add_argument('--data-dir', default='D:/ofc_data/t3_train')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model, states, evs, masks, bests, val_idx = load_model_and_data(
        args.model, args.data_dir, device)
    
    results, regrets = detailed_eval(model, states, evs, masks, bests, val_idx)
    print_results(results)
    
    # Save results
    out_path = Path(args.model).parent / 't3_v2_diagnostics.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")
