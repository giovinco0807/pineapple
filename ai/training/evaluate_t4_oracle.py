"""
Evaluate T4 Oracle Model
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ai.training.train_t4_oracle import T4PolicyValueNet, T4DatasetNPI
from ai.engine.action_space import get_action_from_semantic_index_if_valid, POSITIONS

def main():
    parser = argparse.ArgumentParser(description="Evaluate T4 Oracle Model")
    parser.add_argument("--data-dir", default="ai/data/t4_dataset_v2")
    parser.add_argument("--model", default="ai/data/t4_oracle_v2/t4_policyvalue_best.pt")
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data
    data_dir = Path(args.data_dir)
    ds = T4DatasetNPI(data_dir, n_actions=27, device=device, verbose=False)
    
    # Validation split (consistent with train_t4_oracle.py)
    N = len(ds)
    gen = torch.Generator()
    gen.manual_seed(42)
    perm = torch.randperm(N, generator=gen)
    n_val = min(N - 1, max(1, int(N * 0.2))) if N < 2000 else max(1000, int(N * 0.1))
    val_idx = perm[-n_val:] if n_val > 0 else perm[:0]
    
    # Load model
    model_path = Path(args.model)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    
    model = T4PolicyValueNet(
        state_dim=ckpt['state_dim'],
        n_actions=ckpt['n_actions'],
        hidden=ckpt['hidden'],
        n_blocks=ckpt['n_blocks']
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {ckpt['epoch']}")
    print(f"Validation regret from training: {ckpt['val_regret']:.3f}, Top1: {ckpt['val_top1']:.1%}")
    print("-" * 60)
    
    # Pick a few random samples
    np.random.seed(42)
    sample_size = min(args.samples, len(val_idx))
    sample_indices = np.random.choice(val_idx.cpu().numpy(), size=sample_size, replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            state = ds.states[idx].unsqueeze(0)
            evs = ds.evs[idx]
            mask = ds.masks[idx]
            best_ev = ds.bests[idx].item()
            
            logits, value = model(state, mask.unsqueeze(0))
            logits = logits.squeeze(0)
            value = value.item()
            
            # Predict
            has_ev = (evs > -1e8) & mask
            masked_logits = logits.masked_fill(~has_ev, -1e9)
            pred_best_action = masked_logits.argmax().item()
            true_best_action = evs.masked_fill(~has_ev, -1e9).argmax().item()
            
            pred_ev_for_chosen = evs[pred_best_action].item()
            
            print(f"Sample {i+1}:")
            print(f"  Predicted Value: {value:.2f}")
            print(f"  True Best EV:    {best_ev:.2f}")
            print(f"  Chosen Action EV:{pred_ev_for_chosen:.2f}")
            print(f"  Regret:          {best_ev - pred_ev_for_chosen:.2f}")
            if pred_best_action == true_best_action:
                print("  => CORRECT Top-1 Action!")
            else:
                print("  => Suboptimal Action.")
            print()

if __name__ == "__main__":
    main()
