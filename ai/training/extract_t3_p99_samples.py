#!/usr/bin/env python3
"""
Extract high-regret T3 validation samples.

This script reproduces the train_t3_oracle validation split, evaluates a
checkpoint, and writes the P99 tail samples to JSONL for inspection.

Example:
  python ai/training/extract_t3_p99_samples.py ^
    --data-dir ai/data/t3_oracle_full ^
    --model ai/data/t3_oracle/t3_policyvalue_v2_best.pt ^
    --out ai/data/t3_oracle/t3_p99_samples.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

AI_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = AI_DIR.parent
sys.path.insert(0, str(AI_DIR))
sys.path.insert(0, str(REPO_ROOT))

from engine.encoding import ALL_CARDS  # noqa: E402
from training.train_t3_oracle_v2 import T3PolicyValueNet  # noqa: E402


LOC_NAMES = [
    "my_top",
    "my_middle",
    "my_bottom",
    "opp_top",
    "opp_middle",
    "opp_bottom",
    "dealt",
    "my_discard",
    "unseen",
]


def find_npz_files(data_dir: Path):
    sub_dirs = [d for d in data_dir.iterdir() if d.is_dir() and list(d.glob("*.npz"))]
    if sub_dirs:
        files = []
        for d in sorted(sub_dirs):
            files.extend(sorted(d.glob("*.npz")))
        return files
    return sorted(data_dir.glob("*.npz"))


def load_dataset(data_dir: Path, min_state_dim: int, n_actions: int = 27):
    files = find_npz_files(data_dir)
    
    if not files:
        # Fallback to .npy files in the directory
        states_path = data_dir / "states.npy"
        evs_path = data_dir / "action_evs.npy"
        masks_path = data_dir / "valid_masks.npy"
        
        if not states_path.exists():
            raise FileNotFoundError(f"No NPZ files found, and states.npy is missing in {data_dir}")
            
        states = np.load(states_path).astype(np.float32)
        evs = np.load(evs_path)[:, :n_actions].astype(np.float32)
        masks = np.load(masks_path)[:, :n_actions]
        
        # Calculate bests based on valid mask
        bests = np.max(np.where(masks, evs, -1e9), axis=1).astype(np.float32)
        
        if states.ndim != 2 or states.shape[1] < min_state_dim:
            raise ValueError(f"states.npy has invalid shape {states.shape}")
            
        return {
            "states": states,
            "evs": evs,
            "masks": masks,
            "bests": bests,
            "refs": [("npy", 0, i) for i in range(len(states))],
            "state_dim": states.shape[1],
            "files": [str(data_dir)]
        }

    states_parts = []
    ev_parts = []
    mask_parts = []
    best_parts = []
    refs = []
    state_dim = None

    for file_idx, path in enumerate(files):
        npz = np.load(path)
        states = npz["states"].astype(np.float32)
        evs = npz["action_evs"].astype(np.float32)[:, :n_actions]
        if "action_masks" in npz:
            masks = npz["action_masks"].astype(bool)[:, :n_actions]
        else:
            masks = npz["valid_masks"].astype(bool)[:, :n_actions]
        if "best_evs" in npz:
            bests = npz["best_evs"].astype(np.float32)
        else:
            bests = np.max(np.where(masks, evs, -1e9), axis=1).astype(np.float32)

        if states.ndim != 2 or states.shape[1] < min_state_dim:
            raise ValueError(f"{path} has invalid state shape {states.shape}")
        if state_dim is None:
            state_dim = states.shape[1]
        elif states.shape[1] != state_dim:
            raise ValueError(f"Mixed state dims: {state_dim} and {states.shape[1]} in {path}")

        states_parts.append(states)
        ev_parts.append(evs)
        mask_parts.append(masks)
        best_parts.append(bests)
        refs.extend((str(path), file_idx, local_idx) for local_idx in range(len(states)))

    return {
        "states": np.concatenate(states_parts, axis=0),
        "evs": np.concatenate(ev_parts, axis=0),
        "masks": np.concatenate(mask_parts, axis=0),
        "bests": np.concatenate(best_parts, axis=0),
        "refs": refs,
        "state_dim": state_dim,
        "files": [str(p) for p in files],
    }


def load_model(model_path: Path, state_dim: int, n_actions: int, device: torch.device):
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt_state_dim = ckpt.get("state_dim", state_dim)
        hidden = ckpt.get("hidden", 1024)
        n_blocks = ckpt.get("n_blocks", 4)
        model = T3PolicyValueNet(
            state_dim=ckpt_state_dim,
            n_actions=ckpt.get("n_actions", n_actions),
            hidden=hidden,
            n_blocks=n_blocks,
        )
        if ckpt_state_dim != state_dim:
            raise ValueError(
                f"Checkpoint state_dim={ckpt_state_dim} but dataset state_dim={state_dim}"
            )
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model = T3PolicyValueNet(state_dim=state_dim, n_actions=n_actions)
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    return model


def decode_state(state: np.ndarray):
    card_matrix = state[: 54 * 9].reshape(54, 9)
    locations = {name: [] for name in LOC_NAMES}
    for card_idx, card in enumerate(ALL_CARDS):
        loc_idx = int(np.argmax(card_matrix[card_idx]))
        if card_matrix[card_idx, loc_idx] > 0.5:
            locations[LOC_NAMES[loc_idx]].append(card)

    meta = state[54 * 9 : 54 * 9 + 6].tolist()
    game_features = state[54 * 9 + 6 : 54 * 9 + 36].tolist()
    return {
        "board": {
            "top": locations["my_top"],
            "middle": locations["my_middle"],
            "bottom": locations["my_bottom"],
        },
        "opponent_board": {
            "top": locations["opp_top"],
            "middle": locations["opp_middle"],
            "bottom": locations["opp_bottom"],
        },
        "dealt": locations["dealt"],
        "discards": locations["my_discard"],
        "meta": {
            "turn_scaled": meta[0],
            "is_btn": meta[1],
            "is_fl": meta[2],
            "opp_is_fl": meta[3],
            "chips_self_scaled": meta[4],
            "chips_opponent_scaled": meta[5],
        },
        "features": {
            "row_slots": game_features[0:6],
            "fl_self": game_features[6:12],
            "fl_opp": game_features[12:16],
            "hand_ranks": game_features[16:22],
            "bust_risk": game_features[22:24],
            "draws": game_features[24:30],
        },
    }


def topk_valid(values: np.ndarray, valid_mask: np.ndarray, k: int):
    valid_idx = np.flatnonzero(valid_mask)
    order = valid_idx[np.argsort(values[valid_idx])[::-1]]
    return order[:k].tolist()


def evaluate(model, data, batch_size: int, device: torch.device):
    states = data["states"]
    evs = data["evs"]
    masks = data["masks"]
    n = len(states)

    pred_idx_parts = []
    value_parts = []
    logits_parts = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_states = torch.from_numpy(states[start:end]).to(device)
            batch_masks = torch.from_numpy(masks[start:end]).to(device)
            logits, values = model(batch_states, batch_masks)
            pred_idx_parts.append(torch.argmax(logits, dim=-1).cpu().numpy())
            value_parts.append(values.cpu().numpy())
            logits_parts.append(logits.cpu().numpy())

    pred_idx = np.concatenate(pred_idx_parts)
    logits = np.concatenate(logits_parts)
    pred_ev = evs[np.arange(n), pred_idx]
    true_idx = np.argmax(np.where(masks, evs, -1e9), axis=1)
    best_ev = evs[np.arange(n), true_idx]
    regrets = best_ev - pred_ev
    values = np.concatenate(value_parts)
    return regrets, pred_idx, pred_ev, true_idx, best_ev, values, logits


def main():
    parser = argparse.ArgumentParser(description="Extract T3 P99 high-regret samples")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="ai/data/t3_oracle/t3_p99_samples.jsonl")
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--min-val", type=int, default=1000)
    parser.add_argument("--min-state-dim", type=int, default=520)
    parser.add_argument("--n-actions", type=int, default=27)
    parser.add_argument("--top-n", type=int, default=300)
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    args = parser.parse_args()

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    data = load_dataset(Path(args.data_dir), args.min_state_dim, args.n_actions)
    n_total = len(data["states"])
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    perm = torch.randperm(n_total, generator=gen).numpy()
    n_val = min(max(args.min_val, int(n_total * args.val_frac)), n_total - 1)
    val_idx = perm[n_total - n_val :]

    val_data = {
        "states": data["states"][val_idx],
        "evs": data["evs"][val_idx],
        "masks": data["masks"][val_idx],
        "refs": [data["refs"][int(i)] for i in val_idx],
        "state_dim": data["state_dim"],
    }
    model = load_model(Path(args.model), data["state_dim"], data["evs"].shape[1], device)
    regrets, pred_idx, pred_ev, true_idx, best_ev, values, logits = evaluate(
        model, val_data, args.batch_size, device
    )

    threshold = float(np.percentile(regrets, args.threshold_percentile))
    tail_idx = np.flatnonzero(regrets >= threshold)
    tail_idx = tail_idx[np.argsort(regrets[tail_idx])[::-1]]
    if args.top_n > 0:
        tail_idx = tail_idx[: args.top_n]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rank, i in enumerate(tail_idx, start=1):
            ev_row = val_data["evs"][i]
            mask_row = val_data["masks"][i]
            true_top3 = topk_valid(ev_row, mask_row, 3)
            pred_top3 = topk_valid(logits[i], mask_row, 3)
            ref_path, file_idx, local_idx = val_data["refs"][i]
            record = {
                "tail_rank": rank,
                "val_index": int(i),
                "global_index": int(val_idx[i]),
                "source_file": ref_path,
                "source_file_index": int(file_idx),
                "source_local_index": int(local_idx),
                "regret": float(regrets[i]),
                "best_ev": float(best_ev[i]),
                "pred_ev": float(pred_ev[i]),
                "value_pred": float(values[i]),
                "true_action_idx": int(true_idx[i]),
                "pred_action_idx": int(pred_idx[i]),
                "true_top3_by_ev": [
                    {"action_idx": int(a), "ev": float(ev_row[a])} for a in true_top3
                ],
                "pred_top3_by_logit": [
                    {
                        "action_idx": int(a),
                        "logit": float(logits[i, a]),
                        "ev": float(ev_row[a]),
                    }
                    for a in pred_top3
                ],
                "valid_action_count": int(mask_row.sum()),
                "state": decode_state(val_data["states"][i]),
            }
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    summary = {
        "data_dir": str(args.data_dir),
        "model": str(args.model),
        "state_dim": int(data["state_dim"]),
        "n_total": int(n_total),
        "n_val": int(n_val),
        "threshold_percentile": args.threshold_percentile,
        "threshold": threshold,
        "mean_regret": float(np.mean(regrets)),
        "median_regret": float(np.median(regrets)),
        "p90_regret": float(np.percentile(regrets, 90)),
        "p99_regret": float(np.percentile(regrets, 99)),
        "max_regret": float(np.max(regrets)),
        "zero_regret_pct": float(np.mean(regrets < 0.01)),
        "low_regret_pct": float(np.mean(regrets < 1.0)),
        "written": int(len(tail_idx)),
        "out": str(out_path),
    }
    summary_out = Path(args.summary_out) if args.summary_out else out_path.with_suffix(".summary.json")
    with summary_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
