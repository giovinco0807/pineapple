"""Write per-candidate base model scores for set/listwise reranker features."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.models.action_value_reranker import ActionValueReranker


def parse_weights(value: str, n: int) -> list[float]:
    if not value.strip():
        return [1.0 / max(n, 1)] * n
    weights = [float(part) for part in value.split(",") if part.strip()]
    if len(weights) != n:
        raise ValueError(f"--weights has {len(weights)} values but {n} checkpoints were provided")
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("--weights must sum to a positive value")
    return [w / total for w in weights]


def load_metadata(data_dir: Path) -> dict:
    path = data_dir / "metadata.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@torch.no_grad()
def predict_scores(
    checkpoint: str,
    states: np.ndarray,
    turns: np.ndarray,
    n_samples: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model = ActionValueReranker.from_checkpoint(checkpoint, map_location=device).to(device)
    model.eval()
    scores = np.zeros(n_samples, dtype=np.float32)
    for start in range(0, n_samples, batch_size):
        end = min(n_samples, start + batch_size)
        state = torch.from_numpy(np.array(states[start:end], dtype=np.float32, copy=True)).to(device)
        turn = torch.from_numpy(np.asarray(turns[start:end], dtype=np.int64)).to(device)
        out = model.predict_components(state, turn=turn)
        scores[start:end] = out["score"].detach().cpu().numpy()
    del model
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.empty_cache()
    return scores


@torch.no_grad()
def write_scores(args: argparse.Namespace) -> None:
    data_dir = Path(args.data)
    meta = load_metadata(data_dir)
    states = np.load(data_dir / "states.npy", mmap_mode="r")
    turns = np.asarray(np.load(data_dir / "turns.npy", mmap_mode="r"), dtype=np.int64)
    n_samples = min(int(meta.get("n_samples", states.shape[0])), int(states.shape[0]))
    output = data_dir / args.output_name
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoints = list(args.checkpoints or [])
    if args.checkpoint:
        checkpoints.insert(0, args.checkpoint)
    if not checkpoints:
        raise ValueError("Provide --checkpoint or --checkpoints")
    weights = parse_weights(args.weights, len(checkpoints))
    scores = np.zeros(n_samples, dtype=np.float32)
    for checkpoint, weight in zip(checkpoints, weights):
        scores += float(weight) * predict_scores(
            checkpoint=checkpoint,
            states=states,
            turns=turns,
            n_samples=n_samples,
            device=device,
            batch_size=args.batch_size,
        )
    np.save(output, scores)
    summary = {
        "data": str(data_dir),
        "checkpoint": str(checkpoints[0]) if len(checkpoints) == 1 else "ensemble",
        "checkpoints": [str(path) for path in checkpoints],
        "weights": weights,
        "output": str(output),
        "n_samples": int(n_samples),
        "score_mean": float(scores.mean()) if n_samples else 0.0,
        "score_std": float(scores.std()) if n_samples else 0.0,
    }
    with (data_dir / f"{Path(args.output_name).stem}.metadata.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write base_scores.npy from an action-value reranker checkpoint")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoints", nargs="*")
    parser.add_argument("--weights", default="")
    parser.add_argument("--output-name", default="base_scores.npy")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    write_scores(args)


if __name__ == "__main__":
    main()
