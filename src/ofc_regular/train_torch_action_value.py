"""Train a PyTorch MLP action-value model for regular OFC."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .turn3_model import (
    FEATURE_DIM,
    TorchActionValueModel,
    _build_torch_mlp,
    evaluate_model,
    read_teacher_samples,
    samples_to_matrix,
    split_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="teacher JSONL")
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--hidden-layer-sizes", default="512,256,128")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--log-every", type=int, default=1, help="Emit epoch progress to stderr every N epochs.")
    parser.add_argument(
        "--max-metric-samples",
        type=int,
        default=5000,
        help="Limit sample-level train/holdout metrics after fitting; use 0 for all.",
    )
    return parser.parse_args()


def parse_hidden_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not layers:
        raise ValueError("at least one hidden layer is required")
    return layers


def select_device(torch, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    return requested


def normalize_stats(features: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    feature_sum = np.zeros(features.shape[1], dtype=np.float64)
    feature_sumsq = np.zeros(features.shape[1], dtype=np.float64)
    chunk_size = 65536
    for start in range(0, features.shape[0], chunk_size):
        block = features[start : start + chunk_size]
        feature_sum += block.sum(axis=0, dtype=np.float64)
        feature_sumsq += np.einsum("ij,ij->j", block, block, dtype=np.float64)
    feature_mean64 = feature_sum / float(features.shape[0])
    feature_var64 = feature_sumsq / float(features.shape[0]) - feature_mean64 * feature_mean64
    np.maximum(feature_var64, 0.0, out=feature_var64)
    feature_mean = feature_mean64.astype(np.float32)
    feature_scale = np.sqrt(feature_var64).astype(np.float32)
    feature_scale[feature_scale < 1e-6] = 1.0
    target_mean = float(targets.mean(dtype=np.float64))
    target_scale = float(targets.std(dtype=np.float64))
    if target_scale < 1e-6:
        target_scale = 1.0
    return feature_mean, feature_scale, target_mean, target_scale


def mse_on_matrix(
    torch,
    net,
    *,
    features: np.ndarray,
    targets: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    target_mean: float,
    target_scale: float,
    device: str,
    batch_size: int,
) -> float:
    if features.shape[0] == 0:
        return 0.0
    mean_tensor = torch.from_numpy(feature_mean).to(device)
    scale_tensor = torch.from_numpy(feature_scale).to(device)
    squared_error_sum = 0.0
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            end = min(start + batch_size, features.shape[0])
            x = torch.from_numpy(features[start:end]).to(device)
            x = (x - mean_tensor) / scale_tensor
            pred = net(x).squeeze(-1).detach().cpu().numpy().astype(np.float64)
            pred = pred * target_scale + target_mean
            err = pred - targets[start:end]
            squared_error_sum += float(np.dot(err, err))
    return squared_error_sum / float(features.shape[0])


def limited_samples(samples: Sequence[dict], limit: int) -> Sequence[dict]:
    if limit <= 0 or len(samples) <= limit:
        return samples
    return samples[:limit]


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise SystemExit("epochs must be positive")
    if args.batch_size <= 0:
        raise SystemExit("batch-size must be positive")

    import torch

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = select_device(torch, args.device)
    hidden_layers = parse_hidden_layers(args.hidden_layer_sizes)
    samples = read_teacher_samples(args.input, max_samples=args.max_samples)
    if not samples:
        raise SystemExit("no teacher samples")
    train_samples, holdout_samples = split_samples(
        samples,
        holdout_fraction=args.holdout,
        seed=args.seed,
    )
    if not train_samples:
        raise SystemExit("no training samples after split")

    started_at = time.time()
    train_features, train_targets = samples_to_matrix(train_samples)
    holdout_features = np.zeros((0, FEATURE_DIM), dtype=np.float32)
    holdout_targets = np.zeros(0, dtype=np.float64)
    if holdout_samples:
        holdout_features, holdout_targets = samples_to_matrix(holdout_samples)

    feature_mean, feature_scale, target_mean, target_scale = normalize_stats(
        train_features,
        train_targets,
    )
    train_targets_normalized = ((train_targets - target_mean) / target_scale).astype(np.float32)

    net = _build_torch_mlp(torch, FEATURE_DIM, hidden_layers, args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = torch.nn.MSELoss()
    mean_tensor = torch.from_numpy(feature_mean).to(device)
    scale_tensor = torch.from_numpy(feature_scale).to(device)
    train_features_tensor = torch.from_numpy(train_features)
    train_targets_tensor = torch.from_numpy(train_targets_normalized)

    best_state = None
    best_holdout_mse = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        net.train()
        permutation = torch.randperm(train_features.shape[0])
        loss_sum = 0.0
        seen = 0
        for start in range(0, train_features.shape[0], args.batch_size):
            batch_idx = permutation[start : start + args.batch_size]
            x = train_features_tensor[batch_idx].to(device)
            y = train_targets_tensor[batch_idx].to(device)
            x = (x - mean_tensor) / scale_tensor

            optimizer.zero_grad(set_to_none=True)
            pred = net(x).squeeze(-1)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            batch_size = int(batch_idx.numel())
            loss_sum += float(loss.item()) * batch_size
            seen += batch_size

        net.eval()
        train_loss = loss_sum / max(seen, 1)
        holdout_mse = mse_on_matrix(
            torch,
            net,
            features=holdout_features,
            targets=holdout_targets,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            target_mean=target_mean,
            target_scale=target_scale,
            device=device,
            batch_size=args.batch_size,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss_normalized": train_loss,
                "holdout_mse": holdout_mse,
            }
        )

        if holdout_samples and holdout_mse < best_holdout_mse:
            best_holdout_mse = holdout_mse
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if args.log_every > 0 and (epoch == 1 or epoch % args.log_every == 0):
            print(
                json.dumps(
                    {
                        "event": "epoch",
                        "epoch": epoch,
                        "train_loss_normalized": train_loss,
                        "holdout_mse": holdout_mse,
                        "best_epoch": best_epoch,
                        "best_holdout_mse": best_holdout_mse,
                        "elapsed_seconds": time.time() - started_at,
                    },
                    separators=(",", ":"),
                ),
                file=sys.stderr,
                flush=True,
            )
        if holdout_samples and args.patience > 0 and stale_epochs >= args.patience:
            break

    if best_state is not None:
        net.load_state_dict(best_state)
    else:
        best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
        best_epoch = history[-1]["epoch"]
        best_holdout_mse = history[-1]["holdout_mse"]

    model = TorchActionValueModel(
        state_dict=best_state,
        hidden_layer_sizes=hidden_layers,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        dropout=args.dropout,
        batch_size=args.batch_size,
    )
    model.save(args.model_output)

    metric_limit = int(args.max_metric_samples)
    train_metric_samples = limited_samples(train_samples, metric_limit)
    holdout_metric_samples = limited_samples(holdout_samples, metric_limit)
    metrics = {
        "input": str(args.input),
        "model_output": str(args.model_output),
        "model_type": "torch_mlp",
        "device": device,
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "total_samples": len(samples),
        "train_samples": len(train_samples),
        "holdout_samples": len(holdout_samples),
        "train_actions": int(train_features.shape[0]),
        "holdout_actions": int(holdout_features.shape[0]),
        "best_epoch": best_epoch,
        "best_holdout_mse": best_holdout_mse,
        "elapsed_seconds": time.time() - started_at,
        "epochs_ran": len(history),
        "history": history,
        "train": evaluate_model(model, train_metric_samples),
        "holdout": evaluate_model(model, holdout_metric_samples),
        "metric_sample_limit": metric_limit,
        "seed": args.seed,
        "hidden_layer_sizes": list(hidden_layers),
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    print(json.dumps(metrics, indent=2))

    if args.metrics_output:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
