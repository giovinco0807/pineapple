"""Streaming PyTorch MLP trainer for large regular OFC teacher JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from .train_torch_action_value import parse_hidden_layers, select_device
from .turn3_model import (
    FEATURE_DIM,
    TorchActionValueModel,
    _build_torch_mlp,
    evaluate_model,
    sample_to_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--holdout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--learning-rate", type=float, default=0.0007)
    parser.add_argument("--weight-decay", type=float, default=0.00005)
    parser.add_argument("--hidden-layer-sizes", default="2048,1024,512,256")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--max-metric-samples", type=int, default=5000)
    return parser.parse_args()


def count_samples(path: Path, max_samples: int | None) -> int:
    count = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            count += 1
            if max_samples is not None and count >= max_samples:
                break
    return count


def split_indices(total_samples: int, *, holdout_fraction: float, seed: int) -> set[int]:
    if not 0.0 <= holdout_fraction < 1.0:
        raise ValueError("holdout must be in [0, 1)")
    indices = np.arange(total_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    holdout_count = int(round(total_samples * holdout_fraction))
    return set(int(index) for index in indices[:holdout_count])


def iter_split_samples(
    path: Path,
    *,
    max_samples: int | None,
    holdout_indices: set[int],
    want_holdout: bool,
) -> Iterable[dict]:
    seen = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            if max_samples is not None and seen >= max_samples:
                break
            is_holdout = seen in holdout_indices
            if is_holdout == want_holdout:
                yield json.loads(line)
            seen += 1


def streaming_stats(
    path: Path,
    *,
    max_samples: int | None,
    holdout_indices: set[int],
) -> dict:
    feature_sum = np.zeros(FEATURE_DIM, dtype=np.float64)
    feature_sumsq = np.zeros(FEATURE_DIM, dtype=np.float64)
    target_sum = 0.0
    target_sumsq = 0.0
    train_actions = 0
    holdout_actions = 0
    train_samples = 0
    holdout_samples = 0

    for sample in iter_split_samples(
        path,
        max_samples=max_samples,
        holdout_indices=holdout_indices,
        want_holdout=False,
    ):
        features, targets = sample_to_matrix(sample)
        feature_sum += features.sum(axis=0, dtype=np.float64)
        feature_sumsq += np.einsum("ij,ij->j", features, features, dtype=np.float64)
        target_sum += float(targets.sum(dtype=np.float64))
        target_sumsq += float(np.dot(targets, targets))
        train_actions += int(features.shape[0])
        train_samples += 1

    for sample in iter_split_samples(
        path,
        max_samples=max_samples,
        holdout_indices=holdout_indices,
        want_holdout=True,
    ):
        actions = sample.get("actions", ())
        holdout_actions += len(actions)
        holdout_samples += 1

    feature_mean64 = feature_sum / float(train_actions)
    feature_var64 = feature_sumsq / float(train_actions) - feature_mean64 * feature_mean64
    np.maximum(feature_var64, 0.0, out=feature_var64)
    feature_mean = feature_mean64.astype(np.float32)
    feature_scale = np.sqrt(feature_var64).astype(np.float32)
    feature_scale[feature_scale < 1e-6] = 1.0
    target_mean = target_sum / float(train_actions)
    target_var = target_sumsq / float(train_actions) - target_mean * target_mean
    target_scale = float(np.sqrt(max(target_var, 0.0)))
    if target_scale < 1e-6:
        target_scale = 1.0

    return {
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "target_mean": float(target_mean),
        "target_scale": target_scale,
        "train_actions": train_actions,
        "holdout_actions": holdout_actions,
        "train_samples": train_samples,
        "holdout_samples": holdout_samples,
    }


def train_one_epoch(
    torch,
    net,
    optimizer,
    loss_fn,
    *,
    path: Path,
    max_samples: int | None,
    holdout_indices: set[int],
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    target_mean: float,
    target_scale: float,
    device: str,
    batch_size: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    mean_tensor = torch.from_numpy(feature_mean).to(device)
    scale_tensor = torch.from_numpy(feature_scale).to(device)
    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    buffered = 0
    loss_sum = 0.0
    seen = 0

    def flush() -> None:
        nonlocal feature_parts, target_parts, buffered, loss_sum, seen
        if buffered == 0:
            return
        features = np.vstack(feature_parts).astype(np.float32, copy=False)
        targets = np.concatenate(target_parts).astype(np.float32, copy=False)
        order = rng.permutation(features.shape[0])
        for start in range(0, features.shape[0], batch_size):
            batch_idx = order[start : start + batch_size]
            x = torch.from_numpy(features[batch_idx]).to(device)
            y = torch.from_numpy(targets[batch_idx]).to(device)
            x = (x - mean_tensor) / scale_tensor
            optimizer.zero_grad(set_to_none=True)
            pred = net(x).squeeze(-1)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            batch_count = int(batch_idx.shape[0])
            loss_sum += float(loss.item()) * batch_count
            seen += batch_count
        feature_parts = []
        target_parts = []
        buffered = 0

    for sample in iter_split_samples(
        path,
        max_samples=max_samples,
        holdout_indices=holdout_indices,
        want_holdout=False,
    ):
        features, targets = sample_to_matrix(sample)
        target_norm = ((targets - target_mean) / target_scale).astype(np.float32)
        feature_parts.append(features)
        target_parts.append(target_norm)
        buffered += int(features.shape[0])
        if buffered >= batch_size:
            flush()
    flush()
    return loss_sum / max(seen, 1)


def streaming_mse(
    torch,
    net,
    *,
    path: Path,
    max_samples: int | None,
    holdout_indices: set[int],
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    target_mean: float,
    target_scale: float,
    device: str,
    batch_size: int,
) -> float:
    if not holdout_indices:
        return 0.0
    mean_tensor = torch.from_numpy(feature_mean).to(device)
    scale_tensor = torch.from_numpy(feature_scale).to(device)
    squared_error_sum = 0.0
    seen = 0
    with torch.no_grad():
        for sample in iter_split_samples(
            path,
            max_samples=max_samples,
            holdout_indices=holdout_indices,
            want_holdout=True,
        ):
            features, targets = sample_to_matrix(sample)
            for start in range(0, features.shape[0], batch_size):
                end = min(start + batch_size, features.shape[0])
                x = torch.from_numpy(features[start:end]).to(device)
                x = (x - mean_tensor) / scale_tensor
                pred = net(x).squeeze(-1).detach().cpu().numpy().astype(np.float64)
                pred = pred * target_scale + target_mean
                err = pred - targets[start:end]
                squared_error_sum += float(np.dot(err, err))
                seen += int(end - start)
    return squared_error_sum / max(seen, 1)


def collect_metric_samples(
    path: Path,
    *,
    max_samples: int | None,
    holdout_indices: set[int],
    want_holdout: bool,
    limit: int,
) -> list[dict]:
    if limit <= 0:
        limit = 10**18
    samples: list[dict] = []
    for sample in iter_split_samples(
        path,
        max_samples=max_samples,
        holdout_indices=holdout_indices,
        want_holdout=want_holdout,
    ):
        samples.append(sample)
        if len(samples) >= limit:
            break
    return samples


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
    total_samples = count_samples(args.input, args.max_samples)
    if total_samples <= 0:
        raise SystemExit("no teacher samples")
    holdout_indices = split_indices(total_samples, holdout_fraction=args.holdout, seed=args.seed)

    started_at = time.time()
    stats = streaming_stats(args.input, max_samples=args.max_samples, holdout_indices=holdout_indices)
    feature_mean = stats["feature_mean"]
    feature_scale = stats["feature_scale"]
    target_mean = stats["target_mean"]
    target_scale = stats["target_scale"]

    net = _build_torch_mlp(torch, FEATURE_DIM, hidden_layers, args.dropout).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()

    best_state = None
    best_holdout_mse = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        net.train()
        train_loss = train_one_epoch(
            torch,
            net,
            optimizer,
            loss_fn,
            path=args.input,
            max_samples=args.max_samples,
            holdout_indices=holdout_indices,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            target_mean=target_mean,
            target_scale=target_scale,
            device=device,
            batch_size=args.batch_size,
            seed=args.seed + epoch,
        )
        net.eval()
        holdout_mse = streaming_mse(
            torch,
            net,
            path=args.input,
            max_samples=args.max_samples,
            holdout_indices=holdout_indices,
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
        if holdout_indices and holdout_mse < best_holdout_mse:
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
        if holdout_indices and args.patience > 0 and stale_epochs >= args.patience:
            break

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
        best_epoch = history[-1]["epoch"]
        best_holdout_mse = history[-1]["holdout_mse"]
    net.load_state_dict(best_state)

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
    train_metric_samples = collect_metric_samples(
        args.input,
        max_samples=args.max_samples,
        holdout_indices=holdout_indices,
        want_holdout=False,
        limit=metric_limit,
    )
    holdout_metric_samples = collect_metric_samples(
        args.input,
        max_samples=args.max_samples,
        holdout_indices=holdout_indices,
        want_holdout=True,
        limit=metric_limit,
    )
    metrics = {
        "input": str(args.input),
        "model_output": str(args.model_output),
        "model_type": "torch_mlp_streaming",
        "device": device,
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "total_samples": total_samples,
        "train_samples": stats["train_samples"],
        "holdout_samples": stats["holdout_samples"],
        "train_actions": stats["train_actions"],
        "holdout_actions": stats["holdout_actions"],
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
