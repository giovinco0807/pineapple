"""Streaming PyTorch MLP trainer for HU-aware Turn3 teacher JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from .hu_turn3_model import (
    HU_FEATURE_DIM,
    HuTorchActionValueModel,
    evaluate_model,
    sample_to_matrix,
)
from .train_torch_action_value import parse_hidden_layers, select_device
from .train_hu_turn3 import parse_source_weights
from .turn3_model import _build_torch_mlp


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
    parser.add_argument(
        "--best-action-weight",
        type=float,
        default=2.0,
        help="MSE multiplier for teacher-best action rows. Use 1.0 for unweighted MSE.",
    )
    parser.add_argument(
        "--best-action-tolerance",
        type=float,
        default=1e-9,
        help="Treat actions within this score distance from best as best-action rows.",
    )
    parser.add_argument(
        "--ranking-loss-weight",
        type=float,
        default=0.0,
        help="Optional pairwise ranking loss weight for best-vs-other action gaps.",
    )
    parser.add_argument(
        "--ranking-max-margin",
        type=float,
        default=2.0,
        help="Clamp normalized best-vs-other target gaps used as ranking margins.",
    )
    parser.add_argument(
        "--ranking-gap-tolerance",
        type=float,
        default=1e-6,
        help="Ignore target gaps at or below this normalized value in ranking loss.",
    )
    parser.add_argument(
        "--source-weight",
        action="append",
        default=[],
        metavar="SOURCE=WEIGHT",
        help=(
            "Per-sample source weight applied to all action rows and ranking groups. "
            "May be repeated or comma-separated; unspecified sources use 1.0."
        ),
    )
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
    source_weights: dict[str, float],
) -> dict:
    feature_sum = np.zeros(HU_FEATURE_DIM, dtype=np.float64)
    feature_sumsq = np.zeros(HU_FEATURE_DIM, dtype=np.float64)
    target_sum = 0.0
    target_sumsq = 0.0
    train_actions = 0
    holdout_actions = 0
    train_samples = 0
    holdout_samples = 0
    source_counter: Counter[str] = Counter()
    train_source_counter: Counter[str] = Counter()
    holdout_source_counter: Counter[str] = Counter()
    train_action_weight_sum = 0.0
    train_action_weight_min = float("inf")
    train_action_weight_max = 0.0

    for sample in iter_split_samples(
        path,
        max_samples=max_samples,
        holdout_indices=holdout_indices,
        want_holdout=False,
    ):
        features, targets = sample_to_matrix(sample)
        source = sample_source(sample)
        source_counter[source] += 1
        train_source_counter[source] += 1
        sample_weight = sample_source_weight(sample, source_weights)
        action_weight = sample_weight * float(features.shape[0])
        train_action_weight_sum += action_weight
        train_action_weight_min = min(train_action_weight_min, sample_weight)
        train_action_weight_max = max(train_action_weight_max, sample_weight)
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
        source = sample_source(sample)
        source_counter[source] += 1
        holdout_source_counter[source] += 1
        holdout_actions += len(sample.get("actions", ()))
        holdout_samples += 1

    if train_actions <= 0:
        raise ValueError("no training actions after split")

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
        "source_counts": dict(source_counter),
        "train_source_counts": dict(train_source_counter),
        "holdout_source_counts": dict(holdout_source_counter),
        "train_action_weight_sum": float(train_action_weight_sum),
        "train_action_weight_mean": float(train_action_weight_sum / max(train_actions, 1)),
        "train_action_weight_min": 0.0 if train_action_weight_min == float("inf") else float(train_action_weight_min),
        "train_action_weight_max": float(train_action_weight_max),
    }


def action_weights(
    targets: np.ndarray,
    *,
    best_action_weight: float,
    best_action_tolerance: float,
    source_weight: float = 1.0,
) -> np.ndarray:
    weights = np.full(targets.shape[0], float(source_weight), dtype=np.float32)
    if best_action_weight <= 1.0 or targets.shape[0] == 0:
        return weights
    best = float(np.max(targets))
    weights[targets >= best - best_action_tolerance] *= float(best_action_weight)
    return weights


def sample_source(sample: dict) -> str:
    return str(sample.get("source", "unknown"))


def sample_source_weight(sample: dict, source_weights: dict[str, float]) -> float:
    return float(source_weights.get(sample_source(sample), 1.0))


def ranking_loss_for_grouped_predictions(
    torch,
    predictions,
    targets,
    group_offsets: list[int],
    *,
    max_margin: float,
    gap_tolerance: float,
    group_weights: list[float] | None = None,
):
    losses = []
    weights = []
    for group_index, (start, end) in enumerate(zip(group_offsets, group_offsets[1:])):
        if end - start <= 1:
            continue
        group_pred = predictions[start:end]
        group_targets = targets[start:end]
        best_index = int(torch.argmax(group_targets).item())
        best_pred = group_pred[best_index]
        best_target = group_targets[best_index]
        target_gap = (best_target - group_targets).clamp(min=0.0, max=max_margin)
        mask = target_gap > gap_tolerance
        if bool(mask.any()):
            pred_gap = best_pred - group_pred
            losses.append(torch.relu(target_gap[mask] - pred_gap[mask]).square().mean())
            if group_weights is not None:
                weights.append(float(group_weights[group_index]))
    if not losses:
        return predictions.sum() * 0.0
    if group_weights is not None:
        weight_tensor = torch.tensor(weights, dtype=predictions.dtype, device=predictions.device)
        if bool((weight_tensor > 0.0).any()):
            return (torch.stack(losses) * weight_tensor).sum() / weight_tensor.sum().clamp_min(1.0)
        return predictions.sum() * 0.0
    return torch.stack(losses).mean()


def train_one_epoch(
    torch,
    net,
    optimizer,
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
    best_action_weight: float,
    best_action_tolerance: float,
    ranking_loss_weight: float,
    ranking_max_margin: float,
    ranking_gap_tolerance: float,
    source_weights: dict[str, float],
) -> float:
    rng = np.random.default_rng(seed)
    mean_tensor = torch.from_numpy(feature_mean).to(device)
    scale_tensor = torch.from_numpy(feature_scale).to(device)
    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    group_lengths: list[int] = []
    group_source_weights: list[float] = []
    buffered = 0
    loss_sum = 0.0
    weight_sum = 0.0
    loss_fn = torch.nn.MSELoss(reduction="none")

    def flush() -> None:
        nonlocal feature_parts, target_parts, weight_parts, group_lengths, group_source_weights, buffered, loss_sum, weight_sum
        if buffered == 0:
            return
        features = np.vstack(feature_parts).astype(np.float32, copy=False)
        targets = np.concatenate(target_parts).astype(np.float32, copy=False)
        weights = np.concatenate(weight_parts).astype(np.float32, copy=False)
        order = rng.permutation(features.shape[0])
        for start in range(0, features.shape[0], batch_size):
            batch_idx = order[start : start + batch_size]
            x = torch.from_numpy(features[batch_idx]).to(device)
            y = torch.from_numpy(targets[batch_idx]).to(device)
            w = torch.from_numpy(weights[batch_idx]).to(device)
            x = (x - mean_tensor) / scale_tensor
            optimizer.zero_grad(set_to_none=True)
            pred = net(x).squeeze(-1)
            per_row = loss_fn(pred, y)
            loss = (per_row * w).sum() / w.sum().clamp_min(1.0)
            loss.backward()
            optimizer.step()
            batch_weight = float(weights[batch_idx].sum(dtype=np.float64))
            loss_sum += float(loss.item()) * batch_weight
            weight_sum += batch_weight
        if ranking_loss_weight > 0.0:
            start_group = 0
            start_action = 0
            while start_group < len(group_lengths):
                end_group = start_group
                end_action = start_action
                while end_group < len(group_lengths):
                    next_end = end_action + group_lengths[end_group]
                    if end_group > start_group and next_end - start_action > batch_size:
                        break
                    end_action = next_end
                    end_group += 1
                    if end_action - start_action >= batch_size:
                        break

                x = torch.from_numpy(features[start_action:end_action]).to(device)
                y = torch.from_numpy(targets[start_action:end_action]).to(device)
                x = (x - mean_tensor) / scale_tensor
                offsets = [0]
                cursor = 0
                for length in group_lengths[start_group:end_group]:
                    cursor += length
                    offsets.append(cursor)
                ranking_group_weights = group_source_weights[start_group:end_group]

                optimizer.zero_grad(set_to_none=True)
                pred = net(x).squeeze(-1)
                rank_loss = ranking_loss_for_grouped_predictions(
                    torch,
                    pred,
                    y,
                    offsets,
                    max_margin=ranking_max_margin,
                    gap_tolerance=ranking_gap_tolerance,
                    group_weights=ranking_group_weights,
                )
                loss = rank_loss * ranking_loss_weight
                loss.backward()
                optimizer.step()
                ranking_weight = float(
                    sum(
                        weight * length
                        for weight, length in zip(
                            ranking_group_weights,
                            group_lengths[start_group:end_group],
                        )
                    )
                )
                loss_sum += float(loss.item()) * ranking_weight
                weight_sum += ranking_weight
                start_group = end_group
                start_action = end_action
        feature_parts = []
        target_parts = []
        weight_parts = []
        group_lengths = []
        group_source_weights = []
        buffered = 0

    for sample in iter_split_samples(
        path,
        max_samples=max_samples,
        holdout_indices=holdout_indices,
        want_holdout=False,
    ):
        features, targets = sample_to_matrix(sample)
        target_norm = ((targets - target_mean) / target_scale).astype(np.float32)
        weights = action_weights(
            targets,
            best_action_weight=best_action_weight,
            best_action_tolerance=best_action_tolerance,
            source_weight=sample_source_weight(sample, source_weights),
        )
        feature_parts.append(features)
        target_parts.append(target_norm)
        weight_parts.append(weights)
        group_lengths.append(int(features.shape[0]))
        group_source_weights.append(sample_source_weight(sample, source_weights))
        buffered += int(features.shape[0])
        if buffered >= batch_size:
            flush()
    flush()
    return loss_sum / max(weight_sum, 1.0)


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
    try:
        source_weights = parse_source_weights(args.source_weight)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.epochs <= 0:
        raise SystemExit("epochs must be positive")
    if args.batch_size <= 0:
        raise SystemExit("batch-size must be positive")
    if args.best_action_weight < 1.0:
        raise SystemExit("best-action-weight must be >= 1.0")
    if args.ranking_loss_weight < 0.0:
        raise SystemExit("ranking-loss-weight must be non-negative")
    if args.ranking_max_margin <= 0.0:
        raise SystemExit("ranking-max-margin must be positive")
    if args.ranking_gap_tolerance < 0.0:
        raise SystemExit("ranking-gap-tolerance must be non-negative")

    import torch

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = select_device(torch, args.device)
    hidden_layers = parse_hidden_layers(args.hidden_layer_sizes)
    total_samples = count_samples(args.input, args.max_samples)
    if total_samples <= 0:
        raise SystemExit("no HU teacher samples")
    holdout_indices = split_indices(total_samples, holdout_fraction=args.holdout, seed=args.seed)

    started_at = time.time()
    stats = streaming_stats(
        args.input,
        max_samples=args.max_samples,
        holdout_indices=holdout_indices,
        source_weights=source_weights,
    )
    feature_mean = stats["feature_mean"]
    feature_scale = stats["feature_scale"]
    target_mean = stats["target_mean"]
    target_scale = stats["target_scale"]

    net = _build_torch_mlp(torch, HU_FEATURE_DIM, hidden_layers, args.dropout).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

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
            best_action_weight=args.best_action_weight,
            best_action_tolerance=args.best_action_tolerance,
            ranking_loss_weight=args.ranking_loss_weight,
            ranking_max_margin=args.ranking_max_margin,
            ranking_gap_tolerance=args.ranking_gap_tolerance,
            source_weights=source_weights,
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

    model = HuTorchActionValueModel(
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
        "model_type": "hu_torch_mlp_streaming",
        "device": device,
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "total_samples": total_samples,
        "train_samples": stats["train_samples"],
        "holdout_samples": stats["holdout_samples"],
        "train_actions": stats["train_actions"],
        "holdout_actions": stats["holdout_actions"],
        "source_weights": source_weights,
        "source_counts": stats["source_counts"],
        "train_source_counts": stats["train_source_counts"],
        "holdout_source_counts": stats["holdout_source_counts"],
        "train_action_weight_sum": stats["train_action_weight_sum"],
        "train_action_weight_mean": stats["train_action_weight_mean"],
        "train_action_weight_min": stats["train_action_weight_min"],
        "train_action_weight_max": stats["train_action_weight_max"],
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
        "best_action_weight": args.best_action_weight,
        "best_action_tolerance": args.best_action_tolerance,
        "ranking_loss_weight": args.ranking_loss_weight,
        "ranking_max_margin": args.ranking_max_margin,
        "ranking_gap_tolerance": args.ranking_gap_tolerance,
    }
    print(json.dumps(metrics, indent=2))
    if args.metrics_output:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
