"""Train a HU-aware Turn3 PyTorch MLP from a feature memmap cache."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .hu_turn3_model import HU_FEATURE_DIM, HuTorchActionValueModel
from .train_torch_action_value import parse_hidden_layers, select_device
from .train_torch_hu_turn3_streaming import ranking_loss_for_grouped_predictions
from .turn3_model import _build_torch_mlp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--cache-weight", type=float, default=1.0)
    parser.add_argument("--extra-cache-dir", type=Path, action="append", default=[])
    parser.add_argument("--extra-cache-weight", type=float, action="append", default=[])
    parser.add_argument("--init-model", type=Path)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--holdout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--learning-rate", type=float, default=0.0007)
    parser.add_argument("--weight-decay", type=float, default=0.00005)
    parser.add_argument("--hidden-layer-sizes", default="1024,512,256")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--best-action-weight", type=float, default=2.0)
    parser.add_argument("--best-action-tolerance", type=float, default=1e-9)
    parser.add_argument("--ranking-loss-weight", type=float, default=1.0)
    parser.add_argument("--ranking-max-margin", type=float, default=2.0)
    parser.add_argument("--ranking-gap-tolerance", type=float, default=1e-6)
    parser.add_argument("--max-metric-samples", type=int, default=20_000)
    parser.add_argument(
        "--max-ram-gb",
        type=float,
        default=48.0,
        help="Refuse to load cached feature tensors if estimated RAM exceeds this budget.",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class LoadedCache:
    cache_dir: Path
    weight: float
    metadata: dict
    features: np.ndarray
    targets: np.ndarray
    offsets: np.ndarray


def load_cache(cache_dir: Path):
    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    feature_dim = int(metadata["feature_dim"])
    if feature_dim != HU_FEATURE_DIM:
        raise SystemExit(f"cache feature_dim={feature_dim}, local HU_FEATURE_DIM={HU_FEATURE_DIM}")
    action_count = int(metadata["actions"])
    features = np.memmap(
        metadata["features_path"],
        dtype=np.dtype(metadata["feature_dtype"]),
        mode="r",
        shape=(action_count, feature_dim),
    )
    targets = np.memmap(
        metadata["targets_path"],
        dtype=np.float32,
        mode="r",
        shape=(action_count,),
    )
    offsets = np.load(cache_dir / "sample_offsets.npy")
    return metadata, features, targets, offsets


def load_caches(cache_dirs: list[Path], weights: list[float]) -> list[LoadedCache]:
    caches: list[LoadedCache] = []
    for cache_dir, weight in zip(cache_dirs, weights):
        metadata, features, targets, offsets = load_cache(cache_dir.resolve())
        caches.append(
            LoadedCache(
                cache_dir=cache_dir.resolve(),
                weight=float(weight),
                metadata=metadata,
                features=features,
                targets=targets,
                offsets=offsets,
            )
        )
    return caches


def split_sample_indices(total_samples: int, *, holdout_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= holdout_fraction < 1.0:
        raise ValueError("holdout must be in [0, 1)")
    indices = np.arange(total_samples, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    holdout_count = int(round(total_samples * holdout_fraction))
    holdout = np.sort(indices[:holdout_count])
    train = np.sort(indices[holdout_count:])
    return train, holdout


def materialize_split(
    features,
    targets,
    offsets: np.ndarray,
    sample_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    action_total = int(sum(int(offsets[index + 1] - offsets[index]) for index in sample_indices))
    split_features = np.empty((action_total, HU_FEATURE_DIM), dtype=np.float16)
    split_targets = np.empty(action_total, dtype=np.float32)
    group_offsets = np.zeros(sample_indices.shape[0] + 1, dtype=np.int64)
    cursor = 0
    for group_index, sample_index in enumerate(sample_indices):
        start = int(offsets[sample_index])
        end = int(offsets[sample_index + 1])
        length = end - start
        split_features[cursor : cursor + length] = features[start:end]
        split_targets[cursor : cursor + length] = targets[start:end]
        cursor += length
        group_offsets[group_index + 1] = cursor
    return split_features, split_targets, group_offsets


def count_actions_for_indices(offsets: np.ndarray, sample_indices: np.ndarray) -> int:
    return int(sum(int(offsets[index + 1] - offsets[index]) for index in sample_indices))


def materialize_multi_split(
    caches: list[LoadedCache],
    sample_indices_by_cache: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample_total = int(sum(indices.shape[0] for indices in sample_indices_by_cache))
    action_total = int(
        sum(
            count_actions_for_indices(cache.offsets, indices)
            for cache, indices in zip(caches, sample_indices_by_cache)
        )
    )
    split_features = np.empty((action_total, HU_FEATURE_DIM), dtype=np.float16)
    split_targets = np.empty(action_total, dtype=np.float32)
    group_offsets = np.zeros(sample_total + 1, dtype=np.int64)
    group_cache_indices = np.zeros(sample_total, dtype=np.int16)
    action_cursor = 0
    group_cursor = 0
    for cache_index, (cache, sample_indices) in enumerate(zip(caches, sample_indices_by_cache)):
        for sample_index in sample_indices:
            start = int(cache.offsets[sample_index])
            end = int(cache.offsets[sample_index + 1])
            length = end - start
            split_features[action_cursor : action_cursor + length] = cache.features[start:end]
            split_targets[action_cursor : action_cursor + length] = cache.targets[start:end]
            action_cursor += length
            group_cache_indices[group_cursor] = cache_index
            group_cursor += 1
            group_offsets[group_cursor] = action_cursor
    return split_features, split_targets, group_offsets, group_cache_indices


def normalize_stats(features: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    feature_sum = np.zeros(HU_FEATURE_DIM, dtype=np.float64)
    feature_sumsq = np.zeros(HU_FEATURE_DIM, dtype=np.float64)
    chunk_size = 65536
    for start in range(0, features.shape[0], chunk_size):
        block = features[start : start + chunk_size].astype(np.float32, copy=False)
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


def action_weights(
    targets: np.ndarray,
    group_offsets: np.ndarray,
    *,
    best_action_weight: float,
    best_action_tolerance: float,
) -> np.ndarray:
    weights = np.ones(targets.shape[0], dtype=np.float32)
    if best_action_weight <= 1.0:
        return weights
    for start, end in zip(group_offsets[:-1], group_offsets[1:]):
        group = targets[start:end]
        if group.shape[0] == 0:
            continue
        best = float(group.max())
        weights[start:end][group >= best - best_action_tolerance] *= float(best_action_weight)
    return weights


def apply_group_source_weights(
    weights: np.ndarray,
    group_offsets: np.ndarray,
    group_cache_indices: np.ndarray,
    cache_weights: list[float],
) -> np.ndarray:
    if all(abs(weight - 1.0) < 1e-12 for weight in cache_weights):
        return weights
    for group_index, cache_index in enumerate(group_cache_indices):
        source_weight = float(cache_weights[int(cache_index)])
        if abs(source_weight - 1.0) < 1e-12:
            continue
        start = int(group_offsets[group_index])
        end = int(group_offsets[group_index + 1])
        weights[start:end] *= source_weight
    return weights


def adapted_initial_state_dict(
    torch,
    init_model: HuTorchActionValueModel,
    *,
    hidden_layers: tuple[int, ...],
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    target_mean: float,
    target_scale: float,
) -> dict:
    if tuple(init_model.hidden_layer_sizes) != tuple(hidden_layers):
        raise SystemExit(
            "init model hidden layers do not match: "
            f"{init_model.hidden_layer_sizes} != {hidden_layers}"
        )
    if int(init_model.feature_mean.shape[0]) != HU_FEATURE_DIM:
        raise SystemExit(
            f"init model feature_dim={init_model.feature_mean.shape[0]}, local HU_FEATURE_DIM={HU_FEATURE_DIM}"
        )
    if target_scale <= 0.0:
        raise SystemExit("target_scale must be positive")

    state = {key: value.detach().cpu().clone().float() for key, value in init_model.state_dict.items()}
    weight_keys = [key for key in state if key.endswith(".weight")]
    if not weight_keys:
        raise SystemExit("init model state_dict has no linear weights")

    first_weight_key = weight_keys[0]
    first_bias_key = first_weight_key.removesuffix(".weight") + ".bias"
    last_weight_key = weight_keys[-1]
    last_bias_key = last_weight_key.removesuffix(".weight") + ".bias"
    for key in (first_bias_key, last_bias_key):
        if key not in state:
            raise SystemExit(f"init model state_dict missing {key}")

    old_feature_mean = torch.as_tensor(init_model.feature_mean.astype(np.float32))
    old_feature_scale = torch.as_tensor(init_model.feature_scale.astype(np.float32))
    new_feature_mean = torch.as_tensor(feature_mean.astype(np.float32))
    new_feature_scale = torch.as_tensor(feature_scale.astype(np.float32))
    if bool(torch.any(old_feature_scale <= 0)):
        raise SystemExit("init model feature_scale contains non-positive values")

    first_weight = state[first_weight_key]
    if first_weight.shape[1] != HU_FEATURE_DIM:
        raise SystemExit(
            f"init first layer expects {first_weight.shape[1]} features, local HU_FEATURE_DIM={HU_FEATURE_DIM}"
        )
    feature_multiplier = new_feature_scale / old_feature_scale
    feature_bias_shift = (new_feature_mean - old_feature_mean) / old_feature_scale
    state[first_weight_key] = first_weight * feature_multiplier.reshape(1, -1)
    state[first_bias_key] = state[first_bias_key] + first_weight.matmul(feature_bias_shift)

    old_target_scale = float(init_model.target_scale)
    if old_target_scale <= 0.0:
        raise SystemExit("init model target_scale must be positive")
    target_multiplier = old_target_scale / float(target_scale)
    state[last_weight_key] = state[last_weight_key] * target_multiplier
    state[last_bias_key] = (
        state[last_bias_key] * old_target_scale
        + float(init_model.target_mean)
        - float(target_mean)
    ) / float(target_scale)
    return state


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
            x = torch.from_numpy(features[start:end]).to(device=device, dtype=torch.float32)
            x = (x - mean_tensor) / scale_tensor
            pred = net(x).squeeze(-1).detach().cpu().numpy().astype(np.float64)
            pred = pred * target_scale + target_mean
            err = pred - targets[start:end]
            squared_error_sum += float(np.dot(err, err))
    return squared_error_sum / float(features.shape[0])


def evaluate_cached(
    model: HuTorchActionValueModel,
    features: np.ndarray,
    targets: np.ndarray,
    group_offsets: np.ndarray,
    *,
    limit: int,
    tie_tolerance: float = 1e-9,
) -> dict[str, float]:
    sample_count = int(group_offsets.shape[0] - 1)
    if sample_count <= 0:
        return {
            "samples": 0.0,
            "actions": 0.0,
            "mse": 0.0,
            "top1_accuracy": 0.0,
            "top3_accuracy": 0.0,
            "avg_regret": 0.0,
        }
    if limit > 0:
        sample_count = min(sample_count, limit)
    action_end = int(group_offsets[sample_count])
    predictions = model.predict_matrix(features[:action_end].astype(np.float32, copy=False))
    eval_targets = targets[:action_end].astype(np.float64, copy=False)
    errors = predictions - eval_targets
    squared_error_sum = float(np.dot(errors, errors))
    correct = 0
    top3_correct = 0
    regret_sum = 0.0
    for group_index in range(sample_count):
        start = int(group_offsets[group_index])
        end = int(group_offsets[group_index + 1])
        block_predictions = predictions[start:end]
        block_targets = eval_targets[start:end]
        predicted_idx = int(np.argmax(block_predictions))
        true_best_score = float(np.max(block_targets))
        correct += int(block_targets[predicted_idx] >= true_best_score - tie_tolerance)
        top_k = min(3, end - start)
        top_indices = np.argpartition(block_predictions, -top_k)[-top_k:]
        top3_correct += int(np.any(block_targets[top_indices] >= true_best_score - tie_tolerance))
        regret_sum += float(true_best_score - block_targets[predicted_idx])
    return {
        "samples": float(sample_count),
        "actions": float(action_end),
        "mse": float(squared_error_sum / max(action_end, 1)),
        "top1_accuracy": float(correct / sample_count),
        "top3_accuracy": float(top3_correct / sample_count),
        "avg_regret": float(regret_sum / sample_count),
    }


def train_epoch(
    torch,
    net,
    optimizer,
    *,
    features: np.ndarray,
    targets_norm: np.ndarray,
    weights: np.ndarray,
    group_offsets: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    device: str,
    batch_size: int,
    rng: np.random.Generator,
    ranking_loss_weight: float,
    ranking_max_margin: float,
    ranking_gap_tolerance: float,
) -> float:
    mean_tensor = torch.from_numpy(feature_mean).to(device)
    scale_tensor = torch.from_numpy(feature_scale).to(device)
    loss_fn = torch.nn.MSELoss(reduction="none")
    order = rng.permutation(features.shape[0])
    loss_sum = 0.0
    weight_sum = 0.0
    for start in range(0, order.shape[0], batch_size):
        batch_idx = order[start : start + batch_size]
        x = torch.from_numpy(features[batch_idx]).to(device=device, dtype=torch.float32)
        y = torch.from_numpy(targets_norm[batch_idx]).to(device)
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
        group_order = rng.permutation(group_offsets.shape[0] - 1)
        group_cursor = 0
        while group_cursor < group_order.shape[0]:
            offsets = [0]
            feature_parts: list[np.ndarray] = []
            target_parts: list[np.ndarray] = []
            action_count = 0
            while group_cursor < group_order.shape[0]:
                group_index = int(group_order[group_cursor])
                start = int(group_offsets[group_index])
                end = int(group_offsets[group_index + 1])
                length = end - start
                if feature_parts and action_count + length > batch_size:
                    break
                feature_parts.append(features[start:end])
                target_parts.append(targets_norm[start:end])
                action_count += length
                offsets.append(action_count)
                group_cursor += 1
                if action_count >= batch_size:
                    break
            x_np = np.vstack(feature_parts)
            y_np = np.concatenate(target_parts)
            x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
            y = torch.from_numpy(y_np).to(device)
            x = (x - mean_tensor) / scale_tensor
            optimizer.zero_grad(set_to_none=True)
            pred = net(x).squeeze(-1)
            rank_loss = ranking_loss_for_grouped_predictions(
                torch,
                pred,
                y,
                offsets,
                max_margin=ranking_max_margin,
                gap_tolerance=ranking_gap_tolerance,
            )
            loss = rank_loss * ranking_loss_weight
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * action_count
            weight_sum += float(action_count)

    return loss_sum / max(weight_sum, 1.0)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise SystemExit("epochs must be positive")
    if args.batch_size <= 0:
        raise SystemExit("batch-size must be positive")
    if args.best_action_weight < 1.0:
        raise SystemExit("best-action-weight must be >= 1")
    if args.cache_weight <= 0:
        raise SystemExit("--cache-weight must be positive")
    if len(args.extra_cache_weight) > len(args.extra_cache_dir):
        raise SystemExit("--extra-cache-weight cannot be specified more times than --extra-cache-dir")
    extra_cache_weights = list(args.extra_cache_weight)
    while len(extra_cache_weights) < len(args.extra_cache_dir):
        extra_cache_weights.append(1.0)
    if any(weight <= 0 for weight in extra_cache_weights):
        raise SystemExit("--extra-cache-weight values must be positive")

    import torch

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    started_at = time.time()
    cache_dirs = [args.cache_dir.resolve(), *[path.resolve() for path in args.extra_cache_dir]]
    cache_weights = [float(args.cache_weight), *[float(weight) for weight in extra_cache_weights]]
    caches = load_caches(cache_dirs, cache_weights)
    sample_splits: list[tuple[np.ndarray, np.ndarray]] = []
    train_sample_indices_by_cache: list[np.ndarray] = []
    holdout_sample_indices_by_cache: list[np.ndarray] = []
    cache_summaries = []
    train_actions = 0
    holdout_actions = 0
    total_samples = 0
    for cache_index, cache in enumerate(caches):
        cache_total_samples = int(cache.offsets.shape[0] - 1)
        total_samples += cache_total_samples
        train_sample_indices, holdout_sample_indices = split_sample_indices(
            cache_total_samples,
            holdout_fraction=args.holdout,
            seed=args.seed + cache_index,
        )
        sample_splits.append((train_sample_indices, holdout_sample_indices))
        train_sample_indices_by_cache.append(train_sample_indices)
        holdout_sample_indices_by_cache.append(holdout_sample_indices)
        cache_train_actions = count_actions_for_indices(cache.offsets, train_sample_indices)
        cache_holdout_actions = count_actions_for_indices(cache.offsets, holdout_sample_indices)
        train_actions += cache_train_actions
        holdout_actions += cache_holdout_actions
        cache_summaries.append(
            {
                "cache_dir": str(cache.cache_dir),
                "weight": float(cache.weight),
                "samples": cache_total_samples,
                "actions": int(cache.metadata["actions"]),
                "train_samples": int(train_sample_indices.shape[0]),
                "holdout_samples": int(holdout_sample_indices.shape[0]),
                "train_actions": int(cache_train_actions),
                "holdout_actions": int(cache_holdout_actions),
                "source_counts": cache.metadata.get("source_counts", {}),
                "input": cache.metadata.get("input"),
            }
        )
    estimated_bytes = (
        train_actions * HU_FEATURE_DIM * np.dtype(np.float16).itemsize
        + holdout_actions * HU_FEATURE_DIM * np.dtype(np.float16).itemsize
        + (train_actions + holdout_actions) * np.dtype(np.float32).itemsize
    )
    if estimated_bytes > args.max_ram_gb * (1024**3):
        raise SystemExit(
            f"estimated cached tensors need {estimated_bytes / (1024**3):.1f} GiB; "
            f"increase --max-ram-gb or use a smaller cache"
        )

    train_features, train_targets, train_group_offsets, train_group_cache_indices = materialize_multi_split(
        caches,
        train_sample_indices_by_cache,
    )
    holdout_features, holdout_targets, holdout_group_offsets, holdout_group_cache_indices = materialize_multi_split(
        caches,
        holdout_sample_indices_by_cache,
    )
    feature_mean, feature_scale, target_mean, target_scale = normalize_stats(
        train_features,
        train_targets,
    )
    train_targets_norm = ((train_targets - target_mean) / target_scale).astype(np.float32)
    train_weights = action_weights(
        train_targets,
        train_group_offsets,
        best_action_weight=args.best_action_weight,
        best_action_tolerance=args.best_action_tolerance,
    )
    train_weights = apply_group_source_weights(
        train_weights,
        train_group_offsets,
        train_group_cache_indices,
        cache_weights,
    )

    device = select_device(torch, args.device)
    hidden_layers = parse_hidden_layers(args.hidden_layer_sizes)
    net = _build_torch_mlp(torch, HU_FEATURE_DIM, hidden_layers, args.dropout).to(device)
    if args.init_model is not None:
        init_model = HuTorchActionValueModel.load(args.init_model)
        init_state = adapted_initial_state_dict(
            torch,
            init_model,
            hidden_layers=hidden_layers,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            target_mean=target_mean,
            target_scale=target_scale,
        )
        net.load_state_dict(init_state)
        net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_state = None
    best_holdout_mse = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        net.train()
        train_loss = train_epoch(
            torch,
            net,
            optimizer,
            features=train_features,
            targets_norm=train_targets_norm,
            weights=train_weights,
            group_offsets=train_group_offsets,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            device=device,
            batch_size=args.batch_size,
            rng=rng,
            ranking_loss_weight=args.ranking_loss_weight,
            ranking_max_margin=args.ranking_max_margin,
            ranking_gap_tolerance=args.ranking_gap_tolerance,
        )
        net.eval()
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
        if holdout_group_offsets.shape[0] > 1 and holdout_mse < best_holdout_mse:
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
        if holdout_group_offsets.shape[0] > 1 and args.patience > 0 and stale_epochs >= args.patience:
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

    metrics = {
        "cache_dir": str(cache_dirs[0]),
        "cache_dirs": [str(cache_dir) for cache_dir in cache_dirs],
        "cache_weights": cache_weights,
        "cache_summaries": cache_summaries,
        "cache_metadata": caches[0].metadata if len(caches) == 1 else [cache.metadata for cache in caches],
        "model_output": str(args.model_output),
        "model_type": "hu_torch_mlp_cached",
        "init_model": str(args.init_model) if args.init_model is not None else None,
        "device": device,
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "total_samples": total_samples,
        "train_samples": int(sum(indices.shape[0] for indices in train_sample_indices_by_cache)),
        "holdout_samples": int(sum(indices.shape[0] for indices in holdout_sample_indices_by_cache)),
        "train_actions": int(train_actions),
        "holdout_actions": int(holdout_actions),
        "best_epoch": best_epoch,
        "best_holdout_mse": best_holdout_mse,
        "elapsed_seconds": time.time() - started_at,
        "epochs_ran": len(history),
        "history": history,
        "train": evaluate_cached(
            model,
            train_features,
            train_targets,
            train_group_offsets,
            limit=args.max_metric_samples,
        ),
        "holdout": evaluate_cached(
            model,
            holdout_features,
            holdout_targets,
            holdout_group_offsets,
            limit=args.max_metric_samples,
        ),
        "metric_sample_limit": int(args.max_metric_samples),
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
