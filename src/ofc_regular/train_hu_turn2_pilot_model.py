"""Train and evaluate a HU Turn2 pilot multi-head model from a feature cache."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .build_hu_turn2_pilot_feature_cache import SPLIT_ID_TO_NAME, SPLIT_NAME_TO_ID
from .hu_turn3_model import HU_FEATURE_DIM
from .train_torch_action_value import parse_hidden_layers, select_device
from .turn3_model import _build_torch_mlp

REGRESSION_HEADS = ("ev", "delta_vs_baseline", "delta_vs_reference", "rank_score")
GATE_ID_TO_LABEL = {0: "negative", 1: "gray", 2: "positive"}
THRESHOLD_T2_VALUES = (5.0, 8.0, 10.0, 12.0)
THRESHOLD_REFERENCE_VALUES = (10.0, 15.0, 20.0, 25.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/hu_turn2_stage8_pilot_2000_mc512_reference_override_cached_rank_wide.pt"),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-action-rows", type=int, default=32768)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--hidden-layer-sizes", default="1024,512,256")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026061702)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.05)
    parser.add_argument("--listwise-loss-weight", type=float, default=0.05)
    parser.add_argument("--gate-loss-weight", type=float, default=0.2)
    parser.add_argument("--gate-negative-weight", type=float, default=2.0)
    parser.add_argument("--threshold-split", choices=("val", "test", "holdout"), default="test")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_cache(cache_dir: Path) -> dict[str, Any]:
    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    state_count = int(metadata["state_count"])
    action_count = int(metadata["action_count"])
    feature_dtype = np.dtype(metadata.get("feature_dtype", "float32"))
    features = np.memmap(
        cache_dir / f"features.{feature_dtype.name}.mmap",
        dtype=feature_dtype,
        mode="r",
        shape=(action_count, int(metadata.get("feature_dim", HU_FEATURE_DIM))),
    )
    return {
        "metadata": metadata,
        "state_metadata": read_jsonl(cache_dir / "state_metadata.jsonl"),
        "features": features,
        "target_ev": np.memmap(cache_dir / "target_ev.float32.mmap", dtype=np.float32, mode="r", shape=(action_count,)),
        "target_delta_baseline": np.memmap(cache_dir / "target_delta_baseline.float32.mmap", dtype=np.float32, mode="r", shape=(action_count,)),
        "target_delta_reference": np.memmap(cache_dir / "target_delta_reference.float32.mmap", dtype=np.float32, mode="r", shape=(action_count,)),
        "target_rank_score": np.memmap(cache_dir / "target_rank_score.float32.mmap", dtype=np.float32, mode="r", shape=(action_count,)),
        "action_ev_se": np.memmap(cache_dir / "action_ev_se.float32.mmap", dtype=np.float32, mode="r", shape=(action_count,)),
        "offsets": np.load(cache_dir / "sample_offsets.npy"),
        "action_counts": np.load(cache_dir / "sample_action_counts.npy"),
        "split": np.load(cache_dir / "state_split.npy"),
        "baseline_action_index": np.load(cache_dir / "baseline_action_index.npy"),
        "reference_action_index": np.load(cache_dir / "reference_action_index.npy"),
        "fallback_action_index": np.load(cache_dir / "fallback_action_index.npy"),
        "best_action_index": np.load(cache_dir / "best_action_index.npy"),
        "second_best_action_index": np.load(cache_dir / "second_best_action_index.npy"),
        "gate_label_id": np.load(cache_dir / "gate_label_id.npy"),
    }


def state_indices_for_split(split: np.ndarray, name: str) -> np.ndarray:
    if name == "holdout":
        return np.where(split != SPLIT_NAME_TO_ID["train"])[0]
    return np.where(split == SPLIT_NAME_TO_ID[name])[0]


def action_indices_for_states(offsets: np.ndarray, state_indices: Iterable[int]) -> np.ndarray:
    parts = [np.arange(int(offsets[i]), int(offsets[i + 1]), dtype=np.int64) for i in state_indices]
    if not parts:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(parts)


def target_matrix(cache: dict[str, Any]) -> np.ndarray:
    return np.column_stack(
        [
            np.asarray(cache["target_ev"], dtype=np.float32),
            np.asarray(cache["target_delta_baseline"], dtype=np.float32),
            np.asarray(cache["target_delta_reference"], dtype=np.float32),
            np.asarray(cache["target_rank_score"], dtype=np.float32),
        ]
    ).astype(np.float32, copy=False)


def normalize_stats(features: np.ndarray, targets: np.ndarray, action_indices: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(features[action_indices], dtype=np.float32)
    y = targets[action_indices].astype(np.float32, copy=False)
    feature_mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale[feature_scale < 1e-6] = 1.0
    target_mean = y.mean(axis=0, dtype=np.float64).astype(np.float32)
    target_scale = y.std(axis=0, dtype=np.float64).astype(np.float32)
    target_scale[target_scale < 1e-6] = 1.0
    return {
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "target_mean": target_mean,
        "target_scale": target_scale,
    }


def iter_state_batches(
    state_indices: np.ndarray,
    offsets: np.ndarray,
    *,
    batch_action_rows: int,
    rng: np.random.Generator,
) -> Iterable[np.ndarray]:
    shuffled = state_indices.copy()
    rng.shuffle(shuffled)
    current: list[int] = []
    action_rows = 0
    for state_index in shuffled:
        length = int(offsets[state_index + 1] - offsets[state_index])
        if current and action_rows + length > batch_action_rows:
            yield np.asarray(current, dtype=np.int64)
            current = []
            action_rows = 0
        current.append(int(state_index))
        action_rows += length
    if current:
        yield np.asarray(current, dtype=np.int64)


def grouped_action_indices(offsets: np.ndarray, state_indices: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    parts: list[np.ndarray] = []
    groups: list[tuple[int, int, int]] = []
    cursor = 0
    for state_index in state_indices:
        start = int(offsets[state_index])
        end = int(offsets[state_index + 1])
        length = end - start
        parts.append(np.arange(start, end, dtype=np.int64))
        groups.append((int(state_index), cursor, cursor + length))
        cursor += length
    if not parts:
        return np.zeros(0, dtype=np.int64), []
    return np.concatenate(parts), groups


def ranking_loss(torch, pred_ev, target_ev, groups: list[tuple[int, int, int]]):
    losses = []
    for _state_index, start, end in groups:
        if end - start <= 1:
            continue
        group_pred = pred_ev[start:end]
        group_target = target_ev[start:end]
        best = int(torch.argmax(group_target).item())
        teacher_gap = (group_target[best] - group_target).clamp(min=0.0, max=2.0)
        pred_gap = group_pred[best] - group_pred
        mask = teacher_gap > 1e-6
        if bool(mask.any()):
            losses.append(torch.relu(teacher_gap[mask] - pred_gap[mask]).square().mean())
    if not losses:
        return pred_ev.sum() * 0.0
    return torch.stack(losses).mean()


def listwise_loss(torch, pred_ev, target_ev, groups: list[tuple[int, int, int]], *, temperature: float = 1.0):
    losses = []
    for _state_index, start, end in groups:
        if end - start <= 1:
            continue
        target_probs = torch.softmax(target_ev[start:end] / temperature, dim=0)
        log_probs = torch.log_softmax(pred_ev[start:end] / temperature, dim=0)
        losses.append(-(target_probs * log_probs).sum())
    if not losses:
        return pred_ev.sum() * 0.0
    return torch.stack(losses).mean()


def gate_loss(
    torch,
    gate_logits,
    groups: list[tuple[int, int, int]],
    gate_labels: np.ndarray,
    *,
    negative_weight: float,
):
    logits = []
    labels = []
    weights = []
    for state_index, start, end in groups:
        label_id = int(gate_labels[state_index])
        if label_id == 1:
            continue
        logits.append(gate_logits[start:end].mean())
        labels.append(1.0 if label_id == 2 else 0.0)
        weights.append(1.0 if label_id == 2 else float(negative_weight))
    if not logits:
        return gate_logits.sum() * 0.0
    logit_tensor = torch.stack(logits)
    label_tensor = torch.tensor(labels, dtype=gate_logits.dtype, device=gate_logits.device)
    weight_tensor = torch.tensor(weights, dtype=gate_logits.dtype, device=gate_logits.device)
    return (
        torch.nn.functional.binary_cross_entropy_with_logits(
            logit_tensor,
            label_tensor,
            reduction="none",
        )
        * weight_tensor
    ).sum() / weight_tensor.sum().clamp_min(1.0)


def predict_all(torch, net, cache: dict[str, Any], stats: dict[str, np.ndarray], device: str, batch_size: int) -> np.ndarray:
    features = cache["features"]
    mean = torch.from_numpy(stats["feature_mean"]).to(device)
    scale = torch.from_numpy(stats["feature_scale"]).to(device)
    target_mean = stats["target_mean"].astype(np.float32)
    target_scale = stats["target_scale"].astype(np.float32)
    preds: list[np.ndarray] = []
    net.eval()
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            end = min(start + batch_size, features.shape[0])
            x = torch.from_numpy(np.array(features[start:end], dtype=np.float32, copy=True)).to(device)
            x = (x - mean) / scale
            out = net(x).detach().cpu().numpy().astype(np.float32)
            out[:, :4] = out[:, :4] * target_scale + target_mean
            preds.append(out)
    return np.vstack(preds) if preds else np.zeros((0, 5), dtype=np.float32)


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def corr(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def split_eval(
    cache: dict[str, Any],
    predictions: np.ndarray,
    targets: np.ndarray,
    split_name: str,
    state_indices: np.ndarray,
) -> dict[str, Any]:
    offsets = cache["offsets"]
    baseline_idx = cache["baseline_action_index"]
    gate_labels = cache["gate_label_id"]
    ev_errors: list[float] = []
    delta_b_errors: list[float] = []
    delta_r_errors: list[float] = []
    regrets: list[float] = []
    top1 = 0
    top3 = 0
    pair_correct = 0
    pair_total = 0
    pred_delta_values: list[float] = []
    teacher_delta_values: list[float] = []
    gate_correct = 0
    gate_total = 0
    for state_index in state_indices:
        start = int(offsets[state_index])
        end = int(offsets[state_index + 1])
        pred = predictions[start:end]
        y = targets[start:end]
        ev_errors.extend(np.abs(pred[:, 0] - y[:, 0]).astype(float).tolist())
        delta_b_errors.extend(np.abs(pred[:, 1] - y[:, 1]).astype(float).tolist())
        delta_r_errors.extend(np.abs(pred[:, 2] - y[:, 2]).astype(float).tolist())
        predicted_idx = int(np.argmax(pred[:, 0]))
        best_idx = int(np.argmax(y[:, 0]))
        best_ev = float(y[best_idx, 0])
        regrets.append(best_ev - float(y[predicted_idx, 0]))
        top1 += int(predicted_idx == best_idx)
        top_k = min(3, end - start)
        top_indices = np.argpartition(pred[:, 0], -top_k)[-top_k:]
        top3 += int(best_idx in set(int(i) for i in top_indices))
        for i in range(end - start):
            for j in range(i + 1, end - start):
                target_cmp = float(y[i, 0] - y[j, 0])
                if abs(target_cmp) <= 1e-9:
                    continue
                pred_cmp = float(pred[i, 0] - pred[j, 0])
                pair_correct += int((target_cmp > 0) == (pred_cmp > 0))
                pair_total += 1
        base = int(baseline_idx[state_index])
        pred_delta_values.append(float(pred[predicted_idx, 0] - pred[base, 0]))
        teacher_delta_values.append(float(y[predicted_idx, 0] - y[base, 0]))
        label = int(gate_labels[state_index])
        if label != 1:
            gate_logit = float(pred[:, 4].mean())
            gate_pred = 1 if gate_logit >= 0.0 else 0
            gate_true = 1 if label == 2 else 0
            gate_correct += int(gate_pred == gate_true)
            gate_total += 1

    state_count = int(state_indices.size)
    return {
        "split": split_name,
        "states": state_count,
        "actions": int(sum(int(offsets[i + 1] - offsets[i]) for i in state_indices)),
        "ev_mae": float(np.mean(ev_errors)) if ev_errors else 0.0,
        "ev_rmse": float(np.sqrt(np.mean(np.square(ev_errors)))) if ev_errors else 0.0,
        "delta_vs_baseline_mae": float(np.mean(delta_b_errors)) if delta_b_errors else 0.0,
        "delta_vs_reference_mae": float(np.mean(delta_r_errors)) if delta_r_errors else 0.0,
        "avg_regret": float(np.mean(regrets)) if regrets else 0.0,
        "p90_regret": quantile(regrets, 0.90),
        "p95_regret": quantile(regrets, 0.95),
        "p99_regret": quantile(regrets, 0.99),
        "top1_accuracy": float(top1 / state_count) if state_count else 0.0,
        "top3_recall": float(top3 / state_count) if state_count else 0.0,
        "pairwise_ranking_accuracy": float(pair_correct / pair_total) if pair_total else 0.0,
        "calibration_corr_predicted_delta_teacher_delta": corr(pred_delta_values, teacher_delta_values),
        "gate_accuracy_pos_neg": float(gate_correct / gate_total) if gate_total else 0.0,
        "gate_eval_states": gate_total,
    }


def subset_rows(
    cache: dict[str, Any],
    predictions: np.ndarray,
    targets: np.ndarray,
    split_name: str,
    predicate,
    label: str,
) -> dict[str, Any]:
    split_indices = state_indices_for_split(cache["split"], split_name)
    subset = np.asarray(
        [int(index) for index in split_indices if predicate(cache["state_metadata"][int(index)])],
        dtype=np.int64,
    )
    row = split_eval(cache, predictions, targets, label, subset)
    row["group"] = label
    return row


def threshold_sweep(
    cache: dict[str, Any],
    predictions: np.ndarray,
    targets: np.ndarray,
    split_name: str,
) -> list[dict[str, Any]]:
    state_indices = state_indices_for_split(cache["split"], split_name)
    offsets = cache["offsets"]
    baseline_idx = cache["baseline_action_index"]
    metadata = cache["state_metadata"]
    rows: list[dict[str, Any]] = []
    for min_margin in THRESHOLD_T2_VALUES:
        for reference_margin in THRESHOLD_REFERENCE_VALUES:
            gains: list[float] = []
            losses: list[float] = []
            fired = 0
            evaluated = int(state_indices.size)
            first_fired = 0
            second_fired = 0
            first_false_positive = 0
            second_false_positive = 0
            bucket_counts: Counter[str] = Counter()
            bucket_false_positive: Counter[str] = Counter()
            for state_index in state_indices:
                start = int(offsets[state_index])
                end = int(offsets[state_index + 1])
                base = int(baseline_idx[state_index])
                pred_delta = predictions[start:end, 1]
                candidate = int(np.argmax(pred_delta))
                state_reference_margin = float(metadata[int(state_index)].get("baseline_model_margin", 0.0) or 0.0)
                if candidate == base:
                    continue
                if float(pred_delta[candidate]) < min_margin:
                    continue
                if state_reference_margin < reference_margin:
                    continue
                gain = float(targets[start + candidate, 0] - targets[start + base, 0])
                gains.append(gain)
                losses.append(max(0.0, -gain))
                fired += 1
                seat = str(metadata[int(state_index)].get("seat", "unknown"))
                bucket = str(metadata[int(state_index)].get("bucket_group", "unknown"))
                bucket_counts[bucket] += 1
                false_positive = gain < 0.0
                if false_positive:
                    bucket_false_positive[bucket] += 1
                if seat == "first":
                    first_fired += 1
                    first_false_positive += int(false_positive)
                elif seat == "second":
                    second_fired += 1
                    second_false_positive += int(false_positive)
            false_positive_count = sum(1 for value in gains if value < 0.0)
            rows.append(
                {
                    "split": split_name,
                    "hu_turn2_min_margin": min_margin,
                    "hu_turn2_reference_min_margin": reference_margin,
                    "evaluated_states": evaluated,
                    "override_count": fired,
                    "override_rate": fired / evaluated if evaluated else 0.0,
                    "teacher_avg_gain_on_override": float(np.mean(gains)) if gains else 0.0,
                    "false_positive_count": false_positive_count,
                    "false_positive_rate": false_positive_count / fired if fired else 0.0,
                    "p95_loss": quantile(losses, 0.95),
                    "p99_loss": quantile(losses, 0.99),
                    "first_override_count": first_fired,
                    "second_override_count": second_fired,
                    "first_false_positive_count": first_false_positive,
                    "second_false_positive_count": second_false_positive,
                    "bucket_override_counts": json.dumps(dict(bucket_counts), sort_keys=True),
                    "bucket_false_positive_counts": json.dumps(dict(bucket_false_positive), sort_keys=True),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict[str, Any], threshold_rows: list[dict[str, Any]]) -> None:
    test = summary["eval"]["test"]
    val = summary["eval"]["val"]
    positive_thresholds = [
        row
        for row in threshold_rows
        if row["override_count"] > 0 and row["teacher_avg_gain_on_override"] > 0.0
    ]
    lines = [
        "# HU T2 Stage8 Pilot 2,000 MC512 Training",
        "",
        "This is a pilot pipeline validation, not a production candidate.",
        "",
        f"- model: `{summary['model_output']}`",
        f"- cache: `{summary['cache_dir']}`",
        f"- device: `{summary['device']}`",
        f"- epochs ran: `{summary['epochs_ran']}`",
        f"- best epoch: `{summary['best_epoch']}`",
        f"- train/val/test states: `{summary['split_counts']['train']}` / `{summary['split_counts']['val']}` / `{summary['split_counts']['test']}`",
        "",
        "## Holdout Metrics",
        "",
        f"- val EV MAE / avg_regret / top3: `{val['ev_mae']:.4f}` / `{val['avg_regret']:.4f}` / `{val['top3_recall']:.4f}`",
        f"- test EV MAE / avg_regret / top3: `{test['ev_mae']:.4f}` / `{test['avg_regret']:.4f}` / `{test['top3_recall']:.4f}`",
        f"- test delta baseline MAE: `{test['delta_vs_baseline_mae']:.4f}`",
        f"- test pairwise ranking accuracy: `{test['pairwise_ranking_accuracy']:.4f}`",
        f"- test gate accuracy pos/neg: `{test['gate_accuracy_pos_neg']:.4f}`",
        "",
        "## Threshold Sweep",
        "",
        f"- configs with positive teacher gain and at least one override: `{len(positive_thresholds)}`",
        "- This sweep is teacher-holdout only; do not use it as production evidence.",
        "",
        "## Next Step",
        "",
        summary["recommended_next_step"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise SystemExit("epochs must be positive")
    if args.batch_action_rows <= 0:
        raise SystemExit("batch-action-rows must be positive")
    started_at = time.time()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)

    import torch

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = select_device(torch, args.device)
    hidden_layers = parse_hidden_layers(args.hidden_layer_sizes)

    cache = load_cache(args.cache_dir.resolve())
    targets = target_matrix(cache)
    train_states = state_indices_for_split(cache["split"], "train")
    val_states = state_indices_for_split(cache["split"], "val")
    test_states = state_indices_for_split(cache["split"], "test")
    train_actions = action_indices_for_states(cache["offsets"], train_states)
    stats = normalize_stats(cache["features"], targets, train_actions)

    net = _build_torch_mlp(torch, int(cache["metadata"]["feature_dim"]), hidden_layers, args.dropout, output_dim=5).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    mean_tensor = torch.from_numpy(stats["feature_mean"]).to(device)
    scale_tensor = torch.from_numpy(stats["feature_scale"]).to(device)
    target_mean_tensor = torch.from_numpy(stats["target_mean"]).to(device)
    target_scale_tensor = torch.from_numpy(stats["target_scale"]).to(device)
    regression_weights = torch.tensor([1.0, 0.7, 0.4, 0.25], dtype=torch.float32, device=device)

    best_state = None
    best_val_regret = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed)

    for epoch in range(1, args.epochs + 1):
        net.train()
        loss_sum = 0.0
        batch_count = 0
        for batch_states in iter_state_batches(
            train_states,
            cache["offsets"],
            batch_action_rows=args.batch_action_rows,
            rng=rng,
        ):
            action_indices, groups = grouped_action_indices(cache["offsets"], batch_states)
            x = torch.from_numpy(np.array(cache["features"][action_indices], dtype=np.float32, copy=True)).to(device)
            y = torch.from_numpy(targets[action_indices].astype(np.float32, copy=False)).to(device)
            x = (x - mean_tensor) / scale_tensor
            y_norm = (y - target_mean_tensor) / target_scale_tensor
            optimizer.zero_grad(set_to_none=True)
            pred = net(x)
            regression = torch.nn.functional.smooth_l1_loss(
                pred[:, :4],
                y_norm,
                reduction="none",
            )
            loss = (regression * regression_weights).mean()
            if args.ranking_loss_weight > 0.0:
                loss = loss + args.ranking_loss_weight * ranking_loss(torch, pred[:, 0], y_norm[:, 0], groups)
            if args.listwise_loss_weight > 0.0:
                loss = loss + args.listwise_loss_weight * listwise_loss(torch, pred[:, 0], y_norm[:, 0], groups)
            if args.gate_loss_weight > 0.0:
                loss = loss + args.gate_loss_weight * gate_loss(
                    torch,
                    pred[:, 4],
                    groups,
                    cache["gate_label_id"],
                    negative_weight=args.gate_negative_weight,
                )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            batch_count += 1

        predictions = predict_all(torch, net, cache, stats, device, args.batch_action_rows)
        val_eval = split_eval(cache, predictions, targets, "val", val_states)
        epoch_row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(batch_count, 1),
            "val_avg_regret": val_eval["avg_regret"],
            "val_ev_mae": val_eval["ev_mae"],
            "val_top3_recall": val_eval["top3_recall"],
        }
        history.append(epoch_row)
        if val_eval["avg_regret"] < best_val_regret:
            best_val_regret = float(val_eval["avg_regret"])
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if args.log_every > 0 and (epoch == 1 or epoch % args.log_every == 0):
            print(json.dumps({"event": "epoch", **epoch_row}, separators=(",", ":")), flush=True)
        if args.patience > 0 and stale_epochs >= args.patience:
            break

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
        best_epoch = history[-1]["epoch"]
    net.load_state_dict(best_state)
    predictions = predict_all(torch, net, cache, stats, device, args.batch_action_rows)
    eval_rows = {
        "train": split_eval(cache, predictions, targets, "train", train_states),
        "val": split_eval(cache, predictions, targets, "val", val_states),
        "test": split_eval(cache, predictions, targets, "test", test_states),
    }

    breakdown_rows: list[dict[str, Any]] = []
    for split_name in ("val", "test"):
        for bucket in sorted({str(row["bucket_group"]) for row in cache["state_metadata"]}):
            breakdown_rows.append(
                subset_rows(
                    cache,
                    predictions,
                    targets,
                    split_name,
                    lambda row, bucket=bucket: str(row["bucket_group"]) == bucket,
                    f"{split_name}:bucket:{bucket}",
                )
            )
    position_rows: list[dict[str, Any]] = []
    for split_name in ("val", "test"):
        for seat in ("first", "second"):
            position_rows.append(
                subset_rows(
                    cache,
                    predictions,
                    targets,
                    split_name,
                    lambda row, seat=seat: str(row["seat"]) == seat,
                    f"{split_name}:position:{seat}",
                )
            )
    label_rows: list[dict[str, Any]] = []
    for split_name in ("val", "test"):
        for label in ("positive", "gray", "negative"):
            label_rows.append(
                subset_rows(
                    cache,
                    predictions,
                    targets,
                    split_name,
                    lambda row, label=label: str(row["pilot_gate_label"]) == label,
                    f"{split_name}:label:{label}",
                )
            )
    margin_rows: list[dict[str, Any]] = []
    for split_name in ("val", "test"):
        for bucket in ("lt_0_05", "0_05_0_10", "0_10_0_25", "0_25_0_50", "0_50_1_00", "ge_1_00"):
            margin_rows.append(
                subset_rows(
                    cache,
                    predictions,
                    targets,
                    split_name,
                    lambda row, bucket=bucket: str(row["margin_bucket"]) == bucket,
                    f"{split_name}:margin:{bucket}",
                )
            )

    threshold_split = "holdout" if args.threshold_split == "holdout" else args.threshold_split
    threshold_rows = threshold_sweep(cache, predictions, targets, threshold_split)

    calibration_rows: list[dict[str, Any]] = []
    for state_index in state_indices_for_split(cache["split"], threshold_split):
        start = int(cache["offsets"][state_index])
        end = int(cache["offsets"][state_index + 1])
        base = int(cache["baseline_action_index"][state_index])
        pred = predictions[start:end]
        y = targets[start:end]
        candidate = int(np.argmax(pred[:, 1]))
        calibration_rows.append(
            {
                "state_index": int(state_index),
                "split": threshold_split,
                "predicted_delta": float(pred[candidate, 1]),
                "teacher_delta": float(y[candidate, 0] - y[base, 0]),
                "teacher_best_delta": float(cache["state_metadata"][int(state_index)]["delta_best_vs_baseline"]),
                "bucket_group": cache["state_metadata"][int(state_index)]["bucket_group"],
                "seat": cache["state_metadata"][int(state_index)]["seat"],
                "pilot_gate_label": cache["state_metadata"][int(state_index)]["pilot_gate_label"],
            }
        )

    split_counts = {
        name: int(np.sum(cache["split"] == split_id))
        for split_id, name in SPLIT_ID_TO_NAME.items()
    }
    positive_thresholds = [
        row
        for row in threshold_rows
        if row["override_count"] > 0 and row["teacher_avg_gain_on_override"] > 0.0
    ]
    recommended = (
        "GO to a larger 20k-50k MC512 broad pass only after the same pipeline is repeated with "
        "a larger holdout and then seat-swap validation. This pilot is enough to validate the "
        "cache/training pipeline, not production adoption."
        if eval_rows["test"]["top3_recall"] > 0.5 and eval_rows["test"]["avg_regret"] >= 0.0
        else "NO-GO to 50k until feature/label/model issues are inspected; pilot metrics are weak."
    )
    summary = {
        "schema": "hu_turn2_stage8_pilot_training_v1",
        "cache_dir": str(args.cache_dir.resolve()),
        "output_dir": str(output_dir),
        "model_output": str(args.model_output),
        "device": device,
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "hidden_layer_sizes": list(hidden_layers),
        "dropout": args.dropout,
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "history": history,
        "split_counts": split_counts,
        "eval": eval_rows,
        "threshold_positive_config_count": len(positive_thresholds),
        "elapsed_seconds": time.time() - started_at,
        "recommended_next_step": recommended,
    }

    torch.save(
        {
            "model_kind": "hu_turn2_pilot_multihead_mlp",
            "feature_dim": int(cache["metadata"]["feature_dim"]),
            "hidden_layer_sizes": list(hidden_layers),
            "dropout": args.dropout,
            "state_dict": best_state,
            "feature_mean": stats["feature_mean"],
            "feature_scale": stats["feature_scale"],
            "target_mean": stats["target_mean"],
            "target_scale": stats["target_scale"],
            "heads": ["ev", "delta_vs_baseline", "delta_vs_reference", "rank_score", "override_gate_logit"],
            "t3_continuation_policy": "Stage7_candidate_A_m5_r10",
            "hu_turn3_min_margin": 5.0,
            "hu_turn3_reference_min_margin": 10.0,
            "pilot_only": True,
        },
        args.model_output,
    )

    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "feature_cache_summary.json").write_text(
        json.dumps(cache["metadata"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "split_summary.json").write_text(
        json.dumps(cache["metadata"].get("split_summary", {}), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(output_dir / "training_summary.md", summary, threshold_rows)
    write_csv(output_dir / "training_metrics.csv", history)
    write_csv(output_dir / "holdout_eval.csv", [eval_rows["val"], eval_rows["test"]])
    write_csv(output_dir / "bucket_breakdown.csv", breakdown_rows)
    write_csv(output_dir / "position_breakdown.csv", position_rows)
    write_csv(output_dir / "label_breakdown.csv", label_rows)
    write_csv(output_dir / "margin_bucket_breakdown.csv", margin_rows)
    write_csv(output_dir / "threshold_sweep.csv", threshold_rows)
    write_csv(output_dir / "calibration_plot_data.csv", calibration_rows)
    (output_dir / "recommended_next_step.md").write_text(recommended + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
