"""Train a HU T2 Stage8c realized-delta head.

This is a research-only candidate-generator diagnostic. It predicts realized
whole-game paired delta for Stage8c TopK+confirm rows and evaluates ranking by
observed delta. It is not a production runtime gate.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .analyze_hu_turn2_stage8b_risk_targets import (
    load_rows,
    recommended_use,
    safe_float,
    safe_int,
    target_paths,
    truthy,
)
from .train_hu_turn2_stage8c_risk_head import (
    DEFAULT_TOPK_METRIC_COUNTS,
    FEATURE_MODE_OPPORTUNITY_PROXY_PLUS_PRECONFIRM_META,
    FEATURE_MODES,
    MAX_ROWS_MODE_FIRST,
    MAX_ROWS_MODE_STRATIFIED,
    SPLIT_ID_TO_NAME,
    SPLIT_MODE_ROW_STRATIFIED,
    SPLIT_MODE_SOURCE_LOG,
    SPLIT_MODE_SOURCE_SEED,
    SPLIT_MODES,
    SPLIT_NAME_TO_ID,
    exclude_recommended_use_rows,
    feature_column_names,
    fixed_group_split,
    group_stratified_split,
    limit_rows_for_training,
    risk_feature_vector_for_row,
    split_group_key,
    write_csv,
)
from .train_torch_action_value import parse_hidden_layers, select_device
from .turn3_model import _build_torch_mlp, _import_torch

TARGET_MODE_POLICY_DELTA = "policy_delta"
TARGET_MODE_OBSERVED_DELTA = "observed_delta"
TARGET_MODES = (TARGET_MODE_POLICY_DELTA, TARGET_MODE_OBSERVED_DELTA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", action="append", type=Path, default=[])
    parser.add_argument("--collection-dir", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--target-mode", choices=TARGET_MODES, default=TARGET_MODE_POLICY_DELTA)
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default=FEATURE_MODE_OPPORTUNITY_PROXY_PLUS_PRECONFIRM_META)
    parser.add_argument("--split-mode", choices=SPLIT_MODES, default=SPLIT_MODE_SOURCE_SEED)
    parser.add_argument("--fixed-val-groups", default="")
    parser.add_argument("--fixed-test-groups", default="")
    parser.add_argument("--exclude-recommended-use", action="append", default=[])
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-rows-mode", choices=(MAX_ROWS_MODE_FIRST, MAX_ROWS_MODE_STRATIFIED), default=MAX_ROWS_MODE_FIRST)
    parser.add_argument("--max-rows-seed", type=int, default=2026064518)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-layer-sizes", default="32")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--loss", choices=("huber", "mse"), default="huber")
    parser.add_argument("--target-clip", type=float, default=40.0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=2026061620)
    return parser.parse_args()


def parse_fixed_groups(value: str) -> set[str]:
    return {part.strip() for part in str(value or "").split(",") if part.strip()}


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def delta_observed(row: dict[str, Any]) -> bool:
    use = recommended_use(row)
    if use in {
        "topk_confirm_realized_positive",
        "topk_confirm_realized_loss",
        "topk_confirm_replay_positive",
        "topk_confirm_replay_negative",
        "topk_confirm_replay_gray",
    }:
        return True
    return truthy(row.get("realized_delta_observed")) and truthy(row.get("realized_delta_valid"))


def delta_target_for_row(row: dict[str, Any], target_mode: str) -> tuple[float, str] | None:
    use = recommended_use(row)
    if target_mode == TARGET_MODE_POLICY_DELTA:
        if use in {"topk_confirm_rejected", "topk_confirm_topk_empty"}:
            return 0.0, use
        value = finite_float(row.get("realized_delta", row.get("realized_candidate_seat_delta")))
        if value is None:
            return None
        return value, use
    if target_mode == TARGET_MODE_OBSERVED_DELTA:
        if not delta_observed(row):
            return None
        value = finite_float(row.get("realized_delta", row.get("realized_candidate_seat_delta")))
        if value is None:
            return None
        return value, use
    raise ValueError(f"unknown target mode {target_mode!r}")


def first_present(row: dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def materialize_delta_rows(
    rows: list[dict[str, Any]],
    *,
    feature_mode: str,
    target_mode: str,
    target_clip: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], Counter[str]]:
    features: list[np.ndarray] = []
    targets: list[float] = []
    metadata: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for row in rows:
        target = delta_target_for_row(row, target_mode)
        if target is None:
            skipped["target_missing"] += 1
            continue
        value, target_group = target
        if target_clip > 0:
            value = max(-target_clip, min(target_clip, value))
        try:
            feature_row = risk_feature_vector_for_row(row, feature_mode)
        except Exception as exc:  # pragma: no cover - surfaced through manifest.
            skipped[f"feature_error:{type(exc).__name__}"] += 1
            continue
        features.append(feature_row)
        targets.append(float(value))
        metadata.append(
            {
                "row_index": len(metadata),
                "source_log": row.get("source_log", ""),
                "config_id": row.get("config_id", ""),
                "hand_seed": row.get("hand_seed", ""),
                "split_group_source_log": split_group_key(row, SPLIT_MODE_SOURCE_LOG),
                "split_group_source_seed": split_group_key(row, SPLIT_MODE_SOURCE_SEED),
                "seat": row.get("seat", ""),
                "seat_swap": row.get("seat_swap", ""),
                "candidate_source": row.get("candidate_source", ""),
                "recommended_training_use": recommended_use(row),
                "delta_target_group": target_group,
                "state_signature": row.get("state_signature", ""),
                "action_signature": row.get("action_signature", ""),
                "baseline_action_signature": row.get("baseline_action_signature", ""),
                "candidate_index": safe_int(
                    first_present(row, "candidate_index", "candidate_action_index", "rerank_best_index", default=-1),
                    -1,
                ),
                "baseline_index": safe_int(first_present(row, "baseline_index", "baseline_action_index", default=-1), -1),
                "target_delta": float(value),
                "target_positive": int(value > 0.0),
                "target_negative": int(value < 0.0),
                "target_observed": int(delta_observed(row)),
                "realized_delta": safe_float(row.get("realized_delta", row.get("realized_candidate_seat_delta"))),
                "realized_loss": max(0.0, -safe_float(row.get("realized_delta", row.get("realized_candidate_seat_delta")))),
                "confirm_delta": safe_float(row.get("confirm_delta")),
                "confirm_delta_se": safe_float(row.get("confirm_delta_se")),
                "predicted_delta": safe_float(row.get("predicted_delta")),
                "gate_probability": safe_float(row.get("gate_probability")),
                "candidate_ev_rank": safe_int(row.get("candidate_ev_rank"), 9999),
            }
        )
    if not features:
        raise ValueError(f"no rows for delta target mode {target_mode}; skipped={dict(skipped)}")
    return np.vstack(features).astype(np.float32), np.asarray(targets, dtype=np.float32), metadata, skipped


def split_labels_for_regression(targets: np.ndarray) -> np.ndarray:
    return (targets > 0.0).astype(np.float32)


def mean_and_se(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    mean = float(np.mean(values))
    if values.size == 1:
        return mean, 0.0
    return mean, float(np.std(values, ddof=1) / math.sqrt(values.size))


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return 0.0
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    start = 0
    sorted_values = values[order]
    while start < sorted_values.size:
        end = start + 1
        while end < sorted_values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = float(np.mean(ranks[order[start:end]]))
        start = end
    return ranks


def regression_metrics(targets: np.ndarray, predictions: np.ndarray, *, split_name: str) -> dict[str, Any]:
    if targets.size == 0:
        return {"split": split_name, "rows": 0}
    errors = predictions - targets
    return {
        "split": split_name,
        "rows": int(targets.size),
        "target_mean": float(np.mean(targets)),
        "prediction_mean": float(np.mean(predictions)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(math.sqrt(float(np.mean(errors**2)))),
        "pearson": pearson_corr(targets, predictions),
        "spearman": pearson_corr(rank_values(targets), rank_values(predictions)),
        "positive_rate": float(np.mean(targets > 0.0)),
        "negative_rate": float(np.mean(targets < 0.0)),
    }


def topk_delta_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    split_name: str,
    topk_counts: Iterable[int] = DEFAULT_TOPK_METRIC_COUNTS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if targets.size == 0:
        return rows
    order = np.argsort(-predictions)
    for topk in topk_counts:
        k = min(int(topk), targets.size)
        if k <= 0:
            continue
        idx = order[:k]
        selected = targets[idx]
        losses = np.maximum(0.0, -selected)
        rows.append(
            {
                "split": split_name,
                "topk": int(k),
                "rows": int(targets.size),
                "selected_rows": int(k),
                "selected_target_delta_sum": float(np.sum(selected)),
                "selected_target_delta_mean": float(np.mean(selected)),
                "estimated_target_delta_per_row": float(np.sum(selected) / targets.size),
                "selected_positive_count": int(np.sum(selected > 0.0)),
                "selected_negative_count": int(np.sum(selected < 0.0)),
                "selected_positive_rate": float(np.mean(selected > 0.0)),
                "selected_loss_mean": float(np.mean(losses)) if losses.size else 0.0,
                "selected_max_loss": float(np.max(losses)) if losses.size else 0.0,
                "min_prediction": float(np.min(predictions[idx])),
            }
        )
    return rows


def train_delta_model(
    features: np.ndarray,
    targets: np.ndarray,
    split: np.ndarray,
    *,
    hidden_layer_sizes: tuple[int, ...],
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    seed: int,
    device_choice: str,
    loss_name: str,
    feature_mode: str,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    torch = _import_torch()
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = select_device(torch, device_choice)
    feature_mean = features[split == SPLIT_NAME_TO_ID["train"]].mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = features[split == SPLIT_NAME_TO_ID["train"]].std(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale[feature_scale < 1e-6] = 1.0
    x_norm = ((features - feature_mean) / feature_scale).astype(np.float32)
    net = _build_torch_mlp(
        torch,
        input_dim=features.shape[1],
        hidden_layer_sizes=hidden_layer_sizes,
        dropout=dropout,
        output_dim=1,
    ).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_idx = np.where(split == SPLIT_NAME_TO_ID["train"])[0]
    val_idx = np.where(split == SPLIT_NAME_TO_ID["val"])[0]
    if train_idx.size == 0 or val_idx.size == 0:
        raise ValueError("train and val splits must be non-empty")
    rng = np.random.default_rng(seed)

    def split_loss(indices: np.ndarray) -> float:
        if indices.size == 0:
            return 0.0
        net.eval()
        losses: list[float] = []
        with torch.inference_mode():
            for start in range(0, indices.size, batch_size):
                batch_idx = indices[start : start + batch_size]
                x = torch.from_numpy(x_norm[batch_idx]).to(device)
                y = torch.from_numpy(targets[batch_idx]).to(device)
                pred = net(x).squeeze(-1)
                if loss_name == "huber":
                    loss = torch.nn.functional.smooth_l1_loss(pred, y, reduction="mean")
                else:
                    loss = torch.nn.functional.mse_loss(pred, y, reduction="mean")
                losses.append(float(loss.detach().cpu().item()))
        return float(np.mean(losses)) if losses else 0.0

    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    for epoch in range(1, epochs + 1):
        net.train()
        shuffled = train_idx.copy()
        rng.shuffle(shuffled)
        train_losses: list[float] = []
        for start in range(0, shuffled.size, batch_size):
            batch_idx = shuffled[start : start + batch_size]
            x = torch.from_numpy(x_norm[batch_idx]).to(device)
            y = torch.from_numpy(targets[batch_idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = net(x).squeeze(-1)
            if loss_name == "huber":
                loss = torch.nn.functional.smooth_l1_loss(pred, y, reduction="mean")
            else:
                loss = torch.nn.functional.mse_loss(pred, y, reduction="mean")
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
        val_loss = split_loss(val_idx)
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss})
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    preds: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, features.shape[0], batch_size):
            end = min(start + batch_size, features.shape[0])
            x = torch.from_numpy(x_norm[start:end]).to(device)
            preds.append(net(x).squeeze(-1).detach().cpu().numpy().astype(np.float32))
    predictions = np.concatenate(preds) if preds else np.zeros(0, dtype=np.float32)
    payload = {
        "model_kind": "hu_turn2_stage8c_realized_delta_head_mlp",
        "feature_dim": int(features.shape[1]),
        "feature_mode": feature_mode,
        "feature_column_names": feature_column_names(feature_mode),
        "hidden_layer_sizes": list(hidden_layer_sizes),
        "dropout": float(dropout),
        "state_dict": best_state,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "target": "realized_delta",
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "loss": loss_name,
    }
    return payload, predictions, history


def write_summary(path: Path, manifest: dict[str, Any], metrics: list[dict[str, Any]], topk_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# HU T2 Stage8c Realized-Delta Head Smoke",
        "",
        "This is a research smoke for candidate-generator ranking only. It does not approve runtime use, P2 fixed status, T1, or 50k teacher generation.",
        "",
        "## Inputs",
        "",
        f"- rows: `{manifest['trainable_rows']}`",
        f"- target mode: `{manifest['target_mode']}`",
        f"- feature mode: `{manifest['feature_mode']}`",
        f"- split mode: `{manifest['split_mode']}`",
        f"- model: `{manifest['model_output']}`",
        "",
        "## Regression Metrics",
        "",
        "| split | rows | target mean | pred mean | MAE | RMSE | Pearson | Spearman |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        if "mae" not in row:
            continue
        lines.append(
            "| {split} | {rows} | {target_mean:.4f} | {prediction_mean:.4f} | {mae:.4f} | {rmse:.4f} | {pearson:.4f} | {spearman:.4f} |".format(
                **row
            )
        )
    lines.extend(["", "## TopK Ranking Metrics", "", "| split | topK | mean delta | sum delta | positive rate | max loss |", "|---|---:|---:|---:|---:|---:|"])
    for row in topk_rows:
        if row.get("split") not in {"val", "test"} or int(row.get("topk", 0)) not in {3, 5, 10, 20}:
            continue
        lines.append(
            "| {split} | {topk} | {selected_target_delta_mean:.4f} | {selected_target_delta_sum:.4f} | {selected_positive_rate:.3f} | {selected_max_loss:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- delta-head training smoke: `Pass`",
            "- allowed role: `ranking/triage for selecting rows for expensive TopK+confirm or high-MC labeling`",
            "- runtime gate: `No-Go`",
            "- production / P2 fixed: `No-Go`",
            "- 50k teacher: `No-Go`",
            "- T1 training: `No-Go`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.time()
    paths = target_paths(args.collection_dir, args.input_jsonl)
    rows, source_rows = load_rows(paths)
    loaded_rows = len(rows)
    rows, excluded_recommended_use_counts = exclude_recommended_use_rows(rows, args.exclude_recommended_use)
    rows_after_recommended_use_filter = len(rows)
    rows = limit_rows_for_training(
        rows,
        max_rows=args.max_rows,
        mode=args.max_rows_mode,
        seed=args.max_rows_seed,
        target_mode="topk_confirm_fire",
    )
    features, targets, metadata, skipped = materialize_delta_rows(
        rows,
        feature_mode=args.feature_mode,
        target_mode=args.target_mode,
        target_clip=args.target_clip,
    )
    labels_for_split = split_labels_for_regression(targets)
    fixed_val_groups = parse_fixed_groups(args.fixed_val_groups)
    fixed_test_groups = parse_fixed_groups(args.fixed_test_groups)
    if fixed_val_groups or fixed_test_groups:
        split = fixed_group_split(metadata, split_mode=args.split_mode, val_groups=fixed_val_groups, test_groups=fixed_test_groups)
    else:
        split = group_stratified_split(
            metadata,
            labels_for_split,
            seed=args.seed,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            split_mode=args.split_mode,
        )
    payload, predictions, history = train_delta_model(
        features,
        targets,
        split,
        hidden_layer_sizes=parse_hidden_layers(args.hidden_layer_sizes),
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        device_choice=args.device,
        loss_name=args.loss,
        feature_mode=args.feature_mode,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch = _import_torch()
    torch.save(payload, args.model_output)

    prediction_rows: list[dict[str, Any]] = []
    for index, row in enumerate(metadata):
        prediction_rows.append(
            row
            | {
                "split": SPLIT_ID_TO_NAME[int(split[index])],
                "delta_prediction": float(predictions[index]),
            }
        )

    metrics: list[dict[str, Any]] = []
    topk_rows: list[dict[str, Any]] = []
    for split_id, split_name in SPLIT_ID_TO_NAME.items():
        indices = np.where(split == split_id)[0]
        metrics.append(regression_metrics(targets[indices], predictions[indices], split_name=split_name))
        topk_rows.extend(topk_delta_metrics(targets[indices], predictions[indices], split_name=split_name))
    metrics.append(regression_metrics(targets, predictions, split_name="all"))
    topk_rows.extend(topk_delta_metrics(targets, predictions, split_name="all"))

    manifest = {
        "schema": "hu_turn2_stage8c_realized_delta_head_training_smoke_v1",
        "input_paths": [str(path) for path in paths],
        "source_rows": source_rows,
        "loaded_rows": int(loaded_rows),
        "exclude_recommended_use": [str(value) for value in args.exclude_recommended_use],
        "excluded_recommended_use_rows": int(sum(excluded_recommended_use_counts.values())),
        "excluded_recommended_use_counts": dict(excluded_recommended_use_counts),
        "rows_after_recommended_use_filter": int(rows_after_recommended_use_filter),
        "sampled_input_rows": int(len(rows)),
        "skipped_rows": dict(skipped),
        "output_dir": str(args.output_dir),
        "model_output": str(args.model_output),
        "feature_dim": int(features.shape[1]),
        "feature_mode": args.feature_mode,
        "feature_column_names": feature_column_names(args.feature_mode),
        "target_mode": args.target_mode,
        "target_clip": float(args.target_clip),
        "split_mode": args.split_mode,
        "split_counts": dict(Counter(SPLIT_ID_TO_NAME[int(value)] for value in split)),
        "target_group_counts": dict(Counter(str(row.get("delta_target_group", "")) for row in metadata)),
        "use_counts": dict(Counter(str(row.get("recommended_training_use", "")) for row in metadata)),
        "trainable_rows": int(targets.size),
        "positive_rows": int(np.sum(targets > 0.0)),
        "negative_rows": int(np.sum(targets < 0.0)),
        "zero_rows": int(np.sum(targets == 0.0)),
        "target_mean": float(np.mean(targets)),
        "target_std": float(np.std(targets)),
        "best_epoch": int(payload["best_epoch"]),
        "best_val_loss": float(payload["best_val_loss"]),
        "loss": args.loss,
        "hidden_layer_sizes": list(parse_hidden_layers(args.hidden_layer_sizes)),
        "elapsed_seconds": float(time.time() - started),
        "production_p2_fixed": "No-Go",
        "teacher_50k": "No-Go",
        "t1_training": "No-Go",
    }
    write_csv(args.output_dir / "delta_head_metrics.csv", metrics)
    write_csv(args.output_dir / "delta_head_topk_metrics.csv", topk_rows)
    write_csv(args.output_dir / "delta_head_predictions.csv", prediction_rows)
    write_csv(args.output_dir / "delta_head_training_history.csv", history)
    (args.output_dir / "delta_head_training_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary(args.output_dir / "delta_head_training_summary.md", manifest, metrics, topk_rows)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
