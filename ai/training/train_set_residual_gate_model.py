"""Train a learned confidence gate for set-residual T2 action values.

This is deliberately small and diagnostic.  It learns from per-decision group
features that are available at inference time, predicts the EV delta of using a
set-residual challenger instead of the base ensemble, then chooses a threshold
on a separate validation split.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.models.action_value_set_reranker import ActionValueSetReranker
from ai.training.evaluate_action_value_reranker import parse_topks
from ai.training.evaluate_action_value_set_reranker import summarize_groups
from ai.training.evaluate_set_residual_gate import load_dataset_predictions, parse_named_path


LABEL_KEYS = {
    "group_id",
    "base_regret",
    "challenger_regret",
    "challenger_minus_base_true",
    "challenger_improves",
    "challenger_hurts",
}


class GateMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def feature_names_from_rows(rows: list[dict[str, float]]) -> list[str]:
    if not rows:
        return []
    return [key for key in rows[0].keys() if key not in LABEL_KEYS]


def matrix_from_rows(rows: list[dict[str, float]], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(
        [[float(row.get(name, 0.0)) for name in feature_names] for row in rows],
        dtype=np.float32,
    )
    y = np.asarray([float(row["challenger_minus_base_true"]) for row in rows], dtype=np.float32)
    return x, y


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / np.maximum(std, 1e-6)


def build_prediction_scores(dataset: dict, selected_groups: np.ndarray) -> np.ndarray:
    pred = np.asarray(dataset["base"], dtype=np.float32).copy()
    for group_id, selected in enumerate(selected_groups):
        if not bool(selected):
            continue
        start, end = dataset["bounds"][group_id]
        pred[start:end] = dataset["challenger"][start:end]
    return pred


def threshold_candidates(pred: np.ndarray, max_thresholds: int) -> list[float]:
    values = np.asarray(pred[np.isfinite(pred)], dtype=np.float64)
    if len(values) == 0:
        return [0.0]
    lo = float(values.min())
    hi = float(values.max())
    thresholds = [hi + 1e-6, lo - 1e-6]
    unique = np.unique(values)
    if len(unique) <= max_thresholds:
        thresholds.extend(float(x) for x in unique)
    else:
        qs = np.linspace(0.02, 0.98, max_thresholds)
        thresholds.extend(float(x) for x in np.unique(np.quantile(values, qs)))
    return sorted(set(thresholds))


def evaluate_thresholds(
    dataset: dict,
    pred_delta: np.ndarray,
    topks: list[int],
    max_thresholds: int,
    min_switches: int,
) -> list[dict]:
    rows: list[dict] = []
    for threshold in threshold_candidates(pred_delta, max_thresholds=max_thresholds):
        selected = np.asarray(pred_delta >= float(threshold), dtype=bool)
        pred_scores = build_prediction_scores(dataset, selected)
        metrics, _ = summarize_groups(dataset["scores"], pred_scores, dataset["bounds"], topks)
        switch_groups = int(selected.sum())
        enough = switch_groups >= int(min_switches) or switch_groups in {0, len(selected)}
        rows.append(
            {
                "threshold": float(threshold),
                "groups": int(metrics["groups"]),
                "switch_groups": switch_groups,
                "switch_fraction": float(selected.mean()) if len(selected) else 0.0,
                "enough_switches": bool(enough),
                "metrics": metrics,
            }
        )
    rows.sort(
        key=lambda row: (
            1 if row["enough_switches"] else 0,
            float(row["metrics"].get("group_top1", 0.0)),
            -float(row["metrics"].get("group_top1_regret", 0.0)),
            float(row["metrics"].get("group_top3", 0.0)),
            float(row["metrics"].get("group_top10", 0.0)),
            -abs(float(row["switch_fraction"]) - 0.25),
        ),
        reverse=True,
    )
    return rows


def evaluate_fixed_threshold(dataset: dict, pred_delta: np.ndarray, threshold: float, topks: list[int]) -> dict:
    selected = np.asarray(pred_delta >= float(threshold), dtype=bool)
    pred_scores = build_prediction_scores(dataset, selected)
    metrics, _ = summarize_groups(dataset["scores"], pred_scores, dataset["bounds"], topks)
    return {
        "threshold": float(threshold),
        "groups": int(metrics["groups"]),
        "switch_groups": int(selected.sum()),
        "switch_fraction": float(selected.mean()) if len(selected) else 0.0,
        "metrics": metrics,
    }


def baseline_or_challenger(dataset: dict, use_challenger: bool, topks: list[int]) -> dict:
    selected = np.full(len(dataset["bounds"]), bool(use_challenger), dtype=bool)
    pred_scores = build_prediction_scores(dataset, selected)
    metrics, _ = summarize_groups(dataset["scores"], pred_scores, dataset["bounds"], topks)
    return {
        "groups": int(metrics["groups"]),
        "switch_groups": int(selected.sum()),
        "switch_fraction": float(selected.mean()) if len(selected) else 0.0,
        "metrics": metrics,
    }


@torch.no_grad()
def predict_delta(
    model: GateMLP,
    x: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: float,
    y_std: float,
    device: torch.device,
    batch_size: int,
    target: str = "delta",
) -> np.ndarray:
    model.eval()
    x_norm = normalize(x, x_mean, x_std)
    out = np.zeros(x_norm.shape[0], dtype=np.float32)
    for start in range(0, len(out), batch_size):
        end = min(len(out), start + batch_size)
        xb = torch.from_numpy(x_norm[start:end].astype(np.float32, copy=True)).to(device)
        pred_t = model(xb)
        if target == "improve":
            out[start:end] = torch.sigmoid(pred_t).detach().cpu().numpy().astype(np.float32)
        else:
            pred = pred_t.detach().cpu().numpy().astype(np.float32)
            out[start:end] = pred * float(y_std) + float(y_mean)
    return out


def aggregate_eval(rows: list[dict]) -> dict:
    groups = sum(int(row["groups"]) for row in rows)
    if groups <= 0:
        return {"groups": 0}
    out = {
        "groups": int(groups),
        "switch_groups": int(sum(int(row["switch_groups"]) for row in rows)),
    }
    out["switch_fraction"] = out["switch_groups"] / groups
    for key in (
        "group_top1",
        "group_top3",
        "group_top5",
        "group_top10",
        "group_top15",
        "group_top20",
        "group_top1_regret",
        "group_top3_rerank_regret",
        "group_top10_rerank_regret",
    ):
        out[key] = sum(float(row["metrics"].get(key, 0.0)) * int(row["groups"]) for row in rows) / groups
    return out


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a learned set-residual confidence gate")
    parser.add_argument("--train-data", required=True, help="name=path")
    parser.add_argument("--val-data", required=True, help="name=path")
    parser.add_argument("--eval-dataset", action="append", default=[], help="name=path")
    parser.add_argument("--checkpoint", required=True, help="Set residual reranker checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--target", choices=["delta", "improve"], default="delta")
    parser.add_argument("--topks", default="1,3,5,10,15,20")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--predict-batch-size", type=int, default=512)
    parser.add_argument("--max-thresholds", type=int, default=120)
    parser.add_argument("--min-switches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    topks = parse_topks(args.topks)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    set_model = ActionValueSetReranker.from_checkpoint(args.checkpoint, map_location=device).to(device)

    train_name, train_path = parse_named_path(args.train_data)
    val_name, val_path = parse_named_path(args.val_data)
    train_ds = load_dataset_predictions(train_name, train_path, set_model, args.gamma, device, args.predict_batch_size, topks)
    val_ds = load_dataset_predictions(val_name, val_path, set_model, args.gamma, device, args.predict_batch_size, topks)
    feature_names = feature_names_from_rows(train_ds["features"])
    x_train, y_train = matrix_from_rows(train_ds["features"], feature_names)
    x_val, y_val = matrix_from_rows(val_ds["features"], feature_names)
    y_train_improve = (y_train > 1e-9).astype(np.float32)
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    y_mean = float(y_train.mean())
    y_std = max(float(y_train.std()), 1e-6)
    x_train_norm = normalize(x_train, x_mean, x_std).astype(np.float32)
    if args.target == "improve":
        y_train_norm = y_train_improve.astype(np.float32)
    else:
        y_train_norm = ((y_train - y_mean) / y_std).astype(np.float32)
    weights = (1.0 + np.minimum(np.abs(y_train), 5.0)).astype(np.float32)

    model = GateMLP(input_dim=len(feature_names), hidden=args.hidden, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_threshold = 0.0
    best_val_row: dict | None = None
    n = int(x_train_norm.shape[0])

    for epoch in range(1, int(args.epochs) + 1):
        order = np.random.permutation(n)
        model.train()
        total_loss = 0.0
        total_weight = 0.0
        for start in range(0, n, args.batch_size):
            idx = order[start : start + args.batch_size]
            xb = torch.from_numpy(x_train_norm[idx]).to(device)
            yb = torch.from_numpy(y_train_norm[idx]).to(device)
            wb = torch.from_numpy(weights[idx]).to(device)
            pred = model(xb)
            if args.target == "improve":
                loss = F.binary_cross_entropy_with_logits(pred, yb, reduction="none")
            else:
                loss = F.smooth_l1_loss(pred, yb, reduction="none")
            loss = (loss * wb).sum() / wb.sum().clamp(min=1e-6)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * float(wb.sum().detach().cpu())
            total_weight += float(wb.sum().detach().cpu())

        should_eval = epoch == 1 or epoch % max(int(args.eval_every), 1) == 0 or epoch == args.epochs
        val_row = None
        if should_eval:
            pred_val = predict_delta(
                model,
                x_val,
                x_mean,
                x_std,
                y_mean,
                y_std,
                device,
                args.predict_batch_size,
                args.target,
            )
            val_rows = evaluate_thresholds(
                val_ds,
                pred_val,
                topks,
                max_thresholds=args.max_thresholds,
                min_switches=args.min_switches,
            )
            val_row = val_rows[0]
            score = (
                float(val_row["metrics"].get("group_top1", 0.0)),
                -float(val_row["metrics"].get("group_top1_regret", 0.0)),
                float(val_row["metrics"].get("group_top3", 0.0)),
                float(val_row["metrics"].get("group_top10", 0.0)),
            )
            best_score = (
                float(best_val_row["metrics"].get("group_top1", 0.0)) if best_val_row else -1.0,
                -float(best_val_row["metrics"].get("group_top1_regret", 0.0)) if best_val_row else -1e9,
                float(best_val_row["metrics"].get("group_top3", 0.0)) if best_val_row else -1.0,
                float(best_val_row["metrics"].get("group_top10", 0.0)) if best_val_row else -1.0,
            )
            if score > best_score:
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                best_threshold = float(val_row["threshold"])
                best_val_row = val_row
        if should_eval and val_row is not None:
            avg_loss = total_loss / max(total_weight, 1e-6)
            print(
                f"epoch {epoch:03d}: loss={avg_loss:.4f} "
                f"val_top1={val_row['metrics']['group_top1']:.1%} "
                f"val_reg1={val_row['metrics']['group_top1_regret']:.3f} "
                f"switch={val_row['switch_groups']}/{val_row['groups']} "
                f"threshold={val_row['threshold']:.4f}"
            )

    model.load_state_dict(best_state)
    eval_outputs = {}
    all_selected: list[dict] = []
    all_baseline: list[dict] = []
    all_challenger: list[dict] = []
    for spec in args.eval_dataset:
        name, path = parse_named_path(spec)
        ds = load_dataset_predictions(name, path, set_model, args.gamma, device, args.predict_batch_size, topks)
        x_eval, _ = matrix_from_rows(ds["features"], feature_names)
        pred_eval = predict_delta(
            model,
            x_eval,
            x_mean,
            x_std,
            y_mean,
            y_std,
            device,
            args.predict_batch_size,
            args.target,
        )
        selected = evaluate_fixed_threshold(ds, pred_eval, best_threshold, topks)
        baseline = baseline_or_challenger(ds, False, topks)
        challenger = baseline_or_challenger(ds, True, topks)
        eval_outputs[name] = {
            "data": str(path),
            "baseline": baseline,
            "challenger": challenger,
            "learned_gate": selected,
        }
        all_selected.append(selected)
        all_baseline.append(baseline)
        all_challenger.append(challenger)

    train_pred = predict_delta(model, x_train, x_mean, x_std, y_mean, y_std, device, args.predict_batch_size, args.target)
    val_pred = predict_delta(model, x_val, x_mean, x_std, y_mean, y_std, device, args.predict_batch_size, args.target)
    summary = {
        "checkpoint": str(args.checkpoint),
        "gamma": float(args.gamma),
        "target": args.target,
        "train_data": str(train_path),
        "val_data": str(val_path),
        "feature_names": feature_names,
        "best_epoch": int(best_epoch),
        "selected_threshold": float(best_threshold),
        "train_baseline": baseline_or_challenger(train_ds, False, topks),
        "train_challenger": baseline_or_challenger(train_ds, True, topks),
        "train_learned_gate": evaluate_fixed_threshold(train_ds, train_pred, best_threshold, topks),
        "val_baseline": baseline_or_challenger(val_ds, False, topks),
        "val_challenger": baseline_or_challenger(val_ds, True, topks),
        "val_learned_gate": evaluate_fixed_threshold(val_ds, val_pred, best_threshold, topks),
        "evals": eval_outputs,
        "eval_aggregate": {
            "baseline": aggregate_eval(all_baseline),
            "challenger": aggregate_eval(all_challenger),
            "learned_gate": aggregate_eval(all_selected),
        },
        "normalization": {
            "x_mean": x_mean.tolist(),
            "x_std": x_std.tolist(),
            "y_mean": y_mean,
            "y_std": y_std,
        },
        "elapsed_seconds": time.time() - started,
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_names": feature_names,
            "normalization": summary["normalization"],
            "threshold": best_threshold,
            "hidden": int(args.hidden),
            "dropout": float(args.dropout),
            "gamma": float(args.gamma),
            "target": args.target,
            "source_checkpoint": str(args.checkpoint),
        },
        out_dir / "gate_model.pt",
    )
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("learned set-residual gate")
    print(f"  best_epoch={best_epoch} threshold={best_threshold:.6f}")
    for label, row in summary["eval_aggregate"].items():
        print(
            f"  eval aggregate {label}: groups={row.get('groups', 0)} "
            f"switch={row.get('switch_groups', 0)} "
            f"top1={row.get('group_top1', 0.0):.1%} "
            f"top3={row.get('group_top3', 0.0):.1%} "
            f"top10={row.get('group_top10', 0.0):.1%} "
            f"reg1={row.get('group_top1_regret', 0.0):.3f}"
        )
    print(f"  summary={out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
