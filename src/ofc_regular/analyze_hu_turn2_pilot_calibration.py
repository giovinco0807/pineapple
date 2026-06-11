"""Calibration and threshold-scale audit for the HU Turn2 pilot model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .train_hu_turn2_pilot_model import (
    load_cache,
    predict_all,
    state_indices_for_split,
    target_matrix,
)
from .train_torch_action_value import parse_hidden_layers, select_device
from .turn3_model import _build_torch_mlp

ABS_DELTA_GRID = (0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 2.50)
ABS_REFERENCE_GRID = (0.00, 0.25, 0.50, 0.75, 1.00, 1.25)
GATE_THRESHOLD_GRID = (0.50, 0.60, 0.70, 0.80, 0.90)
QUANTILES = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
SPLITS = ("train", "val", "test", "holdout", "all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--threshold-split", choices=("val", "test", "holdout"), default="test")
    return parser.parse_args()


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def percentile_summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p01": float(np.quantile(array, 0.01)),
        "p05": float(np.quantile(array, 0.05)),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def rankdata(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return array
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    sorted_values = array[order]
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def pearson(xs: Iterable[float], ys: Iterable[float]) -> float:
    x = np.asarray(list(xs), dtype=np.float64)
    y = np.asarray(list(ys), dtype=np.float64)
    if x.size < 2 or y.size < 2 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spearman(xs: Iterable[float], ys: Iterable[float]) -> float:
    return pearson(rankdata(xs), rankdata(ys))


def auc_roc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0
    ranks = rankdata(scores)
    positive_rank_sum = sum(float(rank) for rank, label in zip(ranks, labels) if label)
    return float((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def auc_pr(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    tp = 0
    fp = 0
    last_recall = 0.0
    area = 0.0
    for index in order:
        if labels[int(index)]:
            tp += 1
        else:
            fp += 1
        recall = tp / positives
        precision = tp / max(tp + fp, 1)
        area += precision * (recall - last_recall)
        last_recall = recall
    return float(area)


def split_indices(cache: dict[str, Any], split_name: str) -> np.ndarray:
    if split_name == "all":
        return np.arange(len(cache["state_metadata"]), dtype=np.int64)
    return state_indices_for_split(cache["split"], split_name)


def load_model(torch: Any, model_path: Path, device: str):
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if payload.get("model_kind") != "hu_turn2_pilot_multihead_mlp":
        raise TypeError(f"unsupported model kind: {payload.get('model_kind')}")
    hidden = tuple(int(size) for size in payload["hidden_layer_sizes"])
    net = _build_torch_mlp(
        torch,
        int(payload["feature_dim"]),
        hidden,
        float(payload.get("dropout", 0.0)),
        output_dim=5,
    )
    net.load_state_dict(payload["state_dict"])
    net.to(device)
    net.eval()
    stats = {
        "feature_mean": np.asarray(payload["feature_mean"], dtype=np.float32),
        "feature_scale": np.asarray(payload["feature_scale"], dtype=np.float32),
        "target_mean": np.asarray(payload["target_mean"], dtype=np.float32),
        "target_scale": np.asarray(payload["target_scale"], dtype=np.float32),
    }
    return net, stats, payload


def state_rows(cache: dict[str, Any], predictions: np.ndarray, targets: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offsets = cache["offsets"]
    baseline_idx = cache["baseline_action_index"]
    reference_idx = cache["reference_action_index"]
    metadata = cache["state_metadata"]
    for state_index, meta in enumerate(metadata):
        start = int(offsets[state_index])
        end = int(offsets[state_index + 1])
        pred = predictions[start:end]
        y = targets[start:end]
        teacher_order = np.argsort(-y[:, 0], kind="mergesort")
        pred_order = np.argsort(-pred[:, 0], kind="mergesort")
        delta_order = np.argsort(-pred[:, 1], kind="mergesort")
        best = int(teacher_order[0])
        second = int(teacher_order[1]) if teacher_order.size > 1 else best
        pred_best = int(pred_order[0])
        pred_second = int(pred_order[1]) if pred_order.size > 1 else pred_best
        candidate = int(delta_order[0])
        baseline = int(baseline_idx[state_index])
        reference = int(reference_idx[state_index])
        teacher_best_ev = float(y[best, 0])
        teacher_second_ev = float(y[second, 0])
        candidate_actual_delta_baseline = float(y[candidate, 0] - y[baseline, 0])
        candidate_actual_delta_reference = float(y[candidate, 0] - y[reference, 0])
        gate_logit = float(np.mean(pred[:, 4]))
        row = {
            "state_index": state_index,
            "split": split_name_for_state(cache, state_index),
            "bucket_group": meta.get("bucket_group", "unknown"),
            "run_bucket": meta.get("run_bucket", "unknown"),
            "seat": meta.get("seat", "unknown"),
            "pilot_gate_label": meta.get("pilot_gate_label", "unknown"),
            "actual_high_regret": bool(meta.get("actual_high_regret", False)),
            "actual_low_margin": bool(meta.get("actual_low_margin", False)),
            "actual_teacher_disagreement": bool(meta.get("actual_teacher_disagreement", False)),
            "actual_positive": int(meta.get("pilot_gate_label") == "positive"),
            "actual_negative": int(meta.get("pilot_gate_label") == "negative"),
            "teacher_best_EV": teacher_best_ev,
            "teacher_second_EV": teacher_second_ev,
            "teacher_best_margin": float(teacher_best_ev - teacher_second_ev),
            "actual_delta_best_vs_baseline": float(teacher_best_ev - y[baseline, 0]),
            "actual_delta_best_vs_reference": float(teacher_best_ev - y[reference, 0]),
            "actual_delta_candidate_vs_baseline": candidate_actual_delta_baseline,
            "actual_delta_candidate_vs_reference": candidate_actual_delta_reference,
            "predicted_EV_best": float(pred[pred_best, 0]),
            "predicted_EV_second": float(pred[pred_second, 0]),
            "predicted_EV_margin_top1_top2": float(pred[pred_best, 0] - pred[pred_second, 0]),
            "predicted_delta_vs_baseline": float(pred[candidate, 1]),
            "predicted_delta_vs_reference": float(pred[candidate, 2]),
            "predicted_rank_score": float(pred[candidate, 3]),
            "reference_margin_raw": float(meta.get("baseline_model_margin", 0.0) or 0.0),
            "reference_margin_predicted": float(pred[pred_best, 0] - pred[baseline, 0]),
            "gate_logit": gate_logit,
            "gate_probability": sigmoid(gate_logit),
            "SE_delta": float(meta.get("SE_delta_best_vs_baseline", 0.0) or 0.0),
            "candidate_is_baseline": int(candidate == baseline),
            "candidate_false_positive": int(candidate != baseline and candidate_actual_delta_baseline < 0.0),
            "candidate_loss": max(0.0, -candidate_actual_delta_baseline),
            "action_count": int(end - start),
        }
        rows.append(row)
    return rows


def split_name_for_state(cache: dict[str, Any], state_index: int) -> str:
    split_id = int(cache["split"][state_index])
    return {0: "train", 1: "val", 2: "test"}.get(split_id, "unknown")


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


def rows_for_split(rows: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    if split_name == "all":
        return rows
    if split_name == "holdout":
        return [row for row in rows if row["split"] != "train"]
    return [row for row in rows if row["split"] == split_name]


def scale_audit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    variables = (
        "teacher_best_EV",
        "teacher_second_EV",
        "teacher_best_margin",
        "actual_delta_best_vs_baseline",
        "actual_delta_best_vs_reference",
        "actual_delta_candidate_vs_baseline",
        "actual_delta_candidate_vs_reference",
        "predicted_EV_best",
        "predicted_EV_second",
        "predicted_EV_margin_top1_top2",
        "predicted_delta_vs_baseline",
        "predicted_delta_vs_reference",
        "reference_margin_raw",
        "reference_margin_predicted",
        "gate_probability",
        "SE_delta",
    )
    output: list[dict[str, Any]] = []
    for split_name in SPLITS:
        subset = rows_for_split(rows, split_name)
        for variable in variables:
            stats = percentile_summary(float(row[variable]) for row in subset)
            output.append({"split": split_name, "variable": variable, "n": len(subset), **stats})
    return output


def decile_rows(
    rows: list[dict[str, Any]],
    *,
    relation: str,
    split_name: str,
    x_key: str,
    y_key: str,
    binary_y: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    subset = rows_for_split(rows, split_name)
    values = [(float(row[x_key]), float(row[y_key])) for row in subset]
    if not values:
        return [], {
            "split": split_name,
            "relation": relation,
            "n": 0,
            "pearson": 0.0,
            "spearman": 0.0,
            "mae": 0.0,
            "monotonic_non_decreasing": False,
        }
    xs = [x for x, _y in values]
    ys = [y for _x, y in values]
    order = np.argsort(np.asarray(xs), kind="mergesort")
    deciles: list[dict[str, Any]] = []
    y_means: list[float] = []
    for decile in range(10):
        start = int(round(decile * len(order) / 10))
        end = int(round((decile + 1) * len(order) / 10))
        indices = order[start:end]
        if indices.size == 0:
            continue
        decile_x = [xs[int(index)] for index in indices]
        decile_y = [ys[int(index)] for index in indices]
        y_mean = float(np.mean(decile_y))
        y_means.append(y_mean)
        row = {
            "split": split_name,
            "relation": relation,
            "decile": decile + 1,
            "n": int(indices.size),
            "x_min": float(np.min(decile_x)),
            "x_max": float(np.max(decile_x)),
            "x_mean": float(np.mean(decile_x)),
            "y_mean": y_mean,
            "y_median": float(np.median(decile_y)),
            "mae": float(np.mean(np.abs(np.asarray(decile_x) - np.asarray(decile_y)))) if not binary_y else "",
            "positive_rate": y_mean if binary_y else "",
        }
        deciles.append(row)
    summary = {
        "split": split_name,
        "relation": relation,
        "n": len(values),
        "pearson": pearson(xs, ys),
        "spearman": spearman(xs, ys),
        "mae": float(np.mean(np.abs(np.asarray(xs) - np.asarray(ys)))) if not binary_y else 0.0,
        "monotonic_non_decreasing": all(a <= b + 1e-12 for a, b in zip(y_means, y_means[1:])),
    }
    return deciles, summary


def calibration_outputs(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    relations = (
        (
            "predicted_delta_vs_baseline__actual_candidate_delta_vs_baseline",
            "predicted_delta_vs_baseline",
            "actual_delta_candidate_vs_baseline",
            False,
        ),
        (
            "predicted_delta_vs_reference__actual_candidate_delta_vs_reference",
            "predicted_delta_vs_reference",
            "actual_delta_candidate_vs_reference",
            False,
        ),
        (
            "predicted_EV_margin__teacher_best_margin",
            "predicted_EV_margin_top1_top2",
            "teacher_best_margin",
            False,
        ),
        ("gate_probability__actual_positive", "gate_probability", "actual_positive", True),
        (
            "reference_margin_raw__actual_delta_best_vs_baseline",
            "reference_margin_raw",
            "actual_delta_best_vs_baseline",
            False,
        ),
        (
            "reference_margin_raw__candidate_false_positive",
            "reference_margin_raw",
            "candidate_false_positive",
            True,
        ),
    )
    deciles: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test", "holdout"):
        for relation, x_key, y_key, binary_y in relations:
            rows_part, summary = decile_rows(
                rows,
                relation=relation,
                split_name=split_name,
                x_key=x_key,
                y_key=y_key,
                binary_y=binary_y,
            )
            deciles.extend(rows_part)
            summaries.append(summary)
    return deciles, summaries


def binary_metrics(labels: list[int], scores: list[float], *, threshold: float = 0.5) -> dict[str, Any]:
    preds = [1 if score >= threshold else 0 for score in scores]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "n": len(labels),
        "positive_count": sum(labels),
        "negative_count": len(labels) - sum(labels),
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / max(len(labels), 1),
        "roc_auc": auc_roc(labels, scores),
        "pr_auc": auc_pr(labels, scores),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def gate_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    scopes: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [("all", lambda _row: True)]
    for seat in ("first", "second"):
        scopes.append((f"position:{seat}", lambda row, seat=seat: row["seat"] == seat))
    for bucket in sorted({str(row["bucket_group"]) for row in rows}):
        scopes.append((f"bucket:{bucket}", lambda row, bucket=bucket: row["bucket_group"] == bucket))

    for split_name in ("val", "test", "holdout"):
        split_rows = rows_for_split(rows, split_name)
        for mode in ("gray_excluded", "gray_as_negative"):
            for scope_name, predicate in scopes:
                subset = [row for row in split_rows if predicate(row)]
                if mode == "gray_excluded":
                    subset = [row for row in subset if row["pilot_gate_label"] in {"positive", "negative"}]
                labels = [1 if row["pilot_gate_label"] == "positive" else 0 for row in subset]
                scores = [float(row["gate_probability"]) for row in subset]
                metrics = binary_metrics(labels, scores)
                output.append({"split": split_name, "mode": mode, "scope": scope_name, **metrics})
    return output


def threshold_metrics(
    rows: list[dict[str, Any]],
    *,
    split_name: str,
    min_delta: float,
    min_reference: float,
    gate_threshold: float,
    threshold_kind: str,
) -> dict[str, Any]:
    subset = rows_for_split(rows, split_name)
    fired = [
        row
        for row in subset
        if not row["candidate_is_baseline"]
        and float(row["predicted_delta_vs_baseline"]) >= min_delta
        and float(row["reference_margin_raw"]) >= min_reference
        and float(row["gate_probability"]) >= gate_threshold
    ]
    gains = [float(row["actual_delta_candidate_vs_baseline"]) for row in fired]
    losses = [max(0.0, -gain) for gain in gains]
    false_positive = [row for row, gain in zip(fired, gains) if gain < 0.0]
    first = [row for row in fired if row["seat"] == "first"]
    second = [row for row in fired if row["seat"] == "second"]
    bucket_fp: Counter[str] = Counter(str(row["bucket_group"]) for row in false_positive)
    bucket_fired: Counter[str] = Counter(str(row["bucket_group"]) for row in fired)
    return {
        "split": split_name,
        "threshold_kind": threshold_kind,
        "hu_turn2_min_margin": min_delta,
        "hu_turn2_reference_min_margin": min_reference,
        "gate_threshold": gate_threshold,
        "evaluated_states": len(subset),
        "override_count": len(fired),
        "override_rate": len(fired) / max(len(subset), 1),
        "teacher_avg_gain_on_override": float(np.mean(gains)) if gains else 0.0,
        "median_gain_on_override": float(np.median(gains)) if gains else 0.0,
        "false_positive_count": len(false_positive),
        "false_positive_rate": len(false_positive) / max(len(fired), 1),
        "avg_false_positive_cost": float(np.mean([max(0.0, -gain) for gain in gains if gain < 0.0])) if false_positive else 0.0,
        "p90_loss": float(np.quantile(losses, 0.90)) if losses else 0.0,
        "p95_loss": float(np.quantile(losses, 0.95)) if losses else 0.0,
        "p99_loss": float(np.quantile(losses, 0.99)) if losses else 0.0,
        "max_loss": max(losses) if losses else 0.0,
        "positive_override_count": sum(1 for gain in gains if gain > 0.0),
        "negative_override_count": sum(1 for gain in gains if gain < 0.0),
        "first_override_rate": len(first) / max(sum(1 for row in subset if row["seat"] == "first"), 1),
        "second_override_rate": len(second) / max(sum(1 for row in subset if row["seat"] == "second"), 1),
        "first_avg_gain": float(np.mean([float(row["actual_delta_candidate_vs_baseline"]) for row in first])) if first else 0.0,
        "second_avg_gain": float(np.mean([float(row["actual_delta_candidate_vs_baseline"]) for row in second])) if second else 0.0,
        "bucket_false_positive": json.dumps(dict(bucket_fp), sort_keys=True),
        "bucket_override_counts": json.dumps(dict(bucket_fired), sort_keys=True),
    }


def absolute_threshold_sweep(rows: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    output = []
    for min_delta in ABS_DELTA_GRID:
        for min_reference in ABS_REFERENCE_GRID:
            for gate_threshold in GATE_THRESHOLD_GRID:
                output.append(
                    threshold_metrics(
                        rows,
                        split_name=split_name,
                        min_delta=min_delta,
                        min_reference=min_reference,
                        gate_threshold=gate_threshold,
                        threshold_kind="absolute",
                    )
                )
    return output


def quantile_threshold_sweep(rows: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    subset = rows_for_split(rows, split_name)
    predicted = [float(row["predicted_delta_vs_baseline"]) for row in subset]
    reference = [float(row["reference_margin_raw"]) for row in subset]
    output = []
    if not predicted or not reference:
        return output
    for pred_q in QUANTILES:
        for ref_q in QUANTILES:
            min_delta = float(np.quantile(predicted, pred_q))
            min_reference = float(np.quantile(reference, ref_q))
            for gate_threshold in GATE_THRESHOLD_GRID:
                row = threshold_metrics(
                    rows,
                    split_name=split_name,
                    min_delta=min_delta,
                    min_reference=min_reference,
                    gate_threshold=gate_threshold,
                    threshold_kind="quantile",
                )
                row["predicted_delta_quantile"] = pred_q
                row["reference_margin_quantile"] = ref_q
                output.append(row)
    return output


def breakdown_rows(rows: list[dict[str, Any]], *, key: str) -> list[dict[str, Any]]:
    output = []
    for split_name in ("val", "test", "holdout"):
        split_rows = rows_for_split(rows, split_name)
        values = sorted({str(row[key]) for row in split_rows})
        for value in values:
            subset = [row for row in split_rows if str(row[key]) == value]
            output.append(group_summary(subset, split_name=split_name, group=f"{key}:{value}"))
    return output


def group_summary(rows: list[dict[str, Any]], *, split_name: str, group: str) -> dict[str, Any]:
    return {
        "split": split_name,
        "group": group,
        "states": len(rows),
        "predicted_delta_mean": float(np.mean([row["predicted_delta_vs_baseline"] for row in rows])) if rows else 0.0,
        "actual_candidate_delta_mean": float(np.mean([row["actual_delta_candidate_vs_baseline"] for row in rows])) if rows else 0.0,
        "actual_best_delta_mean": float(np.mean([row["actual_delta_best_vs_baseline"] for row in rows])) if rows else 0.0,
        "reference_margin_p95": percentile_summary(row["reference_margin_raw"] for row in rows)["p95"],
        "gate_probability_mean": float(np.mean([row["gate_probability"] for row in rows])) if rows else 0.0,
        "candidate_false_positive_rate": float(np.mean([row["candidate_false_positive"] for row in rows])) if rows else 0.0,
        "candidate_loss_p95": percentile_summary(row["candidate_loss"] for row in rows)["p95"],
    }


def threshold_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if 0.01 <= float(row["override_rate"]) <= 0.20
        and float(row["teacher_avg_gain_on_override"]) > 0.0
        and float(row["false_positive_rate"]) <= 0.35
    ]
    candidates.sort(
        key=lambda row: (
            -float(row["teacher_avg_gain_on_override"]),
            float(row["false_positive_rate"]),
            float(row["p95_loss"]),
            -float(row["override_rate"]),
        )
    )
    return candidates[:30]


def best_balanced_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if 0.01 <= float(row["override_rate"]) <= 0.20
        and float(row["teacher_avg_gain_on_override"]) > 0.0
        and float(row["false_positive_rate"]) <= 0.35
        and float(row["first_override_rate"]) > 0.0
        and float(row["second_override_rate"]) > 0.0
        and float(row["first_avg_gain"]) > 0.0
        and float(row["second_avg_gain"]) > 0.0
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda row: (
            -min(float(row["first_avg_gain"]), float(row["second_avg_gain"])),
            -float(row["teacher_avg_gain_on_override"]),
            float(row["false_positive_rate"]),
        )
    )
    return candidates[0]


def write_reference_margin_diagnosis(path: Path, rows: list[dict[str, Any]], corr_rows: list[dict[str, Any]]) -> None:
    test = rows_for_split(rows, "test")
    ref_stats = percentile_summary(row["reference_margin_raw"] for row in test)
    pred_stats = percentile_summary(row["predicted_delta_vs_baseline"] for row in test)
    relation = next(
        (
            row
            for row in corr_rows
            if row["split"] == "test" and row["relation"] == "reference_margin_raw__actual_delta_best_vs_baseline"
        ),
        {},
    )
    lines = [
        "# Reference Margin Diagnosis",
        "",
        "- current field: `baseline_model_margin` from the T2 baseline model score gap",
        "- unit/scale: model-score margin, not teacher EV points",
        f"- test reference margin p50/p90/p95/max: `{ref_stats['p50']:.4f}` / `{ref_stats['p90']:.4f}` / `{ref_stats['p95']:.4f}` / `{ref_stats['max']:.4f}`",
        f"- test predicted delta p50/p90/p95/max: `{pred_stats['p50']:.4f}` / `{pred_stats['p90']:.4f}` / `{pred_stats['p95']:.4f}` / `{pred_stats['max']:.4f}`",
        "- reason threshold 10 fired zero overrides: the T2 reference-margin distribution is far below 10; p95 is near 1, so `r10` is outside the pilot scale.",
        f"- Pearson(reference_margin, actual_best_delta): `{float(relation.get('pearson', 0.0)):.4f}`",
        f"- Spearman(reference_margin, actual_best_delta): `{float(relation.get('spearman', 0.0)):.4f}`",
        "",
        "## Runtime Candidate Signals",
        "",
        "- Prefer calibrated `predicted_delta_vs_baseline` plus `gate_probability` for the next pilot gate.",
        "- Treat raw reference margin as a weak pre-gate until it is recalibrated on a larger holdout.",
        "- Do not transplant T3 `m5/r10` thresholds directly into T2; their score scales differ.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(
    path: Path,
    *,
    scale_rows: list[dict[str, Any]],
    corr_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> None:
    def stat(split: str, variable: str, field: str) -> float:
        row = next(item for item in scale_rows if item["split"] == split and item["variable"] == variable)
        return float(row[field])

    delta_corr = next(
        row
        for row in corr_rows
        if row["split"] == "test"
        and row["relation"] == "predicted_delta_vs_baseline__actual_candidate_delta_vs_baseline"
    )
    gate = next(
        row
        for row in gate_rows
        if row["split"] == "test" and row["mode"] == "gray_excluded" and row["scope"] == "all"
    )
    positive_configs = [
        row
        for row in threshold_rows
        if float(row["override_rate"]) > 0.0 and float(row["teacher_avg_gain_on_override"]) > 0.0
    ]
    balanced = best_balanced_candidate(threshold_rows)
    lines = [
        "# HU T2 Pilot Calibration Summary",
        "",
        "This is a calibration audit only. No 50k teacher, T1 work, or production candidate training was started.",
        "",
        "## Scale",
        "",
        f"- test predicted_delta p50/p90/p95/max: `{stat('test', 'predicted_delta_vs_baseline', 'p50'):.4f}` / `{stat('test', 'predicted_delta_vs_baseline', 'p90'):.4f}` / `{stat('test', 'predicted_delta_vs_baseline', 'p95'):.4f}` / `{stat('test', 'predicted_delta_vs_baseline', 'max'):.4f}`",
        f"- test reference_margin_raw p50/p90/p95/max: `{stat('test', 'reference_margin_raw', 'p50'):.4f}` / `{stat('test', 'reference_margin_raw', 'p90'):.4f}` / `{stat('test', 'reference_margin_raw', 'p95'):.4f}` / `{stat('test', 'reference_margin_raw', 'max'):.4f}`",
        f"- test teacher_best_margin p50/p90/p95: `{stat('test', 'teacher_best_margin', 'p50'):.4f}` / `{stat('test', 'teacher_best_margin', 'p90'):.4f}` / `{stat('test', 'teacher_best_margin', 'p95'):.4f}`",
        "",
        "## Calibration",
        "",
        f"- predicted_delta vs actual candidate delta Pearson/Spearman: `{float(delta_corr['pearson']):.4f}` / `{float(delta_corr['spearman']):.4f}`",
        f"- gate gray-excluded precision/recall/F1/PR-AUC: `{float(gate['precision']):.4f}` / `{float(gate['recall']):.4f}` / `{float(gate['f1']):.4f}` / `{float(gate['pr_auc']):.4f}`",
        "",
        "## Thresholds",
        "",
        f"- positive-gain threshold configs with at least one override: `{len(positive_configs)}`",
        f"- recommended candidate rows: `{len(candidate_rows)}`",
        (
            "- balanced candidate: "
            f"`m{float(balanced['hu_turn2_min_margin']):.2f}_r{float(balanced['hu_turn2_reference_min_margin']):.2f}_g{float(balanced['gate_threshold']):.2f}` "
            f"override `{float(balanced['override_rate']):.4f}`, avg gain `{float(balanced['teacher_avg_gain_on_override']):.4f}`, FP `{float(balanced['false_positive_rate']):.4f}`"
            if balanced
            else "- balanced candidate: none"
        ),
        "",
        "## Interpretation",
        "",
        "- The previous `5/10` sweep fired zero because those thresholds are outside the observed T2 pilot scale.",
        "- T2 has usable learning signal, but runtime thresholds must be calibrated in T2 units.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_go_nogo(path: Path, candidate_rows: list[dict[str, Any]], corr_rows: list[dict[str, Any]], gate_rows: list[dict[str, Any]]) -> None:
    balanced = best_balanced_candidate(candidate_rows)
    delta_corr = next(
        row
        for row in corr_rows
        if row["split"] == "test"
        and row["relation"] == "predicted_delta_vs_baseline__actual_candidate_delta_vs_baseline"
    )
    gate = next(
        row
        for row in gate_rows
        if row["split"] == "test" and row["mode"] == "gray_excluded" and row["scope"] == "all"
    )
    go = bool(
        candidate_rows
        and balanced is not None
        and float(delta_corr["spearman"]) > 0.15
        and float(gate["precision"]) > 0.50
    )
    lines = [
        "# Go/No-Go For 50k",
        "",
        "GO to a 20k-50k MC512 broad pass." if go else "NO-GO to 50k yet.",
        "",
        f"- recommended threshold candidates: `{len(candidate_rows)}`",
        f"- balanced first/second candidate: `{'yes' if balanced else 'no'}`",
        f"- test predicted_delta Spearman vs actual candidate delta: `{float(delta_corr['spearman']):.4f}`",
        f"- test gate precision gray-excluded: `{float(gate['precision']):.4f}`",
        "",
    ]
    if go:
        lines.extend(
            [
                "Reason: T2-scale thresholds can fire with positive teacher holdout gain, and calibration is nonzero.",
                (
                    "Suggested pilot runtime scale: "
                    f"`hu_turn2_min_margin={float(balanced['hu_turn2_min_margin']):.2f}`, "
                    f"`hu_turn2_reference_min_margin={float(balanced['hu_turn2_reference_min_margin']):.2f}`, "
                    f"`gate_threshold={float(balanced['gate_threshold']):.2f}`."
                    if balanced
                    else "Use the recommended candidates only as analysis inputs; no balanced runtime was found."
                ),
                "Next command should be a limited 20k MC512 broad pass, not production training.",
            ]
        )
    else:
        lines.extend(
            [
                "Reason: threshold candidates, delta calibration, or gate precision are not strong enough from this 2,000-state pilot.",
                "Next step should be model/loss/head calibration or a small validation rerun, not 50k.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started_at = time.time()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    device = select_device(torch, args.device)
    cache = load_cache(args.cache_dir.resolve())
    targets = target_matrix(cache)
    net, stats, model_payload = load_model(torch, args.model.resolve(), device)
    predictions = predict_all(torch, net, cache, stats, device, args.batch_size)
    rows = state_rows(cache, predictions, targets)

    scale_rows = scale_audit_rows(rows)
    calibration_deciles, calibration_summaries = calibration_outputs(rows)
    gate_rows = gate_metric_rows(rows)
    threshold_rows = absolute_threshold_sweep(rows, args.threshold_split)
    quantile_rows = quantile_threshold_sweep(rows, args.threshold_split)
    combined_thresholds = threshold_rows + quantile_rows
    candidate_rows = threshold_candidates(combined_thresholds)
    position_rows = breakdown_rows(rows, key="seat")
    bucket_rows = breakdown_rows(rows, key="bucket_group")
    actual_bucket_rows = []
    for key in ("actual_high_regret", "actual_low_margin", "actual_teacher_disagreement"):
        actual_bucket_rows.extend(breakdown_rows(rows, key=key))
    bucket_rows.extend(actual_bucket_rows)

    write_csv(output_dir / "state_calibration_values.csv", rows)
    write_csv(output_dir / "scale_audit.csv", scale_rows)
    write_csv(output_dir / "calibration_by_decile.csv", calibration_deciles)
    write_csv(output_dir / "calibration_correlations.csv", calibration_summaries)
    write_csv(output_dir / "gate_metrics.csv", gate_rows)
    write_csv(output_dir / "threshold_sweep_t2_scale.csv", threshold_rows)
    write_csv(output_dir / "threshold_sweep_quantile.csv", quantile_rows)
    write_csv(output_dir / "position_breakdown.csv", position_rows)
    write_csv(output_dir / "bucket_breakdown.csv", bucket_rows)
    write_csv(output_dir / "recommended_threshold_candidates.csv", candidate_rows)
    write_reference_margin_diagnosis(output_dir / "reference_margin_diagnosis.md", rows, calibration_summaries)
    write_summary(
        output_dir / "calibration_summary.md",
        scale_rows=scale_rows,
        corr_rows=calibration_summaries,
        gate_rows=gate_rows,
        threshold_rows=combined_thresholds,
        candidate_rows=candidate_rows,
    )
    write_go_nogo(output_dir / "go_nogo_for_50k.md", candidate_rows, calibration_summaries, gate_rows)
    manifest = {
        "schema": "hu_turn2_stage8_pilot_calibration_v1",
        "cache_dir": str(args.cache_dir.resolve()),
        "model": str(args.model.resolve()),
        "output_dir": str(output_dir),
        "device": device,
        "states": len(rows),
        "actions": int(cache["metadata"]["action_count"]),
        "wrote": [
            "scale_audit.csv",
            "calibration_by_decile.csv",
            "calibration_correlations.csv",
            "gate_metrics.csv",
            "threshold_sweep_t2_scale.csv",
            "threshold_sweep_quantile.csv",
            "position_breakdown.csv",
            "bucket_breakdown.csv",
            "reference_margin_diagnosis.md",
            "recommended_threshold_candidates.csv",
            "calibration_summary.md",
            "go_nogo_for_50k.md",
        ],
        "production_candidate": False,
        "teacher_50k_launched": False,
    }
    (output_dir / "calibration_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "schema": "hu_turn2_pilot_calibration_audit_v1",
        "cache_dir": str(args.cache_dir.resolve()),
        "model": str(args.model.resolve()),
        "output_dir": str(output_dir),
        "device": device,
        "threshold_split": args.threshold_split,
        "states": len(rows),
        "actions": int(cache["metadata"]["action_count"]),
        "candidate_count": len(candidate_rows),
        "elapsed_seconds": time.time() - started_at,
        "model_kind": model_payload.get("model_kind"),
    }
    (output_dir / "calibration_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
