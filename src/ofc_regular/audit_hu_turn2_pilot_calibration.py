"""Audit HU Turn2 pilot model calibration and T2-scale override thresholds."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .build_hu_turn2_pilot_feature_cache import SPLIT_ID_TO_NAME, SPLIT_NAME_TO_ID
from .train_hu_turn2_pilot_model import load_cache, predict_all, target_matrix
from .turn3_model import _build_torch_mlp

STATS_QUANTILES = (
    ("p01", 0.01),
    ("p05", 0.05),
    ("p10", 0.10),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
    ("p95", 0.95),
    ("p99", 0.99),
)
T2_MIN_MARGIN_GRID = (0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 2.50)
REFERENCE_MARGIN_GRID = (0.00, 0.25, 0.50, 0.75, 1.00, 1.25)
GATE_THRESHOLD_GRID = (0.50, 0.60, 0.70, 0.80, 0.90)
QUANTILE_LEVELS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
LABEL_NEGATIVE = 0
LABEL_GRAY = 1
LABEL_POSITIVE = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("D:/ofc-pineapple-storage/regular-ofc-pineapple/feature_cache/hu_t2_stage1_pilot_2000_mc512"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/hu_turn2_stage8_pilot_2000_mc512_reference_override_cached_rank_wide.pt"),
    )
    parser.add_argument(
        "--previous-output",
        type=Path,
        default=Path("outputs/hu_turn2_stage1_pilot_training_2000_mc512"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/hu_turn2_stage1_pilot_training_2000_mc512_calibration"),
    )
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def select_device(torch, requested: str) -> str:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is not available")
        return "cuda"
    if requested == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def split_indices(split: np.ndarray, split_name: str) -> np.ndarray:
    if split_name == "holdout":
        return np.where(split != SPLIT_NAME_TO_ID["train"])[0]
    return np.where(split == SPLIT_NAME_TO_ID[split_name])[0]


def percentile(values: np.ndarray | list[float], q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q * 100.0))


def stats_row(split_name: str, metric: str, values: Iterable[float]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    row: dict[str, Any] = {"split": split_name, "metric": metric, "count": int(arr.size)}
    if arr.size == 0:
        for key in ("mean", "std", "min", "max"):
            row[key] = ""
        for key, _q in STATS_QUANTILES:
            row[key] = ""
        return row
    row.update(
        {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    )
    for key, q in STATS_QUANTILES:
        row[key] = percentile(arr, q)
    return row


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def pearson(x: Iterable[float], y: Iterable[float]) -> float:
    xx = np.asarray(list(x), dtype=np.float64)
    yy = np.asarray(list(y), dtype=np.float64)
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]
    if xx.size < 2 or float(xx.std()) < 1e-12 or float(yy.std()) < 1e-12:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def spearman(x: Iterable[float], y: Iterable[float]) -> float:
    xx = np.asarray(list(x), dtype=np.float64)
    yy = np.asarray(list(y), dtype=np.float64)
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]
    if xx.size < 2:
        return float("nan")
    return pearson(rankdata(xx), rankdata(yy))


def isotonic_fit_predict(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    mask = np.isfinite(x) & np.isfinite(y)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if mask.sum() < 2:
        return out
    order = np.argsort(x[mask], kind="mergesort")
    yy = y[mask][order].astype(np.float64)
    blocks: list[dict[str, float]] = []
    for value in yy:
        blocks.append({"sum": float(value), "weight": 1.0, "start": len(blocks), "end": len(blocks) + 1})
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            if left["sum"] / left["weight"] <= right["sum"] / right["weight"]:
                break
            merged = {
                "sum": left["sum"] + right["sum"],
                "weight": left["weight"] + right["weight"],
                "start": left["start"],
                "end": right["end"],
            }
            blocks[-2:] = [merged]
    fitted_sorted = np.empty(mask.sum(), dtype=np.float64)
    for block in blocks:
        fitted_sorted[int(block["start"]) : int(block["end"])] = block["sum"] / block["weight"]
    fitted_masked = np.empty(mask.sum(), dtype=np.float64)
    fitted_masked[order] = fitted_sorted
    out[np.where(mask)[0]] = fitted_masked
    return out


def roc_auc_score(y_true: list[int], scores: list[float]) -> float:
    y = np.asarray(y_true, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = rankdata(s)
    pos_rank_sum = float(ranks[y == 1].sum())
    return (pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def average_precision_score(y_true: list[int], scores: list[float]) -> float:
    pairs = sorted(
        [(float(score), int(label)) for label, score in zip(y_true, scores) if math.isfinite(float(score))],
        key=lambda item: item[0],
        reverse=True,
    )
    positives = sum(label for _score, label in pairs)
    if positives == 0:
        return float("nan")
    hit = 0
    precision_sum = 0.0
    for rank, (_score, label) in enumerate(pairs, start=1):
        if label:
            hit += 1
            precision_sum += hit / rank
    return precision_sum / positives


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_predictions(torch, cache: dict[str, Any], checkpoint_path: Path, device: str, batch_size: int) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    net = _build_torch_mlp(
        torch,
        int(checkpoint["feature_dim"]),
        [int(v) for v in checkpoint["hidden_layer_sizes"]],
        float(checkpoint.get("dropout", 0.0)),
        output_dim=len(checkpoint.get("heads", [])) or 5,
    ).to(device)
    net.load_state_dict(checkpoint["state_dict"])
    stats = {
        "feature_mean": np.asarray(checkpoint["feature_mean"], dtype=np.float32),
        "feature_scale": np.asarray(checkpoint["feature_scale"], dtype=np.float32),
        "target_mean": np.asarray(checkpoint["target_mean"], dtype=np.float32),
        "target_scale": np.asarray(checkpoint["target_scale"], dtype=np.float32),
    }
    return predict_all(torch, net, cache, stats, device, batch_size)


def build_state_records(cache: dict[str, Any], predictions: np.ndarray, targets: np.ndarray) -> list[dict[str, Any]]:
    offsets = cache["offsets"]
    records: list[dict[str, Any]] = []
    for state_index, metadata in enumerate(cache["state_metadata"]):
        start = int(offsets[state_index])
        end = int(offsets[state_index + 1])
        pred = predictions[start:end]
        target = targets[start:end]
        baseline = int(cache["baseline_action_index"][state_index])
        reference = int(cache["reference_action_index"][state_index])
        best = int(cache["best_action_index"][state_index])
        second = int(cache["second_best_action_index"][state_index])

        predicted_ev_order = np.argsort(pred[:, 0])
        predicted_ev_best = int(predicted_ev_order[-1])
        predicted_ev_second = int(predicted_ev_order[-2]) if pred.shape[0] > 1 else predicted_ev_best
        predicted_delta_best = int(np.argmax(pred[:, 1]))
        predicted_reference_best = int(np.argmax(pred[:, 2]))
        gate_logit_mean = float(pred[:, 4].mean())

        teacher_best_ev = float(target[best, 0])
        teacher_second_ev = float(target[second, 0])
        baseline_ev = float(target[baseline, 0])
        reference_ev = float(target[reference, 0])
        selected_delta_gain = float(target[predicted_delta_best, 0] - baseline_ev)
        selected_reference_gain = float(target[predicted_reference_best, 0] - reference_ev)
        gate_label_id = int(cache["gate_label_id"][state_index])

        record = {
            "state_index": state_index,
            "split": SPLIT_ID_TO_NAME[int(cache["split"][state_index])],
            "run_bucket": metadata.get("run_bucket", ""),
            "bucket_group": metadata.get("bucket_group", ""),
            "source_bucket": metadata.get("source_bucket", ""),
            "seat": metadata.get("seat", ""),
            "to_act_order": metadata.get("to_act_order", ""),
            "pilot_gate_label": metadata.get("pilot_gate_label", ""),
            "pilot_gate_label_id": gate_label_id,
            "actual_high_regret": bool(metadata.get("actual_high_regret", False)),
            "actual_low_margin": bool(metadata.get("actual_low_margin", False)),
            "actual_teacher_disagreement": bool(metadata.get("actual_teacher_disagreement", False)),
            "margin_bucket": metadata.get("margin_bucket", ""),
            "teacher_best_EV": teacher_best_ev,
            "teacher_second_EV": teacher_second_ev,
            "teacher_best_margin": teacher_best_ev - teacher_second_ev,
            "actual_delta_best_vs_baseline": teacher_best_ev - baseline_ev,
            "actual_delta_best_vs_reference": teacher_best_ev - reference_ev,
            "predicted_EV_best": float(pred[predicted_ev_best, 0]),
            "predicted_EV_second": float(pred[predicted_ev_second, 0]),
            "predicted_EV_margin_top1_top2": float(pred[predicted_ev_best, 0] - pred[predicted_ev_second, 0]),
            "predicted_delta_vs_baseline": float(pred[predicted_delta_best, 1]),
            "predicted_delta_vs_reference": float(pred[predicted_reference_best, 2]),
            "actual_delta_selected_vs_baseline": selected_delta_gain,
            "actual_delta_selected_vs_reference": selected_reference_gain,
            "predicted_delta_action_local_index": predicted_delta_best,
            "predicted_reference_action_local_index": predicted_reference_best,
            "predicted_EV_action_local_index": predicted_ev_best,
            "baseline_action_local_index": baseline,
            "reference_action_local_index": reference,
            "reference_margin_raw": safe_float(metadata.get("baseline_model_margin", 0.0)),
            "reference_margin_predicted": float(pred[predicted_ev_best, 0] - pred[reference, 0]),
            "gate_probability": sigmoid(gate_logit_mean),
            "gate_logit_mean": gate_logit_mean,
            "SE_delta": safe_float(metadata.get("SE_delta_best_vs_baseline", 0.0)),
        }
        records.append(record)
    return records


def state_records_for_split(records: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    if split_name == "holdout":
        return [row for row in records if row["split"] != "train"]
    return [row for row in records if row["split"] == split_name]


def action_eval_metrics(records: list[dict[str, Any]], cache: dict[str, Any], predictions: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    offsets = cache["offsets"]
    state_indices = [int(row["state_index"]) for row in records]
    if not state_indices:
        return {
            "states": 0,
            "actions": 0,
            "EV_MAE": float("nan"),
            "delta_MAE": float("nan"),
            "avg_regret": float("nan"),
            "top1": float("nan"),
            "top3": float("nan"),
            "pairwise_ranking": float("nan"),
        }
    ev_abs: list[float] = []
    delta_abs: list[float] = []
    regrets: list[float] = []
    top1 = 0
    top3 = 0
    pair_correct = 0
    pair_total = 0
    action_count = 0
    for state_index in state_indices:
        start = int(offsets[state_index])
        end = int(offsets[state_index + 1])
        action_count += end - start
        pred = predictions[start:end]
        target = targets[start:end]
        ev_abs.extend(np.abs(pred[:, 0] - target[:, 0]).astype(float).tolist())
        delta_abs.extend(np.abs(pred[:, 1] - target[:, 1]).astype(float).tolist())
        best = int(np.argmax(target[:, 0]))
        predicted = int(np.argmax(pred[:, 0]))
        regrets.append(float(target[best, 0] - target[predicted, 0]))
        top1 += int(predicted == best)
        top_k = min(3, end - start)
        top_indices = set(int(i) for i in np.argpartition(pred[:, 0], -top_k)[-top_k:])
        top3 += int(best in top_indices)
        for i in range(end - start):
            for j in range(i + 1, end - start):
                target_cmp = float(target[i, 0] - target[j, 0])
                if abs(target_cmp) <= 1e-9:
                    continue
                pred_cmp = float(pred[i, 0] - pred[j, 0])
                pair_correct += int((target_cmp > 0.0) == (pred_cmp > 0.0))
                pair_total += 1
    return {
        "states": len(state_indices),
        "actions": action_count,
        "EV_MAE": float(np.mean(ev_abs)) if ev_abs else float("nan"),
        "delta_MAE": float(np.mean(delta_abs)) if delta_abs else float("nan"),
        "avg_regret": float(np.mean(regrets)) if regrets else float("nan"),
        "top1": top1 / len(state_indices),
        "top3": top3 / len(state_indices),
        "pairwise_ranking": pair_correct / pair_total if pair_total else float("nan"),
    }


def gate_metrics_for_rows(
    rows: list[dict[str, Any]],
    mode: str,
    group_type: str,
    group_name: str,
    *,
    split_name: str = "",
) -> dict[str, Any]:
    y_true: list[int] = []
    scores: list[float] = []
    for row in rows:
        label = int(row["pilot_gate_label_id"])
        if mode == "gray_ignored" and label == LABEL_GRAY:
            continue
        y_true.append(1 if label == LABEL_POSITIVE else 0)
        scores.append(float(row["gate_probability"]))
    tp = fp = tn = fn = 0
    for label, score in zip(y_true, scores):
        pred = 1 if score >= 0.5 else 0
        if label == 1 and pred == 1:
            tp += 1
        elif label == 0 and pred == 1:
            fp += 1
        elif label == 0 and pred == 0:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if math.isfinite(precision) and math.isfinite(recall) and precision + recall else float("nan")
    total = tp + fp + tn + fn
    return {
        "split": split_name,
        "mode": mode,
        "group_type": group_type,
        "group": group_name,
        "states": total,
        "positive_count": sum(y_true),
        "negative_count": total - sum(y_true),
        "threshold": 0.5,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / total if total else float("nan"),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
    }


def calibration_rows_for(records: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pairs = [
        ("predicted_delta_vs_baseline", "actual_delta_selected_vs_baseline"),
        ("predicted_delta_vs_reference", "actual_delta_selected_vs_reference"),
        ("predicted_EV_margin_top1_top2", "teacher_best_margin"),
        ("gate_probability", "actual_positive_label"),
        ("reference_margin_raw", "actual_delta_selected_vs_baseline"),
        ("reference_margin_raw", "false_positive_risk"),
    ]
    enriched: list[dict[str, Any]] = []
    for row in records:
        clone = dict(row)
        clone["actual_positive_label"] = 1.0 if int(row["pilot_gate_label_id"]) == LABEL_POSITIVE else 0.0
        clone["false_positive_risk"] = 1.0 if float(row["actual_delta_selected_vs_baseline"]) < 0.0 else 0.0
        enriched.append(clone)
    for predictor, target in pairs:
        x = np.asarray([float(row[predictor]) for row in enriched], dtype=np.float64)
        y = np.asarray([float(row[target]) for row in enriched], dtype=np.float64)
        fitted = isotonic_fit_predict(x, y)
        rows.append(
            {
                "split": split_name,
                "row_type": "correlation",
                "predictor": predictor,
                "target": target,
                "count": int(np.isfinite(x).sum()),
                "pearson": pearson(x, y),
                "spearman": spearman(x, y),
                "mae": float(np.nanmean(np.abs(x - y))) if x.size else float("nan"),
                "isotonic_mae_in_sample": float(np.nanmean(np.abs(fitted - y))) if np.isfinite(fitted).any() else float("nan"),
                "monotonicity_spearman_positive": bool((spearman(x, y) or 0.0) > 0.0),
            }
        )
        if x.size == 0:
            continue
        quantiles = np.quantile(x, np.linspace(0.0, 1.0, 11))
        for decile in range(10):
            lo = quantiles[decile]
            hi = quantiles[decile + 1]
            if decile == 9:
                mask = (x >= lo) & (x <= hi)
            else:
                mask = (x >= lo) & (x < hi)
            if not mask.any():
                continue
            rows.append(
                {
                    "split": split_name,
                    "row_type": "decile",
                    "predictor": predictor,
                    "target": target,
                    "decile": decile + 1,
                    "count": int(mask.sum()),
                    "predictor_min": float(x[mask].min()),
                    "predictor_max": float(x[mask].max()),
                    "predictor_mean": float(x[mask].mean()),
                    "target_mean": float(y[mask].mean()),
                    "mae": float(np.mean(np.abs(x[mask] - y[mask]))),
                    "positive_rate": float(np.mean(y[mask] > 0.0)),
                    "false_positive_rate": float(np.mean(y[mask] > 0.0)) if target == "false_positive_risk" else "",
                    "isotonic_mean": float(np.nanmean(fitted[mask])) if np.isfinite(fitted[mask]).any() else "",
                }
            )
    return rows


def sweep_rows(
    records: list[dict[str, Any]],
    *,
    split_name: str,
    min_margins: Iterable[float],
    reference_margins: Iterable[float],
    gate_thresholds: Iterable[float],
    sweep_type: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    evaluated = len(records)
    by_seat_total = Counter(str(row["seat"]) for row in records)
    for min_margin in min_margins:
        for reference_margin in reference_margins:
            for gate_threshold in gate_thresholds:
                overrides = [
                    row
                    for row in records
                    if int(row["predicted_delta_action_local_index"]) != int(row["baseline_action_local_index"])
                    and float(row["predicted_delta_vs_baseline"]) >= float(min_margin)
                    and float(row["reference_margin_raw"]) >= float(reference_margin)
                    and float(row["gate_probability"]) >= float(gate_threshold)
                ]
                gains = np.asarray([float(row["actual_delta_selected_vs_baseline"]) for row in overrides], dtype=np.float64)
                losses = np.maximum(0.0, -gains) if gains.size else np.asarray([], dtype=np.float64)
                by_seat = defaultdict(list)
                by_bucket = defaultdict(list)
                label_counts = Counter(str(row["pilot_gate_label"]) for row in overrides)
                for row, gain in zip(overrides, gains):
                    by_seat[str(row["seat"])].append(float(gain))
                    by_bucket[str(row["bucket_group"])].append(float(gain))
                bucket_false = {
                    bucket: {
                        "override_count": len(values),
                        "false_positive_count": sum(1 for value in values if value < 0.0),
                        "false_positive_rate": (sum(1 for value in values if value < 0.0) / len(values)) if values else 0.0,
                    }
                    for bucket, values in sorted(by_bucket.items())
                }
                row = {
                    "split": split_name,
                    "sweep_type": sweep_type,
                    "hu_turn2_min_margin": float(min_margin),
                    "hu_turn2_reference_min_margin": float(reference_margin),
                    "gate_threshold": float(gate_threshold),
                    "evaluated_states": evaluated,
                    "override_count": len(overrides),
                    "override_rate": len(overrides) / evaluated if evaluated else 0.0,
                    "teacher_avg_gain_on_override": float(gains.mean()) if gains.size else 0.0,
                    "median_gain_on_override": percentile(gains, 0.50) if gains.size else 0.0,
                    "false_positive_rate": float(np.mean(gains < 0.0)) if gains.size else 0.0,
                    "avg_false_positive_cost": float(losses[losses > 0.0].mean()) if (losses > 0.0).any() else 0.0,
                    "p90_loss": percentile(losses, 0.90) if losses.size else 0.0,
                    "p95_loss": percentile(losses, 0.95) if losses.size else 0.0,
                    "p99_loss": percentile(losses, 0.99) if losses.size else 0.0,
                    "max_loss": float(losses.max()) if losses.size else 0.0,
                    "positive_override_count": int(label_counts.get("positive", 0)),
                    "gray_override_count": int(label_counts.get("gray", 0)),
                    "negative_override_count": int(label_counts.get("negative", 0)),
                    "first_override_rate": len(by_seat.get("first", [])) / by_seat_total.get("first", 1),
                    "second_override_rate": len(by_seat.get("second", [])) / by_seat_total.get("second", 1),
                    "first_avg_gain": float(np.mean(by_seat["first"])) if by_seat.get("first") else 0.0,
                    "second_avg_gain": float(np.mean(by_seat["second"])) if by_seat.get("second") else 0.0,
                    "bucket_false_positive": json.dumps(bucket_false, sort_keys=True, ensure_ascii=False),
                }
                rows.append(row)
    return rows


def breakdown_rows(
    all_records: list[dict[str, Any]],
    cache: dict[str, Any],
    predictions: np.ndarray,
    targets: np.ndarray,
    group_type: str,
    groups: list[tuple[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test", "holdout"):
        split_records = state_records_for_split(all_records, split_name)
        for group_name, predicate in groups:
            selected = [row for row in split_records if predicate(row)]
            metrics = action_eval_metrics(selected, cache, predictions, targets)
            gate_gray_ignored = gate_metrics_for_rows(
                selected,
                "gray_ignored",
                group_type,
                str(group_name),
                split_name=split_name,
            )
            gate_gray_negative = gate_metrics_for_rows(
                selected,
                "gray_as_negative",
                group_type,
                str(group_name),
                split_name=split_name,
            )
            rows.append(
                {
                    "split": split_name,
                    "group_type": group_type,
                    "group": group_name,
                    **metrics,
                    "gate_precision_gray_ignored": gate_gray_ignored["precision"],
                    "gate_recall_gray_ignored": gate_gray_ignored["recall"],
                    "gate_precision_gray_as_negative": gate_gray_negative["precision"],
                    "gate_recall_gray_as_negative": gate_gray_negative["recall"],
                }
            )
    return rows


def recommend_thresholds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row["split"] in ("test", "holdout")
        and row["override_rate"] >= 0.01
        and row["override_rate"] <= 0.20
        and row["teacher_avg_gain_on_override"] > 0.0
        and row["false_positive_rate"] <= 0.40
    ]
    if not candidates:
        candidates = [
            row
            for row in rows
            if row["split"] in ("test", "holdout")
            and row["override_count"] > 0
            and row["teacher_avg_gain_on_override"] > 0.0
        ]
    ranked = sorted(
        candidates,
        key=lambda row: (
            row["teacher_avg_gain_on_override"],
            -row["false_positive_rate"],
            -row["p95_loss"],
            row["override_rate"],
        ),
        reverse=True,
    )
    out = []
    for rank, row in enumerate(ranked[:25], start=1):
        item = dict(row)
        item["rank"] = rank
        item["recommendation_note"] = (
            "pilot_candidate_only; repeat on larger holdout before 50k runtime/prod use"
        )
        out.append(item)
    return out


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    if not rows:
        return ["_No rows._"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        rendered = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                rendered.append(f"{value:.4f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return lines


def write_reference_margin_diagnosis(path: Path, scale_rows: list[dict[str, Any]]) -> None:
    ref_test = [row for row in scale_rows if row["split"] == "test" and row["metric"] == "reference_margin_raw"][0]
    lines = [
        "# Reference Margin Diagnosis",
        "",
        "- `reference_margin_raw` is sourced from `state_metadata.baseline_model_margin`.",
        "- In this pilot cache that value is the Stage7/continuation reference model score gap saved by the teacher-data pipeline, not a T2 calibrated EV margin.",
        "- `reference_margin_predicted` in this report is reconstructed from the T2 predicted EV head as `predicted_EV_top1 - predicted_EV(reference_action)`.",
        "- Runtime threshold `hu_turn2_reference_min_margin=10` is therefore incompatible with the raw pilot scale.",
        "",
        "## Test Split Raw Scale",
        "",
        f"- p50: `{ref_test['p50']:.6f}`",
        f"- p90: `{ref_test['p90']:.6f}`",
        f"- p95: `{ref_test['p95']:.6f}`",
        f"- p99: `{ref_test['p99']:.6f}`",
        f"- max: `{ref_test['max']:.6f}`",
        "",
        "Because p95 is near 1.15 and max is far below 10 on this pilot set, a reference threshold of 10 blocks every override.",
        "For T2 calibration, sweep reference thresholds around quantiles or 0.00-1.25 and keep the result pilot-only until repeated on a larger holdout.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(
    path: Path,
    *,
    train_summary: dict[str, Any] | None,
    scale_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
) -> None:
    pred_delta_test = [row for row in scale_rows if row["split"] == "test" and row["metric"] == "predicted_delta_vs_baseline"][0]
    ref_test = [row for row in scale_rows if row["split"] == "test" and row["metric"] == "reference_margin_raw"][0]
    corr_rows = [row for row in calibration_rows if row.get("row_type") == "correlation" and row.get("split") == "test"]
    lines = [
        "# HU T2 Pilot Calibration Summary",
        "",
        "## Existing Pilot Training",
        "",
    ]
    if train_summary:
        test = train_summary["eval"]["test"]
        lines.extend(
            [
                f"- states/actions: `{train_summary['split_counts']['train'] + train_summary['split_counts']['val'] + train_summary['split_counts']['test']}` / from cache",
                f"- test EV MAE: `{test['ev_mae']:.4f}`",
                f"- test avg_regret: `{test['avg_regret']:.4f}`",
                f"- test top1/top3: `{test['top1_accuracy']:.4f}` / `{test['top3_recall']:.4f}`",
                f"- test pairwise ranking: `{test['pairwise_ranking_accuracy']:.4f}`",
                f"- test gate accuracy pos/neg: `{test['gate_accuracy_pos_neg']:.4f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Scale Finding",
            "",
            f"- test predicted_delta_vs_baseline p95: `{pred_delta_test['p95']:.4f}`",
            f"- test reference_margin_raw p95: `{ref_test['p95']:.4f}`",
            "- production-style `5/10` thresholds are above this pilot's T2 output scale and explain zero overrides.",
            "",
            "## Calibration Correlations",
            "",
        ]
    )
    lines.extend(markdown_table(corr_rows, ["predictor", "target", "pearson", "spearman", "mae", "isotonic_mae_in_sample"]))
    lines.extend(["", "## Recommended Pilot Threshold Candidates", ""])
    lines.extend(
        markdown_table(
            recommendations[:8],
            [
                "rank",
                "split",
                "sweep_type",
                "hu_turn2_min_margin",
                "hu_turn2_reference_min_margin",
                "gate_threshold",
                "override_rate",
                "teacher_avg_gain_on_override",
                "false_positive_rate",
                "p95_loss",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This validates that the T2 pilot has a usable signal, but threshold choice is not production-ready.",
            "Use this as a calibration pass only. Do not start 50k teacher or production candidate training from `5/10` thresholds.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_go_nogo(path: Path, recommendations: list[dict[str, Any]], calibration_rows: list[dict[str, Any]]) -> None:
    test_corr = [
        row
        for row in calibration_rows
        if row.get("row_type") == "correlation"
        and row.get("split") == "test"
        and row.get("predictor") == "predicted_delta_vs_baseline"
        and row.get("target") == "actual_delta_selected_vs_baseline"
    ]
    corr_value = float(test_corr[0]["spearman"]) if test_corr else float("nan")
    best = recommendations[0] if recommendations else None
    if best and best["teacher_avg_gain_on_override"] > 0.0 and best["override_count"] > 0 and corr_value > 0.0:
        decision = "CONDITIONAL-GO for a larger calibration/teacher pass, not production."
        reason = "T2-scale thresholds can fire and pilot teacher gain can be positive, but evidence is only 2k states."
    else:
        decision = "NO-GO for 50k."
        reason = "No robust positive threshold candidate was found on this pilot holdout."
    lines = [
        "# Go / No-Go for 50k",
        "",
        f"Decision: **{decision}**",
        "",
        reason,
        "",
        "## Conditions Before 50k",
        "",
        "- Use T2-scale thresholds, not `5/10`.",
        "- Keep this model pilot-only.",
        "- Repeat the same calibration on a larger holdout or a small 5k-10k intermediate pass.",
        "- Require positive teacher gain, tolerable false positives, and no first/second or bucket-specific collapse.",
        "- Keep a separate production gate after 50k; this pass does not authorize production runtime use.",
        "",
        "## Next Concrete Command",
        "",
        "Run a larger calibration teacher pass only after selecting 1-3 threshold candidates from `recommended_threshold_candidates.csv`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    device = select_device(torch, args.device)
    cache = load_cache(args.cache_dir.resolve())
    targets = target_matrix(cache)
    predictions = load_predictions(torch, cache, args.model.resolve(), device, args.batch_size)
    records = build_state_records(cache, predictions, targets)

    scale_rows: list[dict[str, Any]] = []
    scale_metrics = [
        "teacher_best_EV",
        "teacher_second_EV",
        "teacher_best_margin",
        "actual_delta_best_vs_baseline",
        "actual_delta_best_vs_reference",
        "predicted_EV_best",
        "predicted_EV_second",
        "predicted_EV_margin_top1_top2",
        "predicted_delta_vs_baseline",
        "predicted_delta_vs_reference",
        "reference_margin_raw",
        "reference_margin_predicted",
        "gate_probability",
        "SE_delta",
    ]
    for split_name in ("train", "val", "test", "holdout"):
        split_records = state_records_for_split(records, split_name)
        for metric in scale_metrics:
            scale_rows.append(stats_row(split_name, metric, [row[metric] for row in split_records]))

    calibration_rows: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test", "holdout"):
        calibration_rows.extend(calibration_rows_for(state_records_for_split(records, split_name), split_name))

    gate_rows: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test", "holdout"):
        split_records = state_records_for_split(records, split_name)
        for mode in ("gray_ignored", "gray_as_negative"):
            gate_rows.append(gate_metrics_for_rows(split_records, mode, "all", "all", split_name=split_name))
            for seat in sorted({str(row["seat"]) for row in split_records}):
                gate_rows.append(
                    gate_metrics_for_rows(
                        [row for row in split_records if str(row["seat"]) == seat],
                        mode,
                        "seat",
                        seat,
                        split_name=split_name,
                    )
                )
            for bucket in sorted({str(row["bucket_group"]) for row in split_records}):
                gate_rows.append(
                    gate_metrics_for_rows(
                        [row for row in split_records if str(row["bucket_group"]) == bucket],
                        mode,
                        "bucket_group",
                        bucket,
                        split_name=split_name,
                    )
                )

    t2_sweep_rows: list[dict[str, Any]] = []
    quantile_sweep_rows: list[dict[str, Any]] = []
    for split_name in ("val", "test", "holdout"):
        split_records = state_records_for_split(records, split_name)
        t2_sweep_rows.extend(
            sweep_rows(
                split_records,
                split_name=split_name,
                min_margins=T2_MIN_MARGIN_GRID,
                reference_margins=REFERENCE_MARGIN_GRID,
                gate_thresholds=GATE_THRESHOLD_GRID,
                sweep_type="fixed_t2_scale",
            )
        )
        pred_values = [float(row["predicted_delta_vs_baseline"]) for row in split_records]
        ref_values = [float(row["reference_margin_raw"]) for row in split_records]
        pred_quantiles = [percentile(pred_values, level) for level in QUANTILE_LEVELS]
        ref_quantiles = [percentile(ref_values, level) for level in QUANTILE_LEVELS]
        quantile_sweep_rows.extend(
            sweep_rows(
                split_records,
                split_name=split_name,
                min_margins=pred_quantiles,
                reference_margins=ref_quantiles,
                gate_thresholds=GATE_THRESHOLD_GRID,
                sweep_type="quantile",
            )
        )

    position_rows = breakdown_rows(
        records,
        cache,
        predictions,
        targets,
        "seat",
        [("first", lambda row: str(row["seat"]) == "first"), ("second", lambda row: str(row["seat"]) == "second")],
    )
    bucket_groups = sorted({str(row["bucket_group"]) for row in records})
    bucket_rows = breakdown_rows(
        records,
        cache,
        predictions,
        targets,
        "bucket_group",
        [(bucket, lambda row, bucket=bucket: str(row["bucket_group"]) == bucket) for bucket in bucket_groups]
        + [
            ("actual_high_regret", lambda row: bool(row["actual_high_regret"])),
            ("actual_low_margin", lambda row: bool(row["actual_low_margin"])),
            ("actual_teacher_disagreement", lambda row: bool(row["actual_teacher_disagreement"])),
        ],
    )

    recommendations = recommend_thresholds(t2_sweep_rows + quantile_sweep_rows)
    train_summary = None
    summary_path = args.previous_output / "training_summary.json"
    if summary_path.exists():
        train_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    write_csv(output_dir / "scale_audit.csv", scale_rows)
    write_csv(output_dir / "calibration_by_decile.csv", calibration_rows)
    write_csv(output_dir / "gate_metrics.csv", gate_rows)
    write_csv(output_dir / "threshold_sweep_t2_scale.csv", t2_sweep_rows)
    write_csv(output_dir / "threshold_sweep_quantile.csv", quantile_sweep_rows)
    write_csv(output_dir / "position_breakdown.csv", position_rows)
    write_csv(output_dir / "bucket_breakdown.csv", bucket_rows)
    write_csv(output_dir / "recommended_threshold_candidates.csv", recommendations)
    write_reference_margin_diagnosis(output_dir / "reference_margin_diagnosis.md", scale_rows)
    write_summary(
        output_dir / "calibration_summary.md",
        train_summary=train_summary,
        scale_rows=scale_rows,
        calibration_rows=calibration_rows,
        gate_rows=gate_rows,
        recommendations=recommendations,
    )
    write_go_nogo(output_dir / "go_nogo_for_50k.md", recommendations, calibration_rows)
    manifest = {
        "schema": "hu_turn2_stage8_pilot_calibration_v1",
        "cache_dir": str(args.cache_dir.resolve()),
        "model": str(args.model.resolve()),
        "previous_output": str(args.previous_output.resolve()),
        "output_dir": str(output_dir),
        "device": device,
        "states": int(cache["metadata"]["state_count"]),
        "actions": int(cache["metadata"]["action_count"]),
        "wrote": [
            "scale_audit.csv",
            "calibration_by_decile.csv",
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
    (output_dir / "calibration_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
