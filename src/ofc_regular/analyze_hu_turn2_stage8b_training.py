"""Post-training audit for the HU Turn2 Stage8b safe-override head.

This is evaluation-only. It does not generate teacher data, run seat-swap, or
promote a runtime. It checks whether the trained deployable safe-override head
learned the Stage8b labels strongly enough to proceed to C2-small validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .analyze_hu_turn2_gate_c1_followup import enriched_state_rows, load_action_original_indices
from .analyze_hu_turn2_pilot_calibration import auc_pr, load_model, write_csv
from .train_hu_turn2_pilot_model import load_cache, predict_all, target_matrix
from .train_torch_action_value import select_device

DEFAULT_CACHE_DIR = Path("D:/ofc-pineapple-storage/regular-ofc-pineapple/feature_cache/hu_t2_stage8_20k_mc512")
DEFAULT_LABELS = Path("outputs/hu_turn2_stage8b_prelarge_training/stage8b_safe_override_labels.csv")
DEFAULT_MODEL = Path("models/hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt")
DEFAULT_OUTPUT_DIR = Path("outputs/evals/hu_turn2_stage8b_training_audit")

SCORE_THRESHOLDS = (0.50, 0.70, 0.80, 0.85, 0.90, 0.95)
DELTA_GRID = (1.50, 2.00, 2.25, 2.50, 2.75)
RANK_GRID = (1, 2, 3, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--stage8b-labels-csv", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32768)
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def percentile(values: Iterable[float], q: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return 0.0
    return float(np.quantile(array, q))


def read_label_rows(path: Path) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "state_index" not in (reader.fieldnames or ()):
            raise ValueError(f"{path} is missing state_index")
        for row in reader:
            state_index = int(row["state_index"])
            rows[state_index] = row
    return rows


def split_matches(row: dict[str, Any], split_name: str) -> bool:
    split = str(row.get("split", ""))
    if split_name == "all":
        return True
    if split_name == "holdout":
        return split != "train"
    return split == split_name


def join_stage8b_labels(rows: list[dict[str, Any]], labels: dict[int, dict[str, str]]) -> None:
    for row in rows:
        state_index = safe_int(row["state_index"])
        label = labels.get(state_index)
        if label is None:
            raise ValueError(f"missing Stage8b label for state_index={state_index}")
        row["safe_override_probability"] = safe_float(row.get("gate_probability"))
        row["stage8b_safe_lcb196_gate_label_id"] = safe_int(label.get("safe_lcb196_gate_label_id"), 1)
        row["stage8b_safe_lcb164_gate_label_id"] = safe_int(label.get("safe_lcb164_gate_label_id"), 1)
        row["stage8b_safe_lcb196_label"] = label.get("safe_lcb196_label", "")
        row["stage8b_safe_lcb164_label"] = label.get("safe_lcb164_label", "")
        row["stage8b_gate_weight"] = safe_float(label.get("stage8b_gate_weight"), 1.0)
        row["teacher_delta_lcb_196"] = safe_float(label.get("teacher_delta_lcb_196"))
        row["teacher_delta_lcb_164"] = safe_float(label.get("teacher_delta_lcb_164"))
        row["stage8b_hard_negative_label"] = int(truthy(label.get("hard_negative_label")))
        row["stage8b_gray_label"] = int(truthy(label.get("gray_label")))
        row["stage8b_replay_ready"] = int(truthy(label.get("replay_ready")))
        row["stage8b_high_mc_label_source"] = label.get("high_mc_label_source", "")
        row["stage8b_high_mc_diagnosis"] = label.get("high_mc_diagnosis", "")
        row["stage8b_hard_negative_source"] = label.get("hard_negative_source", "")
        row["old_stage8_current_proxy_m2p5_g0p9_fired"] = int(truthy(label.get("current_proxy_m2p5_g0p9_fired")))
        row["old_stage8_predicted_delta_vs_baseline"] = safe_float(label.get("predicted_delta_vs_baseline"))
        row["old_stage8_gate_probability"] = safe_float(label.get("gate_probability"))
        row["old_stage8_predicted_ev_margin"] = safe_float(label.get("predicted_EV_margin_top1_top2"))
        for key in ("c1e_split", "position", "source_group", "source_bucket", "bucket_group", "run_bucket"):
            if label.get(key) not in (None, ""):
                row[key] = label[key]


def add_rank_fields(rows: list[dict[str, Any]], cache: dict[str, Any], predictions: np.ndarray) -> None:
    offsets = cache["offsets"]
    for row in rows:
        state_index = safe_int(row["state_index"])
        start = int(offsets[state_index])
        end = int(offsets[state_index + 1])
        pred = np.asarray(predictions[start:end], dtype=np.float64)
        candidate = safe_int(row["candidate_action_local_index"])
        ev_order = np.argsort(-pred[:, 0], kind="mergesort")
        delta_order = np.argsort(-pred[:, 1], kind="mergesort")
        row["stage8b_model_candidate_ev_rank"] = int(np.where(ev_order == candidate)[0][0]) + 1
        row["stage8b_model_candidate_delta_rank"] = int(np.where(delta_order == candidate)[0][0]) + 1
        row["stage8b_predicted_delta_margin_top1_top2"] = float(
            pred[delta_order[0], 1] - (pred[delta_order[1], 1] if delta_order.size > 1 else pred[delta_order[0], 1])
        )


def build_rows(cache_dir: Path, model_path: Path, labels_csv: Path, *, device_name: str, batch_size: int) -> list[dict[str, Any]]:
    import torch

    device = select_device(torch, device_name)
    cache = load_cache(cache_dir)
    net, stats, _payload = load_model(torch, model_path, device)
    predictions = predict_all(torch, net, cache, stats, device, batch_size)
    targets = target_matrix(cache)
    action_original_indices = load_action_original_indices(cache_dir, int(cache["metadata"]["action_count"]))
    rows = enriched_state_rows(cache, predictions, targets, action_original_indices)
    join_stage8b_labels(rows, read_label_rows(labels_csv))
    add_rank_fields(rows, cache, predictions)
    return rows


def binary_counts(rows: list[dict[str, Any]], threshold: float) -> dict[str, int]:
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "pos": 0, "neg": 0, "gray": 0}
    for row in rows:
        label_id = safe_int(row.get("stage8b_safe_lcb196_gate_label_id"), 1)
        if label_id == 1:
            counts["gray"] += 1
            continue
        score = safe_float(row.get("safe_override_probability"))
        pred = score >= threshold
        actual = label_id == 2
        counts["pos" if actual else "neg"] += 1
        if pred and actual:
            counts["tp"] += 1
        elif pred and not actual:
            counts["fp"] += 1
        elif not pred and actual:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
    return counts


def metric_from_counts(counts: dict[str, int]) -> dict[str, float]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def pr_auc_for_rows(rows: list[dict[str, Any]]) -> float:
    labels: list[int] = []
    scores: list[float] = []
    for row in rows:
        label_id = safe_int(row.get("stage8b_safe_lcb196_gate_label_id"), 1)
        if label_id == 1:
            continue
        labels.append(1 if label_id == 2 else 0)
        scores.append(safe_float(row.get("safe_override_probability")))
    return auc_pr(labels, scores)


def gate_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split in ("train", "val", "test", "holdout", "all"):
        subset = [row for row in rows if split_matches(row, split)]
        pr_auc = pr_auc_for_rows(subset)
        for threshold in SCORE_THRESHOLDS:
            counts = binary_counts(subset, threshold)
            metrics = metric_from_counts(counts)
            hard = [row for row in subset if safe_int(row.get("stage8b_hard_negative_label")) == 1]
            hard_fired = [row for row in hard if safe_float(row.get("safe_override_probability")) >= threshold]
            output.append(
                {
                    "split": split,
                    "threshold": threshold,
                    "rows": len(subset),
                    "positive_labels": counts["pos"],
                    "negative_labels": counts["neg"],
                    "gray_labels": counts["gray"],
                    "tp": counts["tp"],
                    "fp": counts["fp"],
                    "tn": counts["tn"],
                    "fn": counts["fn"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "accuracy": metrics["accuracy"],
                    "pr_auc": pr_auc,
                    "hard_negative_count": len(hard),
                    "hard_negative_rejected": len(hard) - len(hard_fired),
                    "hard_negative_recall_reject": (len(hard) - len(hard_fired)) / max(len(hard), 1),
                    "hard_negative_false_negative_fired": len(hard_fired),
                }
            )
    return output


def group_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    group_fields = ("c1e_split", "position", "source_bucket", "bucket_group", "actual_high_regret", "actual_low_margin", "actual_teacher_disagreement")
    for split in ("test", "holdout", "all"):
        split_rows = [row for row in rows if split_matches(row, split)]
        for field in group_fields:
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in split_rows:
                grouped[str(row.get(field, ""))].append(row)
            for group_value, subset in sorted(grouped.items()):
                pr_auc = pr_auc_for_rows(subset)
                for threshold in (0.80, 0.90, 0.95):
                    counts = binary_counts(subset, threshold)
                    metrics = metric_from_counts(counts)
                    output.append(
                        {
                            "split": split,
                            "group_field": field,
                            "group_value": group_value,
                            "threshold": threshold,
                            "rows": len(subset),
                            "positive_labels": counts["pos"],
                            "negative_labels": counts["neg"],
                            "gray_labels": counts["gray"],
                            "precision": metrics["precision"],
                            "recall": metrics["recall"],
                            "f1": metrics["f1"],
                            "pr_auc": pr_auc,
                        }
                    )
    return output


def runtime_fires(row: dict[str, Any], *, min_delta: float, safe_threshold: float, rank_max: int) -> bool:
    return (
        safe_int(row.get("candidate_is_baseline")) == 0
        and safe_float(row.get("predicted_delta_vs_baseline")) >= min_delta
        and safe_float(row.get("safe_override_probability")) >= safe_threshold
        and safe_int(row.get("stage8b_model_candidate_ev_rank"), 999) <= rank_max
    )


def runtime_gate_sweep(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split in ("test", "holdout"):
        subset = [row for row in rows if split_matches(row, split)]
        oracle = {safe_int(row["state_index"]) for row in subset if safe_int(row.get("stage8b_safe_lcb196_gate_label_id")) == 2}
        hard = {safe_int(row["state_index"]) for row in subset if safe_int(row.get("stage8b_hard_negative_label")) == 1}
        for min_delta in DELTA_GRID:
            for safe_threshold in (0.80, 0.85, 0.90, 0.95):
                for rank_max in RANK_GRID:
                    fired = [row for row in subset if runtime_fires(row, min_delta=min_delta, safe_threshold=safe_threshold, rank_max=rank_max)]
                    fired_ids = {safe_int(row["state_index"]) for row in fired}
                    gains = [safe_float(row.get("actual_delta_candidate_vs_baseline")) for row in fired]
                    losses = [max(0.0, -value) for value in gains]
                    false_positive = [value for value in gains if value < 0.0]
                    overlap = fired_ids & oracle
                    hard_fired = fired_ids & hard
                    output.append(
                        {
                            "split": split,
                            "gate_id": f"m{min_delta:g}_p{safe_threshold:g}_k{rank_max}",
                            "min_delta": min_delta,
                            "safe_probability_threshold": safe_threshold,
                            "candidate_ev_rank_max": rank_max,
                            "rows": len(subset),
                            "fires": len(fired),
                            "fire_rate": len(fired) / max(len(subset), 1),
                            "avg_gain": float(np.mean(gains)) if gains else 0.0,
                            "median_gain": float(np.median(gains)) if gains else 0.0,
                            "false_positive_count": len(false_positive),
                            "false_positive_rate": len(false_positive) / max(len(fired), 1),
                            "p95_loss": percentile(losses, 0.95),
                            "p99_loss": percentile(losses, 0.99),
                            "max_loss": max(losses) if losses else 0.0,
                            "oracle_precision": len(overlap) / max(len(fired), 1),
                            "oracle_recall": len(overlap) / max(len(oracle), 1),
                            "oracle_jaccard": len(overlap) / max(len(fired_ids | oracle), 1),
                            "hard_negative_fired": len(hard_fired),
                            "first_fires": sum(1 for row in fired if str(row.get("position", row.get("seat", ""))) == "first"),
                            "second_fires": sum(1 for row in fired if str(row.get("position", row.get("seat", ""))) == "second"),
                        }
                    )
    return output


def oracle_overlap_rows(rows: list[dict[str, Any]], gate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for gate in gate_rows:
        if gate["split"] not in {"test", "holdout"}:
            continue
        subset = [row for row in rows if split_matches(row, str(gate["split"]))]
        oracle = {safe_int(row["state_index"]) for row in subset if safe_int(row.get("stage8b_safe_lcb196_gate_label_id")) == 2}
        fired = {
            safe_int(row["state_index"])
            for row in subset
            if runtime_fires(
                row,
                min_delta=safe_float(gate["min_delta"]),
                safe_threshold=safe_float(gate["safe_probability_threshold"]),
                rank_max=safe_int(gate["candidate_ev_rank_max"]),
            )
        }
        inter = fired & oracle
        output.append(
            {
                "split": gate["split"],
                "gate_id": gate["gate_id"],
                "oracle_positive_count": len(oracle),
                "runtime_fire_count": len(fired),
                "overlap_count": len(inter),
                "precision_vs_oracle": len(inter) / max(len(fired), 1),
                "recall_vs_oracle": len(inter) / max(len(oracle), 1),
                "jaccard": len(inter) / max(len(fired | oracle), 1),
            }
        )
    return output


def hard_negative_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        if safe_int(row.get("stage8b_hard_negative_label")) != 1:
            continue
        score = safe_float(row.get("safe_override_probability"))
        output.append(
            {
                "state_index": safe_int(row.get("state_index")),
                "split": row.get("split", ""),
                "position": row.get("position", row.get("seat", "")),
                "source_bucket": row.get("source_bucket", ""),
                "bucket_group": row.get("bucket_group", ""),
                "hard_negative_source": row.get("stage8b_hard_negative_source", ""),
                "high_mc_label_source": row.get("stage8b_high_mc_label_source", ""),
                "high_mc_diagnosis": row.get("stage8b_high_mc_diagnosis", ""),
                "safe_override_probability": score,
                "caught_at_p80": int(score < 0.80),
                "caught_at_p90": int(score < 0.90),
                "caught_at_p95": int(score < 0.95),
                "predicted_delta_vs_baseline": safe_float(row.get("predicted_delta_vs_baseline")),
                "actual_delta_candidate_vs_baseline": safe_float(row.get("actual_delta_candidate_vs_baseline")),
                "candidate_loss": safe_float(row.get("candidate_loss")),
            }
        )
    return sorted(output, key=lambda item: -safe_float(item.get("safe_override_probability")))


def old_proxy_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split in ("test", "holdout", "all"):
        subset = [row for row in rows if split_matches(row, split)]
        fired = [row for row in subset if safe_int(row.get("old_stage8_current_proxy_m2p5_g0p9_fired")) == 1]
        gains = [safe_float(row.get("actual_delta_candidate_vs_baseline")) for row in fired]
        output.append(
            {
                "split": split,
                "proxy": "old_stage8_m2p5_g0p9",
                "fires": len(fired),
                "avg_gain": float(np.mean(gains)) if gains else 0.0,
                "false_positive_count": sum(1 for value in gains if value < 0.0),
                "false_positive_rate": sum(1 for value in gains if value < 0.0) / max(len(fired), 1),
            }
        )
    return output


def best_runtime_candidates(gate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    holdout = [row for row in gate_rows if row["split"] == "holdout" and safe_int(row["fires"]) > 0]
    holdout = [
        row
        for row in holdout
        if safe_float(row["avg_gain"]) > 0.0
        and safe_float(row["false_positive_rate"]) <= 0.10
        and safe_float(row["oracle_precision"]) >= 0.50
    ]
    return sorted(
        holdout,
        key=lambda row: (
            -safe_float(row["avg_gain"]),
            safe_float(row["false_positive_rate"]),
            -safe_int(row["fires"]),
        ),
    )[:10]


def write_summary(
    output_dir: Path,
    gate_metrics: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
    hard_rows: list[dict[str, Any]],
    old_rows: list[dict[str, Any]],
) -> None:
    best = best_runtime_candidates(runtime_rows)
    test_p90 = next((row for row in gate_metrics if row["split"] == "test" and safe_float(row["threshold"]) == 0.90), {})
    holdout_p90 = next((row for row in gate_metrics if row["split"] == "holdout" and safe_float(row["threshold"]) == 0.90), {})
    old_holdout = next((row for row in old_rows if row["split"] == "holdout"), {})
    hard_total = len(hard_rows)
    hard_caught_p90 = sum(safe_int(row.get("caught_at_p90")) for row in hard_rows)
    lines = [
        "# HU T2 Stage8b Training Audit",
        "",
        "This is a post-training audit for the deployable safe-override head. It is not production approval.",
        "",
        "## Safe Head",
        "",
        f"- test p=0.90 precision / recall / F1 / PR-AUC: `{safe_float(test_p90.get('precision')):.4f}` / `{safe_float(test_p90.get('recall')):.4f}` / `{safe_float(test_p90.get('f1')):.4f}` / `{safe_float(test_p90.get('pr_auc')):.4f}`",
        f"- holdout p=0.90 precision / recall / F1 / PR-AUC: `{safe_float(holdout_p90.get('precision')):.4f}` / `{safe_float(holdout_p90.get('recall')):.4f}` / `{safe_float(holdout_p90.get('f1')):.4f}` / `{safe_float(holdout_p90.get('pr_auc')):.4f}`",
        f"- hard negatives caught at p=0.90: `{hard_caught_p90}/{hard_total}`",
        "",
        "## Old Proxy Comparison",
        "",
        f"- old Stage8 m2.5/g0.9 holdout fires / avg gain / FP rate: `{safe_int(old_holdout.get('fires'))}` / `{safe_float(old_holdout.get('avg_gain')):.4f}` / `{safe_float(old_holdout.get('false_positive_rate')):.4f}`",
        "",
        "## Runtime Gate Candidates",
        "",
    ]
    if best:
        for row in best[:5]:
            lines.append(
                f"- `{row['gate_id']}`: fires `{row['fires']}`, avg gain `{safe_float(row['avg_gain']):.4f}`, "
                f"FP `{safe_float(row['false_positive_rate']):.4f}`, oracle precision/recall "
                f"`{safe_float(row['oracle_precision']):.4f}` / `{safe_float(row['oracle_recall']):.4f}`"
            )
        decision = "Stage8b C2-small: `Go` for validation-only runtime proxy testing."
    else:
        lines.append("- No holdout runtime gate candidate met the audit filters.")
        decision = "Stage8b C2-small: `No-Go` until the safe head/gate is improved."
    lines.extend(
        [
            "",
            "## Decision",
            "",
            decision,
            "",
            "- 50k teacher: `No-Go`",
            "- T1: `No-Go`",
            "- production / P2 fixed: `No-Go`",
        ]
    )
    (output_dir / "stage8b_training_audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "go_nogo_for_stage8b_c2_small.md").write_text("\n".join(lines[-7:]) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(
        args.cache_dir.resolve(),
        args.model.resolve(),
        args.stage8b_labels_csv.resolve(),
        device_name=args.device,
        batch_size=args.batch_size,
    )
    gate_metrics = gate_metric_rows(rows)
    group_metrics = group_metric_rows(rows)
    runtime_rows = runtime_gate_sweep(rows)
    overlap_rows = oracle_overlap_rows(rows, runtime_rows)
    hard_rows = hard_negative_rows(rows)
    old_rows = old_proxy_comparison_rows(rows)
    write_csv(args.output_dir / "stage8b_gate_metrics.csv", gate_metrics)
    write_csv(args.output_dir / "stage8b_group_metrics.csv", group_metrics)
    write_csv(args.output_dir / "stage8b_runtime_gate_sweep.csv", runtime_rows)
    write_csv(args.output_dir / "stage8b_oracle_overlap.csv", overlap_rows)
    write_csv(args.output_dir / "stage8b_hard_negative_audit.csv", hard_rows)
    write_csv(args.output_dir / "stage8b_old_proxy_comparison.csv", old_rows)
    write_csv(args.output_dir / "stage8b_recommended_runtime_gate_candidates.csv", best_runtime_candidates(runtime_rows))
    write_summary(args.output_dir, gate_metrics, runtime_rows, hard_rows, old_rows)
    summary = {
        "rows": len(rows),
        "label_counts": dict(Counter(safe_int(row.get("stage8b_safe_lcb196_gate_label_id"), 1) for row in rows)),
        "hard_negative_count": len(hard_rows),
        "recommended_candidate_count": len(best_runtime_candidates(runtime_rows)),
    }
    (args.output_dir / "stage8b_training_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
