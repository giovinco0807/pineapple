"""Prepare HU T2 Stage9 candidate-generator diagnostics and hard-miss rows.

This is the first step of the Stage9 T2 candidate-generator redesign.  It
compares a model's TopK action set against the teacher oracle action in an
existing HU T2 all-action feature cache.  The output is intentionally
training/evaluation oriented, not a production runtime gate.
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

from .train_hu_turn2_pilot_model import load_cache, predict_all, scoring_metadata_status
from .train_torch_action_value import select_device
from .turn3_model import _build_torch_mlp, _import_torch

SCORE_HEADS = {
    "ev": 0,
    "delta_vs_baseline": 1,
    "delta_vs_reference": 2,
    "rank_score": 3,
    "gate_logit": 4,
}
DEFAULT_TOPK_VALUES = (1, 3, 5, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-head", choices=tuple(SCORE_HEADS), default="ev")
    parser.add_argument("--topk", default="1,3,5,10")
    parser.add_argument("--hard-miss-topk", type=int, default=5)
    parser.add_argument("--hard-negative-topk", type=int, default=5)
    parser.add_argument("--min-hard-miss-regret", type=float, default=0.25)
    parser.add_argument("--min-hard-negative-loss", type=float, default=0.25)
    parser.add_argument("--missed-positive-weight", type=float, default=4.0)
    parser.add_argument("--hard-negative-weight", type=float, default=3.0)
    parser.add_argument("--oracle-best-weight", type=float, default=1.0)
    parser.add_argument("--batch-action-rows", type=int, default=32768)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--allow-missing-scoring-metadata", action="store_true")
    parser.add_argument("--allow-scoring-mismatch", action="store_true")
    parser.add_argument("--max-states", type=int)
    return parser.parse_args()


def parse_topk_values(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        values = [int(part) for part in parts]
    else:
        values = [int(part) for part in value]
    values = sorted({k for k in values if k > 0})
    if not values:
        raise ValueError("at least one positive TopK value is required")
    return values


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")


def load_stage8_pilot_model(model_path: Path, *, device_choice: str) -> tuple[Any, Any, dict[str, np.ndarray], str]:
    torch = _import_torch()
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if str(payload.get("model_kind")) != "hu_turn2_pilot_multihead_mlp":
        raise ValueError(f"unsupported HU T2 model kind: {payload.get('model_kind')!r}")
    net = _build_torch_mlp(
        torch,
        input_dim=int(payload["feature_dim"]),
        hidden_layer_sizes=tuple(int(v) for v in payload["hidden_layer_sizes"]),
        dropout=float(payload.get("dropout", 0.0)),
        output_dim=5,
    )
    net.load_state_dict(payload["state_dict"])
    device = select_device(torch, device_choice)
    net.to(device)
    stats = {
        "feature_mean": np.asarray(payload["feature_mean"], dtype=np.float32),
        "feature_scale": np.asarray(payload["feature_scale"], dtype=np.float32),
        "target_mean": np.asarray(payload["target_mean"], dtype=np.float32),
        "target_scale": np.asarray(payload["target_scale"], dtype=np.float32),
    }
    return torch, net, stats, device


def state_indices(cache: dict[str, Any], max_states: int | None = None) -> np.ndarray:
    count = int(cache["metadata"]["state_count"])
    if max_states is not None:
        count = min(count, int(max_states))
    return np.arange(count, dtype=np.int64)


def split_name_for_id(value: Any) -> str:
    try:
        split_id = int(value)
    except (TypeError, ValueError):
        return "unknown"
    return {0: "train", 1: "val", 2: "test"}.get(split_id, f"split_{split_id}")


def topk_indices(scores: np.ndarray, k: int) -> list[int]:
    k = min(int(k), int(scores.size))
    if k <= 0:
        return []
    if k == scores.size:
        order = np.argsort(-scores)
    else:
        partial = np.argpartition(scores, -k)[-k:]
        order = partial[np.argsort(-scores[partial])]
    return [int(index) for index in order]


def _json_list(values: Iterable[Any]) -> str:
    return json.dumps(list(values), separators=(",", ":"), ensure_ascii=True)


def build_candidate_rows(
    cache: dict[str, Any],
    predictions: np.ndarray,
    *,
    score_head: str = "ev",
    topk_values: Iterable[int] = DEFAULT_TOPK_VALUES,
    hard_miss_topk: int = 5,
    hard_negative_topk: int = 5,
    min_hard_miss_regret: float = 0.25,
    min_hard_negative_loss: float = 0.25,
    missed_positive_weight: float = 4.0,
    hard_negative_weight: float = 3.0,
    oracle_best_weight: float = 1.0,
    max_states: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    topk_values = parse_topk_values(topk_values)
    score_column = SCORE_HEADS[score_head]
    offsets = cache["offsets"]
    teacher_ev = np.asarray(cache["target_ev"], dtype=np.float32)
    delta_baseline = np.asarray(cache["target_delta_baseline"], dtype=np.float32)
    baseline_indices = np.asarray(cache["baseline_action_index"], dtype=np.int64)
    best_indices = np.asarray(cache["best_action_index"], dtype=np.int64)
    split = np.asarray(cache["split"], dtype=np.int64)
    metadata = cache["state_metadata"]

    state_rows: list[dict[str, Any]] = []
    hard_miss_rows: list[dict[str, Any]] = []
    hard_negative_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []

    for state_index in state_indices(cache, max_states=max_states):
        start = int(offsets[state_index])
        end = int(offsets[state_index + 1])
        action_count = end - start
        if action_count <= 0:
            continue
        state_scores = np.asarray(predictions[start:end, score_column], dtype=np.float32)
        state_teacher_ev = np.asarray(teacher_ev[start:end], dtype=np.float32)
        state_delta_baseline = np.asarray(delta_baseline[start:end], dtype=np.float32)
        best = int(best_indices[state_index])
        if best < 0 or best >= action_count:
            best = int(np.argmax(state_teacher_ev))
        baseline = int(baseline_indices[state_index])
        if baseline < 0 or baseline >= action_count:
            baseline = 0
        top1 = topk_indices(state_scores, 1)[0]
        max_topk = max(max(topk_values), hard_miss_topk, hard_negative_topk)
        ranked = topk_indices(state_scores, max_topk)
        topk_by_value = {k: ranked[: min(k, len(ranked))] for k in topk_values}
        best_ev = float(state_teacher_ev[best])
        baseline_ev = float(state_teacher_ev[baseline])
        top1_ev = float(state_teacher_ev[top1])
        top1_regret = best_ev - top1_ev
        hard_miss_set = set(ranked[: min(hard_miss_topk, len(ranked))])
        topk_best_ev = max((float(state_teacher_ev[i]) for i in hard_miss_set), default=-math.inf)
        topk_regret = best_ev - topk_best_ev if math.isfinite(topk_best_ev) else 0.0
        hard_miss = best not in hard_miss_set and topk_regret >= min_hard_miss_regret
        top_hard_negative = ranked[: min(hard_negative_topk, len(ranked))]
        hard_negative_indices = [
            int(i)
            for i in top_hard_negative
            if float(state_delta_baseline[i]) <= -abs(min_hard_negative_loss)
        ]

        meta = metadata[int(state_index)] if int(state_index) < len(metadata) else {}
        base_row = {
            "schema": "hu_turn2_stage9_candidate_generator_state_v1",
            "state_index": int(state_index),
            "state_hash": meta.get("state_hash", ""),
            "state_id": meta.get("state_id", int(state_index)),
            "sample_id": meta.get("sample_id", ""),
            "hand_id": meta.get("hand_id", ""),
            "seed": meta.get("seed", ""),
            "split": split_name_for_id(split[state_index]),
            "seat": meta.get("seat", ""),
            "bucket_group": meta.get("bucket_group", ""),
            "source_bucket": meta.get("source_bucket", ""),
            "run_bucket": meta.get("run_bucket", ""),
            "pilot_gate_label": meta.get("pilot_gate_label", ""),
            "action_count": int(action_count),
            "score_head": score_head,
            "baseline_index": baseline,
            "oracle_best_index": best,
            "model_top1_index": top1,
            "baseline_ev": baseline_ev,
            "oracle_best_ev": best_ev,
            "oracle_best_delta_vs_baseline": float(state_delta_baseline[best]),
            "model_top1_ev": top1_ev,
            "model_top1_delta_vs_baseline": float(state_delta_baseline[top1]),
            "model_top1_regret": float(top1_regret),
            "hard_miss_topk": int(hard_miss_topk),
            "hard_miss": int(hard_miss),
            "hard_miss_regret": float(topk_regret),
            "hard_negative_topk": int(hard_negative_topk),
            "hard_negative_count": int(len(hard_negative_indices)),
            "model_top_indices": _json_list(ranked),
            "model_top_scores": _json_list(float(state_scores[i]) for i in ranked),
            "model_top_teacher_ev": _json_list(float(state_teacher_ev[i]) for i in ranked),
            "model_top_delta_vs_baseline": _json_list(float(state_delta_baseline[i]) for i in ranked),
            "hard_negative_indices": _json_list(hard_negative_indices),
        }
        for k in topk_values:
            topk = topk_by_value[k]
            topk_set = set(topk)
            best_in_topk = best in topk_set
            best_ev_in_topk = max((float(state_teacher_ev[i]) for i in topk), default=-math.inf)
            base_row[f"oracle_best_in_top{k}"] = int(best_in_topk)
            base_row[f"top{k}_oracle_regret"] = float(best_ev - best_ev_in_topk) if math.isfinite(best_ev_in_topk) else 0.0
        state_rows.append(base_row)

        if hard_miss:
            hard_miss_rows.append(
                base_row
                | {
                    "schema": "hu_turn2_stage9_candidate_generator_hard_miss_v1",
                    "recommended_training_use": "candidate_generator_hard_miss",
                    "oracle_best_row_index": int(start + best),
                    "best_topk_row_index": int(start + max(hard_miss_set, key=lambda i: state_teacher_ev[i])) if hard_miss_set else -1,
                }
            )
            training_rows.append(
                {
                    "schema": "hu_turn2_stage9_candidate_generator_training_row_v1",
                    "state_index": int(state_index),
                    "action_index": best,
                    "action_row_index": int(start + best),
                    "label": "missed_oracle_positive",
                    "target": 1,
                    "weight": float(missed_positive_weight),
                    "target_ev": best_ev,
                    "delta_vs_baseline": float(state_delta_baseline[best]),
                    "split": split_name_for_id(split[state_index]),
                }
            )

        for hard_negative in hard_negative_indices:
            hard_negative_rows.append(
                base_row
                | {
                    "schema": "hu_turn2_stage9_candidate_generator_hard_negative_v1",
                    "recommended_training_use": "candidate_generator_hard_negative",
                    "hard_negative_index": int(hard_negative),
                    "hard_negative_row_index": int(start + hard_negative),
                    "hard_negative_ev": float(state_teacher_ev[hard_negative]),
                    "hard_negative_delta_vs_baseline": float(state_delta_baseline[hard_negative]),
                    "hard_negative_loss_vs_baseline": float(max(0.0, -state_delta_baseline[hard_negative])),
                }
            )
            training_rows.append(
                {
                    "schema": "hu_turn2_stage9_candidate_generator_training_row_v1",
                    "state_index": int(state_index),
                    "action_index": int(hard_negative),
                    "action_row_index": int(start + hard_negative),
                    "label": "model_topk_hard_negative",
                    "target": 0,
                    "weight": float(hard_negative_weight),
                    "target_ev": float(state_teacher_ev[hard_negative]),
                    "delta_vs_baseline": float(state_delta_baseline[hard_negative]),
                    "split": split_name_for_id(split[state_index]),
                }
            )

        training_rows.append(
            {
                "schema": "hu_turn2_stage9_candidate_generator_training_row_v1",
                "state_index": int(state_index),
                "action_index": best,
                "action_row_index": int(start + best),
                "label": "oracle_best",
                "target": 1,
                "weight": float(oracle_best_weight),
                "target_ev": best_ev,
                "delta_vs_baseline": float(state_delta_baseline[best]),
                "split": split_name_for_id(split[state_index]),
            }
        )

    return {
        "state_rows": state_rows,
        "hard_miss_rows": hard_miss_rows,
        "hard_negative_rows": hard_negative_rows,
        "training_rows": training_rows,
    }


def metric_rows(state_rows: list[dict[str, Any]], topk_values: Iterable[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    topk_values = parse_topk_values(topk_values)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in state_rows:
        groups[("all", "all")].append(row)
        groups[("split", str(row.get("split", "unknown")))].append(row)
        groups[("seat", str(row.get("seat", "unknown")))].append(row)
        groups[("bucket_group", str(row.get("bucket_group", "unknown")))].append(row)
        groups[("pilot_gate_label", str(row.get("pilot_gate_label", "unknown")))].append(row)

    for (group_type, group), values in sorted(groups.items()):
        count = len(values)
        row: dict[str, Any] = {
            "group_type": group_type,
            "group": group,
            "states": count,
            "hard_miss_count": sum(int(v.get("hard_miss", 0)) for v in values),
            "hard_miss_rate": sum(int(v.get("hard_miss", 0)) for v in values) / count if count else 0.0,
            "hard_negative_states": sum(1 for v in values if int(v.get("hard_negative_count", 0)) > 0),
            "hard_negative_state_rate": sum(1 for v in values if int(v.get("hard_negative_count", 0)) > 0) / count if count else 0.0,
            "top1_avg_regret": float(np.mean([float(v.get("model_top1_regret", 0.0)) for v in values])) if values else 0.0,
            "top1_p95_regret": float(np.quantile([float(v.get("model_top1_regret", 0.0)) for v in values], 0.95)) if values else 0.0,
        }
        for k in topk_values:
            recall_values = [int(v.get(f"oracle_best_in_top{k}", 0)) for v in values]
            regret_values = [float(v.get(f"top{k}_oracle_regret", 0.0)) for v in values]
            row[f"top{k}_oracle_recall"] = sum(recall_values) / count if count else 0.0
            row[f"top{k}_avg_regret"] = float(np.mean(regret_values)) if regret_values else 0.0
            row[f"top{k}_p95_regret"] = float(np.quantile(regret_values, 0.95)) if regret_values else 0.0
        rows.append(row)
    return rows


def write_summary(path: Path, *, manifest: dict[str, Any], metrics: list[dict[str, Any]]) -> None:
    overall = next((row for row in metrics if row["group_type"] == "all"), {})
    topk_values = manifest["topk_values"]
    lines = [
        "# HU T2 Stage9 Candidate Generator Cache",
        "",
        "This artifact is for candidate-generator redesign. It is not a production runtime gate.",
        "",
        "## Inputs",
        "",
        f"- cache: `{manifest['cache_dir']}`",
        f"- model: `{manifest['model']}`",
        f"- score head: `{manifest['score_head']}`",
        f"- states: `{manifest['states']}`",
        f"- actions: `{manifest['actions']}`",
        "",
        "## Overall Metrics",
        "",
        f"- hard miss states: `{overall.get('hard_miss_count', 0)}` / `{overall.get('states', 0)}`",
        f"- hard negative states: `{overall.get('hard_negative_states', 0)}` / `{overall.get('states', 0)}`",
        f"- top1 avg regret: `{float(overall.get('top1_avg_regret', 0.0)):.4f}`",
    ]
    for k in topk_values:
        lines.append(
            f"- Top{k} oracle recall / avg regret: "
            f"`{float(overall.get(f'top{k}_oracle_recall', 0.0)):.4f}` / "
            f"`{float(overall.get(f'top{k}_avg_regret', 0.0)):.4f}`"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `candidate_generator_state_rows.csv`",
            "- `hard_miss_states.jsonl`",
            "- `hard_negative_actions.jsonl`",
            "- `candidate_generator_training_rows.jsonl`",
            "- `candidate_generator_metrics.csv`",
            "- `candidate_generator_manifest.json`",
            "",
            "## Next Step",
            "",
            "Train a Stage9 candidate generator using the full all-action cache with extra weight on "
            "`missed_oracle_positive` and `model_topk_hard_negative` rows, then re-run this audit on heldout.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    topk_values = parse_topk_values(args.topk)
    cache = load_cache(args.cache_dir)
    scoring = scoring_metadata_status(cache["metadata"])
    if not scoring["training_allowed"]:
        if scoring["status"] == "missing" and not args.allow_missing_scoring_metadata:
            raise SystemExit(f"cache scoring metadata missing; pass --allow-missing-scoring-metadata to override: {scoring}")
        if scoring["status"] == "mismatch" and not args.allow_scoring_mismatch:
            raise SystemExit(f"cache scoring metadata mismatch; pass --allow-scoring-mismatch to override: {scoring}")
    torch, net, stats, device = load_stage8_pilot_model(args.model, device_choice=args.device)
    predictions = predict_all(torch, net, cache, stats, device, args.batch_action_rows)
    built = build_candidate_rows(
        cache,
        predictions,
        score_head=args.score_head,
        topk_values=topk_values,
        hard_miss_topk=args.hard_miss_topk,
        hard_negative_topk=args.hard_negative_topk,
        min_hard_miss_regret=args.min_hard_miss_regret,
        min_hard_negative_loss=args.min_hard_negative_loss,
        missed_positive_weight=args.missed_positive_weight,
        hard_negative_weight=args.hard_negative_weight,
        oracle_best_weight=args.oracle_best_weight,
        max_states=args.max_states,
    )
    metrics = metric_rows(built["state_rows"], topk_values)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "candidate_generator_state_rows.csv", built["state_rows"])
    write_jsonl(output_dir / "hard_miss_states.jsonl", built["hard_miss_rows"])
    write_jsonl(output_dir / "hard_negative_actions.jsonl", built["hard_negative_rows"])
    write_jsonl(output_dir / "candidate_generator_training_rows.jsonl", built["training_rows"])
    write_csv(output_dir / "candidate_generator_metrics.csv", metrics)
    manifest = {
        "schema": "hu_turn2_stage9_candidate_generator_cache_v1",
        "cache_dir": str(args.cache_dir),
        "model": str(args.model),
        "output_dir": str(output_dir),
        "score_head": args.score_head,
        "topk_values": topk_values,
        "hard_miss_topk": int(args.hard_miss_topk),
        "hard_negative_topk": int(args.hard_negative_topk),
        "min_hard_miss_regret": float(args.min_hard_miss_regret),
        "min_hard_negative_loss": float(args.min_hard_negative_loss),
        "missed_positive_weight": float(args.missed_positive_weight),
        "hard_negative_weight": float(args.hard_negative_weight),
        "oracle_best_weight": float(args.oracle_best_weight),
        "states": len(built["state_rows"]),
        "actions": int(cache["metadata"]["action_count"]),
        "hard_miss_rows": len(built["hard_miss_rows"]),
        "hard_negative_rows": len(built["hard_negative_rows"]),
        "training_rows": len(built["training_rows"]),
        "scoring_metadata_status": scoring,
        "primary_metric": "TopK oracle recall and TopK oracle regret; no seat-swap claim",
    }
    (output_dir / "candidate_generator_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_summary(output_dir / "candidate_generator_summary.md", manifest=manifest, metrics=metrics)
    print(json.dumps(manifest, sort_keys=True, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
