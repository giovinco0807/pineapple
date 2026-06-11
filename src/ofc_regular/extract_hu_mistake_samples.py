"""Extract HU Turn3 teacher samples where a model makes high-regret choices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .hu_turn3_model import load_hu_action_value_model, sample_to_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--min-regret", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--require-topk-miss", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--tie-tolerance", type=float, default=1e-9)
    parser.add_argument("--examples", type=int, default=20)
    return parser.parse_args()


def action_text(action: dict[str, Any]) -> str:
    placements = " ".join(f"{card}->{row}" for card, row in action.get("placements", ()))
    discards = ",".join(action.get("discards", ())) or "-"
    return f"{placements}; discard {discards}"


def analyze_sample(
    sample: dict[str, Any],
    *,
    model: Any,
    line_index: int,
    top_k: int,
    tie_tolerance: float,
) -> dict[str, Any]:
    features, targets = sample_to_matrix(sample)
    predictions = model.predict_matrix(features)
    if targets.size == 0:
        raise ValueError(f"line {line_index} has no actions")
    predicted_index = int(np.argmax(predictions))
    true_best_score = float(np.max(targets))
    predicted_true_score = float(targets[predicted_index])
    regret = true_best_score - predicted_true_score
    k = min(top_k, int(targets.size))
    top_indices = [int(index) for index in np.argsort(predictions)[-k:][::-1]]
    topk_hit = any(float(targets[index]) >= true_best_score - tie_tolerance for index in top_indices)
    true_best_indices = [
        int(index)
        for index, target in enumerate(targets)
        if float(target) >= true_best_score - tie_tolerance
    ]
    return {
        "line_index": int(line_index),
        "sample_id": sample.get("sample_id"),
        "seat": sample.get("seat"),
        "to_act_order": sample.get("to_act_order"),
        "dealt": sample.get("dealt"),
        "actions_count": int(targets.size),
        "score_gap": float(sample.get("score_gap", 0.0)),
        "true_best_score": true_best_score,
        "predicted_index": predicted_index,
        "predicted_true_score": predicted_true_score,
        "predicted_ev": float(predictions[predicted_index]),
        "top1_regret": float(regret),
        "topk": int(k),
        "topk_hit": bool(topk_hit),
        "topk_miss": not topk_hit,
        "true_best": [
            {
                "index": int(index),
                "score": float(targets[index]),
                "predicted_ev": float(predictions[index]),
                "text": action_text(sample["actions"][index]),
                "next_board": sample["actions"][index].get("next_board"),
            }
            for index in true_best_indices[:3]
        ],
        "model_topk": [
            {
                "rank": rank + 1,
                "index": int(index),
                "score": float(targets[index]),
                "predicted_ev": float(predictions[index]),
                "regret": float(true_best_score - float(targets[index])),
                "text": action_text(sample["actions"][index]),
                "next_board": sample["actions"][index].get("next_board"),
            }
            for rank, index in enumerate(top_indices)
        ],
    }


def keep_sample(
    analysis: dict[str, Any],
    *,
    min_regret: float,
    require_topk_miss: bool,
) -> bool:
    if float(analysis["top1_regret"]) < min_regret:
        return False
    if require_topk_miss and not bool(analysis["topk_miss"]):
        return False
    return True


def main() -> None:
    args = parse_args()
    if args.min_regret < 0:
        raise SystemExit("--min-regret must be non-negative")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    if args.examples < 0:
        raise SystemExit("--examples must be non-negative")

    model = load_hu_action_value_model(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0
    topk_misses = 0
    regret_sum = 0.0
    kept_regret_sum = 0.0
    examples: list[dict[str, Any]] = []
    with args.teacher_data.open("r", encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as output:
        for line_index, line in enumerate(source):
            if args.max_samples is not None and total >= args.max_samples:
                break
            sample = json.loads(line)
            analysis = analyze_sample(
                sample,
                model=model,
                line_index=line_index,
                top_k=args.top_k,
                tie_tolerance=args.tie_tolerance,
            )
            total += 1
            regret_sum += float(analysis["top1_regret"])
            topk_misses += int(bool(analysis["topk_miss"]))
            if not keep_sample(
                analysis,
                min_regret=args.min_regret,
                require_topk_miss=args.require_topk_miss,
            ):
                continue
            kept = dict(sample)
            kept["source"] = "hu_turn3_mistake_mined"
            kept["mistake_mining"] = {
                "source_teacher_data": str(args.teacher_data),
                "model": str(args.model),
                "line_index": int(line_index),
                "top1_regret": float(analysis["top1_regret"]),
                "topk": int(analysis["topk"]),
                "topk_miss": bool(analysis["topk_miss"]),
                "predicted_index": int(analysis["predicted_index"]),
            }
            output.write(json.dumps(kept, separators=(",", ":")) + "\n")
            written += 1
            kept_regret_sum += float(analysis["top1_regret"])
            examples.append(analysis)

    examples.sort(key=lambda item: float(item["top1_regret"]), reverse=True)
    summary = {
        "teacher_data": str(args.teacher_data),
        "model": str(args.model),
        "output": str(args.output),
        "samples_read": total,
        "samples_written": written,
        "min_regret": float(args.min_regret),
        "top_k": int(args.top_k),
        "require_topk_miss": bool(args.require_topk_miss),
        "topk_miss_count": topk_misses,
        "topk_miss_rate": float(topk_misses / total) if total else 0.0,
        "avg_top1_regret": float(regret_sum / total) if total else 0.0,
        "kept_avg_top1_regret": float(kept_regret_sum / written) if written else 0.0,
        "examples_sorted_by_top1_regret": examples[: args.examples],
    }
    text = json.dumps(summary, indent=2) + "\n"
    if args.summary_output is not None:
        args.summary_output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
