"""Extract Stage9b hard-negative rows from HU T3 joint-exact teacher output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or out in (float("inf"), float("-inf")):
        return default
    return out


def safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _action_by_original_index(sample: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {safe_int(action.get("original_index")): action for action in sample.get("actions", ())}


def extract_hard_negative_rows(
    samples: Iterable[dict[str, Any]],
    *,
    delta_threshold: float = -0.25,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        selection = sample.get("selection") or {}
        baseline_index = safe_int(selection.get("baseline_index"))
        hu_index = safe_int(selection.get("hu_index"))
        if baseline_index < 0 or hu_index < 0 or baseline_index == hu_index:
            continue
        actions = _action_by_original_index(sample)
        baseline = actions.get(baseline_index)
        hu = actions.get(hu_index)
        if baseline is None or hu is None:
            continue
        baseline_score = safe_float(baseline.get("score"))
        hu_score = safe_float(hu.get("score"))
        delta = hu_score - baseline_score
        if delta > delta_threshold:
            continue
        rows.append(
            {
                "schema": "hu_turn3_stage9b_hard_negative_v1",
                "sample_id": sample.get("sample_id"),
                "source": sample.get("source"),
                "source_input_path": sample.get("source_input_path"),
                "state_id": (sample.get("source_state") or {}).get("state_id"),
                "seed": (sample.get("source_state") or {}).get("seed"),
                "hand_seed": (sample.get("source_state") or {}).get("hand_seed"),
                "seat": sample.get("seat"),
                "to_act_order": sample.get("to_act_order"),
                "board": sample.get("board"),
                "opponent_board": sample.get("opponent_board"),
                "dealt": sample.get("dealt"),
                "dead_cards": sample.get("dead_cards"),
                "input_dead_cards": sample.get("input_dead_cards"),
                "baseline_index": baseline_index,
                "hu_index": hu_index,
                "best_index": safe_int(sample.get("best_action_original_index")),
                "baseline_score": baseline_score,
                "hu_score": hu_score,
                "delta_hu_vs_baseline": delta,
                "baseline_regret": safe_float(sample.get("best_score")) - baseline_score,
                "hu_regret": safe_float(sample.get("best_score")) - hu_score,
                "predicted_margin_vs_baseline": selection.get("predicted_margin_vs_baseline"),
                "reference_margin": selection.get("reference_margin"),
                "model_score": selection.get("model_score"),
                "hu_turn3_min_margin": selection.get("hu_turn3_min_margin"),
                "hu_turn3_reference_min_margin": selection.get("hu_turn3_reference_min_margin"),
                "future_count": sample.get("future_count"),
                "future_samples": sample.get("future_samples"),
                "future_digest": sample.get("future_digest"),
                "baseline_action": baseline,
                "hu_action": hu,
                "best_action": (sample.get("actions") or [{}])[0],
                "label": "hard_negative",
                "reason": "joint_exact_hu_worse_than_baseline",
            }
        )
    rows.sort(key=lambda row: safe_float(row.get("delta_hu_vs_baseline")))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="hu_turn3_joint_exact_v1 JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Output hard-negative JSONL.")
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--delta-threshold", type=float, default=-0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = read_jsonl(args.input)
    rows = extract_hard_negative_rows(samples, delta_threshold=args.delta_threshold)
    write_jsonl(args.output, rows)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "samples": len(samples),
        "hard_negatives": len(rows),
        "delta_threshold": args.delta_threshold,
        "mean_delta": sum(safe_float(row.get("delta_hu_vs_baseline")) for row in rows) / len(rows)
        if rows
        else 0.0,
        "worst_delta": safe_float(rows[0].get("delta_hu_vs_baseline")) if rows else None,
    }
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, separators=(",", ":")))


if __name__ == "__main__":
    main()
