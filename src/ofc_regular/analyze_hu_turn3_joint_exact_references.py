"""Compare HU Turn3 sequential-belief teacher values to recorded selections.

The v2 ``score`` field is an independent evaluation-batch estimate under the
recorded continuation policy.  It is not realized match EV and must not be
used to retune a selection threshold on the same rows.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="hu_turn3_sequential_belief_v2 teacher JSONL.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--delta-threshold",
        type=float,
        default=0.25,
        help="Minimum exact EV delta counted as materially positive/negative.",
    )
    return parser.parse_args()


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


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def action_by_original_index(sample: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {safe_int(action.get("original_index")): action for action in sample.get("actions", ())}


def analyze_sample(sample: dict[str, Any], *, delta_threshold: float) -> dict[str, Any] | None:
    actions = list(sample.get("actions", ()))
    selection = sample.get("selection") or {}
    if not actions or not selection:
        return None
    by_index = action_by_original_index(sample)
    best = actions[0]
    best_index = safe_int(best.get("original_index"))
    best_score = safe_float(best.get("score"))
    baseline_index = safe_int(selection.get("baseline_index"))
    hu_index = safe_int(selection.get("hu_index"))
    compare_hu_index = safe_int(selection.get("compare_hu_index"), default=-1)
    baseline = by_index.get(baseline_index)
    hu = by_index.get(hu_index)
    compare_hu = by_index.get(compare_hu_index)
    if baseline is None or hu is None:
        return None
    baseline_score = safe_float(baseline.get("score"))
    hu_score = safe_float(hu.get("score"))
    compare_hu_score = safe_float(compare_hu.get("score")) if compare_hu is not None else 0.0
    delta_hu_vs_baseline = hu_score - baseline_score
    row = {
        "sample_id": sample.get("sample_id"),
        "source": sample.get("source", "unknown"),
        "source_input_path": sample.get("source_input_path")
        or (sample.get("source_state") or {}).get("source_input_path"),
        "state_id": (sample.get("source_state") or {}).get("state_id"),
        "hand_seed": (sample.get("source_state") or {}).get("hand_seed"),
        "hand_index": (sample.get("source_state") or {}).get("hand_index"),
        "seat": sample.get("seat"),
        "to_act_order": sample.get("to_act_order"),
        "future_count": sample.get("future_count"),
        "legal_action_count": sample.get("legal_action_count"),
        "best_index": best_index,
        "baseline_index": baseline_index,
        "hu_index": hu_index,
        "compare_hu_index": compare_hu_index,
        "best_score": best_score,
        "baseline_score": baseline_score,
        "hu_score": hu_score,
        "compare_hu_score": compare_hu_score if compare_hu is not None else "",
        "baseline_regret": best_score - baseline_score,
        "hu_regret": best_score - hu_score,
        "compare_hu_regret": best_score - compare_hu_score if compare_hu is not None else "",
        "delta_hu_vs_baseline": delta_hu_vs_baseline,
        "delta_compare_hu_vs_baseline": compare_hu_score - baseline_score if compare_hu is not None else "",
        "best_is_baseline": int(best_index == baseline_index),
        "best_is_hu": int(best_index == hu_index),
        "hu_materially_positive": int(delta_hu_vs_baseline >= delta_threshold),
        "hu_materially_negative": int(delta_hu_vs_baseline <= -delta_threshold),
        "predicted_margin_vs_baseline": safe_float(selection.get("predicted_margin_vs_baseline")),
        "compare_predicted_margin_vs_baseline": safe_float(selection.get("compare_predicted_margin_vs_baseline")),
        "disagreement": int(bool(selection.get("disagreement"))),
        "compare_hu_disagreement": int(bool(selection.get("compare_hu_disagreement"))),
        "baseline_bust_rate": safe_float(baseline.get("bust_rate")),
        "hu_bust_rate": safe_float(hu.get("bust_rate")),
        "baseline_fl_entry_rate": safe_float(baseline.get("fl_entry_rate")),
        "hu_fl_entry_rate": safe_float(hu.get("fl_entry_rate")),
        "baseline_royalty_mean": safe_float(baseline.get("royalty_mean")),
        "hu_royalty_mean": safe_float(hu.get("royalty_mean")),
    }
    return row


def build_hu_loss_audit_rows(
    samples: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    limit: int = 30,
) -> list[dict[str, Any]]:
    sample_by_id = {sample.get("sample_id"): sample for sample in samples}
    losses = sorted(rows, key=lambda row: safe_float(row.get("delta_hu_vs_baseline")), reverse=False)
    audit_rows: list[dict[str, Any]] = []
    for row in losses[:limit]:
        sample = sample_by_id.get(row.get("sample_id"))
        if sample is None:
            continue
        by_index = action_by_original_index(sample)
        baseline = by_index.get(safe_int(row.get("baseline_index")))
        hu = by_index.get(safe_int(row.get("hu_index")))
        best = by_index.get(safe_int(row.get("best_index")))
        audit_rows.append(
            {
                "sample_id": row.get("sample_id"),
                "source": row.get("source"),
                "source_input_path": row.get("source_input_path"),
                "state_id": row.get("state_id"),
                "hand_seed": row.get("hand_seed"),
                "hand_index": row.get("hand_index"),
                "seat": row.get("seat"),
                "to_act_order": row.get("to_act_order"),
                "delta_hu_vs_baseline": safe_float(row.get("delta_hu_vs_baseline")),
                "baseline_regret": safe_float(row.get("baseline_regret")),
                "hu_regret": safe_float(row.get("hu_regret")),
                "predicted_margin_vs_baseline": safe_float(row.get("predicted_margin_vs_baseline")),
                "board": sample.get("board"),
                "opponent_board": sample.get("opponent_board"),
                "dealt": sample.get("dealt"),
                "dead_cards": sample.get("dead_cards"),
                "baseline_action": baseline,
                "hu_action": hu,
                "best_action": best,
                "selection": sample.get("selection"),
            }
        )
    return audit_rows


def summarize(rows: list[dict[str, Any]], *, delta_threshold: float) -> dict[str, Any]:
    if not rows:
        return {
            "samples": 0,
            "decision": "No-Go",
            "reason": "no_comparable_selection_rows",
        }
    deltas = [safe_float(row.get("delta_hu_vs_baseline")) for row in rows]
    baseline_regrets = [safe_float(row.get("baseline_regret")) for row in rows]
    hu_regrets = [safe_float(row.get("hu_regret")) for row in rows]
    positives = sum(1 for value in deltas if value >= delta_threshold)
    negatives = sum(1 for value in deltas if value <= -delta_threshold)
    zeros = len(deltas) - positives - negatives
    mean_delta = sum(deltas) / len(deltas)
    decision = "Needs larger validation" if mean_delta > 0.0 and positives >= negatives else "No-Go"
    return {
        "samples": len(rows),
        "delta_threshold": delta_threshold,
        "delta_hu_vs_baseline_mean": mean_delta,
        "delta_hu_vs_baseline_min": min(deltas),
        "delta_hu_vs_baseline_max": max(deltas),
        "hu_materially_positive": positives,
        "hu_materially_negative": negatives,
        "hu_near_tie": zeros,
        "best_is_baseline": sum(safe_int(row.get("best_is_baseline"), 0) for row in rows),
        "best_is_hu": sum(safe_int(row.get("best_is_hu"), 0) for row in rows),
        "baseline_regret_mean": sum(baseline_regrets) / len(baseline_regrets),
        "hu_regret_mean": sum(hu_regrets) / len(hu_regrets),
        "decision": decision,
        "reason": "joint_exact_selection_delta_summary",
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# HU T3 Joint Exact Reference Analysis",
        "",
        "This compares joint-exact T3 teacher EVs against recorded Stage3/Stage7-style selection indices. It is teacher diagnostics, not production evidence.",
        "",
        "## Summary",
        "",
        f"- comparable samples: `{summary['samples']}`",
        f"- mean HU-selection delta vs baseline: `{safe_float(summary.get('delta_hu_vs_baseline_mean')):.4f}`",
        f"- materially positive / negative / near tie: `{summary.get('hu_materially_positive')}` / `{summary.get('hu_materially_negative')}` / `{summary.get('hu_near_tie')}`",
        f"- mean baseline regret: `{safe_float(summary.get('baseline_regret_mean')):.4f}`",
        f"- mean HU-selection regret: `{safe_float(summary.get('hu_regret_mean')):.4f}`",
        f"- decision: `{summary.get('decision')}`",
        "",
        "## Interpretation",
        "",
        "- Positive HU-selection delta means the recorded HU selection beats the baseline under this joint-exact T3 teacher.",
        "- Negative delta means the HU selection is worse than baseline under this teacher and should become a hard audit target.",
        "- This analysis does not replace source-heldout seat-swap validation.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_file(input_path: Path, *, delta_threshold: float) -> dict[str, Any]:
    samples = read_jsonl(input_path)
    rows = [
        row
        for sample in samples
        if (row := analyze_sample(sample, delta_threshold=delta_threshold)) is not None
    ]
    summary = summarize(rows, delta_threshold=delta_threshold)
    return {
        "rows": rows,
        "summary": summary,
        "hu_loss_audit_rows": build_hu_loss_audit_rows(samples, rows),
    }


def main() -> None:
    args = parse_args()
    analysis = analyze_file(args.input, delta_threshold=args.delta_threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "joint_exact_reference_rows.csv", analysis["rows"])
    write_jsonl(args.output_dir / "joint_exact_hu_loss_top30.jsonl", analysis["hu_loss_audit_rows"])
    (args.output_dir / "joint_exact_reference_summary.json").write_text(
        json.dumps(analysis["summary"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.output_dir / "joint_exact_reference_summary.md", analysis["summary"])
    print(json.dumps(analysis["summary"], ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
