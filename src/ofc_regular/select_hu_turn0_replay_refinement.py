"""Select a fixed HU T0 MC512 refinement set from broad MC32 replay."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def select_refinement_targets(
    rows: Sequence[dict[str, Any]],
    *,
    gray_top_n: int = 150,
    hard_negative_min_margin: float = 2.0,
    strict_source: str = "p0_strict_f30",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    first_rows = [row for row in rows if str(row.get("seat")) == "first"]
    selected: dict[str, dict[str, Any]] = {}
    reasons: dict[str, set[str]] = {}

    def add(row: dict[str, Any], reason: str) -> None:
        target_id = str(row["target_id"])
        selected[target_id] = row
        reasons.setdefault(target_id, set()).add(reason)

    for row in first_rows:
        label = str(row.get("safe_override_label"))
        if label == "positive":
            add(row, "all_mc32_positive")
        if str(row.get("source_config_id")) == strict_source:
            add(row, "all_strict_source")
        if label == "negative" and float(row.get("predicted_margin", 0.0)) >= hard_negative_min_margin:
            add(row, "high_margin_hard_negative")

    gray_rows = sorted(
        (row for row in first_rows if str(row.get("safe_override_label")) == "gray"),
        key=lambda row: (
            -float(row.get("candidate_delta_lcb196", float("-inf"))),
            str(row["target_id"]),
        ),
    )[:gray_top_n]
    for row in gray_rows:
        add(row, "top_gray_by_lcb196")

    output: list[dict[str, Any]] = []
    for target_id in sorted(selected):
        row = dict(selected[target_id])
        row["refinement_selection_reasons"] = sorted(reasons[target_id])
        row["refinement_source_future_samples"] = int(row["future_samples"])
        output.append(row)
    summary = {
        "schema": "hu_turn0_replay_refinement_selection_v1",
        "input_rows": len(rows),
        "first_rows": len(first_rows),
        "selected_rows": len(output),
        "gray_top_n": int(gray_top_n),
        "hard_negative_min_margin": float(hard_negative_min_margin),
        "strict_source": strict_source,
        "label_counts": dict(
            sorted(Counter(str(row["safe_override_label"]) for row in output).items())
        ),
        "source_counts": dict(
            sorted(Counter(str(row["source_config_id"]) for row in output).items())
        ),
        "reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for row in output
                    for reason in row["refinement_selection_reasons"]
                ).items()
            )
        ),
        "duplicate_target_count": len(output) - len({row["target_id"] for row in output}),
    }
    return output, summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--gray-top-n", type=int, default=150)
    parser.add_argument("--hard-negative-min-margin", type=float, default=2.0)
    parser.add_argument("--strict-source", default="p0_strict_f30")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    rows, summary = select_refinement_targets(
        _read_jsonl(args.input),
        gray_top_n=args.gray_top_n,
        hard_negative_min_margin=args.hard_negative_min_margin,
        strict_source=args.strict_source,
    )
    _write_jsonl(args.output, rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
