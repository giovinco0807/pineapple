"""Prepare HU Turn1 Stage4 safe-selector targets from runtime decision logs."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        required=True,
        help="Decision JSONL. Repeat to merge several fresh validation runs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--top-loss-output", type=Path)
    parser.add_argument("--positive-delta", type=float, default=2.0)
    parser.add_argument("--hard-negative-delta", type=float, default=-5.0)
    parser.add_argument("--top-losses", type=int, default=100)
    parser.add_argument(
        "--include-below-safe-selector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include non-fired rows that reached and failed the safe selector.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
    return rows


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def action_signature(action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return ""
    placements = sorted((str(card), str(row)) for card, row in action.get("placements", ()))
    discards = sorted(str(card) for card in action.get("discards", ()))
    return json.dumps({"placements": placements, "discards": discards}, sort_keys=True, separators=(",", ":"))


def state_action_key(row: dict[str, Any]) -> str:
    payload = {
        "hand_seed": row.get("hand_seed", row.get("hand_id")),
        "seat_swap": row.get("seat_swap"),
        "seat": row.get("seat"),
        "hero_board": row.get("hero_board", row.get("board")),
        "opponent_board": row.get("opponent_board"),
        "cards_to_place": row.get("cards_to_place", row.get("dealt")),
        "candidate": action_signature(row.get("hu_turn1_action") or row.get("runtime_candidate_action")),
        "baseline": action_signature(row.get("baseline_action") or row.get("fallback_action")),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def label_for_delta(delta: float, *, positive_delta: float, hard_negative_delta: float) -> tuple[str, int]:
    if delta >= positive_delta:
        return "positive", 1
    if delta <= hard_negative_delta:
        return "hard_negative", 0
    return "gray", -1


def selector_reached(row: dict[str, Any], *, include_below_safe_selector: bool) -> bool:
    if row.get("override_fired"):
        return True
    if not include_below_safe_selector:
        return False
    return row.get("no_override_reason") == "below_safe_selector" or row.get("safe_selector_score") is not None


def action_differs(row: dict[str, Any]) -> bool:
    candidate = row.get("hu_turn1_action") or row.get("runtime_candidate_action")
    baseline = row.get("baseline_action") or row.get("fallback_action")
    return bool(action_signature(candidate) and action_signature(candidate) != action_signature(baseline))


def select_rows(
    inputs: list[Path],
    *,
    positive_delta: float,
    hard_negative_delta: float,
    include_below_safe_selector: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    seat_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    seen: set[str] = set()

    for input_path in inputs:
        source_name = input_path.parent.name
        for row in read_jsonl(input_path):
            if not row.get("realized_delta_valid"):
                skipped["invalid_realized_delta"] += 1
                continue
            if not selector_reached(row, include_below_safe_selector=include_below_safe_selector):
                skipped["selector_not_reached"] += 1
                continue
            if not action_differs(row):
                skipped["same_as_baseline"] += 1
                continue
            key = state_action_key(row)
            if key in seen:
                skipped["duplicate_state_action"] += 1
                continue
            seen.add(key)

            delta = finite_float(row.get("hu_turn1_realized_delta", row.get("realized_candidate_seat_delta")))
            label, label_id = label_for_delta(
                delta,
                positive_delta=positive_delta,
                hard_negative_delta=hard_negative_delta,
            )
            enriched = dict(row)
            enriched["schema"] = "hu_turn1_stage4_safe_selector_target_v1"
            enriched["source_schema"] = row.get("schema")
            enriched["stage4_source_run"] = source_name
            enriched["stage4_source_path"] = str(input_path)
            enriched["stage4_selection_reason"] = (
                "override_fired" if row.get("override_fired") else "below_safe_selector"
            )
            enriched["hu_turn1_realized_delta"] = delta
            enriched["hu_turn1_safe_override_label"] = label
            enriched["hu_turn1_safe_override_label_id"] = label_id
            enriched["hu_turn1_safe_override_label_policy"] = (
                f"positive_delta>={positive_delta:g};hard_negative_delta<={hard_negative_delta:g};gray_between"
            )
            enriched["stage4_state_action_key"] = key
            output.append(enriched)
            label_counts[label] += 1
            seat_counts[str(row.get("seat", ""))] += 1
            source_counts[source_name] += 1

    deltas = [finite_float(row.get("hu_turn1_realized_delta")) for row in output]
    summary = {
        "schema": "hu_turn1_stage4_safe_selector_target_summary_v1",
        "input_paths": [str(path) for path in inputs],
        "rows_written": len(output),
        "label_counts": dict(sorted(label_counts.items())),
        "seat_counts": dict(sorted(seat_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "skipped_counts": dict(sorted(skipped.items())),
        "positive_delta": positive_delta,
        "hard_negative_delta": hard_negative_delta,
        "mean_delta": sum(deltas) / len(deltas) if deltas else 0.0,
        "min_delta": min(deltas) if deltas else 0.0,
        "max_delta": max(deltas) if deltas else 0.0,
    }
    return output, summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    rows, summary = select_rows(
        args.input,
        positive_delta=args.positive_delta,
        hard_negative_delta=args.hard_negative_delta,
        include_below_safe_selector=args.include_below_safe_selector,
    )
    write_jsonl(args.output, rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.top_loss_output:
        losses = sorted(rows, key=lambda row: finite_float(row.get("hu_turn1_realized_delta")))
        write_jsonl(args.top_loss_output, losses[: max(args.top_losses, 0)])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
