"""Append HU Turn1 runtime relabels to a base teacher JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--addition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument(
        "--addition-label-source",
        default="stage9f_p2_runtime_relabel",
        help="label_source assigned to appended rows.",
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=("error", "skip", "replace"),
        default="error",
        help="How to handle addition rows whose state_key already exists in base.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def state_key(record: dict[str, Any]) -> str:
    if record.get("state_key"):
        return str(record["state_key"])
    payload = {
        "hand_seed": record.get("hand_seed"),
        "player": record.get("player"),
        "seat": record.get("seat"),
        "board": record.get("board"),
        "opponent_board": record.get("opponent_board"),
        "dealt": record.get("dealt"),
        "dead_cards": record.get("dead_cards"),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str(row.get("label_source") or row.get("source") or row.get("profile") or "unknown")] += 1
    return dict(sorted(counts.items()))


def augment_records(
    base_rows: list[dict[str, Any]],
    addition_rows: list[dict[str, Any]],
    *,
    addition_label_source: str,
    duplicate_policy: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_by_key: dict[str, dict[str, Any]] = {}
    duplicate_base_keys = 0
    for row in base_rows:
        key = state_key(row)
        if key in output_by_key:
            duplicate_base_keys += 1
        out = dict(row)
        out["state_key"] = key
        output_by_key[key] = out

    appended = 0
    skipped_duplicates = 0
    replaced_duplicates = 0
    duplicate_addition_keys = 0
    seen_addition: set[str] = set()
    for row in addition_rows:
        key = state_key(row)
        if key in seen_addition:
            duplicate_addition_keys += 1
        seen_addition.add(key)

        out = dict(row)
        out["schema"] = "hu_turn1_stage1_augmented_teacher_v1"
        out["state_key"] = key
        out["label_source"] = addition_label_source
        out["source_schema"] = row.get("schema")

        if key in output_by_key:
            if duplicate_policy == "error":
                raise ValueError(f"addition state_key already exists in base: {key}")
            if duplicate_policy == "skip":
                skipped_duplicates += 1
                continue
            replaced_duplicates += 1
        else:
            appended += 1
        output_by_key[key] = out

    output_rows = list(output_by_key.values())
    summary = {
        "schema": "hu_turn1_stage1_augmented_teacher_summary_v1",
        "base_records": len(base_rows),
        "addition_records": len(addition_rows),
        "output_records": len(output_rows),
        "appended_records": appended,
        "skipped_duplicate_records": skipped_duplicates,
        "replaced_duplicate_records": replaced_duplicates,
        "duplicate_base_keys": duplicate_base_keys,
        "duplicate_addition_keys": duplicate_addition_keys,
        "duplicate_policy": duplicate_policy,
        "addition_label_source": addition_label_source,
        "label_source_counts": source_counts(output_rows),
    }
    return output_rows, summary


def main() -> None:
    args = parse_args()
    base_rows = read_jsonl(args.base)
    addition_rows = read_jsonl(args.addition)
    try:
        output_rows, summary = augment_records(
            base_rows,
            addition_rows,
            addition_label_source=args.addition_label_source,
            duplicate_policy=args.duplicate_policy,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
