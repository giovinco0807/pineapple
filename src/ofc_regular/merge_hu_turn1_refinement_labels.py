"""Merge stronger HU Turn1 refinement labels into a base teacher JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--refinement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
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


def merge_records(
    base_rows: list[dict[str, Any]],
    refinement_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    refinement_by_key: dict[str, dict[str, Any]] = {}
    duplicate_refinement_keys = 0
    for row in refinement_rows:
        key = state_key(row)
        if key in refinement_by_key:
            duplicate_refinement_keys += 1
        refinement_by_key[key] = row

    merged: list[dict[str, Any]] = []
    replaced = 0
    base_keys: Counter[str] = Counter()
    label_source_counts: Counter[str] = Counter()
    for row in base_rows:
        key = state_key(row)
        base_keys[key] += 1
        replacement = refinement_by_key.get(key)
        if replacement is None:
            output = dict(row)
            output["label_source"] = "base_fast_t2"
        else:
            output = dict(replacement)
            output["label_source"] = "stage9f_p2_refinement"
            output["base_schema"] = row.get("schema")
            output["base_profile"] = row.get("profile")
            output["base_opponent_profile"] = row.get("opponent_profile")
            output["base_t2_continuation_profile"] = row.get("t2_continuation_profile")
            replaced += 1
        output["schema"] = "hu_turn1_stage1_merged_teacher_v1"
        output["state_key"] = key
        label_source_counts[str(output["label_source"])] += 1
        merged.append(output)

    duplicate_base_keys = sum(count - 1 for count in base_keys.values() if count > 1)
    missing_refinement_keys = sorted(set(refinement_by_key) - set(base_keys))
    summary = {
        "schema": "hu_turn1_stage1_merged_teacher_summary_v1",
        "base_records": len(base_rows),
        "refinement_records": len(refinement_rows),
        "output_records": len(merged),
        "replaced_records": replaced,
        "duplicate_base_keys": duplicate_base_keys,
        "duplicate_refinement_keys": duplicate_refinement_keys,
        "missing_refinement_keys": len(missing_refinement_keys),
        "label_source_counts": dict(sorted(label_source_counts.items())),
    }
    return merged, summary


def main() -> None:
    args = parse_args()
    base_rows = read_jsonl(args.base)
    refinement_rows = read_jsonl(args.refinement)
    merged, summary = merge_records(base_rows, refinement_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
