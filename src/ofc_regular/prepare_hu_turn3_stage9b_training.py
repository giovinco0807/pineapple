"""Build a balanced HU Turn3 Stage9b teacher dataset.

The Stage9b data keeps full joint-exact teacher samples for every state. Runtime
false-positive fires confirmed by joint-exact MC are tagged with a dedicated
source so existing trainers can up-weight them without changing target format.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def hard_negative_keys(rows: Sequence[dict[str, Any]]) -> set[tuple[str, int, int]]:
    out: set[tuple[str, int, int]] = set()
    for row in rows:
        state_id = str(row.get("state_id") or "")
        if not state_id:
            continue
        try:
            baseline_index = int(row.get("baseline_index"))
            hu_index = int(row.get("hu_index"))
        except (TypeError, ValueError):
            continue
        out.add((state_id, baseline_index, hu_index))
    return out


def sample_key(sample: dict[str, Any]) -> tuple[str, int, int] | None:
    source_state = sample.get("source_state") or {}
    state_id = str(source_state.get("state_id") or "")
    selection = sample.get("selection") or {}
    if not state_id:
        return None
    try:
        baseline_index = int(selection.get("baseline_index"))
        hu_index = int(selection.get("hu_index"))
    except (TypeError, ValueError):
        return None
    return (state_id, baseline_index, hu_index)


def stable_sample_id(sample: dict[str, Any]) -> str:
    source_state = sample.get("source_state") or {}
    state_id = source_state.get("state_id")
    if state_id is not None:
        source_input = source_state.get("source_input_path") or sample.get("source_input_path") or ""
        return f"{source_input}:{state_id}"
    payload = {
        "board": sample.get("board"),
        "opponent_board": sample.get("opponent_board"),
        "dealt": sample.get("dealt"),
        "dead_cards": sample.get("dead_cards"),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def build_stage9b_samples(
    *,
    base_samples: Sequence[dict[str, Any]],
    runtime_samples: Sequence[dict[str, Any]],
    hard_negative_rows: Sequence[dict[str, Any]],
    dedupe: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hard_keys = hard_negative_keys(hard_negative_rows)
    output: list[dict[str, Any]] = []
    seen_base_ids: set[str] = set()
    source_counts: Counter[str] = Counter()
    runtime_hard_negative_count = 0
    runtime_nonnegative_count = 0

    def append_sample(sample: dict[str, Any], source: str | None = None) -> None:
        row = dict(sample)
        if source is not None:
            row["source"] = source
        output.append(row)
        source_counts[str(row.get("source", "unknown"))] += 1

    for sample in base_samples:
        if dedupe:
            key = stable_sample_id(sample)
            if key in seen_base_ids:
                continue
            seen_base_ids.add(key)
        append_sample(sample)

    for sample in runtime_samples:
        key = sample_key(sample)
        if key is not None and key in hard_keys:
            append_sample(sample, "stage9b_runtime_hard_negative")
            runtime_hard_negative_count += 1
        else:
            append_sample(sample, "stage9b_runtime_fire_non_negative")
            runtime_nonnegative_count += 1

    summary = {
        "base_samples": len(base_samples),
        "runtime_samples": len(runtime_samples),
        "hard_negative_rows": len(hard_negative_rows),
        "hard_negative_keys": len(hard_keys),
        "output_samples": len(output),
        "runtime_hard_negative_samples": runtime_hard_negative_count,
        "runtime_fire_non_negative_samples": runtime_nonnegative_count,
        "source_counts": dict(source_counts),
    }
    return output, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-input", type=Path, action="append", required=True)
    parser.add_argument("--runtime-input", type=Path, action="append", required=True)
    parser.add_argument("--hard-negatives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hard-negative-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--no-dedupe", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_samples = [row for path in args.base_input for row in read_jsonl(path)]
    runtime_samples = [row for path in args.runtime_input for row in read_jsonl(path)]
    hard_rows = read_jsonl(args.hard_negatives)
    samples, summary = build_stage9b_samples(
        base_samples=base_samples,
        runtime_samples=runtime_samples,
        hard_negative_rows=hard_rows,
        dedupe=not args.no_dedupe,
    )
    write_jsonl(args.output, samples)
    if args.hard_negative_output is not None:
        write_jsonl(
            args.hard_negative_output,
            [sample for sample in samples if sample.get("source") == "stage9b_runtime_hard_negative"],
        )
    summary.update(
        {
            "base_inputs": [str(path) for path in args.base_input],
            "runtime_inputs": [str(path) for path in args.runtime_input],
            "hard_negatives": str(args.hard_negatives),
            "output": str(args.output),
            "hard_negative_output": str(args.hard_negative_output) if args.hard_negative_output else None,
        }
    )
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, separators=(",", ":")))


if __name__ == "__main__":
    main()
