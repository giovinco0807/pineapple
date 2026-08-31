"""Attach source/teacher board context to mined T2 miss rows.

The mining rows used for miss-weighting intentionally stay small:
``dataset``, ``group_id``, ``target_k`` and loss fields.  For generating more
similar cases, we also need the original board/dealt/opponent context and the
exact teacher action.  This helper keeps the original row shape intact and adds
compact ``source_context`` / ``teacher_context`` objects.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.training.mine_t2_pool_model_disagreements import (  # noqa: E402
    compact_source_context,
    compact_teacher_context,
)
from ai.training.train_t2_selector_feature_ranker import parse_named_path  # noqa: E402


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_record_data(values: list[str]) -> dict[str, list[dict]]:
    records: dict[str, list[dict]] = {}
    for value in values:
        spec = parse_named_path(value)
        records[spec.name] = list(iter_jsonl(spec.path))
    return records


def enrich(args: argparse.Namespace) -> dict:
    source_records = load_record_data(args.source_data)
    teacher_records = load_record_data(args.teacher_data)
    input_path = Path(args.rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_in = 0
    rows_out = 0
    source_hits = 0
    teacher_hits = 0
    with input_path.open("r", encoding="utf-8-sig") as src, output_path.open("w", encoding="utf-8") as out:
        for line in src:
            if not line.strip():
                continue
            rows_in += 1
            row = json.loads(line)
            dataset = str(row.get("dataset"))
            group_id = int(row.get("group_id"))
            best_local = row.get("teacher_best_local_index")
            if dataset in source_records and group_id < len(source_records[dataset]):
                row["source_context"] = compact_source_context(source_records[dataset][group_id])
                source_hits += 1
            if dataset in teacher_records and group_id < len(teacher_records[dataset]):
                row["teacher_context"] = compact_teacher_context(
                    teacher_records[dataset][group_id],
                    int(best_local) if best_local is not None else None,
                )
                teacher_hits += 1
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            rows_out += 1

    summary = {
        "rows": str(input_path),
        "output": str(output_path),
        "rows_in": rows_in,
        "rows_out": rows_out,
        "source_data": {name: len(records) for name, records in source_records.items()},
        "teacher_data": {name: len(records) for name, records in teacher_records.items()},
        "source_hits": source_hits,
        "teacher_hits": teacher_hits,
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, help="Input mined rows JSONL")
    parser.add_argument("--source-data", action="append", default=[], help="Original source JSONL, name=path")
    parser.add_argument("--teacher-data", action="append", default=[], help="Exact teacher JSONL, name=path")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(enrich(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
