"""Relabel saved selector rows with stronger teacher labels for target states."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.tutor.compare_runtime_targets_to_teacher import align_records, iter_jsonl
from ai.tutor.evaluate_hybrid_refinement_teacher import canonical_action, teacher_index


def build_teacher_by_line(targets: list[dict[str, Any]], teachers: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    by_line: dict[int, dict[str, Any]] = {}
    for target, teacher_record in align_records(targets, teachers):
        line = target.get("runtime_result_line") or target.get("line")
        if line is None:
            raise SystemExit("target is missing runtime_result_line")
        teacher = teacher_index(teacher_record)
        if teacher["best_action"] is None:
            continue
        by_line[int(line)] = {
            "index": teacher,
            "eval_mode": teacher_record.get("eval_mode"),
            "source": teacher_record.get("source"),
            "source_line": teacher_record.get("source_line"),
        }
    return by_line


def relabel_row(row: dict[str, Any], teacher_info: dict[str, Any]) -> dict[str, Any]:
    teacher = teacher_info["index"]
    action = canonical_action(row.get("action"))
    out = dict(row)
    out["is_teacher_best"] = action == teacher["best_action"]
    out["teacher_score"] = teacher["score_by_action"].get(action)
    out["teacher_rank"] = teacher.get("rank_by_action", {}).get(action)
    out["teacher_best_score"] = teacher["best_score"]
    out["teacher_margin"] = teacher["margin"]
    out["teacher_samples"] = teacher["samples"]
    out["teacher_eval_mode"] = teacher_info.get("eval_mode")
    out["teacher_source"] = teacher_info.get("source")
    out["teacher_source_line"] = teacher_info.get("source_line")
    out["selector_row_source"] = out.get("selector_row_source") or "relabel_selector_rows_with_teacher"
    return out


def relabel(args: argparse.Namespace) -> dict[str, Any]:
    targets = list(iter_jsonl(Path(args.targets)))
    teachers = list(iter_jsonl(Path(args.teacher_labels)))
    teacher_by_line = build_teacher_by_line(targets, teachers)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "selector_rows": str(args.selector_rows),
        "targets": str(args.targets),
        "teacher_labels": str(args.teacher_labels),
        "output": str(output),
        "target_lines": len(teacher_by_line),
        "input_rows": 0,
        "written": 0,
        "skipped_line_not_target": 0,
        "skipped_unscored_action": 0,
        "groups": 0,
        "teacher_best_rows": 0,
    }
    groups_seen: set[int] = set()
    with output.open("w", encoding="utf-8") as dst:
        for row in iter_jsonl(Path(args.selector_rows)):
            stats["input_rows"] += 1
            line = int(row.get("line", 0))
            teacher_info = teacher_by_line.get(line)
            if teacher_info is None:
                stats["skipped_line_not_target"] += 1
                continue
            out = relabel_row(row, teacher_info)
            if out.get("teacher_score") is None and args.drop_unscored:
                stats["skipped_unscored_action"] += 1
                continue
            dst.write(json.dumps(out, ensure_ascii=False) + "\n")
            stats["written"] += 1
            groups_seen.add(line)
            if bool(out.get("is_teacher_best")):
                stats["teacher_best_rows"] += 1
    stats["groups"] = len(groups_seen)
    output.with_suffix(".summary.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return stats


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Relabel selector rows with stronger teacher labels")
    parser.add_argument("--selector-rows", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--teacher-labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--drop-unscored", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(relabel(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
