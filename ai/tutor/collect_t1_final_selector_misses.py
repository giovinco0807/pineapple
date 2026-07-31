"""Collect T1 final-selection misses back into teacher/target JSONL files."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from ai.tutor.train_t1_final_selector import selector_score


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                yield line_no, json.loads(line)


def group_selector_rows(path: Path, *, min_teacher_margin: float) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for _line_no, row in iter_jsonl(path):
        if int(row.get("turn", -1)) != 1 or not bool(row.get("is_refined")):
            continue
        grouped[int(row.get("line", 0))].append(row)
    return {
        line_no: rows
        for line_no, rows in grouped.items()
        if rows and float(rows[0].get("teacher_margin", 0.0) or 0.0) >= float(min_teacher_margin)
    }


def pick_refined(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            float(row.get("refined_score", float("-inf"))),
            float(row.get("model_score", float("-inf"))),
            -int(row.get("model_rank", 999999)),
        ),
    )


def pick_selector(rows: list[dict[str, Any]], selector: dict[str, Any]) -> dict[str, Any]:
    return max(rows, key=lambda row: selector_score(row, selector))


def teacher_record_by_line(path: Path) -> dict[int, dict[str, Any]]:
    return {line_no: record for line_no, record in iter_jsonl(path)}


def target_from_teacher(record: dict[str, Any], *, source: str, source_line: int, reason: str) -> dict[str, Any]:
    return {
        "source": source,
        "source_line": source_line,
        "turn": record.get("turn"),
        "board": record.get("board"),
        "opponent_board": record.get("opponent_board"),
        "dealt": record.get("dealt"),
        "known_discards": record.get("known_discards"),
        "is_btn": record.get("is_btn", False),
        "is_fl": record.get("is_fl", False),
        "opp_is_fl": record.get("opp_is_fl", False),
        "chips_self": record.get("chips_self", 200),
        "chips_opponent": record.get("chips_opponent", 200),
        "requested_reason": reason,
    }


def collect(args: argparse.Namespace) -> dict[str, Any]:
    selector = json.loads(Path(args.selector).read_text(encoding="utf-8")) if args.selector else None
    groups = group_selector_rows(Path(args.selector_rows), min_teacher_margin=args.min_teacher_margin)
    teachers = teacher_record_by_line(Path(args.teacher_input))
    policy_picker: Callable[[list[dict[str, Any]]], dict[str, Any]]
    if args.policy == "selector":
        if selector is None:
            raise SystemExit("--selector is required for --policy selector")
        policy_picker = lambda rows: pick_selector(rows, selector)
    elif args.policy == "refined_score":
        policy_picker = pick_refined
    else:
        raise SystemExit(f"Unsupported policy: {args.policy}")

    output_teacher = Path(args.output_teacher)
    output_targets = Path(args.output_targets) if args.output_targets else None
    output_teacher.parent.mkdir(parents=True, exist_ok=True)
    if output_targets is not None:
        output_targets.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "selector_rows": str(args.selector_rows),
        "teacher_input": str(args.teacher_input),
        "policy": args.policy,
        "min_teacher_margin": float(args.min_teacher_margin),
        "groups": len(groups),
        "misses": 0,
        "hits": 0,
        "missing_teacher_records": 0,
        "output_teacher": str(output_teacher),
        "output_targets": str(output_targets) if output_targets else "",
    }
    with output_teacher.open("w", encoding="utf-8") as teacher_f:
        target_f = output_targets.open("w", encoding="utf-8") if output_targets is not None else None
        try:
            for line_no, rows in sorted(groups.items()):
                selected = policy_picker(rows)
                if bool(selected.get("is_teacher_best")):
                    stats["hits"] += 1
                    continue
                stats["misses"] += 1
                teacher = teachers.get(line_no)
                if teacher is None:
                    stats["missing_teacher_records"] += 1
                    continue
                reason = (
                    f"t1_final_{args.policy}_miss_margin"
                    f"{float(rows[0].get('teacher_margin', 0.0) or 0.0):.3f}"
                )
                teacher_f.write(json.dumps({**teacher, "active_reason": reason}, ensure_ascii=False) + "\n")
                if target_f is not None:
                    target_f.write(
                        json.dumps(
                            target_from_teacher(
                                teacher,
                                source=str(args.teacher_input),
                                source_line=line_no,
                                reason=reason,
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        finally:
            if target_f is not None:
                target_f.close()
    Path(args.summary).write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return stats


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collect T1 final selector misses")
    parser.add_argument("--teacher-input", required=True)
    parser.add_argument("--selector-rows", required=True)
    parser.add_argument("--output-teacher", required=True)
    parser.add_argument("--output-targets", default="")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--policy", default="selector", choices=["selector", "refined_score"])
    parser.add_argument("--selector", default="")
    parser.add_argument("--min-teacher-margin", type=float, default=0.25)
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(collect(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
