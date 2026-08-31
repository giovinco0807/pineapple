"""Extract runtime-selected T2 actions missing from a teacher JSONL.

The output is a small JSONL that can be sent to ``ai.tutor.run_t2_exact_oracle``.
Each row keeps the original board/dealt context but replaces ``candidates`` with
the unknown selected action plus the current teacher-best action for comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def runtime_result(row: dict[str, Any]) -> dict[str, Any]:
    result = row.get("result")
    if isinstance(result, dict):
        return result
    return row


def action_key(action: dict[str, Any] | None) -> tuple[tuple[tuple[str, str], ...], str]:
    action = action or {}
    placements = tuple(sorted((str(card), str(row)) for card, row in (action.get("placements") or [])))
    return placements, str(action.get("discard") or "")


def candidate_action(candidate: dict[str, Any] | None) -> dict[str, Any]:
    candidate = candidate or {}
    action = candidate.get("action")
    if isinstance(action, dict):
        return {
            "placements": action.get("placements") or [],
            "discard": action.get("discard"),
        }
    return {
        "placements": candidate.get("placements") or [],
        "discard": candidate.get("discard"),
    }


def candidate_score(candidate: dict[str, Any]) -> float:
    for key in ("score", "ev", "teacher_score", "refined_score", "model_score"):
        if candidate.get(key) is not None:
            return float(candidate.get(key) or 0.0)
    return 0.0


def candidate_payload(candidate: dict[str, Any], *, source: str, source_rank: int) -> dict[str, Any]:
    action = candidate_action(candidate)
    return {
        "action": action,
        "placements": action["placements"],
        "discard": action["discard"],
        "score": candidate_score(candidate),
        "source": source,
        "source_rank": int(source_rank),
        "model_score": candidate.get("model_score"),
        "refined_score": candidate.get("refined_score"),
    }


def best_teacher_candidate(row: dict[str, Any]) -> dict[str, Any] | None:
    candidates = row.get("candidates") or []
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: float(candidate.get("score", candidate.get("ev", 0.0)) or 0.0))


def extract(args: argparse.Namespace) -> dict[str, Any]:
    teacher_rows = list(iter_jsonl(Path(args.teacher)))
    runtime_rows = list(iter_jsonl(Path(args.runtime_output)))
    if len(teacher_rows) != len(runtime_rows):
        raise ValueError(f"row count mismatch: teacher={len(teacher_rows)} runtime={len(runtime_rows)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_known = 0
    skipped_no_choice = 0
    with out_path.open("w", encoding="utf-8") as out:
        for row_id, (teacher, raw_runtime) in enumerate(zip(teacher_rows, runtime_rows)):
            runtime = runtime_result(raw_runtime)
            known = {action_key(candidate_action(candidate)) for candidate in (teacher.get("candidates") or [])}
            chosen = runtime.get("best") if isinstance(runtime.get("best"), dict) else None
            if chosen is None:
                skipped_no_choice += 1
                continue
            chosen_key = action_key(candidate_action(chosen))
            if chosen_key in known:
                skipped_known += 1
                continue
            candidates = [candidate_payload(chosen, source="runtime_selected_unknown", source_rank=1)]
            teacher_best = best_teacher_candidate(teacher)
            if teacher_best is not None:
                best_payload = candidate_payload(teacher_best, source="current_teacher_best", source_rank=2)
                if action_key(candidate_action(best_payload)) != chosen_key:
                    candidates.append(best_payload)
            source = dict(teacher)
            source.update(
                {
                    "source": "runtime_selected_unknown_t2",
                    "original_row_id": int(row_id),
                    "runtime_dataset": str(args.dataset_name or ""),
                    "runtime_output": str(Path(args.runtime_output)),
                    "runtime_mode": str(runtime.get("mode") or args.runtime_mode or ""),
                    "runtime_selected_action_key": json.dumps(chosen_key, ensure_ascii=False),
                    "n_candidates": len(candidates),
                    "source_candidate_top_k": len(candidates),
                    "candidates": candidates,
                }
            )
            out.write(json.dumps(source, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1

    summary = {
        "teacher": str(Path(args.teacher)),
        "runtime_output": str(Path(args.runtime_output)),
        "output": str(out_path),
        "dataset_name": str(args.dataset_name or ""),
        "runtime_mode": str(args.runtime_mode or ""),
        "teacher_rows": len(teacher_rows),
        "runtime_rows": len(runtime_rows),
        "written": int(written),
        "skipped_known": int(skipped_known),
        "skipped_no_choice": int(skipped_no_choice),
    }
    out_path.with_suffix(out_path.suffix + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--runtime-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--runtime-mode", default="")
    args = parser.parse_args(argv)
    print(json.dumps(extract(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
