"""Extract unknown exact labels for refined T2 runtime candidates.

This prepares compact JSONL input for ``ai.tutor.run_t2_exact_oracle``.  For
each runtime row, every candidate with a non-null ``refined_score`` is checked
against the source teacher labels plus optional extra teacher JSONL files.  Any
unknown refined candidates are emitted together with the current source teacher
best candidate so the Rust oracle can label the missing actions in one pass.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from ai.training.evaluate_t2_selection_weight import (
    candidate_action_key,
    load_extra_scores,
    parse_eval,
    read_jsonl,
    runtime_result,
    teacher_exact_scores,
)
from ai.training.extract_unknown_runtime_selected_t2 import (
    action_key,
    best_teacher_candidate,
    candidate_action,
    candidate_payload,
)


def runtime_candidates(runtime: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        candidate
        for candidate in (runtime.get("candidates") or [])
        if isinstance(candidate, dict) and candidate.get("refined_score") is not None
    ]


def selection_score(candidate: dict[str, Any], weight: float) -> float:
    refined = float(candidate.get("refined_score", 0.0) or 0.0)
    model = float(candidate.get("model_score", 0.0) or 0.0)
    return refined + float(weight) * model


def refined_payload(
    candidate: dict[str, Any],
    *,
    source: str,
    source_rank: int,
    weight: float,
) -> dict[str, Any]:
    payload = candidate_payload(candidate, source=source, source_rank=source_rank)
    payload["score"] = selection_score(candidate, weight)
    payload["selection_score"] = payload["score"]
    payload["action_idx"] = candidate.get("action_idx")
    payload["model_rank"] = candidate.get("model_rank")
    payload["refined_rank"] = candidate.get("refined_rank")
    payload["predicted_bust"] = candidate.get("predicted_bust")
    payload["predicted_fl"] = candidate.get("predicted_fl")
    payload["predicted_fl_types"] = candidate.get("predicted_fl_types")
    payload["samples"] = candidate.get("samples")
    payload["forced_bust"] = candidate.get("forced_bust")
    return payload


def teacher_payload(candidate: dict[str, Any], *, source_rank: int) -> dict[str, Any]:
    payload = candidate_payload(candidate, source="current_teacher_best", source_rank=source_rank)
    if candidate.get("score") is not None:
        payload["score"] = float(candidate.get("score") or 0.0)
    elif candidate.get("ev") is not None:
        payload["score"] = float(candidate.get("ev") or 0.0)
    return payload


def source_row(
    teacher: dict[str, Any],
    *,
    row_id: int,
    dataset_name: str,
    runtime_path: Path,
    runtime: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    row = dict(teacher)
    row.update(
        {
            "source": "unknown_refined_t2_candidates",
            "original_row_id": int(row_id),
            "runtime_dataset": dataset_name,
            "runtime_output": str(runtime_path),
            "runtime_mode": str(runtime.get("mode") or ""),
            "unknown_refined_candidate_count": len(
                [candidate for candidate in candidates if candidate.get("source") == "unknown_refined_candidate"]
            ),
            "n_candidates": len(candidates),
            "source_candidate_top_k": len(candidates),
            "candidates": candidates,
        }
    )
    return row


def extract(args: argparse.Namespace) -> dict[str, Any]:
    evals = [parse_eval(raw) for raw in args.eval]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "output": str(out_path),
        "selection_weight": float(args.selection_weight),
        "evals": [raw for raw in args.eval],
        "sets": {},
    }
    total_records = 0
    total_unknown_candidates = 0
    total_refined_candidates = 0
    total_known_refined_candidates = 0
    max_candidates_per_record = 0
    candidate_count_hist: Counter[int] = Counter()

    with out_path.open("w", encoding="utf-8") as out:
        for dataset_name, teacher_path, runtime_path, extra_specs in evals:
            teacher_rows = read_jsonl(teacher_path)
            runtime_rows = read_jsonl(runtime_path)
            if len(teacher_rows) != len(runtime_rows):
                raise ValueError(
                    f"row count mismatch for {dataset_name}: "
                    f"teacher={len(teacher_rows)} runtime={len(runtime_rows)}"
                )
            extra_scores = load_extra_scores(extra_specs)
            set_records = 0
            set_unknown_candidates = 0
            set_refined_candidates = 0
            set_known_refined_candidates = 0
            set_candidate_hist: Counter[int] = Counter()
            unknown_by_row: defaultdict[int, int] = defaultdict(int)

            for row_id, (teacher, raw_runtime) in enumerate(zip(teacher_rows, runtime_rows)):
                runtime = runtime_result(raw_runtime)
                refined = runtime_candidates(runtime)
                if not refined:
                    continue
                scores = teacher_exact_scores(teacher, row_id, extra_scores)
                known = set(scores)
                unknown: list[dict[str, Any]] = []
                seen_unknown: set[tuple[tuple[tuple[str, str], ...], str]] = set()
                ranked = sorted(
                    refined,
                    key=lambda candidate: (
                        selection_score(candidate, args.selection_weight),
                        float(candidate.get("refined_score", 0.0) or 0.0),
                        float(candidate.get("model_score", 0.0) or 0.0),
                        -int(candidate.get("model_rank", 999999) or 999999),
                    ),
                    reverse=True,
                )
                for source_rank, candidate in enumerate(ranked, start=1):
                    key = candidate_action_key(candidate)
                    if key in known:
                        set_known_refined_candidates += 1
                        continue
                    if key in seen_unknown:
                        continue
                    seen_unknown.add(key)
                    unknown.append(
                        refined_payload(
                            candidate,
                            source="unknown_refined_candidate",
                            source_rank=source_rank,
                            weight=args.selection_weight,
                        )
                    )
                set_refined_candidates += len(refined)
                if not unknown:
                    continue

                candidates = list(unknown)
                teacher_best = best_teacher_candidate(teacher)
                if teacher_best is not None:
                    best_payload = teacher_payload(teacher_best, source_rank=len(candidates) + 1)
                    if action_key(candidate_action(best_payload)) not in {
                        action_key(candidate_action(candidate)) for candidate in candidates
                    }:
                        candidates.append(best_payload)
                max_candidates_per_record = max(max_candidates_per_record, len(candidates))
                candidate_count_hist[len(candidates)] += 1
                set_candidate_hist[len(candidates)] += 1
                unknown_by_row[len(unknown)] += 1
                out.write(
                    json.dumps(
                        source_row(
                            teacher,
                            row_id=row_id,
                            dataset_name=dataset_name,
                            runtime_path=runtime_path,
                            runtime=runtime,
                            candidates=candidates,
                        ),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                set_records += 1
                set_unknown_candidates += len(unknown)

            total_records += set_records
            total_unknown_candidates += set_unknown_candidates
            total_refined_candidates += set_refined_candidates
            total_known_refined_candidates += set_known_refined_candidates
            summary["sets"][dataset_name] = {
                "teacher": str(teacher_path),
                "runtime_output": str(runtime_path),
                "rows": len(teacher_rows),
                "records": set_records,
                "refined_candidates": set_refined_candidates,
                "known_refined_candidates": set_known_refined_candidates,
                "unknown_refined_candidates": set_unknown_candidates,
                "candidate_count_hist": dict(sorted(set_candidate_hist.items())),
                "unknown_candidates_per_row_hist": dict(sorted(unknown_by_row.items())),
            }

    summary.update(
        {
            "records": total_records,
            "refined_candidates": total_refined_candidates,
            "known_refined_candidates": total_known_refined_candidates,
            "unknown_refined_candidates": total_unknown_candidates,
            "candidate_count_hist": dict(sorted(candidate_count_hist.items())),
            "max_candidates_per_record": max_candidates_per_record,
        }
    )
    summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", action="append", required=True, help="name|teacher|runtime|extra1;extra2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection-weight", type=float, default=1.5)
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(extract(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
