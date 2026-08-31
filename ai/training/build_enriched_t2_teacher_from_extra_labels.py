"""Merge T2 source teacher rows with extra exact-labeled candidates.

Extra labels are usually produced by exacting runtime-selected or refined
candidates that were absent from the original teacher TopK.  This script folds
those labels back into one decision row per original T2 spot, preserving the
group structure needed by ``convert_action_value_teacher`` and ranking losses.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ai.training.evaluate_t2_selection_weight import (
    candidate_action_key,
    parse_eval,
    parse_extra_spec,
    read_jsonl,
)


def candidate_score(candidate: dict[str, Any]) -> float:
    for key in ("score", "ev", "teacher_score", "target_score"):
        if candidate.get(key) is not None:
            return float(candidate.get(key) or 0.0)
    metrics = candidate.get("metrics") or {}
    if metrics.get("score") is not None:
        return float(metrics.get("score") or 0.0)
    return 0.0


def normalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    out = dict(candidate)
    action = out.get("action")
    if isinstance(action, dict):
        out.setdefault("placements", action.get("placements") or [])
        out.setdefault("discard", action.get("discard"))
    score = candidate_score(out)
    out["score"] = score
    out["ev"] = score
    return out


def load_extra_candidates(
    specs: list[tuple[Path, set[str]]],
) -> dict[tuple[str, int], dict[tuple[tuple[tuple[str, str], ...], str], dict[str, Any]]]:
    out: dict[tuple[str, int], dict[tuple[tuple[tuple[str, str], ...], str], dict[str, Any]]] = {}
    for path, dataset_filter in specs:
        for row in read_jsonl(path):
            dataset = str(row.get("runtime_dataset") or "")
            if dataset_filter and dataset not in dataset_filter:
                continue
            row_id = row.get("original_row_id")
            if row_id is None:
                continue
            try:
                row_index = int(row_id)
            except (TypeError, ValueError):
                continue
            bucket = out.setdefault((dataset, row_index), {})
            for raw_candidate in row.get("candidates") or []:
                candidate = normalize_candidate(raw_candidate)
                key = candidate_action_key(candidate)
                current = bucket.get(key)
                if current is None or candidate_score(candidate) > candidate_score(current):
                    candidate["extra_label_source"] = str(path)
                    bucket[key] = candidate
    return out


def merge_candidates(
    base_candidates: list[dict[str, Any]],
    extra_candidates: dict[tuple[tuple[tuple[str, str], ...], str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    by_key: dict[tuple[tuple[tuple[str, str], ...], str], dict[str, Any]] = {}
    for candidate in base_candidates:
        normalized = normalize_candidate(candidate)
        by_key[candidate_action_key(normalized)] = normalized
    added = 0
    for key, candidate in extra_candidates.items():
        if key not in by_key:
            by_key[key] = normalize_candidate(candidate)
            added += 1
            continue
        if candidate_score(candidate) > candidate_score(by_key[key]):
            by_key[key] = normalize_candidate(candidate)
    merged = sorted(by_key.values(), key=candidate_score, reverse=True)
    return merged, added


def build(args: argparse.Namespace) -> dict[str, Any]:
    evals = [parse_eval(raw) for raw in args.eval]
    extra_specs = [parse_extra_spec(raw) for raw in args.extra]
    extra_by_row = load_extra_candidates(extra_specs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "output": str(output),
        "evals": [raw for raw in args.eval],
        "extras": [raw for raw in args.extra],
        "sets": {},
    }
    total_rows = 0
    total_candidates = 0
    total_added = 0
    total_candidate_hist: Counter[int] = Counter()

    with output.open("w", encoding="utf-8") as out:
        for dataset_name, teacher_path, _runtime_path, _extra_specs in evals:
            rows = read_jsonl(teacher_path)
            set_added = 0
            set_candidates = 0
            set_candidate_hist: Counter[int] = Counter()
            for row_id, row in enumerate(rows):
                extra = extra_by_row.get((dataset_name, row_id), {})
                candidates, added = merge_candidates(list(row.get("candidates") or []), extra)
                if not candidates:
                    continue
                best_idx = max(range(len(candidates)), key=lambda idx: candidate_score(candidates[idx]))
                enriched = dict(row)
                enriched.update(
                    {
                        "source": "t2_enriched_source_plus_extra_labels",
                        "runtime_dataset": dataset_name,
                        "original_row_id": int(row_id),
                        "teacher_eval_mode": args.eval_mode,
                        "eval_mode": args.eval_mode,
                        "best_idx": int(best_idx),
                        "n_candidates": len(candidates),
                        "extra_candidates_added": int(added),
                        "candidates": candidates,
                    }
                )
                out.write(json.dumps(enriched, ensure_ascii=False, separators=(",", ":")) + "\n")
                total_rows += 1
                total_candidates += len(candidates)
                total_added += added
                set_added += added
                set_candidates += len(candidates)
                set_candidate_hist[len(candidates)] += 1
                total_candidate_hist[len(candidates)] += 1
            stats["sets"][dataset_name] = {
                "teacher": str(teacher_path),
                "rows": len(rows),
                "extra_candidates_added": set_added,
                "candidates": set_candidates,
                "candidate_count_hist": dict(sorted(set_candidate_hist.items())),
            }

    stats.update(
        {
            "rows": total_rows,
            "candidates": total_candidates,
            "extra_candidates_added": total_added,
            "candidate_count_hist": dict(sorted(total_candidate_hist.items())),
        }
    )
    output.with_suffix(output.suffix + ".summary.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return stats


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", action="append", required=True, help="name|teacher|runtime|extra1;extra2")
    parser.add_argument("--extra", action="append", required=True, help="extra_teacher.jsonl#dataset1,dataset2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-mode", default="exact_t2_cap50_enriched")
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
