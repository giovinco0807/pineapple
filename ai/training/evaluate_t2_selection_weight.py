"""Sweep T2 refined/model blend weights over saved runtime candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def parse_weights(raw: str) -> list[float]:
    return [float(part) for part in raw.split(",") if part.strip()]


def action_key(action: dict[str, Any] | None) -> tuple[tuple[tuple[str, str], ...], str]:
    action = action or {}
    placements = tuple(sorted((str(card), str(row)) for card, row in (action.get("placements") or [])))
    return placements, str(action.get("discard") or "")


def candidate_action_key(candidate: dict[str, Any] | None) -> tuple[tuple[tuple[str, str], ...], str]:
    candidate = candidate or {}
    action = candidate.get("action")
    if isinstance(action, dict):
        return action_key(action)
    return action_key(candidate)


def read_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def runtime_result(row: dict[str, Any]) -> dict[str, Any]:
    result = row.get("result")
    if isinstance(result, dict):
        return result
    return row


def load_extra_scores(
    specs: list[tuple[Path, set[str]]],
) -> dict[tuple[int, tuple[tuple[tuple[str, str], ...], str]], float]:
    out: dict[tuple[int, tuple[tuple[tuple[str, str], ...], str]], float] = {}
    for path, dataset_filter in specs:
        for row in read_jsonl(path):
            if dataset_filter and str(row.get("runtime_dataset") or "") not in dataset_filter:
                continue
            row_id = row.get("original_row_id")
            if row_id is None:
                continue
            try:
                row_index = int(row_id)
            except (TypeError, ValueError):
                continue
            for candidate in row.get("candidates") or []:
                score = candidate.get("score", candidate.get("ev"))
                if score is None:
                    continue
                out[(row_index, candidate_action_key(candidate))] = float(score)
    return out


def teacher_exact_scores(
    teacher: dict[str, Any],
    row_id: int,
    extra: dict[tuple[int, tuple[tuple[tuple[str, str], ...], str]], float],
) -> dict[tuple[tuple[tuple[str, str], ...], str], float]:
    out = {
        candidate_action_key(candidate): float(candidate.get("score", candidate.get("ev", 0.0)) or 0.0)
        for candidate in teacher.get("candidates") or []
    }
    for (extra_row_id, key), score in extra.items():
        if extra_row_id == row_id:
            out[key] = float(score)
    return out


def candidate_selection_score(candidate: dict[str, Any], weight: float) -> tuple[float, float, float, int]:
    refined = candidate.get("refined_score")
    if refined is None:
        return (float("-inf"), float("-inf"), float("-inf"), -999999)
    refined_value = float(refined)
    model_score = float(candidate.get("model_score", 0.0) or 0.0)
    model_rank = int(candidate.get("model_rank", 999999) or 999999)
    return (refined_value + weight * model_score, refined_value, model_score, -model_rank)


def evaluate_weight(
    teacher_rows: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
    extra_scores: dict[tuple[int, tuple[tuple[tuple[str, str], ...], str]], float],
    weight: float,
) -> dict[str, Any]:
    rows = 0
    known = 0
    missing = 0
    hits = 0
    regret = 0.0
    for row_id, (teacher, raw_runtime) in enumerate(zip(teacher_rows, runtime_rows)):
        runtime = runtime_result(raw_runtime)
        candidates = [
            candidate
            for candidate in (runtime.get("candidates") or [])
            if isinstance(candidate, dict) and candidate.get("refined_score") is not None
        ]
        if not candidates:
            continue
        scores = teacher_exact_scores(teacher, row_id, extra_scores)
        if not scores:
            continue
        best_score = max(scores.values())
        chosen = max(candidates, key=lambda candidate: candidate_selection_score(candidate, weight))
        chosen_score = scores.get(candidate_action_key(chosen))
        rows += 1
        if chosen_score is None:
            missing += 1
            continue
        known += 1
        row_regret = max(0.0, best_score - float(chosen_score))
        regret += row_regret
        if row_regret <= 1e-9:
            hits += 1
    return {
        "rows": rows,
        "known": known,
        "missing": missing,
        "hit_rate_known": hits / max(known, 1),
        "hits": hits,
        "avg_regret_known": regret / max(known, 1),
    }


def parse_extra_spec(raw: str) -> tuple[Path, set[str]]:
    path, sep, datasets = raw.partition("#")
    dataset_filter = {part for part in datasets.split(",") if part} if sep else set()
    return Path(path), dataset_filter


def parse_eval(raw: str) -> tuple[str, Path, Path, list[tuple[Path, set[str]]]]:
    parts = raw.split("|")
    if len(parts) < 3:
        raise ValueError("--eval must be name|teacher|runtime|extra1;extra2")
    extras = [parse_extra_spec(part) for part in parts[3].split(";") if part.strip()] if len(parts) >= 4 else []
    return parts[0], Path(parts[1]), Path(parts[2]), extras


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", action="append", required=True, help="name|teacher|runtime|extra1;extra2")
    parser.add_argument("--weights", default="-1,-0.5,0,0.25,0.5,0.75,1,1.25,1.5,2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    weights = parse_weights(args.weights)
    evals = [parse_eval(raw) for raw in args.eval]
    rows_out = []
    for weight in weights:
        total_rows = total_known = total_missing = total_hits = 0
        total_regret = 0.0
        per_set = {}
        for name, teacher_path, runtime_path, extra_paths in evals:
            teacher_rows = read_jsonl(teacher_path, args.limit)
            runtime_rows = read_jsonl(runtime_path, args.limit)
            result = evaluate_weight(
                teacher_rows,
                runtime_rows,
                load_extra_scores(extra_paths),
                weight,
            )
            per_set[name] = result
            total_rows += int(result["rows"])
            total_known += int(result["known"])
            total_missing += int(result["missing"])
            total_hits += int(result["hits"])
            total_regret += float(result["avg_regret_known"]) * int(result["known"])
        rows_out.append(
            {
                "weight": weight,
                "rows": total_rows,
                "known": total_known,
                "missing": total_missing,
                "hit_rate_known": total_hits / max(total_known, 1),
                "avg_regret_known": total_regret / max(total_known, 1),
                "per_set": per_set,
            }
        )
    output = {
        "evals": [raw for raw in args.eval],
        "weights": weights,
        "rows": rows_out,
        "best_known_regret": min(rows_out, key=lambda row: (row["avg_regret_known"], row["missing"])),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    best = output["best_known_regret"]
    print(
        f"best_weight={best['weight']} avg_regret_known={best['avg_regret_known']:.3f} "
        f"hit_known={best['hit_rate_known']:.1%} missing={best['missing']}/{best['rows']}"
    )


if __name__ == "__main__":
    main()
