"""Score hybrid_t1t2 runtime JSONL output against exact/capped teacher rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def parse_topks(value: str) -> list[int]:
    return sorted({int(part) for part in value.split(",") if part.strip() and int(part) > 0})


def action_key(action: dict) -> tuple[tuple[tuple[str, str], ...], str]:
    placements = tuple(sorted((str(card), str(row)) for card, row in (action.get("placements") or [])))
    return placements, str(action.get("discard") or "")


def candidate_action_key(candidate: dict) -> tuple[tuple[tuple[str, str], ...], str]:
    if "action" in candidate and isinstance(candidate.get("action"), dict):
        return action_key(candidate["action"])
    return action_key(candidate)


def action_label_from_key(key: tuple[tuple[tuple[str, str], ...], str]) -> str:
    placements, discard = key
    parts = [f"{card}->{row}" for card, row in placements]
    if discard:
        parts.append(f"discard {discard}")
    return "; ".join(parts)


def read_jsonl(path: Path, limit: int = 0) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def runtime_result(row: dict) -> dict:
    """Accept either a raw hybrid result row or benchmark ``{result: ...}`` row."""
    result = row.get("result")
    if isinstance(result, dict):
        return result
    return row


def load_extra_scores(
    paths: list[str],
    *,
    dataset_name: str = "",
) -> dict[tuple[int, tuple[tuple[tuple[str, str], ...], str]], float]:
    out: dict[tuple[int, tuple[tuple[tuple[str, str], ...], str]], float] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not raw_path:
            continue
        for row in read_jsonl(path):
            if dataset_name and str(row.get("runtime_dataset") or "") not in {"", dataset_name}:
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


def score_rows(
    teacher_rows: list[dict],
    runtime_rows: list[dict],
    topks: list[int],
    extra_scores: dict[tuple[int, tuple[tuple[tuple[str, str], ...], str]], float] | None = None,
) -> dict:
    extra_scores = extra_scores or {}
    hits = {k: 0 for k in topks}
    regrets = {k: 0.0 for k in topks}
    regret_counts = {k: 0 for k in topks}
    missing_topk = {k: 0 for k in topks}
    top1_regret = 0.0
    top1_regret_count = 0
    missing_top1 = 0
    chosen_known = 0
    chosen_missing = 0
    chosen_hit = 0
    chosen_regret = 0.0
    elapsed_ms: list[float] = []
    refinement_evaluated: list[int] = []
    matched_rows = 0
    rows_out = []
    for row_id, (teacher, raw_runtime) in enumerate(zip(teacher_rows, runtime_rows)):
        runtime = runtime_result(raw_runtime)
        teacher_candidates = teacher.get("candidates") or []
        runtime_candidates = runtime.get("candidates") or []
        if not teacher_candidates or not runtime_candidates:
            continue
        exact_by_action = {
            candidate_action_key(candidate): float(candidate.get("score", candidate.get("ev", 0.0)) or 0.0)
            for candidate in teacher_candidates
        }
        for (extra_row_id, extra_key), extra_score in extra_scores.items():
            if extra_row_id == row_id:
                exact_by_action[extra_key] = float(extra_score)
        teacher_by_action = {candidate_action_key(candidate): candidate for candidate in teacher_candidates}
        best_key, best_score = max(exact_by_action.items(), key=lambda item: item[1])
        runtime_keys = [candidate_action_key(candidate) for candidate in runtime_candidates]
        runtime_scores = [exact_by_action.get(key) for key in runtime_keys]
        chosen_candidate = runtime.get("best") if isinstance(runtime.get("best"), dict) else None
        chosen_key = candidate_action_key(chosen_candidate) if chosen_candidate else None
        chosen_score = exact_by_action.get(chosen_key) if chosen_key is not None else None
        chosen_regret_row = None
        if chosen_score is None:
            chosen_missing += 1
        else:
            chosen_known += 1
            chosen_regret_row = best_score - float(chosen_score)
            chosen_regret += chosen_regret_row
            if abs(chosen_regret_row) <= 1e-9:
                chosen_hit += 1
        elapsed_ms.append(float(runtime.get("elapsed_ms", 0.0) or 0.0))
        refinement_evaluated.append(int(runtime.get("refinement_evaluated", 0) or 0))
        if runtime_scores[0] is None:
            missing_top1 += 1
            model_top1_score = None
            top1_regret_row = None
        else:
            model_top1_score = float(runtime_scores[0])
            top1_regret_row = best_score - model_top1_score
            top1_regret += top1_regret_row
            top1_regret_count += 1
        rank = None
        for i, key in enumerate(runtime_keys, start=1):
            if key == best_key:
                rank = i
                break
        for k in topks:
            top_scores = [score for score in runtime_scores[:k] if score is not None]
            if rank is not None and rank <= k:
                hits[k] += 1
            if top_scores:
                best_in_topk = max(top_scores)
                regrets[k] += best_score - best_in_topk
                regret_counts[k] += 1
            else:
                missing_topk[k] += 1
        matched_rows += 1
        rows_out.append(
            {
                "row_id": row_id,
                "turn": teacher.get("turn"),
                "position": teacher.get("position"),
                "board": teacher.get("board"),
                "opponent_board": teacher.get("opponent_board"),
                "dealt": teacher.get("dealt"),
                "teacher_best_score": best_score,
                "teacher_best_action": action_label_from_key(best_key),
                "teacher_best_candidate": teacher_by_action.get(best_key, {}),
                "runtime_top1_teacher_score": model_top1_score,
                "top1_regret": top1_regret_row,
                "chosen_teacher_score": chosen_score,
                "chosen_regret": chosen_regret_row,
                "chosen_action": action_label_from_key(chosen_key) if chosen_key is not None else None,
                "teacher_best_runtime_rank": rank,
                "runtime_best_action_idx": runtime.get("best_action_idx"),
                "runtime_top_candidates": [
                    {
                        "rank": i + 1,
                        "action": action_label_from_key(key),
                        "teacher_score": runtime_scores[i],
                        "model_score": float((runtime_candidates[i] or {}).get("model_score", 0.0) or 0.0),
                    }
                    for i, key in enumerate(runtime_keys[:10])
                ],
            }
        )
    n = max(matched_rows, 1)
    summary = {
        "rows": matched_rows,
        "missing_top1": missing_top1,
        "top1_regret": top1_regret / max(top1_regret_count, 1),
        "top1_regret_known_rows": top1_regret_count,
        "chosen_known_rows": chosen_known,
        "chosen_missing_rows": chosen_missing,
        "chosen_hit_known_rows": chosen_hit,
        "chosen_hit_known_rate": chosen_hit / max(chosen_known, 1),
        "chosen_regret_known": chosen_regret / max(chosen_known, 1),
        "elapsed_ms_mean": sum(elapsed_ms) / max(len(elapsed_ms), 1),
        "elapsed_ms_max": max(elapsed_ms) if elapsed_ms else 0.0,
        "refinement_evaluated_mean": sum(refinement_evaluated) / max(len(refinement_evaluated), 1),
    }
    for k in topks:
        summary[f"top{k}"] = hits[k] / n
        summary[f"top{k}_regret"] = regrets[k] / max(regret_counts[k], 1)
        summary[f"top{k}_regret_known_rows"] = regret_counts[k]
        summary[f"top{k}_missing_all_known"] = missing_topk[k]
    return {"summary": summary, "rows": rows_out}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score hybrid runtime output against teacher JSONL")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--runtime-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--topks", default="1,3,5,10")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--extra-teacher",
        action="append",
        default=[],
        help="Optional teacher JSONL with original_row_id metadata for extra labeled runtime candidates.",
    )
    parser.add_argument(
        "--dataset-name",
        default="",
        help="When using --extra-teacher, only merge rows whose runtime_dataset matches this name.",
    )
    args = parser.parse_args(argv)

    teacher_rows = read_jsonl(Path(args.teacher), args.limit)
    runtime_rows = read_jsonl(Path(args.runtime_output), args.limit)
    if len(teacher_rows) != len(runtime_rows):
        raise ValueError(f"row count mismatch: teacher={len(teacher_rows)} runtime={len(runtime_rows)}")
    result = score_rows(
        teacher_rows,
        runtime_rows,
        parse_topks(args.topks),
        extra_scores=load_extra_scores(list(args.extra_teacher or []), dataset_name=str(args.dataset_name or "")),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = result["summary"]
    print(
        f"rows={summary['rows']} top1={summary.get('top1', 0.0):.1%} "
        f"reg1={summary['top1_regret']:.3f} top3={summary.get('top3', 0.0):.1%} "
        f"reg3={summary.get('top3_regret', 0.0):.3f} top5={summary.get('top5', 0.0):.1%} "
        f"reg5={summary.get('top5_regret', 0.0):.3f} top10={summary.get('top10', 0.0):.1%} "
        f"reg10={summary.get('top10_regret', 0.0):.3f} "
        f"chosen_known={summary['chosen_known_rows']} chosen_missing={summary['chosen_missing_rows']} "
        f"chosen_reg={summary['chosen_regret_known']:.3f} elapsed_mean={summary['elapsed_ms_mean']:.1f}ms"
    )


if __name__ == "__main__":
    main()
