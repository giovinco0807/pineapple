"""Compare runtime target actions against stronger relabeled teacher output."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.action_space import encode_action, get_turn_actions
from ai.tutor.evaluate_hybrid_refinement_teacher import (
    canonical_action,
    candidate_action,
    candidate_score,
)
from ai.tutor.hybrid_t1t2 import action_to_dict, observation_from_payload


RecordKey = tuple[str, int, int]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_runtime_results(path: Path) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        line = row.get("line")
        if line is not None:
            results[int(line)] = row
    return results


def apply_runtime_results(
    targets: list[dict[str, Any]],
    runtime_results: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    missing_lines: list[int] = []
    for target in targets:
        line = target.get("runtime_result_line") or target.get("line")
        if line is None:
            raise SystemExit("target is missing runtime_result_line")
        result = runtime_results.get(int(line))
        if result is None:
            missing_lines.append(int(line))
            continue
        copy = dict(target)
        copy["runtime_model_action_idx"] = result.get("model_top1_action_idx")
        copy["runtime_final_action_idx"] = result.get("best_action_idx")
        copy["runtime_model_teacher_score"] = result.get("model_teacher_score")
        copy["runtime_final_teacher_score"] = result.get("final_teacher_score")
        copy["runtime_teacher_best_score"] = result.get("teacher_best_score")
        updated.append(copy)
    if missing_lines:
        raise SystemExit(f"missing runtime result lines: {sorted(missing_lines)[:3]}")
    return updated


def record_key(record: dict[str, Any]) -> RecordKey | None:
    source = record.get("source")
    source_line = record.get("source_line")
    turn = record.get("turn")
    if source is None or source_line is None or turn is None:
        return None
    normalized_source = str(source).replace("/", "\\").lower()
    return (normalized_source, int(source_line), int(turn))


def align_records(
    targets: list[dict[str, Any]],
    teachers: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    target_keys = [record_key(target) for target in targets]
    teacher_keys = [record_key(teacher) for teacher in teachers]
    if any(key is None for key in target_keys) or any(key is None for key in teacher_keys):
        if len(targets) != len(teachers):
            raise SystemExit(f"target/teacher count mismatch: {len(targets)} != {len(teachers)}")
        return list(zip(targets, teachers))

    teacher_by_key: dict[RecordKey, dict[str, Any]] = {}
    duplicate_keys: set[RecordKey] = set()
    for key, teacher in zip(teacher_keys, teachers):
        assert key is not None
        if key in teacher_by_key:
            duplicate_keys.add(key)
        teacher_by_key[key] = teacher
    if duplicate_keys:
        sample = sorted(duplicate_keys)[:3]
        raise SystemExit(f"duplicate teacher keys: {sample}")

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    missing_keys: list[RecordKey] = []
    for key, target in zip(target_keys, targets):
        assert key is not None
        teacher = teacher_by_key.get(key)
        if teacher is None:
            missing_keys.append(key)
        else:
            pairs.append((target, teacher))
    if missing_keys:
        sample = sorted(missing_keys)[:3]
        raise SystemExit(f"missing teacher keys: {sample}")
    return pairs


def action_map_from_target(target: dict[str, Any]) -> dict[int, tuple[tuple[tuple[str, str], ...], str]]:
    obs = observation_from_payload(target)
    valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    return {
        int(encode_action(action, valid_actions, turn=obs.turn, dealt_cards=obs.dealt_cards)): canonical_action(
            action_to_dict(action)
        )
        for action in valid_actions
    }


def teacher_scores(record: dict[str, Any]) -> dict[tuple[tuple[tuple[str, str], ...], str], float]:
    return {
        canonical_action(candidate_action(candidate)): candidate_score(candidate)
        for candidate in (record.get("candidates") or [])
    }


def compare_pair(index: int, target: dict[str, Any], teacher: dict[str, Any]) -> dict[str, Any]:
    idx_to_action = action_map_from_target(target)
    scores = teacher_scores(teacher)
    best_action, best_score = max(scores.items(), key=lambda item: item[1])

    model_idx = target.get("runtime_model_action_idx")
    final_idx = target.get("runtime_final_action_idx")
    model_action = idx_to_action.get(int(model_idx)) if model_idx is not None else None
    final_action = idx_to_action.get(int(final_idx)) if final_idx is not None else None
    model_score = scores.get(model_action) if model_action is not None else None
    final_score = scores.get(final_action) if final_action is not None else None
    model_regret = None if model_score is None else max(0.0, float(best_score) - float(model_score))
    final_regret = None if final_score is None else max(0.0, float(best_score) - float(final_score))
    model_zero_regret = model_regret is not None and model_regret <= 1e-9
    final_zero_regret = final_regret is not None and final_regret <= 1e-9

    return {
        "index": index,
        "turn": target.get("turn"),
        "source": target.get("source"),
        "source_line": target.get("source_line"),
        "teacher_eval_mode": teacher.get("eval_mode"),
        "teacher_best_score": best_score,
        "teacher_best_action": best_action,
        "runtime_model_action_idx": model_idx,
        "runtime_final_action_idx": final_idx,
        "relabel_model_action": model_action,
        "relabel_final_action": final_action,
        "runtime_model_teacher_score": target.get("runtime_model_teacher_score"),
        "runtime_final_teacher_score": target.get("runtime_final_teacher_score"),
        "runtime_override_delta": target.get("runtime_override_delta"),
        "relabel_model_score": model_score,
        "relabel_final_score": final_score,
        "relabel_model_minus_final": None
        if model_score is None or final_score is None
        else float(model_score) - float(final_score),
        "relabel_model_regret": model_regret,
        "relabel_final_regret": final_regret,
        "relabel_model_is_best": model_action == best_action,
        "relabel_final_is_best": final_action == best_action,
        "relabel_model_zero_regret": model_zero_regret,
        "relabel_final_zero_regret": final_zero_regret,
        "relabel_model_found": model_score is not None,
        "relabel_final_found": final_score is not None,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    model_regrets = [float(row["relabel_model_regret"]) for row in rows if row["relabel_model_regret"] is not None]
    final_regrets = [float(row["relabel_final_regret"]) for row in rows if row["relabel_final_regret"] is not None]
    model_minus_final = [
        float(row["relabel_model_minus_final"]) for row in rows if row["relabel_model_minus_final"] is not None
    ]
    return {
        "records": len(rows),
        "model_found": sum(1 for row in rows if row["relabel_model_found"]),
        "final_found": sum(1 for row in rows if row["relabel_final_found"]),
        "model_is_best": sum(1 for row in rows if row["relabel_model_is_best"]),
        "final_is_best": sum(1 for row in rows if row["relabel_final_is_best"]),
        "model_zero_regret": sum(1 for row in rows if row["relabel_model_zero_regret"]),
        "final_zero_regret": sum(1 for row in rows if row["relabel_final_zero_regret"]),
        "model_better_than_final": sum(
            1 for row in rows if row["relabel_model_minus_final"] is not None and row["relabel_model_minus_final"] > 0
        ),
        "final_better_than_model": sum(
            1 for row in rows if row["relabel_model_minus_final"] is not None and row["relabel_model_minus_final"] < 0
        ),
        "model_final_tie": sum(
            1 for row in rows if row["relabel_model_minus_final"] is not None and row["relabel_model_minus_final"] == 0
        ),
        "avg_model_regret": statistics.fmean(model_regrets) if model_regrets else 0.0,
        "avg_final_regret": statistics.fmean(final_regrets) if final_regrets else 0.0,
        "max_model_regret": max(model_regrets) if model_regrets else 0.0,
        "max_final_regret": max(final_regrets) if final_regrets else 0.0,
        "avg_model_minus_final": statistics.fmean(model_minus_final) if model_minus_final else 0.0,
    }


def compare(args: argparse.Namespace) -> dict[str, Any]:
    targets = list(iter_jsonl(Path(args.targets)))
    teachers = list(iter_jsonl(Path(args.teacher_labels)))
    if args.limit > 0:
        targets = targets[: args.limit]
    if args.runtime_results:
        targets = apply_runtime_results(targets, load_runtime_results(Path(args.runtime_results)))
    pairs = align_records(targets, teachers)

    rows = [compare_pair(i, target, teacher) for i, (target, teacher) in enumerate(pairs, start=1)]
    summary = {
        "targets": str(args.targets),
        "teacher_labels": str(args.teacher_labels),
        "runtime_results": str(args.runtime_results) if args.runtime_results else "",
        "output": str(args.output) if args.output else None,
        **summarize(rows),
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare runtime target actions to relabeled teacher records")
    parser.add_argument("--targets", required=True)
    parser.add_argument("--teacher-labels", required=True)
    parser.add_argument("--runtime-results", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(compare(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
