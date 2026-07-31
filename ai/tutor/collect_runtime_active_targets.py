"""Collect high-confidence active targets from runtime refinement results.

This script is intended for the Top1 active loop.  It can merge multiple
runtime-evaluation result files, require MC1000+ or exact teacher labels, and
deduplicate states before writing active-teacher targets.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.tutor.weak_groups_to_active_targets import expand_suits, target_from_record, target_key, teacher_margin


def parse_turns(raw: str) -> set[int]:
    return {int(part) for part in raw.split(",") if part.strip()}


def parse_reasons(raw: str) -> set[str]:
    return {part.strip() for part in raw.split(",") if part.strip()}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_teacher_records(path: Path) -> dict[int, dict[str, Any]]:
    return {line_no: row for line_no, row in enumerate(iter_jsonl(path), start=1)}


def split_path_args(raw_values: list[str] | None) -> list[Path]:
    paths: list[Path] = []
    for raw in raw_values or []:
        for part in raw.split(";"):
            if part.strip():
                paths.append(Path(part.strip()))
    return paths


def load_excluded_target_keys(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        for target in iter_jsonl(path):
            excluded.add(target_key(target))
    return excluded


def parse_pair(raw: str) -> dict[str, str]:
    parts = raw.split("::")
    if len(parts) < 2:
        raise ValueError("pair must be RESULTS::TEACHER[::TAG]")
    return {
        "results": parts[0],
        "teacher": parts[1],
        "tag": parts[2] if len(parts) > 2 and parts[2] else Path(parts[0]).parent.name,
    }


def result_regret(result: dict[str, Any]) -> float:
    best = result.get("teacher_best_score")
    final = result.get("final_teacher_score")
    if best is None or final is None:
        return 0.0
    return max(0.0, float(best) - float(final))


def override_delta(result: dict[str, Any]) -> float:
    model = result.get("model_teacher_score")
    final = result.get("final_teacher_score")
    if model is None or final is None:
        return 0.0
    return float(model) - float(final)


def selected_reasons(result: dict[str, Any], requested: set[str], *, min_regret: float, min_override_delta: float) -> list[str]:
    reasons: list[str] = []
    regret = result_regret(result)
    delta = override_delta(result)
    if "final_miss" in requested and not bool(result.get("final_top1_hit")) and regret >= min_regret:
        reasons.append("runtime_final_top1_miss")
    if (
        "sync_hit_final_miss" in requested
        and result.get("teacher_best_in_sync") is True
        and not bool(result.get("final_top1_hit"))
        and regret >= min_regret
    ):
        reasons.append("runtime_sync_hit_final_top1_miss")
    if (
        "bad_override" in requested
        and bool(result.get("model_top1_overridden"))
        and delta > 0.0
        and delta >= min_override_delta
    ):
        reasons.append("runtime_bad_override")
    if (
        "good_override" in requested
        and bool(result.get("model_top1_overridden"))
        and delta < 0.0
        and -delta >= min_override_delta
    ):
        reasons.append("runtime_good_override")
    if "model_miss" in requested and not bool(result.get("model_top1_hit")):
        reasons.append("runtime_model_top1_miss")
    if "sync_miss" in requested and result.get("teacher_best_in_sync") is False:
        reasons.append("runtime_sync_miss")
    if "pool_miss" in requested and result.get("teacher_best_in_pool") is False:
        reasons.append("runtime_pool_miss")
    return reasons


def weak_from_result(result: dict[str, Any], min_teacher_margin: float, min_teacher_sims: int) -> dict[str, Any]:
    regret = result_regret(result)
    return {
        "group_id": int(result.get("line", 0)) - 1,
        "turn": int(result.get("turn", -1)),
        "rank": 999,
        "regret": regret,
        "topk_exact_rerank_regret": {"1": regret, "3": regret, "5": regret, "10": regret, "20": regret},
        "selection_topk": 1,
        "min_teacher_margin": float(min_teacher_margin),
        "min_teacher_sims": int(min_teacher_sims),
    }


def convert(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    teacher_output = Path(args.teacher_output) if args.teacher_output else None
    if teacher_output is not None:
        teacher_output.parent.mkdir(parents=True, exist_ok=True)
    if teacher_output is not None and args.suit_permutations != "none":
        raise SystemExit("--teacher-output currently requires --suit-permutations none")
    turns = parse_turns(args.turns)
    requested = parse_reasons(args.reasons)
    stats = Counter()
    seen: set[str] = set()
    exclude_paths = split_path_args(args.exclude_targets)
    excluded_keys = load_excluded_target_keys(exclude_paths)
    written = 0
    teacher_written = 0
    pairs = [parse_pair(raw) for raw in args.pair]

    with output.open("w", encoding="utf-8") as dst:
        teacher_dst = teacher_output.open("w", encoding="utf-8") if teacher_output is not None else None
        for pair in pairs:
            results_path = Path(pair["results"])
            teacher_path = Path(pair["teacher"])
            tag = pair["tag"]
            teacher_records = load_teacher_records(teacher_path)
            stats["pairs"] += 1
            for result in iter_jsonl(results_path):
                stats["results"] += 1
                turn = int(result.get("turn", -1))
                if turn not in turns:
                    stats["skipped_turn"] += 1
                    continue
                line_no = int(result.get("line", 0))
                record = teacher_records.get(line_no)
                if record is None:
                    stats["skipped_missing_teacher"] += 1
                    continue

                margin_info = teacher_margin(record)
                eval_mode = str(record.get("eval_mode") or result.get("teacher_eval_mode") or "")
                is_exact = "exact" in eval_mode.lower()
                margin = margin_info.get("teacher_margin")
                if (
                    args.min_teacher_margin > 0.0
                    and (args.margin_applies_to == "all" or not is_exact)
                    and (margin is None or float(margin) < float(args.min_teacher_margin))
                ):
                    stats["skipped_low_teacher_margin"] += 1
                    continue
                sims = margin_info.get("teacher_sims")
                if (
                    args.min_teacher_sims > 0
                    and (args.teacher_sims_applies_to == "all" or not is_exact)
                    and (sims is None or int(sims) < int(args.min_teacher_sims))
                ):
                    stats["skipped_low_teacher_sims"] += 1
                    continue

                runtime_reasons = selected_reasons(
                    result,
                    requested,
                    min_regret=args.min_regret,
                    min_override_delta=args.min_override_delta,
                )
                if not runtime_reasons:
                    stats["skipped_no_requested_reason"] += 1
                    continue

                target = target_from_record(record, weak_from_result(result, args.min_teacher_margin, args.min_teacher_sims))
                reasons = list(target.get("reasons") or [])
                for reason in runtime_reasons:
                    if reason not in reasons:
                        reasons.append(reason)
                if args.min_teacher_sims > 0 and sims is not None and int(sims) >= int(args.min_teacher_sims):
                    reasons.append(f"teacher_sims_ge_{int(args.min_teacher_sims)}")
                if margin is not None and float(margin) >= float(args.min_teacher_margin):
                    reasons.append("teacher_margin_pass")
                target["reasons"] = sorted(set(reasons))
                target["runtime_source_tag"] = tag
                target["runtime_results"] = str(results_path)
                target["runtime_teacher_labels"] = str(teacher_path)
                target["runtime_result_line"] = line_no
                target["runtime_teacher_eval_mode"] = eval_mode
                target["runtime_teacher_sims"] = sims
                target["runtime_teacher_margin"] = margin
                target["runtime_model_top1_hit"] = bool(result.get("model_top1_hit"))
                target["runtime_final_top1_hit"] = bool(result.get("final_top1_hit"))
                target["runtime_teacher_best_in_pool"] = result.get("teacher_best_in_pool")
                target["runtime_teacher_best_in_sync"] = result.get("teacher_best_in_sync")
                target["runtime_model_action_idx"] = result.get("model_top1_action_idx")
                target["runtime_final_action_idx"] = result.get("best_action_idx")
                target["runtime_model_teacher_score"] = result.get("model_teacher_score")
                target["runtime_final_teacher_score"] = result.get("final_teacher_score")
                target["runtime_teacher_best_score"] = result.get("teacher_best_score")
                target["runtime_regret"] = result_regret(result)
                target["runtime_override_delta"] = override_delta(result)
                target["runtime_exact_evaluated"] = result.get("exact_evaluated")
                target["runtime_elapsed_ms"] = result.get("elapsed_ms")

                for expanded in expand_suits(target, args.suit_permutations):
                    key = target_key(expanded)
                    if key in excluded_keys:
                        stats["skipped_excluded_target"] += 1
                        continue
                    if key in seen:
                        stats["skipped_duplicate"] += 1
                        continue
                    seen.add(key)
                    dst.write(json.dumps(expanded, ensure_ascii=False) + "\n")
                    if teacher_dst is not None:
                        teacher_record = dict(record)
                        teacher_record["runtime_source_tag"] = tag
                        teacher_record["runtime_results"] = str(results_path)
                        teacher_record["runtime_result_line"] = line_no
                        teacher_record["runtime_active_reasons"] = runtime_reasons
                        teacher_record["runtime_regret"] = result_regret(result)
                        teacher_record["runtime_override_delta"] = override_delta(result)
                        teacher_dst.write(json.dumps(teacher_record, ensure_ascii=False) + "\n")
                        teacher_written += 1
                    written += 1
                    stats[f"turn_{expanded['turn']}"] += 1
                    stats[f"tag_{tag}"] += 1
                    for reason in runtime_reasons:
                        stats[f"reason_{reason}"] += 1
        if teacher_dst is not None:
            teacher_dst.close()

    summary = {
        "pairs": pairs,
        "output": str(output),
        "teacher_output": str(teacher_output) if teacher_output is not None else "",
        "turns": sorted(turns),
        "reasons": sorted(requested),
        "min_teacher_margin": float(args.min_teacher_margin),
        "margin_applies_to": args.margin_applies_to,
        "min_teacher_sims": int(args.min_teacher_sims),
        "teacher_sims_applies_to": args.teacher_sims_applies_to,
        "min_regret": float(args.min_regret),
        "min_override_delta": float(args.min_override_delta),
        "suit_permutations": args.suit_permutations,
        "exclude_targets": [str(path) for path in exclude_paths],
        "excluded_target_keys": len(excluded_keys),
        "written": written,
        "teacher_written": teacher_written,
        "counts": dict(stats),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collect high-confidence runtime active targets")
    parser.add_argument("--pair", action="append", required=True, help="RESULTS::TEACHER[::TAG]; repeatable")
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher-output", default="", help="Optional matching teacher JSONL for the written targets")
    parser.add_argument("--turns", default="1")
    parser.add_argument("--reasons", default="final_miss,bad_override,sync_miss")
    parser.add_argument("--min-teacher-margin", type=float, default=0.25)
    parser.add_argument("--margin-applies-to", choices=("estimated", "all"), default="estimated")
    parser.add_argument("--min-teacher-sims", type=int, default=1000)
    parser.add_argument("--teacher-sims-applies-to", choices=("estimated", "all"), default="estimated")
    parser.add_argument("--min-regret", type=float, default=0.1)
    parser.add_argument("--min-override-delta", type=float, default=0.1)
    parser.add_argument("--suit-permutations", choices=("none", "all"), default="none")
    parser.add_argument(
        "--exclude-targets",
        action="append",
        default=[],
        help="Target JSONL path(s) to skip by stable target key; repeatable or semicolon-separated",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(convert(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
