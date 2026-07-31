"""Convert runtime hybrid-refinement misses into active teacher targets."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.tutor.weak_groups_to_active_targets import (  # noqa: E402
    expand_suits,
    target_from_record,
    target_key,
    teacher_margin,
)


def parse_turns(raw: str) -> set[int]:
    return {int(part) for part in raw.split(",") if part.strip()}


def load_teacher_records(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                records[line_no] = json.loads(line)
    return records


def iter_results(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def result_regret(result: dict[str, Any]) -> float:
    best = result.get("teacher_best_score")
    final = result.get("final_teacher_score")
    if best is None or final is None:
        return 0.0
    return max(0.0, float(best) - float(final))


def override_delta(result: dict[str, Any]) -> float:
    """Positive when the runtime final action is worse than model Top1."""
    model = result.get("model_teacher_score")
    final = result.get("final_teacher_score")
    if model is None or final is None:
        return 0.0
    return float(model) - float(final)


def result_to_weak(result: dict[str, Any], min_teacher_margin: float) -> dict[str, Any]:
    regret = result_regret(result)
    return {
        "group_id": int(result.get("line", 0)) - 1,
        "turn": int(result.get("turn", -1)),
        "rank": 999,
        "regret": regret,
        "topk_exact_rerank_regret": {"1": regret, "3": regret, "5": regret, "10": regret, "20": regret},
        "selection_topk": 1,
        "min_teacher_margin": float(min_teacher_margin),
    }


def convert(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    turns = parse_turns(args.turns)
    teacher_records = load_teacher_records(Path(args.teacher_labels))
    stats = Counter()
    seen: set[str] = set()
    written = 0

    with output.open("w", encoding="utf-8") as dst:
        for result in iter_results(Path(args.results)):
            stats["results"] += 1
            turn = int(result.get("turn", -1))
            if turn not in turns:
                stats["skipped_turn"] += 1
                continue
            if bool(result.get("final_top1_hit")):
                stats["skipped_hit"] += 1
                continue
            if args.override_worse_only:
                if not bool(result.get("model_top1_overridden")):
                    stats["skipped_not_override"] += 1
                    continue
                delta = override_delta(result)
                if delta <= 0.0:
                    stats["skipped_not_worse_override"] += 1
                    continue
                if args.min_override_delta > 0.0 and delta < args.min_override_delta:
                    stats["skipped_low_override_delta"] += 1
                    continue
            if args.model_misses_only and bool(result.get("model_top1_hit")):
                stats["skipped_model_hit"] += 1
                continue
            if args.min_regret > 0.0 and result_regret(result) < args.min_regret:
                stats["skipped_low_regret"] += 1
                continue

            line_no = int(result.get("line", 0))
            record = teacher_records.get(line_no)
            if record is None:
                stats["skipped_missing_teacher"] += 1
                continue
            margin_info = teacher_margin(record)
            margin = margin_info.get("teacher_margin")
            eval_mode = str(record.get("eval_mode") or result.get("teacher_eval_mode") or "")
            is_exact = "exact" in eval_mode.lower()
            if (
                args.min_teacher_margin > 0.0
                and (args.margin_applies_to == "all" or not is_exact)
                and (margin is None or float(margin) < args.min_teacher_margin)
            ):
                stats["skipped_low_teacher_margin"] += 1
                continue
            sims = margin_info.get("teacher_sims")
            if (
                args.min_teacher_sims > 0
                and (args.teacher_sims_applies_to == "all" or not is_exact)
                and (sims is None or int(sims) < args.min_teacher_sims)
            ):
                stats["skipped_low_teacher_sims"] += 1
                continue

            weak = result_to_weak(result, args.min_teacher_margin)
            weak["min_teacher_sims"] = int(args.min_teacher_sims)
            target = target_from_record(record, weak)
            reasons = list(target.get("reasons") or [])
            for reason in (
                "runtime_final_top1_miss",
                f"runtime_{args.refinement_tag}_miss",
                "high_confidence_teacher_margin"
                if margin is not None and float(margin) >= args.min_teacher_margin
                else "",
                "high_confidence_teacher_sims"
                if args.min_teacher_sims > 0 and sims is not None and int(sims) >= args.min_teacher_sims
                else "",
            ):
                if reason and reason not in reasons:
                    reasons.append(reason)
            target["reasons"] = reasons
            target["runtime_refinement_tag"] = args.refinement_tag
            target["runtime_result_line"] = line_no
            target["runtime_model_top1_hit"] = bool(result.get("model_top1_hit"))
            target["runtime_final_top1_hit"] = bool(result.get("final_top1_hit"))
            target["runtime_model_action_idx"] = result.get("model_top1_action_idx")
            target["runtime_final_action_idx"] = result.get("best_action_idx")
            target["runtime_model_teacher_score"] = result.get("model_teacher_score")
            target["runtime_final_teacher_score"] = result.get("final_teacher_score")
            target["runtime_override_worse"] = bool(override_delta(result) > 0.0)
            target["runtime_override_delta"] = override_delta(result)
            target["runtime_teacher_best_score"] = result.get("teacher_best_score")
            target["runtime_exact_evaluated"] = result.get("exact_evaluated")
            target["runtime_elapsed_ms"] = result.get("elapsed_ms")

            for expanded in expand_suits(target, args.suit_permutations):
                key = target_key(expanded)
                if key in seen:
                    stats["skipped_duplicate"] += 1
                    continue
                seen.add(key)
                dst.write(json.dumps(expanded, ensure_ascii=False) + "\n")
                written += 1
                stats[f"turn_{expanded['turn']}"] += 1
                for reason in expanded.get("reasons", []):
                    stats[f"reason_{reason}"] += 1

    summary = {
        "results": str(args.results),
        "teacher_labels": str(args.teacher_labels),
        "output": str(output),
        "turns": sorted(turns),
        "min_teacher_margin": float(args.min_teacher_margin),
        "margin_applies_to": args.margin_applies_to,
        "min_teacher_sims": int(args.min_teacher_sims),
        "teacher_sims_applies_to": args.teacher_sims_applies_to,
        "min_regret": float(args.min_regret),
        "model_misses_only": bool(args.model_misses_only),
        "override_worse_only": bool(args.override_worse_only),
        "min_override_delta": float(args.min_override_delta),
        "refinement_tag": args.refinement_tag,
        "suit_permutations": args.suit_permutations,
        "written": written,
        "counts": dict(stats),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert hybrid refinement misses to active teacher targets")
    parser.add_argument("--results", required=True, help="Runtime evaluation results.jsonl")
    parser.add_argument("--teacher-labels", required=True, help="Teacher JSONL used by the runtime evaluation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--turns", default="1,2")
    parser.add_argument("--min-teacher-margin", type=float, default=1.0)
    parser.add_argument("--margin-applies-to", choices=("estimated", "all"), default="estimated")
    parser.add_argument("--min-teacher-sims", type=int, default=0,
                        help="Skip estimated/all targets whose teacher best has fewer simulations")
    parser.add_argument("--teacher-sims-applies-to", choices=("estimated", "all"), default="estimated")
    parser.add_argument("--min-regret", type=float, default=0.0)
    parser.add_argument("--model-misses-only", action="store_true")
    parser.add_argument("--override-worse-only", action="store_true")
    parser.add_argument("--min-override-delta", type=float, default=0.0)
    parser.add_argument("--refinement-tag", default="mc_board")
    parser.add_argument("--suit-permutations", choices=("none", "all"), default="none")
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(convert(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
