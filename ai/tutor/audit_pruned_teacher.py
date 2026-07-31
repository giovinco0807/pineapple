"""Audit whether a full teacher's best action survived top-k pruning.

Inputs:
  1. targets from ai.tutor.extract_selfplay_targets, preserving
     top_k_indices/action_evs from the pruned self-play run.
  2. full teacher labels from ai.training.generate_active_teacher for those
     same targets.

The main metric is top-k hit rate: full-teacher best action index is present in
the pruned self-play candidate set.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.training.convert_mc_teacher import (
    action_to_index_t0,
    action_to_index_t1plus,
    get_candidate_bust,
    get_candidate_ev,
    get_candidate_fl,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _source_key(record: dict[str, Any]) -> tuple[str, int] | None:
    source = record.get("source")
    source_line = record.get("source_line")
    if source is None or source_line is None:
        return None
    try:
        return (str(Path(str(source)).resolve()).lower(), int(source_line))
    except (OSError, TypeError, ValueError):
        return (str(source).lower(), int(source_line))


def _candidate_action_index(record: dict[str, Any], candidate: dict[str, Any]) -> int:
    turn = int(record.get("turn", 0))
    dealt = list(record.get("dealt") or [])
    if turn == 0:
        return int(action_to_index_t0(candidate.get("placements", []), dealt))
    return int(action_to_index_t1plus(candidate, dealt))


def _fmt_action(candidate: dict[str, Any]) -> str:
    placements = "; ".join(f"{card}->{pos}" for card, pos in candidate.get("placements", []))
    discard = candidate.get("discard")
    if discard:
        return f"{placements}; discard {discard}"
    return placements


def audit(args: argparse.Namespace) -> dict[str, Any]:
    targets = _load_jsonl(Path(args.targets))
    labels = _load_jsonl(Path(args.labels))
    targets_by_key = {
        key: target
        for target in targets
        if (key := _source_key(target)) is not None
    }

    stats: Counter[str] = Counter()
    by_turn: dict[int, Counter[str]] = defaultdict(Counter)
    misses: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = []

    for label in labels:
        key = _source_key(label)
        target = targets_by_key.get(key) if key is not None else None
        if target is None:
            stats["missing_target"] += 1
            continue
        turn = int(label.get("turn", target.get("turn", -1)))
        candidates = list(label.get("candidates") or [])
        if not candidates:
            stats["empty_labels"] += 1
            continue
        best_idx = int(label.get("best_idx", 0) or 0)
        if best_idx < 0 or best_idx >= len(candidates):
            best_idx = 0
        best = candidates[best_idx]
        best_action_idx = _candidate_action_index(label, best)
        pruned_indices = {int(idx) for idx in target.get("top_k_indices") or []}
        evaluated_actions = int(target.get("evaluated_actions") or len(pruned_indices))
        n_actions = int(target.get("n_actions") or 0)
        top_k_hit = best_action_idx in pruned_indices
        selfplay_action_idx = target.get("selfplay_action_idx")
        selfplay_hit = selfplay_action_idx == best_action_idx

        stats["records"] += 1
        by_turn[turn]["records"] += 1
        if evaluated_actions and n_actions and evaluated_actions < n_actions:
            stats["pruned_records"] += 1
            by_turn[turn]["pruned_records"] += 1
        if top_k_hit:
            stats["topk_hit"] += 1
            by_turn[turn]["topk_hit"] += 1
        else:
            stats["topk_miss"] += 1
            by_turn[turn]["topk_miss"] += 1
        if selfplay_hit:
            stats["selfplay_best"] += 1
            by_turn[turn]["selfplay_best"] += 1

        row = {
            "source": target.get("source"),
            "source_line": target.get("source_line"),
            "turn": turn,
            "is_btn": bool(target.get("is_btn", False)),
            "n_actions": n_actions,
            "evaluated_actions": evaluated_actions,
            "pruned_top_k": int(target.get("pruned_top_k") or 0),
            "best_action_idx": best_action_idx,
            "selfplay_action_idx": selfplay_action_idx,
            "best_action": _fmt_action(best),
            "best_ev": float(get_candidate_ev(best, str(label.get("eval_mode", "")))),
            "best_fl": float(get_candidate_fl(best, str(label.get("eval_mode", "")))),
            "best_bust": float(get_candidate_bust(best, str(label.get("eval_mode", "")))),
            "dealt": list(target.get("dealt") or []),
            "board": target.get("board"),
            "opponent_board": target.get("opponent_board"),
        }
        if top_k_hit:
            hits.append(row)
        else:
            misses.append(row)

    def rate(num: int, den: int) -> float:
        return 0.0 if den <= 0 else num / den

    turn_summary = {}
    for turn, counts in sorted(by_turn.items()):
        total = int(counts["records"])
        turn_summary[str(turn)] = {
            "records": total,
            "pruned_records": int(counts["pruned_records"]),
            "topk_hit": int(counts["topk_hit"]),
            "topk_miss": int(counts["topk_miss"]),
            "topk_hit_rate": rate(int(counts["topk_hit"]), total),
            "selfplay_best": int(counts["selfplay_best"]),
            "selfplay_best_rate": rate(int(counts["selfplay_best"]), total),
        }

    summary = {
        "targets": args.targets,
        "labels": args.labels,
        "records": int(stats["records"]),
        "pruned_records": int(stats["pruned_records"]),
        "topk_hit": int(stats["topk_hit"]),
        "topk_miss": int(stats["topk_miss"]),
        "topk_hit_rate": rate(int(stats["topk_hit"]), int(stats["records"])),
        "selfplay_best": int(stats["selfplay_best"]),
        "selfplay_best_rate": rate(int(stats["selfplay_best"]), int(stats["records"])),
        "missing_target": int(stats["missing_target"]),
        "empty_labels": int(stats["empty_labels"]),
        "turns": turn_summary,
        "miss_examples": misses[: args.max_examples],
        "hit_examples": hits[: args.max_examples],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    miss_path = output.with_suffix(".misses.jsonl")
    with miss_path.open("w", encoding="utf-8") as f:
        for item in misses:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    md_path = output.with_suffix(".md")
    lines = [
        "# Top-K Pruning Audit",
        "",
        f"- targets: `{Path(args.targets).name}`",
        f"- labels: `{Path(args.labels).name}`",
        f"- records: {summary['records']}",
        f"- top-k hit: {summary['topk_hit']}/{summary['records']} ({summary['topk_hit_rate']:.2%})",
        f"- top-k miss: {summary['topk_miss']}",
        f"- self-play chose full best: {summary['selfplay_best']}/{summary['records']} ({summary['selfplay_best_rate']:.2%})",
        "",
        "## By Turn",
        "",
        "| turn | records | pruned | top-k hit | miss | hit rate | self-play best |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for turn, counts in summary["turns"].items():
        lines.append(
            f"| T{turn} | {counts['records']} | {counts['pruned_records']} | "
            f"{counts['topk_hit']} | {counts['topk_miss']} | {counts['topk_hit_rate']:.2%} | "
            f"{counts['selfplay_best_rate']:.2%} |"
        )
    if misses:
        lines.extend(["", "## Miss Examples", ""])
        for item in misses[: args.max_examples]:
            lines.extend(
                [
                    f"### source line {item['source_line']} / T{item['turn']}",
                    "",
                    f"- dealt: {' '.join(item['dealt'])}",
                    f"- board: T:{' '.join(item['board']['top'])} | M:{' '.join(item['board']['mid'])} | B:{' '.join(item['board']['bot'])}",
                    f"- best: {item['best_action']}",
                    f"- best EV: {item['best_ev']:.3f}, FL: {item['best_fl']:.1%}, bust: {item['best_bust']:.1%}",
                    f"- best action idx: {item['best_action_idx']}, self-play idx: {item['selfplay_action_idx']}",
                    "",
                ]
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary["misses_path"] = str(miss_path)
    summary["markdown_path"] = str(md_path)
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit full-teacher best action against pruned self-play candidates")
    parser.add_argument("--targets", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(audit(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
