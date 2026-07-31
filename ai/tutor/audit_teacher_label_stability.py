"""Audit Top1 label stability between weak and strong teacher JSONL files.

This is intended for MC300 -> MC1000/exact checks.  A weak MC Top1 label should
not be treated as a hard Top1 target if the stronger teacher picks a different
candidate on the same state.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROW_NAMES = {"middle": "mid", "mid": "mid", "bottom": "bot", "bot": "bot", "top": "top"}


def row_name(value: str) -> str:
    return ROW_NAMES.get(str(value), str(value))


def normalize_cards(cards: Iterable[Any]) -> tuple[str, ...]:
    return tuple(str(card) for card in cards if str(card))


def board_key(board: dict[str, Any] | None) -> tuple[tuple[str, tuple[str, ...]], ...]:
    board = board or {}
    rows = {
        "top": normalize_cards(board.get("top", []) or []),
        "mid": normalize_cards((board.get("mid", []) or []) + (board.get("middle", []) or [])),
        "bot": normalize_cards((board.get("bot", []) or []) + (board.get("bottom", []) or [])),
    }
    return tuple(sorted(rows.items()))


def state_key(record: dict[str, Any]) -> str:
    stable = {
        "turn": int(record.get("turn", -1)),
        "board": board_key(record.get("board") or {}),
        "opponent_board": board_key(record.get("opponent_board") or record.get("board_opponent") or {}),
        "dealt": tuple(sorted(str(card) for card in (record.get("dealt") or []))),
        "known_discards": tuple(sorted(str(card) for card in (record.get("known_discards") or []))),
        "exclude": tuple(sorted(str(card) for card in (record.get("exclude") or []))),
        "is_btn": bool(record.get("is_btn", str(record.get("position", "")).lower() == "btn")),
    }
    return json.dumps(stable, sort_keys=True, separators=(",", ":"))


def source_key(record: dict[str, Any]) -> str | None:
    source = record.get("source")
    source_line = record.get("source_line")
    if source is None or source_line is None:
        return None
    decision_index = record.get("source_decision_index")
    return json.dumps(
        {
            "source": str(source),
            "source_line": int(source_line),
            "source_decision_index": None if decision_index is None else int(decision_index),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def action_key(action: dict[str, Any]) -> str:
    placements = tuple(
        sorted((str(card), row_name(str(pos))) for card, pos in (action.get("placements") or []))
    )
    discard = action.get("discard")
    stable = {"placements": placements, "discard": None if discard is None else str(discard)}
    return json.dumps(stable, sort_keys=True, separators=(",", ":"))


def action_text(action: dict[str, Any]) -> str:
    placements = ", ".join(f"{card}->{row_name(pos)}" for card, pos in (action.get("placements") or []))
    return f"{placements}; discard {action.get('discard')}"


def candidate_score(candidate: dict[str, Any]) -> float | None:
    for key in ("target_score", "score", "ev", "refined_score"):
        if key in candidate and candidate[key] is not None:
            return float(candidate[key])
    for block_name in ("recursive", "mc", "metrics"):
        block = candidate.get(block_name) or {}
        if block_name == "recursive":
            block = block.get("mc") or block
        for key in ("avg_score", "score", "ev"):
            if key in block and block[key] is not None:
                return float(block[key])
    return None


def teacher_mode(record: dict[str, Any]) -> str:
    return str(record.get("eval_mode") or record.get("teacher_eval_mode") or "")


def teacher_sims(candidate: dict[str, Any] | None) -> int | None:
    if candidate is None:
        return None
    for block_name in ("recursive", "mc", "metrics"):
        block = candidate.get(block_name) or {}
        if block_name == "recursive":
            block = block.get("mc") or block
        for key in ("simulations", "n_rollouts", "samples", "sims"):
            if key in block and block[key] is not None:
                return int(block[key])
    return None


def candidate_ranking(record: dict[str, Any]) -> list[dict[str, Any]]:
    ranked: list[tuple[int, float, dict[str, Any]]] = []
    for idx, candidate in enumerate(record.get("candidates") or []):
        score = candidate_score(candidate)
        if score is not None:
            ranked.append((idx, float(score), candidate))
    ranked.sort(key=lambda item: item[1], reverse=True)
    if not ranked:
        return []
    best_idx = record.get("best_idx")
    if best_idx is not None:
        best_idx = int(best_idx)
        # Keep externally supplied best_idx authoritative, then sort the rest by score.
        best_item = next((item for item in ranked if item[0] == best_idx), None)
        if best_item is not None:
            ranked = [best_item] + [item for item in ranked if item[0] != best_idx]
    return [
        {
            "idx": idx,
            "rank": rank,
            "score": score,
            "candidate": candidate,
            "key": action_key(candidate),
            "action": action_text(candidate),
        }
        for rank, (idx, score, candidate) in enumerate(ranked, start=1)
    ]


def margin(ranking: list[dict[str, Any]]) -> float | None:
    if len(ranking) < 2:
        return None
    return float(ranking[0]["score"]) - float(ranking[1]["score"])


def score_map(ranking: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item["key"]): item for item in ranking}


def load_records(path: Path, turns: set[int]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            record.setdefault("_line_no", line_no)
            if int(record.get("turn", -1)) in turns:
                records.append(record)
    return records


def index_records(records: list[dict[str, Any]], mode: str) -> tuple[dict[str, dict[str, Any]], Counter]:
    index: dict[str, dict[str, Any]] = {}
    stats = Counter()
    for record in records:
        key = source_key(record) if mode == "source" else state_key(record)
        if not key:
            stats["missing_key"] += 1
            continue
        if key in index:
            stats["duplicate_key"] += 1
            continue
        index[key] = record
    return index, stats


def margin_bin(value: float | None) -> str:
    if value is None:
        return "none"
    if value < 0.25:
        return "<0.25"
    if value < 0.5:
        return "0.25-0.5"
    if value < 1.0:
        return "0.5-1"
    if value < 2.0:
        return "1-2"
    return ">=2"


def topk_contains(ranking_map: dict[str, dict[str, Any]], key: str, k: int) -> bool:
    item = ranking_map.get(key)
    return bool(item and int(item["rank"]) <= k)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    turns = {int(part) for part in args.turns.split(",") if part.strip()}
    weak_records = load_records(Path(args.weak), turns)
    strong_records = load_records(Path(args.strong), turns)
    weak_index, weak_index_stats = index_records(weak_records, args.match)
    strong_index, strong_index_stats = index_records(strong_records, args.match)

    output_mismatches = Path(args.output_mismatches) if args.output_mismatches else None
    if output_mismatches:
        output_mismatches.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    by_turn: dict[int, Counter] = defaultdict(Counter)
    by_margin_bin: dict[str, Counter] = defaultdict(Counter)
    mismatch_rows: list[dict[str, Any]] = []

    common_keys = [key for key in strong_index if key in weak_index]
    if args.max_records:
        common_keys = common_keys[: int(args.max_records)]

    for key in common_keys:
        weak = weak_index[key]
        strong = strong_index[key]
        turn = int(strong.get("turn", weak.get("turn", -1)))
        weak_rank = candidate_ranking(weak)
        strong_rank = candidate_ranking(strong)
        if not weak_rank or not strong_rank:
            stats["missing_ranking"] += 1
            by_turn[turn]["missing_ranking"] += 1
            continue
        weak_map = score_map(weak_rank)
        strong_map = score_map(strong_rank)
        weak_best = weak_rank[0]
        strong_best = strong_rank[0]
        weak_margin = margin(weak_rank)
        strong_margin = margin(strong_rank)
        weak_best_in_strong = strong_map.get(str(weak_best["key"]))
        strong_best_in_weak = weak_map.get(str(strong_best["key"]))
        if weak_best_in_strong is None or strong_best_in_weak is None:
            stats["action_key_mismatch"] += 1
            by_turn[turn]["action_key_mismatch"] += 1
            continue
        same = str(weak_best["key"]) == str(strong_best["key"])
        stats["matched"] += 1
        by_turn[turn]["matched"] += 1
        bin_name = margin_bin(weak_margin)
        by_margin_bin[bin_name]["matched"] += 1
        if same:
            stats["top1_same"] += 1
            by_turn[turn]["top1_same"] += 1
            by_margin_bin[bin_name]["top1_same"] += 1
        else:
            stats["top1_changed"] += 1
            by_turn[turn]["top1_changed"] += 1
            by_margin_bin[bin_name]["top1_changed"] += 1
            row = {
                "match_key": key,
                "turn": turn,
                "weak_line": weak.get("_line_no"),
                "strong_line": strong.get("_line_no"),
                "source": strong.get("source") or weak.get("source"),
                "source_line": strong.get("source_line") or weak.get("source_line"),
                "weak_eval_mode": teacher_mode(weak),
                "strong_eval_mode": teacher_mode(strong),
                "weak_sims": teacher_sims(weak_best["candidate"]),
                "strong_sims": teacher_sims(strong_best["candidate"]),
                "weak_margin": weak_margin,
                "strong_margin": strong_margin,
                "weak_best_score": weak_best["score"],
                "strong_best_score": strong_best["score"],
                "weak_best_strong_score": weak_best_in_strong["score"],
                "strong_best_weak_score": strong_best_in_weak["score"],
                "weak_best_strong_rank": weak_best_in_strong["rank"],
                "strong_best_weak_rank": strong_best_in_weak["rank"],
                "weak_best_action": weak_best["action"],
                "strong_best_action": strong_best["action"],
            }
            mismatch_rows.append(row)
        for k in (3, 5, 10, 15):
            if topk_contains(strong_map, str(weak_best["key"]), k):
                stats[f"weak_top1_in_strong_top{k}"] += 1
                by_turn[turn][f"weak_top1_in_strong_top{k}"] += 1
            if topk_contains(weak_map, str(strong_best["key"]), k):
                stats[f"strong_top1_in_weak_top{k}"] += 1
                by_turn[turn][f"strong_top1_in_weak_top{k}"] += 1

    if output_mismatches:
        with output_mismatches.open("w", encoding="utf-8") as f:
            for row in mismatch_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    matched = int(stats["matched"])
    report = {
        "weak": str(args.weak),
        "strong": str(args.strong),
        "match": args.match,
        "turns": sorted(turns),
        "weak_records": len(weak_records),
        "strong_records": len(strong_records),
        "weak_index": len(weak_index),
        "strong_index": len(strong_index),
        "weak_index_stats": dict(weak_index_stats),
        "strong_index_stats": dict(strong_index_stats),
        "common_keys": len(common_keys),
        "matched": matched,
        "top1_same": int(stats["top1_same"]),
        "top1_changed": int(stats["top1_changed"]),
        "top1_same_rate": (float(stats["top1_same"]) / matched) if matched else 0.0,
        "weak_top1_in_strong_top3_rate": (float(stats["weak_top1_in_strong_top3"]) / matched) if matched else 0.0,
        "strong_top1_in_weak_top3_rate": (float(stats["strong_top1_in_weak_top3"]) / matched) if matched else 0.0,
        "counts": dict(stats),
        "by_turn": {str(turn): dict(counter) for turn, counter in sorted(by_turn.items())},
        "by_weak_margin_bin": {name: dict(counter) for name, counter in sorted(by_margin_bin.items())},
        "mismatches": str(output_mismatches) if output_mismatches else None,
    }
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit weak-vs-strong teacher Top1 label stability")
    parser.add_argument("--weak", required=True, help="Weak teacher JSONL, e.g. MC300")
    parser.add_argument("--strong", required=True, help="Strong teacher JSONL, e.g. MC1000 or exact")
    parser.add_argument("--match", choices=("state", "source"), default="state")
    parser.add_argument("--turns", default="0,1,2,3,4")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-mismatches", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = audit(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
