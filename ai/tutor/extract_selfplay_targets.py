"""Extract teacher/audit targets from self-play JSONL.

This is intentionally broader than the older active-target extractor: it can
write every decision state from a top-k-pruned self-play run, preserving the
pruning metadata needed to audit whether the full teacher's best action was
inside the pruned candidate set.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def _row(board: dict[str, Any], *names: str) -> list[str]:
    out: list[str] = []
    for name in names:
        out.extend(board.get(name, []) or [])
    return out


def _teacher_board(board: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "top": list(board.get("top", [])),
        "mid": _row(board, "middle", "mid"),
        "bot": _row(board, "bottom", "bot"),
    }


def _flat_board(board: dict[str, list[str]]) -> list[str]:
    return list(board.get("top", [])) + list(board.get("mid", [])) + list(board.get("bot", []))


def _unique(cards: Iterable[str], blocked: set[str] | None = None) -> list[str]:
    blocked = blocked or set()
    seen: set[str] = set()
    out: list[str] = []
    for card in cards:
        card = str(card)
        if not card or card in blocked or card in seen:
            continue
        seen.add(card)
        out.append(card)
    return out


def _norm_action(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "placements": [[str(card), str(pos)] for card, pos in action.get("placements", [])],
        "discard": action.get("discard"),
    }


def _target_from_selfplay(input_path: Path, line_no: int, record: dict[str, Any]) -> dict[str, Any] | None:
    turn_log = record.get("turn_log") or {}
    turn = int(turn_log.get("turn", -1))
    dealt = list(turn_log.get("dealt_cards") or [])
    if turn < 0 or not dealt:
        return None

    board = _teacher_board(turn_log.get("board_self") or {})
    opponent_board = _teacher_board(turn_log.get("board_opponent") or {})
    own_cards = set(_flat_board(board)) | set(dealt)
    known_discards = _unique(turn_log.get("discards_self") or [], blocked=own_cards)
    exclude = _unique(_flat_board(opponent_board) + known_discards, blocked=own_cards)

    reasons = [f"turn_{turn}", "selfplay_decision"]
    pruned_top_k = int(turn_log.get("pruned_top_k") or 0)
    evaluated_actions = int(turn_log.get("evaluated_actions") or 0)
    n_actions = int(turn_log.get("n_actions") or 0)
    if pruned_top_k > 0 and evaluated_actions and evaluated_actions < n_actions:
        reasons.append(f"pruned_top_{pruned_top_k}")
    if bool(turn_log.get("estimated", True)):
        reasons.append("estimated")

    player = str(turn_log.get("player", 0))
    hand = record.get("hand_result") or {}
    return {
        "source": str(input_path),
        "source_line": int(line_no),
        "turn": turn,
        "board": board,
        "opponent_board": opponent_board,
        "dealt": dealt,
        "known_discards": known_discards,
        "exclude": exclude,
        "is_btn": bool(turn_log.get("is_btn", False)),
        "player": int(turn_log.get("player", 0)),
        "reasons": reasons,
        "selfplay_action": _norm_action(turn_log.get("action") or {}),
        "selfplay_action_idx": turn_log.get("action_idx"),
        "n_actions": n_actions,
        "evaluated_actions": evaluated_actions,
        "top_k_indices": list(turn_log.get("top_k_indices") or []),
        "action_evs": list(turn_log.get("action_evs") or []),
        "eval_mode": turn_log.get("eval_mode"),
        "sims": int(turn_log.get("sims") or 0),
        "estimated": bool(turn_log.get("estimated", True)),
        "pruned_top_k": pruned_top_k,
        "prune_model": turn_log.get("prune_model"),
        "busted": bool((hand.get("busted") or {}).get(player, False)),
        "fl_entry": bool((hand.get("fl_entry") or {}).get(player, False)),
        "fl_cards": int((hand.get("fl_cards") or {}).get(player, 0)),
        "reward": float(record.get("reward", 0.0)),
    }


def extract(args: argparse.Namespace) -> dict[str, Any]:
    input_paths = [Path(path) for path in args.inputs]
    turns = {int(part) for part in args.turns.split(",") if part.strip()}
    rng = random.Random(args.seed)
    by_turn: dict[int, list[dict[str, Any]]] = defaultdict(list)
    stats: Counter[str] = Counter()

    for input_path in input_paths:
        with input_path.open("r", encoding="utf-8") as src:
            for line_no, line in enumerate(src, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    stats["bad_json"] += 1
                    continue
                target = _target_from_selfplay(input_path, line_no, record)
                if target is None:
                    stats["skipped"] += 1
                    continue
                turn = int(target["turn"])
                if turn not in turns:
                    continue
                if args.require_pruned and int(target.get("pruned_top_k") or 0) <= 0:
                    stats["not_pruned"] += 1
                    continue
                if args.sample_rate < 1.0 and rng.random() > args.sample_rate:
                    stats["sampled_out"] += 1
                    continue
                by_turn[turn].append(target)
                stats[f"turn_{turn}_available"] += 1

    selected: list[dict[str, Any]] = []
    for turn in sorted(by_turn):
        items = by_turn[turn]
        rng.shuffle(items)
        if args.per_turn > 0:
            items = items[: args.per_turn]
        selected.extend(items)

    rng.shuffle(selected)
    if args.max_records > 0:
        selected = selected[: args.max_records]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as dst:
        for target in selected:
            dst.write(json.dumps(target, ensure_ascii=False) + "\n")
            stats[f"turn_{target['turn']}_written"] += 1

    summary = {
        "inputs": [str(path) for path in input_paths],
        "output": str(output),
        "turns": sorted(turns),
        "seed": int(args.seed),
        "sample_rate": float(args.sample_rate),
        "per_turn": int(args.per_turn),
        "max_records": int(args.max_records),
        "require_pruned": bool(args.require_pruned),
        "written": len(selected),
        "counts": dict(stats),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract self-play states for teacher generation or top-k audit")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--turns", default="0,1,2,3,4")
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--per-turn", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--require-pruned", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(extract(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
