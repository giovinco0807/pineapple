"""Extract aligned self-play action differences as active-teacher targets."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def action_key(record: dict[str, Any]) -> tuple[tuple[tuple[str, str], ...], str | None]:
    action = record["turn_log"].get("action", {})
    placements = tuple(sorted((str(card), row_name(str(pos))) for card, pos in action.get("placements", [])))
    discard = action.get("discard")
    return placements, None if discard is None else str(discard)


def row_name(value: str) -> str:
    return "middle" if value in {"mid", "middle"} else ("bottom" if value in {"bot", "bottom"} else value)


def board_key(board: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(board.get("top", [])),
        tuple(board.get("middle", board.get("mid", []))),
        tuple(board.get("bottom", board.get("bot", []))),
    )


def same_state(left: dict[str, Any], right: dict[str, Any]) -> bool:
    a = left["turn_log"]
    b = right["turn_log"]
    return (
        a.get("dealt_cards") == b.get("dealt_cards")
        and board_key(a.get("board_self", {})) == board_key(b.get("board_self", {}))
        and board_key(a.get("board_opponent", {})) == board_key(b.get("board_opponent", {}))
    )


def teacher_board(board: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "top": list(board.get("top", [])),
        "mid": list(board.get("middle", board.get("mid", []))),
        "bot": list(board.get("bottom", board.get("bot", []))),
    }


def flat_board_cards(board: dict[str, list[str]]) -> list[str]:
    return list(board.get("top", [])) + list(board.get("mid", [])) + list(board.get("bot", []))


def unique_cards(cards: Iterable[str], blocked: set[str] | None = None) -> list[str]:
    blocked = blocked or set()
    seen: set[str] = set()
    out: list[str] = []
    for card in cards:
        card = str(card)
        if not card or card in seen or card in blocked:
            continue
        seen.add(card)
        out.append(card)
    return out


def normalize_action(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "placements": [[str(card), row_name(str(pos))] for card, pos in action.get("placements", [])],
        "discard": action.get("discard"),
    }


def target_from_record(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    record_index: int,
    hand_index: int,
    reasons: list[str],
) -> dict[str, Any]:
    turn_log = baseline["turn_log"]
    board = teacher_board(turn_log.get("board_self", {}))
    opponent_board = teacher_board(turn_log.get("board_opponent", {}))
    dealt = list(turn_log.get("dealt_cards", []))
    own_cards = set(flat_board_cards(board)) | set(dealt)
    known_discards = unique_cards(turn_log.get("discards_self", []) or [], blocked=own_cards)
    exclude = unique_cards(flat_board_cards(opponent_board) + known_discards, blocked=own_cards)
    return {
        "source": "aligned_selfplay_route_diff",
        "source_line": record_index,
        "hand": hand_index,
        "record": record_index,
        "turn": int(turn_log.get("turn", -1)),
        "board": board,
        "opponent_board": opponent_board,
        "dealt": dealt,
        "known_discards": known_discards,
        "exclude": exclude,
        "is_btn": bool(turn_log.get("is_btn", False)),
        "player": int(turn_log.get("player", 0)),
        "reasons": reasons,
        "baseline_action": normalize_action(turn_log.get("action", {})),
        "candidate_action": normalize_action(candidate["turn_log"].get("action", {})),
        "baseline_action_idx": turn_log.get("action_idx"),
        "candidate_action_idx": candidate["turn_log"].get("action_idx"),
    }


def extract(args: argparse.Namespace) -> dict[str, Any]:
    baseline_records = load_records(Path(args.baseline))
    candidate_records = load_records(Path(args.candidate))
    if len(baseline_records) != len(candidate_records):
        raise ValueError(f"Record count mismatch: {len(baseline_records)} vs {len(candidate_records)}")

    turns = {int(part) for part in args.turns.split(",") if part.strip()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter()
    written = 0
    with output.open("w", encoding="utf-8") as f:
        for index, (baseline, candidate) in enumerate(zip(baseline_records, candidate_records), start=1):
            turn = int(baseline["turn_log"].get("turn", -1))
            if turn not in turns:
                continue
            if action_key(baseline) == action_key(candidate):
                continue
            state_matches = same_state(baseline, candidate)
            if args.same_state_only and not state_matches:
                stats["skipped_state_mismatch"] += 1
                continue
            reasons = ["model_action_diff", f"turn_{turn}"]
            if state_matches:
                reasons.append("same_state")
            else:
                reasons.append("state_mismatch")
            target = target_from_record(
                baseline,
                candidate,
                record_index=index,
                hand_index=(index - 1) // args.records_per_hand + 1,
                reasons=reasons,
            )
            f.write(json.dumps(target, ensure_ascii=False) + "\n")
            written += 1
            stats[f"turn_{turn}"] += 1
            stats["same_state" if state_matches else "state_mismatch"] += 1

    summary = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "output": str(output),
        "records": len(baseline_records),
        "records_per_hand": args.records_per_hand,
        "turns": sorted(turns),
        "same_state_only": bool(args.same_state_only),
        "written": written,
        "counts": dict(stats),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract aligned self-play action diffs as active-teacher targets")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--turns", default="2")
    parser.add_argument("--records-per-hand", type=int, default=10)
    parser.add_argument("--same-state-only", action="store_true", default=True)
    parser.add_argument("--include-state-mismatch", dest="same_state_only", action="store_false")
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(extract(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
