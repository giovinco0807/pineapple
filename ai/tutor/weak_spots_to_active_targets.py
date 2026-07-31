"""Convert model-evaluation weak spots into active-teacher target states."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def row(board: Dict[str, Any], *names: str) -> List[str]:
    out: List[str] = []
    for name in names:
        out.extend(board.get(name, []) or [])
    return [str(card) for card in out if card]


def teacher_board(board: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    board = board or {}
    return {
        "top": row(board, "top"),
        "mid": row(board, "middle", "mid"),
        "bot": row(board, "bottom", "bot"),
    }


def unique_cards(cards: Iterable[str], blocked: Optional[set[str]] = None) -> List[str]:
    blocked = blocked or set()
    seen: set[str] = set()
    out: List[str] = []
    for card in cards:
        card = str(card)
        if not card or card in seen or card in blocked:
            continue
        seen.add(card)
        out.append(card)
    return out


def flat_cards(board: Dict[str, Any]) -> List[str]:
    return row(board, "top") + row(board, "mid", "middle") + row(board, "bot", "bottom")


def convert_item(item: Dict[str, Any]) -> Dict[str, Any]:
    board = teacher_board(item.get("board") or {})
    opponent = teacher_board(item.get("opponent_board") or {})
    dealt = unique_cards(item.get("dealt", []) or [])
    own_cards = set(flat_cards(board)) | set(dealt)
    opp_cards = flat_cards(opponent)
    known = unique_cards(item.get("known_discards", []) or [], blocked=own_cards)
    exclude = unique_cards(opp_cards + known, blocked=own_cards)
    tags = list(item.get("tags", []) or [])
    if "high_regret" not in tags and float(item.get("regret", 0.0)) >= 2.0:
        tags.append("high_regret")
    if "miss_top3" not in tags and int(item.get("true_best_pred_rank", 1)) > 3:
        tags.append("miss_top3")

    return {
        "source": str(item.get("source") or "weak_spots"),
        "source_line": item.get("decision_index"),
        "turn": int(item.get("turn", -1)),
        "board": board,
        "opponent_board": opponent,
        "dealt": dealt,
        "known_discards": known,
        "exclude": exclude,
        "is_btn": str(item.get("position", "")).lower() == "btn",
        "player": 0,
        "reasons": tags,
        "regret": float(item.get("regret", 0.0)),
        "true_best_pred_rank": int(item.get("true_best_pred_rank", 0)),
        "teacher_best": item.get("teacher_best"),
        "model_pick": item.get("model_pick"),
        "teacher_gap": float(item.get("teacher_gap", 0.0)),
    }


def load_weak_spots(path: Path, turns: set[int], min_regret: float) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            turn = int(item.get("turn", -1))
            if turn not in turns:
                continue
            if float(item.get("regret", 0.0)) < min_regret:
                continue
            items.append(item)
    items.sort(
        key=lambda item: (
            float(item.get("regret", 0.0)),
            int(item.get("true_best_pred_rank", 0)),
            float(item.get("fl_rate_mae", 0.0)),
            float(item.get("bust_rate_mae", 0.0)),
        ),
        reverse=True,
    )
    return items


def convert(args: argparse.Namespace) -> Dict[str, Any]:
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    turns = {int(t) for t in args.turns.split(",") if t.strip()}
    items = load_weak_spots(input_path, turns, args.min_regret)
    if args.max_records:
        items = items[: args.max_records]

    stats = Counter()
    with output_path.open("w", encoding="utf-8") as out:
        for item in items:
            target = convert_item(item)
            out.write(json.dumps(target, ensure_ascii=False) + "\n")
            stats[f"turn_{target['turn']}"] += 1
            for reason in target.get("reasons", []):
                stats[f"reason_{reason}"] += 1

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "records": len(items),
        "turns": sorted(turns),
        "min_regret": args.min_regret,
        "max_records": args.max_records,
        "counts": dict(stats),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Convert weak_spots.jsonl to active-teacher targets")
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--turns", default="1,3")
    parser.add_argument("--min-regret", type=float, default=1.0)
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(convert(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
