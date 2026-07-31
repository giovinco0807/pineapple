"""Convert oracle/model p99 sample JSONL rows into active-teacher targets."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def row(board: dict[str, Any], *names: str) -> list[str]:
    cards: list[str] = []
    for name in names:
        cards.extend(board.get(name, []) or [])
    return [str(card) for card in cards if card]


def normalize_board(board: dict[str, Any] | None) -> dict[str, list[str]]:
    board = board or {}
    return {
        "top": row(board, "top"),
        "mid": row(board, "middle", "mid"),
        "bot": row(board, "bottom", "bot"),
    }


def flat_cards(board: dict[str, Any]) -> list[str]:
    return row(board, "top") + row(board, "mid", "middle") + row(board, "bot", "bottom")


def unique_cards(cards: Iterable[str], blocked: set[str] | None = None) -> list[str]:
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


def infer_is_btn(state: dict[str, Any]) -> bool:
    meta = state.get("meta") or {}
    try:
        return float(meta.get("is_btn", 0.0)) >= 0.5
    except (TypeError, ValueError):
        return bool(meta.get("is_btn", False))


def convert_item(item: dict[str, Any], turn: int, source: str, line_no: int) -> dict[str, Any] | None:
    state = item.get("state") or {}
    board = normalize_board(state.get("board") or item.get("board"))
    opponent = normalize_board(state.get("opponent_board") or item.get("opponent_board"))
    dealt = unique_cards(state.get("dealt", []) or item.get("dealt", []))
    if not dealt:
        return None

    own_cards = set(flat_cards(board)) | set(dealt)
    known = unique_cards(state.get("discards", []) or item.get("known_discards", []), blocked=own_cards)
    exclude = unique_cards(flat_cards(opponent) + known, blocked=own_cards)
    regret = float(item.get("regret", 0.0) or 0.0)
    valid_action_count = int(item.get("valid_action_count", 0) or 0)

    reasons = ["p99_sample"]
    if regret >= 2.0:
        reasons.append("high_regret")
    if valid_action_count >= 20:
        reasons.append("wide_action_set")
    if turn == 2:
        reasons.append("t2")
    elif turn == 3:
        reasons.append("t3")

    return {
        "source": source,
        "source_line": line_no,
        "turn": int(turn),
        "board": board,
        "opponent_board": opponent,
        "dealt": dealt,
        "known_discards": known,
        "exclude": exclude,
        "is_btn": infer_is_btn(state),
        "player": 0,
        "reasons": reasons,
        "regret": regret,
        "valid_action_count": valid_action_count,
        "true_action_cards": item.get("true_action_cards"),
        "pred_action_cards": item.get("pred_action_cards"),
        "true_top3_by_ev": item.get("true_top3_by_ev"),
        "pred_top3_by_logit": item.get("pred_top3_by_logit"),
    }


def convert(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    written = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as out:
        for line_no, line in enumerate(src, start=1):
            if args.max_records and written >= args.max_records:
                break
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                stats["skipped_bad_json"] += 1
                continue
            if float(item.get("regret", 0.0) or 0.0) < args.min_regret:
                stats["skipped_low_regret"] += 1
                continue
            target = convert_item(item, args.turn, str(input_path), line_no)
            if target is None:
                stats["skipped_invalid"] += 1
                continue
            out.write(json.dumps(target, ensure_ascii=False) + "\n")
            written += 1
            stats[f"turn_{target['turn']}"] += 1
            stats[f"is_btn_{int(bool(target['is_btn']))}"] += 1
            for reason in target["reasons"]:
                stats[f"reason_{reason}"] += 1

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "turn": int(args.turn),
        "records": int(written),
        "min_regret": float(args.min_regret),
        "max_records": int(args.max_records),
        "counts": dict(stats),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert p99 sample rows to active-teacher targets")
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--turn", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--min-regret", type=float, default=1.0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(convert(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
