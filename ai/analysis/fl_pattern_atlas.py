"""Build a decision-level FL pattern atlas from self-play JSONL."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


RANK_VALUE = {r: i for i, r in enumerate("23456789TJQKA", start=2)}


def rank(card: str) -> str:
    if str(card).startswith("X"):
        return "X"
    return str(card)[0] if card else ""


def top_bucket(cards: list[str]) -> str:
    if not cards:
        return "top_empty"
    ranks = [rank(c) for c in cards]
    jokers = ranks.count("X")
    counts = Counter(r for r in ranks if r != "X")
    for r, name in (("A", "aa"), ("K", "kk"), ("Q", "qq")):
        if counts.get(r, 0) + jokers >= 2:
            return f"top_fl_{name}"
    if any(c + jokers >= 3 for c in counts.values()) or jokers >= 2:
        return "top_fl_trips"
    if any(c >= 2 for c in counts.values()) or (jokers and counts):
        return "top_pair_low"
    high = sorted((r for r in ranks if r in {"Q", "K", "A", "X"}))
    if high:
        return "top_high_" + "".join(high)
    if len(cards) >= 3:
        return "top_filled_other"
    return "top_low_partial"


def has_flush_draw(cards: list[str]) -> bool:
    suits = [c[-1] for c in cards if len(c) == 2 and not c.startswith("X")]
    return bool(suits) and max(Counter(suits).values()) >= min(4, max(2, len(cards)))


def has_straight_draw(cards: list[str]) -> bool:
    vals = sorted({RANK_VALUE.get(rank(c), 0) for c in cards if rank(c) not in {"", "X"}})
    if len(vals) < 3:
        return False
    for i in range(len(vals)):
        window = vals[i:i + 4]
        if len(window) >= 3 and max(window) - min(window) <= 4:
            return True
    return False


def five_row_bucket(cards: list[str], prefix: str) -> str:
    if not cards:
        return f"{prefix}_empty"
    ranks = [rank(c) for c in cards]
    jokers = ranks.count("X")
    counts = sorted(Counter(r for r in ranks if r != "X").values(), reverse=True)
    top_count = (counts[0] if counts else 0) + jokers
    if top_count >= 4:
        made = "quads"
    elif top_count >= 3 and len(counts) > 1 and counts[1] >= 2:
        made = "full_house"
    elif top_count >= 3:
        made = "trips"
    elif sum(1 for c in counts if c >= 2) >= 2:
        made = "two_pair"
    elif top_count >= 2:
        made = "pair"
    else:
        made = "high"

    draws = []
    if has_flush_draw(cards):
        draws.append("flush_draw")
    if has_straight_draw(cards):
        draws.append("straight_draw")
    draw_suffix = "_" + "_".join(draws) if draws else ""
    size = "complete" if len(cards) == 5 else f"{len(cards)}c"
    return f"{prefix}_{made}_{size}{draw_suffix}"


def fl_card_name(cards: int) -> str:
    return {14: "qq", 15: "kk", 16: "aa", 17: "trips"}.get(int(cards), "none")


def analyze(args: argparse.Namespace) -> None:
    input_paths = [Path(path) for path in args.inputs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    buckets = defaultdict(lambda: {
        "n": 0,
        "fl": 0,
        "bust": 0,
        "reward_sum": 0.0,
        "fl_shape_but_foul": 0,
        "fl_cards": Counter(),
    })

    total_lines = 0
    for input_path in input_paths:
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                total_lines += 1
                if args.max_lines and total_lines > args.max_lines:
                    break
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                turn_log = record.get("turn_log", {})
                board = turn_log.get("board_self", {})
                player = str(turn_log.get("player", 0))
                hand = record.get("hand_result", {})
                key = (
                    int(turn_log.get("turn", -1)),
                    top_bucket(list(board.get("top", []))),
                    five_row_bucket(list(board.get("middle", [])), "mid"),
                    five_row_bucket(list(board.get("bottom", [])), "bot"),
                )
                entry = buckets[key]
                fl = bool(hand.get("fl_entry", {}).get(player, False))
                bust = bool(hand.get("busted", {}).get(player, False))
                fl_cards = int(hand.get("fl_cards", {}).get(player, 0))
                entry["n"] += 1
                entry["fl"] += int(fl)
                entry["bust"] += int(bust)
                entry["reward_sum"] += float(record.get("reward", 0.0))
                entry["fl_cards"][fl_card_name(fl_cards)] += int(fl)
                if bust and key[1].startswith("top_fl_"):
                    entry["fl_shape_but_foul"] += 1
            if args.max_lines and total_lines > args.max_lines:
                break

    rows = []
    for key, entry in buckets.items():
        n = entry["n"]
        if n < args.min_count:
            continue
        rows.append({
            "turn": key[0],
            "top": key[1],
            "middle": key[2],
            "bottom": key[3],
            "n": n,
            "fl_rate": entry["fl"] / n,
            "bust_rate": entry["bust"] / n,
            "avg_reward": entry["reward_sum"] / n,
            "fl_shape_but_foul_rate": entry["fl_shape_but_foul"] / n,
            "fl_cards": dict(entry["fl_cards"]),
        })
    rows.sort(key=lambda r: (r["turn"], -r["n"], -r["fl_rate"]))

    json_path = output_dir / "fl_pattern_atlas.json"
    md_path = output_dir / "fl_pattern_atlas.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"inputs": [str(path) for path in input_paths], "rows": rows}, f, indent=2)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# FL Pattern Atlas\n\n")
        f.write("- inputs:\n")
        for input_path in input_paths:
            f.write(f"  - `{input_path}`\n")
        f.write(f"- buckets: {len(rows):,}\n")
        f.write(f"- min_count: {args.min_count}\n\n")
        f.write("| turn | top | middle | bottom | n | FL | bust | FL-shape foul | avg reward |\n")
        f.write("| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for r in rows[:args.markdown_rows]:
            f.write(
                f"| {r['turn']} | {r['top']} | {r['middle']} | {r['bottom']} | "
                f"{r['n']} | {r['fl_rate']:.1%} | {r['bust_rate']:.1%} | "
                f"{r['fl_shape_but_foul_rate']:.1%} | {r['avg_reward']:+.2f} |\n"
            )

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build FL pattern atlas from self-play JSONL")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--markdown-rows", type=int, default=80)
    parser.add_argument("--max-lines", type=int, default=0)
    args = parser.parse_args(argv)
    analyze(args)


if __name__ == "__main__":
    main()
