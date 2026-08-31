"""Summarize self-play JSONL at the normal-hand level.

`ai/self_play.py` writes one JSON record per seat/turn, so each normal hand is
usually represented by 10 records.  This script chunks those records back into
hands and reports FL entry rates without counting each turn as a separate hand.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


FL_CARD_NAMES = {
    14: "qq",
    15: "kk",
    16: "aa",
    17: "trips",
}
HIGH_FL_RANKS = {"Q", "K", "A"}


def rank(card: str) -> str:
    if str(card).startswith("X"):
        return "X"
    return str(card)[0] if card else ""


def top_has_fl_shape(top: list[str]) -> bool:
    ranks = [rank(card) for card in top]
    jokers = ranks.count("X")
    counts = Counter(r for r in ranks if r != "X")
    for high_rank in HIGH_FL_RANKS:
        if counts.get(high_rank, 0) + jokers >= 2:
            return True
    return any(count + jokers >= 3 for count in counts.values()) or jokers >= 2


def empty_stats() -> dict:
    return {
        "hands": 0,
        "seat_games": 0,
        "fl_entries": 0,
        "busts": 0,
        "hand_any_fl": 0,
        "hand_both_fl": 0,
        "fl_shape_but_foul": 0,
        "fl_types": Counter(),
        "reward0_sum": 0.0,
        "reward1_sum": 0.0,
    }


def merge_stats(dst: dict, src: dict) -> None:
    for key in (
        "hands",
        "seat_games",
        "fl_entries",
        "busts",
        "hand_any_fl",
        "hand_both_fl",
        "fl_shape_but_foul",
    ):
        dst[key] += src[key]
    dst["reward0_sum"] += src["reward0_sum"]
    dst["reward1_sum"] += src["reward1_sum"]
    dst["fl_types"].update(src["fl_types"])


def reconstruct_final_boards(block: list[dict]) -> dict[str, dict[str, list[str]]]:
    boards: dict[str, dict[str, list[str]]] = {}
    for record in block:
        turn_log = record.get("turn_log", {})
        player = str(turn_log.get("player", 0))
        before = turn_log.get("board_self", {})
        board = {
            "top": list(before.get("top", [])),
            "middle": list(before.get("middle", [])),
            "bottom": list(before.get("bottom", [])),
        }
        action = turn_log.get("action", {})
        for card, pos in action.get("placements", []):
            if pos == "top":
                board["top"].append(card)
            elif pos in {"middle", "mid"}:
                board["middle"].append(card)
            else:
                board["bottom"].append(card)
        boards[player] = board
    return boards


def summarize_block(block: list[dict], stats: dict) -> None:
    if not block:
        return

    hand_result = block[0].get("hand_result", {})
    boards = reconstruct_final_boards(block)
    fl_by_seat = []
    rewards = {}

    for record in block:
        turn_log = record.get("turn_log", {})
        player = str(turn_log.get("player", 0))
        rewards.setdefault(player, float(record.get("reward", 0.0)))

    stats["hands"] += 1
    stats["seat_games"] += 2
    stats["reward0_sum"] += rewards.get("0", 0.0)
    stats["reward1_sum"] += rewards.get("1", 0.0)

    for seat in ("0", "1"):
        fl = bool(hand_result.get("fl_entry", {}).get(seat, False))
        busted = bool(hand_result.get("busted", {}).get(seat, False))
        fl_cards = int(hand_result.get("fl_cards", {}).get(seat, 0))
        fl_by_seat.append(fl)

        stats["fl_entries"] += int(fl)
        stats["busts"] += int(busted)
        if fl:
            stats["fl_types"][FL_CARD_NAMES.get(fl_cards, "other")] += 1
        if busted and top_has_fl_shape(boards.get(seat, {}).get("top", [])):
            stats["fl_shape_but_foul"] += 1

    stats["hand_any_fl"] += int(any(fl_by_seat))
    stats["hand_both_fl"] += int(all(fl_by_seat))


def summarize_file(path: Path, records_per_hand: int) -> dict:
    stats = empty_stats()
    block: list[dict] = []
    bad_json = 0
    lines = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            lines += 1
            try:
                block.append(json.loads(line))
            except json.JSONDecodeError:
                bad_json += 1
                continue
            if len(block) >= records_per_hand:
                summarize_block(block, stats)
                block = []

    if block:
        summarize_block(block, stats)

    stats["input"] = str(path)
    stats["lines"] = lines
    stats["bad_json"] = bad_json
    return stats


def pct(num: float, den: float) -> float:
    return 0.0 if den <= 0 else num / den


def serializable(stats: dict) -> dict:
    return {
        **{k: v for k, v in stats.items() if k != "fl_types"},
        "fl_types": dict(stats["fl_types"]),
        "seat_fl_rate": pct(stats["fl_entries"], stats["seat_games"]),
        "seat_bust_rate": pct(stats["busts"], stats["seat_games"]),
        "hand_any_fl_rate": pct(stats["hand_any_fl"], stats["hands"]),
        "hand_both_fl_rate": pct(stats["hand_both_fl"], stats["hands"]),
        "fl_shape_but_foul_rate": pct(stats["fl_shape_but_foul"], stats["seat_games"]),
        "avg_reward0": pct(stats["reward0_sum"], stats["hands"]),
        "avg_reward1": pct(stats["reward1_sum"], stats["hands"]),
    }


def write_markdown(path: Path, combined: dict, per_file: list[dict]) -> None:
    def fmt_rate(value: float) -> str:
        return f"{value:.1%}"

    rows = [serializable(s) for s in per_file]
    combined_s = serializable(combined)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Self-Play FL Summary\n\n")
        f.write("## Combined\n\n")
        f.write(f"- hands: {combined_s['hands']:,}\n")
        f.write(f"- seat games: {combined_s['seat_games']:,}\n")
        f.write(f"- seat FL entry rate: {fmt_rate(combined_s['seat_fl_rate'])}\n")
        f.write(f"- hand any-FL rate: {fmt_rate(combined_s['hand_any_fl_rate'])}\n")
        f.write(f"- hand both-FL rate: {fmt_rate(combined_s['hand_both_fl_rate'])}\n")
        f.write(f"- seat foul rate: {fmt_rate(combined_s['seat_bust_rate'])}\n")
        f.write(f"- FL-shape-but-foul rate: {fmt_rate(combined_s['fl_shape_but_foul_rate'])}\n")
        f.write(f"- avg reward seat0: {combined_s['avg_reward0']:+.3f}\n")
        f.write(f"- FL types: {combined_s['fl_types']}\n\n")

        f.write("## By File\n\n")
        f.write("| input | hands | seat FL | any-FL | foul | shape-foul | QQ | KK | AA | Trips | avg r0 |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            fl_types = defaultdict(int, row["fl_types"])
            f.write(
                f"| `{Path(row['input']).name}` | {row['hands']:,} | "
                f"{fmt_rate(row['seat_fl_rate'])} | {fmt_rate(row['hand_any_fl_rate'])} | "
                f"{fmt_rate(row['seat_bust_rate'])} | {fmt_rate(row['fl_shape_but_foul_rate'])} | "
                f"{fl_types['qq']} | {fl_types['kk']} | {fl_types['aa']} | {fl_types['trips']} | "
                f"{row['avg_reward0']:+.3f} |\n"
            )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize self-play JSONL FL metrics")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--records-per-hand", type=int, default=10)
    args = parser.parse_args(argv)

    per_file = [summarize_file(Path(path), args.records_per_hand) for path in args.inputs]
    combined = empty_stats()
    for stats in per_file:
        merge_stats(combined, stats)

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "combined": serializable(combined),
                "per_file": [serializable(stats) for stats in per_file],
                "records_per_hand": args.records_per_hand,
            },
            f,
            indent=2,
        )
    write_markdown(output_md, combined, per_file)
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
