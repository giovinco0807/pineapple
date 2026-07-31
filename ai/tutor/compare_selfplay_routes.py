"""Compare two aligned self-play JSONL route outputs.

Use this on self-play files generated with the same seed and `--workers 1`.
The script compares records by position, so multi-worker outputs are only useful
for aggregate summaries, not exact route diffs.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def action_key(record: dict[str, Any]) -> tuple[tuple[tuple[str, str], ...], str | None]:
    action = record["turn_log"].get("action", {})
    placements = tuple(sorted((str(card), str(pos)) for card, pos in action.get("placements", [])))
    discard = action.get("discard")
    return placements, None if discard is None else str(discard)


def board_key(board: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(board.get("top", [])),
        tuple(board.get("middle", [])),
        tuple(board.get("bottom", [])),
    )


def same_state(left: dict[str, Any], right: dict[str, Any]) -> bool:
    a = left["turn_log"]
    b = right["turn_log"]
    return (
        a.get("dealt_cards") == b.get("dealt_cards")
        and board_key(a.get("board_self", {})) == board_key(b.get("board_self", {}))
        and board_key(a.get("board_opponent", {})) == board_key(b.get("board_opponent", {}))
    )


def compare_records(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    records_per_hand: int,
    max_examples: int,
) -> dict[str, Any]:
    if len(baseline) != len(candidate):
        raise ValueError(f"Record count mismatch: {len(baseline)} vs {len(candidate)}")

    turn_total: Counter[int] = Counter()
    turn_action_diff: Counter[int] = Counter()
    turn_action_idx_diff: Counter[int] = Counter()
    hand_diff: Counter[int] = Counter()
    mismatched_states = 0
    examples: list[dict[str, Any]] = []

    for index, (left, right) in enumerate(zip(baseline, candidate)):
        a = left["turn_log"]
        b = right["turn_log"]
        turn = int(a.get("turn", -1))
        turn_total[turn] += 1
        hand = index // records_per_hand
        state_matches = same_state(left, right)
        if not state_matches:
            mismatched_states += 1

        if action_key(left) != action_key(right):
            turn_action_diff[turn] += 1
            hand_diff[hand] += 1
            if len(examples) < max_examples:
                examples.append(
                    {
                        "hand": hand + 1,
                        "record": index + 1,
                        "player": a.get("player"),
                        "turn": turn,
                        "same_state": state_matches,
                        "dealt": a.get("dealt_cards"),
                        "board_self": a.get("board_self"),
                        "baseline_action": a.get("action"),
                        "candidate_action": b.get("action"),
                        "baseline_action_idx": a.get("action_idx"),
                        "candidate_action_idx": b.get("action_idx"),
                    }
                )

        if a.get("action_idx") != b.get("action_idx"):
            turn_action_idx_diff[turn] += 1

    return {
        "records": len(baseline),
        "hands": len(baseline) // records_per_hand,
        "records_per_hand": records_per_hand,
        "mismatched_states": mismatched_states,
        "action_differences": int(sum(turn_action_diff.values())),
        "hands_with_action_difference": int(len(hand_diff)),
        "turn_total": {str(k): int(v) for k, v in sorted(turn_total.items())},
        "turn_action_differences": {str(k): int(v) for k, v in sorted(turn_action_diff.items())},
        "turn_action_idx_differences": {str(k): int(v) for k, v in sorted(turn_action_idx_diff.items())},
        "examples": examples,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    baseline_name = Path(report["baseline"]).name
    candidate_name = Path(report["candidate"]).name
    lines = [
        "# Aligned Self-Play Route Compare",
        "",
        f"- baseline: `{baseline_name}`",
        f"- candidate: `{candidate_name}`",
        f"- records: {report['records']:,}",
        f"- hands: {report['hands']:,}",
        f"- mismatched states: {report['mismatched_states']:,}",
        f"- action differences: {report['action_differences']:,}",
        f"- hands with action difference: {report['hands_with_action_difference']:,}",
        "",
        "| turn | records | action diffs | action_idx diffs |",
        "|---:|---:|---:|---:|",
    ]
    totals = report["turn_total"]
    action_diffs = report["turn_action_differences"]
    action_idx_diffs = report["turn_action_idx_differences"]
    for turn in sorted(totals, key=int):
        lines.append(
            f"| T{turn} | {totals[turn]} | {action_diffs.get(turn, 0)} | {action_idx_diffs.get(turn, 0)} |"
        )

    lines.extend(["", "## First Differences", ""])
    if not report["examples"]:
        lines.append("No action differences found.")
    else:
        for item in report["examples"]:
            lines.append(
                f"- hand {item['hand']} record {item['record']} P{item['player']} "
                f"T{item['turn']} same_state={item['same_state']}: "
                f"baseline_idx={item['baseline_action_idx']} candidate_idx={item['candidate_action_idx']}"
            )
            lines.append(f"  - dealt: {item['dealt']}")
            lines.append(f"  - baseline: {item['baseline_action']}")
            lines.append(f"  - candidate: {item['candidate_action']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare aligned self-play route JSONL files")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--records-per-hand", type=int, default=10)
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = compare_records(
        load_records(Path(args.baseline)),
        load_records(Path(args.candidate)),
        records_per_hand=args.records_per_hand,
        max_examples=args.max_examples,
    )
    report["baseline"] = args.baseline
    report["candidate"] = args.candidate

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(output_md, report)
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
