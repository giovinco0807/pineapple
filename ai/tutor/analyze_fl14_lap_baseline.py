"""Summarise one played FL14 chain before the next lap is trained.

The terminal file is produced by ``fl_solver finish-t4``.  The optional T4
position file is the eleven-card board selected at T3.  Joining the two makes
it possible to attribute realised forced fouls to the board shape that exposed
the final draw, without pretending that a realised draw is the position's
full foul probability.

This is deliberately a baseline tool, not a promotion gate.  A later lap must
still be compared with its predecessor on fresh paired deals that neither lap
trained on.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


FL_EV = {14: 0.0, 15: 10.7, 16: 29.9, 17: 63.5}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            ident = str(record["id"])
            if ident in seen:
                raise ValueError(f"{path}:{number}: duplicate id {ident}")
            seen.add(ident)
            record["id"] = ident
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def _mean_ci95(values: Iterable[float]) -> dict[str, float]:
    data = [float(value) for value in values]
    if not data:
        return {"mean": math.nan, "stderr": math.nan, "ci95_low": math.nan, "ci95_high": math.nan}
    mean = sum(data) / len(data)
    if len(data) == 1:
        stderr = math.nan
        half = math.nan
    else:
        variance = sum((value - mean) ** 2 for value in data) / (len(data) - 1)
        stderr = math.sqrt(variance / len(data))
        half = 1.959963984540054 * stderr
    return {
        "mean": mean,
        "stderr": stderr,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def _cards(record: dict[str, Any]) -> list[str]:
    cards: list[str] = []
    for row in record.get("rows", []):
        cards.extend(row)
    for field in ("dead", "draw"):
        value = record.get(field, [])
        if isinstance(value, str):
            cards.extend(card for card in value.split(",") if card)
        else:
            cards.extend(value)
    return cards


def _shape(position: dict[str, Any]) -> str:
    rows = position.get("rows")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError(f"position {position.get('id')} has no three-row board")
    return "-".join(str(len(row)) for row in rows)


def _shape_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    fouls = sum(bool(row["finished"]["busted"]) for row in rows)
    forced = sum(
        bool(row["finished"]["busted"]) and int(row["finished"]["unfouled"]) == 0
        for row in rows
    )
    legal_counts = [int(row["finished"]["unfouled"]) for row in rows]
    return {
        "n": n,
        "share": n,
        "fouls": fouls,
        "foul_rate": fouls / n,
        "forced_fouls": forced,
        "foul_contribution": forced,
        "mean_legal_t4_actions_on_realised_draw": sum(legal_counts) / n,
        "zero_legal_rate": sum(value == 0 for value in legal_counts) / n,
        "one_legal_rate": sum(value == 1 for value in legal_counts) / n,
    }


def summarise(
    finished: list[dict[str, Any]],
    positions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a machine-readable baseline with honest realised-draw semantics."""

    finished_by_id = {record["id"]: record for record in finished}
    n = len(finished)
    scores = [float(record["value"]) for record in finished]
    royalties = [0.0 if record["busted"] else float(record["royalty"]) for record in finished]
    fouls = [1.0 if record["busted"] else 0.0 for record in finished]
    entries = [1.0 if int(record.get("entry_width", 0)) else 0.0 for record in finished]
    entry_ev = [FL_EV.get(int(record.get("entry_width", 0)), 0.0) for record in finished]
    forced = sum(bool(record["busted"]) and int(record["unfouled"]) == 0 for record in finished)
    chosen = sum(bool(record["busted"]) and int(record["unfouled"]) > 0 for record in finished)
    legal = Counter(int(record["unfouled"]) for record in finished)
    widths = Counter(int(record.get("entry_width", 0)) for record in finished)

    summary: dict[str, Any] = {
        "schema": "ofc_fl14_lap_baseline/v1",
        "semantics": {
            "foul": "realised final T4 draw",
            "forced_foul": "the realised T4 draw offered zero non-fouling placements",
            "warning": "this does not estimate the full T3 position foul probability",
        },
        "hands": n,
        "score": _mean_ci95(scores),
        "royalty_foul_zero": _mean_ci95(royalties),
        "foul_rate": _mean_ci95(fouls),
        "fl_entry_rate": _mean_ci95(entries),
        "fl_entry_ev": _mean_ci95(entry_ev),
        "forced_fouls": forced,
        "chosen_fouls": chosen,
        "legal_t4_actions_on_realised_draw": {str(key): legal[key] for key in sorted(legal)},
        "entry_widths": {str(key): widths[key] for key in sorted(widths)},
    }

    if positions is None:
        return summary

    position_by_id = {record["id"]: record for record in positions}
    missing_positions = sorted(set(finished_by_id) - set(position_by_id))
    extra_positions = sorted(set(position_by_id) - set(finished_by_id))
    if missing_positions or extra_positions:
        raise ValueError(
            "finished/position ids differ: "
            f"missing_positions={missing_positions[:5]} extra_positions={extra_positions[:5]}"
        )

    joined: list[dict[str, Any]] = []
    for ident, final in finished_by_id.items():
        position = position_by_id[ident]
        joined.append({"finished": final, "position": position})

    by_shape: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_jokers: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        by_shape[_shape(row["position"])].append(row)
        joker_count = sum(card.startswith("X") or card == "JK" for card in _cards(row["position"]))
        by_jokers[joker_count].append(row)

    shape_summary = {shape: _shape_rows(rows) for shape, rows in sorted(by_shape.items())}
    for values in shape_summary.values():
        values["share"] /= n
        values["foul_contribution"] /= max(forced, 1)
    summary["t3_board_shapes"] = shape_summary
    joker_summary = {
        str(count): _shape_rows(rows) for count, rows in sorted(by_jokers.items())
    }
    for values in joker_summary.values():
        values["share"] /= n
        values["foul_contribution"] /= max(forced, 1)
    summary["seen_jokers"] = joker_summary
    return summary


def _print_human(summary: dict[str, Any]) -> None:
    def metric(name: str, scale: float = 1.0, suffix: str = "") -> None:
        item = summary[name]
        print(
            f"{name:22} {item['mean'] * scale:+.4f}{suffix} "
            f"[{item['ci95_low'] * scale:+.4f}, {item['ci95_high'] * scale:+.4f}]"
        )

    print(f"hands                  {summary['hands']}")
    metric("score")
    metric("royalty_foul_zero")
    metric("foul_rate", 100.0, "%")
    metric("fl_entry_rate", 100.0, "%")
    metric("fl_entry_ev")
    print(f"forced/chosen fouls    {summary['forced_fouls']} / {summary['chosen_fouls']}")
    if "t3_board_shapes" in summary:
        print("\nT3 board shape   n    share   foul    forced-foul contribution   mean legal T4 actions")
        ordered = sorted(
            summary["t3_board_shapes"].items(),
            key=lambda item: (-item[1]["forced_fouls"], item[0]),
        )
        for shape, row in ordered:
            print(
                f"{shape:14} {row['n']:5d}  {row['share'] * 100:6.2f}%  "
                f"{row['foul_rate'] * 100:6.2f}%  {row['foul_contribution'] * 100:9.2f}%"
                f"                 {row['mean_legal_t4_actions_on_realised_draw']:.2f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finished", type=Path, required=True)
    parser.add_argument("--positions", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    summary = summarise(
        _load_jsonl(args.finished),
        _load_jsonl(args.positions) if args.positions else None,
    )
    _print_human(summary)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
