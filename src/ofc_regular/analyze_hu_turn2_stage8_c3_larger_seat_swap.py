"""Gate C3 larger seat-swap decision for HU Turn2 Stage8.

C3 consumes the larger seat-swap output from
``evaluate_hu_turn2_stage8_seat_swap`` and decides whether the project is ready
for selected high-MC refinement / C4 validation. It never launches 50k teacher
generation, T1 training, or production training.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path("outputs/evals/hu_turn2_stage8_c3_larger_seat_swap")
GRID_FILE = "hu_turn2_stage8_20k_threshold_grid_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def c3_row_status(row: dict[str, Any]) -> tuple[str, str]:
    blockers: list[str] = []
    if safe_int(row.get("paired_seeds")) < 2000:
        blockers.append("paired_seeds_lt_2000")
    if safe_float(row.get("aggregate_ev_per_hand")) <= 0.0:
        blockers.append("seat_swap_ev_not_positive")
    if safe_float(row.get("ci95_low_seed_means")) < -0.05:
        blockers.append("ci95_low_lt_minus_0p05")
    if safe_float(row.get("avg_gain_on_override")) <= 0.0:
        blockers.append("teacher_avg_gain_not_positive")
    if safe_float(row.get("false_positive_override_rate"), 1.0) > 0.10:
        blockers.append("false_positive_rate_gt_10pct")
    if safe_float(row.get("p95_loss")) > 5.0:
        blockers.append("p95_loss_gt_5")
    if safe_int(row.get("override_count")) == 0:
        blockers.append("runtime_override_zero")
    return ("conditional_go" if not blockers else "no_go", ";".join(blockers))


def ranked_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        status, blockers = c3_row_status(row)
        item: dict[str, Any] = dict(row)
        item["c3_status"] = status
        item["c3_blockers"] = blockers
        output.append(item)
    output.sort(
        key=lambda row: (
            {"conditional_go": 0, "no_go": 1}.get(str(row["c3_status"]), 2),
            -safe_float(row.get("ci95_low_seed_means"), -999.0),
            -safe_float(row.get("aggregate_ev_per_hand"), -999.0),
            safe_float(row.get("false_positive_override_rate"), 999.0),
            -safe_float(row.get("avg_gain_on_override"), -999.0),
        )
    )
    return output


def c3_decision(rows: list[dict[str, Any]]) -> tuple[str, list[str], dict[str, Any] | None]:
    if not rows:
        return "No-Go", ["threshold_grid_missing_or_empty"], None
    viable = [row for row in rows if row["c3_status"] == "conditional_go"]
    if viable:
        return "Conditional-Go", [], viable[0]
    blockers = sorted({part for row in rows for part in str(row.get("c3_blockers", "")).split(";") if part})
    return "No-Go", blockers or ["no_viable_c3_candidate"], rows[0]


def write_summary(output_dir: Path, rows: list[dict[str, Any]], decision: str, blockers: list[str], recommended: dict[str, Any] | None) -> None:
    lines = [
        "# HU Turn2 Stage8 C3 Larger Seat-Swap",
        "",
        "C3 is validation-only. It does not start 50k teacher, T1 training, production training, or production runtime changes.",
        "",
        f"- selected refinement / C4 preparation: `{decision}`",
        f"- blockers: `{';'.join(blockers) if blockers else 'none'}`",
        "- 50k teacher: `No-Go`",
        "- 50k teacher launched: `False`",
        "- T1 training: `No-Go`",
        "- production training: `No-Go`",
        "",
        "## Ranked Candidates",
        "",
        "| config | status | blockers | paired | EV/hand | CI low | CI high | override rate | FP rate | avg gain | p95 loss |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {config} | {status} | {blockers} | {paired} | {ev:.5f} | {low:.5f} | {high:.5f} | {orate:.5f} | {fp:.5f} | {gain:.5f} | {p95:.5f} |".format(
                config=row.get("config_id", ""),
                status=row.get("c3_status", ""),
                blockers=row.get("c3_blockers", ""),
                paired=safe_int(row.get("paired_seeds")),
                ev=safe_float(row.get("aggregate_ev_per_hand")),
                low=safe_float(row.get("ci95_low_seed_means")),
                high=safe_float(row.get("ci95_high_seed_means")),
                orate=safe_float(row.get("runtime_override_rate")),
                fp=safe_float(row.get("false_positive_override_rate")),
                gain=safe_float(row.get("avg_gain_on_override")),
                p95=safe_float(row.get("p95_loss")),
            )
        )
    if recommended:
        lines.extend(
            [
                "",
                "## Recommended C3 Follow-Up Candidate",
                "",
                f"- config: `{recommended.get('config_id', '')}`",
                f"- hu_turn2_min_margin: `{recommended.get('hu_turn2_min_margin', '')}`",
                f"- hu_turn2_reference_min_margin: `{recommended.get('hu_turn2_reference_min_margin', '')}`",
                f"- hu_turn2_gate_threshold: `{recommended.get('hu_turn2_gate_threshold', '')}`",
            ]
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "c3_larger_seat_swap_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "go_nogo_for_c3_followup.md").write_text(
        "\n".join(
            [
                "# Go / No-Go For C3 Follow-Up",
                "",
                f"- selected refinement / C4 preparation: `{decision}`",
                f"- blockers: `{';'.join(blockers) if blockers else 'none'}`",
                "- 50k teacher: `No-Go`",
                "- 50k teacher launched: `False`",
                "- T1 training: `No-Go`",
                "- production training: `No-Go`",
                "",
                "This authorizes only preparing selected high-MC refinement / C4 validation. It does not start 50k teacher or production work.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.input_dir
    rows = ranked_rows(read_csv(args.input_dir / GRID_FILE))
    decision, blockers, recommended = c3_decision(rows)
    write_csv(output_dir / "c3_candidate_ranked.csv", rows)
    write_summary(output_dir, rows, decision, blockers, recommended)
    print(
        json.dumps(
            {
                "gate": "C3",
                "candidate_count": len(rows),
                "c3_followup_preparation": decision,
                "blockers": blockers,
                "recommended_config": recommended.get("config_id") if recommended else None,
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
