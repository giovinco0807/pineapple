"""Aggregate GCP shards for HU Turn2 Stage8b TopK + MC rerank validation.

Each GCP shard runs ``ofc_regular.evaluate_hu_turn2_stage8b_topk_mc_rerank``
for a single (config, seed) pair and uploads its output directory. This
aggregator concatenates the per-shard ``seed_breakdown.csv`` rows and
produces the cross-seed grid plus a go/no-go style summary.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .evaluate_hu_turn2_stage8b_topk_mc_rerank import aggregate_topk_seed_rows, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Downloaded GCP run directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def collect_seed_rows(input_dir: Path) -> list[dict[str, Any]]:
    roots = [input_dir / "results", input_dir]
    seen: set[Path] = set()
    rows: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("seed_breakdown.csv")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            rows.extend(read_csv_rows(path))
    return rows


def summary_markdown(grid: list[dict[str, Any]], seed_row_count: int) -> str:
    lines = [
        "# HU T2 Stage8b TopK + MC Rerank — GCP Aggregate",
        "",
        f"- seed rows aggregated: {seed_row_count}",
        "",
        "| config | paired | EV/hand | CI low | CI high | overrides | override rate | avg confirm gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in grid:
        lines.append(
            "| `{config}` | {paired} | {ev:+.4f} | {lo:+.4f} | {hi:+.4f} | {n} | {rate:.2%} | {gain:+.4f} |".format(
                config=row["config_id"],
                paired=row["paired_seeds"],
                ev=float(row["aggregate_ev_per_hand"]),
                lo=float(row["ci95_low_seed_means"]),
                hi=float(row["ci95_high_seed_means"]),
                n=row["override_count"],
                rate=float(row["runtime_override_rate"]),
                gain=float(row["avg_confirm_gain_on_override"]),
            )
        )
    lines += [
        "",
        "Interpretation notes:",
        "- `avg confirm gain` comes from the independent confirmation MC stream",
        "  (unbiased); do not quote the selection-stage rerank delta as gain.",
        "- Go requires CI low > 0 on the aggregate paired EV, not just a",
        "  positive point estimate.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    seed_rows = collect_seed_rows(args.input_dir)
    if not seed_rows:
        raise SystemExit(f"no seed_breakdown.csv rows found under {args.input_dir}")
    grid = aggregate_topk_seed_rows(seed_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "topk_rerank_aggregate.csv", grid)
    write_csv(args.output_dir / "seed_breakdown_all.csv", seed_rows)
    (args.output_dir / "topk_rerank_aggregate.md").write_text(
        summary_markdown(grid, len(seed_rows)),
        encoding="utf-8",
    )
    print(json.dumps({"configs": len(grid), "seed_rows": len(seed_rows), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
