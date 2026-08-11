"""Audit runtime-available risk guards for Stage8c TopK fired decisions.

This is analysis-only. It does not train a model and it does not approve a T2
runtime. It applies extra runtime-available filters to already-fired TopK
decisions, then reports realized whole-game deltas for the remaining fires.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-log", type=Path, action="append", default=[])
    parser.add_argument("--input-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-kept", type=int, default=50)
    parser.add_argument("--top-n", type=int, default=50)
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                row["_source_log"] = str(path)
                rows.append(row)
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil((q / 100.0) * len(ordered)) - 1))
    return ordered[index]


def mean_and_se(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance / len(values))


@dataclass(frozen=True)
class RiskGuard:
    min_predicted_delta: float | None = None
    min_confirm_delta: float | None = None
    min_confirm_z: float | None = None
    max_confirm_se: float | None = None
    max_candidate_rank: int | None = None
    min_gate_probability: float | None = None

    @property
    def guard_id(self) -> str:
        parts = []
        if self.min_predicted_delta is not None:
            parts.append(f"pd{self.min_predicted_delta:g}")
        if self.min_confirm_delta is not None:
            parts.append(f"cd{self.min_confirm_delta:g}")
        if self.min_confirm_z is not None:
            parts.append(f"cz{self.min_confirm_z:g}")
        if self.max_confirm_se is not None:
            parts.append(f"semax{self.max_confirm_se:g}")
        if self.max_candidate_rank is not None:
            parts.append(f"rank{self.max_candidate_rank:g}")
        if self.min_gate_probability is not None:
            parts.append(f"g{self.min_gate_probability:g}")
        return "_".join(parts) if parts else "no_extra_guard"


def confirm_z(row: dict[str, Any]) -> float:
    delta = confirm_delta(row)
    se = confirm_delta_se(row)
    return delta / se if se > 0.0 else 0.0


def confirm_delta(row: dict[str, Any]) -> float:
    return safe_float(row.get("confirm_delta", row.get("rerank_delta")))


def confirm_delta_se(row: dict[str, Any]) -> float:
    return safe_float(row.get("confirm_delta_se", row.get("rerank_delta_se")))


def guard_passes(row: dict[str, Any], guard: RiskGuard) -> bool:
    if guard.min_predicted_delta is not None and safe_float(row.get("predicted_delta")) < guard.min_predicted_delta:
        return False
    if guard.min_confirm_delta is not None and confirm_delta(row) < guard.min_confirm_delta:
        return False
    if guard.min_confirm_z is not None and confirm_z(row) < guard.min_confirm_z:
        return False
    if guard.max_confirm_se is not None and confirm_delta_se(row) > guard.max_confirm_se:
        return False
    if guard.max_candidate_rank is not None and safe_int(row.get("candidate_ev_rank"), 9999) > guard.max_candidate_rank:
        return False
    if guard.min_gate_probability is not None and safe_float(row.get("gate_probability")) < guard.min_gate_probability:
        return False
    return True


def fired_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if truthy(row.get("override_fired"))
        and truthy(row.get("realized_delta_valid"))
        and row.get("realized_candidate_seat_delta") not in (None, "")
    ]


def metric_row(config_id: str, all_rows: list[dict[str, Any]], fired: list[dict[str, Any]], guard: RiskGuard) -> dict[str, Any]:
    kept = [row for row in fired if guard_passes(row, guard)]
    blocked = len(fired) - len(kept)
    deltas = [safe_float(row.get("realized_candidate_seat_delta")) for row in kept]
    losses = [max(0.0, -value) for value in deltas]
    mean, se = mean_and_se(deltas)
    decision_count = len(all_rows)
    ev_per_hand = sum(deltas) / decision_count if decision_count else 0.0
    kept_loss_count = sum(1 for value in deltas if value < 0.0)
    base_deltas = [safe_float(row.get("realized_candidate_seat_delta")) for row in fired]
    base_sum = sum(base_deltas)
    blocked_losses = [
        row for row in fired if not guard_passes(row, guard) and safe_float(row.get("realized_candidate_seat_delta")) < 0.0
    ]
    blocked_gains = [
        row for row in fired if not guard_passes(row, guard) and safe_float(row.get("realized_candidate_seat_delta")) > 0.0
    ]
    return {
        "config_id": config_id,
        "guard_id": guard.guard_id,
        "decision_count": decision_count,
        "input_fires": len(fired),
        "kept_fires": len(kept),
        "blocked_fires": blocked,
        "keep_rate": len(kept) / len(fired) if fired else 0.0,
        "ev_per_hand_after_guard": ev_per_hand,
        "ev_per_hand_delta_vs_unfiltered": ev_per_hand - (base_sum / decision_count if decision_count else 0.0),
        "per_fire_delta_mean": mean,
        "per_fire_delta_ci95_low": mean - 1.96 * se,
        "per_fire_delta_ci95_high": mean + 1.96 * se,
        "realized_loss_count": kept_loss_count,
        "loss_rate": kept_loss_count / len(kept) if kept else 0.0,
        "p95_loss": percentile(losses, 95),
        "p99_loss": percentile(losses, 99),
        "max_loss": max(losses, default=0.0),
        "blocked_loss_count": len(blocked_losses),
        "blocked_gain_count": len(blocked_gains),
        "blocked_loss_delta_sum": sum(safe_float(row.get("realized_candidate_seat_delta")) for row in blocked_losses),
        "blocked_gain_delta_sum": sum(safe_float(row.get("realized_candidate_seat_delta")) for row in blocked_gains),
        "min_predicted_delta": guard.min_predicted_delta,
        "min_confirm_delta": guard.min_confirm_delta,
        "min_confirm_z": guard.min_confirm_z,
        "max_confirm_se": guard.max_confirm_se,
        "max_candidate_rank": guard.max_candidate_rank,
        "min_gate_probability": guard.min_gate_probability,
    }


def guard_grid() -> list[RiskGuard]:
    guards = [RiskGuard()]
    for min_pd in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0):
        for min_cd in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
            for min_cz in (1.0, 1.25, 1.5, 1.75, 2.0):
                for max_se in (None, 0.75, 1.0, 1.25, 1.5, 2.0):
                    for max_rank in (None, 1, 2, 3, 5, 10):
                        for min_gate in (None, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01):
                            guards.append(
                                RiskGuard(
                                    min_predicted_delta=min_pd,
                                    min_confirm_delta=min_cd,
                                    min_confirm_z=min_cz,
                                    max_confirm_se=max_se,
                                    max_candidate_rank=max_rank,
                                    min_gate_probability=min_gate,
                                )
                            )
    return guards


def discover_logs(args: argparse.Namespace) -> list[Path]:
    logs = list(args.decision_log or [])
    for directory in args.input_dir or []:
        candidate = directory / "runtime_decisions.jsonl"
        if candidate.exists():
            logs.append(candidate)
    unique: list[Path] = []
    seen = set()
    for path in logs:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    if not unique:
        raise SystemExit("at least one --decision-log or --input-dir with runtime_decisions.jsonl is required")
    return unique


def analyze(rows: list[dict[str, Any]], *, min_kept: int, top_n: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_config[str(row.get("config_id", ""))].append(row)
    grid_rows: list[dict[str, Any]] = []
    for config_id, config_rows in by_config.items():
        fired = fired_rows(config_rows)
        for guard in guard_grid():
            grid_rows.append(metric_row(config_id, config_rows, fired, guard))
    eligible = [row for row in grid_rows if safe_int(row.get("kept_fires")) >= min_kept]
    eligible.sort(
        key=lambda row: (
            safe_float(row.get("ev_per_hand_after_guard")),
            -safe_float(row.get("max_loss")),
            safe_int(row.get("kept_fires")),
        ),
        reverse=True,
    )
    return grid_rows, eligible[:top_n]


def write_summary(path: Path, *, logs: list[Path], grid_rows: list[dict[str, Any]], top_rows: list[dict[str, Any]], min_kept: int) -> None:
    base_rows = [row for row in grid_rows if row.get("guard_id") == "no_extra_guard"]
    lines = [
        "# HU T2 Stage8c Runtime Risk Filter Audit",
        "",
        "This is post-hoc analysis only. It does not approve production, P2 fixed status, T1, or 50k teacher generation.",
        "",
        "## Inputs",
        "",
    ]
    lines.extend(f"- `{path}`" for path in logs)
    lines.extend(
        [
            "",
            "## Unfiltered Baseline",
            "",
            "| config | fires | EV/hand | per-fire | losses | max loss |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in base_rows:
        lines.append(
            "| {config_id} | {kept_fires} | {ev:.4f} | {per_fire:.4f} | {losses} | {max_loss:.4f} |".format(
                config_id=row["config_id"],
                kept_fires=safe_int(row.get("kept_fires")),
                ev=safe_float(row.get("ev_per_hand_after_guard")),
                per_fire=safe_float(row.get("per_fire_delta_mean")),
                losses=safe_int(row.get("realized_loss_count")),
                max_loss=safe_float(row.get("max_loss")),
            )
        )
    lines.extend(
        [
            "",
            f"## Top Guards With kept_fires >= {min_kept}",
            "",
            "| guard | kept | EV/hand | per-fire | CI low | losses | max loss | blocked losses | blocked gains |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in top_rows[:10]:
        lines.append(
            "| {guard} | {kept} | {ev:.4f} | {per_fire:.4f} | {ci_low:.4f} | {losses} | {max_loss:.4f} | {blocked_losses} | {blocked_gains} |".format(
                guard=row["guard_id"],
                kept=safe_int(row.get("kept_fires")),
                ev=safe_float(row.get("ev_per_hand_after_guard")),
                per_fire=safe_float(row.get("per_fire_delta_mean")),
                ci_low=safe_float(row.get("per_fire_delta_ci95_low")),
                losses=safe_int(row.get("realized_loss_count")),
                max_loss=safe_float(row.get("max_loss")),
                blocked_losses=safe_int(row.get("blocked_loss_count")),
                blocked_gains=safe_int(row.get("blocked_gain_count")),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- Use realized deltas only; confirm deltas are gate inputs, not performance claims.",
            "- Treat any selected guard as a hypothesis requiring non-overlapping validation.",
            "- Stronger EV here can be overfit to these fired rows.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    logs = discover_logs(args)
    rows: list[dict[str, Any]] = []
    for path in logs:
        rows.extend(read_jsonl(path))
    grid_rows, top_rows = analyze(rows, min_kept=args.min_kept, top_n=args.top_n)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "risk_filter_grid.csv", grid_rows)
    write_csv(args.output_dir / "risk_filter_top.csv", top_rows)
    write_summary(
        args.output_dir / "risk_filter_summary.md",
        logs=logs,
        grid_rows=grid_rows,
        top_rows=top_rows,
        min_kept=args.min_kept,
    )
    manifest = {
        "decision_logs": [str(path) for path in logs],
        "rows": len(rows),
        "grid_rows": len(grid_rows),
        "top_rows": len(top_rows),
        "min_kept": args.min_kept,
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "risk_filter_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
