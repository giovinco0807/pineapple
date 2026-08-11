"""Compare Stage8c TopK candidate generators by realized per-fire metrics.

This analyzer is evaluation-only. It compares already-produced
``evaluate_hu_turn2_stage8b_topk_mc_rerank`` output directories and keeps the
primary metric on realized fired-hand deltas. Confirm MC deltas remain gate
diagnostics only.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any

from ofc_regular.analyze_hu_turn2_stage8c_topk_per_fire import aggregate, read_jsonl, safe_float, safe_int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        help="Candidate group in name=eval_output_dir form. Repeat to add seeds or groups.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evals/hu_turn2_current_fl_ev_stage8c_candidate_generator_comparison"),
    )
    return parser.parse_args()


def parse_groups(group_specs: list[str]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for spec in group_specs:
        if "=" not in spec:
            raise SystemExit(f"--group must be name=path: {spec}")
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        path = Path(raw_path.strip())
        if not name:
            raise SystemExit(f"--group name is empty: {spec}")
        if not path.exists() or not path.is_dir():
            raise SystemExit(f"--group path is not a directory: {path}")
        groups.setdefault(name, []).append(path)
    if len(groups) < 2:
        raise SystemExit("at least two candidate groups are required")
    return groups


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def action_signature(action: Any) -> str:
    return json.dumps(action, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def fired_state_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("config_id"),
        row.get("hand_id"),
        row.get("seat_swap"),
        row.get("street"),
        row.get("seat"),
    )


def fired_sets_by_config(paths: list[Path]) -> dict[str, dict[tuple[Any, ...], str]]:
    by_config: dict[str, dict[tuple[Any, ...], str]] = {}
    for directory in paths:
        for row in read_jsonl(directory / "runtime_decisions.jsonl"):
            if not row.get("override_fired"):
                continue
            config_id = str(row.get("config_id", ""))
            key = fired_state_key(row)
            by_config.setdefault(config_id, {})[key] = action_signature(row.get("final_action"))
    return by_config


def cancellation_status(cancel: dict[str, Any], fires: int) -> dict[str, Any]:
    present = bool(cancel) and safe_int(cancel.get("input_rows")) > 0
    nonzero = safe_int(cancel.get("non_fired_nonzero_count")) if present else 0
    delta_sum = safe_float(cancel.get("non_fired_delta_sum")) if present else 0.0
    max_abs = safe_float(cancel.get("non_fired_delta_max_abs")) if present else 0.0
    clean = present and nonzero == 0 and abs(delta_sum) <= 1e-9 and max_abs <= 1e-9
    if not present:
        reason = "missing_cancellation_audit"
    elif not clean:
        reason = "dirty_non_fired_cancellation"
    elif fires <= 0:
        reason = "no_realized_fires"
    else:
        reason = ""
    return {
        "cancellation_audit_present": present,
        "cancellation_clean": clean,
        "primary_metric_valid": clean and fires > 0,
        "metric_exclusion_reason": reason,
        "non_fired_nonzero_count": nonzero,
        "non_fired_delta_sum": delta_sum,
        "non_fired_delta_max_abs": max_abs,
    }


def candidate_metrics(groups: dict[str, list[Path]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_name, paths in sorted(groups.items()):
        summary_rows, cancellation_rows, seed_rows = aggregate(paths)
        cancellation_by_config = {str(row["config_id"]): row for row in cancellation_rows}
        seed_count_by_config: dict[str, int] = {}
        for row in seed_rows:
            config_id = str(row.get("config_id", ""))
            seed_count_by_config[config_id] = seed_count_by_config.get(config_id, 0) + 1
        for row in summary_rows:
            config_id = str(row["config_id"])
            cancel = cancellation_by_config.get(config_id, {})
            fires = safe_int(row.get("realized_override_count"))
            cancellation = cancellation_status(cancel, fires)
            rows.append(
                {
                    "group": group_name,
                    "config_id": config_id,
                    "input_dir_count": len(paths),
                    "seed_rows": seed_count_by_config.get(config_id, row.get("seed_rows", 0)),
                    "paired_seeds": row["paired_seeds"],
                    "decision_count": row["decision_count"],
                    "fires": fires,
                    "override_rate": row["override_rate"],
                    "estimated_ev_per_hand": row["estimated_ev_per_hand"],
                    "per_fire_delta_mean": row["per_fire_delta_mean"],
                    "per_fire_delta_ci95_low": row["per_fire_delta_ci95_low"],
                    "per_fire_delta_ci95_high": row["per_fire_delta_ci95_high"],
                    "realized_loss_count": row["realized_loss_count"],
                    "p95_loss": row["p95_loss"],
                    "max_loss": row["max_loss"],
                    **cancellation,
                }
            )
    rows.sort(key=lambda row: (str(row["config_id"]), -safe_float(row["estimated_ev_per_hand"])))
    return rows


def overlap_metrics(groups: dict[str, list[Path]]) -> list[dict[str, Any]]:
    fired_by_group = {name: fired_sets_by_config(paths) for name, paths in sorted(groups.items())}
    rows: list[dict[str, Any]] = []
    for left_name, right_name in combinations(sorted(fired_by_group), 2):
        left_configs = fired_by_group[left_name]
        right_configs = fired_by_group[right_name]
        for config_id in sorted(set(left_configs) | set(right_configs)):
            left = left_configs.get(config_id, {})
            right = right_configs.get(config_id, {})
            left_keys = set(left)
            right_keys = set(right)
            overlap = left_keys & right_keys
            union = left_keys | right_keys
            rows.append(
                {
                    "left_group": left_name,
                    "right_group": right_name,
                    "config_id": config_id,
                    "left_fires": len(left_keys),
                    "right_fires": len(right_keys),
                    "overlap": len(overlap),
                    "union": len(union),
                    "jaccard": len(overlap) / len(union) if union else 0.0,
                    "left_only": len(left_keys - right_keys),
                    "right_only": len(right_keys - left_keys),
                    "same_action_overlap": sum(1 for key in overlap if left.get(key) == right.get(key)),
                }
            )
    return rows


def write_markdown(path: Path, metric_rows: list[dict[str, Any]], overlap_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# HU T2 Stage8c Candidate Generator Comparison",
        "",
        "This is validation-only. It does not authorize production, P2 fixed status, T1, or 50k teacher generation.",
        "",
        "## Metrics",
        "",
        "| group | paired | fires | primary valid | fire rate | EV/hand | per-fire delta | per-fire CI | losses | max loss | non-fired nonzero | exclusion |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in metric_rows:
        lines.append(
            "| {group} | {paired} | {fires} | {valid} | {rate:.4%} | {ev:+.4f} | {pf:+.4f} | [{low:+.4f}, {high:+.4f}] | {losses} | {max_loss:.4f} | {nonzero} | `{reason}` |".format(
                group=row["group"],
                paired=row["paired_seeds"],
                fires=row["fires"],
                valid=row["primary_metric_valid"],
                rate=safe_float(row["override_rate"]),
                ev=safe_float(row["estimated_ev_per_hand"]),
                pf=safe_float(row["per_fire_delta_mean"]),
                low=safe_float(row["per_fire_delta_ci95_low"]),
                high=safe_float(row["per_fire_delta_ci95_high"]),
                losses=row["realized_loss_count"],
                max_loss=safe_float(row["max_loss"]),
                nonzero=row["non_fired_nonzero_count"],
                reason=row.get("metric_exclusion_reason", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Fired Set Overlap",
            "",
            "| left | right | overlap | union | Jaccard | left only | right only | same action overlap |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in overlap_rows:
        lines.append(
            "| {left} | {right} | {overlap} | {union} | {jaccard:.4f} | {left_only} | {right_only} | {same_action} |".format(
                left=row["left_group"],
                right=row["right_group"],
                overlap=row["overlap"],
                union=row["union"],
                jaccard=safe_float(row["jaccard"]),
                left_only=row["left_only"],
                right_only=row["right_only"],
                same_action=row["same_action_overlap"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- execution: `Pass`",
            "- production / P2 fixed: `No-Go`",
            "- 50k teacher: `No-Go`",
            "- T1 training: `No-Go`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    groups = parse_groups(args.group)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = candidate_metrics(groups)
    overlap_rows = overlap_metrics(groups)
    write_csv(args.output_dir / "candidate_generator_metrics.csv", metric_rows)
    write_csv(args.output_dir / "candidate_generator_overlap.csv", overlap_rows)
    write_markdown(args.output_dir / "candidate_generator_comparison.md", metric_rows, overlap_rows)
    print(
        json.dumps(
            {
                "groups": {name: len(paths) for name, paths in groups.items()},
                "metric_rows": len(metric_rows),
                "overlap_rows": len(overlap_rows),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
