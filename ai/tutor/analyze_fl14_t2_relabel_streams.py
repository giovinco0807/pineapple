"""Audit two independent FL14 T2 relabel streams.

This tool does not merge labels.  It answers whether a two-stream pilot is
precise enough to scale and partitions roots into stable labels versus roots
that still need adjudication.  All ranking comparisons are made within a root,
so a harmless stream-wide value offset is removed from the action-difference
RMSE.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _read_jsonl(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    rows: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            root_id = str(row["id"])
            if root_id in rows:
                duplicates.append(root_id)
                continue
            rows[root_id] = row
    return rows, sorted(set(duplicates))


def _expected_ids(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    rows, duplicates = _read_jsonl(path)
    if duplicates:
        raise ValueError(f"expected-roots contains duplicate ids: {duplicates[:5]}")
    return list(rows)


def _action_values(row: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    values: dict[str, float] = {}
    duplicates: list[str] = []
    for action in row.get("actions", []):
        key = str(action["action_key"])
        if key in values:
            duplicates.append(key)
            continue
        values[key] = float(action["value"])
    return values, sorted(set(duplicates))


def _best_key(values: dict[str, float]) -> str:
    # Key is a deterministic tie break; normal labels almost never tie exactly.
    return max(values, key=lambda key: (values[key], key))


def _best_second_margin(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values)
    return ordered[-1] - ordered[-2] if len(ordered) >= 2 else 0.0


def _rmse(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else 0.0


def analyze(
    stream_a_path: Path,
    stream_b_path: Path,
    *,
    expected_roots_path: Path | None = None,
    expected_opponents: int | None = None,
    expected_t3_draws: int | None = None,
    stable_margin: float = 0.5,
    min_best_agreement_rate: float = 0.59,
    max_root_centered_rmse: float = 0.65,
    max_top5_pair_gap_rmse: float = 1.10,
) -> dict[str, Any]:
    if not 0.0 <= min_best_agreement_rate <= 1.0:
        raise ValueError("min_best_agreement_rate must be in [0, 1]")
    if stable_margin < 0 or max_root_centered_rmse < 0 or max_top5_pair_gap_rmse < 0:
        raise ValueError("margins and RMSE thresholds must be non-negative")

    stream_a, duplicates_a = _read_jsonl(stream_a_path)
    stream_b, duplicates_b = _read_jsonl(stream_b_path)
    expected = _expected_ids(expected_roots_path)
    expected_set = set(expected) if expected is not None else set(stream_a) | set(stream_b)
    ids_a, ids_b = set(stream_a), set(stream_b)
    missing_a = sorted(expected_set - ids_a)
    missing_b = sorted(expected_set - ids_b)
    extra_a = sorted(ids_a - expected_set) if expected is not None else []
    extra_b = sorted(ids_b - expected_set) if expected is not None else []

    action_set_mismatch: list[str] = []
    duplicate_action_ids: list[str] = []
    short_or_setting_mismatch: list[str] = []
    valid: list[dict[str, Any]] = []
    centered_differences: list[float] = []
    top5_pair_gap_differences: list[float] = []

    for root_id in sorted(expected_set & ids_a & ids_b):
        row_a, row_b = stream_a[root_id], stream_b[root_id]
        values_a, action_duplicates_a = _action_values(row_a)
        values_b, action_duplicates_b = _action_values(row_b)
        if action_duplicates_a or action_duplicates_b:
            duplicate_action_ids.append(root_id)
            continue
        if not values_a or set(values_a) != set(values_b):
            action_set_mismatch.append(root_id)
            continue

        setting_bad = False
        if expected_opponents is not None:
            setting_bad = (
                int(row_a.get("opponents", -1)) != expected_opponents
                or int(row_b.get("opponents", -1)) != expected_opponents
            )
        if expected_t3_draws is not None:
            setting_bad = setting_bad or any(
                int(action.get("t3_draws", -1)) != expected_t3_draws
                for row in (row_a, row_b)
                for action in row.get("actions", [])
            )
        if setting_bad:
            short_or_setting_mismatch.append(root_id)
            continue

        keys = sorted(values_a)
        differences = [values_a[key] - values_b[key] for key in keys]
        mean_difference = sum(differences) / len(differences)
        centered = [difference - mean_difference for difference in differences]
        centered_differences.extend(centered)

        average = {key: (values_a[key] + values_b[key]) / 2.0 for key in keys}
        top = sorted(keys, key=lambda key: (average[key], key), reverse=True)[:5]
        root_pair_differences: list[float] = []
        for first in range(len(top)):
            for second in range(first + 1, len(top)):
                left, right = top[first], top[second]
                gap_difference = (
                    (values_a[left] - values_a[right])
                    - (values_b[left] - values_b[right])
                )
                root_pair_differences.append(gap_difference)
                top5_pair_gap_differences.append(gap_difference)

        best_a = _best_key(values_a)
        best_b = _best_key(values_b)
        average_best = _best_key(average)
        margin = _best_second_margin(average.values())
        stable = best_a == best_b and margin >= stable_margin
        valid.append(
            {
                "id": root_id,
                "actions": len(keys),
                "best_a": best_a,
                "best_b": best_b,
                "average_best": average_best,
                "best_agrees": best_a == best_b,
                "average_best_second_margin": margin,
                "root_centered_action_difference_rmse": _rmse(centered),
                "top5_pair_gap_difference_rmse": _rmse(root_pair_differences),
                "stable": stable,
            }
        )

    agreement_count = sum(row["best_agrees"] for row in valid)
    agreement_rate = agreement_count / len(valid) if valid else 0.0
    centered_rmse = _rmse(centered_differences)
    top5_rmse = _rmse(top5_pair_gap_differences)
    margins = [row["average_best_second_margin"] for row in valid]
    stable_ids = [row["id"] for row in valid if row["stable"]]
    adjudicate_ids = [row["id"] for row in valid if not row["stable"]]

    integrity_pass = not any(
        (
            duplicates_a,
            duplicates_b,
            missing_a,
            missing_b,
            extra_a,
            extra_b,
            action_set_mismatch,
            duplicate_action_ids,
            short_or_setting_mismatch,
        )
    )
    checks = {
        "integrity": integrity_pass,
        "best_agreement_rate": agreement_rate >= min_best_agreement_rate,
        "root_centered_action_difference_rmse": centered_rmse <= max_root_centered_rmse,
        "top5_pair_gap_difference_rmse": top5_rmse <= max_top5_pair_gap_rmse,
    }
    return {
        "schema": "ofc_fl14_t2_relabel_stream_audit/v1",
        "inputs": {
            "stream_a": str(stream_a_path),
            "stream_b": str(stream_b_path),
            "expected_roots": str(expected_roots_path) if expected_roots_path else None,
        },
        "thresholds": {
            "expected_opponents": expected_opponents,
            "expected_t3_draws": expected_t3_draws,
            "stable_margin": stable_margin,
            "min_best_agreement_rate": min_best_agreement_rate,
            "max_root_centered_action_difference_rmse": max_root_centered_rmse,
            "max_top5_pair_gap_difference_rmse": max_top5_pair_gap_rmse,
        },
        "integrity": {
            "expected_roots": len(expected_set),
            "stream_a_roots": len(stream_a),
            "stream_b_roots": len(stream_b),
            "valid_roots": len(valid),
            "duplicate_ids_a": duplicates_a,
            "duplicate_ids_b": duplicates_b,
            "missing_ids_a": missing_a,
            "missing_ids_b": missing_b,
            "extra_ids_a": extra_a,
            "extra_ids_b": extra_b,
            "action_set_mismatch_ids": action_set_mismatch,
            "duplicate_action_ids": duplicate_action_ids,
            "short_or_setting_mismatch_ids": short_or_setting_mismatch,
        },
        "metrics": {
            "best_agreement_count": agreement_count,
            "best_agreement_rate": agreement_rate,
            "root_centered_action_difference_rmse": centered_rmse,
            "top5_pair_gap_difference_rmse": top5_rmse,
            "mean_average_best_second_margin": sum(margins) / len(margins) if margins else 0.0,
            "stable_roots": len(stable_ids),
            "adjudicate_roots": len(adjudicate_ids),
        },
        "stable_ids": stable_ids,
        "adjudicate_ids": adjudicate_ids,
        "checks": checks,
        "passed": all(checks.values()),
        "per_root": valid,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-a", type=Path, required=True)
    parser.add_argument("--stream-b", type=Path, required=True)
    parser.add_argument("--expected-roots", type=Path)
    parser.add_argument("--expected-opponents", type=int)
    parser.add_argument("--expected-t3-draws", type=int)
    parser.add_argument("--stable-margin", type=float, default=0.5)
    parser.add_argument("--min-best-agreement-rate", type=float, default=0.59)
    parser.add_argument("--max-root-centered-rmse", type=float, default=0.65)
    parser.add_argument("--max-top5-pair-gap-rmse", type=float, default=1.10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        args.stream_a,
        args.stream_b,
        expected_roots_path=args.expected_roots,
        expected_opponents=args.expected_opponents,
        expected_t3_draws=args.expected_t3_draws,
        stable_margin=args.stable_margin,
        min_best_agreement_rate=args.min_best_agreement_rate,
        max_root_centered_rmse=args.max_root_centered_rmse,
        max_top5_pair_gap_rmse=args.max_top5_pair_gap_rmse,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("metrics", "checks", "passed")}, indent=2))


if __name__ == "__main__":
    main()
