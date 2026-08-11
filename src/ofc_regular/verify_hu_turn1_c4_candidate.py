"""Verify a fixed HU Turn1 C4 candidate against predeclared acceptance gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_checks(
    row: dict[str, Any],
    matchup: dict[str, Any],
    *,
    expected_config: str,
    expected_decisions: int,
    min_fires: int,
    max_p95_loss: float,
    max_p99_loss: float,
    max_loss: float,
) -> list[dict[str, Any]]:
    def check(name: str, passed: bool, actual: Any, expected: str) -> dict[str, Any]:
        return {"check": name, "passed": bool(passed), "actual": actual, "expected": expected}

    config_id = str(row.get("config_id") or "")
    decisions = safe_int(row.get("decision_count"))
    valid_decisions = safe_int(row.get("valid_decision_count"))
    fires = safe_int(row.get("valid_override_count"))
    per_fire_mean = safe_float(row.get("realized_per_fire_delta_mean"))
    per_fire_ci_low = safe_float(row.get("realized_per_fire_ci95_low"))
    ev_ci_low = safe_float(row.get("estimated_ev_per_decision_ci95_low"))
    whole_hand_ev = safe_float(matchup.get("avg_score_per_hand_for_a"))

    return [
        check("config_id", config_id == expected_config, config_id, expected_config),
        check("decision_count", decisions == expected_decisions, decisions, f"== {expected_decisions}"),
        check(
            "paired_realized_coverage",
            decisions - valid_decisions == fires,
            decisions - valid_decisions,
            f"== valid fires ({fires})",
        ),
        check("minimum_fires", fires >= min_fires, fires, f">= {min_fires}"),
        check("invalid_overrides", safe_int(row.get("invalid_override_count")) == 0, row.get("invalid_override_count"), "== 0"),
        check("per_fire_mean_positive", per_fire_mean > 0.0, per_fire_mean, "> 0"),
        check("per_fire_ci95_low_positive", per_fire_ci_low > 0.0, per_fire_ci_low, "> 0"),
        check("ev_per_decision_ci95_low_positive", ev_ci_low > 0.0, ev_ci_low, "> 0"),
        check("whole_hand_ev_positive", whole_hand_ev > 0.0, whole_hand_ev, "> 0"),
        check(
            "non_fired_counterfactual_zero",
            safe_int(row.get("non_fired_counterfactual_nonzero_count")) == 0,
            row.get("non_fired_counterfactual_nonzero_count"),
            "== 0",
        ),
        check(
            "non_fired_final_mismatch_zero",
            safe_int(row.get("non_fired_final_mismatch_count")) == 0,
            row.get("non_fired_final_mismatch_count"),
            "== 0",
        ),
        check(
            "non_fired_final_unknown_zero",
            safe_int(row.get("non_fired_final_match_unknown_count")) == 0,
            row.get("non_fired_final_match_unknown_count"),
            "== 0",
        ),
        check("p95_loss", safe_float(row.get("p95_loss")) <= max_p95_loss, row.get("p95_loss"), f"<= {max_p95_loss}"),
        check("p99_loss", safe_float(row.get("p99_loss")) <= max_p99_loss, row.get("p99_loss"), f"<= {max_p99_loss}"),
        check("max_loss", safe_float(row.get("max_loss")) <= max_loss, row.get("max_loss"), f"<= {max_loss}"),
    ]


def _read_summary_row(path: Path, expected_config: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if str(row.get("config_id") or "") == expected_config]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one summary row for {expected_config!r}; found {len(matches)}")
    return matches[0]


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# HU T1 C4 Acceptance",
        "",
        f"- decision: `{'Go' if result['passed'] else 'No-Go'}`",
        f"- config: `{result['expected_config']}`",
        "- performance source: `realized_seat_swap_counterfactual`",
        "- confirm delta role: `gate_diagnostic_only`",
        "",
        "| check | passed | actual | expected |",
        "|---|---:|---:|---|",
    ]
    for row in result["checks"]:
        lines.append(
            f"| `{row['check']}` | {row['passed']} | `{row['actual']}` | `{row['expected']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--matchup-summary", type=Path, required=True)
    parser.add_argument("--expected-config", required=True)
    parser.add_argument("--expected-decisions", type=int, default=60000)
    parser.add_argument("--min-fires", type=int, default=300)
    parser.add_argument("--max-p95-loss", type=float, default=25.0)
    parser.add_argument("--max-p99-loss", type=float, default=40.0)
    parser.add_argument("--max-loss", type=float, default=50.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fail-on-no-go", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row = _read_summary_row(args.summary_csv, args.expected_config)
    matchup = json.loads(args.matchup_summary.read_text(encoding="utf-8"))
    checks = build_checks(
        row,
        matchup,
        expected_config=args.expected_config,
        expected_decisions=args.expected_decisions,
        min_fires=args.min_fires,
        max_p95_loss=args.max_p95_loss,
        max_p99_loss=args.max_p99_loss,
        max_loss=args.max_loss,
    )
    result = {
        "schema": "hu_turn1_c4_acceptance_v1",
        "passed": all(check["passed"] for check in checks),
        "expected_config": args.expected_config,
        "summary_csv": str(args.summary_csv),
        "matchup_summary": str(args.matchup_summary),
        "checks": checks,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "acceptance.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown(args.output_dir / "acceptance.md", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.fail_on_no_go and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
