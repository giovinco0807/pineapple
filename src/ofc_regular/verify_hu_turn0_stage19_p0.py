"""Verify the preregistered HU T0 Stage19 P0 final holdout."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _check(name: str, passed: bool, actual: Any, expected: str) -> dict[str, Any]:
    return {
        "check": name,
        "passed": bool(passed),
        "actual": actual,
        "expected": expected,
    }


def _normalized_path_text(value: Any) -> str:
    return str(value or "").replace("\\", "/")


def _numeric_check(
    checks: list[dict[str, Any]],
    row: dict[str, Any],
    key: str,
    predicate: Any,
    expected: str,
) -> None:
    value = _finite_float(row.get(key))
    checks.append(_check(key, value is not None and predicate(value), value, expected))


def _candidate_checks(
    *,
    candidate: dict[str, Any],
    row: dict[str, Any] | None,
    plan: dict[str, Any],
    summary: dict[str, Any],
    tail_audit: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if row is None:
        return [_check("config_present", False, None, candidate["id"])]

    acceptance = plan["acceptance"]
    evaluation = plan["evaluation"]
    expected_margins = {
        key: float(value) for key, value in candidate["min_margin_by_seat"].items()
    }
    actual_margins = {
        key: float(value) for key, value in (row.get("min_margin_by_seat") or {}).items()
    }
    expected_seats = tuple(candidate["allowed_seats"])
    actual_seats = tuple(row.get("allowed_seats") or ())
    by_seat = row.get("by_seat") or {}
    enabled_seats_nonnegative = all(
        _finite_float((by_seat.get(seat) or {}).get("avg_delta_per_hand")) is not None
        and float((by_seat.get(seat) or {})["avg_delta_per_hand"]) >= 0.0
        for seat in expected_seats
    )

    checks = [
        _check("config_present", True, row.get("config_id"), candidate["id"]),
        _check(
            "candidate_model",
            _normalized_path_text(row.get("candidate_model"))
            == _normalized_path_text(plan["candidate_model"]),
            row.get("candidate_model"),
            plan["candidate_model"],
        ),
        _check(
            "candidate_topk",
            int(row.get("candidate_topk", -1)) == int(plan["candidate_topk"]),
            row.get("candidate_topk"),
            f"== {plan['candidate_topk']}",
        ),
        _check(
            "paired_seeds",
            int(row.get("paired_seeds", -1))
            == int(evaluation["paired_seeds_per_config"]),
            row.get("paired_seeds"),
            f"== {evaluation['paired_seeds_per_config']}",
        ),
        _check(
            "min_margin_by_seat",
            actual_margins == expected_margins,
            actual_margins,
            str(expected_margins),
        ),
        _check(
            "allowed_seats",
            actual_seats == expected_seats,
            list(actual_seats),
            str(list(expected_seats)),
        ),
        _check(
            "minimum_fires",
            int(row.get("fires", -1)) >= int(acceptance["minimum_fires"]),
            row.get("fires"),
            f">= {acceptance['minimum_fires']}",
        ),
        _check(
            "non_fired_nonzero_count",
            int(row.get("non_fired_nonzero_count", -1))
            == int(acceptance["non_fired_nonzero_count"]),
            row.get("non_fired_nonzero_count"),
            f"== {acceptance['non_fired_nonzero_count']}",
        ),
        _check(
            "enabled_seat_point_estimates_nonnegative",
            enabled_seats_nonnegative,
            {
                seat: (by_seat.get(seat) or {}).get("avg_delta_per_hand")
                for seat in expected_seats
            },
            ">= 0 for every enabled seat",
        ),
        _check(
            "duplicate_event_count",
            int(summary.get("duplicate_event_count", -1))
            == int(acceptance["duplicate_event_count"]),
            summary.get("duplicate_event_count"),
            f"== {acceptance['duplicate_event_count']}",
        ),
        _check(
            "seed_set_mismatch_configs",
            list(summary.get("seed_set_mismatch_configs") or [])
            == list(acceptance["seed_set_mismatch_configs"]),
            summary.get("seed_set_mismatch_configs"),
            str(acceptance["seed_set_mismatch_configs"]),
        ),
    ]
    expected_selector_model = plan.get("safe_selector_model")
    if expected_selector_model is not None:
        checks.extend(
            [
                _check(
                    "safe_selector_model",
                    _normalized_path_text(row.get("safe_selector_model"))
                    == _normalized_path_text(expected_selector_model),
                    row.get("safe_selector_model"),
                    str(expected_selector_model),
                ),
                _check(
                    "safe_selector_threshold_by_seat",
                    {
                        key: float(value)
                        for key, value in (
                            row.get("safe_selector_threshold_by_seat") or {}
                        ).items()
                    }
                    == {
                        key: float(value)
                        for key, value in candidate[
                            "safe_selector_threshold_by_seat"
                        ].items()
                    },
                    row.get("safe_selector_threshold_by_seat"),
                    str(candidate["safe_selector_threshold_by_seat"]),
                ),
            ]
        )
    _numeric_check(
        checks,
        row,
        "avg_delta_per_hand",
        lambda value: value
        > float(acceptance["aggregate_avg_delta_per_hand_min_exclusive"]),
        f"> {acceptance['aggregate_avg_delta_per_hand_min_exclusive']}",
    )
    _numeric_check(
        checks,
        row,
        "ci95_low",
        lambda value: value >= float(acceptance["aggregate_ci95_low_min"]),
        f">= {acceptance['aggregate_ci95_low_min']}",
    )
    _numeric_check(
        checks,
        row,
        "avg_delta_per_fire",
        lambda value: value > float(acceptance["avg_delta_per_fire_min_exclusive"]),
        f"> {acceptance['avg_delta_per_fire_min_exclusive']}",
    )
    _numeric_check(
        checks,
        row,
        "per_fire_ci95_low",
        lambda value: value >= float(acceptance["per_fire_ci95_low_min"]),
        f">= {acceptance['per_fire_ci95_low_min']}",
    )
    _numeric_check(
        checks,
        row,
        "p99_fire_loss",
        lambda value: value <= float(acceptance["p99_fire_loss_max"]),
        f"<= {acceptance['p99_fire_loss_max']}",
    )
    if "max_fire_loss_max" in acceptance:
        _numeric_check(
            checks,
            row,
            "max_fire_loss",
            lambda value: value <= float(acceptance["max_fire_loss_max"]),
            f"<= {acceptance['max_fire_loss_max']}",
        )
    if plan.get("tail_audit") is not None:
        expected_tail = plan["tail_audit"]
        labels = (tail_audit or {}).get("label_counts") or {}
        records = int((tail_audit or {}).get("records", 0))
        negative_rate = (
            int(labels.get("negative", 0)) / records if records > 0 else None
        )
        checks.extend(
            [
                _check(
                    "tail_audit_records",
                    records >= int(expected_tail["minimum_records"]),
                    records,
                    f">= {expected_tail['minimum_records']}",
                ),
                _check(
                    "tail_audit_complete",
                    not list((tail_audit or {}).get("missing_shards") or []),
                    (tail_audit or {}).get("missing_shards"),
                    "[]",
                ),
                _check(
                    "tail_audit_common_random_futures",
                    (tail_audit or {}).get("common_random_futures_verified") is True,
                    (tail_audit or {}).get("common_random_futures_verified"),
                    "true",
                ),
                _check(
                    "tail_audit_action_mapping",
                    (tail_audit or {}).get("action_mapping_verified") is True,
                    (tail_audit or {}).get("action_mapping_verified"),
                    "true",
                ),
                _check(
                    "tail_audit_delta_mean",
                    _finite_float((tail_audit or {}).get("delta_mean")) is not None
                    and float(tail_audit["delta_mean"])
                    > float(expected_tail["delta_mean_min_exclusive"]),
                    (tail_audit or {}).get("delta_mean"),
                    f"> {expected_tail['delta_mean_min_exclusive']}",
                ),
                _check(
                    "tail_audit_negative_rate",
                    negative_rate is not None
                    and negative_rate <= float(expected_tail["negative_rate_max"]),
                    negative_rate,
                    f"<= {expected_tail['negative_rate_max']}",
                ),
            ]
        )
    return checks


def verify_holdout(
    plan: dict[str, Any],
    summary: dict[str, Any],
    tail_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = {str(row.get("config_id")): row for row in summary.get("config_results", [])}
    candidate_results: list[dict[str, Any]] = []
    selected: str | None = None
    for candidate in sorted(plan["candidates"], key=lambda item: int(item["priority"])):
        checks = _candidate_checks(
            candidate=candidate,
            row=rows.get(candidate["id"]),
            plan=plan,
            summary=summary,
            tail_audit=tail_audit,
        )
        passed = all(check["passed"] for check in checks)
        candidate_results.append(
            {
                "candidate_id": candidate["id"],
                "priority": candidate["priority"],
                "passed": passed,
                "checks": checks,
            }
        )
        if selected is None and passed:
            selected = candidate["id"]

    return {
        "schema": "hu_turn0_stage19_p0_acceptance_v1",
        "passed": selected is not None,
        "selected_candidate": selected,
        "decision": "Go" if selected is not None else "No-Go",
        "performance_source": "fresh_same_seed_whole_game_counterfactual",
        "selection_order": plan["acceptance"]["selection_order"],
        "candidate_results": candidate_results,
    }


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# HU T0 Stage19 P0 Final Holdout",
        "",
        f"- decision: `{result['decision']}`",
        f"- selected candidate: `{result['selected_candidate']}`",
        "- performance source: `fresh_same_seed_whole_game_counterfactual`",
        "- threshold adaptation after holdout: `disabled`",
        "",
    ]
    for candidate in result["candidate_results"]:
        lines.extend(
            [
                f"## {candidate['candidate_id']}",
                "",
                f"- passed: `{candidate['passed']}`",
                "",
                "| check | passed | actual | expected |",
                "|---|---:|---|---|",
            ]
        )
        for check in candidate["checks"]:
            lines.append(
                f"| `{check['check']}` | {check['passed']} | "
                f"`{check['actual']}` | `{check['expected']}` |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--tail-audit-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fail-on-no-go", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    tail_audit = (
        json.loads(args.tail_audit_summary.read_text(encoding="utf-8-sig"))
        if args.tail_audit_summary is not None
        else None
    )
    result = verify_holdout(plan, summary, tail_audit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "acceptance.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown(args.output_dir / "acceptance.md", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.fail_on_no_go and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
