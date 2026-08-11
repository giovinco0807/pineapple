"""Verify HU T2 Stage9f guarded-production preset evidence.

The guarded preset is intentionally not enabled by default.  This verifier
checks that the preset remains opt-in only and that its production-like canary
audit artifacts still satisfy the promotion-review thresholds recorded in the
config.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ofc_regular.evaluate_hu_turn2_stage8b_topk_mc_rerank import parse_topk_configs


DEFAULT_PRESET = Path(
    "configs/hu_turn2_stage9f_cse2_csemax2_bothseat_guarded_production_preset.json"
)
DEFAULT_AUDIT_DIR = Path(
    "outputs/evals/hu_turn2_stage9f_bothseat_profile_production_like_canary30000/audit"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_single_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}, found {len(rows)}")
    return rows[0]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing numeric column {key}")
    return float(value)


def _int(row: dict[str, str], key: str) -> int:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing integer column {key}")
    return int(float(value))


def _add_check(
    checks: list[dict[str, Any]],
    failures: list[str],
    name: str,
    passed: bool,
    detail: str,
) -> None:
    checks.append({"name": name, "passed": passed, "detail": detail})
    if not passed:
        failures.append(f"{name}: {detail}")


def verify_guarded_preset(
    *,
    preset_path: Path = DEFAULT_PRESET,
    audit_dir: Path = DEFAULT_AUDIT_DIR,
    minimum_decisions: int | None = None,
    minimum_realized_fires: int | None = None,
    verification_scope: str = "promotion",
) -> dict[str, Any]:
    preset = _load_json(preset_path)
    decision = _read_single_csv_row(audit_dir / "profile_canary_decision_summary.csv")
    seats = _read_csv_rows(audit_dir / "profile_canary_seat_breakdown.csv")
    config_breakdown = _read_csv_rows(audit_dir / "profile_canary_config_breakdown.csv")

    failures: list[str] = []
    warnings: list[str] = []
    checks: list[dict[str, Any]] = []

    _add_check(
        checks,
        failures,
        "preset_status",
        preset.get("status") == "guarded_production_preset_ready_not_enabled",
        str(preset.get("status")),
    )
    _add_check(
        checks,
        failures,
        "explicit_opt_in_only",
        preset.get("requires_explicit_enable") is True
        and preset.get("production_default") is False
        and preset.get("production_p2_fixed") is False,
        "requires explicit enable and keeps production/P2 defaults false",
    )
    _add_check(
        checks,
        failures,
        "selective_override_only",
        preset.get("full_replacement_enabled") is False
        and preset.get("runtime", {}).get("full_replacement_enabled") is False
        and preset.get("safety", {}).get("stage9f_selective_override_only") is True,
        "full replacement disabled",
    )
    _add_check(
        checks,
        failures,
        "rollback_configured",
        preset.get("rollback", {}).get("preset") == "stage9f_off",
        str(preset.get("rollback", {})),
    )
    _add_check(
        checks,
        failures,
        "t3_continuation",
        preset.get("t3_continuation", {}).get("profile") == "stage7_m5_r10"
        and preset.get("t3_continuation", {}).get("hu_turn3_min_margin") == 5.0
        and preset.get("t3_continuation", {}).get("hu_turn3_reference_min_margin") == 10.0,
        str(preset.get("t3_continuation", {})),
    )

    parsed = parse_topk_configs(preset["runtime"]["config_string"])[0]
    _add_check(
        checks,
        failures,
        "runtime_shape",
        parsed.top_k == 3
        and parsed.mc_samples == 16
        and parsed.confirm_mc_samples == 32
        and parsed.confirm_se_multiplier == 2.0
        and parsed.max_confirm_se == 2.0
        and parsed.allowed_seats == ("first", "second"),
        preset["runtime"]["config_string"],
    )

    monitoring = preset["monitoring"]
    warning_thresholds = monitoring["warning_thresholds"]
    rollback_thresholds = monitoring["rollback_thresholds"]
    required_decisions = (
        int(monitoring["promotion_minimum_paired_decisions"])
        if minimum_decisions is None
        else int(minimum_decisions)
    )
    required_fires = (
        int(monitoring["promotion_minimum_fired_decisions"])
        if minimum_realized_fires is None
        else int(minimum_realized_fires)
    )
    count_check_prefix = "promotion" if verification_scope == "promotion" else verification_scope

    decisions = _int(decision, "decision_count")
    fires = _int(decision, "realized_override_count")
    replay_ready = _int(decision, "replay_ready_count")
    non_fired_nonzero = _int(decision, "non_fired_nonzero_count")
    per_fire_ci_low = _float(decision, "realized_per_fire_delta_ci95_low")
    p95_loss = _float(decision, "realized_per_fire_loss_p95")
    max_loss = _float(decision, "realized_per_fire_loss_max")
    latency_p95 = _float(decision, "latency_ms_p95")
    stage_a_latency_p95 = _float(decision, "latency_component_stage_a_p95_ms")
    confirm_latency_p95 = _float(decision, "latency_component_confirm_p95_ms")
    overhead_latency_p95 = _float(decision, "latency_component_overhead_p95_ms")
    missing_fields = decision.get("missing_replay_field_counts", "")

    _add_check(
        checks,
        failures,
        f"{count_check_prefix}_decision_count",
        decisions >= required_decisions,
        f"{decisions} decisions, required {required_decisions}",
    )
    _add_check(
        checks,
        failures,
        f"{count_check_prefix}_fire_count",
        fires >= required_fires,
        f"{fires} realized fires, required {required_fires}",
    )
    _add_check(
        checks,
        failures,
        "aggregate_per_fire_positive",
        per_fire_ci_low > rollback_thresholds["per_fire_ci95_low"],
        f"ci95_low={per_fire_ci_low}",
    )
    _add_check(
        checks,
        failures,
        "replay_ready_complete",
        replay_ready == decisions and missing_fields == "{}",
        f"replay_ready={replay_ready}/{decisions}, missing={missing_fields}",
    )
    _add_check(
        checks,
        failures,
        "non_fired_cancellation",
        non_fired_nonzero == monitoring["required"]["non_fired_nonzero_count"],
        f"non_fired_nonzero={non_fired_nonzero}",
    )
    _add_check(
        checks,
        failures,
        "tail_loss_below_rollback",
        p95_loss <= rollback_thresholds["p95_fired_loss"]
        and max_loss < rollback_thresholds["max_fired_loss"],
        f"p95_loss={p95_loss}, max_loss={max_loss}",
    )
    _add_check(
        checks,
        failures,
        "latency_below_rollback",
        latency_p95 < rollback_thresholds["latency_p95_ms"],
        f"p95_latency_ms={latency_p95}",
    )
    if latency_p95 > warning_thresholds["latency_p95_ms"]:
        warnings.append(
            f"latency_p95_ms {latency_p95:.2f} exceeds warning threshold "
            f"{warning_thresholds['latency_p95_ms']:.2f}; components p95 ms: "
            f"stage_a={stage_a_latency_p95:.2f}, "
            f"confirm={confirm_latency_p95:.2f}, "
            f"overhead={overhead_latency_p95:.2f}"
        )

    seat_by_name = {row["seat"]: row for row in seats}
    for seat in ("first", "second"):
        row = seat_by_name.get(seat)
        _add_check(
            checks,
            failures,
            f"{seat}_seat_present",
            row is not None,
            f"seat={seat}",
        )
        if row is None:
            continue
        seat_ci_low = _float(row, "realized_per_fire_delta_ci95_low")
        seat_non_fired = _int(row, "non_fired_nonzero_count")
        _add_check(
            checks,
            failures,
            f"{seat}_seat_positive",
            seat_ci_low > rollback_thresholds[f"{seat}_per_fire_ci95_low"],
            f"ci95_low={seat_ci_low}",
        )
        _add_check(
            checks,
            failures,
            f"{seat}_seat_non_fired_cancellation",
            seat_non_fired == 0,
            f"non_fired_nonzero={seat_non_fired}",
        )

    config_map = {(row["bucket_type"], row["bucket"]): int(row["decision_count"]) for row in config_breakdown}
    _add_check(
        checks,
        failures,
        "profile_a_matches",
        _int(decision, "decision_count")
        == config_map.get(("runtime_profile", "stage9f_cse2_csemax2_bothseat")),
        str(config_map),
    )
    _add_check(
        checks,
        failures,
        "t3_policy_matches",
        _int(decision, "decision_count")
        == config_map.get(("t3_continuation_policy", "Stage7_candidate_A_m5_r10")),
        str(config_map),
    )

    result = {
        "status": "pass" if not failures else "fail",
        "preset_path": str(preset_path),
        "audit_dir": str(audit_dir),
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
        "decision": {
            "guarded_production_preset": "Ready" if not failures else "No-Go",
            "production_default": "No-Go until explicitly enabled",
            "p2_fixed": "No-Go until explicitly accepted",
            "teacher_50k": "No-Go",
            "turn1_training": "No-Go",
        },
        "metrics": {
            "verification_scope": verification_scope,
            "minimum_decisions": required_decisions,
            "minimum_realized_fires": required_fires,
            "decisions": decisions,
            "realized_fires": fires,
            "per_fire_ci95_low": per_fire_ci_low,
            "p95_loss": p95_loss,
            "max_loss": max_loss,
            "latency_p95_ms": latency_p95,
            "stage_a_latency_p95_ms": stage_a_latency_p95,
            "confirm_latency_p95_ms": confirm_latency_p95,
            "overhead_latency_p95_ms": overhead_latency_p95,
            "replay_ready": replay_ready,
            "non_fired_nonzero_count": non_fired_nonzero,
        },
    }
    return result


def _write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "guarded_preset_verification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = [
        "# HU T2 Stage9f Guarded Preset Verification",
        "",
        f"- status: `{result['status']}`",
        f"- preset: `{result['preset_path']}`",
        f"- audit dir: `{result['audit_dir']}`",
        f"- decisions: `{result['metrics']['decisions']}`",
        f"- realized fires: `{result['metrics']['realized_fires']}`",
        f"- per-fire CI low: `{result['metrics']['per_fire_ci95_low']}`",
        f"- p95/max loss: `{result['metrics']['p95_loss']}` / `{result['metrics']['max_loss']}`",
        f"- latency p95 ms: `{result['metrics']['latency_p95_ms']}`",
        f"- latency component p95 ms: stage A `{result['metrics']['stage_a_latency_p95_ms']}`, "
        f"confirm `{result['metrics']['confirm_latency_p95_ms']}`, "
        f"overhead `{result['metrics']['overhead_latency_p95_ms']}`",
        "",
        "## Decision",
        "",
    ]
    for key, value in result["decision"].items():
        lines.append(f"- {key}: `{value}`")
    if result["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in result["warnings"])
    if result["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in result["failures"])
    (output_dir / "guarded_preset_verification.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", type=Path, default=DEFAULT_PRESET)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--minimum-decisions", type=int, default=None)
    parser.add_argument("--minimum-realized-fires", type=int, default=None)
    parser.add_argument("--verification-scope", default="promotion")
    args = parser.parse_args(argv)

    result = verify_guarded_preset(
        preset_path=args.preset,
        audit_dir=args.audit_dir,
        minimum_decisions=args.minimum_decisions,
        minimum_realized_fires=args.minimum_realized_fires,
        verification_scope=args.verification_scope,
    )
    if args.output_dir is not None:
        _write_outputs(result, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
