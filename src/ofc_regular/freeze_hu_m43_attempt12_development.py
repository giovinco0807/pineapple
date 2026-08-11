"""Freeze an all-gates-pass Attempt12 Development200 decision for Audit50."""

from __future__ import annotations

import argparse
import json
import math
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from . import freeze_hu_m43_attempt09_development as _base
from . import hu_m43_attempt12_spot as _spot
from .hu_m43_attempt12_contract import M43_ATTEMPT12_PLAN_SHA256, M43_ATTEMPT12_PROFILES
from .select_hu_m43_attempt12_development import (
    ATTEMPT12_DEVELOPMENT_DECISION_SCHEMA,
    ATTEMPT12_DEVELOPMENT_RECEIPT_SCHEMA,
)


ATTEMPT12_DEVELOPMENT_GO_FREEZE_SCHEMA = "hu_m43_attempt12_development_go_freeze_v1"
ATTEMPT12_DEVELOPMENT_GO_FREEZE_STATUS = "go_freeze_attempt12_development"
_LOCK = threading.RLock()
_BASE_VALIDATE_SELECTOR = _base._validate_selector
_ASSESSMENT_SOURCE = (
    "disjoint_E512_locked_final_nonbaseline_output_vs_explicit_baseline"
)
_INTEGRITY_NAMES = (
    "action_mapping",
    "rng_domain",
    "hidden_information",
    "risk_reserve_contract",
    "retained_order",
    "phase_filter",
    "pooled_phase",
    "locked_action_change",
)
_BINDINGS: dict[str, Any] = {
    "spot": _spot,
    "M43_ATTEMPT09_PLAN_SHA256": M43_ATTEMPT12_PLAN_SHA256,
    "M43_ATTEMPT09_PROFILES": M43_ATTEMPT12_PROFILES,
    "ATTEMPT09_DEVELOPMENT_DECISION_SCHEMA": ATTEMPT12_DEVELOPMENT_DECISION_SCHEMA,
    "ATTEMPT09_DEVELOPMENT_RECEIPT_SCHEMA": ATTEMPT12_DEVELOPMENT_RECEIPT_SCHEMA,
    "ATTEMPT09_DEVELOPMENT_GO_FREEZE_SCHEMA": ATTEMPT12_DEVELOPMENT_GO_FREEZE_SCHEMA,
    "DEVELOPMENT_GO_FREEZE_STATUS": ATTEMPT12_DEVELOPMENT_GO_FREEZE_STATUS,
    "SELECTOR_SOURCE_NAME": "select_hu_m43_attempt12_development.py",
}


@contextmanager
def _attempt12_freeze_bindings() -> Iterator[None]:
    with _LOCK:
        bindings = {**_BINDINGS, "_validate_selector": _validate_attempt12_selector}
        prior = {name: getattr(_base, name) for name in bindings}
        try:
            for name, value in bindings.items():
                setattr(_base, name, value)
            yield
        finally:
            for name, value in prior.items():
                setattr(_base, name, value)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Attempt12 {label} must be a mapping")
    return value


def _finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"Attempt12 {label} must be finite")
    return float(value)


def _validate_metric_block(
    metrics: Mapping[str, Any],
    *,
    states: int,
    fires_min: int,
    enforce_quality_gates: bool,
) -> None:
    if metrics.get("states") != states:
        raise ValueError("Attempt12 E512 metric state count changed")
    fires = metrics.get("fires")
    if type(fires) is not int or fires < fires_min or fires > states:
        raise ValueError("Attempt12 E512 fire count changed")
    mean_state = _finite_number(
        metrics.get("mean_delta_per_state"), "mean delta per state"
    )
    mean_fire = _finite_number(
        metrics.get("mean_delta_per_fire"), "mean delta per fire"
    )
    if (
        (enforce_quality_gates and (mean_state <= 0 or mean_fire <= 0))
        or not math.isclose(
            mean_state,
            mean_fire * fires / states,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("Attempt12 nonfire-zero E512 mean identity changed")
    false_count = metrics.get("false_positive_fires")
    false_rate = _finite_number(
        metrics.get("false_positive_rate_per_fire"), "false positive rate"
    )
    if (
        type(false_count) is not int
        or not 0 <= false_count <= fires
        or not math.isclose(
            false_rate, false_count / fires, rel_tol=1e-12, abs_tol=1e-12
        )
        or (enforce_quality_gates and false_rate > 0.4)
    ):
        raise ValueError("Attempt12 E512 false-positive metrics changed")
    tails = _mapping(
        metrics.get("maximum_per_fired_root_loss"), "E512 tail metrics"
    )
    if set(tails) != {"p95", "p99", "max"}:
        raise ValueError("Attempt12 E512 tail metric keys changed")
    for name, limit in (("p95", 25.0), ("p99", 40.0), ("max", 50.0)):
        observed = _finite_number(tails.get(name), f"E512 {name} loss")
        if not 0 <= observed <= limit:
            raise ValueError(f"Attempt12 E512 {name} loss gate changed")


def _diagnostic_metrics(
    rows: Sequence[Mapping[str, Any]], *, states: int
) -> dict[str, Any]:
    means = [
        _finite_number(row.get("e512_paired_delta_mean"), "E512 root mean")
        for row in rows
    ]
    losses: dict[str, list[float]] = {"p95": [], "p99": [], "max": []}
    for row in rows:
        loss = _mapping(row.get("e512_per_root_loss"), "E512 root loss")
        if set(loss) != set(losses):
            raise ValueError("Attempt12 E512 root loss keys changed")
        for name in losses:
            value = _finite_number(loss.get(name), f"E512 root {name} loss")
            if value < 0:
                raise ValueError("Attempt12 E512 root loss became negative")
            losses[name].append(value)
    false_count = sum(mean <= 0 for mean in means)
    fires = len(rows)
    return {
        "states": states,
        "fires": fires,
        "mean_delta_per_state": sum(means) / states,
        "mean_delta_per_fire": sum(means) / fires if fires else None,
        "false_positive_fires": false_count,
        "false_positive_rate_per_fire": false_count / fires if fires else None,
        "maximum_per_fired_root_loss": {
            name: max(values, default=None) for name, values in losses.items()
        },
    }


def _same_metrics(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    if set(actual) != set(expected):
        return False
    for name, expected_value in expected.items():
        actual_value = actual.get(name)
        if isinstance(expected_value, Mapping):
            if not isinstance(actual_value, Mapping) or not _same_metrics(
                actual_value, expected_value
            ):
                return False
        elif isinstance(expected_value, (int, float)) and not isinstance(
            expected_value, bool
        ):
            if (
                isinstance(actual_value, bool)
                or not isinstance(actual_value, (int, float))
                or not math.isclose(
                    float(actual_value),
                    float(expected_value),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                return False
        elif actual_value != expected_value or type(actual_value) is not type(
            expected_value
        ):
            return False
    return True


def _validate_attempt12_selector(
    decision_path: Path, receipt_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reopen the exact Attempt12 E512 Go report before freezing Audit50."""

    decision, receipt = _BASE_VALIDATE_SELECTOR(decision_path, receipt_path)
    population = _mapping(
        decision.get("development_population"), "development population"
    )
    if population != {
        "roots": 200,
        "profile_counts": {profile: 40 for profile in M43_ATTEMPT12_PROFILES},
    }:
        raise ValueError("Attempt12 Development200 balance changed")
    metrics = _mapping(decision.get("metrics"), "metrics")
    overall = _mapping(metrics.get("overall"), "overall metrics")
    _validate_metric_block(
        overall, states=200, fires_min=40, enforce_quality_gates=True
    )
    by_profile = _mapping(metrics.get("by_profile"), "profile metrics")
    if set(by_profile) != set(M43_ATTEMPT12_PROFILES):
        raise ValueError("Attempt12 profile metric set changed")
    for profile in M43_ATTEMPT12_PROFILES:
        _validate_metric_block(
            _mapping(by_profile.get(profile), f"{profile} metrics"),
            states=40,
            fires_min=3,
            enforce_quality_gates=False,
        )
    diagnostics = metrics.get("fired_root_diagnostics")
    if not isinstance(diagnostics, list) or len(diagnostics) != overall["fires"]:
        raise ValueError("Attempt12 fired E512 diagnostic count changed")
    expected_diagnostic_keys = {
        "root_index",
        "profile",
        "selected_action_key",
        "e512_paired_delta_mean",
        "e512_per_root_loss",
        "false_positive",
    }
    for row in diagnostics:
        if not isinstance(row, Mapping) or set(row) != expected_diagnostic_keys:
            raise ValueError("Attempt12 fired E512 diagnostic schema changed")
        mean = _finite_number(row.get("e512_paired_delta_mean"), "E512 root mean")
        if row.get("false_positive") is not (mean <= 0):
            raise ValueError("Attempt12 fired E512 false-positive label changed")
        if (
            type(row.get("root_index")) is not int
            or not 0 <= row["root_index"] < 200
            or row.get("profile") not in M43_ATTEMPT12_PROFILES
            or not isinstance(row.get("selected_action_key"), str)
            or not row["selected_action_key"]
        ):
            raise ValueError("Attempt12 fired E512 identity changed")
    if len({row["root_index"] for row in diagnostics}) != len(diagnostics):
        raise ValueError("Attempt12 fired E512 roots repeat")
    recomputed_overall = _diagnostic_metrics(diagnostics, states=200)
    if not _same_metrics(overall, recomputed_overall):
        raise ValueError("Attempt12 E512 overall metrics were not recomputed")
    for profile in M43_ATTEMPT12_PROFILES:
        recomputed_profile = _diagnostic_metrics(
            [row for row in diagnostics if row["profile"] == profile], states=40
        )
        if not _same_metrics(by_profile[profile], recomputed_profile):
            raise ValueError("Attempt12 E512 profile metrics were not recomputed")

    integrity = _mapping(decision.get("integrity"), "integrity")
    expected_integrity = {
        **{f"{name}_violation_count": 0 for name in _INTEGRITY_NAMES},
        "nonfire_exact_baseline_action_fallback_verified": True,
    }
    if integrity != expected_integrity:
        raise ValueError("Attempt12 selector integrity counts changed")
    science = _mapping(decision.get("science_boundary"), "science boundary")
    if science.get("assessment_source") != _ASSESSMENT_SOURCE:
        raise ValueError("Attempt12 freeze assessment source changed")
    gate_map = {str(gate.get("name")): gate for gate in decision["gates"]}
    gate_names = set(gate_map)
    expected_gate_names = {
        "fires_total",
        "fires_each_profile",
        "mean_delta_per_state",
        "mean_delta_per_fire",
        "false_positive_rate_per_fire",
        "maximum_per_fired_root_p95_loss",
        "maximum_per_fired_root_p99_loss",
        "maximum_per_fired_root_max_loss",
        *(f"{name}_violation_count" for name in _INTEGRITY_NAMES),
        "nonfire_exact_baseline_action_fallback",
    }
    if gate_names != expected_gate_names or len(decision["gates"]) != len(
        expected_gate_names
    ):
        raise ValueError("Attempt12 selector gate set changed")
    expected_observed = {
        "fires_total": overall["fires"],
        "fires_each_profile": {
            profile: by_profile[profile]["fires"]
            for profile in M43_ATTEMPT12_PROFILES
        },
        "mean_delta_per_state": overall["mean_delta_per_state"],
        "mean_delta_per_fire": overall["mean_delta_per_fire"],
        "false_positive_rate_per_fire": overall[
            "false_positive_rate_per_fire"
        ],
        "maximum_per_fired_root_p95_loss": overall[
            "maximum_per_fired_root_loss"
        ]["p95"],
        "maximum_per_fired_root_p99_loss": overall[
            "maximum_per_fired_root_loss"
        ]["p99"],
        "maximum_per_fired_root_max_loss": overall[
            "maximum_per_fired_root_loss"
        ]["max"],
        **{f"{name}_violation_count": 0 for name in _INTEGRITY_NAMES},
        "nonfire_exact_baseline_action_fallback": True,
    }
    for name, expected in expected_observed.items():
        observed = gate_map[name].get("observed")
        if isinstance(expected, Mapping):
            matches = observed == expected
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            matches = (
                not isinstance(observed, bool)
                and isinstance(observed, (int, float))
                and math.isclose(
                    float(observed),
                    float(expected),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
        else:
            matches = observed is expected
        if not matches:
            raise ValueError("Attempt12 selector gate observation changed")
    return decision, receipt


def build_attempt12_development_go_freeze(
    *,
    run_dir: str | Path,
    received_dir: str | Path,
    decision_path: str | Path,
    decision_receipt_path: str | Path,
) -> dict[str, Any]:
    with _attempt12_freeze_bindings():
        return _base.build_attempt09_development_go_freeze(
            run_dir=run_dir,
            received_dir=received_dir,
            decision_path=decision_path,
            decision_receipt_path=decision_receipt_path,
        )


def freeze_attempt12_development(
    *,
    run_dir: str | Path,
    received_dir: str | Path,
    decision_path: str | Path,
    decision_receipt_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    with _attempt12_freeze_bindings():
        return _base.freeze_attempt09_development(
            run_dir=run_dir,
            received_dir=received_dir,
            decision_path=decision_path,
            decision_receipt_path=decision_receipt_path,
            output=output,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--received-dir", required=True, type=Path)
    parser.add_argument("--decision", required=True, type=Path)
    parser.add_argument("--decision-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = freeze_attempt12_development(
        run_dir=args.run_dir,
        received_dir=args.received_dir,
        decision_path=args.decision,
        decision_receipt_path=args.decision_receipt,
        output=args.output,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTEMPT12_DEVELOPMENT_GO_FREEZE_SCHEMA",
    "build_attempt12_development_go_freeze",
    "freeze_attempt12_development",
]
