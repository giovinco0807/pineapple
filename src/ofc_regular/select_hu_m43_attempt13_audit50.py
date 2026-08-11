"""Apply the predeclared one-shot Attempt13 Audit50 search-quality gates."""

from __future__ import annotations

import argparse
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from . import select_hu_m43_attempt09_development as _engine
from . import select_hu_m43_attempt12_audit50 as _base
from .select_hu_m43_attempt13_development import _attempt13_selector_bindings


ATTEMPT13_AUDIT50_DECISION_SCHEMA = "hu_m43_attempt13_audit50_go_no_go_v1"
ATTEMPT13_AUDIT50_RECEIPT_SCHEMA = "hu_m43_attempt13_audit50_selector_receipt_v1"
ATTEMPT13_AUDIT50_RECEIPT_KEYS = _base.ATTEMPT12_AUDIT50_RECEIPT_KEYS

_LOCK = threading.RLock()
_AUDIT_GATES = {
    "fires_total_min": 10,
    "fires_each_profile_min": 1,
    "mean_delta_per_state_strictly_greater_than": 0,
    "mean_delta_per_fire_strictly_greater_than": 0,
    "false_positive_rate_per_fire_max": 0.4,
    "override_loss_p95_max": 25,
    "override_loss_p99_max": 40,
    "override_loss_max": 50,
    "nonfire_exact_baseline_action_fallback_required": True,
    "teacher_values_reported_as_realized_match_ev": False,
    "realized_population_acceptance_still_required": True,
}


def _audit_engine_bindings() -> dict[str, Any]:
    return {
        "ATTEMPT09_DEVELOPMENT_DECISION_SCHEMA": ATTEMPT13_AUDIT50_DECISION_SCHEMA,
        "ATTEMPT09_DEVELOPMENT_RECEIPT_SCHEMA": ATTEMPT13_AUDIT50_RECEIPT_SCHEMA,
        "_ROOTS": 50,
        "POPULATION": "future_audit",
        "ROOT_INDEX_FIRST": 200,
        "ROOTS_PER_PROFILE": 10,
        "GATE_SECTION": "future_audit_go_no_go",
        "POPULATION_REPORT_KEY": "audit_population",
        "GO_STATUS": "go_attempt13_audit50_search_quality",
        "NO_GO_STATUS": "no_go_close_attempt13_audit50",
        "_GATES": _AUDIT_GATES,
        "SELECTOR_SOURCE_PATH": Path(__file__),
    }


def _module_bindings() -> dict[str, Any]:
    return {
        "ATTEMPT12_AUDIT50_DECISION_SCHEMA": ATTEMPT13_AUDIT50_DECISION_SCHEMA,
        "ATTEMPT12_AUDIT50_RECEIPT_SCHEMA": ATTEMPT13_AUDIT50_RECEIPT_SCHEMA,
        "ATTEMPT12_AUDIT50_RECEIPT_KEYS": ATTEMPT13_AUDIT50_RECEIPT_KEYS,
        "_AUDIT_GATES": _AUDIT_GATES,
        "_AUDIT_BINDINGS": _audit_engine_bindings(),
        "_attempt12_selector_bindings": _attempt13_selector_bindings,
    }


@contextmanager
def _attempt13_audit_bindings() -> Iterator[None]:
    bindings = _module_bindings()
    with _LOCK:
        prior = {name: getattr(_base, name) for name in bindings}
        try:
            for name, value in bindings.items():
                setattr(_base, name, value)
            with _base._attempt12_audit_bindings():
                yield
        finally:
            for name, value in prior.items():
                setattr(_base, name, value)


def aggregate_attempt13_audit50_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    source_input_sha256: str,
    source_plan_sha256: str,
    authorization_sha256: str,
    source_package_sha256: str,
    run_name: str,
) -> dict[str, Any]:
    with _attempt13_audit_bindings():
        return _engine.aggregate_attempt09_development_rows(
            rows,
            plan=plan,
            source_input_sha256=source_input_sha256,
            source_plan_sha256=source_plan_sha256,
            authorization_sha256=authorization_sha256,
            source_package_sha256=source_package_sha256,
            run_name=run_name,
        )


def select_attempt13_audit50(
    *,
    input_path: str | Path,
    plan_path: str | Path,
    authorization_path: str | Path,
    source_package_sha256: str,
    run_name: str,
) -> dict[str, Any]:
    """Recompute the frozen Audit50 decision without publishing artifacts."""

    with _attempt13_audit_bindings():
        return _engine.select_attempt09_development(
            input_path=input_path,
            plan_path=plan_path,
            authorization_path=authorization_path,
            source_package_sha256=source_package_sha256,
            run_name=run_name,
        )


def execute_attempt13_audit50_selector(
    *,
    input_path: str | Path,
    plan_path: str | Path,
    authorization_path: str | Path,
    source_package_sha256: str,
    run_name: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    with _attempt13_audit_bindings():
        # The inherited implementation supplies the atomic, write-once output
        # directory and validates the fresh Attempt13 receipt identity.
        return _base.execute_attempt12_audit50_selector(
            input_path=input_path,
            plan_path=plan_path,
            authorization_path=authorization_path,
            source_package_sha256=source_package_sha256,
            run_name=run_name,
            output_dir=output_dir,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--source-package-sha256", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = execute_attempt13_audit50_selector(
        input_path=args.input,
        plan_path=args.plan,
        authorization_path=args.authorization,
        source_package_sha256=args.source_package_sha256,
        run_name=args.run_name,
        output_dir=args.output_dir,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTEMPT13_AUDIT50_DECISION_SCHEMA",
    "ATTEMPT13_AUDIT50_RECEIPT_KEYS",
    "ATTEMPT13_AUDIT50_RECEIPT_SCHEMA",
    "aggregate_attempt13_audit50_rows",
    "execute_attempt13_audit50_selector",
    "select_attempt13_audit50",
]
