"""Apply the predeclared one-shot Attempt11 Audit50 search-quality gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from . import select_hu_m43_attempt09_development as _base
from .select_hu_m43_attempt11_development import _attempt11_selector_bindings


ATTEMPT11_AUDIT50_DECISION_SCHEMA = "hu_m43_attempt11_audit50_go_no_go_v1"
ATTEMPT11_AUDIT50_RECEIPT_SCHEMA = "hu_m43_attempt11_audit50_selector_receipt_v1"
ATTEMPT11_AUDIT50_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "decision_sha256",
        "decision",
        "search_freeze_authorized",
        "gate_evaluation_count",
        "selector_executed",
        "future_audit_authorized",
        "audit_rows_used_for_fit",
        "fit_performed",
        "threshold_selected",
        "runtime_policy_activated",
        "current_profile_mutated",
    }
)
_AUDIT_GATES = {
    "fires_total_min": 10,
    "fires_each_profile_min": 1,
    "mean_delta_per_state_strictly_greater_than": 0.0,
    "mean_delta_per_fire_strictly_greater_than": 0.0,
    "false_positive_rate_per_fire_max": 0.4,
    "override_loss_p95_max": 25.0,
    "override_loss_p99_max": 40.0,
    "override_loss_max": 50.0,
    "nonfire_exact_baseline_action_fallback_required": True,
    "teacher_values_reported_as_realized_match_ev": False,
    "realized_population_acceptance_still_required": True,
}
_AUDIT_BINDINGS: dict[str, Any] = {
    "ATTEMPT09_DEVELOPMENT_DECISION_SCHEMA": ATTEMPT11_AUDIT50_DECISION_SCHEMA,
    "ATTEMPT09_DEVELOPMENT_RECEIPT_SCHEMA": ATTEMPT11_AUDIT50_RECEIPT_SCHEMA,
    "_ROOTS": 50,
    "POPULATION": "future_audit",
    "ROOT_INDEX_FIRST": 200,
    "ROOTS_PER_PROFILE": 10,
    "GATE_SECTION": "future_audit_go_no_go",
    "POPULATION_REPORT_KEY": "audit_population",
    "GO_STATUS": "go_attempt11_audit50_search_quality",
    "NO_GO_STATUS": "no_go_close_attempt11_audit50",
    "_GATES": _AUDIT_GATES,
    "SELECTOR_SOURCE_PATH": Path(__file__),
}


@contextmanager
def _attempt11_audit_bindings() -> Iterator[None]:
    with _attempt11_selector_bindings():
        prior = {name: getattr(_base, name) for name in _AUDIT_BINDINGS}
        try:
            for name, value in _AUDIT_BINDINGS.items():
                setattr(_base, name, value)
            yield
        finally:
            for name, value in prior.items():
                setattr(_base, name, value)


def aggregate_attempt11_audit50_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    source_input_sha256: str,
    source_plan_sha256: str,
    authorization_sha256: str,
    source_package_sha256: str,
    run_name: str,
) -> dict[str, Any]:
    with _attempt11_audit_bindings():
        return _base.aggregate_attempt09_development_rows(
            rows,
            plan=plan,
            source_input_sha256=source_input_sha256,
            source_plan_sha256=source_plan_sha256,
            authorization_sha256=authorization_sha256,
            source_package_sha256=source_package_sha256,
            run_name=run_name,
        )


def execute_attempt11_audit50_selector(
    *,
    input_path: str | Path,
    plan_path: str | Path,
    authorization_path: str | Path,
    source_package_sha256: str,
    run_name: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    with _attempt11_audit_bindings():
        destination = Path(output_dir)
        if destination.exists():
            raise FileExistsError(
                f"Attempt11 audit50 selector output already exists: {destination}"
            )
        report = _base.select_attempt09_development(
            input_path=input_path,
            plan_path=plan_path,
            authorization_path=authorization_path,
            source_package_sha256=source_package_sha256,
            run_name=run_name,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.with_name(
            f".{destination.name}.{uuid.uuid4().hex}.tmp"
        )
        staging.mkdir()
        try:
            decision_bytes = _canonical(report)
            receipt = {
                "schema": ATTEMPT11_AUDIT50_RECEIPT_SCHEMA,
                "status": "single_frozen_gate_evaluation_complete",
                "run_name": run_name,
                "decision_sha256": hashlib.sha256(decision_bytes).hexdigest(),
                "decision": report["decision"],
                "search_freeze_authorized": report["search_freeze_authorized"],
                "gate_evaluation_count": 1,
                "selector_executed": True,
                "future_audit_authorized": False,
                "audit_rows_used_for_fit": False,
                "fit_performed": False,
                "threshold_selected": False,
                "runtime_policy_activated": False,
                "current_profile_mutated": False,
            }
            _validate_attempt11_audit50_receipt(
                receipt,
                run_name=run_name,
                decision=report["decision"],
                decision_sha256=hashlib.sha256(decision_bytes).hexdigest(),
                search_freeze_authorized=report["search_freeze_authorized"],
            )
            for name, data in (
                ("decision.json", decision_bytes),
                ("decision_receipt.json", _canonical(receipt)),
            ):
                with (staging / name).open("xb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
            if destination.exists():
                raise FileExistsError(
                    f"Attempt11 audit50 selector output already exists: {destination}"
                )
            os.rename(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return receipt


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _validate_attempt11_audit50_receipt(
    receipt: Mapping[str, Any],
    *,
    run_name: str,
    decision: str,
    decision_sha256: str,
    search_freeze_authorized: bool,
) -> None:
    if (
        set(receipt) != ATTEMPT11_AUDIT50_RECEIPT_KEYS
        or receipt.get("schema") != ATTEMPT11_AUDIT50_RECEIPT_SCHEMA
        or receipt.get("status") != "single_frozen_gate_evaluation_complete"
        or receipt.get("run_name") != run_name
        or receipt.get("decision_sha256") != decision_sha256
        or receipt.get("decision") != decision
        or receipt.get("search_freeze_authorized")
        is not search_freeze_authorized
        or receipt.get("gate_evaluation_count") != 1
        or receipt.get("selector_executed") is not True
        or receipt.get("future_audit_authorized") is not False
        or receipt.get("audit_rows_used_for_fit") is not False
        or receipt.get("fit_performed") is not False
        or receipt.get("threshold_selected") is not False
        or receipt.get("runtime_policy_activated") is not False
        or receipt.get("current_profile_mutated") is not False
    ):
        raise ValueError("Attempt11 audit50 selector receipt contract changed")


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
    receipt = execute_attempt11_audit50_selector(
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
    "ATTEMPT11_AUDIT50_DECISION_SCHEMA",
    "ATTEMPT11_AUDIT50_RECEIPT_KEYS",
    "ATTEMPT11_AUDIT50_RECEIPT_SCHEMA",
    "aggregate_attempt11_audit50_rows",
    "execute_attempt11_audit50_selector",
]
