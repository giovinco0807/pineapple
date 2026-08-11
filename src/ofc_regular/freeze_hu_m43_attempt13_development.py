"""Freeze an all-gates-pass Attempt13 Development200 decision for Audit50."""

from __future__ import annotations

import argparse
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from . import freeze_hu_m43_attempt09_development as _engine
from . import freeze_hu_m43_attempt12_development as _base
from . import hu_m43_attempt13_spot as _spot
from .hu_m43_attempt13_contract import M43_ATTEMPT13_PLAN_SHA256, M43_ATTEMPT13_PROFILES
from .select_hu_m43_attempt13_development import (
    ATTEMPT13_DEVELOPMENT_DECISION_SCHEMA,
    ATTEMPT13_DEVELOPMENT_RECEIPT_SCHEMA,
)


ATTEMPT13_DEVELOPMENT_GO_FREEZE_SCHEMA = "hu_m43_attempt13_development_go_freeze_v1"
ATTEMPT13_DEVELOPMENT_GO_FREEZE_STATUS = "go_freeze_attempt13_development"

_LOCK = threading.RLock()
_INTEGRITY_NAMES = (
    "action_mapping",
    "rng_domain",
    "hidden_information",
    "risk_reserve_contract",
    "retained_order",
    "phase_filter",
    "pooled_phase",
    "locked_action_change",
    "extreme_tail_statistic",
)


def _freeze_engine_bindings() -> dict[str, Any]:
    return {
        "spot": _spot,
        "M43_ATTEMPT09_PLAN_SHA256": M43_ATTEMPT13_PLAN_SHA256,
        "M43_ATTEMPT09_PROFILES": M43_ATTEMPT13_PROFILES,
        "ATTEMPT09_DEVELOPMENT_DECISION_SCHEMA": ATTEMPT13_DEVELOPMENT_DECISION_SCHEMA,
        "ATTEMPT09_DEVELOPMENT_RECEIPT_SCHEMA": ATTEMPT13_DEVELOPMENT_RECEIPT_SCHEMA,
        "ATTEMPT09_DEVELOPMENT_GO_FREEZE_SCHEMA": ATTEMPT13_DEVELOPMENT_GO_FREEZE_SCHEMA,
        "DEVELOPMENT_GO_FREEZE_STATUS": ATTEMPT13_DEVELOPMENT_GO_FREEZE_STATUS,
        "SELECTOR_SOURCE_NAME": "select_hu_m43_attempt13_development.py",
    }


def _module_bindings() -> dict[str, Any]:
    return {
        "_spot": _spot,
        "M43_ATTEMPT12_PLAN_SHA256": M43_ATTEMPT13_PLAN_SHA256,
        "M43_ATTEMPT12_PROFILES": M43_ATTEMPT13_PROFILES,
        "ATTEMPT12_DEVELOPMENT_DECISION_SCHEMA": ATTEMPT13_DEVELOPMENT_DECISION_SCHEMA,
        "ATTEMPT12_DEVELOPMENT_RECEIPT_SCHEMA": ATTEMPT13_DEVELOPMENT_RECEIPT_SCHEMA,
        "ATTEMPT12_DEVELOPMENT_GO_FREEZE_SCHEMA": ATTEMPT13_DEVELOPMENT_GO_FREEZE_SCHEMA,
        "ATTEMPT12_DEVELOPMENT_GO_FREEZE_STATUS": ATTEMPT13_DEVELOPMENT_GO_FREEZE_STATUS,
        "_INTEGRITY_NAMES": _INTEGRITY_NAMES,
        "_BINDINGS": _freeze_engine_bindings(),
    }


@contextmanager
def _attempt13_freeze_bindings() -> Iterator[None]:
    bindings = _module_bindings()
    with _LOCK:
        prior = {name: getattr(_base, name) for name in bindings}
        try:
            for name, value in bindings.items():
                setattr(_base, name, value)
            with _base._attempt12_freeze_bindings():
                yield
        finally:
            for name, value in prior.items():
                setattr(_base, name, value)


def build_attempt13_development_go_freeze(
    *,
    run_dir: str | Path,
    received_dir: str | Path,
    decision_path: str | Path,
    decision_receipt_path: str | Path,
) -> dict[str, Any]:
    with _attempt13_freeze_bindings():
        return _engine.build_attempt09_development_go_freeze(
            run_dir=run_dir,
            received_dir=received_dir,
            decision_path=decision_path,
            decision_receipt_path=decision_receipt_path,
        )


def freeze_attempt13_development(
    *,
    run_dir: str | Path,
    received_dir: str | Path,
    decision_path: str | Path,
    decision_receipt_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    with _attempt13_freeze_bindings():
        return _engine.freeze_attempt09_development(
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
    payload = freeze_attempt13_development(
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
    "ATTEMPT13_DEVELOPMENT_GO_FREEZE_SCHEMA",
    "build_attempt13_development_go_freeze",
    "freeze_attempt13_development",
]
