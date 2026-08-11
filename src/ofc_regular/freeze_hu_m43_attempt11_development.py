"""Freeze an all-gates-pass Attempt11 Development200 decision for Audit50."""

from __future__ import annotations

import argparse
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from . import freeze_hu_m43_attempt09_development as _base
from . import hu_m43_attempt11_spot as _spot
from .hu_m43_attempt11_contract import M43_ATTEMPT11_PLAN_SHA256, M43_ATTEMPT11_PROFILES
from .select_hu_m43_attempt11_development import (
    ATTEMPT11_DEVELOPMENT_DECISION_SCHEMA,
    ATTEMPT11_DEVELOPMENT_RECEIPT_SCHEMA,
)


ATTEMPT11_DEVELOPMENT_GO_FREEZE_SCHEMA = "hu_m43_attempt11_development_go_freeze_v1"
ATTEMPT11_DEVELOPMENT_GO_FREEZE_STATUS = "go_freeze_attempt11_development"
_LOCK = threading.RLock()
_BINDINGS: dict[str, Any] = {
    "spot": _spot,
    "M43_ATTEMPT09_PLAN_SHA256": M43_ATTEMPT11_PLAN_SHA256,
    "M43_ATTEMPT09_PROFILES": M43_ATTEMPT11_PROFILES,
    "ATTEMPT09_DEVELOPMENT_DECISION_SCHEMA": ATTEMPT11_DEVELOPMENT_DECISION_SCHEMA,
    "ATTEMPT09_DEVELOPMENT_RECEIPT_SCHEMA": ATTEMPT11_DEVELOPMENT_RECEIPT_SCHEMA,
    "ATTEMPT09_DEVELOPMENT_GO_FREEZE_SCHEMA": ATTEMPT11_DEVELOPMENT_GO_FREEZE_SCHEMA,
    "DEVELOPMENT_GO_FREEZE_STATUS": ATTEMPT11_DEVELOPMENT_GO_FREEZE_STATUS,
    "SELECTOR_SOURCE_NAME": "select_hu_m43_attempt11_development.py",
}


@contextmanager
def _attempt11_freeze_bindings() -> Iterator[None]:
    with _LOCK:
        prior = {name: getattr(_base, name) for name in _BINDINGS}
        try:
            for name, value in _BINDINGS.items():
                setattr(_base, name, value)
            yield
        finally:
            for name, value in prior.items():
                setattr(_base, name, value)


def build_attempt11_development_go_freeze(
    *,
    run_dir: str | Path,
    received_dir: str | Path,
    decision_path: str | Path,
    decision_receipt_path: str | Path,
) -> dict[str, Any]:
    with _attempt11_freeze_bindings():
        return _base.build_attempt09_development_go_freeze(
            run_dir=run_dir,
            received_dir=received_dir,
            decision_path=decision_path,
            decision_receipt_path=decision_receipt_path,
        )


def freeze_attempt11_development(
    *,
    run_dir: str | Path,
    received_dir: str | Path,
    decision_path: str | Path,
    decision_receipt_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    with _attempt11_freeze_bindings():
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
    payload = freeze_attempt11_development(
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
    "ATTEMPT11_DEVELOPMENT_GO_FREEZE_SCHEMA",
    "build_attempt11_development_go_freeze",
    "freeze_attempt11_development",
]
