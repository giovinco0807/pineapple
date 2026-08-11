#!/usr/bin/env python3
"""Write-once GET-only closeout for the terminal Step12c failure."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12c_failure_closeout_v1
    as closeout,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / "step12c_pair_v1_actual"
)
RECEIPT_PATH = OUTPUT_ROOT / "phase2_schema_failure_closeout_receipt.json"
POLICY_REGISTRY_PATH = REPO_ROOT / "src" / "ofc_regular" / "ai_profiles.py"
OLD_VERIFIER_PATH = (
    REPO_ROOT
    / "scripts"
    / "verify_hu_m31_t3_step6d_rearm2_diagnostic_step12b_failure_closeout_v1.py"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object changed: {path}")
    return value


def _load_old_observer_module() -> Any:
    name = "_hu_m31_step12b_read_only_closeout_observer"
    spec = importlib.util.spec_from_file_location(name, OLD_VERIFIER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("read-only closeout observer cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def build_live_receipt() -> dict[str, Any]:
    observer_module = _load_old_observer_module()
    deployment = _read_json(OUTPUT_ROOT / "deployment_contract.json")
    plan = _read_json(OUTPUT_ROOT / "phase2_iam_plan.json")
    source = _read_json(
        OUTPUT_ROOT / "bootstrap_source_provision_receipt.json"
    )
    observer = observer_module.LiveReadOnlyCloseoutObserver(
        deployment=deployment,
        phase2_plan=plan,
        source_receipt=source,
    )
    return closeout.build_closeout_receipt(
        output_root=OUTPUT_ROOT,
        observer=observer,
        policy_registry_path=POLICY_REGISTRY_PATH,
    )


def write_once_or_validate_identical(
    path: Path, value: Mapping[str, Any]
) -> dict[str, Any]:
    checked = closeout.validate_closeout_receipt(value)
    raw = (
        json.dumps(
            checked,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    if path.exists() or path.is_symlink():
        if not path.is_file() or path.read_bytes() != raw:
            raise FileExistsError("different Step12c closeout receipt exists")
        return closeout.validate_closeout_receipt(_read_json(path))
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return checked


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise ValueError("Step12c closeout accepts no arguments")
    receipt = write_once_or_validate_identical(
        RECEIPT_PATH, build_live_receipt()
    )
    sys.stdout.write(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt_sha256": receipt["receipt_sha256"],
                "receipt_path": str(RECEIPT_PATH.resolve()),
                "all_closeout_gates_passed": receipt[
                    "all_closeout_gates_passed"
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
