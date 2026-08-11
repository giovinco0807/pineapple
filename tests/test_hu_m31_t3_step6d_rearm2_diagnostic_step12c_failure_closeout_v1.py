from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12c_failure_closeout_v1
    as subject,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (
    ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / "step12c_pair_v1_actual"
)
RECEIPT_PATH = OUTPUT_ROOT / "phase2_schema_failure_closeout_receipt.json"
SCRIPT_PATH = (
    ROOT
    / "scripts"
    / "verify_hu_m31_t3_step6d_rearm2_diagnostic_step12c_failure_closeout_v1.py"
)


def _receipt() -> dict[str, Any]:
    value = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_actual_closeout_proves_exact_zero_boundary() -> None:
    receipt = subject.validate_closeout_receipt(_receipt())
    assert receipt["receipt_sha256"] == (
        "31e675593ca168fc899a6dc34083a6f54c8258aabbb62dcd71906d56d77161ef"
    )
    assert receipt["all_closeout_gates_passed"] is True
    assert receipt["phase2_install_receipt_present"] is False
    assert len(receipt["phase2_condition_title_readbacks"]) == 8
    assert all(
        row["member_occurrences"] == 0
        for row in receipt["phase2_condition_title_readbacks"]
    )
    assert receipt["instance_final_statuses"] == [404, 404]
    assert receipt["disk_final_statuses"] == [404, 404]
    assert receipt["controller_service_account_absent"] is True
    assert receipt["direct_v2_result_object_count"] == 0
    assert receipt["bootstrap_source_object_count"] == 3
    assert receipt["token_creator_zero_observed"] is True
    assert receipt["token_creator_second_add_performed"] is False
    assert receipt["controller_token_reminted"] is False
    assert receipt["current_profile_changed"] is False


def test_resealed_closeout_boundary_tamper_fails() -> None:
    changed = _receipt()
    changed["all_phase2_condition_title_bindings_zero"] = False
    changed.pop("receipt_sha256")
    changed["receipt_sha256"] = subject.canonical_sha256(changed)
    with pytest.raises(ValueError, match="boundary"):
        subject.validate_closeout_receipt(changed)


def test_write_once_accepts_only_identical_existing_receipt() -> None:
    name = "step12c_failure_closeout_script_under_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        assert (
            module.write_once_or_validate_identical(
                RECEIPT_PATH, _receipt()
            )["receipt_sha256"]
            == _receipt()["receipt_sha256"]
        )
        changed = _receipt()
        changed["read_only_operations"].append("extra_read_only_probe")
        changed["read_only_operation_count"] += 1
        changed.pop("receipt_sha256")
        changed["receipt_sha256"] = subject.canonical_sha256(changed)
        with pytest.raises(FileExistsError):
            module.write_once_or_validate_identical(RECEIPT_PATH, changed)
    finally:
        sys.modules.pop(name, None)
