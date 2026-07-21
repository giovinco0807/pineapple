from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_deployment_contract_v2
    as deployment_v2,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12c_fresh_identity_v1
    as step12c_identity,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12d_fresh_identity_v1
    as subject,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "scripts"
    / "run_hu_m31_t3_step6d_rearm2_diagnostic_step12d_pair_v1.py"
)


@pytest.fixture(scope="module")
def runner() -> Any:
    name = "step12d_pair_v1_runner_identity_test"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _terminal_json(root: Path, name: str) -> dict[str, Any]:
    value = json.loads((root / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_real_prepared_identity_is_disjoint_from_both_terminals(
    runner: Any,
) -> None:
    prepared = runner.step12b.prepare_run(
        now_unix_seconds=1_900_000_000,
        run_nonce="d5" * 32,
    )
    receipt = subject.validate_fresh_identity(
        deployment_contract=prepared.deployment_contract,
        controller_public_key_record=prepared.controller_public_key_record,
        source_plan=prepared.source_plan,
        require_output_absent=False,
    )
    assert receipt["status"] == subject.STATUS
    for root in (
        subject.TERMINAL_STEP12B_OUTPUT_ROOT,
        subject.TERMINAL_STEP12C_OUTPUT_ROOT,
    ):
        old = _terminal_json(root, "deployment_contract.json")
        assert receipt["deployment_contract_sha256"] != old[
            "deployment_contract_sha256"
        ]
        assert receipt["run_identity_sha256"] != old["run_identity_sha256"]
        assert receipt["direct_stage_identity_sha256"] != old[
            "direct_stage_identity_sha256"
        ]
        assert set(receipt["instance_names"]).isdisjoint(
            row["instance_name"] for row in old["instances"]
        )
    assert prepared.deployment_contract["attempt_index"] == 0
    assert prepared.deployment_contract["vm_count"] == 2
    assert deployment_v2.MAX_ATTEMPTS == 1
    assert deployment_v2.ACTUAL_MEMORY_MB == 30_720


@pytest.mark.parametrize(
    "terminal_root_name",
    ["TERMINAL_STEP12B_OUTPUT_ROOT", "TERMINAL_STEP12C_OUTPUT_ROOT"],
)
def test_terminal_public_key_and_nonce_are_rejected(
    runner: Any, terminal_root_name: str
) -> None:
    terminal_root = getattr(subject, terminal_root_name)
    old_deployment = _terminal_json(terminal_root, "deployment_contract.json")
    old_public = _terminal_json(terminal_root, "controller_public_key.json")
    old_signer = SimpleNamespace(public_record=old_public)
    prepared_key_reuse = runner.step12b.prepare_run(
        now_unix_seconds=1_900_000_000,
        run_nonce="e6" * 32,
        signer=old_signer,
    )
    with pytest.raises(ValueError, match="controller_key_id"):
        subject.validate_fresh_identity(
            deployment_contract=prepared_key_reuse.deployment_contract,
            controller_public_key_record=(
                prepared_key_reuse.controller_public_key_record
            ),
            source_plan=prepared_key_reuse.source_plan,
            require_output_absent=False,
        )

    prepared_nonce_reuse = runner.step12b.prepare_run(
        now_unix_seconds=1_900_000_000,
        run_nonce=old_deployment["run_nonce"],
    )
    with pytest.raises(ValueError, match="run_nonce"):
        subject.validate_fresh_identity(
            deployment_contract=prepared_nonce_reuse.deployment_contract,
            controller_public_key_record=(
                prepared_nonce_reuse.controller_public_key_record
            ),
            source_plan=prepared_nonce_reuse.source_plan,
            require_output_absent=False,
        )


def test_only_exact_fresh_output_root_is_accepted(tmp_path: Path) -> None:
    assert subject.exact_output_path(subject.EXPECTED_OUTPUT_ROOT) == (
        subject.EXPECTED_OUTPUT_ROOT.resolve()
    )
    with pytest.raises(PermissionError):
        subject.exact_output_root(subject.TERMINAL_STEP12B_OUTPUT_ROOT)
    with pytest.raises(PermissionError):
        subject.exact_output_root(subject.TERMINAL_STEP12C_OUTPUT_ROOT)
    with pytest.raises(PermissionError):
        subject.exact_output_root(
            subject.TERMINAL_STEP12C_OUTPUT_ROOT / "nested"
        )
    with pytest.raises(PermissionError):
        subject.exact_output_root(tmp_path / "arbitrary")
    with pytest.raises(PermissionError):
        subject.exact_output_root(
            step12c_identity.EXPECTED_OUTPUT_ROOT
        )


def test_step12d_expected_root_is_new_and_absent() -> None:
    assert subject.EXPECTED_OUTPUT_ROOT.name == "step12d_pair_v1_actual"
    assert subject.EXPECTED_OUTPUT_ROOT != (
        step12c_identity.EXPECTED_OUTPUT_ROOT
    )
    assert subject.EXPECTED_OUTPUT_ROOT != (
        step12c_identity.TERMINAL_OUTPUT_ROOT
    )
    assert not subject.EXPECTED_OUTPUT_ROOT.exists()


def test_terminal_trees_and_closeouts_are_frozen() -> None:
    snapshot = subject.terminal_tree_snapshot()
    assert snapshot["step12b"]["file_count"] > 0
    assert snapshot["step12c"]["file_count"] > 0

    step12b_closeout = _terminal_json(
        subject.TERMINAL_STEP12B_OUTPUT_ROOT,
        "token_barrier_failure_closeout_receipt.json",
    )
    assert (
        step12b_closeout["receipt_sha256"]
        == step12c_identity.TERMINAL_CLOSEOUT_RECEIPT_SHA256
    )
    step12c_closeout = _terminal_json(
        subject.TERMINAL_STEP12C_OUTPUT_ROOT,
        "phase2_schema_failure_closeout_receipt.json",
    )
    assert (
        step12c_closeout["receipt_sha256"]
        == subject.TERMINAL_STEP12C_CLOSEOUT_RECEIPT_SHA256
    )
    assert (
        step12c_closeout["deployment_contract_sha256"]
        == subject.TERMINAL_STEP12C_DEPLOYMENT_CONTRACT_SHA256
    )
    step12c_deployment = _terminal_json(
        subject.TERMINAL_STEP12C_OUTPUT_ROOT, "deployment_contract.json"
    )
    assert (
        step12c_deployment["deployment_contract_sha256"]
        == subject.TERMINAL_STEP12C_DEPLOYMENT_CONTRACT_SHA256
    )


def test_tampered_step12c_closeout_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_read = subject._read_json

    def tampered(path: Path) -> dict[str, Any]:
        value = real_read(path)
        if path.name == "phase2_schema_failure_closeout_receipt.json":
            value = dict(value)
            value["all_closeout_gates_passed"] = False
        return value

    monkeypatch.setattr(subject, "_read_json", tampered)
    with pytest.raises(ValueError, match="Step12c closeout"):
        subject._validate_terminal_closeouts()
