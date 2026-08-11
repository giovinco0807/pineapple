from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12l_fresh_identity_v1
    as step12l_identity,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12m_fresh_identity_v1
    as subject,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "scripts"
    / "run_hu_m31_t3_step6d_rearm2_diagnostic_step12m_pair_v1.py"
)


@pytest.fixture(scope="module")
def runner() -> Any:
    name = "step12m_pair_v1_runner_identity_test"
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


_ALL_TERMINALS = (
    "TERMINAL_STEP12B_OUTPUT_ROOT",
    "TERMINAL_STEP12C_OUTPUT_ROOT",
    "TERMINAL_STEP12D_OUTPUT_ROOT",
    "TERMINAL_STEP12E_OUTPUT_ROOT",
    "TERMINAL_STEP12F_OUTPUT_ROOT",
    "TERMINAL_STEP12G_OUTPUT_ROOT",
    "TERMINAL_STEP12H_OUTPUT_ROOT",
    "TERMINAL_STEP12I_OUTPUT_ROOT",
    "TERMINAL_STEP12J_OUTPUT_ROOT",
    "TERMINAL_STEP12K_OUTPUT_ROOT",
    "TERMINAL_STEP12L_OUTPUT_ROOT",
)


def test_real_prepared_identity_is_disjoint_from_all_terminals(
    runner: Any,
) -> None:
    prepared = runner.step12b.prepare_run(
        now_unix_seconds=1_900_000_000,
        run_nonce="e5" * 32,
    )
    receipt = subject.validate_fresh_identity(
        deployment_contract=prepared.deployment_contract,
        controller_public_key_record=prepared.controller_public_key_record,
        source_plan=prepared.source_plan,
        require_output_absent=False,
    )
    assert receipt["status"] == subject.STATUS
    assert receipt["phase2_failure_instrumentation"] == (
        "underlying_step11_error_code_status_operation_persisted"
    )
    for name in _ALL_TERMINALS:
        old = _terminal_json(
            getattr(subject, name), "deployment_contract.json"
        )
        assert receipt["deployment_contract_sha256"] != old[
            "deployment_contract_sha256"
        ]
        assert set(receipt["instance_names"]).isdisjoint(
            row["instance_name"] for row in old["instances"]
        )


@pytest.mark.parametrize("terminal_root_name", _ALL_TERMINALS)
def test_terminal_public_key_and_nonce_are_rejected(
    runner: Any, terminal_root_name: str
) -> None:
    terminal_root = getattr(subject, terminal_root_name)
    old_deployment = _terminal_json(terminal_root, "deployment_contract.json")
    old_public = _terminal_json(terminal_root, "controller_public_key.json")
    old_signer = SimpleNamespace(public_record=old_public)
    prepared_key_reuse = runner.step12b.prepare_run(
        now_unix_seconds=1_900_000_000,
        run_nonce="f6" * 32,
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
    for name in _ALL_TERMINALS:
        terminal = getattr(subject, name)
        with pytest.raises(PermissionError):
            subject.exact_output_root(terminal)
    with pytest.raises(PermissionError):
        subject.exact_output_root(tmp_path / "arbitrary")
    with pytest.raises(PermissionError):
        subject.exact_output_root(step12l_identity.EXPECTED_OUTPUT_ROOT)


def test_step12m_expected_root_is_distinct_terminal_identity() -> None:
    assert subject.EXPECTED_OUTPUT_ROOT.name == "step12m_pair_v1_actual"
    assert subject.EXPECTED_OUTPUT_ROOT != step12l_identity.EXPECTED_OUTPUT_ROOT


def test_prelaunch_freshness_is_checked_on_an_unconsumed_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "step12m_pair_v1_actual"
    monkeypatch.setattr(subject, "EXPECTED_OUTPUT_ROOT", root)
    assert subject.exact_output_root(root) == root.resolve()
    root.mkdir()
    with pytest.raises(FileExistsError, match="not fresh"):
        subject.exact_output_root(root)


def test_terminal_trees_and_closeouts_are_frozen() -> None:
    snapshot = subject.terminal_tree_snapshot()
    for label in (
        "step12b", "step12c", "step12d", "step12e", "step12f",
        "step12g", "step12h", "step12i", "step12j", "step12k", "step12l",
    ):
        assert snapshot[label]["file_count"] > 0

    step12l_closeout = _terminal_json(
        subject.TERMINAL_STEP12L_OUTPUT_ROOT,
        "external_preflight_token_field_bug_closeout_receipt.json",
    )
    assert (
        step12l_closeout["receipt_sha256"]
        == subject.TERMINAL_STEP12L_CLOSEOUT_RECEIPT_SHA256
    )
    assert step12l_closeout["step12l_identity_terminal"] is True
    step12l_deployment = _terminal_json(
        subject.TERMINAL_STEP12L_OUTPUT_ROOT, "deployment_contract.json"
    )
    assert (
        step12l_deployment["deployment_contract_sha256"]
        == subject.TERMINAL_STEP12L_DEPLOYMENT_CONTRACT_SHA256
    )


def test_tampered_step12l_closeout_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_read = subject._read_json

    def tampered(path: Path) -> dict[str, Any]:
        value = real_read(path)
        if path.name == "external_preflight_token_field_bug_closeout_receipt.json":
            value = dict(value)
            value["step12l_identity_terminal"] = False
        return value

    monkeypatch.setattr(subject, "_read_json", tampered)
    with pytest.raises(ValueError, match="Step12l closeout"):
        subject._validate_terminal_closeouts()
