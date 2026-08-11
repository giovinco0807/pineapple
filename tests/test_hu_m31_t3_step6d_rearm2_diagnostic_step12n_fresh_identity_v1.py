from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12n_fresh_identity_v1
    as subject,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "scripts"
    / "run_hu_m31_t3_step6d_rearm2_diagnostic_step12n_pair_v1.py"
)


@pytest.fixture(scope="module")
def runner() -> Any:
    name = "step12n_pair_v1_runner_identity_test"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_terminal_step12m_is_frozen_and_clean() -> None:
    receipt = subject.inspect_terminal_step12m()
    assert receipt["closeout_receipt_sha256"] == (
        subject.TERMINAL_STEP12M_CLOSEOUT_RECEIPT_SHA256
    )
    assert receipt["tree"] == {
        "file_count": subject.TERMINAL_STEP12M_TREE_FILE_COUNT,
        "tree_sha256": subject.TERMINAL_STEP12M_TREE_SHA256,
    }
    assert receipt["current_profile_changed"] is False


def test_step12n_path_identity_is_distinct_without_prelaunch_assumption() -> None:
    assert subject.EXPECTED_OUTPUT_ROOT.name == "step12n_pair_v1_actual"
    assert subject.EXPECTED_OUTPUT_ROOT != subject.TERMINAL_STEP12M_OUTPUT_ROOT
    assert subject.exact_output_path(subject.EXPECTED_OUTPUT_ROOT) == (
        subject.EXPECTED_OUTPUT_ROOT.resolve()
    )
    with pytest.raises(PermissionError):
        subject.exact_output_path(subject.TERMINAL_STEP12M_OUTPUT_ROOT)


def test_prelaunch_freshness_state_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "step12n_pair_v1_actual"
    monkeypatch.setattr(subject, "EXPECTED_OUTPUT_ROOT", root)
    assert subject.validate_prelaunch_freshness(root) == root.resolve()
    root.mkdir()
    with pytest.raises(FileExistsError, match="not fresh"):
        subject.validate_prelaunch_freshness(root)


def test_real_prepared_identity_is_disjoint_and_diagnostic(
    runner: Any,
) -> None:
    prepared = runner.step12b.prepare_run(
        now_unix_seconds=1_900_000_000,
        run_nonce="9a" * 32,
    )
    receipt = subject.validate_fresh_identity(
        deployment_contract=prepared.deployment_contract,
        controller_public_key_record=prepared.controller_public_key_record,
        source_plan=prepared.source_plan,
        require_output_absent=False,
    )
    assert receipt["status"] == subject.STATUS
    assert receipt["worker_diagnostic_contract"] == {
        "bounded_host_prerequisite_install": True,
        "python3_venv_required": True,
        "sanitized_serial_failure_marker": True,
        "serial_contents_persisted": False,
        "failure_hold_seconds": 600,
        "done_readback_before_success_self_delete": True,
    }
    assert receipt["attempt1_authorized"] is False
    assert receipt["third_vm_authorized"] is False
    assert receipt["automatic_retry_authorized"] is False


def test_dry_run_uses_unconsumed_temp_root(
    runner: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "step12n_pair_v1_actual"
    monkeypatch.setattr(subject, "EXPECTED_OUTPUT_ROOT", root)
    prepared = runner.step12b.prepare_run(
        now_unix_seconds=1_900_000_000,
        run_nonce="ab" * 32,
    )
    receipt = runner.dry_run_receipt(prepared, output_root=root)
    assert receipt["status"] == (
        "step12n_offline_fresh_identity_no_cloud_adapter"
    )
    assert receipt["cloud_adapter_constructed"] is False
    assert receipt["cloud_mutation_performed"] is False
    assert not root.exists()


def test_wrong_confirmation_fails_before_prepare(
    runner: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def forbidden_prepare() -> None:
        nonlocal called
        called = True
        raise AssertionError("prepare_run must not be called")

    monkeypatch.setattr(runner.step12b, "prepare_run", forbidden_prepare)
    with pytest.raises(PermissionError, match="confirmation is missing"):
        runner.main(["--execute", "--confirm", "wrong"])
    assert called is False
