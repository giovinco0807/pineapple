from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "scripts"
    / "run_hu_m31_t3_step6d_rearm2_diagnostic_step12d_pair_v1.py"
)
STEP12C_RUNNER_PATH = (
    ROOT
    / "scripts"
    / "run_hu_m31_t3_step6d_rearm2_diagnostic_step12c_pair_v1.py"
)


@pytest.fixture(scope="module")
def runner() -> Any:
    name = "step12d_pair_v1_runner_under_test"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_confirmation_token_is_new_and_step12d_specific(runner: Any) -> None:
    assert runner.EXECUTION_CONFIRMATION == (
        "EXECUTE_STEP12D_DIRECT_V2_EXACT_PAIR_ATTEMPT0"
    )
    step12c_source = STEP12C_RUNNER_PATH.read_text(encoding="utf-8")
    assert runner.EXECUTION_CONFIRMATION not in step12c_source
    assert runner.FINAL_SCHEMA != "hu_m31_t3_step6d_step12c_pair_v1_final"
    assert runner.FAILURE_SCHEMA != (
        "hu_m31_t3_step6d_step12c_pair_v1_failure"
    )


def test_wrong_confirmation_fails_before_prepare(
    runner: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def forbidden_prepare() -> None:
        nonlocal called
        called = True
        raise AssertionError("prepare must not run")

    monkeypatch.setattr(runner.step12b, "prepare_run", forbidden_prepare)
    with pytest.raises(PermissionError, match="confirmation"):
        runner.main(["--execute", "--confirm", "wrong"])
    with pytest.raises(PermissionError, match="confirmation"):
        runner.main(
            [
                "--execute",
                "--confirm",
                "EXECUTE_STEP12C_DIRECT_V2_EXACT_PAIR_ATTEMPT0",
            ]
        )
    assert called is False


def test_dry_run_never_constructs_live_backend(
    runner: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    def forbidden_backend(_: Any) -> None:
        raise AssertionError("dry-run constructed a cloud backend")

    monkeypatch.setattr(
        runner.step12b, "LiveExecutionBackend", forbidden_backend
    )
    monkeypatch.setattr(
        runner.fresh_identity, "exact_output_root", lambda _: tmp_path
    )
    monkeypatch.setattr(runner.step12b, "prepare_run", lambda: object())
    monkeypatch.setattr(
        runner,
        "dry_run_receipt",
        lambda _prepared, output_root: {
            "status": "step12d_offline_fresh_identity_no_cloud_adapter",
            "cloud_adapter_constructed": False,
            "cloud_mutation_performed": False,
            "attempt1_authorized": False,
            "third_vm_authorized": False,
        },
    )
    assert runner.main([]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == (
        "step12d_offline_fresh_identity_no_cloud_adapter"
    )
    assert receipt["cloud_adapter_constructed"] is False
    assert receipt["cloud_mutation_performed"] is False
    assert receipt["attempt1_authorized"] is False
    assert receipt["third_vm_authorized"] is False


def test_execute_once_forwards_exactly_once_and_wraps_success(
    runner: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = SimpleNamespace(
        deployment_contract={},
        controller_public_key_record={},
        source_plan={},
    )
    fresh = {
        "receipt_sha256": "21" * 32,
        "deployment_contract_sha256": "22" * 32,
        "instance_names": ["candidate", "reference"],
    }
    trees = {
        "step12b": {"file_count": 10, "tree_sha256": "23" * 32},
        "step12c": {"file_count": 11, "tree_sha256": "24" * 32},
    }
    calls: list[str] = []
    writes: list[tuple[Path, dict[str, Any]]] = []

    monkeypatch.setattr(
        runner.fresh_identity, "exact_output_root", lambda _: tmp_path
    )
    monkeypatch.setattr(
        runner, "_fresh_receipt", lambda _prepared, _root: fresh
    )
    monkeypatch.setattr(
        runner.fresh_identity,
        "terminal_tree_snapshot",
        lambda: json.loads(json.dumps(trees)),
    )
    monkeypatch.setattr(
        runner.step12b,
        "LiveExecutionBackend",
        lambda _: calls.append("backend") or object(),
    )
    monkeypatch.setattr(
        runner.step12b,
        "execute_prepared",
        lambda **_: calls.append("execute")
        or {"receipt_sha256": "25" * 32, "result_object_count": 88},
    )
    monkeypatch.setattr(
        runner.step12b,
        "_exclusive_write_json",
        lambda path, value: writes.append((path, dict(value))),
    )

    result = runner.execute_once(prepared=prepared, output_root=tmp_path)
    assert calls == ["backend", "execute"]
    assert result["status"] == "step12d_pair_received_cleaned_and_disjoint"
    assert result["result_object_count"] == 88
    assert result["terminal_trees_unchanged"] is True
    assert result["terminal_trees_before"] == trees
    assert [path.name for path, _ in writes] == ["STEP12D_FINAL.json"]
    assert result["attempt1_authorized"] is False
    assert result["third_vm_authorized"] is False


def test_execute_once_registers_both_terminal_roots_immutable(
    runner: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = SimpleNamespace(
        deployment_contract={},
        controller_public_key_record={},
        source_plan={},
    )
    fresh = {"receipt_sha256": "31" * 32}
    monkeypatch.setattr(
        runner.fresh_identity, "exact_output_root", lambda _: tmp_path
    )
    monkeypatch.setattr(
        runner, "_fresh_receipt", lambda _prepared, _root: fresh
    )
    monkeypatch.setattr(
        runner.fresh_identity,
        "terminal_tree_snapshot",
        lambda: {"step12b": {}, "step12c": {}},
    )

    def stop_before_backend(_: Any) -> None:
        raise RuntimeError("stop before any cloud construction")

    monkeypatch.setattr(
        runner.step12b, "LiveExecutionBackend", stop_before_backend
    )
    with pytest.raises(RuntimeError, match="stop before"):
        runner.execute_once(prepared=prepared, output_root=tmp_path)
    immutable = runner.step12b._IMMUTABLE_ROOTS
    assert (
        runner.fresh_identity.TERMINAL_STEP12B_OUTPUT_ROOT.resolve()
        in immutable
    )
    assert (
        runner.fresh_identity.TERMINAL_STEP12C_OUTPUT_ROOT.resolve()
        in immutable
    )
