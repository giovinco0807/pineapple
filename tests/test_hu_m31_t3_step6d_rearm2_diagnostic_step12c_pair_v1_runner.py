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
    / "run_hu_m31_t3_step6d_rearm2_diagnostic_step12c_pair_v1.py"
)


@pytest.fixture(scope="module")
def runner() -> Any:
    name = "step12c_pair_v1_runner_under_test"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


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
    assert called is False


def test_dry_run_never_constructs_live_backend(
    runner: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    def forbidden_backend(_: Any) -> None:
        raise AssertionError("dry-run constructed a cloud backend")

    monkeypatch.setattr(runner.step12b, "LiveExecutionBackend", forbidden_backend)
    monkeypatch.setattr(
        runner.fresh_identity, "exact_output_root", lambda _: tmp_path
    )
    monkeypatch.setattr(runner.step12b, "prepare_run", lambda: object())
    monkeypatch.setattr(
        runner,
        "dry_run_receipt",
        lambda _prepared, output_root: {
            "status": "step12c_offline_fresh_identity_no_cloud_adapter",
            "cloud_adapter_constructed": False,
            "cloud_mutation_performed": False,
            "attempt1_authorized": False,
            "third_vm_authorized": False,
        },
    )
    assert runner.main([]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == (
        "step12c_offline_fresh_identity_no_cloud_adapter"
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
        "receipt_sha256": "11" * 32,
        "deployment_contract_sha256": "12" * 32,
        "instance_names": ["candidate", "reference"],
    }
    tree = {"file_count": 10, "tree_sha256": "13" * 32}
    calls: list[str] = []
    writes: list[tuple[Path, dict[str, Any]]] = []

    monkeypatch.setattr(
        runner.fresh_identity, "exact_output_root", lambda _: tmp_path
    )
    monkeypatch.setattr(
        runner, "_fresh_receipt", lambda _prepared, _root: fresh
    )
    monkeypatch.setattr(
        runner.fresh_identity, "terminal_tree_snapshot", lambda: dict(tree)
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
        or {"receipt_sha256": "14" * 32, "result_object_count": 88},
    )
    monkeypatch.setattr(
        runner.step12b,
        "_exclusive_write_json",
        lambda path, value: writes.append((path, dict(value))),
    )

    result = runner.execute_once(prepared=prepared, output_root=tmp_path)
    assert calls == ["backend", "execute"]
    assert result["status"] == "step12c_pair_received_cleaned_and_disjoint"
    assert result["result_object_count"] == 88
    assert result["terminal_tree_unchanged"] is True
    assert [path.name for path, _ in writes] == ["STEP12C_FINAL.json"]
    assert result["attempt1_authorized"] is False
    assert result["third_vm_authorized"] is False
