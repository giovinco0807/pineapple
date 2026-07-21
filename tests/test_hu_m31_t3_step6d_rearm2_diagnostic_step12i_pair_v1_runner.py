from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_run_scoped_controller_sa_v2
    as sa_v2,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12f_sa_create_poll_v1
    as sa_create_poll,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12h_readonly_retry_v3
    as readonly_retry_v3,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "scripts"
    / "run_hu_m31_t3_step6d_rearm2_diagnostic_step12i_pair_v1.py"
)
IDENTITY_RECORD = {
    "project": "ofc-solver-485418",
    "account_id": "ofc-m31-s2b-a1b2c3d4e5f6",
    "email": (
        "ofc-m31-s2b-a1b2c3d4e5f6@"
        "ofc-solver-485418.iam.gserviceaccount.com"
    ),
    "principal": (
        "serviceAccount:ofc-m31-s2b-a1b2c3d4e5f6@"
        "ofc-solver-485418.iam.gserviceaccount.com"
    ),
    "derived_from_run_tag": "a1b2c3d4e5f6",
    "run_scoped": True,
    "legacy_shared_controller_reused": False,
}


@pytest.fixture(scope="module")
def runner() -> Any:
    name = "step12i_pair_v1_runner_under_test"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_confirmation_token_is_new_and_step12i_specific(runner: Any) -> None:
    assert runner.EXECUTION_CONFIRMATION == (
        "EXECUTE_STEP12I_DIRECT_V2_EXACT_PAIR_ATTEMPT0"
    )
    for older in ("step12d", "step12e", "step12f", "step12g", "step12h"):
        older_path = (
            ROOT
            / "scripts"
            / f"run_hu_m31_t3_step6d_rearm2_diagnostic_{older}_pair_v1.py"
        )
        assert runner.EXECUTION_CONFIRMATION not in older_path.read_text(
            encoding="utf-8"
        )


def test_wrong_or_older_confirmation_fails_before_prepare(
    runner: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def forbidden_prepare() -> None:
        nonlocal called
        called = True
        raise AssertionError("prepare must not run")

    monkeypatch.setattr(runner.step12b, "prepare_run", forbidden_prepare)
    for bad in (
        "wrong",
        "EXECUTE_STEP12H_DIRECT_V2_EXACT_PAIR_ATTEMPT0",
        "EXECUTE_STEP12G_DIRECT_V2_EXACT_PAIR_ATTEMPT0",
    ):
        with pytest.raises(PermissionError, match="confirmation"):
            runner.main(["--execute", "--confirm", bad])
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
            "status": "step12i_offline_fresh_identity_no_cloud_adapter",
            "cloud_adapter_constructed": False,
        },
    )
    assert runner.main([]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["cloud_adapter_constructed"] is False


def test_execute_once_installs_both_hardening_wrappers(
    runner: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = SimpleNamespace(
        deployment_contract={},
        controller_public_key_record={},
        source_plan={},
    )
    identity = sa_v2.validate_controller_identity(IDENTITY_RECORD)
    inner_iam = object()
    inner_sa = SimpleNamespace(identity=identity)
    fresh = {
        "receipt_sha256": "c1" * 32,
        "deployment_contract_sha256": "c2" * 32,
        "instance_names": ["candidate", "reference"],
    }
    trees = {
        label: {} for label in
        ("step12b", "step12c", "step12d", "step12e",
         "step12f", "step12g", "step12h")
    }
    seen: dict[str, Any] = {}
    writes: list[str] = []

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
        lambda _: SimpleNamespace(
            iam_admin=inner_iam,
            controller_sa_admin=inner_sa,
            http=object(),
            user_tokens=object(),
        ),
    )

    def capture_execute(**kwargs: Any) -> dict[str, Any]:
        seen["backend"] = kwargs["backend"]
        return {"receipt_sha256": "c3" * 32, "result_object_count": 2}

    monkeypatch.setattr(runner.step12b, "execute_prepared", capture_execute)
    monkeypatch.setattr(
        runner.step12b,
        "_exclusive_write_json",
        lambda path, _value: writes.append(path.name),
    )

    result = runner.execute_once(prepared=prepared, output_root=tmp_path)
    backend = seen["backend"]
    assert isinstance(
        backend.iam_admin, readonly_retry_v3.ReadOnlyRetryIamAdminV3
    )
    assert isinstance(
        backend.controller_sa_admin,
        sa_create_poll.CreateReadbackPollingControllerAdmin,
    )
    assert result["status"] == "step12i_pair_received_cleaned_and_disjoint"
    assert writes == ["STEP12I_FINAL.json"]


def test_execute_once_registers_all_terminal_roots_immutable(
    runner: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = SimpleNamespace(
        deployment_contract={},
        controller_public_key_record={},
        source_plan={},
    )
    monkeypatch.setattr(
        runner.fresh_identity, "exact_output_root", lambda _: tmp_path
    )
    monkeypatch.setattr(
        runner,
        "_fresh_receipt",
        lambda _prepared, _root: {"receipt_sha256": "d1" * 32},
    )
    monkeypatch.setattr(
        runner.fresh_identity,
        "terminal_tree_snapshot",
        lambda: {
            label: {} for label in
            ("step12b", "step12c", "step12d", "step12e",
             "step12f", "step12g", "step12h")
        },
    )

    def stop_before_backend(_: Any) -> None:
        raise RuntimeError("stop before any cloud construction")

    monkeypatch.setattr(
        runner.step12b, "LiveExecutionBackend", stop_before_backend
    )
    with pytest.raises(RuntimeError, match="stop before"):
        runner.execute_once(prepared=prepared, output_root=tmp_path)
    immutable = runner.step12b._IMMUTABLE_ROOTS
    for name in (
        "TERMINAL_STEP12B_OUTPUT_ROOT",
        "TERMINAL_STEP12H_OUTPUT_ROOT",
    ):
        assert getattr(runner.fresh_identity, name).resolve() in immutable
