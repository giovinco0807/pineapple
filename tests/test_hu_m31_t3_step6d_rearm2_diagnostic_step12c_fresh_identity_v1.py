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
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_phase2_iam_lifecycle_v2
    as phase2_lifecycle,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_token_barrier_v1
    as token_barrier,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12c_fresh_identity_v1
    as subject,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "scripts"
    / "run_hu_m31_t3_step6d_rearm2_diagnostic_step12c_pair_v1.py"
)


@pytest.fixture(scope="module")
def runner() -> Any:
    name = "step12c_pair_v1_runner_identity_test"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _old_json(name: str) -> dict[str, Any]:
    value = json.loads(
        (subject.TERMINAL_OUTPUT_ROOT / name).read_text(encoding="utf-8")
    )
    assert isinstance(value, dict)
    return value


def test_real_prepared_identity_is_fully_disjoint(runner: Any) -> None:
    prepared = runner.step12b.prepare_run(
        now_unix_seconds=1_900_000_000,
        run_nonce="b3" * 32,
    )
    receipt = subject.validate_fresh_identity(
        deployment_contract=prepared.deployment_contract,
        controller_public_key_record=prepared.controller_public_key_record,
        source_plan=prepared.source_plan,
        require_output_absent=False,
    )
    old = _old_json("deployment_contract.json")
    deployment = prepared.deployment_contract
    assert receipt["status"] == subject.STATUS
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
    assert deployment["attempt_index"] == 0
    assert deployment["vm_count"] == 2
    assert deployment_v2.MAX_ATTEMPTS == 1
    assert deployment_v2.ACTUAL_MEMORY_MB == 30_720


def test_terminal_public_key_and_nonce_are_rejected(runner: Any) -> None:
    old_deployment = _old_json("deployment_contract.json")
    old_public = _old_json("controller_public_key.json")
    old_signer = SimpleNamespace(public_record=old_public)
    prepared_key_reuse = runner.step12b.prepare_run(
        now_unix_seconds=1_900_000_000,
        run_nonce="c4" * 32,
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
    with pytest.raises(FileExistsError):
        subject.exact_output_root(subject.EXPECTED_OUTPUT_ROOT)
    with pytest.raises(PermissionError):
        subject.exact_output_root(subject.TERMINAL_OUTPUT_ROOT)
    with pytest.raises(PermissionError):
        subject.exact_output_root(tmp_path / "arbitrary")


def test_terminal_tree_and_closeout_are_frozen() -> None:
    snapshot = subject.terminal_tree_snapshot()
    closeout = _old_json("token_barrier_failure_closeout_receipt.json")
    assert snapshot["file_count"] > 0
    assert (
        closeout["receipt_sha256"]
        == subject.TERMINAL_CLOSEOUT_RECEIPT_SHA256
    )
    assert (
        closeout["deployment_contract_sha256"]
        == subject.TERMINAL_DEPLOYMENT_CONTRACT_SHA256
    )


def test_live_step12c_token_receipt_crosses_corrected_phase2_boundary() -> None:
    root = subject.ARTIFACT_PARENT / "step12c_pair_v1_actual"
    receipt = json.loads(
        (root / "token_barrier_receipt.json").read_text(encoding="utf-8")
    )
    plan = json.loads(
        (root / "phase2_iam_plan.json").read_text(encoding="utf-8")
    )
    outcome = token_barrier.TokenBarrierOutcome(object(), receipt)
    assert (
        phase2_lifecycle._validate_token_barrier(outcome, plan)
        == receipt["receipt_sha256"]
    )

    changed = json.loads(json.dumps(receipt))
    changed["token_creator_revoke_zero_readback"][
        "second_add_performed"
    ] = True
    changed.pop("receipt_sha256")
    changed["receipt_sha256"] = token_barrier.canonical_sha256(changed)
    with pytest.raises(ValueError, match="zero readback"):
        phase2_lifecycle._validate_token_barrier(
            token_barrier.TokenBarrierOutcome(object(), changed), plan
        )
