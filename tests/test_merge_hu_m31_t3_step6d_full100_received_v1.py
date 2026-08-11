from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import hu_m31_t3_step6d_candidate02_full100_plan as full_plan
from ofc_regular import merge_hu_m31_t3_step6d_full100_received_v1 as subject
from ofc_regular import run_hu_m31_t3_step6d_performance_v2 as runner


RUN_NAME = "candidate02-full100-receive-unit-001"


def _receive_fixture(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    receive = (tmp_path / "receive").resolve()
    receive.mkdir()
    (receive / "receive_receipt.json").write_bytes(
        runner.canonical_bytes({"unit": "full100-receipt"})
    )
    paths: dict[str, list[str]] = {"candidate": [], "reference": []}
    for role in runner.SOURCE_ROLES:
        for shard in range(full_plan.SHARD_COUNT_PER_ROLE):
            done = receive / "jobs" / f"{role}-shard-{shard:02d}" / "DONE.json"
            done.parent.mkdir(parents=True)
            done.write_bytes(runner.canonical_bytes({"role": role, "shard": shard}))
            paths[role].append(str(done.resolve()))
    validation = {
        "schema": subject.RECEIVED_DIRECTORY_VALIDATION_SCHEMA,
        "status": "exact_full100_receive_directory_revalidated",
        "run_name": RUN_NAME,
        "receive_receipt_sha256": subject.lifecycle.sha256_file(
            receive / "receive_receipt.json"
        ),
        "run_contract_digest": full_plan.FULL_RUN_CONTRACT_DIGEST,
        "candidate_done_paths": paths["candidate"],
        "reference_done_paths": paths["reference"],
        "paired_hand_count": 100,
        "root_count": 200,
        "source_isolation_validated": True,
        "root_pairing_validated": True,
        "performance_lock_authorized": False,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
    }
    return receive, validation


def _scientific_summary(
    *,
    candidate_done_paths: list[Path],
    reference_done_paths: list[Path],
    all_gates: bool,
    forbidden: bool = False,
) -> dict[str, Any]:
    def rows(paths: list[Path]) -> list[dict[str, Any]]:
        return [{"path": str(Path(path).resolve())} for path in paths]

    return {
        "schema": subject.scientific.MERGE_SCHEMA,
        "status": "pass" if all_gates else "no_go",
        "decision": "unit-scientific-decision",
        "scope": subject.scientific.FULL_SCOPE,
        "candidate_variant": runner.CANDIDATE02_VARIANT,
        "full100_plan": {"unit_plan": True},
        "run_contract_digest": full_plan.FULL_RUN_CONTRACT_DIGEST,
        "paired_hand_count": 100,
        "root_count": 200,
        "source_done_inputs": {
            "candidate": rows(candidate_done_paths),
            "reference": rows(reference_done_paths),
        },
        "all_gates_passed": all_gates,
        "performance_candidate_frozen": all_gates,
        "performance_lock_authorized": all_gates,
        "quality_pilot_authorized": forbidden,
        "artifact_fanout_authorized": False,
        "training_authorized": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "m31_complete": False,
    }


def _install_validators(
    monkeypatch: pytest.MonkeyPatch,
    validation: dict[str, Any],
    *,
    all_gates: bool,
    forbidden: bool = False,
    captures: list[dict[str, Any]] | None = None,
) -> None:
    monkeypatch.setattr(
        subject.lifecycle,
        "validate_received_directory",
        lambda _receive, expected_run_name=None: dict(validation),
    )

    def fake_merge(**kwargs: Any) -> dict[str, Any]:
        if captures is not None:
            captures.append(kwargs)
        return _scientific_summary(
            candidate_done_paths=list(kwargs["candidate_done_paths"]),
            reference_done_paths=list(kwargs["reference_done_paths"]),
            all_gates=all_gates,
            forbidden=forbidden,
        )

    monkeypatch.setattr(subject.scientific, "merge_candidate02_full100", fake_merge)


@pytest.mark.parametrize("all_gates", [True, False])
def test_received_bridge_binds_receipt_run_and_only_opens_lock_on_go(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    all_gates: bool,
) -> None:
    receive, received_validation = _receive_fixture(tmp_path)
    captures: list[dict[str, Any]] = []
    _install_validators(
        monkeypatch,
        received_validation,
        all_gates=all_gates,
        captures=captures,
    )
    summary_path = tmp_path / "merge" / "summary.json"
    validation_path = tmp_path / "merge" / "validation.json"
    summary, validation = subject.merge_and_validate_received_full100(
        receive_dir=receive,
        expected_run_name=RUN_NAME,
        summary_output_path=summary_path,
        validation_output_path=validation_path,
    )

    assert summary["status"] == ("pass" if all_gates else "no_go")
    assert summary["run_name"] == RUN_NAME
    assert summary["receive_dir"] == str(receive)
    assert (
        summary["receive_receipt_sha256"]
        == received_validation["receive_receipt_sha256"]
    )
    assert summary["all_gates_passed"] is all_gates
    assert summary["performance_candidate_frozen"] is all_gates
    assert summary["performance_lock_authorized"] is all_gates
    assert summary["quality_pilot_authorized"] is False
    assert summary["artifact_fanout_authorized"] is False
    assert summary["training_authorized"] is False
    assert summary["current_profile_changed"] is False
    assert validation["source_shard_count"] == 20
    assert validation["receipt_and_sources_recomputed"] is True
    assert all(
        path.is_absolute()
        for call in captures
        for role in ("candidate_done_paths", "reference_done_paths")
        for path in call[role]
    )
    assert (
        subject.validate_received_full100_merge(summary_path=summary_path) == validation
    )
    with pytest.raises(FileExistsError, match="write-once"):
        subject.merge_and_validate_received_full100(
            receive_dir=receive,
            expected_run_name=RUN_NAME,
            summary_output_path=summary_path,
            validation_output_path=tmp_path / "other-validation.json",
        )


def test_received_bridge_rejects_receipt_tamper_after_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receive, received_validation = _receive_fixture(tmp_path)
    _install_validators(monkeypatch, received_validation, all_gates=True)
    summary_path = tmp_path / "summary.json"
    subject.merge_and_validate_received_full100(
        receive_dir=receive,
        expected_run_name=RUN_NAME,
        summary_output_path=summary_path,
        validation_output_path=tmp_path / "validation.json",
    )
    (receive / "receive_receipt.json").write_bytes(
        runner.canonical_bytes({"unit": "tampered"})
    )
    with pytest.raises(ValueError, match="receipt SHA-256 binding changed"):
        subject.validate_received_full100_merge(summary_path=summary_path)


def test_received_bridge_rejects_wrong_run_outside_done_and_downstream_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receive, received_validation = _receive_fixture(tmp_path)

    wrong_run = dict(received_validation)
    wrong_run["run_name"] = "different-full100-run"
    _install_validators(monkeypatch, wrong_run, all_gates=True)
    with pytest.raises(ValueError, match="validation boundary changed"):
        subject.merge_received_full100(receive_dir=receive, expected_run_name=RUN_NAME)

    outside = tmp_path / "outside" / "DONE.json"
    outside.parent.mkdir()
    outside.write_bytes(runner.canonical_bytes({"outside": True}))
    escaped = dict(received_validation)
    escaped["candidate_done_paths"] = list(escaped["candidate_done_paths"])
    escaped["candidate_done_paths"][0] = str(outside.resolve())
    _install_validators(monkeypatch, escaped, all_gates=True)
    with pytest.raises(ValueError, match="escapes receive directory"):
        subject.merge_received_full100(receive_dir=receive, expected_run_name=RUN_NAME)

    _install_validators(
        monkeypatch, received_validation, all_gates=True, forbidden=True
    )
    with pytest.raises(ValueError, match="authorization boundary changed"):
        subject.merge_received_full100(receive_dir=receive, expected_run_name=RUN_NAME)


def test_received_bridge_rejects_outputs_inside_receive_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receive, received_validation = _receive_fixture(tmp_path)
    _install_validators(monkeypatch, received_validation, all_gates=True)
    with pytest.raises(ValueError, match="outside the receive tree"):
        subject.merge_and_validate_received_full100(
            receive_dir=receive,
            expected_run_name=RUN_NAME,
            summary_output_path=receive / "merge" / "summary.json",
            validation_output_path=tmp_path / "validation.json",
        )
