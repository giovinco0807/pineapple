from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import hu_m31_t3_dataset_source_binding_v1 as subject
from ofc_regular import hu_m31_t3_dataset_supervisor_v1 as supervisor


FRESH_RUN = "regular-hu-m31-t3-fqv1-wsl-20260724-005"
DATASET_RUN = "regular-hu-m31-t3-dataset-v1-20260724-001"


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(subject.canonical_bytes(value))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(base: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(base).as_posix(),
        "sha256": _sha(path),
        "bytes": path.stat().st_size,
    }


def _install_valid_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    controller_fs = (tmp_path / "controller-fs").resolve()
    root = controller_fs / "closeout"
    controller_root = root / "controller"
    smoke_shard = root / "dataset" / "shards" / "train-0000"
    smoke_shard.mkdir(parents=True)
    (smoke_shard / "SHARD_DONE.json").write_bytes(b"{}")

    fresh_plan_core = {
        "schema": subject.quality_bridge.PLAN_SCHEMA,
        "status": "immutable_quality_cloud_plan_ready_not_authorized",
        "run_name": FRESH_RUN,
        "cloud_launch_authorized": False,
        "cloud_execution_started": False,
        "training_eligible": False,
        "current_profile_changed": False,
    }
    fresh_plan = {
        **fresh_plan_core,
        "plan_sha256": subject.quality_bridge.canonical_sha256(fresh_plan_core),
    }
    fresh_plan_path = controller_fs / "quality-gcp-plan.json"
    _write(fresh_plan_path, fresh_plan)

    dataset_plan = {"schema": subject.dataset.DATASET_PLAN_SCHEMA}
    dataset_plan_path = root / "dataset" / "dataset-plan.json"
    _write(dataset_plan_path, dataset_plan)
    smoke = {
        "schema": subject.dataset.SMOKE_GATE_SCHEMA,
        "status": "pass",
        "decision": "open_remaining_8975_paired_fanout",
        "metrics": {
            "paired_hand_count": 25,
            "root_count": 50,
            "seat_counts": {"first": 25, "second": 25},
        },
        "all_gates_passed": True,
        "full_9000_paired_fanout_authorized": True,
        "current_profile_changed": False,
    }
    smoke_path = root / "dataset" / "smoke-gate.json"
    _write(smoke_path, smoke)
    fresh_gate = {
        "schema": subject.quality_gate.GATE_SCHEMA,
        "status": "pass",
        "all_gates_passed": True,
        "data_pilot_25_paired_authorized": True,
        "full_9000_paired_fanout_authorized": False,
        "current_profile_changed": False,
    }
    fresh_gate_path = root / "quality" / "fresh-quality-gate.json"
    _write(fresh_gate_path, fresh_gate)
    final = {
        "schema": subject.quality_bridge.FINAL_SCHEMA,
        "status": "qualified",
        "accepted_job_count": 15,
        "data_pilot_25_paired_authorized": True,
        "full_9000_paired_fanout_authorized": False,
        "current_profile_changed": False,
    }
    final_path = root / "quality" / "final.json"
    _write(final_path, final)
    portable_path = controller_root / "portable_authorization.json"
    _write(portable_path, {"schema": "portable"})

    content_paths = {
        "dataset_plan": dataset_plan_path,
        "fresh_quality_gate": fresh_gate_path,
        "smoke_gate": smoke_path,
        "portable_authorization": portable_path,
    }
    content_sources = [
        {
            "kind": kind,
            "source_path": str(path),
            "sha256": _sha(path),
            "bytes": path.stat().st_size,
        }
        for kind, path in content_paths.items()
    ]
    transport_plan = {
        "schema": subject.transport.TRANSPORT_PLAN_SCHEMA,
        "status": "qualified_ready_cloud_not_started",
        "run_name": DATASET_RUN,
        "plan_sha256": "1" * 64,
        "execution_identity_sha256": "2" * 64,
        "cloud_execution_started": False,
        "quality_and_smoke_source_replayed": True,
        "cloud_launch_authorized": True,
        "current_profile_changed": False,
        "content_sources": content_sources,
        "source_paths": {"smoke_shard_directory": str(smoke_shard)},
        "source_identity": {
            "dataset_plan_sha256": "3" * 64,
            "portable_authorization_sha256": "4" * 64,
        },
    }
    transport_path = controller_root / "transport_plan.json"
    _write(transport_path, transport_plan)
    contract = {
        "schema": subject.controller.CONTROLLER_CONTRACT_SCHEMA,
        "contract_sha256": "5" * 64,
        "run_name": DATASET_RUN,
        "quality_and_smoke_source_replayed": True,
        "full_fanout_authorized_after_smoke_only": True,
        "current_profile_changed": False,
    }
    contract_path = controller_root / "controller_contract.json"
    _write(contract_path, contract)

    bundle_paths = [
        fresh_gate_path,
        final_path,
        dataset_plan_path,
        smoke_path,
        portable_path,
        transport_path,
        contract_path,
    ]
    bundle_records = [_record(root, path) for path in bundle_paths]
    source_records = [_record(controller_fs, fresh_plan_path)]
    manifest = {
        "schema": subject.closeout.FANOUT_BUNDLE_SCHEMA,
        "status": ("portable_359_shard_authorization_ready_cloud_not_started"),
        "required_restore_root": str(controller_fs),
        "restore_contract": "same_persistent_disk_exact_absolute_mount_only",
        "source_paths_all_within_restore_root": True,
        "portable_authorization_sha256": "4" * 64,
        "transport_plan_sha256": "1" * 64,
        "dataset_plan_sha256": "3" * 64,
        "smoke_shard_id": subject.dataset.SMOKE_SHARD_ID,
        "completed_smoke_pair_count": 25,
        "fanout_shard_count": 359,
        "bundle_files": bundle_records,
        "bundle_file_aggregate_sha256": subject.canonical_sha256(bundle_records),
        "persistent_source_closure_files": source_records,
        "persistent_source_closure_sha256": subject.canonical_sha256(source_records),
        "performance_receipt_validation": {"mode": "pinned"},
        "performance_source_paths_dereferenced": False,
        "fresh_quality_worker_vm_reuse_allowed": False,
        "dedicated_controller_context_required": True,
        "controller_cloud_mutation_authorized": False,
        "controller_cleanup_condition": "final-and-export",
        "hidden_information_field_count": 0,
        "opponent_private_discards_used": False,
        "teacher_values_are_realized_match_ev": False,
        "training_eligible": False,
        "current_profile_changed": False,
    }
    manifest_path = root / subject.closeout.FANOUT_MANIFEST_NAME
    _write(manifest_path, manifest)
    ready = {
        "schema": subject.closeout.CLOSEOUT_SCHEMA,
        "status": ("complete_same_linux_ready_359_shard_fanout_not_started"),
        "output_root": str(root),
        "fresh_quality_final_receipt": _record(root, final_path),
        "fresh_quality_gate": _record(root, fresh_gate_path),
        "dataset_smoke_gate": _record(root, smoke_path),
        "portable_authorization": _record(root, portable_path),
        "fanout_bundle_manifest": _record(root, manifest_path),
        "quality_job_count": 15,
        "dataset_smoke_pair_count": 25,
        "fanout_shard_count": 359,
        "performance_receipt_validation_mode": "pinned",
        "performance_source_paths_dereferenced": False,
        "fresh_quality_worker_vm_reuse_allowed": False,
        "dedicated_controller_context": True,
        "cloud_called": False,
        "cloud_execution_started": False,
        "full_fanout_started": False,
        "training_eligible": False,
        "current_profile_changed": False,
    }
    ready_path = root / subject.closeout.READY_NAME
    _write(ready_path, ready)

    monkeypatch.setattr(
        subject.transport,
        "validate_transport_plan",
        lambda value, replay_sources: dict(value),
    )
    monkeypatch.setattr(
        subject.controller,
        "_validate_contract",
        lambda value, transport_plan: dict(value),
    )
    monkeypatch.setattr(
        subject.dataset,
        "validate_dataset_plan",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        subject.dataset,
        "validate_smoke_gate_receipt",
        lambda value, plan, smoke_shard_directory: dict(value),
    )
    monkeypatch.setattr(
        subject.quality_gate,
        "validate_fresh_quality_gate_value",
        lambda value, replay_sources: dict(value),
    )
    monkeypatch.setattr(
        subject.quality_bridge,
        "validate_final_receipt",
        lambda value: dict(value),
    )
    return {
        "closeout_ready_path": ready_path,
        "expected_closeout_file_sha256": _sha(ready_path),
        "expected_fresh_quality_plan_file_sha256": _sha(fresh_plan_path),
        "expected_smoke_gate_file_sha256": _sha(smoke_path),
        "expected_fresh_quality_run_name": FRESH_RUN,
        "expected_dataset_run_name": DATASET_RUN,
        "controller_root": controller_root,
        "ready": ready,
    }


def test_exact_attempt005_closeout_and_smoke_binding_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _install_valid_tree(tmp_path, monkeypatch)
    kwargs = {
        key: value
        for key, value in values.items()
        if key not in {"controller_root", "ready"}
    }
    binding = subject.build_source_binding(**kwargs)
    assert binding["fresh_quality_run_name"] == FRESH_RUN
    assert binding["dataset_run_name"] == DATASET_RUN
    assert binding["dataset_smoke_pair_count"] == 25
    assert binding["dataset_smoke_gate_passed"] is True
    assert binding["cloud_mutated"] is False
    assert binding["current_profile_changed"] is False
    assert (
        subject.validate_source_binding(
            binding,
            expected_dataset_run_name=DATASET_RUN,
            expected_controller_root=values["controller_root"],
        )
        == binding
    )


def test_pending_or_no_go_attempt005_cannot_bind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _install_valid_tree(tmp_path, monkeypatch)
    ready_path = values["closeout_ready_path"]
    pending = dict(values["ready"])
    pending["status"] = "attempt005_fresh_quality_not_closed"
    _write(ready_path, pending)
    values["expected_closeout_file_sha256"] = _sha(ready_path)
    kwargs = {
        key: value
        for key, value in values.items()
        if key not in {"controller_root", "ready"}
    }
    with pytest.raises(PermissionError, match="not production-ready"):
        subject.build_source_binding(**kwargs)


def test_attempt004_identity_cannot_be_substituted_for_attempt005(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _install_valid_tree(tmp_path, monkeypatch)
    values["expected_fresh_quality_run_name"] = (
        "regular-hu-m31-t3-fqv1-wsl-20260724-004"
    )
    kwargs = {
        key: value
        for key, value in values.items()
        if key not in {"controller_root", "ready"}
    }
    with pytest.raises(PermissionError, match="run identity"):
        subject.build_source_binding(**kwargs)


def test_dataset_run_and_controller_identity_are_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _install_valid_tree(tmp_path, monkeypatch)
    kwargs = {
        key: value
        for key, value in values.items()
        if key not in {"controller_root", "ready"}
    }
    binding = subject.build_source_binding(**kwargs)
    with pytest.raises(PermissionError, match="run name mismatch"):
        subject.validate_source_binding(
            binding,
            expected_dataset_run_name=("regular-hu-m31-t3-dataset-v1-20260724-002"),
            expected_controller_root=values["controller_root"],
        )
    with pytest.raises(PermissionError, match="controller mismatch"):
        subject.validate_source_binding(
            binding,
            expected_dataset_run_name=DATASET_RUN,
            expected_controller_root=(tmp_path / "other-controller").resolve(),
        )


def test_closeout_and_smoke_digest_tamper_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _install_valid_tree(tmp_path, monkeypatch)
    values["expected_closeout_file_sha256"] = "a" * 64
    kwargs = {
        key: value
        for key, value in values.items()
        if key not in {"controller_root", "ready"}
    }
    with pytest.raises(ValueError, match="closeout file SHA changed"):
        subject.build_source_binding(**kwargs)

    values = _install_valid_tree(tmp_path / "second", monkeypatch)
    values["expected_smoke_gate_file_sha256"] = "b" * 64
    kwargs = {
        key: value
        for key, value in values.items()
        if key not in {"controller_root", "ready"}
    }
    with pytest.raises(ValueError, match="smoke-gate digest changed"):
        subject.build_source_binding(**kwargs)


def test_supervisor_cli_validates_binding_before_cloud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def reject(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise PermissionError("injected source-binding rejection")

    def run(**_kwargs: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        raise AssertionError("cloud runner must remain closed")

    monkeypatch.setattr(
        supervisor.source_binding, "validate_source_binding_file", reject
    )
    monkeypatch.setattr(supervisor, "run_until_terminal", run)
    result = supervisor.main(
        [
            "run-all",
            "--controller-root",
            str((tmp_path / "controller").resolve()),
            "--supervisor-root",
            str((tmp_path / "supervisor").resolve()),
            "--confirm-run-name",
            DATASET_RUN,
            "--source-binding",
            str((tmp_path / "binding.json").resolve()),
            "--confirm-total-cost-cap-usd",
            "500.000000",
            "--confirm-spot-rate-guard-usd-per-vm-hour",
            "0.500000",
            "--allow-cloud-mutation",
        ]
    )
    assert result == 2
    assert called is False


def test_raw_controller_prepare_wrapper_is_disabled(
    tmp_path: Path,
) -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_hu_m31_t3_dataset_gcp_v1.py"
    )
    spec = importlib.util.spec_from_file_location(
        f"_m31_bound_wrapper_{tmp_path.name}", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.main(["prepare"]) == 2
