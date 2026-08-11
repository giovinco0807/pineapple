from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ofc_regular import hu_m31_t3_dataset_launch_preflight_v1 as subject
from ofc_regular import hu_m31_t3_dataset_supervisor_v1 as supervisor
from ofc_regular import hu_m31_t3_dataset_transport_selection_v1 as selection


RUN_NAME = "regular-hu-m31-t3-dataset-v1-20260724-001"


def _selection(tmp_path: Path) -> Path:
    evidence = tmp_path / "quota-evidence.json"
    evidence.write_bytes(subject.canonical_bytes({"immutable": True}))
    source = supervisor.build_source_receipt()
    hashes = {
        row["relative_path"]: row["observed_sha256"]
        for row in source["files"]
        if row["relative_path"]
        != "src/ofc_regular/hu_m31_t3_dataset_source_binding_v1.py"
    }
    receipt = selection.build_selection_receipt(
        c4_limit_vcpus=128,
        c4_usage_vcpus=0,
        spot_limit_vcpus=468,
        spot_usage_vcpus=0,
        global_limit_vcpus=512,
        global_usage_vcpus=0,
        observed_worker_service_account_count=8,
        evidence_source=str(evidence.resolve()),
        evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
        source_hashes=hashes,
    )
    path = tmp_path / "selection.json"
    path.write_bytes(subject.canonical_bytes(receipt))
    return path


def _arguments(tmp_path: Path) -> dict[str, object]:
    return {
        "dataset_run_name": RUN_NAME,
        "selection_receipt_path": _selection(tmp_path),
        "planned_source_binding_path": tmp_path / "binding.json",
        "planned_controller_root": tmp_path / "controller",
        "planned_supervisor_root": tmp_path / "supervisor",
    }


def test_pending_preflight_is_create_only_and_has_exact_run_next(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    receipt = subject.build_launch_preflight(**arguments)
    assert receipt["status"] == subject.PENDING_STATUS
    assert receipt["blocking_reasons"] == [
        "fresh_quality_closeout_and_25_pair_smoke_source_binding_missing"
    ]
    assert receipt["ready_to_run_next"] is False
    assert receipt["scientific_boundary"]["paired_hand_count"] == 9_000
    assert receipt["scientific_boundary"]["cloud_shard_count"] == 359
    assert receipt["transport_boundary"]["maximum_concurrent_vms"] == 8
    assert receipt["transport_boundary"]["wave_count"] == 45
    assert receipt["cost_boundary"]["total_cloud_cost_cap_micro_usd"] == (500_000_000)
    assert receipt["storage_boundary"] == {
        "production_temp_staging_source_replay_filesystem": (
            "native_ext4_under_/home/wner"
        ),
        "mnt_d_allowed_usage": (
            "final_receipts_and_final_artifacts_create_only_only"
        ),
        "production_temp_on_mnt_d_allowed": False,
        "production_staging_on_mnt_d_allowed": False,
        "production_source_replay_on_mnt_d_allowed": False,
        "wsl_vhd_physical_backing": "D:",
    }
    assert receipt["run_next_argv"][-1] == "--allow-cloud-mutation"
    assert receipt["cloud_opened"] is False
    assert not Path(arguments["planned_controller_root"]).exists()
    assert not Path(arguments["planned_supervisor_root"]).exists()
    assert not Path(arguments["planned_source_binding_path"]).exists()


def test_ready_preflight_replays_binding_and_exact_initial_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _arguments(tmp_path)
    binding = Path(arguments["planned_source_binding_path"])
    binding.write_bytes(b"placeholder")
    controller_root = Path(arguments["planned_controller_root"])
    controller_root.mkdir()
    expected_binding = {
        "binding_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        subject.source_binding,
        "validate_source_binding_file",
        lambda path, **kwargs: (
            expected_binding
            if Path(path) == binding
            and kwargs["expected_dataset_run_name"] == RUN_NAME
            and Path(kwargs["expected_controller_root"]) == controller_root
            else (_ for _ in ()).throw(AssertionError("binding mismatch"))
        ),
    )
    monkeypatch.setattr(
        subject.controller,
        "controller_status",
        lambda path: {
            "resume_status": "wave_ready",
            "accepted_lifecycle_count": 0,
            "complete_shard_count": 0,
            "exhausted_shard_count": 0,
            "selected_count": 8,
        },
    )
    receipt = subject.build_launch_preflight(**arguments)
    assert receipt["status"] == subject.READY_STATUS
    assert receipt["ready_to_run_next"] is True
    assert receipt["blocking_reasons"] == []
    assert receipt["source_binding_sha256"] == "b" * 64
    argv = receipt["run_next_argv"]
    assert argv[2:4] == [
        "run-next",
        "--controller-root",
    ]
    assert "--confirm-total-cost-cap-usd" in argv
    assert "500.000000" in argv
    assert "--confirm-spot-rate-guard-usd-per-vm-hour" in argv
    assert "0.500000" in argv


def test_preflight_output_is_absolute_write_once(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    with pytest.raises(ValueError, match="must be absolute"):
        subject.write_launch_preflight(
            output_path="relative-preflight.json",
            **arguments,
        )
    output = tmp_path / "preflight.json"
    first = subject.write_launch_preflight(
        output_path=output,
        **arguments,
    )
    assert subject.write_launch_preflight(output_path=output, **arguments) == first
    output.write_bytes(output.read_bytes() + b" ")
    with pytest.raises(FileExistsError, match="conflicts"):
        subject.write_launch_preflight(output_path=output, **arguments)
