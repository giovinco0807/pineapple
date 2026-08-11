from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import hu_m31_t3_dataset_executor_v1 as executor
from ofc_regular import hu_m31_t3_dataset_portable_worker_v1 as portable
from ofc_regular import hu_m31_t3_same_linux_closeout_v1 as subject


def _json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ),
        encoding="ascii",
    )


def test_cross_contract_readers_accept_only_the_two_exact_canonical_forms(
    tmp_path: Path,
) -> None:
    gate = {"schema": "fresh-quality-gate"}
    fresh_raw = subject.quality_gate.canonical_bytes(gate)
    fresh_path = tmp_path / "fresh.json"
    fresh_path.write_bytes(fresh_raw)
    assert executor._read_quality_gate(fresh_path) == gate
    fresh_path.write_bytes(fresh_raw + b"\n")
    assert executor._read_quality_gate(fresh_path) == gate
    fresh_path.write_bytes(fresh_raw + b"\n\n")
    with pytest.raises(ValueError, match="not canonical JSON"):
        executor._read_quality_gate(fresh_path)

    authorization = {"schema": "portable"}
    dataset_raw = portable.canonical_bytes(authorization)
    portable_path = tmp_path / "portable.json"
    portable_path.write_bytes(dataset_raw)
    assert portable._read_canonical(portable_path, "portable") == authorization
    portable_path.write_bytes(dataset_raw.removesuffix(b"\n"))
    assert portable._read_canonical(portable_path, "portable") == authorization
    portable_path.write_bytes(dataset_raw + b"\n")
    with pytest.raises(ValueError, match="not canonical JSON"):
        portable._read_canonical(portable_path, "portable")


def test_real_fresh_canonical_gate_reaches_dataset_source_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = {
        "schema": executor.quality_gate.GATE_SCHEMA,
        "status": "pass",
        "decision": "fresh_quality_pass_open_25_paired_data_shard_only",
        "all_gates_passed": True,
        "quality_pilot_passed": True,
        "data_pilot_25_paired_authorized": True,
        "full_9000_paired_fanout_authorized": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
        "teacher_values_are_realized_match_ev": False,
        "merge_sha256": "a" * 64,
    }
    gate_path = tmp_path / "fresh-quality-gate.json"
    gate_path.write_bytes(executor.quality_gate.canonical_bytes(gate))
    replayed: list[bool] = []

    def validate(
        value: dict[str, Any], *, replay_sources: bool
    ) -> dict[str, Any]:
        replayed.append(replay_sources)
        assert value == gate
        return dict(value)

    monkeypatch.setattr(
        executor.quality_gate,
        "validate_fresh_quality_gate_value",
        validate,
    )
    observed, file_sha = executor._gate_value(gate_path)
    assert observed == gate
    assert file_sha == hashlib.sha256(gate_path.read_bytes()).hexdigest()
    assert replayed == [True]


def _install_closeout_fakes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    controller_fs = (tmp_path / "controller-fs").resolve()
    controller_fs.mkdir()
    repository = controller_fs / "repo"
    repository.mkdir()
    staging = controller_fs / "fresh-staging"
    staging.mkdir()
    accepted = controller_fs / "accepted"
    accepted.mkdir()
    for index in range(15):
        _json(
            accepted / f"quality-job-{index:02d}.json",
            {"status": "accepted", "index": index},
        )
    provider = controller_fs / "provider.json"
    _json(provider, {"schema": "provider"})
    gcp_plan_path = controller_fs / "gcp-plan.json"
    ledger_path = controller_fs / "ledger.json"

    artifact_names = {
        field: staging / f"{field}.bin"
        for field in subject.quality_staging.ARTIFACT_FIELDS
    }
    for field, path in artifact_names.items():
        if field in {
            "plan",
            "materialization",
            "root_seal",
            "performance_receipt",
            "profile_registry",
            "launch_manifest",
            "runtime_source_manifest",
            "wheelhouse_manifest",
        }:
            _json(path, {"field": field})
        else:
            path.write_bytes(field.encode("ascii"))
    ready_path = staging / subject.quality_staging.READY_NAME
    _json(ready_path, {"status": "ready"})

    expected_ids = [
        job_id
        for ids in subject.quality_bridge.WAVE_JOB_IDS
        for job_id in ids
    ]
    plan = {
        "plan_sha256": "1" * 64,
        "source_paths": {
            "staging_directory": str(staging),
            "launch_manifest": str(artifact_names["launch_manifest"]),
            "performance_receipt": str(
                artifact_names["performance_receipt"]
            ),
            "profile_registry": str(artifact_names["profile_registry"]),
        },
        "machine_contract": {"provisioning_model": "SPOT"},
        "waves": [
            {
                "requires_owned_compute_absent": True,
                "requires_worker_iam_removed_before_receive": True,
            },
            {
                "requires_owned_compute_absent": True,
                "requires_worker_iam_removed_before_receive": True,
            },
        ],
        "launch_manifest": {
            "quality_package": {
                "sha256": hashlib.sha256(b"quality_package").hexdigest()
            },
            "runtime_source": {"sha256": "3" * 64},
            "wheelhouse": {"sha256": "4" * 64},
            "candidate_library": {
                "path": "native/candidate/release/libofc_hu_m3_engine.so",
                "bytes": len(b"candidate02"),
            },
        },
    }
    ledger = {
        "ledger_sha256": "5" * 64,
        "accepted_jobs": [{"job_id": item} for item in expected_ids],
        "transitions": [
            {"transition_sha256": "6" * 64},
            {"transition_sha256": "7" * 64},
        ],
    }
    _json(gcp_plan_path, plan)
    _json(ledger_path, ledger)

    monkeypatch.setattr(
        subject.quality_bridge,
        "validate_gcp_plan",
        lambda value, replay_sources: dict(value),
    )
    monkeypatch.setattr(
        subject.quality_bridge,
        "validate_attempt_ledger",
        lambda _plan, value: dict(value),
    )
    monkeypatch.setattr(
        subject.quality_bridge,
        "build_resume_plan",
        lambda _plan, _ledger: {"all_jobs_accepted": True},
    )
    staging_receipt = {
        "artifacts": {
            field: {"path": field, "sha256": "8" * 64, "bytes": 1}
            for field in subject.quality_staging.ARTIFACT_FIELDS
        }
    }
    monkeypatch.setattr(
        subject.quality_staging,
        "validate_local_staging_receipt",
        lambda _path: staging_receipt,
    )
    monkeypatch.setattr(
        subject.quality_staging,
        "_file_from_record",
        lambda _root, _record, label: artifact_names[label],
    )
    performance_audit = {
        "schema": subject.performance_portable.AUDIT_SCHEMA,
        "mode": "exact_pinned_portable_no_source_path_dereference",
        "receipt_file_sha256": (
            subject.performance_portable.PINNED_RECEIPT_FILE_SHA256
        ),
        "receipt_sha256": subject.performance_portable.PINNED_RECEIPT_SHA256,
        "expected_profile_sha256": (
            subject.quality.CURRENT_PROFILE_REGISTRY_SHA256
        ),
        "source_paths_dereferenced": False,
        "new_seed_authorized": False,
        "replacement_binary_authorized": False,
        "current_profile_changed": False,
        "audit_sha256": "f" * 64,
    }
    monkeypatch.setattr(
        subject.performance_portable,
        "load_preferred_or_pinned_receipt",
        lambda _path, *, expected_profile_sha256: (
            {"receipt_sha256": subject.performance_portable.PINNED_RECEIPT_SHA256},
            dict(performance_audit),
        ),
    )
    monkeypatch.setattr(
        subject.quality,
        "validate_plan_authorization",
        lambda value, performance_receipt_path: {
            **dict(value),
            "performance": str(performance_receipt_path),
        },
    )
    materialized_root = controller_fs / "materialized-roots"
    materialized_root.mkdir()
    for index in range(55):
        _json(
            materialized_root / "roots" / f"pair-{index:03d}.json",
            {"index": index},
        )
    monkeypatch.setattr(
        subject.quality,
        "validate_materialization_receipt",
        lambda _value, plan, replay_roots: {
            "root_directory": str(materialized_root)
        },
    )
    monkeypatch.setattr(
        subject.quality,
        "validate_root_seal",
        lambda value, plan, materialization: dict(value),
    )
    merge = {
        "source_paths": {
            "plan": str(artifact_names["plan"]),
            "materialization": str(artifact_names["materialization"]),
            "root_seal": str(artifact_names["root_seal"]),
            "results_directory": str(accepted),
            "performance_receipt": str(
                artifact_names["performance_receipt"]
            ),
        }
    }
    gate = {
        "status": "pass",
        "all_gates_passed": True,
        "quality_pilot_passed": True,
        "data_pilot_25_paired_authorized": True,
        "full_9000_paired_fanout_authorized": False,
    }
    monkeypatch.setattr(
        subject.quality_gate,
        "build_fresh_quality_merge",
        lambda **_kwargs: dict(merge),
    )
    monkeypatch.setattr(
        subject.quality_gate,
        "build_fresh_quality_gate",
        lambda **_kwargs: dict(gate),
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

    candidate_raw = b"candidate02"
    candidate_hash = hashlib.sha256(candidate_raw).hexdigest()
    monkeypatch.setattr(
        subject.dataset,
        "ACCEPTED_CANDIDATE_LIBRARY_SHA256",
        candidate_hash,
    )
    runtime_payloads = {
        name: f"# {name}\n".encode("ascii")
        for name in subject._REQUIRED_DATASET_RUNTIME_PATHS
    }
    runtime_payloads[
        "native/candidate/release/libofc_hu_m3_engine.so"
    ] = candidate_raw
    wheelhouse_payloads = {"numpy.whl": b"wheel"}

    def archive_payloads(path: Path, **_kwargs: Any) -> dict[str, bytes]:
        if Path(path) == artifact_names["runtime_source"]:
            return dict(runtime_payloads)
        return dict(wheelhouse_payloads)

    monkeypatch.setattr(
        subject.quality_transport, "_archive_payloads", archive_payloads
    )
    monkeypatch.setattr(
        subject.dataset,
        "validate_dataset_plan",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        subject.dataset,
        "validate_completed_shard",
        lambda **_kwargs: {
            "pair_count": 25,
            "root_count": 50,
            "seat_counts": {"first": 25, "second": 25},
        },
    )

    cloud_calls: list[str] = []

    def pilot_runner(**kwargs: Any) -> dict[str, Any]:
        pilot_root = Path(kwargs["output_root"])
        _json(
            pilot_root / subject.local_pilot.PLAN_NAME,
            {"schema": "dataset-plan"},
        )
        shard = (
            pilot_root
            / subject.local_pilot.SHARDS_DIRECTORY_NAME
            / subject.dataset.SMOKE_SHARD_ID
        )
        _json(shard / "SHARD_DONE.json", {"status": "done"})
        _json(
            pilot_root / subject.local_pilot.SMOKE_GATE_NAME,
            {"status": "pass"},
        )
        return {
            "status": "passed_source_replayed_25_paired_local_pilot",
            "paired_hand_count": 25,
            "root_count": 50,
            "full_fanout_started": False,
            "current_profile_changed": False,
        }

    def controller_prepare(**kwargs: Any) -> dict[str, Any]:
        controller = Path(kwargs["output_root"])
        _json(
            controller / "portable_authorization.json",
            {
                "authorization_sha256": "9" * 64,
                "full_9000_paired_fanout_authorized": True,
            },
        )
        source_paths = [
            kwargs["dataset_plan_path"],
            kwargs["fresh_quality_gate_path"],
            kwargs["smoke_gate_path"],
            kwargs["smoke_shard_archive_path"],
            kwargs["runtime_archive_path"],
            kwargs["wheelhouse_archive_path"],
            kwargs["candidate_library_path"],
        ]
        _json(
            controller / "transport_plan.json",
            {
                "plan_sha256": "a" * 64,
                "cloud_shard_count": 359,
                "precompleted_smoke_shard_id": subject.dataset.SMOKE_SHARD_ID,
                "cloud_execution_started": False,
                "content_sources": [
                    {"source_path": str(Path(item).resolve())}
                    for item in source_paths
                ],
                "source_paths": {
                    "smoke_shard_directory": str(
                        Path(kwargs["smoke_shard_directory"]).resolve()
                    )
                },
            },
        )
        contract = {
            "status": "prepared_cloud_not_started",
            "quality_and_smoke_source_replayed": True,
            "current_profile_changed": False,
        }
        _json(controller / "controller_contract.json", contract)
        return contract

    return {
        "controller_filesystem_root": controller_fs,
        "output_root": controller_fs / "closeout",
        "repository_root": repository,
        "fresh_quality_gcp_plan_path": gcp_plan_path,
        "fresh_quality_ledger_path": ledger_path,
        "accepted_results_directory": accepted,
        "dataset_run_name": "regular-hu-m31-dataset-20260724-001",
        "dataset_bucket": "ofc-m31-artifacts",
        "provider_config_path": provider,
        "raw_controller_token": "0d89ae50-310a-4de4-bb20-19c8c3ca666c",
        "platform_name": "linux",
        "pilot_runner": pilot_runner,
        "controller_prepare": controller_prepare,
        "cloud_calls": cloud_calls,
    }


def test_same_linux_closeout_is_replayable_and_stops_before_cloud(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _install_closeout_fakes(tmp_path, monkeypatch)
    first = subject.run_same_linux_closeout(
        **{key: value for key, value in kwargs.items() if key != "cloud_calls"}
    )
    second = subject.run_same_linux_closeout(
        **{key: value for key, value in kwargs.items() if key != "cloud_calls"}
    )

    assert second == first
    assert first["status"] == (
        "complete_same_linux_ready_359_shard_fanout_not_started"
    )
    assert first["quality_job_count"] == 15
    assert first["dataset_smoke_pair_count"] == 25
    assert first["fanout_shard_count"] == 359
    assert first["performance_receipt_validation_mode"] == (
        "exact_pinned_portable_no_source_path_dereference"
    )
    assert first["performance_source_paths_dereferenced"] is False
    assert first["fresh_quality_worker_vm_reuse_allowed"] is False
    assert first["cloud_called"] is False
    assert first["full_fanout_started"] is False
    assert kwargs["cloud_calls"] == []

    root = Path(kwargs["output_root"])
    manifest = json.loads(
        (root / subject.FANOUT_MANIFEST_NAME).read_text(encoding="ascii")
    )
    assert manifest["required_restore_root"] == str(
        kwargs["controller_filesystem_root"]
    )
    assert manifest["restore_contract"] == (
        "same_persistent_disk_exact_absolute_mount_only"
    )
    assert manifest["fresh_quality_worker_vm_reuse_allowed"] is False
    assert manifest["hidden_information_field_count"] == 0
    assert manifest["performance_receipt_validation"][
        "receipt_file_sha256"
    ] == subject.performance_portable.PINNED_RECEIPT_FILE_SHA256
    assert manifest["performance_source_paths_dereferenced"] is False


def test_non_linux_fails_before_any_artifact(
    tmp_path: Path,
) -> None:
    output = (tmp_path / "must-not-exist").resolve()
    with pytest.raises(PermissionError, match="native Linux"):
        subject.run_same_linux_closeout(
            controller_filesystem_root=tmp_path.resolve(),
            output_root=output,
            repository_root=tmp_path.resolve(),
            fresh_quality_gcp_plan_path=tmp_path / "missing-plan",
            fresh_quality_ledger_path=tmp_path / "missing-ledger",
            accepted_results_directory=tmp_path.resolve(),
            dataset_run_name="unused",
            dataset_bucket="unused",
            provider_config_path=tmp_path / "missing-provider",
            raw_controller_token="unused",
            platform_name="win32",
        )
    assert not output.exists()
