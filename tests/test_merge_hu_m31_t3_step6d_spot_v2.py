from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import hu_m31_t3_step6d_spot_v2 as spot
from ofc_regular import merge_hu_m31_t3_step6d_spot_v2 as bridge
from ofc_regular import run_hu_m31_t3_step6d_performance_v2 as runner


RUN_NAME = "step6d-v2-merge-unit"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(runner.canonical_bytes(value))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract(
    variant: str = runner.CANDIDATE01_VARIANT,
) -> dict[str, Any]:
    return runner.build_run_contract(
        candidate_library_sha256="1" * 64,
        reference_library_sha256=spot.REFERENCE_NATIVE_LIBRARY_SHA256,
        workers=1,
        rayon_threads_per_worker=16,
        variant=variant,
    )


def _source_entries(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = {
        name: {"sha256": "a" * 64, "bytes": 1} for name in spot._REQUIRED_SOURCE_PATHS
    }
    entries[spot.REFERENCE_PACKAGE_PATH] = {
        "sha256": contract["reference_library_sha256"],
        "bytes": 20,
    }
    entries[spot.CANDIDATE_PACKAGE_PATH] = {
        "sha256": contract["candidate_library_sha256"],
        "bytes": 20,
    }
    entries[spot.FEATURE_PACKAGE_PATH] = {
        "sha256": spot.EXPECTED_FEATURE_ENCODER_SHA256,
        "bytes": 20,
    }
    if runner.contract_variant(contract) == runner.CANDIDATE02_TAIL_V2_VARIANT:
        entries[spot.CANDIDATE02_TAIL_V2_SELECTION_PACKAGE_PATH] = {
            "sha256": runner.CANDIDATE02_TAIL_V2_SELECTION_MANIFEST_SHA256,
            "bytes": spot.CANDIDATE02_TAIL_V2_SELECTION_BYTES,
        }
    return entries


def _build_receive(
    root: Path,
    *,
    variant: str = runner.CANDIDATE01_VARIANT,
) -> dict[str, Any]:
    contract = _contract(variant)
    contract_digest = runner.canonical_sha256(contract)
    tail_indices = spot._contract_tail_hand_indices(contract)
    authorized_ids = list(spot.authorized_job_ids(contract))
    records: list[dict[str, Any]] = []
    receipt_jobs: list[dict[str, Any]] = []
    candidate_done: list[str] = []
    reference_done: list[str] = []
    for manifest in spot.build_job_manifests(contract):
        role = str(manifest["source_role"])
        hand_index = int(manifest["work_hand_indices"][0])
        identifier = spot.job_id(role, hand_index, contract)
        manifest_path = root / "jobs" / f"{identifier}.json"
        _write(manifest_path, manifest)
        record = spot._job_record(manifest)
        record.update(
            {"sha256": _sha256(manifest_path), "bytes": manifest_path.stat().st_size}
        )
        records.append(record)

        job_dir = root / "jobs" / identifier
        root_path = job_dir / "roots" / f"hand_{hand_index:03d}.json"
        hand_path = job_dir / "hands" / role / f"hand_{hand_index:03d}.json"
        done_path = job_dir / "DONE.json"
        done = {
            "schema": runner._done_schema(contract),
            "status": "complete_source_isolated_shard",
            "run_contract_digest": contract_digest,
            "source_role": role,
            "work_hand_indices": [hand_index],
            "completed_hand_indices": [hand_index],
            "reference_library_sha256": contract["reference_library_sha256"],
            "candidate_library_sha256": contract["candidate_library_sha256"],
            "native_library_sha256": contract[f"{role}_library_sha256"],
            "training_eligible": False,
            "quality_evidence": False,
            "promotion_evidence": False,
            "current_profile_changed": False,
        }
        _write(job_dir / "run_contract.json", contract)
        _write(job_dir / "shard_manifest.json", manifest)
        _write(root_path, {})
        _write(hand_path, {})
        _write(done_path, done)
        relative_done = f"jobs/{identifier}/DONE.json"
        receipt_jobs.append(
            {
                "job_id": identifier,
                "source_role": role,
                "work_hand_indices": [hand_index],
                "done_path": relative_done,
                "done_sha256": _sha256(done_path),
                "source_hand_path": (
                    f"jobs/{identifier}/hands/{role}/hand_{hand_index:03d}.json"
                ),
                "source_hand_sha256": _sha256(hand_path),
                "root_sha256": _sha256(root_path),
                "run_contract_digest": contract_digest,
            }
        )
        (candidate_done if role == "candidate" else reference_done).append(
            relative_done
        )

    entries = _source_entries(contract)
    launch_target = spot.build_launch_target()
    package = {
        "schema": spot.SPOT_PACKAGE_SCHEMA,
        "status": "immutable_package_ready_not_authorized",
        "run_name": RUN_NAME,
        "source_name": spot.SOURCE_NAME,
        "source_sha256": "b" * 64,
        "source_bytes": 100,
        "startup_name": spot.STARTUP_NAME,
        "startup_sha256": "c" * 64,
        "run_contract": contract,
        "run_contract_digest": contract_digest,
        "accepted_reference": {
            "package_path": spot.REFERENCE_PACKAGE_PATH,
            "sha256": contract["reference_library_sha256"],
        },
        "accepted_candidate": {
            "package_path": spot.CANDIDATE_PACKAGE_PATH,
            "sha256": contract["candidate_library_sha256"],
        },
        "feature_encoder": {
            "package_path": spot.FEATURE_PACKAGE_PATH,
            "sha256": spot.EXPECTED_FEATURE_ENCODER_SHA256,
        },
        "image": {
            "project": "debian-cloud",
            "name": spot.EXPECTED_IMAGE_NAME,
            "id": spot.EXPECTED_IMAGE_ID,
            "self_link": spot.EXPECTED_IMAGE_SELF_LINK,
        },
        "allocation": {
            "machine_type": spot.EXPECTED_MACHINE_TYPE,
            "process_count": 1,
            "rayon_threads_per_process": 16,
            "omp_threads": 1,
            "m3_batch_threads": 1,
        },
        "launch_target": launch_target,
        "cost_guard": spot.build_cost_guard(),
        "tail_schedule": {
            "contract_hand_indices": list(runner.CONTRACT_HAND_INDICES),
            "work_hand_indices": list(tail_indices),
            "source_roles": list(runner.SOURCE_ROLES),
            "mapping": "one_source_hand_per_vm",
            "logical_job_count": 20,
            "max_logical_jobs": 20,
        },
        "job_manifests": records,
        "source_entries": entries,
        "source_entry_count": len(entries),
        "checkpoint": {
            "unit": "completed_source_hand",
            "upload_after_each_hand": True,
            "immutable_write_once": True,
            "resume_verifies_all_artifacts": True,
            "done_uploaded_last": True,
        },
        "heartbeat": {
            "schema": spot.HEARTBEAT_SCHEMA,
            "required": True,
            "interval_seconds": spot.HEARTBEAT_INTERVAL_SECONDS,
            "remote_scope": "progress_only",
        },
        "spot_execution_authorized": False,
        "production_fanout_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "m31_complete": False,
        "gcloud_invoked": False,
    }
    package_path = root / "package_manifest.json"
    _write(package_path, package)
    package_sha = _sha256(package_path)
    authorization = {
        "schema": spot.LAUNCH_AUTHORIZATION_SCHEMA,
        "status": "explicit_tail_spot_authorization",
        "run_name": RUN_NAME,
        "package_manifest_sha256": package_sha,
        "source_sha256": package["source_sha256"],
        "startup_sha256": package["startup_sha256"],
        "run_contract_digest": contract_digest,
        "launch_target": launch_target,
        "cost_guard": package["cost_guard"],
        "authorized_job_ids": authorized_ids,
        "logical_job_count": 20,
        "tail_only": True,
        "spot_execution_authorized": True,
        "production_fanout_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "authorized_unix_seconds": 1.0,
    }
    authorization_path = root / spot.AUTHORIZATION_NAME
    _write(authorization_path, authorization)
    _write(
        root / spot.PACKAGE_READY_NAME,
        {
            "schema": spot.PACKAGE_READY_SCHEMA,
            "status": "immutable_local_package_complete",
            "run_name": RUN_NAME,
            "package_manifest_sha256": package_sha,
            "source_sha256": package["source_sha256"],
            "startup_sha256": package["startup_sha256"],
            "run_contract_digest": contract_digest,
            "logical_job_count": 20,
            "gcloud_invoked": False,
            "spot_vm_started": False,
            "current_profile_changed": False,
        },
    )
    expected_ids = authorized_ids
    result_prefix = f"gs://{launch_target['bucket']}/runs/{RUN_NAME}/results"
    preflight = {
        "schema": spot.LAUNCH_PREFLIGHT_SCHEMA,
        "status": "quota_and_collision_checks_passed",
        "checked_unix_seconds": 1.0,
        "region": spot.DEFAULT_REGION,
        "selected_job_ids": expected_ids,
        "selected_instance_names": [
            f"{RUN_NAME}-j{ordinal:02d}" for ordinal in range(20)
        ],
        "selected_done_uris": [
            f"{result_prefix}/jobs/{identifier}/DONE.json"
            for identifier in expected_ids
        ],
        "required_vcpus": 320,
        "quota": {
            "CPUS": {"limit": 400, "usage": 0, "available": 400},
            "PREEMPTIBLE_CPUS": {
                "limit": 400,
                "usage": 0,
                "available": 400,
            },
        },
        "instances_absent": True,
        "done_objects_absent": True,
        "cost_guard_sha256": runner.canonical_sha256(package["cost_guard"]),
        "selected_estimated_max_compute_usd": package["cost_guard"][
            "all_20_estimated_max_compute_usd"
        ],
        "hard_tail_compute_cap_usd": package["cost_guard"]["hard_tail_compute_cap_usd"],
    }
    claim = {
        "schema": spot.LAUNCH_CLAIM_SCHEMA,
        "status": "exclusive_claim_acquired_before_remote_mutation",
        "run_name": RUN_NAME,
        "selected_job_ids": expected_ids,
        "run_contract_digest": contract_digest,
        "launch_target": launch_target,
        "cost_guard_sha256": runner.canonical_sha256(package["cost_guard"]),
        "preflight_sha256": runner.canonical_sha256(preflight),
        "claimed_unix_seconds": 2.0,
        "crash_reuse_authorized": False,
    }
    claim_path = root / spot.LAUNCH_CLAIM_NAME
    _write(claim_path, claim)
    created = []
    for ordinal, identifier in enumerate(expected_ids):
        role = identifier.split("-hand-", 1)[0]
        created.append(
            {
                "job_id": identifier,
                "source_role": role,
                "work_hand_indices": [int(identifier.rsplit("-", 1)[-1])],
                "instance": f"{RUN_NAME}-j{ordinal:02d}",
                "zone": spot.DEFAULT_ZONES[ordinal % len(spot.DEFAULT_ZONES)],
                "attempt_index": 0,
                "status": "created",
            }
        )
    result = {
        "schema": spot.LAUNCH_RESULT_SCHEMA,
        "status": "selected_tail_jobs_created",
        "run_name": RUN_NAME,
        "selected_job_ids": expected_ids,
        "created": created,
        "logical_job_count": 20,
        "max_logical_jobs": 20,
        "machine_type": spot.EXPECTED_MACHINE_TYPE,
        "process_count": 1,
        "rayon_threads_per_process": 16,
        "run_contract_digest": contract_digest,
        "launch_target": launch_target,
        "cost_guard": package["cost_guard"],
        "preflight": preflight,
        "launch_claim_sha256": runner.canonical_sha256(claim),
        "failures": [],
        "cleanup": [],
        "cleanup_absence_proven": True,
        "cleanup_compute_stopped_or_absent": True,
        "production_fanout_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "runtime_policy_activated": False,
    }
    result_path = root / spot.LAUNCH_RESULT_NAME
    _write(result_path, result)
    receipt = {
        "schema": spot.RECEIVE_SCHEMA,
        "status": "exact_tail_source_jobs_received_and_validated",
        "run_name": RUN_NAME,
        "launch_target": launch_target,
        "result_prefix": result_prefix,
        "package_manifest_sha256": package_sha,
        "launch_authorization_sha256": _sha256(authorization_path),
        "launch_claim_sha256": _sha256(claim_path),
        "launch_result_sha256": _sha256(result_path),
        "resume_claim_sha256": None,
        "resume_result_sha256": None,
        "run_contract_digest": contract_digest,
        "work_hand_indices": list(tail_indices),
        "source_roles": list(runner.SOURCE_ROLES),
        "logical_job_count": 20,
        "jobs": receipt_jobs,
        "candidate_done_paths": candidate_done,
        "reference_done_paths": reference_done,
        "source_isolation_validated": True,
        "merge_executed": False,
        "production_fanout_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
    }
    _write(root / "receive_receipt.json", receipt)
    return receipt


def _add_resume_evidence(
    root: Path, selected: tuple[str, ...] = ("candidate-hand-002",)
) -> None:
    package = json.loads((root / "package_manifest.json").read_text("utf-8"))
    receipt_path = root / "receive_receipt.json"
    receipt = json.loads(receipt_path.read_text("utf-8"))
    allowed = list(spot.authorized_job_ids(package["run_contract"]))
    initial_names = [
        f"{RUN_NAME}-j{allowed.index(identifier):02d}" for identifier in selected
    ]
    resume_names = [f"{name}-a01" for name in initial_names]
    preflight = {
        "schema": spot.RESUME_PREFLIGHT_SCHEMA,
        "status": "attempt1_quota_done_and_instance_checks_passed",
        "checked_unix_seconds": 3.0,
        "region": spot.DEFAULT_REGION,
        "attempt_index": 1,
        "selected_job_ids": list(selected),
        "selected_initial_instance_names": initial_names,
        "selected_resume_instance_names": resume_names,
        "selected_done_uris": [
            f"{receipt['result_prefix']}/jobs/{identifier}/DONE.json"
            for identifier in selected
        ],
        "required_vcpus": len(selected) * spot.VCPUS_PER_VM,
        "quota": {
            "CPUS": {"limit": 400, "usage": 0, "available": 400},
            "PREEMPTIBLE_CPUS": {
                "limit": 400,
                "usage": 0,
                "available": 400,
            },
        },
        "no_active_initial_instances": True,
        "resume_instance_names_absent": True,
        "done_objects_absent": True,
        "validated_completed_job_ids": [
            identifier for identifier in allowed if identifier not in selected
        ],
        "all_incomplete_jobs_selected": True,
        "cost_guard_sha256": runner.canonical_sha256(package["cost_guard"]),
        "max_cumulative_vm_jobs": spot.MAX_CUMULATIVE_VM_JOBS,
        "all_attempts_estimated_max_compute_usd": (
            spot.ALL_ATTEMPTS_ESTIMATED_MAX_COMPUTE_USD
        ),
        "hard_tail_compute_cap_usd": spot.HARD_TAIL_COMPUTE_CAP_USD,
    }
    initial_claim_path = root / spot.LAUNCH_CLAIM_NAME
    initial_result_path = root / spot.LAUNCH_RESULT_NAME
    resume_claim = {
        "schema": spot.RESUME_CLAIM_SCHEMA,
        "status": "exclusive_attempt1_claim_acquired_before_remote_mutation",
        "run_name": RUN_NAME,
        "attempt_index": 1,
        "selected_job_ids": list(selected),
        "initial_launch_claim_sha256": _sha256(initial_claim_path),
        "initial_launch_result_sha256": _sha256(initial_result_path),
        "run_contract_digest": receipt["run_contract_digest"],
        "launch_target": receipt["launch_target"],
        "cost_guard_sha256": runner.canonical_sha256(package["cost_guard"]),
        "preflight_sha256": runner.canonical_sha256(preflight),
        "claimed_unix_seconds": 4.0,
        "third_attempt_authorized": False,
    }
    resume_claim_path = root / spot.RESUME_CLAIM_NAME
    _write(resume_claim_path, resume_claim)
    created = []
    for identifier, _initial_name, resume_name in zip(
        selected, initial_names, resume_names, strict=True
    ):
        ordinal = allowed.index(identifier)
        created.append(
            {
                "job_id": identifier,
                "source_role": identifier.split("-hand-", 1)[0],
                "work_hand_indices": [int(identifier.rsplit("-", 1)[-1])],
                "instance": resume_name,
                "zone": spot.DEFAULT_ZONES[ordinal % len(spot.DEFAULT_ZONES)],
                "attempt_index": 1,
                "status": "created",
            }
        )
    resume_result = {
        "schema": spot.RESUME_RESULT_SCHEMA,
        "status": "selected_resume_jobs_created",
        "run_name": RUN_NAME,
        "attempt_index": 1,
        "selected_job_ids": list(selected),
        "created": created,
        "logical_job_count": len(created),
        "max_resume_jobs": spot.MAX_LOGICAL_JOBS,
        "max_cumulative_vm_jobs": spot.MAX_CUMULATIVE_VM_JOBS,
        "run_contract_digest": receipt["run_contract_digest"],
        "launch_target": receipt["launch_target"],
        "cost_guard": package["cost_guard"],
        "preflight": preflight,
        "resume_claim_sha256": _sha256(resume_claim_path),
        "initial_launch_claim_sha256": _sha256(initial_claim_path),
        "initial_launch_result_sha256": _sha256(initial_result_path),
        "failures": [],
        "cleanup": [],
        "cleanup_absence_proven": True,
        "cleanup_compute_stopped_or_absent": True,
        "third_attempt_authorized": False,
        "production_fanout_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "runtime_policy_activated": False,
    }
    resume_result_path = root / spot.RESUME_RESULT_NAME
    _write(resume_result_path, resume_result)
    receipt["resume_claim_sha256"] = _sha256(resume_claim_path)
    receipt["resume_result_sha256"] = _sha256(resume_result_path)
    _write(receipt_path, receipt)


@pytest.fixture
def receive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "receive"
    _build_receive(root)

    def validate_completed(output_dir: Path) -> dict[str, Any]:
        return json.loads((Path(output_dir) / "DONE.json").read_text("utf-8"))

    monkeypatch.setattr(bridge.runner, "validate_completed_output", validate_completed)
    return root


def test_receipt_bridge_resolves_exact_tail_then_calls_existing_merger(
    receive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_merge(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        captured.update(kwargs)
        return {"status": "pass"}, {"schema": "unit-validation"}

    monkeypatch.setattr(bridge.merger, "merge_and_validate_performance_v2", fake_merge)
    summary, validation = bridge.merge_received_spot_v2(
        receive_dir=receive,
        summary_output_path=tmp_path / "merge" / "summary.json",
        validation_output_path=tmp_path / "merge" / "validation.json",
        expected_run_name=RUN_NAME,
    )

    assert summary == {"status": "pass"}
    assert validation == {"schema": "unit-validation"}
    assert len(captured["candidate_done_paths"]) == 10
    assert len(captured["reference_done_paths"]) == 10
    assert all(
        path.is_absolute() and path.is_relative_to(receive)
        for path in captured["candidate_done_paths"]
    )
    assert captured["scope"] == "auto"


def test_receipt_bridge_routes_candidate02_to_dedicated_merger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receive = tmp_path / "candidate02-receive"
    _build_receive(receive, variant=runner.CANDIDATE02_VARIANT)

    def validate_completed(output_dir: Path) -> dict[str, Any]:
        return json.loads((Path(output_dir) / "DONE.json").read_text("utf-8"))

    monkeypatch.setattr(bridge.runner, "validate_completed_output", validate_completed)
    captured: dict[str, Any] = {}

    def fake_candidate02_merge(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        captured.update(kwargs)
        return {"status": "pass"}, {"schema": "candidate02-unit-validation"}

    monkeypatch.setattr(
        bridge.candidate02_merger,
        "merge_and_validate_candidate02_performance",
        fake_candidate02_merge,
    )
    monkeypatch.setattr(
        bridge.merger,
        "merge_and_validate_performance_v2",
        lambda **_kwargs: pytest.fail("candidate01 merger received candidate02"),
    )
    summary, validation = bridge.merge_received_spot_v2(
        receive_dir=receive,
        summary_output_path=tmp_path / "merge" / "summary.json",
        validation_output_path=tmp_path / "merge" / "validation.json",
        scope=bridge.candidate02_merger.TAIL_DIAGNOSTIC_SCOPE,
        expected_run_name=RUN_NAME,
    )
    assert summary == {"status": "pass"}
    assert validation == {"schema": "candidate02-unit-validation"}
    assert len(captured["candidate_done_paths"]) == 10
    assert len(captured["reference_done_paths"]) == 10
    assert "scope" not in captured


def test_receipt_bridge_routes_candidate02_tail_v2_to_new_dedicated_merger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receive = tmp_path / "candidate02-tail-v2-receive"
    receipt = _build_receive(receive, variant=runner.CANDIDATE02_TAIL_V2_VARIANT)

    def validate_completed(output_dir: Path) -> dict[str, Any]:
        return json.loads((Path(output_dir) / "DONE.json").read_text("utf-8"))

    monkeypatch.setattr(bridge.runner, "validate_completed_output", validate_completed)
    captured: dict[str, Any] = {}

    def fake_tail_v2_merge(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        captured.update(kwargs)
        return {"status": "pass"}, {"schema": "candidate02-tail-v2-validation"}

    monkeypatch.setattr(
        bridge.candidate02_tail_v2_merger,
        "merge_and_validate_candidate02_tail_v2",
        fake_tail_v2_merge,
    )
    monkeypatch.setattr(
        bridge.candidate02_merger,
        "merge_and_validate_candidate02_performance",
        lambda **_kwargs: pytest.fail("candidate02-v1 merger received tail-v2"),
    )
    monkeypatch.setattr(
        bridge.merger,
        "merge_and_validate_performance_v2",
        lambda **_kwargs: pytest.fail("candidate01 merger received tail-v2"),
    )
    summary, validation = bridge.merge_received_spot_v2(
        receive_dir=receive,
        summary_output_path=tmp_path / "merge" / "summary.json",
        validation_output_path=tmp_path / "merge" / "validation.json",
        scope=bridge.candidate02_tail_v2_merger.TAIL_DIAGNOSTIC_SCOPE,
        expected_run_name=RUN_NAME,
    )
    assert summary == {"status": "pass"}
    assert validation == {"schema": "candidate02-tail-v2-validation"}
    assert receipt["work_hand_indices"] == list(
        runner.CANDIDATE02_TAIL_V2_TAIL_HAND_INDICES
    )
    assert [path.parent.name for path in captured["candidate_done_paths"]] == [
        f"candidate-hand-{index:03d}"
        for index in runner.CANDIDATE02_TAIL_V2_TAIL_HAND_INDICES
    ]
    assert len(captured["reference_done_paths"]) == 10


def test_candidate02_tail_v2_receive_rejects_mixed_v1_job_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receive = tmp_path / "candidate02-tail-v2-receive"
    receipt = _build_receive(receive, variant=runner.CANDIDATE02_TAIL_V2_VARIANT)
    mixed_job = receive / receipt["jobs"][0]["done_path"]
    _write(mixed_job.parent / "run_contract.json", _contract())

    def validate_completed(output_dir: Path) -> dict[str, Any]:
        return json.loads((Path(output_dir) / "DONE.json").read_text("utf-8"))

    monkeypatch.setattr(bridge.runner, "validate_completed_output", validate_completed)
    with pytest.raises(ValueError, match="contract chain changed"):
        bridge.resolve_received_done_paths(receive, expected_run_name=RUN_NAME)


def test_candidate02_tail_v2_receive_rejects_old_tail_receipt_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receive = tmp_path / "candidate02-tail-v2-receive"
    _build_receive(receive, variant=runner.CANDIDATE02_TAIL_V2_VARIANT)
    receipt_path = receive / "receive_receipt.json"
    receipt = json.loads(receipt_path.read_text("utf-8"))
    receipt["work_hand_indices"] = list(runner.TAIL_HAND_INDICES)
    _write(receipt_path, receipt)

    monkeypatch.setattr(
        bridge.runner,
        "validate_completed_output",
        lambda output_dir: json.loads(
            (Path(output_dir) / "DONE.json").read_text("utf-8")
        ),
    )
    with pytest.raises(ValueError, match="tail cardinality changed"):
        bridge.resolve_received_done_paths(receive, expected_run_name=RUN_NAME)


def test_candidate02_tail_v2_receive_rejects_selection_manifest_record_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receive = tmp_path / "candidate02-tail-v2-receive"
    _build_receive(receive, variant=runner.CANDIDATE02_TAIL_V2_VARIANT)
    package_path = receive / "package_manifest.json"
    package = json.loads(package_path.read_text("utf-8"))
    package["source_entries"][spot.CANDIDATE02_TAIL_V2_SELECTION_PACKAGE_PATH][
        "sha256"
    ] = ("0" * 64)
    _write(package_path, package)

    monkeypatch.setattr(
        bridge.runner,
        "validate_completed_output",
        lambda output_dir: json.loads(
            (Path(output_dir) / "DONE.json").read_text("utf-8")
        ),
    )
    with pytest.raises(ValueError, match="package/authorization chain changed"):
        bridge.resolve_received_done_paths(receive, expected_run_name=RUN_NAME)


def test_candidate02_receive_rejects_candidate01_done_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receive = tmp_path / "candidate02-receive"
    _build_receive(receive, variant=runner.CANDIDATE02_VARIANT)
    receipt_path = receive / "receive_receipt.json"
    receipt = json.loads(receipt_path.read_text("utf-8"))
    done_path = receive / receipt["jobs"][0]["done_path"]
    done = json.loads(done_path.read_text("utf-8"))
    done["schema"] = runner.DONE_SCHEMA
    _write(done_path, done)
    receipt["jobs"][0]["done_sha256"] = _sha256(done_path)
    _write(receipt_path, receipt)

    def validate_completed(output_dir: Path) -> dict[str, Any]:
        return json.loads((Path(output_dir) / "DONE.json").read_text("utf-8"))

    monkeypatch.setattr(bridge.runner, "validate_completed_output", validate_completed)
    with pytest.raises(ValueError, match="DONE binding changed"):
        bridge.resolve_received_done_paths(receive, expected_run_name=RUN_NAME)


@pytest.mark.parametrize(
    "mutation",
    ("traversal", "duplicate_job", "wrong_hand", "wrong_hash", "wrong_prefix"),
)
def test_receipt_bridge_fails_closed_before_merger(
    receive: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    receipt_path = receive / "receive_receipt.json"
    receipt = json.loads(receipt_path.read_text("utf-8"))
    if mutation == "traversal":
        receipt["jobs"][0]["done_path"] = "../escape/DONE.json"
    elif mutation == "duplicate_job":
        receipt["jobs"][1]["job_id"] = receipt["jobs"][0]["job_id"]
    elif mutation == "wrong_hand":
        receipt["jobs"][0]["source_hand_path"] = receipt["jobs"][1]["source_hand_path"]
    elif mutation == "wrong_hash":
        receipt["jobs"][0]["done_sha256"] = "0" * 64
    else:
        receipt["result_prefix"] = "gs://wrong-bucket/runs/wrong/results"
    _write(receipt_path, receipt)
    monkeypatch.setattr(
        bridge.merger,
        "merge_and_validate_performance_v2",
        lambda **_kwargs: pytest.fail("merger called before receipt validation"),
    )

    with pytest.raises(ValueError):
        bridge.merge_received_spot_v2(
            receive_dir=receive,
            summary_output_path=receive.parent / "summary.json",
            validation_output_path=receive.parent / "validation.json",
            expected_run_name=RUN_NAME,
        )


def test_receipt_bridge_rejects_package_tamper_and_extra_receive_file(
    receive: Path,
) -> None:
    package_path = receive / "package_manifest.json"
    package = json.loads(package_path.read_text("utf-8"))
    package["source_sha256"] = "0" * 64
    _write(package_path, package)
    with pytest.raises(ValueError):
        bridge.resolve_received_done_paths(receive, expected_run_name=RUN_NAME)

    _build_receive(receive)
    (receive / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable receive tree"):
        bridge.resolve_received_done_paths(receive, expected_run_name=RUN_NAME)


@pytest.mark.parametrize(
    "mutation", ("claim_subset", "result_failure", "created_instance", "quota")
)
def test_receipt_bridge_revalidates_launch_claim_and_result_semantics(
    receive: Path, mutation: str
) -> None:
    receipt_path = receive / "receive_receipt.json"
    receipt = json.loads(receipt_path.read_text("utf-8"))
    if mutation == "claim_subset":
        claim_path = receive / spot.LAUNCH_CLAIM_NAME
        claim = json.loads(claim_path.read_text("utf-8"))
        claim["selected_job_ids"] = claim["selected_job_ids"][:-1]
        _write(claim_path, claim)
        receipt["launch_claim_sha256"] = _sha256(claim_path)
    else:
        result_path = receive / spot.LAUNCH_RESULT_NAME
        result = json.loads(result_path.read_text("utf-8"))
        if mutation == "result_failure":
            result["failures"] = [
                {"job_id": "candidate-hand-002", "error_type": "Unit", "error": "x"}
            ]
        elif mutation == "created_instance":
            result["created"][0]["instance"] = "wrong-instance"
        else:
            result["preflight"]["quota"]["CPUS"]["available"] = 319
        _write(result_path, result)
        receipt["launch_result_sha256"] = _sha256(result_path)
    _write(receipt_path, receipt)

    with pytest.raises(ValueError):
        bridge.resolve_received_done_paths(receive, expected_run_name=RUN_NAME)


def test_receipt_bridge_accepts_one_bound_resume_attempt(receive: Path) -> None:
    _add_resume_evidence(receive, ("candidate-hand-002", "reference-hand-050"))

    candidate, reference, receipt = bridge.resolve_received_done_paths(
        receive, expected_run_name=RUN_NAME
    )

    assert len(candidate) == len(reference) == 10
    assert receipt["resume_claim_sha256"] == _sha256(receive / spot.RESUME_CLAIM_NAME)
    assert receipt["resume_result_sha256"] == _sha256(receive / spot.RESUME_RESULT_NAME)


def test_receipt_bridge_schema_exactly_tracks_lifecycle_schema() -> None:
    assert bridge._RECEIPT_KEYS == spot._RECEIVE_RECEIPT_KEYS


@pytest.mark.parametrize("resumed", (False, True))
def test_actual_receive_jobs_output_resolves_through_bridge(
    receive: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resumed: bool,
) -> None:
    run_dir = receive
    if resumed:
        _add_resume_evidence(run_dir)
    package = json.loads((run_dir / "package_manifest.json").read_text("utf-8"))
    authorization = json.loads((run_dir / spot.AUTHORIZATION_NAME).read_text("utf-8"))
    _write(run_dir / spot.PACKAGE_MANIFEST_NAME, package)
    monkeypatch.setattr(spot, "validate_launch", lambda _path: (package, authorization))

    def copy_received_job(command: list[str], **_kwargs: Any) -> None:
        values = list(command)
        assert values[:4] == ["gcloud", "storage", "rsync", "--recursive"]
        destination = Path(values[5])
        shutil.copytree(
            run_dir / "jobs" / destination.name,
            destination,
            dirs_exist_ok=True,
        )

    monkeypatch.setattr(spot, "_run", copy_received_job)
    output = tmp_path / "actual-receive"
    receipt = spot.receive_jobs(run_dir=run_dir, output_dir=output)

    candidate, reference, resolved_receipt = bridge.resolve_received_done_paths(
        output, expected_run_name=RUN_NAME
    )

    assert resolved_receipt == receipt
    assert len(candidate) == len(reference) == 10
    assert (output / spot.LAUNCH_CLAIM_NAME).is_file()
    assert (output / spot.LAUNCH_RESULT_NAME).is_file()
    assert (output / spot.RESUME_CLAIM_NAME).exists() is resumed
    assert (output / spot.RESUME_RESULT_NAME).exists() is resumed


def test_receipt_bridge_rejects_resume_incomplete_set_mismatch(receive: Path) -> None:
    _add_resume_evidence(receive)
    claim_path = receive / spot.RESUME_CLAIM_NAME
    result_path = receive / spot.RESUME_RESULT_NAME
    receipt_path = receive / "receive_receipt.json"
    claim = json.loads(claim_path.read_text("utf-8"))
    result = json.loads(result_path.read_text("utf-8"))
    receipt = json.loads(receipt_path.read_text("utf-8"))
    result["preflight"]["validated_completed_job_ids"] = []
    claim["preflight_sha256"] = runner.canonical_sha256(result["preflight"])
    _write(claim_path, claim)
    result["resume_claim_sha256"] = _sha256(claim_path)
    _write(result_path, result)
    receipt["resume_claim_sha256"] = _sha256(claim_path)
    receipt["resume_result_sha256"] = _sha256(result_path)
    _write(receipt_path, receipt)

    with pytest.raises(ValueError, match="resume"):
        bridge.resolve_received_done_paths(receive, expected_run_name=RUN_NAME)


@pytest.mark.parametrize("present", ("claim", "result"))
def test_receipt_bridge_rejects_one_sided_resume_hash_pair(
    receive: Path, present: str
) -> None:
    receipt_path = receive / "receive_receipt.json"
    receipt = json.loads(receipt_path.read_text("utf-8"))
    receipt[f"resume_{present}_sha256"] = "0" * 64
    _write(receipt_path, receipt)

    with pytest.raises(ValueError, match="receipt boundary"):
        bridge.resolve_received_done_paths(receive, expected_run_name=RUN_NAME)


def test_receipt_bridge_rejects_unbound_or_tampered_resume_evidence(
    receive: Path,
) -> None:
    _add_resume_evidence(receive)
    receipt_path = receive / "receive_receipt.json"
    receipt = json.loads(receipt_path.read_text("utf-8"))
    receipt["resume_claim_sha256"] = None
    receipt["resume_result_sha256"] = None
    _write(receipt_path, receipt)
    with pytest.raises(ValueError, match="unbound resume evidence"):
        bridge.resolve_received_done_paths(receive, expected_run_name=RUN_NAME)

    _build_receive(receive)
    _add_resume_evidence(receive)
    result_path = receive / spot.RESUME_RESULT_NAME
    result = json.loads(result_path.read_text("utf-8"))
    result["third_attempt_authorized"] = True
    _write(result_path, result)
    receipt = json.loads(receipt_path.read_text("utf-8"))
    receipt["resume_result_sha256"] = _sha256(result_path)
    _write(receipt_path, receipt)
    with pytest.raises(ValueError, match="resume chain"):
        bridge.resolve_received_done_paths(receive, expected_run_name=RUN_NAME)


def test_receipt_bridge_rejects_outputs_inside_immutable_receive(receive: Path) -> None:
    with pytest.raises(ValueError, match="outside the receive tree"):
        bridge.merge_received_spot_v2(
            receive_dir=receive,
            summary_output_path=receive / "summary.json",
            validation_output_path=receive.parent / "validation.json",
            expected_run_name=RUN_NAME,
        )


def test_merge_wrapper_uses_bridge_and_contains_no_cloud_commands() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "Merge-GcpHuM31T3Step6dV2SpotRun.ps1"
    ).read_text(encoding="utf-8")

    assert "ofc_regular.merge_hu_m31_t3_step6d_spot_v2" in source
    assert "--expected-run-name" in source
    assert "spot_v2_merge/$RunName" in source
    assert "gcloud" not in source.casefold()
