from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_local as local,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as plan,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_vm_adapter as subject,
)


@pytest.fixture(scope="module")
def package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    target = tmp_path_factory.mktemp("diag-vm-adapter") / "package"
    local.build_local_package(output_dir=target)
    return target


@pytest.fixture(scope="module")
def stage1_receive(
    package: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    root = tmp_path_factory.mktemp("diag-vm-stage1-receive")
    opened = root / "stage1-open.json"
    local.open_local_stage(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
        output_path=opened,
    )
    output = root / "job"
    job = local.initialize_local_job(
        package_dir=package,
        stage_open_path=opened,
        stage_id=plan.STAGE1_ID,
        job_id=plan.STAGE1_JOB_IDS[0],
        output_dir=output,
    )
    for index in job["work_hand_indices"]:
        identity = f"{plan.STAGE1_ID}|{plan.STAGE1_JOB_IDS[0]}|{index}"
        local.record_local_upload(
            package_dir=package,
            output_dir=output,
            hand_index=index,
            payload={
                "schema": "diagnostic_fake_solver_output_v1",
                "identity": identity,
                "portable_decision_sha256": hashlib.sha256(
                    identity.encode("ascii")
                ).hexdigest(),
            },
        )
    local.complete_local_job(package_dir=package, output_dir=output)
    received = root / "received"
    local.receive_stage(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
        job_outputs={plan.STAGE1_JOB_IDS[0]: output},
        destination=received,
    )
    return received


def _result(job_id: str, sequence: int) -> dict[str, object]:
    identity = f"{job_id}|{sequence}"
    return {
        "schema": "diagnostic_t3_result_v1",
        "action_key": (
            f"rak1:{sequence:013x}:0000000000000:"
            "0000000000000:0000000000000"
        ),
        "q_milli": sequence,
        "portable_sha256": hashlib.sha256(identity.encode("ascii")).hexdigest(),
    }


def _record(uri: str, content: dict[str, object]) -> dict[str, object]:
    return {
        "uri": uri,
        "generation": 1,
        "content_sha256": subject.canonical_sha256(content),
        "content": content,
    }


def _job_objects(
    preview: dict[str, object],
    job: dict[str, object],
    *,
    complete: bool,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    uploads: list[dict[str, object]] = []
    heartbeats: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    count = len(job["work_hand_indices"]) if complete else 1
    for sequence in range(1, count + 1):
        upload = subject.build_upload(
            preview,
            job_id=job["job_id"],
            sequence=sequence,
            result=_result(job["job_id"], sequence),
        )
        uploads.append(upload)
        heartbeat = subject.build_heartbeat(
            preview,
            job_id=job["job_id"],
            uploads=uploads,
        )
        heartbeats.append(heartbeat)
        records.append(_record(job["upload_uris"][sequence - 1], upload))
        records.append(_record(job["heartbeat_uris"][sequence - 1], heartbeat))
    done = None
    if complete:
        done = subject.build_done(
            preview,
            job_id=job["job_id"],
            uploads=uploads,
            heartbeats=heartbeats,
        )
        records.append(_record(job["done_uri"], done))
    return records, done


def _snapshot(
    preview: dict[str, object],
    objects: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema": subject.SNAPSHOT_SCHEMA,
        "preview_sha256": subject.canonical_sha256(preview),
        "stage_id": preview["stage_id"],
        "run_name": preview["run_name"],
        "objects": objects,
        "cloud_query_performed": False,
        "cloud_write_performed": False,
    }


def test_exact_stage_preview_cost_and_production_separation(
    package: Path, stage1_receive: Path
) -> None:
    stage1 = subject.build_preview(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
    )
    stage2 = subject.build_preview(
        package_dir=package,
        stage_id=plan.STAGE2_ID,
        prerequisite_receive_dir=stage1_receive,
    )
    assert stage1["selected_job_ids"] == list(plan.STAGE1_JOB_IDS)
    assert stage2["selected_job_ids"] == list(plan.STAGE2_JOB_IDS)
    assert [stage1["vm_count"], stage2["vm_count"]] == [1, 2]
    assert stage1["cost_guard"]["initial_estimated_max_compute_usd"] == pytest.approx(
        0.665
    )
    assert stage2["cost_guard"]["all_attempts_estimated_max_compute_usd"] == pytest.approx(
        2.66
    )
    audit = subject.audit(package_dir=package)
    assert audit[
        "combined_all_attempts_estimated_max_compute_usd"
    ] == pytest.approx(3.99)
    assert audit["diagnostic_compute_cap_usd"] == 4.0
    assert audit["remote_collision_state_queried"] is False
    assert audit["launch_ready"] is False
    assert stage1["remote_manifest"]["prefix"] != stage2["remote_manifest"]["prefix"]
    assert plan.EXPECTED_REARM2_RUN_NAME not in stage1["remote_manifest"]["prefix"]
    assert stage1["remote_manifest"]["freshness_preflight"] == {
        "entire_stage_prefix_must_be_empty": True,
        "conditional_create_generation_match": 0,
        "unknown_object_is_fatal": True,
        "permission_or_query_error_is_fatal": True,
        "performed_by_this_dry_run_module": False,
    }
    assert all(
        job["success_policy"]["self_delete_requested_after_done_validation"]
        and job["failure_policy"]["preserve_vm_for_diagnosis"]
        for job in stage1["jobs"] + stage2["jobs"]
    )
    assert all(value == "0" for value in (
        stage1["jobs"][0]["metadata"]["CLOUD_LAUNCH_AUTHORIZED"],
        stage1["jobs"][0]["metadata"]["GCLOUD_INVOCATION_AUTHORIZED"],
    ))


def test_stage2_binds_actual_validated_stage1_receipt(
    package: Path, stage1_receive: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="actual validated stage1"):
        subject.build_preview(package_dir=package, stage_id=plan.STAGE2_ID)
    forged = tmp_path / "forged"
    forged.mkdir()
    (forged / local.RECEIPT_NAME).write_text(
        '{"receipt_sha256":"' + "0" * 64 + '"}\n',
        encoding="utf-8",
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        subject.build_preview(
            package_dir=package,
            stage_id=plan.STAGE2_ID,
            prerequisite_receive_dir=forged,
        )
    preview = subject.build_preview(
        package_dir=package,
        stage_id=plan.STAGE2_ID,
        prerequisite_receive_dir=stage1_receive,
    )
    binding = preview["stage_token"]["prerequisite_stage1_receive"]
    receipt = local.validate_received_stage(
        package_dir=package,
        receive_dir=stage1_receive,
        stage_id=plan.STAGE1_ID,
    )
    assert binding["receipt_sha256"] == local.sha256_file(
        stage1_receive / local.RECEIPT_NAME
    )
    assert binding["job_record_aggregate_sha256"] == receipt[
        "job_record_aggregate_sha256"
    ]


def test_preview_tamper_collision_price_and_exact_result_schema_fail_closed(
    package: Path,
) -> None:
    preview = subject.build_preview(package_dir=package, stage_id=plan.STAGE1_ID)
    tampered = deepcopy(preview)
    tampered["selected_job_ids"].append("reference-shard-00")
    with pytest.raises(ValueError, match="preview changed"):
        subject.validate_preview(tampered, package_dir=package)
    collision = deepcopy(preview)
    collision["jobs"][0]["done_uri"] = collision["jobs"][0]["upload_uris"][0]
    with pytest.raises(ValueError):
        subject.validate_preview(collision, package_dir=package)
    observed = subject.build_preview(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
        observed_spot_price=0.55932,
    )
    assert (
        observed["cost_guard"]["observed_spot_price_usd_per_vm_hour"]
        == 0.55932
    )
    with pytest.raises(ValueError, match="price"):
        subject.build_preview(
            package_dir=package,
            stage_id=plan.STAGE1_ID,
            observed_spot_price=0.570001,
        )
    with pytest.raises(ValueError, match="frozen project"):
        subject.build_preview(
            package_dir=package,
            stage_id=plan.STAGE1_ID,
            project="other-project",
        )
    result = _result(plan.STAGE1_JOB_IDS[0], 1)
    result["serialized_payload"] = '{"villain_discards":["As"]}'
    with pytest.raises(ValueError, match="keys changed"):
        subject.build_upload(
            preview,
            job_id=plan.STAGE1_JOB_IDS[0],
            sequence=1,
            result=result,
        )


def test_resume_done_order_receive_and_failure_preservation(package: Path) -> None:
    preview = subject.build_preview(package_dir=package, stage_id=plan.STAGE1_ID)
    job = preview["jobs"][0]
    partial, _done = _job_objects(preview, job, complete=False)
    checked = subject.validate_snapshot(preview, _snapshot(preview, partial))
    assert checked["job_states"][0]["state"] == "resumable_same_attempt"
    assert subject.canonical_bytes(checked) == subject.canonical_bytes(
        subject.validate_snapshot(preview, _snapshot(preview, partial))
    )

    failure = subject.build_failure(
        preview,
        job_id=job["job_id"],
        failure_code="watchdog_timeout",
        completed_sequences=1,
    )
    failure_checked = subject.validate_snapshot(
        preview,
        _snapshot(preview, partial + [_record(job["failure_uri"], failure)]),
    )
    assert failure_checked["job_states"][0]["state"] == "failure_vm_preserved"
    assert failure["self_delete_requested"] is False
    assert failure["done_published"] is False
    assert failure["bounded_shutdown_required"] is True
    assert failure["shutdown_deadline_seconds"] == 4200

    complete, done = _job_objects(preview, job, complete=True)
    assert done is not None
    receive = subject.build_receive(preview, done_records=[done])
    complete.append(_record(preview["remote_manifest"]["receive_uri"], receive))
    final = subject.validate_snapshot(preview, _snapshot(preview, complete))
    assert final["job_states"][0]["state"] == "success_self_delete_requested"
    assert final["receive_validated"] is True

    forged_done = deepcopy(done)
    forged_done["serialized_hidden_payload"] = '{"opponent_discard":["As"]}'
    with pytest.raises(ValueError, match="DONE keys changed"):
        subject.build_receive(preview, done_records=[forged_done])

    early_done = deepcopy(done)
    early_done["completed_hand_indices"] = early_done["completed_hand_indices"][:-1]
    with pytest.raises(ValueError, match="DONE"):
        subject.validate_snapshot(
            preview,
            _snapshot(preview, partial + [_record(job["done_uri"], early_done)]),
        )


def test_stale_identity_unknown_object_and_duplicate_collision_are_rejected(
    package: Path,
) -> None:
    preview = subject.build_preview(package_dir=package, stage_id=plan.STAGE1_ID)
    job = preview["jobs"][0]
    partial, _ = _job_objects(preview, job, complete=False)
    stale = deepcopy(partial)
    stale[0]["content"]["attempt_id"] = "old-attempt"
    stale[0]["content_sha256"] = subject.canonical_sha256(stale[0]["content"])
    with pytest.raises(ValueError, match="upload content"):
        subject.validate_snapshot(preview, _snapshot(preview, stale))
    unknown = deepcopy(partial)
    unknown[0]["uri"] += ".unknown"
    with pytest.raises(ValueError, match="collision or stale"):
        subject.validate_snapshot(preview, _snapshot(preview, unknown))
    duplicate = partial + [deepcopy(partial[0])]
    with pytest.raises(ValueError, match="collision or stale"):
        subject.validate_snapshot(preview, _snapshot(preview, duplicate))


def test_adapter_preview_never_calls_subprocess_or_changes_profiles(
    package: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = plan.DEFAULT_CURRENT_PROFILE
    before = local.sha256_file(profile)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("dry-run adapter called subprocess/cloud path")

    monkeypatch.setattr(local.subprocess, "run", forbidden)
    preview = subject.build_preview(package_dir=package, stage_id=plan.STAGE1_ID)
    subject.validate_snapshot(preview, _snapshot(preview, []))
    assert local.sha256_file(profile) == before == plan.EXPECTED_CURRENT_PROFILE_FILE_SHA256
    assert not hasattr(subject, "launch")
    assert not hasattr(subject, "authorize_launch")
    assert not hasattr(subject, "write_claim")
    assert preview["capabilities"]["vm_create_authorized"] is False
    assert preview["capabilities"]["object_write_authorized"] is False
