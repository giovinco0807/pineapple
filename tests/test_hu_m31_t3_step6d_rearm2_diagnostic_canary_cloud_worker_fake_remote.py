from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_package as cloud_package,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_adapter as adapter,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_fake_remote as subject,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as plan,
)


@pytest.fixture(scope="module")
def package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    target = tmp_path_factory.mktemp("diag-worker-fake-remote") / "package"
    cloud_package.build_package(output_dir=target)
    return target


def _complete_job(
    stage: subject.FakeCloudWorkerStage, job_id: str
) -> dict[str, object]:
    stage.publish_remaining(job_id=job_id)
    done = stage.publish_done(job_id=job_id)
    stage.controller_mark_vm_absent(job_id=job_id)
    return done


def _complete_stage1(
    package: Path,
) -> tuple[
    subject.FakeCloudWorkerStage,
    dict[str, object],
    subject.ControllerReceiptProof,
]:
    stage = subject.FakeCloudWorkerStage(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
    )
    stage.start_initial_exact()
    _complete_job(stage, plan.STAGE1_JOB_IDS[0])
    receive = stage.receive()
    return stage, receive, stage.controller_proof()


def _created_uris(
    store: subject.InMemoryGenerationMatchStore,
) -> list[str]:
    return [
        event["uri"]
        for event in store.events
        if event["operation"] == "conditional_create_generation_match_zero"
    ]


def test_generation_match_zero_is_idempotent_and_collision_safe() -> None:
    store = subject.InMemoryGenerationMatchStore()
    uri = "gs://fake-r2diag/unit/object.json"
    value = {"schema": "unit_v1", "value": 7}

    first = store.put_once(uri, value)
    second = store.put_once(uri, value)

    assert first.created is True
    assert second.created is False
    assert first.generation == second.generation == 1
    assert store.read(uri) == value
    with pytest.raises(FileExistsError, match="different bytes"):
        store.put_once(uri, {"schema": "unit_v1", "value": 8})


def test_first_object_is_exact_empty_prefix_sentinel_and_auth_is_separate(
    package: Path,
) -> None:
    store = subject.InMemoryGenerationMatchStore()
    stage = subject.FakeCloudWorkerStage(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
        store=store,
    )
    prefix = stage.preview["remote_manifest"]["prefix"]
    objects = store.list_prefix(prefix)
    assert len(objects) == 1
    assert store.read(objects[0])["schema"] == subject.STAGE_SENTINEL_SCHEMA

    stage.start_initial_exact()
    authorization = [
        uri
        for uri in store.list_prefix(prefix)
        if uri.endswith("/fake-authorization.json")
    ]
    claims = [
        uri
        for uri in store.list_prefix(prefix)
        if "/fake-claims/" in uri
    ]
    assert len(authorization) == len(claims) == 1
    assert authorization[0] != claims[0]
    assert (
        store.read(authorization[0])["schema"]
        == subject.SIMULATION_AUTHORIZATION_SCHEMA
    )
    claim = store.read(claims[0])
    assert claim["schema"] == subject.SIMULATION_CLAIM_SCHEMA
    assert claim["simulation_only"] is True
    assert claim["real_vm_claim"] is False
    assert claim["cloud_launch_authorized"] is False


def test_nonempty_stage_prefix_fails_before_sentinel(package: Path) -> None:
    preview = adapter.build_preview(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
    )
    store = subject.InMemoryGenerationMatchStore()
    rogue = f"{preview['remote_manifest']['prefix']}/rogue-before-sentinel.json"
    store.put_once(rogue, {"schema": "rogue_v1"})
    created_before = list(_created_uris(store))

    with pytest.raises(FileExistsError, match="exactly empty"):
        subject.FakeCloudWorkerStage(
            package_dir=package,
            stage_id=plan.STAGE1_ID,
            store=store,
        )
    assert _created_uris(store) == created_before


@pytest.mark.parametrize("attempt", [False, 1.0, 2, -1])
def test_attempt_index_is_strict_and_bounded(
    package: Path, attempt: object
) -> None:
    stage = subject.FakeCloudWorkerStage(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
    )
    with pytest.raises(ValueError, match="attempt"):
        stage.start_attempt(
            job_id=plan.STAGE1_JOB_IDS[0],
            attempt_index=attempt,  # type: ignore[arg-type]
        )


def test_unknown_object_fails_before_any_worker_mutation(package: Path) -> None:
    stage = subject.FakeCloudWorkerStage(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
    )
    prefix = stage.preview["remote_manifest"]["prefix"]
    stage.store.put_once(
        f"{prefix}/results/unknown-tree-or-control.json",
        {"schema": "rogue_v1"},
    )
    created_before = list(_created_uris(stage.store))

    with pytest.raises(ValueError, match="unknown object"):
        stage.start_initial_exact()

    assert _created_uris(stage.store) == created_before
    assert not any("/fake-authorization.json" in uri for uri in created_before)
    assert not any("/fake-claims/" in uri for uri in created_before)


def test_pending_upload_attempt1_adds_only_missing_heartbeat(
    package: Path,
) -> None:
    stage = subject.FakeCloudWorkerStage(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
    )
    job_id = plan.STAGE1_JOB_IDS[0]
    stage.start_initial_exact()
    attempt0 = stage.preview
    job0 = attempt0["jobs"][0]
    stage.publish_upload(
        job_id=job_id,
        sequence=1,
        publish_heartbeat=False,
    )
    upload_uri = job0["upload_uris"][0]
    heartbeat_uri = job0["heartbeat_uris"][0]
    upload_generation = stage.store.record(upload_uri)["generation"]
    created_before_resume = _created_uris(stage.store)
    assert upload_uri in created_before_resume
    assert heartbeat_uri not in created_before_resume

    stage.interrupt(job_id=job_id)
    stage.controller_mark_vm_absent(job_id=job_id)
    attempt1 = stage.resume_attempt1([job_id])
    assert attempt1["attempt_index"] == 1
    assert attempt1["jobs"][0]["upload_uris"] == job0["upload_uris"]
    assert attempt1["jobs"][0]["heartbeat_uris"] == job0["heartbeat_uris"]
    assert attempt1["jobs"][0]["done_uri"] == job0["done_uri"]
    assert (
        attempt1["remote_manifest"]["attempt_control_prefix"]
        != attempt0["remote_manifest"]["attempt_control_prefix"]
    )

    stage.publish_remaining(job_id=job_id)
    created_after_resume = _created_uris(stage.store)
    assert created_after_resume.count(upload_uri) == 1
    assert created_after_resume.count(heartbeat_uri) == 1
    assert stage.store.record(upload_uri)["generation"] == upload_generation
    _complete_job(stage, job_id)
    receive = stage.receive()
    assert receive["transport_fixture_only"] is True
    assert receive["scientific_payload_present"] is False


def test_full_prefix_attempt1_publishes_only_done_not_artifacts(
    package: Path,
) -> None:
    stage = subject.FakeCloudWorkerStage(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
    )
    job_id = plan.STAGE1_JOB_IDS[0]
    stage.start_initial_exact()
    stage.publish_remaining(job_id=job_id)
    job = stage.preview["jobs"][0]
    scientific_uris = {
        *job["upload_uris"],
        *job["heartbeat_uris"],
        *(row["uri"] for row in job["tree_object_manifest"]),
    }
    scientific_creates_before = [
        uri for uri in _created_uris(stage.store) if uri in scientific_uris
    ]
    assert not stage.store.contains(job["done_uri"])

    stage.interrupt(job_id=job_id)
    stage.controller_mark_vm_absent(job_id=job_id)
    stage.resume_attempt1([job_id])
    done = stage.publish_done(job_id=job_id)

    scientific_creates_after = [
        uri for uri in _created_uris(stage.store) if uri in scientific_uris
    ]
    # The only new tree object is runner DONE.json; uploads, heartbeats, roots,
    # and source-hand fixtures retain their original generations.
    new_scientific = scientific_creates_after[
        len(scientific_creates_before) :
    ]
    assert len(new_scientific) == 1
    assert new_scientific[0].endswith("/DONE.json")
    assert _created_uris(stage.store).count(job["done_uri"]) == 1
    assert done["done_published_last"] is True
    lifecycle = stage._success_records[job_id]
    assert lifecycle["self_delete_requested"] is True
    assert lifecycle["preserve_vm_on_failure"] is False


def test_clean_and_full_prefix_resume_have_identical_done_and_receive(
    package: Path,
) -> None:
    clean = subject.FakeCloudWorkerStage(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
    )
    job_id = plan.STAGE1_JOB_IDS[0]
    clean.start_initial_exact()
    clean_done = _complete_job(clean, job_id)
    clean_receive = clean.receive()

    resumed = subject.FakeCloudWorkerStage(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
    )
    resumed.start_initial_exact()
    resumed.publish_remaining(job_id=job_id)
    resumed.interrupt(job_id=job_id)
    resumed.controller_mark_vm_absent(job_id=job_id)
    resumed.resume_attempt1([job_id])
    resumed_done = _complete_job(resumed, job_id)
    resumed_receive = resumed.receive()

    assert resumed_done == clean_done
    assert resumed_receive == clean_receive


def test_done_is_last_read_back_and_receive_requires_vm_absence(
    package: Path,
) -> None:
    stage = subject.FakeCloudWorkerStage(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
    )
    job_id = plan.STAGE1_JOB_IDS[0]
    stage.start_initial_exact()
    with pytest.raises(ValueError, match="full artifact"):
        stage.publish_done(job_id=job_id)
    stage.publish_remaining(job_id=job_id)
    done = stage.publish_done(job_id=job_id)
    job = stage.preview["jobs"][0]
    done_create_position = next(
        index
        for index, event in enumerate(stage.store.events)
        if event["operation"] == "conditional_create_generation_match_zero"
        and event["uri"] == job["done_uri"]
    )
    for uri in (
        *job["upload_uris"],
        *job["heartbeat_uris"],
        *(row["uri"] for row in job["tree_object_manifest"]),
    ):
        create_position = next(
            index
            for index, event in enumerate(stage.store.events)
            if event["operation"] == "conditional_create_generation_match_zero"
            and event["uri"] == uri
        )
        assert create_position < done_create_position
    assert stage.store.read(job["done_uri"]) == done
    with pytest.raises(ValueError, match="VM absence"):
        stage.receive()
    stage.controller_mark_vm_absent(job_id=job_id)
    assert stage.receive()["selected_job_ids"] == [job_id]


def test_direct_valid_done_cannot_bypass_success_lifecycle(
    package: Path,
) -> None:
    stage = subject.FakeCloudWorkerStage(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
    )
    job_id = plan.STAGE1_JOB_IDS[0]
    stage.start_initial_exact()
    stage.publish_remaining(job_id=job_id)
    job = stage.preview["jobs"][0]
    uploads, heartbeats, pending = stage._job_prefix_content(job)
    assert pending is False
    done = adapter.build_done(
        stage.preview,
        job_id=job_id,
        uploads=uploads,
        heartbeats=heartbeats,
    )
    stage.interrupt(job_id=job_id)
    stage.controller_mark_vm_absent(job_id=job_id)
    # Adversary writes a byte-valid DONE envelope without invoking
    # publish_done, so no success/self-delete lifecycle token exists.
    stage.store.put_once(job["done_uri"], done)

    with pytest.raises(ValueError, match="lifecycle ownership"):
        stage.receive()
    with pytest.raises(ValueError, match="receive"):
        stage.controller_proof()


def test_known_uri_preseed_cannot_be_adopted_by_lifecycle(package: Path) -> None:
    oracle = subject.FakeCloudWorkerStage(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
    )
    job_id = plan.STAGE1_JOB_IDS[0]
    oracle.start_initial_exact()
    _complete_job(oracle, job_id)
    oracle.receive()
    oracle_job = oracle.preview["jobs"][0]
    oracle_uris = {
        "authorization": oracle._authorization_uri(0),
        "claim": oracle._claim_uri(job_id, 0),
        "upload": oracle_job["upload_uris"][0],
        "tree": oracle_job["tree_object_manifest"][2]["uri"],
        "done": oracle_job["done_uri"],
        "lifecycle": oracle._lifecycle_uri(job_id, 0),
        "receive": oracle.preview["remote_manifest"]["receive_uri"],
    }

    for label, oracle_uri in oracle_uris.items():
        target = subject.FakeCloudWorkerStage(
            package_dir=package,
            stage_id=plan.STAGE1_ID,
        )
        target_job = target.preview["jobs"][0]
        target_uris = {
            "authorization": target._authorization_uri(0),
            "claim": target._claim_uri(job_id, 0),
            "upload": target_job["upload_uris"][0],
            "tree": target_job["tree_object_manifest"][2]["uri"],
            "done": target_job["done_uri"],
            "lifecycle": target._lifecycle_uri(job_id, 0),
            "receive": target.preview["remote_manifest"]["receive_uri"],
        }
        assert target_uris[label] == oracle_uri
        target.store.put_once(
            target_uris[label],
            oracle.store.read(oracle_uri),
        )
        created_before = list(_created_uris(target.store))
        with pytest.raises(ValueError, match="lifecycle ownership"):
            target.start_initial_exact()
        assert _created_uris(target.store) == created_before


def test_preseeded_second_job_claim_is_not_adopted_after_first_claim(
    package: Path,
) -> None:
    _stage1, _receive1, proof = _complete_stage1(package)
    oracle = subject.FakeCloudWorkerStage(
        package_dir=package,
        stage_id=plan.STAGE2_ID,
        stage1_proof=proof,
    )
    first_id, second_id = plan.STAGE2_JOB_IDS
    oracle.start_attempt(job_id=first_id, attempt_index=0)
    oracle_second_claim = oracle.start_attempt(
        job_id=second_id, attempt_index=0
    )

    target = subject.FakeCloudWorkerStage(
        package_dir=package,
        stage_id=plan.STAGE2_ID,
        stage1_proof=proof,
    )
    target.start_attempt(job_id=first_id, attempt_index=0)
    claim_uri = target._claim_uri(second_id, 0)
    target.store.put_once(claim_uri, oracle_second_claim)
    created_before = list(_created_uris(target.store))
    with pytest.raises(ValueError, match="lifecycle ownership"):
        target.start_attempt(job_id=second_id, attempt_index=0)
    assert _created_uris(target.store) == created_before


def test_failure_preserves_vm_then_bounded_controller_shutdown_and_no_done(
    package: Path,
) -> None:
    stage = subject.FakeCloudWorkerStage(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
    )
    job_id = plan.STAGE1_JOB_IDS[0]
    stage.start_initial_exact()
    with pytest.raises(ValueError, match="integer"):
        stage.publish_failure(job_id=job_id, shutdown_seconds=True)
    with pytest.raises(ValueError, match="maximum"):
        stage.publish_failure(
            job_id=job_id,
            shutdown_seconds=subject.MAX_CONTROLLER_SHUTDOWN_SECONDS + 1,
        )
    failure = stage.publish_failure(
        job_id=job_id,
        shutdown_seconds=subject.MAX_CONTROLLER_SHUTDOWN_SECONDS,
    )
    assert failure["self_delete_requested"] is False
    assert failure["preserve_vm_on_failure"] is True
    assert failure["controller_shutdown_bounded"] is True
    assert not stage.store.contains(stage.preview["jobs"][0]["done_uri"])
    stage.controller_mark_vm_absent(job_id=job_id)
    stage.resume_attempt1([job_id])
    _complete_job(stage, job_id)
    assert stage.receive()["transport_fixture_only"] is True


def test_third_attempt_and_wrong_resume_set_fail_closed(package: Path) -> None:
    stage = subject.FakeCloudWorkerStage(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
    )
    job_id = plan.STAGE1_JOB_IDS[0]
    stage.start_initial_exact()
    stage.interrupt(job_id=job_id)
    stage.controller_mark_vm_absent(job_id=job_id)
    with pytest.raises(ValueError, match="exact incomplete"):
        stage.resume_attempt1([])
    stage.resume_attempt1([job_id])
    with pytest.raises(ValueError, match="attempt1 can only follow"):
        stage.resume_attempt1([job_id])
    with pytest.raises(ValueError, match="attempt"):
        stage.start_attempt(job_id=job_id, attempt_index=2)


def test_stage2_partial_resume_selects_only_incomplete_job(package: Path) -> None:
    _stage1, _receive1, proof = _complete_stage1(package)
    stage2 = subject.FakeCloudWorkerStage(
        package_dir=package,
        stage_id=plan.STAGE2_ID,
        stage1_proof=proof,
    )
    candidate_id, reference_id = plan.STAGE2_JOB_IDS
    stage2.start_initial_exact()
    _complete_job(stage2, candidate_id)
    stage2.publish_upload(
        job_id=reference_id,
        sequence=1,
        publish_heartbeat=False,
    )
    stage2.interrupt(job_id=reference_id)
    stage2.controller_mark_vm_absent(job_id=reference_id)

    with pytest.raises(ValueError, match="exact incomplete"):
        stage2.resume_attempt1([])
    with pytest.raises(ValueError, match="exact incomplete"):
        stage2.resume_attempt1([candidate_id, reference_id])
    stage2.resume_attempt1([reference_id])
    assert stage2._vm[candidate_id]["state"] == "absent"
    assert stage2._vm[reference_id]["state"] == "active"
    _complete_job(stage2, reference_id)
    receive = stage2.receive()
    assert receive["selected_job_ids"] == [candidate_id, reference_id]
    proof2 = stage2.controller_proof()
    assert proof2.content["all_worker_vms_absent"] is True
    assert [
        row["attempt_index"]
        for row in proof2.content["success_lifecycle_records"]
    ] == [0, 1]


def test_stage2_requires_opaque_received_and_vm_absent_stage1_proof(
    package: Path,
) -> None:
    stage1, receive, proof = _complete_stage1(package)
    receipt = proof.content
    assert receipt["receive"] == receive
    assert receipt["all_worker_vms_absent"] is True
    assert receipt["success_lifecycle_records"][0][
        "self_delete_requested"
    ] is True
    for field in (
        "cloud_receive_performed",
        "performance_lock_evidence",
        "quality_evidence",
        "training_eligible",
        "promotion_evidence",
    ):
        assert receipt[field] is False

    stage2 = subject.FakeCloudWorkerStage(
        package_dir=package,
        stage_id=plan.STAGE2_ID,
        stage1_proof=proof,
    )
    assert stage2.preview["stage_id"] == plan.STAGE2_ID
    with pytest.raises(ValueError, match="opaque"):
        subject.FakeCloudWorkerStage(
            package_dir=package,
            stage_id=plan.STAGE2_ID,
            stage1_proof=receive,  # type: ignore[arg-type]
        )
    forged_bytes = bytearray(proof.receipt_bytes)
    forged_bytes[-2] = ord("0") if forged_bytes[-2] != ord("0") else ord("1")
    forged = subject.ControllerReceiptProof(
        receipt_bytes=bytes(forged_bytes),
        _source_stage=stage1,
        _issuer_nonce=stage1._issuer_nonce,
    )
    with pytest.raises(ValueError, match="bytes changed|canonical"):
        subject.FakeCloudWorkerStage(
            package_dir=package,
            stage_id=plan.STAGE2_ID,
            stage1_proof=forged,
        )
    stage1.store.put_once(
        f"{stage1.preview['remote_manifest']['prefix']}/rogue-after-proof.json",
        {"schema": "rogue_after_proof_v1"},
    )
    with pytest.raises(ValueError, match="unknown object"):
        subject.FakeCloudWorkerStage(
            package_dir=package,
            stage_id=plan.STAGE2_ID,
            stage1_proof=proof,
        )


def test_fake_lifecycle_never_calls_subprocess_or_system(
    package: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("production process API was called")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(os, "system", forbidden)

    stage = subject.FakeCloudWorkerStage(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
    )
    stage.start_initial_exact()
    _complete_job(stage, plan.STAGE1_JOB_IDS[0])
    receive = stage.receive()
    assert receive["cloud_receive_performed"] is False
    assert receive["remote_write_performed"] is False


def test_bool_publish_heartbeat_is_rejected(package: Path) -> None:
    stage = subject.FakeCloudWorkerStage(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
    )
    job_id = plan.STAGE1_JOB_IDS[0]
    stage.start_initial_exact()
    with pytest.raises(ValueError, match="strict boolean"):
        stage.publish_upload(
            job_id=job_id,
            sequence=1,
            publish_heartbeat=1,  # type: ignore[arg-type]
        )
