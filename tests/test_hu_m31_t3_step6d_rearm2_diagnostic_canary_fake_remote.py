from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from ofc_regular import hu_m31_t3_step6d_full100_spot_v1 as production_full100
from ofc_regular import (
    hu_m31_t3_step6d_performance_lock_rearm2_spot as production_rearm2,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_fake_remote as subject,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_local as local,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as plan,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_vm_adapter as adapter,
)


@pytest.fixture(scope="module")
def package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    target = tmp_path_factory.mktemp("diag-fake-remote") / "package"
    local.build_local_package(output_dir=target)
    return target


@pytest.fixture(scope="module")
def local_stage1_receive(
    package: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    root = tmp_path_factory.mktemp("diag-fake-local-stage1")
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


def _upload_all(
    stage: subject.FakeRemoteStage,
    *,
    job_id: str,
    start_sequence: int = 1,
) -> None:
    job = next(row for row in stage.preview["jobs"] if row["job_id"] == job_id)
    for sequence in range(start_sequence, len(job["work_hand_indices"]) + 1):
        stage.upload_sequence(
            job_id=job_id,
            sequence=sequence,
            result=_result(job_id, sequence),
        )


def _complete_stage1(
    preview: dict[str, object],
    *,
    resume_after: int | None,
) -> tuple[dict[str, object], subject.ControllerReceiptProof]:
    stage = subject.FakeRemoteStage(
        preview,
        store=subject.InMemoryGenerationStore(),
    )
    job_id = plan.STAGE1_JOB_IDS[0]
    stage.start_initial_exact()
    if resume_after is not None:
        for sequence in range(1, resume_after + 1):
            stage.upload_sequence(
                job_id=job_id,
                sequence=sequence,
                result=_result(job_id, sequence),
            )
        stage.interrupt_for_resume(job_id=job_id)
        stage.controller_mark_vm_absent(job_id=job_id)
        stage.resume_exact([job_id])
        # A restarted worker first replays the immutable completed prefix.
        for sequence in range(1, resume_after + 1):
            stage.upload_sequence(
                job_id=job_id,
                sequence=sequence,
                result=_result(job_id, sequence),
            )
        _upload_all(stage, job_id=job_id, start_sequence=resume_after + 1)
    else:
        _upload_all(stage, job_id=job_id)
    stage.publish_done(job_id=job_id)
    stage.controller_mark_vm_absent(job_id=job_id)
    receive = stage.receive()
    return receive, stage.controller_proof()


def test_generation_zero_store_prefix_and_collision_semantics() -> None:
    store = subject.InMemoryGenerationStore()
    uri = "gs://diagnostic-test/prefix/object.json"
    content = {"schema": "test_v1", "value": 1}

    first = store.put_once(uri, content)
    second = store.put_once(uri, content)

    assert first.created is True
    assert second.created is False
    assert first.generation == second.generation == 1
    assert store.read(uri) == content
    assert store.list_prefix("gs://diagnostic-test/prefix") == [uri]
    with pytest.raises(FileExistsError, match="different bytes"):
        store.put_once(uri, {"schema": "test_v1", "value": 2})


def test_preview_zero_hash_outside_prefix_and_rehashed_metadata_fail_closed(
    package: Path,
) -> None:
    preview = adapter.build_preview(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
    )
    zero = deepcopy(preview)
    zero["package_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="nonzero"):
        subject.FakeRemoteStage(
            zero,
            store=subject.InMemoryGenerationStore(),
        )

    outside = deepcopy(preview)
    rogue_uri = "gs://outside-bucket/escaped/hand_000.json"
    outside["jobs"][0]["upload_uris"][0] = rogue_uri
    outside["remote_manifest"]["job_prefixes"][0]["upload_uris"][0] = rogue_uri
    outside["remote_manifest_sha256"] = adapter.canonical_sha256(
        outside["remote_manifest"]
    )
    with pytest.raises(ValueError, match="metadata, or URI"):
        subject.FakeRemoteStage(
            outside,
            store=subject.InMemoryGenerationStore(),
        )

    identity = deepcopy(preview)
    identity["jobs"][0]["metadata"]["INSTANCE_NAME"] = "attacker-instance"
    identity["jobs"][0]["metadata_sha256"] = adapter.canonical_sha256(
        identity["jobs"][0]["metadata"]
    )
    with pytest.raises(ValueError, match="metadata, or URI"):
        subject.FakeRemoteStage(
            identity,
            store=subject.InMemoryGenerationStore(),
        )


@pytest.mark.parametrize(
    "tamper",
    (
        "outer_vm_count",
        "cost_guard_vm_count",
        "work_hand_zero",
        "root_hand_zero",
        "generation_match_zero",
        "false_capability",
        "true_policy",
    ),
)
def test_preview_bool_for_numeric_or_boolean_contract_field_fails_closed(
    package: Path,
    tamper: str,
) -> None:
    preview = deepcopy(
        adapter.build_preview(
            package_dir=package,
            stage_id=plan.STAGE1_ID,
        )
    )
    if tamper == "outer_vm_count":
        preview["vm_count"] = True
    elif tamper == "cost_guard_vm_count":
        preview["cost_guard"]["vm_count"] = True
        preview["cost_guard_sha256"] = adapter.canonical_sha256(
            preview["cost_guard"]
        )
    elif tamper == "work_hand_zero":
        assert preview["jobs"][0]["work_hand_indices"][0] == 0
        preview["jobs"][0]["work_hand_indices"][0] = False
    elif tamper == "root_hand_zero":
        assert preview["jobs"][0]["root_records"][0]["hand_index"] == 0
        preview["jobs"][0]["root_records"][0]["hand_index"] = False
    elif tamper == "generation_match_zero":
        preview["remote_manifest"]["freshness_preflight"][
            "conditional_create_generation_match"
        ] = False
        preview["remote_manifest_sha256"] = adapter.canonical_sha256(
            preview["remote_manifest"]
        )
    elif tamper == "false_capability":
        preview["capabilities"]["vm_create_authorized"] = 0
    elif tamper == "true_policy":
        preview["jobs"][0]["success_policy"]["publish_done_last"] = 1
    else:  # pragma: no cover - the frozen parameter set is exhaustive.
        raise AssertionError(f"unknown tamper case: {tamper}")

    with pytest.raises(ValueError, match="exact|boolean"):
        subject.FakeRemoteStage(
            preview,
            store=subject.InMemoryGenerationStore(),
        )


def test_runtime_bool_attempt_and_upload_sequence_fail_closed(
    package: Path,
) -> None:
    preview = adapter.build_preview(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
    )
    stage = subject.FakeRemoteStage(
        preview,
        store=subject.InMemoryGenerationStore(),
    )
    job_id = plan.STAGE1_JOB_IDS[0]

    with pytest.raises(ValueError, match="zero and one"):
        stage.start_attempt(job_id=job_id, attempt_index=True)
    stage.start_initial_exact()
    with pytest.raises(ValueError, match="exact next"):
        stage.upload_sequence(
            job_id=job_id,
            sequence=True,
            result=_result(job_id, 1),
        )


def test_prefix_empty_sentinel_auth_claim_and_production_launchers_are_isolated(
    package: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("fake transport reached a production all20 path")

    monkeypatch.setattr(production_full100, "launch_jobs", forbidden)
    monkeypatch.setattr(production_full100, "_launch_selected", forbidden)
    monkeypatch.setattr(production_rearm2, "launch_jobs", forbidden)
    preview = adapter.build_preview(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
    )

    stale_store = subject.InMemoryGenerationStore()
    stale_store.put_once(
        preview["remote_manifest"]["prefix"] + "/logs/stale.json",
        {"schema": "stale_v1"},
    )
    with pytest.raises(FileExistsError, match="not exactly empty"):
        subject.FakeRemoteStage(preview, store=stale_store)
    assert not stale_store.contains(
        preview["remote_manifest"]["prefix"]
        + "/"
        + subject.SENTINEL_NAME
    )

    store = subject.InMemoryGenerationStore()
    stage = subject.FakeRemoteStage(preview, store=store)
    assert stage.authorization["schema"] == subject.SIMULATED_AUTHORIZATION_SCHEMA
    assert stage.claim["schema"] == subject.SIMULATED_CLAIM_SCHEMA
    assert stage.authorization["cloud_launch_authorized"] is False
    assert stage.claim["vm_create_authorized"] is False
    assert store.read(stage.authorization_uri) == stage.authorization
    assert store.read(stage.claim_uri) == stage.claim
    assert not hasattr(subject, "launch")
    assert not hasattr(subject, "authorize_launch")


def test_stage2_requires_exact_controller_receipt_and_vm_absence(
    package: Path,
    local_stage1_receive: Path,
) -> None:
    stage1_preview = adapter.build_preview(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
    )
    _receive, controller = _complete_stage1(stage1_preview, resume_after=None)
    stage2_preview = adapter.build_preview(
        package_dir=package,
        stage_id=plan.STAGE2_ID,
        prerequisite_receive_dir=local_stage1_receive,
    )

    with pytest.raises(ValueError, match="controller-proven"):
        subject.FakeRemoteStage(
            stage2_preview,
            store=subject.InMemoryGenerationStore(),
        )
    tampered = deepcopy(controller.content)
    tampered["all_vms_absent"] = False
    with pytest.raises(ValueError, match="opaque stage1"):
        subject.FakeRemoteStage(
            stage2_preview,
            store=subject.InMemoryGenerationStore(),
            prerequisite_stage1_controller_receipt=tampered,
        )
    forged_proof = subject.ControllerReceiptProof(
        receipt_bytes=subject.canonical_bytes(tampered),
        _source_stage=controller._source_stage,
        _issuer_nonce=controller._issuer_nonce,
    )
    with pytest.raises(ValueError, match="provenance changed"):
        subject.FakeRemoteStage(
            stage2_preview,
            store=subject.InMemoryGenerationStore(),
            prerequisite_stage1_controller_receipt=forged_proof,
        )
    opened = subject.FakeRemoteStage(
        stage2_preview,
        store=subject.InMemoryGenerationStore(),
        prerequisite_stage1_controller_receipt=controller,
    )
    assert opened.authorization[
        "prerequisite_stage1_controller_receipt_sha256"
    ] == subject.canonical_sha256(controller.content)


def test_attempt_zero_one_exact_partial_resume_and_third_attempt_rejected(
    package: Path,
    local_stage1_receive: Path,
) -> None:
    stage1_preview = adapter.build_preview(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
    )
    _receive, controller = _complete_stage1(stage1_preview, resume_after=None)
    preview = adapter.build_preview(
        package_dir=package,
        stage_id=plan.STAGE2_ID,
        prerequisite_receive_dir=local_stage1_receive,
    )
    stage = subject.FakeRemoteStage(
        preview,
        store=subject.InMemoryGenerationStore(),
        prerequisite_stage1_controller_receipt=controller,
    )
    stage.start_initial_exact()
    candidate, reference = plan.STAGE2_JOB_IDS
    _upload_all(stage, job_id=candidate)
    stage.publish_done(job_id=candidate)
    stage.controller_mark_vm_absent(job_id=candidate)
    stage.upload_sequence(
        job_id=reference,
        sequence=1,
        result=_result(reference, 1),
    )
    stage.interrupt_for_resume(job_id=reference)
    stage.controller_mark_vm_absent(job_id=reference)

    with pytest.raises(ValueError, match="exact ordered incomplete"):
        stage.resume_exact([candidate, reference])
    resumed = stage.resume_exact([reference])
    assert [row["attempt_index"] for row in resumed] == [1]
    with pytest.raises(ValueError, match="limited to zero and one"):
        stage.start_attempt(job_id=reference, attempt_index=2)
    with pytest.raises(ValueError, match="order or reuse"):
        stage.start_attempt(job_id=reference, attempt_index=1)


def test_done_is_last_read_back_and_receive_waits_for_vm_absence(
    package: Path,
) -> None:
    preview = adapter.build_preview(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
    )
    store = subject.InMemoryGenerationStore()
    stage = subject.FakeRemoteStage(preview, store=store)
    job_id = plan.STAGE1_JOB_IDS[0]
    job = preview["jobs"][0]
    stage.start_initial_exact()
    _upload_all(stage, job_id=job_id)
    with pytest.raises(ValueError, match="requires DONE"):
        stage.receive()

    done = stage.publish_done(job_id=job_id)
    result_creates = [
        row
        for row in store.events
        if row["operation"] == "conditional_create_generation_match_zero"
        and str(row["uri"]).startswith(job["metadata"]["RESULT_PREFIX"])
    ]
    assert result_creates[-1]["uri"] == job["done_uri"]
    assert any(
        row["operation"] == "readback" and row["uri"] == job["done_uri"]
        for row in store.events
    )
    assert store.read(job["done_uri"]) == done
    with pytest.raises(ValueError, match="controller-proven absence"):
        stage.receive()

    stopped = stage.controller_mark_vm_absent(job_id=job_id)
    assert stopped["controller_vm_absent"] is True
    receive = stage.receive()
    assert receive["diagnostic_only"] is True
    assert stage.validate_data_state()["receive_validated"] is True
    assert stage.controller_receipt()["all_vms_absent"] is True


def test_failure_record_has_no_done_and_requires_bounded_shutdown(
    package: Path,
) -> None:
    preview = adapter.build_preview(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
    )
    store = subject.InMemoryGenerationStore()
    stage = subject.FakeRemoteStage(preview, store=store)
    job_id = plan.STAGE1_JOB_IDS[0]
    job = preview["jobs"][0]
    stage.start_initial_exact()
    stage.upload_sequence(
        job_id=job_id,
        sequence=1,
        result=_result(job_id, 1),
    )
    failure = stage.publish_failure(
        job_id=job_id,
        failure_code="watchdog_timeout",
    )
    assert store.contains(job["failure_uri"])
    assert not store.contains(job["done_uri"])
    assert failure["done_published"] is False
    assert failure["self_delete_requested"] is False
    assert failure["bounded_shutdown_required"] is True
    assert stage.validate_data_state()["job_states"][0]["state"] == (
        "failure_vm_preserved"
    )
    with pytest.raises(ValueError, match="requires DONE"):
        stage.receive()
    stopped = stage.controller_mark_vm_absent(job_id=job_id)
    assert stopped["shutdown_deadline_seconds"] == 4200
    assert stopped["vm_present"] is False


@pytest.mark.parametrize(
    "fault_after",
    ["before_upload", "after_upload", "after_heartbeat"],
)
def test_fault_injection_never_publishes_done(
    package: Path,
    fault_after: str,
) -> None:
    preview = adapter.build_preview(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
    )
    store = subject.InMemoryGenerationStore()
    stage = subject.FakeRemoteStage(preview, store=store)
    job_id = plan.STAGE1_JOB_IDS[0]
    job = preview["jobs"][0]
    stage.start_initial_exact()
    with pytest.raises(RuntimeError, match="injected failure"):
        stage.upload_sequence(
            job_id=job_id,
            sequence=1,
            result=_result(job_id, 1),
            fault_after=fault_after,
        )
    assert not store.contains(job["done_uri"])
    assert not store.contains(preview["remote_manifest"]["receive_uri"])


def test_after_upload_crash_recovers_pending_step_on_exact_attempt_one(
    package: Path,
) -> None:
    preview = adapter.build_preview(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
    )
    store = subject.InMemoryGenerationStore()
    stage = subject.FakeRemoteStage(preview, store=store)
    job_id = plan.STAGE1_JOB_IDS[0]
    job = preview["jobs"][0]
    stage.start_initial_exact()
    with pytest.raises(RuntimeError, match="after upload"):
        stage.upload_sequence(
            job_id=job_id,
            sequence=1,
            result=_result(job_id, 1),
            fault_after="after_upload",
        )
    upload_record = store.record(job["upload_uris"][0])
    interrupted = stage.interrupt_for_resume(job_id=job_id)
    assert interrupted["completed_sequences"] == 0
    assert interrupted["pending_upload_sequence"] == 1
    stage.controller_mark_vm_absent(job_id=job_id)
    stage.resume_exact([job_id])

    stage.upload_sequence(
        job_id=job_id,
        sequence=1,
        result=_result(job_id, 1),
    )
    assert store.record(job["upload_uris"][0])["generation"] == (
        upload_record["generation"]
    )
    _upload_all(stage, job_id=job_id, start_sequence=2)
    stage.publish_done(job_id=job_id)
    stage.controller_mark_vm_absent(job_id=job_id)
    assert stage.receive()["diagnostic_only"] is True


def test_full_upload_prefix_crash_attempt_one_publishes_done_without_reupload(
    package: Path,
) -> None:
    preview = adapter.build_preview(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
    )
    store = subject.InMemoryGenerationStore()
    stage = subject.FakeRemoteStage(preview, store=store)
    job_id = plan.STAGE1_JOB_IDS[0]
    job = preview["jobs"][0]
    stage.start_initial_exact()
    final_sequence = len(job["work_hand_indices"])
    for sequence in range(1, final_sequence):
        stage.upload_sequence(
            job_id=job_id,
            sequence=sequence,
            result=_result(job_id, sequence),
        )
    with pytest.raises(RuntimeError, match="after heartbeat"):
        stage.upload_sequence(
            job_id=job_id,
            sequence=final_sequence,
            result=_result(job_id, final_sequence),
            fault_after="after_heartbeat",
        )
    data_creates_before = [
        row
        for row in store.events
        if row["operation"] == "conditional_create_generation_match_zero"
        and (
            row["uri"] in job["upload_uris"]
            or row["uri"] in job["heartbeat_uris"]
        )
    ]
    interrupted = stage.interrupt_for_resume(job_id=job_id)
    assert interrupted["completed_sequences"] == final_sequence
    assert interrupted["pending_upload_sequence"] is None
    stage.controller_mark_vm_absent(job_id=job_id)
    stage.resume_exact([job_id])
    stage.publish_done(job_id=job_id)
    data_creates_after = [
        row
        for row in store.events
        if row["operation"] == "conditional_create_generation_match_zero"
        and (
            row["uri"] in job["upload_uris"]
            or row["uri"] in job["heartbeat_uris"]
        )
    ]
    assert data_creates_after == data_creates_before
    stage.controller_mark_vm_absent(job_id=job_id)
    assert stage.receive()["diagnostic_only"] is True


def test_rogue_object_is_fatal_and_invalid_prefix_does_not_mutate_terminal(
    package: Path,
) -> None:
    preview = adapter.build_preview(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
    )
    store = subject.InMemoryGenerationStore()
    stage = subject.FakeRemoteStage(preview, store=store)
    job_id = plan.STAGE1_JOB_IDS[0]
    stage.start_initial_exact()
    rogue_uri = preview["remote_manifest"]["prefix"] + "/rogue/object.json"
    store.put_once(rogue_uri, {"schema": "rogue_v1"})
    with pytest.raises(ValueError, match="unknown or rogue"):
        stage.validate_data_state()
    with pytest.raises(ValueError, match="unknown or rogue"):
        stage.interrupt_for_resume(job_id=job_id)
    assert stage._terminal[job_id] is None


def test_clean_and_resumed_semantic_receipts_are_byte_identical(
    package: Path,
) -> None:
    preview = adapter.build_preview(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
    )
    clean_receive, clean_controller = _complete_stage1(
        preview,
        resume_after=None,
    )
    resumed_receive, resumed_controller = _complete_stage1(
        preview,
        resume_after=3,
    )

    assert subject.canonical_bytes(clean_receive) == subject.canonical_bytes(
        resumed_receive
    )
    assert subject.canonical_bytes(
        clean_controller.content
    ) == subject.canonical_bytes(
        resumed_controller.content
    )
