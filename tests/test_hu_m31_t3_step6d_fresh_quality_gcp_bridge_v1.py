from __future__ import annotations

import hashlib
import runpy
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from ofc_regular import hu_m31_t3_step6d_fresh_quality_gcp_bridge_v1 as subject
from ofc_regular import hu_m31_t3_step6d_fresh_quality_gcp_provider_v1 as provider
from ofc_regular import hu_m31_t3_step6d_fresh_quality_transport_v1 as transport
from ofc_regular import hu_m31_t3_step6d_fresh_quality_v1 as quality
from ofc_regular import hu_rl_c4_gcp_lifecycle as c4_gcp


ROOT = Path(__file__).resolve().parents[1]


def _record(path: Path, root: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


@pytest.fixture()
def frozen_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> dict[str, object]:
    staging = tmp_path / "staging"
    staging.mkdir()
    package_path = staging / "fresh-quality.zip"
    package_path.write_bytes(b"immutable-quality-package")
    source_path = staging / "source.zip"
    source_path.write_bytes(b"source")
    source_manifest_path = staging / "source.json"
    source_manifest_path.write_bytes(b"{}")
    wheels_path = staging / "wheels.zip"
    wheels_path.write_bytes(b"wheels")
    wheels_manifest_path = staging / "wheels.json"
    wheels_manifest_path.write_bytes(b"{}")
    startup_path = staging / "startup.sh"
    startup_path.write_bytes(b"#!/usr/bin/env bash\nexit 0\n")
    launch_path = staging / "launch.json"
    performance_path = tmp_path / "performance.json"
    performance_path.write_bytes(b'{"qualified":true}')
    profile_path = tmp_path / "ai_profiles.py"
    profile_path.write_bytes(b"# pinned profile fixture\n")
    profile_sha = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    monkeypatch.setattr(quality, "CURRENT_PROFILE_REGISTRY_SHA256", profile_sha)

    jobs = []
    ordinal = 0
    for wave_index, ids in enumerate(subject.WAVE_JOB_IDS):
        for job_id in ids:
            jobs.append(
                {
                    "job_id": job_id,
                    "phase": (
                        quality.PRIMARY_PHASE
                        if job_id.startswith("primary-")
                        else quality.CONFIRMATION_PHASE
                    ),
                    "wave_index": wave_index,
                    "package_job_path": f"jobs/{job_id}.json",
                    "job_manifest_sha256": hashlib.sha256(
                        job_id.encode("ascii")
                    ).hexdigest(),
                    "result_path": f"results/{job_id}.json",
                    "output_prefix": f"runs/test/jobs/{job_id}/attempt-a00",
                    "rayon_threads": 16,
                    "processes": 1,
                    "create_only": True,
                }
            )
            ordinal += 1
    authorization = {
        "schema": quality.PERFORMANCE_RECEIPT_SCHEMA,
        "status": "qualified",
        "decision": quality.QUALIFIED_DECISION,
        "receipt_sha256": "a" * 64,
        "performance_lock_qualified": True,
        "quality_pilot_authorized": True,
        "performance_lock_finalized": True,
        "one_shot_lock_consumed": True,
        "current_profile_changed": False,
    }
    run_name = "regular-hu-m31-t3-fqv1-test001"
    launch = {
        "schema": transport.LAUNCH_MANIFEST_SCHEMA,
        "status": "local_transport_ready_cloud_not_authorized",
        "run_name": run_name,
        "quality_package": _record(package_path, staging),
        "runtime_source": _record(source_path, staging),
        "runtime_source_manifest": _record(source_manifest_path, staging),
        "wheelhouse": _record(wheels_path, staging),
        "wheelhouse_manifest": _record(wheels_manifest_path, staging),
        "startup": _record(startup_path, staging),
        "candidate_library": {
            "path": "native/candidate.so",
            "sha256": "1" * 64,
            "bytes": 1,
        },
        "feature_encoder": {
            "path": "native/feature.so",
            "sha256": "2" * 64,
            "bytes": 1,
        },
        "allocation": {"processes": 1, "rayon_threads": 16},
        "wave_job_counts": [8, 7],
        "waves": [
            {"wave_index": i, "job_ids": list(ids), "job_count": len(ids)}
            for i, ids in enumerate(subject.WAVE_JOB_IDS)
        ],
        "jobs": jobs,
        "job_count": 15,
        "package_plan_sha256": "3" * 64,
        "package_root_seal_sha256": "4" * 64,
        "profile_registry_sha256": profile_sha,
        "transport_ready": True,
        "cloud_launch_authorized": False,
        "cloud_execution_started": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }
    launch_path.write_bytes(subject.canonical_bytes(launch))

    fake_package = {
        "plan": {"authorizing_performance_receipt": authorization},
        "seal": {},
        "jobs": [
            {
                "job_id": row["job_id"],
                "phase": row["phase"],
                "result_path": row["result_path"],
            }
            for row in jobs
        ],
        "package_root": str(tmp_path / "fake-package-root"),
    }
    monkeypatch.setattr(
        transport,
        "validate_local_launch_manifest",
        lambda value, *, staging_directory: deepcopy(dict(value)),
    )
    monkeypatch.setattr(
        transport,
        "extract_and_validate_package",
        lambda **_kwargs: deepcopy(fake_package),
    )
    monkeypatch.setattr(
        quality,
        "validate_plan_authorization",
        lambda plan, *, performance_receipt_path: deepcopy(dict(plan)),
    )
    plan = subject.build_gcp_plan(
        launch_manifest_path=launch_path,
        staging_directory=staging,
        performance_receipt_path=performance_path,
        profile_registry_path=profile_path,
        run_name=run_name,
        identity_salt="01" * 16,
        project="ofc-solver-485418",
        region="asia-northeast1",
        zone="asia-northeast1-b",
        bucket="pokerhu-ofc-solver-485418-training",
    )
    return {
        "plan": plan,
        "fake_package": fake_package,
        "tmp_path": tmp_path,
    }


def _object(name: str, payload: bytes, generation: int) -> dict[str, object]:
    return {
        "name": name,
        "generation": str(generation),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _lifecycle(
    plan: dict,
    ledger: dict,
    resume: dict,
    request: dict,
    *,
    failed: set[str] | None = None,
) -> tuple[dict, dict[str, bytes]]:
    failed = failed or set()
    expected = {row["job_id"]: row for row in request["expected_result_objects"]}
    payloads: dict[str, bytes] = {}
    rows = []
    generation = 1
    for selected in resume["selected_attempts"]:
        job_id = selected["job_id"]
        if job_id in failed:
            rows.append(
                {
                    "job_id": job_id,
                    "attempt_id": selected["attempt_id"],
                    "instance_id": selected["instance_id"],
                    "status": "failed",
                    "result_object": None,
                    "done_object": None,
                    "task_objects": [],
                }
            )
            continue
        result_payload = subject.canonical_bytes(
            {"schema": "test-result", "job_id": job_id}
        )
        done_payload = subject.canonical_bytes(
            {
                "schema": transport.DONE_SCHEMA,
                "job_id": job_id,
                "task_record_aggregate_sha256": "5" * 64,
            }
        )
        task_payload = subject.canonical_bytes(
            {"schema": transport.TASK_SCHEMA, "job_id": job_id}
        )
        result = _object(expected[job_id]["result_object"], result_payload, generation)
        generation += 1
        task_count = 10 if job_id.startswith("primary-") else 2
        tasks = []
        for task_index in range(task_count):
            task = _object(
                expected[job_id]["task_object_prefix"]
                + f"root_{task_index:03d}.json",
                task_payload,
                generation,
            )
            generation += 1
            tasks.append(task)
            payloads[task["name"]] = task_payload
        done = _object(expected[job_id]["done_object"], done_payload, generation)
        generation += 1
        payloads[result["name"]] = result_payload
        payloads[done["name"]] = done_payload
        rows.append(
            {
                "job_id": job_id,
                "attempt_id": selected["attempt_id"],
                "instance_id": selected["instance_id"],
                "status": "ready",
                "result_object": result,
                "done_object": done,
                "task_objects": tasks,
            }
        )
    core = {
        "schema": subject.LIFECYCLE_SCHEMA,
        "status": "exact_wave_terminal_cleanup_and_receiver_handoff_ready",
        "plan_sha256": plan["plan_sha256"],
        "ledger_sha256": ledger["ledger_sha256"],
        "resume_sha256": resume["resume_sha256"],
        "request_sha256": request["request_sha256"],
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_index": resume["resume_wave_index"],
        "attempt_rows": rows,
        "launch_create_only": True,
        "one_vm_per_job": True,
        "unlisted_vm_created": 0,
        "wildcard_delete_used": False,
        "unrelated_resource_touched": False,
        "owned_compute_absent": True,
        "worker_iam_removed": True,
        "receiver_handoff_ready": True,
        "current_profile_changed": False,
    }
    return {**core, "receipt_sha256": subject.canonical_sha256(core)}, payloads


class _Reader:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.generations = {
            name: str(index)
            for index, name in enumerate(payloads, start=1)
        }

    def read_object(self, *, bucket: str, object_name: str) -> dict:
        del bucket
        raw = self.payloads[object_name]
        return {
            "name": object_name,
            "generation": self.generations[object_name],
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "payload": raw,
        }


def test_plan_is_exactly_8_plus_7_and_two_attempts(
    frozen_plan: dict[str, object],
) -> None:
    plan = frozen_plan["plan"]
    assert [row["job_count"] for row in plan["waves"]] == [8, 7]
    assert plan["machine_contract"]["max_concurrent_vms"] == 8
    assert len(plan["jobs"]) == 15
    assert all(
        [attempt["attempt_id"] for attempt in job["attempts"]] == ["a00", "a01"]
        for job in plan["jobs"]
    )
    assert len(
        {
            attempt["instance_id"]
            for job in plan["jobs"]
            for attempt in job["attempts"]
        }
    ) == 30
    assert plan["current_profile_changed"] is False
    assert plan["reused_safety_contract"]["full100_science_plan_reused"] is False


def test_resume_enforces_earliest_wave_and_attempt_limit(
    frozen_plan: dict[str, object],
) -> None:
    plan = frozen_plan["plan"]
    ledger = subject.initial_attempt_ledger(plan)
    resume = subject.build_resume_plan(plan, ledger)
    assert resume["resume_wave_index"] == 0
    assert len(resume["selected_attempts"]) == 8
    assert {row["attempt_id"] for row in resume["selected_attempts"]} == {"a00"}
    request = subject.build_wave_request(plan, ledger, resume)
    lifecycle, _ = _lifecycle(
        plan,
        ledger,
        resume,
        request,
        failed={row["job_id"] for row in resume["selected_attempts"]},
    )
    rows = lifecycle["attempt_rows"]
    transition_core = {
        "sequence": 1,
        "previous_transition_sha256": None,
        "wave_index": 0,
        "resume_sha256": resume["resume_sha256"],
        "lifecycle_receipt_sha256": lifecycle["receipt_sha256"],
        "receive_receipt_sha256": "6" * 64,
        "attempt_rows": rows,
        "owned_compute_absent": True,
        "worker_iam_removed": True,
    }
    transition = {
        **transition_core,
        "transition_sha256": subject.canonical_sha256(transition_core),
    }
    ledger_core = {
        "schema": subject.LEDGER_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "transitions": [transition],
        "accepted_jobs": [],
    }
    retry_ledger = {
        **ledger_core,
        "ledger_sha256": subject.canonical_sha256(ledger_core),
    }
    retry = subject.build_resume_plan(plan, retry_ledger)
    assert {row["attempt_id"] for row in retry["selected_attempts"]} == {"a01"}


def test_lifecycle_must_prove_cleanup_absence_and_iam_removal(
    frozen_plan: dict[str, object],
) -> None:
    plan = frozen_plan["plan"]
    ledger = subject.initial_attempt_ledger(plan)
    resume = subject.build_resume_plan(plan, ledger)
    request = subject.build_wave_request(plan, ledger, resume)
    lifecycle, _ = _lifecycle(plan, ledger, resume, request)
    checked = subject.validate_lifecycle_receipt(
        plan, ledger, resume, request, lifecycle
    )
    assert checked["receiver_handoff_ready"] is True
    tampered = deepcopy(lifecycle)
    tampered["worker_iam_removed"] = False
    core = dict(tampered)
    core.pop("receipt_sha256")
    tampered["receipt_sha256"] = subject.canonical_sha256(core)
    with pytest.raises(PermissionError, match="cleanup boundary"):
        subject.validate_lifecycle_receipt(
            plan, ledger, resume, request, tampered
        )


def test_receive_source_replays_wave_and_opens_second_wave_only(
    monkeypatch: pytest.MonkeyPatch,
    frozen_plan: dict[str, object],
) -> None:
    plan = frozen_plan["plan"]
    ledger = subject.initial_attempt_ledger(plan)
    resume = subject.build_resume_plan(plan, ledger)
    request = subject.build_wave_request(plan, ledger, resume)
    lifecycle, payloads = _lifecycle(plan, ledger, resume, request)
    generation_by_name = {
        record["name"]: record["generation"]
        for row in lifecycle["attempt_rows"]
        for record in [
            row["result_object"],
            row["done_object"],
            *row["task_objects"],
        ]
    }

    class Reader(_Reader):
        def read_object(self, *, bucket: str, object_name: str) -> dict:
            value = super().read_object(bucket=bucket, object_name=object_name)
            value["generation"] = generation_by_name[object_name]
            return value

    fake_jobs = {
        row["job_id"]: row for row in frozen_plan["fake_package"]["jobs"]
    }
    monkeypatch.setattr(
        transport,
        "validate_selected_job",
        lambda *, package, job_id, expected_job_manifest_sha256: deepcopy(
            fake_jobs[job_id]
        ),
    )
    root_indices_by_job: dict[str, list[int]] = {}

    def actual_transport_shape(_package_root, job, _plan):
        ordinal = int(job["job_id"].split("-")[1])
        start = ordinal * 10
        indices = list(range(start, start + 10))
        root_indices_by_job[job["job_id"]] = indices
        return (
            {index: ({"pair_index": index // 2}, object()) for index in indices},
            [{"root_indices": indices}],
        )

    monkeypatch.setattr(transport, "_root_lookup", actual_transport_shape)

    validated_lookup_indices: dict[str, list[int]] = {}

    def validate_done_with_mapping(value, *, job, lookup, **_kwargs):
        assert isinstance(lookup, dict)
        validated_lookup_indices[job["job_id"]] = list(lookup)
        return {
            **deepcopy(dict(value)),
            "task_record_aggregate_sha256": "5" * 64,
        }

    monkeypatch.setattr(
        transport,
        "_validate_done",
        validate_done_with_mapping,
    )
    receipt = subject.receive_ready_jobs(
        plan=plan,
        ledger=ledger,
        resume=resume,
        request=request,
        lifecycle_receipt=lifecycle,
        reader=Reader(payloads),
        output_directory=Path(frozen_plan["tmp_path"]) / "receive-wave0",
    )
    assert receipt["accepted_job_count"] == 8
    assert receipt["failed_job_count"] == 0
    assert root_indices_by_job["primary-01"] == list(range(10, 20))
    assert validated_lookup_indices["primary-01"] == list(range(10, 20))
    assert receipt["source_replay_complete"] is True
    assert receipt["cloud_mutated"] is False
    assert receipt["current_profile_changed"] is False
    assert (
        Path(frozen_plan["tmp_path"]) / "receive-wave0" / "receive_receipt.json"
    ).is_file()
    next_resume = subject.build_resume_plan(plan, receipt["output_ledger"])
    assert next_resume["resume_wave_index"] == 1
    assert [row["job_id"] for row in next_resume["selected_attempts"]] == list(
        subject.WAVE_JOB_IDS[1]
    )
    assert {row["attempt_id"] for row in next_resume["selected_attempts"]} == {
        "a00"
    }


def test_receive_failure_preserves_create_only_partial_evidence(
    monkeypatch: pytest.MonkeyPatch,
    frozen_plan: dict[str, object],
) -> None:
    plan = frozen_plan["plan"]
    ledger = subject.initial_attempt_ledger(plan)
    resume = subject.build_resume_plan(plan, ledger)
    request = subject.build_wave_request(plan, ledger, resume)
    lifecycle, payloads = _lifecycle(plan, ledger, resume, request)
    generation_by_name = {
        record["name"]: record["generation"]
        for row in lifecycle["attempt_rows"]
        for record in [
            row["result_object"],
            row["done_object"],
            *row["task_objects"],
        ]
    }

    class Reader(_Reader):
        def read_object(self, *, bucket: str, object_name: str) -> dict:
            value = super().read_object(bucket=bucket, object_name=object_name)
            value["generation"] = generation_by_name[object_name]
            return value

    fake_jobs = {
        row["job_id"]: row for row in frozen_plan["fake_package"]["jobs"]
    }
    monkeypatch.setattr(
        transport,
        "validate_selected_job",
        lambda *, package, job_id, expected_job_manifest_sha256: deepcopy(
            fake_jobs[job_id]
        ),
    )
    monkeypatch.setattr(
        transport,
        "_root_lookup",
        lambda *_args: ({index: ({}, object()) for index in range(10)}, []),
    )

    def fail_second_job(value, *, job, **_kwargs):
        if job["job_id"] == "primary-01":
            raise ValueError("forced receive regression failure")
        return {
            **deepcopy(dict(value)),
            "task_record_aggregate_sha256": "5" * 64,
        }

    monkeypatch.setattr(transport, "_validate_done", fail_second_job)
    output = Path(frozen_plan["tmp_path"]) / "receive-partial"
    with pytest.raises(ValueError, match="forced receive regression failure"):
        subject.receive_ready_jobs(
            plan=plan,
            ledger=ledger,
            resume=resume,
            request=request,
            lifecycle_receipt=lifecycle,
            reader=Reader(payloads),
            output_directory=output,
        )

    assert (output / "jobs/primary-00/DONE.json").is_file()
    assert (output / "jobs/primary-01/DONE.json").is_file()
    assert (output / "accepted-results/primary-00.json").is_file()
    assert not (output / "accepted-results/primary-01.json").exists()
    assert not (output / "receive_receipt.json").exists()
    with pytest.raises(FileExistsError, match="create-only"):
        subject.receive_ready_jobs(
            plan=plan,
            ledger=ledger,
            resume=resume,
            request=request,
            lifecycle_receipt=lifecycle,
            reader=Reader(payloads),
            output_directory=output,
        )


def test_append_only_controller_journal_is_reused(
    frozen_plan: dict[str, object],
) -> None:
    plan = frozen_plan["plan"]
    ledger = subject.initial_attempt_ledger(plan)
    resume = subject.build_resume_plan(plan, ledger)
    request = subject.build_wave_request(plan, ledger, resume)
    journal = subject.open_controller_journal(
        Path(frozen_plan["tmp_path"]) / "journal", wave_request=request
    )
    event = journal.append(
        phase="quality-wave-0",
        mode="prepare",
        status="complete",
        operation_key="quality-wave-0-prepare",
        predecessor_event_sha256=None,
        evidence={"request_sha256": request["request_sha256"]},
        output={"cloud_mutated": False},
        mutation_requested=False,
        mutation_outcome="not_requested",
        recorded_at_utc="2026-07-23T12:00:00Z",
    )
    assert event.value["schema"] == subject.controller_v2.EVENT_SCHEMA
    assert subject.open_controller_journal(
        Path(frozen_plan["tmp_path"]) / "journal",
        wave_request=request,
        create=False,
    ).load()[0].value == event.value


class _FakeCloud:
    def __init__(self, *, image_link: str, image_id: str, features: list[str]) -> None:
        self.objects: dict[str, tuple[str, bytes]] = {}
        self.next_generation = 1
        self.instances: dict[str, dict] = {}
        self.disks: dict[str, dict] = {}
        self.operations: dict[str, dict] = {}
        self.policy = {
            "version": 3,
            "etag": "etag-1",
            "bindings": [
                {
                    "role": "roles/storage.legacyBucketReader",
                    "members": ["projectViewer:unrelated"],
                }
            ],
        }
        self.policy_version = 1
        self.image = {
            "selfLink": image_link,
            "id": image_id,
            "status": "READY",
            "guestOsFeatures": [{"type": item} for item in features],
        }

    def get_object_metadata(self, *, bucket: str, object_name: str):
        del bucket
        value = self.objects.get(object_name)
        if value is None:
            return None
        generation, raw = value
        return {"name": object_name, "generation": generation, "size": str(len(raw))}

    def get_object_bytes(
        self, *, bucket: str, object_name: str, generation: str | None = None
    ):
        del bucket
        value = self.objects.get(object_name)
        if value is None or (generation is not None and value[0] != generation):
            return None
        return value[1]

    def put_object_new(
        self, *, bucket: str, object_name: str, payload: bytes, content_type: str
    ):
        del bucket, content_type
        if object_name in self.objects:
            raise FileExistsError(object_name)
        generation = str(self.next_generation)
        self.next_generation += 1
        self.objects[object_name] = (generation, bytes(payload))
        return {"name": object_name, "generation": generation}

    def get_instance(self, *, instance_name: str):
        value = self.instances.get(instance_name)
        return None if value is None else deepcopy(value)

    def create_instance(self, *, instance_spec: dict, request_id: str):
        name = instance_spec["name"]
        assert name not in self.instances
        instance_id = str(10_000 + len(self.instances))
        disk_id = str(20_000 + len(self.disks))
        instance = deepcopy(instance_spec)
        instance["id"] = instance_id
        instance["status"] = "RUNNING"
        instance["disks"][0]["source"] = (
            f"projects/{provider.PROJECT}/zones/{provider.ZONE}/disks/{name}"
        )
        self.instances[name] = instance
        initialize = instance_spec["disks"][0]["initializeParams"]
        self.disks[name] = {
            "name": name,
            "id": disk_id,
            "type": initialize["diskType"],
            "sizeGb": initialize["diskSizeGb"],
            "sourceImage": initialize["sourceImage"],
            "labels": deepcopy(initialize["labels"]),
        }
        operation = {
            "name": f"insert-{len(self.operations):03d}",
            "status": "DONE",
            "operationType": "insert",
            "targetLink": (
                f"projects/{provider.PROJECT}/zones/{provider.ZONE}/instances/{name}"
            ),
            "requestId": request_id,
        }
        self.operations[operation["name"]] = operation
        return deepcopy(operation)

    def get_zone_operation(self, *, operation_name: str):
        return deepcopy(self.operations[operation_name])

    def delete_instance(self, *, instance_name: str, request_id: str):
        self.instances.pop(instance_name, None)
        self.disks.pop(instance_name, None)
        operation = {
            "name": f"delete-{len(self.operations):03d}",
            "status": "DONE",
            "operationType": "delete",
            "targetLink": (
                f"projects/{provider.PROJECT}/zones/{provider.ZONE}/instances/"
                f"{instance_name}"
            ),
            "requestId": request_id,
        }
        self.operations[operation["name"]] = operation
        return deepcopy(operation)

    def get_disk_optional(self, *, disk_name: str):
        value = self.disks.get(disk_name)
        return None if value is None else deepcopy(value)

    def delete_disk(self, *, disk_name: str, request_id: str):
        self.disks.pop(disk_name, None)
        operation = {
            "name": f"delete-disk-{len(self.operations):03d}",
            "status": "DONE",
            "operationType": "delete",
            "targetLink": (
                f"projects/{provider.PROJECT}/zones/{provider.ZONE}/disks/"
                f"{disk_name}"
            ),
            "requestId": request_id,
        }
        self.operations[operation["name"]] = operation
        return deepcopy(operation)

    def get_bucket_iam_policy(self, *, bucket: str):
        del bucket
        return deepcopy(self.policy)

    def set_bucket_iam_policy(self, *, bucket: str, policy: dict):
        del bucket
        assert policy["etag"] == self.policy["etag"]
        self.policy_version += 1
        self.policy = deepcopy(policy)
        self.policy["etag"] = f"etag-{self.policy_version}"
        return deepcopy(self.policy)

    def get_service_account(self, *, email: str):
        return {"email": email, "uniqueId": str(30_000 + len(email))}

    def test_service_account_act_as(self, *, email: str):
        del email
        return True

    def get_region_quota(self, *, region: str):
        assert region == provider.REGION
        return {
            "quotas": [
                {"metric": "PREEMPTIBLE_CPUS", "limit": 355, "usage": 0},
            ]
        }

    def get_image(self, *, self_link: str):
        assert self_link == self.image["selfLink"]
        return deepcopy(self.image)

    def get_router(self, *, region: str, router_name: str):
        assert region == provider.REGION
        assert router_name == provider.NAT_ROUTER_NAME
        return {
            "name": provider.NAT_ROUTER_NAME,
            "network": (
                f"projects/{provider.PROJECT}/global/networks/{provider.NETWORK}"
            ),
            "nats": [
                {
                    "name": provider.NAT_NAME,
                    "natIpAllocateOption": "AUTO_ONLY",
                    "sourceSubnetworkIpRangesToNat": (
                        "ALL_SUBNETWORKS_ALL_IP_RANGES"
                    ),
                }
            ],
        }

    def get_cloud_quota(self, *, quota_id: str):
        if quota_id == provider.C4_QUOTA_ID:
            return {
                "quotaId": quota_id,
                "metric": "compute.googleapis.com/cpus_per_vm_family",
                "isPrecise": True,
                "dimensionsInfos": [
                    {
                        "dimensions": {
                            "region": provider.REGION,
                            "vm_family": "C4",
                        },
                        "details": {"value": "128"},
                    }
                ],
            }
        assert quota_id == provider.GLOBAL_QUOTA_ID
        return {
            "quotaId": quota_id,
            "metric": "compute.googleapis.com/cpus_all_regions",
            "isPrecise": True,
            "dimensionsInfos": [
                {"dimensions": {}, "details": {"value": "500"}}
            ],
        }

    def list_instances(self):
        return [
            {**deepcopy(row), "_scope": f"zones/{provider.ZONE}"}
            for row in self.instances.values()
        ]

    def get_machine_type_url(self, *, self_link: str):
        return {"selfLink": self_link, "guestCpus": 16}

    def publish_quality_results(self, *, plan: dict, resume: dict) -> None:
        jobs = {row["job_id"]: row for row in plan["jobs"]}
        for selected in resume["selected_attempts"]:
            job = jobs[selected["job_id"]]
            attempt = next(
                row
                for row in job["attempts"]
                if row["attempt_id"] == selected["attempt_id"]
            )
            result_payload = subject.canonical_bytes(
                {"schema": "test-result", "job_id": selected["job_id"]}
            )
            result = self.put_object_new(
                bucket=plan["bucket"],
                object_name=attempt["result_object"],
                payload=result_payload,
                content_type="application/json",
            )
            del result
            task_count = 10 if job["phase"] == "primary" else 2
            task_records = []
            for index in range(task_count):
                task_payload = subject.canonical_bytes(
                    {
                        "schema": transport.TASK_SCHEMA,
                        "job_id": selected["job_id"],
                        "root_index": index,
                    }
                )
                path = f"tasks/root_{index:03d}.json"
                self.put_object_new(
                    bucket=plan["bucket"],
                    object_name=attempt["task_object_prefix"]
                    + f"root_{index:03d}.json",
                    payload=task_payload,
                    content_type="application/json",
                )
                task_records.append(
                    {
                        "path": path,
                        "sha256": hashlib.sha256(task_payload).hexdigest(),
                        "bytes": len(task_payload),
                    }
                )
            done_payload = subject.canonical_bytes(
                {
                    "schema": transport.DONE_SCHEMA,
                    "status": "complete_validated_quality_job",
                    "job_id": selected["job_id"],
                    "phase": job["phase"],
                    "job_manifest_sha256": job["job_manifest_sha256"],
                    "result_sha256": hashlib.sha256(result_payload).hexdigest(),
                    "result_bytes": len(result_payload),
                    "task_records": task_records,
                    "task_record_aggregate_sha256": subject.canonical_sha256(
                        task_records
                    ),
                    "completed_root_count": task_count,
                    "done_published_last": True,
                    "training_eligible": False,
                    "promotion_evidence": False,
                    "current_profile_changed": False,
                }
            )
            self.put_object_new(
                bucket=plan["bucket"],
                object_name=attempt["done_object"],
                payload=done_payload,
                content_type="application/json",
            )


def test_provider_executes_polls_cleans_and_receives_exact_wave(
    monkeypatch: pytest.MonkeyPatch,
    frozen_plan: dict[str, object],
) -> None:
    plan = frozen_plan["plan"]
    ledger = subject.initial_attempt_ledger(plan)
    resume = subject.build_resume_plan(plan, ledger)
    request = subject.build_wave_request(plan, ledger, resume)
    image_link = (
        "https://www.googleapis.com/compute/v1/projects/debian-cloud/global/"
        "images/debian-12-bookworm-v20260715"
    )
    image_id = "123456789"
    features = ["GVNIC", "UEFI_COMPATIBLE", "VIRTIO_SCSI_MULTIQUEUE"]
    accounts = [
        f"ofc-fq-worker-{index:02d}@{provider.PROJECT}.iam.gserviceaccount.com"
        for index in range(8)
    ]
    provider_plan = provider.build_provider_plan(
        bridge_plan=plan,
        ledger=ledger,
        resume=resume,
        wave_request=request,
        image_self_link=image_link,
        image_id=image_id,
        guest_os_features=features,
        worker_service_accounts=accounts,
    )
    cloud = _FakeCloud(
        image_link=image_link, image_id=image_id, features=features
    )
    launch = provider.execute_wave(
        provider_plan=provider_plan,
        bridge_plan=plan,
        ledger=ledger,
        resume=resume,
        wave_request=request,
        raw_nonce="11111111-1111-4111-8111-111111111111",
        transport=cloud,
        now_unix_seconds=1_784_800_000,
        observed_at_utc="2026-07-23T12:00:00Z",
        sleep=lambda _seconds: None,
    )
    assert launch["create_complete"] is True
    assert len(cloud.instances) == 8
    assert len(launch["iam_receipt"]["condition_titles"]) == 9
    assert all(
        len(binding["members"]) == 1
        for binding in cloud.policy["bindings"]
        if binding.get("role") == "roles/storage.objectCreator"
    )
    before = provider.poll_wave(
        provider_plan=provider_plan,
        bridge_plan=plan,
        ledger=ledger,
        resume=resume,
        wave_request=request,
        launch_receipt=launch,
        transport=cloud,
    )
    assert before["done_count"] == 0
    cloud.publish_quality_results(plan=plan, resume=resume)
    # GCS generation values are object-version identities, not a global
    # publication sequence.  A valid DONE-last worker run must remain
    # receivable even when DONE has a numerically smaller generation.
    for selected in resume["selected_attempts"]:
        job = next(row for row in plan["jobs"] if row["job_id"] == selected["job_id"])
        attempt = next(
            row
            for row in job["attempts"]
            if row["attempt_id"] == selected["attempt_id"]
        )
        done_generation, done_payload = cloud.objects[attempt["done_object"]]
        assert int(done_generation) > 1
        cloud.objects[attempt["done_object"]] = ("1", done_payload)
    after = provider.poll_wave(
        provider_plan=provider_plan,
        bridge_plan=plan,
        ledger=ledger,
        resume=resume,
        wave_request=request,
        launch_receipt=launch,
        transport=cloud,
    )
    assert after["done_count"] == 8
    lifecycle = provider.cleanup_wave(
        provider_plan=provider_plan,
        bridge_plan=plan,
        ledger=ledger,
        resume=resume,
        wave_request=request,
        launch_receipt=launch,
        transport=cloud,
        sleep=lambda _seconds: None,
    )
    assert not cloud.instances
    assert not cloud.disks
    assert lifecycle["owned_compute_absent"] is True
    assert lifecycle["worker_iam_removed"] is True
    assert len(cloud.policy["bindings"]) == 1

    fake_jobs = {
        row["job_id"]: row for row in frozen_plan["fake_package"]["jobs"]
    }
    monkeypatch.setattr(
        transport,
        "validate_selected_job",
        lambda *, package, job_id, expected_job_manifest_sha256: deepcopy(
            fake_jobs[job_id]
        ),
    )
    monkeypatch.setattr(transport, "_root_lookup", lambda *_args: ({}, []))
    monkeypatch.setattr(
        transport,
        "_validate_done",
        lambda value, **_kwargs: {
            **deepcopy(dict(value)),
            "task_record_aggregate_sha256": value[
                "task_record_aggregate_sha256"
            ],
        },
    )
    received = provider.receive_wave(
        bridge_plan=plan,
        ledger=ledger,
        resume=resume,
        wave_request=request,
        lifecycle_receipt=lifecycle,
        transport=cloud,
        output_directory=Path(frozen_plan["tmp_path"]) / "provider-receive",
        accepted_results_directory=Path(frozen_plan["tmp_path"])
        / "accepted-results",
    )
    assert received["accepted_job_count"] == 8
    next_resume = subject.build_resume_plan(plan, received["output_ledger"])
    assert next_resume["resume_wave_index"] == 1
    assert len(next_resume["selected_attempts"]) == 7
    next_request = subject.build_wave_request(
        plan, received["output_ledger"], next_resume
    )
    next_provider_plan = provider.build_provider_plan(
        bridge_plan=plan,
        ledger=received["output_ledger"],
        resume=next_resume,
        wave_request=next_request,
        image_self_link=image_link,
        image_id=image_id,
        guest_os_features=features,
        worker_service_accounts=accounts,
    )
    second_launch = provider.execute_wave(
        provider_plan=next_provider_plan,
        bridge_plan=plan,
        ledger=received["output_ledger"],
        resume=next_resume,
        wave_request=next_request,
        raw_nonce="22222222-2222-4222-8222-222222222222",
        transport=cloud,
        now_unix_seconds=1_784_800_100,
        observed_at_utc="2026-07-23T12:01:40Z",
        sleep=lambda _seconds: None,
    )
    assert second_launch["create_complete"] is True
    assert len(cloud.instances) == 7
    orphan_name = next(iter(cloud.instances))
    cloud.instances.pop(orphan_name)
    assert orphan_name in cloud.disks
    abort_lifecycle = provider.cleanup_wave(
        provider_plan=next_provider_plan,
        bridge_plan=plan,
        ledger=received["output_ledger"],
        resume=next_resume,
        wave_request=next_request,
        launch_receipt=None,
        transport=cloud,
        sleep=lambda _seconds: None,
    )
    assert not cloud.instances
    assert not cloud.disks
    assert abort_lifecycle["owned_compute_absent"] is True
    assert abort_lifecycle["worker_iam_removed"] is True
    assert all(row["status"] == "failed" for row in abort_lifecycle["attempt_rows"])
    assert len(cloud.policy["bindings"]) == 1


def test_cli_exposes_plan_execute_poll_cleanup_receive_and_env_only_token(
    frozen_plan: dict[str, object],
) -> None:
    script = ROOT / "scripts/run_hu_m31_t3_step6d_fresh_quality_gcp_v1.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    source = script.read_text(encoding="utf-8")
    for command in ("prepare", "plan", "execute", "poll", "cleanup", "receive"):
        assert f'"{command}"' in source
    assert "GOOGLE_OAUTH_ACCESS_TOKEN" in source
    assert "--access-token" not in source
    assert "--journal-directory" in source
    assert "--abort-without-launch-receipt" in source
    assert provider.worker_startup_script().count("pip install") == 0
    namespace = runpy.run_path(str(script), run_name="fresh_quality_cli_test")
    plan = frozen_plan["plan"]
    ledger = subject.initial_attempt_ledger(plan)
    resume = subject.build_resume_plan(plan, ledger)
    request = subject.build_wave_request(plan, ledger, resume)
    journal = namespace["_journal"](
        Path(frozen_plan["tmp_path"]) / "cli-journal",
        wave_request=request,
        create=True,
    )
    event = namespace["_journal_append"](
        journal,
        phase="quality-wave-0",
        mode="poll",
        status="complete",
        operation_key="quality-wave-0-poll-000001",
        evidence={"request_sha256": request["request_sha256"]},
        output={"cloud_mutated": False, "current_profile_changed": False},
        mutation_requested=False,
        mutation_outcome="not_requested",
    )
    assert journal.load()[0].value == event


def _minimal_cli_wave_files(
    tmp_path: Path, namespace: dict[str, object]
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    plan = {
        "run_name": "regular-hu-m31-t3-fqv1-cli-regression",
        "plan_sha256": "1" * 64,
    }
    ledger = {"ledger_sha256": "2" * 64}
    resume = {
        "resume_sha256": "3" * 64,
        "selected_attempts": [
            {"job_id": "primary-00", "wave_index": 0},
            {"job_id": "primary-01", "wave_index": 0},
        ],
    }
    request = {
        "request_sha256": "4" * 64,
        "selected_attempts": resume["selected_attempts"],
    }
    writer = namespace["_write"]
    assert callable(writer)
    writer(tmp_path / "bridge.json", plan)
    writer(tmp_path / "ledger.json", ledger)
    writer(tmp_path / "resume.json", resume)
    writer(tmp_path / "request.json", request)
    return plan, ledger, resume, request


def test_cli_phase_derives_one_nonempty_selected_attempt_wave_fail_closed() -> None:
    script = ROOT / "scripts/run_hu_m31_t3_step6d_fresh_quality_gcp_v1.py"
    namespace = runpy.run_path(str(script), run_name="fresh_quality_cli_phase_test")
    phase = namespace["_phase"]
    assert callable(phase)
    assert phase(
        {
            "selected_attempts": [
                {"job_id": "primary-00", "wave_index": 0},
                {"job_id": "primary-01", "wave_index": 0},
            ]
        }
    ) == "quality-wave-0"
    with pytest.raises(ValueError, match="at least one"):
        phase({"selected_attempts": []})
    with pytest.raises(ValueError, match="multiple waves"):
        phase(
            {
                "selected_attempts": [
                    {"job_id": "primary-00", "wave_index": 0},
                    {"job_id": "primary-08", "wave_index": 1},
                ]
            }
        )
    with pytest.raises(ValueError, match="wave index is invalid"):
        phase(
            {
                "selected_attempts": [
                    {"job_id": "primary-00", "wave_index": True}
                ]
            }
        )


def test_cli_plan_journals_wave_derived_from_selected_attempts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = ROOT / "scripts/run_hu_m31_t3_step6d_fresh_quality_gcp_v1.py"
    namespace = runpy.run_path(str(script), run_name="fresh_quality_cli_plan_test")
    _plan, _ledger, resume, request = _minimal_cli_wave_files(
        tmp_path, namespace
    )
    provider_plan = {
        "provider_plan_sha256": "5" * 64,
        "current_profile_changed": False,
    }
    monkeypatch.setattr(subject, "build_resume_plan", lambda *_args: resume)
    monkeypatch.setattr(subject, "build_wave_request", lambda *_args: request)
    monkeypatch.setattr(
        provider, "build_provider_plan", lambda **_kwargs: provider_plan
    )
    main = namespace["main"]
    assert callable(main)
    assert main(
        [
            "plan",
            "--bridge-plan",
            str(tmp_path / "bridge.json"),
            "--ledger",
            str(tmp_path / "ledger.json"),
            "--image-self-link",
            "https://example.invalid/image",
            "--image-id",
            "123",
            "--guest-os-feature",
            "GVNIC",
            "--worker-service-account",
            "worker@example.invalid",
            "--output-directory",
            str(tmp_path / "wave-plan"),
            "--journal-directory",
            str(tmp_path / "journal"),
        ]
    ) == 0
    journal = namespace["_journal"](
        tmp_path / "journal", wave_request=request, create=False
    )
    events = journal.load()
    assert len(events) == 1
    assert events[0].value["phase"] == "quality-wave-0"
    assert events[0].value["operation_key"] == "quality-wave-0-plan"
    assert events[0].value["mutation_requested"] is False


def test_cli_execute_journals_wave_before_any_provider_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = ROOT / "scripts/run_hu_m31_t3_step6d_fresh_quality_gcp_v1.py"
    namespace = runpy.run_path(
        str(script), run_name="fresh_quality_cli_execute_test"
    )
    plan, _ledger, _resume, request = _minimal_cli_wave_files(
        tmp_path, namespace
    )
    provider_plan = {
        "provider_plan_sha256": "5" * 64,
        "current_profile_changed": False,
    }
    writer = namespace["_write"]
    assert callable(writer)
    writer(tmp_path / "provider.json", provider_plan)
    namespace["_journal"](
        tmp_path / "journal", wave_request=request, create=True
    )
    transport_marker = object()
    monkeypatch.setenv("GOOGLE_OAUTH_ACCESS_TOKEN", "test-token")
    monkeypatch.setattr(
        provider,
        "GcpQualityRestAdapter",
        lambda *, access_token, token_provider=None: (
            transport_marker
            if access_token == "test-token"
            # The CLI supplies a refresher so a long wave survives token expiry.
            and callable(token_provider)
            else pytest.fail("unexpected transport construction")
        ),
    )
    observed: dict[str, object] = {}

    def fake_execute_wave(**kwargs):
        observed.update(kwargs)
        return {
            "create_complete": True,
            "cloud_mutated": True,
            "current_profile_changed": False,
        }

    monkeypatch.setattr(provider, "execute_wave", fake_execute_wave)
    monkeypatch.setenv(
        "OFC_M31_FQ_OPERATION_NONCE",
        "11111111-1111-4111-8111-111111111111",
    )
    main = namespace["main"]
    assert callable(main)
    assert main(
        [
            "execute",
            "--provider-plan",
            str(tmp_path / "provider.json"),
            "--bridge-plan",
            str(tmp_path / "bridge.json"),
            "--ledger",
            str(tmp_path / "ledger.json"),
            "--resume",
            str(tmp_path / "resume.json"),
            "--wave-request",
            str(tmp_path / "request.json"),
            "--journal-directory",
            str(tmp_path / "journal"),
            "--confirm-run-name",
            plan["run_name"],
            "--execute-cloud-mutations",
            "CREATE_EXACT_FRESH_QUALITY_WAVE",
            "--output",
            str(tmp_path / "launch.json"),
        ]
    ) == 0
    assert observed["transport"] is transport_marker
    events = namespace["_journal"](
        tmp_path / "journal", wave_request=request, create=False
    ).load()
    assert [event.value["phase"] for event in events] == [
        "quality-wave-0",
        "quality-wave-0",
    ]
    assert [event.value["status"] for event in events] == [
        "pending",
        "complete",
    ]
    assert all(
        event.value["operation_key"] == "quality-wave-0-execute"
        for event in events
    )


def test_live_adapter_delete_races_are_absence_safe() -> None:
    requests: list[tuple[str, str]] = []

    def request(method, url, _headers, _payload, _timeout):
        requests.append((method, url))
        if "/instances/" in url:
            return c4_gcp.HttpResponse(404, b"{}", {})
        return c4_gcp.HttpResponse(
            200, b'{"name":"delete-disk-op","status":"DONE"}', {}
        )

    adapter = provider.GcpQualityRestAdapter(
        access_token="test-token", request=request
    )
    request_id = "11111111-1111-4111-8111-111111111111"
    assert (
        adapter.delete_instance(instance_name="fq-owned-vm", request_id=request_id)
        is None
    )
    assert adapter.delete_disk(
        disk_name="fq-owned-disk", request_id=request_id
    ) == {"name": "delete-disk-op", "status": "DONE"}
    assert [method for method, _url in requests] == ["DELETE", "DELETE"]
    assert all(f"requestId={request_id}" in url for _method, url in requests)
