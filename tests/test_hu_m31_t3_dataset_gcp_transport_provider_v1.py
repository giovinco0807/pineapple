from __future__ import annotations

import hashlib
import json
import tarfile
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import pytest

from ofc_regular import hu_m31_t3_dataset_contract_v1 as dataset
from ofc_regular import hu_m31_t3_dataset_gcp_provider_v1 as provider
from ofc_regular import hu_m31_t3_dataset_gcp_transport_v1 as transport
from ofc_regular import hu_m31_t3_dataset_portable_worker_v1 as portable


def _tar(path: Path, root: str, files: Mapping[str, bytes]) -> None:
    source = path.parent / f"{path.stem}-source" / root
    source.mkdir(parents=True)
    for relative, payload in files.items():
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    with tarfile.open(path, "w:gz") as archive:
        archive.add(source, arcname=root)


def _zip(path: Path, root: str, files: Mapping[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for relative, payload in files.items():
            archive.writestr(f"{root}/{relative}", payload)


def _transport_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    plan = dataset.build_dataset_plan()
    plan_path = tmp_path / "dataset-plan.json"
    plan_path.write_bytes(dataset.canonical_bytes(plan))
    fresh = tmp_path / "fresh-quality-gate.json"
    fresh.write_bytes(transport.canonical_bytes({"stub": "fresh"}))
    smoke_gate = tmp_path / "smoke-gate.json"
    smoke_gate.write_bytes(transport.canonical_bytes({"stub": "smoke"}))
    smoke_directory = tmp_path / "smoke-shard"
    smoke_directory.mkdir()
    smoke_file = smoke_directory / "smoke.txt"
    smoke_file.write_bytes(b"source-replayed-smoke")
    inventory = portable._inventory(smoke_directory)  # type: ignore[attr-defined]
    portable_value = {
        "authorization_sha256": "a" * 64,
        "smoke_shard_inventory": inventory,
    }
    portable_path = tmp_path / "portable.json"
    portable_path.write_bytes(transport.canonical_bytes(portable_value))
    monkeypatch.setattr(
        portable,
        "build_portable_fanout_authorization",
        lambda **_kwargs: deepcopy(portable_value),
    )
    smoke_archive = tmp_path / "smoke.tar.gz"
    _tar(smoke_archive, "smoke_shard", {"smoke.txt": smoke_file.read_bytes()})
    runtime_archive = tmp_path / "runtime.tar.gz"
    _tar(runtime_archive, "runtime", {"src/runtime.py": b"# pinned"})
    wheelhouse = tmp_path / "wheelhouse.zip"
    _zip(wheelhouse, "wheelhouse", {"runtime.whl": b"pinned-wheel"})
    library = tmp_path / "candidate02.dll"
    library.write_bytes(b"accepted-candidate02")
    monkeypatch.setattr(
        transport.dataset,
        "ACCEPTED_CANDIDATE_LIBRARY_SHA256",
        hashlib.sha256(library.read_bytes()).hexdigest(),
    )
    return transport.build_transport_plan(
        run_name="regular-hu-m31-dataset-test",
        bucket="ofc-m31-dataset-test",
        dataset_plan_path=plan_path,
        fresh_quality_gate_path=fresh,
        smoke_gate_path=smoke_gate,
        portable_authorization_path=portable_path,
        smoke_shard_directory=smoke_directory,
        smoke_shard_archive_path=smoke_archive,
        runtime_archive_path=runtime_archive,
        wheelhouse_archive_path=wheelhouse,
        candidate_library_path=library,
    )


def test_transport_plan_is_exact_359_cloud_plus_smoke_and_max8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _transport_plan(tmp_path, monkeypatch)
    assert transport.validate_transport_plan(plan, replay_sources=True) == plan
    assert plan["total_dataset_shard_count"] == 360
    assert plan["cloud_shard_count"] == 359
    assert plan["precompleted_smoke_shard_id"] == dataset.SMOKE_SHARD_ID
    assert plan["wave_count"] == 45
    assert [wave["shard_count"] for wave in plan["waves"][:2]] == [8, 8]
    assert plan["waves"][-1]["shard_count"] == 7
    assert all(wave["max_vm_count"] <= 8 for wave in plan["waves"])
    assert all(job["paired_hand_count"] == 25 for job in plan["jobs"])
    assert all(len(job["attempts"]) == 2 for job in plan["jobs"])
    assert plan["checkpoint_contract"]["heartbeat_create_only_sequence"] is True
    assert plan["cleanup_contract"]["vm_and_boot_disk_absence_required"] is True

    changed = deepcopy(plan)
    changed["waves"][0]["max_vm_count"] = 9
    core = dict(changed)
    core.pop("plan_sha256")
    changed["plan_sha256"] = transport.canonical_sha256(core)
    with pytest.raises(ValueError, match="wave binding"):
        transport.validate_transport_plan(changed, replay_sources=False)


def _checkpoint_lifecycle(
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    rows = []
    payloads = {}
    for generation, selected in enumerate(
        request["selected_attempts"], start=1
    ):
        auth = transport.canonical_bytes(
            {"portable": selected["shard_id"]}
        )
        file_object = (
            f"{selected['file_object_prefix']}DATASET_AUTHORIZATION.json"
        )
        manifest_core = {
            "schema": transport.CHECKPOINT_SCHEMA,
            "plan_sha256": plan["plan_sha256"],
            "shard_id": selected["shard_id"],
            "attempt_id": selected["attempt_id"],
            "sequence": 0,
            "completed_pair_count": 0,
            "complete": False,
            "files": [
                {
                    "relative_path": "DATASET_AUTHORIZATION.json",
                    "object_name": file_object,
                    "sha256": hashlib.sha256(auth).hexdigest(),
                    "bytes": len(auth),
                }
            ],
            "checkpoint_published_after_files": True,
            "create_only": True,
            "teacher_values_are_realized_match_ev": False,
            "current_profile_changed": False,
        }
        manifest = {
            **manifest_core,
            "checkpoint_sha256": transport.canonical_sha256(manifest_core),
        }
        checkpoint = {
            "object": selected["checkpoint_object_format"] % 0,
            "generation": str(generation * 2),
            "sha256": transport.canonical_sha256(manifest),
            "bytes": len(transport.canonical_bytes(manifest)),
            "manifest": manifest,
            "file_objects": [
                {
                    "object": file_object,
                    "generation": str(generation * 2 - 1),
                    "sha256": hashlib.sha256(auth).hexdigest(),
                    "bytes": len(auth),
                }
            ],
        }
        rows.append(
            {
                "shard_id": selected["shard_id"],
                "attempt_id": selected["attempt_id"],
                "instance_id": selected["instance_id"],
                "status": "checkpointed",
                "completed_pair_count": 0,
                "checkpoint": checkpoint,
                "heartbeat": None,
                "owned_compute_absent": True,
                "worker_iam_removed": True,
            }
        )
        payloads[file_object] = auth
    core = {
        "schema": transport.LIFECYCLE_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "ledger_sha256": ledger["ledger_sha256"],
        "resume_sha256": resume["resume_sha256"],
        "request_sha256": request["request_sha256"],
        "wave_index": request["wave_index"],
        "selected_attempt_count": len(rows),
        "attempt_rows": rows,
        "max_concurrent_vms": 8,
        "create_only_gcs": True,
        "exact_owned_cleanup": True,
        "owned_vm_disk_absent": True,
        "worker_iam_removed_before_receive": True,
        "receiver_handoff_ready": True,
        "wildcard_delete_used": False,
        "unrelated_resource_touched": False,
        "gcs_evidence_deleted": False,
        "current_profile_changed": False,
    }
    return (
        {**core, "receipt_sha256": transport.canonical_sha256(core)},
        payloads,
    )


def test_checkpoint_receive_and_retry_are_source_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _transport_plan(tmp_path, monkeypatch)
    ledger = transport.build_attempt_ledger(plan)
    resume = transport.build_resume_plan(plan, ledger)
    request = transport.build_wave_request(plan, ledger, resume)
    assert request["selected_count"] == 8
    assert all(row["attempt_id"] == "a00" for row in request["selected_attempts"])
    lifecycle, payloads = _checkpoint_lifecycle(
        plan, ledger, resume, request
    )
    transport.validate_lifecycle_receipt(
        lifecycle,
        plan=plan,
        ledger=ledger,
        resume=resume,
        wave_request=request,
    )

    object_root = tmp_path / "objects"
    for file_object, authorization in payloads.items():
        remote_file = object_root.joinpath(*PurePath(file_object).parts)
        remote_file.parent.mkdir(parents=True)
        remote_file.write_bytes(authorization)
    received = transport.receive_wave(
        plan=plan,
        lifecycle_receipt=lifecycle,
        object_root=object_root,
        local_root=tmp_path / "received",
    )
    assert received["partial_count"] == 8
    assert received["rows"][0]["status"] == "partial_safe_to_resume"

    lifecycle_path = tmp_path / "lifecycle.json"
    lifecycle_path.write_bytes(transport.canonical_bytes(lifecycle))
    after = transport.build_attempt_ledger(
        plan, lifecycle_receipt_paths=[lifecycle_path]
    )
    retry = transport.build_resume_plan(plan, after)
    assert retry["resume_wave_index"] == 0
    by_id = {row["shard_id"]: row for row in retry["selected_attempts"]}
    assert by_id[request["selected_attempts"][0]["shard_id"]]["attempt_id"] == "a01"
    assert retry["selected_count"] == 8


class _ObjectStore:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], tuple[str, bytes]] = {}
        self.next_generation = 1
        self.iam_policy: dict[str, Any] = {
            "version": 3,
            "etag": "etag-0",
            "bindings": [
                {
                    "role": "roles/storage.legacyBucketReader",
                    "members": ["projectViewer:test"],
                }
            ],
        }
        self.iam_revision = 0

    def get_object_metadata(
        self, *, bucket: str, object_name: str
    ) -> Mapping[str, Any] | None:
        value = self.objects.get((bucket, object_name))
        if value is None:
            return None
        return {"name": object_name, "generation": value[0]}

    def get_object_bytes(
        self, *, bucket: str, object_name: str, generation: str | None = None
    ) -> bytes | None:
        value = self.objects.get((bucket, object_name))
        if value is None or (generation is not None and generation != value[0]):
            return None
        return value[1]

    def put_object_new(
        self, *, bucket: str, object_name: str, payload: bytes, content_type: str
    ) -> Mapping[str, Any]:
        del content_type
        key = (bucket, object_name)
        if key in self.objects:
            raise FileExistsError(object_name)
        generation = str(self.next_generation)
        self.next_generation += 1
        self.objects[key] = (generation, payload)
        return {"name": object_name, "generation": generation}

    def get_service_account(self, *, email: str) -> Mapping[str, Any]:
        return {"email": email, "uniqueId": "123456789"}

    def test_service_account_act_as(self, *, email: str) -> bool:
        del email
        return True

    def get_bucket_iam_policy(self, *, bucket: str) -> Mapping[str, Any]:
        del bucket
        return deepcopy(self.iam_policy)

    def set_bucket_iam_policy(
        self, *, bucket: str, policy: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        del bucket
        self.iam_revision += 1
        self.iam_policy = {
            **deepcopy(dict(policy)),
            "etag": f"etag-{self.iam_revision}",
        }
        return deepcopy(self.iam_policy)

    def get_instance(self, *, instance_name: str) -> None:
        del instance_name
        return None

    def get_disk_optional(self, *, disk_name: str) -> None:
        del disk_name
        return None


def test_provider_plan_and_stage_are_cloud_neutral_then_create_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _transport_plan(tmp_path, monkeypatch)
    ledger = transport.build_attempt_ledger(plan)
    resume = transport.build_resume_plan(plan, ledger)
    request = transport.build_wave_request(plan, ledger, resume)
    accounts = [
        f"m31ds{index}@{provider.PROJECT}.iam.gserviceaccount.com"
        for index in range(8)
    ]
    provider_plan = provider.build_provider_plan(
        transport_plan=plan,
        ledger=ledger,
        resume=resume,
        wave_request=request,
        image_self_link=(
            "https://www.googleapis.com/compute/v1/projects/"
            "debian-cloud/global/images/debian-12-bookworm-v20260701"
        ),
        image_id="123456789",
        guest_os_features=["GVNIC", "UEFI_COMPATIBLE"],
        worker_service_accounts=accounts,
        local_shard_root=tmp_path / "local-shards",
    )
    assert provider_plan["cloud_mutated"] is False
    assert len(provider_plan["workers"]) == 8
    assert provider_plan["runtime_contract"]["machine_type"] == "c4-standard-16"
    assert provider_plan["runtime_contract"]["checkpoint_pair_granularity"] == 1
    provider.validate_provider_plan(
        provider_plan,
        transport_plan=plan,
        ledger=ledger,
        resume=resume,
        wave_request=request,
    )

    cloud = _ObjectStore()
    first = provider.stage_content(
        provider_plan=provider_plan,
        transport_plan=plan,
        ledger=ledger,
        resume=resume,
        wave_request=request,
        transport=cloud,  # type: ignore[arg-type]
    )
    assert first["cloud_mutated"] is True
    assert all(row["created"] for row in first["content_records"])
    second = provider.stage_content(
        provider_plan=provider_plan,
        transport_plan=plan,
        ledger=ledger,
        resume=resume,
        wave_request=request,
        transport=cloud,  # type: ignore[arg-type]
    )
    assert second["cloud_mutated"] is False
    assert all(not row["created"] for row in second["content_records"])
    assert len(cloud.objects) == len(provider_plan["content_entries"])
    assert provider_plan["startup_script_sha256"] == hashlib.sha256(
        provider.worker_startup_script().encode("utf-8")
    ).hexdigest()
    startup = provider.worker_startup_script()
    assert "ifGenerationMatch=0" in startup
    assert "stop.wait(60)" in startup
    assert 'cmd[-1]="1"' in startup
    assert "pip\" install --no-index" in startup


def test_preempted_wave_cleanup_proves_absence_removes_iam_and_opens_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _transport_plan(tmp_path, monkeypatch)
    ledger = transport.build_attempt_ledger(plan)
    resume = transport.build_resume_plan(plan, ledger)
    request = transport.build_wave_request(plan, ledger, resume)
    accounts = [
        f"m31ds{index}@{provider.PROJECT}.iam.gserviceaccount.com"
        for index in range(8)
    ]
    provider_plan = provider.build_provider_plan(
        transport_plan=plan,
        ledger=ledger,
        resume=resume,
        wave_request=request,
        image_self_link=(
            "https://www.googleapis.com/compute/v1/projects/"
            "debian-cloud/global/images/debian-12-bookworm-v20260701"
        ),
        image_id="123456789",
        guest_os_features=["GVNIC", "UEFI_COMPATIBLE"],
        worker_service_accounts=accounts,
        local_shard_root=tmp_path / "local-shards",
    )
    cloud = _ObjectStore()
    stage = provider.stage_content(
        provider_plan=provider_plan,
        transport_plan=plan,
        ledger=ledger,
        resume=resume,
        wave_request=request,
        transport=cloud,  # type: ignore[arg-type]
    )
    iam = provider.install_worker_iam(
        provider_plan=provider_plan,
        transport_plan=plan,
        ledger=ledger,
        resume=resume,
        wave_request=request,
        transport=cloud,  # type: ignore[arg-type]
        now_unix_seconds=1_800_000_000,
    )
    workers = {row["shard_id"]: row for row in provider_plan["workers"]}
    launch_rows = []
    for index, selected in enumerate(request["selected_attempts"], start=1):
        spec = provider._instance_spec(  # type: ignore[attr-defined]
            provider=provider_plan,
            worker=workers[selected["shard_id"]],
            selected=selected,
            stage_receipt=stage,
        )
        launch_rows.append(
            {
                "shard_id": selected["shard_id"],
                "attempt_id": selected["attempt_id"],
                "instance_id": selected["instance_id"],
                "request_id": provider._uuid_for(  # type: ignore[attr-defined]
                    provider_plan["provider_plan_sha256"],
                    selected["shard_id"],
                    selected["attempt_id"],
                    "create",
                ),
                "spec_sha256": provider.canonical_sha256(spec),
                "provider_instance_id": str(1000 + index),
                "provider_boot_disk_id": str(2000 + index),
                "observed_status": "TERMINATED",
                "created": True,
            }
        )
    launch_core = {
        "schema": provider.LAUNCH_RECEIPT_SCHEMA,
        "provider_plan_sha256": provider_plan["provider_plan_sha256"],
        "wave_request_sha256": request["request_sha256"],
        "claim": {
            "object_name": provider_plan["claim_object"],
            "generation": "999",
            "sha256": "a" * 64,
            "bytes": 1,
            "nonce_sha256": "b" * 64,
            "response_reconciled": False,
        },
        "stage_receipt": stage,
        "iam_receipt": iam,
        "quota_receipt": {"sufficient": True},
        "rows": launch_rows,
        "selected_shard_count": 8,
        "created_instance_count": 8,
        "at_most_eight_c4": True,
        "one_shot_claim_consumed": True,
        "unlisted_vm_created": 0,
        "cloud_mutated": True,
        "current_profile_changed": False,
    }
    launch = {
        **launch_core,
        "receipt_sha256": provider.canonical_sha256(launch_core),
    }
    lifecycle = provider.cleanup_wave(
        provider_plan=provider_plan,
        transport_plan=plan,
        ledger=ledger,
        resume=resume,
        wave_request=request,
        launch_receipt=launch,
        transport=cloud,  # type: ignore[arg-type]
        sleep=lambda _seconds: None,
    )
    assert lifecycle["owned_vm_disk_absent"] is True
    assert lifecycle["worker_iam_removed_before_receive"] is True
    assert all(row["status"] == "failed" for row in lifecycle["attempt_rows"])
    assert cloud.iam_policy["bindings"] == [
        {
            "role": "roles/storage.legacyBucketReader",
            "members": ["projectViewer:test"],
        }
    ]
    lifecycle_path = tmp_path / "preempted-lifecycle.json"
    lifecycle_path.write_bytes(transport.canonical_bytes(lifecycle))
    retry_ledger = transport.build_attempt_ledger(
        plan, lifecycle_receipt_paths=[lifecycle_path]
    )
    retry = transport.build_resume_plan(plan, retry_ledger)
    assert retry["selected_count"] == 8
    assert all(row["attempt_id"] == "a01" for row in retry["selected_attempts"])


# pathlib.PurePath is selected at runtime and does not touch the filesystem.
from pathlib import PurePath
