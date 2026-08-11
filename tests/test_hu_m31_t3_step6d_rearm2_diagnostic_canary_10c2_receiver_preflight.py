from __future__ import annotations

import json
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_10c2_receiver_preflight as subject,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_package as cloud_package,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_adapter as adapter,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as plan,
)

EVALUATION_UNIX_SECONDS = 2_000_000_000
OBSERVED_AT_UNIX_SECONDS = EVALUATION_UNIX_SECONDS - 60


class MemoryReadOnlyBackend:
    backend_id = "unit-memory-read-only-v1"
    fixture_only = True

    def __init__(self, objects: Mapping[str, bytes]) -> None:
        self.objects = dict(objects)
        self.list_calls: list[str] = []
        self.read_calls: list[str] = []

    def list_prefix(self, prefix: str) -> Sequence[Mapping[str, Any]]:
        self.list_calls.append(prefix)
        return [
            {
                "uri": uri,
                "generation": 7,
                "metageneration": 1,
                "bytes": len(self.objects[uri]),
                "sha256": subject.hashlib.sha256(
                    self.objects[uri]
                ).hexdigest(),
                "crc32c": "unit-crc32c",
                "etag": f"unit-etag-{index}",
            }
            for index, uri in enumerate(
                sorted(
                    uri
                    for uri in self.objects
                    if uri.startswith(prefix.rstrip("/") + "/")
                )
            )
        ]

    def read_bytes(self, uri: str, generation: int) -> bytes:
        assert generation == 7
        self.read_calls.append(f"{uri}#{generation}")
        return self.objects[uri]


@pytest.fixture(scope="module")
def package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    target = tmp_path_factory.mktemp("r2diag-10c2-receiver") / "package"
    cloud_package.build_package(output_dir=target)
    return target


def _stage_fixture(
    package: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    preview = adapter.build_preview(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
    )
    job = preview["jobs"][0]
    uploads = [
        adapter.build_fixture_upload(
            preview, job_id=job["job_id"], sequence=sequence
        )
        for sequence in range(1, len(job["work_hand_indices"]) + 1)
    ]
    heartbeats = [
        adapter.build_heartbeat(
            preview, job_id=job["job_id"], uploads=uploads[:sequence]
        )
        for sequence in range(1, len(uploads) + 1)
    ]
    done = adapter.build_done(
        preview,
        job_id=job["job_id"],
        uploads=uploads,
        heartbeats=heartbeats,
    )
    receive = adapter.build_receive(preview, done_records=[done])
    return preview, done, receive


def _tree_bytes(
    package: Path,
    preview: Mapping[str, Any],
    done: Mapping[str, Any],
    *,
    outer_package_manifest: Mapping[str, Any] | None = None,
    direct_stage_identity: Mapping[str, Any] | None = None,
) -> dict[str, bytes]:
    manifest = json.loads((package / "manifest.json").read_text("utf-8"))
    job = preview["jobs"][0]
    source_zip = package / manifest["source_name"]
    by_path: dict[str, bytes] = {}
    by_path["run_contract.json"] = adapter.canonical_bytes(
        manifest["run_contract"]
    )
    by_path["shard_manifest.json"] = (
        package / job["runner_job_manifest"]["path"]
    ).read_bytes()
    with zipfile.ZipFile(source_zip) as archive:
        for index in job["work_hand_indices"]:
            member = next(
                name
                for name in archive.namelist()
                if name.endswith(f"/roots/hand_{index:03d}.json")
            )
            by_path[f"roots/hand_{index:03d}.json"] = archive.read(member)
            identity = {
                "schema": adapter.FIXTURE_SCHEMA,
                "source_role": job["source_role"],
                "hand_index": index,
                "path": f"hands/{job['source_role']}/hand_{index:03d}.json",
                "transport_fixture_only": True,
                "scientific_payload_present": False,
            }
            by_path[identity["path"]] = adapter.canonical_bytes(identity)
    done_identity = {
        "schema": adapter.FIXTURE_SCHEMA,
        "source_role": job["source_role"],
        "path": "DONE.json",
        "artifact_manifest_sha256": adapter.canonical_sha256(
            done["runner_artifact_manifest"]
        ),
        "transport_fixture_only": True,
        "scientific_payload_present": False,
    }
    by_path["DONE.json"] = adapter.canonical_bytes(done_identity)
    records = done["tree_object_records"]
    assert set(by_path) == {row["path"] for row in records}
    for row in records:
        raw = by_path[row["path"]]
        assert len(raw) == row["bytes"]
        assert subject.hashlib.sha256(raw).hexdigest() == row["sha256"]
    direct = subject.build_direct_v1_remote_layout(
        preview,
        outer_package_manifest=outer_package_manifest,
        direct_stage_identity=direct_stage_identity,
    )
    tree_prefix = direct["job_layouts"][0]["tree_prefix"]
    return {
        f"{tree_prefix}/{row['path']}": by_path[row["path"]]
        for row in records
    }


def _outer_bindings(
    package: Path,
    preview: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    outer = subject.build_outer_package_manifest(
        package_dir=package,
        offline_wheel_record={
            "path": (
                "wheels/"
                + subject.direct_transport.EXPECTED_NUMPY_WHEEL_FILENAME
            ),
            "uri_suffix": (
                "wheels/"
                + subject.direct_transport.EXPECTED_NUMPY_WHEEL_FILENAME
            ),
            "sha256": (
                subject.direct_transport.EXPECTED_NUMPY_WHEEL_SHA256
            ),
            "bytes": subject.direct_transport.EXPECTED_NUMPY_WHEEL_BYTES,
            "mode": "0644",
            "kind": "offline_numpy_cp311_manylinux_x86_64_wheel",
        },
    )
    direct = subject.build_direct_stage_identity(
        preview,
        outer_package_manifest=outer,
    )
    return outer, direct


def _bound_preflight(
    package: Path,
    preview: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    outer, direct = _outer_bindings(package, preview)
    preflight = subject.build_read_only_preflight_plan(
        preview,
        outer_package_manifest=outer,
        direct_stage_identity=direct,
    )
    return preflight, outer, direct


def test_remote_inventory_is_exact_and_contains_no_scientific_payload(
    package: Path,
) -> None:
    preview, _, receive = _stage_fixture(package)
    inventory = subject.build_remote_object_inventory(
        preview, receive=receive
    )
    assert inventory["selected_job_ids"] == ["candidate-shard-00"]
    assert inventory["object_count"] == 23
    assert inventory["jobs"][0]["objects"][0]["path"] == "run_contract.json"
    assert inventory["jobs"][0]["objects"][-1]["path"] == "DONE.json"
    assert inventory["identity_envelope_is_not_content_proof"] is True
    assert inventory["remote_write_performed"] is False
    assert inventory["training_eligible"] is False
    assert (
        subject.validate_remote_object_inventory(
            inventory, preview=preview, receive=receive
        )
        == inventory
    )


def test_receiver_downloads_all_bytes_then_materializes_and_calls_runner_validator(
    package: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preview, done, receive = _stage_fixture(package)
    outer, direct = _outer_bindings(package, preview)
    backend = MemoryReadOnlyBackend(
        _tree_bytes(
            package,
            preview,
            done,
            outer_package_manifest=outer,
            direct_stage_identity=direct,
        )
    )
    validator_calls: list[Path] = []

    def validator(path: str | Path) -> dict[str, Any]:
        root = Path(path)
        validator_calls.append(root)
        assert (root / "run_contract.json").is_file()
        assert (root / "shard_manifest.json").is_file()
        assert (root / "DONE.json").is_file()
        return {"schema": "unit_runner_done_v1", "validated": True}

    monkeypatch.setattr(
        subject.runner, "validate_completed_output", validator
    )
    destination = tmp_path / "received-stage"
    result = subject.materialize_and_validate_received_stage(
        preview,
        receive=receive,
        destination_root=destination,
        backend=backend,
        outer_package_manifest=outer,
        direct_stage_identity=direct,
    )
    assert len(validator_calls) == 1
    assert validator_calls[0].name == "candidate-shard-00"
    assert validator_calls[0].parent.name == "jobs"
    assert validator_calls[0].parents[1].name.startswith(
        ".received-stage.10c2-"
    )
    assert (destination / "jobs/candidate-shard-00/DONE.json").is_file()
    assert len(backend.list_calls) == 1
    assert len(backend.read_calls) == 23
    assert result["runner_validate_completed_output_performed"] is True
    assert result["generation_bound_reads_performed"] is True
    assert result["exclusive_hidden_staging_performed"] is True
    assert result["file_fsync_performed"] is True
    assert result["directory_fsync_performed"] is True
    assert result["atomic_rename_performed"] is True
    assert result["destination_noreplace_finalize_performed"] is True
    assert result["staging_artifact_remaining"] is False
    assert result["backend_fixture_only"] is True
    assert result["external_cloud_read_performed"] is False
    assert result["signed_external_query_receipt_present"] is False
    assert result["contract_or_fake_evidence_only"] is True
    assert result["cloud_mutation_performed"] is False
    assert result["claim_created"] is False
    assert result["authorization_created"] is False
    assert result["vm_created"] is False
    assert (
        subject.validate_materialization_result(
            result,
            preview=preview,
            receive=receive,
            outer_package_manifest=outer,
            direct_stage_identity=direct,
        )
        == result
    )


@pytest.mark.parametrize("failure", ["unknown", "missing", "bytes", "sha"])
def test_receiver_fails_before_local_write_on_remote_inventory_or_content_error(
    package: Path, tmp_path: Path, failure: str
) -> None:
    preview, done, receive = _stage_fixture(package)
    objects = _tree_bytes(package, preview, done)
    first_uri = next(iter(objects))
    if failure == "unknown":
        direct = subject.build_direct_v1_remote_layout(preview)
        objects[direct["job_layouts"][0]["tree_prefix"] + "/rogue.json"] = (
            b"rogue"
        )
    elif failure == "missing":
        objects.pop(first_uri)
    elif failure == "bytes":
        objects[first_uri] += b"x"
    else:
        changed = bytearray(objects[first_uri])
        changed[0] ^= 1
        objects[first_uri] = bytes(changed)
    destination = tmp_path / f"bad-{failure}"
    with pytest.raises((ValueError, KeyError)):
        subject.materialize_and_validate_received_stage(
            preview,
            receive=receive,
            destination_root=destination,
            backend=MemoryReadOnlyBackend(objects),
        )
    assert not destination.exists()


def test_receiver_rejects_nonfresh_destination(
    package: Path, tmp_path: Path
) -> None:
    preview, done, receive = _stage_fixture(package)
    destination = tmp_path / "already-exists"
    destination.mkdir()
    with pytest.raises(FileExistsError, match="fresh"):
        subject.materialize_and_validate_received_stage(
            preview,
            receive=receive,
            destination_root=destination,
            backend=MemoryReadOnlyBackend(
                _tree_bytes(package, preview, done)
            ),
        )


def test_runner_validation_failure_removes_hidden_staging_and_leaves_final_absent(
    package: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview, done, receive = _stage_fixture(package)
    destination = tmp_path / "validation-fails"

    def reject(_: str | Path) -> dict[str, Any]:
        raise ValueError("injected runner validation failure")

    monkeypatch.setattr(subject.runner, "validate_completed_output", reject)
    with pytest.raises(ValueError, match="runner validation failure"):
        subject.materialize_and_validate_received_stage(
            preview,
            receive=receive,
            destination_root=destination,
            backend=MemoryReadOnlyBackend(
                _tree_bytes(package, preview, done)
            ),
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".validation-fails.10c2-*.staging"))


def test_noreplace_primitive_preserves_concurrent_destination(
    tmp_path: Path,
) -> None:
    staging = tmp_path / ".direct-finalize.staging"
    destination = tmp_path / "direct-finalize"
    staging.mkdir()
    (staging / "receiver.txt").write_text("receiver", encoding="utf-8")
    destination.mkdir()
    (destination / "concurrent.txt").write_text("concurrent", encoding="utf-8")

    with pytest.raises(FileExistsError, match="no-replace"):
        subject._finalize_directory_noreplace(staging, destination)

    assert (staging / "receiver.txt").read_text(encoding="utf-8") == "receiver"
    assert (
        destination / "concurrent.txt"
    ).read_text(encoding="utf-8") == "concurrent"


def test_receiver_finalization_race_preserves_other_destination_and_cleans_staging(
    package: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview, done, receive = _stage_fixture(package)
    destination = tmp_path / "finalize-race"
    real_finalize = subject._finalize_directory_noreplace
    monkeypatch.setattr(
        subject.runner,
        "validate_completed_output",
        lambda _: {"schema": "unit_runner_done_v1", "validated": True},
    )

    def race(source: Path, target: Path) -> None:
        assert target == destination
        target.mkdir()
        (target / "other-owner.txt").write_text("other", encoding="utf-8")
        real_finalize(source, target)

    monkeypatch.setattr(subject, "_finalize_directory_noreplace", race)
    with pytest.raises(FileExistsError, match="no-replace"):
        subject.materialize_and_validate_received_stage(
            preview,
            receive=receive,
            destination_root=destination,
            backend=MemoryReadOnlyBackend(
                _tree_bytes(package, preview, done)
            ),
        )

    assert (destination / "other-owner.txt").read_text(
        encoding="utf-8"
    ) == "other"
    assert not list(tmp_path.glob(".finalize-race.10c2-*.staging"))


def test_receiver_fsyncs_created_directory_tree_and_final_parent(
    package: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview, done, receive = _stage_fixture(package)
    destination = tmp_path / "fsync-tree"
    synced: list[Path] = []
    monkeypatch.setattr(
        subject.runner,
        "validate_completed_output",
        lambda _: {"schema": "unit_runner_done_v1", "validated": True},
    )

    def record(path: Path) -> None:
        synced.append(Path(path))

    monkeypatch.setattr(subject, "_fsync_directory", record)
    result = subject.materialize_and_validate_received_stage(
        preview,
        receive=receive,
        destination_root=destination,
        backend=MemoryReadOnlyBackend(_tree_bytes(package, preview, done)),
    )

    assert result["directory_fsync_performed"] is True
    assert destination in synced
    assert synced.count(tmp_path) >= 3
    assert any(path.name == "candidate-shard-00" for path in synced)
    assert any(
        path.name.startswith(".fsync-tree.10c2-")
        and path.name.endswith(".staging")
        for path in synced
    )


def test_receiver_directory_fsync_failure_removes_owned_staging(
    package: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview, done, receive = _stage_fixture(package)
    destination = tmp_path / "fsync-fails"
    real_fsync = subject._fsync_directory
    injected = False
    monkeypatch.setattr(
        subject.runner,
        "validate_completed_output",
        lambda _: {"schema": "unit_runner_done_v1", "validated": True},
    )

    def fail_once(path: Path) -> None:
        nonlocal injected
        if path.name == "candidate-shard-00" and not injected:
            injected = True
            raise OSError("injected directory fsync failure")
        real_fsync(path)

    monkeypatch.setattr(subject, "_fsync_directory", fail_once)
    with pytest.raises(OSError, match="directory fsync failure"):
        subject.materialize_and_validate_received_stage(
            preview,
            receive=receive,
            destination_root=destination,
            backend=MemoryReadOnlyBackend(
                _tree_bytes(package, preview, done)
            ),
        )

    assert injected is True
    assert not destination.exists()
    assert not list(tmp_path.glob(".fsync-fails.10c2-*.staging"))


def test_controller_receipt_is_impossible_until_all_vms_absent(
    package: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preview, done, receive = _stage_fixture(package)
    monkeypatch.setattr(
        subject.runner,
        "validate_completed_output",
        lambda _: {
            "schema": "unit_runner_done_v1",
            "validated": True,
        },
    )
    result = subject.materialize_and_validate_received_stage(
        preview,
        receive=receive,
        destination_root=tmp_path / "received",
        backend=MemoryReadOnlyBackend(_tree_bytes(package, preview, done)),
    )
    instance_names = {
        "candidate-shard-00": "r2diag-s1-candidate-shard-00"
    }
    present = subject.build_vm_absence_observation(
        preview,
        instance_names_by_job=instance_names,
        present_instance_names=["r2diag-s1-candidate-shard-00"],
        observation_source="injected-unit-read-only-backend",
        query_performed=False,
        fixture_only=True,
    )
    with pytest.raises(ValueError, match="every worker VM absent"):
        subject.build_controller_receipt(
            preview,
            receive=receive,
            materialization_result=result,
            vm_absence_observation=present,
        )
    absent = subject.build_vm_absence_observation(
        preview,
        instance_names_by_job=instance_names,
        present_instance_names=[],
        observation_source="injected-unit-read-only-backend",
        query_performed=False,
        fixture_only=True,
    )
    receipt = subject.build_controller_receipt(
        preview,
        receive=receive,
        materialization_result=result,
        vm_absence_observation=absent,
    )
    assert receipt["all_worker_vms_absent"] is True
    assert receipt["runner_content_validated"] is True
    assert receipt["fixture_only"] is True
    assert receipt["external_read_only_evidence"] is False
    assert receipt["launch_authorized"] is False
    assert receipt["cloud_mutation_performed"] is False


def _external_preflight_observations(
    preflight: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    prefix = subject.build_prefix_observation(
        preflight,
        package_objects=[
            {
                **row,
                "generation": index + 1,
                "metageneration": 1,
                "crc32c": f"unit-crc32c-{index}",
                "etag": f"unit-etag-{index}",
            }
            for index, row in enumerate(
                preflight["requirements"]["prefix"][
                    "expected_package_objects"
                ]
            )
        ],
        stage_object_uris=[],
        observation_source="injected-external-read-only-observation",
        source_identity="unit-prefix-observer-v1",
        observed_at_unix_seconds=OBSERVED_AT_UNIX_SECONDS,
        query_performed=True,
        fixture_only=False,
    )
    capacity = subject.build_capacity_observation(
        preflight,
        available_vcpu=preflight["requirements"]["capacity"][
            "required_vcpu"
        ],
        provider_metric_mapping=subject.CAPACITY_PROVIDER_METRIC_MAPPING,
        observation_source="injected-external-read-only-observation",
        source_identity="unit-capacity-observer-v1",
        observed_at_unix_seconds=OBSERVED_AT_UNIX_SECONDS,
        query_performed=True,
        fixture_only=False,
    )
    instances = subject.build_instance_observation(
        preflight,
        observed_instances=[],
        observation_source="injected-external-read-only-observation",
        source_identity="unit-instance-observer-v1",
        observed_at_unix_seconds=OBSERVED_AT_UNIX_SECONDS,
        query_performed=True,
        fixture_only=False,
    )
    price = subject.build_spot_price_observation(
        preflight,
        observed_price_usd_per_vm_hour=0.40,
        observation_source="injected-external-read-only-observation",
        source_identity="unit-price-observer-v1",
        official_source_url="https://cloud.google.com/spot-vms/pricing",
        observed_at_unix_seconds=OBSERVED_AT_UNIX_SECONDS,
        sku_effective_at_unix_seconds=(
            OBSERVED_AT_UNIX_SECONDS - 3600
        ),
        query_performed=True,
        fixture_only=False,
    )
    return prefix, instances, capacity, price


def test_preflight_separates_content_addressed_package_from_fresh_stage(
    package: Path,
) -> None:
    preview, _, _ = _stage_fixture(package)
    preflight, outer, direct = _bound_preflight(package, preview)
    prefix = preflight["requirements"]["prefix"]
    assert "/packages/" in prefix["package_prefix"]
    assert outer["outer_package_identity_sha256"] in prefix["package_prefix"]
    assert len(prefix["expected_package_objects"]) == len(
        subject.direct_transport.build_outer_package_inventory(outer)[
            "records"
        ]
    )
    assert prefix["allowed_package_inventory_states"] == [
        "exact_empty_provisioning_required",
        "exact_expected_subset_provisioning_required",
        "exact_complete_immutable_reuse",
    ]
    assert prefix["stage_prefix_must_be_exactly_empty"] is True
    assert "/stages/" in prefix["stage_prefix"]
    assert direct["direct_stage_identity_sha256"] in prefix["stage_prefix"]
    assert prefix["result_prefix"].startswith(prefix["stage_prefix"] + "/")
    assert prefix["attempt_control_prefix"].startswith(
        prefix["stage_prefix"] + "/"
    )
    assert prefix["package_prefix"] != prefix["stage_prefix"]
    assert prefix["package_and_stage_prefix_disjoint"] is True
    assert (
        preflight["requirements"]["spot_price"][
            "ceiling_usd_per_vm_hour"
        ]
        == adapter.legacy_vm.SPOT_PRICE_CEILING_USD_PER_VM_HOUR
        == 0.57
    )
    assert preflight["cloud_query_performed"] is False
    assert preflight["launch_authorized"] is False
    assert preflight["launch_ready"] is False
    assert (
        subject.validate_read_only_preflight_plan(
            preflight,
            preview=preview,
            outer_package_manifest=outer,
            direct_stage_identity=direct,
        )
        == preflight
    )


def test_outer_bindings_are_required_for_any_real_preflight_contract(
    package: Path,
) -> None:
    preview, _, _ = _stage_fixture(package)
    local_plan = subject.build_read_only_preflight_plan(preview)
    assert local_plan["status"] == (
        "legacy_local_contract_only_outer_bindings_required"
    )
    assert local_plan["real_read_only_preflight_contract_eligible"] is False
    prefix, instances, capacity, price = _external_preflight_observations(
        local_plan
    )
    result = subject.evaluate_read_only_preflight(
        local_plan,
        preview=preview,
        prefix_observation=prefix,
        instance_observation=instances,
        capacity_observation=capacity,
        price_observation=price,
        evaluation_unix_seconds=EVALUATION_UNIX_SECONDS,
    )
    assert result["observation_contract_passed"] is False
    assert "outer_package_and_direct_stage_identity_missing" in result[
        "failures"
    ]
    assert result["external_read_only_evidence"] is False
    assert result["launch_authorized"] is False


def test_outer_manifest_is_non_self_referential_and_tamper_rejected(
    package: Path,
) -> None:
    preview, _, _ = _stage_fixture(package)
    outer, direct = _outer_bindings(package, preview)
    assert all(row["path"] != "outer-manifest.json" for row in outer["objects"])
    identity_before = outer["outer_package_identity_sha256"]
    layout = subject.build_direct_v1_remote_layout(
        preview,
        outer_package_manifest=outer,
        direct_stage_identity=direct,
    )
    manifest_record = next(
        row
        for row in layout["package_objects"]
        if row["path"] == "outer-manifest.json"
    )
    assert manifest_record["sha256"] == subject.hashlib.sha256(
        subject.direct_transport.canonical_bytes(outer)
    ).hexdigest()
    assert outer["outer_package_identity_sha256"] == identity_before
    tampered = deepcopy(outer)
    tampered["objects"][0]["bytes"] += 1
    with pytest.raises(ValueError, match="outer package"):
        subject.validate_outer_package_manifest(tampered, preview=preview)


def test_retry_keeps_direct_stage_and_result_identity_but_changes_control(
    package: Path,
) -> None:
    attempt0, _, _ = _stage_fixture(package)
    job = attempt0["jobs"][0]
    prior = adapter.fixture_progress(
        attempt0,
        adapter.empty_snapshot(attempt0),
        job_id=job["job_id"],
        completed_count=1,
    )
    attempt1 = adapter.build_preview(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
        attempt_index=1,
        prior_preview=attempt0,
        prior_snapshot=prior,
    )
    outer0, direct0 = _outer_bindings(package, attempt0)
    outer1, direct1 = _outer_bindings(package, attempt1)
    layout0 = subject.build_direct_v1_remote_layout(
        attempt0,
        outer_package_manifest=outer0,
        direct_stage_identity=direct0,
    )
    layout1 = subject.build_direct_v1_remote_layout(
        attempt1,
        outer_package_manifest=outer1,
        direct_stage_identity=direct1,
    )
    assert outer0["outer_package_identity_sha256"] == outer1[
        "outer_package_identity_sha256"
    ]
    assert direct0["direct_stage_identity_sha256"] == direct1[
        "direct_stage_identity_sha256"
    ]
    assert layout0["stage_prefix"] == layout1["stage_prefix"]
    assert layout0["result_prefix"] == layout1["result_prefix"]
    assert layout0["attempt_control_prefix"] != layout1[
        "attempt_control_prefix"
    ]
    assert layout0["job_layouts"][0]["instance_name"] != layout1[
        "job_layouts"
    ][0]["instance_name"]


def test_local_preflight_contract_never_claims_external_pass_or_authorizes_launch(
    package: Path,
) -> None:
    preview, _, _ = _stage_fixture(package)
    preflight, outer, direct = _bound_preflight(package, preview)
    prefix, instances, capacity, price = _external_preflight_observations(
        preflight
    )
    result = subject.evaluate_read_only_preflight(
        preflight,
        preview=preview,
        outer_package_manifest=outer,
        direct_stage_identity=direct,
        prefix_observation=prefix,
        instance_observation=instances,
        capacity_observation=capacity,
        price_observation=price,
        evaluation_unix_seconds=EVALUATION_UNIX_SECONDS,
    )
    assert result["observation_contract_passed"] is True
    assert result["preflight_passed"] is False
    assert result["status"] == (
        "local_observation_contract_passed_external_receipt_required"
    )
    assert result["failures"] == [
        "signed_external_query_receipt_missing"
    ]
    assert result["external_read_only_evidence"] is False
    assert result["signed_external_query_receipt_present"] is False
    assert result["cloud_query_performed"] is False
    assert result["launch_authorized"] is False
    assert result["launch_ready"] is False
    assert result["package_provisioning_required"] is False
    assert result["cloud_mutation_performed"] is False
    assert result["claim_created"] is False
    assert result["authorization_created"] is False
    assert result["vm_created"] is False
    assert (
        subject.validate_read_only_preflight_result(
            result,
            plan=preflight,
            preview=preview,
            outer_package_manifest=outer,
            direct_stage_identity=direct,
        )
        == result
    )


def test_empty_package_prefix_passes_only_as_provisioning_required(
    package: Path,
) -> None:
    preview, _, _ = _stage_fixture(package)
    preflight, outer, direct = _bound_preflight(package, preview)
    _, instances, capacity, price = _external_preflight_observations(
        preflight
    )
    prefix = subject.build_prefix_observation(
        preflight,
        package_objects=[],
        stage_object_uris=[],
        observation_source="injected-external-read-only-observation",
        source_identity="unit-prefix-observer-v1",
        observed_at_unix_seconds=OBSERVED_AT_UNIX_SECONDS,
        query_performed=True,
        fixture_only=False,
    )
    result = subject.evaluate_read_only_preflight(
        preflight,
        preview=preview,
        outer_package_manifest=outer,
        direct_stage_identity=direct,
        prefix_observation=prefix,
        instance_observation=instances,
        capacity_observation=capacity,
        price_observation=price,
        evaluation_unix_seconds=EVALUATION_UNIX_SECONDS,
    )
    assert prefix["package_inventory_state"] == (
        "exact_empty_provisioning_required"
    )
    assert result["observation_contract_passed"] is True
    assert result["preflight_passed"] is False
    assert result["package_provisioning_required"] is True
    assert result["package_provisioning_authorized"] is False
    assert result["launch_ready"] is False
    assert result["launch_authorized"] is False


def test_exact_package_subset_is_restart_safe_and_lists_missing_objects(
    package: Path,
) -> None:
    preview, _, _ = _stage_fixture(package)
    preflight, outer, direct = _bound_preflight(package, preview)
    prefix, instances, capacity, price = _external_preflight_observations(
        preflight
    )
    subset = prefix["package_objects"][:-2]
    prefix = subject.build_prefix_observation(
        preflight,
        package_objects=subset,
        stage_object_uris=[],
        observation_source="injected-external-read-only-observation",
        source_identity="unit-prefix-observer-v1",
        observed_at_unix_seconds=OBSERVED_AT_UNIX_SECONDS,
        query_performed=True,
        fixture_only=False,
    )
    result = subject.evaluate_read_only_preflight(
        preflight,
        preview=preview,
        outer_package_manifest=outer,
        direct_stage_identity=direct,
        prefix_observation=prefix,
        instance_observation=instances,
        capacity_observation=capacity,
        price_observation=price,
        evaluation_unix_seconds=EVALUATION_UNIX_SECONDS,
    )
    assert prefix["package_inventory_state"] == (
        "exact_expected_subset_provisioning_required"
    )
    assert len(prefix["missing_package_objects"]) == 2
    assert result["observation_contract_passed"] is True
    assert result["preflight_passed"] is False
    assert result["package_provisioning_required"] is True
    assert result["launch_ready"] is False


def test_expected_instance_name_collision_is_a_prelaunch_no_go(
    package: Path,
) -> None:
    preview, _, _ = _stage_fixture(package)
    preflight, outer, direct = _bound_preflight(package, preview)
    prefix, _, capacity, price = _external_preflight_observations(preflight)
    expected = preflight["requirements"]["instances"][
        "expected_instances"
    ][0]["instance_name"]
    instances = subject.build_instance_observation(
        preflight,
        observed_instances=[
            {
                "instance_name": expected,
                "zone": adapter.DEFAULT_ZONE,
                "status": "RUNNING",
            }
        ],
        observation_source="injected-external-read-only-observation",
        source_identity="unit-instance-observer-v1",
        observed_at_unix_seconds=OBSERVED_AT_UNIX_SECONDS,
        query_performed=True,
        fixture_only=False,
    )
    result = subject.evaluate_read_only_preflight(
        preflight,
        preview=preview,
        outer_package_manifest=outer,
        direct_stage_identity=direct,
        prefix_observation=prefix,
        instance_observation=instances,
        capacity_observation=capacity,
        price_observation=price,
        evaluation_unix_seconds=EVALUATION_UNIX_SECONDS,
    )
    assert result["preflight_passed"] is False
    assert "expected_instance_name_collision" in result["failures"]
    assert result["launch_authorized"] is False


def test_expired_observations_fail_closed(package: Path) -> None:
    preview, _, _ = _stage_fixture(package)
    preflight, outer, direct = _bound_preflight(package, preview)
    prefix, instances, capacity, price = _external_preflight_observations(
        preflight
    )
    result = subject.evaluate_read_only_preflight(
        preflight,
        preview=preview,
        outer_package_manifest=outer,
        direct_stage_identity=direct,
        prefix_observation=prefix,
        instance_observation=instances,
        capacity_observation=capacity,
        price_observation=price,
        evaluation_unix_seconds=(
            OBSERVED_AT_UNIX_SECONDS
            + subject.PRICE_OBSERVATION_MAX_AGE_SECONDS
            + 1
        ),
    )
    assert result["preflight_passed"] is False
    assert result["failures"] == [
        "prefix_observation_stale_or_from_future",
        "instance_observation_stale_or_from_future",
        "capacity_observation_stale_or_from_future",
        "spot_price_observation_stale_or_from_future",
        "signed_external_query_receipt_missing",
    ]
    assert result["launch_authorized"] is False


def test_missing_price_observation_fails_closed_without_inventing_value(
    package: Path,
) -> None:
    preview, _, _ = _stage_fixture(package)
    preflight, outer, direct = _bound_preflight(package, preview)
    prefix, instances, capacity, _ = _external_preflight_observations(
        preflight
    )
    price = subject.build_spot_price_observation(
        preflight,
        observed_price_usd_per_vm_hour=None,
        observation_source="not-observed",
        source_identity="unit-unobserved-price-v1",
        official_source_url=None,
        observed_at_unix_seconds=OBSERVED_AT_UNIX_SECONDS,
        sku_effective_at_unix_seconds=None,
        query_performed=False,
        fixture_only=True,
    )
    result = subject.evaluate_read_only_preflight(
        preflight,
        preview=preview,
        outer_package_manifest=outer,
        direct_stage_identity=direct,
        prefix_observation=prefix,
        instance_observation=instances,
        capacity_observation=capacity,
        price_observation=price,
        evaluation_unix_seconds=EVALUATION_UNIX_SECONDS,
    )
    assert result["preflight_passed"] is False
    assert result["failures"] == [
        "spot_price_external_observation_missing",
        "signed_external_query_receipt_missing",
    ]
    assert result["price_invented_or_cached"] is False
    assert result["launch_authorized"] is False


@pytest.mark.parametrize(
    ("field", "expected_failure"),
    [
        ("package", "content_addressed_package_inventory_invalid"),
        ("stage", "direct_v1_stage_prefix_not_empty"),
    ],
)
def test_prefix_mismatch_or_prior_result_fails_closed(
    package: Path, field: str, expected_failure: str
) -> None:
    preview, _, _ = _stage_fixture(package)
    preflight, outer, direct = _bound_preflight(package, preview)
    prefix, instances, capacity, price = _external_preflight_observations(
        preflight
    )
    if field == "package":
        package_objects = deepcopy(prefix["package_objects"])
        package_objects[0]["sha256"] = "f" * 64
        with pytest.raises(
            ValueError, match="unknown or mismatched object"
        ):
            subject.build_prefix_observation(
                preflight,
                package_objects=package_objects,
                stage_object_uris=[],
                observation_source="injected-external-read-only-observation",
                source_identity="unit-prefix-observer-v1",
                observed_at_unix_seconds=OBSERVED_AT_UNIX_SECONDS,
                query_performed=True,
                fixture_only=False,
            )
        return
    else:
        prefix = subject.build_prefix_observation(
            preflight,
            package_objects=prefix["package_objects"],
            stage_object_uris=[
                preflight["requirements"]["prefix"]["stage_prefix"]
                + "/rogue-before-launch"
            ],
            observation_source="injected-external-read-only-observation",
            source_identity="unit-prefix-observer-v1",
            observed_at_unix_seconds=OBSERVED_AT_UNIX_SECONDS,
            query_performed=True,
            fixture_only=False,
        )
    result = subject.evaluate_read_only_preflight(
        preflight,
        preview=preview,
        outer_package_manifest=outer,
        direct_stage_identity=direct,
        prefix_observation=prefix,
        instance_observation=instances,
        capacity_observation=capacity,
        price_observation=price,
        evaluation_unix_seconds=EVALUATION_UNIX_SECONDS,
    )
    assert result["preflight_passed"] is False
    assert expected_failure in result["failures"]
    assert result["launch_authorized"] is False


@pytest.mark.parametrize("bad", [False, True, -1, float("nan")])
def test_spot_price_is_strict_and_never_implicitly_observed(
    package: Path, bad: object
) -> None:
    preview, _, _ = _stage_fixture(package)
    preflight, _, _ = _bound_preflight(package, preview)
    with pytest.raises(ValueError):
        subject.build_spot_price_observation(
            preflight,
            observed_price_usd_per_vm_hour=bad,  # type: ignore[arg-type]
            observation_source="invalid",
            source_identity="unit-invalid-price-v1",
            official_source_url="https://cloud.google.com/spot-vms/pricing",
            observed_at_unix_seconds=OBSERVED_AT_UNIX_SECONDS,
            sku_effective_at_unix_seconds=(
                OBSERVED_AT_UNIX_SECONDS - 3600
            ),
            query_performed=True,
            fixture_only=False,
        )


def test_spot_price_rejects_nonofficial_source_url(package: Path) -> None:
    preview, _, _ = _stage_fixture(package)
    preflight, _, _ = _bound_preflight(package, preview)
    with pytest.raises(ValueError, match="official"):
        subject.build_spot_price_observation(
            preflight,
            observed_price_usd_per_vm_hour=0.40,
            observation_source="invalid",
            source_identity="unit-invalid-price-source-v1",
            official_source_url="https://example.com/pricing",
            observed_at_unix_seconds=OBSERVED_AT_UNIX_SECONDS,
            sku_effective_at_unix_seconds=(
                OBSERVED_AT_UNIX_SECONDS - 3600
            ),
            query_performed=True,
            fixture_only=False,
        )
