from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tarfile
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import hu_m31_t3_post_training_gcp_controller_v1 as controller
from ofc_regular import hu_m31_t3_post_training_gcp_v1 as subject


IMAGE = (
    "https://www.googleapis.com/compute/v1/projects/"
    "debian-cloud/global/images/debian-12-bookworm-v20260701"
)
TOKEN = "12345678-1234-4234-9234-123456789abc"


def _module(name: str, filename: str) -> Any:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _tar(path: Path, top: str) -> None:
    source = path.parent / f"{path.stem}-src" / top
    source.mkdir(parents=True)
    (source / "marker").write_bytes(b"pinned")
    with tarfile.open(path, "w:gz") as archive:
        archive.add(source, arcname=top)


def _zip(path: Path, top: str) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{top}/pip-99-py3-none-any.whl", b"wheel")


def _fake_bundle_resolution(
    *,
    runtime: Path,
    wheels: Path,
    accepted: Path,
    diagnostic: Path,
    manifest: Path,
    ready: Path,
    wheelhouse_source_manifest: Path,
) -> dict[str, Any]:
    return {
        "manifest": {
            "manifest_sha256": "1" * 64,
            "source_inventory_sha256": "2" * 64,
            "model_inventory_sha256": "3" * 64,
            "model_inventory": [{} for _ in range(11)],
            "wheelhouse": {"entries_sha256": "4" * 64},
        },
        "ready": {"ready_sha256": "5" * 64},
        "manifest_path": manifest.resolve(),
        "ready_path": ready.resolve(),
        "runtime_archive_path": runtime.resolve(),
        "wheelhouse_archive_path": wheels.resolve(),
        "wheelhouse_source_manifest_path": (
            wheelhouse_source_manifest.resolve()
        ),
        "accepted_library_path": accepted.resolve(),
        "diagnostic_library_path": diagnostic.resolve(),
    }


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_path = tmp_path / "abr-plan.json"
    plan_path.write_bytes(subject.canonical_bytes({"test": "plan"}))
    runtime = tmp_path / "runtime.tar.gz"
    wheels = tmp_path / "wheelhouse.zip"
    accepted = tmp_path / "accepted.so"
    diagnostic = tmp_path / "diagnostic.so"
    bundle_manifest = tmp_path / "ABR_RUNTIME_BUNDLE_MANIFEST.json"
    bundle_ready = tmp_path / "ABR_RUNTIME_BUNDLE_READY.json"
    wheelhouse_source_manifest = tmp_path / "wheelhouse-source.json"
    _tar(runtime, "runtime")
    _zip(wheels, "wheelhouse")
    accepted.write_bytes(b"accepted")
    diagnostic.write_bytes(b"diagnostic")
    bundle_manifest.write_bytes(subject.canonical_bytes({"bundle": "manifest"}))
    bundle_ready.write_bytes(subject.canonical_bytes({"bundle": "ready"}))
    wheelhouse_source_manifest.write_bytes(
        subject.canonical_bytes({"wheelhouse": "source"})
    )
    resolution = _fake_bundle_resolution(
        runtime=runtime,
        wheels=wheels,
        accepted=accepted,
        diagnostic=diagnostic,
        manifest=bundle_manifest,
        ready=bundle_ready,
        wheelhouse_source_manifest=wheelhouse_source_manifest,
    )
    monkeypatch.setattr(
        subject.runtime_bundle,
        "resolve_bundle",
        lambda *_args, **_kwargs: resolution,
    )
    sources = [
        subject._source(  # type: ignore[attr-defined]
            kind="abr_plan",
            path=plan_path,
            relative_path="static/abr_plan/PLAN.json",
        ),
        subject._source(  # type: ignore[attr-defined]
            kind="runtime_archive",
            path=runtime,
            relative_path="static/runtime/runtime.tar.gz",
        ),
        subject._source(  # type: ignore[attr-defined]
            kind="abr_runtime_bundle_manifest",
            path=bundle_manifest,
            relative_path=(
                "static/runtime_bundle/ABR_RUNTIME_BUNDLE_MANIFEST.json"
            ),
        ),
        subject._source(  # type: ignore[attr-defined]
            kind="abr_runtime_bundle_ready",
            path=bundle_ready,
            relative_path=(
                "static/runtime_bundle/ABR_RUNTIME_BUNDLE_READY.json"
            ),
        ),
        subject._source(  # type: ignore[attr-defined]
            kind="abr_runtime_wheelhouse_source_manifest",
            path=wheelhouse_source_manifest,
            relative_path=(
                "static/runtime_bundle/wheelhouse_source_manifest.json"
            ),
        ),
        subject._source(  # type: ignore[attr-defined]
            kind="wheelhouse_archive",
            path=wheels,
            relative_path="static/wheelhouse/wheelhouse.zip",
        ),
        subject._source(  # type: ignore[attr-defined]
            kind="accepted_library",
            path=accepted,
            relative_path="static/accepted_library/accepted.so",
        ),
        subject._source(  # type: ignore[attr-defined]
            kind="diagnostic_library",
            path=diagnostic,
            relative_path="static/diagnostic_library/diagnostic.so",
        ),
    ]
    jobs = [
        {
            "ordinal": index,
            "job_id": f"abr-pair-{index:04d}",
            "pair_index": index,
            "output_relative_path": f"pairs/pair-{index:06d}.json",
        }
        for index in range(50)
    ]
    contract = subject._base_contract(  # type: ignore[attr-defined]
        run_name="m31-post-training-test",
        bucket="m31-post-training-test",
        workload="abr_pilot50",
        sources=sources,
        jobs=jobs,
        scientific_identity={
            "pair_count": 50,
            "accepted_library_sha256": hashlib.sha256(b"accepted").hexdigest(),
            "diagnostic_library_sha256": hashlib.sha256(b"diagnostic").hexdigest(),
            "abr_runtime_bundle_required": True,
            "abr_runtime_bundle_manifest_sha256": "1" * 64,
            "abr_runtime_bundle_manifest_file_sha256": subject.sha256_file(
                bundle_manifest
            ),
            "abr_runtime_bundle_ready_sha256": "5" * 64,
            "abr_runtime_source_inventory_sha256": "2" * 64,
            "abr_runtime_model_inventory_sha256": "3" * 64,
            "abr_runtime_wheelhouse_entries_sha256": "4" * 64,
            "real_behavior_model_count": 11,
            "real_behavior_models_deserialized": True,
            "actor_observation_only": True,
            "synthetic_values_allowed": False,
        },
    )
    monkeypatch.setattr(
        subject.abr_teacher, "validate_plan", lambda value: deepcopy(dict(value))
    )
    monkeypatch.setattr(
        subject.abr_teacher,
        "validate_pair_evidence",
        lambda value, **_kwargs: deepcopy(dict(value)),
    )
    output = tmp_path / "smoke-output"
    pair = output / "pairs" / "pair-000000.json"
    pair.parent.mkdir(parents=True)
    pair.write_bytes(subject.canonical_bytes({"pair_index": 0}))
    record = {
        "relative_path": "pairs/pair-000000.json",
        "sha256": subject.sha256_file(pair),
        "bytes": pair.stat().st_size,
    }
    manifest_core = {
        "schema": subject.RESULT_MANIFEST_SCHEMA,
        "contract_sha256": contract["contract_sha256"],
        "job_id": "abr-pair-0000",
        "attempt_id": "local",
        "files": [record],
        "file_count": 1,
        "scientific": {
            "new_pairs": 1,
            "synthetic_values_used": False,
            "opponent_private_discards_used": False,
        },
        "manifest_published_after_files": True,
        "create_only": True,
        "hidden_opponent_discard_used": False,
        "synthetic_abr_used": False,
        "current_profile_changed": False,
    }
    manifest = {
        **manifest_core,
        "manifest_sha256": subject.canonical_sha256(manifest_core),
    }
    smoke = subject.build_local_smoke_receipt(
        contract=contract,
        output_root=output,
        manifest=manifest,
        filesystem_type="ext4",
        production_executor=True,
    )
    contract_path = tmp_path / "contract.json"
    smoke_path = tmp_path / "smoke.json"
    contract_path.write_bytes(subject.canonical_bytes(contract))
    smoke_path.write_bytes(subject.canonical_bytes(smoke))
    cloud_plan = subject.build_cloud_plan(
        contract_path=contract_path,
        smoke_receipt_path=smoke_path,
        image_self_link=IMAGE,
        image_id="123456789",
        guest_os_features=["GVNIC", "UEFI_COMPATIBLE"],
        worker_service_accounts=[
            f"m31pt{index:02d}@{subject.PROJECT}.iam.gserviceaccount.com"
            for index in range(8)
        ],
    )
    return contract, cloud_plan


def _cloud(plan: dict[str, Any], tmp_path: Path) -> Any:
    helper = _module(
        f"_m31_post_cloud_helper_{tmp_path.name}",
        "test_hu_m31_t3_step6d_fresh_quality_gcp_bridge_v1.py",
    )
    return helper._FakeCloud(
        image_link=plan["image"]["self_link"],
        image_id=plan["image"]["id"],
        features=plan["image"]["guest_os_features"],
    )


def _publish(
    cloud: Any,
    plan: dict[str, Any],
    wave: dict[str, Any],
) -> None:
    for selected in wave["selected"]:
        pair_index = int(selected["job_id"].rsplit("-", 1)[1])
        payload = subject.canonical_bytes({"pair_index": pair_index})
        relative = f"pairs/pair-{pair_index:06d}.json"
        cloud.put_object_new(
            bucket=plan["bucket"],
            object_name=f"{selected['result_prefix']}/files/{relative}",
            payload=payload,
            content_type="application/json",
        )
        record = {
            "relative_path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
        manifest_core = {
            "schema": subject.RESULT_MANIFEST_SCHEMA,
            "contract_sha256": plan["contract"]["contract_sha256"],
            "job_id": selected["job_id"],
            "attempt_id": selected["attempt_id"],
            "files": [record],
            "file_count": 1,
            "scientific": {
                "new_pairs": 1,
                "synthetic_values_used": False,
                "opponent_private_discards_used": False,
            },
            "manifest_published_after_files": True,
            "create_only": True,
            "hidden_opponent_discard_used": False,
            "synthetic_abr_used": False,
            "current_profile_changed": False,
        }
        manifest = {
            **manifest_core,
            "manifest_sha256": subject.canonical_sha256(manifest_core),
        }
        cloud.put_object_new(
            bucket=plan["bucket"],
            object_name=f"{selected['result_prefix']}/manifest.json",
            payload=subject.canonical_bytes(manifest),
            content_type="application/json",
        )


def test_cloud_plan_is_exact_max8_and_real_smoke_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, plan = _fixture(tmp_path, monkeypatch)
    checked = subject.validate_cloud_plan(plan, replay_sources=True)
    assert checked["max_concurrent_vms"] == 8
    assert checked["machine_type"] == "c4-standard-16"
    assert checked["contract"]["job_count"] == 50
    assert checked["wave_count"] == 7
    assert [len(row["job_ids"]) for row in checked["waves"]] == [
        8,
        8,
        8,
        8,
        8,
        8,
        2,
    ]
    diagnostic = deepcopy(plan["smoke_receipt"])
    diagnostic["production_executor"] = False
    diagnostic["status"] = "diagnostic_only"
    diagnostic["smoke_sha256"] = subject._self_digest(  # type: ignore[attr-defined]
        diagnostic, "smoke_sha256"
    )
    with pytest.raises(PermissionError, match="local smoke"):
        subject.validate_local_smoke(diagnostic, contract=contract)
    tampered = deepcopy(plan)
    tampered["waves"][0]["job_ids"][0] = "abr-pair-0049"
    tampered["cloud_plan_sha256"] = subject._self_digest(  # type: ignore[attr-defined]
        tampered, "cloud_plan_sha256"
    )
    with pytest.raises(ValueError, match="wave grid"):
        subject.validate_cloud_plan(tampered, replay_sources=False)


@pytest.mark.parametrize(
    ("mode", "pair_count"), [("pilot50", 50), ("production250", 250)]
)
def test_abr_builder_dual_pins_exact_real_pair_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    pair_count: int,
) -> None:
    runtime = tmp_path / "runtime.tar.gz"
    wheels = tmp_path / "wheelhouse.zip"
    accepted = tmp_path / "accepted.so"
    diagnostic = tmp_path / "diagnostic.so"
    bundle_manifest = tmp_path / "ABR_RUNTIME_BUNDLE_MANIFEST.json"
    bundle_ready = tmp_path / "ABR_RUNTIME_BUNDLE_READY.json"
    wheelhouse_source_manifest = tmp_path / "wheelhouse-source.json"
    plan_path = tmp_path / "plan.json"
    _tar(runtime, "runtime")
    _zip(wheels, "wheelhouse")
    accepted.write_bytes(b"accepted-real")
    diagnostic.write_bytes(b"diagnostic-real")
    bundle_manifest.write_bytes(subject.canonical_bytes({"bundle": "manifest"}))
    bundle_ready.write_bytes(subject.canonical_bytes({"bundle": "ready"}))
    wheelhouse_source_manifest.write_bytes(
        subject.canonical_bytes({"wheelhouse": "source"})
    )
    resolution = _fake_bundle_resolution(
        runtime=runtime,
        wheels=wheels,
        accepted=accepted,
        diagnostic=diagnostic,
        manifest=bundle_manifest,
        ready=bundle_ready,
        wheelhouse_source_manifest=wheelhouse_source_manifest,
    )
    monkeypatch.setattr(
        subject.runtime_bundle,
        "resolve_bundle",
        lambda *_args, **_kwargs: resolution,
    )
    plan = {
        "pair_count": pair_count,
        "accepted_library_sha256": subject.sha256_file(accepted),
        "diagnostic_library_sha256": subject.sha256_file(diagnostic),
        "synthetic_values_allowed": False,
        "opponent_private_discards_used": False,
    }
    plan_path.write_bytes(subject.canonical_bytes(plan))
    monkeypatch.setattr(
        subject.abr_teacher,
        "validate_plan",
        lambda value: deepcopy(dict(value)),
    )
    contract = subject.build_abr_contract(
        run_name=f"m31-abr-{mode}",
        bucket="m31-post-training-test",
        mode=mode,
        abr_plan_path=plan_path,
        runtime_bundle_manifest_path=bundle_manifest,
        runtime_archive_path=runtime,
        wheelhouse_archive_path=wheels,
        accepted_library_path=accepted,
        diagnostic_library_path=diagnostic,
    )
    assert contract["job_count"] == pair_count
    assert contract["scientific_identity"]["accepted_library_sha256"] == (
        subject.sha256_file(accepted)
    )
    assert contract["scientific_identity"]["diagnostic_library_sha256"] == (
        subject.sha256_file(diagnostic)
    )
    assert contract["scientific_identity"]["synthetic_values_allowed"] is False
    assert contract["jobs"][-1]["pair_index"] == pair_count - 1


def test_locked_builder_pins_closure_threshold_model_abr_and_exact_260(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime.tar.gz"
    wheels = tmp_path / "wheelhouse.zip"
    abr_archive = tmp_path / "abr.tar.gz"
    _tar(runtime, "runtime")
    _zip(wheels, "wheelhouse")
    abr_source = tmp_path / "abr-source" / "abr_bundle"
    abr_source.mkdir(parents=True)
    bundle_raw = subject.canonical_bytes({"bundle": "real"})
    build_receipt_raw = subject.canonical_bytes({"receipt": "real"})
    (abr_source / "bundle.json").write_bytes(bundle_raw)
    (
        abr_source / subject.abr_cli.ABR_PRODUCTION_BUILD_RECEIPT_FILE
    ).write_bytes(build_receipt_raw)
    with tarfile.open(abr_archive, "w:gz") as archive:
        archive.add(abr_source, arcname="abr_bundle")
    closure = tmp_path / "closure.tar.gz"
    threshold = tmp_path / "threshold.json"
    registry = tmp_path / "registry.json"
    promotion_path = tmp_path / "promotion.json"
    execution_path = tmp_path / "execution.json"
    closure.write_bytes(b"closure")
    threshold.write_bytes(subject.canonical_bytes({"threshold": 1}))
    registry.write_bytes(subject.canonical_bytes({"model_sha256": "1" * 64}))
    promotion_path.write_bytes(subject.canonical_bytes({"promotion": True}))
    work_items = [
        {
            "ordinal": index,
            "work_id": f"work-{index:04d}",
            "output_filename": f"shard-{index:04d}.json",
            "seed_index_start": index * 50,
            "seed_index_stop_exclusive": (index + 1) * 50,
        }
        for index in range(260)
    ]
    execution_path.write_bytes(
        subject.canonical_bytes({"work_items": work_items})
    )
    monkeypatch.setattr(
        subject.promotion,
        "validate_locked_promotion_plan",
        lambda value: deepcopy(dict(value)),
    )
    monkeypatch.setattr(
        subject.promotion_execution,
        "validate_execution_plan",
        lambda value, **_kwargs: deepcopy(dict(value)),
    )
    contract = subject.build_locked_promotion_contract(
        run_name="m31-locked-promotion",
        bucket="m31-post-training-test",
        promotion_plan_path=promotion_path,
        execution_plan_path=execution_path,
        runtime_archive_path=runtime,
        wheelhouse_archive_path=wheels,
        closure_package_path=closure,
        compatibility_threshold_lock_path=threshold,
        policy_registry_path=registry,
        abr_bundle_archive_path=abr_archive,
        expected_closure_sha256=subject.sha256_file(closure),
        expected_abr_bundle_archive_sha256=subject.sha256_file(abr_archive),
        expected_abr_bundle_file_sha256=hashlib.sha256(bundle_raw).hexdigest(),
        expected_abr_production_build_receipt_sha256=hashlib.sha256(
            build_receipt_raw
        ).hexdigest(),
    )
    science = contract["scientific_identity"]
    assert contract["job_count"] == 260
    assert sum(row["row_count"] for row in contract["jobs"]) == 13_000
    assert science["closure_package_sha256"] == subject.sha256_file(closure)
    assert science["threshold_lock_file_sha256"] == subject.sha256_file(threshold)
    assert science["policy_registry_file_sha256"] == subject.sha256_file(registry)
    assert science["abr_bundle_archive_sha256"] == subject.sha256_file(
        abr_archive
    )
    assert science["abr_bundle_file_sha256"] == hashlib.sha256(
        bundle_raw
    ).hexdigest()
    assert science["abr_production_build_receipt_file_sha256"] == hashlib.sha256(
        build_receipt_raw
    ).hexdigest()


def test_fake_cloud_full_wave_cleanup_receive_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _contract, plan = _fixture(tmp_path, monkeypatch)
    wave = subject.build_wave(plan)
    cloud = _cloud(plan, tmp_path)
    original_policy = deepcopy(cloud.policy)
    launch = subject.execute_wave(
        cloud_plan=plan,
        wave=wave,
        transport=cloud,
        raw_nonce="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        now_unix_seconds=1_800_000_000,
        observed_at_utc="2027-01-15T08:00:00Z",
        sleep=lambda _seconds: None,
    )
    assert launch["created_instance_count"] == 8
    assert len(cloud.instances) == 8
    assert all(
        row["scheduling"]["maxRunDuration"]["seconds"] == "21600"
        for row in cloud.instances.values()
    )
    _publish(cloud, plan, wave)
    assert subject.poll_wave(
        cloud_plan=plan, wave=wave, transport=cloud
    )["complete_count"] == 8
    lifecycle = subject.cleanup_wave(
        cloud_plan=plan,
        wave=wave,
        launch=launch,
        transport=cloud,
        allow_incomplete=False,
        sleep=lambda _seconds: None,
    )
    assert len(cloud.instances) == 0
    assert len(cloud.disks) == 0
    assert cloud.policy["bindings"] == original_policy["bindings"]
    received = subject.receive_lifecycle(
        cloud_plan=plan,
        lifecycle=lifecycle,
        transport=cloud,
        output_root=tmp_path / "received",
    )
    assert received["received_file_count"] == 8
    retry_or_next = subject.build_wave(plan, accepted_lifecycles=[lifecycle])
    assert retry_or_next["selected"][0]["job_id"] == "abr-pair-0008"
    assert {row["attempt_id"] for row in retry_or_next["selected"]} == {"a00"}


def test_incomplete_cleanup_retries_only_failed_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _contract, plan = _fixture(tmp_path, monkeypatch)
    wave = subject.build_wave(plan)
    cloud = _cloud(plan, tmp_path)
    launch = subject.execute_wave(
        cloud_plan=plan,
        wave=wave,
        transport=cloud,
        raw_nonce="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        now_unix_seconds=1_800_000_000,
        observed_at_utc="2027-01-15T08:00:00Z",
        sleep=lambda _seconds: None,
    )
    lifecycle = subject.cleanup_wave(
        cloud_plan=plan,
        wave=wave,
        launch=launch,
        transport=cloud,
        allow_incomplete=True,
        sleep=lambda _seconds: None,
    )
    assert {row["status"] for row in lifecycle["rows"]} == {"failed"}
    retry = subject.build_wave(plan, accepted_lifecycles=[lifecycle])
    assert [row["job_id"] for row in retry["selected"]] == [
        row["job_id"] for row in wave["selected"]
    ]
    assert {row["attempt_id"] for row in retry["selected"]} == {"a01"}
    assert not cloud.instances
    assert not cloud.disks


def test_missing_launch_receipt_abort_deletes_only_exact_owned_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _contract, plan = _fixture(tmp_path, monkeypatch)
    wave = subject.build_wave(plan)
    cloud = _cloud(plan, tmp_path)
    original_bindings = deepcopy(cloud.policy["bindings"])
    # Simulate a controller crash after the provider completed launch but
    # before the local launch receipt was durably written.
    subject.execute_wave(
        cloud_plan=plan,
        wave=wave,
        transport=cloud,
        raw_nonce="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
        now_unix_seconds=1_800_000_000,
        observed_at_utc="2027-01-15T08:00:00Z",
        sleep=lambda _seconds: None,
    )
    lifecycle = subject.abort_wave(
        cloud_plan=plan,
        wave=wave,
        transport=cloud,
        sleep=lambda _seconds: None,
    )
    assert lifecycle["recovered_after_missing_launch_receipt"] is True
    assert {row["status"] for row in lifecycle["rows"]} == {"failed"}
    assert not cloud.instances
    assert not cloud.disks
    assert cloud.policy["bindings"] == original_bindings
    assert lifecycle["wildcard_delete_used"] is False
    assert lifecycle["unrelated_resource_touched"] is False


def test_controller_requires_exact_sentinel_and_redacted_fresh_oauth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _contract, plan = _fixture(tmp_path, monkeypatch)
    root = tmp_path / "controller"
    controller.prepare_controller(
        cloud_plan=plan,
        output_root=root,
        raw_controller_token=TOKEN,
    )
    monkeypatch.setenv(subject.CONTROLLER_TOKEN_ENV, TOKEN)
    context = controller.plan_next_wave(root)
    assert isinstance(context, controller.Context)
    cloud = _cloud(plan, tmp_path)
    monkeypatch.setenv(subject.PHASE_SENTINEL_ENV, "wrong")
    with pytest.raises(PermissionError, match="exactly equal"):
        controller.execute_next_wave(
            root,
            confirm_run_name=plan["contract"]["run_name"],
            allow_cloud_mutation=True,
            raw_nonce="cccccccc-cccc-4ccc-8ccc-cccccccccccc",
            cloud=cloud,
            now_unix_seconds=1_800_000_000,
            sleep=lambda _seconds: None,
        )
    assert not cloud.objects
    assert not cloud.instances
    monkeypatch.setenv(
        subject.PHASE_SENTINEL_ENV,
        controller.phase_sentinel(root, "execute"),
    )
    launch = controller.execute_next_wave(
        root,
        confirm_run_name=plan["contract"]["run_name"],
        allow_cloud_mutation=True,
        raw_nonce="cccccccc-cccc-4ccc-8ccc-cccccccccccc",
        cloud=cloud,
        now_unix_seconds=1_800_000_000,
        sleep=lambda _seconds: None,
    )
    assert launch["created_instance_count"] == 8
    with pytest.raises(FileExistsError, match="one-shot"):
        controller.execute_next_wave(
            root,
            confirm_run_name=plan["contract"]["run_name"],
            allow_cloud_mutation=True,
            raw_nonce="dddddddd-dddd-4ddd-8ddd-dddddddddddd",
            cloud=cloud,
        )
    receipt = controller.oauth_preflight(
        expires_at_unix=1_800_003_000,
        now_unix=1_800_000_000,
    )
    assert receipt["remaining_seconds"] == 3000
    assert receipt["token_recorded"] is False
    assert set(receipt).isdisjoint({"token", "access_token", "token_sha256"})
    with pytest.raises(PermissionError, match="2700"):
        controller.oauth_preflight(
            expires_at_unix=1_800_002_699,
            now_unix=1_800_000_000,
        )
