from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import hu_m31_t3_dataset_gcp_controller_v1 as subject
from ofc_regular import hu_m31_t3_dataset_gcp_provider_v1 as provider
from ofc_regular import hu_m31_t3_dataset_gcp_transport_v1 as transport


TOKEN = "12345678-1234-4234-9234-123456789abc"
IMAGE = (
    "https://www.googleapis.com/compute/v1/projects/"
    "debian-cloud/global/images/debian-12-bookworm-v20260701"
)


def _module(name: str, filename: str) -> Any:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load test helper: {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    helper = _module(
        f"_m31_dataset_provider_helper_{tmp_path.name}",
        "test_hu_m31_t3_dataset_gcp_transport_provider_v1.py",
    )
    sources = tmp_path / "sources"
    sources.mkdir()
    plan = helper._transport_plan(sources, monkeypatch)
    config = {
        "schema": subject.PROVIDER_CONFIG_SCHEMA,
        "image_self_link": IMAGE,
        "image_id": "123456789",
        "guest_os_features": ["GVNIC", "UEFI_COMPATIBLE"],
        "worker_service_accounts": [
            f"m31ds{index}@{provider.PROJECT}.iam.gserviceaccount.com"
            for index in range(8)
        ],
        "current_profile_changed": False,
    }
    root = tmp_path / "controller"
    subject.initialize_controller_from_transport_plan(
        transport_plan=plan,
        provider_config=config,
        local_shard_root=tmp_path / "local-shards",
        output_root=root,
        raw_controller_token=TOKEN,
    )
    monkeypatch.setenv(subject.CONTROLLER_TOKEN_ENV, TOKEN)
    return root, plan, config


class _NoCloud:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"unexpected cloud call: {name}")


def _fake_cloud(config: dict[str, Any], tmp_path: Path) -> Any:
    helper = _module(
        f"_m31_quality_cloud_helper_{tmp_path.name}",
        "test_hu_m31_t3_step6d_fresh_quality_gcp_bridge_v1.py",
    )
    return helper._FakeCloud(
        image_link=config["image_self_link"],
        image_id=config["image_id"],
        features=config["guest_os_features"],
    )


def test_prepare_and_plan_are_cloud_neutral_gate_bound_and_max8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, plan, config = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        subject,
        "_live_transport",
        lambda: (_ for _ in ()).throw(AssertionError("cloud called")),
    )

    # The production prepare path is local-only.  Any accidental adapter
    # construction makes this test fail before a controller can be persisted.
    by_kind = {row["kind"]: row for row in plan["content_sources"]}
    config_path = tmp_path / "provider-config.json"
    config_path.write_bytes(subject.canonical_bytes(config))
    prepared_root = tmp_path / "prepared-controller"
    prepared = subject.prepare_controller(
        run_name=plan["run_name"],
        bucket=plan["bucket"],
        dataset_plan_path=by_kind["dataset_plan"]["source_path"],
        fresh_quality_gate_path=by_kind["fresh_quality_gate"]["source_path"],
        smoke_gate_path=by_kind["smoke_gate"]["source_path"],
        smoke_shard_directory=plan["source_paths"][
            "smoke_shard_directory"
        ],
        smoke_shard_archive_path=by_kind["smoke_shard_archive"][
            "source_path"
        ],
        runtime_archive_path=by_kind["runtime_archive"]["source_path"],
        wheelhouse_archive_path=by_kind["wheelhouse_archive"]["source_path"],
        candidate_library_path=by_kind["candidate_library"]["source_path"],
        provider_config_path=config_path,
        local_shard_root=tmp_path / "prepared-local-shards",
        output_root=prepared_root,
        raw_controller_token=TOKEN,
    )
    assert prepared["full_fanout_authorized_after_smoke_only"] is True
    assert prepared["current_profile_changed"] is False

    context = subject.plan_next_wave(root)
    assert isinstance(context, subject.WaveContext)
    assert context.wave_request["wave_index"] == 0
    assert context.wave_request["selected_count"] == 8
    assert context.provider_plan["runtime_contract"]["machine_type"] == (
        "c4-standard-16"
    )
    assert context.provider_plan["runtime_contract"]["max_vm_count"] == 8
    assert plan["wave_count"] == 45
    assert all(row["max_vm_count"] <= 8 for row in plan["waves"])
    assert subject.controller_status(root)["accepted_lifecycle_count"] == 0

    # Every later local or cloud phase replays the exact gate bytes.  Changing
    # the qualified fresh-quality source closes the controller before a cloud
    # adapter can be constructed.
    fresh = next(
        Path(row["source_path"])
        for row in plan["content_sources"]
        if row["kind"] == "fresh_quality_gate"
    )
    fresh.write_bytes(subject.canonical_bytes({"tampered": True}))
    with pytest.raises(ValueError, match="content source changed"):
        subject.controller_status(root)


def test_exact_sentinels_fake_cloud_full_attempt_and_automatic_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, plan, config = _fixture(tmp_path, monkeypatch)
    context = subject.plan_next_wave(root)
    assert isinstance(context, subject.WaveContext)
    cloud = _fake_cloud(config, tmp_path)
    original_policy = deepcopy(cloud.policy)

    monkeypatch.setenv(subject.PHASE_SENTINEL_ENV, "wrong")
    with pytest.raises(PermissionError, match="exactly equal"):
        subject.execute_next_wave(
            root,
            confirm_run_name=plan["run_name"],
            allow_cloud_mutation=True,
            cloud=cloud,
            now_unix_seconds=1_800_000_000,
            sleep=lambda _seconds: None,
        )
    assert cloud.objects == {}
    assert cloud.instances == {}
    assert cloud.policy == original_policy

    monkeypatch.setenv(
        subject.PHASE_SENTINEL_ENV,
        subject.expected_phase_sentinel(context, "execute"),
    )
    launch = subject.execute_next_wave(
        root,
        confirm_run_name=plan["run_name"],
        allow_cloud_mutation=True,
        cloud=cloud,
        now_unix_seconds=1_800_000_000,
        sleep=lambda _seconds: None,
    )
    assert launch["created_instance_count"] == 8
    assert launch["at_most_eight_c4"] is True
    assert len(cloud.instances) == 8
    assert all(
        row["scheduling"]["maxRunDuration"]["seconds"] == "21600"
        for row in cloud.instances.values()
    )

    # A completed create-only launch receipt resumes without any cloud call.
    assert (
        subject.execute_next_wave(
            root,
            confirm_run_name=plan["run_name"],
            allow_cloud_mutation=True,
            cloud=_NoCloud(),  # type: ignore[arg-type]
        )
        == launch
    )

    monkeypatch.setenv(
        subject.PHASE_SENTINEL_ENV,
        subject.expected_phase_sentinel(context, "poll"),
    )
    poll = subject.poll_next_wave(
        root,
        confirm_run_name=plan["run_name"],
        allow_cloud_read=True,
        cloud=cloud,
    )
    assert poll["complete_count"] == 0
    assert all(row["instance_present"] for row in poll["rows"])

    monkeypatch.setenv(
        subject.PHASE_SENTINEL_ENV,
        subject.expected_phase_sentinel(context, "cleanup"),
    )
    with pytest.raises(PermissionError, match="incomplete running"):
        subject.cleanup_next_wave(
            root,
            confirm_run_name=plan["run_name"],
            allow_cloud_mutation=True,
            cloud=cloud,
            sleep=lambda _seconds: None,
        )
    assert len(cloud.instances) == 8

    monkeypatch.setenv(
        subject.PHASE_SENTINEL_ENV,
        subject.expected_phase_sentinel(context, "cleanup-incomplete"),
    )
    lifecycle = subject.cleanup_next_wave(
        root,
        confirm_run_name=plan["run_name"],
        allow_cloud_mutation=True,
        allow_incomplete_cleanup=True,
        cloud=cloud,
        sleep=lambda _seconds: None,
    )
    assert len(cloud.instances) == 0
    assert all(row["status"] == "failed" for row in lifecycle["attempt_rows"])
    assert lifecycle["worker_iam_removed_before_receive"] is True

    monkeypatch.setenv(
        subject.PHASE_SENTINEL_ENV,
        subject.expected_phase_sentinel(context, "receive"),
    )
    received = subject.receive_next_wave(
        root,
        confirm_run_name=plan["run_name"],
        allow_cloud_read=True,
        cloud=cloud,
    )
    assert received["failed_count"] == 8
    accepted = subject.accept_next_wave(root)
    assert accepted["next_status"] == "wave_ready"
    assert accepted["next_wave_index"] == 0
    assert accepted["next_selected_count"] == 8

    retry = subject.plan_next_wave(root)
    assert isinstance(retry, subject.WaveContext)
    assert {
        row["attempt_id"] for row in retry.wave_request["selected_attempts"]
    } == {"a01"}
    status = subject.controller_status(root)
    assert status["accepted_lifecycle_count"] == 1
    assert status["resume_wave_index"] == 0
    assert status["max_concurrent_vms"] == 8
    assert len(subject._load_events(root)) == 8  # type: ignore[attr-defined]
    with pytest.raises(PermissionError, match="all cloud shards"):
        subject.finalize_dataset(root)


def test_receive_is_required_before_accepted_ledger_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, plan, config = _fixture(tmp_path, monkeypatch)
    context = subject.plan_next_wave(root)
    assert isinstance(context, subject.WaveContext)
    cloud = _fake_cloud(config, tmp_path)
    monkeypatch.setenv(
        subject.PHASE_SENTINEL_ENV,
        subject.expected_phase_sentinel(context, "execute"),
    )
    subject.execute_next_wave(
        root,
        confirm_run_name=plan["run_name"],
        allow_cloud_mutation=True,
        cloud=cloud,
        now_unix_seconds=1_800_000_000,
        sleep=lambda _seconds: None,
    )
    monkeypatch.setenv(
        subject.PHASE_SENTINEL_ENV,
        subject.expected_phase_sentinel(context, "poll"),
    )
    subject.poll_next_wave(
        root,
        confirm_run_name=plan["run_name"],
        allow_cloud_read=True,
        cloud=cloud,
    )
    monkeypatch.setenv(
        subject.PHASE_SENTINEL_ENV,
        subject.expected_phase_sentinel(context, "cleanup-incomplete"),
    )
    subject.cleanup_next_wave(
        root,
        confirm_run_name=plan["run_name"],
        allow_cloud_mutation=True,
        allow_incomplete_cleanup=True,
        cloud=cloud,
        sleep=lambda _seconds: None,
    )
    with pytest.raises(PermissionError, match="cleanup plus receive"):
        subject.accept_next_wave(root)
    assert subject.controller_status(root)["accepted_lifecycle_count"] == 0


def test_worker_iam_crash_windows_reconcile_exact_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _plan, config = _fixture(tmp_path, monkeypatch)
    context = subject.plan_next_wave(root)
    assert isinstance(context, subject.WaveContext)
    cloud = _fake_cloud(config, tmp_path)
    provider.stage_content(
        provider_plan=context.provider_plan,
        transport_plan=context.transport_plan,
        ledger=context.ledger,
        resume=context.resume,
        wave_request=context.wave_request,
        transport=cloud,
    )
    first = provider.install_worker_iam(
        provider_plan=context.provider_plan,
        transport_plan=context.transport_plan,
        ledger=context.ledger,
        resume=context.resume,
        wave_request=context.wave_request,
        transport=cloud,
        now_unix_seconds=1_800_000_000,
    )
    assert first["cloud_mutated"] is True
    assert first["response_reconciled"] is False

    exact_policy = deepcopy(cloud.policy)
    cloud.policy["bindings"][-1]["condition"]["expression"] += " "
    with pytest.raises(PermissionError, match="binding changed"):
        provider.install_worker_iam(
            provider_plan=context.provider_plan,
            transport_plan=context.transport_plan,
            ledger=context.ledger,
            resume=context.resume,
            wave_request=context.wave_request,
            transport=cloud,
            now_unix_seconds=1_800_000_000,
        )
    cloud.policy = exact_policy

    # Crash after policy mutation but before the create-only receipt.
    replayed = provider.install_worker_iam(
        provider_plan=context.provider_plan,
        transport_plan=context.transport_plan,
        ledger=context.ledger,
        resume=context.resume,
        wave_request=context.wave_request,
        transport=cloud,
        now_unix_seconds=1_800_000_000,
    )
    assert replayed["expires_at_utc"] == first["expires_at_utc"]
    assert replayed["cloud_mutated"] is False
    assert replayed["response_reconciled"] is True
    provider.validate_iam_receipt(
        replayed, provider_plan=context.provider_plan
    )

    removed = provider.remove_worker_iam(
        provider_plan=context.provider_plan,
        transport_plan=context.transport_plan,
        ledger=context.ledger,
        resume=context.resume,
        wave_request=context.wave_request,
        iam_receipt=replayed,
        transport=cloud,
    )
    assert removed["cloud_mutated"] is True
    assert removed["response_reconciled"] is False

    # Symmetric crash after removal but before its create-only receipt.
    removed_replay = provider.remove_worker_iam(
        provider_plan=context.provider_plan,
        transport_plan=context.transport_plan,
        ledger=context.ledger,
        resume=context.resume,
        wave_request=context.wave_request,
        iam_receipt=replayed,
        transport=cloud,
    )
    assert removed_replay["cloud_mutated"] is False
    assert removed_replay["response_reconciled"] is True
    provider.validate_iam_cleanup_receipt(
        removed_replay,
        provider_plan=context.provider_plan,
        iam_receipt=replayed,
    )


def test_dataset_owned_instance_uses_dataset_duration_not_quality_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _plan, _config = _fixture(tmp_path, monkeypatch)
    context = subject.plan_next_wave(root)
    assert isinstance(context, subject.WaveContext)
    selected = context.wave_request["selected_attempts"][0]
    worker = context.provider_plan["workers"][0]
    stage_core = {
        "schema": provider.STAGE_RECEIPT_SCHEMA,
        "status": "all_content_hash_replayed",
        "provider_plan_sha256": context.provider_plan[
            "provider_plan_sha256"
        ],
        "content_records": [
            {
                "kind": entry["kind"],
                "object_name": entry["object_name"],
                "generation": str(index + 1),
                "sha256": entry["sha256"],
                "bytes": entry["bytes"],
                "created": True,
            }
            for index, entry in enumerate(
                context.provider_plan["content_entries"]
            )
        ],
        "record_count": len(context.provider_plan["content_entries"]),
        "all_bytes_replayed": True,
        "create_only_or_identical_reuse": True,
        "cloud_mutated": True,
        "current_profile_changed": False,
    }
    stage = {
        **stage_core,
        "receipt_sha256": provider.canonical_sha256(stage_core),
    }
    spec = provider._instance_spec(  # type: ignore[attr-defined]
        provider=context.provider_plan,
        worker=worker,
        selected=selected,
        stage_receipt=stage,
    )
    assert spec["scheduling"]["maxRunDuration"]["seconds"] == "21600"
