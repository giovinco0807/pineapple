from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ofc_regular import hu_m31_t3_dataset_gcp_controller_v1 as controller
from ofc_regular import hu_m31_t3_dataset_supervisor_v1 as subject


def _module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, Any], dict[str, Any], Any]:
    helper = _module(
        f"_m31_dataset_supervisor_helper_{tmp_path.name}",
        Path(__file__).with_name("test_hu_m31_t3_dataset_gcp_controller_v1.py"),
    )
    root, plan, config = helper._fixture(tmp_path, monkeypatch)
    cloud = helper._fake_cloud(config, tmp_path)
    cloud.get_oauth_token_ttl_seconds = lambda: 3600
    return root, plan, config, cloud


def _preparation_module(name: str) -> Any:
    return _module(
        name,
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "prepare_hu_m31_t3_dataset_gcp_v1.py",
    )


def _run_one_failed_wave(
    *,
    root: Path,
    plan: dict[str, Any],
    cloud: Any,
    supervisor_root: Path,
) -> dict[str, Any]:
    return subject.run_next_wave(
        controller_root=root,
        supervisor_root=supervisor_root,
        confirm_run_name=plan["run_name"],
        allow_cloud_mutation=True,
        cloud_factory=lambda: cloud,
        max_poll_attempts=1,
        poll_interval_seconds=0,
        sleep=lambda _seconds: None,
        now=lambda: 1_800_000_000,
    )


def _cost_context(index: int, selected_count: int = 8) -> Any:
    request = hashlib.sha256(f"cost-wave-{index}".encode("ascii")).hexdigest()
    return SimpleNamespace(
        contract={"run_name": "m31-dataset-cost-test"},
        wave_request={
            "wave_index": index,
            "request_sha256": request,
            "selected_count": selected_count,
        },
    )


def _cost_lifecycle(request_sha256: str) -> dict[str, Any]:
    core = {
        "request_sha256": request_sha256,
        "owned_vm_disk_absent": True,
        "worker_iam_removed_before_receive": True,
    }
    return {**core, "receipt_sha256": subject.canonical_sha256(core)}


def test_source_pin_mutation_flag_and_attempt_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, plan, _config, _cloud = _fixture(tmp_path, monkeypatch)
    source = subject.build_source_receipt()
    assert source["all_match"] is True
    assert source["current_profile_changed"] is False
    assert source["cost_contract"] == {
        "total_cloud_cost_cap_micro_usd": 500_000_000,
        "non_compute_reserve_micro_usd": 25_000_000,
        "compute_cost_cap_micro_usd": 475_000_000,
        "spot_rate_guard_micro_usd_per_vm_hour": 500_000,
        "live_spot_price_recheck_required_before_launch": True,
        "reservation_before_cloud_factory": True,
        "settlement_after_exact_cleanup": True,
    }
    assert subject.MAX_SUPERVISED_LIFECYCLES == 90
    ai = next(
        row
        for row in source["files"]
        if row["relative_path"] == "src/ofc_regular/ai_profiles.py"
    )
    assert ai["observed_sha256"] == (
        "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
    )
    called = False

    def cloud_factory() -> Any:
        nonlocal called
        called = True
        raise AssertionError("cloud must remain unopened")

    with pytest.raises(PermissionError, match="execution boundary"):
        subject.run_next_wave(
            controller_root=root,
            supervisor_root=tmp_path / "supervisor",
            confirm_run_name=plan["run_name"],
            allow_cloud_mutation=False,
            cloud_factory=cloud_factory,
        )
    assert called is False


def test_cost_guard_reserves_settles_resumes_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cost-supervisor"
    root.mkdir()
    context = _cost_context(0)
    wave = root / "wave-00-cost"
    reservation = subject._authorize_cost_reservation(
        supervisor_root=root,
        wave_root=wave,
        context=context,
        reserved_at_unix_seconds=1_800_000_000,
    )
    assert reservation["reserved_compute_cost_micro_usd"] == 24_000_000
    assert subject.build_cost_status(root)["committed_compute_cost_micro_usd"] == (
        24_000_000
    )

    lifecycle = _cost_lifecycle(context.wave_request["request_sha256"])
    (wave / "lifecycle-receipt.json").write_bytes(subject.canonical_bytes(lifecycle))
    settlement = subject._settle_cost(
        wave_root=wave,
        reservation=reservation,
        lifecycle=lifecycle,
        settled_at_unix_seconds=1_800_003_600,
    )
    assert settlement["settled_compute_cost_micro_usd"] == 4_000_000
    assert subject.build_cost_status(root)["guarded_total_cost_micro_usd"] == (
        29_000_000
    )
    assert (
        subject._authorize_cost_reservation(
            supervisor_root=root,
            wave_root=wave,
            context=context,
            reserved_at_unix_seconds=1_900_000_000,
        )
        == reservation
    )

    path = wave / "cost-settlement.json"
    tampered = json.loads(path.read_text(encoding="ascii"))
    tampered["settled_compute_cost_micro_usd"] -= 1
    path.write_bytes(subject.canonical_bytes(tampered))
    with pytest.raises(ValueError, match="cost settlement changed"):
        subject.build_cost_status(root)


def test_cost_guard_stops_before_twentieth_worst_case_wave(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cost-cap"
    root.mkdir()
    for index in range(19):
        context = _cost_context(index)
        subject._authorize_cost_reservation(
            supervisor_root=root,
            wave_root=root / f"wave-{index:02d}-cost",
            context=context,
            reserved_at_unix_seconds=1_800_000_000 + index,
        )
    status = subject.build_cost_status(root)
    assert status["committed_compute_cost_micro_usd"] == 456_000_000
    assert status["guarded_total_cost_micro_usd"] == 481_000_000
    with pytest.raises(PermissionError, match=r"\$500 cloud cost cap"):
        subject._authorize_cost_reservation(
            supervisor_root=root,
            wave_root=root / "wave-19-cost",
            context=_cost_context(19),
            reserved_at_unix_seconds=1_800_000_019,
        )
    assert not (root / "wave-19-cost").exists()


def test_cost_confirmation_is_exact() -> None:
    subject._require_cost_confirmation(
        total_cost_cap_usd="500.000000",
        spot_rate_guard_usd_per_vm_hour="0.500000",
    )
    with pytest.raises(PermissionError, match="cost cap"):
        subject._require_cost_confirmation(
            total_cost_cap_usd="500",
            spot_rate_guard_usd_per_vm_hour="0.500000",
        )


def test_cloud_neutral_preparation_builds_v1_config_and_selection(
    tmp_path: Path,
) -> None:
    prepare = _preparation_module(f"_m31_v1_prepare_{tmp_path.name}")
    config = prepare.build_provider_config(
        image_self_link=(
            "https://www.googleapis.com/compute/v1/projects/"
            "debian-cloud/global/images/debian-12-bookworm-v20260701"
        ),
        image_id="123456789",
        guest_os_features=["UEFI_COMPATIBLE", "GVNIC"],
    )
    assert config["schema"] == controller.PROVIDER_CONFIG_SCHEMA
    assert config["worker_service_accounts"] == list(
        prepare.EXPECTED_V1_WORKER_SERVICE_ACCOUNTS
    )
    provider_core = {
        "schema": "hu_m31_t3_step6d_fresh_quality_gcp_provider_plan_v1",
        "status": "provider_plan_ready_cloud_not_mutated",
        "project": "ofc-solver-485418",
        "region": "asia-northeast1",
        "zone": "asia-northeast1-b",
        "image": {
            "self_link": config["image_self_link"],
            "id": config["image_id"],
            "guest_os_features": config["guest_os_features"],
        },
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    provider_plan = {
        **provider_core,
        "provider_plan_sha256": controller.canonical_sha256(provider_core),
    }
    provider_path = tmp_path / "provider-plan.json"
    provider_path.write_bytes(controller.canonical_bytes(provider_plan))
    assert (
        prepare.build_provider_config_from_fresh_quality_plan(provider_path) == config
    )

    launch_core = {
        "schema": "hu_m31_t3_step6d_fresh_quality_gcp_launch_v1",
        "status": "exact_selected_quality_wave_created",
        "create_complete": True,
        "unlisted_vm_created": 0,
        "quota_receipt": {
            "metrics": [
                {
                    "metric": "c4_family_vcpus",
                    "limit_vcpus": 128,
                    "usage_vcpus": 0,
                },
                {
                    "metric": "spot_vcpus",
                    "limit_vcpus": 468,
                    "usage_vcpus": 0,
                },
                {
                    "metric": "global_vcpus",
                    "limit_vcpus": 512,
                    "usage_vcpus": 0,
                },
            ]
        },
        "current_profile_changed": False,
    }
    launch = {
        **launch_core,
        "receipt_sha256": controller.canonical_sha256(launch_core),
    }
    launch_path = tmp_path / "launch.json"
    launch_path.write_bytes(controller.canonical_bytes(launch))
    selected = prepare.build_selection_from_fresh_quality_launch(
        launch_path=launch_path,
        observed_worker_service_accounts=(prepare.EXPECTED_V1_WORKER_SERVICE_ACCOUNTS),
    )
    assert selected["status"] == "v1_8vm_selected_v2_parallel20_no_go"
    assert selected["v1"]["wave_count"] == 45
    assert selected["v2"]["c4_vcpu_deficit"] == 227
    assert selected["service_account_08_19_creation_authorized"] is False


def test_low_ttl_and_lock_stop_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, plan, _config, cloud = _fixture(tmp_path, monkeypatch)
    cloud.get_oauth_token_ttl_seconds = lambda: 2699
    with pytest.raises(PermissionError, match="below 2700"):
        subject.run_next_wave(
            controller_root=root,
            supervisor_root=tmp_path / "low-ttl",
            confirm_run_name=plan["run_name"],
            allow_cloud_mutation=True,
            cloud_factory=lambda: cloud,
            now=lambda: 1_800_000_000,
        )
    assert cloud.instances == {}

    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / subject.LOCK_FILENAME).write_text("occupied", encoding="ascii")
    called = False

    def cloud_factory() -> Any:
        nonlocal called
        called = True
        return cloud

    with pytest.raises(PermissionError, match="another M3.1"):
        subject.run_next_wave(
            controller_root=root,
            supervisor_root=locked,
            confirm_run_name=plan["run_name"],
            allow_cloud_mutation=True,
            cloud_factory=cloud_factory,
        )
    assert called is False


def test_force_refresh_lease_never_persists_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [1_800_000_000]
    calls: list[list[str]] = []
    token = "memory-only-oauth-token-value-123456"

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        assert kwargs == {
            "check": True,
            "capture_output": True,
            "text": True,
        }
        calls.append(command)
        expiry = (
            datetime.fromtimestamp(clock[0] + 3600, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "credential": {
                        "access_token": token,
                        "token_expiry": expiry,
                    }
                }
            )
        )

    monkeypatch.setattr(subject.subprocess, "run", fake_run)
    lease = subject._GcloudCredentialLease(now=lambda: clock[0])
    first = lease.get(force_refresh=True)
    assert first.get_oauth_token_ttl_seconds() == 3600
    assert calls[0] == [
        "gcloud",
        "config",
        "config-helper",
        "--force-auth-refresh",
        "--format=json",
    ]
    assert lease.get(force_refresh=False) is first
    clock[0] += 2800
    second = lease.get(force_refresh=False)
    assert second is not first
    assert len(calls) == 2

    root, _plan, _config, _cloud = _fixture(tmp_path, monkeypatch)
    context = controller.plan_next_wave(root)
    assert isinstance(context, controller.WaveContext)
    receipt = subject.build_oauth_preflight(
        context=context,
        cloud=second,
        observed_at_utc="2027-01-15T08:46:40Z",
    )
    assert token.encode("ascii") not in subject.canonical_bytes(receipt)
    assert receipt["access_token_persisted"] is False


def test_failed_wave_is_cleaned_accepted_and_checkpointed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, plan, _config, cloud = _fixture(tmp_path, monkeypatch)
    original_bindings = deepcopy(cloud.policy["bindings"])
    supervisor_root = tmp_path / "supervisor"
    result = _run_one_failed_wave(
        root=root,
        plan=plan,
        cloud=cloud,
        supervisor_root=supervisor_root,
    )
    assert result["schema"] == subject.SUPERVISOR_CHECKPOINT_SCHEMA
    assert result["cleanup_incomplete"] is True
    assert result["next_status"] == "wave_ready"
    assert cloud.instances == {}
    assert cloud.disks == {}
    assert cloud.policy["bindings"] == original_bindings
    assert len(list(supervisor_root.rglob("checkpoint-intent.json"))) == 1
    assert len(list(supervisor_root.rglob("SUPERVISOR_CHECKPOINT.json"))) == 1
    assert controller.controller_status(root)["accepted_lifecycle_count"] == 1


def test_existing_launch_replays_without_duplicate_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, plan, _config, cloud = _fixture(tmp_path, monkeypatch)
    context = controller.plan_next_wave(root)
    assert isinstance(context, controller.WaveContext)
    create = cloud.create_instance
    count = 0

    def counted(*, instance_spec: dict, request_id: str) -> dict:
        nonlocal count
        count += 1
        return create(instance_spec=instance_spec, request_id=request_id)

    cloud.create_instance = counted
    monkeypatch.setenv(
        controller.PHASE_SENTINEL_ENV,
        controller.expected_phase_sentinel(context, "execute"),
    )
    controller.execute_next_wave(
        root,
        confirm_run_name=plan["run_name"],
        allow_cloud_mutation=True,
        cloud=cloud,
        now_unix_seconds=1_800_000_000,
        sleep=lambda _seconds: None,
    )
    assert count == 8
    _run_one_failed_wave(
        root=root,
        plan=plan,
        cloud=cloud,
        supervisor_root=tmp_path / "supervisor",
    )
    assert count == 8
    assert cloud.instances == {}


def test_run_all_retries_then_stops_on_exhaustion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, plan, _config, cloud = _fixture(tmp_path, monkeypatch)
    result = subject.run_until_terminal(
        controller_root=root,
        supervisor_root=tmp_path / "supervisor",
        confirm_run_name=plan["run_name"],
        allow_cloud_mutation=True,
        cloud_factory=lambda: cloud,
        max_poll_attempts=1,
        poll_interval_seconds=0,
        sleep=lambda _seconds: None,
    )
    assert result["status"] == "no_go_attempts_exhausted"
    assert result["checkpoint_count"] == 2
    assert result["controller_status"]["exhausted_shard_count"] == 8
    assert cloud.instances == {}


def test_pre_accept_crash_recovers_intent_without_cloud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, plan, _config, cloud = _fixture(tmp_path, monkeypatch)
    accept = controller.accept_next_wave
    monkeypatch.setattr(
        controller,
        "accept_next_wave",
        lambda _root: (_ for _ in ()).throw(RuntimeError("injected pre-accept crash")),
    )
    supervisor_root = tmp_path / "supervisor"
    with pytest.raises(RuntimeError, match="pre-accept"):
        _run_one_failed_wave(
            root=root,
            plan=plan,
            cloud=cloud,
            supervisor_root=supervisor_root,
        )
    assert controller.controller_status(root)["accepted_lifecycle_count"] == 0
    assert len(list(supervisor_root.rglob("checkpoint-intent.json"))) == 1

    monkeypatch.setattr(controller, "accept_next_wave", accept)
    called = False

    def no_cloud() -> Any:
        nonlocal called
        called = True
        raise AssertionError("intent recovery reopened cloud")

    launch_path = next(supervisor_root.rglob("launch-receipt.json"))
    launch_raw = launch_path.read_bytes()
    launch_path.write_bytes(launch_raw + b" ")
    with pytest.raises(ValueError, match="canonical JSON"):
        subject.run_next_wave(
            controller_root=root,
            supervisor_root=supervisor_root,
            confirm_run_name=plan["run_name"],
            allow_cloud_mutation=True,
            cloud_factory=no_cloud,
        )
    assert called is False
    launch_path.write_bytes(launch_raw)

    recovered = subject.run_next_wave(
        controller_root=root,
        supervisor_root=supervisor_root,
        confirm_run_name=plan["run_name"],
        allow_cloud_mutation=True,
        cloud_factory=no_cloud,
    )
    assert recovered["wave_index"] == 0
    assert called is False
    assert controller.controller_status(root)["accepted_lifecycle_count"] == 1


def test_post_ledger_crash_backfills_checkpoint_without_next_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, plan, _config, cloud = _fixture(tmp_path, monkeypatch)
    write_once = subject._write_once
    injected = False

    def crash_checkpoint(path: Path, value: Any) -> None:
        nonlocal injected
        if path.name == "SUPERVISOR_CHECKPOINT.json" and not injected:
            injected = True
            raise RuntimeError("injected post-ledger crash")
        write_once(path, value)

    monkeypatch.setattr(subject, "_write_once", crash_checkpoint)
    supervisor_root = tmp_path / "supervisor"
    with pytest.raises(RuntimeError, match="post-ledger"):
        _run_one_failed_wave(
            root=root,
            plan=plan,
            cloud=cloud,
            supervisor_root=supervisor_root,
        )
    assert controller.controller_status(root)["accepted_lifecycle_count"] == 1
    assert not list(supervisor_root.rglob("SUPERVISOR_CHECKPOINT.json"))

    monkeypatch.setattr(subject, "_write_once", write_once)
    called = False

    def no_cloud() -> Any:
        nonlocal called
        called = True
        raise AssertionError("checkpoint recovery launched next wave")

    recovered = subject.run_next_wave(
        controller_root=root,
        supervisor_root=supervisor_root,
        confirm_run_name=plan["run_name"],
        allow_cloud_mutation=True,
        cloud_factory=no_cloud,
    )
    assert recovered["wave_index"] == 0
    assert recovered["next_status"] == "wave_ready"
    assert called is False
    assert controller.controller_status(root)["accepted_lifecycle_count"] == 1


def test_partial_launch_exact_cleanup_and_abort_blocks_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, plan, _config, cloud = _fixture(tmp_path, monkeypatch)
    original_bindings = deepcopy(cloud.policy["bindings"])
    create = cloud.create_instance
    count = 0

    def fail_after_three(*, instance_spec: dict, request_id: str) -> dict:
        nonlocal count
        if count == 3:
            raise RuntimeError("injected partial launch")
        count += 1
        return create(instance_spec=instance_spec, request_id=request_id)

    cloud.create_instance = fail_after_three
    supervisor_root = tmp_path / "supervisor"
    with pytest.raises(RuntimeError, match="partial launch"):
        subject.run_next_wave(
            controller_root=root,
            supervisor_root=supervisor_root,
            confirm_run_name=plan["run_name"],
            allow_cloud_mutation=True,
            cloud_factory=lambda: cloud,
            sleep=lambda _seconds: None,
            now=lambda: 1_800_000_000,
        )
    assert count == 3
    assert cloud.instances == {}
    assert cloud.disks == {}
    assert cloud.policy["bindings"] == original_bindings
    abort = subject._read(next(supervisor_root.rglob("abort-cleanup.json")), "abort")
    assert abort["owned_vm_disk_absent"] is True
    assert abort["worker_iam_absent"] is True
    with pytest.raises(PermissionError, match="explicit new controller"):
        subject.run_next_wave(
            controller_root=root,
            supervisor_root=supervisor_root,
            confirm_run_name=plan["run_name"],
            allow_cloud_mutation=True,
            cloud_factory=lambda: cloud,
        )


def test_unwritten_iam_is_reconciled_and_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, plan, _config, cloud = _fixture(tmp_path, monkeypatch)
    original_bindings = deepcopy(cloud.policy["bindings"])
    write_once = controller._write_once

    def fail_iam_receipt(path: Path, value: Any) -> None:
        if path.name == "iam_receipt.json":
            raise OSError("injected crash before IAM receipt")
        write_once(path, value)

    monkeypatch.setattr(controller, "_write_once", fail_iam_receipt)
    supervisor_root = tmp_path / "supervisor"
    with pytest.raises(OSError, match="before IAM receipt"):
        subject.run_next_wave(
            controller_root=root,
            supervisor_root=supervisor_root,
            confirm_run_name=plan["run_name"],
            allow_cloud_mutation=True,
            cloud_factory=lambda: cloud,
            sleep=lambda _seconds: None,
            now=lambda: 1_800_000_000,
        )
    assert cloud.instances == {}
    assert cloud.disks == {}
    assert cloud.policy["bindings"] == original_bindings
    abort = subject._read(next(supervisor_root.rglob("abort-cleanup.json")), "abort")
    assert len(abort["iam_cleanup_receipt_sha256"]) == 64


def test_orphan_owned_disk_is_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, plan, _config, cloud = _fixture(tmp_path, monkeypatch)
    create = cloud.create_instance
    first = True

    def orphan(*, instance_spec: dict, request_id: str) -> dict:
        nonlocal first
        if not first:
            raise RuntimeError("stop after orphan")
        first = False
        create(instance_spec=instance_spec, request_id=request_id)
        cloud.instances.pop(instance_spec["name"])
        raise RuntimeError("orphaned exact boot disk")

    cloud.create_instance = orphan
    with pytest.raises(RuntimeError, match="orphaned exact boot disk"):
        subject.run_next_wave(
            controller_root=root,
            supervisor_root=tmp_path / "supervisor",
            confirm_run_name=plan["run_name"],
            allow_cloud_mutation=True,
            cloud_factory=lambda: cloud,
            sleep=lambda _seconds: None,
            now=lambda: 1_800_000_000,
        )
    assert cloud.instances == {}
    assert cloud.disks == {}
