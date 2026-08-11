from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import hu_m31_t3_dataset_contract_v1 as dataset
from ofc_regular import hu_m31_t3_dataset_executor_v1 as executor_v1
from ofc_regular import hu_m31_t3_dataset_gcp_provider_v2 as provider
from ofc_regular import hu_m31_t3_dataset_gcp_controller_v2 as controller
from ofc_regular import hu_m31_t3_dataset_gcp_transport_v1 as transport_v1
from ofc_regular import hu_m31_t3_dataset_gcp_transport_v2 as transport
from ofc_regular import hu_m31_t3_dataset_portable_worker_v1 as worker_v1
from ofc_regular import hu_m31_t3_dataset_worker_sa_plan_v2 as sa_plan
from ofc_regular import hu_m31_t3_dataset_transport_selection_v1 as selection
from ofc_regular import (
    hu_m31_t3_step6d_fresh_quality_gcp_provider_v1 as quality_provider,
)


IMAGE = (
    "https://www.googleapis.com/compute/v1/projects/"
    "debian-cloud/global/images/debian-12-bookworm-v20260701"
)


def _module(name: str, filename: str) -> Any:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], dict[str, Any]]:
    helper = _module(
        f"_m31_ds_v1_helper_{tmp_path.name}",
        "test_hu_m31_t3_dataset_gcp_transport_provider_v1.py",
    )
    v1_root = tmp_path / "v1"
    v1_root.mkdir()
    original = helper._transport_plan(v1_root, monkeypatch)
    by_kind = {row["kind"]: row for row in original["content_sources"]}
    monkeypatch.setattr(transport, "_filesystem_type", lambda _path: "ext2/ext3")
    relocation = transport.relocate_sources_to_ext4(
        source_files={
            kind: by_kind[kind]["source_path"] for kind in transport_v1.CONTENT_KINDS
        },
        smoke_shard_directory=original["source_paths"]["smoke_shard_directory"],
        destination_root=tmp_path / "ext4",
    )
    relocated = {row["kind"]: row["destination_path"] for row in relocation["files"]}
    plan = transport.build_transport_plan(
        source_relocation_receipt=relocation,
        run_name=original["run_name"],
        bucket=original["bucket"],
        dataset_plan_path=relocated["dataset_plan"],
        fresh_quality_gate_path=relocated["fresh_quality_gate"],
        smoke_gate_path=relocated["smoke_gate"],
        portable_authorization_path=relocated["portable_authorization"],
        smoke_shard_directory=relocation["smoke_shard_directory"]["destination_path"],
        smoke_shard_archive_path=relocated["smoke_shard_archive"],
        runtime_archive_path=relocated["runtime_archive"],
        wheelhouse_archive_path=relocated["wheelhouse_archive"],
        candidate_library_path=relocated["candidate_library"],
    )
    return plan, relocation


def test_parallel20_changes_only_fanout_and_reuses_v1_byte_producers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, relocation = _fixture(tmp_path, monkeypatch)
    assert transport.validate_transport_plan(plan, replay_sources=True) == plan
    assert plan["wave_count"] == 18
    assert [row["shard_count"] for row in plan["waves"]] == [20] * 17 + [19]
    assert plan["machine_contract"]["max_concurrent_vms"] == 20
    assert plan["machine_contract"]["max_concurrent_vcpus"] == 320
    assert len(plan["jobs"]) == 359
    assert all(row["paired_hand_count"] == 25 for row in plan["jobs"])
    assert all(row["wave_index"] == row["ordinal"] // 20 for row in plan["jobs"])
    assert transport.run_portable_dataset_shard is worker_v1.run_portable_dataset_shard
    assert transport.build_merge_manifest is dataset.build_merge_manifest
    lock = plan["scientific_v1_lock"]
    assert lock["dataset_plan_sha256"] == dataset.EXPECTED_DATASET_PLAN_SHA256
    assert lock["same_shard_bytes_as_v1"] is True
    assert lock["same_merge_bytes_as_v1"] is True
    assert lock["dataset_executor_module_sha256"] == transport.sha256_file(
        executor_v1.__file__
    )
    assert plan["source_ext4_relocation_receipt_sha256"] == relocation["receipt_sha256"]

    v1_plan = transport_v1.build_transport_plan(
        **transport._v1_build_kwargs(plan)  # type: ignore[attr-defined]
    )
    v2_normalized = deepcopy(plan["jobs"])
    for row in v2_normalized:
        row["wave_index"] = row["ordinal"] // 8
    assert v2_normalized == v1_plan["jobs"]


def test_parallel20_retry_preemption_and_tamper_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _relocation = _fixture(tmp_path, monkeypatch)
    ledger = transport.build_attempt_ledger(plan)
    resume = transport.build_resume_plan(plan, ledger)
    request = transport.build_wave_request(plan, ledger, resume)
    assert request["selected_count"] == 20
    assert request["required_vcpus"] == 320
    assert {row["attempt_id"] for row in request["selected_attempts"]} == {"a00"}
    rows = [
        {
            "shard_id": row["shard_id"],
            "attempt_id": row["attempt_id"],
            "instance_id": row["instance_id"],
            "status": "failed",
            "completed_pair_count": 0,
            "checkpoint": None,
            "heartbeat": None,
            "owned_compute_absent": True,
            "worker_iam_removed": True,
        }
        for row in request["selected_attempts"]
    ]
    lifecycle = transport.build_lifecycle_receipt(
        plan=plan,
        ledger=ledger,
        resume=resume,
        wave_request=request,
        attempt_rows=rows,
    )
    after = transport.build_attempt_ledger(plan, lifecycle_receipts=[lifecycle])
    retry = transport.build_resume_plan(plan, after)
    assert retry["selected_count"] == 20
    assert {row["attempt_id"] for row in retry["selected_attempts"]} == {"a01"}

    changed = deepcopy(plan)
    changed["waves"][0]["shard_count"] = 19
    core = dict(changed)
    core.pop("plan_sha256")
    changed["plan_sha256"] = transport.canonical_sha256(core)
    with pytest.raises(ValueError, match="cardinality"):
        transport.validate_transport_plan(changed, replay_sources=False)

    lifecycle_changed = deepcopy(lifecycle)
    lifecycle_changed["partial_launch_cleanup_complete"] = False
    core = dict(lifecycle_changed)
    core.pop("receipt_sha256")
    lifecycle_changed["receipt_sha256"] = transport.canonical_sha256(core)
    with pytest.raises(ValueError, match="boundary"):
        transport.validate_lifecycle_receipt(lifecycle_changed, plan=plan)


def test_ext4_relocation_is_create_only_and_detects_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan, receipt = _fixture(tmp_path, monkeypatch)
    assert transport.validate_source_relocation_receipt(receipt) == receipt
    first = Path(receipt["files"][0]["destination_path"])
    first.write_bytes(first.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="relocated file"):
        transport.validate_source_relocation_receipt(receipt)


def test_service_account_plan_exposes_exact_missing_08_through_19() -> None:
    plan = sa_plan.build_provisioning_plan()
    assert sa_plan.validate_provisioning_plan(plan) == plan
    assert plan["observed_existing_service_accounts"] == list(
        provider.EXPECTED_WORKER_SERVICE_ACCOUNTS[:8]
    )
    assert plan["missing_service_accounts"] == list(
        provider.EXPECTED_WORKER_SERVICE_ACCOUNTS[8:]
    )
    assert plan["missing_count"] == 12
    assert plan["service_account_creation_authorized"] is False
    assert plan["cloud_mutated"] is False
    receipt = sa_plan.build_inventory_receipt(
        provisioning_plan=plan,
        observed_existing_service_accounts=plan["observed_existing_service_accounts"],
    )
    assert receipt["worker_pool_ready"] is False
    assert receipt["status"] == "parallel20_worker_pool_incomplete_no_go"


def test_live_c4_128_selects_v1_and_forbids_08_19_provisioning() -> None:
    receipt = selection.build_selection_receipt(
        c4_limit_vcpus=128,
        c4_usage_vcpus=0,
        spot_limit_vcpus=468,
        spot_usage_vcpus=0,
        global_limit_vcpus=512,
        global_usage_vcpus=0,
        observed_worker_service_account_count=8,
        evidence_source="attempt003_live_quota_receipt",
        evidence_sha256="a" * 64,
        source_hashes={"ai_profiles": "b" * 64},
    )
    assert selection.validate_selection_receipt(receipt) == receipt
    assert receipt["v1"]["selected_for_m31_production"] is True
    assert receipt["v1"]["required_vcpus"] == 128
    assert receipt["v2"]["quota_gate_pass"] is False
    assert receipt["v2"]["c4_vcpu_deficit"] == 227
    assert receipt["service_account_08_19_creation_authorized"] is False


def test_local_wave0_rehearsal_aligns_v1_first_eight_and_keeps_v2_dormant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    v2_plan, _relocation = _fixture(tmp_path, monkeypatch)
    v1_plan = transport_v1.build_transport_plan(
        **transport._v1_build_kwargs(v2_plan)  # type: ignore[attr-defined]
    )
    v1_ledger = transport_v1.build_attempt_ledger(v1_plan)
    v1_resume = transport_v1.build_resume_plan(v1_plan, v1_ledger)
    v1_request = transport_v1.build_wave_request(v1_plan, v1_ledger, v1_resume)
    v2_ledger = transport.build_attempt_ledger(v2_plan)
    v2_resume = transport.build_resume_plan(v2_plan, v2_ledger)
    v2_request = transport.build_wave_request(v2_plan, v2_ledger, v2_resume)
    selected = selection.build_selection_receipt(
        c4_limit_vcpus=128,
        c4_usage_vcpus=0,
        spot_limit_vcpus=468,
        spot_usage_vcpus=0,
        global_limit_vcpus=512,
        global_usage_vcpus=0,
        observed_worker_service_account_count=8,
        evidence_source="attempt003_live_quota_receipt",
        evidence_sha256="a" * 64,
        source_hashes={"ai_profiles": "b" * 64},
    )
    rehearsal = selection.build_wave0_rehearsal_receipt(
        selection_receipt=selected,
        v1_wave_request=v1_request,
        v2_wave_request=v2_request,
        fixture_kind="synthetic_local",
    )
    assert rehearsal["v1_selected_count"] == 8
    assert rehearsal["v1_required_vcpus"] == 128
    assert rehearsal["v2_selected_count"] == 20
    assert rehearsal["v2_required_vcpus"] == 320
    assert rehearsal["first_eight_shards_identical"] is True
    assert rehearsal["selected_for_execution"] == "v1"
    assert rehearsal["v2_execution_authorized"] is False
    assert rehearsal["cloud_mutated"] is False


class _Cloud:
    def __init__(self, *, fail_after: int | None = None) -> None:
        helper = _module(
            f"_m31_fq_cloud_{id(self)}",
            "test_hu_m31_t3_step6d_fresh_quality_gcp_bridge_v1.py",
        )
        base = helper._FakeCloud(
            image_link=IMAGE,
            image_id="123456789",
            features=["GVNIC", "UEFI_COMPATIBLE"],
        )
        self.__dict__.update(base.__dict__)
        self._base_type = type(base)
        self.fail_after = fail_after
        self.create_calls = 0

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._base_type, name)
        if hasattr(attribute, "__get__"):
            return attribute.__get__(self, type(self))
        return attribute

    def get_oauth_token_ttl_seconds(self) -> int:
        return 3600

    def get_service_account(self, *, email: str) -> dict[str, str]:
        index = provider.EXPECTED_WORKER_SERVICE_ACCOUNTS.index(email)
        return {"email": email, "uniqueId": str(30000 + index)}

    def get_cloud_quota(self, *, quota_id: str) -> dict[str, Any]:
        if quota_id == quality_provider.C4_QUOTA_ID:
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
                        "details": {"value": "355"},
                    }
                ],
            }
        return {
            "quotaId": quota_id,
            "metric": "compute.googleapis.com/cpus_all_regions",
            "isPrecise": True,
            "dimensionsInfos": [{"dimensions": {}, "details": {"value": "1000"}}],
        }

    def create_instance(self, *, instance_spec: dict, request_id: str) -> dict:
        if self.fail_after is not None and self.create_calls >= self.fail_after:
            raise RuntimeError("injected partial launch")
        self.create_calls += 1
        return self._base_type.create_instance(
            self, instance_spec=instance_spec, request_id=request_id
        )


def _provider_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    plan, _relocation = _fixture(tmp_path, monkeypatch)
    ledger = transport.build_attempt_ledger(plan)
    resume = transport.build_resume_plan(plan, ledger)
    request = transport.build_wave_request(plan, ledger, resume)
    provider_plan = provider.build_provider_plan(
        transport_plan=plan,
        ledger=ledger,
        resume=resume,
        wave_request=request,
        image_self_link=IMAGE,
        image_id="123456789",
        guest_os_features=["GVNIC", "UEFI_COMPATIBLE"],
        worker_service_accounts=provider.EXPECTED_WORKER_SERVICE_ACCOUNTS,
        local_shard_root=tmp_path / "shards",
    )
    return plan, ledger, resume, request, provider_plan


def test_readonly_preflight_requires_ttl_20_identities_and_35_vcpu_reserve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, ledger, resume, request, provider_plan = _provider_fixture(
        tmp_path, monkeypatch
    )
    cloud = _Cloud()
    receipt = provider.build_readonly_preflight(
        provider_plan=provider_plan,
        transport_plan=plan,
        ledger=ledger,
        resume=resume,
        wave_request=request,
        transport=cloud,  # type: ignore[arg-type]
        observed_at_utc="2026-07-24T00:00:00Z",
    )
    assert (
        provider.validate_readonly_preflight(receipt, provider_plan=provider_plan)
        == receipt
    )
    assert receipt["oauth_ttl_seconds"] == 3600
    assert receipt["required_vcpus"] == 320
    assert all(row["remaining_vcpus"] >= 35 for row in receipt["post_launch_reserve"])
    assert receipt["service_account_count"] == 20
    assert receipt["cloud_mutated"] is False

    cloud.get_oauth_token_ttl_seconds = lambda: 2699  # type: ignore[method-assign]
    with pytest.raises(PermissionError, match="45 minutes"):
        provider.build_readonly_preflight(
            provider_plan=provider_plan,
            transport_plan=plan,
            ledger=ledger,
            resume=resume,
            wave_request=request,
            transport=cloud,  # type: ignore[arg-type]
            observed_at_utc="2026-07-24T00:00:00Z",
        )


def test_partial_launch_is_cleaned_with_exact_iam_and_compute_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, ledger, resume, request, provider_plan = _provider_fixture(
        tmp_path, monkeypatch
    )
    cloud = _Cloud(fail_after=5)
    preflight = provider.build_readonly_preflight(
        provider_plan=provider_plan,
        transport_plan=plan,
        ledger=ledger,
        resume=resume,
        wave_request=request,
        transport=cloud,  # type: ignore[arg-type]
        observed_at_utc="2026-07-24T00:00:00Z",
    )
    stage = provider.stage_content(
        provider_plan=provider_plan, transport=cloud  # type: ignore[arg-type]
    )
    iam = provider.install_worker_iam(
        provider_plan=provider_plan,
        transport=cloud,  # type: ignore[arg-type]
        now_unix_seconds=1_800_000_000,
    )
    launch = provider.execute_wave_atomic(
        provider_plan=provider_plan,
        preflight_receipt=preflight,
        stage_receipt=stage,
        iam_receipt=iam,
        transport=cloud,  # type: ignore[arg-type]
        raw_nonce="12345678-1234-4234-9234-123456789abc",
        sleep=lambda _seconds: None,
    )
    assert launch["status"] == "partial_launch_cleaned_no_go"
    assert launch["created_instance_count"] == 5
    assert launch["partial_launch_cleanup_complete"] is True
    assert cloud.instances == {}
    assert cloud.disks == {}
    titles = set(provider_plan["iam_contract"]["condition_titles"])
    assert all(
        not (
            isinstance(row.get("condition"), dict)
            and row["condition"].get("title") in titles
        )
        for row in cloud.policy["bindings"]
    )


def _controller_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, Any], dict[str, Any], str]:
    plan, relocation = _fixture(tmp_path, monkeypatch)
    config = {
        "schema": controller.PROVIDER_CONFIG_SCHEMA,
        "image_self_link": IMAGE,
        "image_id": "123456789",
        "guest_os_features": ["GVNIC", "UEFI_COMPATIBLE"],
        "worker_service_accounts": list(provider.EXPECTED_WORKER_SERVICE_ACCOUNTS),
        "service_accounts_preexisting": True,
        "service_account_creation_authorized": False,
        "current_profile_changed": False,
    }
    token = "12345678-1234-4234-9234-123456789abc"
    root = tmp_path / "controller"
    controller.initialize_controller(
        transport_plan=plan,
        source_relocation_receipt=relocation,
        provider_config=config,
        local_shard_root=tmp_path / "local-shards",
        output_root=root,
        raw_controller_token=token,
    )
    monkeypatch.setenv(controller.CONTROLLER_TOKEN_ENV, token)
    return root, plan, config, token


def test_controller_is_cloud_neutral_then_resumes_failed_wave(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, plan, _config, _token = _controller_fixture(tmp_path, monkeypatch)
    context = controller.plan_next_wave(root)
    assert isinstance(context, controller.WaveContext)
    assert context.wave_request["selected_count"] == 20
    assert context.provider_plan["runtime_contract"]["required_vcpus"] == 320
    status = controller.controller_status(root)
    assert status["wave_count"] == 18
    assert status["max_concurrent_vms"] == 20
    assert status["worker_service_account_count"] == 20
    assert status["accepted_lifecycle_count"] == 0

    rows = [
        {
            "shard_id": row["shard_id"],
            "attempt_id": row["attempt_id"],
            "instance_id": row["instance_id"],
            "status": "failed",
            "completed_pair_count": 0,
            "checkpoint": None,
            "heartbeat": None,
            "owned_compute_absent": True,
            "worker_iam_removed": True,
        }
        for row in context.wave_request["selected_attempts"]
    ]
    lifecycle = transport.build_lifecycle_receipt(
        plan=plan,
        ledger=context.ledger,
        resume=context.resume,
        wave_request=context.wave_request,
        attempt_rows=rows,
    )
    path = tmp_path / "lifecycle.json"
    path.write_bytes(transport.canonical_bytes(lifecycle))
    accepted = controller.accept_lifecycle_receipt(root, lifecycle_receipt_path=path)
    assert accepted["next_status"] == "wave_ready"
    retry = controller.plan_next_wave(root)
    assert isinstance(retry, controller.WaveContext)
    assert {row["attempt_id"] for row in retry.resume["selected_attempts"]} == {"a01"}
    assert controller.controller_status(root)["accepted_lifecycle_count"] == 1


def test_controller_requires_exact_phase_and_fresh_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, plan, _config, _token = _controller_fixture(tmp_path, monkeypatch)
    context = controller.plan_next_wave(root)
    assert isinstance(context, controller.WaveContext)
    cloud = _Cloud(fail_after=2)
    monkeypatch.setenv(controller.PHASE_SENTINEL_ENV, "wrong")
    with pytest.raises(PermissionError, match="sentinel"):
        controller.readonly_preflight_next(
            root,
            confirm_run_name=plan["run_name"],
            allow_cloud_read=True,
            cloud=cloud,  # type: ignore[arg-type]
            now_unix_seconds=1_800_000_000,
        )
    assert cloud.objects == {}
    assert cloud.instances == {}

    monkeypatch.setenv(
        controller.PHASE_SENTINEL_ENV,
        controller.expected_phase_sentinel(context, "preflight"),
    )
    controller.readonly_preflight_next(
        root,
        confirm_run_name=plan["run_name"],
        allow_cloud_read=True,
        cloud=cloud,  # type: ignore[arg-type]
        now_unix_seconds=1_800_000_000,
    )
    monkeypatch.setenv(
        controller.PHASE_SENTINEL_ENV,
        controller.expected_phase_sentinel(context, "stage"),
    )
    controller.stage_next(
        root,
        confirm_run_name=plan["run_name"],
        allow_cloud_mutation=True,
        cloud=cloud,  # type: ignore[arg-type]
    )
    monkeypatch.setenv(
        controller.PHASE_SENTINEL_ENV,
        controller.expected_phase_sentinel(context, "install-iam"),
    )
    controller.install_iam_next(
        root,
        confirm_run_name=plan["run_name"],
        allow_cloud_mutation=True,
        cloud=cloud,  # type: ignore[arg-type]
        now_unix_seconds=1_800_000_000,
    )
    monkeypatch.setenv(
        controller.PHASE_SENTINEL_ENV,
        controller.expected_phase_sentinel(context, "execute"),
    )
    with pytest.raises(PermissionError, match="older than 5 minutes"):
        controller.execute_next(
            root,
            confirm_run_name=plan["run_name"],
            allow_cloud_mutation=True,
            raw_nonce="12345678-1234-4234-9234-123456789abd",
            cloud=cloud,  # type: ignore[arg-type]
            now_unix_seconds=1_800_000_301,
            sleep=lambda _seconds: None,
        )
    launch = controller.execute_next(
        root,
        confirm_run_name=plan["run_name"],
        allow_cloud_mutation=True,
        raw_nonce="12345678-1234-4234-9234-123456789abd",
        cloud=cloud,  # type: ignore[arg-type]
        now_unix_seconds=1_800_000_299,
        sleep=lambda _seconds: None,
    )
    assert launch["status"] == "partial_launch_cleaned_no_go"
    assert launch["partial_launch_cleanup_complete"] is True
    assert cloud.instances == {}
    monkeypatch.setenv(
        controller.PHASE_SENTINEL_ENV,
        controller.expected_phase_sentinel(context, "receive"),
    )
    received = controller.receive_next(
        root,
        confirm_run_name=plan["run_name"],
        allow_cloud_read=True,
        cloud=cloud,  # type: ignore[arg-type]
    )
    assert received["failed_count"] == 20
    accepted = controller.accept_next(root)
    assert accepted["next_status"] == "wave_ready"
    retry = controller.plan_next_wave(root)
    assert isinstance(retry, controller.WaveContext)
    assert {row["attempt_id"] for row in retry.resume["selected_attempts"]} == {"a01"}
