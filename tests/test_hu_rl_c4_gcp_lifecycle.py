from __future__ import annotations

import copy
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Mapping

import pytest

from ofc_regular import hu_rl_c4_gcp_lifecycle as subject


NOW = 1_782_000_000
LAUNCH_NONCE = "12345678-1234-4234-8234-123456789abc"
CLEANUP_NONCE = "22345678-1234-4234-8234-123456789abc"
PRINCIPAL_SHA = hashlib.sha256(b"controller@example.invalid").hexdigest()
IMAGE_LINK = (
    "https://www.googleapis.com/compute/v1/projects/debian-cloud/global/images/"
    "debian-12-bookworm-v20260721"
)
IMAGE_ID = "1234567890123456789"
MANIFEST_SHA = "a" * 64


def _package() -> dict[str, Any]:
    return {
        "archive_filename": "package.tar.gz",
        "archive_sha256": hashlib.sha256(b"package-bytes").hexdigest(),
        "archive_size_bytes": len(b"package-bytes"),
        "archive_root": "package-root",
        "file_count": 5,
        "files_sha256": "b" * 64,
        "manifest_sha256": MANIFEST_SHA,
        "native_wheel_sha256": "c" * 64,
    }


def _plan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[dict[str, Any], Path, Path]:
    package_root = tmp_path / "package-root"
    package_root.mkdir()
    archive = tmp_path / "package.tar.gz"
    archive.write_bytes(b"package-bytes")
    package = _package()
    monkeypatch.setattr(subject, "verify_frozen_package", lambda **_: dict(package))
    plan = subject.build_execution_plan(
        run_name="rlc-formal-20260722-001",
        package_root=package_root,
        archive_path=archive,
        expected_archive_sha256=package["archive_sha256"],
        bucket="pokerhu-ofc-solver-485418-training",
        worker_service_account="ofc-m31-t3-diagnostic@ofc-solver-485418.iam.gserviceaccount.com",
        source_image_self_link=IMAGE_LINK,
        image_id=IMAGE_ID,
        guest_os_features=["UEFI_COMPATIBLE", "GVNIC"],
    )
    return plan, package_root, archive


def _manifest() -> dict[str, Any]:
    return {"manifest_sha256": MANIFEST_SHA}


class FakeCloud:
    def __init__(self, plan: Mapping[str, Any]) -> None:
        self.plan = plan
        self.objects: dict[str, tuple[bytes, str]] = {}
        self.instances: dict[str, dict[str, Any]] = {}
        self.operations: dict[str, dict[str, Any]] = {}
        self.create_calls = 0
        self.delete_calls = 0
        self.put_calls: list[str] = []
        self.create_response_lost = False
        self.create_response_lost_without_instance = False
        self.delete_response_lost = False
        self.object_response_lost: set[str] = set()
        self.operation_stuck = False
        self.next_generation = 100

    def get_object_metadata(self, *, bucket: str, object_name: str) -> Mapping[str, Any] | None:
        assert bucket == self.plan["bucket"]
        value = self.objects.get(object_name)
        return None if value is None else {"name": object_name, "generation": value[1], "size": str(len(value[0]))}

    def get_object_bytes(self, *, bucket: str, object_name: str, generation: str | None = None) -> bytes | None:
        assert bucket == self.plan["bucket"]
        value = self.objects.get(object_name)
        if value is None or (generation is not None and generation != value[1]):
            return None
        return value[0]

    def put_object_new(
        self, *, bucket: str, object_name: str, payload: bytes, content_type: str
    ) -> Mapping[str, Any]:
        assert bucket == self.plan["bucket"]
        if object_name in self.objects:
            raise subject.HuRlC4GcpLifecycleError("create-only collision")
        generation = str(self.next_generation)
        self.next_generation += 1
        self.objects[object_name] = (bytes(payload), generation)
        self.put_calls.append(object_name)
        if object_name in self.object_response_lost:
            raise subject.ResponseLostError("fixture response lost")
        return {"name": object_name, "generation": generation, "size": str(len(payload))}

    @staticmethod
    def _metadata(spec: Mapping[str, Any]) -> dict[str, str]:
        return {row["key"]: row["value"] for row in spec["metadata"]["items"]}

    def _instance_from_spec(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        disk_name = spec["name"] + "-boot"
        return {
            **copy.deepcopy(dict(spec)),
            "id": "9876543210123456789",
            "status": "RUNNING",
            "cpuPlatform": "Intel Emerald Rapids",
            "machineType": (
                f"https://www.googleapis.com/compute/v1/projects/{subject.PROJECT}/zones/"
                f"{subject.ZONE}/machineTypes/{subject.MACHINE_TYPE}"
            ),
            "disks": [
                {
                    "boot": True,
                    "autoDelete": True,
                    "interface": subject.BOOT_DISK_INTERFACE,
                    "source": (
                        f"https://www.googleapis.com/compute/v1/projects/{subject.PROJECT}/zones/"
                        f"{subject.ZONE}/disks/{disk_name}"
                    ),
                }
            ],
            "networkInterfaces": [
                {
                    "network": spec["networkInterfaces"][0]["network"],
                    "subnetwork": spec["networkInterfaces"][0]["subnetwork"],
                    "nicType": subject.NIC_TYPE,
                }
            ],
        }

    def get_instance(self, *, instance_name: str) -> Mapping[str, Any] | None:
        value = self.instances.get(instance_name)
        return None if value is None else copy.deepcopy(value)

    def create_instance(self, *, instance_spec: Mapping[str, Any], request_id: str) -> Mapping[str, Any]:
        assert request_id == LAUNCH_NONCE
        self.create_calls += 1
        if not self.create_response_lost_without_instance:
            self.instances[instance_spec["name"]] = self._instance_from_spec(instance_spec)
        if self.create_response_lost or self.create_response_lost_without_instance:
            raise subject.ResponseLostError("fixture insert response lost")
        operation = {
            "name": "insert-operation-1",
            "operationType": "insert",
            "targetLink": f"https://compute.googleapis.com/compute/v1/projects/p/zones/z/instances/{instance_spec['name']}",
            "status": "PENDING" if self.operation_stuck else "DONE",
        }
        self.operations[operation["name"]] = operation
        return copy.deepcopy(operation)

    def get_zone_operation(self, *, operation_name: str) -> Mapping[str, Any]:
        return copy.deepcopy(self.operations[operation_name])

    def get_machine_type(self, *, machine_type: str) -> Mapping[str, Any]:
        assert machine_type == subject.MACHINE_TYPE
        return {"name": machine_type, "guestCpus": 16, "memoryMb": 61440}

    def get_disk(self, *, disk_name: str) -> Mapping[str, Any]:
        return {
            "name": disk_name,
            "type": (
                f"https://www.googleapis.com/compute/v1/projects/{subject.PROJECT}/zones/"
                f"{subject.ZONE}/diskTypes/{subject.BOOT_DISK_TYPE}"
            ),
            "sourceImage": IMAGE_LINK,
            "sourceImageId": IMAGE_ID,
        }

    def delete_instance(self, *, instance_name: str, request_id: str) -> Mapping[str, Any]:
        assert request_id == CLEANUP_NONCE
        self.delete_calls += 1
        self.instances.pop(instance_name, None)
        if self.delete_response_lost:
            raise subject.ResponseLostError("fixture delete response lost")
        operation = {
            "name": "delete-operation-1",
            "operationType": "delete",
            "targetLink": f"https://compute.googleapis.com/compute/v1/projects/p/zones/z/instances/{instance_name}",
            "status": "DONE",
        }
        self.operations[operation["name"]] = operation
        return operation


def _launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, fake: FakeCloud | None = None
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, FakeCloud]:
    plan, package_root, archive = _plan(monkeypatch, tmp_path)
    monkeypatch.setattr(subject, "_load_manifest_from_package", lambda _: _manifest())
    cloud = fake or FakeCloud(plan)
    auth = subject.build_operation_authorization(
        plan=plan,
        operation="launch",
        raw_nonce=LAUNCH_NONCE,
        now_unix_seconds=NOW,
    )
    receipt = subject.launch_or_recover(
        plan=plan,
        authorization=auth,
        raw_nonce=LAUNCH_NONCE,
        package_root=package_root,
        archive_path=archive,
        controller_principal_sha256=PRINCIPAL_SHA,
        transport=cloud,
        now_unix_seconds=NOW,
        observed_at_utc="2026-07-22T08:00:00Z",
        sleep=lambda _: None,
    )
    return plan, receipt, package_root, archive, cloud


def test_plan_locks_exact_one_c4_spot_hyperdisk_nvme_gvnic_no_external_ip_and_ttl(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan, _, _ = _plan(monkeypatch, tmp_path)
    subject.validate_execution_plan(plan)
    assert plan["instance_count"] == 1
    assert plan["machine_type"] == "c4-standard-16"
    assert plan["network"]["nic_type"] == "GVNIC"
    assert plan["network"]["external_ip"] is False
    assert plan["vm"]["boot_disk_type"] == "hyperdisk-balanced"
    assert plan["vm"]["boot_disk_interface"] == "NVME"
    assert plan["vm"]["max_run_duration_seconds"] == 4500
    assert plan["current_profile_changed"] is False


def test_plan_rejects_image_without_gvnic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(subject, "verify_frozen_package", lambda **_: _package())
    with pytest.raises(subject.HuRlC4GcpLifecycleError, match="lacks GVNIC"):
        subject.build_execution_plan(
            run_name="rlc-test-001",
            package_root=tmp_path,
            archive_path=tmp_path / "missing",
            expected_archive_sha256=_package()["archive_sha256"],
            bucket="valid-bucket",
            worker_service_account="worker1@ofc-solver-485418.iam.gserviceaccount.com",
            source_image_self_link=IMAGE_LINK,
            image_id=IMAGE_ID,
            guest_os_features=["UEFI_COMPATIBLE"],
        )


def test_real_package_003_archive_sha_is_frozen_when_available() -> None:
    root = Path(r"D:\ofc-gcp-runs\hu-rl-c4-formal-package-20260722-003")
    archive = Path(r"D:\ofc-gcp-runs\hu-rl-c4-formal-package-20260722-003.tar.gz")
    if not root.is_dir() or not archive.is_file():
        pytest.skip("local package _003 is not available")
    result = subject.verify_frozen_package(
        package_root=root,
        archive_path=archive,
        expected_archive_sha256="fa02c8bbea3930887dfc119b2d64827aac5cd4e011bebc9e6df927f30861074d",
    )
    assert result["manifest_sha256"] == "0c94a40c1f687ac51f501a6f9f575db39852f3906aec8f1c5c9d0d219bcad242"
    assert result["file_count"] == 492


def test_package_sha_mismatch_fails_before_any_cloud_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan, package_root, archive = _plan(monkeypatch, tmp_path)
    cloud = FakeCloud(plan)
    auth = subject.build_operation_authorization(
        plan=plan, operation="launch", raw_nonce=LAUNCH_NONCE, now_unix_seconds=NOW
    )
    monkeypatch.setattr(
        subject,
        "verify_frozen_package",
        lambda **_: {**_package(), "archive_sha256": "f" * 64},
    )
    with pytest.raises(subject.HuRlC4GcpLifecycleError, match="no longer matches"):
        subject.launch_or_recover(
            plan=plan,
            authorization=auth,
            raw_nonce=LAUNCH_NONCE,
            package_root=package_root,
            archive_path=archive,
            controller_principal_sha256=PRINCIPAL_SHA,
            transport=cloud,
            now_unix_seconds=NOW,
            sleep=lambda _: None,
        )
    assert cloud.create_calls == 0
    assert cloud.put_calls == []


def test_happy_launch_is_exact_one_and_external_observation_attests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan, receipt, _, _, cloud = _launch(monkeypatch, tmp_path)
    subject.validate_launch_receipt(receipt, plan=plan)
    assert cloud.create_calls == 1
    assert len(cloud.instances) == 1
    assert receipt["instance_count"] == 1
    assert receipt["external_ip_present"] is False
    assert receipt["external_observation"]["vcpu_count"] == 16
    assert receipt["external_observation"]["nic_type"] == "GVNIC"
    assert receipt["external_observation"]["boot_disk_type"] == "hyperdisk-balanced"
    assert receipt["machine_attestation"]["manifest_sha256"] == MANIFEST_SHA
    spec = next(iter(cloud.instances.values()))
    assert "accessConfigs" not in spec["networkInterfaces"][0]
    assert spec["disks"][0]["interface"] == "NVME"


def test_response_lost_after_insert_reconciles_only_exact_owned_instance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan, package_root, archive = _plan(monkeypatch, tmp_path)
    monkeypatch.setattr(subject, "_load_manifest_from_package", lambda _: _manifest())
    cloud = FakeCloud(plan)
    cloud.create_response_lost = True
    auth = subject.build_operation_authorization(
        plan=plan, operation="launch", raw_nonce=LAUNCH_NONCE, now_unix_seconds=NOW
    )
    receipt = subject.launch_or_recover(
        plan=plan,
        authorization=auth,
        raw_nonce=LAUNCH_NONCE,
        package_root=package_root,
        archive_path=archive,
        controller_principal_sha256=PRINCIPAL_SHA,
        transport=cloud,
        now_unix_seconds=NOW,
        observed_at_utc="2026-07-22T08:00:00Z",
        sleep=lambda _: None,
    )
    assert receipt["reconciled_after_response_loss"] is True
    assert cloud.create_calls == 1
    assert len(cloud.instances) == 1


def test_response_lost_without_instance_fails_closed_and_never_claims_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan, package_root, archive = _plan(monkeypatch, tmp_path)
    monkeypatch.setattr(subject, "_load_manifest_from_package", lambda _: _manifest())
    cloud = FakeCloud(plan)
    cloud.create_response_lost_without_instance = True
    auth = subject.build_operation_authorization(
        plan=plan, operation="launch", raw_nonce=LAUNCH_NONCE, now_unix_seconds=NOW
    )
    with pytest.raises(subject.HuRlC4GcpLifecycleError, match="exact instance is absent"):
        subject.launch_or_recover(
            plan=plan,
            authorization=auth,
            raw_nonce=LAUNCH_NONCE,
            package_root=package_root,
            archive_path=archive,
            controller_principal_sha256=PRINCIPAL_SHA,
            transport=cloud,
            now_unix_seconds=NOW,
            sleep=lambda _: None,
        )
    assert cloud.create_calls == 1
    assert cloud.instances == {}


def test_object_put_response_lost_reconciles_exact_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan, package_root, archive = _plan(monkeypatch, tmp_path)
    monkeypatch.setattr(subject, "_load_manifest_from_package", lambda _: _manifest())
    cloud = FakeCloud(plan)
    cloud.object_response_lost.add(plan["objects"]["package"])
    auth = subject.build_operation_authorization(
        plan=plan, operation="launch", raw_nonce=LAUNCH_NONCE, now_unix_seconds=NOW
    )
    receipt = subject.launch_or_recover(
        plan=plan,
        authorization=auth,
        raw_nonce=LAUNCH_NONCE,
        package_root=package_root,
        archive_path=archive,
        controller_principal_sha256=PRINCIPAL_SHA,
        transport=cloud,
        now_unix_seconds=NOW,
        observed_at_utc="2026-07-22T08:00:00Z",
        sleep=lambda _: None,
    )
    assert receipt["package_object"]["generation"] == "100"
    assert cloud.create_calls == 1


def test_preexisting_wrong_owned_instance_fails_before_second_create(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan, package_root, archive = _plan(monkeypatch, tmp_path)
    monkeypatch.setattr(subject, "_load_manifest_from_package", lambda _: _manifest())
    cloud = FakeCloud(plan)
    spec = subject.build_instance_spec(plan, package_generation="100")
    wrong = cloud._instance_from_spec(spec)
    wrong["labels"]["ofc-owner"] = "somebody-else"
    cloud.instances[plan["instance_name"]] = wrong
    auth = subject.build_operation_authorization(
        plan=plan, operation="launch", raw_nonce=LAUNCH_NONCE, now_unix_seconds=NOW
    )
    with pytest.raises(subject.HuRlC4GcpLifecycleError, match="ownership"):
        subject.launch_or_recover(
            plan=plan,
            authorization=auth,
            raw_nonce=LAUNCH_NONCE,
            package_root=package_root,
            archive_path=archive,
            controller_principal_sha256=PRINCIPAL_SHA,
            transport=cloud,
            now_unix_seconds=NOW,
            sleep=lambda _: None,
        )
    assert cloud.create_calls == 0
    assert len(cloud.instances) == 1


def _fake_result(receipt: Mapping[str, Any], *, passed: bool) -> dict[str, Any]:
    value: dict[str, Any] = {
        "manifest_sha256": MANIFEST_SHA,
        "machine_attestation": receipt["machine_attestation"],
        "gates": {"overall_pass": passed},
        "result_sha256": "d" * 64,
    }
    return value


def _status(plan: Mapping[str, Any], *, exit_code: int) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": subject.WORKER_STATUS_SCHEMA,
        "execution_plan_sha256": plan["execution_plan_sha256"],
        "benchmark_exit_code": exit_code,
        "result_uploaded": True,
        "worker_status_sha256": None,
    }
    value["worker_status_sha256"] = subject._self_digest(value, "worker_status_sha256")
    return value


@pytest.mark.parametrize(("passed", "exit_code", "expected"), [(True, 0, "pass"), (False, 2, "no_go")])
def test_collection_accepts_validated_formal_pass_or_no_go(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    passed: bool,
    exit_code: int,
    expected: str,
) -> None:
    plan, launch, package_root, _, cloud = _launch(monkeypatch, tmp_path)
    result = _fake_result(launch, passed=passed)
    cloud.objects[plan["objects"]["result"]] = (subject.canonical_bytes(result), "200")
    cloud.objects[plan["objects"]["worker_status"]] = (
        subject.canonical_bytes(_status(plan, exit_code=exit_code)),
        "201",
    )
    receipt, raw = subject.collect_result(
        plan=plan,
        launch_receipt=launch,
        package_root=package_root,
        transport=cloud,
        sleep=lambda _: None,
        result_validator=lambda value, manifest: None,
    )
    assert expected in receipt["status"]
    assert receipt["formal_gate_pass"] is passed
    assert hashlib.sha256(raw).hexdigest() == receipt["result_bytes_sha256"]


def test_collection_rejects_result_from_another_attestation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan, launch, package_root, _, cloud = _launch(monkeypatch, tmp_path)
    result = _fake_result(launch, passed=True)
    result["machine_attestation"] = {**launch["machine_attestation"], "instance_id": "1"}
    cloud.objects[plan["objects"]["result"]] = (subject.canonical_bytes(result), "200")
    cloud.objects[plan["objects"]["worker_status"]] = (
        subject.canonical_bytes(_status(plan, exit_code=0)),
        "201",
    )
    with pytest.raises(subject.HuRlC4GcpLifecycleError, match="attestation differs"):
        subject.collect_result(
            plan=plan,
            launch_receipt=launch,
            package_root=package_root,
            transport=cloud,
            sleep=lambda _: None,
            result_validator=lambda value, manifest: None,
        )


def test_cleanup_requires_collection_or_explicit_abort(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan, launch, _, _, cloud = _launch(monkeypatch, tmp_path)
    auth = subject.build_operation_authorization(
        plan=plan, operation="cleanup", raw_nonce=CLEANUP_NONCE, now_unix_seconds=NOW
    )
    with pytest.raises(subject.HuRlC4GcpLifecycleError, match="validated collection or explicit abort"):
        subject.cleanup_owned_instance(
            plan=plan,
            authorization=auth,
            raw_nonce=CLEANUP_NONCE,
            launch_receipt=launch,
            collection_receipt=None,
            explicit_abort=False,
            package_generation=launch["package_object"]["generation"],
            transport=cloud,
            now_unix_seconds=NOW,
            sleep=lambda _: None,
        )
    assert cloud.delete_calls == 0


def test_explicit_abort_cleanup_deletes_only_exact_owned_instance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan, launch, _, _, cloud = _launch(monkeypatch, tmp_path)
    unrelated = copy.deepcopy(next(iter(cloud.instances.values())))
    unrelated["name"] = "unrelated-c4-vm"
    unrelated["id"] = "1111111111111111111"
    cloud.instances[unrelated["name"]] = unrelated
    auth = subject.build_operation_authorization(
        plan=plan, operation="cleanup", raw_nonce=CLEANUP_NONCE, now_unix_seconds=NOW
    )
    receipt = subject.cleanup_owned_instance(
        plan=plan,
        authorization=auth,
        raw_nonce=CLEANUP_NONCE,
        launch_receipt=launch,
        collection_receipt=None,
        explicit_abort=True,
        package_generation=launch["package_object"]["generation"],
        transport=cloud,
        now_unix_seconds=NOW,
        sleep=lambda _: None,
    )
    assert receipt["operator_abort"] is True
    assert receipt["scientific_result_claimed"] is False
    assert cloud.delete_calls == 1
    assert plan["instance_name"] not in cloud.instances
    assert "unrelated-c4-vm" in cloud.instances


def test_delete_response_lost_accepts_only_measured_absence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan, launch, _, _, cloud = _launch(monkeypatch, tmp_path)
    cloud.delete_response_lost = True
    auth = subject.build_operation_authorization(
        plan=plan, operation="cleanup", raw_nonce=CLEANUP_NONCE, now_unix_seconds=NOW
    )
    receipt = subject.cleanup_owned_instance(
        plan=plan,
        authorization=auth,
        raw_nonce=CLEANUP_NONCE,
        launch_receipt=launch,
        collection_receipt=None,
        explicit_abort=True,
        package_generation=launch["package_object"]["generation"],
        transport=cloud,
        now_unix_seconds=NOW,
        sleep=lambda _: None,
    )
    assert receipt["instance_confirmed_absent"] is True
    assert receipt["delete_operation"] is None


def test_cleanup_wrong_ownership_never_deletes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan, launch, _, _, cloud = _launch(monkeypatch, tmp_path)
    cloud.instances[plan["instance_name"]]["labels"]["ofc-plan"] = "wrong-plan"
    auth = subject.build_operation_authorization(
        plan=plan, operation="cleanup", raw_nonce=CLEANUP_NONCE, now_unix_seconds=NOW
    )
    with pytest.raises(subject.HuRlC4GcpLifecycleError, match="ownership"):
        subject.cleanup_owned_instance(
            plan=plan,
            authorization=auth,
            raw_nonce=CLEANUP_NONCE,
            launch_receipt=launch,
            collection_receipt=None,
            explicit_abort=True,
            package_generation=launch["package_object"]["generation"],
            transport=cloud,
            now_unix_seconds=NOW,
            sleep=lambda _: None,
        )
    assert cloud.delete_calls == 0


def test_startup_script_verifies_package_waits_for_attestation_and_uploads_result() -> None:
    script = subject.worker_startup_script()
    assert "sha256sum -c" in script
    assert "ifGenerationMatch=0" in script
    assert "c4-attestation-object" in script
    assert "startup_hu_rl_c4_formal_benchmark.sh" in script
    assert "maxRunDuration" not in script
    assert "systemctl poweroff" in script
    assert "curl" not in script


def test_authorization_nonce_expiry_and_plan_tamper_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan, _, _ = _plan(monkeypatch, tmp_path)
    auth = subject.build_operation_authorization(
        plan=plan, operation="launch", raw_nonce=LAUNCH_NONCE, now_unix_seconds=NOW
    )
    with pytest.raises(subject.HuRlC4GcpLifecycleError, match="invalid, expired"):
        subject.validate_operation_authorization(
            auth,
            plan=plan,
            operation="launch",
            raw_nonce=LAUNCH_NONCE,
            now_unix_seconds=NOW + 7200,
        )
    changed = copy.deepcopy(plan)
    changed["instance_count"] = 2
    with pytest.raises(subject.HuRlC4GcpLifecycleError):
        subject.validate_execution_plan(changed)


def test_principal_raw_value_is_not_in_launch_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, receipt, _, _, _ = _launch(monkeypatch, tmp_path)
    encoded = json.dumps(receipt, sort_keys=True)
    assert "controller@example.invalid" not in encoded
    assert PRINCIPAL_SHA in encoded


class _ScriptedRequester:
    """Replays a fixed sequence of outcomes and records what was attempted."""

    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[tuple[str, str]] = []
        self.tokens: list[str] = []

    def __call__(self, method, url, headers, payload, timeout):
        self.calls.append((method, url))
        self.tokens.append(headers["Authorization"])
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _ok(body: bytes = b"{}") -> subject.HttpResponse:
    return subject.HttpResponse(200, body, {})


def _http(code: int) -> subject.HttpResponse:
    return subject.HttpResponse(code, b"{}", {})


def _transport(outcomes: list[Any], **kwargs: Any):
    requester = _ScriptedRequester(outcomes)
    transport = subject.GcpRestTransport(
        access_token="token-one",
        request=requester,
        sleep=lambda _seconds: None,
        **kwargs,
    )
    return transport, requester


def test_read_retries_a_dropped_response_then_succeeds() -> None:
    transport, requester = _transport([TimeoutError("read timed out"), _ok()])
    assert transport.get_instance(instance_name="vm-a") == {}
    assert len(requester.calls) == 2


def test_read_retries_throttling_and_server_errors() -> None:
    transport, requester = _transport([_http(429), _http(503), _ok()])
    assert transport.get_instance(instance_name="vm-a") == {}
    assert len(requester.calls) == 3


def test_read_gives_up_after_the_retry_budget() -> None:
    outcomes: list[Any] = [TimeoutError("read timed out")] * 12
    transport, requester = _transport(outcomes)
    with pytest.raises(TimeoutError):
        transport.get_instance(instance_name="vm-a")
    assert len(requester.calls) == subject._READ_RETRY_ATTEMPTS + 1


def test_a_lost_mutating_response_is_never_retried() -> None:
    """Insert carries a requestId, but retrying is the caller's decision."""

    transport, requester = _transport([TimeoutError("read timed out")])
    with pytest.raises(subject.ResponseLostError):
        transport.create_instance(instance_spec={"name": "vm-a"}, request_id="r-1")
    assert len(requester.calls) == 1


def test_expired_token_is_refreshed_once_and_the_call_retried() -> None:
    transport, requester = _transport(
        [_http(401), _ok()], token_provider=lambda: "token-two"
    )
    assert transport.get_instance(instance_name="vm-a") == {}
    assert requester.tokens == ["Bearer token-one", "Bearer token-two"]


def test_repeated_401_still_fails_rather_than_looping() -> None:
    transport, requester = _transport(
        [_http(401), _http(401)], token_provider=lambda: "token-two"
    )
    with pytest.raises(subject.HuRlC4GcpLifecycleError):
        transport.get_instance(instance_name="vm-a")
    assert len(requester.calls) == 2


def test_without_a_token_provider_401_stays_fatal() -> None:
    transport, requester = _transport([_http(401)])
    with pytest.raises(subject.HuRlC4GcpLifecycleError):
        transport.get_instance(instance_name="vm-a")
    assert len(requester.calls) == 1


def test_missing_resource_is_still_reported_as_absent() -> None:
    transport, _requester = _transport([_http(404)])
    assert transport.get_instance(instance_name="vm-a") is None
