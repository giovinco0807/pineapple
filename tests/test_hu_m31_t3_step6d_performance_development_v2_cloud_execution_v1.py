from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_performance_development_v2_cloud_execution_v1 as subject,
)
from ofc_regular import (
    hu_m31_t3_step6d_performance_development_v2_local_package as local,
)
from ofc_regular import (
    hu_m31_t3_step6d_performance_development_v2_preflight as preflight,
)
from ofc_regular import (
    hu_m31_t3_step6d_performance_development_v2_worker_iam as worker_iam,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "regular-hu-m31-c02-perfdev-v2-20260722-cloud01"
IDENTITY = "perfdev-v2-20260722-cloud01"
OBSERVED_AT = "2026-07-22T15:00:00Z"
OBSERVED_UNIX = int(datetime.fromisoformat(OBSERVED_AT.replace("Z", "+00:00")).timestamp())
NONCE = "26c3d7e2-c7a4-4e5b-8279-7cc782fb04f5"
CLOSEOUT_NONCE = "b9a2d1e1-c6d0-4497-ab92-e50bec1fa538"


def _image() -> dict[str, Any]:
    name = "debian-12-bookworm-v20260721"
    return {
        "schema": preflight.IMAGE_OBSERVATION_SCHEMA,
        "observation_id": "image-cloud-fixture-20260722",
        "observed_at_utc": OBSERVED_AT,
        "project": "debian-cloud",
        "name": name,
        "id": "9021508813201755912",
        "selfLink": f"https://www.googleapis.com/compute/v1/projects/debian-cloud/global/images/{name}",
        "status": "READY",
        "deprecation": {"state": "ACTIVE", "replacement": None},
        "guest_os_features": ["GVNIC", "UEFI_COMPATIBLE", "VIRTIO_SCSI_MULTIQUEUE"],
        "read_only": True,
    }


def _runtime() -> dict[str, Any]:
    return {
        "schema": preflight.RUNTIME_OBSERVATION_SCHEMA,
        "observation_id": "runtime-cloud-fixture-20260722",
        "observed_at_utc": OBSERVED_AT,
        "project": "ofc-solver-485418",
        "region": "asia-northeast1",
        "zone": "asia-northeast1-b",
        "machine_type": {"name": "c4-standard-16", "guest_cpus": 16, "memory_mb": 61_440},
        "spot_price": {"machine_type": "c4-standard-16", "provisioning_model": "SPOT", "currency": "USD", "unit": "vm_hour", "value": "0.53146"},
        "quota": {"c4_cpus": {"limit": 128, "usage": 0}, "spot_cpus": {"limit": 468, "usage": 32}},
        "namespace": {
            "run_name": RUN_NAME,
            "identity_namespace": IDENTITY,
            "result_prefix": f"hu-m31-t3/perfdev-v2/{RUN_NAME}/",
            "run_name_collision_count": 0,
            "identity_collision_count": 0,
            "result_prefix_collision_count": 0,
            "inventory_read_only": True,
        },
        "read_only": True,
        "cloud_mutated": False,
    }


def _dummy_wheel(path: Path, name: str, version: str) -> None:
    dist = name.replace("-", "_")
    filename = f"{dist}-{version}-py3-none-any.whl"
    metadata_dir = f"{dist}-{version}.dist-info"
    with zipfile.ZipFile(path / filename, "w") as archive:
        archive.writestr(f"{metadata_dir}/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        archive.writestr(f"{metadata_dir}/WHEEL", "Wheel-Version: 1.0\nTag: py3-none-any\n")


def _wheelhouse(path: Path) -> Path:
    path.mkdir()
    requirements = REPO_ROOT / "configs/hu_m43_attempt08_runtime_requirements.txt"
    for raw in requirements.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("--"):
            continue
        name, version = line.split("==", 1)
        _dummy_wheel(path, name, version)
    return path


@pytest.fixture(scope="module")
def cloud_package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("perfdev-v2-cloud-package")
    receipt = preflight.build_dry_run_receipt(
        image_observation=_image(), runtime_observation=_runtime()
    )
    receipt_path = root / "dry_receipt.json"
    receipt_path.write_bytes(local.canonical_bytes(receipt))
    local_parent = root / "local"
    local.package_local(
        output_parent=local_parent,
        run_name=RUN_NAME,
        dry_run_receipt_path=receipt_path,
        now_unix_seconds=OBSERVED_UNIX,
    )
    wheelhouse = _wheelhouse(root / "wheelhouse")
    cloud_parent = root / "cloud"
    result = subject.build_cloud_executable_package(
        local_package_dir=local_parent / RUN_NAME,
        wheelhouse_dir=wheelhouse,
        output_parent=cloud_parent,
        now_unix_seconds=OBSERVED_UNIX,
    )
    assert result["cloud_executable"] is True
    return cloud_parent / RUN_NAME


def _plan(package: Path) -> dict[str, Any]:
    return json.loads((package / subject.PLAN_NAME).read_text(encoding="ascii"))


class FakeCloud:
    def __init__(self, plan: Mapping[str, Any], *, fail_second: bool = False) -> None:
        self.plan = dict(plan)
        self.fail_second = fail_second
        self.objects: dict[str, bytes] = {}
        self.created: list[dict[str, Any]] = []
        self.deleted: list[str] = []
        self.instances: dict[str, dict[str, Any]] = {}
        self.collisions = False

    def inspect_namespace(self, *, run_name: str, identity_namespace: str, result_prefix: str, instance_names: Sequence[str]) -> Mapping[str, Any]:
        assert run_name == self.plan["run_name"]
        return {
            "read_only": True,
            "cloud_mutated": False,
            "run_name_collision_count": 1 if self.collisions else 0,
            "identity_collision_count": 0,
            "result_prefix_object_count": 0,
            "instance_collision_counts": {name: 0 for name in instance_names},
            "worker_iam": {
                "read_only": True,
                "service_account": subject.WORKER_SERVICE_ACCOUNT,
                "bucket": subject.WORKER_BUCKET,
                "required_reader_prefix": f"{result_prefix}control/",
                "required_creator_prefix": result_prefix,
                "exact_conditional_binding_count": 2,
                "reader_binding_count": 1,
                "creator_binding_count": 1,
                "excess_worker_binding_count": 0,
                "iam_expiry_unix_seconds": OBSERVED_UNIX + subject.WORKER_IAM_EXPIRY_OFFSET_SECONDS,
                "object_get_allowed": True,
                "object_create_allowed": True,
                "object_list_required": False,
                "exact_prefix_condition": True,
                "all_required_permissions_present": True,
            },
            "network_path": {
                "read_only": True,
                "router_name": subject.NAT_ROUTER_NAME,
                "router_region": self.plan["region"],
                "router_network_exact": True,
                "nat_name": subject.NAT_NAME,
                "nat_count_with_name": 1,
                "nat_ip_allocate_option": "AUTO_ONLY",
                "source_subnetwork_ip_ranges_to_nat": "ALL_SUBNETWORKS_ALL_IP_RANGES",
                "external_ipv4_on_vm": False,
                "path_ready": True,
            },
        }

    def put_if_absent(self, *, object_name: str, payload: bytes) -> Mapping[str, Any]:
        created = object_name not in self.objects
        if created:
            self.objects[object_name] = payload
        return {"created": created, "object_name": object_name, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload), "generation": str(len(self.objects)) if created else "0"}

    def create_instance(self, *, specification: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.fail_second and len(self.created) == 1:
            raise RuntimeError("fixture second create failure")
        spec = deepcopy(dict(specification))
        row = next(row for row in self.plan["instances"] if row["instance_name"] == spec["name"])
        response = {"created": True, "name": spec["name"], "status": "PROVISIONING", "ownership_label": row["ownership_label"], "execution_plan_sha256": subject.canonical_sha256(self.plan)}
        self.created.append(spec)
        self.instances[spec["name"]] = {"name": spec["name"], "ownership_label": row["ownership_label"], "execution_plan_sha256": subject.canonical_sha256(self.plan)}
        return response

    def copy_if_absent(self, *, source_object: str, destination_object: str, expected_sha256: str, expected_bytes: int) -> Mapping[str, Any]:
        raw = self.objects.get(source_object)
        assert raw is not None
        assert hashlib.sha256(raw).hexdigest() == expected_sha256
        assert len(raw) == expected_bytes
        created = destination_object not in self.objects
        if created:
            self.objects[destination_object] = raw
        return {"created": created, "source_object": source_object, "object_name": destination_object, "sha256": expected_sha256, "bytes": expected_bytes, "generation": str(len(self.objects)) if created else "0"}

    def get_object(self, *, object_name: str) -> bytes | None:
        return self.objects.get(object_name)

    def list_objects(self, *, prefix: str) -> Sequence[str]:
        return sorted(name for name in self.objects if name.startswith(prefix))

    def get_instance(self, *, instance_name: str) -> Mapping[str, Any] | None:
        return self.instances.get(instance_name)

    def delete_instance_exact(self, *, instance_name: str, ownership_label: str, execution_plan_sha256: str) -> Mapping[str, Any]:
        observed = self.instances[instance_name]
        assert observed == {"name": instance_name, "ownership_label": ownership_label, "execution_plan_sha256": execution_plan_sha256}
        del self.instances[instance_name]
        self.deleted.append(instance_name)
        return {"instance_name": instance_name, "deleted": True, "ownership_label": ownership_label}


def _worker_iam_readback(plan: Mapping[str, Any]) -> dict[str, Any]:
    nonce_sha256 = hashlib.sha256(NONCE.encode("ascii")).hexdigest()
    opaque_sha256 = hashlib.sha256(b"fixture").hexdigest()
    roles = [
        {
            "role_name": role,
            "included_permissions": list(worker_iam.ROLE_PERMISSIONS[role]),
            "stage": "GA",
            "deleted": False,
            "role_etag_sha256": opaque_sha256,
            "role_fingerprint_sha256": opaque_sha256,
        }
        for role in (worker_iam.READER_ROLE, worker_iam.CREATOR_ROLE)
    ]
    unsigned = {
        "schema": worker_iam.READBACK_RECEIPT_SCHEMA,
        "status": "launch_preflight_exact_two_bindings_validated",
        "iam_plan_sha256": opaque_sha256,
        "execution_plan_sha256": subject.canonical_sha256(plan),
        "one_shot_nonce_sha256": nonce_sha256,
        "prepare_receipt_sha256": opaque_sha256,
        "install_receipt_sha256": opaque_sha256,
        "role_readbacks": roles,
        "project_worker_zero_readback": {
            "read_only": True,
            "worker_principal": worker_iam.WORKER_PRINCIPAL,
            "direct_worker_binding_count": 0,
            "project_policy_fingerprint_sha256": opaque_sha256,
            "project_policy_etag_sha256": opaque_sha256,
        },
        "bucket_policy_fingerprint_sha256": opaque_sha256,
        "bucket_policy_etag_sha256": opaque_sha256,
        "unrelated_policy_fingerprint_sha256": opaque_sha256,
        "exact_binding_count": 2,
        "observed_unix_seconds": OBSERVED_UNIX,
        "launch_preflight": {
            "read_only": True,
            "service_account": subject.WORKER_SERVICE_ACCOUNT,
            "bucket": subject.WORKER_BUCKET,
            "required_reader_prefix": f"{plan['result_prefix']}control/",
            "required_creator_prefix": plan["result_prefix"],
            "exact_conditional_binding_count": 2,
            "reader_binding_count": 1,
            "creator_binding_count": 1,
            "excess_worker_binding_count": 0,
            "iam_expiry_unix_seconds": OBSERVED_UNIX
            + subject.WORKER_IAM_EXPIRY_OFFSET_SECONDS,
            "object_get_allowed": True,
            "object_create_allowed": True,
            "object_list_required": False,
            "exact_prefix_condition": True,
            "all_required_permissions_present": True,
        },
        "cloud_mutation_performed": False,
        "current_profile_changed": False,
    }
    return {**unsigned, "receipt_sha256": subject.canonical_sha256(unsigned)}


def _launch_material(package: Path, fake: FakeCloud | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], FakeCloud]:
    plan = _plan(package)
    cloud = fake or FakeCloud(plan)
    stage_auth = subject.build_content_stage_authorization(cloud_package_dir=package, explicit_stage_authorized=True, one_shot_nonce=NONCE, now_unix_seconds=OBSERVED_UNIX)
    staged = subject.stage_content(cloud_package_dir=package, authorization=stage_auth, raw_one_shot_nonce=NONCE, transport=cloud, now_unix_seconds=OBSERVED_UNIX)
    iam_readback = _worker_iam_readback(plan)
    with tempfile.TemporaryDirectory() as temporary:
        receipt_path = Path(temporary) / "fresh.json"
        receipt_path.write_bytes(subject.canonical_bytes(preflight.build_dry_run_receipt(image_observation=_image(), runtime_observation=_runtime())))
        overlay = subject.build_fresh_launch_overlay(cloud_package_dir=package, content_stage_receipt=staged, worker_iam_readback=iam_readback, expected_one_shot_nonce_sha256=hashlib.sha256(NONCE.encode("ascii")).hexdigest(), fresh_dry_run_receipt_path=receipt_path, now_unix_seconds=OBSERVED_UNIX)
    authorization = subject.build_launch_authorization(cloud_package_dir=package, launch_overlay=overlay, explicit_launch_authorized=True, one_shot_nonce=NONCE, now_unix_seconds=OBSERVED_UNIX)
    return staged, overlay, authorization, cloud


def _authorization(package: Path) -> dict[str, Any]:
    return _launch_material(package)[2]


def test_cloud_package_binds_offline_runtime_and_exact_pair(cloud_package: Path) -> None:
    result = subject.validate_cloud_package(
        cloud_package, now_unix_seconds=OBSERVED_UNIX
    )
    plan = _plan(cloud_package)
    assert result["source_roles"] == ["candidate", "reference"]
    assert result["tail_hand_indices"] == [0, 4, 5, 12, 14, 16, 17, 23, 41, 43]
    assert result["heavy_hand_indices"] == [0, 5, 12, 16, 17, 23, 41, 43]
    assert result["random_hand_indices"] == [4, 14]
    assert result["run_contract_digest"] == subject.TAIL_RUN_CONTRACT_DIGEST
    assert result["run_contract_schema"] == subject.TAIL_RUN_CONTRACT_SCHEMA
    assert result["run_contract_variant"] == subject.TAIL_RUN_CONTRACT_VARIANT
    assert (
        result["selection_manifest_sha256"]
        == subject.TAIL_SELECTION_MANIFEST_SHA256
    )
    assert plan["tail_hand_indices"] == result["tail_hand_indices"]
    assert plan["heavy_hand_indices"] == result["heavy_hand_indices"]
    assert plan["random_hand_indices"] == result["random_hand_indices"]
    assert plan["roadmap_amendment"] == {
        "schema": "hu_m31_t3_step6d_perfdev_v2_tail_v2_roadmap_amendment_v1",
        "reason": "old_profile_tail_is_mixed_geometry_and_not_tail_qualification",
        "superseded_mixed_diagnostic_hand_indices": [
            2,
            6,
            7,
            9,
            13,
            20,
            21,
            29,
            33,
            50,
        ],
        "qualification_hand_indices": result["tail_hand_indices"],
        "heavy_21x21_hand_indices": result["heavy_hand_indices"],
        "random_hand_indices": result["random_hand_indices"],
        "old_results_reused": False,
        "fresh_rerun_required": True,
    }
    assert plan["allocation"] == {
        "machine_type": "c4-standard-16",
        "vm_count": 2,
        "worker_processes_per_vm": 1,
        "rayon_threads_per_process": 16,
        "spot": True,
        "boot_disk_type": "hyperdisk-balanced",
        "boot_disk_interface": "NVME",
        "boot_disk_size_gb": 20,
        "network_nic_type": "GVNIC",
    }
    assert plan["runtime_wheelhouse"]["offline_install_only"] is True
    assert plan["feature_encoder"]["sha256"] == local.FEATURE_LIBRARY_SHA256
    assert plan["startup"]["sha256"] != local.REARM2_STARTUP_SHA256
    assert result["launch_authorized"] is False
    assert result["cloud_mutated"] is False


@pytest.mark.parametrize(
    ("field", "legacy_value"),
    (
        (
            "schema",
            "hu_m31_t3_step6d_perfdev_v2_execution_plan_v1",
        ),
        ("run_contract_digest", "92a87f1544a73104d5256db39b022094c8c7f7b11f77bd19dcf1ab1442581ebd"),
        ("run_contract_schema", subject.runner.CANDIDATE02_RUN_CONTRACT_SCHEMA),
        ("run_contract_variant", subject.runner.CANDIDATE02_VARIANT),
        ("selection_manifest_sha256", "0" * 64),
        (
            "tail_hand_indices",
            [2, 6, 7, 9, 13, 20, 21, 29, 33, 50],
        ),
    ),
)
def test_execution_plan_rejects_legacy_candidate02_tail_identity(
    cloud_package: Path, field: str, legacy_value: Any
) -> None:
    plan = _plan(cloud_package)
    plan[field] = legacy_value
    with pytest.raises(ValueError, match="frozen boundary"):
        subject.validate_execution_plan(plan, require_fresh_receipt=False)


def test_wheel_platform_gate_rejects_windows_cp312_and_aarch64(tmp_path: Path) -> None:
    for index, tag in enumerate(("cp311-cp311-win_amd64", "cp312-cp312-manylinux_2_17_x86_64", "cp311-cp311-manylinux_2_17_aarch64")):
        path = tmp_path / f"bad{index}-1.0-py3-none-any.whl"
        dist = f"bad{index}-1.0.dist-info"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(f"{dist}/METADATA", f"Metadata-Version: 2.1\nName: bad{index}\nVersion: 1.0\n")
            archive.writestr(f"{dist}/WHEEL", f"Wheel-Version: 1.0\nTag: {tag}\n")
        with pytest.raises(ValueError, match="CPython 3.11 Linux x86_64"):
            subject._wheel_distribution_identity(path)


def test_wheel_identity_ignores_vendored_nested_dist_info(tmp_path: Path) -> None:
    path = tmp_path / "setuptools-83.0.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "setuptools-83.0.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: setuptools\nVersion: 83.0.0\n",
        )
        archive.writestr(
            "setuptools-83.0.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nTag: py3-none-any\n",
        )
        archive.writestr(
            "setuptools/_vendor/vendored-1.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: vendored\nVersion: 1.0\n",
        )
        archive.writestr(
            "setuptools/_vendor/vendored-1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nTag: py3-none-any\n",
        )
    assert subject._wheel_distribution_identity(path) == (
        "setuptools",
        "83.0.0",
        ("py3-none-any",),
    )


def test_authorization_requires_explicit_flag_raw_nonce_and_freshness(cloud_package: Path) -> None:
    _staged, overlay, auth, _fake = _launch_material(cloud_package)
    with pytest.raises(PermissionError, match="explicit"):
        subject.build_launch_authorization(cloud_package_dir=cloud_package, launch_overlay=overlay, explicit_launch_authorized=False, one_shot_nonce=NONCE, now_unix_seconds=OBSERVED_UNIX)
    plan = _plan(cloud_package)
    subject.validate_launch_authorization(auth, execution_plan=plan, raw_nonce=NONCE, now_unix_seconds=OBSERVED_UNIX)
    with pytest.raises(ValueError, match="nonce"):
        subject.validate_launch_authorization(auth, execution_plan=plan, raw_nonce=str(uuid.uuid4()), now_unix_seconds=OBSERVED_UNIX)
    with pytest.raises(ValueError, match="raw one-shot nonce"):
        subject.validate_launch_authorization(auth, execution_plan=plan, raw_nonce=None, now_unix_seconds=OBSERVED_UNIX)  # type: ignore[arg-type]


def test_worker_iam_readback_binds_nonce_and_project_zero(cloud_package: Path) -> None:
    plan = _plan(cloud_package)
    expected_nonce_sha256 = hashlib.sha256(NONCE.encode("ascii")).hexdigest()
    receipt = _worker_iam_readback(plan)
    subject.validate_worker_iam_readback(
        receipt,
        execution_plan=plan,
        expected_one_shot_nonce_sha256=expected_nonce_sha256,
        now_unix_seconds=OBSERVED_UNIX,
    )
    with pytest.raises(ValueError, match="lifecycle readback"):
        subject.validate_worker_iam_readback(
            receipt,
            execution_plan=plan,
            expected_one_shot_nonce_sha256="0" * 64,
            now_unix_seconds=OBSERVED_UNIX,
        )
    body = deepcopy(receipt)
    body.pop("receipt_sha256")
    body["project_worker_zero_readback"]["direct_worker_binding_count"] = 1
    drifted = {**body, "receipt_sha256": subject.canonical_sha256(body)}
    with pytest.raises(ValueError, match="project worker-zero"):
        subject.validate_worker_iam_readback(
            drifted,
            execution_plan=plan,
            expected_one_shot_nonce_sha256=expected_nonce_sha256,
            now_unix_seconds=OBSERVED_UNIX,
        )
def test_fake_launch_creates_exact_two_and_pins_startup_wheels(cloud_package: Path) -> None:
    plan = _plan(cloud_package)
    fake = FakeCloud(plan)
    staged, overlay, auth, _ = _launch_material(cloud_package, fake)
    receipt = subject.launch_pair(cloud_package_dir=cloud_package, content_stage_receipt=staged, launch_overlay=overlay, authorization=auth, raw_one_shot_nonce=NONCE, bucket=subject.WORKER_BUCKET, transport=fake, now_unix_seconds=OBSERVED_UNIX)
    assert receipt["status"] == "launched_exact_candidate_reference_pair"
    assert receipt["all_instances_started"] is True
    assert [row["name"] for row in fake.created] == [row["instance_name"] for row in plan["instances"]]
    for spec in fake.created:
        assert spec["machine_type"] == "c4-standard-16"
        assert spec["metadata"]["WHEELHOUSE_SHA256"] == plan["runtime_wheelhouse"]["sha256"]
        assert hashlib.sha256(spec["metadata"]["startup-script"].encode()).hexdigest() == plan["startup"]["sha256"]


def test_runtime_binding_rebuilds_all_object_wheel_and_limit_metadata(cloud_package: Path) -> None:
    plan = _plan(cloud_package)
    auth = _authorization(cloud_package)
    instance = plan["instances"][0]
    control = f"{plan['result_prefix']}control/"
    expected = {
        "project_id": plan["project"], "bucket": subject.WORKER_BUCKET,
        "run_name": plan["run_name"], "identity_namespace": plan["identity_namespace"],
        "result_prefix": plan["result_prefix"], "source_role": "candidate",
        "instance_name": instance["instance_name"],
        "source_object": f"{control}{subject.SOURCE_NAME}",
        "source_sha256": plan["source_archive"]["sha256"],
        "wheelhouse_object": f"{control}{subject.WHEELHOUSE_NAME}",
        "wheelhouse_sha256": plan["runtime_wheelhouse"]["sha256"],
        "wheelhouse_manifest_sha256": plan["runtime_wheelhouse"]["manifest_sha256"],
        "plan_object": f"{control}{subject.PLAN_NAME}",
        "plan_sha256": subject.canonical_sha256(plan),
        "authorization_object": f"{control}launch_authorization.json",
        "authorization_sha256": subject.canonical_sha256(auth),
        "authorization_nonce_sha256": auth["one_shot_nonce_sha256"],
        "startup_sha256": plan["startup"]["sha256"],
        "ownership_label": instance["ownership_label"], "attempt_index": 0,
        "resume_object": "none", "resume_sha256": "none",
        "max_runtime_seconds": subject.MAX_VM_RUNTIME_SECONDS,
        "heartbeat_seconds": subject.HEARTBEAT_SECONDS,
    }
    assert subject.validate_runtime_binding(plan, auth, expected) == expected
    tampered = deepcopy(expected)
    tampered["wheelhouse_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="runtime metadata"):
        subject.validate_runtime_binding(plan, auth, tampered)


def test_collision_fails_before_any_publish_or_create(cloud_package: Path) -> None:
    fake = FakeCloud(_plan(cloud_package))
    staged, overlay, auth, _ = _launch_material(cloud_package, fake)
    staged_names = set(fake.objects)
    fake.collisions = True
    with pytest.raises(ValueError, match="collision"):
        subject.launch_pair(cloud_package_dir=cloud_package, content_stage_receipt=staged, launch_overlay=overlay, authorization=auth, raw_one_shot_nonce=NONCE, bucket=subject.WORKER_BUCKET, transport=fake, now_unix_seconds=OBSERVED_UNIX)
    assert set(fake.objects) == staged_names
    assert fake.created == []


def test_launch_time_iam_drift_fails_before_nonce_claim_or_vm_create(
    cloud_package: Path,
) -> None:
    staged, overlay, auth, fake = _launch_material(cloud_package)
    before_objects = dict(fake.objects)
    original_inspect = fake.inspect_namespace

    def drifted_inspect(**kwargs: Any) -> Mapping[str, Any]:
        observed = deepcopy(dict(original_inspect(**kwargs)))
        observed["worker_iam"]["excess_worker_binding_count"] = 1
        observed["worker_iam"]["all_required_permissions_present"] = False
        return observed

    fake.inspect_namespace = drifted_inspect  # type: ignore[method-assign]
    with pytest.raises(PermissionError, match="worker IAM"):
        subject.launch_pair(
            cloud_package_dir=cloud_package,
            content_stage_receipt=staged,
            launch_overlay=overlay,
            authorization=auth,
            raw_one_shot_nonce=NONCE,
            bucket=subject.WORKER_BUCKET,
            transport=fake,
            now_unix_seconds=OBSERVED_UNIX,
        )
    assert fake.objects == before_objects
    assert fake.created == []


def test_partial_launch_is_not_success_and_only_created_owned_vm_can_be_cleaned(cloud_package: Path) -> None:
    plan = _plan(cloud_package)
    fake = FakeCloud(plan, fail_second=True)
    staged, overlay, auth, _ = _launch_material(cloud_package, fake)
    receipt = subject.launch_pair(cloud_package_dir=cloud_package, content_stage_receipt=staged, launch_overlay=overlay, authorization=auth, raw_one_shot_nonce=NONCE, bucket=subject.WORKER_BUCKET, transport=fake, now_unix_seconds=OBSERVED_UNIX)
    assert receipt["status"] == "partial_launch_failure_not_success"
    assert receipt["all_instances_started"] is False
    cleanup = subject.cleanup_partial_launch(partial_launch_receipt=receipt, execution_plan=plan, transport=fake)
    assert cleanup["status"] == "partial_launch_owned_cleanup_complete"
    assert fake.deleted == [plan["instances"][0]["instance_name"]]
    assert plan["instances"][1]["instance_name"] not in fake.deleted


def test_full_launch_failure_closeout_is_explicit_exact_and_never_scientific(
    cloud_package: Path, tmp_path: Path,
) -> None:
    plan = _plan(cloud_package)
    fake = FakeCloud(plan)
    staged, overlay, auth, _ = _launch_material(cloud_package, fake)
    launch = subject.launch_pair(
        cloud_package_dir=cloud_package,
        content_stage_receipt=staged,
        launch_overlay=overlay,
        authorization=auth,
        raw_one_shot_nonce=NONCE,
        bucket=subject.WORKER_BUCKET,
        transport=fake,
        now_unix_seconds=OBSERVED_UNIX,
    )
    assert launch["all_instances_started"] is True
    with pytest.raises(PermissionError, match="operator-abort"):
        subject.closeout_failed_owned_launch_pair(
            launch_receipt=launch,
            execution_plan=plan,
            raw_cleanup_nonce=NONCE,
            explicit_operator_abort=False,
            transport=fake,
        )
    assert fake.deleted == []

    single_body = deepcopy(launch)
    single_body.pop("receipt_content_sha256")
    single_body["created_instances"] = single_body["created_instances"][:1]
    single = {
        **single_body,
        "receipt_content_sha256": subject.canonical_sha256(single_body),
    }
    with pytest.raises(ValueError, match="full pair launch"):
        subject._validate_full_launch_receipt(single, execution_plan=plan)
    unrelated_body = deepcopy(launch)
    unrelated_body.pop("receipt_content_sha256")
    unrelated_body["created_instances"][0]["ownership_label"] = (
        "pdv2-00000000000000000000"
    )
    unrelated = {
        **unrelated_body,
        "receipt_content_sha256": subject.canonical_sha256(unrelated_body),
    }
    with pytest.raises(ValueError, match="ownership"):
        subject._validate_full_launch_receipt(unrelated, execution_plan=plan)

    closeout = subject.closeout_failed_owned_launch_pair(
        launch_receipt=launch,
        execution_plan=plan,
        raw_cleanup_nonce=NONCE,
        explicit_operator_abort=True,
        transport=fake,
    )
    checked = subject._validate_owned_launch_failure_closeout_receipt(
        closeout, execution_plan=plan
    )
    assert checked["operator_abort"] is True
    assert checked["automatic_closeout"] is False
    assert checked["scientific_result_claimed"] is False
    assert checked["artifact_validation_claimed"] is False
    assert checked["training_eligible"] is False
    assert checked["all_owned_instances_deleted"] is True
    assert sorted(fake.deleted) == sorted(
        row["instance_name"] for row in plan["instances"]
    )
    with pytest.raises(ValueError, match="validated complete pair"):
        subject._validate_collection_receipt(closeout)
    launch_path = tmp_path / "full_launch.json"
    output_path = tmp_path / "abort_dry_run.json"
    launch_path.write_bytes(subject.canonical_bytes(launch))

    def forbidden_factory(**_: Any) -> FakeCloud:
        raise AssertionError("operator-abort dry-run must not construct transport")

    assert subject.main(
        [
            "cleanup-launched-failure",
            "--cloud-package", str(cloud_package),
            "--launch-receipt", str(launch_path),
            "--cleanup-nonce", NONCE,
            "--bucket", subject.WORKER_BUCKET,
            "--authorize",
            "OPERATOR_ABORT_DELETE_ONLY_EXACT_OWNED_LAUNCHED_PAIR_NO_SCIENTIFIC_RESULT",
            "--output", str(output_path),
            "--dry-run",
        ],
        transport_factory=forbidden_factory,
    ) == 0
    dry = json.loads(output_path.read_text(encoding="utf-8"))
    assert dry["operator_abort"] is True
    assert dry["scientific_result_claimed"] is False
    assert dry["cleanup_executed"] is False


def test_full_launch_closeout_requires_post_delete_absence(
    cloud_package: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(cloud_package)

    class StickyDeleteFake(FakeCloud):
        def delete_instance_exact(
            self, *, instance_name: str, ownership_label: str,
            execution_plan_sha256: str,
        ) -> Mapping[str, Any]:
            observed = self.instances[instance_name]
            assert observed["ownership_label"] == ownership_label
            assert observed["execution_plan_sha256"] == execution_plan_sha256
            self.deleted.append(instance_name)
            return {
                "instance_name": instance_name,
                "deleted": True,
                "ownership_label": ownership_label,
            }

    fake = StickyDeleteFake(plan)
    staged, overlay, auth, _ = _launch_material(cloud_package, fake)
    launch = subject.launch_pair(
        cloud_package_dir=cloud_package,
        content_stage_receipt=staged,
        launch_overlay=overlay,
        authorization=auth,
        raw_one_shot_nonce=NONCE,
        bucket=subject.WORKER_BUCKET,
        transport=fake,
        now_unix_seconds=OBSERVED_UNIX,
    )
    monkeypatch.setattr(subject, "DELETE_CONFIRMATION_ATTEMPTS", 2)
    monkeypatch.setattr(subject.time, "sleep", lambda _seconds: None)
    with pytest.raises(TimeoutError, match="not confirmed absent"):
        subject.closeout_failed_owned_launch_pair(
            launch_receipt=launch,
            execution_plan=plan,
            raw_cleanup_nonce=NONCE,
            explicit_operator_abort=True,
            transport=fake,
        )


def test_real_closeout_polls_delete_done_then_measures_404(
    cloud_package: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(cloud_package)
    fake = FakeCloud(plan)
    staged, overlay, auth, _ = _launch_material(cloud_package, fake)
    launch = subject.launch_pair(
        cloud_package_dir=cloud_package,
        content_stage_receipt=staged,
        launch_overlay=overlay,
        authorization=auth,
        raw_one_shot_nonce=NONCE,
        bucket=subject.WORKER_BUCKET,
        transport=fake,
        now_unix_seconds=OBSERVED_UNIX,
    )
    present = {
        row["instance_name"]: {
            "ofc-owner": row["ownership_label"],
            "ofc-plan": subject.canonical_sha256(plan)[:32],
        }
        for row in plan["instances"]
    }
    operations: dict[str, str] = {}

    def requester(
        method: str, url: str, headers: Mapping[str, str], body: bytes | None,
        timeout: int,
    ) -> subject.HttpResponse:
        if method == "DELETE":
            name = url.rsplit("/", 1)[1]
            operation_name = f"delete-{len(operations) + 1}"
            target = (
                "https://www.googleapis.com/compute/v1/projects/"
                f"{plan['project']}/zones/{plan['zone']}/instances/{name}"
            )
            operations[operation_name] = name
            return subject.HttpResponse(
                200,
                json.dumps(
                    {
                        "name": operation_name,
                        "status": "PENDING",
                        "operationType": "delete",
                        "targetLink": target,
                    }
                ).encode(),
                {},
            )
        if method == "GET" and "/operations/" in url:
            operation_name = url.rsplit("/", 1)[1]
            name = operations[operation_name]
            present.pop(name, None)
            target = (
                "https://www.googleapis.com/compute/v1/projects/"
                f"{plan['project']}/zones/{plan['zone']}/instances/{name}"
            )
            return subject.HttpResponse(
                200,
                json.dumps(
                    {
                        "name": operation_name,
                        "status": "DONE",
                        "operationType": "delete",
                        "targetLink": target,
                    }
                ).encode(),
                {},
            )
        if method == "GET" and "/instances/" in url:
            name = url.rsplit("/", 1)[1]
            if name not in present:
                return subject.HttpResponse(404, b"", {})
            return subject.HttpResponse(
                200,
                json.dumps({"name": name, "labels": present[name]}).encode(),
                {},
            )
        raise AssertionError((method, url))

    monkeypatch.setenv(
        subject.GcpHttpTransport.TOKEN_ENV, "fixture-secret-token-123456789"
    )
    adapter = subject.GcpHttpTransport(
        project=plan["project"], zone=plan["zone"], bucket=subject.WORKER_BUCKET,
        mode="launch-failure-cleanup", execution_plan=plan,
        owned_launch_receipt=launch, requester=requester,
        sleep=lambda _seconds: None,
    )
    closeout = subject.closeout_failed_owned_launch_pair(
        launch_receipt=launch,
        execution_plan=plan,
        raw_cleanup_nonce=NONCE,
        explicit_operator_abort=True,
        transport=adapter,
    )
    assert closeout["all_owned_instances_deleted"] is True
    assert present == {}


def _populate_complete_results(fake: FakeCloud, plan: Mapping[str, Any], auth: Mapping[str, Any]) -> None:
    contract = subject.runner.build_run_contract(
        candidate_library_sha256=subject.contract_v1.CANDIDATE_LIBRARY_SHA256,
        reference_library_sha256=subject.contract_v1.REFERENCE_LIBRARY_SHA256,
        variant=subject.TAIL_RUN_CONTRACT_VARIANT,
    )
    for role in subject.SOURCE_ROLES:
        shard_manifest = subject.runner.build_shard_manifest(
            run_contract=contract,
            source_role=role,
            work_hand_indices=subject.TAIL_HAND_INDICES,
        )
        prefix = f"{plan['result_prefix']}results/{role}/"
        records = []
        for relative in subject._required_artifact_paths(role):
            if relative == "run_contract.json":
                raw = subject.canonical_bytes(contract)
            elif relative == "shard_manifest.json":
                raw = subject.canonical_bytes(shard_manifest)
            elif relative.startswith("roots/"):
                raw = f"same-{relative}\n".encode()
            else:
                raw = f"{role}-{relative}\n".encode()
            fake.objects[f"{prefix}{relative}"] = raw
            records.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
        instance = next(row for row in plan["instances"] if row["source_role"] == role)
        manifest = {
            "schema": subject.ROLE_RESULT_SCHEMA,
            "status": "complete_validated_role_result",
            "run_name": plan["run_name"], "source_role": role,
            "instance_name": instance["instance_name"], "attempt_index": 0,
            "source_sha256": plan["source_archive"]["sha256"],
            "execution_plan_sha256": subject.canonical_sha256(plan),
            "authorization_sha256": subject.canonical_sha256(auth),
            "authorization_nonce_sha256": auth["one_shot_nonce_sha256"],
            "startup_sha256": plan["startup"]["sha256"],
            "ownership_label": instance["ownership_label"],
            "work_hand_indices": list(subject.TAIL_HAND_INDICES),
            "artifact_count": len(records), "artifacts": records,
            "artifact_manifest_sha256": subject.canonical_sha256(records),
            "heartbeat_count": 1, "runner_validation_passed": True,
            "partial_result": False,
        }
        fake.objects[f"{prefix}RESULT_MANIFEST.json"] = subject.canonical_bytes(manifest)
        fake.objects[f"{plan['result_prefix']}heartbeats/{role}/attempt-0/1.json"] = b"heartbeat"
        fake.objects[f"{plan['result_prefix']}progress/{role}/run_contract.json"] = b"contract"
        fake.objects[f"{plan['result_prefix']}progress/{role}/shard_manifest.json"] = b"manifest"
        fake.instances[instance["instance_name"]] = {"name": instance["instance_name"], "ownership_label": instance["ownership_label"], "execution_plan_sha256": subject.canonical_sha256(plan)}


def test_collect_requires_both_complete_then_cleanup_exact_owned_pair(cloud_package: Path) -> None:
    plan = _plan(cloud_package)
    auth = _authorization(cloud_package)
    fake = FakeCloud(plan)
    with pytest.raises(ValueError, match="partial result"):
        subject.collect_pair(cloud_package_dir=cloud_package, authorization=auth, transport=fake, completed_output_validator=lambda _: {})
    _populate_complete_results(fake, plan, auth)
    collection = subject.collect_pair(cloud_package_dir=cloud_package, authorization=auth, transport=fake, completed_output_validator=lambda _: {})
    assert collection["artifact_validation_passed"] is True
    cleanup = subject.cleanup_collected_pair(collection_receipt=collection, transport=fake)
    assert cleanup["status"] == "exact_owned_pair_cleanup_complete"
    assert sorted(fake.deleted) == sorted(row["instance_name"] for row in plan["instances"])


@pytest.mark.parametrize(
    ("field", "legacy_value"),
    (
        (
            "schema",
            "hu_m31_t3_step6d_perfdev_v2_role_result_manifest_v1",
        ),
        (
            "work_hand_indices",
            [2, 6, 7, 9, 13, 20, 21, 29, 33, 50],
        ),
    ),
)
def test_collect_rejects_legacy_role_result_manifest(
    cloud_package: Path, field: str, legacy_value: Any
) -> None:
    plan = _plan(cloud_package)
    auth = _authorization(cloud_package)
    fake = FakeCloud(plan)
    _populate_complete_results(fake, plan, auth)
    object_name = f"{plan['result_prefix']}results/candidate/RESULT_MANIFEST.json"
    manifest = json.loads(fake.objects[object_name])
    manifest[field] = legacy_value
    fake.objects[object_name] = subject.canonical_bytes(manifest)
    with pytest.raises(ValueError, match="complete frozen result"):
        subject.collect_pair(
            cloud_package_dir=cloud_package,
            authorization=auth,
            transport=fake,
            completed_output_validator=lambda _: {},
        )


def test_collect_rejects_legacy_candidate02_run_contract_even_with_stub_validator(
    cloud_package: Path,
) -> None:
    plan = _plan(cloud_package)
    auth = _authorization(cloud_package)
    fake = FakeCloud(plan)
    _populate_complete_results(fake, plan, auth)
    legacy = subject.runner.build_run_contract(
        candidate_library_sha256=subject.contract_v1.CANDIDATE_LIBRARY_SHA256,
        reference_library_sha256=subject.contract_v1.REFERENCE_LIBRARY_SHA256,
        variant=subject.runner.CANDIDATE02_VARIANT,
    )
    prefix = f"{plan['result_prefix']}results/candidate/"
    artifact_object = f"{prefix}run_contract.json"
    raw = subject.canonical_bytes(legacy)
    fake.objects[artifact_object] = raw
    manifest_object = f"{prefix}RESULT_MANIFEST.json"
    manifest = json.loads(fake.objects[manifest_object])
    record = next(
        item for item in manifest["artifacts"] if item["path"] == "run_contract.json"
    )
    record["sha256"] = hashlib.sha256(raw).hexdigest()
    record["bytes"] = len(raw)
    manifest["artifact_manifest_sha256"] = subject.canonical_sha256(
        manifest["artifacts"]
    )
    fake.objects[manifest_object] = subject.canonical_bytes(manifest)
    with pytest.raises(ValueError, match="frozen Candidate02 tail-v2 contract"):
        subject.collect_pair(
            cloud_package_dir=cloud_package,
            authorization=auth,
            transport=fake,
            completed_output_validator=lambda _: {},
        )


def test_real_http_adapter_is_mode_separated_and_token_is_not_returned(cloud_package: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _plan(cloud_package)
    auth = _authorization(cloud_package)
    calls: list[tuple[str, str, Mapping[str, str], bytes | None, int]] = []
    inserted: dict[str, Mapping[str, Any]] = {}

    def requester(method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: int) -> subject.HttpResponse:
        calls.append((method, url, dict(headers), body, timeout))
        if method == "GET" and f"/routers/{subject.NAT_ROUTER_NAME}" in url:
            router = {"name": subject.NAT_ROUTER_NAME, "network": f"https://www.googleapis.com/compute/v1/projects/{plan['project']}/global/networks/default", "nats": [{"name": subject.NAT_NAME, "natIpAllocateOption": "AUTO_ONLY", "sourceSubnetworkIpRangesToNat": "ALL_SUBNETWORKS_ALL_IP_RANGES"}]}
            return subject.HttpResponse(200, json.dumps(router).encode(), {})
        if method == "GET" and "/operations/op-1" in url:
            name = next(iter(inserted))
            operation = {
                "name": "op-1",
                "status": "DONE",
                "operationType": "insert",
                "targetLink": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    f"{plan['project']}/zones/{plan['zone']}/instances/{name}"
                ),
            }
            return subject.HttpResponse(200, json.dumps(operation).encode(), {})
        if method == "GET" and "compute.googleapis.com" in url:
            for name, labels in inserted.items():
                if url.endswith(f"/instances/{name}"):
                    return subject.HttpResponse(
                        200,
                        json.dumps({"name": name, "labels": labels}).encode(),
                        {},
                    )
            return subject.HttpResponse(404, b"", {})
        if method == "GET" and "/iam?" in url:
            assert "optionsRequestedPolicyVersion=3" in url
            expiry = datetime.fromtimestamp(OBSERVED_UNIX + subject.WORKER_IAM_EXPIRY_OFFSET_SECONDS, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            reader_expression = f'resource.name.startsWith("projects/_/buckets/{subject.WORKER_BUCKET}/objects/{plan["result_prefix"]}control/") && request.time < timestamp("{expiry}")'
            creator_expression = f'resource.name.startsWith("projects/_/buckets/{subject.WORKER_BUCKET}/objects/{plan["result_prefix"]}") && request.time < timestamp("{expiry}")'
            policy = {"bindings": [
                {"role": "projects/ofc-solver-485418/roles/ofcM31T3ObjectReaderV1", "members": [f"serviceAccount:{subject.WORKER_SERVICE_ACCOUNT}"], "condition": {"expression": reader_expression}},
                {"role": "projects/ofc-solver-485418/roles/ofcM31T3ResultCreatorV1", "members": [f"serviceAccount:{subject.WORKER_SERVICE_ACCOUNT}"], "condition": {"expression": creator_expression}},
            ]}
            return subject.HttpResponse(200, json.dumps(policy).encode(), {})
        if method == "GET" and "storage.googleapis.com/storage/v1" in url:
            return subject.HttpResponse(200, b"{}", {})
        if method == "POST" and "upload/storage" in url:
            query = urllib_parse(url)
            return subject.HttpResponse(200, json.dumps({"name": query["name"][0], "generation": "1"}).encode(), {})
        if method == "POST" and "compute.googleapis.com" in url:
            assert body is not None
            request = json.loads(body)
            inserted[request["name"]] = request["labels"]
            operation = {
                "name": "op-1",
                "status": "PENDING",
                "operationType": "insert",
                "targetLink": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    f"{plan['project']}/zones/{plan['zone']}/instances/"
                    f"{request['name']}"
                ),
            }
            return subject.HttpResponse(200, json.dumps(operation).encode(), {})
        raise AssertionError((method, url))

    monkeypatch.setenv(subject.GcpHttpTransport.TOKEN_ENV, "fixture-secret-token-123456789")
    adapter = subject.GcpHttpTransport(project=plan["project"], zone=plan["zone"], bucket=subject.WORKER_BUCKET, mode="launch", execution_plan=plan, authorization=auth, raw_one_shot_nonce=NONCE, now_unix_seconds=OBSERVED_UNIX, requester=requester, clock=lambda: OBSERVED_UNIX, sleep=lambda _seconds: None)
    observation = adapter.inspect_namespace(run_name=plan["run_name"], identity_namespace=plan["identity_namespace"], result_prefix=plan["result_prefix"], instance_names=[row["instance_name"] for row in plan["instances"]])
    assert observation["cloud_mutated"] is False
    adapter.put_if_absent(object_name=f"{plan['result_prefix']}control/test.json", payload=b"{}")
    row = plan["instances"][0]
    control = f"{plan['result_prefix']}control/"
    spec = subject._instance_spec(
        plan,
        auth,
        row,
        source_object=f"{control}{subject.SOURCE_NAME}",
        wheelhouse_object=f"{control}{subject.WHEELHOUSE_NAME}",
        plan_object=f"{control}{subject.PLAN_NAME}",
        authorization_object=f"{control}launch_authorization.json",
    )
    spec["metadata"]["startup-script"] = (cloud_package / subject.STARTUP_NAME).read_text(
        encoding="utf-8"
    )
    spec["metadata"]["BUCKET"] = subject.WORKER_BUCKET
    adapter.create_instance(specification=spec)
    gce_bodies = [
        json.loads(body)
        for method, url, _headers, body, _timeout in calls
        if method == "POST" and "compute.googleapis.com" in url and body is not None
    ]
    assert len(gce_bodies) == 1
    assert gce_bodies[0]["disks"] == [
        {
            "boot": True,
            "autoDelete": True,
            "type": "PERSISTENT",
            "interface": "NVME",
            "initializeParams": {
                "sourceImage": plan["image"]["selfLink"],
                "diskSizeGb": "20",
                "diskType": f"zones/{plan['zone']}/diskTypes/hyperdisk-balanced",
            },
        }
    ]
    assert gce_bodies[0]["networkInterfaces"] == [
        {
            "network": "global/networks/default",
            "subnetwork": f"regions/{plan['region']}/subnetworks/default",
            "nicType": "GVNIC",
        }
    ]
    assert {method for method, *_ in calls} <= {"GET", "POST"}
    assert "fixture-secret-token" not in json.dumps(observation)
    receive = subject.GcpHttpTransport(project=plan["project"], zone=plan["zone"], bucket=subject.WORKER_BUCKET, mode="receive", execution_plan=plan, requester=requester)
    with pytest.raises(PermissionError, match="mutation"):
        receive.put_if_absent(object_name=f"{plan['result_prefix']}control/x", payload=b"x")


def test_stage_http_put_uses_bounded_long_timeout_and_rechecks_expiry(
    cloud_package: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(cloud_package)
    authorization = subject.build_content_stage_authorization(
        cloud_package_dir=cloud_package,
        explicit_stage_authorized=True,
        one_shot_nonce=NONCE,
        now_unix_seconds=OBSERVED_UNIX,
    )
    observed: list[tuple[str, int]] = []

    def requester(
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: int,
    ) -> subject.HttpResponse:
        observed.append((method, timeout))
        name = urllib_parse(url)["name"][0]
        return subject.HttpResponse(
            200, json.dumps({"name": name, "generation": "1"}).encode(), {}
        )

    now = [OBSERVED_UNIX]
    monkeypatch.setenv(
        subject.GcpHttpTransport.TOKEN_ENV, "fixture-secret-token-123456789"
    )
    adapter = subject.GcpHttpTransport(
        project=plan["project"],
        zone=plan["zone"],
        bucket=subject.WORKER_BUCKET,
        mode="stage",
        execution_plan=plan,
        authorization=authorization,
        raw_one_shot_nonce=NONCE,
        now_unix_seconds=OBSERVED_UNIX,
        requester=requester,
        clock=lambda: now[0],
    )
    object_name = f"{plan['content_staging']['object_prefix']}timeout-probe.bin"
    adapter.put_if_absent(object_name=object_name, payload=b"probe")
    assert observed == [("POST", 600)]
    now[0] = authorization["expires_unix_seconds"] + 1
    with pytest.raises(PermissionError, match="expired"):
        adapter.put_if_absent(object_name=object_name, payload=b"probe")
    assert observed == [("POST", 600)]


def test_gce_insert_operation_error_and_timeout_fail_closed(
    cloud_package: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(cloud_package)
    auth = _authorization(cloud_package)
    row = plan["instances"][0]
    control = f"{plan['result_prefix']}control/"
    spec = subject._instance_spec(
        plan,
        auth,
        row,
        source_object=f"{control}{subject.SOURCE_NAME}",
        wheelhouse_object=f"{control}{subject.WHEELHOUSE_NAME}",
        plan_object=f"{control}{subject.PLAN_NAME}",
        authorization_object=f"{control}launch_authorization.json",
    )
    spec["metadata"]["startup-script"] = (cloud_package / subject.STARTUP_NAME).read_text(
        encoding="utf-8"
    )
    spec["metadata"]["BUCKET"] = subject.WORKER_BUCKET
    target = (
        "https://www.googleapis.com/compute/v1/projects/"
        f"{plan['project']}/zones/{plan['zone']}/instances/{row['instance_name']}"
    )
    monkeypatch.setenv(
        subject.GcpHttpTransport.TOKEN_ENV, "fixture-secret-token-123456789"
    )

    def error_requester(
        method: str, url: str, headers: Mapping[str, str], body: bytes | None,
        timeout: int,
    ) -> subject.HttpResponse:
        assert method == "POST"
        return subject.HttpResponse(
            200,
            json.dumps(
                {
                    "name": "op-error",
                    "status": "DONE",
                    "operationType": "insert",
                    "targetLink": target,
                    "error": {"errors": [{"code": "INVALID_ARGUMENT"}]},
                }
            ).encode(),
            {},
        )

    error_adapter = subject.GcpHttpTransport(
        project=plan["project"], zone=plan["zone"], bucket=subject.WORKER_BUCKET,
        mode="launch", execution_plan=plan, authorization=auth,
        raw_one_shot_nonce=NONCE, now_unix_seconds=OBSERVED_UNIX,
        requester=error_requester, clock=lambda: OBSERVED_UNIX,
        sleep=lambda _seconds: None,
    )
    with pytest.raises(RuntimeError, match="operation identity"):
        error_adapter.create_instance(specification=spec)

    operation_gets = 0

    def pending_requester(
        method: str, url: str, headers: Mapping[str, str], body: bytes | None,
        timeout: int,
    ) -> subject.HttpResponse:
        nonlocal operation_gets
        if method == "GET":
            operation_gets += 1
        return subject.HttpResponse(
            200,
            json.dumps(
                {
                    "name": "op-timeout",
                    "status": "PENDING",
                    "operationType": "insert",
                    "targetLink": target,
                }
            ).encode(),
            {},
        )

    timeout_adapter = subject.GcpHttpTransport(
        project=plan["project"], zone=plan["zone"], bucket=subject.WORKER_BUCKET,
        mode="launch", execution_plan=plan, authorization=auth,
        raw_one_shot_nonce=NONCE, now_unix_seconds=OBSERVED_UNIX,
        requester=pending_requester, clock=lambda: OBSERVED_UNIX,
        sleep=lambda _seconds: None,
    )
    with pytest.raises(TimeoutError, match="did not complete"):
        timeout_adapter.create_instance(specification=spec)
    assert operation_gets == subject.GCE_ZONE_OPERATION_ATTEMPTS


def urllib_parse(url: str) -> dict[str, list[str]]:
    from urllib.parse import parse_qs, urlparse

    return parse_qs(urlparse(url).query)


def test_cli_launch_dry_run_constructs_no_transport_and_refuses_overwrite(cloud_package: Path, tmp_path: Path) -> None:
    staged, overlay, auth, _fake = _launch_material(cloud_package)
    auth_path = tmp_path / "auth.json"
    staged_path = tmp_path / "staged.json"
    overlay_path = tmp_path / "overlay.json"
    iam_path = tmp_path / "iam.json"
    auth_path.write_bytes(subject.canonical_bytes(auth))
    staged_path.write_bytes(subject.canonical_bytes(staged))
    overlay_path.write_bytes(subject.canonical_bytes(overlay))
    iam_path.write_bytes(subject.canonical_bytes(overlay["worker_iam_readback"]))
    output = tmp_path / "dry.json"

    def forbidden_factory(**_: Any) -> FakeCloud:
        raise AssertionError("dry-run must not construct a cloud transport")

    argv = ["launch", "--cloud-package", str(cloud_package), "--authorization", str(auth_path), "--content-stage-receipt", str(staged_path), "--launch-overlay", str(overlay_path), "--worker-iam-readback", str(iam_path), "--nonce", NONCE, "--bucket", subject.WORKER_BUCKET, "--authorize", "EXECUTE_T3_PERFDEV_V2_EXACT_TWO_VM_TAIL", "--output", str(output), "--now-unix-seconds", str(OBSERVED_UNIX), "--dry-run"]
    assert subject.main(argv, transport_factory=forbidden_factory) == 0
    assert json.loads(output.read_text())["cloud_mutation_count"] == 0
    with pytest.raises(FileExistsError):
        subject.main(argv, transport_factory=forbidden_factory)


def test_startup_is_fresh_shell_valid_offline_and_probes_feature_encoder() -> None:
    startup = subject.DEFAULT_STARTUP_PATH
    raw = startup.read_bytes()
    assert hashlib.sha256(raw).hexdigest() not in subject._FORBIDDEN_STARTUP_HASHES
    text = raw.decode("utf-8")
    assert "--no-index" in text
    assert "WHEELHOUSE_SHA256" in text
    assert "cp --no-preserve" in text
    assert "rust_direct_available" in text
    assert "82510e563ee290cd536e7aebe6bcf0efefac061af5488851f0a582bb4ef83411" in text
    assert "successful resume" in text
    assert 'if download_object "$object"' not in text
    assert text.count(
        "https://snapshot.debian.org/archive/debian/20260722T000000Z"
    ) == 1
    assert text.count(
        "https://snapshot.debian.org/archive/debian-security/20260722T000000Z"
    ) == 1
    assert "20260609T000000Z" not in text
    assert "shutdown -h" not in text
    assert "systemctl poweroff" not in text
    assert "force_shutdown" not in text
    assert "instanceTerminationAction=DELETE" in text
    assert "cloud._validate_tail_v2_runtime_artifacts" in text
    assert '"work_hand_indices": shard_manifest["work_hand_indices"]' in text
    assert "[2, 6, 7, 9, 13, 20, 21, 29, 33, 50]" not in text
    package_validation = text.index("local.validate_local_package")
    execution_plan_validation = text.index("cloud.validate_execution_plan")
    no_bytecode = text.index("export PYTHONDONTWRITEBYTECODE=1")
    feature_runtime_copy = text.index(
        'cp --no-preserve=mode,ownership,timestamps "$FEATURE_SOURCE" '
        '"$FEATURE_RUNTIME"'
    )
    assert no_bytecode < package_validation < execution_plan_validation < feature_runtime_copy
    assert 20260722 >= int(_image()["name"].rsplit("v", 1)[1])
    final_manifest_upload = text.index(
        'upload_once "$WORK/RESULT_MANIFEST.json"'
    )
    final_pump_check = text.index(
        "require_progress_pump_alive", final_manifest_upload
    )
    assert final_manifest_upload < final_pump_check < text.index(
        "VALIDATED_DONE=1", final_pump_check
    )
    if os.name == "nt":
        wsl = shutil.which("wsl.exe")
        if wsl is None:
            pytest.skip("WSL is unavailable for the Linux startup-script syntax check")
        drive, tail = os.path.splitdrive(str(startup.resolve()))
        tail_posix = tail.lstrip("\\/").replace("\\", "/")
        wsl_path = f"/mnt/{drive[0].lower()}/{tail_posix}"
        subprocess.run([wsl, "bash", "-n", wsl_path], check=True)
    else:
        bash = shutil.which("bash")
        if bash is None:
            pytest.skip("bash is unavailable for startup-script syntax check")
        subprocess.run([bash, "-n", str(startup)], check=True)


def test_progress_pump_two_passes_do_not_repeat_create_only_upload(
    tmp_path: Path,
) -> None:
    startup_text = subject.DEFAULT_STARTUP_PATH.read_text(encoding="utf-8")
    start = startup_text.index("upload_progress_checkpoint_once() {")
    end = startup_text.index("\nstop_background() {", start)
    functions = startup_text[start:end]
    probe = tmp_path / "progress_registry_probe.sh"
    probe.write_text(
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        + functions
        + r'''
sha() { sha256sum "$1" | awk '{print $1}'; }
upload_once() {
  printf '%s\t%s\n' "$2" "$(sha "$1")" >> "$UPLOAD_LOG"
}
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
OUTPUT="$WORK/output"
PROGRESS_UPLOAD_REGISTRY="$WORK/registry"
RESULT_PREFIX='fixture/run/'
SOURCE_ROLE='candidate'
UPLOAD_LOG="$WORK/upload.log"
mkdir -p "$OUTPUT/nested" "$PROGRESS_UPLOAD_REGISTRY"
printf '{"version":1}\n' > "$OUTPUT/nested/checkpoint.json"
progress_upload_pass
progress_upload_pass
[[ "$(wc -l < "$UPLOAD_LOG")" -eq 1 ]]
printf '{"version":2}\n' > "$OUTPUT/nested/checkpoint.json"
if progress_upload_pass; then
  echo 'changed published path was accepted' >&2
  exit 71
fi
[[ "$(wc -l < "$UPLOAD_LOG")" -eq 1 ]]
sleep 2 &
PUMP_PID=$!
require_progress_pump_alive
kill "$PUMP_PID"
wait "$PUMP_PID" 2>/dev/null || true
sleep 0.05 &
PUMP_PID=$!
sleep 0.1
if require_progress_pump_alive; then
  echo 'pump failure during final publish was accepted' >&2
  exit 72
fi
printf 'REGISTRY_AND_LIVENESS_PASS\n'
''',
        encoding="utf-8",
        newline="\n",
    )
    if os.name == "nt":
        wsl = shutil.which("wsl.exe")
        if wsl is None:
            pytest.skip("WSL is unavailable for startup pump executable test")
        drive, tail = os.path.splitdrive(str(probe.resolve()))
        tail_posix = tail.lstrip("\\/").replace("\\", "/")
        command = [wsl, "bash", f"/mnt/{drive[0].lower()}/{tail_posix}"]
    else:
        bash = shutil.which("bash")
        if bash is None:
            pytest.skip("bash is unavailable for startup pump executable test")
        command = [bash, str(probe)]
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=30
    )
    assert "REGISTRY_AND_LIVENESS_PASS" in completed.stdout


def _publish_control_prefix(
    package: Path,
    fake: FakeCloud,
    authorization: Mapping[str, Any],
    count: int,
) -> list[str]:
    plan = _plan(package)
    nonce_sha = hashlib.sha256(NONCE.encode("ascii")).hexdigest()
    result_control = f"{plan['result_prefix']}control/"
    content = plan["content_staging"]["object_prefix"]
    claim = {
        "schema": "hu_m31_t3_step6d_perfdev_v2_nonce_claim_v1",
        "run_name": plan["run_name"],
        "one_shot_nonce_sha256": nonce_sha,
        "execution_plan_sha256": subject.canonical_sha256(plan),
        "claimed_unix_seconds": OBSERVED_UNIX,
    }
    rows = [
        (
            f"{result_control}nonce-{nonce_sha}.json",
            subject.canonical_bytes(claim),
        ),
        (
            f"{result_control}{subject.SOURCE_NAME}",
            fake.objects[f"{content}{subject.SOURCE_NAME}"],
        ),
        (
            f"{result_control}{subject.WHEELHOUSE_NAME}",
            fake.objects[f"{content}{subject.WHEELHOUSE_NAME}"],
        ),
        (
            f"{result_control}{subject.WHEELHOUSE_MANIFEST_NAME}",
            fake.objects[f"{content}{subject.WHEELHOUSE_MANIFEST_NAME}"],
        ),
        (f"{result_control}{subject.PLAN_NAME}", subject.canonical_bytes(plan)),
        (
            f"{result_control}launch_authorization.json",
            subject.canonical_bytes(authorization),
        ),
        (
            f"{result_control}{subject.MANIFEST_NAME}",
            (package / subject.MANIFEST_NAME).read_bytes(),
        ),
    ]
    for name, raw in rows[:count]:
        fake.objects[name] = raw
    return [name for name, _raw in rows]


def test_zero_created_closeout_accepts_empty_pre_authorization_boundary(
    cloud_package: Path,
) -> None:
    plan = _plan(cloud_package)
    fake = FakeCloud(plan)
    readback = _worker_iam_readback(plan)
    receipt = subject.closeout_zero_created_launch_boundary(
        cloud_package_dir=cloud_package,
        worker_iam_readback=readback,
        raw_launch_nonce=NONCE,
        launch_authorization=None,
        raw_closeout_nonce=CLOSEOUT_NONCE,
        failure_stage="overlay_failed",
        controller_insert_attempt_count=0,
        explicit_operator_abort=True,
        transport=fake,
        sleep=lambda _seconds: None,
    )
    checked = subject._validate_zero_created_closeout_receipt(
        receipt, execution_plan=plan
    )
    assert checked["owned_instance_count"] == 0
    assert checked["all_planned_instances_absent"] is True
    assert checked["zero_created_readback_attempt_count"] == 12
    assert checked["prior_result_prefix_object_count"] == 0
    assert checked["closeout_cloud_mutation_performed"] is False
    assert checked["instance_delete_attempt_count"] == 0
    assert checked["controller_declared_insert_attempt_count"] == 0
    assert checked["insert_attempt_count_independently_verified"] is False
    assert checked["historical_zero_created_claimed"] is False
    assert checked["diagnostic_only"] is True
    assert checked["iam_cleanup_authorized"] is False
    assert fake.created == []
    assert fake.deleted == []


def test_zero_created_closeout_accepts_only_exact_partial_control_prefix(
    cloud_package: Path,
) -> None:
    plan = _plan(cloud_package)
    staged, _overlay, authorization, fake = _launch_material(cloud_package)
    assert staged["object_count"] == 3
    allowed = _publish_control_prefix(cloud_package, fake, authorization, 3)
    receipt = subject.closeout_zero_created_launch_boundary(
        cloud_package_dir=cloud_package,
        worker_iam_readback=_worker_iam_readback(plan),
        raw_launch_nonce=NONCE,
        launch_authorization=authorization,
        raw_closeout_nonce=CLOSEOUT_NONCE,
        failure_stage="control_publish_failed",
        controller_insert_attempt_count=0,
        explicit_operator_abort=True,
        transport=fake,
        sleep=lambda _seconds: None,
    )
    assert [row["object_name"] for row in receipt["prior_control_objects"]] == (
        allowed[:3]
    )
    assert receipt["prior_control_publish_prefix_length"] == 3
    subject._validate_zero_created_closeout_receipt(receipt, execution_plan=plan)

    fake.objects[f"{plan['result_prefix']}results/candidate/DONE.json"] = b"bad"
    with pytest.raises(ValueError, match="exact launch-control prefix"):
        subject.closeout_zero_created_launch_boundary(
            cloud_package_dir=cloud_package,
            worker_iam_readback=_worker_iam_readback(plan),
            raw_launch_nonce=NONCE,
            launch_authorization=authorization,
            raw_closeout_nonce=CLOSEOUT_NONCE,
            failure_stage="control_publish_failed",
            controller_insert_attempt_count=0,
            explicit_operator_abort=True,
            transport=fake,
            sleep=lambda _seconds: None,
        )


def test_zero_created_closeout_accepts_full_control_set_before_first_insert(
    cloud_package: Path,
) -> None:
    plan = _plan(cloud_package)
    _staged, _overlay, authorization, fake = _launch_material(cloud_package)
    allowed = _publish_control_prefix(cloud_package, fake, authorization, 7)
    receipt = subject.closeout_zero_created_launch_boundary(
        cloud_package_dir=cloud_package,
        worker_iam_readback=_worker_iam_readback(plan),
        raw_launch_nonce=NONCE,
        launch_authorization=authorization,
        raw_closeout_nonce=CLOSEOUT_NONCE,
        failure_stage="before_first_insert_aborted",
        controller_insert_attempt_count=0,
        explicit_operator_abort=True,
        transport=fake,
        sleep=lambda _seconds: None,
    )
    assert [row["object_name"] for row in receipt["prior_control_objects"]] == allowed
    assert receipt["prior_result_prefix_object_count"] == 7
    assert receipt["namespace_list_readback_count"] == 2
    assert receipt["namespace_stable_during_confirmation"] is True
    subject._validate_zero_created_closeout_receipt(receipt, execution_plan=plan)


def test_zero_created_closeout_rejects_any_measured_planned_instance(
    cloud_package: Path,
) -> None:
    plan = _plan(cloud_package)
    fake = FakeCloud(plan)
    row = plan["instances"][0]
    fake.instances[row["instance_name"]] = {
        "name": row["instance_name"],
        "ownership_label": row["ownership_label"],
        "execution_plan_sha256": subject.canonical_sha256(plan),
    }
    with pytest.raises(ValueError, match="zero-created closeout is forbidden"):
        subject.closeout_zero_created_launch_boundary(
            cloud_package_dir=cloud_package,
            worker_iam_readback=_worker_iam_readback(plan),
            raw_launch_nonce=NONCE,
            launch_authorization=None,
            raw_closeout_nonce=CLOSEOUT_NONCE,
            failure_stage="overlay_failed",
            controller_insert_attempt_count=0,
            explicit_operator_abort=True,
            transport=fake,
            sleep=lambda _seconds: None,
        )
    assert fake.deleted == []


def test_zero_created_closeout_rejects_namespace_race(
    cloud_package: Path,
) -> None:
    plan = _plan(cloud_package)

    class RacingNamespace(FakeCloud):
        list_count = 0

        def list_objects(self, *, prefix: str) -> Sequence[str]:
            self.list_count += 1
            if self.list_count == 2:
                self.objects[f"{prefix}results/candidate/DONE.json"] = b"late"
            return super().list_objects(prefix=prefix)

    fake = RacingNamespace(plan)
    with pytest.raises(ValueError, match="changed during confirmation"):
        subject.closeout_zero_created_launch_boundary(
            cloud_package_dir=cloud_package,
            worker_iam_readback=_worker_iam_readback(plan),
            raw_launch_nonce=NONCE,
            launch_authorization=None,
            raw_closeout_nonce=CLOSEOUT_NONCE,
            failure_stage="overlay_failed",
            controller_insert_attempt_count=0,
            explicit_operator_abort=True,
            transport=fake,
            sleep=lambda _seconds: None,
        )


def test_zero_created_closeout_stage_and_receipt_tamper_fail_closed(
    cloud_package: Path,
) -> None:
    plan = _plan(cloud_package)
    fake = FakeCloud(plan)
    with pytest.raises(ValueError, match="pre-authorization"):
        subject.closeout_zero_created_launch_boundary(
            cloud_package_dir=cloud_package,
            worker_iam_readback=_worker_iam_readback(plan),
            raw_launch_nonce=NONCE,
            launch_authorization=_authorization(cloud_package),
            raw_closeout_nonce=CLOSEOUT_NONCE,
            failure_stage="overlay_failed",
            controller_insert_attempt_count=0,
            explicit_operator_abort=True,
            transport=fake,
            sleep=lambda _seconds: None,
        )
    with pytest.raises(ValueError, match="zero declared inserts"):
        subject.closeout_zero_created_launch_boundary(
            cloud_package_dir=cloud_package,
            worker_iam_readback=_worker_iam_readback(plan),
            raw_launch_nonce=NONCE,
            launch_authorization=None,
            raw_closeout_nonce=CLOSEOUT_NONCE,
            failure_stage="overlay_failed",
            controller_insert_attempt_count=1,
            explicit_operator_abort=True,
            transport=fake,
            sleep=lambda _seconds: None,
        )
    receipt = subject.closeout_zero_created_launch_boundary(
        cloud_package_dir=cloud_package,
        worker_iam_readback=_worker_iam_readback(plan),
        raw_launch_nonce=NONCE,
        launch_authorization=None,
        raw_closeout_nonce=CLOSEOUT_NONCE,
        failure_stage="authorization_failed",
        controller_insert_attempt_count=0,
        explicit_operator_abort=True,
        transport=fake,
        sleep=lambda _seconds: None,
    )
    changed = deepcopy(receipt)
    changed["worker_iam_readback_receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="receipt changed"):
        subject._validate_zero_created_closeout_receipt(
            changed, execution_plan=plan
        )


def test_zero_created_real_adapter_is_get_only_and_delete_incapable(
    cloud_package: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(cloud_package)
    calls: list[tuple[str, str]] = []

    def requester(
        method: str, url: str, _headers: Mapping[str, str],
        _body: bytes | None, _timeout: int,
    ) -> subject.HttpResponse:
        calls.append((method, url))
        if "storage.googleapis.com/storage/v1" in url:
            return subject.HttpResponse(200, b"{}", {})
        if "/instances/" in url:
            return subject.HttpResponse(404, b"", {})
        raise AssertionError((method, url))

    monkeypatch.setenv(
        subject.GcpHttpTransport.TOKEN_ENV, "fixture-secret-token-123456789"
    )
    monkeypatch.setattr(subject, "ZERO_CREATED_CONFIRMATION_ATTEMPTS", 2)
    adapter = subject.GcpHttpTransport(
        project=plan["project"],
        zone=plan["zone"],
        bucket=subject.WORKER_BUCKET,
        mode="zero-created-closeout",
        execution_plan=plan,
        worker_iam_readback=_worker_iam_readback(plan),
        raw_one_shot_nonce=NONCE,
        requester=requester,
    )
    receipt = subject.closeout_zero_created_launch_boundary(
        cloud_package_dir=cloud_package,
        worker_iam_readback=_worker_iam_readback(plan),
        raw_launch_nonce=NONCE,
        launch_authorization=None,
        raw_closeout_nonce=CLOSEOUT_NONCE,
        failure_stage="overlay_failed",
        controller_insert_attempt_count=0,
        explicit_operator_abort=True,
        transport=adapter,
        sleep=lambda _seconds: None,
    )
    assert receipt["zero_created_readback_attempt_count"] == 2
    assert {method for method, _url in calls} == {"GET"}
    with pytest.raises(PermissionError, match="GCE delete"):
        adapter.delete_instance_exact(
            instance_name=plan["instances"][0]["instance_name"],
            ownership_label=plan["instances"][0]["ownership_label"],
            execution_plan_sha256=subject.canonical_sha256(plan),
        )


def test_zero_created_cli_requires_explicit_frozen_surface() -> None:
    args = subject._cli_parser().parse_args(
        [
            "closeout-zero-created",
            "--cloud-package", "cloud",
            "--worker-iam-readback", "iam.json",
            "--nonce", NONCE,
            "--closeout-nonce", CLOSEOUT_NONCE,
            "--failure-stage", "overlay_failed",
            "--controller-insert-attempt-count", "0",
            "--bucket", subject.WORKER_BUCKET,
            "--authorize",
            "OPERATOR_ABORT_RECORD_PREINSERT_DIAGNOSTIC_NO_IAM_CLEANUP",
            "--output", "zero-closeout.json",
        ]
    )
    assert args.command == "closeout-zero-created"
    assert args.launch_authorization is None
    assert args.failure_stage == "overlay_failed"
    assert args.controller_insert_attempt_count == 0
