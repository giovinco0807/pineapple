from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_performance_development_v2_contract as contract_v1,
)
from ofc_regular import (
    hu_m31_t3_step6d_performance_development_v2_preflight as subject,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _image() -> dict:
    name = "debian-12-bookworm-v20260715"
    return {
        "schema": subject.IMAGE_OBSERVATION_SCHEMA,
        "observation_id": "image-ro-20260722-001",
        "observed_at_utc": "2026-07-22T12:10:00Z",
        "project": "debian-cloud",
        "name": name,
        "id": "9876543210123456789",
        "selfLink": (
            "https://www.googleapis.com/compute/v1/projects/debian-cloud/"
            f"global/images/{name}"
        ),
        "status": "READY",
        "deprecation": {"state": "ACTIVE", "replacement": None},
        "guest_os_features": ["GVNIC", "UEFI_COMPATIBLE", "VIRTIO_SCSI_MULTIQUEUE"],
        "read_only": True,
    }


def _runtime() -> dict:
    run_name = "regular-hu-m31-c02-perfdev-v2-20260722-001"
    return {
        "schema": subject.RUNTIME_OBSERVATION_SCHEMA,
        "observation_id": "runtime-ro-20260722-001",
        "observed_at_utc": "2026-07-22T12:11:00Z",
        "project": "pokerhu-ofc-solver-485418",
        "region": "asia-northeast1",
        "zone": "asia-northeast1-b",
        "machine_type": {
            "name": "c4-standard-16",
            "guest_cpus": 16,
            "memory_mb": 61_440,
        },
        "spot_price": {
            "machine_type": "c4-standard-16",
            "provisioning_model": "SPOT",
            "currency": "USD",
            "unit": "vm_hour",
            "value": "0.45638",
        },
        "quota": {
            "c4_cpus": {"limit": 64, "usage": 16},
            "spot_cpus": {"limit": 64, "usage": 0},
        },
        "namespace": {
            "run_name": run_name,
            "identity_namespace": "perfdev-v2-20260722-001",
            "result_prefix": f"hu-m31-t3/perfdev-v2/{run_name}/",
            "run_name_collision_count": 0,
            "identity_collision_count": 0,
            "result_prefix_collision_count": 0,
            "inventory_read_only": True,
        },
        "read_only": True,
        "cloud_mutated": False,
    }


def test_preflight_freezes_exact_image_price_quota_topology_and_namespace() -> None:
    receipt = subject.build_dry_run_receipt(
        image_observation=_image(), runtime_observation=_runtime()
    )

    assert receipt["schema"] == subject.DRY_RUN_RECEIPT_SCHEMA
    assert receipt["contract_sha256"] == contract_v1.canonical_sha256(
        receipt["contract"]
    )
    assert receipt["image_preflight"]["observation"] == _image()
    runtime = receipt["runtime_preflight"]
    assert runtime["spot_price_usd_per_vm_hour"] == "0.45638"
    assert runtime["available_c4_vcpus"] == 48
    assert runtime["available_spot_vcpus"] == 64
    assert runtime["validated_topology"]["execution_mode"] == (
        "one_source_one_process_scalar_1x16"
    )
    assert receipt["rearm2_resources_reused"] is False
    assert receipt["gcloud_invoked"] is False
    assert receipt["cloud_executable"] is False
    assert receipt["launch_authorized"] is False
    assert receipt["cloud_mutated"] is False
    assert receipt["instances_created"] is False
    assert receipt["current_profile_changed"] is False


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda value: value["spot_price"].__setitem__("value", "0.57001"), "gate"),
        (lambda value: value["quota"]["c4_cpus"].__setitem__("usage", 33), "gate"),
        (
            lambda value: value["machine_type"].__setitem__("guest_cpus", 8),
            "gate",
        ),
        (
            lambda value: value["namespace"].__setitem__(
                "run_name_collision_count", 1
            ),
            "not fresh",
        ),
        (
            lambda value: value["namespace"].__setitem__(
                "identity_collision_count", False
            ),
            "not fresh",
        ),
        (
            lambda value: value["namespace"].__setitem__(
                "run_name", contract_v1.REARM2_PACKAGE_RUN_NAME
            ),
            "not fresh",
        ),
    ],
)
def test_runtime_gate_fails_closed(mutator, message: str) -> None:
    value = _runtime()
    mutator(value)
    with pytest.raises(ValueError, match=message):
        subject.build_runtime_preflight(value)


def test_old_deprecated_or_implicitly_substituted_image_is_rejected() -> None:
    old = _image()
    old.update(
        {
            "name": contract_v1.OLD_DEPRECATED_IMAGE["name"],
            "id": contract_v1.OLD_DEPRECATED_IMAGE["id"],
            "selfLink": contract_v1.OLD_DEPRECATED_IMAGE["self_link"],
            "deprecation": {
                "state": "DEPRECATED",
                "replacement": _image()["selfLink"],
            },
        }
    )
    with pytest.raises(ValueError, match="active READY"):
        subject.build_image_preflight(old)
    unknown = _image()
    unknown.pop("deprecation")
    with pytest.raises(ValueError, match="fields changed"):
        subject.build_image_preflight(unknown)


def test_image_guest_features_require_canonical_gvnic() -> None:
    missing = _image()
    missing["guest_os_features"] = ["UEFI_COMPATIBLE"]
    with pytest.raises(ValueError, match="active READY image"):
        subject.build_image_preflight(missing)
    unordered = _image()
    unordered["guest_os_features"] = ["UEFI_COMPATIBLE", "GVNIC"]
    with pytest.raises(ValueError, match="active READY image"):
        subject.build_image_preflight(unordered)

def test_receipt_tamper_fails_closed() -> None:
    receipt = subject.build_dry_run_receipt(
        image_observation=_image(), runtime_observation=_runtime()
    )
    tampered = deepcopy(receipt)
    tampered["launch_authorized"] = True
    with pytest.raises(ValueError, match="receipt changed"):
        subject.validate_dry_run_receipt(tampered)


def test_dry_run_entrypoint_writes_once_without_cloud_execution(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.json"
    runtime_path = tmp_path / "runtime.json"
    output_path = tmp_path / "receipt.json"
    image_path.write_text(json.dumps(_image()), encoding="utf-8")
    runtime_path.write_text(json.dumps(_runtime()), encoding="utf-8")
    command = [
        sys.executable,
        str(
            REPO_ROOT
            / "scripts/prepare_hu_m31_t3_step6d_performance_development_v2.py"
        ),
        "--image-observation",
        str(image_path),
        "--runtime-observation",
        str(runtime_path),
        "--output",
        str(output_path),
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    summary = json.loads(completed.stdout)
    receipt = json.loads(output_path.read_text(encoding="utf-8"))

    assert summary["output_written"] is True
    assert summary["receipt_sha256"] == contract_v1.canonical_sha256(receipt)
    assert summary["cloud_executable"] is False
    assert summary["launch_authorized"] is False
    assert summary["cloud_mutated"] is False
    assert summary["instances_created"] is False

    repeated = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert repeated.returncode != 0
    assert "refusing to overwrite dry-run receipt" in repeated.stderr


def test_entrypoint_does_not_import_a_cloud_or_process_adapter() -> None:
    source = (
        REPO_ROOT
        / "scripts/prepare_hu_m31_t3_step6d_performance_development_v2.py"
    ).read_text(encoding="utf-8")
    assert "subprocess" not in source
    assert "google.cloud" not in source
    assert "live_cloud" not in source
