from __future__ import annotations

import copy
from datetime import datetime, timezone

import pytest

from ofc_regular import hu_m31_t3_step6d_full100_wave_plan_v2 as wave
from ofc_regular import hu_m31_t3_step6d_full100_wave_runtime_preflight_v2 as subject


OBSERVED = "2027-01-20T10:00:00Z"
CURRENT = "2027-01-20T10:02:00Z"
LIVE_SHAPED_STORAGE_LOCATIONS = [
    "me-central2",
    "australia-southeast1",
    "europe-west9",
    "me-west1",
    "europe-central2",
    "us-east7",
    "asia-northeast1",
    "us-west8",
    "europe-west10",
    "southamerica-west1",
    "asia-northeast2",
    "europe-north2",
    "europe-west12",
    "eu",
    "asia-southeast1",
    "europe-west4",
    "us-central1",
    "us-west2",
    "europe-west3",
    "us-east4",
    "europe-west2",
    "us-west1",
    "us-south1",
    "us-west4",
    "us-west3",
    "europe-west6",
    "africa-south1",
    "asia-east1",
    "us-east1",
    "asia",
    "me-central1",
    "europe-west1",
    "asia-south2",
    "europe-north1",
    "asia-east2",
    "northamerica-northeast1",
    "asia-south1",
    "us",
    "asia-southeast2",
    "europe-west8",
    "asia-southeast3",
    "asia-northeast3",
    "southamerica-east1",
    "europe-west15",
    "northamerica-south1",
    "australia-southeast2",
    "us-central2",
    "europe-southwest1",
    "northamerica-northeast2",
    "us-east5",
]


def _image() -> dict:
    name = "debian-12-bookworm-v20260721"
    return {
        "kind": "compute#image",
        "id": "9021508813201755912",
        "name": name,
        "family": "debian-12",
        "status": "READY",
        "architecture": "X86_64",
        "guestOsFeatures": [
            {"type": "GVNIC"},
            {"type": "SEV_CAPABLE"},
            {"type": "SEV_LIVE_MIGRATABLE_V2"},
            {"type": "UEFI_COMPATIBLE"},
            {"type": "VIRTIO_SCSI_MULTIQUEUE"},
        ],
        # The live Debian image is replicated to many regions.  Provider order
        # is not stable and must not become identity-bearing.
        "storageLocations": list(LIVE_SHAPED_STORAGE_LOCATIONS),
        "selfLink": (
            "https://www.googleapis.com/compute/v1/projects/debian-cloud/"
            f"global/images/{name}"
        ),
    }


def _machine() -> dict:
    return {
        "kind": "compute#machineType",
        "id": "16001",
        "name": "c4-standard-16",
        "guestCpus": 16,
        "memoryMb": 61_440,
        "architecture": "X86_64",
        "zone": subject.ZONE_SELF_LINK,
        "selfLink": subject.MACHINE_TYPE_SELF_LINK,
    }


def _network() -> dict:
    return {
        "kind": "compute#network",
        "id": "10001",
        "name": "default",
        "selfLink": subject.NETWORK_SELF_LINK,
        "autoCreateSubnetworks": True,
    }


def _subnetwork() -> dict:
    return {
        "kind": "compute#subnetwork",
        "id": "10002",
        "name": "default",
        "region": subject.REGION_SELF_LINK,
        "network": subject.NETWORK_SELF_LINK,
        "selfLink": subject.SUBNETWORK_SELF_LINK,
        "stackType": "IPV4_ONLY",
    }


def _nat() -> dict:
    return {
        "kind": "compute#router",
        "id": "10003",
        "name": subject.NAT_ROUTER_NAME,
        "region": subject.REGION_SELF_LINK,
        "network": subject.NETWORK_SELF_LINK,
        "selfLink": subject.ROUTER_SELF_LINK,
        "nats": [
            {
                "name": subject.NAT_NAME,
                "natIpAllocateOption": "AUTO_ONLY",
                "sourceSubnetworkIpRangesToNat": (
                    "ALL_SUBNETWORKS_ALL_IP_RANGES"
                ),
            }
        ],
    }


def _bucket() -> dict:
    return {
        "kind": "storage#bucket",
        "id": subject.BUCKET,
        "name": subject.BUCKET,
        "projectNumber": subject.PROJECT_NUMBER,
        "location": "ASIA-NORTHEAST1",
        "locationType": "region",
        "storageClass": "STANDARD",
        "iamConfiguration": {
            "uniformBucketLevelAccess": {"enabled": True}
        },
    }


def _plan(image: dict | None = None, *, digest: str | None = None) -> dict:
    observed_image = _image() if image is None else image
    identity = (
        subject.derive_image_identity_sha256(observed_image)
        if digest is None
        else digest
    )
    return wave.build_wave_plan(
        run_name="regular-hu-m31-c02-f100wv2-runtimepf-001",
        identity_salt="1234567890abcdef1234567890abcdef",
        package_sha256="2" * 64,
        image_digest="sha256:" + identity,
    )


def _build(plan: dict | None = None, **overrides) -> dict:
    values = {
        "wave_plan": _plan() if plan is None else plan,
        "image_observation": _image(),
        "machine_type_observation": _machine(),
        "network_observation": _network(),
        "subnetwork_observation": _subnetwork(),
        "cloud_nat_observation": _nat(),
        "bucket_observation": _bucket(),
        "observed_at_utc": OBSERVED,
        "current_utc": CURRENT,
    }
    values.update(overrides)
    return subject.build_runtime_preflight_receipt(**values)


def _reseal(value: dict) -> None:
    value["receipt_sha256"] = subject.canonical_sha256(
        {key: item for key, item in value.items() if key != "receipt_sha256"}
    )


def test_pass_receipt_binds_exact_image_machine_network_nat_bucket_and_plan() -> None:
    plan = _plan()
    receipt = _build(plan)
    assert subject.validate_runtime_preflight_receipt(
        wave_plan=plan,
        value=receipt,
        current_utc="2027-01-20T10:04:59Z",
    ) == receipt
    assert receipt["wave_plan_sha256"] == plan["schedule_sha256"]
    assert receipt["execution_identity_sha256"] == plan["execution_identity_sha256"]
    assert receipt["runtime_image_digest"] == plan["runtime_binding"]["image_digest"]
    assert receipt["image"]["architecture"] == "X86_64"
    assert receipt["image"]["storage_locations"] == sorted(
        LIVE_SHAPED_STORAGE_LOCATIONS
    )
    assert receipt["image"]["guest_os_features"] == sorted(
        receipt["image"]["guest_os_features"]
    )
    assert "GVNIC" in receipt["image"]["guest_os_features"]
    assert receipt["machine_type"]["guest_cpus"] == 16
    assert receipt["machine_type"]["memory_mb"] == 61_440
    assert receipt["network"]["self_link"] == subject.NETWORK_SELF_LINK
    assert receipt["subnetwork"]["self_link"] == subject.SUBNETWORK_SELF_LINK
    assert receipt["cloud_nat"]["nat_ip_allocate_option"] == "AUTO_ONLY"
    assert receipt["cloud_nat"]["source_subnetwork_ip_ranges_to_nat"] == (
        "ALL_SUBNETWORKS_ALL_IP_RANGES"
    )
    assert receipt["bucket"]["uniform_bucket_level_access"] is True
    assert receipt["launch_network_design"]["access_configs"] == []
    assert receipt["launch_network_design"]["external_ipv4"] is False
    assert receipt["expires_at_utc"] == "2027-01-20T10:07:00Z"
    assert receipt["standalone_launch_authorized"] is False
    assert receipt["cloud_mutated"] is False
    assert receipt["current_profile_changed"] is False


def test_established_candidate02_image_identity_is_preserved() -> None:
    assert subject.derive_image_identity_sha256(_image()) == (
        "9dd85299f559ea3b143b1a764a9c69e0e535672036c2b45bf1cff25b88da3c0d"
    )


def test_live_shaped_multilocation_image_requires_unique_safe_tokyo_membership() -> None:
    observed = _image()
    normalized = subject.normalize_image_observation(observed)
    assert len(normalized["storage_locations"]) > 1
    assert normalized["storage_locations"] == sorted(
        LIVE_SHAPED_STORAGE_LOCATIONS
    )
    missing_tokyo = _image()
    missing_tokyo["storageLocations"] = ["us-central1", "europe-west1"]
    with pytest.raises(ValueError, match="omit Tokyo"):
        subject.normalize_image_observation(missing_tokyo)
    duplicated = _image()
    duplicated["storageLocations"].append("asia-northeast1")
    with pytest.raises(ValueError, match="unsafe"):
        subject.normalize_image_observation(duplicated)


@pytest.mark.parametrize(
    ("observed", "current", "message"),
    [
        ("2027-01-20T09:56:59Z", CURRENT, "stale"),
        ("2027-01-20T10:02:01Z", CURRENT, "future"),
    ],
)
def test_stale_or_future_provider_observation_is_rejected(
    observed: str, current: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _build(observed_at_utc=observed, current_utc=current)


def test_receipt_expires_exactly_five_minutes_after_issue() -> None:
    plan = _plan()
    receipt = _build(plan)
    with pytest.raises(ValueError, match="changed or expired"):
        subject.validate_runtime_preflight_receipt(
            wave_plan=plan,
            value=receipt,
            current_utc=receipt["expires_at_utc"],
        )


@pytest.mark.parametrize(
    ("field", "mutator", "message"),
    [
        (
            "image_observation",
            lambda value: value.update(
                name="debian-12",
                selfLink=(
                    "https://www.googleapis.com/compute/v1/projects/debian-cloud/"
                    "global/images/family/debian-12"
                ),
            ),
            "pinned",
        ),
        (
            "image_observation",
            lambda value: value.__setitem__("architecture", "ARM64"),
            "pinned",
        ),
        (
            "image_observation",
            lambda value: value.__setitem__("storageLocations", ["us-central1"]),
            "omit Tokyo",
        ),
        (
            "machine_type_observation",
            lambda value: value.__setitem__("guestCpus", 8),
            "machine",
        ),
        (
            "network_observation",
            lambda value: value.__setitem__(
                "selfLink", subject.NETWORK_SELF_LINK.replace("default", "other")
            ),
            "network",
        ),
        (
            "subnetwork_observation",
            lambda value: value.__setitem__(
                "selfLink", subject.SUBNETWORK_SELF_LINK.replace("default", "other")
            ),
            "subnetwork",
        ),
        (
            "cloud_nat_observation",
            lambda value: value["nats"][0].__setitem__(
                "natIpAllocateOption", "MANUAL_ONLY"
            ),
            "Cloud NAT",
        ),
        (
            "cloud_nat_observation",
            lambda value: value["nats"][0].__setitem__(
                "sourceSubnetworkIpRangesToNat", "LIST_OF_SUBNETWORKS"
            ),
            "Cloud NAT",
        ),
        (
            "bucket_observation",
            lambda value: value.__setitem__("location", "US-CENTRAL1"),
            "bucket",
        ),
        (
            "bucket_observation",
            lambda value: value["iamConfiguration"][
                "uniformBucketLevelAccess"
            ].__setitem__("enabled", False),
            "bucket",
        ),
    ],
)
def test_wrong_runtime_infrastructure_fails_closed(field, mutator, message: str) -> None:
    factories = {
        "image_observation": _image,
        "machine_type_observation": _machine,
        "network_observation": _network,
        "subnetwork_observation": _subnetwork,
        "cloud_nat_observation": _nat,
        "bucket_observation": _bucket,
    }
    value = factories[field]()
    mutator(value)
    with pytest.raises(ValueError, match=message):
        _build(**{field: value})


def test_wave_plan_image_digest_must_match_exact_provider_image() -> None:
    plan = _plan(digest="3" * 64)
    with pytest.raises(ValueError, match="digest does not match"):
        _build(plan)


def test_resealed_receipt_cannot_change_no_external_ip_or_fixed_shape() -> None:
    plan = _plan()
    receipt = _build(plan)
    forged = copy.deepcopy(receipt)
    forged["launch_network_design"]["external_ipv4"] = True
    _reseal(forged)
    with pytest.raises(ValueError, match="changed or expired"):
        subject.validate_runtime_preflight_receipt(
            wave_plan=plan,
            value=forged,
            current_utc=CURRENT,
        )
    forged = copy.deepcopy(receipt)
    forged["machine_type"]["guest_cpus"] = 8
    body = {
        key: item
        for key, item in forged["machine_type"].items()
        if key != "identity_sha256"
    }
    forged["machine_type"]["identity_sha256"] = subject.canonical_sha256(body)
    _reseal(forged)
    with pytest.raises(ValueError, match="changed or expired"):
        subject.validate_runtime_preflight_receipt(
            wave_plan=plan,
            value=forged,
            current_utc=CURRENT,
        )


def test_module_has_no_http_or_subprocess_execution_surface() -> None:
    import inspect

    source = inspect.getsource(subject)
    assert "import requests" not in source
    assert "import urllib" not in source
    assert "import subprocess" not in source
    assert "gcloud" not in source
    assert "GOOGLE_OAUTH_ACCESS_TOKEN" not in source
