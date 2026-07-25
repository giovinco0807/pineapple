"""Read-only runtime-infrastructure preflight for full100 wave v2.

The caller supplies provider JSON obtained immediately before launch.  This
module normalizes and seals the exact image, machine, VPC, subnet, Cloud NAT,
and bucket identities.  It contains no HTTP client and performs no mutation.
Passing this preflight is only one input to a later launch bundle; by itself it
never authorizes VM creation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_step6d_full100_wave_plan_v2 as wave


SCHEMA = "hu_m31_t3_step6d_full100_wave_runtime_preflight_receipt_v2"
STATUS = "exact_read_only_runtime_infrastructure_preflight_passed"

PROJECT = "ofc-solver-485418"
PROJECT_NUMBER = "783381566570"
REGION = "asia-northeast1"
ZONE = "asia-northeast1-b"
IMAGE_PROJECT = "debian-cloud"
MACHINE_TYPE = "c4-standard-16"
MACHINE_GUEST_CPUS = 16
MACHINE_MEMORY_MB = 61_440
ARCHITECTURE = "X86_64"
NETWORK_NAME = "default"
SUBNETWORK_NAME = "default"
NAT_ROUTER_NAME = "ofc-t3-nat-router-asia-northeast1"
NAT_NAME = "ofc-t3-nat-asia-northeast1"
BUCKET = "pokerhu-ofc-solver-485418-training"
BUCKET_LOCATION = "ASIA-NORTHEAST1"
PREFLIGHT_VALIDITY_SECONDS = 300

NETWORK_SELF_LINK = (
    f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
    f"global/networks/{NETWORK_NAME}"
)
SUBNETWORK_SELF_LINK = (
    f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/regions/"
    f"{REGION}/subnetworks/{SUBNETWORK_NAME}"
)
MACHINE_TYPE_SELF_LINK = (
    f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/zones/{ZONE}/"
    f"machineTypes/{MACHINE_TYPE}"
)
ROUTER_SELF_LINK = (
    f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/regions/{REGION}/"
    f"routers/{NAT_ROUTER_NAME}"
)
REGION_SELF_LINK = (
    f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/regions/{REGION}"
)
ZONE_SELF_LINK = (
    f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/zones/{ZONE}"
)

_SHA = re.compile(r"^[0-9a-f]{64}$")
_DIGITS = re.compile(r"^[1-9][0-9]*$")
_IMAGE_NAME = re.compile(r"^debian-12-bookworm-v[0-9]{8}$")
_FEATURE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_STORAGE_LOCATION = re.compile(
    r"^(?:asia|eu|us|[a-z]+-[a-z0-9]+[0-9])$"
)
_UTC_SECONDS = re.compile(
    r"^(?:19|20)[0-9]{2}-(?:0[1-9]|1[0-2])-"
    r"(?:0[1-9]|[12][0-9]|3[01])T(?:[01][0-9]|2[0-3]):"
    r"[0-5][0-9]:[0-5][0-9]Z$"
)

_RECEIPT_KEYS = frozenset(
    {
        "schema", "status", "wave_plan_sha256", "run_name",
        "execution_identity_sha256", "runtime_image_digest", "project",
        "project_number", "region", "zone", "observed_at_utc", "issued_at_utc",
        "expires_at_utc", "validity_seconds", "image", "machine_type", "network",
        "subnetwork", "cloud_nat", "bucket", "launch_network_design",
        "all_provider_identities_exact", "image_digest_matches_wave_plan",
        "no_external_ip_design", "read_only_observation", "standalone_launch_authorized",
        "cloud_mutated", "current_profile_changed", "receipt_sha256",
    }
)
_IMAGE_KEYS = frozenset(
    {
        "project", "name", "id", "self_link", "status", "deprecation",
        "architecture", "guest_os_features", "storage_locations",
        "image_identity_sha256",
    }
)
_DEPRECATION_KEYS = frozenset({"state", "replacement"})
_MACHINE_KEYS = frozenset(
    {
        "project", "zone", "name", "id", "self_link", "guest_cpus",
        "memory_mb", "architecture", "identity_sha256",
    }
)
_NETWORK_KEYS = frozenset(
    {"project", "name", "id", "self_link", "identity_sha256"}
)
_SUBNETWORK_KEYS = frozenset(
    {
        "project", "region", "name", "id", "self_link", "network_self_link",
        "identity_sha256",
    }
)
_NAT_KEYS = frozenset(
    {
        "project", "region", "router_name", "router_id", "router_self_link",
        "network_self_link", "nat_name", "nat_ip_allocate_option",
        "source_subnetwork_ip_ranges_to_nat", "nat_ips", "identity_sha256",
    }
)
_BUCKET_KEYS = frozenset(
    {
        "project", "project_number", "name", "id", "location", "location_type",
        "uniform_bucket_level_access", "identity_sha256",
    }
)
_LAUNCH_NETWORK_KEYS = frozenset(
    {
        "network_self_link", "subnetwork_self_link", "nic_type", "access_configs",
        "external_ipv4", "cloud_nat_required", "cloud_nat_router", "cloud_nat_name",
    }
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields changed")


def _provider_id(value: Any, label: str) -> str:
    rendered = str(value)
    if isinstance(value, bool) or _DIGITS.fullmatch(rendered) is None:
        raise ValueError(f"{label} is not a positive provider identity")
    return rendered


def _timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or _UTC_SECONDS.fullmatch(value) is None:
        raise ValueError(f"{label} must be canonical UTC seconds")
    parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    if parsed.tzinfo != timezone.utc:
        raise ValueError(f"{label} must be UTC")
    return parsed


def _render_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _one_of(
    value: Mapping[str, Any], names: Sequence[str], *, required: bool, label: str
) -> Any:
    found = [name for name in names if name in value]
    if len(found) > 1:
        raise ValueError(f"{label} aliases are ambiguous")
    if not found:
        if required:
            raise ValueError(f"{label} is missing")
        return None
    return value[found[0]]


def _sha_identity(value: Mapping[str, Any]) -> str:
    return canonical_sha256(value)


def _image_features(value: Mapping[str, Any]) -> list[str]:
    raw = _one_of(
        value,
        ("guestOsFeatures", "guest_os_features"),
        required=True,
        label="image guest features",
    )
    if not isinstance(raw, list):
        raise ValueError("image guest features are malformed")
    features: list[str] = []
    for item in raw:
        if isinstance(item, Mapping) and set(item) == {"type"}:
            feature = item.get("type")
        else:
            feature = item
        if not isinstance(feature, str) or _FEATURE.fullmatch(feature) is None:
            raise ValueError("image guest feature is malformed")
        features.append(feature)
    if features != sorted(set(features)):
        # Provider order is not identity-bearing; normalize it, but duplicate
        # entries are ambiguous and remain forbidden.
        if len(features) != len(set(features)):
            raise ValueError("image guest features are duplicated")
        features = sorted(features)
    if "GVNIC" not in features:
        raise ValueError("pinned Debian image does not support GVNIC")
    return features


def _image_deprecation(value: Mapping[str, Any]) -> dict[str, Any]:
    if "deprecation" in value and "deprecated" in value:
        raise ValueError("image deprecation aliases are ambiguous")
    if "deprecation" in value:
        raw = value["deprecation"]
    else:
        raw = value.get("deprecated")
    if raw is None:
        return {"state": "ACTIVE", "replacement": None}
    if not isinstance(raw, Mapping):
        raise ValueError("image deprecation observation is malformed")
    if not set(raw).issubset({"state", "replacement"}) or "state" not in raw:
        raise ValueError("image deprecation fields changed")
    normalized = {"state": raw.get("state"), "replacement": raw.get("replacement")}
    if normalized != {"state": "ACTIVE", "replacement": None}:
        raise ValueError("pinned Debian image is deprecated or replaced")
    return normalized


def normalize_image_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("image provider observation must be an object")
    name = value.get("name")
    image_id = _provider_id(value.get("id"), "image ID")
    self_link = _one_of(
        value, ("selfLink", "self_link"), required=True, label="image selfLink"
    )
    expected_link = (
        f"https://www.googleapis.com/compute/v1/projects/{IMAGE_PROJECT}/"
        f"global/images/{name}"
    )
    project = value.get("project", IMAGE_PROJECT)
    architecture = value.get("architecture")
    locations = _one_of(
        value,
        ("storageLocations", "storage_locations"),
        required=True,
        label="image storage locations",
    )
    if (
        not isinstance(locations, list)
        or not 1 <= len(locations) <= 128
        or any(
            not isinstance(location, str)
            or _STORAGE_LOCATION.fullmatch(location) is None
            for location in locations
        )
        or len(locations) != len(set(locations))
        or REGION not in locations
    ):
        raise ValueError("Debian image storage locations are unsafe or omit Tokyo")
    normalized_locations = sorted(locations)
    deprecation = _image_deprecation(value)
    features = _image_features(value)
    if (
        project != IMAGE_PROJECT
        or not isinstance(name, str)
        or _IMAGE_NAME.fullmatch(name) is None
        or value.get("family", "debian-12") != "debian-12"
        or self_link != expected_link
        or value.get("status") != "READY"
        or architecture != ARCHITECTURE
    ):
        raise ValueError("Debian image is not an exact pinned active runtime image")
    identity_body = {
        # This is the established Candidate02 image identity surface.  The
        # architecture and storage facts are fixed gates and are additionally
        # sealed by the containing receipt.
        "project": IMAGE_PROJECT,
        "name": name,
        "id": image_id,
        "selfLink": self_link,
        "status": "READY",
        "deprecation": deprecation,
        "guest_os_features": features,
    }
    return {
        "project": IMAGE_PROJECT,
        "name": name,
        "id": image_id,
        "self_link": self_link,
        "status": "READY",
        "deprecation": deprecation,
        "architecture": ARCHITECTURE,
        "guest_os_features": features,
        "storage_locations": normalized_locations,
        "image_identity_sha256": canonical_sha256(identity_body),
    }


def derive_image_identity_sha256(value: Mapping[str, Any]) -> str:
    return normalize_image_observation(value)["image_identity_sha256"]


def normalize_machine_type_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("machine provider observation must be an object")
    guest_cpus = _one_of(
        value, ("guestCpus", "guest_cpus"), required=True, label="machine guest CPUs"
    )
    memory_mb = _one_of(
        value, ("memoryMb", "memory_mb"), required=True, label="machine memory"
    )
    self_link = _one_of(
        value, ("selfLink", "self_link"), required=True, label="machine selfLink"
    )
    zone = value.get("zone")
    if zone == ZONE_SELF_LINK:
        zone = ZONE
    if (
        value.get("project", PROJECT) != PROJECT
        or zone != ZONE
        or value.get("name") != MACHINE_TYPE
        or guest_cpus != MACHINE_GUEST_CPUS
        or memory_mb != MACHINE_MEMORY_MB
        or value.get("architecture") != ARCHITECTURE
        or self_link != MACHINE_TYPE_SELF_LINK
        or value.get("deprecated") not in (None, {})
    ):
        raise ValueError("c4-standard-16 machine identity or shape changed")
    body = {
        "project": PROJECT,
        "zone": ZONE,
        "name": MACHINE_TYPE,
        "id": _provider_id(value.get("id"), "machine ID"),
        "self_link": MACHINE_TYPE_SELF_LINK,
        "guest_cpus": MACHINE_GUEST_CPUS,
        "memory_mb": MACHINE_MEMORY_MB,
        "architecture": ARCHITECTURE,
    }
    return {**body, "identity_sha256": _sha_identity(body)}


def normalize_network_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("network provider observation must be an object")
    self_link = _one_of(
        value, ("selfLink", "self_link"), required=True, label="network selfLink"
    )
    if (
        value.get("project", PROJECT) != PROJECT
        or value.get("name") != NETWORK_NAME
        or self_link != NETWORK_SELF_LINK
    ):
        raise ValueError("default network identity changed")
    body = {
        "project": PROJECT,
        "name": NETWORK_NAME,
        "id": _provider_id(value.get("id"), "network ID"),
        "self_link": NETWORK_SELF_LINK,
    }
    return {**body, "identity_sha256": _sha_identity(body)}


def normalize_subnetwork_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("subnetwork provider observation must be an object")
    self_link = _one_of(
        value, ("selfLink", "self_link"), required=True, label="subnetwork selfLink"
    )
    network_link = _one_of(
        value,
        ("network", "network_self_link"),
        required=True,
        label="subnetwork network",
    )
    region = value.get("region")
    if region == REGION_SELF_LINK:
        region = REGION
    if (
        value.get("project", PROJECT) != PROJECT
        or region != REGION
        or value.get("name") != SUBNETWORK_NAME
        or self_link != SUBNETWORK_SELF_LINK
        or network_link != NETWORK_SELF_LINK
    ):
        raise ValueError("default Tokyo subnetwork identity changed")
    body = {
        "project": PROJECT,
        "region": REGION,
        "name": SUBNETWORK_NAME,
        "id": _provider_id(value.get("id"), "subnetwork ID"),
        "self_link": SUBNETWORK_SELF_LINK,
        "network_self_link": NETWORK_SELF_LINK,
    }
    return {**body, "identity_sha256": _sha_identity(body)}


def normalize_cloud_nat_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Cloud NAT provider observation must be an object")
    router_link = _one_of(
        value, ("selfLink", "self_link"), required=True, label="router selfLink"
    )
    region = value.get("region")
    if region == REGION_SELF_LINK:
        region = REGION
    nats = value.get("nats")
    if not isinstance(nats, list) or len(nats) != 1 or not isinstance(nats[0], Mapping):
        raise ValueError("Cloud NAT router must expose exactly one target NAT")
    nat = nats[0]
    nat_ips = nat.get("natIps", nat.get("nat_ips", []))
    if (
        value.get("project", PROJECT) != PROJECT
        or region != REGION
        or value.get("name") != NAT_ROUTER_NAME
        or router_link != ROUTER_SELF_LINK
        or value.get("network") != NETWORK_SELF_LINK
        or nat.get("name") != NAT_NAME
        or nat.get("natIpAllocateOption", nat.get("nat_ip_allocate_option"))
        != "AUTO_ONLY"
        or nat.get(
            "sourceSubnetworkIpRangesToNat",
            nat.get("source_subnetwork_ip_ranges_to_nat"),
        )
        != "ALL_SUBNETWORKS_ALL_IP_RANGES"
        or nat_ips != []
    ):
        raise ValueError("Cloud NAT identity or routing contract changed")
    body = {
        "project": PROJECT,
        "region": REGION,
        "router_name": NAT_ROUTER_NAME,
        "router_id": _provider_id(value.get("id"), "router ID"),
        "router_self_link": ROUTER_SELF_LINK,
        "network_self_link": NETWORK_SELF_LINK,
        "nat_name": NAT_NAME,
        "nat_ip_allocate_option": "AUTO_ONLY",
        "source_subnetwork_ip_ranges_to_nat": "ALL_SUBNETWORKS_ALL_IP_RANGES",
        "nat_ips": [],
    }
    return {**body, "identity_sha256": _sha_identity(body)}


def normalize_bucket_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("bucket provider observation must be an object")
    project_number = _one_of(
        value,
        ("projectNumber", "project_number"),
        required=True,
        label="bucket project number",
    )
    iam = _one_of(
        value,
        ("iamConfiguration", "iam_configuration"),
        required=True,
        label="bucket IAM configuration",
    )
    if not isinstance(iam, Mapping):
        raise ValueError("bucket IAM configuration is malformed")
    ubla = _one_of(
        iam,
        ("uniformBucketLevelAccess", "uniform_bucket_level_access"),
        required=True,
        label="uniform bucket-level access",
    )
    if not isinstance(ubla, Mapping):
        raise ValueError("uniform bucket-level access observation is malformed")
    location_type = value.get("locationType", value.get("location_type"))
    if (
        value.get("project", PROJECT) != PROJECT
        or str(project_number) != PROJECT_NUMBER
        or value.get("name") != BUCKET
        or value.get("id", BUCKET) != BUCKET
        or value.get("location") != BUCKET_LOCATION
        or str(location_type).lower() != "region"
        or ubla.get("enabled") is not True
    ):
        raise ValueError("result bucket identity, location, or UBLA changed")
    body = {
        "project": PROJECT,
        "project_number": PROJECT_NUMBER,
        "name": BUCKET,
        "id": BUCKET,
        "location": BUCKET_LOCATION,
        "location_type": "region",
        "uniform_bucket_level_access": True,
    }
    return {**body, "identity_sha256": _sha_identity(body)}


def _launch_network_design() -> dict[str, Any]:
    return {
        "network_self_link": NETWORK_SELF_LINK,
        "subnetwork_self_link": SUBNETWORK_SELF_LINK,
        "nic_type": "GVNIC",
        "access_configs": [],
        "external_ipv4": False,
        "cloud_nat_required": True,
        "cloud_nat_router": NAT_ROUTER_NAME,
        "cloud_nat_name": NAT_NAME,
    }


def _validated_normalized(value: Mapping[str, Any], keys: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    checked = copy.deepcopy(dict(value))
    _exact_keys(checked, keys, label)
    return checked


def _validate_normalized_image(value: Mapping[str, Any]) -> dict[str, Any]:
    checked = _validated_normalized(value, _IMAGE_KEYS, "normalized image")
    deprecation = checked.get("deprecation")
    if not isinstance(deprecation, Mapping):
        raise ValueError("normalized image deprecation is malformed")
    _exact_keys(deprecation, _DEPRECATION_KEYS, "normalized image deprecation")
    raw = {
        "project": checked.get("project"),
        "name": checked.get("name"),
        "id": checked.get("id"),
        "self_link": checked.get("self_link"),
        "status": checked.get("status"),
        "deprecation": copy.deepcopy(dict(deprecation)),
        "architecture": checked.get("architecture"),
        "guest_os_features": checked.get("guest_os_features"),
        "storage_locations": checked.get("storage_locations"),
    }
    expected = normalize_image_observation(raw)
    if checked != expected:
        raise ValueError("normalized image identity changed")
    return checked


def _validate_identity_sha(value: Mapping[str, Any], keys: frozenset[str], label: str) -> dict[str, Any]:
    checked = _validated_normalized(value, keys, label)
    supplied = checked.get("identity_sha256")
    if not isinstance(supplied, str) or _SHA.fullmatch(supplied) is None:
        raise ValueError(f"{label} identity digest changed")
    body = {key: item for key, item in checked.items() if key != "identity_sha256"}
    if canonical_sha256(body) != supplied:
        raise ValueError(f"{label} identity digest changed")
    return checked


def build_runtime_preflight_receipt(
    *,
    wave_plan: Mapping[str, Any],
    image_observation: Mapping[str, Any],
    machine_type_observation: Mapping[str, Any],
    network_observation: Mapping[str, Any],
    subnetwork_observation: Mapping[str, Any],
    cloud_nat_observation: Mapping[str, Any],
    bucket_observation: Mapping[str, Any],
    observed_at_utc: str,
    current_utc: str,
) -> dict[str, Any]:
    plan = wave.validate_wave_plan(wave_plan)
    observed = _timestamp(observed_at_utc, "runtime observation time")
    current = _timestamp(current_utc, "runtime preflight current time")
    age = (current - observed).total_seconds()
    if age < 0:
        raise ValueError("runtime provider observation is from the future")
    if age > PREFLIGHT_VALIDITY_SECONDS:
        raise ValueError("runtime provider observation is stale")
    image = normalize_image_observation(image_observation)
    expected_digest = f"sha256:{image['image_identity_sha256']}"
    if plan["runtime_binding"]["image_digest"] != expected_digest:
        raise ValueError("wave plan image digest does not match pinned provider image")
    machine = normalize_machine_type_observation(machine_type_observation)
    network = normalize_network_observation(network_observation)
    subnetwork = normalize_subnetwork_observation(subnetwork_observation)
    cloud_nat = normalize_cloud_nat_observation(cloud_nat_observation)
    bucket = normalize_bucket_observation(bucket_observation)
    body = {
        "schema": SCHEMA,
        "status": STATUS,
        "wave_plan_sha256": plan["schedule_sha256"],
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "runtime_image_digest": expected_digest,
        "project": PROJECT,
        "project_number": PROJECT_NUMBER,
        "region": REGION,
        "zone": ZONE,
        "observed_at_utc": observed_at_utc,
        "issued_at_utc": current_utc,
        "expires_at_utc": _render_timestamp(
            current + timedelta(seconds=PREFLIGHT_VALIDITY_SECONDS)
        ),
        "validity_seconds": PREFLIGHT_VALIDITY_SECONDS,
        "image": image,
        "machine_type": machine,
        "network": network,
        "subnetwork": subnetwork,
        "cloud_nat": cloud_nat,
        "bucket": bucket,
        "launch_network_design": _launch_network_design(),
        "all_provider_identities_exact": True,
        "image_digest_matches_wave_plan": True,
        "no_external_ip_design": True,
        "read_only_observation": True,
        "standalone_launch_authorized": False,
        "cloud_mutated": False,
        "current_profile_changed": False,
    }
    return validate_runtime_preflight_receipt(
        wave_plan=plan,
        value={**body, "receipt_sha256": canonical_sha256(body)},
        current_utc=current_utc,
    )


def validate_runtime_preflight_receipt(
    *,
    wave_plan: Mapping[str, Any],
    value: Mapping[str, Any],
    current_utc: str,
) -> dict[str, Any]:
    plan = wave.validate_wave_plan(wave_plan)
    if not isinstance(value, Mapping):
        raise ValueError("runtime preflight receipt must be an object")
    checked = copy.deepcopy(dict(value))
    _exact_keys(checked, _RECEIPT_KEYS, "runtime preflight receipt")
    supplied = checked.pop("receipt_sha256", None)
    if (
        not isinstance(supplied, str)
        or _SHA.fullmatch(supplied) is None
        or supplied == "0" * 64
        or canonical_sha256(checked) != supplied
    ):
        raise ValueError("runtime preflight receipt digest changed")
    image = _validate_normalized_image(checked.get("image"))
    machine = _validate_identity_sha(
        checked.get("machine_type"), _MACHINE_KEYS, "normalized machine"
    )
    network = _validate_identity_sha(
        checked.get("network"), _NETWORK_KEYS, "normalized network"
    )
    subnetwork = _validate_identity_sha(
        checked.get("subnetwork"), _SUBNETWORK_KEYS, "normalized subnetwork"
    )
    cloud_nat = _validate_identity_sha(
        checked.get("cloud_nat"), _NAT_KEYS, "normalized Cloud NAT"
    )
    bucket = _validate_identity_sha(
        checked.get("bucket"), _BUCKET_KEYS, "normalized bucket"
    )
    design = _validated_normalized(
        checked.get("launch_network_design"),
        _LAUNCH_NETWORK_KEYS,
        "launch network design",
    )
    issued = _timestamp(checked.get("issued_at_utc"), "preflight issue time")
    observed = _timestamp(checked.get("observed_at_utc"), "preflight observation time")
    expires = _timestamp(checked.get("expires_at_utc"), "preflight expiry time")
    current = _timestamp(current_utc, "preflight validation current time")
    expected_digest = f"sha256:{image['image_identity_sha256']}"
    if (
        checked.get("schema") != SCHEMA
        or checked.get("status") != STATUS
        or checked.get("wave_plan_sha256") != plan["schedule_sha256"]
        or checked.get("run_name") != plan["run_name"]
        or checked.get("execution_identity_sha256")
        != plan["execution_identity_sha256"]
        or checked.get("runtime_image_digest") != expected_digest
        or plan["runtime_binding"]["image_digest"] != expected_digest
        or checked.get("project") != PROJECT
        or checked.get("project_number") != PROJECT_NUMBER
        or checked.get("region") != REGION
        or checked.get("zone") != ZONE
        or checked.get("validity_seconds") != PREFLIGHT_VALIDITY_SECONDS
        or expires != issued + timedelta(seconds=PREFLIGHT_VALIDITY_SECONDS)
        or observed > issued
        or (issued - observed).total_seconds() > PREFLIGHT_VALIDITY_SECONDS
        or current < issued
        or current >= expires
        or machine.get("project") != PROJECT
        or machine.get("zone") != ZONE
        or machine.get("name") != MACHINE_TYPE
        or machine.get("guest_cpus") != MACHINE_GUEST_CPUS
        or machine.get("memory_mb") != MACHINE_MEMORY_MB
        or machine.get("architecture") != ARCHITECTURE
        or machine.get("self_link") != MACHINE_TYPE_SELF_LINK
        or network.get("self_link") != NETWORK_SELF_LINK
        or subnetwork.get("self_link") != SUBNETWORK_SELF_LINK
        or subnetwork.get("network_self_link") != NETWORK_SELF_LINK
        or cloud_nat.get("router_name") != NAT_ROUTER_NAME
        or cloud_nat.get("nat_name") != NAT_NAME
        or cloud_nat.get("network_self_link") != NETWORK_SELF_LINK
        or cloud_nat.get("nat_ip_allocate_option") != "AUTO_ONLY"
        or cloud_nat.get("source_subnetwork_ip_ranges_to_nat")
        != "ALL_SUBNETWORKS_ALL_IP_RANGES"
        or bucket.get("name") != BUCKET
        or bucket.get("location") != BUCKET_LOCATION
        or bucket.get("uniform_bucket_level_access") is not True
        or design != _launch_network_design()
        or checked.get("all_provider_identities_exact") is not True
        or checked.get("image_digest_matches_wave_plan") is not True
        or checked.get("no_external_ip_design") is not True
        or checked.get("read_only_observation") is not True
        or checked.get("standalone_launch_authorized") is not False
        or checked.get("cloud_mutated") is not False
        or checked.get("current_profile_changed") is not False
    ):
        raise ValueError("runtime preflight receipt contract changed or expired")
    return {**checked, "receipt_sha256": supplied}


# Short aliases for controller code.
build_runtime_preflight = build_runtime_preflight_receipt
validate_runtime_preflight = validate_runtime_preflight_receipt


__all__ = [
    "ARCHITECTURE",
    "BUCKET",
    "BUCKET_LOCATION",
    "MACHINE_TYPE",
    "NAT_NAME",
    "NAT_ROUTER_NAME",
    "NETWORK_SELF_LINK",
    "PREFLIGHT_VALIDITY_SECONDS",
    "PROJECT",
    "PROJECT_NUMBER",
    "REGION",
    "SCHEMA",
    "SUBNETWORK_SELF_LINK",
    "ZONE",
    "build_runtime_preflight",
    "build_runtime_preflight_receipt",
    "canonical_bytes",
    "canonical_sha256",
    "derive_image_identity_sha256",
    "normalize_bucket_observation",
    "normalize_cloud_nat_observation",
    "normalize_image_observation",
    "normalize_machine_type_observation",
    "normalize_network_observation",
    "normalize_subnetwork_observation",
    "validate_runtime_preflight",
    "validate_runtime_preflight_receipt",
]
