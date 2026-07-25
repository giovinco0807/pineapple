"""Read-only image/runtime preflight for T3 performance-development v2.

The caller must supply exact observations.  This module performs no discovery,
subprocess execution, authentication, package construction, or cloud mutation.
Passing this preflight creates only a dry-run receipt and never authorizes a
launch.
"""

from __future__ import annotations

import json
import os
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

from . import hu_m31_t3_step6d_performance_development_v2_contract as contract_v1


IMAGE_OBSERVATION_SCHEMA = (
    "hu_m31_t3_step6d_performance_development_v2_image_observation_v2"
)
IMAGE_PREFLIGHT_SCHEMA = (
    "hu_m31_t3_step6d_performance_development_v2_image_preflight_v2"
)
RUNTIME_OBSERVATION_SCHEMA = (
    "hu_m31_t3_step6d_performance_development_v2_runtime_observation_v1"
)
RUNTIME_PREFLIGHT_SCHEMA = (
    "hu_m31_t3_step6d_performance_development_v2_runtime_preflight_v1"
)
DRY_RUN_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_performance_development_v2_dry_run_receipt_v1"
)

_IMAGE_KEYS = frozenset(
    {
        "schema",
        "observation_id",
        "observed_at_utc",
        "project",
        "name",
        "id",
        "selfLink",
        "status",
        "deprecation",
        "guest_os_features",
        "read_only",
    }
)
_DEPRECATION_KEYS = frozenset({"state", "replacement"})
_RUNTIME_KEYS = frozenset(
    {
        "schema",
        "observation_id",
        "observed_at_utc",
        "project",
        "region",
        "zone",
        "machine_type",
        "spot_price",
        "quota",
        "namespace",
        "read_only",
        "cloud_mutated",
    }
)
_MACHINE_KEYS = frozenset({"name", "guest_cpus", "memory_mb"})
_PRICE_KEYS = frozenset(
    {"machine_type", "provisioning_model", "currency", "unit", "value"}
)
_QUOTA_KEYS = frozenset({"c4_cpus", "spot_cpus"})
_ONE_QUOTA_KEYS = frozenset({"limit", "usage"})
_NAMESPACE_KEYS = frozenset(
    {
        "run_name",
        "identity_namespace",
        "result_prefix",
        "run_name_collision_count",
        "identity_collision_count",
        "result_prefix_collision_count",
        "inventory_read_only",
    }
)

_OBSERVATION_ID = re.compile(r"[a-z0-9][a-z0-9._-]{7,127}")
_RFC3339_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z")
_PROJECT = re.compile(r"[a-z][a-z0-9-]{4,62}[a-z0-9]")
_IMAGE_NAME = re.compile(r"debian-12-bookworm-v\d{8}")
_RUN_NAME = re.compile(r"regular-hu-m31-c02-perfdev-v2-[a-z0-9][a-z0-9-]{7,47}")
_IDENTITY = re.compile(r"perfdev-v2-[a-z0-9][a-z0-9-]{7,47}")
_FORBIDDEN_NAMESPACE_TERMS = (
    "performance-lock",
    "rearm2",
    "lock-r2",
    contract_v1.REARM2_RUN_ID,
    contract_v1.REARM2_PACKAGE_RUN_NAME,
)


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields changed")


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _decimal_string(value: Any, label: str) -> Decimal:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an exact decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{label} is invalid") from exc
    if not parsed.is_finite() or parsed <= 0 or format(parsed, "f") != value:
        raise ValueError(f"{label} is not a canonical positive decimal")
    return parsed


def _common_observation_fields(value: Mapping[str, Any]) -> None:
    observation_id = value.get("observation_id")
    observed_at = value.get("observed_at_utc")
    if (
        not isinstance(observation_id, str)
        or _OBSERVATION_ID.fullmatch(observation_id) is None
        or not isinstance(observed_at, str)
        or _RFC3339_UTC.fullmatch(observed_at) is None
        or value.get("read_only") is not True
    ):
        raise ValueError("read-only observation identity changed")


def validate_image_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    observation = dict(value)
    _exact_keys(observation, _IMAGE_KEYS, "replacement image observation")
    _common_observation_fields(observation)
    deprecation = observation.get("deprecation")
    if not isinstance(deprecation, Mapping):
        raise ValueError("replacement image deprecation observation is missing")
    _exact_keys(deprecation, _DEPRECATION_KEYS, "replacement image deprecation")

    project = observation.get("project")
    name = observation.get("name")
    image_id = observation.get("id")
    expected_self_link = (
        f"https://www.googleapis.com/compute/v1/projects/{project}/"
        f"global/images/{name}"
    )
    old = contract_v1.OLD_DEPRECATED_IMAGE
    guest_os_features = observation.get("guest_os_features")
    if (
        observation.get("schema") != IMAGE_OBSERVATION_SCHEMA
        or project != "debian-cloud"
        or _PROJECT.fullmatch(str(project)) is None
        or not isinstance(name, str)
        or _IMAGE_NAME.fullmatch(name) is None
        or not isinstance(image_id, str)
        or not image_id.isdigit()
        or observation.get("selfLink") != expected_self_link
        or observation.get("status") != "READY"
        or dict(deprecation) != {"state": "ACTIVE", "replacement": None}
        or not isinstance(guest_os_features, list)
        or any(
            not isinstance(feature, str)
            or not feature
            or feature != feature.upper()
            for feature in guest_os_features
        )
        or guest_os_features != sorted(set(guest_os_features))
        or "GVNIC" not in guest_os_features
        or name == old["name"]
        or image_id == old["id"]
        or observation.get("selfLink") == old["self_link"]
    ):
        raise ValueError("replacement image is not an exact active READY image")
    return observation


def build_image_preflight(value: Mapping[str, Any]) -> dict[str, Any]:
    observation = validate_image_observation(value)
    return {
        "schema": IMAGE_PREFLIGHT_SCHEMA,
        "status": "pass_read_only_replacement_image_exact_and_active",
        "observation": observation,
        "image_identity_sha256": contract_v1.canonical_sha256(
            {
                "project": observation["project"],
                "name": observation["name"],
                "id": observation["id"],
                "selfLink": observation["selfLink"],
                "status": observation["status"],
                "deprecation": observation["deprecation"],
                "guest_os_features": observation["guest_os_features"],
            }
        ),
        "default_or_substituted_image_used": False,
        "old_deprecated_image_reused": False,
        "cloud_mutated": False,
        "launch_authorized": False,
    }


def _quota_available(value: Mapping[str, Any], label: str) -> int:
    _exact_keys(value, _ONE_QUOTA_KEYS, label)
    limit = value.get("limit")
    usage = value.get("usage")
    if (
        not _is_plain_int(limit)
        or not _is_plain_int(usage)
        or limit < 0
        or usage < 0
        or usage > limit
    ):
        raise ValueError(f"{label} values changed")
    return limit - usage


def _validate_namespace(value: Mapping[str, Any]) -> dict[str, Any]:
    namespace = dict(value)
    _exact_keys(namespace, _NAMESPACE_KEYS, "fresh namespace observation")
    run_name = namespace.get("run_name")
    identity = namespace.get("identity_namespace")
    prefix = namespace.get("result_prefix")
    if (
        not isinstance(run_name, str)
        or _RUN_NAME.fullmatch(run_name) is None
        or not isinstance(identity, str)
        or _IDENTITY.fullmatch(identity) is None
        or not isinstance(prefix, str)
        or prefix != f"hu-m31-t3/perfdev-v2/{run_name}/"
        or not _is_plain_int(namespace.get("run_name_collision_count"))
        or namespace.get("run_name_collision_count") != 0
        or not _is_plain_int(namespace.get("identity_collision_count"))
        or namespace.get("identity_collision_count") != 0
        or not _is_plain_int(namespace.get("result_prefix_collision_count"))
        or namespace.get("result_prefix_collision_count") != 0
        or namespace.get("inventory_read_only") is not True
        or any(term in run_name for term in _FORBIDDEN_NAMESPACE_TERMS)
        or any(term in identity for term in _FORBIDDEN_NAMESPACE_TERMS)
        or any(term in prefix for term in _FORBIDDEN_NAMESPACE_TERMS)
    ):
        raise ValueError("performance-development v2 namespace is not fresh")
    return namespace


def validate_runtime_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    observation = dict(value)
    _exact_keys(observation, _RUNTIME_KEYS, "runtime observation")
    _common_observation_fields(observation)
    machine = observation.get("machine_type")
    price = observation.get("spot_price")
    quota = observation.get("quota")
    namespace = observation.get("namespace")
    if (
        not isinstance(machine, Mapping)
        or not isinstance(price, Mapping)
        or not isinstance(quota, Mapping)
        or not isinstance(namespace, Mapping)
    ):
        raise ValueError("runtime observation nested fields are missing")
    _exact_keys(machine, _MACHINE_KEYS, "machine observation")
    _exact_keys(price, _PRICE_KEYS, "Spot price observation")
    _exact_keys(quota, _QUOTA_KEYS, "quota observation")

    price_value = _decimal_string(price.get("value"), "Spot price")
    maximum = Decimal(contract_v1.MAX_SPOT_PRICE_USD_PER_VM_HOUR)
    c4 = quota.get("c4_cpus")
    spot = quota.get("spot_cpus")
    if not isinstance(c4, Mapping) or not isinstance(spot, Mapping):
        raise ValueError("quota observations are missing")
    c4_available = _quota_available(c4, "C4 CPU quota")
    spot_available = _quota_available(spot, "Spot CPU quota")
    if (
        observation.get("schema") != RUNTIME_OBSERVATION_SCHEMA
        or not isinstance(observation.get("project"), str)
        or _PROJECT.fullmatch(observation["project"]) is None
        or observation.get("region") != "asia-northeast1"
        or observation.get("zone") != "asia-northeast1-b"
        or dict(machine)
        != {
            "name": contract_v1.MACHINE_TYPE,
            "guest_cpus": contract_v1.GUEST_VCPUS,
            "memory_mb": 61_440,
        }
        or dict(price)
        != {
            "machine_type": contract_v1.MACHINE_TYPE,
            "provisioning_model": "SPOT",
            "currency": "USD",
            "unit": "vm_hour",
            "value": price["value"],
        }
        or price_value > maximum
        or c4_available < contract_v1.MIN_REQUIRED_C4_VCPUS
        or spot_available < contract_v1.MIN_REQUIRED_C4_VCPUS
        or observation.get("cloud_mutated") is not False
    ):
        raise ValueError("runtime price, quota, or exact 1x16 topology gate failed")
    observation["namespace"] = _validate_namespace(namespace)
    return observation


def build_runtime_preflight(value: Mapping[str, Any]) -> dict[str, Any]:
    observation = validate_runtime_observation(value)
    quota = observation["quota"]
    c4_available = quota["c4_cpus"]["limit"] - quota["c4_cpus"]["usage"]
    spot_available = (
        quota["spot_cpus"]["limit"] - quota["spot_cpus"]["usage"]
    )
    return {
        "schema": RUNTIME_PREFLIGHT_SCHEMA,
        "status": "pass_read_only_price_quota_topology_and_namespace",
        "observation": observation,
        "validated_topology": {
            "machine_type": contract_v1.MACHINE_TYPE,
            "source_instance_count": 2,
            "workers_per_source_process": contract_v1.WORKERS_PER_SOURCE,
            "rayon_threads_per_worker": contract_v1.RAYON_THREADS_PER_WORKER,
            "guest_vcpus_per_instance": contract_v1.GUEST_VCPUS,
            "execution_mode": "one_source_one_process_scalar_1x16",
        },
        "spot_price_usd_per_vm_hour": observation["spot_price"]["value"],
        "spot_price_ceiling": contract_v1.MAX_SPOT_PRICE_USD_PER_VM_HOUR,
        "available_c4_vcpus": c4_available,
        "available_spot_vcpus": spot_available,
        "minimum_required_vcpus": contract_v1.MIN_REQUIRED_C4_VCPUS,
        "rearm2_namespace_or_seed_reused": False,
        "cloud_mutated": False,
        "launch_authorized": False,
    }


def build_dry_run_receipt(
    *,
    image_observation: Mapping[str, Any],
    runtime_observation: Mapping[str, Any],
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    frozen_contract = (
        contract_v1.build_contract()
        if contract is None
        else contract_v1.validate_contract(contract)
    )
    image = build_image_preflight(image_observation)
    runtime = build_runtime_preflight(runtime_observation)
    value = {
        "schema": DRY_RUN_RECEIPT_SCHEMA,
        "status": "local_dry_run_pass_cloud_execution_still_forbidden",
        "contract": frozen_contract,
        "contract_sha256": contract_v1.canonical_sha256(frozen_contract),
        "image_preflight": image,
        "runtime_preflight": runtime,
        "image_preflight_sha256": contract_v1.canonical_sha256(image),
        "runtime_preflight_sha256": contract_v1.canonical_sha256(runtime),
        "rearm2_resources_reused": False,
        "root_content_read": False,
        "package_built": False,
        "gcloud_invoked": False,
        "cloud_executable": False,
        "launch_authorized": False,
        "cloud_mutated": False,
        "instances_created": False,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
        "runtime_policy_activated": False,
        "m31_complete": False,
    }
    return validate_dry_run_receipt(value)


def validate_dry_run_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    frozen_contract = contract_v1.validate_contract(payload.get("contract", {}))
    image_record = payload.get("image_preflight")
    runtime_record = payload.get("runtime_preflight")
    if not isinstance(image_record, Mapping) or not isinstance(
        runtime_record, Mapping
    ):
        raise ValueError("dry-run preflight records are missing")
    expected_image = build_image_preflight(image_record.get("observation", {}))
    expected_runtime = build_runtime_preflight(runtime_record.get("observation", {}))
    expected = build_dry_run_receipt_unchecked(
        contract=frozen_contract,
        image_preflight=expected_image,
        runtime_preflight=expected_runtime,
    )
    if payload != expected:
        raise ValueError("performance-development v2 dry-run receipt changed")
    return payload


def build_dry_run_receipt_unchecked(
    *,
    contract: Mapping[str, Any],
    image_preflight: Mapping[str, Any],
    runtime_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble an already-validated receipt without recursive validation."""

    return {
        "schema": DRY_RUN_RECEIPT_SCHEMA,
        "status": "local_dry_run_pass_cloud_execution_still_forbidden",
        "contract": dict(contract),
        "contract_sha256": contract_v1.canonical_sha256(contract),
        "image_preflight": dict(image_preflight),
        "runtime_preflight": dict(runtime_preflight),
        "image_preflight_sha256": contract_v1.canonical_sha256(image_preflight),
        "runtime_preflight_sha256": contract_v1.canonical_sha256(
            runtime_preflight
        ),
        "rearm2_resources_reused": False,
        "root_content_read": False,
        "package_built": False,
        "gcloud_invoked": False,
        "cloud_executable": False,
        "launch_authorized": False,
        "cloud_mutated": False,
        "instances_created": False,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
        "runtime_policy_activated": False,
        "m31_complete": False,
    }


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    target = Path(path).resolve()
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def build_dry_run_receipt_from_files(
    *, image_path: str | Path, runtime_path: str | Path
) -> dict[str, Any]:
    return build_dry_run_receipt(
        image_observation=_read_json(image_path, "image observation"),
        runtime_observation=_read_json(runtime_path, "runtime observation"),
    )


def write_once(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite dry-run receipt: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(contract_v1.canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, target)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite dry-run receipt: {target}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "DRY_RUN_RECEIPT_SCHEMA",
    "IMAGE_OBSERVATION_SCHEMA",
    "IMAGE_PREFLIGHT_SCHEMA",
    "RUNTIME_OBSERVATION_SCHEMA",
    "RUNTIME_PREFLIGHT_SCHEMA",
    "build_dry_run_receipt",
    "build_dry_run_receipt_from_files",
    "build_image_preflight",
    "build_runtime_preflight",
    "validate_dry_run_receipt",
    "validate_image_observation",
    "validate_runtime_observation",
    "write_once",
]
