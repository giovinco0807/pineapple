"""Offline freeze and validator for the Step 11 one-VM lifecycle launch.

This module cannot provision objects, sign controller records, mutate IAM,
create a VM, release a claim, or receive results.  It freezes the exact
Compute Engine insert template and the two-phase metadata contract which a
separately authorized controller must satisfy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as transport,
)
from . import hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as plan
from . import hu_m31_t3_step6d_rearm2_diagnostic_canary_vm_adapter as legacy_vm


SCHEMA = "hu_m31_t3_step6d_rearm2_diagnostic_step11_launch_contract_v1"
STATUS = "offline_frozen_launch_contract_controller_authorization_required"

PROJECT = transport.PROJECT
ZONE = transport.ZONE
REGION = "asia-northeast1"
MACHINE_TYPE = transport.MACHINE_TYPE
WORKER_SERVICE_ACCOUNT = transport.WORKER_SERVICE_ACCOUNT
OAUTH_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

NETWORK_SELF_LINK = (
    f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/global/"
    "networks/default"
)
SUBNETWORK_SELF_LINK = (
    f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/regions/"
    f"{REGION}/subnetworks/default"
)
NAT_ROUTER_SELF_LINK = (
    f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/regions/"
    f"{REGION}/routers/ofc-t3-nat-router-asia-northeast1"
)
NAT_ROUTER_NAME = "ofc-t3-nat-router-asia-northeast1"
NAT_NAME = "ofc-t3-nat-asia-northeast1"

MAX_RUNTIME_SECONDS = legacy_vm.MAX_RUNTIME_SECONDS_PER_VM
FAILURE_SHUTDOWN_SECONDS = transport.MAX_FAILURE_SHUTDOWN_SECONDS

INITIAL_METADATA_FROM_FILE_KEYS = (
    "startup-script",
    "ofc-step11-transport-contract",
    "ofc-step11-controller-authorization",
    "ofc-step11-controller-public-key",
    "ofc-step11-prebootstrap",
)
POSTCREATE_CLAIM_METADATA_KEY = "ofc-step11-controller-claim"
STARTUP_RELATIVE = (
    "scripts/"
    "bootstrap_hu_m31_t3_step6d_rearm2_diagnostic_step11_v1.sh"
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STARTUP = _REPO_ROOT / STARTUP_RELATIVE
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _read_canonical(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    raw = source.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not JSON") from error
    if (
        not isinstance(value, dict)
        or canonical_bytes(value) != raw
    ):
        raise ValueError(f"{label} must use exact canonical JSON bytes")
    return value


def _file_record(
    path: str | Path,
    *,
    expected_suffix: str,
    kind: str,
    maximum_bytes: int = 262_144,
) -> dict[str, Any]:
    source = Path(path)
    if (
        not source.is_file()
        or source.is_symlink()
        or source.suffix != expected_suffix
    ):
        raise ValueError(f"{kind} must be a regular {expected_suffix} file")
    size = source.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise ValueError(f"{kind} size escaped metadata-from-file limit")
    return {
        "kind": kind,
        "sha256": sha256_file(source),
        "bytes": size,
        "metadata_value_max_bytes": maximum_bytes,
    }


def _transport_stage1_candidate(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    checked = transport.validate_job_contract(value)
    preview = checked["adapter_preview"]
    binding = checked["metadata_binding"]
    if (
        preview["stage_id"] != plan.STAGE1_ID
        or preview["run_name"] != plan.STAGE1_RUN_NAME
        or preview["selected_job_ids"] != list(plan.STAGE1_JOB_IDS)
        or preview["vm_count"] != 1
        or preview["attempt_index"] != 0
        or binding["stage_id"] != plan.STAGE1_ID
        or binding["job_id"] != plan.STAGE1_JOB_IDS[0]
        or binding["source_role"] != "candidate"
        or binding["attempt_index"] != 0
    ):
        raise ValueError("Step 11 accepts exactly stage1 candidate attempt 0")
    if (
        binding["project"] != PROJECT
        or binding["zone"] != ZONE
        or binding["worker_service_account"] != WORKER_SERVICE_ACCOUNT
        or binding["image"]
        != {
            "project": transport.IMAGE_PROJECT,
            "name": transport.IMAGE_NAME,
            "id": transport.IMAGE_ID,
            "self_link": transport.IMAGE_SELF_LINK,
            "family_resolution_permitted": False,
        }
    ):
        raise ValueError("Step 11 transport target changed")
    cost = preview["cost_guard"]
    if (
        cost["vm_count"] != 1
        or cost["max_runtime_seconds_per_vm"] != MAX_RUNTIME_SECONDS
        or cost["max_attempts_per_job"] != transport.MAX_ATTEMPTS
        or cost["diagnostic_compute_cap_usd"]
        != legacy_vm.DIAGNOSTIC_COMPUTE_CAP_USD
        or cost["spot_price_ceiling_usd_per_vm_hour"]
        != legacy_vm.SPOT_PRICE_CEILING_USD_PER_VM_HOUR
    ):
        raise ValueError("Step 11 cost guard changed")
    return checked


def _public_key_binding(
    *,
    direct_contract: Mapping[str, Any],
    public_key_record: Mapping[str, Any],
) -> dict[str, Any]:
    public_key = transport.validate_rsa_public_key_record(public_key_record)
    public_key_sha = canonical_sha256(public_key)
    trust = direct_contract["authorization_contract"]
    if (
        trust["controller_public_key_sha256"] != public_key_sha
        or trust["controller_key_id"] != public_key["key_id"]
        or trust["signature_algorithm"] != transport.RSA_SIGNATURE_ALGORITHM
        or trust["controller_private_key_embedded"] is not False
        or trust["worker_shared_signing_secret_present"] is not False
    ):
        raise ValueError("Step 11 public key is not pinned by transport contract")
    return {
        "schema": transport.RSA_PUBLIC_KEY_SCHEMA,
        "key_id": public_key["key_id"],
        "public_key_sha256": public_key_sha,
        "signature_algorithm": transport.RSA_SIGNATURE_ALGORITHM,
        "private_key_location": "controller_only_outside_workspace_vm_and_gcs",
        "private_key_embedded": False,
        "shared_secret_present": False,
    }


def _instance_insert_static_body(
    direct_contract: Mapping[str, Any],
) -> dict[str, Any]:
    binding = direct_contract["metadata_binding"]
    metadata_items = [
        {"key": key, "value": value}
        for key, value in sorted(direct_contract["metadata_values"].items())
    ]
    metadata_items.extend(
        [
            {"key": "block-project-ssh-keys", "value": "true"},
            {
                "key": "ofc-step11-claim-release-state",
                "value": "pending-post-create",
            },
        ]
    )
    return {
        "name": binding["instance_name"],
        "machineType": (
            f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/zones/"
            f"{ZONE}/machineTypes/{MACHINE_TYPE}"
        ),
        "canIpForward": False,
        "deletionProtection": False,
        "disks": [
            {
                "boot": True,
                "autoDelete": True,
                "type": "PERSISTENT",
                "initializeParams": {
                    "sourceImage": transport.IMAGE_SELF_LINK,
                },
            }
        ],
        "networkInterfaces": [
            {
                "network": NETWORK_SELF_LINK,
                "subnetwork": SUBNETWORK_SELF_LINK,
                "accessConfigs": [],
            }
        ],
        "scheduling": {
            "provisioningModel": "SPOT",
            "instanceTerminationAction": "DELETE",
            "automaticRestart": False,
            "onHostMaintenance": "TERMINATE",
            "maxRunDuration": {
                "seconds": str(MAX_RUNTIME_SECONDS),
                "nanos": 0,
            },
        },
        "serviceAccounts": [
            {
                "email": WORKER_SERVICE_ACCOUNT,
                "scopes": [OAUTH_SCOPE],
            }
        ],
        "metadata": {
            "scalarItems": metadata_items,
            "metadataFromFileKeys": list(INITIAL_METADATA_FROM_FILE_KEYS),
            "forbiddenAtInsert": [POSTCREATE_CLAIM_METADATA_KEY],
        },
        "labels": {
            "ofc-workload": "m31-t3-step11",
            "ofc-stage": "lifecycle-smoke",
            "ofc-diagnostic-only": "true",
        },
    }


def build_launch_contract(
    *,
    transport_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    prebootstrap_path: str | Path,
    startup_path: str | Path = DEFAULT_STARTUP,
    _skip_validation: bool = False,
) -> dict[str, Any]:
    direct = _transport_stage1_candidate(transport_contract)
    trust = _public_key_binding(
        direct_contract=direct,
        public_key_record=controller_public_key_record,
    )
    startup = _file_record(
        startup_path,
        expected_suffix=".sh",
        kind="step11_stage0_startup_shell",
    )
    prebootstrap = _file_record(
        prebootstrap_path,
        expected_suffix=".py",
        kind="step11_prebootstrap_validator",
    )
    direct_raw = canonical_bytes(direct)
    public_key = transport.validate_rsa_public_key_record(
        controller_public_key_record
    )
    public_key_raw = canonical_bytes(public_key)
    body = _instance_insert_static_body(direct)
    value = {
        "schema": SCHEMA,
        "status": STATUS,
        "diagnostic_only": True,
        "stage_id": plan.STAGE1_ID,
        "run_name": plan.STAGE1_RUN_NAME,
        "selected_job_ids": list(plan.STAGE1_JOB_IDS),
        "source_roles": ["candidate"],
        "vm_count": 1,
        "attempt_index": 0,
        "transport_contract_sha256": canonical_sha256(direct),
        "direct_stage_identity_sha256": direct[
            "direct_stage_identity_sha256"
        ],
        "outer_package_identity_sha256": direct[
            "outer_package_manifest"
        ]["outer_package_identity_sha256"],
        "controller_trust": trust,
        "target": {
            "project": PROJECT,
            "region": REGION,
            "zone": ZONE,
            "instance_name": direct["metadata_binding"]["instance_name"],
            "machine_type": MACHINE_TYPE,
            "image_project": transport.IMAGE_PROJECT,
            "image_name": transport.IMAGE_NAME,
            "image_id": transport.IMAGE_ID,
            "image_self_link": transport.IMAGE_SELF_LINK,
            "image_family_resolution_permitted": False,
            "worker_service_account": WORKER_SERVICE_ACCOUNT,
        },
        "network_egress_contract": {
            "network_self_link": NETWORK_SELF_LINK,
            "subnetwork_self_link": SUBNETWORK_SELF_LINK,
            "external_ip_permitted": False,
            "access_configs": [],
            "private_google_access_observed": False,
            "cloud_nat_required": True,
            "nat_router_name": NAT_ROUTER_NAME,
            "nat_router_self_link": NAT_ROUTER_SELF_LINK,
            "nat_name": NAT_NAME,
            "nat_type": "PUBLIC",
            "nat_source_subnetwork_ip_ranges": (
                "ALL_SUBNETWORKS_ALL_IP_RANGES"
            ),
            "nat_endpoint_types": ["ENDPOINT_TYPE_VM"],
            "network_observation_required_before_insert": True,
        },
        "instance_insert": {
            "method": "POST",
            "url": (
                "https://compute.googleapis.com/compute/v1/projects/"
                f"{PROJECT}/zones/{ZONE}/instances"
            ),
            "request_id_uuid_v4_required": True,
            "static_body": body,
            "static_body_sha256": canonical_sha256(body),
        },
        "runtime_guards": {
            "provisioning_model": "SPOT",
            "max_runtime_seconds": MAX_RUNTIME_SECONDS,
            "instance_termination_action": "DELETE",
            "automatic_restart": False,
            "delete_protection": False,
            "boot_disk_auto_delete": True,
            "failure_shutdown_seconds": FAILURE_SHUTDOWN_SECONDS,
            "startup_shell_record": startup,
            "prebootstrap_record": prebootstrap,
        },
        "metadata_from_file_contract": {
            "initial_exact_keys": list(INITIAL_METADATA_FROM_FILE_KEYS),
            "initial_static_records": {
                "startup-script": startup,
                "ofc-step11-transport-contract": {
                    "kind": "canonical_transport_contract",
                    "sha256": hashlib.sha256(direct_raw).hexdigest(),
                    "bytes": len(direct_raw),
                    "metadata_value_max_bytes": 262_144,
                },
                "ofc-step11-controller-public-key": {
                    "kind": "canonical_controller_public_key",
                    "sha256": hashlib.sha256(public_key_raw).hexdigest(),
                    "bytes": len(public_key_raw),
                    "metadata_value_max_bytes": 262_144,
                },
                "ofc-step11-prebootstrap": prebootstrap,
            },
            "dynamic_initial_key": "ofc-step11-controller-authorization",
            "dynamic_initial_schema": transport.AUTHORIZATION_SCHEMA,
            "claim_key": POSTCREATE_CLAIM_METADATA_KEY,
            "claim_schema": transport.CLAIM_SCHEMA,
            "claim_present_at_insert": False,
            "claim_release_phase": "post_instance_insert_get",
            "claim_release_requires_provider_instance_id": True,
            "claim_release_requires_metadata_fingerprint_cas": True,
            "claim_release_may_add_only_claim_key": True,
            "claim_release_must_preserve_all_initial_metadata": True,
            "unknown_metadata_key_is_fatal": True,
        },
        "permission_contract": {
            "worker_oauth_scope": OAUTH_SCOPE,
            "worker_required_iam_permissions": [
                "compute.instances.delete",
                "storage.objects.create",
                "storage.objects.get",
            ],
            "compute_delete_must_be_conditioned_to_exact_instance": True,
            "worker_permission_proof_required_before_insert": True,
            "service_account_inherited_iam_audit_required": True,
            "uniform_bucket_level_access_required": True,
        },
        "authorization_boundary": {
            "package_provisioning_performed": False,
            "controller_authorization_created": False,
            "claim_created": False,
            "claim_released": False,
            "instance_insert_performed": False,
            "vm_created": False,
            "cloud_mutation_performed": False,
            "launch_authorized": False,
            "launch_ready": False,
            "current_profile_changed": False,
            "performance_lock_evidence": False,
            "quality_evidence": False,
            "training_eligible": False,
            "promotion_evidence": False,
        },
    }
    if len(direct_raw) > 262_144 or len(public_key_raw) > 262_144:
        raise ValueError("initial metadata file escaped Compute metadata limit")
    if _skip_validation:
        return value
    return validate_launch_contract(
        value,
        transport_contract=direct,
        controller_public_key_record=public_key,
        prebootstrap_path=prebootstrap_path,
        startup_path=startup_path,
    )


def validate_launch_contract(
    value: Mapping[str, Any],
    *,
    transport_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    prebootstrap_path: str | Path,
    startup_path: str | Path = DEFAULT_STARTUP,
) -> dict[str, Any]:
    checked = dict(value)
    expected = build_launch_contract(
        transport_contract=transport_contract,
        controller_public_key_record=controller_public_key_record,
        prebootstrap_path=prebootstrap_path,
        startup_path=startup_path,
        _skip_validation=True,
    )
    if checked != expected:
        raise ValueError("Step 11 launch contract changed")
    if (
        checked["instance_insert"]["static_body"]["networkInterfaces"][0][
            "accessConfigs"
        ]
        != []
        or checked["metadata_from_file_contract"][
            "claim_present_at_insert"
        ]
        is not False
        or checked["metadata_from_file_contract"]["claim_key"]
        in checked["metadata_from_file_contract"]["initial_exact_keys"]
        or any(checked["authorization_boundary"].values())
    ):
        raise ValueError("Step 11 launch boundary changed")
    return checked


def freeze_launch_contract(
    *,
    transport_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    prebootstrap_path: str | Path,
    output: str | Path,
    startup_path: str | Path = DEFAULT_STARTUP,
) -> dict[str, Any]:
    value = build_launch_contract(
        transport_contract=transport_contract,
        controller_public_key_record=controller_public_key_record,
        prebootstrap_path=prebootstrap_path,
        startup_path=startup_path,
    )
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Step 11 launch contract output must be fresh")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--transport-contract", type=Path, required=True)
        command.add_argument("--controller-public-key", type=Path, required=True)
        command.add_argument("--prebootstrap", type=Path, required=True)
        command.add_argument("--startup", type=Path, default=DEFAULT_STARTUP)
        if name == "freeze":
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("--contract", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    direct = _read_canonical(args.transport_contract, "transport contract")
    public_key = _read_canonical(
        args.controller_public_key, "controller public key"
    )
    if args.command == "freeze":
        value = freeze_launch_contract(
            transport_contract=direct,
            controller_public_key_record=public_key,
            prebootstrap_path=args.prebootstrap,
            startup_path=args.startup,
            output=args.output,
        )
    else:
        value = validate_launch_contract(
            _read_canonical(args.contract, "Step 11 launch contract"),
            transport_contract=direct,
            controller_public_key_record=public_key,
            prebootstrap_path=args.prebootstrap,
            startup_path=args.startup,
        )
    print(
        json.dumps(
            {
                "schema": value["schema"],
                "status": value["status"],
                "contract_sha256": canonical_sha256(value),
                "launch_ready": value["authorization_boundary"][
                    "launch_ready"
                ],
            },
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "INITIAL_METADATA_FROM_FILE_KEYS",
    "MAX_RUNTIME_SECONDS",
    "POSTCREATE_CLAIM_METADATA_KEY",
    "SCHEMA",
    "STATUS",
    "build_launch_contract",
    "canonical_bytes",
    "canonical_sha256",
    "freeze_launch_contract",
    "main",
    "sha256_file",
    "validate_launch_contract",
]
