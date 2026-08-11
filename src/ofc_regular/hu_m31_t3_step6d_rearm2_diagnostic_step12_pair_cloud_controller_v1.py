"""Exact two-VM controller for the Step 12 diagnostic Stage-2 pair.

This module deliberately reuses the frozen Step-11 transport and worker
package.  The inner package remains c4-standard-16/Rayon-16, while the actual
diagnostic launch topology is separately and explicitly bound to two
c4-standard-8 Spot VMs.  Consequently every result produced here is lifecycle
diagnostic evidence only and is inadmissible as performance, quality, training,
or promotion evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_10c2_receiver_preflight
    as receiver,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_adapter
    as adapter,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as transport,
)
from . import hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as plan
from . import hu_m31_t3_step6d_rearm2_diagnostic_canary_vm_adapter as legacy_vm
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_cloud_controller_v1
    as step11_cloud,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_controller_v1
    as controller,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_launch_contract_v1
    as step11_launch,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12_pair_contract_v1
    as pair_contract_module,
)


SCHEMA = "hu_m31_t3_step6d_rearm2_diagnostic_step12_pair_controller_v1"
FINAL_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_step12_pair_lifecycle_receipt_v1"
)
POST_PAIR_CLAIM_CALLBACK_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_step12_post_pair_claim_callback_v1"
)
POST_PAIR_CLAIM_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_step12_post_pair_claim_revoke_v1"
)
EXECUTION_CONFIRMATION = "EXECUTE_STEP12_STAGE2_CANDIDATE_REFERENCE_ATTEMPT0"

ACTUAL_MACHINE_TYPE = "c4-standard-8"
INNER_MACHINE_TYPE = transport.MACHINE_TYPE
INNER_RAYON_THREADS = 16
PROJECT = transport.PROJECT
PROJECT_NUMBER = step11_cloud.PROJECT_NUMBER
ZONE = transport.ZONE
REGION = step11_launch.REGION
WORKER_SERVICE_ACCOUNT = transport.WORKER_SERVICE_ACCOUNT
OAUTH_SCOPE = transport.REQUIRED_WORKER_OAUTH_SCOPE
JOB_IDS = tuple(plan.STAGE2_JOB_IDS)
SOURCE_ROLES = ("candidate", "reference")
ATTEMPT_INDEX = 0
VM_COUNT = 2

STARTUP_METADATA_KEY = "startup-script"
TRANSPORT_METADATA_KEY = "ofc-step11-transport-contract"
AUTHORIZATION_METADATA_KEY = "ofc-step11-controller-authorization"
PUBLIC_KEY_METADATA_KEY = "ofc-step11-controller-public-key"
PREBOOTSTRAP_METADATA_KEY = "ofc-step11-prebootstrap"
BASE_PREBOOTSTRAP_METADATA_KEY = "ofc-step12-prebootstrap-base"
BLOCK_PROJECT_SSH_KEYS = "block-project-ssh-keys"
RELEASE_STATE_KEY = step11_cloud.RELEASE_STATE_KEY
CLAIM_KEY = step11_cloud.CLAIM_KEY

MAX_METADATA_VALUE_BYTES = 262_144
MAX_TOTAL_METADATA_BYTES = 512 * 1024
DONE_TIMEOUT_SECONDS = 4_500
ABSENCE_TIMEOUT_SECONDS = 600
DEFAULT_POLL_SECONDS = 2.0
DIRECT_RESULT_MARKER = "/hu-m31-r2diag-direct-v1/stages/"
LEGACY_RESULT_MARKER = "/hu-m31-r2diag-worker-v1/"


class JsonClient(Protocol):
    def request(
        self,
        *,
        method: str,
        url: str,
        body: bytes | None = None,
        content_type: str | None = None,
        timeout_seconds: int = 60,
    ) -> controller.HttpResponse: ...


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


def _request_id(value: str | None = None) -> str:
    candidate = str(uuid.uuid4()) if value is None else value
    parsed = uuid.UUID(candidate)
    if parsed.version != 4 or str(parsed) != candidate:
        raise ValueError("request ID must be canonical UUIDv4")
    return candidate


def _read_file(path: str | Path, *, maximum: int) -> bytes:
    source = Path(path)
    if (
        not source.is_file()
        or source.is_symlink()
        or source.stat().st_size <= 0
        or source.stat().st_size > maximum
    ):
        raise ValueError(f"metadata file identity changed: {source}")
    return source.read_bytes()


def _metadata_map(items: Any) -> dict[str, str]:
    if not isinstance(items, list):
        raise ValueError("provider metadata items changed")
    result: dict[str, str] = {}
    for item in items:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"key", "value"}
            or not isinstance(item["key"], str)
            or not isinstance(item["value"], str)
            or item["key"] in result
        ):
            raise ValueError("provider metadata item changed")
        result[item["key"]] = item["value"]
    return result


def _strict_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not an object")
    return value


def _instance_url(instance_name: str) -> str:
    if instance_name not in {
        contract_name
        for contract_name in getattr(_instance_url, "_allowed", ())
    }:
        raise ValueError("instance URL escaped frozen Step12 pair")
    return (
        "https://compute.googleapis.com/compute/v1/projects/"
        f"{PROJECT}/zones/{ZONE}/instances/{instance_name}"
    )


def _bind_instance_names(names: Sequence[str]) -> None:
    frozen = tuple(names)
    if len(frozen) != VM_COUNT or len(set(frozen)) != VM_COUNT:
        raise ValueError("Step12 instance-name set changed")
    setattr(_instance_url, "_allowed", frozen)


def _checked_transport_pair(
    candidate_transport_contract: Mapping[str, Any],
    reference_transport_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = transport.validate_job_contract(candidate_transport_contract)
    reference = transport.validate_job_contract(reference_transport_contract)
    values = (candidate, reference)
    preview = candidate["adapter_preview"]
    if (
        preview != reference["adapter_preview"]
        or preview.get("stage_id") != plan.STAGE2_ID
        or preview.get("run_name") != plan.STAGE2_RUN_NAME
        or preview.get("selected_job_ids") != list(JOB_IDS)
        or preview.get("attempt_index") != ATTEMPT_INDEX
        or preview.get("vm_count") != VM_COUNT
        or [row["metadata_binding"]["job_id"] for row in values]
        != list(JOB_IDS)
        or [row["metadata_binding"]["source_role"] for row in values]
        != list(SOURCE_ROLES)
        or any(
            row["metadata_binding"]["attempt_index"] != ATTEMPT_INDEX
            for row in values
        )
        or candidate["direct_stage_identity"]
        != reference["direct_stage_identity"]
        or candidate["remote_layout"] != reference["remote_layout"]
        or candidate["outer_package_manifest"]
        != reference["outer_package_manifest"]
    ):
        raise ValueError("Step12 transport pair escaped frozen Stage2 topology")
    names = [row["metadata_binding"]["instance_name"] for row in values]
    _bind_instance_names(names)
    return candidate, reference


def _validate_pair_execution_contract(
    pair_contract: Mapping[str, Any],
    *,
    transports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    checked = dict(pair_contract)
    digest = checked.get("pair_contract_sha256")
    unsigned = dict(checked)
    unsigned.pop("pair_contract_sha256", None)
    instances = checked.get("instances")
    topology = checked.get("execution_topology")
    underlying = checked.get("underlying_frozen_identity")
    namespace = checked.get("result_namespace")
    boundary = checked.get("authorization_boundary")
    preview = transports[0]["adapter_preview"]
    layout = transports[0]["remote_layout"]
    direct_jobs = layout["jobs"]
    if (
        not isinstance(digest, str)
        or digest != canonical_sha256(unsigned)
        or checked.get("stage_id") != plan.STAGE2_ID
        or checked.get("run_name") != plan.STAGE2_RUN_NAME
        or checked.get("selected_job_ids") != list(JOB_IDS)
        or checked.get("source_roles") != list(SOURCE_ROLES)
        or checked.get("vm_count") != VM_COUNT
        or checked.get("attempt_index") != ATTEMPT_INDEX
        or checked.get("diagnostic_only") is not True
        or checked.get("scientific_payload_present") is not False
        or checked.get("evidence_admissibility")
        != {
            "performance_lock_evidence": False,
            "quality_evidence": False,
            "training_eligible": False,
            "promotion_evidence": False,
        }
        or not isinstance(topology, Mapping)
        or topology.get("actual_machine_type") != ACTUAL_MACHINE_TYPE
        or topology.get("inner_declared_machine_type") != INNER_MACHINE_TYPE
        or topology.get("inner_rayon_threads_per_process")
        != INNER_RAYON_THREADS
        or topology.get("cpu_oversubscribed") is not True
        or topology.get("diagnostic_only") is not True
        or topology.get("admissible_as_performance_evidence") is not False
        or not isinstance(instances, list)
        or [row.get("job_id") for row in instances] != list(JOB_IDS)
        or [row.get("source_role") for row in instances]
        != list(SOURCE_ROLES)
        or [row.get("instance_name") for row in instances]
        != [row["metadata_binding"]["instance_name"] for row in transports]
        or any(
            row.get("machine_type") != ACTUAL_MACHINE_TYPE
            or row.get("actual_vcpus") != 8
            or row.get("worker_process_count") != 1
            or row.get("inner_rayon_threads") != INNER_RAYON_THREADS
            or row.get("attempt_index") != ATTEMPT_INDEX
            for row in instances
        )
        or any(
            instance.get("transport_contract_sha256")
            != canonical_sha256(contract)
            or instance.get("metadata_binding_sha256")
            != contract["metadata_binding_sha256"]
            or instance.get("runner_job_manifest_sha256")
            != next(
                row["runner_job_manifest"]["sha256"]
                for row in preview["jobs"]
                if row["job_id"] == instance["job_id"]
            )
            or instance.get("work_hand_indices")
            != list(plan.STAGE2_HAND_INDICES)
            or instance.get("root_records_sha256")
            != canonical_sha256(
                next(
                    row["root_records"]
                    for row in preview["jobs"]
                    if row["job_id"] == instance["job_id"]
                )
            )
            or instance.get("result_prefix") != direct_job["result_prefix"]
            or instance.get("tree_prefix") != direct_job["tree_prefix"]
            or instance.get("done_uri") != direct_job["done_uri"]
            for instance, contract, direct_job in zip(
                instances, transports, direct_jobs, strict=True
            )
        )
        or not isinstance(underlying, Mapping)
        or underlying.get("outer_package_identity_sha256")
        != transports[0]["outer_package_manifest"][
            "outer_package_identity_sha256"
        ]
        or underlying.get("outer_package_manifest_sha256")
        != transports[0]["outer_package_manifest_sha256"]
        or underlying.get("direct_stage_identity_sha256")
        != transports[0]["direct_stage_identity_sha256"]
        or underlying.get("preview_stage_identity_sha256")
        != preview["stage_identity_sha256"]
        or underlying.get("adapter_preview_sha256")
        != transports[0]["adapter_preview_sha256"]
        or underlying.get("inner_machine_type") != INNER_MACHINE_TYPE
        or underlying.get("inner_rayon_num_threads")
        != str(INNER_RAYON_THREADS)
        or underlying.get("immutable_fix3_outer_package_reused") is not True
        or underlying.get("underlying_direct_identity_changed") is not False
        or not isinstance(namespace, Mapping)
        or namespace.get("authoritative_source")
        != "remote_layout_direct_v1"
        or namespace.get("stage_prefix") != layout["stage_prefix"]
        or namespace.get("result_prefix") != layout["result_prefix"]
        or namespace.get("receive_uri") != layout["receive_uri"]
        or namespace.get("direct_stage_identity_sha256")
        != layout["direct_stage_identity_sha256"]
        or namespace.get(
            "adapter_preview_result_uris_runtime_authoritative"
        )
        is not False
        or namespace.get("entire_stage_prefix_must_be_empty_before_attempt0")
        is not True
        or namespace.get("ordered_done_job_ids") != list(JOB_IDS)
        or not isinstance(boundary, Mapping)
        or boundary.get("vm_limit") != VM_COUNT
        or boundary.get("attempt_limit_per_job") != 1
        or boundary.get("attempt1_authorized") is not False
        or boundary.get("resume_authorized") is not False
        or boundary.get("third_vm_authorized") is not False
        or checked.get("current_profile_sha256")
        != pair_contract_module.EXPECTED_CURRENT_PROFILE_SHA256
        or checked.get("current_profile_changed") is not False
    ):
        raise ValueError("Step12 pair execution contract changed")
    return checked


def build_insert_body(
    *,
    transport_contract: Mapping[str, Any],
    authorization: Mapping[str, Any],
    public_key_record: Mapping[str, Any],
    startup_path: str | Path,
    prebootstrap_path: str | Path,
    base_prebootstrap_path: str | Path,
) -> dict[str, Any]:
    checked = transport.validate_job_contract(transport_contract)
    binding = checked["metadata_binding"]
    if (
        binding["stage_id"] != plan.STAGE2_ID
        or binding["job_id"] not in JOB_IDS
        or binding["attempt_index"] != ATTEMPT_INDEX
    ):
        raise ValueError("Step12 insert escaped the frozen pair")
    public_key = transport.validate_rsa_public_key_record(public_key_record)
    if (
        checked["authorization_contract"]["controller_public_key_sha256"]
        != canonical_sha256(public_key)
        or checked["authorization_contract"]["controller_key_id"]
        != public_key["key_id"]
    ):
        raise ValueError("Step12 public key is not pinned by transport")
    startup = _read_file(startup_path, maximum=MAX_METADATA_VALUE_BYTES)
    prebootstrap = _read_file(
        prebootstrap_path, maximum=MAX_METADATA_VALUE_BYTES
    )
    base = _read_file(
        base_prebootstrap_path, maximum=MAX_METADATA_VALUE_BYTES
    )
    values = {
        STARTUP_METADATA_KEY: startup.decode("utf-8"),
        TRANSPORT_METADATA_KEY: canonical_bytes(checked).decode("ascii"),
        AUTHORIZATION_METADATA_KEY: canonical_bytes(authorization).decode(
            "ascii"
        ),
        PUBLIC_KEY_METADATA_KEY: canonical_bytes(public_key).decode("ascii"),
        PREBOOTSTRAP_METADATA_KEY: prebootstrap.decode("utf-8"),
        BASE_PREBOOTSTRAP_METADATA_KEY: base.decode("utf-8"),
        BLOCK_PROJECT_SSH_KEYS: "true",
        RELEASE_STATE_KEY: "pending-post-create",
        **dict(checked["metadata_values"]),
    }
    if (
        CLAIM_KEY in values
        or len(values) != len(set(values))
        or any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or len(value.encode("utf-8")) > MAX_METADATA_VALUE_BYTES
            for key, value in values.items()
        )
        or sum(
            len(key.encode("utf-8")) + len(value.encode("utf-8"))
            for key, value in values.items()
        )
        > MAX_TOTAL_METADATA_BYTES
    ):
        raise ValueError("Step12 initial metadata escaped exact limits")
    return {
        "name": binding["instance_name"],
        "machineType": (
            "https://www.googleapis.com/compute/v1/projects/"
            f"{PROJECT}/zones/{ZONE}/machineTypes/{ACTUAL_MACHINE_TYPE}"
        ),
        "canIpForward": False,
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
                "network": step11_launch.NETWORK_SELF_LINK,
                "subnetwork": step11_launch.SUBNETWORK_SELF_LINK,
                "accessConfigs": [],
            }
        ],
        "reservationAffinity": {"consumeReservationType": "NO_RESERVATION"},
        "scheduling": {
            "provisioningModel": "SPOT",
            "instanceTerminationAction": "DELETE",
            "automaticRestart": False,
            "onHostMaintenance": "TERMINATE",
            "maxRunDuration": {
                "seconds": str(legacy_vm.MAX_RUNTIME_SECONDS_PER_VM),
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
            "items": [
                {"key": key, "value": values[key]} for key in sorted(values)
            ]
        },
    }


def _validate_provider(
    value: Mapping[str, Any],
    *,
    expected_name: str,
    expected_initial_metadata: Mapping[str, str],
) -> dict[str, Any]:
    record = step11_cloud._validate_provider_instance(
        value,
        expected_name=expected_name,
        expected_initial_metadata=expected_initial_metadata,
    )
    machine = value.get("machineType")
    if (
        not isinstance(machine, str)
        or urllib.parse.urlsplit(machine).path.rsplit("/", 1)[-1]
        != ACTUAL_MACHINE_TYPE
    ):
        raise RuntimeError("provider machine type escaped c4-standard-8")
    return {**record, "machine_type": ACTUAL_MACHINE_TYPE}


def _validate_claimed_provider(
    value: Mapping[str, Any],
    *,
    expected_name: str,
    expected_instance_id: str,
    expected_initial_metadata: Mapping[str, str],
    claim: Mapping[str, Any],
) -> dict[str, Any]:
    record = step11_cloud._validate_claimed_provider_instance(
        value,
        expected_name=expected_name,
        expected_instance_id=expected_instance_id,
        expected_initial_metadata=expected_initial_metadata,
        claim=claim,
    )
    machine = value.get("machineType")
    if (
        not isinstance(machine, str)
        or urllib.parse.urlsplit(machine).path.rsplit("/", 1)[-1]
        != ACTUAL_MACHINE_TYPE
    ):
        raise RuntimeError("claimed provider machine type changed")
    return {**record, "machine_type": ACTUAL_MACHINE_TYPE}


def _provider_get(
    client: JsonClient, instance_name: str
) -> tuple[controller.HttpResponse, dict[str, Any] | None]:
    response = client.request(method="GET", url=_instance_url(instance_name))
    if response.status == 404:
        return response, None
    if response.status != 200:
        raise RuntimeError(
            f"provider instance GET returned HTTP {response.status}"
        )
    return response, _strict_json(response.body, "provider instance")


def _read_done_once(
    *,
    client: JsonClient,
    done_uri: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    pinned = step11_cloud.read_generation_pinned_object(
        client=client, uri=done_uri, allow_missing=True
    )
    if pinned is None:
        return None
    record, raw = pinned
    return record, _strict_json(raw, "worker DONE")


def wait_for_pair_done(
    *,
    collector_client: JsonClient,
    instance_client: JsonClient,
    preview: Mapping[str, Any],
    direct_jobs: Sequence[Mapping[str, Any]],
    instance_names: Sequence[str],
    timeout_seconds: int = DONE_TIMEOUT_SECONDS,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if (
        type(timeout_seconds) is not int
        or not 1 <= timeout_seconds <= DONE_TIMEOUT_SECONDS
        or not math.isfinite(poll_seconds)
        or not 0 < poll_seconds <= 10
        or preview.get("stage_id") != plan.STAGE2_ID
        or preview.get("selected_job_ids") != list(JOB_IDS)
        or preview.get("attempt_index") != ATTEMPT_INDEX
        or len(direct_jobs) != VM_COUNT
        or [row.get("job_id") for row in direct_jobs] != list(JOB_IDS)
        or len(instance_names) != VM_COUNT
        or len(set(instance_names)) != VM_COUNT
        or len({row.get("done_uri") for row in direct_jobs}) != VM_COUNT
        or any(
            not isinstance(row.get("done_uri"), str)
            or DIRECT_RESULT_MARKER not in row["done_uri"]
            or LEGACY_RESULT_MARKER in row["done_uri"]
            for row in direct_jobs
        )
    ):
        raise ValueError("Step12 DONE monitor escaped exact pair")
    observed: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    deadline = now() + timeout_seconds
    while len(observed) != VM_COUNT:
        for job, instance_name in zip(
            direct_jobs, instance_names, strict=True
        ):
            job_id = str(job["job_id"])
            if job_id in observed:
                continue
            done_uri = str(job["done_uri"])
            expected_suffix = (
                f"/results/jobs/{job_id}/DONE.envelope.json"
            )
            if not done_uri.endswith(expected_suffix):
                raise ValueError("Step12 monitor rejected non-direct DONE URI")
            item = _read_done_once(
                client=collector_client, done_uri=done_uri
            )
            if item is not None:
                if item[1].get("job_id") != job_id:
                    raise RuntimeError("Step12 DONE job identity changed")
                observed[job_id] = item
                continue
            response, provider = _provider_get(
                instance_client, instance_name
            )
            terminal: str | None = None
            if response.status == 404:
                terminal = "ABSENT"
            else:
                assert provider is not None
                status = provider.get("status")
                if provider.get("name") != instance_name or status not in {
                    "PROVISIONING",
                    "STAGING",
                    "RUNNING",
                    "STOPPING",
                    "SUSPENDING",
                    "SUSPENDED",
                    "REPAIRING",
                    "TERMINATED",
                }:
                    raise RuntimeError("Step12 provider monitor changed")
                if status in {
                    "STOPPING",
                    "SUSPENDING",
                    "SUSPENDED",
                    "TERMINATED",
                }:
                    terminal = str(status)
            if terminal is not None:
                item = _read_done_once(
                    client=collector_client, done_uri=done_uri
                )
                if item is not None:
                    if item[1].get("job_id") != job_id:
                        raise RuntimeError("late DONE job identity changed")
                    observed[job_id] = item
                    continue
                raise RuntimeError(
                    f"{job_id} entered {terminal} before publishing DONE"
                )
        if len(observed) == VM_COUNT:
            break
        if now() >= deadline:
            raise TimeoutError("Step12 pair DONE timed out")
        sleep(min(poll_seconds, max(0.0, deadline - now())))
    ordered_records = [observed[job_id][0] for job_id in JOB_IDS]
    ordered_done = [observed[job_id][1] for job_id in JOB_IDS]
    receive = adapter.build_receive(preview, done_records=ordered_done)
    return ordered_records, receive


@dataclass
class Step12PairExecutionError(RuntimeError):
    message: str
    cleanup: Sequence[Mapping[str, Any]]

    def __str__(self) -> str:
        return self.message


def execute_pair_attempt0(
    *,
    execute: bool,
    execution_confirmation: str,
    client: JsonClient,
    collector_client: JsonClient,
    pair_contract: Mapping[str, Any],
    candidate_transport_contract: Mapping[str, Any],
    reference_transport_contract: Mapping[str, Any],
    signer: controller.EphemeralControllerKey,
    package_generations: Mapping[str, int],
    external_preflight_receipt_sha256: str,
    issued_unix_seconds: int,
    expires_unix_seconds: int,
    post_pair_claim_callback: Callable[
        [Mapping[str, Any]], Mapping[str, Any]
    ],
    startup_path: str | Path,
    prebootstrap_path: str | Path,
    base_prebootstrap_path: str | Path,
    destination_root: str | Path,
    final_receipt_path: str | Path | None = None,
    request_ids: Sequence[str] | None = None,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if execute is not True or execution_confirmation != EXECUTION_CONFIRMATION:
        raise PermissionError("Step12 pair execution confirmation is missing")
    candidate, reference = _checked_transport_pair(
        candidate_transport_contract, reference_transport_contract
    )
    transports = (candidate, reference)
    checked_pair = _validate_pair_execution_contract(
        pair_contract, transports=transports
    )
    expected_package_uris = [
        row["uri"]
        for row in candidate["remote_layout"]["package_inventory"]["records"]
    ]
    if (
        not isinstance(package_generations, Mapping)
        or set(package_generations) != set(expected_package_uris)
        or any(
            type(package_generations[uri]) is not int
            or package_generations[uri] <= 0
            for uri in expected_package_uris
        )
    ):
        raise ValueError("Step12 package generations are incomplete")
    ids = (
        [_request_id() for _ in range(6)]
        if request_ids is None
        else [_request_id(item) for item in request_ids]
    )
    if len(ids) != 6 or len(set(ids)) != 6:
        raise ValueError("Step12 requires six unique request IDs")

    authorizations = [
        controller.build_controller_authorization(
            contract=row,
            external_preflight_receipt_sha256=(
                external_preflight_receipt_sha256
            ),
            issued_unix_seconds=issued_unix_seconds,
            expires_unix_seconds=expires_unix_seconds,
            signer=signer,
        )
        for row in transports
    ]
    bodies = [
        build_insert_body(
            transport_contract=row,
            authorization=authorization,
            public_key_record=signer.public_record,
            startup_path=startup_path,
            prebootstrap_path=prebootstrap_path,
            base_prebootstrap_path=base_prebootstrap_path,
        )
        for row, authorization in zip(
            transports, authorizations, strict=True
        )
    ]
    initial_metadata = [
        _metadata_map(body["metadata"]["items"]) for body in bodies
    ]
    instance_names = [
        row["metadata_binding"]["instance_name"] for row in transports
    ]
    inserted = [False, False]
    provider_raw: list[dict[str, Any] | None] = [None, None]
    provider_records: list[dict[str, Any] | None] = [None, None]
    insert_operations: list[dict[str, Any] | None] = [None, None]
    claims: list[dict[str, Any] | None] = [None, None]
    claim_operations: list[dict[str, Any] | None] = [None, None]
    claimed_records: list[dict[str, Any] | None] = [None, None]
    cleanup: list[Mapping[str, Any]] = []
    try:
        insert_url = (
            "https://compute.googleapis.com/compute/v1/projects/"
            f"{PROJECT}/zones/{ZONE}/instances"
        )
        for index, (name, body) in enumerate(
            zip(instance_names, bodies, strict=True)
        ):
            inserted[index] = True
            response = client.request(
                method="POST",
                url=f"{insert_url}?requestId={ids[index]}",
                body=canonical_bytes(body),
                content_type="application/json",
            )
            if response.status not in (200, 202):
                raise RuntimeError(
                    f"{name} insert returned HTTP {response.status}"
                )
            operation = step11_cloud.wait_zone_operation(
                client=client,
                initial=_strict_json(response.body, "insert operation"),
                expected_instance_url=_instance_url(name),
                now=now,
                sleep=sleep,
            )
            insert_operations[index] = operation
            _, provider = _provider_get(client, name)
            if provider is None:
                raise RuntimeError("inserted provider disappeared before claim")
            provider_raw[index] = provider
            provider_records[index] = _validate_provider(
                provider,
                expected_name=name,
                expected_initial_metadata=initial_metadata[index],
            )

        for index, (row, authorization, name) in enumerate(
            zip(transports, authorizations, instance_names, strict=True)
        ):
            provider = provider_raw[index]
            provider_record = provider_records[index]
            assert provider is not None and provider_record is not None
            claim = controller.build_worker_claim(
                contract=row,
                authorization=authorization,
                project_number=PROJECT_NUMBER,
                instance_id=provider_record["instance_id"],
                package_generations=package_generations,
                signer=signer,
            )
            claims[index] = claim
            cas_body = step11_cloud.build_claim_cas_body(
                provider_instance=provider,
                expected_initial_metadata=initial_metadata[index],
                claim=claim,
            )
            response = client.request(
                method="POST",
                url=(
                    f"{_instance_url(name)}/setMetadata"
                    f"?requestId={ids[2 + index]}"
                ),
                body=canonical_bytes(cas_body),
                content_type="application/json",
            )
            if response.status == 412:
                raise RuntimeError(f"{name} claim CAS precondition failed")
            if response.status not in (200, 202):
                raise RuntimeError(
                    f"{name} claim CAS returned HTTP {response.status}"
                )
            claim_operations[index] = step11_cloud.wait_zone_operation(
                client=client,
                initial=_strict_json(response.body, "claim operation"),
                expected_instance_url=_instance_url(name),
                now=now,
                sleep=sleep,
            )
            _, claimed = _provider_get(client, name)
            if claimed is None:
                raise RuntimeError("provider disappeared during claim readback")
            claimed_records[index] = _validate_claimed_provider(
                claimed,
                expected_name=name,
                expected_instance_id=provider_record["instance_id"],
                expected_initial_metadata=initial_metadata[index],
                claim=claim,
            )

        callback_input = {
            "schema": POST_PAIR_CLAIM_CALLBACK_SCHEMA,
            "instance_names": instance_names,
            "provider_instance_ids": [
                row["instance_id"] for row in provider_records if row
            ],
            "claim_sha256s": [
                canonical_sha256(row) for row in claims if row
            ],
            "both_claims_provider_readback_verified": True,
        }
        callback = dict(post_pair_claim_callback(callback_input))
        unsigned_callback = dict(callback)
        callback_sha = unsigned_callback.pop("receipt_sha256", None)
        if (
            callback.get("schema") != POST_PAIR_CLAIM_RECEIPT_SCHEMA
            or callback.get("status")
            != "launch_and_worker_actas_removed_after_both_claims"
            or callback.get("instance_names") != instance_names
            or callback.get("claim_sha256s")
            != callback_input["claim_sha256s"]
            or callback.get("launch_binding_removed") is not True
            or callback.get("worker_actas_binding_removed") is not True
            or callback.get("readback_verified") is not True
            or callback_sha != canonical_sha256(unsigned_callback)
        ):
            raise RuntimeError("Step12 post-pair claim revocation failed")

        direct_jobs = candidate["remote_layout"]["jobs"]
        done_records, receive_record = wait_for_pair_done(
            collector_client=collector_client,
            instance_client=client,
            preview=candidate["adapter_preview"],
            direct_jobs=direct_jobs,
            instance_names=instance_names,
            now=now,
            sleep=sleep,
        )
        destination = Path(destination_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        backend = step11_cloud.GenerationPinnedGcsBackend(
            read_client=client,
            list_client=collector_client,
            allowed_prefixes=[row["tree_prefix"] for row in direct_jobs],
        )
        materialization = receiver.materialize_and_validate_received_stage(
            candidate["adapter_preview"],
            receive=receive_record,
            destination_root=destination,
            backend=backend,
            outer_package_manifest=candidate["outer_package_manifest"],
            direct_stage_identity=candidate["direct_stage_identity"],
        )
        materialization = step11_cloud._validate_cloud_materialization(
            value=materialization,
            transport_contract=candidate,
            receive_record=receive_record,
        )
        absences = [
            step11_cloud.wait_for_instance_absence(
                client=client,
                instance_name=name,
                timeout_seconds=ABSENCE_TIMEOUT_SECONDS,
                now=now,
                sleep=sleep,
            )
            for name in instance_names
        ]
        receipt_body = {
            "schema": FINAL_RECEIPT_SCHEMA,
            "status": (
                "step12_candidate_reference_pair_received_validated_and_absent"
            ),
            "diagnostic_only": True,
            "stage_id": plan.STAGE2_ID,
            "run_name": plan.STAGE2_RUN_NAME,
            "selected_job_ids": list(JOB_IDS),
            "source_roles": list(SOURCE_ROLES),
            "attempt_index": ATTEMPT_INDEX,
            "vm_count": VM_COUNT,
            "actual_machine_type": ACTUAL_MACHINE_TYPE,
            "inner_machine_type": INNER_MACHINE_TYPE,
            "inner_rayon_threads": INNER_RAYON_THREADS,
            "oversubscribed_diagnostic_only": True,
            "pair_contract_sha256": checked_pair["pair_contract_sha256"],
            "transport_contract_sha256s": [
                canonical_sha256(row) for row in transports
            ],
            "outer_package_identity_sha256": candidate[
                "outer_package_manifest"
            ]["outer_package_identity_sha256"],
            "direct_stage_identity_sha256": candidate[
                "direct_stage_identity_sha256"
            ],
            "external_preflight_receipt_sha256": (
                external_preflight_receipt_sha256
            ),
            "package_generations_sha256": canonical_sha256(
                dict(package_generations)
            ),
            "authorization_sha256s": [
                canonical_sha256(row) for row in authorizations
            ],
            "claim_sha256s": callback_input["claim_sha256s"],
            "post_pair_claim_revoke_receipt_sha256": callback[
                "receipt_sha256"
            ],
            "provider_instances": provider_records,
            "provider_instances_after_claim": claimed_records,
            "insert_request_body_sha256s": [
                canonical_sha256(row) for row in bodies
            ],
            "insert_operations": insert_operations,
            "claim_operations": claim_operations,
            "done_generation_records": done_records,
            "receive_sha256": canonical_sha256(receive_record),
            "materialization_result_sha256": canonical_sha256(
                materialization
            ),
            "provider_absences": absences,
            "provider_get_404_count": sum(
                row["provider_get_status"] == 404 for row in absences
            ),
            "worker_self_delete_observed_for_both": True,
            "controller_cleanup_delete_used": False,
            "scientific_payload_present": False,
            "performance_lock_evidence": False,
            "quality_evidence": False,
            "training_eligible": False,
            "promotion_evidence": False,
            "current_profile_changed": False,
            "cloud_mutation_performed": True,
        }
        receipt = {
            **receipt_body,
            "receipt_sha256": canonical_sha256(receipt_body),
        }
        if final_receipt_path is not None:
            controller.exclusive_write_json(final_receipt_path, receipt)
        return receipt
    except BaseException as error:
        for index, name in enumerate(instance_names):
            try:
                cleanup.append(
                    step11_cloud.delete_exact_instance(
                        client=client,
                        instance_name=name,
                        request_id=ids[4 + index],
                        now=now,
                        sleep=sleep,
                    )
                )
            except BaseException as cleanup_error:
                cleanup.append(
                    {
                        "instance_name": name,
                        "delete_requested": inserted[index],
                        "cleanup_failed": True,
                        "error_type": type(cleanup_error).__name__,
                        "error": str(cleanup_error),
                    }
                )
        raise Step12PairExecutionError(
            "Step12 pair lifecycle failed; cleanup attempted for both names",
            cleanup=tuple(cleanup),
        ) from error


__all__ = [
    "ACTUAL_MACHINE_TYPE",
    "EXECUTION_CONFIRMATION",
    "FINAL_RECEIPT_SCHEMA",
    "INNER_MACHINE_TYPE",
    "INNER_RAYON_THREADS",
    "POST_PAIR_CLAIM_RECEIPT_SCHEMA",
    "SCHEMA",
    "Step12PairExecutionError",
    "build_insert_body",
    "canonical_bytes",
    "canonical_sha256",
    "execute_pair_attempt0",
    "wait_for_pair_done",
]
