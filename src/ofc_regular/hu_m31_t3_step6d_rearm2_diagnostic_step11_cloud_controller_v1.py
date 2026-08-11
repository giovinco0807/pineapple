"""Executable, fail-closed controller for the Step 11 one-VM lifecycle smoke.

Importing this module cannot contact Google Cloud.  The only mutating entry
point requires both an explicit execution confirmation and an injected client
owned by the dedicated Step 11 controller service account.

The controller is deliberately limited to stage-1 candidate attempt 0.  It
materializes a valid Compute Engine REST ``metadata.items`` body, waits for
zonal operations, adds only the signed post-create claim with a metadata
fingerprint CAS, receives the generation-pinned result tree, runs the existing
independent materializer/validator, and requires an exact provider GET 404
after the worker deletes itself.  Any failure after insert triggers a bounded
controller delete of that exact instance.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_10c2_real_readonly_preflight_v1
    as real_preflight,
)
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
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_step11_iam_capacity_gate_v1
    as iam_gate,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_controller_v1
    as controller,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_launch_contract_v1
    as launch,
)


FINAL_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_step11_cloud_lifecycle_receipt_v1"
)
EXECUTION_CONFIRMATION = "EXECUTE_STEP11_ONE_VM_STAGE1_CANDIDATE_ATTEMPT0"
SHARED_PROJECT_EXCEPTION_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_step11_shared_project_exception_v2"
)
SHARED_PROJECT_EXCEPTION_STATUS = (
    "bounded_shared_project_risk_exception_for_authorized_step11_one_vm"
)
SHARED_PROJECT_LIVE_EVIDENCE_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_step11_live_iam_evidence_v1"
)
POST_CLAIM_REVOKE_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_step11_post_claim_revoke_v1"
)
RESULT_BACKEND_ID = "gcs-json-generation-pinned-step11-v1"

PROJECT = transport.PROJECT
ZONE = transport.ZONE
PROJECT_NUMBER = iam_gate.PROJECT_NUMBER
CONTROLLER_SERVICE_ACCOUNT = controller.CONTROLLER_SERVICE_ACCOUNT
CLAIM_KEY = launch.POSTCREATE_CLAIM_METADATA_KEY
RELEASE_STATE_KEY = "ofc-step11-claim-release-state"
BLOCK_PROJECT_SSH_KEYS = "block-project-ssh-keys"
INITIAL_FILE_KEYS = frozenset(launch.INITIAL_METADATA_FROM_FILE_KEYS)
STATIC_INITIAL_KEYS = frozenset({BLOCK_PROJECT_SSH_KEYS, RELEASE_STATE_KEY})
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_METADATA_VALUE_BYTES = 262_144
MAX_TOTAL_METADATA_BYTES = 512 * 1024
DEFAULT_OPERATION_TIMEOUT_SECONDS = 300
DEFAULT_DONE_TIMEOUT_SECONDS = 4_200
DEFAULT_ABSENCE_TIMEOUT_SECONDS = 180
DEFAULT_POLL_SECONDS = 2.0
LIVE_EVIDENCE_MAX_AGE_SECONDS = 300
MIN_EXECUTION_REMAINING_SECONDS = 5_600

_SAFE_NAME = re.compile(r"^[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_LIVE_EVIDENCE_FIELDS = {
    "schema",
    "collected_at_unix_seconds",
    "collected_via_get_only",
    "cloud_mutation_performed",
    "project",
    "bucket",
    "worker_service_account",
    "controller_service_account",
    "initiating_principal",
    "project_policy",
    "bucket_policy",
    "worker_service_account_policy",
    "controller_service_account_policy",
    "custom_roles",
    "enabled_services",
    "evidence_sha256",
}


class JsonClient(Protocol):
    """The narrow HTTP surface shared by the real and fake clients."""

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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _exact(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    if set(value) != fields:
        raise ValueError(f"{label} fields changed")


def _strict_int(
    value: Any,
    label: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if (
        type(value) is not int
        or (minimum is not None and value < minimum)
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"{label} is not an exact bounded integer")
    return value


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA256.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise ValueError(f"{label} is not a nonzero SHA-256")
    return value


def _strict_json(raw: bytes, label: str) -> dict[str, Any]:
    if not isinstance(raw, bytes) or len(raw) > MAX_JSON_BYTES:
        raise ValueError(f"{label} JSON response size changed")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} response is not JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON response is not an object")
    return value


def _request_json(
    client: JsonClient,
    *,
    method: str,
    url: str,
    body: Mapping[str, Any] | None = None,
    accepted: Sequence[int] = (200,),
) -> tuple[controller.HttpResponse, dict[str, Any]]:
    response = client.request(
        method=method,
        url=url,
        body=None if body is None else canonical_bytes(body),
        content_type=None if body is None else "application/json",
    )
    if response.status not in accepted:
        raise RuntimeError(
            f"{method} provider request returned HTTP {response.status}"
        )
    return response, _strict_json(response.body, "provider")


def _request_id(value: str | None = None) -> str:
    generated = str(uuid.uuid4()) if value is None else value
    try:
        parsed = uuid.UUID(generated)
    except (ValueError, AttributeError) as error:
        raise ValueError("Compute request ID is not UUID") from error
    if parsed.version != 4 or str(parsed) != generated.lower():
        raise ValueError("Compute request ID must be canonical UUID v4")
    return generated


def _instance_url(instance_name: str) -> str:
    if _SAFE_NAME.fullmatch(instance_name) is None:
        raise ValueError("instance name is unsafe")
    return (
        "https://compute.googleapis.com/compute/v1/projects/"
        f"{PROJECT}/zones/{ZONE}/instances/{instance_name}"
    )


def _operation_url(operation_name: str) -> str:
    if _SAFE_NAME.fullmatch(operation_name) is None:
        raise ValueError("zonal operation name is unsafe")
    return (
        "https://compute.googleapis.com/compute/v1/projects/"
        f"{PROJECT}/zones/{ZONE}/operations/{operation_name}"
    )


def _metadata_items(value: Any, *, label: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError(f"{label} metadata items are missing")
    rows: list[dict[str, str]] = []
    keys: list[str] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} metadata item is not an object")
        row = dict(raw)
        _exact(row, {"key", "value"}, f"{label} metadata item")
        key = row["key"]
        item_value = row["value"]
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(item_value, str)
            or len(item_value.encode("utf-8")) > MAX_METADATA_VALUE_BYTES
        ):
            raise ValueError(f"{label} metadata item shape changed")
        keys.append(key)
        rows.append({"key": key, "value": item_value})
    if len(keys) != len(set(keys)):
        raise ValueError(f"{label} metadata contains duplicate keys")
    return rows


def _metadata_map(items: Sequence[Mapping[str, str]]) -> dict[str, str]:
    rows = _metadata_items(list(items), label="provider")
    return {row["key"]: row["value"] for row in rows}


def _expected_initial_metadata_keys(
    transport_contract: Mapping[str, Any],
) -> frozenset[str]:
    metadata_values = transport_contract.get("metadata_values")
    if not isinstance(metadata_values, Mapping):
        raise ValueError("transport metadata values are missing")
    keys = frozenset(metadata_values)
    if (
        len(keys) != 8
        or any(not isinstance(key, str) or not key for key in keys)
        or keys & INITIAL_FILE_KEYS
        or keys & STATIC_INITIAL_KEYS
        or CLAIM_KEY in keys
    ):
        raise ValueError("transport metadata key contract changed")
    return frozenset((*keys, *INITIAL_FILE_KEYS, *STATIC_INITIAL_KEYS))


def _checked_file_value(
    path: str | Path,
    *,
    expected: Mapping[str, Any],
    label: str,
) -> str:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"{label} must be a real file")
    raw = source.read_bytes()
    if (
        len(raw) != expected.get("bytes")
        or _sha256_bytes(raw) != expected.get("sha256")
        or len(raw) > MAX_METADATA_VALUE_BYTES
    ):
        raise ValueError(f"{label} identity changed")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8") from error


def validate_package_provision_receipt(
    *,
    transport_contract: Mapping[str, Any],
    outer_root: str | Path,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild the package plan and verify every immutable generation."""

    checked_contract = transport.validate_job_contract(transport_contract)
    plan = controller.build_package_provision_plan(
        contract=checked_contract, outer_root=outer_root
    )
    candidate = dict(receipt)
    receipt_sha = _sha(
        candidate.pop("receipt_sha256", None), "package provision receipt"
    )
    if canonical_sha256(candidate) != receipt_sha:
        raise ValueError("package provision receipt digest changed")
    records = candidate.get("records")
    if not isinstance(records, list):
        raise ValueError("package provision receipt records are missing")
    expected = controller.build_package_provision_receipt(
        plan=plan, readbacks=records
    )
    if dict(receipt) != expected:
        raise ValueError("package provision receipt changed")
    if (
        set(expected["package_generations"])
        != {
            row["uri"]
            for row in checked_contract["remote_layout"][
                "package_inventory"
            ]["records"]
        }
        or any(
            type(generation) is not int or generation <= 0
            for generation in expected["package_generations"].values()
        )
    ):
        raise ValueError("package generations are incomplete")
    return expected


def validate_authoritative_preflight(
    *,
    artifact: Mapping[str, Any],
    transport_contract: Mapping[str, Any],
    now_unix_seconds: int,
) -> dict[str, Any]:
    """Revalidate the concrete GET-only bundle, never an arbitrary hash."""

    checked_contract = transport.validate_job_contract(transport_contract)
    value = dict(artifact)
    artifact_sha = _sha(
        value.pop("artifact_sha256", None), "actual preflight artifact"
    )
    if canonical_sha256(value) != artifact_sha:
        raise ValueError("actual preflight artifact digest changed")
    value["artifact_sha256"] = artifact_sha
    required = {
        "schema",
        "status",
        "stage_id",
        "retrieved_at_unix_seconds",
        "preview_identity",
        "outer_package_manifest",
        "direct_stage_identity",
        "read_only_preflight_plan",
        "observation_bundle",
        "receiver_preflight_result",
        "read_only_observation_passed",
        "launch_permission_ready",
        "credential_provenance",
        "access_token_recorded",
        "authorization_header_recorded",
        "collector_subprocess_used",
        "collector_gcloud_used",
        "cloud_mutation_performed",
        "claim_created",
        "authorization_created",
        "vm_created",
        "launch_authorized",
        "launch_ready",
        "diagnostic_only",
        "artifact_sha256",
    }
    _exact(value, required, "actual preflight artifact")
    if (
        value["schema"] != real_preflight.ACTUAL_RUN_SCHEMA
        or value["status"]
        != "pass_read_only_preflight_launch_still_unauthorized"
        or value["read_only_observation_passed"] is not True
        or value["launch_permission_ready"] is not False
        or value["diagnostic_only"] is not True
        or value["outer_package_manifest"]
        != checked_contract["outer_package_manifest"]
        or value["direct_stage_identity"]
        != checked_contract["direct_stage_identity"]
    ):
        raise ValueError("actual preflight artifact is not authoritative")
    for field in (
        "access_token_recorded",
        "authorization_header_recorded",
        "collector_subprocess_used",
        "collector_gcloud_used",
        "cloud_mutation_performed",
        "claim_created",
        "authorization_created",
        "vm_created",
        "launch_authorized",
        "launch_ready",
    ):
        if value[field] is not False:
            raise ValueError(f"actual preflight {field} changed")
    preview = checked_contract["adapter_preview"]
    identity = value["preview_identity"]
    if identity != {
        "package_manifest_sha256": preview["package_manifest_sha256"],
        "stage_identity_sha256": preview["stage_identity_sha256"],
        "stage_id": preview["stage_id"],
        "run_name": preview["run_name"],
    }:
        raise ValueError("actual preflight preview identity changed")
    plan = value["read_only_preflight_plan"]
    rebuilt_plan = receiver.build_read_only_preflight_plan(
        preview,
        outer_package_manifest=value["outer_package_manifest"],
        direct_stage_identity=value["direct_stage_identity"],
    )
    if plan != rebuilt_plan:
        raise ValueError("actual preflight plan changed")
    bundle = real_preflight.validate_observation_bundle(
        value["observation_bundle"],
        plan=plan,
        preview=preview,
        outer_package_manifest=value["outer_package_manifest"],
        direct_stage_identity=value["direct_stage_identity"],
        evaluation_unix_seconds=_strict_int(
            now_unix_seconds, "current time", minimum=1
        ),
    )
    if (
        bundle["concrete_stdlib_transport_used"] is not True
        or bundle["external_cloud_read_performed"] is not True
        or bundle["fixture_only"] is not False
        or bundle["read_only_observation_passed"] is not True
        or value["receiver_preflight_result"]
        != bundle["receiver_contract_result"]
        or value["receiver_preflight_result"][
            "observation_contract_passed"
        ]
        is not True
    ):
        raise ValueError("actual preflight lacks concrete external evidence")
    return value


def validate_authoritative_gate(
    *,
    plan: Mapping[str, Any],
    observation: Mapping[str, Any],
    transport_contract: Mapping[str, Any],
    now_unix_seconds: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Re-run the IAM/capacity gate against the exact contract and time."""

    checked_contract = transport.validate_job_contract(transport_contract)
    checked_plan = iam_gate.validate_step11_gate_plan(plan)
    source = checked_plan["source_contract"]
    binding = checked_contract["metadata_binding"]
    if (
        checked_plan["principals"]["controller_principal"]
        != f"serviceAccount:{CONTROLLER_SERVICE_ACCOUNT}"
        or source["direct_stage_identity_sha256"]
        != checked_contract["direct_stage_identity_sha256"]
        or source["outer_package_identity_sha256"]
        != checked_contract["outer_package_manifest"][
            "outer_package_identity_sha256"
        ]
        or source["metadata_binding_sha256"]
        != checked_contract["metadata_binding_sha256"]
        or source["instance_names"] != [binding["instance_name"]]
        or checked_plan["instance_contract"]["max_attempts"] != 1
        or checked_plan["instance_contract"]["allowed_launch_requests"][0][
            "attempt_index"
        ]
        != 0
    ):
        raise ValueError("IAM/capacity gate is not bound to exact attempt 0")
    result = iam_gate.validate_step11_gate(
        checked_plan,
        observation,
        evaluated_at_unix_seconds=_strict_int(
            now_unix_seconds, "current time", minimum=1
        ),
    )
    if (
        result["iam_ubla_capacity_gate_passed"] is not True
        or result["step11_prelaunch_ready"] is not True
        or result["failures"] != []
        or result["launch_authorized"] is not False
        or result["cloud_mutation_performed"] is not False
    ):
        raise ValueError("IAM/capacity gate did not pass")
    return checked_plan, result


def seal_shared_project_live_evidence(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    body = dict(value)
    if "evidence_sha256" in body:
        raise ValueError("live IAM evidence is already sealed")
    if set(body) != _LIVE_EVIDENCE_FIELDS - {"evidence_sha256"}:
        raise ValueError("live IAM evidence fields changed")
    return {**body, "evidence_sha256": canonical_sha256(body)}


def _policy_bindings(
    value: Any, *, label: str
) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} policy is missing")
    bindings = value.get("bindings", [])
    if not isinstance(bindings, list):
        raise ValueError(f"{label} policy bindings changed")
    normalized: list[dict[str, Any]] = []
    for raw in bindings:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} policy binding changed")
        role = raw.get("role")
        members = raw.get("members")
        condition = raw.get("condition")
        if (
            not isinstance(role, str)
            or not isinstance(members, list)
            or not members
            or any(not isinstance(member, str) for member in members)
            or (
                condition is not None
                and not isinstance(condition, Mapping)
            )
        ):
            raise ValueError(f"{label} policy binding shape changed")
        normalized.append(
            {
                "role": role,
                "members": sorted(members),
                "condition": (
                    dict(condition) if condition is not None else None
                ),
            }
        )
    return normalized


def _target_binding_triples(
    policy: Any,
    *,
    principals: set[str],
    label: str,
) -> set[tuple[str, str, str | None]]:
    triples: set[tuple[str, str, str | None]] = set()
    for binding in _policy_bindings(policy, label=label):
        targeted = principals.intersection(binding["members"])
        if not targeted:
            continue
        if len(targeted) != 1 or binding["members"] != sorted(targeted):
            raise ValueError(f"{label} targeted binding members changed")
        condition = binding["condition"]
        expression: str | None = None
        if condition is not None:
            expression = condition.get("expression")
            title = condition.get("title")
            if (
                not isinstance(expression, str)
                or not expression
                or not isinstance(title, str)
                or not title
                or set(condition)
                not in ({"title", "expression"}, {"title", "expression", "description"})
            ):
                raise ValueError(f"{label} binding condition changed")
        triples.add(
            (binding["role"], next(iter(targeted)), expression)
        )
    return triples


def validate_shared_project_live_evidence(
    value: Mapping[str, Any],
    *,
    transport_contract: Mapping[str, Any],
    actual_preflight_artifact: Mapping[str, Any],
    gate_plan: Mapping[str, Any],
    now_unix_seconds: int,
) -> dict[str, Any]:
    checked_contract = transport.validate_job_contract(transport_contract)
    checked_plan = iam_gate.validate_step11_gate_plan(gate_plan)
    evidence = dict(value)
    _exact(evidence, _LIVE_EVIDENCE_FIELDS, "live IAM evidence")
    raw_sha = evidence.pop("evidence_sha256")
    if _sha(raw_sha, "live IAM evidence") != canonical_sha256(evidence):
        raise ValueError("live IAM evidence SHA-256 changed")
    evidence["evidence_sha256"] = raw_sha
    now = _strict_int(now_unix_seconds, "current time", minimum=1)
    collected = _strict_int(
        evidence["collected_at_unix_seconds"],
        "live IAM evidence time",
        minimum=1,
    )
    if not 0 <= now - collected <= LIVE_EVIDENCE_MAX_AGE_SECONDS:
        raise ValueError("live IAM evidence is stale or from the future")
    source = checked_plan["source_contract"]
    worker_email = source["worker_service_account"]
    worker = f"serviceAccount:{worker_email}"
    controller_member = (
        f"serviceAccount:{CONTROLLER_SERVICE_ACCOUNT}"
    )
    if (
        evidence["schema"] != SHARED_PROJECT_LIVE_EVIDENCE_SCHEMA
        or evidence["collected_via_get_only"] is not True
        or evidence["cloud_mutation_performed"] is not False
        or evidence["project"] != PROJECT
        or evidence["bucket"] != source["bucket"]
        or evidence["worker_service_account"] != worker_email
        or evidence["controller_service_account"]
        != CONTROLLER_SERVICE_ACCOUNT
        or not isinstance(evidence["initiating_principal"], str)
        or not evidence["initiating_principal"].startswith("user:")
    ):
        raise ValueError("live IAM evidence identity changed")

    worker_bindings = checked_plan["iam_contract"]["worker_bindings"]
    controller_bindings = checked_plan["iam_contract"][
        "controller_bindings"
    ]
    expected_project = {
        (
            worker_bindings[2]["role"],
            worker,
            worker_bindings[2]["condition"]["expression"],
        ),
        *{
            (
                binding["role"],
                controller_member,
                binding["condition"]["expression"],
            )
            for binding in controller_bindings[:3]
        },
        (
            "roles/serviceusage.serviceUsageConsumer",
            controller_member,
            controller_bindings[0]["condition"]["expression"],
        ),
    }
    if _target_binding_triples(
        evidence["project_policy"],
        principals={worker, controller_member},
        label="project",
    ) != expected_project:
        raise ValueError("live targeted project IAM changed")

    if _target_binding_triples(
        evidence["worker_service_account_policy"],
        principals={controller_member},
        label="worker service account",
    ) != {
        (
            "roles/iam.serviceAccountUser",
            controller_member,
            controller_bindings[3]["condition"]["expression"],
        )
    }:
        raise ValueError("live worker actAs IAM changed")
    initiating = evidence["initiating_principal"]
    if _target_binding_triples(
        evidence["controller_service_account_policy"],
        principals={initiating},
        label="controller service account",
    ) != {
        (
            "roles/iam.serviceAccountTokenCreator",
            initiating,
            controller_bindings[0]["condition"]["expression"],
        )
    }:
        raise ValueError("live controller impersonation IAM changed")

    bucket_resource_prefix = (
        f"projects/_/buckets/{source['bucket']}/objects/"
    )
    package_resource = (
        bucket_resource_prefix + source["package_object_prefix"] + "/"
    )
    result_resource = (
        bucket_resource_prefix + source["result_object_prefix"] + "/"
    )
    time_expression = controller_bindings[0]["condition"]["expression"]
    expected_bucket = {
        (
            worker_bindings[0]["role"],
            worker,
            worker_bindings[0]["condition"]["expression"],
        ),
        (
            worker_bindings[1]["role"],
            worker,
            worker_bindings[1]["condition"]["expression"],
        ),
        (
            f"projects/{PROJECT}/roles/ofcM31T3ObjectReaderV1",
            controller_member,
            (
                f'(resource.name.startsWith("{package_resource}") || '
                f'resource.name.startsWith("{result_resource}")) && '
                f"{time_expression}"
            ),
        ),
    }
    if _target_binding_triples(
        evidence["bucket_policy"],
        principals={worker, controller_member},
        label="bucket",
    ) != expected_bucket:
        raise ValueError("live targeted bucket IAM changed")
    for binding in _policy_bindings(
        evidence["bucket_policy"], label="bucket"
    ):
        if {"allUsers", "allAuthenticatedUsers"}.intersection(
            binding["members"]
        ):
            raise ValueError("live bucket has a public principal")

    roles = evidence["custom_roles"]
    expected_roles = checked_plan["iam_contract"]["custom_roles"]
    if not isinstance(roles, Mapping) or set(roles) != set(expected_roles):
        raise ValueError("live custom role inventory changed")
    for key, expected in expected_roles.items():
        role = roles[key]
        if (
            not isinstance(role, Mapping)
            or role.get("name") != expected["name"]
            or sorted(role.get("includedPermissions", []))
            != expected["permissions"]
            or role.get("stage") != expected["stage"]
            or role.get("deleted", False) is not False
        ):
            raise ValueError(f"live custom role changed: {key}")
    enabled = evidence["enabled_services"]
    if (
        not isinstance(enabled, list)
        or any(not isinstance(name, str) for name in enabled)
        or not {
            "compute.googleapis.com",
            "storage.googleapis.com",
            "iamcredentials.googleapis.com",
            "serviceusage.googleapis.com",
        }.issubset(enabled)
    ):
        raise ValueError("required live service API is not enabled")

    project_bindings = _policy_bindings(
        evidence["project_policy"], label="project"
    )
    editor_members = {
        member
        for binding in project_bindings
        if binding["role"] == "roles/editor"
        for member in binding["members"]
    }
    if not {
        iam_gate.DEFAULT_COMPUTE_SERVICE_ACCOUNT_PRINCIPAL,
        iam_gate.CLOUD_SERVICES_SERVICE_ACCOUNT_PRINCIPAL,
    }.issubset(editor_members):
        raise ValueError("known shared-project Editor findings changed")

    checked_preflight = validate_authoritative_preflight(
        artifact=actual_preflight_artifact,
        transport_contract=checked_contract,
        now_unix_seconds=now,
    )
    facts = checked_preflight["observation_bundle"]["facts"]
    bucket_facts = facts["bucket_and_iam"]
    observed_worker_bindings = {
        (
            row["role"],
            row["member"],
            row["condition"]["expression"],
        )
        for row in bucket_facts["qualifying_conditioned_bindings"]
    }
    expected_worker_bindings = {
        (
            binding["role"],
            binding["member"],
            binding["condition"]["expression"],
        )
        for binding in worker_bindings[:2]
    }
    target_absence = {
        row["instance_name"]: row
        for row in facts["expected_instance_absence"]
    }
    target_name = checked_contract["metadata_binding"]["instance_name"]
    if (
        observed_worker_bindings != expected_worker_bindings
        or bucket_facts["uniform_bucket_level_access_enabled"] is not True
        or target_name not in target_absence
        or target_absence[target_name]["absent"] is not True
        or facts["regional_capacity"]["available_vcpu"] < 16
    ):
        raise ValueError("live preflight facts do not support the exception")
    return evidence


def build_shared_project_exception(
    *,
    transport_contract: Mapping[str, Any],
    actual_preflight_artifact: Mapping[str, Any],
    gate_plan: Mapping[str, Any],
    live_iam_evidence: Mapping[str, Any],
    now_unix_seconds: int,
) -> dict[str, Any]:
    """Record the known shared-project actAs risk without claiming gate pass."""

    checked_contract = transport.validate_job_contract(transport_contract)
    checked_plan = iam_gate.validate_step11_gate_plan(gate_plan)
    checked_live = validate_shared_project_live_evidence(
        live_iam_evidence,
        transport_contract=checked_contract,
        actual_preflight_artifact=actual_preflight_artifact,
        gate_plan=checked_plan,
        now_unix_seconds=now_unix_seconds,
    )
    preflight_sha = _sha(
        actual_preflight_artifact.get("artifact_sha256"),
        "actual preflight artifact",
    )
    binding = checked_contract["metadata_binding"]
    source = checked_plan["source_contract"]
    if (
        binding["attempt_index"] != 0
        or binding["source_role"] != "candidate"
        or checked_contract["adapter_preview"]["vm_count"] != 1
        or checked_plan["instance_contract"]["max_attempts"] != 1
        or checked_plan["instance_contract"]["max_concurrent_vms"] != 1
        or checked_plan["capacity_contract"]["target_instance_names"]
        != [binding["instance_name"]]
    ):
        raise ValueError("shared-project exception escaped exact attempt 0")
    body = {
        "schema": SHARED_PROJECT_EXCEPTION_SCHEMA,
        "status": SHARED_PROJECT_EXCEPTION_STATUS,
        "authorization_scope": (
            "user_authorized_step11_one_vm_lifecycle_smoke_only"
        ),
        "transport_contract_sha256": canonical_sha256(checked_contract),
        "actual_preflight_artifact_sha256": preflight_sha,
        "gate_plan_sha256": checked_plan["plan_sha256"],
        "live_iam_evidence_sha256": checked_live["evidence_sha256"],
        "stage_id": binding["stage_id"],
        "job_id": binding["job_id"],
        "source_role": "candidate",
        "attempt_index": 0,
        "vm_count": 1,
        "known_shared_project_findings": [
            "default_compute_sa_editor_may_act_as_worker",
            "cloud_services_sa_editor_may_act_as_worker",
        ],
        "strict_gate_unverified_checks": [
            "ancestor_deny_policy_effectiveness_not_proven",
            "effective_permission_simulation_not_run",
            "strict_gate_observation_not_collected",
        ],
        "bounded_mitigations": {
            "instance_name": binding["instance_name"],
            "package_prefix": source["package_prefix"],
            "stage_prefix": source["stage_prefix"],
            "result_prefix": source["result_prefix"],
            "max_runtime_seconds": 4_200,
            "external_ip_permitted": False,
            "worker_object_delete_permitted": False,
            "attempt1_authorized": False,
            "controller_and_worker_bindings_expire_at": checked_plan[
                "authorization_window"
            ]["expires_at_rfc3339"],
            "exact_binding_cleanup_required": True,
            "preexisting_editor_bindings_changed": False,
        },
        "strict_iam_gate_passed": False,
        "exception_authorizes_step12": False,
        "exception_authorizes_retry": False,
        "current_profile_changed": False,
        "cloud_mutation_performed": False,
    }
    return {**body, "exception_sha256": canonical_sha256(body)}


def validate_shared_project_exception(
    value: Mapping[str, Any],
    *,
    transport_contract: Mapping[str, Any],
    actual_preflight_artifact: Mapping[str, Any],
    gate_plan: Mapping[str, Any],
    live_iam_evidence: Mapping[str, Any],
    now_unix_seconds: int,
) -> dict[str, Any]:
    expected = build_shared_project_exception(
        transport_contract=transport_contract,
        actual_preflight_artifact=actual_preflight_artifact,
        gate_plan=gate_plan,
        live_iam_evidence=live_iam_evidence,
        now_unix_seconds=now_unix_seconds,
    )
    if dict(value) != expected:
        raise ValueError("shared-project exception record changed")
    return expected


def materialize_insert_body(
    *,
    transport_contract: Mapping[str, Any],
    launch_contract: Mapping[str, Any],
    authorization: Mapping[str, Any],
    public_key_record: Mapping[str, Any],
    startup_path: str | Path,
    prebootstrap_path: str | Path,
) -> dict[str, Any]:
    """Turn the offline blueprint into the exact valid GCE REST request body."""

    checked_contract = transport.validate_job_contract(transport_contract)
    checked_launch = launch.validate_launch_contract(
        launch_contract,
        transport_contract=checked_contract,
        controller_public_key_record=public_key_record,
        prebootstrap_path=prebootstrap_path,
        startup_path=startup_path,
    )
    binding = checked_contract["metadata_binding"]
    if (
        binding["attempt_index"] != 0
        or binding["source_role"] != "candidate"
        or checked_launch["attempt_index"] != 0
        or checked_launch["vm_count"] != 1
    ):
        raise ValueError("actual insert escaped stage1 candidate attempt 0")
    file_contract = checked_launch["metadata_from_file_contract"]
    static_records = file_contract["initial_static_records"]
    values = {
        "startup-script": _checked_file_value(
            startup_path,
            expected=static_records["startup-script"],
            label="startup script",
        ),
        "ofc-step11-transport-contract": canonical_bytes(
            checked_contract
        ).decode("ascii"),
        "ofc-step11-controller-authorization": canonical_bytes(
            authorization
        ).decode("ascii"),
        "ofc-step11-controller-public-key": canonical_bytes(
            transport.validate_rsa_public_key_record(public_key_record)
        ).decode("ascii"),
        "ofc-step11-prebootstrap": _checked_file_value(
            prebootstrap_path,
            expected=static_records["ofc-step11-prebootstrap"],
            label="prebootstrap",
        ),
        BLOCK_PROJECT_SSH_KEYS: "true",
        RELEASE_STATE_KEY: "pending-post-create",
        **dict(checked_contract["metadata_values"]),
    }
    # Validate every static canonical value against the frozen records.
    for key in (
        "ofc-step11-transport-contract",
        "ofc-step11-controller-public-key",
    ):
        raw = values[key].encode("utf-8")
        record = static_records[key]
        if (
            len(raw) != record["bytes"]
            or _sha256_bytes(raw) != record["sha256"]
        ):
            raise ValueError(f"{key} changed after launch freeze")
    if (
        set(values) != _expected_initial_metadata_keys(checked_contract)
        or CLAIM_KEY in values
        or len(values) != len(set(values))
    ):
        raise ValueError("initial instance metadata allowlist changed")
    for key, value in values.items():
        if (
            not isinstance(value, str)
            or len(value.encode("utf-8")) > MAX_METADATA_VALUE_BYTES
        ):
            raise ValueError(f"metadata value changed: {key}")
    if (
        sum(
            len(key.encode("utf-8")) + len(value.encode("utf-8"))
            for key, value in values.items()
        )
        > MAX_TOTAL_METADATA_BYTES
    ):
        raise ValueError("aggregate instance metadata escaped Compute limit")

    static = checked_launch["instance_insert"]["static_body"]
    body = {
        "name": static["name"],
        "machineType": static["machineType"],
        "canIpForward": False,
        "disks": static["disks"],
        "networkInterfaces": static["networkInterfaces"],
        "reservationAffinity": {"consumeReservationType": "NO_RESERVATION"},
        "scheduling": static["scheduling"],
        "serviceAccounts": static["serviceAccounts"],
        "metadata": {
            "items": [
                {"key": key, "value": values[key]}
                for key in sorted(values)
            ]
        },
    }
    if (
        set(body["metadata"]) != {"items"}
        or "scalarItems" in body["metadata"]
        or "metadataFromFileKeys" in body["metadata"]
        or "forbiddenAtInsert" in body["metadata"]
        or "labels" in body
        or "deletionProtection" in body
        or body["reservationAffinity"]
        != {"consumeReservationType": "NO_RESERVATION"}
        or body["networkInterfaces"][0].get("accessConfigs") != []
    ):
        raise ValueError("actual GCE insert body escaped least privilege")
    return body


def _operation_record(
    value: Mapping[str, Any],
    *,
    expected_instance_url: str,
) -> dict[str, Any]:
    name = value.get("name")
    status = value.get("status")
    target = value.get("targetLink")
    zone = value.get("zone")
    def compute_path(url: Any) -> str | None:
        if not isinstance(url, str):
            return None
        parsed = urllib.parse.urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.netloc
            not in {"compute.googleapis.com", "www.googleapis.com"}
            or parsed.query
            or parsed.fragment
            or not parsed.path.startswith("/compute/v1/projects/")
        ):
            return None
        return parsed.path

    expected_zone_path = f"/compute/v1/projects/{PROJECT}/zones/{ZONE}"
    if (
        not isinstance(name, str)
        or _SAFE_NAME.fullmatch(name) is None
        or status not in {"PENDING", "RUNNING", "DONE"}
        or compute_path(target) != compute_path(expected_instance_url)
        or compute_path(zone) != expected_zone_path
    ):
        raise RuntimeError("zonal operation identity changed")
    if status == "DONE" and value.get("error"):
        raise RuntimeError("zonal operation completed with an error")
    return {
        "name": name,
        "id": value.get("id"),
        "operation_type": value.get("operationType"),
        "target_link": target,
        "status": status,
        "insert_time": value.get("insertTime"),
        "start_time": value.get("startTime"),
        "end_time": value.get("endTime"),
        "http_error_status_code": value.get("httpErrorStatusCode"),
    }


def wait_zone_operation(
    *,
    client: JsonClient,
    initial: Mapping[str, Any],
    expected_instance_url: str,
    timeout_seconds: int = DEFAULT_OPERATION_TIMEOUT_SECONDS,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
) -> dict[str, Any]:
    """Poll the exact returned zonal operation until DONE."""

    _strict_int(timeout_seconds, "operation timeout", minimum=1, maximum=900)
    if not math.isfinite(poll_seconds) or not 0 < poll_seconds <= 10:
        raise ValueError("operation poll interval changed")
    record = _operation_record(
        initial, expected_instance_url=expected_instance_url
    )
    deadline = now() + timeout_seconds
    while record["status"] != "DONE":
        if now() >= deadline:
            raise TimeoutError("zonal operation timed out")
        sleep(min(poll_seconds, max(0.0, deadline - now())))
        _, current = _request_json(
            client,
            method="GET",
            url=_operation_url(record["name"]),
        )
        record = _operation_record(
            current, expected_instance_url=expected_instance_url
        )
    return record


def _validate_provider_instance(
    value: Mapping[str, Any],
    *,
    expected_name: str,
    expected_initial_metadata: Mapping[str, str],
) -> dict[str, Any]:
    instance_id = value.get("id")
    metadata = value.get("metadata")
    if (
        value.get("name") != expected_name
        or not isinstance(instance_id, str)
        or not instance_id.isdigit()
        or not isinstance(metadata, Mapping)
        or not isinstance(metadata.get("fingerprint"), str)
        or not metadata["fingerprint"]
    ):
        raise RuntimeError("provider instance identity changed")
    provider_metadata = _metadata_map(metadata.get("items"))
    if (
        provider_metadata != dict(expected_initial_metadata)
        or set(provider_metadata) != set(expected_initial_metadata)
        or CLAIM_KEY in provider_metadata
    ):
        raise RuntimeError("provider initial metadata changed")
    service_accounts = value.get("serviceAccounts")
    if (
        not isinstance(service_accounts, list)
        or len(service_accounts) != 1
        or service_accounts[0].get("email")
        != transport.WORKER_SERVICE_ACCOUNT
        or service_accounts[0].get("scopes")
        != [transport.REQUIRED_WORKER_OAUTH_SCOPE]
    ):
        raise RuntimeError("provider service account or OAuth scope changed")
    interfaces = value.get("networkInterfaces")
    if (
        not isinstance(interfaces, list)
        or len(interfaces) != 1
        or interfaces[0].get("accessConfigs", []) != []
    ):
        raise RuntimeError("provider instance acquired an external IP")
    scheduling = value.get("scheduling")
    if (
        not isinstance(scheduling, Mapping)
        or scheduling.get("provisioningModel") != "SPOT"
        or scheduling.get("instanceTerminationAction") != "DELETE"
        or scheduling.get("automaticRestart") is not False
        or scheduling.get("onHostMaintenance") != "TERMINATE"
    ):
        raise RuntimeError("provider scheduling contract changed")
    return {
        "instance_id": instance_id,
        "name": expected_name,
        "status": value.get("status"),
        "metadata_fingerprint": metadata["fingerprint"],
        "initial_metadata_sha256": canonical_sha256(provider_metadata),
        "service_account": service_accounts[0]["email"],
        "oauth_scopes": service_accounts[0]["scopes"],
        "external_access_configs": [],
        "provisioning_model": "SPOT",
    }


def build_claim_cas_body(
    *,
    provider_instance: Mapping[str, Any],
    expected_initial_metadata: Mapping[str, str],
    claim: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve the complete initial set and add only the signed claim."""

    metadata = provider_instance.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("provider metadata is missing")
    fingerprint = metadata.get("fingerprint")
    current = _metadata_map(metadata.get("items"))
    if (
        not isinstance(fingerprint, str)
        or not fingerprint
        or current != dict(expected_initial_metadata)
        or set(current) != set(expected_initial_metadata)
        or CLAIM_KEY in current
    ):
        raise ValueError("claim CAS provider metadata changed")
    claim_raw = canonical_bytes(claim).decode("ascii")
    updated = {**current, CLAIM_KEY: claim_raw}
    if (
        set(updated) != set(current) | {CLAIM_KEY}
        or any(updated[key] != value for key, value in current.items())
    ):
        raise ValueError("claim CAS changed initial metadata")
    return {
        "fingerprint": fingerprint,
        "items": [
            {"key": key, "value": updated[key]} for key in sorted(updated)
        ],
    }


def _validate_claimed_provider_instance(
    value: Mapping[str, Any],
    *,
    expected_name: str,
    expected_instance_id: str,
    expected_initial_metadata: Mapping[str, str],
    claim: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = value.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("claimed provider metadata is missing")
    actual = _metadata_map(metadata.get("items"))
    expected = {
        **dict(expected_initial_metadata),
        CLAIM_KEY: canonical_bytes(claim).decode("ascii"),
    }
    if (
        actual != expected
        or set(actual) != set(expected_initial_metadata) | {CLAIM_KEY}
    ):
        raise RuntimeError("provider claim metadata changed")
    initial_view = dict(value)
    initial_view["metadata"] = {
        "fingerprint": metadata.get("fingerprint"),
        "items": [
            {"key": key, "value": expected_initial_metadata[key]}
            for key in sorted(expected_initial_metadata)
        ],
    }
    provider = _validate_provider_instance(
        initial_view,
        expected_name=expected_name,
        expected_initial_metadata=expected_initial_metadata,
    )
    if provider["instance_id"] != expected_instance_id:
        raise RuntimeError("provider instance ID changed after claim CAS")
    return {
        **provider,
        "metadata_fingerprint": metadata["fingerprint"],
        "claimed_metadata_sha256": canonical_sha256(actual),
        "claim_present": True,
    }


def _gs_parts(uri: str) -> tuple[str, str]:
    if not isinstance(uri, str) or not uri.startswith("gs://"):
        raise ValueError("GCS URI must use gs://")
    bucket, separator, name = uri[5:].partition("/")
    if not separator or not bucket or not name:
        raise ValueError("GCS URI is incomplete")
    return bucket, name


def _gcs_metadata_url(uri: str) -> str:
    bucket, name = _gs_parts(uri)
    return (
        "https://storage.googleapis.com/storage/v1/b/"
        f"{urllib.parse.quote(bucket, safe='')}/o/"
        f"{urllib.parse.quote(name, safe='')}?fields=bucket,name,generation,"
        "metageneration,size,crc32c,etag"
    )


def _gcs_media_url(uri: str, generation: int) -> str:
    bucket, name = _gs_parts(uri)
    return (
        "https://storage.googleapis.com/download/storage/v1/b/"
        f"{urllib.parse.quote(bucket, safe='')}/o/"
        f"{urllib.parse.quote(name, safe='')}?alt=media&generation={generation}"
    )


def _gcs_list_url(prefix: str, page_token: str | None) -> str:
    bucket, name = _gs_parts(prefix)
    query = {
        "prefix": name.rstrip("/") + "/",
        "fields": (
            "nextPageToken,items(bucket,name,generation,metageneration,size,"
            "crc32c,etag)"
        ),
    }
    if page_token is not None:
        query["pageToken"] = page_token
    return (
        "https://storage.googleapis.com/storage/v1/b/"
        f"{urllib.parse.quote(bucket, safe='')}/o?"
        f"{urllib.parse.urlencode(query)}"
    )


def read_generation_pinned_object(
    *,
    client: JsonClient,
    uri: str,
    allow_missing: bool = False,
) -> tuple[dict[str, Any], bytes] | None:
    """GET metadata, then read that exact immutable object generation."""

    metadata_response = client.request(method="GET", url=_gcs_metadata_url(uri))
    if allow_missing and metadata_response.status == 404:
        return None
    if metadata_response.status != 200:
        raise RuntimeError(
            f"GCS metadata read returned HTTP {metadata_response.status}"
        )
    metadata = _strict_json(metadata_response.body, "GCS metadata")
    bucket, name = _gs_parts(uri)
    generation = metadata.get("generation")
    metageneration = metadata.get("metageneration")
    size = metadata.get("size")
    if (
        metadata.get("bucket") != bucket
        or metadata.get("name") != name
        or not isinstance(generation, str)
        or not generation.isdigit()
        or int(generation) <= 0
        or not isinstance(metageneration, str)
        or not metageneration.isdigit()
        or int(metageneration) <= 0
        or not isinstance(size, str)
        or not size.isdigit()
        or int(size) <= 0
        or not isinstance(metadata.get("crc32c"), str)
        or not metadata["crc32c"]
        or not isinstance(metadata.get("etag"), str)
        or not metadata["etag"]
    ):
        raise RuntimeError("GCS metadata identity changed")
    media = client.request(
        method="GET", url=_gcs_media_url(uri, int(generation))
    )
    if media.status != 200 or len(media.body) != int(size):
        raise RuntimeError("generation-pinned GCS media read changed")
    record = {
        "uri": uri,
        "generation": int(generation),
        "metageneration": int(metageneration),
        "bytes": len(media.body),
        "sha256": _sha256_bytes(media.body),
        "crc32c": metadata["crc32c"],
        "etag": metadata["etag"],
    }
    return record, media.body


class GenerationPinnedGcsBackend:
    """Read-only receiver backend with an optional separate list credential."""

    backend_id = RESULT_BACKEND_ID
    fixture_only = False

    def __init__(
        self,
        *,
        read_client: JsonClient,
        allowed_prefixes: Sequence[str],
        list_client: JsonClient | None = None,
    ) -> None:
        prefixes = tuple(prefix.rstrip("/") for prefix in allowed_prefixes)
        if (
            not prefixes
            or len(prefixes) != len(set(prefixes))
            or any(not prefix.startswith("gs://") for prefix in prefixes)
        ):
            raise ValueError("receiver GCS prefix allowlist changed")
        self._read_client = read_client
        self._list_client = list_client or read_client
        self._allowed_prefixes = prefixes
        self._cache: dict[tuple[str, int], bytes] = {}

    def _allowed(self, uri: str) -> bool:
        return any(
            uri == prefix or uri.startswith(prefix + "/")
            for prefix in self._allowed_prefixes
        )

    def list_prefix(self, prefix: str) -> Sequence[Mapping[str, Any]]:
        if prefix.rstrip("/") not in self._allowed_prefixes:
            raise ValueError("receiver list escaped exact tree prefix")
        listed: list[dict[str, Any]] = []
        page_token: str | None = None
        seen_tokens: set[str] = set()
        while True:
            response = self._list_client.request(
                method="GET", url=_gcs_list_url(prefix, page_token)
            )
            if response.status != 200:
                raise RuntimeError(
                    f"GCS object list returned HTTP {response.status}"
                )
            payload = _strict_json(response.body, "GCS object list")
            items = payload.get("items", [])
            if not isinstance(items, list):
                raise RuntimeError("GCS object list items changed")
            for item in items:
                if not isinstance(item, Mapping):
                    raise RuntimeError("GCS listed object is not an object")
                uri = f"gs://{item.get('bucket')}/{item.get('name')}"
                generation = item.get("generation")
                if (
                    not self._allowed(uri)
                    or not isinstance(generation, str)
                    or not generation.isdigit()
                ):
                    raise RuntimeError("GCS listed object escaped receiver")
                pinned = read_generation_pinned_object(
                    client=self._read_client, uri=uri
                )
                assert pinned is not None
                record, raw = pinned
                if (
                    record["generation"] != int(generation)
                    or str(record["metageneration"])
                    != item.get("metageneration")
                    or str(record["bytes"]) != item.get("size")
                    or record["crc32c"] != item.get("crc32c")
                    or record["etag"] != item.get("etag")
                ):
                    raise RuntimeError(
                        "GCS list and generation metadata disagree"
                    )
                self._cache[(uri, record["generation"])] = raw
                listed.append(record)
            token = payload.get("nextPageToken")
            if token is None:
                break
            if (
                not isinstance(token, str)
                or not token
                or token in seen_tokens
            ):
                raise RuntimeError("GCS list pagination changed")
            seen_tokens.add(token)
            page_token = token
        if len({row["uri"] for row in listed}) != len(listed):
            raise RuntimeError("GCS list returned duplicate object")
        return listed

    def read_bytes(self, uri: str, generation: int) -> bytes:
        if (
            not self._allowed(uri)
            or type(generation) is not int
            or generation <= 0
        ):
            raise ValueError("receiver read escaped exact generation")
        cached = self._cache.get((uri, generation))
        if cached is not None:
            return cached
        pinned = read_generation_pinned_object(
            client=self._read_client, uri=uri
        )
        assert pinned is not None
        record, raw = pinned
        if record["generation"] != generation:
            raise RuntimeError("receiver object generation changed")
        return raw


def _validate_cloud_materialization(
    *,
    value: Mapping[str, Any],
    transport_contract: Mapping[str, Any],
    receive_record: Mapping[str, Any],
) -> dict[str, Any]:
    checked_contract = transport.validate_job_contract(transport_contract)
    checked = receiver.validate_materialization_result(
        value,
        preview=checked_contract["adapter_preview"],
        receive=receive_record,
        outer_package_manifest=checked_contract[
            "outer_package_manifest"
        ],
        direct_stage_identity=checked_contract["direct_stage_identity"],
    )
    if (
        checked["backend_id"] != RESULT_BACKEND_ID
        or checked["backend_fixture_only"] is not False
        or checked["generation_bound_reads_performed"] is not True
        or checked["download_bytes_and_hash_verified"] is not True
        or checked["exact_prefix_inventory_verified"] is not True
        or checked["runner_validate_completed_output_performed"] is not True
        or checked["staging_artifact_remaining"] is not False
    ):
        raise ValueError("real cloud materialization evidence changed")
    return checked


def wait_for_done(
    *,
    client: JsonClient,
    preview: Mapping[str, Any],
    done_uri: str | None = None,
    instance_client: JsonClient | None = None,
    instance_name: str | None = None,
    timeout_seconds: int = DEFAULT_DONE_TIMEOUT_SECONDS,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Wait for DONE while failing fast if the exact VM becomes terminal."""

    _strict_int(timeout_seconds, "DONE timeout", minimum=1, maximum=4_500)
    if (instance_client is None) != (instance_name is None):
        raise ValueError("DONE monitor instance inputs must be paired")
    if (
        preview.get("attempt_index") != 0
        or preview.get("selected_job_ids") != ["candidate-shard-00"]
        or preview.get("vm_count") != 1
    ):
        raise ValueError("DONE monitor escaped exact Step 11 job")
    job = preview["jobs"][0]
    polled_done_uri = job["done_uri"] if done_uri is None else done_uri
    _gs_parts(polled_done_uri)
    expected_done_suffix = (
        f"/results/jobs/{job['job_id']}/DONE.envelope.json"
    )
    if not polled_done_uri.endswith(expected_done_suffix):
        raise ValueError("DONE monitor URI escaped exact Step 11 job")
    instance_url = (
        None if instance_name is None else _instance_url(instance_name)
    )
    deadline = now() + timeout_seconds

    def read_done_once() -> tuple[dict[str, Any], dict[str, Any]] | None:
        pinned = read_generation_pinned_object(
            client=client, uri=polled_done_uri, allow_missing=True
        )
        if pinned is None:
            return None
        record, raw = pinned
        done = _strict_json(raw, "worker DONE")
        receive = adapter.build_receive(
            preview, done_records=[done]
        )
        return record, receive

    while True:
        observed_done = read_done_once()
        if observed_done is not None:
            return observed_done
        if instance_client is not None and instance_url is not None:
            response = instance_client.request(method="GET", url=instance_url)
            if response.status == 404:
                # The worker publishes DONE and then immediately self-deletes.
                # A first GCS miss can race with that publication while the
                # following Compute read already observes absence.  Re-read
                # DONE once before classifying the lifecycle as failed.
                observed_done = read_done_once()
                if observed_done is not None:
                    return observed_done
                raise RuntimeError(
                    "worker instance became absent before publishing DONE"
                )
            if response.status != 200:
                raise RuntimeError(
                    "worker instance monitor returned HTTP "
                    f"{response.status}"
                )
            provider = _strict_json(response.body, "worker instance monitor")
            status = provider.get("status")
            if (
                provider.get("name") != instance_name
                or status
                not in {
                    "PROVISIONING",
                    "STAGING",
                    "RUNNING",
                    "STOPPING",
                    "SUSPENDING",
                    "SUSPENDED",
                    "REPAIRING",
                    "TERMINATED",
                }
            ):
                raise RuntimeError("worker instance monitor identity changed")
            if status in {
                "STOPPING",
                "SUSPENDING",
                "SUSPENDED",
                "TERMINATED",
            }:
                observed_done = read_done_once()
                if observed_done is not None:
                    return observed_done
                raise RuntimeError(
                    f"worker instance entered {status} before publishing DONE"
                )
        if now() >= deadline:
            raise TimeoutError("worker DONE timed out")
        sleep(min(poll_seconds, max(0.0, deadline - now())))


def wait_for_instance_absence(
    *,
    client: JsonClient,
    instance_name: str,
    timeout_seconds: int = DEFAULT_ABSENCE_TIMEOUT_SECONDS,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
) -> dict[str, Any]:
    """Require an exact provider GET 404, not a list-derived inference."""

    _strict_int(timeout_seconds, "absence timeout", minimum=1, maximum=600)
    url = _instance_url(instance_name)
    deadline = now() + timeout_seconds
    queries = 0
    while True:
        response = client.request(method="GET", url=url)
        queries += 1
        if response.status == 404:
            return {
                "instance_name": instance_name,
                "instance_url": url,
                "provider_get_status": 404,
                "query_count": queries,
                "all_expected_instances_absent": True,
            }
        if response.status != 200:
            raise RuntimeError(
                f"instance absence GET returned HTTP {response.status}"
            )
        if now() >= deadline:
            raise TimeoutError("worker instance did not self-delete")
        sleep(min(poll_seconds, max(0.0, deadline - now())))


def delete_exact_instance(
    *,
    client: JsonClient,
    instance_name: str,
    request_id: str | None = None,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Controller cleanup for the one exact instance; 404 is idempotent."""

    url = _instance_url(instance_name)
    response = client.request(
        method="DELETE",
        url=f"{url}?requestId={_request_id(request_id)}",
    )
    if response.status == 404:
        return {
            "instance_name": instance_name,
            "delete_requested": False,
            "already_absent": True,
            "provider_get_status": 404,
        }
    if response.status not in (200, 202):
        raise RuntimeError(
            f"controller cleanup delete returned HTTP {response.status}"
        )
    operation = wait_zone_operation(
        client=client,
        initial=_strict_json(response.body, "delete operation"),
        expected_instance_url=url,
        now=now,
        sleep=sleep,
    )
    absence = wait_for_instance_absence(
        client=client,
        instance_name=instance_name,
        now=now,
        sleep=sleep,
    )
    return {
        "instance_name": instance_name,
        "delete_requested": True,
        "already_absent": False,
        "operation": operation,
        "provider_get_status": absence["provider_get_status"],
    }


def _exact_direct_result_jobs(
    checked_contract: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    remote_layout = checked_contract.get("remote_layout")
    jobs = (
        remote_layout.get("jobs")
        if isinstance(remote_layout, Mapping)
        else None
    )
    metadata_binding = checked_contract.get("metadata_binding")
    preview = checked_contract.get("adapter_preview")
    selected_job_ids = (
        preview.get("selected_job_ids")
        if isinstance(preview, Mapping)
        else None
    )
    if (
        not isinstance(jobs, list)
        or len(jobs) != 1
        or not isinstance(jobs[0], Mapping)
        or not isinstance(metadata_binding, Mapping)
        or not isinstance(selected_job_ids, list)
        or len(selected_job_ids) != 1
        or jobs[0].get("job_id") != metadata_binding.get("job_id")
        or jobs[0].get("job_id") != selected_job_ids[0]
    ):
        raise ValueError("Step 11 direct result layout escaped exact job")
    return tuple(jobs)


@dataclass
class Step11ExecutionError(RuntimeError):
    """Failure plus the bounded cleanup evidence retained by the caller."""

    message: str
    cleanup: Mapping[str, Any] | None = None

    def __str__(self) -> str:
        return self.message


def execute_step11_attempt0(
    *,
    execute: bool,
    execution_confirmation: str,
    client: JsonClient | None,
    collector_client: JsonClient | None,
    transport_contract: Mapping[str, Any],
    launch_contract: Mapping[str, Any],
    signer: controller.EphemeralControllerKey,
    package_provision_receipt: Mapping[str, Any],
    outer_root: str | Path,
    actual_preflight_artifact: Mapping[str, Any],
    gate_plan: Mapping[str, Any],
    gate_observation: Mapping[str, Any] | None,
    live_iam_evidence: Mapping[str, Any] | None,
    post_claim_callback: Callable[
        [Mapping[str, Any]], Mapping[str, Any]
    ]
    | None,
    startup_path: str | Path,
    prebootstrap_path: str | Path,
    destination_root: str | Path,
    final_receipt_path: str | Path | None = None,
    shared_project_exception: Mapping[str, Any] | None = None,
    now_unix_seconds: Callable[[], int] = lambda: int(time.time()),
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    request_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the single authorized VM lifecycle.

    Package provisioning and IAM application are deliberately separate
    phases.  This function accepts only a revalidated generation receipt and a
    fresh, passing GET-only IAM/capacity observation.
    """

    if (
        execute is not True
        or execution_confirmation != EXECUTION_CONFIRMATION
        or client is None
    ):
        raise PermissionError(
            "Step 11 cloud execution requires explicit confirmation and client"
        )
    if (
        getattr(client, "_token_source", None) is not None
        and getattr(client._token_source, "controller_principal", None)
        != CONTROLLER_SERVICE_ACCOUNT
    ):
        raise PermissionError("Step 11 client is not the dedicated controller")
    if shared_project_exception is not None and post_claim_callback is None:
        raise PermissionError(
            "shared-project execution requires post-claim launch revocation"
        )
    ids = list(request_ids or [str(uuid.uuid4()) for _ in range(3)])
    if len(ids) != 3:
        raise ValueError("Step 11 requires three bounded request IDs")
    ids = [_request_id(value) for value in ids]
    checked_contract = transport.validate_job_contract(transport_contract)
    if (
        checked_contract["metadata_binding"]["attempt_index"] != 0
        or checked_contract["metadata_binding"]["source_role"] != "candidate"
        or checked_contract["adapter_preview"]["vm_count"] != 1
    ):
        raise ValueError("Step 11 execution accepts only attempt 0 candidate")
    direct_result_jobs = _exact_direct_result_jobs(checked_contract)
    direct_result_job = direct_result_jobs[0]
    now_seconds = _strict_int(
        now_unix_seconds(), "execution time", minimum=1
    )
    checked_preflight = validate_authoritative_preflight(
        artifact=actual_preflight_artifact,
        transport_contract=checked_contract,
        now_unix_seconds=now_seconds,
    )
    if shared_project_exception is None:
        if gate_observation is None:
            raise ValueError("passing IAM gate observation is missing")
        checked_gate, gate_result = validate_authoritative_gate(
            plan=gate_plan,
            observation=gate_observation,
            transport_contract=checked_contract,
            now_unix_seconds=now_seconds,
        )
        gate_mode = "strict_iam_ubla_capacity_gate_passed"
        gate_observation_sha256 = gate_result["observation_sha256"]
        gate_result_sha256 = gate_result["result_sha256"]
        gate_exception_sha256 = None
        live_iam_evidence_sha256 = None
    else:
        if gate_observation is not None:
            raise ValueError("gate pass and shared-project exception conflict")
        if live_iam_evidence is None:
            raise ValueError("shared-project live IAM evidence is missing")
        checked_gate = iam_gate.validate_step11_gate_plan(gate_plan)
        checked_exception = validate_shared_project_exception(
            shared_project_exception,
            transport_contract=checked_contract,
            actual_preflight_artifact=checked_preflight,
            gate_plan=checked_gate,
            live_iam_evidence=live_iam_evidence,
            now_unix_seconds=now_seconds,
        )
        window = checked_gate["authorization_window"]
        if not (
            window["issued_at_unix_seconds"]
            <= now_seconds
            < window["expires_at_unix_seconds"]
        ):
            raise ValueError("shared-project exception window is not active")
        gate_mode = "bounded_shared_project_exception_no_gate_pass_claim"
        gate_observation_sha256 = None
        gate_result_sha256 = None
        gate_exception_sha256 = checked_exception["exception_sha256"]
        live_iam_evidence_sha256 = checked_exception[
            "live_iam_evidence_sha256"
        ]
    package_receipt = validate_package_provision_receipt(
        transport_contract=checked_contract,
        outer_root=outer_root,
        receipt=package_provision_receipt,
    )
    window = checked_gate["authorization_window"]
    if (
        window["expires_at_unix_seconds"] - now_seconds
        < MIN_EXECUTION_REMAINING_SECONDS
    ):
        raise ValueError(
            "authorization window lacks bounded lifecycle cleanup margin"
        )
    authorization = controller.build_controller_authorization(
        contract=checked_contract,
        external_preflight_receipt_sha256=checked_preflight[
            "artifact_sha256"
        ],
        issued_unix_seconds=window["issued_at_unix_seconds"],
        expires_unix_seconds=window["expires_at_unix_seconds"],
        signer=signer,
    )
    insert_body = materialize_insert_body(
        transport_contract=checked_contract,
        launch_contract=launch_contract,
        authorization=authorization,
        public_key_record=signer.public_record,
        startup_path=startup_path,
        prebootstrap_path=prebootstrap_path,
    )
    initial_metadata = _metadata_map(insert_body["metadata"]["items"])
    instance_name = checked_contract["metadata_binding"]["instance_name"]
    instance_url = _instance_url(instance_name)
    insert_attempted = False
    cleanup: Mapping[str, Any] | None = None
    try:
        insert_url = (
            checked_launch_url(launch_contract)
            + f"?requestId={ids[0]}"
        )
        # A transport failure can occur after Compute accepted the insert but
        # before the response reached this process. From this point onward the
        # exact-name delete path must run even when no response is available.
        insert_attempted = True
        insert_response = client.request(
            method="POST",
            url=insert_url,
            body=canonical_bytes(insert_body),
            content_type="application/json",
        )
        if insert_response.status not in (200, 202):
            raise RuntimeError(
                f"instance insert returned HTTP {insert_response.status}"
            )
        insert_operation = wait_zone_operation(
            client=client,
            initial=_strict_json(insert_response.body, "insert operation"),
            expected_instance_url=instance_url,
            now=monotonic,
            sleep=sleep,
        )
        _, provider = _request_json(
            client, method="GET", url=instance_url
        )
        provider_record = _validate_provider_instance(
            provider,
            expected_name=instance_name,
            expected_initial_metadata=initial_metadata,
        )
        claim = controller.build_worker_claim(
            contract=checked_contract,
            authorization=authorization,
            project_number=PROJECT_NUMBER,
            instance_id=provider_record["instance_id"],
            package_generations=package_receipt["package_generations"],
            signer=signer,
        )
        transport.validate_controller_approval(
            contract=checked_contract,
            authorization=authorization,
            claim=claim,
            verifier=transport.RsaSha256ControllerTrustVerifier(
                signer.public_record
            ),
            now_unix_seconds=now_seconds,
        )
        cas_body = build_claim_cas_body(
            provider_instance=provider,
            expected_initial_metadata=initial_metadata,
            claim=claim,
        )
        set_response = client.request(
            method="POST",
            url=f"{instance_url}/setMetadata?requestId={ids[1]}",
            body=canonical_bytes(cas_body),
            content_type="application/json",
        )
        if set_response.status == 412:
            raise RuntimeError("claim metadata CAS precondition failed")
        if set_response.status not in (200, 202):
            raise RuntimeError(
                f"claim metadata update returned HTTP {set_response.status}"
            )
        claim_operation = wait_zone_operation(
            client=client,
            initial=_strict_json(set_response.body, "claim operation"),
            expected_instance_url=instance_url,
            now=monotonic,
            sleep=sleep,
        )
        _, claimed_provider = _request_json(
            client, method="GET", url=instance_url
        )
        claimed_provider_record = _validate_claimed_provider_instance(
            claimed_provider,
            expected_name=instance_name,
            expected_instance_id=provider_record["instance_id"],
            expected_initial_metadata=initial_metadata,
            claim=claim,
        )
        post_claim_revoke: Mapping[str, Any] | None = None
        if post_claim_callback is not None:
            callback_input = {
                "schema": (
                    "hu_m31_t3_step6d_rearm2_diagnostic_"
                    "step11_post_claim_callback_v1"
                ),
                "instance_name": instance_name,
                "provider_instance_id": provider_record["instance_id"],
                "claim_sha256": canonical_sha256(claim),
            }
            callback_result = dict(post_claim_callback(callback_input))
            expected_callback_fields = {
                "schema",
                "status",
                "instance_name",
                "provider_instance_id",
                "claim_sha256",
                "launch_binding_removed",
                "worker_actas_binding_removed",
                "readback_verified",
                "cloud_mutation_performed",
                "receipt_sha256",
            }
            _exact(
                callback_result,
                expected_callback_fields,
                "post-claim revoke receipt",
            )
            unsigned_callback = dict(callback_result)
            callback_sha = unsigned_callback.pop("receipt_sha256")
            if (
                callback_result["schema"] != POST_CLAIM_REVOKE_SCHEMA
                or callback_result["status"]
                != "launch_and_worker_actas_removed_after_claim"
                or callback_result["instance_name"] != instance_name
                or callback_result["provider_instance_id"]
                != provider_record["instance_id"]
                or callback_result["claim_sha256"]
                != canonical_sha256(claim)
                or callback_result["launch_binding_removed"] is not True
                or callback_result["worker_actas_binding_removed"] is not True
                or callback_result["readback_verified"] is not True
                or callback_result["cloud_mutation_performed"] is not True
                or _sha(callback_sha, "post-claim revoke receipt")
                != canonical_sha256(unsigned_callback)
            ):
                raise RuntimeError("post-claim launch revocation failed")
            post_claim_revoke = callback_result
        done_record, receive_record = wait_for_done(
            # A conditional object-reader binding cannot authorize GET on an
            # object that does not exist yet because resource.name is not
            # available for condition evaluation; GCS returns 403 rather than
            # 404.  The independently injected GET-only collector is therefore
            # used for the polling edge.  Existing result objects continue to
            # be generation-pinned and validated below.
            client=collector_client or client,
            preview=checked_contract["adapter_preview"],
            done_uri=direct_result_job["done_uri"],
            instance_client=client,
            instance_name=instance_name,
            now=monotonic,
            sleep=sleep,
        )
        tree_prefixes = [
            job["tree_prefix"]
            for job in direct_result_jobs
        ]
        backend = GenerationPinnedGcsBackend(
            read_client=client,
            list_client=collector_client,
            allowed_prefixes=tree_prefixes,
        )
        materialization = receiver.materialize_and_validate_received_stage(
            checked_contract["adapter_preview"],
            receive=receive_record,
            destination_root=destination_root,
            backend=backend,
            outer_package_manifest=checked_contract[
                "outer_package_manifest"
            ],
            direct_stage_identity=checked_contract[
                "direct_stage_identity"
            ],
        )
        materialization = _validate_cloud_materialization(
            value=materialization,
            transport_contract=checked_contract,
            receive_record=receive_record,
        )
        absence = wait_for_instance_absence(
            client=client,
            instance_name=instance_name,
            now=monotonic,
            sleep=sleep,
        )
        receipt_body = {
            "schema": FINAL_RECEIPT_SCHEMA,
            "status": (
                "step11_one_vm_attempt0_received_validated_and_absent"
            ),
            "diagnostic_only": True,
            "stage_id": checked_contract["metadata_binding"]["stage_id"],
            "run_name": checked_contract["metadata_binding"]["run_name"],
            "job_id": checked_contract["metadata_binding"]["job_id"],
            "source_role": "candidate",
            "attempt_index": 0,
            "vm_count": 1,
            "transport_contract_sha256": canonical_sha256(
                checked_contract
            ),
            "launch_contract_sha256": canonical_sha256(launch_contract),
            "actual_preflight_artifact_sha256": checked_preflight[
                "artifact_sha256"
            ],
            "gate_plan_sha256": checked_gate["plan_sha256"],
            "gate_mode": gate_mode,
            "gate_observation_sha256": gate_observation_sha256,
            "gate_result_sha256": gate_result_sha256,
            "gate_exception_sha256": gate_exception_sha256,
            "live_iam_evidence_sha256": live_iam_evidence_sha256,
            "strict_iam_gate_passed": (
                gate_mode == "strict_iam_ubla_capacity_gate_passed"
            ),
            "package_provision_receipt_sha256": package_receipt[
                "receipt_sha256"
            ],
            "package_generations_sha256": package_receipt[
                "package_generations_sha256"
            ],
            "authorization_sha256": canonical_sha256(authorization),
            "claim_sha256": canonical_sha256(claim),
            "controller_public_key_sha256": canonical_sha256(
                signer.public_record
            ),
            "controller_private_key_serialized": False,
            "controller_service_account": CONTROLLER_SERVICE_ACCOUNT,
            "provider_instance": provider_record,
            "provider_instance_after_claim": claimed_provider_record,
            "insert_request_body_sha256": canonical_sha256(insert_body),
            "insert_operation": insert_operation,
            "claim_cas_body_sha256": canonical_sha256(cas_body),
            "claim_operation": claim_operation,
            "post_claim_revoke_receipt_sha256": (
                post_claim_revoke["receipt_sha256"]
                if post_claim_revoke is not None
                else None
            ),
            "done_generation_record": done_record,
            "receive_sha256": canonical_sha256(receive_record),
            "materialization_result_sha256": receiver.canonical_sha256(
                materialization
            ),
            "runner_content_validated": True,
            "provider_absence": absence,
            "provider_get_404_observed": True,
            "worker_self_delete_observed": True,
            "controller_cleanup_delete_used": False,
            "cloud_mutation_performed": True,
            "current_profile_changed": False,
            "performance_lock_evidence": False,
            "quality_evidence": False,
            "training_eligible": False,
            "promotion_evidence": False,
        }
        receipt = {
            **receipt_body,
            "receipt_sha256": canonical_sha256(receipt_body),
        }
        if final_receipt_path is not None:
            controller.exclusive_write_json(final_receipt_path, receipt)
        return receipt
    except BaseException as error:
        if insert_attempted:
            try:
                cleanup = delete_exact_instance(
                    client=client,
                    instance_name=instance_name,
                    request_id=ids[2],
                    now=monotonic,
                    sleep=sleep,
                )
            except BaseException:
                cleanup = {
                    "instance_name": instance_name,
                    "delete_requested": True,
                    "cleanup_failed": True,
                }
        raise Step11ExecutionError(
            "Step 11 lifecycle failed; exact-instance cleanup was attempted",
            cleanup=cleanup,
        ) from error


def checked_launch_url(value: Mapping[str, Any]) -> str:
    """Return only the exact fixed insert URL from a validated-shape record."""

    url = value.get("instance_insert", {}).get("url")
    expected = (
        "https://compute.googleapis.com/compute/v1/projects/"
        f"{PROJECT}/zones/{ZONE}/instances"
    )
    if url != expected:
        raise ValueError("launch insert URL changed")
    return expected


__all__ = [
    "EXECUTION_CONFIRMATION",
    "FINAL_RECEIPT_SCHEMA",
    "GenerationPinnedGcsBackend",
    "MIN_EXECUTION_REMAINING_SECONDS",
    "POST_CLAIM_REVOKE_SCHEMA",
    "SHARED_PROJECT_EXCEPTION_SCHEMA",
    "SHARED_PROJECT_LIVE_EVIDENCE_SCHEMA",
    "Step11ExecutionError",
    "build_shared_project_exception",
    "build_claim_cas_body",
    "canonical_bytes",
    "canonical_sha256",
    "checked_launch_url",
    "delete_exact_instance",
    "execute_step11_attempt0",
    "materialize_insert_body",
    "read_generation_pinned_object",
    "seal_shared_project_live_evidence",
    "validate_authoritative_gate",
    "validate_authoritative_preflight",
    "validate_package_provision_receipt",
    "validate_shared_project_exception",
    "validate_shared_project_live_evidence",
    "wait_for_done",
    "wait_for_instance_absence",
    "wait_zone_operation",
]
