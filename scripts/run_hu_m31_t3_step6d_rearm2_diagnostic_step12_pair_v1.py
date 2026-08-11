#!/usr/bin/env python3
"""Run the explicitly confirmed Step 12 candidate/reference diagnostic pair.

The default mode is a local-only dry run.  The cloud path requires both
``--execute`` and the exact confirmation string exported by the Step 12 cloud
controller.  A run has no resume path: it creates one fresh output root, two
fresh attempt-0 identities, and at most two VMs.

This script intentionally reuses the proven Step 11 REST IAM administrator,
token sources, and generation-pinned storage primitives.  Step 12-specific
code adds the pair-wide claim/revoke boundary and independently convergent
cleanup for both exact VM and boot-disk names.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_10c2_real_readonly_preflight_v1
    as real_preflight,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_adapter
    as adapter,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as transport,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as canary_plan,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_cloud_controller_v1
    as step11_cloud,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_controller_v1
    as controller,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_rest_iam_admin_v1
    as rest_iam,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12_pair_cloud_controller_v1
    as pair_cloud,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12_pair_contract_v1
    as pair_contract_module,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12_pair_iam_capacity_gate_v1
    as pair_gate,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
STEP11_ROOT = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / "step11_one_vm_v12_fix3_actual"
)
PACKAGE_DIR = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / "rearm2_diagnostic_cloud_worker_package_v1"
)
WHEEL_PATH = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / "rearm2_diagnostic_10c2_local_preflight_v4"
    / "outer"
    / "wheels"
    / transport.EXPECTED_NUMPY_WHEEL_FILENAME
)
STARTUP_PATH = (
    REPO_ROOT
    / "scripts"
    / "bootstrap_hu_m31_t3_step6d_rearm2_diagnostic_step11_v1.sh"
)
PREBOOTSTRAP_PATH = (
    REPO_ROOT
    / "scripts"
    / "prebootstrap_hu_m31_t3_step6d_rearm2_diagnostic_step12_pair_v1.py"
)
BASE_PREBOOTSTRAP_PATH = (
    REPO_ROOT
    / "scripts"
    / "prebootstrap_hu_m31_t3_step6d_rearm2_diagnostic_step11_v1.py"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / "step12_pair_v1_actual"
)
POLICY_REGISTRY_PATH = REPO_ROOT / "src" / "ofc_regular" / "ai_profiles.py"
EXPECTED_POLICY_REGISTRY_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)
EXPECTED_OUTER_IDENTITY = (
    "593a1f1caaa8710811fc3044c32c66d681ccfe484095b9c94b3c6ba886aa6548"
)
NAT_ROUTER_RESOURCE = (
    "projects/ofc-solver-485418/regions/asia-northeast1/"
    "routers/ofc-t3-nat-router-asia-northeast1"
)
AUTHORIZATION_WINDOW_SECONDS = 7_200

RUNNER_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_step12_pair_runner_plan_v1"
)
PACKAGE_READBACK_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_step12_package_readback_v1"
)
CAPACITY_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_step12_live_capacity_v1"
)
AUTHORIZATION_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_step12_authorization_v1"
)
PRE_MUTATION_SECURITY_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12_pre_mutation_security_evidence_v1"
)
BOUNDED_EXCEPTION_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12_shared_project_bounded_exception_v1"
)
CLEANUP_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_step12_final_cleanup_v1"
)
REQUIRED_ENABLED_SERVICES = frozenset(
    {
        "compute.googleapis.com",
        "iamcredentials.googleapis.com",
        "serviceusage.googleapis.com",
        "storage.googleapis.com",
    }
)

_SAFE_NAME = re.compile(r"^[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)

_REST_ADMIN: rest_iam.Step11RestIamAdmin | None = None
_USER_COMPUTE_CLIENT: "ExactPairUserComputeClient | None" = None
_ACTIVE_GATE_PLAN: dict[str, Any] | None = None
_ACTIVE_OUTPUT_ROOT: Path | None = None
_POST_CLAIM_REVOKED = False
_CLOUD_MUTATION_STARTED = False


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"required JSON is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"required JSON is not an object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    controller.exclusive_write_json(path, value)


def _seal(body: Mapping[str, Any], *, field: str = "receipt_sha256") -> dict[str, Any]:
    checked = dict(body)
    if field in checked:
        raise ValueError("receipt body is already sealed")
    return {**checked, field: controller.canonical_sha256(checked)}


def _validate_sealed(
    value: Mapping[str, Any], *, field: str = "receipt_sha256"
) -> dict[str, Any]:
    checked = dict(value)
    digest = checked.pop(field, None)
    if (
        not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or controller.canonical_sha256(checked) != digest
    ):
        raise ValueError("sealed Step12 receipt digest changed")
    return {**checked, field: digest}


def _assert_profile_unchanged() -> None:
    if (
        not POLICY_REGISTRY_PATH.is_file()
        or POLICY_REGISTRY_PATH.is_symlink()
        or hashlib.sha256(POLICY_REGISTRY_PATH.read_bytes()).hexdigest()
        != EXPECTED_POLICY_REGISTRY_SHA256
    ):
        raise RuntimeError("current/profile registry hash changed")


def _validate_local_files() -> None:
    for path in (
        STARTUP_PATH,
        PREBOOTSTRAP_PATH,
        BASE_PREBOOTSTRAP_PATH,
        WHEEL_PATH,
    ):
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            raise ValueError(f"Step12 local input changed: {path}")
    if not PACKAGE_DIR.is_dir() or PACKAGE_DIR.is_symlink():
        raise ValueError("Step12 package mirror changed")


def _load_step11_runner_primitives() -> ModuleType:
    """Load the Step 11 runner as definitions only; its main guard stays inert."""

    path = (
        REPO_ROOT
        / "scripts"
        / "run_hu_m31_t3_step6d_rearm2_diagnostic_step11_one_vm_v1.py"
    )
    name = "_ofc_step11_runner_primitives"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Step11 runner primitives could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _stage1_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    stage1 = _read_json(STEP11_ROOT / "transport_contract.json")
    done = _read_json(STEP11_ROOT / "late_done_envelope.json")
    recovery = _read_json(STEP11_ROOT / "late_done_recovery_receipt.json")
    receive = adapter.build_receive(
        stage1["adapter_preview"],
        done_records=[done],
    )
    if (
        controller.canonical_sha256(receive) != recovery.get("receive_sha256")
        or recovery.get("status")
        != "step11_fix3_late_done_received_validated_after_self_delete_race"
        or recovery.get("provider_instance_and_disk_get_404") is not True
        or recovery.get("all_step11_targeted_iam_bindings_absent") is not True
        or recovery.get("current_profile_changed") is not False
        or recovery.get("policy_registry_sha256")
        != EXPECTED_POLICY_REGISTRY_SHA256
    ):
        raise ValueError("Step11 recovery prerequisite changed")
    return stage1, receive, recovery


def _build_local_bundle(
    *, issued_at_unix_seconds: int
) -> tuple[
    controller.EphemeralControllerKey,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    _assert_profile_unchanged()
    _validate_local_files()
    stage1, stage1_receive, recovery = _stage1_inputs()
    signer = controller.generate_ephemeral_controller_key(key_size=3_072)
    wheel_record = transport.build_offline_wheel_record(WHEEL_PATH)
    contracts = [
        transport.build_job_contract(
            package_dir=PACKAGE_DIR,
            stage_id=canary_plan.STAGE2_ID,
            job_id=job_id,
            attempt_index=0,
            offline_wheel_record=wheel_record,
            controller_public_key_record=signer.public_record,
            prerequisite_stage1_preview=stage1["adapter_preview"],
            prerequisite_stage1_receive=stage1_receive,
        )
        for job_id in canary_plan.STAGE2_JOB_IDS
    ]
    candidate, reference = contracts
    pair = pair_contract_module.build_pair_contract(
        candidate_transport_contract=candidate,
        reference_transport_contract=reference,
        stage1_recovery_receipt=recovery,
    )
    expires = issued_at_unix_seconds + AUTHORIZATION_WINDOW_SECONDS
    gate = pair_gate.build_step12_pair_gate_plan(
        pair,
        candidate_transport_contract=candidate,
        reference_transport_contract=reference,
        stage1_recovery_receipt=recovery,
        issued_at_unix_seconds=issued_at_unix_seconds,
        expires_at_unix_seconds=expires,
        nat_router_resource=NAT_ROUTER_RESOURCE,
    )
    if (
        gate["iam_contract"]["binding_count"] != 10
        or gate["capacity_contract"]["requested_c4_vcpu"] != 16
        or gate["instance_contract"]["max_concurrent_vms"] != 2
        or gate["instance_contract"]["max_attempts_per_job"] != 1
        or gate["instance_contract"]["attempt1_authorized"] is not False
        or pair["current_profile_sha256"]
        != EXPECTED_POLICY_REGISTRY_SHA256
        or candidate["outer_package_manifest"][
            "outer_package_identity_sha256"
        ]
        != EXPECTED_OUTER_IDENTITY
    ):
        raise RuntimeError("Step12 local execution boundary changed")
    return signer, candidate, reference, pair, gate, recovery


def _runner_plan(
    *,
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    pair: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    names = list(gate["instance_contract"]["authorized_instance_names"])
    direct_jobs = candidate["remote_layout"]["jobs"]
    body = {
        "schema": RUNNER_SCHEMA,
        "status": "local_pair_plan_validated_cloud_launch_not_authorized",
        "stage_id": canary_plan.STAGE2_ID,
        "run_name": canary_plan.STAGE2_RUN_NAME,
        "job_ids": list(canary_plan.STAGE2_JOB_IDS),
        "source_roles": ["candidate", "reference"],
        "attempt_index": 0,
        "instance_names": names,
        "actual_machine_type": "c4-standard-8",
        "requested_c4_vcpu": 16,
        "max_concurrent_vms": 2,
        "transport_contract_sha256s": [
            controller.canonical_sha256(candidate),
            controller.canonical_sha256(reference),
        ],
        "pair_contract_sha256": pair["pair_contract_sha256"],
        "gate_plan_sha256": gate["plan_sha256"],
        "outer_package_identity_sha256": candidate[
            "outer_package_manifest"
        ]["outer_package_identity_sha256"],
        "direct_stage_identity_sha256": candidate[
            "direct_stage_identity_sha256"
        ],
        "direct_stage_prefix": candidate["remote_layout"]["stage_prefix"],
        "direct_done_uris": [row["done_uri"] for row in direct_jobs],
        "binding_count": 10,
        "pre_mutation_live_custom_role_get_required": True,
        "pre_mutation_nat_get_required": True,
        "pre_mutation_enabled_services_get_required": True,
        "step11_exception_authorizes_step12": False,
        "step12_bounded_exception_required": True,
        "strict_effective_iam_simulation_passed": False,
        "ancestor_deny_policy_effectiveness_proven": False,
        "both_claims_required_before_revoke": True,
        "final_exact_instance_get_404_required": names,
        "final_exact_disk_get_404_required": names,
        "final_targeted_iam_binding_count_required": 0,
        "cloud_api_calls_performed": False,
        "cloud_mutation_performed": False,
        "launch_authorized": False,
        "diagnostic_only": True,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }
    return _seal(body, field="plan_sha256")


def _gs_parts(uri: str) -> tuple[str, str]:
    parsed = urllib.parse.urlsplit(uri)
    if (
        parsed.scheme != "gs"
        or parsed.netloc != transport.BUCKET
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/")
        or not parsed.path[1:]
    ):
        raise ValueError("GCS URI escaped Step12 bucket")
    return parsed.netloc, parsed.path[1:]


def _list_prefix_url(uri: str, *, page_token: str | None = None) -> str:
    bucket, prefix = _gs_parts(uri)
    query = {
        "prefix": prefix.rstrip("/") + "/",
        "fields": (
            "nextPageToken,items(bucket,name,generation,metageneration,"
            "size,crc32c,etag)"
        ),
        "maxResults": "1000",
    }
    if page_token is not None:
        query["pageToken"] = page_token
    return (
        "https://storage.googleapis.com/storage/v1/b/"
        f"{urllib.parse.quote(bucket, safe='')}/o?"
        f"{urllib.parse.urlencode(query)}"
    )


def _strict_response_json(
    response: controller.HttpResponse, *, label: str
) -> dict[str, Any]:
    if response.status != 200 or len(response.body) > 16 * 1024 * 1024:
        raise RuntimeError(f"{label} returned HTTP {response.status}")
    try:
        value = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError(f"{label} is not JSON") from None
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not an object")
    return value


def _require_direct_stage_prefix_empty(
    client: Any,
    *,
    stage_prefix: str,
    observed_at_unix_seconds: int | None = None,
) -> dict[str, Any]:
    if "/hu-m31-r2diag-direct-v1/stages/" not in stage_prefix:
        raise ValueError("Step12 stage prefix is not direct-v1")
    page_token: str | None = None
    page_count = 0
    while True:
        page_count += 1
        if page_count > 10:
            raise RuntimeError("Step12 exact-prefix pagination exceeded bound")
        response = client.request(
            method="GET",
            url=_list_prefix_url(stage_prefix, page_token=page_token),
        )
        value = _strict_response_json(
            response, label="Step12 direct-prefix inventory"
        )
        rows = value.get("items", [])
        if not isinstance(rows, list) or rows:
            raise FileExistsError("Step12 direct stage prefix is not exactly empty")
        token = value.get("nextPageToken")
        if token is None:
            break
        if (
            not isinstance(token, str)
            or not token
            or len(token) > 2_048
            or page_token == token
        ):
            raise RuntimeError("Step12 prefix pagination token changed")
        page_token = token
    body = {
        "schema": (
            "hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12_direct_prefix_empty_v1"
        ),
        "status": "fresh_direct_stage_prefix_exactly_empty",
        "stage_prefix": stage_prefix,
        "observed_at_unix_seconds": (
            int(time.time())
            if observed_at_unix_seconds is None
            else observed_at_unix_seconds
        ),
        "observation_max_age_seconds": (
            pair_gate.OBSERVATION_MAX_AGE_SECONDS
        ),
        "page_count": page_count,
        "object_count": 0,
        "collected_via_get_only": True,
        "cloud_mutation_performed": False,
    }
    return _seal(body)


def _package_generation_readback(
    client: Any, *, candidate: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, int]]:
    source = _read_json(STEP11_ROOT / "package_provision_receipt.json")
    source_unsigned = dict(source)
    source_digest = source_unsigned.pop("receipt_sha256", None)
    expected_records = {
        row["uri"]: row
        for row in candidate["remote_layout"]["package_inventory"]["records"]
    }
    generations = source.get("package_generations")
    if (
        source_digest != controller.canonical_sha256(source_unsigned)
        or source.get("outer_package_identity_sha256")
        != EXPECTED_OUTER_IDENTITY
        or source.get("all_generation_bound") is not True
        or source.get("all_bytes_and_sha256_read_back") is not True
        or not isinstance(generations, Mapping)
        or set(generations) != set(expected_records)
    ):
        raise ValueError("retained package receipt changed")
    records: list[dict[str, Any]] = []
    for uri in sorted(expected_records):
        expected = expected_records[uri]
        pinned_generation = generations[uri]
        if type(pinned_generation) is not int or pinned_generation <= 0:
            raise ValueError("retained package generation changed")
        result = step11_cloud.read_generation_pinned_object(
            client=client, uri=uri
        )
        if result is None:
            raise RuntimeError("retained package object disappeared")
        record, raw = result
        if (
            record["generation"] != pinned_generation
            or record["bytes"] != expected["bytes"]
            or record["sha256"] != expected["sha256"]
            or hashlib.sha256(raw).hexdigest() != expected["sha256"]
        ):
            raise RuntimeError("retained package generation/readback changed")
        records.append(record)
    body = {
        "schema": PACKAGE_READBACK_SCHEMA,
        "status": "retained_package_generation_pinned_readback_passed",
        "source_package_receipt_sha256": source["receipt_sha256"],
        "outer_package_identity_sha256": EXPECTED_OUTER_IDENTITY,
        "object_count": len(records),
        "records": records,
        "package_generations": dict(generations),
        "package_generations_sha256": controller.canonical_sha256(
            dict(generations)
        ),
        "collected_via_get_only": True,
        "cloud_mutation_performed": False,
    }
    return _seal(body), dict(generations)


def _integer_quota(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} quota is not numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError(f"{label} quota is not a nonnegative integer")
    return int(number)


def _quota_entry(
    values: Mapping[str, Any], *, metric: str, label: str
) -> dict[str, int]:
    rows = values.get("quotas")
    if not isinstance(rows, list):
        raise ValueError(f"{label} quota inventory is missing")
    matches = [row for row in rows if row.get("metric") == metric]
    if len(matches) != 1:
        raise ValueError(f"{label} quota metric {metric} changed")
    row = matches[0]
    limit = _integer_quota(row.get("limit"), label=f"{metric} limit")
    usage = _integer_quota(row.get("usage"), label=f"{metric} usage")
    if usage > limit:
        raise ValueError(f"{metric} usage exceeds limit")
    return {"limit": limit, "usage": usage, "available": limit - usage}


def _authoritative_capacity_target(
    gate: Mapping[str, Any],
) -> Any:
    names = gate["instance_contract"]["authorized_instance_names"]
    return real_preflight._Target(
        project=transport.PROJECT,
        bucket=transport.BUCKET,
        region=pair_gate.REGION,
        zone=transport.ZONE,
        machine_type="c4-standard-8",
        # These fields are deliberately inert: the fixed query allowlist below
        # contains only Cloud Quotas and aggregated Compute endpoints.
        package_object_prefix="capacity-only/package",
        stage_object_prefix="capacity-only/stage",
        expected_instance_names=tuple(names),
        service_account_email=transport.WORKER_SERVICE_ACCOUNT,
        image_project=real_preflight.IMAGE_PROJECT,
        image_name=real_preflight.IMAGE_NAME,
        image_id=real_preflight.IMAGE_ID,
        image_self_link=real_preflight.IMAGE_SELF_LINK,
    )


def _collect_authoritative_capacity_observations(
    token_source: Any,
    *,
    gate: Mapping[str, Any],
    collected_at_unix_seconds: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    target = _authoritative_capacity_target(gate)
    active_transport = real_preflight.StdlibGoogleJsonReadOnlyTransport()
    access_token = real_preflight._validate_token(
        token_source.access_token()
    )
    valid_until = (
        collected_at_unix_seconds + pair_gate.OBSERVATION_MAX_AGE_SECONDS
    )
    endpoint_ids = (
        "cloud_quotas_global_cpu",
        "cloud_quotas_c4_cpu",
        *real_preflight.GLOBAL_CPU_EMPTY_INVENTORY_ENDPOINTS,
    )
    observations: dict[str, Any] = {}
    transcripts: list[dict[str, Any]] = []
    for endpoint_id in endpoint_ids:
        observation, transcript = real_preflight._single_query(
            transport=active_transport,
            access_token=access_token,
            endpoint_id=endpoint_id,
            target=target,
            retrieved_at_unix_seconds=collected_at_unix_seconds,
            valid_until_unix_seconds=valid_until,
        )
        observations[endpoint_id] = observation
        transcripts.append(transcript)
    return observations, transcripts, active_transport.backend_id


def _authoritative_capacity_facts(
    observations: Mapping[str, Any],
    *,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    expected_endpoints = {
        "cloud_quotas_global_cpu",
        "cloud_quotas_c4_cpu",
        *real_preflight.GLOBAL_CPU_EMPTY_INVENTORY_ENDPOINTS,
    }
    if set(observations) != expected_endpoints:
        raise ValueError(
            "authoritative capacity observation endpoint set changed"
        )
    target = _authoritative_capacity_target(gate)
    inventory_facts: list[dict[str, Any]] = []
    for endpoint_id in real_preflight.GLOBAL_CPU_EMPTY_INVENTORY_ENDPOINTS:
        raw = observations[endpoint_id]
        if not isinstance(raw, Mapping):
            raise ValueError("authoritative Compute inventory changed")
        inventory_facts.append(
            real_preflight._empty_compute_inventory_facts(
                raw,
                endpoint_id=endpoint_id,
                target=target,
            )
        )
    global_facts = real_preflight._global_cpu_quota_facts(
        observations["cloud_quotas_global_cpu"],
        inventory_facts,
        target=target,
        project_number=pair_gate.PROJECT_NUMBER,
    )
    c4_facts = real_preflight._regional_c4_quota_facts(
        observations["cloud_quotas_c4_cpu"],
        target=target,
        project_number=pair_gate.PROJECT_NUMBER,
        usage_inventory_proof=global_facts["usage_inventory_proof"],
    )
    instance_inventory = observations["compute_aggregated_instances"]
    records: list[dict[str, Any]] = []
    for scope, scope_value in instance_inventory["items"].items():
        if "instances" not in scope_value:
            continue
        records.extend(
            real_preflight._instance_inventory_record(
                row,
                scope=scope,
                target=target,
            )
            for row in scope_value["instances"]
        )
    return {
        "global_cpu": global_facts,
        "regional_c4": c4_facts,
        "inventory_facts": inventory_facts,
        "instance_records": records,
    }


def _capacity_transcript_digest_evidence(
    observations: Mapping[str, Any],
    transcripts: Sequence[Mapping[str, Any]],
    *,
    external_read: bool,
) -> dict[str, Any]:
    expected_endpoints = set(observations)
    if external_read:
        by_endpoint: dict[str, str] = {}
        for row in transcripts:
            if not isinstance(row, Mapping):
                raise ValueError("capacity transcript is not an object")
            endpoint_id = row.get("endpoint_id")
            digest = row.get("transcript_sha256")
            unsigned = dict(row)
            unsigned.pop("transcript_sha256", None)
            if (
                row.get("schema") != real_preflight.TRANSCRIPT_SCHEMA
                or not isinstance(endpoint_id, str)
                or endpoint_id not in expected_endpoints
                or endpoint_id in by_endpoint
                or not isinstance(digest, str)
                or _SHA256.fullmatch(digest) is None
                or real_preflight.canonical_sha256(unsigned) != digest
                or row.get("page_tokens_exhausted") is not True
                or row.get("cloud_mutation_performed") is not False
                or row.get("normalized_observation_sha256")
                != real_preflight.canonical_sha256(
                    observations[endpoint_id]
                )
            ):
                raise ValueError("capacity transcript evidence changed")
            by_endpoint[endpoint_id] = digest
        evidence_kind = "provider_get_transcript_sha256"
    else:
        if transcripts:
            raise ValueError("fixture capacity transcripts are forbidden")
        by_endpoint = {
            endpoint_id: real_preflight.canonical_sha256(observation)
            for endpoint_id, observation in observations.items()
        }
        evidence_kind = "fixture_observation_sha256"
    if set(by_endpoint) != expected_endpoints or len(by_endpoint) != 6:
        raise ValueError("capacity transcript endpoint set changed")
    ordered = dict(sorted(by_endpoint.items()))
    return {
        "evidence_kind": evidence_kind,
        "endpoint_count": len(ordered),
        "sha256_by_endpoint": ordered,
        "aggregate_sha256": controller.canonical_sha256(ordered),
    }


def _collect_fresh_capacity(
    step11_runner: ModuleType,
    *,
    gate: Mapping[str, Any],
    collected_at_unix_seconds: int,
    token_source: Any | None = None,
    authoritative_observations: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    project = transport.PROJECT
    zone = transport.ZONE
    region = pair_gate.REGION
    capacity_contract = gate.get("capacity_contract")
    if (
        not isinstance(capacity_contract, Mapping)
        or capacity_contract.get("quota_metrics")
        != [
            "CPUS",
            "CPUS_ALL_REGIONS",
            "CPUS_PER_VM_FAMILY_C4",
            "PREEMPTIBLE_CPUS",
        ]
        or capacity_contract.get("authoritative_inventory_endpoints")
        != list(real_preflight.GLOBAL_CPU_EMPTY_INVENTORY_ENDPOINTS)
        or capacity_contract.get(
            "authoritative_inventory_complete_required"
        )
        is not True
        or capacity_contract.get(
            "authoritative_inventory_same_scope_set_required"
        )
        is not True
        or capacity_contract.get(
            "authoritative_inventory_unreachable_count_required"
        )
        != 0
        or capacity_contract.get(
            "authoritative_inventory_page_tokens_exhausted_required"
        )
        is not True
        or capacity_contract.get(
            "authoritative_noninstance_inventory_empty_required"
        )
        is not True
        or capacity_contract.get(
            "authoritative_transcript_endpoint_digest_map_required"
        )
        is not True
    ):
        raise RuntimeError(
            "Step12 authoritative capacity evidence contract changed"
        )
    machine = step11_runner._gcloud_json(
        [
            "compute",
            "machine-types",
            "describe",
            "c4-standard-8",
            f"--project={project}",
            f"--zone={zone}",
        ]
    )
    regional = step11_runner._gcloud_json(
        [
            "compute",
            "regions",
            "describe",
            region,
            f"--project={project}",
        ]
    )
    if (
        not isinstance(machine, Mapping)
        or machine.get("name") != "c4-standard-8"
        or _integer_quota(machine.get("guestCpus"), label="machine vCPU") != 8
        or not isinstance(regional, Mapping)
        or regional.get("status") != "UP"
    ):
        raise RuntimeError("fresh capacity provider shape changed")
    if authoritative_observations is None:
        if token_source is None:
            raise RuntimeError(
                "authoritative capacity token source is required"
            )
        (
            observations,
            transcripts,
            backend_id,
        ) = _collect_authoritative_capacity_observations(
            token_source,
            gate=gate,
            collected_at_unix_seconds=collected_at_unix_seconds,
        )
        external_read = True
    else:
        observations = dict(authoritative_observations)
        transcripts = []
        backend_id = "fixture-injected-authoritative-shape-v1"
        external_read = False
    authoritative = _authoritative_capacity_facts(
        observations, gate=gate
    )
    transcript_evidence = _capacity_transcript_digest_evidence(
        observations, transcripts, external_read=external_read
    )
    global_facts = authoritative["global_cpu"]
    c4_facts = authoritative["regional_c4"]
    records = authoritative["instance_records"]
    global_limit = int(global_facts["limit"])
    global_usage = int(global_facts["usage"])
    global_available = int(global_facts["available"])
    c4_limit = int(c4_facts["limit"])
    c4_usage = int(c4_facts["usage"])
    c4_available = int(c4_facts["available"])
    target_names = set(
        gate["instance_contract"]["authorized_instance_names"]
    )
    collisions = sorted(
        row["name"] for row in records if row["name"] in target_names
    )
    interfering = sorted(
        row["name"]
        for row in records
        if row["target_region_c4"] and row["status"] != "TERMINATED"
    )
    if collisions or interfering:
        raise RuntimeError("fresh capacity inventory is not isolated")
    quota = {
        "CPUS": _quota_entry(
            regional,
            metric="CPUS",
            label="region",
        ),
        "C4_CPUS_PER_VM_FAMILY": {
            "limit": c4_limit,
            "usage": c4_usage,
            "available": c4_available,
            "limit_source": (
                "cloudquotas.googleapis.com/v1/quotaInfos.get"
            ),
            "usage_source": (
                "compute.googleapis.com/compute/v1/aggregated:"
                "conservative-target-region-c4-instance-vcpu-upper-bound-"
                "with-empty-reservations-nodeGroups-futureReservations"
            ),
        },
        "CPUS_ALL_REGIONS": {
            "limit": global_limit,
            "usage": global_usage,
            "available": global_available,
            "limit_source": (
                "cloudquotas.googleapis.com/v1/quotaInfos.get"
            ),
            "usage_source": (
                "compute.googleapis.com/compute/v1/aggregated:"
                "conservative-standard-instance-vcpu-upper-bound-with-"
                "empty-reservations-nodeGroups-futureReservations"
            ),
        },
        "PREEMPTIBLE_CPUS": _quota_entry(
            regional, metric="PREEMPTIBLE_CPUS", label="region"
        ),
    }
    available = min(row["available"] for row in quota.values())
    requested = gate["capacity_contract"]["requested_c4_vcpu"]
    if requested != 16 or available < requested:
        raise RuntimeError("fresh Step12 C4/Spot quota is insufficient")
    body = {
        "schema": CAPACITY_SCHEMA,
        "status": "fresh_get_only_capacity_covers_exact_pair",
        "collected_at_unix_seconds": collected_at_unix_seconds,
        "observation_max_age_seconds": pair_gate.OBSERVATION_MAX_AGE_SECONDS,
        "project": project,
        "zone": zone,
        "region": region,
        "machine_type": "c4-standard-8",
        "machine_vcpu": 8,
        "requested_concurrent_vms": 2,
        "requested_c4_vcpu": requested,
        "quota": quota,
        "available_vcpu": available,
        "target_instance_names": sorted(target_names),
        "target_name_collisions": collisions,
        "nonterminated_c4_interference": interfering,
        "global_cpu_inventory_proof": global_facts[
            "usage_inventory_proof"
        ],
        "global_cpu_inventory_endpoint_count": len(
            authoritative["inventory_facts"]
        ),
        "global_cpu_inventory_resource_counts": {
            row["resource_collection"]: row["resource_count"]
            for row in authoritative["inventory_facts"]
        },
        "inventory_fully_enumerated": True,
        "authoritative_inventory_contract_satisfied": True,
        "authoritative_capacity_backend_id": backend_id,
        "authoritative_capacity_transcript_evidence": transcript_evidence,
        "authoritative_external_cloud_read_performed": external_read,
        "spot_stock_proven": False,
        "spot_stock_only_provable_by_insert": True,
        "collected_via_get_only": True,
        "cloud_mutation_performed": False,
    }
    return _seal(body)


def _validate_live_custom_roles(
    *,
    gate: Mapping[str, Any],
    observed_roles: Mapping[str, Any],
) -> dict[str, Any]:
    expected = gate["iam_contract"]["custom_roles"]
    if set(observed_roles) != set(expected):
        raise ValueError("live Step12 custom-role key set changed")
    normalized: dict[str, Any] = {}
    for key in sorted(expected):
        wanted = expected[key]
        observed = observed_roles[key]
        if not isinstance(observed, Mapping):
            raise ValueError("live Step12 custom role is not an object")
        permissions = observed.get("includedPermissions")
        if (
            observed.get("name") != wanted["name"]
            or observed.get("stage") != wanted["stage"]
            or not isinstance(permissions, list)
            or sorted(permissions) != sorted(wanted["permissions"])
            or len(permissions) != len(set(permissions))
            or observed.get("deleted") is True
        ):
            raise ValueError(f"live Step12 custom role drifted: {key}")
        normalized[key] = {
            "name": observed["name"],
            "stage": observed["stage"],
            "included_permissions": sorted(permissions),
            "etag": observed.get("etag"),
            "provider_record_sha256": controller.canonical_sha256(
                dict(observed)
            ),
        }
    return normalized


def _validate_nat_router(
    *, gate: Mapping[str, Any], observed_router: Mapping[str, Any]
) -> dict[str, Any]:
    capacity = gate["capacity_contract"]
    expected_router = capacity["nat_router_resource"].rsplit("/", 1)[-1]
    expected_network_path = (
        f"/compute/v1/projects/{transport.PROJECT}/global/networks/default"
    )
    region = observed_router.get("region")
    network = observed_router.get("network")
    nats = observed_router.get("nats")
    matches = (
        [
            row
            for row in nats
            if isinstance(row, Mapping)
            and row.get("name") == capacity["nat_name"]
        ]
        if isinstance(nats, list)
        else []
    )
    if (
        observed_router.get("name") != expected_router
        or not isinstance(region, str)
        or region.rsplit("/", 1)[-1] != capacity["region"]
        or not isinstance(network, str)
        or urllib.parse.urlsplit(network).path != expected_network_path
        or len(matches) != 1
    ):
        raise ValueError("live Step12 Cloud NAT router identity changed")
    nat = matches[0]
    if (
        nat.get("sourceSubnetworkIpRangesToNat")
        != capacity["nat_source_subnetwork_ip_ranges"]
        or nat.get("natIpAllocateOption") != "AUTO_ONLY"
    ):
        raise ValueError("live Step12 Cloud NAT configuration changed")
    return {
        "router_name": expected_router,
        "router_resource": capacity["nat_router_resource"],
        "region": capacity["region"],
        "network_resource": capacity["network_resource"],
        "nat_name": capacity["nat_name"],
        "source_subnetwork_ip_ranges": nat[
            "sourceSubnetworkIpRangesToNat"
        ],
        "nat_ip_allocate_option": nat["natIpAllocateOption"],
        "provider_record_sha256": controller.canonical_sha256(
            dict(observed_router)
        ),
    }


def _validate_enabled_services(
    observed_services: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled: set[str] = set()
    for row in observed_services:
        config = row.get("config") if isinstance(row, Mapping) else None
        name = config.get("name") if isinstance(config, Mapping) else None
        if row.get("state") != "ENABLED" or not isinstance(name, str):
            raise ValueError("enabled-service inventory row changed")
        if name in enabled:
            raise ValueError("enabled-service inventory is duplicated")
        enabled.add(name)
    missing = sorted(REQUIRED_ENABLED_SERVICES - enabled)
    if missing:
        raise ValueError(f"required Step12 services are not enabled: {missing}")
    return {
        "required_services": sorted(REQUIRED_ENABLED_SERVICES),
        "enabled_services": sorted(enabled),
        "missing_required_services": [],
        "required_services_enabled": True,
    }


def _validate_step11_does_not_authorize_step12(
    value: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = (
        _read_json(STEP11_ROOT / "shared_project_exception.json")
        if value is None
        else dict(value)
    )
    checked = dict(source)
    digest = checked.pop("exception_sha256", None)
    if (
        digest != controller.canonical_sha256(checked)
        or source.get("schema")
        != step11_cloud.SHARED_PROJECT_EXCEPTION_SCHEMA
        or source.get("status")
        != step11_cloud.SHARED_PROJECT_EXCEPTION_STATUS
        or source.get("authorization_scope")
        != "user_authorized_step11_one_vm_lifecycle_smoke_only"
        or source.get("strict_iam_gate_passed") is not False
        or source.get("exception_authorizes_step12") is not False
        or source.get("exception_authorizes_retry") is not False
        or source.get("vm_count") != 1
        or source.get("attempt_index") != 0
        or source.get("current_profile_changed") is not False
    ):
        raise ValueError("Step11 exception/non-authorization record changed")
    unverified = source.get("strict_gate_unverified_checks")
    if (
        not isinstance(unverified, list)
        or "ancestor_deny_policy_effectiveness_not_proven" not in unverified
        or "effective_permission_simulation_not_run" not in unverified
    ):
        raise ValueError("Step11 strict-gate limitations changed")
    return {
        "schema": source["schema"],
        "status": source["status"],
        "exception_sha256": source["exception_sha256"],
        "authorization_scope": source["authorization_scope"],
        "strict_iam_gate_passed": False,
        "exception_authorizes_step12": False,
        "exception_authorizes_retry": False,
        "strict_gate_unverified_checks": list(unverified),
    }


def _collect_pre_mutation_security_evidence(
    step11_runner: ModuleType,
    *,
    user_token: Any,
    gate: Mapping[str, Any],
    collected_at_unix_seconds: int,
) -> dict[str, Any]:
    evidence_client = step11_runner.ActiveUserEvidenceClient(user_token)
    observed_roles: dict[str, Any] = {}
    for key, expected in sorted(
        gate["iam_contract"]["custom_roles"].items()
    ):
        role_id = expected["name"].rsplit("/", 1)[-1]
        observed_roles[key] = evidence_client.get_json(
            "https://iam.googleapis.com/v1/projects/"
            f"{transport.PROJECT}/roles/"
            f"{urllib.parse.quote(role_id, safe='')}"
        )
    custom_roles = _validate_live_custom_roles(
        gate=gate, observed_roles=observed_roles
    )

    raw_services: list[Mapping[str, Any]] = []
    seen_tokens: set[str] = set()
    page_token: str | None = None
    for _ in range(10):
        query = {"filter": "state:ENABLED", "pageSize": "200"}
        if page_token is not None:
            query["pageToken"] = page_token
        page = evidence_client.get_json(
            "https://serviceusage.googleapis.com/v1/projects/"
            f"{transport.PROJECT}/services?"
            f"{urllib.parse.urlencode(query)}"
        )
        rows = page.get("services", [])
        if not isinstance(rows, list) or any(
            not isinstance(row, Mapping) for row in rows
        ):
            raise ValueError("enabled-service inventory changed")
        raw_services.extend(rows)
        next_token = page.get("nextPageToken")
        if next_token is None:
            break
        if (
            not isinstance(next_token, str)
            or not next_token
            or len(next_token) > 2_048
            or next_token in seen_tokens
        ):
            raise ValueError("enabled-service pagination changed")
        seen_tokens.add(next_token)
        page_token = next_token
    else:
        raise ValueError("enabled-service pagination exceeded bound")
    services = _validate_enabled_services(raw_services)

    router_name = gate["capacity_contract"][
        "nat_router_resource"
    ].rsplit("/", 1)[-1]
    router = step11_runner._gcloud_json(
        [
            "compute",
            "routers",
            "describe",
            router_name,
            f"--project={transport.PROJECT}",
            f"--region={pair_gate.REGION}",
        ]
    )
    if not isinstance(router, Mapping):
        raise ValueError("live Cloud NAT router is not an object")
    nat = _validate_nat_router(gate=gate, observed_router=router)
    step11_non_authorization = (
        _validate_step11_does_not_authorize_step12()
    )
    body = {
        "schema": PRE_MUTATION_SECURITY_SCHEMA,
        "status": (
            "live_roles_nat_services_verified_step11_still_non_authorizing"
        ),
        "collected_at_unix_seconds": collected_at_unix_seconds,
        "gate_plan_sha256": gate["plan_sha256"],
        "custom_roles": custom_roles,
        "custom_role_count": len(custom_roles),
        "nat": nat,
        "enabled_services": services,
        "step11_non_authorization": step11_non_authorization,
        "strict_effective_iam_simulation_passed": False,
        "ancestor_deny_policy_effectiveness_proven": False,
        "collected_via_get_only": True,
        "cloud_mutation_performed": False,
        "current_profile_changed": False,
    }
    return _seal(body)


def _build_step12_bounded_exception(
    *,
    pair: Mapping[str, Any],
    gate: Mapping[str, Any],
    prefix_empty: Mapping[str, Any],
    package_readback: Mapping[str, Any],
    capacity: Mapping[str, Any],
    security_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    checked_prefix = _validate_sealed(prefix_empty)
    checked_package = _validate_sealed(package_readback)
    checked_capacity = _validate_sealed(capacity)
    checked_security = _validate_sealed(security_evidence)
    names = gate["instance_contract"]["authorized_instance_names"]
    if (
        pair_gate.canonical_sha256(pair)
        != gate["source_pair_contract"]["pair_contract_sha256"]
        or checked_prefix["stage_prefix"]
        != gate["source_pair_contract"]["stage_prefix"]
        or checked_package["outer_package_identity_sha256"]
        != gate["source_pair_contract"]["outer_package_identity_sha256"]
        or checked_capacity["target_instance_names"] != sorted(names)
        or checked_capacity["available_vcpu"] < 16
        or checked_security["gate_plan_sha256"] != gate["plan_sha256"]
        or checked_security["step11_non_authorization"][
            "exception_authorizes_step12"
        ]
        is not False
    ):
        raise ValueError("Step12 bounded-exception prerequisites changed")
    body = {
        "schema": BOUNDED_EXCEPTION_SCHEMA,
        "status": (
            "user_approved_exact_c4_8_pair_attempt0_bounded_exception"
        ),
        "authorization_scope": (
            "step12_stage2_candidate_reference_attempt0_diagnostic_only"
        ),
        "user_approved": True,
        "user_approval_evidence": "exact_cli_execution_confirmation",
        "execution_confirmation_sha256": hashlib.sha256(
            pair_cloud.EXECUTION_CONFIRMATION.encode("ascii")
        ).hexdigest(),
        "pair_contract_sha256": pair["pair_contract_sha256"],
        "gate_plan_sha256": gate["plan_sha256"],
        "pre_mutation_security_receipt_sha256": checked_security[
            "receipt_sha256"
        ],
        "direct_prefix_empty_receipt_sha256": checked_prefix[
            "receipt_sha256"
        ],
        "package_readback_receipt_sha256": checked_package[
            "receipt_sha256"
        ],
        "capacity_receipt_sha256": checked_capacity["receipt_sha256"],
        "source_step11_exception_sha256": checked_security[
            "step11_non_authorization"
        ]["exception_sha256"],
        "source_step11_exception_authorizes_step12": False,
        "strict_iam_gate_passed": False,
        "strict_effective_iam_simulation_passed": False,
        "ancestor_deny_policy_effectiveness_proven": False,
        "bounded_exception_authorizes_exact_step12_pair": True,
        "exception_authorizes_retry": False,
        "exception_authorizes_attempt1": False,
        "instance_names": names,
        "vm_count": 2,
        "actual_machine_type": "c4-standard-8",
        "requested_c4_vcpu": 16,
        "attempt_index": 0,
        "max_attempts_per_job": 1,
        "shared_worker_service_account": gate["principals"][
            "worker_service_account"
        ],
        "shared_worker_identity_residual_risk": True,
        "known_shared_project_findings": [
            "candidate_and_reference_share_one_worker_service_account",
            "default_compute_sa_editor_may_act_as_worker",
            "cloud_services_sa_editor_may_act_as_worker",
            "ancestor_deny_policy_effectiveness_not_proven",
            "effective_permission_simulation_not_run",
        ],
        "bounded_mitigations": {
            "exact_two_instance_names": names,
            "external_ip_permitted": False,
            "attempt1_authorized": False,
            "resume_authorized": False,
            "both_provider_readback_claims_required_before_revoke": True,
            "launch_and_worker_actas_revoked_after_claim_count": 2,
            "all_temporary_iam_bindings_removed": True,
            "final_instance_get_404_required": names,
            "final_disk_get_404_required": names,
            "final_targeted_iam_binding_count_required": 0,
        },
        "diagnostic_only": True,
        "scientific_payload_present": False,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "cloud_mutation_performed": False,
        "current_profile_changed": False,
    }
    return _seal(body)


def _validate_step12_bounded_exception(
    value: Mapping[str, Any],
    *,
    pair: Mapping[str, Any],
    gate: Mapping[str, Any],
    prefix_empty: Mapping[str, Any],
    package_readback: Mapping[str, Any],
    capacity: Mapping[str, Any],
    security_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    checked = _validate_sealed(value)
    expected = _build_step12_bounded_exception(
        pair=pair,
        gate=gate,
        prefix_empty=prefix_empty,
        package_readback=package_readback,
        capacity=capacity,
        security_evidence=security_evidence,
    )
    if (
        checked != expected
        or checked["bounded_exception_authorizes_exact_step12_pair"]
        is not True
        or checked["strict_iam_gate_passed"] is not False
        or checked["strict_effective_iam_simulation_passed"] is not False
        or checked["ancestor_deny_policy_effectiveness_proven"] is not False
        or checked["shared_worker_identity_residual_risk"] is not True
        or checked["exception_authorizes_retry"] is not False
        or checked["exception_authorizes_attempt1"] is not False
    ):
        raise ValueError("Step12 bounded exception changed")
    return checked


def _target(value: str) -> rest_iam.PolicyTarget:
    try:
        return rest_iam.PolicyTarget(value)
    except ValueError as error:
        raise ValueError("Step12 IAM target changed") from error


def _binding_specs(gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    checked = pair_gate.validate_step12_pair_gate_plan(
        gate,
        pair_contract=_ACTIVE_PAIR_INPUTS["pair"],
        candidate_transport_contract=_ACTIVE_PAIR_INPUTS["candidate"],
        reference_transport_contract=_ACTIVE_PAIR_INPUTS["reference"],
        stage1_recovery_receipt=_ACTIVE_PAIR_INPUTS["recovery"],
    )
    expected_resources = {
        "project": f"projects/{transport.PROJECT}",
        "bucket": f"projects/_/buckets/{transport.BUCKET}",
        "worker_service_account": (
            f"projects/{transport.PROJECT}/serviceAccounts/"
            f"{transport.WORKER_SERVICE_ACCOUNT}"
        ),
        "controller_service_account": (
            f"projects/{transport.PROJECT}/serviceAccounts/"
            f"{pair_gate.CONTROLLER_SERVICE_ACCOUNT}"
        ),
    }
    specs: list[dict[str, Any]] = []
    for row in checked["iam_contract"]["binding_specs"]:
        target = _target(row["target"])
        if row["resource"] != expected_resources[target.value]:
            raise ValueError("Step12 IAM resource changed")
        specs.append(
            {
                "purpose": row["purpose"],
                "target": target,
                "role": row["role"],
                "member": row["member"],
                "condition": dict(row["condition"]),
            }
        )
    if (
        len(specs) != 10
        or len({row["purpose"] for row in specs}) != 10
        or any("*" in row["condition"]["expression"] for row in specs)
    ):
        raise ValueError("Step12 exact 10-binding set changed")
    return specs


_ACTIVE_PAIR_INPUTS: dict[str, dict[str, Any]] = {}


def _principals_by_target(
    gate: Mapping[str, Any],
) -> dict[rest_iam.PolicyTarget, frozenset[str]]:
    principals = gate["principals"]
    controller_member = principals["controller_principal"]
    worker_member = principals["worker_principal"]
    initiator = principals["initiating_principal"]
    return {
        rest_iam.PolicyTarget.PROJECT: frozenset(
            {controller_member, worker_member}
        ),
        rest_iam.PolicyTarget.BUCKET: frozenset(
            {controller_member, worker_member}
        ),
        rest_iam.PolicyTarget.WORKER_SERVICE_ACCOUNT: frozenset(
            {controller_member}
        ),
        rest_iam.PolicyTarget.CONTROLLER_SERVICE_ACCOUNT: frozenset(
            {initiator}
        ),
    }


def _targeted_bindings(
    policy: Mapping[str, Any], *, principals: frozenset[str]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in policy.get("bindings", []):
        members = raw.get("members", [])
        matched = sorted(member for member in members if member in principals)
        if matched:
            result.append(
                {
                    "role": raw.get("role"),
                    "members": matched,
                    "all_members": sorted(members),
                    "condition": raw.get("condition"),
                }
            )
    return result


def _exact_authorization_readback_sha256(
    specs: Sequence[Mapping[str, Any]],
    live: Sequence[Mapping[str, Any]],
) -> str:
    expected = [
        {
            "target": row["target"].value,
            "role": row["role"],
            "member": row["member"],
            "condition": dict(row["condition"]),
        }
        for row in specs
    ]
    observed: list[dict[str, Any]] = []
    for row in live:
        members = row.get("members")
        all_members = row.get("all_members")
        condition = row.get("condition")
        if (
            not isinstance(row.get("target"), str)
            or not isinstance(row.get("role"), str)
            or not isinstance(members, list)
            or not isinstance(all_members, list)
            or len(members) != 1
            or members != all_members
            or not isinstance(members[0], str)
            or not isinstance(condition, Mapping)
        ):
            raise RuntimeError(
                "Step12 authorization readback is not one exact binding"
            )
        observed.append(
            {
                "target": row["target"],
                "role": row["role"],
                "member": members[0],
                "condition": dict(condition),
            }
        )
    expected_bytes = sorted(
        controller.canonical_bytes(row) for row in expected
    )
    observed_bytes = sorted(
        controller.canonical_bytes(row) for row in observed
    )
    if (
        len(expected) != 10
        or len(observed) != 10
        or len(set(expected_bytes)) != 10
        or len(set(observed_bytes)) != 10
        or observed_bytes != expected_bytes
    ):
        raise RuntimeError(
            "Step12 exact target/role/member/condition set changed"
        )
    return controller.canonical_sha256(
        sorted(expected, key=controller.canonical_bytes)
    )


def _clear_pair_bindings(
    admin: rest_iam.Step11RestIamAdmin,
    *,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    specs = _binding_specs(gate)
    allowed = {
        (row["target"], row["role"], row["member"]) for row in specs
    }
    removed: list[dict[str, Any]] = []
    principals = _principals_by_target(gate)
    for target in rest_iam.PolicyTarget:
        policy = admin.get_policy(target)
        for raw in list(policy.get("bindings", [])):
            role = raw.get("role")
            condition = raw.get("condition")
            for member in list(raw.get("members", [])):
                if (target, role, member) not in allowed:
                    continue
                result = admin.remove_binding(
                    target,
                    role=role,
                    member=member,
                    condition=condition,
                )
                if result.changed:
                    removed.append(
                        {
                            "target": target.value,
                            "role": role,
                            "member": member,
                            "condition_sha256": (
                                controller.canonical_sha256(condition)
                                if condition is not None
                                else None
                            ),
                        }
                    )
    residual: dict[str, Any] = {}
    for target in rest_iam.PolicyTarget:
        matches = _targeted_bindings(
            admin.get_policy(target),
            principals=principals[target],
        )
        if matches:
            residual[target.value] = matches
    if residual:
        raise RuntimeError("Step12 targeted IAM cleanup did not converge")
    body = {
        "schema": (
            "hu_m31_t3_step6d_rearm2_diagnostic_"
            "step12_targeted_binding_cleanup_v1"
        ),
        "status": "all_step12_target_principals_absent",
        "removed_bindings": removed,
        "removed_binding_count": len(removed),
        "targeted_binding_count": 0,
        "readback_verified": True,
        "cloud_mutation_performed": bool(removed),
    }
    return _seal(body)


def _clear_pair_bindings_with_retry(
    admin: rest_iam.Step11RestIamAdmin,
    *,
    gate: Mapping[str, Any],
    attempts: int = 3,
) -> dict[str, Any]:
    failures: list[BaseException] = []
    for attempt in range(1, attempts + 1):
        try:
            return _clear_pair_bindings(admin, gate=gate)
        except BaseException as error:
            failures.append(error)
            if attempt != attempts:
                time.sleep(min(4, 2 ** (attempt - 1)))
    raise BaseExceptionGroup(
        "Step12 IAM revoke/readback did not converge", failures
    )


def _install_pair_authorization(
    admin: rest_iam.Step11RestIamAdmin,
    *, gate: Mapping[str, Any]
) -> dict[str, Any]:
    specs = _binding_specs(gate)
    records: list[dict[str, Any]] = []
    for spec in specs:
        result = admin.add_binding(
            spec["target"],
            role=spec["role"],
            member=spec["member"],
            condition=spec["condition"],
        )
        if result.changed is not True:
            raise RuntimeError("Step12 authorization was not fresh")
        records.append(
            {
                "purpose": spec["purpose"],
                "target": spec["target"].value,
                "role": spec["role"],
                "member": spec["member"],
                "condition_sha256": controller.canonical_sha256(
                    spec["condition"]
                ),
                "attempts": result.attempts,
                "created": True,
            }
        )
    principals = _principals_by_target(gate)
    live: list[dict[str, Any]] = []
    for target in rest_iam.PolicyTarget:
        live.extend(
            {
                "target": target.value,
                **row,
            }
            for row in _targeted_bindings(
                admin.get_policy(target),
                principals=principals[target],
            )
        )
    live_binding_set_sha256 = _exact_authorization_readback_sha256(
        specs, live
    )
    if len(records) != 10:
        raise RuntimeError("Step12 exact authorization write set changed")
    body = {
        "schema": AUTHORIZATION_SCHEMA,
        "status": "fresh_exact_pair_authorization_installed_and_read_back",
        "binding_count": 10,
        "bindings": records,
        "live_targeted_binding_count": len(live),
        "live_binding_set_sha256": live_binding_set_sha256,
        "instance_names": gate["instance_contract"][
            "authorized_instance_names"
        ],
        "vm_limit": 2,
        "attempt_limit_per_job": 1,
        "expires_at": gate["authorization_window"]["expires_at_rfc3339"],
        "readback_verified": True,
        "cloud_mutation_performed": True,
    }
    return _seal(body)


def _remove_exact_binding(
    admin: rest_iam.Step11RestIamAdmin,
    *,
    spec: Mapping[str, Any],
) -> None:
    result = admin.remove_binding(
        spec["target"],
        role=spec["role"],
        member=spec["member"],
        condition=spec["condition"],
    )
    if result.changed is not True:
        raise RuntimeError(f"{spec['purpose']} binding was not live")


def _post_pair_claim_revoke(
    callback_input: Mapping[str, Any],
) -> dict[str, Any]:
    global _POST_CLAIM_REVOKED

    if (
        _POST_CLAIM_REVOKED
        or _REST_ADMIN is None
        or _ACTIVE_GATE_PLAN is None
        or _ACTIVE_OUTPUT_ROOT is None
    ):
        raise RuntimeError("Step12 post-pair claim callback state changed")
    names = _ACTIVE_GATE_PLAN["instance_contract"][
        "authorized_instance_names"
    ]
    provider_ids = callback_input.get("provider_instance_ids")
    claim_sha256s = callback_input.get("claim_sha256s")
    if (
        callback_input.get("schema")
        != pair_cloud.POST_PAIR_CLAIM_CALLBACK_SCHEMA
        or callback_input.get("instance_names") != names
        or callback_input.get("both_claims_provider_readback_verified")
        is not True
        or not isinstance(provider_ids, list)
        or len(provider_ids) != 2
        or len(set(provider_ids)) != 2
        or any(
            not isinstance(value, str) or not value.isdigit()
            for value in provider_ids
        )
        or not isinstance(claim_sha256s, list)
        or len(claim_sha256s) != 2
        or len(set(claim_sha256s)) != 2
        or any(
            not isinstance(value, str)
            or _SHA256.fullmatch(value) is None
            for value in claim_sha256s
        )
    ):
        raise RuntimeError("both provider-readback claims were not proven")
    specs = {
        row["purpose"]: row for row in _binding_specs(_ACTIVE_GATE_PLAN)
    }
    # Mark before mutation: a retry after a partial revoke is forbidden.
    _POST_CLAIM_REVOKED = True
    _remove_exact_binding(
        _REST_ADMIN, spec=specs["controller_vm_launch"]
    )
    _remove_exact_binding(
        _REST_ADMIN, spec=specs["controller_worker_act_as"]
    )
    principals = _principals_by_target(_ACTIVE_GATE_PLAN)
    project = _targeted_bindings(
        _REST_ADMIN.get_policy(rest_iam.PolicyTarget.PROJECT),
        principals=principals[rest_iam.PolicyTarget.PROJECT],
    )
    worker = _targeted_bindings(
        _REST_ADMIN.get_policy(
            rest_iam.PolicyTarget.WORKER_SERVICE_ACCOUNT
        ),
        principals=principals[
            rest_iam.PolicyTarget.WORKER_SERVICE_ACCOUNT
        ],
    )
    controller_member = _ACTIVE_GATE_PLAN["principals"][
        "controller_principal"
    ]
    launch_role = specs["controller_vm_launch"]["role"]
    if any(
        row["role"] == launch_role and controller_member in row["members"]
        for row in project
    ) or any(
        row["role"] == "roles/iam.serviceAccountUser"
        and controller_member in row["members"]
        for row in worker
    ):
        raise RuntimeError("Step12 launch authority survived pair revoke")
    body = {
        "schema": pair_cloud.POST_PAIR_CLAIM_RECEIPT_SCHEMA,
        "status": "launch_and_worker_actas_removed_after_both_claims",
        "instance_names": names,
        "provider_instance_ids": provider_ids,
        "claim_sha256s": claim_sha256s,
        "both_claims_provider_readback_verified": True,
        "revoke_callback_invocation_count": 1,
        "launch_binding_removed": True,
        "worker_actas_binding_removed": True,
        "readback_verified": True,
        "cloud_mutation_performed": True,
    }
    receipt = _seal(body)
    _write(
        _ACTIVE_OUTPUT_ROOT / "post_pair_claim_revoke_receipt.json",
        receipt,
    )
    return receipt


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> None:
        return None


class ExactPairUserComputeClient:
    """User-token client restricted to two exact VMs, disks, and zone ops."""

    def __init__(self, source: Any, *, instance_names: Sequence[str]) -> None:
        names = tuple(instance_names)
        if (
            len(names) != 2
            or len(set(names)) != 2
            or any(_SAFE_NAME.fullmatch(name) is None for name in names)
        ):
            raise ValueError("exact pair compute client names changed")
        self._source = source
        self._names = frozenset(names)
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect()
        )

    def _allowed(self, method: str, url: str) -> bool:
        parsed = urllib.parse.urlsplit(url)
        root = (
            f"/compute/v1/projects/{transport.PROJECT}/zones/"
            f"{transport.ZONE}/"
        )
        exact = {
            root + f"{kind}/{name}"
            for kind in ("instances", "disks")
            for name in self._names
        }
        if (
            parsed.scheme != "https"
            or parsed.netloc != "compute.googleapis.com"
            or parsed.fragment
        ):
            return False
        if parsed.path in exact:
            if method == "GET":
                return not parsed.query
            if method != "DELETE":
                return False
            query = urllib.parse.parse_qs(
                parsed.query,
                keep_blank_values=True,
                strict_parsing=True,
            )
            return (
                set(query) == {"requestId"}
                and len(query["requestId"]) == 1
                and _UUID.fullmatch(query["requestId"][0]) is not None
            )
        operation_prefix = root + "operations/"
        return (
            method == "GET"
            and not parsed.query
            and parsed.path.startswith(operation_prefix)
            and re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}",
                parsed.path[len(operation_prefix) :],
            )
            is not None
        )

    def request(
        self,
        *,
        method: str,
        url: str,
        body: bytes | None = None,
        content_type: str | None = None,
        timeout_seconds: int = 60,
    ) -> controller.HttpResponse:
        if (
            not self._allowed(method, url)
            or body is not None
            or content_type is not None
            or type(timeout_seconds) is not int
            or not 1 <= timeout_seconds <= 120
        ):
            raise ValueError("pair user compute request escaped exact surface")
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self._source.access_token()}"
            },
            method=method,
        )
        try:
            with self._opener.open(
                request, timeout=timeout_seconds
            ) as response:
                return controller.HttpResponse(
                    status=int(response.status),
                    headers=dict(response.headers.items()),
                    body=response.read(4 * 1024 * 1024 + 1),
                )
        except urllib.error.HTTPError as error:
            return controller.HttpResponse(
                status=int(error.code),
                headers=(
                    dict(error.headers.items()) if error.headers else {}
                ),
                body=error.read(4 * 1024 * 1024 + 1),
            )


def _compute_url(kind: str, name: str) -> str:
    if (
        kind not in {"instances", "disks"}
        or _SAFE_NAME.fullmatch(name) is None
    ):
        raise ValueError("exact compute resource changed")
    return (
        "https://compute.googleapis.com/compute/v1/projects/"
        f"{transport.PROJECT}/zones/{transport.ZONE}/{kind}/{name}"
    )


def _wait_absent(
    client: ExactPairUserComputeClient,
    *,
    kind: str,
    name: str,
    timeout_seconds: int = 600,
) -> int:
    deadline = time.monotonic() + timeout_seconds
    url = _compute_url(kind, name)
    while True:
        response = client.request(method="GET", url=url)
        if response.status == 404:
            return 404
        if response.status != 200:
            raise RuntimeError(
                f"exact {kind} GET returned HTTP {response.status}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(f"exact {kind} did not become absent")
        time.sleep(2)


def _delete_disk(
    client: ExactPairUserComputeClient, *, name: str
) -> dict[str, Any]:
    url = _compute_url("disks", name)
    initial = client.request(method="GET", url=url)
    if initial.status == 404:
        return {
            "disk_name": name,
            "delete_requested": False,
            "already_absent": True,
            "provider_get_status": 404,
        }
    if initial.status != 200:
        raise RuntimeError(
            f"exact disk cleanup GET returned HTTP {initial.status}"
        )
    request_id = str(uuid.uuid4())
    response = client.request(
        method="DELETE", url=f"{url}?requestId={request_id}"
    )
    if response.status == 404:
        return {
            "disk_name": name,
            "delete_requested": False,
            "already_absent": True,
            "provider_get_status": 404,
        }
    if response.status not in {200, 202}:
        raise RuntimeError(
            f"exact disk delete returned HTTP {response.status}"
        )
    operation = _strict_response_json(
        controller.HttpResponse(
            status=200,
            headers=response.headers,
            body=response.body,
        ),
        label="exact disk delete operation",
    )
    completed = step11_cloud.wait_zone_operation(
        client=client,
        initial=operation,
        expected_instance_url=url,
    )
    return {
        "disk_name": name,
        "delete_requested": True,
        "already_absent": False,
        "operation": completed,
        "provider_get_status": _wait_absent(
            client, kind="disks", name=name
        ),
    }


def _final_cleanup() -> dict[str, Any]:
    errors: list[BaseException] = []
    try:
        _assert_profile_unchanged()
    except BaseException as error:
        errors.append(error)
    if _ACTIVE_GATE_PLAN is None:
        if errors:
            raise BaseExceptionGroup(
                "Step12 pre-cloud invariant failed", errors
            )
        return _seal(
            {
                "schema": CLEANUP_SCHEMA,
                "status": "no_gate_or_cloud_administrator_created",
                "cloud_mutation_performed": False,
                "current_profile_changed": False,
            }
        )
    names = _ACTIVE_GATE_PLAN["instance_contract"][
        "authorized_instance_names"
    ]
    if not _CLOUD_MUTATION_STARTED:
        body = {
            "schema": CLEANUP_SCHEMA,
            "status": (
                "pre_mutation_gate_failed_no_cloud_cleanup_mutation_needed"
            ),
            "instance_names": names,
            "disk_names": names,
            "iam_revoke_attempted": False,
            "compute_delete_attempted": False,
            "cloud_mutation_performed": False,
            "current_profile_sha256": EXPECTED_POLICY_REGISTRY_SHA256,
            "current_profile_changed": False,
        }
        receipt = _seal(body)
        if _ACTIVE_OUTPUT_ROOT is not None:
            path = _ACTIVE_OUTPUT_ROOT / "final_cloud_cleanup_receipt.json"
            if not path.exists():
                _write(path, receipt)
        if errors:
            raise BaseExceptionGroup(
                "Step12 pre-mutation cleanup invariant failed", errors
            )
        return receipt
    binding_cleanup: Mapping[str, Any] | None = None
    instance_cleanup: list[Mapping[str, Any]] = []
    disk_cleanup: list[Mapping[str, Any]] = []
    final_instance_status: dict[str, int | None] = {
        name: None for name in names
    }
    final_disk_status: dict[str, int | None] = {name: None for name in names}
    targeted: dict[str, list[dict[str, Any]]] = {}
    iam_revoked_and_zero = False

    # Temporary authority is always revoked before user-credential compute
    # cleanup.  If revoke/readback cannot converge after bounded retries, no
    # user-token delete is allowed: otherwise a still-authorized controller
    # could recreate a just-deleted exact name.
    if _REST_ADMIN is not None:
        try:
            binding_cleanup = _clear_pair_bindings_with_retry(
                _REST_ADMIN, gate=_ACTIVE_GATE_PLAN
            )
            iam_revoked_and_zero = True
        except BaseException as error:
            errors.append(error)
    else:
        errors.append(RuntimeError("pair IAM administrator is unavailable"))
    if iam_revoked_and_zero and _USER_COMPUTE_CLIENT is not None:
        for name in names:
            try:
                instance_cleanup.append(
                    step11_cloud.delete_exact_instance(
                        client=_USER_COMPUTE_CLIENT,
                        instance_name=name,
                    )
                )
            except BaseException as error:
                errors.append(error)
                instance_cleanup.append(
                    {
                        "instance_name": name,
                        "cleanup_failed": True,
                        "error_type": type(error).__name__,
                    }
                )
        for name in names:
            try:
                disk_cleanup.append(
                    _delete_disk(_USER_COMPUTE_CLIENT, name=name)
                )
            except BaseException as error:
                errors.append(error)
                disk_cleanup.append(
                    {
                        "disk_name": name,
                        "cleanup_failed": True,
                        "error_type": type(error).__name__,
                    }
                )
    elif iam_revoked_and_zero:
        errors.append(RuntimeError("pair compute cleanup client is unavailable"))
    else:
        for name in names:
            instance_cleanup.append(
                {
                    "instance_name": name,
                    "cleanup_skipped": True,
                    "reason": "iam_revoke_and_zero_readback_not_proven",
                }
            )
            disk_cleanup.append(
                {
                    "disk_name": name,
                    "cleanup_skipped": True,
                    "reason": "iam_revoke_and_zero_readback_not_proven",
                }
            )

    if _REST_ADMIN is not None:
        principals = _principals_by_target(_ACTIVE_GATE_PLAN)
        for target in rest_iam.PolicyTarget:
            try:
                targeted[target.value] = _targeted_bindings(
                    _REST_ADMIN.get_policy(target),
                    principals=principals[target],
                )
                if targeted[target.value]:
                    raise RuntimeError(
                        f"{target.value} targeted IAM bindings remain"
                    )
            except BaseException as error:
                errors.append(error)
    if iam_revoked_and_zero and _USER_COMPUTE_CLIENT is not None:
        for name in names:
            try:
                final_instance_status[name] = _wait_absent(
                    _USER_COMPUTE_CLIENT, kind="instances", name=name
                )
            except BaseException as error:
                errors.append(error)
            try:
                final_disk_status[name] = _wait_absent(
                    _USER_COMPUTE_CLIENT, kind="disks", name=name
                )
            except BaseException as error:
                errors.append(error)
    try:
        _assert_profile_unchanged()
    except BaseException as error:
        errors.append(error)

    body = {
        "schema": CLEANUP_SCHEMA,
        "status": (
            "exact_pair_vm_disk_and_all_temporary_iam_absent"
            if not errors
            else "step12_cleanup_attempted_but_not_fully_converged"
        ),
        "instance_names": names,
        "disk_names": names,
        "binding_cleanup_receipt_sha256": (
            binding_cleanup.get("receipt_sha256")
            if binding_cleanup is not None
            else None
        ),
        "instance_cleanup": instance_cleanup,
        "disk_cleanup": disk_cleanup,
        "final_instance_get_status": final_instance_status,
        "final_disk_get_status": final_disk_status,
        "targeted_iam_bindings": targeted,
        "final_targeted_iam_binding_count": sum(
            len(rows) for rows in targeted.values()
        ),
        "iam_revoked_before_user_credential_compute_cleanup": (
            iam_revoked_and_zero
        ),
        "independent_instance_cleanup_attempt_count": len(instance_cleanup),
        "independent_disk_cleanup_attempt_count": len(disk_cleanup),
        "cleanup_error_types": [type(error).__name__ for error in errors],
        "package_and_results_retained": True,
        "current_profile_sha256": EXPECTED_POLICY_REGISTRY_SHA256,
        "current_profile_changed": False,
        "cloud_mutation_performed": (
            _REST_ADMIN is not None or _USER_COMPUTE_CLIENT is not None
        ),
    }
    receipt = _seal(body)
    if _ACTIVE_OUTPUT_ROOT is not None:
        path = _ACTIVE_OUTPUT_ROOT / "final_cloud_cleanup_receipt.json"
        if not path.exists():
            _write(path, receipt)
    if errors:
        raise BaseExceptionGroup(
            "Step12 final cleanup did not fully converge", errors
        )
    if (
        len(instance_cleanup) != 2
        or len(disk_cleanup) != 2
        or set(final_instance_status.values()) != {404}
        or set(final_disk_status.values()) != {404}
        or body["final_targeted_iam_binding_count"] != 0
    ):
        raise RuntimeError("Step12 final cleanup evidence is incomplete")
    return receipt


def _prelaunch_exact_compute_empty(
    client: ExactPairUserComputeClient, *, names: Sequence[str]
) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    for kind in ("instances", "disks"):
        for name in names:
            response = client.request(
                method="GET", url=_compute_url(kind, name)
            )
            statuses[f"{kind}/{name}"] = response.status
            if response.status != 404:
                raise FileExistsError(
                    f"fresh Step12 {kind}/{name} is not absent"
                )
    return _seal(
        {
            "schema": (
                "hu_m31_t3_step6d_rearm2_diagnostic_"
                "step12_prelaunch_compute_empty_v1"
            ),
            "status": "exact_two_instances_and_disks_absent",
            "provider_status": statuses,
            "provider_get_404_count": len(statuses),
            "collected_via_get_only": True,
            "cloud_mutation_performed": False,
        }
    )


def _require_launch_evidence_and_authorization_fresh(
    *,
    gate: Mapping[str, Any],
    capacity: Mapping[str, Any],
    prefix: Mapping[str, Any],
    now_unix_seconds: int,
) -> int:
    capacity_age = (
        now_unix_seconds - capacity["collected_at_unix_seconds"]
    )
    prefix_age = now_unix_seconds - prefix["observed_at_unix_seconds"]
    remaining = (
        gate["authorization_window"]["expires_at_unix_seconds"]
        - now_unix_seconds
    )
    if (
        capacity_age < 0
        or prefix_age < 0
        or capacity_age > pair_gate.OBSERVATION_MAX_AGE_SECONDS
        or prefix_age > pair_gate.OBSERVATION_MAX_AGE_SECONDS
    ):
        raise RuntimeError(
            "Step12 launch-time GET-only evidence became stale"
        )
    if remaining < step11_cloud.MIN_EXECUTION_REMAINING_SECONDS:
        raise RuntimeError(
            "Step12 authorization lacks runtime and cleanup margin"
        )
    return remaining


def _execute(
    *,
    execution_confirmation: str,
    output_root: Path,
    signer: controller.EphemeralControllerKey,
    candidate: dict[str, Any],
    reference: dict[str, Any],
    pair: dict[str, Any],
    gate: dict[str, Any],
    recovery: dict[str, Any],
) -> int:
    global _REST_ADMIN
    global _USER_COMPUTE_CLIENT
    global _ACTIVE_GATE_PLAN
    global _ACTIVE_OUTPUT_ROOT
    global _CLOUD_MUTATION_STARTED

    if execution_confirmation != pair_cloud.EXECUTION_CONFIRMATION:
        raise PermissionError("exact Step12 execution confirmation is missing")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Step12 output root is not fresh: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir()
    _ACTIVE_GATE_PLAN = gate
    _ACTIVE_OUTPUT_ROOT = output_root
    _ACTIVE_PAIR_INPUTS.update(
        {
            "candidate": candidate,
            "reference": reference,
            "pair": pair,
            "recovery": recovery,
        }
    )
    for filename, value in (
        ("controller_public_key.json", signer.public_record),
        ("candidate_transport_contract.json", candidate),
        ("reference_transport_contract.json", reference),
        ("pair_contract.json", pair),
        ("iam_capacity_gate_plan.json", gate),
        (
            "runner_plan.json",
            _runner_plan(
                candidate=candidate,
                reference=reference,
                pair=pair,
                gate=gate,
            ),
        ),
    ):
        _write(output_root / filename, value)

    step11_runner = _load_step11_runner_primitives()
    user_token = step11_runner.ActiveUserToken()
    user_token.access_token()
    user_read = step11_runner.ActiveUserReadOnlyClient(user_token)
    names = gate["instance_contract"]["authorized_instance_names"]
    user_compute = ExactPairUserComputeClient(
        user_token, instance_names=names
    )
    admin = rest_iam.Step11RestIamAdmin(
        http_client=rest_iam.StdlibJsonHttpsClient(),
        user_token_source=user_token,
        project=transport.PROJECT,
        bucket=transport.BUCKET,
        worker_service_account=transport.WORKER_SERVICE_ACCOUNT,
        controller_service_account=pair_gate.CONTROLLER_SERVICE_ACCOUNT,
    )
    _REST_ADMIN = admin
    _USER_COMPUTE_CLIENT = user_compute

    compute_empty = _prelaunch_exact_compute_empty(
        user_compute, names=names
    )
    _write(output_root / "prelaunch_compute_empty.json", compute_empty)
    prefix_empty = _require_direct_stage_prefix_empty(
        user_read, stage_prefix=candidate["remote_layout"]["stage_prefix"]
    )
    _write(output_root / "direct_prefix_empty.json", prefix_empty)
    package_readback, package_generations = _package_generation_readback(
        user_read, candidate=candidate
    )
    _write(output_root / "package_generation_readback.json", package_readback)
    capacity = _collect_fresh_capacity(
        step11_runner,
        gate=gate,
        collected_at_unix_seconds=int(time.time()),
        token_source=user_token,
    )
    if (
        int(time.time()) - capacity["collected_at_unix_seconds"]
        > pair_gate.OBSERVATION_MAX_AGE_SECONDS
    ):
        raise RuntimeError("Step12 capacity observation became stale")
    _write(output_root / "fresh_live_capacity.json", capacity)

    security_evidence = _collect_pre_mutation_security_evidence(
        step11_runner,
        user_token=user_token,
        gate=gate,
        collected_at_unix_seconds=int(time.time()),
    )
    _write(
        output_root / "pre_mutation_security_evidence.json",
        security_evidence,
    )
    bounded_exception = _build_step12_bounded_exception(
        pair=pair,
        gate=gate,
        prefix_empty=prefix_empty,
        package_readback=package_readback,
        capacity=capacity,
        security_evidence=security_evidence,
    )
    _write(
        output_root / "step12_shared_project_bounded_exception.json",
        bounded_exception,
    )
    durable_exception = _validate_step12_bounded_exception(
        _read_json(
            output_root / "step12_shared_project_bounded_exception.json"
        ),
        pair=pair,
        gate=gate,
        prefix_empty=prefix_empty,
        package_readback=package_readback,
        capacity=capacity,
        security_evidence=security_evidence,
    )

    # The first cloud mutation is forbidden until every GET-only input above
    # has been sealed into the exact Step12 bounded exception.
    _CLOUD_MUTATION_STARTED = True
    stale = _clear_pair_bindings(admin, gate=gate)
    _write(output_root / "stale_binding_cleanup.json", stale)
    authorization = _install_pair_authorization(admin, gate=gate)
    _write(output_root / "authorization_receipt.json", authorization)

    token_source = step11_runner.CachedControllerToken(admin)
    step11_runner._wait_for_controller_token(token_source)
    controller_client = controller.GoogleJsonClient(token_source)
    launch_capacity = _collect_fresh_capacity(
        step11_runner,
        gate=gate,
        collected_at_unix_seconds=int(time.time()),
        token_source=user_token,
    )
    _write(output_root / "launch_fresh_live_capacity.json", launch_capacity)
    launch_prefix = _require_direct_stage_prefix_empty(
        user_read,
        stage_prefix=candidate["remote_layout"]["stage_prefix"],
        observed_at_unix_seconds=int(time.time()),
    )
    _write(output_root / "launch_direct_prefix_empty.json", launch_prefix)
    _require_launch_evidence_and_authorization_fresh(
        gate=gate,
        capacity=launch_capacity,
        prefix=launch_prefix,
        now_unix_seconds=int(time.time()),
    )
    if _validate_step12_bounded_exception(
        _read_json(
            output_root / "step12_shared_project_bounded_exception.json"
        ),
        pair=pair,
        gate=gate,
        prefix_empty=prefix_empty,
        package_readback=package_readback,
        capacity=capacity,
        security_evidence=security_evidence,
    ) != durable_exception:
        raise RuntimeError("Step12 bounded exception changed before insert")
    launch_compute_empty = _prelaunch_exact_compute_empty(
        user_compute, names=names
    )
    _write(
        output_root / "launch_preinsert_compute_empty.json",
        launch_compute_empty,
    )
    # The four exact GETs above may each consume up to 60 seconds.  Recheck
    # both GET-only observations and the cleanup margin after those calls,
    # immediately before the first insert.
    _require_launch_evidence_and_authorization_fresh(
        gate=gate,
        capacity=launch_capacity,
        prefix=launch_prefix,
        now_unix_seconds=int(time.time()),
    )
    receipt = pair_cloud.execute_pair_attempt0(
        execute=True,
        execution_confirmation=pair_cloud.EXECUTION_CONFIRMATION,
        client=controller_client,
        collector_client=user_read,
        pair_contract=pair,
        candidate_transport_contract=candidate,
        reference_transport_contract=reference,
        signer=signer,
        package_generations=package_generations,
        external_preflight_receipt_sha256=durable_exception[
            "receipt_sha256"
        ],
        issued_unix_seconds=gate["authorization_window"][
            "issued_at_unix_seconds"
        ],
        expires_unix_seconds=gate["authorization_window"][
            "expires_at_unix_seconds"
        ],
        post_pair_claim_callback=_post_pair_claim_revoke,
        startup_path=STARTUP_PATH,
        prebootstrap_path=PREBOOTSTRAP_PATH,
        base_prebootstrap_path=BASE_PREBOOTSTRAP_PATH,
        destination_root=output_root / "received",
        final_receipt_path=output_root / "final_lifecycle_receipt.json",
    )
    if (
        receipt["provider_get_404_count"] != 2
        or receipt["current_profile_changed"] is not False
    ):
        raise RuntimeError("Step12 lifecycle receipt changed")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step12 exact candidate/reference diagnostic pair runner"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="enable the guarded cloud path",
    )
    parser.add_argument(
        "--execute-confirmation",
        default="",
        help="exact Step12 execution confirmation string",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="fresh durable output root used only by --execute",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    issued = int(time.time())
    signer, candidate, reference, pair, gate, recovery = _build_local_bundle(
        issued_at_unix_seconds=issued
    )
    plan = _runner_plan(
        candidate=candidate,
        reference=reference,
        pair=pair,
        gate=gate,
    )
    if not args.execute:
        if args.execute_confirmation:
            raise ValueError(
                "execution confirmation is invalid without --execute"
            )
        sys.stdout.write(json.dumps(plan, sort_keys=True) + "\n")
        return 0
    if args.execute_confirmation != pair_cloud.EXECUTION_CONFIRMATION:
        raise PermissionError("exact Step12 execution confirmation is missing")
    return _execute(
        execution_confirmation=args.execute_confirmation,
        output_root=args.output_root.resolve(),
        signer=signer,
        candidate=candidate,
        reference=reference,
        pair=pair,
        gate=gate,
        recovery=recovery,
    )


def _run_with_cleanup(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.execute:
        return main(argv)
    primary: BaseException | None = None
    result = 1
    try:
        result = main(argv)
    except BaseException as error:
        primary = error
    try:
        cleanup = _final_cleanup()
    except BaseException as cleanup_error:
        if primary is not None:
            raise BaseExceptionGroup(
                "Step12 execution and final cleanup both failed",
                [primary, cleanup_error],
            ) from None
        raise
    if primary is not None:
        raise primary
    lifecycle = _read_json(
        args.output_root.resolve() / "final_lifecycle_receipt.json"
    )
    sys.stdout.write(
        json.dumps(
            {
                "status": lifecycle["status"],
                "receipt_sha256": lifecycle["receipt_sha256"],
                "cleanup_receipt_sha256": cleanup["receipt_sha256"],
                "instance_names": [
                    row["name"] for row in lifecycle["provider_instances"]
                ],
                "provider_get_404_count": lifecycle[
                    "provider_get_404_count"
                ],
                "final_targeted_iam_binding_count": cleanup[
                    "final_targeted_iam_binding_count"
                ],
                "output_root": str(args.output_root.resolve()),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return result


if __name__ == "__main__":
    raise SystemExit(_run_with_cleanup())
