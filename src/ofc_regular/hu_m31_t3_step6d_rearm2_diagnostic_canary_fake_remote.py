"""In-memory fault model for the bounded diagnostic VM preview.

This module is deliberately not a cloud transport.  It has no filesystem,
network, subprocess, gcloud, or VM API.  It exercises the remote-state
contract that a later implementation must satisfy:

* an empty, stage-specific prefix is claimed with generation-match zero;
* simulated authorization and claim objects are separate and immutable;
* uploads are ordered, immutable, and read back before ``DONE``;
* attempts are limited to zero and one;
* receive and stage escalation require controller-proven VM absence.

All authorization-like records in this module explicitly deny real cloud
execution.  They are test artifacts only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_step6d_rearm2_diagnostic_canary_local as local
from . import hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as plan
from . import hu_m31_t3_step6d_rearm2_diagnostic_canary_vm_adapter as adapter


PREFIX_SENTINEL_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_fake_prefix_sentinel_v1"
)
SIMULATED_AUTHORIZATION_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_fake_authorization_v1"
)
SIMULATED_CLAIM_SCHEMA = "hu_m31_t3_step6d_rearm2_diagnostic_fake_claim_v1"
ATTEMPT_CLAIM_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_fake_attempt_claim_v1"
)
CONTROLLER_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_fake_controller_receipt_v1"
)

SENTINEL_NAME = "control/FAKE_PREFIX_SENTINEL.json"
AUTHORIZATION_NAME = "control/FAKE_SIMULATED_AUTHORIZATION.json"
CLAIM_NAME = "control/FAKE_SIMULATED_CLAIM.json"


def canonical_bytes(value: Any) -> bytes:
    return adapter.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return adapter.canonical_sha256(value)


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_bytes(value))


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} keys changed")


def _nonzero_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or set(value) - set("0123456789abcdef")
        or value == "0" * 64
    ):
        raise ValueError(f"{label} must be a nonzero lowercase sha256")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _exact_int(value: Any, expected: int, label: str) -> int:
    if type(value) is not int or value != expected:
        raise ValueError(f"{label} must be the exact integer {expected}")
    return value


def _exact_float(value: Any, expected: float, label: str) -> float:
    if (
        type(value) is not float
        or not math.isfinite(value)
        or value != expected
    ):
        raise ValueError(f"{label} must be the exact finite float {expected}")
    return value


def _exact_bool(value: Any, expected: bool, label: str) -> bool:
    if value is not expected:
        raise ValueError(f"{label} must be the exact boolean {expected}")
    return value


def _uri(prefix: str, relative: str) -> str:
    return f"{prefix.rstrip('/')}/{relative.lstrip('/')}"


@dataclass(frozen=True)
class PutResult:
    uri: str
    generation: int
    content_sha256: str
    created: bool


@dataclass(frozen=True)
class ControllerReceiptProof:
    """Opaque in-memory proof backed by the stage that issued the receipt."""

    receipt_bytes: bytes
    _source_stage: Any
    _issuer_nonce: object

    @property
    def content(self) -> dict[str, Any]:
        value = json.loads(self.receipt_bytes)
        if not isinstance(value, dict) or canonical_bytes(value) != self.receipt_bytes:
            raise ValueError("fake controller proof receipt is not canonical")
        return value


class InMemoryGenerationStore:
    """Canonical-JSON object store with generation-match-zero semantics."""

    def __init__(self) -> None:
        self._objects: dict[str, tuple[int, bytes]] = {}
        self._next_generation = 1
        self.events: list[dict[str, Any]] = []

    @staticmethod
    def _under_prefix(uri: str, prefix: str) -> bool:
        base = prefix.rstrip("/")
        return uri == base or uri.startswith(base + "/")

    def list_prefix(self, prefix: str) -> list[str]:
        return sorted(
            uri for uri in self._objects if self._under_prefix(uri, prefix)
        )

    def contains(self, uri: str) -> bool:
        return uri in self._objects

    def put_once(self, uri: str, content: Mapping[str, Any]) -> PutResult:
        if not isinstance(uri, str) or not uri.startswith("gs://"):
            raise ValueError("fake remote URI must be an exact gs:// identity")
        raw = canonical_bytes(dict(content))
        digest = canonical_sha256(dict(content))
        existing = self._objects.get(uri)
        if existing is not None:
            generation, previous = existing
            if previous != raw:
                raise FileExistsError(
                    f"generation-match-zero collision with different bytes: {uri}"
                )
            self.events.append(
                {
                    "operation": "idempotent_readback",
                    "uri": uri,
                    "generation": generation,
                    "content_sha256": digest,
                }
            )
            return PutResult(uri, generation, digest, False)
        generation = self._next_generation
        self._next_generation += 1
        self._objects[uri] = (generation, raw)
        self.events.append(
            {
                "operation": "conditional_create_generation_match_zero",
                "uri": uri,
                "generation": generation,
                "content_sha256": digest,
            }
        )
        return PutResult(uri, generation, digest, True)

    def read(self, uri: str) -> dict[str, Any]:
        try:
            generation, raw = self._objects[uri]
        except KeyError as exc:
            raise FileNotFoundError(f"fake remote object is absent: {uri}") from exc
        value = json.loads(raw)
        if not isinstance(value, dict) or canonical_bytes(value) != raw:
            raise ValueError("fake remote object is not canonical JSON")
        self.events.append(
            {
                "operation": "readback",
                "uri": uri,
                "generation": generation,
                "content_sha256": canonical_sha256(value),
            }
        )
        return _copy_json(value)

    def record(self, uri: str) -> dict[str, Any]:
        content = self.read(uri)
        generation, _raw = self._objects[uri]
        return {
            "uri": uri,
            "generation": generation,
            "content_sha256": canonical_sha256(content),
            "content": content,
        }


def _stage_spec(stage_id: str) -> dict[str, Any]:
    if stage_id == plan.STAGE1_ID:
        return {
            "run_name": plan.STAGE1_RUN_NAME,
            "job_ids": list(plan.STAGE1_JOB_IDS),
            "roles": ["candidate"],
            "hand_indices": list(plan.STAGE1_HAND_INDICES),
            "result_name": plan.STAGE1_RESULT_NAME,
            "stage_number": 1,
        }
    if stage_id == plan.STAGE2_ID:
        return {
            "run_name": plan.STAGE2_RUN_NAME,
            "job_ids": list(plan.STAGE2_JOB_IDS),
            "roles": ["candidate", "reference"],
            "hand_indices": list(plan.STAGE2_HAND_INDICES),
            "result_name": plan.STAGE2_RESULT_NAME,
            "stage_number": 2,
        }
    raise ValueError("fake transport permits only the frozen diagnostic stages")


def _validate_cost_guard(
    value: Mapping[str, Any], *, vm_count: int
) -> dict[str, Any]:
    guard = dict(value)
    _exact(
        guard,
        {
            "currency",
            "planning_price_source",
            "observed_spot_price_usd_per_vm_hour",
            "spot_price_ceiling_usd_per_vm_hour",
            "max_runtime_seconds_per_vm",
            "watchdog_seconds_per_vm",
            "max_attempts_per_job",
            "vm_count",
            "initial_estimated_max_compute_usd",
            "all_attempts_estimated_max_compute_usd",
            "diagnostic_compute_cap_usd",
        },
        "fake preview cost guard",
    )
    observed = guard["observed_spot_price_usd_per_vm_hour"]
    if observed is not None and (
        type(observed) is not float
        or not math.isfinite(observed)
        or observed <= 0
        or observed > adapter.SPOT_PRICE_CEILING_USD_PER_VM_HOUR
    ):
        raise ValueError("fake preview observed Spot price changed")
    initial = (
        vm_count
        * adapter.SPOT_PRICE_CEILING_USD_PER_VM_HOUR
        * adapter.MAX_RUNTIME_SECONDS_PER_VM
        / 3600
    )
    _exact_float(
        guard["spot_price_ceiling_usd_per_vm_hour"],
        adapter.SPOT_PRICE_CEILING_USD_PER_VM_HOUR,
        "fake preview Spot price ceiling",
    )
    _exact_int(
        guard["max_runtime_seconds_per_vm"],
        adapter.MAX_RUNTIME_SECONDS_PER_VM,
        "fake preview max runtime",
    )
    _exact_int(
        guard["watchdog_seconds_per_vm"],
        adapter.WATCHDOG_SECONDS_PER_VM,
        "fake preview watchdog",
    )
    _exact_int(
        guard["max_attempts_per_job"],
        adapter.MAX_ATTEMPTS_PER_JOB,
        "fake preview max attempts",
    )
    _exact_int(guard["vm_count"], vm_count, "fake preview cost-guard VM count")
    _exact_float(
        guard["initial_estimated_max_compute_usd"],
        initial,
        "fake preview initial compute estimate",
    )
    _exact_float(
        guard["all_attempts_estimated_max_compute_usd"],
        initial * adapter.MAX_ATTEMPTS_PER_JOB,
        "fake preview all-attempt compute estimate",
    )
    _exact_float(
        guard["diagnostic_compute_cap_usd"],
        adapter.DIAGNOSTIC_COMPUTE_CAP_USD,
        "fake preview diagnostic compute cap",
    )
    expected = {
        "currency": "USD",
        "planning_price_source": (
            "caller_supplied_read_only_observation"
            if observed is not None
            else "frozen_ceiling_only_no_live_price_query"
        ),
        "observed_spot_price_usd_per_vm_hour": observed,
        "spot_price_ceiling_usd_per_vm_hour": (
            adapter.SPOT_PRICE_CEILING_USD_PER_VM_HOUR
        ),
        "max_runtime_seconds_per_vm": adapter.MAX_RUNTIME_SECONDS_PER_VM,
        "watchdog_seconds_per_vm": adapter.WATCHDOG_SECONDS_PER_VM,
        "max_attempts_per_job": adapter.MAX_ATTEMPTS_PER_JOB,
        "vm_count": vm_count,
        "initial_estimated_max_compute_usd": initial,
        "all_attempts_estimated_max_compute_usd": (
            initial * adapter.MAX_ATTEMPTS_PER_JOB
        ),
        "diagnostic_compute_cap_usd": adapter.DIAGNOSTIC_COMPUTE_CAP_USD,
    }
    if guard != expected:
        raise ValueError("fake preview cost guard changed")
    return guard


def _validate_preview_boundary(preview: Mapping[str, Any]) -> dict[str, Any]:
    """Independently validate every identity needed by the fake transport."""

    value = dict(preview)
    _exact(
        value,
        {
            "schema",
            "status",
            "adapter_schema",
            "package_manifest_sha256",
            "stage_id",
            "run_name",
            "selected_job_ids",
            "vm_count",
            "stage_token",
            "stage_token_sha256",
            "remote_manifest",
            "remote_manifest_sha256",
            "jobs",
            "cost_guard",
            "cost_guard_sha256",
            "capabilities",
        },
        "fake VM preview",
    )
    if (
        value["schema"] != adapter.PREVIEW_SCHEMA
        or value["status"] != "local_dry_run_preview_cloud_not_authorized"
        or value["adapter_schema"] != adapter.ADAPTER_SCHEMA
        or not isinstance(value["stage_id"], str)
    ):
        raise ValueError("fake transport requires the accepted local VM preview")
    spec = _stage_spec(value["stage_id"])
    package_sha = _nonzero_sha(
        value["package_manifest_sha256"], "preview package manifest"
    )
    _exact_int(
        value["vm_count"],
        len(spec["job_ids"]),
        "fake preview VM count",
    )
    if (
        value["run_name"] != spec["run_name"]
        or value["selected_job_ids"] != spec["job_ids"]
    ):
        raise ValueError("fake preview stage/run/job identity changed")

    token_raw = value["stage_token"]
    if not isinstance(token_raw, Mapping):
        raise ValueError("fake preview stage token must be an object")
    token = dict(token_raw)
    _exact(
        token,
        {
            "schema",
            "status",
            "package_manifest_sha256",
            "stage_id",
            "run_name",
            "selected_job_ids",
            "prerequisite_stage1_receive",
            "cloud_launch_authorized",
            "gcloud_invocation_authorized",
            "claim_written",
            "authorization_written",
            "diagnostic_only",
        },
        "fake preview stage token",
    )
    prerequisite = token["prerequisite_stage1_receive"]
    if value["stage_id"] == plan.STAGE1_ID:
        prerequisite_ok = prerequisite is None
    else:
        prerequisite_ok = isinstance(prerequisite, Mapping)
        if prerequisite_ok:
            prerequisite = dict(prerequisite)
            _exact(
                prerequisite,
                {
                    "stage_id",
                    "run_name",
                    "selected_job_ids",
                    "package_manifest_sha256",
                    "receipt_sha256",
                    "job_record_aggregate_sha256",
                },
                "fake preview stage1 receipt binding",
            )
            prerequisite_ok = (
                prerequisite["stage_id"] == plan.STAGE1_ID
                and prerequisite["run_name"] == plan.STAGE1_RUN_NAME
                and prerequisite["selected_job_ids"] == list(plan.STAGE1_JOB_IDS)
                and prerequisite["package_manifest_sha256"] == package_sha
                and _nonzero_sha(
                    prerequisite["receipt_sha256"],
                    "stage1 receipt binding",
                )
                == prerequisite["receipt_sha256"]
                and _nonzero_sha(
                    prerequisite["job_record_aggregate_sha256"],
                    "stage1 job aggregate binding",
                )
                == prerequisite["job_record_aggregate_sha256"]
            )
    if (
        token["schema"] != adapter.STAGE_TOKEN_SCHEMA
        or token["status"] != "dry_run_identity_only_cloud_not_authorized"
        or token["package_manifest_sha256"] != package_sha
        or token["stage_id"] != value["stage_id"]
        or token["run_name"] != spec["run_name"]
        or token["selected_job_ids"] != spec["job_ids"]
        or not prerequisite_ok
        or token["cloud_launch_authorized"] is not False
        or token["gcloud_invocation_authorized"] is not False
        or token["claim_written"] is not False
        or token["authorization_written"] is not False
        or token["diagnostic_only"] is not True
        or value["stage_token_sha256"] != canonical_sha256(token)
    ):
        raise ValueError("fake preview stage token identity changed")

    guard_raw = value["cost_guard"]
    if not isinstance(guard_raw, Mapping):
        raise ValueError("fake preview cost guard must be an object")
    guard = _validate_cost_guard(guard_raw, vm_count=len(spec["job_ids"]))
    if value["cost_guard_sha256"] != canonical_sha256(guard):
        raise ValueError("fake preview cost-guard digest changed")

    capabilities_raw = value["capabilities"]
    if not isinstance(capabilities_raw, Mapping):
        raise ValueError("fake preview capabilities must be an object")
    capabilities = dict(capabilities_raw)
    expected_capabilities = {
        "cloud_launch_authorized": False,
        "gcloud_invocation_authorized": False,
        "subprocess_invocation_authorized": False,
        "claim_write_authorized": False,
        "launch_authorization_write_authorized": False,
        "object_write_authorized": False,
        "vm_create_authorized": False,
        "production_all20_launcher_reused": False,
        "current_profile_changed": False,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
    }
    for field, expected in expected_capabilities.items():
        _exact_bool(
            capabilities.get(field),
            expected,
            f"fake preview capability {field}",
        )
    if capabilities != expected_capabilities:
        raise ValueError("fake preview capability boundary changed")

    remote_raw = value["remote_manifest"]
    if not isinstance(remote_raw, Mapping):
        raise ValueError("fake preview remote manifest must be an object")
    remote = dict(remote_raw)
    _exact(
        remote,
        {
            "schema",
            "prefix",
            "stage_id",
            "run_name",
            "freshness_preflight",
            "control_objects",
            "job_prefixes",
            "receive_uri",
        },
        "fake preview remote manifest",
    )
    prefix = (
        f"gs://{adapter.DEFAULT_BUCKET}/hu-m31-r2diag-vm-v1/"
        f"{spec['run_name']}/{package_sha[:16]}"
    )
    freshness = {
        "entire_stage_prefix_must_be_empty": True,
        "conditional_create_generation_match": 0,
        "unknown_object_is_fatal": True,
        "permission_or_query_error_is_fatal": True,
        "performed_by_this_dry_run_module": False,
    }
    freshness_raw = remote["freshness_preflight"]
    if not isinstance(freshness_raw, Mapping):
        raise ValueError("fake preview freshness preflight must be an object")
    freshness_value = dict(freshness_raw)
    _exact(
        freshness_value,
        set(freshness),
        "fake preview freshness preflight",
    )
    _exact_bool(
        freshness_value["entire_stage_prefix_must_be_empty"],
        True,
        "fake preview empty-prefix requirement",
    )
    _exact_int(
        freshness_value["conditional_create_generation_match"],
        0,
        "fake preview generation match",
    )
    for field in (
        "unknown_object_is_fatal",
        "permission_or_query_error_is_fatal",
    ):
        _exact_bool(
            freshness_value[field],
            True,
            f"fake preview freshness field {field}",
        )
    _exact_bool(
        freshness_value["performed_by_this_dry_run_module"],
        False,
        "fake preview dry-run preflight flag",
    )
    if not isinstance(remote["control_objects"], list):
        raise ValueError("fake preview control objects must be a list")
    controls: list[dict[str, Any]] = []
    for raw in remote["control_objects"]:
        if not isinstance(raw, Mapping):
            raise ValueError("fake preview control object must be an object")
        record = dict(raw)
        _exact(record, {"uri", "sha256", "bytes"}, "fake preview control object")
        _nonzero_sha(record["sha256"], "fake preview control object")
        _positive_int(record["bytes"], "fake preview control object bytes")
        controls.append(record)
    expected_control_uris = [
        _uri(prefix, f"source/{local.MANIFEST_NAME}"),
        _uri(prefix, f"source/{local.READY_NAME}"),
        _uri(prefix, f"source/{local.SOURCE_NAME}"),
        _uri(prefix, f"source/{local.STARTUP_NAME}"),
        _uri(prefix, "control/stage_token.json"),
        *[
            _uri(
                prefix,
                f"source/jobs/{value['stage_id']}/{job_id}.json",
            )
            for job_id in spec["job_ids"]
        ],
    ]
    if (
        [record["uri"] for record in controls] != expected_control_uris
        or controls[0]["sha256"] != package_sha
        or controls[4]["sha256"] != canonical_sha256(token)
        or controls[4]["bytes"] != len(canonical_bytes(token))
    ):
        raise ValueError("fake preview control-object identities changed")
    source_sha = controls[2]["sha256"]

    jobs_raw = value["jobs"]
    if not isinstance(jobs_raw, list) or len(jobs_raw) != len(spec["job_ids"]):
        raise ValueError("fake preview exact stage job set changed")
    jobs: list[dict[str, Any]] = []
    job_prefixes: list[dict[str, Any]] = []
    for offset, (raw, job_id, role) in enumerate(
        zip(jobs_raw, spec["job_ids"], spec["roles"], strict=True)
    ):
        if not isinstance(raw, Mapping):
            raise ValueError("fake preview job must be an object")
        job = dict(raw)
        _exact(
            job,
            {
                "job_id",
                "source_role",
                "instance_name",
                "attempt_id",
                "work_hand_indices",
                "root_records",
                "metadata",
                "metadata_sha256",
                "control_objects",
                "upload_uris",
                "heartbeat_uris",
                "done_uri",
                "failure_uri",
                "success_policy",
                "failure_policy",
                "resume_policy",
            },
            "fake preview job",
        )
        work_hand_indices = job["work_hand_indices"]
        if (
            job["job_id"] != job_id
            or job["source_role"] != role
            or not isinstance(work_hand_indices, list)
            or len(work_hand_indices) != len(spec["hand_indices"])
            or not isinstance(job["root_records"], list)
            or len(job["root_records"]) != len(spec["hand_indices"])
        ):
            raise ValueError("fake preview job membership changed")
        for actual, expected in zip(
            work_hand_indices, spec["hand_indices"], strict=True
        ):
            _exact_int(
                actual,
                expected,
                "fake preview work hand index",
            )
        for root, hand_index in zip(
            job["root_records"], spec["hand_indices"], strict=True
        ):
            if not isinstance(root, Mapping):
                raise ValueError("fake preview root record must be an object")
            _exact(
                root,
                {"hand_index", "archive_path", "sha256", "bytes"},
                "fake preview root record",
            )
            _exact_int(
                root["hand_index"],
                hand_index,
                "fake preview root hand index",
            )
            if (
                root["archive_path"] != f"roots/hand_{hand_index:03d}.json"
            ):
                raise ValueError("fake preview root identity changed")
            _nonzero_sha(root["sha256"], "fake preview root")
            _positive_int(root["bytes"], "fake preview root bytes")

        output_prefix = _uri(prefix, f"results/jobs/{job_id}")
        progress_prefix = _uri(prefix, f"progress/jobs/{job_id}")
        upload_uris = [
            _uri(output_prefix, f"uploads/hand_{index:03d}.json")
            for index in spec["hand_indices"]
        ]
        heartbeat_uris = [
            _uri(progress_prefix, f"heartbeats/{sequence:06d}.json")
            for sequence in range(1, len(spec["hand_indices"]) + 1)
        ]
        done_uri = _uri(output_prefix, "DONE.json")
        failure_uri = _uri(output_prefix, "FAILURE.json")
        attempt_id = f"{spec['run_name']}|{job_id}|attempt-0"
        instance_name = (
            f"r2diag-{spec['stage_number']}-"
            f"{'c' if role == 'candidate' else 'r'}-"
            f"{job_id.rsplit('-', 1)[-1]}-a0"
        )
        job_control = job["control_objects"]
        if (
            not isinstance(job_control, list)
            or len(job_control) != 1
            or not isinstance(job_control[0], Mapping)
        ):
            raise ValueError("fake preview job control binding changed")
        control = dict(job_control[0])
        _exact(control, {"uri", "sha256", "bytes"}, "fake preview job control")
        if control != controls[5 + offset]:
            raise ValueError("fake preview job/control ordering changed")
        job_manifest_sha = _nonzero_sha(
            control["sha256"], "fake preview job manifest"
        )
        _positive_int(control["bytes"], "fake preview job manifest bytes")
        metadata = {
            "ADAPTER_SCHEMA": adapter.ADAPTER_SCHEMA,
            "ATTEMPT_ID": attempt_id,
            "ATTEMPT_INDEX": "0",
            "BUCKET": adapter.DEFAULT_BUCKET,
            "CLOUD_LAUNCH_AUTHORIZED": "0",
            "COST_GUARD_SHA256": canonical_sha256(guard),
            "DIAGNOSTIC_ONLY": "1",
            "DONE_URI": done_uri,
            "FAILURE_URI": failure_uri,
            "GCLOUD_INVOCATION_AUTHORIZED": "0",
            "HEARTBEAT_INTERVAL_SECONDS": str(
                adapter.HEARTBEAT_INTERVAL_SECONDS
            ),
            "HEARTBEAT_PREFIX": _uri(progress_prefix, "heartbeats"),
            "INSTANCE_NAME": instance_name,
            "JOB_ID": job_id,
            "JOB_MANIFEST_SHA256": job_manifest_sha,
            "JOB_MANIFEST_URI": control["uri"],
            "MACHINE_TYPE": adapter.MACHINE_TYPE,
            "MAX_RUNTIME_SECONDS": str(adapter.MAX_RUNTIME_SECONDS_PER_VM),
            "PACKAGE_MANIFEST_SHA256": package_sha,
            "PACKAGE_MANIFEST_URI": controls[0]["uri"],
            "PRESERVE_VM_ON_FAILURE": "1",
            "PROJECT_ID": adapter.DEFAULT_PROJECT,
            "PUBLISH_DONE_LAST": "1",
            "RESULT_PREFIX": output_prefix,
            "RUN_NAME": spec["run_name"],
            "SELF_DELETE_ON_SUCCESS": "1",
            "SOURCE_ROLE": role,
            "SOURCE_SHA256": source_sha,
            "SOURCE_URI": controls[2]["uri"],
            "STAGE_ID": value["stage_id"],
            "STAGE_TOKEN_SHA256": canonical_sha256(token),
            "STAGE_TOKEN_URI": controls[4]["uri"],
            "UPLOAD_PREFIX": _uri(output_prefix, "uploads"),
            "WATCHDOG_SECONDS": str(adapter.WATCHDOG_SECONDS_PER_VM),
            "ZONE": adapter.DEFAULT_ZONE,
        }
        success_policy = job["success_policy"]
        failure_policy = job["failure_policy"]
        resume_policy = job["resume_policy"]
        if not isinstance(success_policy, Mapping):
            raise ValueError("fake preview success policy must be an object")
        if not isinstance(failure_policy, Mapping):
            raise ValueError("fake preview failure policy must be an object")
        if not isinstance(resume_policy, Mapping):
            raise ValueError("fake preview resume policy must be an object")
        success_policy = dict(success_policy)
        failure_policy = dict(failure_policy)
        resume_policy = dict(resume_policy)
        expected_success_policy = {
            "publish_done_last": True,
            "self_delete_requested_after_done_validation": True,
            "preserve_vm": False,
        }
        expected_failure_policy = {
            "publish_done": False,
            "self_delete_requested": False,
            "preserve_vm_for_diagnosis": True,
            "bounded_shutdown_required": True,
            "shutdown_deadline_seconds": adapter.MAX_RUNTIME_SECONDS_PER_VM,
        }
        expected_resume_policy = {
            "same_attempt_identity_only": True,
            "ordered_upload_heartbeat_prefix_only": True,
            "same_content_is_idempotent": True,
            "different_content_collision_is_fatal": True,
            "max_attempts": adapter.MAX_ATTEMPTS_PER_JOB,
        }
        _exact(
            success_policy,
            set(expected_success_policy),
            "fake preview success policy",
        )
        _exact(
            failure_policy,
            set(expected_failure_policy),
            "fake preview failure policy",
        )
        _exact(
            resume_policy,
            set(expected_resume_policy),
            "fake preview resume policy",
        )
        for field, expected in expected_success_policy.items():
            _exact_bool(
                success_policy[field],
                expected,
                f"fake preview success policy {field}",
            )
        for field in (
            "publish_done",
            "self_delete_requested",
            "preserve_vm_for_diagnosis",
            "bounded_shutdown_required",
        ):
            _exact_bool(
                failure_policy[field],
                expected_failure_policy[field],
                f"fake preview failure policy {field}",
            )
        _exact_int(
            failure_policy["shutdown_deadline_seconds"],
            adapter.MAX_RUNTIME_SECONDS_PER_VM,
            "fake preview failure shutdown deadline",
        )
        for field in (
            "same_attempt_identity_only",
            "ordered_upload_heartbeat_prefix_only",
            "same_content_is_idempotent",
            "different_content_collision_is_fatal",
        ):
            _exact_bool(
                resume_policy[field],
                expected_resume_policy[field],
                f"fake preview resume policy {field}",
            )
        _exact_int(
            resume_policy["max_attempts"],
            adapter.MAX_ATTEMPTS_PER_JOB,
            "fake preview resume max attempts",
        )
        if (
            job["instance_name"] != instance_name
            or job["attempt_id"] != attempt_id
            or job["metadata"] != metadata
            or job["metadata_sha256"] != canonical_sha256(metadata)
            or job["upload_uris"] != upload_uris
            or job["heartbeat_uris"] != heartbeat_uris
            or job["done_uri"] != done_uri
            or job["failure_uri"] != failure_uri
            or success_policy != expected_success_policy
            or failure_policy != expected_failure_policy
            or resume_policy != expected_resume_policy
        ):
            raise ValueError("fake preview job hash, metadata, or URI changed")
        jobs.append(job)
        job_prefixes.append(
            {
                "job_id": job_id,
                "upload_uris": upload_uris,
                "heartbeat_uris": heartbeat_uris,
                "done_uri": done_uri,
                "failure_uri": failure_uri,
            }
        )

    expected_remote = {
        "schema": adapter.REMOTE_MANIFEST_SCHEMA,
        "prefix": prefix,
        "stage_id": value["stage_id"],
        "run_name": spec["run_name"],
        "freshness_preflight": freshness,
        "control_objects": controls,
        "job_prefixes": job_prefixes,
        "receive_uri": _uri(
            prefix, f"received/{spec['result_name']}"
        ),
    }
    if (
        remote != expected_remote
        or value["remote_manifest_sha256"] != canonical_sha256(remote)
    ):
        raise ValueError("fake preview remote-manifest identity changed")
    return value


def _validate_controller_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = dict(value)
    _exact(
        receipt,
        {
            "schema",
            "status",
            "package_manifest_sha256",
            "stage_id",
            "run_name",
            "selected_job_ids",
            "receive_sha256",
            "all_vms_absent",
            "diagnostic_only",
            "performance_lock_evidence",
            "quality_evidence",
            "training_eligible",
            "promotion_evidence",
        },
        "fake controller receipt",
    )
    if (
        receipt["schema"] != CONTROLLER_RECEIPT_SCHEMA
        or receipt["status"]
        != "fake_stage_received_and_controller_proved_all_vms_absent"
        or receipt["stage_id"] != plan.STAGE1_ID
        or receipt["run_name"] != plan.STAGE1_RUN_NAME
        or receipt["selected_job_ids"] != list(plan.STAGE1_JOB_IDS)
        or not isinstance(receipt["package_manifest_sha256"], str)
        or len(receipt["package_manifest_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in receipt["package_manifest_sha256"]
        )
        or not isinstance(receipt["receive_sha256"], str)
        or len(receipt["receive_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in receipt["receive_sha256"]
        )
        or receipt["all_vms_absent"] is not True
        or receipt["diagnostic_only"] is not True
        or any(
            receipt[field] is not False
            for field in (
                "performance_lock_evidence",
                "quality_evidence",
                "training_eligible",
                "promotion_evidence",
            )
        )
    ):
        raise ValueError("fake stage1 controller receipt changed")
    return receipt


class FakeRemoteStage:
    """State machine for one stage of the diagnostic VM preview."""

    def __init__(
        self,
        preview: Mapping[str, Any],
        *,
        store: InMemoryGenerationStore,
        prerequisite_stage1_controller_receipt: ControllerReceiptProof | None = None,
    ) -> None:
        self.preview = _validate_preview_boundary(preview)
        self.store = store
        self.prefix = str(self.preview["remote_manifest"]["prefix"])
        self._jobs = {
            str(job["job_id"]): dict(job) for job in self.preview["jobs"]
        }
        self._attempts: dict[str, list[int]] = {
            job_id: [] for job_id in self._jobs
        }
        self._terminal: dict[str, str | None] = {
            job_id: None for job_id in self._jobs
        }
        self._vm_present: dict[str, bool] = {
            job_id: False for job_id in self._jobs
        }
        self._shutdown_state: dict[str, dict[str, Any] | None] = {
            job_id: None for job_id in self._jobs
        }
        self._proof_nonce = object()
        prerequisite_sha: str | None = None
        if self.preview["stage_id"] == plan.STAGE1_ID:
            if prerequisite_stage1_controller_receipt is not None:
                raise ValueError("fake stage1 must not receive a stage1 prerequisite")
            if self.preview["stage_token"]["prerequisite_stage1_receive"] is not None:
                raise ValueError("fake stage1 preview unexpectedly has a prerequisite")
        else:
            if prerequisite_stage1_controller_receipt is None:
                raise ValueError(
                    "fake stage2 requires controller-proven stage1 receive and VM absence"
                )
            if not isinstance(
                prerequisite_stage1_controller_receipt, ControllerReceiptProof
            ):
                raise ValueError(
                    "fake stage2 requires an opaque stage1 controller proof"
                )
            source = prerequisite_stage1_controller_receipt._source_stage
            if (
                not isinstance(source, FakeRemoteStage)
                or prerequisite_stage1_controller_receipt._issuer_nonce
                is not source._proof_nonce
                or source.preview["stage_id"] != plan.STAGE1_ID
                or source.controller_receipt()
                != prerequisite_stage1_controller_receipt.content
                or source.validate_data_state()["receive_validated"] is not True
                or any(source._vm_present.values())
            ):
                raise ValueError("fake stage1 controller proof provenance changed")
            receipt = _validate_controller_receipt(
                prerequisite_stage1_controller_receipt.content
            )
            local_binding = self.preview["stage_token"].get(
                "prerequisite_stage1_receive"
            )
            if (
                not isinstance(local_binding, Mapping)
                or receipt["package_manifest_sha256"]
                != self.preview["package_manifest_sha256"]
                or local_binding.get("package_manifest_sha256")
                != self.preview["package_manifest_sha256"]
                or local_binding.get("stage_id") != plan.STAGE1_ID
                or local_binding.get("run_name") != plan.STAGE1_RUN_NAME
                or local_binding.get("selected_job_ids")
                != list(plan.STAGE1_JOB_IDS)
            ):
                raise ValueError("fake stage2 prerequisite binding changed")
            prerequisite_sha = canonical_sha256(receipt)

        if self.store.list_prefix(self.prefix):
            raise FileExistsError(
                "fake stage prefix is not exactly empty before sentinel acquisition"
            )
        self.sentinel_uri = _uri(self.prefix, SENTINEL_NAME)
        self.authorization_uri = _uri(self.prefix, AUTHORIZATION_NAME)
        self.claim_uri = _uri(self.prefix, CLAIM_NAME)
        sentinel = {
            "schema": PREFIX_SENTINEL_SCHEMA,
            "status": "fake_generation_match_zero_prefix_claimed",
            "preview_sha256": canonical_sha256(self.preview),
            "package_manifest_sha256": self.preview["package_manifest_sha256"],
            "stage_id": self.preview["stage_id"],
            "run_name": self.preview["run_name"],
            "prefix": self.prefix,
            "generation_match": 0,
            "simulation_only": True,
            "cloud_launch_authorized": False,
        }
        self._put_and_readback(self.sentinel_uri, sentinel)
        authorization = {
            "schema": SIMULATED_AUTHORIZATION_SCHEMA,
            "status": "fake_authorization_record_no_cloud_capability",
            "preview_sha256": canonical_sha256(self.preview),
            "package_manifest_sha256": self.preview["package_manifest_sha256"],
            "stage_id": self.preview["stage_id"],
            "run_name": self.preview["run_name"],
            "selected_job_ids": self.preview["selected_job_ids"],
            "prefix": self.prefix,
            "prefix_sentinel_sha256": canonical_sha256(sentinel),
            "prerequisite_stage1_controller_receipt_sha256": prerequisite_sha,
            "simulation_only": True,
            "cloud_launch_authorized": False,
            "gcloud_invocation_authorized": False,
            "vm_create_authorized": False,
        }
        self._put_and_readback(self.authorization_uri, authorization)
        claim = {
            "schema": SIMULATED_CLAIM_SCHEMA,
            "status": "exclusive_fake_claim_acquired_before_simulated_vm_state",
            "preview_sha256": canonical_sha256(self.preview),
            "stage_id": self.preview["stage_id"],
            "run_name": self.preview["run_name"],
            "selected_job_ids": self.preview["selected_job_ids"],
            "prefix": self.prefix,
            "simulated_authorization_sha256": canonical_sha256(authorization),
            "prefix_sentinel_sha256": canonical_sha256(sentinel),
            "simulation_only": True,
            "cloud_launch_authorized": False,
            "gcloud_invocation_authorized": False,
            "vm_create_authorized": False,
        }
        self._put_and_readback(self.claim_uri, claim)
        self.sentinel = sentinel
        self.authorization = authorization
        self.claim = claim
        self._assert_prefix_inventory()

    def _put_and_readback(
        self, uri: str, content: Mapping[str, Any]
    ) -> PutResult:
        result = self.store.put_once(uri, content)
        if self.store.read(uri) != dict(content):
            raise ValueError("fake remote immutable put failed readback")
        return result

    def _job(self, job_id: str) -> dict[str, Any]:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise ValueError("fake attempt escaped the exact stage job set") from exc

    def _attempt_uri(self, job_id: str, attempt_index: int) -> str:
        return _uri(
            self.prefix,
            f"control/attempts/{job_id}/attempt-{attempt_index}.json",
        )

    def _attempt_record(
        self, *, job_id: str, attempt_index: int
    ) -> dict[str, Any]:
        return {
            "schema": ATTEMPT_CLAIM_SCHEMA,
            "status": "fake_attempt_claimed_generation_match_zero",
            "preview_sha256": canonical_sha256(self.preview),
            "simulated_claim_sha256": canonical_sha256(self.claim),
            "stage_id": self.preview["stage_id"],
            "run_name": self.preview["run_name"],
            "job_id": job_id,
            "attempt_index": attempt_index,
            "simulation_only": True,
            "cloud_launch_authorized": False,
            "vm_create_authorized": False,
        }

    def _assert_prefix_inventory(self) -> None:
        allowed = {
            self.sentinel_uri,
            self.authorization_uri,
            self.claim_uri,
            str(self.preview["remote_manifest"]["receive_uri"]),
        }
        for job in self.preview["jobs"]:
            allowed.update(job["upload_uris"])
            allowed.update(job["heartbeat_uris"])
            allowed.update((job["done_uri"], job["failure_uri"]))
            allowed.update(
                self._attempt_uri(job["job_id"], attempt_index)
                for attempt_index in self._attempts[job["job_id"]]
            )
        actual = set(self.store.list_prefix(self.prefix))
        unknown = sorted(actual - allowed)
        if unknown:
            raise ValueError(
                "fake remote prefix contains an unknown or rogue object: "
                + ",".join(unknown)
            )
        required = {
            self.sentinel_uri: self.sentinel,
            self.authorization_uri: self.authorization,
            self.claim_uri: self.claim,
        }
        for uri, expected in required.items():
            if not self.store.contains(uri) or self.store.read(uri) != expected:
                raise ValueError("fake remote control chain changed")
        for job_id, attempts in self._attempts.items():
            for attempt_index in attempts:
                expected = self._attempt_record(
                    job_id=job_id, attempt_index=attempt_index
                )
                uri = self._attempt_uri(job_id, attempt_index)
                if not self.store.contains(uri) or self.store.read(uri) != expected:
                    raise ValueError("fake remote attempt claim changed")

    def start_attempt(self, *, job_id: str, attempt_index: int) -> dict[str, Any]:
        self._job(job_id)
        self._assert_prefix_inventory()
        if type(attempt_index) is not int or attempt_index not in (0, 1):
            raise ValueError("fake diagnostic attempts are limited to zero and one")
        attempts = self._attempts[job_id]
        if attempt_index != len(attempts) or attempt_index in attempts:
            raise ValueError("fake attempt order or reuse changed")
        if self._vm_present[job_id]:
            raise ValueError("fake job already has a present VM")
        if self._terminal[job_id] in ("success", "failure"):
            raise ValueError("terminal fake job cannot be attempted again")
        if attempt_index == 0:
            uploads, _heartbeats, pending = self._inspect_progress(job_id)
            if uploads or pending is not None:
                raise ValueError("attempt zero requires an empty job result prefix")
        else:
            if self._terminal[job_id] != "interrupted":
                raise ValueError("attempt one requires an interrupted attempt zero")
            # A complete upload/heartbeat prefix without DONE is still
            # resumable: attempt one may validate it and publish DONE without
            # rewriting any scientific object.
            self._inspect_progress(job_id)
        record = self._attempt_record(
            job_id=job_id, attempt_index=attempt_index
        )
        self._put_and_readback(
            self._attempt_uri(job_id, attempt_index), record
        )
        attempts.append(attempt_index)
        self._terminal[job_id] = None
        self._vm_present[job_id] = True
        self._shutdown_state[job_id] = None
        self._assert_prefix_inventory()
        return record

    def start_initial_exact(
        self, selected_job_ids: Sequence[str] | None = None
    ) -> list[dict[str, Any]]:
        expected = list(self.preview["selected_job_ids"])
        selected = expected if selected_job_ids is None else list(selected_job_ids)
        if selected != expected:
            raise ValueError("fake initial stage requires the exact ordered job set")
        return [
            self.start_attempt(job_id=job_id, attempt_index=0)
            for job_id in selected
        ]

    def _object_content(self, uri: str) -> dict[str, Any] | None:
        return self.store.read(uri) if self.store.contains(uri) else None

    def _inspect_progress(
        self, job_id: str
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        tuple[int, dict[str, Any]] | None,
    ]:
        job = self._job(job_id)
        uploads: list[dict[str, Any]] = []
        heartbeats: list[dict[str, Any]] = []
        pending_upload: tuple[int, dict[str, Any]] | None = None
        gap_seen = False
        for sequence, (upload_uri, heartbeat_uri) in enumerate(
            zip(job["upload_uris"], job["heartbeat_uris"], strict=True), 1
        ):
            upload = self._object_content(upload_uri)
            heartbeat = self._object_content(heartbeat_uri)
            if upload is None and heartbeat is not None:
                raise ValueError("fake heartbeat exists without its upload")
            if upload is None:
                gap_seen = True
                continue
            if gap_seen:
                raise ValueError("fake job has a non-contiguous upload prefix")
            expected_upload = adapter.build_upload(
                self.preview,
                job_id=job_id,
                sequence=sequence,
                result=upload.get("result", {}),
            )
            if upload != expected_upload:
                raise ValueError("fake immutable upload changed")
            if heartbeat is None:
                if pending_upload is not None or sequence != len(uploads) + 1:
                    raise ValueError("fake job has multiple pending upload steps")
                pending_upload = (sequence, upload)
                gap_seen = True
                continue
            uploads.append(upload)
            expected_heartbeat = adapter.build_heartbeat(
                self.preview, job_id=job_id, uploads=uploads
            )
            if heartbeat != expected_heartbeat:
                raise ValueError("fake immutable heartbeat changed")
            heartbeats.append(heartbeat)
        return uploads, heartbeats, pending_upload

    def _completed_content(
        self, job_id: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        uploads, heartbeats, pending = self._inspect_progress(job_id)
        if pending is not None:
            raise ValueError("fake job has an upload awaiting its heartbeat")
        return uploads, heartbeats

    def _completed_count(self, job_id: str) -> int:
        uploads, _heartbeats, _pending = self._inspect_progress(job_id)
        return len(uploads)

    def upload_sequence(
        self,
        *,
        job_id: str,
        sequence: int,
        result: Mapping[str, Any],
        fault_after: str | None = None,
    ) -> dict[str, Any]:
        job = self._job(job_id)
        self._assert_prefix_inventory()
        if not self._vm_present[job_id] or self._terminal[job_id] is not None:
            raise ValueError("fake upload requires an active non-terminal attempt")
        if fault_after not in (None, "before_upload", "after_upload", "after_heartbeat"):
            raise ValueError("unknown fake fault injection point")
        if fault_after == "before_upload":
            raise RuntimeError("injected failure before upload")
        completed_uploads, _heartbeats, pending = self._inspect_progress(job_id)
        completed = len(completed_uploads)
        permitted_next = pending[0] if pending is not None else completed + 1
        if (
            type(sequence) is not int
            or sequence < 1
            or sequence > permitted_next
            or (pending is not None and sequence > completed and sequence != pending[0])
        ):
            raise ValueError("fake upload sequence is not the exact next/idempotent item")
        upload = adapter.build_upload(
            self.preview, job_id=job_id, sequence=sequence, result=result
        )
        self._put_and_readback(job["upload_uris"][sequence - 1], upload)
        if fault_after == "after_upload":
            raise RuntimeError("injected failure after upload before heartbeat")
        prior_uploads, _prior_heartbeats = self._completed_content_for_heartbeat(
            job_id, sequence, upload
        )
        heartbeat = adapter.build_heartbeat(
            self.preview, job_id=job_id, uploads=prior_uploads
        )
        self._put_and_readback(job["heartbeat_uris"][sequence - 1], heartbeat)
        if fault_after == "after_heartbeat":
            raise RuntimeError("injected failure after heartbeat")
        return upload

    def _completed_content_for_heartbeat(
        self,
        job_id: str,
        sequence: int,
        current: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        job = self._job(job_id)
        uploads: list[dict[str, Any]] = []
        heartbeats: list[dict[str, Any]] = []
        for offset in range(1, sequence + 1):
            upload = (
                dict(current)
                if offset == sequence
                else self.store.read(job["upload_uris"][offset - 1])
            )
            uploads.append(upload)
            if offset < sequence:
                heartbeats.append(
                    self.store.read(job["heartbeat_uris"][offset - 1])
                )
        return uploads, heartbeats

    def interrupt_for_resume(self, *, job_id: str) -> dict[str, Any]:
        job = self._job(job_id)
        self._assert_prefix_inventory()
        if (
            not self._vm_present[job_id]
            or self._terminal[job_id] is not None
            or self.store.contains(job["done_uri"])
            or self.store.contains(job["failure_uri"])
        ):
            raise ValueError("fake job is not interruptible")
        uploads, _heartbeats, pending = self._inspect_progress(job_id)
        state = {
            "job_id": job_id,
            "state": "interrupted_requires_bounded_shutdown",
            "attempt_index": self._attempts[job_id][-1],
            "completed_sequences": len(uploads),
            "pending_upload_sequence": (
                None if pending is None else pending[0]
            ),
            "done_published": False,
            "self_delete_requested": False,
            "bounded_shutdown_required": True,
            "shutdown_deadline_seconds": adapter.MAX_RUNTIME_SECONDS_PER_VM,
        }
        # Mutate terminal state only after the complete remote prefix has
        # passed validation.
        self._terminal[job_id] = "interrupted"
        self._shutdown_state[job_id] = state
        return dict(state)

    def publish_failure(
        self, *, job_id: str, failure_code: str
    ) -> dict[str, Any]:
        job = self._job(job_id)
        self._assert_prefix_inventory()
        if not self._vm_present[job_id] or self._terminal[job_id] is not None:
            raise ValueError("fake failure requires an active attempt")
        if self.store.contains(job["done_uri"]):
            raise ValueError("fake failure cannot coexist with DONE")
        failure = adapter.build_failure(
            self.preview,
            job_id=job_id,
            failure_code=failure_code,
            completed_sequences=self._completed_count(job_id),
        )
        self._put_and_readback(job["failure_uri"], failure)
        if self.store.contains(job["done_uri"]):
            raise AssertionError("fake failure unexpectedly published DONE")
        self._terminal[job_id] = "failure"
        self._shutdown_state[job_id] = {
            "job_id": job_id,
            "state": "failure_preserved_requires_bounded_shutdown",
            "attempt_index": self._attempts[job_id][-1],
            "completed_sequences": self._completed_count(job_id),
            "done_published": False,
            "self_delete_requested": False,
            "bounded_shutdown_required": True,
            "shutdown_deadline_seconds": adapter.MAX_RUNTIME_SECONDS_PER_VM,
        }
        return failure

    def publish_done(self, *, job_id: str) -> dict[str, Any]:
        job = self._job(job_id)
        self._assert_prefix_inventory()
        if not self._vm_present[job_id] or self._terminal[job_id] is not None:
            raise ValueError("fake DONE requires an active attempt")
        if self.store.contains(job["failure_uri"]):
            raise ValueError("fake DONE cannot coexist with FAILURE")
        uploads, heartbeats = self._completed_content(job_id)
        done = adapter.build_done(
            self.preview,
            job_id=job_id,
            uploads=uploads,
            heartbeats=heartbeats,
        )
        self._put_and_readback(job["done_uri"], done)
        if self.store.read(job["done_uri"]) != done:
            raise ValueError("fake DONE failed post-write readback")
        self._terminal[job_id] = "success"
        self._shutdown_state[job_id] = {
            "job_id": job_id,
            "state": "success_done_validated_self_delete_requested",
            "attempt_index": self._attempts[job_id][-1],
            "completed_sequences": len(uploads),
            "done_published": True,
            "self_delete_requested": True,
            "bounded_shutdown_required": True,
            "shutdown_deadline_seconds": adapter.MAX_RUNTIME_SECONDS_PER_VM,
        }
        return done

    def controller_mark_vm_absent(self, *, job_id: str) -> dict[str, Any]:
        self._job(job_id)
        self._assert_prefix_inventory()
        if self._terminal[job_id] is None or not self._vm_present[job_id]:
            raise ValueError("controller cannot prove absence for this fake VM state")
        self._vm_present[job_id] = False
        state = dict(self._shutdown_state[job_id] or {})
        state["controller_vm_absent"] = True
        state["vm_present"] = False
        self._shutdown_state[job_id] = state
        return state

    def resume_exact(self, selected_job_ids: Sequence[str]) -> list[dict[str, Any]]:
        self._assert_prefix_inventory()
        expected = [
            job_id
            for job_id in self.preview["selected_job_ids"]
            if self._terminal[job_id] == "interrupted"
            and self._attempts[job_id] == [0]
            and not self._vm_present[job_id]
        ]
        if list(selected_job_ids) != expected or not expected:
            raise ValueError("fake resume requires the exact ordered incomplete set")
        return [
            self.start_attempt(job_id=job_id, attempt_index=1)
            for job_id in expected
        ]

    def _snapshot(self) -> dict[str, Any]:
        self._assert_prefix_inventory()
        allowed: list[str] = []
        for job in self.preview["jobs"]:
            allowed.extend(job["upload_uris"])
            allowed.extend(job["heartbeat_uris"])
            allowed.extend((job["done_uri"], job["failure_uri"]))
        allowed.append(self.preview["remote_manifest"]["receive_uri"])
        records = [
            self.store.record(uri) for uri in allowed if self.store.contains(uri)
        ]
        return {
            "schema": adapter.SNAPSHOT_SCHEMA,
            "preview_sha256": canonical_sha256(self.preview),
            "stage_id": self.preview["stage_id"],
            "run_name": self.preview["run_name"],
            "objects": records,
            "cloud_query_performed": False,
            "cloud_write_performed": False,
        }

    def validate_data_state(self) -> dict[str, Any]:
        return adapter.validate_snapshot(self.preview, self._snapshot())

    def receive(self) -> dict[str, Any]:
        self._assert_prefix_inventory()
        if any(
            self._terminal[job_id] != "success"
            for job_id in self.preview["selected_job_ids"]
        ):
            raise ValueError("fake receive requires DONE for the exact stage job set")
        if any(
            self._vm_present[job_id]
            for job_id in self.preview["selected_job_ids"]
        ):
            raise ValueError(
                "fake receive requires controller-proven absence for every VM"
            )
        done_records = [
            self.store.read(self._jobs[job_id]["done_uri"])
            for job_id in self.preview["selected_job_ids"]
        ]
        receive = adapter.build_receive(
            self.preview, done_records=done_records
        )
        uri = str(self.preview["remote_manifest"]["receive_uri"])
        self._put_and_readback(uri, receive)
        checked = self.validate_data_state()
        if checked["receive_validated"] is not True:
            raise ValueError("fake receive did not validate after readback")
        return receive

    def controller_receipt(self) -> dict[str, Any]:
        self._assert_prefix_inventory()
        receive_uri = str(self.preview["remote_manifest"]["receive_uri"])
        if not self.store.contains(receive_uri):
            raise ValueError("fake controller receipt requires validated receive")
        if any(self._vm_present.values()):
            raise ValueError("fake controller receipt requires every VM absent")
        receive = self.store.read(receive_uri)
        value = {
            "schema": CONTROLLER_RECEIPT_SCHEMA,
            "status": "fake_stage_received_and_controller_proved_all_vms_absent",
            "package_manifest_sha256": self.preview["package_manifest_sha256"],
            "stage_id": self.preview["stage_id"],
            "run_name": self.preview["run_name"],
            "selected_job_ids": self.preview["selected_job_ids"],
            "receive_sha256": canonical_sha256(receive),
            "all_vms_absent": True,
            "diagnostic_only": True,
            "performance_lock_evidence": False,
            "quality_evidence": False,
            "training_eligible": False,
            "promotion_evidence": False,
        }
        if self.preview["stage_id"] == plan.STAGE1_ID:
            _validate_controller_receipt(value)
        return value

    def controller_proof(self) -> ControllerReceiptProof:
        if self.preview["stage_id"] != plan.STAGE1_ID:
            raise ValueError("only fake stage1 can issue the stage2 prerequisite")
        receipt = self.controller_receipt()
        return ControllerReceiptProof(
            receipt_bytes=canonical_bytes(receipt),
            _source_stage=self,
            _issuer_nonce=self._proof_nonce,
        )


__all__ = [
    "ATTEMPT_CLAIM_SCHEMA",
    "AUTHORIZATION_NAME",
    "CLAIM_NAME",
    "CONTROLLER_RECEIPT_SCHEMA",
    "ControllerReceiptProof",
    "FakeRemoteStage",
    "InMemoryGenerationStore",
    "PREFIX_SENTINEL_SCHEMA",
    "PutResult",
    "SIMULATED_AUTHORIZATION_SCHEMA",
    "SIMULATED_CLAIM_SCHEMA",
    "canonical_bytes",
    "canonical_sha256",
]
