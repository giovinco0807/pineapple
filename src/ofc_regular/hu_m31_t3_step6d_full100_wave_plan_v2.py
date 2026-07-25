"""Fail-closed local contract for the Candidate02 full-100 wave transport.

The scientific full-100 plan is frozen.  This module only groups its twenty
source-isolated jobs into quota-bounded 8+8+4 waves and models immutable
attempt/readback evidence.  It deliberately cannot authorize a cloud launch:
the live quota readback, atomic persistent launch claim, launch-receipt
reconciliation, and owned-root fsync checks are not connected here.

No function invokes gcloud or changes an AI profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_step6d_candidate02_full100_plan as full100
from . import hu_m31_t3_step6d_full100_wave_science_registry_v2 as science_registry


WAVE_PLAN_SCHEMA = "hu_m31_t3_step6d_full100_wave_plan_v2"
OBSERVED_TRANSITION_SCHEMA = "hu_m31_t3_step6d_full100_observed_transition_v1"
ATTEMPT_LEDGER_SCHEMA = "hu_m31_t3_step6d_full100_attempt_ledger_v1"
RESUME_PLAN_SCHEMA = "hu_m31_t3_step6d_full100_wave_resume_plan_v3"
EXPECTED_INVENTORY_SCHEMA = "hu_m31_t3_step6d_full100_expected_inventory_v3"
OBSERVED_INVENTORY_SCHEMA = "hu_m31_t3_step6d_full100_observed_inventory_v1"
ARTIFACT_INVENTORY_SCHEMA = EXPECTED_INVENTORY_SCHEMA

WAVE_PLAN_STATUS = science_registry.DEVELOPMENT_WAVE_STATUS
WAVE_PLAN_DECISION = science_registry.DEVELOPMENT_WAVE_DECISION
FULL100_EXECUTION_SCOPE = science_registry.DEVELOPMENT_EXECUTION_SCOPE
PERFORMANCE_LOCK_V4_EXECUTION_SCOPE = (
    science_registry.PERFORMANCE_LOCK_V4_EXECUTION_SCOPE
)
STARTUP_CANARY_SCOPE = science_registry.STARTUP_CANARY_EXECUTION_SCOPE
STARTUP_CANARY_JOB_ID = "candidate-shard-00"
STARTUP_CANARY_SOURCE_ROLE = "candidate"
STARTUP_CANARY_ATTEMPT_ID = "a00"
_EXECUTION_SCOPES = frozenset(
    {
        FULL100_EXECUTION_SCOPE,
        PERFORMANCE_LOCK_V4_EXECUTION_SCOPE,
        STARTUP_CANARY_SCOPE,
    }
)

MACHINE_TYPE = "c4-standard-16"
VCPUS_PER_VM = 16
TOKYO_C4_FAMILY_QUOTA_VCPUS = 128
MAX_CONCURRENT_VMS = TOKYO_C4_FAMILY_QUOTA_VCPUS // VCPUS_PER_VM
WAVE_SHARD_GROUPS = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9))
WAVE_VM_COUNTS = tuple(2 * len(group) for group in WAVE_SHARD_GROUPS)
MAX_ATTEMPTS_PER_JOB = 2
SOURCE_ROLES = ("candidate", "reference")
ATTEMPT_IDS = ("a00", "a01")

AUTHORIZATION_BLOCKERS = (
    "persistent_atomic_launch_claim_not_implemented",
    "live_quota_and_headroom_readback_not_connected",
    "launch_receipt_one_vm_one_job_one_role_not_connected",
    "parent_directory_fsync_and_owned_root_verification_not_connected",
)

RUN_NAME_PREFIX = "regular-hu-m31-c02-f100wv2-"
PREVIOUS_RUN_NAMES = frozenset({"regular-hu-m31-c02-full100-dev-20260717-002"})
_SAFE_RUN = re.compile(r"^[a-z0-9][a-z0-9-]{2,62}[a-z0-9]$")
_SAFE_INSTANCE = re.compile(r"^[a-z0-9][a-z0-9-]{2,61}[a-z0-9]$")
_SAFE_PROJECT = re.compile(r"^[a-z][a-z0-9-]{4,61}[a-z0-9]$")
_SAFE_ZONE = re.compile(r"^[a-z]+-[a-z0-9]+[0-9]-[a-z]$")
_IDENTITY_SALT = re.compile(r"^[0-9a-f]{32}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_UTC_SECONDS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FULL100_PLAN_PATH = science_registry.descriptor_for_kind(
    science_registry.DEVELOPMENT_SCIENCE_KIND
).resolved_default_plan_path()

_PLAN_KEYS = frozenset(
    {
        "schema", "status", "decision", "scope", "run_name",
        "execution_identity_sha256", "identity_tag", "full100_plan",
        "full100_plan_sha256", "run_contract_digest", "root_set",
        "seed_contract", "runtime_binding", "quota_contract",
        "pair_co_location_policy", "artifact_contract", "resume_contract",
        "wave_count", "wave_vm_counts", "waves", "coverage",
        "authorization_blockers", "cloud_launcher_ready",
        "cloud_launch_authorized", "cloud_started",
        "performance_lock_authorized", "quality_pilot_authorized",
        "training_eligible", "current_profile_changed", "named_profile_added",
        "runtime_policy_activated", "schedule_sha256",
    }
)
_WAVE_KEYS = frozenset(
    {
        "wave_index", "wave_id", "prerequisite_wave_ids", "shard_indices",
        "job_ids", "candidate_reference_pairs", "vm_count", "vcpu_count",
        "max_concurrent_vms", "launch_requires_prior_wave_complete",
        "launch_requires_all_owned_vms_quiescent", "create_only",
    }
)
_PAIR_KEYS = frozenset(
    {
        "pair_id", "shard_index", "work_hand_indices", "candidate_job_id",
        "reference_job_id", "candidate_attempt_instance_ids",
        "reference_attempt_instance_ids", "same_wave_required",
        "distinct_vm_required", "distinct_process_required",
    }
)
_RUNTIME_KEYS = frozenset(
    {"package_sha256", "image_digest", "binary_sha256_by_role", "allocation_digest"}
)
_TRANSITION_KEYS = frozenset(
    {
        "schema", "run_name", "execution_identity_sha256", "project_id",
        "zone", "observed_at_utc", "readback_source",
        "previous_transition_digest", "owned_vm_readback_complete",
        "owned_vm_quiescent", "owned_vms", "attempt_history", "done_objects",
        "acceptance_records", "transition_digest",
    }
)
_HISTORY_KEYS = frozenset({"job_id", "source_role", "attempts"})
_ATTEMPT_KEYS = frozenset(
    {"attempt_id", "instance_id", "launch_receipt_sha256", "terminal_status"}
)
_OWNED_VM_KEYS = frozenset(
    {"instance_id", "job_id", "source_role", "attempt_id", "status"}
)
_DONE_KEYS = frozenset(
    {
        "job_id", "source_role", "attempt_id", "path", "generation", "bytes",
        "sha256", "done_identity_sha256", "package_sha256", "image_digest",
        "binary_sha256", "allocation_digest", "root_digest",
    }
)
_ACCEPTANCE_KEYS = frozenset(
    {
        "job_id", "source_role", "attempt_id", "path", "generation", "bytes",
        "sha256", "done_generation", "done_sha256", "create_only",
    }
)
_LEDGER_KEYS = frozenset(
    {
        "schema", "wave_plan_sha256", "run_name", "execution_identity_sha256",
        "transitions", "latest_transition_digest", "consumed_transition_digests",
        "cloud_claim_persistent", "ledger_sha256",
    }
)
_RESUME_KEYS = frozenset(
    {
        "schema", "status", "wave_plan_sha256", "attempt_ledger_sha256",
        "run_name", "execution_identity_sha256", "observed_transition_digest",
        "resume_wave_index", "selected_attempts", "all_jobs_complete",
        "owned_vm_quiescence_proven", "transition_unconsumed",
        "authorization_blockers", "cloud_launcher_ready",
        "cloud_launch_authorized", "third_attempt_authorized", "resume_sha256",
    }
)
_SELECTED_KEYS = frozenset(
    {"job_id", "source_role", "attempt_id", "instance_id", "artifact_prefix"}
)
_EXPECTED_INVENTORY_KEYS = frozenset(
    {
        "schema", "run_name", "execution_identity_sha256", "wave_plan_sha256",
        "attempt_ledger_sha256", "artifact_prefix", "job_count",
        "paired_hand_count", "source_root_count", "source_hand_count",
        "object_count", "records", "content_sha256_required",
        "object_generation_required", "overwrite_forbidden", "inventory_sha256",
    }
)
_EXPECTED_RECORD_KEYS = frozenset(
    {
        "job_id", "source_role", "shard_index", "work_hand_indices",
        "accepted_attempt_id", "attempt_prefix", "done_path", "acceptance_path",
        "root_paths", "source_hand_paths", "run_contract_digest",
        "done_identity_sha256", "package_sha256", "image_digest",
        "binary_sha256", "allocation_digest", "root_digest",
        "done_generation", "done_bytes", "done_sha256",
        "acceptance_generation", "acceptance_bytes", "acceptance_sha256",
        "immutable_write_once", "done_written_last", "acceptance_create_only",
    }
)
_OBSERVED_INVENTORY_KEYS = frozenset(
    {
        "schema", "run_name", "execution_identity_sha256",
        "expected_inventory_sha256", "observed_at_utc", "readback_complete",
        "objects", "object_count", "inventory_sha256",
    }
)
_OBSERVED_OBJECT_KEYS = frozenset(
    {
        "path", "job_id", "source_role", "attempt_id", "object_kind",
        "hand_index", "generation", "bytes", "sha256", "done_identity_sha256",
        "package_sha256", "image_digest", "binary_sha256",
        "allocation_digest", "root_digest",
    }
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields changed")


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _parse_utc_seconds(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or _UTC_SECONDS.fullmatch(value) is None:
        raise ValueError(f"{label} must be canonical UTC seconds")
    parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    if parsed.tzinfo != timezone.utc:
        raise ValueError(f"{label} must be UTC")
    return parsed


def _read_frozen_plan(path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve()
    if target.is_symlink() or not target.is_file():
        raise ValueError("frozen full100 plan is missing or unsafe")
    raw = target.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError("frozen full100 plan is not canonical JSON")
    descriptor, frozen = science_registry.validate_scientific_plan(value)
    if sha256_file(target) != descriptor.plan_sha256:
        raise ValueError("frozen full100 plan hash changed")
    return frozen


def _validate_run_and_salt(run_name: Any, identity_salt: Any) -> tuple[str, str]:
    if (
        not isinstance(run_name, str)
        or _SAFE_RUN.fullmatch(run_name) is None
        or not run_name.startswith(RUN_NAME_PREFIX)
        or run_name in PREVIOUS_RUN_NAMES
    ):
        raise ValueError("full100 wave run name is not a fresh v2 identity")
    if (
        not isinstance(identity_salt, str)
        or _IDENTITY_SALT.fullmatch(identity_salt) is None
        or identity_salt == "0" * 32
    ):
        raise ValueError("identity salt must be nonzero 128-bit lowercase hex")
    return run_name, identity_salt


def _runtime_binding(
    frozen: Mapping[str, Any], package_sha256: Any, image_digest: Any
) -> dict[str, Any]:
    package = _require_sha(package_sha256, "package SHA-256")
    if not isinstance(image_digest, str) or _IMAGE_DIGEST.fullmatch(image_digest) is None:
        raise ValueError("image digest must be sha256:<64 lowercase hex>")
    contract = frozen["run_contract"]
    result = {
        "package_sha256": package,
        "image_digest": image_digest,
        "binary_sha256_by_role": {
            "candidate": _require_sha(
                contract["candidate_library_sha256"], "candidate binary SHA-256"
            ),
            "reference": _require_sha(
                contract["reference_library_sha256"], "reference binary SHA-256"
            ),
        },
        "allocation_digest": canonical_sha256(contract["allocation"]),
    }
    _exact_keys(result, _RUNTIME_KEYS, "full100 runtime binding")
    return result


def _identity_sha256(
    run_name: str, identity_salt: str, runtime_binding: Mapping[str, Any],
    science: science_registry.ScienceDescriptor,
    execution_scope: str = FULL100_EXECUTION_SCOPE,
) -> str:
    if science.legacy_development_identity and execution_scope == FULL100_EXECUTION_SCOPE:
        # Keep the established full100 identity preimage bit-exact.  The
        # diagnostic scope below deliberately uses a different schema and an
        # explicit scope field so its artifacts cannot replay into full100.
        identity_material = {
            "schema": "hu_m31_t3_step6d_full100_wave_identity_salt_v1",
            "run_name": run_name,
            "identity_salt": identity_salt,
            "runtime_binding": runtime_binding,
            "full100_plan_sha256": science.plan_sha256,
            "run_contract_digest": science.run_contract_digest,
        }
    elif (
        science.legacy_development_identity
        and execution_scope == STARTUP_CANARY_SCOPE
    ):
        identity_material = {
            "schema": "hu_m31_t3_step6d_full100_wave_scoped_identity_salt_v2",
            "execution_scope": execution_scope,
            "run_name": run_name,
            "identity_salt": identity_salt,
            "runtime_binding": runtime_binding,
            "full100_plan_sha256": science.plan_sha256,
            "run_contract_digest": science.run_contract_digest,
        }
    elif science.allows_execution_scope(execution_scope):
        identity_material = {
            "schema": "hu_m31_t3_step6d_full100_wave_science_identity_salt_v3",
            "science_kind": science.science_kind,
            "scientific_plan_schema": science.plan_schema,
            "execution_scope": execution_scope,
            "run_name": run_name,
            "identity_salt": identity_salt,
            "runtime_binding": runtime_binding,
            "full100_plan_sha256": science.plan_sha256,
            "run_contract_digest": science.run_contract_digest,
        }
    else:
        raise ValueError("unsupported full100 execution scope")
    return canonical_sha256(identity_material)


def _instance_prefix(run_name: str, identity_sha256: str) -> str:
    run_tag = hashlib.sha256(run_name.encode("ascii")).hexdigest()[:10]
    return f"f100-{run_tag}-{identity_sha256[:12]}"


def _instance_id(
    run_name: str, identity_sha256: str, wave_index: int, job_ordinal: int,
    attempt_id: str,
) -> str:
    if attempt_id not in ATTEMPT_IDS:
        raise ValueError("full100 attempt ID changed")
    value = (
        f"{_instance_prefix(run_name, identity_sha256)}-"
        f"w{wave_index:02d}-j{job_ordinal:02d}-{attempt_id}"
    )
    if len(value) > 63 or _SAFE_INSTANCE.fullmatch(value) is None:
        raise ValueError("full100 wave instance identity is not GCE safe")
    return value


def _build_plan_from_identity(
    *, run_name: str, identity_sha256: str, scientific_plan: Mapping[str, Any],
    runtime_binding: Mapping[str, Any],
    execution_scope: str = FULL100_EXECUTION_SCOPE,
) -> dict[str, Any]:
    if (
        not isinstance(run_name, str)
        or _SAFE_RUN.fullmatch(run_name) is None
        or not run_name.startswith(RUN_NAME_PREFIX)
    ):
        raise ValueError("full100 wave run name is not a fresh v2 identity")
    _require_sha(identity_sha256, "execution identity")
    science, frozen = science_registry.validate_scientific_plan(scientific_plan)
    if (
        execution_scope not in _EXECUTION_SCOPES
        or not science.allows_execution_scope(execution_scope)
    ):
        raise ValueError("scientific plan and execution scope do not match")
    binding = deepcopy(dict(runtime_binding))
    _exact_keys(binding, _RUNTIME_KEYS, "full100 runtime binding")
    _require_sha(binding["package_sha256"], "package SHA-256")
    if _IMAGE_DIGEST.fullmatch(str(binding["image_digest"])) is None:
        raise ValueError("image digest changed")
    if set(binding["binary_sha256_by_role"]) != set(SOURCE_ROLES):
        raise ValueError("binary role binding changed")
    for role in SOURCE_ROLES:
        _require_sha(binding["binary_sha256_by_role"][role], f"{role} binary")
    _require_sha(binding["allocation_digest"], "allocation digest")
    if binding != _runtime_binding(
        frozen, binding["package_sha256"], binding["image_digest"]
    ):
        raise ValueError("runtime binding does not match frozen science")

    jobs = {str(row["job_id"]): dict(row) for row in frozen["jobs"]}
    if len(jobs) != 20:
        raise ValueError("frozen full100 job IDs are duplicated or missing")
    ordinals = {str(row["job_id"]): i for i, row in enumerate(frozen["jobs"])}
    waves: list[dict[str, Any]] = []
    all_jobs: list[str] = []
    all_instances: list[str] = []
    for wave_index, shard_indices in enumerate(WAVE_SHARD_GROUPS):
        pairs: list[dict[str, Any]] = []
        wave_jobs: list[str] = []
        for shard_index in shard_indices:
            candidate_id = f"candidate-shard-{shard_index:02d}"
            reference_id = f"reference-shard-{shard_index:02d}"
            candidate = jobs.get(candidate_id)
            reference = jobs.get(reference_id)
            if candidate is None or reference is None:
                raise ValueError("frozen full100 candidate/reference pair is missing")
            work = list(candidate["work_hand_indices"])
            if (
                candidate["source_role"] != "candidate"
                or reference["source_role"] != "reference"
                or candidate["shard_index"] != shard_index
                or reference["shard_index"] != shard_index
                or list(reference["work_hand_indices"]) != work
            ):
                raise ValueError("frozen full100 candidate/reference pairing changed")
            c_instances = {
                attempt: _instance_id(
                    run_name, identity_sha256, wave_index, ordinals[candidate_id], attempt
                ) for attempt in ATTEMPT_IDS
            }
            r_instances = {
                attempt: _instance_id(
                    run_name, identity_sha256, wave_index, ordinals[reference_id], attempt
                ) for attempt in ATTEMPT_IDS
            }
            pairs.append(
                {
                    "pair_id": f"paired-shard-{shard_index:02d}",
                    "shard_index": shard_index,
                    "work_hand_indices": work,
                    "candidate_job_id": candidate_id,
                    "reference_job_id": reference_id,
                    "candidate_attempt_instance_ids": c_instances,
                    "reference_attempt_instance_ids": r_instances,
                    "same_wave_required": True,
                    "distinct_vm_required": True,
                    "distinct_process_required": True,
                }
            )
            wave_jobs.extend((candidate_id, reference_id))
            all_instances.extend((*c_instances.values(), *r_instances.values()))
        vm_count = len(wave_jobs)
        waves.append(
            {
                "wave_index": wave_index,
                "wave_id": f"wave-{wave_index + 1:02d}",
                "prerequisite_wave_ids": [row["wave_id"] for row in waves],
                "shard_indices": list(shard_indices),
                "job_ids": wave_jobs,
                "candidate_reference_pairs": pairs,
                "vm_count": vm_count,
                "vcpu_count": vm_count * VCPUS_PER_VM,
                "max_concurrent_vms": MAX_CONCURRENT_VMS,
                "launch_requires_prior_wave_complete": wave_index > 0,
                "launch_requires_all_owned_vms_quiescent": True,
                "create_only": True,
            }
        )
        all_jobs.extend(wave_jobs)
    if len(all_instances) != len(set(all_instances)):
        raise ValueError("full100 wave attempt identities are not unique")

    identity_tag = identity_sha256[:12]
    artifact_prefix = f"runs/{run_name}-{identity_tag}/full100-wave-v2"
    if execution_scope == STARTUP_CANARY_SCOPE:
        artifact_prefix += "/startup-canary"
    artifact_contract: dict[str, Any] = {
        "prefix": artifact_prefix,
        "attempt_path_template": (
            f"{artifact_prefix}/results/jobs/{{job_id}}/attempts/{{attempt_id}}"
        ),
        "job_acceptance_path_template": (
            f"{artifact_prefix}/results/jobs/{{job_id}}/ACCEPTED.json"
        ),
        "attempt_outputs_create_only": True,
        "job_acceptance_create_only": True,
        "one_accepted_attempt_per_job": True,
        "done_written_last": True,
        "content_sha256_required": True,
        "object_generation_precondition": 0,
        "receive_destination_immutable": True,
        "complete_observed_inventory_required_before_merge": True,
    }
    resume_contract: dict[str, Any] = {
        "max_attempts_per_job": MAX_ATTEMPTS_PER_JOB,
        "attempt_ids": list(ATTEMPT_IDS),
        "untouched_job_attempt": "a00",
        "one_failed_attempt_next_attempt": "a01",
        "two_failed_attempts_fail_closed": True,
        "same_observed_transition_reuse_forbidden": True,
        "resume_only_after_owned_vm_readback_quiescent": True,
        "earliest_incomplete_wave_only": True,
        "accepted_jobs_never_relaunched": True,
        "later_wave_before_prior_complete_forbidden": True,
        "third_attempt_authorized": False,
    }
    if execution_scope == STARTUP_CANARY_SCOPE:
        artifact_contract.update(
            {
                "startup_canary_only": True,
                "performance_evaluation_eligible": False,
                "scientific_merge_eligible": False,
                "training_eligible": False,
            }
        )
        resume_contract.update(
            {
                "execution_scope": STARTUP_CANARY_SCOPE,
                "startup_canary_job_id": STARTUP_CANARY_JOB_ID,
                "startup_canary_source_role": STARTUP_CANARY_SOURCE_ROLE,
                "startup_canary_attempt_id": STARTUP_CANARY_ATTEMPT_ID,
                "selected_attempt_count": 1,
                "a01_authorized": False,
                "other_jobs_authorized": False,
                "retry_authorized": False,
                "performance_evaluation_eligible": False,
                "scientific_merge_eligible": False,
                "training_eligible": False,
            }
        )
    value: dict[str, Any] = {
        "schema": WAVE_PLAN_SCHEMA,
        "status": (
            science.wave_status
            if execution_scope != STARTUP_CANARY_SCOPE
            else "local_single_vm_startup_canary_contract_ready_cloud_not_authorized"
        ),
        "decision": (
            science.wave_decision
            if execution_scope != STARTUP_CANARY_SCOPE
            else "validate_startup_transport_once_without_scientific_promotion"
        ),
        "scope": execution_scope,
        "run_name": run_name,
        "execution_identity_sha256": identity_sha256,
        "identity_tag": identity_tag,
        "full100_plan": deepcopy(frozen),
        "full100_plan_sha256": science.plan_sha256,
        "run_contract_digest": science.run_contract_digest,
        "root_set": deepcopy(frozen["root_set"]),
        "seed_contract": deepcopy(frozen["run_contract"]["seed_contract"]),
        "runtime_binding": binding,
        "quota_contract": {
            "region": "asia-northeast1", "quota_metric": "C4_CPUS",
            "planned_quota_vcpus": TOKYO_C4_FAMILY_QUOTA_VCPUS,
            "machine_type": MACHINE_TYPE, "vcpus_per_vm": VCPUS_PER_VM,
            "planned_max_concurrent_vms": MAX_CONCURRENT_VMS,
            "planned_max_concurrent_vcpus": MAX_CONCURRENT_VMS * VCPUS_PER_VM,
            "wave_vm_counts": list(WAVE_VM_COUNTS),
            "live_quota_readback_required": True,
            "live_headroom_readback_required": True,
        },
        "pair_co_location_policy": {
            "mode": "same_wave_distinct_vm", "same_wave_required": True,
            "same_zone_required": False, "same_vm_forbidden": True,
            "same_process_forbidden": True, "role_output_prefixes_isolated": True,
        },
        "artifact_contract": artifact_contract,
        "resume_contract": resume_contract,
        "wave_count": len(waves),
        "wave_vm_counts": list(WAVE_VM_COUNTS),
        "waves": waves,
        "coverage": {
            "job_ids": all_jobs, "logical_job_count": 20,
            "paired_hand_indices": list(range(100)), "paired_hand_count": 100,
            "candidate_hand_count": 100, "reference_hand_count": 100,
            "source_root_count": 200, "candidate_reference_pairs": 10,
            "duplicate_jobs_allowed": False, "missing_jobs_allowed": False,
        },
        "authorization_blockers": list(AUTHORIZATION_BLOCKERS),
        "cloud_launcher_ready": False,
        "cloud_launch_authorized": False,
        "cloud_started": False,
        "performance_lock_authorized": False,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
    }
    value["schedule_sha256"] = canonical_sha256(value)
    return value


def build_wave_plan(
    *, run_name: str, identity_salt: str, package_sha256: str,
    image_digest: str, full100_plan: Mapping[str, Any] | None = None,
    full100_plan_path: str | Path = DEFAULT_FULL100_PLAN_PATH,
    execution_scope: str | None = None,
) -> dict[str, Any]:
    """Build a local-only schedule; ``identity_salt`` is never emitted."""

    run_name, identity_salt = _validate_run_and_salt(run_name, identity_salt)
    frozen = (
        _read_frozen_plan(full100_plan_path)
        if full100_plan is None
        else science_registry.validate_scientific_plan(full100_plan)[1]
    )
    science = science_registry.descriptor_for_plan(frozen)
    selected_scope = (
        science.execution_scope if execution_scope is None else execution_scope
    )
    binding = _runtime_binding(frozen, package_sha256, image_digest)
    if (
        selected_scope not in _EXECUTION_SCOPES
        or not science.allows_execution_scope(selected_scope)
    ):
        raise ValueError("scientific plan and execution scope do not match")
    identity = _identity_sha256(
        run_name,
        identity_salt,
        binding,
        science,
        execution_scope=selected_scope,
    )
    return validate_wave_plan(
        _build_plan_from_identity(
            run_name=run_name, identity_sha256=identity,
            scientific_plan=frozen, runtime_binding=binding,
            execution_scope=selected_scope,
        )
    )


_VALIDATED_WAVE_PLAN_CACHE: dict[str, dict[str, Any]] = {}
_VALIDATED_WAVE_PLAN_CACHE_MAX = 8


def validate_wave_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a wave plan, memoized on the exact canonical plan content.

    Validation is a pure function of the plan bytes, but one pre-claim read
    path validates the same frozen plan seven times and each pass re-derives
    the entire seed/shard contract (tens of millions of iterations), which
    pushed the fresh-read chain past its freshness window. Memoizing on the
    canonical content hash removes only that repetition: a different plan, or
    any tampered byte, hashes differently and is validated in full, so no
    check is weakened and no result is shared across distinct inputs.
    """

    if not isinstance(value, Mapping):
        raise ValueError("full100 wave plan must be an object")
    try:
        key = canonical_sha256(value)
    except (TypeError, ValueError):
        return _validate_wave_plan_uncached(value)
    cached = _VALIDATED_WAVE_PLAN_CACHE.get(key)
    if cached is not None:
        return deepcopy(cached)
    payload = _validate_wave_plan_uncached(value)
    if len(_VALIDATED_WAVE_PLAN_CACHE) >= _VALIDATED_WAVE_PLAN_CACHE_MAX:
        _VALIDATED_WAVE_PLAN_CACHE.clear()
    _VALIDATED_WAVE_PLAN_CACHE[key] = deepcopy(payload)
    return payload


def _validate_wave_plan_uncached(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("full100 wave plan must be an object")
    payload = deepcopy(dict(value))
    _exact_keys(payload, _PLAN_KEYS, "full100 wave plan")
    if any("salt" in key or "nonce" in key for key in payload):
        raise ValueError("raw identity material must not appear in the wave plan")
    scientific = payload.get("full100_plan")
    if not isinstance(scientific, Mapping):
        raise ValueError("full100 wave scientific plan is missing")
    expected = _build_plan_from_identity(
        run_name=payload.get("run_name"),
        identity_sha256=payload.get("execution_identity_sha256"),
        scientific_plan=scientific,
        runtime_binding=payload.get("runtime_binding", {}),
        execution_scope=payload.get("scope"),
    )
    if payload != expected:
        raise ValueError("full100 wave plan contract changed")
    if payload["identity_tag"] != payload["execution_identity_sha256"][:12]:
        raise ValueError("full100 identity tag changed")
    jobs: list[str] = []
    hands = {role: [] for role in SOURCE_ROLES}
    instances: list[str] = []
    for wave_index, wave in enumerate(payload["waves"]):
        _exact_keys(wave, _WAVE_KEYS, "full100 wave entry")
        if wave["wave_index"] != wave_index or wave["vm_count"] > MAX_CONCURRENT_VMS:
            raise ValueError("full100 wave concurrency changed")
        for pair in wave["candidate_reference_pairs"]:
            _exact_keys(pair, _PAIR_KEYS, "full100 wave pair")
            if pair["candidate_job_id"] == pair["reference_job_id"]:
                raise ValueError("full100 source pair is not isolated")
            for role in SOURCE_ROLES:
                mapping = pair[f"{role}_attempt_instance_ids"]
                if set(mapping) != set(ATTEMPT_IDS):
                    raise ValueError("full100 attempt lane changed")
                instances.extend(mapping.values())
                hands[role].extend(pair["work_hand_indices"])
        jobs.extend(wave["job_ids"])
    if (
        len(jobs) != len(set(jobs)) != 20
        or len(jobs) != 20
        or len(instances) != len(set(instances))
        or sorted(hands["candidate"]) != list(range(100))
        or sorted(hands["reference"]) != list(range(100))
        or payload["wave_vm_counts"] != [8, 8, 4]
        or payload["authorization_blockers"] != list(AUTHORIZATION_BLOCKERS)
        or payload["cloud_launcher_ready"] is not False
        or payload["cloud_launch_authorized"] is not False
    ):
        raise ValueError("full100 wave coverage or authorization changed")
    return payload


def validate_startup_canary_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the diagnostic plan and its exact single-attempt boundary."""

    plan = validate_wave_plan(value)
    contract = plan["resume_contract"]
    artifact = plan["artifact_contract"]
    if (
        plan["scope"] != STARTUP_CANARY_SCOPE
        or contract.get("execution_scope") != STARTUP_CANARY_SCOPE
        or contract.get("startup_canary_job_id") != STARTUP_CANARY_JOB_ID
        or contract.get("startup_canary_source_role") != STARTUP_CANARY_SOURCE_ROLE
        or contract.get("startup_canary_attempt_id") != STARTUP_CANARY_ATTEMPT_ID
        or contract.get("selected_attempt_count") != 1
        or contract.get("a01_authorized") is not False
        or contract.get("other_jobs_authorized") is not False
        or contract.get("retry_authorized") is not False
        or contract.get("performance_evaluation_eligible") is not False
        or contract.get("scientific_merge_eligible") is not False
        or contract.get("training_eligible") is not False
        or artifact.get("startup_canary_only") is not True
        or artifact.get("performance_evaluation_eligible") is not False
        or artifact.get("scientific_merge_eligible") is not False
        or artifact.get("training_eligible") is not False
        or not artifact["prefix"].endswith("/full100-wave-v2/startup-canary")
        or plan["wave_vm_counts"] != [8, 8, 4]
        or plan["coverage"]["logical_job_count"] != 20
        or plan["coverage"]["paired_hand_count"] != 100
    ):
        raise ValueError("startup canary scope contract changed")
    return plan


def _job_metadata(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    scientific = {row["job_id"]: row for row in plan["full100_plan"]["jobs"]}
    for wave in plan["waves"]:
        for pair in wave["candidate_reference_pairs"]:
            for role in SOURCE_ROLES:
                job_id = pair[f"{role}_job_id"]
                result[job_id] = {
                    "job_id": job_id, "source_role": role,
                    "wave_index": wave["wave_index"],
                    "shard_index": pair["shard_index"],
                    "work_hand_indices": list(scientific[job_id]["work_hand_indices"]),
                    "instance_ids": deepcopy(pair[f"{role}_attempt_instance_ids"]),
                }
    return result


def _attempt_prefix(plan: Mapping[str, Any], job_id: str, attempt_id: str) -> str:
    return plan["artifact_contract"]["attempt_path_template"].format(
        job_id=job_id, attempt_id=attempt_id
    )


def _acceptance_path(plan: Mapping[str, Any], job_id: str) -> str:
    return plan["artifact_contract"]["job_acceptance_path_template"].format(
        job_id=job_id
    )


def empty_attempt_history(wave_plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    plan = validate_wave_plan(wave_plan)
    metadata = _job_metadata(plan)
    return [
        {"job_id": job, "source_role": metadata[job]["source_role"], "attempts": []}
        for job in plan["coverage"]["job_ids"]
    ]


def expected_done_identity_sha256(
    wave_plan: Mapping[str, Any], *, job_id: str, attempt_id: str,
    root_digest: str,
) -> str:
    plan = validate_wave_plan(wave_plan)
    metadata = _job_metadata(plan)
    if job_id not in metadata or attempt_id not in ATTEMPT_IDS:
        raise ValueError("unknown full100 job or attempt")
    _require_sha(root_digest, "root digest")
    return _expected_done_identity_unchecked(
        plan, metadata, job_id=job_id, attempt_id=attempt_id,
        root_digest=root_digest,
    )


def _expected_done_identity_unchecked(
    plan: Mapping[str, Any], metadata: Mapping[str, Mapping[str, Any]], *,
    job_id: str, attempt_id: str, root_digest: str,
) -> str:
    role = metadata[job_id]["source_role"]
    binding = plan["runtime_binding"]
    return canonical_sha256(
        {
            "schema": "hu_m31_t3_step6d_full100_done_identity_v1",
            "run_name": plan["run_name"],
            "execution_identity_sha256": plan["execution_identity_sha256"],
            "job_id": job_id, "source_role": role, "attempt_id": attempt_id,
            "package_sha256": binding["package_sha256"],
            "image_digest": binding["image_digest"],
            "binary_sha256": binding["binary_sha256_by_role"][role],
            "allocation_digest": binding["allocation_digest"],
            "root_digest": root_digest,
        }
    )


def _validate_history(
    plan: Mapping[str, Any], history: Any
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    if not isinstance(history, list):
        raise ValueError("attempt history is missing")
    metadata = _job_metadata(plan)
    allowed = plan["coverage"]["job_ids"]
    if len(history) != len(allowed):
        raise ValueError("attempt history must cover every logical job exactly once")
    result: list[dict[str, Any]] = []
    by_job: dict[str, list[dict[str, Any]]] = {}
    instance_ids: set[str] = set()
    receipts: set[str] = set()
    for expected_job, raw in zip(allowed, history, strict=True):
        if not isinstance(raw, Mapping):
            raise ValueError("attempt history entry is not an object")
        row = deepcopy(dict(raw))
        _exact_keys(row, _HISTORY_KEYS, "attempt history entry")
        meta = metadata[expected_job]
        if row["job_id"] != expected_job or row["source_role"] != meta["source_role"]:
            raise ValueError("attempt history order or role changed")
        attempts = row["attempts"]
        if not isinstance(attempts, list) or len(attempts) > MAX_ATTEMPTS_PER_JOB:
            raise ValueError("attempt history exceeded two attempts")
        clean_attempts: list[dict[str, Any]] = []
        for index, raw_attempt in enumerate(attempts):
            if not isinstance(raw_attempt, Mapping):
                raise ValueError("attempt history lane is not an object")
            attempt = deepcopy(dict(raw_attempt))
            _exact_keys(attempt, _ATTEMPT_KEYS, "attempt history lane")
            attempt_id = ATTEMPT_IDS[index]
            if (
                attempt["attempt_id"] != attempt_id
                or attempt["instance_id"] != meta["instance_ids"][attempt_id]
                or attempt["terminal_status"] not in {"failed", "accepted"}
            ):
                raise ValueError("attempt history sequence or identity changed")
            _require_sha(attempt["launch_receipt_sha256"], "launch receipt")
            if attempt["instance_id"] in instance_ids or attempt["launch_receipt_sha256"] in receipts:
                raise ValueError("attempt history reused an instance or launch receipt")
            instance_ids.add(attempt["instance_id"])
            receipts.add(attempt["launch_receipt_sha256"])
            if index == 1 and attempts[0]["terminal_status"] != "failed":
                raise ValueError("a01 requires a failed a00")
            if attempt["terminal_status"] == "accepted" and index != len(attempts) - 1:
                raise ValueError("an accepted job cannot be attempted again")
            clean_attempts.append(attempt)
        row["attempts"] = clean_attempts
        result.append(row)
        by_job[expected_job] = clean_attempts
    return result, by_job


def _validate_done(
    plan: Mapping[str, Any], value: Any, history: Mapping[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("DONE readback is missing")
    metadata = _job_metadata(plan)
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError("DONE readback entry is not an object")
        row = deepcopy(dict(raw))
        _exact_keys(row, _DONE_KEYS, "DONE readback entry")
        job = row["job_id"]
        if job not in metadata or job in seen:
            raise ValueError("both attempts have DONE or a DONE job is duplicated")
        seen.add(job)
        meta = metadata[job]
        attempts = history[job]
        if (
            row["source_role"] != meta["source_role"]
            or not attempts
            or attempts[-1]["terminal_status"] != "accepted"
            or row["attempt_id"] != attempts[-1]["attempt_id"]
            or row["path"] != f"{_attempt_prefix(plan, job, row['attempt_id'])}/DONE.json"
        ):
            raise ValueError("DONE readback is not the accepted terminal attempt")
        _require_positive_int(row["generation"], "DONE generation")
        _require_positive_int(row["bytes"], "DONE bytes")
        for key in ("sha256", "done_identity_sha256", "package_sha256",
                    "binary_sha256", "allocation_digest", "root_digest"):
            _require_sha(row[key], f"DONE {key}")
        binding = plan["runtime_binding"]
        role = meta["source_role"]
        if (
            row["package_sha256"] != binding["package_sha256"]
            or row["image_digest"] != binding["image_digest"]
            or row["binary_sha256"] != binding["binary_sha256_by_role"][role]
            or row["allocation_digest"] != binding["allocation_digest"]
            or row["done_identity_sha256"] != _expected_done_identity_unchecked(
                plan, metadata, job_id=job, attempt_id=row["attempt_id"],
                root_digest=row["root_digest"]
            )
        ):
            raise ValueError("DONE runtime or identity binding changed")
        result.append(row)
    accepted = {
        job for job, attempts in history.items()
        if attempts and attempts[-1]["terminal_status"] == "accepted"
    }
    if seen != accepted:
        raise ValueError("DONE readback does not exactly match accepted jobs")
    return result


def _validate_acceptance(
    plan: Mapping[str, Any], value: Any, done: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("acceptance readback is missing")
    metadata = _job_metadata(plan)
    done_by_job = {row["job_id"]: row for row in done}
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError("acceptance readback entry is not an object")
        row = deepcopy(dict(raw))
        _exact_keys(row, _ACCEPTANCE_KEYS, "acceptance readback entry")
        job = row["job_id"]
        if job not in metadata or job in seen:
            raise ValueError("multiple accepted attempts exist for one job")
        seen.add(job)
        done_row = done_by_job.get(job)
        if (
            done_row is None
            or row["source_role"] != metadata[job]["source_role"]
            or row["attempt_id"] != done_row["attempt_id"]
            or row["path"] != _acceptance_path(plan, job)
            or row["done_generation"] != done_row["generation"]
            or row["done_sha256"] != done_row["sha256"]
            or row["create_only"] is not True
        ):
            raise ValueError("acceptance record is not bound to exactly one DONE")
        _require_positive_int(row["generation"], "acceptance generation")
        _require_positive_int(row["bytes"], "acceptance bytes")
        _require_sha(row["sha256"], "acceptance SHA-256")
        result.append(row)
    if seen != set(done_by_job):
        raise ValueError("acceptance records do not exactly match DONE readback")
    return result


def build_observed_transition(
    wave_plan: Mapping[str, Any], *, project_id: str, zone: str,
    observed_at_utc: str, previous_transition_digest: str | None,
    attempt_history: Sequence[Mapping[str, Any]],
    owned_vms: Sequence[Mapping[str, Any]] = (),
    done_objects: Sequence[Mapping[str, Any]] = (),
    acceptance_records: Sequence[Mapping[str, Any]] = (),
    readback_source: str = "local_observed_fixture",
) -> dict[str, Any]:
    plan = validate_wave_plan(wave_plan)
    core: dict[str, Any] = {
        "schema": OBSERVED_TRANSITION_SCHEMA,
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "project_id": project_id, "zone": zone,
        "observed_at_utc": observed_at_utc, "readback_source": readback_source,
        "previous_transition_digest": previous_transition_digest,
        "owned_vm_readback_complete": True,
        "owned_vm_quiescent": len(owned_vms) == 0,
        "owned_vms": deepcopy(list(owned_vms)),
        "attempt_history": deepcopy(list(attempt_history)),
        "done_objects": deepcopy(list(done_objects)),
        "acceptance_records": deepcopy(list(acceptance_records)),
    }
    core["transition_digest"] = canonical_sha256(core)
    return validate_observed_transition(plan, core)


def validate_observed_transition(
    wave_plan: Mapping[str, Any], value: Mapping[str, Any]
) -> dict[str, Any]:
    plan = validate_wave_plan(wave_plan)
    if not isinstance(value, Mapping):
        raise ValueError("observed transition must be an object")
    payload = deepcopy(dict(value))
    _exact_keys(payload, _TRANSITION_KEYS, "observed transition")
    if (
        payload["schema"] != OBSERVED_TRANSITION_SCHEMA
        or payload["run_name"] != plan["run_name"]
        or payload["execution_identity_sha256"] != plan["execution_identity_sha256"]
        or not isinstance(payload["project_id"], str)
        or _SAFE_PROJECT.fullmatch(payload["project_id"]) is None
        or not isinstance(payload["zone"], str)
        or _SAFE_ZONE.fullmatch(payload["zone"]) is None
        or payload["readback_source"] not in {"local_observed_fixture", "gcloud_readback"}
        or payload["owned_vm_readback_complete"] is not True
    ):
        raise ValueError("observed transition identity or readback fields changed")
    _parse_utc_seconds(payload["observed_at_utc"], "observed time")
    previous = payload["previous_transition_digest"]
    if previous is not None:
        _require_sha(previous, "previous transition digest")
    history, by_job = _validate_history(plan, payload["attempt_history"])
    payload["attempt_history"] = history
    done = _validate_done(plan, payload["done_objects"], by_job)
    acceptance = _validate_acceptance(plan, payload["acceptance_records"], done)
    payload["done_objects"] = done
    payload["acceptance_records"] = acceptance

    owned = payload["owned_vms"]
    if not isinstance(owned, list):
        raise ValueError("owned VM readback is missing")
    metadata = _job_metadata(plan)
    seen_instances: set[str] = set()
    for raw in owned:
        if not isinstance(raw, Mapping):
            raise ValueError("owned VM readback entry is not an object")
        _exact_keys(raw, _OWNED_VM_KEYS, "owned VM readback entry")
        job = raw["job_id"]
        attempt = raw["attempt_id"]
        if (
            job not in metadata or attempt not in ATTEMPT_IDS
            or raw["source_role"] != metadata[job]["source_role"]
            or raw["instance_id"] != metadata[job]["instance_ids"][attempt]
            or raw["instance_id"] in seen_instances
            or raw["status"] not in {
                "PROVISIONING", "STAGING", "RUNNING", "STOPPING", "TERMINATED"
            }
            or any(row["attempt_id"] == attempt for row in by_job[job])
        ):
            raise ValueError("owned VM readback is duplicated or inconsistent")
        seen_instances.add(raw["instance_id"])
    if payload["owned_vm_quiescent"] is not (len(owned) == 0):
        raise ValueError("owned VM quiescence does not match exact readback")
    digest_payload = {k: v for k, v in payload.items() if k != "transition_digest"}
    if payload["transition_digest"] != canonical_sha256(digest_payload):
        raise ValueError("observed transition digest changed")
    return payload


def build_attempt_ledger(
    wave_plan: Mapping[str, Any], *,
    transitions: Sequence[Mapping[str, Any]],
    consumed_transition_digests: Sequence[str] = (),
) -> dict[str, Any]:
    plan = validate_wave_plan(wave_plan)
    validated = [validate_observed_transition(plan, row) for row in transitions]
    core: dict[str, Any] = {
        "schema": ATTEMPT_LEDGER_SCHEMA,
        "wave_plan_sha256": plan["schedule_sha256"],
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "transitions": validated,
        "latest_transition_digest": (
            validated[-1]["transition_digest"] if validated else None
        ),
        "consumed_transition_digests": list(consumed_transition_digests),
        "cloud_claim_persistent": False,
    }
    core["ledger_sha256"] = canonical_sha256(core)
    return validate_attempt_ledger(plan, core)


def _history_map(transition: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {row["job_id"]: list(row["attempts"]) for row in transition["attempt_history"]}


def _accepted_jobs(transition: Mapping[str, Any]) -> set[str]:
    return {row["job_id"] for row in transition["acceptance_records"]}


def _earliest_incomplete_wave(
    plan: Mapping[str, Any], accepted: set[str]
) -> int | None:
    for wave in plan["waves"]:
        if not set(wave["job_ids"]).issubset(accepted):
            return int(wave["wave_index"])
    return None


def validate_attempt_ledger(
    wave_plan: Mapping[str, Any], value: Mapping[str, Any]
) -> dict[str, Any]:
    plan = validate_wave_plan(wave_plan)
    if not isinstance(value, Mapping):
        raise ValueError("attempt ledger must be an object")
    payload = deepcopy(dict(value))
    _exact_keys(payload, _LEDGER_KEYS, "attempt ledger")
    if (
        payload["schema"] != ATTEMPT_LEDGER_SCHEMA
        or payload["wave_plan_sha256"] != plan["schedule_sha256"]
        or payload["run_name"] != plan["run_name"]
        or payload["execution_identity_sha256"] != plan["execution_identity_sha256"]
        or payload["cloud_claim_persistent"] is not False
        or not isinstance(payload["transitions"], list)
        or not payload["transitions"]
    ):
        raise ValueError("attempt ledger identity or persistence state changed")
    transitions = [validate_observed_transition(plan, row) for row in payload["transitions"]]
    consumed = payload["consumed_transition_digests"]
    if not isinstance(consumed, list) or len(consumed) != len(set(consumed)):
        raise ValueError("attempt ledger consumed transition set changed")
    for digest in consumed:
        _require_sha(digest, "consumed transition digest")
    consumed_set = set(consumed)
    first_history = _history_map(transitions[0])
    if (
        transitions[0]["previous_transition_digest"] is not None
        or any(first_history.values())
        or transitions[0]["done_objects"]
        or transitions[0]["acceptance_records"]
        or transitions[0]["owned_vms"]
    ):
        raise ValueError("attempt ledger must start from an empty observed baseline")
    previous = transitions[0]
    for current in transitions[1:]:
        if (
            current["previous_transition_digest"] != previous["transition_digest"]
            or previous["transition_digest"] not in consumed_set
            or current["project_id"] != previous["project_id"]
            or current["zone"] != previous["zone"]
            or _parse_utc_seconds(current["observed_at_utc"], "observed time")
            <= _parse_utc_seconds(previous["observed_at_utc"], "previous observed time")
        ):
            raise ValueError("attempt ledger transition chain or observation scope changed")
        before = _history_map(previous)
        after = _history_map(current)
        accepted_before = _accepted_jobs(previous)
        wave_index = _earliest_incomplete_wave(plan, accepted_before)
        changed_jobs: list[str] = []
        for job in plan["coverage"]["job_ids"]:
            if after[job][: len(before[job])] != before[job] or len(after[job]) > len(before[job]) + 1:
                raise ValueError("attempt history was rewritten or skipped an attempt")
            if len(after[job]) == len(before[job]) + 1:
                changed_jobs.append(job)
        owned_changed = current["owned_vms"] != previous["owned_vms"]
        if not changed_jobs and not owned_changed:
            raise ValueError("attempt ledger transition did not advance observed evidence")
        affected_jobs = [*changed_jobs, *(row["job_id"] for row in current["owned_vms"])]
        if wave_index is None or any(
            job not in plan["waves"][wave_index]["job_ids"] for job in affected_jobs
        ):
            raise ValueError("attempt evidence advanced outside the earliest incomplete wave")
        if not accepted_before.issubset(_accepted_jobs(current)):
            raise ValueError("accepted job evidence disappeared")
        previous = current
    latest = transitions[-1]["transition_digest"]
    all_digests = {row["transition_digest"] for row in transitions}
    if (
        payload["latest_transition_digest"] != latest
        or any(value not in all_digests for value in consumed)
    ):
        raise ValueError("attempt ledger latest or consumed transition set changed")
    payload["transitions"] = transitions
    digest_payload = {k: v for k, v in payload.items() if k != "ledger_sha256"}
    if payload["ledger_sha256"] != canonical_sha256(digest_payload):
        raise ValueError("attempt ledger digest changed")
    return payload


def mark_latest_transition_consumed(
    wave_plan: Mapping[str, Any], attempt_ledger: Mapping[str, Any]
) -> dict[str, Any]:
    ledger = validate_attempt_ledger(wave_plan, attempt_ledger)
    latest = ledger["latest_transition_digest"]
    if latest in ledger["consumed_transition_digests"]:
        raise ValueError("same observed transition cannot be consumed twice")
    return build_attempt_ledger(
        wave_plan, transitions=ledger["transitions"],
        consumed_transition_digests=[*ledger["consumed_transition_digests"], latest],
    )


def build_resume_plan(
    wave_plan: Mapping[str, Any], *, attempt_ledger: Mapping[str, Any]
) -> dict[str, Any]:
    plan = validate_wave_plan(wave_plan)
    ledger = validate_attempt_ledger(plan, attempt_ledger)
    if plan["scope"] == STARTUP_CANARY_SCOPE:
        return validate_resume_plan(plan, ledger, _build_resume_unchecked(plan, ledger))
    latest = ledger["transitions"][-1]
    if not latest["owned_vm_quiescent"]:
        raise ValueError("resume requires complete owned-VM quiescence")
    if ledger["latest_transition_digest"] in ledger["consumed_transition_digests"]:
        raise ValueError("same observed transition was already consumed")
    accepted = _accepted_jobs(latest)
    wave_index = _earliest_incomplete_wave(plan, accepted)
    histories = _history_map(latest)
    selected: list[dict[str, Any]] = []
    if wave_index is not None:
        metadata = _job_metadata(plan)
        for job in plan["waves"][wave_index]["job_ids"]:
            if job in accepted:
                continue
            attempts = histories[job]
            if len(attempts) >= MAX_ATTEMPTS_PER_JOB:
                raise ValueError("unfinished job exhausted its two-attempt budget")
            if attempts and attempts[-1]["terminal_status"] != "failed":
                raise ValueError("unfinished job has a non-failed terminal attempt")
            attempt_id = ATTEMPT_IDS[len(attempts)]
            selected.append(
                {
                    "job_id": job,
                    "source_role": metadata[job]["source_role"],
                    "attempt_id": attempt_id,
                    "instance_id": metadata[job]["instance_ids"][attempt_id],
                    "artifact_prefix": _attempt_prefix(plan, job, attempt_id),
                }
            )
    if len(selected) > MAX_CONCURRENT_VMS:
        raise ValueError("resume exceeds the eight-VM wave boundary")
    core: dict[str, Any] = {
        "schema": RESUME_PLAN_SCHEMA,
        "status": (
            "all_jobs_complete_local_readback_only"
            if wave_index is None
            else "local_attempt_selection_ready_cloud_not_authorized"
        ),
        "wave_plan_sha256": plan["schedule_sha256"],
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "observed_transition_digest": ledger["latest_transition_digest"],
        "resume_wave_index": wave_index,
        "selected_attempts": selected,
        "all_jobs_complete": wave_index is None,
        "owned_vm_quiescence_proven": True,
        "transition_unconsumed": True,
        "authorization_blockers": list(AUTHORIZATION_BLOCKERS),
        "cloud_launcher_ready": False,
        "cloud_launch_authorized": False,
        "third_attempt_authorized": False,
    }
    core["resume_sha256"] = canonical_sha256(core)
    return validate_resume_plan(plan, ledger, core)


def validate_resume_plan(
    wave_plan: Mapping[str, Any], attempt_ledger: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("resume plan must be an object")
    payload = deepcopy(dict(value))
    _exact_keys(payload, _RESUME_KEYS, "resume plan")
    # Rebuild once with a private helper guard to avoid accepting caller state.
    plan = validate_wave_plan(wave_plan)
    ledger = validate_attempt_ledger(plan, attempt_ledger)
    digest = payload.pop("resume_sha256", None)
    if digest != canonical_sha256(payload):
        raise ValueError("resume plan digest changed")
    payload["resume_sha256"] = digest
    if (
        payload["schema"] != RESUME_PLAN_SCHEMA
        or payload["wave_plan_sha256"] != plan["schedule_sha256"]
        or payload["attempt_ledger_sha256"] != ledger["ledger_sha256"]
        or payload["observed_transition_digest"] != ledger["latest_transition_digest"]
        or payload["authorization_blockers"] != list(AUTHORIZATION_BLOCKERS)
        or payload["cloud_launcher_ready"] is not False
        or payload["cloud_launch_authorized"] is not False
        or payload["third_attempt_authorized"] is not False
    ):
        raise ValueError("resume authorization or evidence binding changed")
    for selected in payload["selected_attempts"]:
        _exact_keys(selected, _SELECTED_KEYS, "selected attempt")
    # Compare to a fresh derived plan without recursive validation.
    expected = _build_resume_unchecked(plan, ledger)
    if payload != expected:
        raise ValueError("resume plan selection changed")
    return payload


def _build_resume_unchecked(
    plan: Mapping[str, Any], ledger: Mapping[str, Any]
) -> dict[str, Any]:
    latest = ledger["transitions"][-1]
    if not latest["owned_vm_quiescent"]:
        raise ValueError("resume requires complete owned-VM quiescence")
    if ledger["latest_transition_digest"] in ledger["consumed_transition_digests"]:
        raise ValueError("same observed transition was already consumed")
    accepted = _accepted_jobs(latest)
    metadata = _job_metadata(plan)
    histories = _history_map(latest)
    selected: list[dict[str, Any]] = []
    if plan["scope"] == STARTUP_CANARY_SCOPE:
        validate_startup_canary_plan(plan)
        if (
            accepted
            or latest["done_objects"]
            or latest["acceptance_records"]
            or any(histories.values())
        ):
            raise PermissionError(
                "startup canary is single-attempt only; retry and resume are forbidden"
            )
        wave_index: int | None = 0
        job = STARTUP_CANARY_JOB_ID
        attempt_id = STARTUP_CANARY_ATTEMPT_ID
        selected.append(
            {
                "job_id": job,
                "source_role": STARTUP_CANARY_SOURCE_ROLE,
                "attempt_id": attempt_id,
                "instance_id": metadata[job]["instance_ids"][attempt_id],
                "artifact_prefix": _attempt_prefix(plan, job, attempt_id),
            }
        )
    else:
        wave_index = _earliest_incomplete_wave(plan, accepted)
    if wave_index is not None and plan["scope"] != STARTUP_CANARY_SCOPE:
        for job in plan["waves"][wave_index]["job_ids"]:
            if job in accepted:
                continue
            attempts = histories[job]
            if len(attempts) >= MAX_ATTEMPTS_PER_JOB:
                raise ValueError("unfinished job exhausted its two-attempt budget")
            if attempts and attempts[-1]["terminal_status"] != "failed":
                raise ValueError("unfinished job has a non-failed terminal attempt")
            attempt_id = ATTEMPT_IDS[len(attempts)]
            selected.append(
                {
                    "job_id": job, "source_role": metadata[job]["source_role"],
                    "attempt_id": attempt_id,
                    "instance_id": metadata[job]["instance_ids"][attempt_id],
                    "artifact_prefix": _attempt_prefix(plan, job, attempt_id),
                }
            )
    result: dict[str, Any] = {
        "schema": RESUME_PLAN_SCHEMA,
        "status": (
            "startup_canary_candidate_shard_00_a00_ready_cloud_not_authorized"
            if plan["scope"] == STARTUP_CANARY_SCOPE
            else (
                "all_jobs_complete_local_readback_only" if wave_index is None
                else "local_attempt_selection_ready_cloud_not_authorized"
            )
        ),
        "wave_plan_sha256": plan["schedule_sha256"],
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "observed_transition_digest": ledger["latest_transition_digest"],
        "resume_wave_index": wave_index,
        "selected_attempts": selected,
        "all_jobs_complete": wave_index is None,
        "owned_vm_quiescence_proven": True,
        "transition_unconsumed": True,
        "authorization_blockers": list(AUTHORIZATION_BLOCKERS),
        "cloud_launcher_ready": False,
        "cloud_launch_authorized": False,
        "third_attempt_authorized": False,
    }
    result["resume_sha256"] = canonical_sha256(result)
    return result


def expected_artifact_inventory(
    wave_plan: Mapping[str, Any], *, attempt_ledger: Mapping[str, Any]
) -> dict[str, Any]:
    plan = validate_wave_plan(wave_plan)
    if plan["scope"] == STARTUP_CANARY_SCOPE:
        raise PermissionError(
            "startup canary is ineligible for scientific merge or performance inventory"
        )
    ledger = validate_attempt_ledger(plan, attempt_ledger)
    latest = ledger["transitions"][-1]
    accepted = _accepted_jobs(latest)
    if accepted != set(plan["coverage"]["job_ids"]):
        raise ValueError("complete accepted job evidence is required for inventory")
    metadata = _job_metadata(plan)
    done = {row["job_id"]: row for row in latest["done_objects"]}
    acceptance = {row["job_id"]: row for row in latest["acceptance_records"]}
    records: list[dict[str, Any]] = []
    for job in plan["coverage"]["job_ids"]:
        meta = metadata[job]
        done_row = done[job]
        accept_row = acceptance[job]
        attempt = done_row["attempt_id"]
        prefix = _attempt_prefix(plan, job, attempt)
        work = meta["work_hand_indices"]
        records.append(
            {
                "job_id": job, "source_role": meta["source_role"],
                "shard_index": meta["shard_index"], "work_hand_indices": work,
                "accepted_attempt_id": attempt, "attempt_prefix": prefix,
                "done_path": f"{prefix}/DONE.json",
                "acceptance_path": _acceptance_path(plan, job),
                "root_paths": [f"{prefix}/roots/hand_{hand:03d}.json" for hand in work],
                "source_hand_paths": [
                    f"{prefix}/hands/{meta['source_role']}/hand_{hand:03d}.json"
                    for hand in work
                ],
                "run_contract_digest": plan["run_contract_digest"],
                "done_identity_sha256": done_row["done_identity_sha256"],
                "package_sha256": done_row["package_sha256"],
                "image_digest": done_row["image_digest"],
                "binary_sha256": done_row["binary_sha256"],
                "allocation_digest": done_row["allocation_digest"],
                "root_digest": done_row["root_digest"],
                "done_generation": done_row["generation"],
                "done_bytes": done_row["bytes"], "done_sha256": done_row["sha256"],
                "acceptance_generation": accept_row["generation"],
                "acceptance_bytes": accept_row["bytes"],
                "acceptance_sha256": accept_row["sha256"],
                "immutable_write_once": True, "done_written_last": True,
                "acceptance_create_only": True,
            }
        )
    value: dict[str, Any] = {
        "schema": EXPECTED_INVENTORY_SCHEMA,
        "run_name": plan["run_name"],
        "execution_identity_sha256": plan["execution_identity_sha256"],
        "wave_plan_sha256": plan["schedule_sha256"],
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "artifact_prefix": plan["artifact_contract"]["prefix"],
        "job_count": 20, "paired_hand_count": 100,
        "source_root_count": 200, "source_hand_count": 200,
        "object_count": 440, "records": records,
        "content_sha256_required": True, "object_generation_required": True,
        "overwrite_forbidden": True,
    }
    value["inventory_sha256"] = canonical_sha256(value)
    return validate_artifact_inventory(plan, ledger, value)


def validate_artifact_inventory(
    wave_plan: Mapping[str, Any], attempt_ledger: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("expected artifact inventory must be an object")
    payload = deepcopy(dict(value))
    _exact_keys(payload, _EXPECTED_INVENTORY_KEYS, "expected artifact inventory")
    if not isinstance(payload["records"], list):
        raise ValueError("expected artifact records are missing")
    seen_jobs: set[str] = set()
    seen_paths: set[str] = set()
    for row in payload["records"]:
        _exact_keys(row, _EXPECTED_RECORD_KEYS, "expected artifact record")
        paths = [row["done_path"], row["acceptance_path"], *row["root_paths"],
                 *row["source_hand_paths"]]
        if row["job_id"] in seen_jobs or any(path in seen_paths for path in paths):
            raise ValueError("expected artifact inventory contains a duplicate")
        seen_jobs.add(row["job_id"])
        seen_paths.update(paths)
    plan = validate_wave_plan(wave_plan)
    ledger = validate_attempt_ledger(plan, attempt_ledger)
    # Rebuild without recursion by comparing deterministic core with digest removed.
    latest = ledger["transitions"][-1]
    if _accepted_jobs(latest) != set(plan["coverage"]["job_ids"]):
        raise ValueError("complete accepted job evidence is required for inventory")
    if (
        payload["schema"] != EXPECTED_INVENTORY_SCHEMA
        or payload["run_name"] != plan["run_name"]
        or payload["execution_identity_sha256"]
        != plan["execution_identity_sha256"]
        or payload["wave_plan_sha256"] != plan["schedule_sha256"]
        or payload["attempt_ledger_sha256"] != ledger["ledger_sha256"]
        or payload["artifact_prefix"] != plan["artifact_contract"]["prefix"]
        or payload["job_count"] != 20
        or payload["paired_hand_count"] != 100
        or payload["source_root_count"] != 200
        or payload["source_hand_count"] != 200
        or payload["object_count"] != 440
        or payload["content_sha256_required"] is not True
        or payload["object_generation_required"] is not True
        or payload["overwrite_forbidden"] is not True
        or payload["inventory_sha256"] != canonical_sha256(
            {k: v for k, v in payload.items() if k != "inventory_sha256"}
        )
        or len(seen_jobs) != 20 or len(seen_paths) != 440
    ):
        raise ValueError("expected artifact inventory is missing or changed")
    # Deterministic path and binding checks against the terminal readback.
    done = {row["job_id"]: row for row in latest["done_objects"]}
    accepts = {row["job_id"]: row for row in latest["acceptance_records"]}
    meta = _job_metadata(plan)
    for row in payload["records"]:
        job = row["job_id"]
        d = done[job]
        a = accepts[job]
        attempt = d["attempt_id"]
        prefix = _attempt_prefix(plan, job, attempt)
        expected_paths = [f"{prefix}/roots/hand_{h:03d}.json" for h in meta[job]["work_hand_indices"]]
        expected_hands = [
            f"{prefix}/hands/{meta[job]['source_role']}/hand_{h:03d}.json"
            for h in meta[job]["work_hand_indices"]
        ]
        if (
            row["source_role"] != meta[job]["source_role"]
            or row["shard_index"] != meta[job]["shard_index"]
            or row["work_hand_indices"] != meta[job]["work_hand_indices"]
            or row["accepted_attempt_id"] != attempt
            or row["attempt_prefix"] != prefix
            or row["done_path"] != d["path"]
            or row["acceptance_path"] != a["path"]
            or row["root_paths"] != expected_paths
            or row["source_hand_paths"] != expected_hands
            or row["run_contract_digest"] != plan["run_contract_digest"]
            or any(row[key] != d[key] for key in (
                "done_identity_sha256", "package_sha256", "image_digest",
                "binary_sha256", "allocation_digest", "root_digest"
            ))
            or row["done_generation"] != d["generation"]
            or row["done_bytes"] != d["bytes"] or row["done_sha256"] != d["sha256"]
            or row["acceptance_generation"] != a["generation"]
            or row["acceptance_bytes"] != a["bytes"]
            or row["acceptance_sha256"] != a["sha256"]
            or row["immutable_write_once"] is not True
            or row["done_written_last"] is not True
            or row["acceptance_create_only"] is not True
        ):
            raise ValueError("expected artifact record path or binding changed")
    return payload


def build_observed_artifact_inventory(
    wave_plan: Mapping[str, Any], *, attempt_ledger: Mapping[str, Any],
    observed_at_utc: str, objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = expected_artifact_inventory(
        wave_plan, attempt_ledger=attempt_ledger
    )
    core: dict[str, Any] = {
        "schema": OBSERVED_INVENTORY_SCHEMA,
        "run_name": expected["run_name"],
        "execution_identity_sha256": expected["execution_identity_sha256"],
        "expected_inventory_sha256": expected["inventory_sha256"],
        "observed_at_utc": observed_at_utc, "readback_complete": True,
        "objects": deepcopy(list(objects)), "object_count": len(objects),
    }
    core["inventory_sha256"] = canonical_sha256(core)
    return validate_observed_artifact_inventory(
        wave_plan, attempt_ledger, core
    )


def _expected_object_specs(expected: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for row in expected["records"]:
        common = {
            "job_id": row["job_id"], "source_role": row["source_role"],
            "attempt_id": row["accepted_attempt_id"],
        }
        for hand, path in zip(row["work_hand_indices"], row["root_paths"], strict=True):
            specs[path] = {**common, "object_kind": "root", "hand_index": hand}
        for hand, path in zip(row["work_hand_indices"], row["source_hand_paths"], strict=True):
            specs[path] = {**common, "object_kind": "source_hand", "hand_index": hand}
        specs[row["done_path"]] = {**common, "object_kind": "done", "hand_index": None}
        specs[row["acceptance_path"]] = {
            **common, "object_kind": "acceptance", "hand_index": None
        }
    return specs


def validate_observed_artifact_inventory(
    wave_plan: Mapping[str, Any], attempt_ledger: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    plan = validate_wave_plan(wave_plan)
    ledger = validate_attempt_ledger(plan, attempt_ledger)
    expected = expected_artifact_inventory(plan, attempt_ledger=ledger)
    if not isinstance(value, Mapping):
        raise ValueError("observed artifact inventory must be an object")
    payload = deepcopy(dict(value))
    _exact_keys(payload, _OBSERVED_INVENTORY_KEYS, "observed artifact inventory")
    _parse_utc_seconds(payload["observed_at_utc"], "inventory observed time")
    if (
        payload["schema"] != OBSERVED_INVENTORY_SCHEMA
        or payload["run_name"] != plan["run_name"]
        or payload["execution_identity_sha256"] != plan["execution_identity_sha256"]
        or payload["expected_inventory_sha256"] != expected["inventory_sha256"]
        or payload["readback_complete"] is not True
        or not isinstance(payload["objects"], list)
        or payload["object_count"] != len(payload["objects"])
        or payload["inventory_sha256"] != canonical_sha256(
            {k: v for k, v in payload.items() if k != "inventory_sha256"}
        )
    ):
        raise ValueError("observed inventory identity or digest changed")
    specs = _expected_object_specs(expected)
    if len(payload["objects"]) != len(specs):
        raise ValueError("observed inventory has missing or extra objects")
    observed: dict[str, dict[str, Any]] = {}
    for raw in payload["objects"]:
        if not isinstance(raw, Mapping):
            raise ValueError("observed object is not an object")
        row = deepcopy(dict(raw))
        _exact_keys(row, _OBSERVED_OBJECT_KEYS, "observed object")
        path = row["path"]
        if path in observed:
            raise ValueError("observed inventory contains a duplicate path")
        spec = specs.get(path)
        if spec is None:
            raise ValueError("observed inventory contains an extra path")
        if any(row[key] != spec[key] for key in (
            "job_id", "source_role", "attempt_id", "object_kind", "hand_index"
        )):
            raise ValueError("observed object identity changed")
        _require_positive_int(row["generation"], "object generation")
        _require_positive_int(row["bytes"], "object bytes")
        _require_sha(row["sha256"], "object SHA-256")
        observed[path] = row
    if set(observed) != set(specs):
        raise ValueError("observed inventory has missing or extra objects")

    records = {row["job_id"]: row for row in expected["records"]}
    root_hashes_by_hand: dict[int, dict[str, str]] = {}
    for job, record in records.items():
        done = observed[record["done_path"]]
        acceptance = observed[record["acceptance_path"]]
        for key in ("done_identity_sha256", "package_sha256", "image_digest",
                    "binary_sha256", "allocation_digest", "root_digest"):
            if done[key] != record[key]:
                raise ValueError("observed DONE binding changed")
        if (
            done["generation"] != record["done_generation"]
            or done["bytes"] != record["done_bytes"]
            or done["sha256"] != record["done_sha256"]
            or acceptance["generation"] != record["acceptance_generation"]
            or acceptance["bytes"] != record["acceptance_bytes"]
            or acceptance["sha256"] != record["acceptance_sha256"]
        ):
            raise ValueError("observed DONE or acceptance metadata changed")
        for key in ("done_identity_sha256", "package_sha256", "image_digest",
                    "binary_sha256", "allocation_digest", "root_digest"):
            if acceptance[key] is not None:
                raise ValueError("acceptance object carries unexpected DONE fields")
        root_rows = [observed[path] for path in record["root_paths"]]
        bundle = canonical_sha256(
            [
                {"hand_index": hand, "sha256": row["sha256"]}
                for hand, row in zip(record["work_hand_indices"], root_rows, strict=True)
            ]
        )
        if bundle != record["root_digest"]:
            raise ValueError("observed root object digest does not match DONE")
        for hand, row in zip(record["work_hand_indices"], root_rows, strict=True):
            root_hashes_by_hand.setdefault(hand, {})[record["source_role"]] = row["sha256"]
        for path in [*record["root_paths"], *record["source_hand_paths"]]:
            row = observed[path]
            for key in ("done_identity_sha256", "package_sha256", "image_digest",
                        "binary_sha256", "allocation_digest", "root_digest"):
                if row[key] is not None:
                    raise ValueError("non-DONE object carries unexpected DONE fields")
    if any(
        set(by_role) != set(SOURCE_ROLES)
        or by_role["candidate"] != by_role["reference"]
        for by_role in root_hashes_by_hand.values()
    ):
        raise ValueError("candidate/reference root object hashes differ")
    return payload


def write_once(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(f"immutable full100 wave artifact exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    except FileExistsError as exc:
        raise FileExistsError(f"immutable full100 wave artifact exists: {target}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--identity-salt", required=True)
    parser.add_argument("--package-sha256", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--full100-plan", type=Path, default=DEFAULT_FULL100_PLAN_PATH)
    parser.add_argument(
        "--startup-canary",
        action="store_true",
        help="prepare only candidate-shard-00/a00 as a non-scientific startup canary",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = build_wave_plan(
        run_name=args.run_name, identity_salt=args.identity_salt,
        package_sha256=args.package_sha256, image_digest=args.image_digest,
        full100_plan_path=args.full100_plan,
        execution_scope=STARTUP_CANARY_SCOPE if args.startup_canary else None,
    )
    if args.output is not None:
        write_once(args.output, plan)
    print(json.dumps(plan, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_INVENTORY_SCHEMA", "ATTEMPT_LEDGER_SCHEMA",
    "AUTHORIZATION_BLOCKERS", "DEFAULT_FULL100_PLAN_PATH",
    "EXPECTED_INVENTORY_SCHEMA", "FULL100_EXECUTION_SCOPE", "MAX_CONCURRENT_VMS",
    "OBSERVED_INVENTORY_SCHEMA", "OBSERVED_TRANSITION_SCHEMA",
    "PERFORMANCE_LOCK_V4_EXECUTION_SCOPE", "RESUME_PLAN_SCHEMA",
    "STARTUP_CANARY_ATTEMPT_ID", "STARTUP_CANARY_JOB_ID",
    "STARTUP_CANARY_SCOPE", "STARTUP_CANARY_SOURCE_ROLE",
    "TOKYO_C4_FAMILY_QUOTA_VCPUS", "WAVE_PLAN_SCHEMA",
    "WAVE_SHARD_GROUPS", "WAVE_VM_COUNTS", "build_attempt_ledger",
    "build_observed_artifact_inventory", "build_observed_transition",
    "build_resume_plan", "build_wave_plan", "canonical_sha256",
    "empty_attempt_history", "expected_artifact_inventory",
    "expected_done_identity_sha256", "main", "mark_latest_transition_consumed",
    "validate_artifact_inventory", "validate_attempt_ledger",
    "validate_observed_artifact_inventory", "validate_observed_transition",
    "validate_resume_plan", "validate_startup_canary_plan",
    "validate_wave_plan", "write_once",
]
