"""Fresh, retry-free execution identity for the Step 12b diagnostic pair.

The accepted worker package is immutable and still contains the two frozen
Stage-2 source jobs.  This module gives a *new execution* a distinct run name,
GCE instance names, direct-v1 result prefix, controller principal, output root,
and externally visible job IDs.  The source job IDs are retained only as an
explicit one-to-one payload mapping.

No cloud API is called here.  The temporary patch context is required because
the immutable transport reconstructs its run name and deterministic instance
names during validation.  It changes imported module objects only while a new
contract is built or validated and restores every value on exit; no old source
file or artifact is rewritten.

Important: these rebound contracts are local alias-planning evidence only.
The patch cannot cross into a fresh VM Python process, so this module never
authorizes launching them as old transport contracts.  A separate v2 metadata
bridge must reproduce the validation context inside the worker and publish
only to the external fresh result identity.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_adapter
    as adapter,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as transport,
)
from . import hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as plan


SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_retry_free_attempt0_execution_identity_v1"
)
STATUS = "fresh_step12b_attempt0_alias_identity_local_planning_only"

EXECUTION_RUN_NAME = "regular-hu-m31-r2diag-s2b-20260720-001"
EXECUTION_STAGE_ID = "step12b_candidate_reference_pair_attempt0"
EXECUTION_CONFIRMATION = (
    "EXECUTE_STEP12B_STAGE2_CANDIDATE_REFERENCE_ATTEMPT0"
)
DEFAULT_OUTPUT_ROOT_NAME = "step12b_pair_v1_attempt0_actual"
LOCAL_SMOKE_OUTPUT_ROOT_NAME = "step12b_pair_v1_local_smoke"
LOCAL_SMOKE_IDENTITY = "step12b_local_smoke_never_cloud_executable"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ACTUAL_OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / DEFAULT_OUTPUT_ROOT_NAME
).resolve()

SOURCE_JOB_IDS = ("candidate-shard-01", "reference-shard-01")
EXECUTION_JOB_IDS = (
    "candidate-s2b-shard-01",
    "reference-s2b-shard-01",
)
SOURCE_TO_EXECUTION_JOB = MappingProxyType(
    dict(zip(SOURCE_JOB_IDS, EXECUTION_JOB_IDS, strict=True))
)
EXECUTION_TO_SOURCE_JOB = MappingProxyType(
    dict(zip(EXECUTION_JOB_IDS, SOURCE_JOB_IDS, strict=True))
)

PROJECT = transport.PROJECT
RUN_SCOPED_CONTROLLER_SERVICE_ACCOUNT = (
    f"ofc-m31-s2b-controller-001@{PROJECT}.iam.gserviceaccount.com"
)

ATTEMPT_INDEX = 0
MAX_ATTEMPTS_PER_JOB = 1
VM_COUNT = 2
TOKEN_LIFETIME_SECONDS = 3_600
MIN_EXECUTION_REMAINING_SECONDS = 5_600

EXPECTED_DIRECT_STAGE_IDENTITY_SHA256 = (
    "f36acce08acee8b7a8644015ce08ca426430dd415c3fb4def055a40b425e4e2c"
)
EXPECTED_DIRECT_STAGE_PREFIX = (
    "gs://pokerhu-ofc-solver-485418-training/"
    "hu-m31-r2diag-direct-v1/stages/"
    f"{EXECUTION_RUN_NAME}/{EXPECTED_DIRECT_STAGE_IDENTITY_SHA256}"
)

OLD_STEP12_RUN_NAME = "regular-hu-m31-r2diag-s2-20260718-001"
OLD_STEP12_DIRECT_STAGE_IDENTITY_SHA256 = (
    "b70bcad3f7df8721d544f45fcd4bc513ce27ea6c199f0cb1a5adc749e1d029e4"
)
OLD_STEP12_INSTANCE_NAMES = frozenset(
    {
        "r2d-10c2-s2-candidate-01-a0-29e3c6f8",
        "r2d-10c2-s2-reference-01-a0-29e3c6f8",
    }
)
OLD_STEP12_OUTPUT_ROOT_NAME = "step12_pair_v1_actual"
OLD_STEP12_DIRECT_STAGE_PREFIX = (
    "gs://pokerhu-ofc-solver-485418-training/"
    "hu-m31-r2diag-direct-v1/stages/"
    f"{OLD_STEP12_RUN_NAME}/{OLD_STEP12_DIRECT_STAGE_IDENTITY_SHA256}"
)

PACKAGE_READBACK_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_immutable_package_generation_readback_v1"
)
PACKAGE_OBSERVATION_MAX_AGE_SECONDS = 120

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_INSTANCE = re.compile(r"^[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?$")
_PATCH_LOCK = threading.RLock()
_PATCH_ACTIVE = False


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def execution_instance_name(source_job_id: str, attempt_index: int) -> str:
    """Return the only GCE name allowed for one rebound source job/attempt."""

    if source_job_id not in SOURCE_JOB_IDS:
        raise ValueError("Step12b source job escaped the frozen pair")
    if type(attempt_index) is not int or attempt_index not in (0, 1):
        raise ValueError("Step12b deterministic-name attempt is invalid")
    role = "cand" if source_job_id == SOURCE_JOB_IDS[0] else "ref"
    suffix = hashlib.sha256(
        (
            f"{EXECUTION_RUN_NAME}|{source_job_id}|attempt-{attempt_index}|"
            "step12b-attempt0-v1"
        ).encode("ascii")
    ).hexdigest()[:8]
    value = f"r2d-10c2-s2b-{role}-01-a{attempt_index}-{suffix}"
    if _SAFE_INSTANCE.fullmatch(value) is None:
        raise AssertionError("Step12b deterministic instance name is invalid")
    return value


ATTEMPT0_INSTANCE_NAMES = tuple(
    execution_instance_name(job_id, ATTEMPT_INDEX)
    for job_id in SOURCE_JOB_IDS
)


def _patched_stage(
    original: Any,
    manifest: Mapping[str, Any],
    stage_id: str,
) -> dict[str, Any]:
    stage = original(manifest, stage_id)
    if stage_id == plan.STAGE2_ID:
        stage = dict(stage)
        stage["run_name"] = EXECUTION_RUN_NAME
    return stage


def _patched_instance_name(
    original: Any,
    *,
    stage_id: str,
    job_id: str,
    attempt_index: int,
    preview_stage_identity_sha256: str,
) -> str:
    _sha(preview_stage_identity_sha256, "preview stage identity")
    if stage_id == plan.STAGE2_ID and job_id in SOURCE_JOB_IDS:
        return execution_instance_name(job_id, attempt_index)
    return original(
        stage_id=stage_id,
        job_id=job_id,
        attempt_index=attempt_index,
        preview_stage_identity_sha256=preview_stage_identity_sha256,
    )


@contextmanager
def rebound_transport_identity() -> Iterator[None]:
    """Temporarily install the exact Step12b run/name reconstruction.

    This is intentionally process-global and therefore serialized.  Nested
    entry is rejected so a caller cannot accidentally validate under a partly
    restored identity.
    """

    global _PATCH_ACTIVE
    with _PATCH_LOCK:
        if _PATCH_ACTIVE:
            raise RuntimeError("Step12b identity patch is not re-entrant")
        _PATCH_ACTIVE = True
        original_stage = adapter._stage
        original_name = transport.deterministic_instance_name
        original_run_name = transport.STAGE2_RUN_NAME
        try:
            adapter._stage = lambda manifest, stage_id: _patched_stage(
                original_stage, manifest, stage_id
            )
            transport.deterministic_instance_name = (
                lambda **kwargs: _patched_instance_name(
                    original_name, **kwargs
                )
            )
            transport.STAGE2_RUN_NAME = EXECUTION_RUN_NAME
            yield
        finally:
            adapter._stage = original_stage
            transport.deterministic_instance_name = original_name
            transport.STAGE2_RUN_NAME = original_run_name
            _PATCH_ACTIVE = False


def build_rebound_job_contract(**kwargs: Any) -> dict[str, Any]:
    """Build one immutable-payload contract under the fresh execution ID."""

    if kwargs.get("stage_id") != plan.STAGE2_ID:
        raise ValueError("Step12b may build only the frozen Stage2 payload")
    if kwargs.get("job_id") not in SOURCE_JOB_IDS:
        raise ValueError("Step12b may build only the two frozen source jobs")
    if kwargs.get("attempt_index", 0) != ATTEMPT_INDEX:
        raise ValueError("Step12b attempt1 is not authorized")
    with rebound_transport_identity():
        contract = transport.build_job_contract(**kwargs)
        return _validate_rebound_job_contract_in_context(contract)


def _validate_rebound_job_contract_in_context(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    checked = transport.validate_job_contract(value)
    binding = checked["metadata_binding"]
    source_job_id = binding["job_id"]
    if (
        checked["adapter_preview"]["run_name"] != EXECUTION_RUN_NAME
        or source_job_id not in SOURCE_JOB_IDS
        or binding["attempt_index"] != ATTEMPT_INDEX
        or binding["instance_name"]
        != execution_instance_name(source_job_id, ATTEMPT_INDEX)
        or checked["direct_stage_identity_sha256"]
        != EXPECTED_DIRECT_STAGE_IDENTITY_SHA256
        or checked["remote_layout"]["stage_prefix"]
        != EXPECTED_DIRECT_STAGE_PREFIX
        or checked["remote_layout"]["stage_prefix"]
        == OLD_STEP12_DIRECT_STAGE_PREFIX
        or OLD_STEP12_RUN_NAME
        in checked["remote_layout"]["stage_prefix"]
        or binding["instance_name"] in OLD_STEP12_INSTANCE_NAMES
    ):
        raise ValueError("Step12b rebound transport identity changed")
    return checked


def validate_rebound_job_contract(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    with rebound_transport_identity():
        return _validate_rebound_job_contract_in_context(value)


def _exact_output_root(
    path: str | Path,
    *,
    must_be_fresh: bool,
) -> Path:
    target = Path(path).resolve()
    if (
        target != EXPECTED_ACTUAL_OUTPUT_ROOT
        or target.name != DEFAULT_OUTPUT_ROOT_NAME
        or target.name == OLD_STEP12_OUTPUT_ROOT_NAME
        or target.name == LOCAL_SMOKE_OUTPUT_ROOT_NAME
        or target.is_symlink()
        or must_be_fresh
        and target.exists()
    ):
        raise FileExistsError(
            "Step12b requires its exact new actual output root"
        )
    return target


def require_fresh_output_root(path: str | Path) -> Path:
    return _exact_output_root(path, must_be_fresh=True)


def require_local_smoke_output_root(path: str | Path) -> Path:
    """Keep local smoke output disjoint from the static actual identity."""

    target = Path(path).resolve()
    if (
        target.name != LOCAL_SMOKE_OUTPUT_ROOT_NAME
        or target.name in {
            DEFAULT_OUTPUT_ROOT_NAME,
            OLD_STEP12_OUTPUT_ROOT_NAME,
        }
        or target.exists()
        or target.is_symlink()
    ):
        raise FileExistsError(
            "Step12b local smoke requires its disjoint nonexistent root"
        )
    return target


def _package_reuse_requirements(
    contracts: tuple[Mapping[str, Any], Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_inventory = contracts[0]["remote_layout"][
        "package_inventory"
    ]
    reference_inventory = contracts[1]["remote_layout"][
        "package_inventory"
    ]
    if candidate_inventory != reference_inventory:
        raise ValueError("Step12b immutable package inventories differ")
    records = [
        {
            "uri": row["uri"],
            "sha256": _sha(row["sha256"], "package object sha256"),
            "bytes": row["bytes"],
        }
        for row in candidate_inventory["records"]
    ]
    if (
        not records
        or len({row["uri"] for row in records}) != len(records)
        or any(
            type(row["bytes"]) is not int or row["bytes"] <= 0
            for row in records
        )
        or candidate_inventory["package_prefix"]
        != contracts[0]["remote_layout"]["package_prefix"]
    ):
        raise ValueError("Step12b immutable package inventory changed")
    return {
        "package_prefix": candidate_inventory["package_prefix"],
        "outer_package_identity_sha256": contracts[0][
            "outer_package_manifest"
        ]["outer_package_identity_sha256"],
        "record_count": len(records),
        "records": records,
        "records_sha256": canonical_sha256(records),
        "old_immutable_package_bytes_may_be_reused": True,
        "generation_pinned_get_required_for_every_object": True,
        "exact_generation_sha256_bytes_readback_required": True,
        "package_write_forbidden": True,
        "old_result_stage_prefix_reuse_forbidden": True,
    }


def _build_execution_identity(
    candidate_transport_contract: Mapping[str, Any],
    reference_transport_contract: Mapping[str, Any],
    *,
    controller_public_key_sha256: str,
    gate_plan_sha256: str,
    immutable_package_source_receipt_sha256: str,
    immutable_package_generations_sha256: str,
    issued_at_unix_seconds: int,
    expires_at_unix_seconds: int,
    output_root: str | Path,
    require_fresh_output: bool,
) -> dict[str, Any]:
    candidate = validate_rebound_job_contract(candidate_transport_contract)
    reference = validate_rebound_job_contract(reference_transport_contract)
    contracts = (candidate, reference)
    if type(require_fresh_output) is not bool:
        raise ValueError("Step12b output freshness mode changed")
    root = _exact_output_root(
        output_root, must_be_fresh=require_fresh_output
    )
    issued = issued_at_unix_seconds
    expires = expires_at_unix_seconds
    if (
        type(issued) is not int
        or type(expires) is not int
        or issued <= 0
        or expires - issued > 7_200
        or expires - issued < MIN_EXECUTION_REMAINING_SECONDS
    ):
        raise ValueError("Step12b authorization window is invalid")
    configured_keys = {
        contract["authorization_contract"][
            "controller_public_key_sha256"
        ]
        for contract in contracts
    }
    if (
        [row["metadata_binding"]["job_id"] for row in contracts]
        != list(SOURCE_JOB_IDS)
        or [row["metadata_binding"]["instance_name"] for row in contracts]
        != list(ATTEMPT0_INSTANCE_NAMES)
        or candidate["remote_layout"] != reference["remote_layout"]
        or candidate["direct_stage_identity"]
        != reference["direct_stage_identity"]
        or configured_keys != {controller_public_key_sha256}
    ):
        raise ValueError("Step12b candidate/reference payload mapping changed")
    package_reuse = _package_reuse_requirements(contracts)
    body = {
        "schema": SCHEMA,
        "status": STATUS,
        "execution_stage_id": EXECUTION_STAGE_ID,
        "execution_run_name": EXECUTION_RUN_NAME,
        "execution_job_ids": list(EXECUTION_JOB_IDS),
        "source_job_ids": list(SOURCE_JOB_IDS),
        "job_bindings": [
            {
                "execution_job_id": execution_job_id,
                "source_job_id": source_job_id,
                "source_role": contract["metadata_binding"]["source_role"],
                "instance_name": contract["metadata_binding"][
                    "instance_name"
                ],
                "transport_contract_sha256": canonical_sha256(contract),
            }
            for execution_job_id, source_job_id, contract in zip(
                EXECUTION_JOB_IDS, SOURCE_JOB_IDS, contracts, strict=True
            )
        ],
        "attempt_index": ATTEMPT_INDEX,
        "max_attempts_per_job": MAX_ATTEMPTS_PER_JOB,
        "vm_count": VM_COUNT,
        "instance_names": list(ATTEMPT0_INSTANCE_NAMES),
        "direct_stage_identity_sha256": (
            EXPECTED_DIRECT_STAGE_IDENTITY_SHA256
        ),
        "direct_stage_prefix": EXPECTED_DIRECT_STAGE_PREFIX,
        "old_step12_direct_stage_prefix": OLD_STEP12_DIRECT_STAGE_PREFIX,
        "immutable_package_reuse": package_reuse,
        "immutable_package_reuse_sha256": canonical_sha256(package_reuse),
        "immutable_package_source_receipt_sha256": _sha(
            immutable_package_source_receipt_sha256,
            "immutable package source receipt",
        ),
        "immutable_package_generations_sha256": _sha(
            immutable_package_generations_sha256,
            "immutable package generations",
        ),
        "output_root": str(root),
        "controller_service_account": (
            RUN_SCOPED_CONTROLLER_SERVICE_ACCOUNT
        ),
        "run_scoped_controller_required": True,
        "prior_controller_token_reuse_forbidden": True,
        "controller_public_key_sha256": _sha(
            controller_public_key_sha256, "controller public key"
        ),
        "gate_plan_sha256": _sha(gate_plan_sha256, "gate plan"),
        "issued_at_unix_seconds": issued,
        "expires_at_unix_seconds": expires,
        "minimum_execution_remaining_seconds": (
            MIN_EXECUTION_REMAINING_SECONDS
        ),
        "execution_confirmation": EXECUTION_CONFIRMATION,
        "execution_confirmation_sha256": hashlib.sha256(
            EXECUTION_CONFIRMATION.encode("ascii")
        ).hexdigest(),
        "retry_authorized": False,
        "resume_authorized": False,
        "attempt1_authorized": False,
        "third_vm_authorized": False,
        "old_step12_run_reuse_forbidden": True,
        "old_step12_prefix_reuse_forbidden": True,
        "old_step12_instance_reuse_forbidden": True,
        "old_step12_output_root_reuse_forbidden": True,
        "local_alias_planning_only": True,
        "inner_fresh_process_rebind_implemented": False,
        "rebound_transport_contract_cloud_executable": False,
        "v2_alias_bridge_required_before_cloud_launch": True,
        "actual_identity_must_be_claimed_before_first_cloud_mutation": True,
        "actual_identity_reuse_after_mutation_forbidden": True,
        "local_smoke_identity": False,
        "local_smoke_output_root_name": LOCAL_SMOKE_OUTPUT_ROOT_NAME,
        "local_smoke_may_consume_actual_identity": False,
        "cloud_launch_authorized": False,
        "cloud_mutation_performed": False,
        "diagnostic_only": True,
        "scientific_payload_present": False,
        "performance_lock_evidence": False,
        "quality_evidence": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }
    return {**body, "identity_sha256": canonical_sha256(body)}


def build_execution_identity(
    candidate_transport_contract: Mapping[str, Any],
    reference_transport_contract: Mapping[str, Any],
    *,
    controller_public_key_sha256: str,
    gate_plan_sha256: str,
    immutable_package_source_receipt_sha256: str,
    immutable_package_generations_sha256: str,
    issued_at_unix_seconds: int,
    expires_at_unix_seconds: int,
    output_root: str | Path,
) -> dict[str, Any]:
    return _build_execution_identity(
        candidate_transport_contract,
        reference_transport_contract,
        controller_public_key_sha256=controller_public_key_sha256,
        gate_plan_sha256=gate_plan_sha256,
        immutable_package_source_receipt_sha256=(
            immutable_package_source_receipt_sha256
        ),
        immutable_package_generations_sha256=(
            immutable_package_generations_sha256
        ),
        issued_at_unix_seconds=issued_at_unix_seconds,
        expires_at_unix_seconds=expires_at_unix_seconds,
        output_root=output_root,
        require_fresh_output=True,
    )


def validate_execution_identity(
    value: Mapping[str, Any],
    candidate_transport_contract: Mapping[str, Any],
    reference_transport_contract: Mapping[str, Any],
    *,
    controller_public_key_sha256: str,
    gate_plan_sha256: str,
    immutable_package_source_receipt_sha256: str,
    immutable_package_generations_sha256: str,
    issued_at_unix_seconds: int,
    expires_at_unix_seconds: int,
    output_root: str | Path,
) -> dict[str, Any]:
    checked = copy.deepcopy(dict(value))
    digest = checked.pop("identity_sha256", None)
    if digest != canonical_sha256(checked):
        raise ValueError("Step12b execution identity digest changed")
    expected = _build_execution_identity(
        candidate_transport_contract,
        reference_transport_contract,
        controller_public_key_sha256=controller_public_key_sha256,
        gate_plan_sha256=gate_plan_sha256,
        immutable_package_source_receipt_sha256=(
            immutable_package_source_receipt_sha256
        ),
        immutable_package_generations_sha256=(
            immutable_package_generations_sha256
        ),
        issued_at_unix_seconds=issued_at_unix_seconds,
        expires_at_unix_seconds=expires_at_unix_seconds,
        output_root=output_root,
        require_fresh_output=False,
    )
    if value != expected:
        raise ValueError("Step12b execution identity changed")
    return copy.deepcopy(dict(value))


def build_immutable_package_readback_receipt(
    candidate_transport_contract: Mapping[str, Any],
    reference_transport_contract: Mapping[str, Any],
    *,
    execution_identity_sha256: str,
    source_package_receipt_sha256: str,
    expected_package_generations: Mapping[str, int],
    observed_at_unix_seconds: int,
    observed_records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Seal a fresh exact GET of the only old bytes Step12b may reuse."""

    contracts = (
        validate_rebound_job_contract(candidate_transport_contract),
        validate_rebound_job_contract(reference_transport_contract),
    )
    requirements = _package_reuse_requirements(contracts)
    expected = {row["uri"]: row for row in requirements["records"]}
    if (
        not isinstance(expected_package_generations, Mapping)
        or set(expected_package_generations) != set(expected)
        or any(
            type(generation) is not int or generation <= 0
            for generation in expected_package_generations.values()
        )
    ):
        raise ValueError("Step12b pinned package generation map changed")
    if (
        type(observed_at_unix_seconds) is not int
        or observed_at_unix_seconds <= 0
        or not isinstance(observed_records, list)
        or len(observed_records) != len(expected)
    ):
        raise ValueError("Step12b package readback observation changed")
    checked_records: list[dict[str, Any]] = []
    for raw in observed_records:
        if not isinstance(raw, Mapping):
            raise ValueError("Step12b package readback record changed")
        uri = raw.get("uri")
        wanted = expected.get(uri)
        if (
            wanted is None
            or type(raw.get("generation")) is not int
            or raw["generation"] <= 0
            or raw["generation"] != expected_package_generations[uri]
            or raw.get("sha256") != wanted["sha256"]
            or raw.get("bytes") != wanted["bytes"]
        ):
            raise ValueError(
                "Step12b package generation/SHA/bytes readback changed"
            )
        checked_records.append(
            {
                "uri": uri,
                "generation": raw["generation"],
                "sha256": raw["sha256"],
                "bytes": raw["bytes"],
            }
        )
    checked_records.sort(key=lambda row: row["uri"])
    if len({row["uri"] for row in checked_records}) != len(expected):
        raise ValueError("Step12b package readback URI set changed")
    body = {
        "schema": PACKAGE_READBACK_SCHEMA,
        "status": (
            "fresh_generation_pinned_exact_immutable_package_readback"
        ),
        "execution_run_name": EXECUTION_RUN_NAME,
        "execution_identity_sha256": _sha(
            execution_identity_sha256, "execution identity"
        ),
        "source_package_receipt_sha256": _sha(
            source_package_receipt_sha256,
            "source package receipt",
        ),
        "package_prefix": requirements["package_prefix"],
        "outer_package_identity_sha256": requirements[
            "outer_package_identity_sha256"
        ],
        "observed_at_unix_seconds": observed_at_unix_seconds,
        "observation_max_age_seconds": (
            PACKAGE_OBSERVATION_MAX_AGE_SECONDS
        ),
        "record_count": len(checked_records),
        "records": checked_records,
        "records_sha256": canonical_sha256(checked_records),
        "package_generations": {
            uri: expected_package_generations[uri]
            for uri in sorted(expected_package_generations)
        },
        "package_generations_sha256": canonical_sha256(
            {
                uri: expected_package_generations[uri]
                for uri in sorted(expected_package_generations)
            }
        ),
        "all_gets_generation_pinned": True,
        "all_sha256_verified_from_response_bytes": True,
        "all_byte_lengths_verified": True,
        "package_reuse_only": True,
        "result_stage_prefix_reuse": False,
        "collected_via_get_only": True,
        "cloud_mutation_performed": False,
        "access_token_stored": False,
        "authorization_header_stored": False,
        "current_profile_changed": False,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def validate_immutable_package_readback_receipt(
    value: Mapping[str, Any],
    candidate_transport_contract: Mapping[str, Any],
    reference_transport_contract: Mapping[str, Any],
    *,
    execution_identity_sha256: str,
    source_package_receipt_sha256: str,
    expected_package_generations: Mapping[str, int],
    now_unix_seconds: int,
) -> dict[str, Any]:
    checked = copy.deepcopy(dict(value))
    digest = checked.pop("receipt_sha256", None)
    if digest != canonical_sha256(checked):
        raise ValueError("Step12b package readback receipt digest changed")
    observed_at = checked.get("observed_at_unix_seconds")
    if (
        type(now_unix_seconds) is not int
        or type(observed_at) is not int
        or now_unix_seconds < observed_at
        or now_unix_seconds - observed_at
        > PACKAGE_OBSERVATION_MAX_AGE_SECONDS
    ):
        raise ValueError("Step12b package readback receipt is stale")
    expected = build_immutable_package_readback_receipt(
        candidate_transport_contract,
        reference_transport_contract,
        execution_identity_sha256=execution_identity_sha256,
        source_package_receipt_sha256=source_package_receipt_sha256,
        expected_package_generations=expected_package_generations,
        observed_at_unix_seconds=observed_at,
        observed_records=checked.get("records"),
    )
    if value != expected:
        raise ValueError("Step12b package readback receipt changed")
    return copy.deepcopy(dict(value))


__all__ = [
    "ATTEMPT0_INSTANCE_NAMES",
    "ATTEMPT_INDEX",
    "DEFAULT_OUTPUT_ROOT_NAME",
    "EXECUTION_CONFIRMATION",
    "EXECUTION_JOB_IDS",
    "EXECUTION_RUN_NAME",
    "EXPECTED_ACTUAL_OUTPUT_ROOT",
    "EXPECTED_DIRECT_STAGE_IDENTITY_SHA256",
    "EXPECTED_DIRECT_STAGE_PREFIX",
    "LOCAL_SMOKE_IDENTITY",
    "LOCAL_SMOKE_OUTPUT_ROOT_NAME",
    "OLD_STEP12_DIRECT_STAGE_PREFIX",
    "PACKAGE_OBSERVATION_MAX_AGE_SECONDS",
    "PACKAGE_READBACK_SCHEMA",
    "RUN_SCOPED_CONTROLLER_SERVICE_ACCOUNT",
    "SCHEMA",
    "SOURCE_JOB_IDS",
    "build_execution_identity",
    "build_immutable_package_readback_receipt",
    "build_rebound_job_contract",
    "canonical_sha256",
    "execution_instance_name",
    "rebound_transport_identity",
    "require_fresh_output_root",
    "require_local_smoke_output_root",
    "validate_execution_identity",
    "validate_immutable_package_readback_receipt",
    "validate_rebound_job_contract",
]
