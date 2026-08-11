"""Fail-closed Spot adapter for the fresh Candidate02 rearm2 lock.

This module reuses the audited v1 transport and the rearm1 actual-package
smoke helpers, but gives rearm2 its own producer modules, archive proof,
global Spot claim, and authorization binding.  Importing it has no effect on
v1 or rearm1 globals.

The cloud boundary is intentionally two-phase:

1. execute the packaged startup pre-content verifier for all 20 actual job
   manifests while temporary root payloads are poisoned, then persist one
   canonical write-once receipt;
2. bind that receipt's SHA-256 into both the rearm2 global Spot claim and the
   launch authorization before any cloud-capable operation is allowed.

The receipt body does not contain its own digest.  Its deterministic digest is
computed before the 20 verifier invocations and supplied to their preview
claim/authorization.  The receipt is written only after every invocation
passes, avoiding a circular self-hash while exercising the exact cloud
contract.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Mapping, Sequence

from . import hu_m31_t3_step6d_performance_lock_rearm1_spot as rearm1_spot
from . import hu_m31_t3_step6d_performance_lock_spot_v1 as spot_v1


PREAUTHORIZE_SMOKE_SCHEMA = rearm1_spot.PREAUTHORIZE_SMOKE_SCHEMA
PREAUTHORIZE_SMOKE_STATUS = rearm1_spot.PREAUTHORIZE_SMOKE_STATUS
PREAUTHORIZE_SMOKE_NAME = rearm1_spot.PREAUTHORIZE_SMOKE_NAME
PREAUTHORIZE_SMOKE_SHA_FIELD = "preauthorize_smoke_receipt_sha256"
PREAUTHORIZE_SMOKE_KEYS = frozenset(
    {
        "actual_package_manifest_validated",
        "candidate_variant",
        "cloud_mutation_executed",
        "current_profile_changed",
        "done_schema",
        "full_no_root_read_startup_contract_executed",
        "global_spot_claim_name",
        "job_manifest_aggregate_sha256",
        "package_manifest_sha256",
        "persistent_claim_or_authorization_written",
        "plan_sha256",
        "preview_authorization_validated",
        "receipt_sha_bound_before_cloud_mutation",
        "root_schema",
        "run_contract_digest",
        "run_name",
        "schedule",
        "schema",
        "shard_manifest_schema",
        "signed_root_claim_strings_preserved",
        "source_sha256",
        "startup_invocation_count",
        "startup_package_phase",
        "startup_root_member_read",
        "startup_sha256",
        "status",
        "step6d_run_id",
        "temporary_poisoned_root_count",
        "verified_job_count",
        "verified_job_ids",
        "verified_job_manifests",
        "write_once_receipt_required",
    }
)

GLOBAL_SPOT_CLAIM_NAME = "GLOBAL_PERFORMANCE_LOCK_REARM2_SPOT_CLAIM.json"
GLOBAL_SPOT_CLAIM_SCHEMA = (
    "hu_m31_t3_step6d_performance_lock_rearm2_global_spot_claim_v1"
)
GLOBAL_SPOT_CLAIM_STATUS = (
    "global_rearm2_one_shot_spot_identity_claimed_after_exhaustive_smoke_"
    "before_authorization"
)
REARM2_PLAN_SCOPE = "performance_lock_rearm2_fresh_roots_only"
REARM2_STARTUP_PACKAGE_PHASE = "lock_rearm2"

REARM2_PARITY_VALIDATOR_PACKAGE_PATH = (
    "src/ofc_regular/"
    "verify_hu_m31_t3_feature_encoder_platform_parity_rearm2.py"
)
REARM2_PARITY_VALIDATOR_SCHEMA = (
    "hu_m31_t3_feature_encoder_rearm2_chain_validator_v1"
)
REARM2_PARITY_VALIDATOR_STATUS = (
    "verified_rearm2_chain_and_zero_v1_rearm1_reuse_before_packaging"
)
EXPECTED_REARM2_PARITY_VALIDATOR_SHA256 = (
    "45b33acbc1673199ff0c8ab0d68f5d938398a4f8013004b80cee93924cc87897"
)
EXPECTED_REARM2_PARITY_VALIDATOR_BYTES = 17_649

DEFAULT_PROJECT = spot_v1.DEFAULT_PROJECT
DEFAULT_BUCKET = spot_v1.DEFAULT_BUCKET
DEFAULT_REGION = spot_v1.DEFAULT_REGION
DEFAULT_ZONES = spot_v1.DEFAULT_ZONES

_PLAN_MODULE = "ofc_regular.hu_m31_t3_step6d_candidate02_performance_lock_rearm2_plan"
_OPEN_MODULE = "ofc_regular.hu_m31_t3_step6d_performance_lock_rearm2_open"
_CONTEXT_LOCK = rearm1_spot._CONTEXT_LOCK
_BASE_VALIDATE_ARCHIVE = spot_v1._validate_archive
_BASE_GLOBAL_SPOT_CLAIM_PAYLOAD = spot_v1._global_spot_claim_payload
_BASE_GLOBAL_SPOT_CLAIM_KEYS = spot_v1._GLOBAL_SPOT_CLAIM_KEYS
_BASE_AUTHORIZATION_KEYS = spot_v1._AUTHORIZATION_KEYS
_BASE_VALIDATE_LAUNCH_AUTHORIZATION = spot_v1.validate_launch_authorization

_BOUND_GLOBAL_SPOT_CLAIM_KEYS = frozenset(
    set(_BASE_GLOBAL_SPOT_CLAIM_KEYS) | {PREAUTHORIZE_SMOKE_SHA_FIELD}
)
_BOUND_AUTHORIZATION_KEYS = frozenset(
    set(_BASE_AUTHORIZATION_KEYS) | {PREAUTHORIZE_SMOKE_SHA_FIELD}
)


def _load_rearm_modules() -> tuple[ModuleType, ModuleType]:
    """Load producer-owned rearm2 modules without import-time mutation."""

    plan = importlib.import_module(_PLAN_MODULE)
    lock_open = importlib.import_module(_OPEN_MODULE)
    required_plan = (
        "CANDIDATE_VARIANT",
        "RUN_ID",
        "SCHEDULE",
        "ROOT_SCHEMA",
        "DONE_SCHEMA",
        "SHARD_MANIFEST_SCHEMA",
        "LOCK_RUN_CONTRACT_DIGEST",
        "PRECONTENT_PLAN_SHA256",
        "PLAN_SCOPE",
        "CANDIDATE_LIBRARY_SHA256",
        "REFERENCE_LIBRARY_SHA256",
        "FEATURE_ENCODER_SHA256",
        "CURRENT_PROFILE_REGISTRY_SHA256",
        "validate_precontent_plan",
    )
    required_open = (
        "CLAIM_SCHEMA",
        "CLAIM_STATUS",
        "MATERIALIZATION_SCHEMA",
        "MATERIALIZATION_STATUS",
        "SEAL_SCHEMA",
        "SEAL_STATUS",
        "PerformanceLockInputs",
        "DEFAULT_GLOBAL_CLAIM_PATH",
        "DEFAULT_PRECONTENT_PLAN_PATH",
        "validate_open_claim",
        "validate_root_seal",
    )
    missing_plan = [name for name in required_plan if not hasattr(plan, name)]
    missing_open = [name for name in required_open if not hasattr(lock_open, name)]
    if missing_plan or missing_open:
        raise RuntimeError(
            "rearm2 Spot producer API is incomplete: "
            f"plan={missing_plan}, open={missing_open}"
        )
    if plan.PLAN_SCOPE != REARM2_PLAN_SCOPE:
        raise ValueError("rearm2 plan scope changed")
    return plan, lock_open


def _prior_no_reuse_fields(
    value: Mapping[str, Any],
    *,
    label: str,
    require_old_v1_root: bool = True,
) -> None:
    """Require explicit false reuse flags for both consumed lock identities."""

    fields = (
        "rearm1_attempt1_reused",
        "rearm1_package_reused",
        "rearm1_root_reused",
        "rearm1_seed_reused",
        "rearm1_claim_reused",
    )
    if require_old_v1_root and value.get("old_v1_root_reused") is not False:
        raise ValueError(f"{label} permits prior lock reuse")
    if any(value.get(field) is not False for field in fields):
        raise ValueError(f"{label} permits prior lock reuse")


def _validate_rearm2_parity_binding(
    source: Path,
    *,
    claim: Mapping[str, Any],
    materialization: Mapping[str, Any],
    seal: Mapping[str, Any],
) -> None:
    """Bind the packaged base-compatible receipt to the rearm2 validator."""

    with zipfile.ZipFile(source) as archive:
        raw_receipt = archive.read(
            spot_v1.MATERIALIZER_PARITY_RECEIPT_PACKAGE_PATH
        )
        receipt = json.loads(raw_receipt.decode("utf-8"))
        validator_bytes = archive.read(REARM2_PARITY_VALIDATOR_PACKAGE_PATH)
    if (
        not isinstance(receipt, dict)
        or raw_receipt != spot_v1.canonical_bytes(receipt)
    ):
        raise ValueError("packaged rearm2 parity receipt is not canonical")
    if (
        not EXPECTED_REARM2_PARITY_VALIDATOR_SHA256
        or EXPECTED_REARM2_PARITY_VALIDATOR_BYTES <= 0
    ):
        raise ValueError("rearm2 parity validator identity is not frozen")

    chain = receipt.get("lock_chain")
    validator_sha = hashlib.sha256(validator_bytes).hexdigest()
    if (
        validator_sha != EXPECTED_REARM2_PARITY_VALIDATOR_SHA256
        or len(validator_bytes) != EXPECTED_REARM2_PARITY_VALIDATOR_BYTES
        or not isinstance(chain, Mapping)
        or chain.get("contract") != "performance_lock_rearm2_v1"
        or chain.get("validator_schema") != REARM2_PARITY_VALIDATOR_SCHEMA
        or chain.get("validator_status") != REARM2_PARITY_VALIDATOR_STATUS
        or chain.get("validator_bytes") != EXPECTED_REARM2_PARITY_VALIDATOR_BYTES
        or chain.get("validator_sha256")
        != EXPECTED_REARM2_PARITY_VALIDATOR_SHA256
        or chain.get("global_claim_sha256") != spot_v1.canonical_sha256(claim)
        or chain.get("materialization_sha256")
        != spot_v1.canonical_sha256(materialization)
        or chain.get("seal_sha256") != spot_v1.canonical_sha256(seal)
        or chain.get("v1_reused") is not False
        or chain.get("rearm1_reused") is not False
        or chain.get("fresh_recovery_v3_seed_schedule") is not True
        or chain.get("reseeded") is not False
        or chain.get("current_profile_changed") is not False
    ):
        raise ValueError("packaged rearm2 parity validator binding changed")


def _validate_rearm2_archive(
    source: Path,
    entries: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Replay v1 package validation, then require v1+rearm1 non-reuse."""

    plan, claim, seal, materialization = _BASE_VALIDATE_ARCHIVE(source, entries)
    claim_guards = claim.get("rearm_guards")
    comparison = seal.get("prior_lock_comparison")
    seal_guards = seal.get("rearm_guards")
    if not isinstance(claim_guards, Mapping):
        raise ValueError("packaged rearm2 claim no-reuse proof is missing")
    _prior_no_reuse_fields(claim_guards, label="packaged rearm2 claim")
    _prior_no_reuse_fields(
        materialization,
        label="packaged rearm2 materialization",
    )
    if (
        materialization.get("fresh_recovery_v3_seed_schedule") is not True
        or materialization.get("reseeded") is not False
    ):
        raise ValueError("packaged rearm2 materialization was reseeded")

    overlap_fields = tuple(
        f"{prior}_{kind}_overlap_count"
        for prior in ("old_v1", "rearm1")
        for kind in ("root_hash", "fingerprint", "seed")
    )
    if (
        not isinstance(comparison, Mapping)
        or any(comparison.get(field) != 0 for field in overlap_fields)
        or comparison.get("old_v1_global_claim_sha256")
        != spot_v1.lock_open.OLD_V1_GLOBAL_CLAIM_SHA256
        or comparison.get("old_v1_seal_sha256")
        != spot_v1.lock_open.OLD_V1_SEAL_SHA256
        or comparison.get("rearm1_global_claim_sha256")
        != spot_v1.lock_open.REARM1_GLOBAL_CLAIM_SHA256
        or comparison.get("rearm1_seal_sha256")
        != spot_v1.lock_open.REARM1_SEAL_SHA256
        or not isinstance(seal_guards, Mapping)
    ):
        raise ValueError("packaged rearm2 prior-lock overlap proof changed")
    _prior_no_reuse_fields(
        comparison,
        label="packaged rearm2 prior comparison",
        require_old_v1_root=False,
    )
    _prior_no_reuse_fields(
        seal_guards,
        label="packaged rearm2 seal",
    )
    if (
        seal_guards.get("fresh_710_series_seed_schedule") is not True
        or seal_guards.get("same_identity_resume_only") is not True
        or seal_guards.get("post_claim_reseeded") is not False
    ):
        raise ValueError("packaged rearm2 seal guards changed")
    _validate_rearm2_parity_binding(
        source,
        claim=claim,
        materialization=materialization,
        seal=seal,
    )
    return plan, claim, seal, materialization


def _smoke_receipt_path(target: Path) -> Path:
    return target / PREAUTHORIZE_SMOKE_NAME


def _read_bound_smoke_receipt(target: Path) -> tuple[dict[str, Any], str]:
    path = _smoke_receipt_path(target)
    receipt = _validate_preauthorize_smoke_receipt_unlocked(
        path,
        run_dir=target,
    )
    digest = spot_v1.sha256_file(path)
    if digest != spot_v1.canonical_sha256(receipt):
        raise ValueError("rearm2 preauthorize smoke receipt hash changed")
    return receipt, digest


def _base_global_spot_claim_payload(
    *,
    target: Path,
    manifest: Mapping[str, Any],
    root_claim: Mapping[str, Any],
    claimed_unix_ns: int,
) -> dict[str, Any]:
    """Call the frozen v1 constructor against its own exact key set."""

    current_keys = spot_v1._GLOBAL_SPOT_CLAIM_KEYS
    spot_v1._GLOBAL_SPOT_CLAIM_KEYS = _BASE_GLOBAL_SPOT_CLAIM_KEYS
    try:
        return _BASE_GLOBAL_SPOT_CLAIM_PAYLOAD(
            target=target,
            manifest=manifest,
            root_claim=root_claim,
            claimed_unix_ns=claimed_unix_ns,
        )
    finally:
        spot_v1._GLOBAL_SPOT_CLAIM_KEYS = current_keys


def _bound_global_spot_claim_payload(
    *,
    target: Path,
    manifest: Mapping[str, Any],
    root_claim: Mapping[str, Any],
    claimed_unix_ns: int,
) -> dict[str, Any]:
    """Extend the v1 payload with the immutable exhaustive-smoke binding."""

    _receipt, receipt_sha = _read_bound_smoke_receipt(target)
    payload = _base_global_spot_claim_payload(
        target=target,
        manifest=manifest,
        root_claim=root_claim,
        claimed_unix_ns=claimed_unix_ns,
    )
    payload["schema"] = GLOBAL_SPOT_CLAIM_SCHEMA
    payload["status"] = GLOBAL_SPOT_CLAIM_STATUS
    payload[PREAUTHORIZE_SMOKE_SHA_FIELD] = receipt_sha
    if set(payload) != _BOUND_GLOBAL_SPOT_CLAIM_KEYS:
        raise AssertionError("rearm2 global Spot claim schema changed")
    return payload


def _validate_bound_launch_authorization(
    run_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, authorization = _BASE_VALIDATE_LAUNCH_AUTHORIZATION(run_dir)
    target = Path(run_dir).resolve()
    _receipt, receipt_sha = _read_bound_smoke_receipt(target)
    global_claim = authorization.get("global_spot_claim")
    if (
        authorization.get(PREAUTHORIZE_SMOKE_SHA_FIELD) != receipt_sha
        or not isinstance(global_claim, Mapping)
        or global_claim.get(PREAUTHORIZE_SMOKE_SHA_FIELD) != receipt_sha
        or global_claim.get("schema") != GLOBAL_SPOT_CLAIM_SCHEMA
        or global_claim.get("status") != GLOBAL_SPOT_CLAIM_STATUS
    ):
        raise ValueError("rearm2 launch authorization smoke binding changed")
    return manifest, authorization


@contextmanager
def _rearm2_context(
    *,
    bind_smoke_receipt: bool = False,
) -> Iterator[tuple[ModuleType, ModuleType]]:
    """Activate rearm2 only for one serialized adapter operation."""

    with _CONTEXT_LOCK:
        plan, lock_open = _load_rearm_modules()
        previous = {
            "plan": spot_v1.lock_plan,
            "open": spot_v1.lock_open,
            "claim_name": spot_v1.GLOBAL_SPOT_CLAIM_NAME,
            "archive": spot_v1._validate_archive,
            "claim_keys": spot_v1._GLOBAL_SPOT_CLAIM_KEYS,
            "auth_keys": spot_v1._AUTHORIZATION_KEYS,
            "claim_payload": spot_v1._global_spot_claim_payload,
            "validate_auth": spot_v1.validate_launch_authorization,
            "r1_claim_name": rearm1_spot.GLOBAL_SPOT_CLAIM_NAME,
            "r1_scope": rearm1_spot.REARM1_PLAN_SCOPE,
            "r1_phase": rearm1_spot.REARM1_STARTUP_PACKAGE_PHASE,
        }
        spot_v1.lock_plan = plan
        spot_v1.lock_open = lock_open
        spot_v1.GLOBAL_SPOT_CLAIM_NAME = GLOBAL_SPOT_CLAIM_NAME
        spot_v1._validate_archive = _validate_rearm2_archive
        rearm1_spot.GLOBAL_SPOT_CLAIM_NAME = GLOBAL_SPOT_CLAIM_NAME
        rearm1_spot.REARM1_PLAN_SCOPE = REARM2_PLAN_SCOPE
        rearm1_spot.REARM1_STARTUP_PACKAGE_PHASE = (
            REARM2_STARTUP_PACKAGE_PHASE
        )
        if bind_smoke_receipt:
            spot_v1._GLOBAL_SPOT_CLAIM_KEYS = _BOUND_GLOBAL_SPOT_CLAIM_KEYS
            spot_v1._AUTHORIZATION_KEYS = _BOUND_AUTHORIZATION_KEYS
            spot_v1._global_spot_claim_payload = (
                _bound_global_spot_claim_payload
            )
            spot_v1.validate_launch_authorization = (
                _validate_bound_launch_authorization
            )
        try:
            yield plan, lock_open
        finally:
            spot_v1.validate_launch_authorization = previous["validate_auth"]
            spot_v1._global_spot_claim_payload = previous["claim_payload"]
            spot_v1._AUTHORIZATION_KEYS = previous["auth_keys"]
            spot_v1._GLOBAL_SPOT_CLAIM_KEYS = previous["claim_keys"]
            rearm1_spot.REARM1_STARTUP_PACKAGE_PHASE = previous["r1_phase"]
            rearm1_spot.REARM1_PLAN_SCOPE = previous["r1_scope"]
            rearm1_spot.GLOBAL_SPOT_CLAIM_NAME = previous["r1_claim_name"]
            spot_v1._validate_archive = previous["archive"]
            spot_v1.GLOBAL_SPOT_CLAIM_NAME = previous["claim_name"]
            spot_v1.lock_open = previous["open"]
            spot_v1.lock_plan = previous["plan"]


def _preview_global_claim(
    *,
    target: Path,
    manifest: Mapping[str, Any],
    root_claim: Mapping[str, Any],
    receipt_sha: str,
) -> dict[str, Any]:
    payload = _base_global_spot_claim_payload(
        target=target,
        manifest=manifest,
        root_claim=root_claim,
        claimed_unix_ns=1,
    )
    payload["schema"] = GLOBAL_SPOT_CLAIM_SCHEMA
    payload["status"] = GLOBAL_SPOT_CLAIM_STATUS
    payload[PREAUTHORIZE_SMOKE_SHA_FIELD] = receipt_sha
    if set(payload) != _BOUND_GLOBAL_SPOT_CLAIM_KEYS:
        raise AssertionError("rearm2 preview global Spot claim schema changed")
    return payload


def _preview_authorization(
    *,
    target: Path,
    manifest: Mapping[str, Any],
    preview_claim: Mapping[str, Any],
    receipt_sha: str,
) -> dict[str, Any]:
    qualification = manifest["tail_qualification"]
    value = {
        "schema": spot_v1.AUTHORIZATION_SCHEMA,
        "status": "explicit_one_shot_performance_lock_spot_authorization",
        "run_name": manifest["run_name"],
        "package_manifest_sha256": spot_v1.sha256_file(
            target / spot_v1.MANIFEST_NAME
        ),
        "source_sha256": manifest["source_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "plan_sha256": manifest["plan_sha256"],
        "tail_summary_sha256": qualification["summary_sha256"],
        "tail_validation_sha256": qualification["validation_sha256"],
        "run_contract_digest": manifest["run_contract_digest"],
        "launch_target": manifest["launch_target"],
        "cost_guard": manifest["cost_guard"],
        "authorized_job_ids": list(spot_v1.authorized_job_ids()),
        "logical_job_count": spot_v1.MAX_LOGICAL_JOBS,
        "global_spot_claim": dict(preview_claim),
        PREAUTHORIZE_SMOKE_SHA_FIELD: receipt_sha,
        "performance_development_only": False,
        "spot_execution_authorized": True,
        "performance_lock_authorized": True,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "authorized_unix_seconds": 1.0,
    }
    if set(value) != _BOUND_AUTHORIZATION_KEYS:
        raise AssertionError("rearm2 preview authorization schema changed")
    return value


def _smoke_receipt_body(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    verified_jobs: list[dict[str, str]],
    poisoned_root_count: int,
    plan: ModuleType,
) -> dict[str, Any]:
    value = {
        "schema": PREAUTHORIZE_SMOKE_SCHEMA,
        "status": PREAUTHORIZE_SMOKE_STATUS,
        "run_name": manifest["run_name"],
        "package_manifest_sha256": spot_v1.sha256_file(manifest_path),
        "source_sha256": manifest["source_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "plan_sha256": manifest["plan_sha256"],
        "run_contract_digest": manifest["run_contract_digest"],
        "candidate_variant": plan.CANDIDATE_VARIANT,
        "step6d_run_id": plan.RUN_ID,
        "schedule": plan.SCHEDULE,
        "root_schema": plan.ROOT_SCHEMA,
        "shard_manifest_schema": plan.SHARD_MANIFEST_SCHEMA,
        "done_schema": plan.DONE_SCHEMA,
        "global_spot_claim_name": GLOBAL_SPOT_CLAIM_NAME,
        "startup_package_phase": REARM2_STARTUP_PACKAGE_PHASE,
        "verified_job_manifests": verified_jobs,
        "job_manifest_aggregate_sha256": spot_v1.canonical_sha256(verified_jobs),
        "verified_job_ids": [row["job_id"] for row in verified_jobs],
        "verified_job_count": len(verified_jobs),
        "startup_invocation_count": len(verified_jobs),
        "actual_package_manifest_validated": True,
        "preview_authorization_validated": True,
        "full_no_root_read_startup_contract_executed": True,
        "write_once_receipt_required": True,
        "receipt_sha_bound_before_cloud_mutation": True,
        "signed_root_claim_strings_preserved": True,
        "temporary_poisoned_root_count": poisoned_root_count,
        "startup_root_member_read": False,
        "persistent_claim_or_authorization_written": False,
        "cloud_mutation_executed": False,
        "current_profile_changed": False,
    }
    if set(value) != PREAUTHORIZE_SMOKE_KEYS:
        raise AssertionError("rearm2 preauthorize smoke receipt schema changed")
    return value


def _preauthorize_smoke_unlocked(run_dir: str | Path) -> dict[str, Any]:
    """Execute all actual job verifiers without reading persistent roots."""

    target = Path(run_dir).resolve()
    manifest_path = target / spot_v1.MANIFEST_NAME
    source_path = target / spot_v1.SOURCE_NAME
    startup_path = target / spot_v1.STARTUP_NAME
    manifest = spot_v1._read_canonical(
        manifest_path, "rearm2 preauthorize package manifest"
    )
    for path, expected_sha, label in (
        (source_path, manifest.get("source_sha256"), "source"),
        (startup_path, manifest.get("startup_sha256"), "startup"),
    ):
        if (
            not path.is_file()
            or path.is_symlink()
            or spot_v1.sha256_file(path) != expected_sha
        ):
            raise ValueError(f"rearm2 preauthorize {label} binding changed")

    packaged_root_claim = spot_v1._read_packaged_open_claim(target, manifest)
    root_claim_path = Path(
        str(packaged_root_claim["global_claim_path"])
    ).resolve()
    actual_root_claim = spot_v1._read_canonical(
        root_claim_path, "rearm2 global root claim"
    )
    if (
        actual_root_claim != packaged_root_claim
        or spot_v1.canonical_sha256(actual_root_claim)
        != manifest["tail_qualification"]["open_claim_sha256"]
    ):
        raise ValueError("rearm2 global root claim changed before smoke")

    global_spot_claim_path = spot_v1._global_spot_claim_path(actual_root_claim)
    persistent_paths = (
        global_spot_claim_path,
        target / spot_v1.AUTHORIZATION_NAME,
        target / spot_v1.LAUNCH_CLAIM_NAME,
        target / spot_v1.LAUNCH_RESULT_NAME,
        target / spot_v1.RESUME_CLAIM_NAME,
        target / spot_v1.RESUME_RESULT_NAME,
        target / spot_v1.RESULT_OPEN_CLAIM_NAME,
    )
    existing = [str(path) for path in persistent_paths if path.exists()]
    if existing:
        raise ValueError(
            "rearm2 smoke must precede every Spot/authorization write: "
            f"{existing}"
        )
    protected_paths = (
        root_claim_path,
        manifest_path,
        source_path,
        startup_path,
        *persistent_paths,
    )
    before = rearm1_spot._canonical_file_snapshot(protected_paths)
    verified_jobs = rearm1_spot._validated_actual_job_manifests(
        target=target,
        manifest=manifest,
    )
    verifier = rearm1_spot._extract_precontent_verifier(startup_path)
    expected_output = (
        f"{REARM2_STARTUP_PACKAGE_PHASE}|{spot_v1.lock_plan.DONE_SCHEMA}"
    )

    with tempfile.TemporaryDirectory(
        prefix="rearm2-preauthorize-smoke-"
    ) as temporary:
        scratch = Path(temporary)
        poisoned_source = scratch / spot_v1.SOURCE_NAME
        preview_authorization_path = scratch / spot_v1.AUTHORIZATION_NAME
        poisoned_root_count = rearm1_spot._poison_temporary_root_payloads(
            source_path,
            poisoned_source,
        )
        receipt = _smoke_receipt_body(
            manifest=manifest,
            manifest_path=manifest_path,
            verified_jobs=verified_jobs,
            poisoned_root_count=poisoned_root_count,
            plan=spot_v1.lock_plan,
        )
        prospective_receipt_sha = spot_v1.canonical_sha256(receipt)
        preview_claim = _preview_global_claim(
            target=target,
            manifest=manifest,
            root_claim=actual_root_claim,
            receipt_sha=prospective_receipt_sha,
        )
        if (
            preview_claim["global_root_claim_path"]
            != actual_root_claim["global_claim_path"]
            or preview_claim["lock_output_directory"]
            != actual_root_claim["lock_output_directory"]
        ):
            raise AssertionError("rearm2 preview changed signed root-claim strings")
        preview_authorization = _preview_authorization(
            target=target,
            manifest=manifest,
            preview_claim=preview_claim,
            receipt_sha=prospective_receipt_sha,
        )
        preview_authorization_path.write_bytes(
            spot_v1.canonical_bytes(preview_authorization)
        )
        for job in verified_jobs:
            job_id = job["job_id"]
            job_path = target / "jobs" / f"{job_id}.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-",
                    str(poisoned_source),
                    str(manifest_path),
                    str(preview_authorization_path),
                    str(job_path),
                    job_id,
                    job["sha256"],
                ],
                input=verifier,
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )
            if (
                completed.returncode != 0
                or completed.stdout.strip() != expected_output
            ):
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise ValueError(
                    "rearm2 startup full no-root-read verifier rejected actual "
                    f"job {job_id}: returncode={completed.returncode}, "
                    f"detail={detail}"
                )
        if spot_v1.canonical_sha256(receipt) != prospective_receipt_sha:
            raise AssertionError("rearm2 smoke receipt changed during execution")

    after = rearm1_spot._canonical_file_snapshot(protected_paths)
    if after != before:
        raise AssertionError("rearm2 preauthorize smoke mutated persistent state")
    return receipt


def preauthorize_smoke(run_dir: str | Path) -> dict[str, Any]:
    with _rearm2_context():
        return _preauthorize_smoke_unlocked(run_dir)


def write_preauthorize_smoke_receipt(
    run_dir: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the exhaustive smoke, then write exactly one canonical receipt."""

    target = Path(run_dir).resolve()
    required_destination = _smoke_receipt_path(target)
    destination = (
        Path(output_path).resolve()
        if output_path is not None
        else required_destination
    )
    if destination != required_destination:
        raise ValueError(
            "rearm2 smoke receipt must use the package-bound fixed path"
        )
    if destination.exists():
        raise FileExistsError(
            f"rearm2 preauthorize smoke receipt already exists: {destination}"
        )
    with _rearm2_context():
        receipt = _preauthorize_smoke_unlocked(target)
        spot_v1._write_once(destination, receipt)
        if spot_v1.sha256_file(destination) != spot_v1.canonical_sha256(receipt):
            raise AssertionError("rearm2 written smoke receipt is not canonical")
        return receipt


def _validate_preauthorize_smoke_receipt_unlocked(
    receipt_path: str | Path,
    *,
    run_dir: str | Path,
) -> dict[str, Any]:
    target = Path(run_dir).resolve()
    receipt = spot_v1._read_canonical(
        receipt_path,
        "rearm2 actual-package preauthorize smoke receipt",
    )
    manifest_path = target / spot_v1.MANIFEST_NAME
    manifest = spot_v1._read_canonical(
        manifest_path,
        "rearm2 actual-package preauthorize manifest",
    )
    verified = rearm1_spot._validated_actual_job_manifests(
        target=target,
        manifest=manifest,
    )
    expected = _smoke_receipt_body(
        manifest=manifest,
        manifest_path=manifest_path,
        verified_jobs=verified,
        poisoned_root_count=int(
            receipt.get("temporary_poisoned_root_count", -1)
        ),
        plan=spot_v1.lock_plan,
    )
    if (
        receipt != expected
        or receipt.get("verified_job_count") != spot_v1.MAX_LOGICAL_JOBS
        or receipt.get("startup_invocation_count") != spot_v1.MAX_LOGICAL_JOBS
        or receipt.get("verified_job_ids")
        != list(spot_v1.authorized_job_ids())
        or receipt.get("temporary_poisoned_root_count") != 100
    ):
        raise ValueError("rearm2 actual-package smoke receipt changed")
    return receipt


def validate_preauthorize_smoke_receipt(
    receipt_path: str | Path,
    *,
    run_dir: str | Path,
) -> dict[str, Any]:
    with _rearm2_context():
        return _validate_preauthorize_smoke_receipt_unlocked(
            receipt_path,
            run_dir=run_dir,
        )


def package_performance_lock(
    *,
    output_dir: str | Path,
    run_name: str,
    lock_inputs: Any,
    startup_script: str | Path | None = None,
) -> dict[str, Any]:
    with _rearm2_context():
        return spot_v1.package_performance_lock(
            output_dir=output_dir,
            run_name=run_name,
            lock_inputs=lock_inputs,
            startup_script=startup_script,
        )


def validate_package(run_dir: str | Path) -> dict[str, Any]:
    with _rearm2_context():
        return spot_v1.validate_package(run_dir)


def _authorize_launch_unlocked(run_dir: str | Path) -> dict[str, Any]:
    target = Path(run_dir).resolve()
    _receipt, receipt_sha = _read_bound_smoke_receipt(target)
    manifest = spot_v1.validate_package(target)
    global_spot_claim = spot_v1._acquire_global_spot_claim(
        target=target,
        manifest=manifest,
    )
    qualification = manifest["tail_qualification"]
    authorization = {
        "schema": spot_v1.AUTHORIZATION_SCHEMA,
        "status": "explicit_one_shot_performance_lock_spot_authorization",
        "run_name": manifest["run_name"],
        "package_manifest_sha256": spot_v1.sha256_file(
            target / spot_v1.MANIFEST_NAME
        ),
        "source_sha256": manifest["source_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "plan_sha256": manifest["plan_sha256"],
        "tail_summary_sha256": qualification["summary_sha256"],
        "tail_validation_sha256": qualification["validation_sha256"],
        "run_contract_digest": manifest["run_contract_digest"],
        "launch_target": manifest["launch_target"],
        "cost_guard": manifest["cost_guard"],
        "authorized_job_ids": list(spot_v1.authorized_job_ids()),
        "logical_job_count": spot_v1.MAX_LOGICAL_JOBS,
        "global_spot_claim": global_spot_claim,
        PREAUTHORIZE_SMOKE_SHA_FIELD: receipt_sha,
        "performance_development_only": False,
        "spot_execution_authorized": True,
        "performance_lock_authorized": True,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "authorized_unix_seconds": time.time(),
    }
    if set(authorization) != _BOUND_AUTHORIZATION_KEYS:
        raise AssertionError("rearm2 launch authorization schema changed")
    spot_v1._write_once(target / spot_v1.AUTHORIZATION_NAME, authorization)
    _validate_bound_launch_authorization(target)
    return authorization


def authorize_launch(run_dir: str | Path) -> dict[str, Any]:
    """Require the write-once smoke receipt before creating either claim."""

    with _rearm2_context(bind_smoke_receipt=True):
        return _authorize_launch_unlocked(run_dir)


def validate_global_spot_claim(
    run_dir: str | Path,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    with _rearm2_context(bind_smoke_receipt=True):
        return spot_v1.validate_global_spot_claim(run_dir, manifest)


def validate_launch_authorization(
    run_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with _rearm2_context(bind_smoke_receipt=True):
        return spot_v1.validate_launch_authorization(run_dir)


def launch_jobs(
    *,
    run_dir: str | Path,
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    with _rearm2_context(bind_smoke_receipt=True):
        return spot_v1.launch_jobs(
            run_dir=run_dir,
            project=project,
            bucket=bucket,
        )


def validate_launch_chain(
    run_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    with _rearm2_context(bind_smoke_receipt=True):
        return spot_v1.validate_launch_chain(run_dir)


def preflight_resume(
    *,
    run_dir: str | Path,
    selected: Sequence[str],
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    with _rearm2_context(bind_smoke_receipt=True):
        return spot_v1.preflight_resume(
            run_dir=run_dir,
            selected=selected,
            project=project,
            bucket=bucket,
        )


def resume_jobs(
    *,
    run_dir: str | Path,
    selected: Sequence[str],
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    with _rearm2_context(bind_smoke_receipt=True):
        return spot_v1.resume_jobs(
            run_dir=run_dir,
            selected=selected,
            project=project,
            bucket=bucket,
        )


def validate_resume_chain(
    *,
    run_dir: str | Path,
    manifest: Mapping[str, Any] | None = None,
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    with _rearm2_context(bind_smoke_receipt=True):
        return spot_v1.validate_resume_chain(
            run_dir=run_dir,
            manifest=manifest,
            project=project,
            bucket=bucket,
        )


def cloud_status(
    *,
    run_dir: str | Path,
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    with _rearm2_context(bind_smoke_receipt=True):
        return spot_v1.cloud_status(
            run_dir=run_dir,
            project=project,
            bucket=bucket,
        )


def claim_results(
    *,
    run_dir: str | Path,
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    with _rearm2_context(bind_smoke_receipt=True):
        return spot_v1.claim_results(
            run_dir=run_dir,
            project=project,
            bucket=bucket,
        )


def validate_result_open_claim(
    *,
    run_dir: str | Path,
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
    verify_remote: bool = True,
) -> dict[str, Any]:
    with _rearm2_context(bind_smoke_receipt=True):
        return spot_v1.validate_result_open_claim(
            run_dir=run_dir,
            project=project,
            bucket=bucket,
            verify_remote=verify_remote,
        )


def receive_jobs(
    *,
    run_dir: str | Path,
    destination: str | Path,
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    with _rearm2_context(bind_smoke_receipt=True):
        return spot_v1.receive_jobs(
            run_dir=run_dir,
            destination=destination,
            project=project,
            bucket=bucket,
        )


def validate_received_directory(
    *,
    receive_dir: str | Path,
    run_dir: str | Path,
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    with _rearm2_context(bind_smoke_receipt=True):
        return spot_v1.validate_received_directory(
            receive_dir=receive_dir,
            run_dir=run_dir,
            project=project,
            bucket=bucket,
        )


def _parser() -> argparse.ArgumentParser:
    plan, lock_open = _load_rearm_modules()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def cloud_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--run-dir", type=Path, required=True)
        command.add_argument("--project", default=DEFAULT_PROJECT)
        command.add_argument("--bucket", default=DEFAULT_BUCKET)

    package = commands.add_parser("package")
    package.add_argument("--output-dir", type=Path, required=True)
    package.add_argument("--run-name", required=True)
    package.add_argument(
        "--repository-root",
        type=Path,
        default=lock_open.DEFAULT_REPOSITORY_ROOT,
    )
    package.add_argument(
        "--plan",
        type=Path,
        default=lock_open.DEFAULT_PRECONTENT_PLAN_PATH,
    )
    package.add_argument("--lock-output", type=Path, required=True)
    package.add_argument("--candidate-library", type=Path, required=True)
    package.add_argument("--reference-library", type=Path, required=True)
    package.add_argument("--feature-encoder", type=Path, required=True)
    package.add_argument("--startup-source", type=Path, required=True)
    package.add_argument(
        "--incident-receipt",
        type=Path,
        default=plan.DEFAULT_INCIDENT_RECEIPT_PATH,
    )
    package.add_argument("--old-v1-roots", type=Path, required=True)
    package.add_argument("--rearm1-roots", type=Path, required=True)
    package.add_argument(
        "--old-v1-global-claim",
        type=Path,
        default=lock_open.DEFAULT_OLD_V1_GLOBAL_CLAIM_PATH,
    )
    package.add_argument(
        "--rearm1-global-claim",
        type=Path,
        default=lock_open.DEFAULT_REARM1_GLOBAL_CLAIM_PATH,
    )
    package.add_argument(
        "--development-summary",
        type=Path,
        default=lock_open.DEFAULT_DEVELOPMENT_MERGE_DIR / "summary.json",
    )
    package.add_argument(
        "--development-validation",
        type=Path,
        default=lock_open.DEFAULT_DEVELOPMENT_MERGE_DIR / "validation.json",
    )
    package.add_argument(
        "--development-roots",
        type=Path,
        default=lock_open.DEFAULT_DEVELOPMENT_ROOT_DIR,
    )
    package.add_argument(
        "--global-claim",
        type=Path,
        default=lock_open.DEFAULT_GLOBAL_CLAIM_PATH,
    )
    preauthorize = commands.add_parser("preauthorize-smoke")
    preauthorize.add_argument("--run-dir", type=Path, required=True)
    write = commands.add_parser("write-preauthorize-smoke-receipt")
    write.add_argument("--run-dir", type=Path, required=True)
    write.add_argument("--output", type=Path)
    validate = commands.add_parser("validate-preauthorize-smoke-receipt")
    validate.add_argument("--run-dir", type=Path, required=True)
    validate.add_argument("--receipt", type=Path, required=True)
    validate_package_command = commands.add_parser("validate-package")
    validate_package_command.add_argument("--run-dir", type=Path, required=True)
    validate_global = commands.add_parser("validate-global-spot-claim")
    validate_global.add_argument("--run-dir", type=Path, required=True)
    authorize = commands.add_parser("authorize")
    authorize.add_argument("--run-dir", type=Path, required=True)
    validate_authorization = commands.add_parser("validate-authorization")
    validate_authorization.add_argument("--run-dir", type=Path, required=True)
    launch = commands.add_parser("launch")
    cloud_arguments(launch)
    launch_chain = commands.add_parser("validate-launch-chain")
    launch_chain.add_argument("--run-dir", type=Path, required=True)
    preflight = commands.add_parser("preflight-resume")
    cloud_arguments(preflight)
    preflight.add_argument("--job", action="append", required=True)
    resume = commands.add_parser("resume")
    cloud_arguments(resume)
    resume.add_argument("--job", action="append", required=True)
    validate_resume = commands.add_parser("validate-resume-chain")
    cloud_arguments(validate_resume)
    status = commands.add_parser("status")
    cloud_arguments(status)
    claim = commands.add_parser("claim-results")
    cloud_arguments(claim)
    validate_claim = commands.add_parser("validate-result-claim")
    cloud_arguments(validate_claim)
    validate_claim.add_argument("--no-verify-remote", action="store_true")
    receive = commands.add_parser("receive")
    cloud_arguments(receive)
    receive.add_argument("--destination", type=Path, required=True)
    validate_receive = commands.add_parser("validate-received")
    cloud_arguments(validate_receive)
    validate_receive.add_argument("--receive-dir", type=Path, required=True)
    return parser


def _build_cli_inputs(args: argparse.Namespace) -> Any:
    _, lock_open = _load_rearm_modules()
    return lock_open.PerformanceLockRearm2Inputs(
        repository_root=args.repository_root,
        plan_path=args.plan,
        lock_output_directory=args.lock_output,
        candidate_library=args.candidate_library,
        reference_library=args.reference_library,
        feature_encoder=args.feature_encoder,
        startup_source=args.startup_source,
        incident_receipt_path=args.incident_receipt,
        old_v1_root_directory=args.old_v1_roots,
        rearm1_root_directory=args.rearm1_roots,
        development_summary_path=args.development_summary,
        development_validation_path=args.development_validation,
        development_root_directory=args.development_roots,
        old_v1_global_claim_path=args.old_v1_global_claim,
        rearm1_global_claim_path=args.rearm1_global_claim,
        global_claim_path=args.global_claim,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "package":
        result: Any = package_performance_lock(
            output_dir=args.output_dir,
            run_name=args.run_name,
            lock_inputs=_build_cli_inputs(args),
        )
    elif args.command == "preauthorize-smoke":
        result: Any = preauthorize_smoke(args.run_dir)
    elif args.command == "write-preauthorize-smoke-receipt":
        result = write_preauthorize_smoke_receipt(
            args.run_dir,
            output_path=args.output,
        )
    elif args.command == "validate-preauthorize-smoke-receipt":
        result = validate_preauthorize_smoke_receipt(
            args.receipt,
            run_dir=args.run_dir,
        )
    elif args.command == "validate-package":
        result = validate_package(args.run_dir)
    elif args.command == "validate-global-spot-claim":
        result = validate_global_spot_claim(args.run_dir)
    elif args.command == "authorize":
        result = authorize_launch(args.run_dir)
    elif args.command == "validate-authorization":
        result = validate_launch_authorization(args.run_dir)
    elif args.command == "launch":
        result = launch_jobs(
            run_dir=args.run_dir,
            project=args.project,
            bucket=args.bucket,
        )
    elif args.command == "validate-launch-chain":
        result = validate_launch_chain(args.run_dir)
    elif args.command == "preflight-resume":
        result = preflight_resume(
            run_dir=args.run_dir,
            selected=args.job,
            project=args.project,
            bucket=args.bucket,
        )
    elif args.command == "resume":
        result = resume_jobs(
            run_dir=args.run_dir,
            selected=args.job,
            project=args.project,
            bucket=args.bucket,
        )
    elif args.command == "validate-resume-chain":
        result = validate_resume_chain(
            run_dir=args.run_dir,
            project=args.project,
            bucket=args.bucket,
        )
    elif args.command == "status":
        result = cloud_status(
            run_dir=args.run_dir,
            project=args.project,
            bucket=args.bucket,
        )
    elif args.command == "claim-results":
        result = claim_results(
            run_dir=args.run_dir,
            project=args.project,
            bucket=args.bucket,
        )
    elif args.command == "validate-result-claim":
        result = validate_result_open_claim(
            run_dir=args.run_dir,
            project=args.project,
            bucket=args.bucket,
            verify_remote=not args.no_verify_remote,
        )
    elif args.command == "receive":
        result = receive_jobs(
            run_dir=args.run_dir,
            destination=args.destination,
            project=args.project,
            bucket=args.bucket,
        )
    else:
        result = validate_received_directory(
            receive_dir=args.receive_dir,
            run_dir=args.run_dir,
            project=args.project,
            bucket=args.bucket,
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "GLOBAL_SPOT_CLAIM_NAME",
    "GLOBAL_SPOT_CLAIM_SCHEMA",
    "GLOBAL_SPOT_CLAIM_STATUS",
    "PREAUTHORIZE_SMOKE_NAME",
    "PREAUTHORIZE_SMOKE_SCHEMA",
    "PREAUTHORIZE_SMOKE_STATUS",
    "REARM2_PLAN_SCOPE",
    "REARM2_STARTUP_PACKAGE_PHASE",
    "authorize_launch",
    "claim_results",
    "cloud_status",
    "launch_jobs",
    "main",
    "package_performance_lock",
    "preauthorize_smoke",
    "preflight_resume",
    "receive_jobs",
    "resume_jobs",
    "validate_global_spot_claim",
    "validate_launch_authorization",
    "validate_launch_chain",
    "validate_package",
    "validate_preauthorize_smoke_receipt",
    "validate_received_directory",
    "validate_result_open_claim",
    "validate_resume_chain",
    "write_preauthorize_smoke_receipt",
]
