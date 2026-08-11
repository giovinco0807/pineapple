"""Scoped Spot lifecycle adapter for the fresh Candidate02 rearm1 lock.

The audited v1 lifecycle remains the sole implementation of packaging,
authorization, launch, resume, status, result claiming, and receive.  This
module activates the rearm1 plan/open producers only while one adapter call is
running, under a process-wide re-entrant lock, and restores every v1 global in
``finally``.  Importing this module never changes the v1 lifecycle.

``preauthorize_smoke`` is deliberately local and non-persistent.  It builds an
in-memory preview of the global Spot claim, preserves the exact signed lexical
paths from the root claim, poisons only temporary copies of root payloads, and
runs the exact pre-content Python verifier embedded in the packaged startup
script.  No global Spot claim, launch authorization, cloud resource, or result
artifact is created by the smoke.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import zipfile
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Mapping, Sequence

from . import hu_m31_t3_step6d_performance_lock_spot_v1 as spot_v1


PREAUTHORIZE_SMOKE_SCHEMA = (
    "hu_m31_t3_step6d_performance_lock_actual_package_preauthorize_smoke_v2"
)
PREAUTHORIZE_SMOKE_STATUS = (
    "actual_package_manifest_authorization_all_20_jobs_full_no_root_read_"
    "startup_contract_passed_before_spot_authorization"
)
PREAUTHORIZE_SMOKE_NAME = "ACTUAL_PACKAGE_PREAUTHORIZE_SMOKE.json"
_PREAUTHORIZE_SMOKE_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "package_manifest_sha256",
        "source_sha256",
        "startup_sha256",
        "plan_sha256",
        "run_contract_digest",
        "candidate_variant",
        "step6d_run_id",
        "schedule",
        "root_schema",
        "shard_manifest_schema",
        "done_schema",
        "global_spot_claim_name",
        "startup_package_phase",
        "verified_job_manifests",
        "job_manifest_aggregate_sha256",
        "verified_job_ids",
        "verified_job_count",
        "startup_invocation_count",
        "actual_package_manifest_validated",
        "preview_authorization_validated",
        "full_no_root_read_startup_contract_executed",
        "write_once_receipt_required",
        "receipt_sha_bound_before_cloud_mutation",
        "signed_root_claim_strings_preserved",
        "temporary_poisoned_root_count",
        "startup_root_member_read",
        "persistent_claim_or_authorization_written",
        "cloud_mutation_executed",
        "current_profile_changed",
    }
)
GLOBAL_SPOT_CLAIM_NAME = "GLOBAL_PERFORMANCE_LOCK_REARM1_SPOT_CLAIM.json"
REARM1_PLAN_SCOPE = "performance_lock_rearm1_fresh_roots_only"
REARM1_STARTUP_PACKAGE_PHASE = "lock_rearm1"
REARM1_PARITY_VALIDATOR_PACKAGE_PATH = (
    "src/ofc_regular/"
    "verify_hu_m31_t3_feature_encoder_platform_parity_rearm1.py"
)
REARM1_PARITY_VALIDATOR_SCHEMA = (
    "hu_m31_t3_feature_encoder_rearm1_chain_validator_v1"
)
REARM1_PARITY_VALIDATOR_STATUS = (
    "verified_rearm1_v2_chain_and_fresh_no_reuse_guards_before_packaging"
)
EXPECTED_REARM1_PARITY_VALIDATOR_SHA256 = (
    "487451a6d593e44a37236dfd9e8a3b7850a4edf1c733b50cd87ccc5aa07a29b1"
)
EXPECTED_REARM1_PARITY_VALIDATOR_BYTES = 14_387

DEFAULT_PROJECT = spot_v1.DEFAULT_PROJECT
DEFAULT_BUCKET = spot_v1.DEFAULT_BUCKET
DEFAULT_REGION = spot_v1.DEFAULT_REGION
DEFAULT_ZONES = spot_v1.DEFAULT_ZONES

_PLAN_MODULE = "ofc_regular.hu_m31_t3_step6d_candidate02_performance_lock_rearm1_plan"
_OPEN_MODULE = "ofc_regular.hu_m31_t3_step6d_performance_lock_rearm1_open"
_CONTEXT_LOCK = threading.RLock()
_PRECONTENT_MARKER = "PRE-CONTENT ORDERING"
_HEREDOC_PATTERN = re.compile(r"<<'PY'\r?\n(.*?)\r?\nPY", flags=re.DOTALL)
_LOCAL_ZIP_HEADER = struct.Struct("<IHHHHHIIIHH")
_LOCAL_ZIP_SIGNATURE = 0x04034B50
_BASE_VALIDATE_ARCHIVE = spot_v1._validate_archive


def _validate_rearm_parity_binding(
    source: Path,
    *,
    claim: Mapping[str, Any],
    materialization: Mapping[str, Any],
    seal: Mapping[str, Any],
) -> None:
    """Require the packaged receipt to bind the audited rearm1 validator."""

    with zipfile.ZipFile(source) as archive:
        raw_receipt = archive.read(
            spot_v1.MATERIALIZER_PARITY_RECEIPT_PACKAGE_PATH
        )
        receipt = json.loads(raw_receipt.decode("utf-8"))
        validator_bytes = archive.read(REARM1_PARITY_VALIDATOR_PACKAGE_PATH)
    if (
        not isinstance(receipt, dict)
        or raw_receipt != spot_v1.canonical_bytes(receipt)
    ):
        raise ValueError("packaged rearm1 parity receipt is not canonical")

    validator_sha = hashlib.sha256(validator_bytes).hexdigest()
    chain = receipt.get("lock_chain")
    if (
        validator_sha != EXPECTED_REARM1_PARITY_VALIDATOR_SHA256
        or len(validator_bytes) != EXPECTED_REARM1_PARITY_VALIDATOR_BYTES
        or not isinstance(chain, Mapping)
        or chain.get("contract") != "performance_lock_rearm1_v2"
        or chain.get("validator_schema") != REARM1_PARITY_VALIDATOR_SCHEMA
        or chain.get("validator_status") != REARM1_PARITY_VALIDATOR_STATUS
        or chain.get("validator_bytes")
        != EXPECTED_REARM1_PARITY_VALIDATOR_BYTES
        or chain.get("validator_sha256")
        != EXPECTED_REARM1_PARITY_VALIDATOR_SHA256
        or chain.get("global_claim_sha256")
        != spot_v1.canonical_sha256(claim)
        or chain.get("materialization_sha256")
        != spot_v1.canonical_sha256(materialization)
        or chain.get("seal_sha256") != spot_v1.canonical_sha256(seal)
        or chain.get("old_v1_global_claim_sha256")
        != spot_v1.lock_open.OLD_V1_GLOBAL_CLAIM_SHA256
        or chain.get("old_v1_seal_sha256")
        != spot_v1.lock_open.OLD_V1_SEAL_SHA256
        or chain.get("startup_failure_receipt_sha256")
        != spot_v1.lock_open.STARTUP_FAILURE_RECEIPT_SHA256
        or chain.get("old_v1_attempt1_reused") is not False
        or chain.get("old_v1_root_reused") is not False
        or chain.get("fresh_recovery_seed_schedule") is not True
        or chain.get("reseeded") is not False
        or chain.get("current_profile_changed") is not False
    ):
        raise ValueError("packaged rearm1 parity validator binding changed")


def _load_rearm_modules() -> tuple[ModuleType, ModuleType]:
    """Load producer-owned rearm modules without import-time side effects."""

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
        "MATERIALIZATION_SCHEMA",
        "MATERIALIZATION_STATUS",
        "SEAL_SCHEMA",
        "PerformanceLockInputs",
        "OLD_V1_GLOBAL_CLAIM_SHA256",
        "OLD_V1_SEAL_SHA256",
        "STARTUP_FAILURE_RECEIPT_SHA256",
        "DEFAULT_GLOBAL_CLAIM_PATH",
        "DEFAULT_PRECONTENT_PLAN_PATH",
        "validate_open_claim",
        "validate_root_seal",
    )
    missing_plan = [name for name in required_plan if not hasattr(plan, name)]
    missing_open = [name for name in required_open if not hasattr(lock_open, name)]
    if missing_plan or missing_open:
        raise RuntimeError(
            "rearm1 Spot producer API is incomplete: "
            f"plan={missing_plan}, open={missing_open}"
        )
    return plan, lock_open


def _validate_rearm_archive(
    source: Path, entries: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Replay the v1 archive validator, then require rearm-only no-reuse proof."""

    plan, claim, seal, materialization = _BASE_VALIDATE_ARCHIVE(source, entries)
    old_comparison = seal.get("old_performance_lock_comparison")
    guards = seal.get("rearm_guards")
    expected_old_keys = {
        "old_v1_global_claim_sha256",
        "old_v1_seal_sha256",
        "old_v1_root_count",
        "rearm1_fingerprint_overlap_count",
        "rearm1_root_hash_overlap_count",
        "rearm1_seed_overlap_count",
        "old_v1_attempt1_reused",
        "old_v1_root_reused",
    }
    expected_guard_keys = {
        "incident_receipt_sha256",
        "fresh_700_series_seed_schedule",
        "same_identity_resume_only",
        "old_v1_attempt1_reused",
        "old_v1_root_reused",
        "post_claim_reseeded",
    }
    if (
        materialization.get("schema") != spot_v1.lock_open.MATERIALIZATION_SCHEMA
        or materialization.get("status") != spot_v1.lock_open.MATERIALIZATION_STATUS
        or materialization.get("fresh_recovery_seed_schedule") is not True
        or materialization.get("old_v1_attempt1_reused") is not False
        or materialization.get("old_v1_root_reused") is not False
        or materialization.get("reseeded") is not False
        or not isinstance(old_comparison, Mapping)
        or set(old_comparison) != expected_old_keys
        or old_comparison.get("old_v1_global_claim_sha256")
        != spot_v1.lock_open.OLD_V1_GLOBAL_CLAIM_SHA256
        or old_comparison.get("old_v1_seal_sha256")
        != spot_v1.lock_open.OLD_V1_SEAL_SHA256
        or old_comparison.get("old_v1_root_count") != 100
        or any(
            old_comparison.get(field) != 0
            for field in (
                "rearm1_fingerprint_overlap_count",
                "rearm1_root_hash_overlap_count",
                "rearm1_seed_overlap_count",
            )
        )
        or old_comparison.get("old_v1_attempt1_reused") is not False
        or old_comparison.get("old_v1_root_reused") is not False
        or not isinstance(guards, Mapping)
        or set(guards) != expected_guard_keys
        or guards.get("incident_receipt_sha256")
        != spot_v1.lock_open.STARTUP_FAILURE_RECEIPT_SHA256
        or guards.get("fresh_700_series_seed_schedule") is not True
        or guards.get("same_identity_resume_only") is not True
        or guards.get("old_v1_attempt1_reused") is not False
        or guards.get("old_v1_root_reused") is not False
        or guards.get("post_claim_reseeded") is not False
    ):
        raise ValueError("packaged rearm1 no-reuse proof changed")
    _validate_rearm_parity_binding(
        source,
        claim=claim,
        materialization=materialization,
        seal=seal,
    )
    return plan, claim, seal, materialization


@contextmanager
def _rearm1_context() -> Iterator[tuple[ModuleType, ModuleType]]:
    """Temporarily activate rearm producers and always restore v1 globals."""

    with _CONTEXT_LOCK:
        plan, lock_open = _load_rearm_modules()
        previous_plan = spot_v1.lock_plan
        previous_open = spot_v1.lock_open
        previous_claim_name = spot_v1.GLOBAL_SPOT_CLAIM_NAME
        previous_validate_archive = spot_v1._validate_archive
        spot_v1.lock_plan = plan
        spot_v1.lock_open = lock_open
        spot_v1.GLOBAL_SPOT_CLAIM_NAME = GLOBAL_SPOT_CLAIM_NAME
        spot_v1._validate_archive = _validate_rearm_archive
        try:
            yield plan, lock_open
        finally:
            spot_v1._validate_archive = previous_validate_archive
            spot_v1.GLOBAL_SPOT_CLAIM_NAME = previous_claim_name
            spot_v1.lock_open = previous_open
            spot_v1.lock_plan = previous_plan


def _extract_precontent_verifier(startup_path: Path) -> str:
    if not startup_path.is_file() or startup_path.is_symlink():
        raise ValueError("rearm1 startup script is missing or unsafe")
    source = startup_path.read_text(encoding="utf-8")
    matches = [
        block
        for block in _HEREDOC_PATTERN.findall(source)
        if _PRECONTENT_MARKER in block
    ]
    if len(matches) != 1:
        raise ValueError("rearm1 startup pre-content verifier is not unique")
    verifier = matches[0]
    if (
        "zipfile.ZipFile" not in verifier
        or "archive.read(ROOT_PREFIX" in verifier
        or "Do not archive.read() a root here." not in verifier
    ):
        raise ValueError("rearm1 startup pre-content root-read guard changed")
    return verifier


def _poison_temporary_root_payloads(source: Path, destination: Path) -> int:
    """Corrupt temporary root payload bytes while preserving the ZIP directory."""

    shutil.copy2(source, destination)
    with zipfile.ZipFile(destination) as archive:
        roots = [
            info
            for info in archive.infolist()
            if info.filename.startswith(f"{spot_v1.ROOT_PACKAGE_DIR}/")
        ]
    if not roots:
        raise ValueError("rearm1 package has no root members to guard")
    with destination.open("r+b") as handle:
        for info in roots:
            handle.seek(info.header_offset)
            header = handle.read(_LOCAL_ZIP_HEADER.size)
            if len(header) != _LOCAL_ZIP_HEADER.size:
                raise ValueError("rearm1 temporary ZIP root header is truncated")
            fields = _LOCAL_ZIP_HEADER.unpack(header)
            if fields[0] != _LOCAL_ZIP_SIGNATURE or info.compress_size <= 0:
                raise ValueError("rearm1 temporary ZIP root header changed")
            filename_bytes = fields[-2]
            extra_bytes = fields[-1]
            payload_offset = (
                info.header_offset
                + _LOCAL_ZIP_HEADER.size
                + filename_bytes
                + extra_bytes
            )
            handle.seek(payload_offset)
            original = handle.read(1)
            if len(original) != 1:
                raise ValueError("rearm1 temporary ZIP root payload is empty")
            handle.seek(payload_offset)
            handle.write(bytes((original[0] ^ 0xFF,)))
    return len(roots)


def _preview_authorization(
    *,
    target: Path,
    manifest: Mapping[str, Any],
    preview_claim: Mapping[str, Any],
) -> dict[str, Any]:
    qualification = manifest["tail_qualification"]
    value = {
        "schema": spot_v1.AUTHORIZATION_SCHEMA,
        "status": "explicit_one_shot_performance_lock_spot_authorization",
        "run_name": manifest["run_name"],
        "package_manifest_sha256": spot_v1.sha256_file(target / spot_v1.MANIFEST_NAME),
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
    if set(value) != spot_v1._AUTHORIZATION_KEYS:
        raise AssertionError("rearm1 preview authorization schema changed")
    return value


def _canonical_file_snapshot(paths: Sequence[Path]) -> dict[str, str | None]:
    return {
        str(path): (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
        for path in paths
    }


def _validated_actual_job_manifests(
    *,
    target: Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Validate every external job file with the producer-owned runner."""

    plan = spot_v1.lock_plan
    contract_raw = manifest.get("run_contract")
    if not isinstance(contract_raw, Mapping):
        raise ValueError("rearm1 actual-package run contract is missing")
    contract = spot_v1.runner.validate_run_contract(contract_raw)
    if (
        spot_v1.canonical_sha256(contract)
        != manifest.get("run_contract_digest")
        or spot_v1.runner.contract_variant(contract) != plan.CANDIDATE_VARIANT
        or contract.get("step6d_run_id") != plan.RUN_ID
        or contract.get("schedule") != plan.SCHEDULE
    ):
        raise ValueError("rearm1 producer-owned recovery constants changed")

    records = manifest.get("job_manifests")
    expected_ids = list(spot_v1.authorized_job_ids())
    if (
        not isinstance(records, list)
        or len(records) != len(expected_ids)
        or [row.get("job_id") for row in records if isinstance(row, Mapping)]
        != expected_ids
    ):
        raise ValueError("rearm1 actual-package job record set changed")

    verified: list[dict[str, str]] = []
    for record, job_id in zip(records, expected_ids, strict=True):
        if not isinstance(record, Mapping) or set(record) != spot_v1._JOB_RECORD_KEYS:
            raise ValueError(f"rearm1 actual-package job record changed: {job_id}")
        relative = record.get("path")
        if relative != f"jobs/{job_id}.json":
            raise ValueError(f"rearm1 actual-package job path changed: {job_id}")
        job_path = (target / str(relative)).resolve()
        if (
            not job_path.is_relative_to(target)
            or not job_path.is_file()
            or job_path.is_symlink()
        ):
            raise ValueError(f"rearm1 actual-package job file is unsafe: {job_id}")
        job_sha256 = spot_v1.sha256_file(job_path)
        if (
            record.get("sha256") != job_sha256
            or record.get("bytes") != job_path.stat().st_size
        ):
            raise ValueError(f"rearm1 actual-package job bytes changed: {job_id}")
        job = spot_v1._read_canonical(
            job_path, f"rearm1 actual-package job {job_id}"
        )
        validated = spot_v1.runner.validate_shard_manifest(job)
        expected = spot_v1.runner.build_shard_manifest(
            run_contract=contract,
            source_role=str(record.get("source_role")),
            work_hand_indices=record.get("work_hand_indices", []),
        )
        if (
            validated != expected
            or validated.get("schema") != plan.SHARD_MANIFEST_SCHEMA
            or validated.get("run_contract") != contract
            or validated.get("run_contract_digest")
            != manifest.get("run_contract_digest")
            or validated.get("source_role") != record.get("source_role")
            or validated.get("work_hand_indices")
            != record.get("work_hand_indices")
            or record.get("output_prefix") != f"jobs/{job_id}"
        ):
            raise ValueError(
                f"rearm1 actual-package producer job contract changed: {job_id}"
            )
        verified.append({"job_id": job_id, "sha256": job_sha256})
    return verified


def _preauthorize_smoke_unlocked(run_dir: str | Path) -> dict[str, Any]:
    target = Path(run_dir).resolve()
    manifest_path = target / spot_v1.MANIFEST_NAME
    source_path = target / spot_v1.SOURCE_NAME
    startup_path = target / spot_v1.STARTUP_NAME
    manifest = spot_v1._read_canonical(
        manifest_path, "rearm1 preauthorize package manifest"
    )
    for path, expected_sha256, label in (
        (source_path, manifest.get("source_sha256"), "source"),
        (startup_path, manifest.get("startup_sha256"), "startup"),
    ):
        if (
            not path.is_file()
            or path.is_symlink()
            or spot_v1.sha256_file(path) != expected_sha256
        ):
            raise ValueError(f"rearm1 preauthorize {label} binding changed")

    packaged_root_claim = spot_v1._read_packaged_open_claim(target, manifest)
    root_claim_path = Path(str(packaged_root_claim["global_claim_path"])).resolve()
    actual_root_claim = spot_v1._read_canonical(
        root_claim_path, "rearm1 global root claim"
    )
    if (
        actual_root_claim != packaged_root_claim
        or spot_v1.canonical_sha256(actual_root_claim)
        != manifest["tail_qualification"]["open_claim_sha256"]
    ):
        raise ValueError("rearm1 global root claim changed before smoke")

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
            "rearm1 preauthorize smoke must precede every Spot/authorization "
            f"write: {existing}"
        )
    protected_paths = (
        root_claim_path,
        manifest_path,
        source_path,
        startup_path,
        *persistent_paths,
    )
    before = _canonical_file_snapshot(protected_paths)

    preview_claim = spot_v1._global_spot_claim_payload(
        target=target,
        manifest=manifest,
        root_claim=actual_root_claim,
        claimed_unix_ns=1,
    )
    if (
        preview_claim["global_root_claim_path"]
        != actual_root_claim["global_claim_path"]
        or preview_claim["lock_output_directory"]
        != actual_root_claim["lock_output_directory"]
    ):
        raise AssertionError("rearm1 preview changed signed root-claim strings")
    preview_authorization = _preview_authorization(
        target=target,
        manifest=manifest,
        preview_claim=preview_claim,
    )
    if spot_v1.lock_plan.PLAN_SCOPE != REARM1_PLAN_SCOPE:
        raise ValueError("rearm1 startup package phase scope changed")
    verified_jobs = _validated_actual_job_manifests(
        target=target,
        manifest=manifest,
    )
    verifier = _extract_precontent_verifier(startup_path)
    expected_startup_output = (
        f"{REARM1_STARTUP_PACKAGE_PHASE}|{spot_v1.lock_plan.DONE_SCHEMA}"
    )

    with tempfile.TemporaryDirectory(prefix="rearm1-preauthorize-smoke-") as temporary:
        scratch = Path(temporary)
        poisoned_source = scratch / spot_v1.SOURCE_NAME
        preview_authorization_path = scratch / spot_v1.AUTHORIZATION_NAME
        poisoned_root_count = _poison_temporary_root_payloads(
            source_path, poisoned_source
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
                or completed.stdout.strip() != expected_startup_output
            ):
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise ValueError(
                    "rearm1 startup full no-root-read verifier rejected actual "
                    f"job {job_id}: returncode={completed.returncode}, "
                    f"detail={detail}"
                )
    after = _canonical_file_snapshot(protected_paths)
    if after != before:
        raise AssertionError("rearm1 preauthorize smoke mutated persistent state")
    return {
        "schema": PREAUTHORIZE_SMOKE_SCHEMA,
        "status": PREAUTHORIZE_SMOKE_STATUS,
        "run_name": manifest["run_name"],
        "package_manifest_sha256": spot_v1.sha256_file(manifest_path),
        "source_sha256": manifest["source_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "plan_sha256": manifest["plan_sha256"],
        "run_contract_digest": manifest["run_contract_digest"],
        "candidate_variant": spot_v1.lock_plan.CANDIDATE_VARIANT,
        "step6d_run_id": spot_v1.lock_plan.RUN_ID,
        "schedule": spot_v1.lock_plan.SCHEDULE,
        "root_schema": spot_v1.lock_plan.ROOT_SCHEMA,
        "shard_manifest_schema": spot_v1.lock_plan.SHARD_MANIFEST_SCHEMA,
        "done_schema": spot_v1.lock_plan.DONE_SCHEMA,
        "global_spot_claim_name": GLOBAL_SPOT_CLAIM_NAME,
        "startup_package_phase": REARM1_STARTUP_PACKAGE_PHASE,
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


def preauthorize_smoke(run_dir: str | Path) -> dict[str, Any]:
    with _rearm1_context():
        return _preauthorize_smoke_unlocked(run_dir)


def write_preauthorize_smoke_receipt(
    run_dir: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Execute the exhaustive smoke and persist one canonical write-once receipt."""

    target = Path(run_dir).resolve()
    receipt = preauthorize_smoke(target)
    destination = (
        Path(output_path).resolve()
        if output_path is not None
        else target / PREAUTHORIZE_SMOKE_NAME
    )
    spot_v1._write_once(destination, receipt)
    return receipt


def _validate_preauthorize_smoke_receipt_unlocked(
    receipt_path: str | Path,
    *,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Replay immutable package/job bindings without reading root payloads."""

    target = Path(run_dir).resolve()
    receipt = spot_v1._read_canonical(
        receipt_path, "actual-package preauthorize smoke receipt"
    )
    manifest_path = target / spot_v1.MANIFEST_NAME
    manifest = spot_v1._read_canonical(
        manifest_path, "actual-package preauthorize manifest"
    )
    verified = _validated_actual_job_manifests(target=target, manifest=manifest)
    expected_ids = list(spot_v1.authorized_job_ids())
    if (
        set(receipt) != _PREAUTHORIZE_SMOKE_KEYS
        or receipt.get("schema") != PREAUTHORIZE_SMOKE_SCHEMA
        or receipt.get("status") != PREAUTHORIZE_SMOKE_STATUS
        or receipt.get("run_name") != manifest.get("run_name")
        or receipt.get("package_manifest_sha256")
        != spot_v1.sha256_file(manifest_path)
        or receipt.get("source_sha256") != manifest.get("source_sha256")
        or receipt.get("startup_sha256") != manifest.get("startup_sha256")
        or receipt.get("plan_sha256") != manifest.get("plan_sha256")
        or receipt.get("run_contract_digest")
        != manifest.get("run_contract_digest")
        or receipt.get("candidate_variant")
        != spot_v1.lock_plan.CANDIDATE_VARIANT
        or receipt.get("step6d_run_id") != spot_v1.lock_plan.RUN_ID
        or receipt.get("schedule") != spot_v1.lock_plan.SCHEDULE
        or receipt.get("root_schema") != spot_v1.lock_plan.ROOT_SCHEMA
        or receipt.get("shard_manifest_schema")
        != spot_v1.lock_plan.SHARD_MANIFEST_SCHEMA
        or receipt.get("done_schema") != spot_v1.lock_plan.DONE_SCHEMA
        or receipt.get("verified_job_manifests") != verified
        or receipt.get("job_manifest_aggregate_sha256")
        != spot_v1.canonical_sha256(verified)
        or receipt.get("verified_job_ids") != expected_ids
        or receipt.get("verified_job_count") != 20
        or receipt.get("startup_invocation_count") != 20
        or receipt.get("actual_package_manifest_validated") is not True
        or receipt.get("preview_authorization_validated") is not True
        or receipt.get("full_no_root_read_startup_contract_executed") is not True
        or receipt.get("write_once_receipt_required") is not True
        or receipt.get("receipt_sha_bound_before_cloud_mutation") is not True
        or receipt.get("signed_root_claim_strings_preserved") is not True
        or receipt.get("temporary_poisoned_root_count") != 100
        or receipt.get("startup_root_member_read") is not False
        or receipt.get("persistent_claim_or_authorization_written") is not False
        or receipt.get("cloud_mutation_executed") is not False
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("actual-package preauthorize smoke receipt changed")
    return receipt


def validate_preauthorize_smoke_receipt(
    receipt_path: str | Path,
    *,
    run_dir: str | Path,
) -> dict[str, Any]:
    with _rearm1_context():
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
    with _rearm1_context():
        return spot_v1.package_performance_lock(
            output_dir=output_dir,
            run_name=run_name,
            lock_inputs=lock_inputs,
            startup_script=startup_script,
        )


def validate_package(run_dir: str | Path) -> dict[str, Any]:
    with _rearm1_context():
        return spot_v1.validate_package(run_dir)


def validate_global_spot_claim(
    run_dir: str | Path, manifest: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    with _rearm1_context():
        return spot_v1.validate_global_spot_claim(run_dir, manifest)


def authorize_launch(run_dir: str | Path) -> dict[str, Any]:
    with _rearm1_context():
        _preauthorize_smoke_unlocked(run_dir)
        return spot_v1.authorize_launch(run_dir)


def validate_launch_authorization(
    run_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with _rearm1_context():
        return spot_v1.validate_launch_authorization(run_dir)


def launch_jobs(
    *,
    run_dir: str | Path,
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    with _rearm1_context():
        return spot_v1.launch_jobs(
            run_dir=run_dir,
            project=project,
            bucket=bucket,
        )


def validate_launch_chain(
    run_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    with _rearm1_context():
        return spot_v1.validate_launch_chain(run_dir)


def preflight_resume(
    *,
    run_dir: str | Path,
    selected: Sequence[str],
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    with _rearm1_context():
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
    with _rearm1_context():
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
    with _rearm1_context():
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
    with _rearm1_context():
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
    with _rearm1_context():
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
    with _rearm1_context():
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
    with _rearm1_context():
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
    with _rearm1_context():
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
    package.add_argument(
        "--old-v1-roots",
        type=Path,
        required=True,
    )
    package.add_argument(
        "--old-v1-global-claim",
        type=Path,
        default=lock_open.DEFAULT_OLD_V1_GLOBAL_CLAIM_PATH,
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
    write_smoke = commands.add_parser("write-preauthorize-smoke-receipt")
    write_smoke.add_argument("--run-dir", type=Path, required=True)
    write_smoke.add_argument("--output", type=Path)
    validate_smoke = commands.add_parser("validate-preauthorize-smoke-receipt")
    validate_smoke.add_argument("--run-dir", type=Path, required=True)
    validate_smoke.add_argument("--receipt", type=Path, required=True)
    validate = commands.add_parser("validate-package")
    validate.add_argument("--run-dir", type=Path, required=True)
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
    validate_claim.add_argument(
        "--no-verify-remote",
        action="store_true",
    )
    receive = commands.add_parser("receive")
    cloud_arguments(receive)
    receive.add_argument("--destination", type=Path, required=True)
    validate_receive = commands.add_parser("validate-received")
    cloud_arguments(validate_receive)
    validate_receive.add_argument("--receive-dir", type=Path, required=True)
    return parser


def _build_cli_inputs(args: argparse.Namespace) -> Any:
    _, lock_open = _load_rearm_modules()
    return lock_open.PerformanceLockRearm1Inputs(
        repository_root=args.repository_root,
        plan_path=args.plan,
        lock_output_directory=args.lock_output,
        candidate_library=args.candidate_library,
        reference_library=args.reference_library,
        feature_encoder=args.feature_encoder,
        startup_source=args.startup_source,
        incident_receipt_path=args.incident_receipt,
        old_v1_root_directory=args.old_v1_roots,
        development_summary_path=args.development_summary,
        development_validation_path=args.development_validation,
        development_root_directory=args.development_roots,
        old_v1_global_claim_path=args.old_v1_global_claim,
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
        result = preauthorize_smoke(args.run_dir)
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
    "PREAUTHORIZE_SMOKE_NAME",
    "PREAUTHORIZE_SMOKE_SCHEMA",
    "PREAUTHORIZE_SMOKE_STATUS",
    "REARM1_PLAN_SCOPE",
    "REARM1_STARTUP_PACKAGE_PHASE",
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
