"""Freeze the fresh Candidate02 performance-lock rearm2 pre-content plan.

Rearm1 attempt0 is terminal.  This plan authorizes only a new claim, new
roots, and the recovery-v3 710-series seed namespaces.  Spot authorization
remains forbidden until a write-once exhaustive smoke receipt proves the
actual package and all twenty packaged job manifests pass the startup
contract without reading roots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import (
    close_hu_m31_t3_step6d_performance_lock_rearm1_startup_failure as closeout,
)
from . import (
    hu_m31_t3_step6d_candidate02_performance_lock_rearm1_plan as rearm1_plan,
)
from . import run_hu_m31_t3_step6d_performance_v2 as runner
from .hu_m31_t3_behavior_roots import M31_T3_BEHAVIOR_PROFILES


PLAN_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_lock_rearm2_precontent_plan_v1"
)
PLAN_STATUS = "frozen_rearm2_precontent_plan_fresh_lock_not_opened"
PLAN_DECISION = (
    "terminal_rearm1_startup_failure_authorizes_one_fresh_v3_seed_lock_only"
)
PLAN_SCOPE = "performance_lock_rearm2_fresh_roots_only"
ASSIGNMENT_METHOD = "consecutive_ten_hand_arithmetic_rearm2_v1"

ACTUAL_PACKAGE_SMOKE_SCHEMA = (
    "hu_m31_t3_step6d_performance_lock_actual_package_preauthorize_smoke_v2"
)
ACTUAL_PACKAGE_SMOKE_STATUS = (
    "actual_package_manifest_authorization_all_20_jobs_full_no_root_read_"
    "startup_contract_passed_before_spot_authorization"
)

CANDIDATE_VARIANT = runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_VARIANT
RUN_ID = runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_RUN_ID
SCHEDULE = runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_SCHEDULE
ROOT_SCHEMA = runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_ROOT_SCHEMA
DONE_SCHEMA = runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_DONE_SCHEMA
SHARD_MANIFEST_SCHEMA = (
    runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_SHARD_MANIFEST_SCHEMA
)
RECOVERY_V3_RUN_CONTRACT_DIGEST = (
    "39bc01820c4dd690f3e971112dca181b45028f67a45893ea0708391ca06280a5"
)
LOCK_RUN_CONTRACT_DIGEST = RECOVERY_V3_RUN_CONTRACT_DIGEST
STARTUP_FAILURE_RECEIPT_SHA256 = (
    "fa9041b064a11db2b24e9b2051aee0bc72661b384fd4ff84b89c62331eeb4214"
)
PRECONTENT_PLAN_SHA256_PLACEHOLDER = "__SET_AFTER_CANONICAL_REARM1_CLOSEOUT__"
PRECONTENT_PLAN_SHA256 = (
    "8e67f3443208b5fddea20eb574f6ea760d8e9d7f518180f906944103796569d5"
)

CANDIDATE_LIBRARY_SHA256 = rearm1_plan.CANDIDATE_LIBRARY_SHA256
REFERENCE_LIBRARY_SHA256 = rearm1_plan.REFERENCE_LIBRARY_SHA256
FEATURE_ENCODER_SHA256 = rearm1_plan.FEATURE_ENCODER_SHA256
CURRENT_PROFILE_REGISTRY_SHA256 = rearm1_plan.CURRENT_PROFILE_REGISTRY_SHA256
STEP6D_CONTRACT_SHA256 = rearm1_plan.STEP6D_CONTRACT_SHA256
SOURCE_ROLES = tuple(runner.SOURCE_ROLES)
SHARD_COUNT_PER_ROLE = 10
HANDS_PER_SHARD = 10
LOGICAL_JOB_COUNT = 20
JOB_IDS = tuple(
    f"{role}-shard-{index:02d}"
    for role in SOURCE_ROLES
    for index in range(SHARD_COUNT_PER_ROLE)
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INCIDENT_RECEIPT_PATH = (
    _REPO_ROOT
    / "outputs/hu_joint_policy/m31_t3_step6d/performance_lock_rearm2/"
    "performance_lock_rearm1_startup_failure_closeout.json"
)
INCIDENT_RECEIPT_REPO_PATH = (
    "outputs/hu_joint_policy/m31_t3_step6d/performance_lock_rearm2/"
    "performance_lock_rearm1_startup_failure_closeout.json"
)
DEFAULT_OUTPUT_PATH = (
    _REPO_ROOT
    / "outputs/hu_joint_policy/m31_t3_step6d/performance_lock_rearm2/"
    "precontent_plan_v1.json"
)


def canonical_bytes(value: Any) -> bytes:
    return runner.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return runner.canonical_sha256(value)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_canonical(path: str | Path, label: str) -> dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    raw = target.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"precontent plan already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise FileExistsError(f"precontent plan already exists: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def validate_startup_failure_closeout(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = dict(value)
    logs = receipt.get("startup_logs")
    counts = receipt.get("control_plane_counts")
    incident = receipt.get("incident")
    access = receipt.get("content_access")
    disposition = receipt.get("disposition")
    if (
        canonical_sha256(receipt) != STARTUP_FAILURE_RECEIPT_SHA256
        or receipt.get("schema") != closeout.RECEIPT_SCHEMA
        or receipt.get("status") != closeout.RECEIPT_STATUS
        or receipt.get("run_name") != closeout.RUN_NAME
        or receipt.get("run_contract_digest") != closeout.RUN_CONTRACT_DIGEST
        or not isinstance(logs, Mapping)
        or logs.get("count") != 20
        or logs.get("job_ids") != list(closeout.JOB_IDS)
        or logs.get("all_exact_schema_failures") is not True
        or not _is_sha256(logs.get("aggregate_sha256"))
        or not isinstance(counts, Mapping)
        or set(counts) != set(closeout.COUNT_KEYS)
        or any(counts.get(key) != 0 for key in closeout.COUNT_KEYS)
        or not isinstance(incident, Mapping)
        or incident.get("attempt_index") != 0
        or incident.get("failed_jobs") != 20
        or incident.get("error") != closeout.ERROR_LINE
        or incident.get("actual_job_schema")
        != runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SHARD_MANIFEST_SCHEMA
        or incident.get("startup_expected_job_schema") != runner.SHARD_MANIFEST_SCHEMA
        or incident.get("deterministic_with_claimed_bytes") is not True
        or not isinstance(access, Mapping)
        or access.get("root_content_opened") is not False
        or access.get("hand_content_opened") is not False
        or access.get("result_content_opened") is not False
        or access.get("cloud_mutated") is not False
        or not isinstance(disposition, Mapping)
        or disposition.get("attempt0_consumed") is not True
        or disposition.get("current_run_irrecoverable") is not True
        or disposition.get("rearm1_attempt1_authorized") is not False
        or disposition.get("rearm1_package_reuse_authorized") is not False
        or disposition.get("rearm1_root_reuse_authorized") is not False
        or disposition.get("rearm1_seed_reuse_authorized") is not False
        or disposition.get("rearm1_claim_reuse_authorized") is not False
        or disposition.get("fresh_rearm2_plan_authorized") is not False
        or disposition.get("training_eligible") is not False
        or disposition.get("current_profile_changed") is not False
    ):
        raise ValueError("rearm1 startup-failure closeout boundary changed")
    return receipt


def load_startup_failure_closeout(
    path: str | Path = DEFAULT_INCIDENT_RECEIPT_PATH,
) -> dict[str, Any]:
    target = Path(path)
    if sha256_file(target) != STARTUP_FAILURE_RECEIPT_SHA256:
        raise ValueError("rearm1 startup-failure closeout file hash changed")
    return validate_startup_failure_closeout(
        _read_canonical(target, "rearm1 startup-failure closeout")
    )


def _run_contract() -> dict[str, Any]:
    contract = runner.build_run_contract(
        candidate_library_sha256=CANDIDATE_LIBRARY_SHA256,
        reference_library_sha256=REFERENCE_LIBRARY_SHA256,
        variant=CANDIDATE_VARIANT,
    )
    if (
        runner.canonical_sha256(contract) != RECOVERY_V3_RUN_CONTRACT_DIGEST
        or runner.contract_variant(contract) != CANDIDATE_VARIANT
    ):
        raise RuntimeError("rearm2 recovery-v3 run contract changed")
    return contract


def _shards() -> list[dict[str, Any]]:
    values = []
    for index in range(SHARD_COUNT_PER_ROLE):
        hands = list(range(index * HANDS_PER_SHARD, (index + 1) * HANDS_PER_SHARD))
        values.append(
            {
                "shard_index": index,
                "work_hand_indices": hands,
                "profile_counts": {
                    profile: sum(
                        1
                        for hand in hands
                        if M31_T3_BEHAVIOR_PROFILES[
                            hand % len(M31_T3_BEHAVIOR_PROFILES)
                        ]
                        == profile
                    )
                    for profile in M31_T3_BEHAVIOR_PROFILES
                },
            }
        )
    return values


def _jobs(
    contract: Mapping[str, Any], shards: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    jobs = []
    for role in SOURCE_ROLES:
        for shard in shards:
            index = int(shard["shard_index"])
            hands = list(shard["work_hand_indices"])
            manifest = runner.build_shard_manifest(
                run_contract=contract,
                source_role=role,
                work_hand_indices=hands,
            )
            jobs.append(
                {
                    "job_id": f"{role}-shard-{index:02d}",
                    "source_role": role,
                    "shard_index": index,
                    "work_hand_indices": hands,
                    "shard_manifest_sha256": canonical_sha256(manifest),
                }
            )
    return jobs


def _rearm1_disposition() -> dict[str, Any]:
    return {
        "run_name": closeout.RUN_NAME,
        "run_contract_digest": closeout.RUN_CONTRACT_DIGEST,
        "attempt0_consumed": True,
        "current_run_irrecoverable": True,
        "rearm1_attempt1_authorized": False,
        "rearm1_package_reuse_authorized": False,
        "rearm1_root_reuse_authorized": False,
        "rearm1_seed_reuse_authorized": False,
        "rearm1_claim_reuse_authorized": False,
        "rearm1_result_reuse_authorized": False,
        "rearm1_cloud_execution_authorized": False,
    }


def _smoke_requirement() -> dict[str, Any]:
    return {
        "schema": ACTUAL_PACKAGE_SMOKE_SCHEMA,
        "status": ACTUAL_PACKAGE_SMOKE_STATUS,
        "authorization_requires_exhaustive_actual_package_smoke": True,
        "write_once_receipt_required": True,
        "verified_job_count": 20,
        "startup_invocation_count": 20,
        "root_read": False,
        "cloud_mutated": False,
        "global_spot_claim_and_authorization_bind_receipt_sha256": True,
    }


def _fresh_lock_authority() -> dict[str, Any]:
    return {
        "fresh_lock_ordinal": "rearm2",
        "authorized_fresh_lock_count": 1,
        "fresh_lock_plan_authorized": True,
        "fresh_root_set_required": True,
        "fresh_seed_set_required": True,
        "fresh_global_claim_required_before_root_content": True,
        "fresh_root_content_authorized_before_global_claim": False,
        "fresh_spot_execution_authorized": False,
        "authorization_before_actual_package_smoke_allowed": False,
        "rearm1_attempt1_authorized": False,
        "rearm1_package_authorized": False,
        "rearm1_roots_authorized": False,
        "rearm1_seeds_authorized": False,
        "rearm1_claim_authorized": False,
        "alternate_seed_authorized": False,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
    }


@lru_cache(maxsize=1)
def _base_rearm1_plan() -> dict[str, Any]:
    return rearm1_plan.build_precontent_plan()


def _build_payload(receipt: Mapping[str, Any]) -> dict[str, Any]:
    validated_receipt = validate_startup_failure_closeout(receipt)
    base = deepcopy(_base_rearm1_plan())
    contract = _run_contract()
    seeds = runner.candidate02_performance_lock_recovery_v3_seed_contract()
    shards = _shards()
    jobs = _jobs(contract, shards)
    value = {
        "schema": PLAN_SCHEMA,
        "status": PLAN_STATUS,
        "decision": PLAN_DECISION,
        "scope": PLAN_SCOPE,
        "development_qualification": base["development_qualification"],
        "startup_failure_closeout": {
            "path": INCIDENT_RECEIPT_REPO_PATH,
            "sha256": STARTUP_FAILURE_RECEIPT_SHA256,
            "receipt": validated_receipt,
        },
        "rearm1_disposition": _rearm1_disposition(),
        "prior_lock_evidence": {
            "v1_global_claim_sha256": (
                rearm1_plan.closeout.EXPECTED_HASHES[
                    rearm1_plan.closeout.GLOBAL_ROOT_CLAIM_NAME
                ]
            ),
            "v1_seed_set_sha256": (
                runner.CANDIDATE02_PERFORMANCE_LOCK_SEED_SET_SHA256
            ),
            "rearm1_global_claim_sha256": closeout.EXPECTED_HASHES[
                closeout.GLOBAL_ROOT_CLAIM_NAME
            ],
            "rearm1_materialization_sha256": closeout.EXPECTED_HASHES[
                closeout.MATERIALIZATION_NAME
            ],
            "rearm1_seal_sha256": closeout.EXPECTED_HASHES[closeout.SEAL_NAME],
            "rearm1_package_manifest_sha256": closeout.EXPECTED_HASHES[
                closeout.MANIFEST_NAME
            ],
            "rearm1_seed_set_sha256": (
                runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SEED_SET_SHA256
            ),
        },
        "fresh_lock_authority": _fresh_lock_authority(),
        "actual_package_smoke_requirement": _smoke_requirement(),
        "candidate_variant": CANDIDATE_VARIANT,
        "step6d_contract": base["step6d_contract"],
        "run_contract": contract,
        "run_contract_digest": RECOVERY_V3_RUN_CONTRACT_DIGEST,
        "source_identity": base["source_identity"],
        "image": base["image"],
        "allocation": base["allocation"],
        "root_contract": {
            "schema": ROOT_SCHEMA,
            "run_id": RUN_ID,
            "schedule": SCHEDULE,
            "seed_contract": seeds,
            "hand_indices": list(runner.CONTRACT_HAND_INDICES),
            "root_indices": list(range(200)),
            "hands": 100,
            "roots": 200,
            "fresh_root_set": True,
            "fresh_seed_set": True,
            "root_content_addressed": False,
            "root_content_hashes_known": False,
            "development_overlap_required": 0,
            "v1_overlap_required": 0,
            "rearm1_overlap_required": 0,
        },
        "assignment": {
            "method": ASSIGNMENT_METHOD,
            "source_role_isolated_process": True,
            "candidate_reference_same_hand_mapping": True,
        },
        "source_roles": list(SOURCE_ROLES),
        "shard_count_per_role": SHARD_COUNT_PER_ROLE,
        "hands_per_shard": HANDS_PER_SHARD,
        "logical_job_count": LOGICAL_JOB_COUNT,
        "shards": shards,
        "jobs": jobs,
        "open_claim_required_before_root_content": True,
        "root_content_opened": False,
        "cloud_started": False,
        "performance_lock_passed": False,
        "training_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "m31_complete": False,
    }
    return value


def compute_precontent_plan_sha256(
    *,
    incident_receipt: Mapping[str, Any] | None = None,
    incident_receipt_path: str | Path = DEFAULT_INCIDENT_RECEIPT_PATH,
) -> str:
    receipt = (
        load_startup_failure_closeout(incident_receipt_path)
        if incident_receipt is None
        else dict(incident_receipt)
    )
    return canonical_sha256(_build_payload(receipt))


def build_precontent_plan(
    *,
    incident_receipt: Mapping[str, Any] | None = None,
    incident_receipt_path: str | Path = DEFAULT_INCIDENT_RECEIPT_PATH,
) -> dict[str, Any]:
    if not _is_sha256(PRECONTENT_PLAN_SHA256):
        raise RuntimeError("rearm2 precontent plan SHA is not frozen")
    receipt = (
        load_startup_failure_closeout(incident_receipt_path)
        if incident_receipt is None
        else dict(incident_receipt)
    )
    value = _build_payload(receipt)
    return validate_precontent_plan(value, incident_receipt=receipt)


def validate_precontent_plan(
    value: Mapping[str, Any],
    *,
    incident_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(value)
    closeout_record = payload.get("startup_failure_closeout")
    receipt = (
        closeout_record.get("receipt")
        if incident_receipt is None and isinstance(closeout_record, Mapping)
        else incident_receipt
    )
    if not isinstance(receipt, Mapping):
        raise ValueError("rearm2 closeout receipt is absent")
    expected = _build_payload(validate_startup_failure_closeout(receipt))
    if payload != expected:
        raise ValueError("rearm2 precontent plan changed")
    if canonical_sha256(payload) != PRECONTENT_PLAN_SHA256:
        raise ValueError("rearm2 precontent plan SHA changed")
    if (
        payload.get("schema") != PLAN_SCHEMA
        or payload.get("status") != PLAN_STATUS
        or payload.get("scope") != PLAN_SCOPE
        or payload.get("run_contract_digest") != RECOVERY_V3_RUN_CONTRACT_DIGEST
        or runner.validate_run_contract(payload["run_contract"])
        != payload["run_contract"]
        or payload.get("rearm1_disposition") != _rearm1_disposition()
        or payload.get("fresh_lock_authority") != _fresh_lock_authority()
        or payload.get("actual_package_smoke_requirement")
        != _smoke_requirement()
        or payload.get("jobs") != _jobs(payload["run_contract"], _shards())
    ):
        raise ValueError("rearm2 plan contract boundary changed")
    return payload


def load_and_validate_precontent_plan(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if sha256_file(target) != PRECONTENT_PLAN_SHA256:
        raise ValueError("rearm2 precontent plan file hash changed")
    return validate_precontent_plan(_read_canonical(target, "rearm2 plan"))


def write_precontent_plan(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    incident_receipt_path: str | Path = DEFAULT_INCIDENT_RECEIPT_PATH,
) -> dict[str, Any]:
    value = build_precontent_plan(incident_receipt_path=incident_receipt_path)
    _write_once(Path(output_path), value)
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("digest", "write", "validate"))
    parser.add_argument("--incident-receipt", type=Path, default=DEFAULT_INCIDENT_RECEIPT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "digest":
        print(
            compute_precontent_plan_sha256(
                incident_receipt_path=args.incident_receipt
            )
        )
    elif args.command == "write":
        value = write_precontent_plan(
            output_path=args.output,
            incident_receipt_path=args.incident_receipt,
        )
        print(canonical_sha256(value))
    else:
        print(canonical_sha256(load_and_validate_precontent_plan(args.output)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ACTUAL_PACKAGE_SMOKE_SCHEMA",
    "ACTUAL_PACKAGE_SMOKE_STATUS",
    "CANDIDATE_VARIANT",
    "DEFAULT_INCIDENT_RECEIPT_PATH",
    "DEFAULT_OUTPUT_PATH",
    "DONE_SCHEMA",
    "JOB_IDS",
    "LOCK_RUN_CONTRACT_DIGEST",
    "PLAN_SCHEMA",
    "PLAN_SCOPE",
    "PLAN_STATUS",
    "PRECONTENT_PLAN_SHA256",
    "PRECONTENT_PLAN_SHA256_PLACEHOLDER",
    "RECOVERY_V3_RUN_CONTRACT_DIGEST",
    "ROOT_SCHEMA",
    "RUN_ID",
    "SCHEDULE",
    "SHARD_MANIFEST_SCHEMA",
    "STARTUP_FAILURE_RECEIPT_SHA256",
    "build_precontent_plan",
    "canonical_bytes",
    "canonical_sha256",
    "compute_precontent_plan_sha256",
    "load_and_validate_precontent_plan",
    "load_startup_failure_closeout",
    "validate_precontent_plan",
    "validate_startup_failure_closeout",
    "write_precontent_plan",
]
