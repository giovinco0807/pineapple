"""One-shot local open/materialize/seal lifecycle for M3.1 lock rearm1.

Rearm1 is a new, explicitly claimed identity after the immutable v1 Spot run
was consumed by a deterministic pre-content startup failure.  This module
never repairs or resumes that run.  It durably creates the distinct rearm1
global claim before constructing or touching the new root output path.

Only deterministic root creation/resume and root sealing are implemented
here.  Cloud launch, training, quality promotion, profile changes, and
opponent-private discard access are outside this contract.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

from . import close_hu_m31_t3_step6d_performance_lock_startup_failure as closeout
from . import hu_m31_t3_step6d_performance_lock_open as legacy_open
from . import run_hu_m31_t3_step6d_performance_v2 as runner
from . import select_hu_m31_t3_step6d_candidate02_tail_v2 as development_roots
from .hu_m31_t3_behavior_roots import M31_T3_BEHAVIOR_PROFILES


CLAIM_SCHEMA = "hu_m31_t3_step6d_performance_lock_rearm1_global_claim_v2"
MATERIALIZATION_SCHEMA = (
    "hu_m31_t3_step6d_performance_lock_rearm1_root_materialization_v2"
)
SEAL_SCHEMA = "hu_m31_t3_step6d_performance_lock_rearm1_root_seal_v2"

CLAIM_STATUS = (
    "global_rearm1_claim_persisted_before_new_root_touch_crash_consumes_claim"
)
MATERIALIZATION_STATUS = (
    "all_100_rearm1_roots_materialized_exact_claimed_identity"
)
SEAL_STATUS = (
    "sealed_100_fresh_disjoint_hidden_safe_performance_lock_rearm1_roots"
)

STARTUP_FAILURE_RECEIPT_SHA256 = (
    "b529f87a301ebb588d3e965b01bae08ea887ab13bc1f305c72058c7a90bafbd7"
)
PRECONTENT_PLAN_SHA256 = (
    "3b8a4230531f0d81c5320b2b0d051878113ac57a38ad97fdb0b1d89e0f17a886"
)
LOCK_RUN_CONTRACT_DIGEST = (
    "f02e8845401eef92ba617f2c832204bdc74c20aa9e91810c52bf99687c4886b5"
)
OLD_V1_GLOBAL_CLAIM_SHA256 = (
    "8b5db48a71d39eb8b85cc9fab43cf4ee43a9350c7b1fa49b3d004816ec397f31"
)
OLD_V1_MATERIALIZATION_SHA256 = (
    "002b79449136e156b0aa560304988fe90df22bb5ea567917d80fb95e753cf118"
)
OLD_V1_SEAL_SHA256 = (
    "804ad0ee10a2ff2d6a75336ef8878ce87e27e723888dd55f6970d2c8bf0e389b"
)

CANDIDATE_LIBRARY_SHA256 = legacy_open.CANDIDATE_LIBRARY_SHA256
REFERENCE_LIBRARY_SHA256 = legacy_open.REFERENCE_LIBRARY_SHA256
FEATURE_ENCODER_SHA256 = legacy_open.FEATURE_ENCODER_SHA256
STEP6D_CONTRACT_BYTE_SHA256 = legacy_open.STEP6D_CONTRACT_BYTE_SHA256
AI_PROFILES_CURRENT_SHA256 = legacy_open.AI_PROFILES_CURRENT_SHA256
IMAGE = dict(legacy_open.IMAGE)
ALLOCATION = dict(legacy_open.ALLOCATION)
MODEL_INPUT_PATHS = legacy_open.MODEL_INPUT_PATHS
ROOT_GENERATOR_INPUT_PATHS = legacy_open.ROOT_GENERATOR_INPUT_PATHS

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPOSITORY_ROOT = _REPO_ROOT
DEFAULT_PRECONTENT_PLAN_PATH = (
    _REPO_ROOT
    / "outputs/hu_joint_policy/m31_t3_step6d/performance_lock_rearm1/"
    "precontent_plan_v1.json"
)
DEFAULT_GLOBAL_CLAIM_PATH = (
    _REPO_ROOT
    / "outputs/hu_joint_policy/m31_t3_step6d/"
    "GLOBAL_PERFORMANCE_LOCK_REARM1_CLAIM.json"
)
DEFAULT_OLD_V1_GLOBAL_CLAIM_PATH = legacy_open.DEFAULT_GLOBAL_CLAIM_PATH
DEFAULT_DEVELOPMENT_MERGE_DIR = legacy_open.DEFAULT_DEVELOPMENT_MERGE_DIR
DEFAULT_DEVELOPMENT_ROOT_DIR = legacy_open.DEFAULT_DEVELOPMENT_ROOT_DIR

_PLAN_MODULE = (
    ".hu_m31_t3_step6d_candidate02_performance_lock_rearm1_plan"
)

_CLAIM_KEYS = frozenset(
    {
        "schema",
        "status",
        "scope",
        "opened_unix_ns",
        "global_claim_path",
        "lock_output_directory",
        "precontent_plan",
        "startup_failure_closeout",
        "old_v1_lock_evidence",
        "development_go",
        "step6d_contract",
        "lock_run_contract",
        "lock_run_contract_digest",
        "runner_source",
        "ai_profiles_current",
        "accepted_binaries",
        "model_inputs",
        "root_generator_inputs",
        "image",
        "allocation",
        "seed_contract",
        "startup_source",
        "rearm_guards",
        "restrictions",
    }
)
_MATERIALIZATION_KEYS = frozenset(
    {
        "schema",
        "status",
        "global_claim_sha256",
        "plan_sha256",
        "run_contract_digest",
        "hand_indices",
        "root_count",
        "root_artifact_sha256",
        "aggregate_root_sha256",
        "same_identity_resume_only",
        "fresh_recovery_seed_schedule",
        "old_v1_attempt1_reused",
        "old_v1_root_reused",
        "reseeded",
        "training_eligible",
        "current_profile_changed",
    }
)
_SEAL_KEYS = frozenset(
    {
        "schema",
        "status",
        "global_claim_sha256",
        "materialization_sha256",
        "plan_sha256",
        "run_contract_digest",
        "hand_indices",
        "root_count",
        "observation_count",
        "profile_counts",
        "seat_counts",
        "root_artifact_sha256",
        "aggregate_root_sha256",
        "root_topology_sha256",
        "observation_fingerprint_sha256",
        "root_artifact_unique",
        "observation_fingerprint_unique",
        "development_comparison",
        "old_performance_lock_comparison",
        "visibility",
        "selection_inputs",
        "rearm_guards",
        "training_eligible",
        "quality_evidence",
        "promotion_evidence",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
    }
)


@dataclass(frozen=True)
class PerformanceLockRearm1Inputs:
    repository_root: Path
    plan_path: Path
    lock_output_directory: Path
    candidate_library: Path
    reference_library: Path
    feature_encoder: Path
    startup_source: Path
    incident_receipt_path: Path
    old_v1_root_directory: Path
    development_summary_path: Path = DEFAULT_DEVELOPMENT_MERGE_DIR / "summary.json"
    development_validation_path: Path = (
        DEFAULT_DEVELOPMENT_MERGE_DIR / "validation.json"
    )
    development_root_directory: Path = DEFAULT_DEVELOPMENT_ROOT_DIR
    old_v1_global_claim_path: Path = DEFAULT_OLD_V1_GLOBAL_CLAIM_PATH
    global_claim_path: Path = DEFAULT_GLOBAL_CLAIM_PATH


# Compatibility alias for the shared Spot wrapper.  It maps to the distinct
# rearm1 input contract and cannot construct a legacy-v1 identity.
PerformanceLockInputs = PerformanceLockRearm1Inputs


canonical_bytes = legacy_open.canonical_bytes
canonical_sha256 = legacy_open.canonical_sha256
sha256_file = legacy_open.sha256_file
_lexical_absolute = legacy_open._lexical_absolute
_read_canonical = legacy_open._read_canonical
_write_once_durable = legacy_open._write_once_durable
_write_or_validate_durable = legacy_open._write_or_validate_durable
_file_record = legacy_open._file_record
_input_records = legacy_open._input_records


def _load_plan_module() -> ModuleType:
    try:
        return importlib.import_module(_PLAN_MODULE, package=__package__)
    except ModuleNotFoundError as exc:
        raise RuntimeError("rearm1 pre-content plan module is unavailable") from exc


def _module_string(module: ModuleType, names: Sequence[str], label: str) -> str:
    for name in names:
        value = getattr(module, name, None)
        if isinstance(value, str) and value:
            return value
    raise RuntimeError(f"rearm1 plan does not expose {label}")


def _load_and_validate_plan(path: Path) -> tuple[dict[str, Any], ModuleType]:
    raw = _read_canonical(path, "rearm1 pre-content plan")
    module = _load_plan_module()
    validator = getattr(module, "validate_precontent_plan", None)
    if not callable(validator):
        raise RuntimeError("rearm1 plan validator API is unavailable")
    validated = validator(raw)
    if not isinstance(validated, Mapping) or dict(validated) != raw:
        raise ValueError("rearm1 plan validator changed the plan")
    expected_sha = _module_string(
        module,
        ("PRECONTENT_PLAN_SHA256", "PLAN_SHA256"),
        "pre-content plan SHA",
    )
    if expected_sha != PRECONTENT_PLAN_SHA256 or sha256_file(path) != expected_sha:
        raise ValueError("rearm1 pre-content plan SHA changed")
    return raw, module


def _validate_incident(path: Path) -> dict[str, Any]:
    if sha256_file(path) != STARTUP_FAILURE_RECEIPT_SHA256:
        raise ValueError("startup-failure closeout receipt SHA changed")
    receipt = _read_canonical(path, "startup-failure closeout receipt")
    counts = receipt.get("control_plane_counts")
    disposition = receipt.get("disposition")
    access = receipt.get("content_access")
    incident = receipt.get("incident")
    logs = receipt.get("startup_logs")
    if (
        receipt.get("schema") != closeout.RECEIPT_SCHEMA
        or receipt.get("status") != closeout.RECEIPT_STATUS
        or receipt.get("run_name") != closeout.RUN_NAME
        or not isinstance(counts, Mapping)
        or set(counts) != set(closeout.COUNT_KEYS)
        or any(counts.get(key) != 0 for key in closeout.COUNT_KEYS)
        or not isinstance(disposition, Mapping)
        or disposition.get("current_run_irrecoverable") is not True
        or disposition.get("attempt1_authorized") is not False
        or disposition.get("alternate_seed_authorized") is not False
        or disposition.get("reseed_authorized") is not False
        or disposition.get("alternate_package_authorized") is not False
        or disposition.get("quality_pilot_authorized") is not False
        or disposition.get("training_eligible") is not False
        or disposition.get("current_profile_changed") is not False
        or not isinstance(access, Mapping)
        or access.get("hand_content_opened") is not False
        or access.get("root_content_opened") is not False
        or access.get("result_content_opened") is not False
        or access.get("cloud_query_performed") is not False
        or access.get("cloud_mutated") is not False
        or not isinstance(incident, Mapping)
        or incident.get("attempt_index") != 0
        or incident.get("failed_jobs") != 20
        or incident.get("deterministic_with_claimed_bytes") is not True
        or not isinstance(logs, Mapping)
        or logs.get("count") != 20
        or logs.get("job_ids") != list(closeout.JOB_IDS)
    ):
        raise ValueError("startup-failure closeout receipt boundary changed")
    return receipt


def _validate_old_v1_controls(inputs: PerformanceLockRearm1Inputs) -> dict[str, Any]:
    claim_path = Path(inputs.old_v1_global_claim_path)
    output = Path(inputs.old_v1_root_directory)
    materialization_path = output / "materialization.json"
    seal_path = output / "seal.json"
    expected = (
        (claim_path, OLD_V1_GLOBAL_CLAIM_SHA256, "old v1 global claim"),
        (
            materialization_path,
            OLD_V1_MATERIALIZATION_SHA256,
            "old v1 materialization",
        ),
        (seal_path, OLD_V1_SEAL_SHA256, "old v1 seal"),
    )
    for path, digest, label in expected:
        if sha256_file(path) != digest:
            raise ValueError(f"{label} SHA changed")
    claim = _read_canonical(claim_path, "old v1 global claim")
    materialization = _read_canonical(
        materialization_path, "old v1 materialization"
    )
    seal = _read_canonical(seal_path, "old v1 seal")
    if (
        claim.get("schema") != legacy_open.CLAIM_SCHEMA
        or claim.get("status")
        != (
            "global_one_shot_claim_persisted_before_lock_root_touch_"
            "crash_consumes_claim"
        )
        or claim.get("lock_run_contract_digest")
        != legacy_open.LOCK_RUN_CONTRACT_DIGEST
        or claim.get("ai_profiles_current", {}).get("sha256")
        != AI_PROFILES_CURRENT_SHA256
        or claim.get("image") != IMAGE
        or claim.get("allocation") != ALLOCATION
        or claim.get("seed_contract", {}).get("seed_set_sha256")
        != runner.CANDIDATE02_PERFORMANCE_LOCK_SEED_SET_SHA256
        or claim.get("restrictions", {}).get("reseed_allowed") is not False
        or claim.get("restrictions", {}).get("cloud_authorized") is not False
        or materialization.get("schema") != legacy_open.MATERIALIZATION_SCHEMA
        or materialization.get("root_count") != 100
        or seal.get("schema") != legacy_open.SEAL_SCHEMA
        or seal.get("root_count") != 100
        or seal.get("observation_count") != 200
        or seal.get("visibility", {}).get("opponent_private_discards_used")
        is not False
    ):
        raise ValueError("old v1 control evidence changed")
    if _lexical_absolute(output) != str(claim.get("lock_output_directory")):
        raise ValueError("old v1 root directory does not match its claim")
    return {
        "global_claim": {
            **_file_record(claim_path),
            "canonical_sha256": canonical_sha256(claim),
        },
        "materialization": {
            **_file_record(materialization_path),
            "canonical_sha256": canonical_sha256(materialization),
        },
        "seal": {
            **_file_record(seal_path),
            "canonical_sha256": canonical_sha256(seal),
        },
        "run_contract_digest": legacy_open.LOCK_RUN_CONTRACT_DIGEST,
        "seed_set_sha256": runner.CANDIDATE02_PERFORMANCE_LOCK_SEED_SET_SHA256,
        "attempt1_used": False,
        "root_reuse_authorized": False,
    }


def _plan_contract(
    plan: Mapping[str, Any], module: ModuleType
) -> tuple[dict[str, Any], str]:
    contract = runner.build_run_contract(
        candidate_library_sha256=CANDIDATE_LIBRARY_SHA256,
        reference_library_sha256=REFERENCE_LIBRARY_SHA256,
        variant=runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT,
    )
    digest = runner.canonical_sha256(contract)
    expected_digest = _module_string(
        module,
        ("RECOVERY_RUN_CONTRACT_DIGEST", "RUN_CONTRACT_DIGEST"),
        "recovery run-contract digest",
    )
    candidate_contracts = [
        plan.get(name)
        for name in ("recovery_run_contract", "run_contract", "lock_run_contract")
        if isinstance(plan.get(name), Mapping)
    ]
    candidate_digests = [
        plan.get(name)
        for name in (
            "recovery_run_contract_digest",
            "run_contract_digest",
            "lock_run_contract_digest",
        )
        if isinstance(plan.get(name), str)
    ]
    if (
        digest != expected_digest
        or expected_digest != LOCK_RUN_CONTRACT_DIGEST
        or contract not in candidate_contracts
        or digest not in candidate_digests
        or runner.contract_variant(contract)
        != runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT
    ):
        raise ValueError("rearm1 recovery runner contract changed")
    return contract, digest


def _validate_plan_rearm_boundaries(
    plan: Mapping[str, Any],
    receipt: Mapping[str, Any],
    module: ModuleType,
) -> None:
    closeout_record = plan.get("startup_failure_closeout")
    if (
        not isinstance(closeout_record, Mapping)
        or closeout_record.get("sha256") != STARTUP_FAILURE_RECEIPT_SHA256
        or closeout_record.get("receipt") != receipt
    ):
        raise ValueError("rearm1 plan startup-failure receipt binding changed")
    old = plan.get("old_run_disposition")
    fresh = plan.get("fresh_lock_authority")
    if not isinstance(old, Mapping) or not isinstance(fresh, Mapping):
        raise ValueError("rearm1 plan authority boundary is absent")
    old_producer = getattr(module, "_old_run_disposition", None)
    fresh_producer = getattr(module, "_fresh_lock_authority", None)
    if (
        not callable(old_producer)
        or not callable(fresh_producer)
        or dict(old) != old_producer()
        or dict(fresh) != fresh_producer()
        or old.get("old_attempt1_authorized") is not False
        or old.get("old_root_reuse_authorized") is not False
        or old.get("old_seed_reuse_authorized") is not False
        or old.get("old_package_reuse_authorized") is not False
        or fresh.get("fresh_lock_ordinal") != "rearm1"
        or fresh.get("authorized_fresh_lock_count") != 1
        or fresh.get("fresh_lock_plan_authorized") is not True
        or fresh.get("fresh_root_set_required") is not True
        or fresh.get("fresh_seed_set_required") is not True
        or fresh.get("fresh_global_claim_required_before_root_content") is not True
        or fresh.get("fresh_root_content_authorized_before_global_claim")
        is not False
        or fresh.get("old_attempt1_authorized") is not False
        or fresh.get("old_roots_authorized") is not False
        or fresh.get("old_seeds_authorized") is not False
    ):
        raise ValueError("rearm1 plan permits old attempt/root reuse")
    for field in (
        "cloud_started",
        "training_eligible",
        "quality_evidence",
        "promotion_evidence",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
        "m31_complete",
    ):
        if field in plan and plan.get(field) is not False:
            raise ValueError(f"rearm1 plan widened authorization: {field}")


def _claim_payload(
    inputs: PerformanceLockRearm1Inputs, *, opened_unix_ns: int
) -> dict[str, Any]:
    root = Path(inputs.repository_root).resolve()
    plan_path = Path(inputs.plan_path).resolve()
    plan, plan_module = _load_and_validate_plan(plan_path)
    receipt = _validate_incident(Path(inputs.incident_receipt_path).resolve())
    _validate_plan_rearm_boundaries(plan, receipt, plan_module)
    contract, contract_digest = _plan_contract(plan, plan_module)
    development_go = legacy_open._validate_official_development_go(
        Path(inputs.development_summary_path).resolve(),
        Path(inputs.development_validation_path).resolve(),
    )
    old_v1 = _validate_old_v1_controls(inputs)

    step6d_contract_path = root / "configs/hu_joint_policy_m31_t3_step6d_contract.json"
    step6d_contract = json.loads(step6d_contract_path.read_text(encoding="utf-8"))
    if (
        sha256_file(step6d_contract_path) != STEP6D_CONTRACT_BYTE_SHA256
        or step6d_contract["anchors"]["policy_registry"]["byte_sha256"]
        != AI_PROFILES_CURRENT_SHA256
    ):
        raise ValueError("Step6d/current registry anchor changed")
    ai_profiles = _file_record(root / "src/ofc_regular/ai_profiles.py", root=root)
    if ai_profiles["sha256"] != AI_PROFILES_CURRENT_SHA256:
        raise ValueError("ai_profiles/current bytes changed")

    accepted = {
        "candidate": _file_record(inputs.candidate_library),
        "reference": _file_record(inputs.reference_library),
        "feature_encoder": _file_record(inputs.feature_encoder),
    }
    if {
        name: record["sha256"] for name, record in accepted.items()
    } != {
        "candidate": CANDIDATE_LIBRARY_SHA256,
        "reference": REFERENCE_LIBRARY_SHA256,
        "feature_encoder": FEATURE_ENCODER_SHA256,
    }:
        raise ValueError("accepted rearm1 binary identity changed")

    seed_contract = runner.candidate02_performance_lock_recovery_seed_contract()
    if (
        seed_contract.get("seed_set_sha256")
        != runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SEED_SET_SHA256
        or seed_contract.get("seed_min") != 700_108_071_901
        or seed_contract.get("seed_max") != 705_207_072_198
        or seed_contract.get("seed_count") != 600
        or seed_contract.get("all_values_unique") is not True
        or seed_contract.get("locked_before_content_read") is not True
        or seed_contract.get("candidate02_development_overlap_count") != 0
        or seed_contract.get("performance_lock_v1_overlap_count") != 0
    ):
        raise ValueError("rearm1 recovery seed contract changed")

    claim_identity = _lexical_absolute(inputs.global_claim_path)
    output_identity = _lexical_absolute(inputs.lock_output_directory)
    old_output_identity = _lexical_absolute(inputs.old_v1_root_directory)
    if claim_identity != _lexical_absolute(DEFAULT_GLOBAL_CLAIM_PATH):
        raise ValueError("rearm1 claim path is not the fixed global path")
    if output_identity == old_output_identity:
        raise ValueError("rearm1 output would reuse the old v1 root path")
    try:
        root_common = os.path.commonpath((output_identity, old_output_identity))
    except ValueError:
        root_common = ""
    if root_common in (output_identity, old_output_identity):
        raise ValueError("rearm1 and old v1 root trees must be disjoint")
    for unsafe in (output_identity, old_output_identity):
        try:
            common = os.path.commonpath((claim_identity, unsafe))
        except ValueError:
            common = ""
        if common == unsafe:
            raise ValueError("rearm1 global claim must be outside root trees")

    startup_record = _file_record(inputs.startup_source)
    plan_startup_sha = plan.get("startup_source_sha256")
    if isinstance(plan_startup_sha, str) and startup_record["sha256"] != plan_startup_sha:
        raise ValueError("rearm1 startup source differs from the plan")

    payload = {
        "schema": CLAIM_SCHEMA,
        "status": CLAIM_STATUS,
        "scope": "candidate02_performance_lock_rearm1_fresh_roots_only",
        "opened_unix_ns": opened_unix_ns,
        "global_claim_path": claim_identity,
        "lock_output_directory": output_identity,
        "precontent_plan": {
            **_file_record(plan_path),
            "schema": plan.get("schema"),
            "canonical_sha256": canonical_sha256(plan),
        },
        "startup_failure_closeout": {
            **_file_record(inputs.incident_receipt_path),
            "schema": receipt["schema"],
            "status": receipt["status"],
            "canonical_sha256": canonical_sha256(receipt),
        },
        "old_v1_lock_evidence": old_v1,
        "development_go": development_go,
        "step6d_contract": {
            **_file_record(step6d_contract_path, root=root),
            "canonical_sha256": canonical_sha256(step6d_contract),
        },
        "lock_run_contract": contract,
        "lock_run_contract_digest": contract_digest,
        "runner_source": _file_record(
            root / "src/ofc_regular/run_hu_m31_t3_step6d_performance_v2.py",
            root=root,
        ),
        "ai_profiles_current": ai_profiles,
        "accepted_binaries": accepted,
        "model_inputs": _input_records(root, MODEL_INPUT_PATHS),
        "root_generator_inputs": _input_records(root, ROOT_GENERATOR_INPUT_PATHS),
        "image": dict(IMAGE),
        "allocation": dict(ALLOCATION),
        "seed_contract": seed_contract,
        "startup_source": startup_record,
        "rearm_guards": {
            "new_global_claim": True,
            "claim_before_new_root_path_touch": True,
            "crash_consumes_claim": True,
            "deterministic_exact_identity_resume_only": True,
            "fresh_700_series_seed_schedule": True,
            "old_v1_attempt1_used": False,
            "old_v1_attempt1_reused": False,
            "old_v1_root_reused": False,
            "old_v1_package_reused": False,
        },
        "restrictions": {
            "timing_used_for_root_selection": False,
            "q_used_for_root_selection": False,
            "ev_used_for_root_selection": False,
            "old_v1_attempt1_allowed": False,
            "old_v1_root_reuse_allowed": False,
            "alternate_seed_allowed": False,
            # Retained for the audited shared Spot archive validator.  The
            # stricter rearm-specific field below fixes the post-claim scope.
            "reseed_allowed": False,
            "post_claim_reseed_allowed": False,
            "cloud_authorized": False,
            "training_authorized": False,
            "quality_authorized": False,
            "promotion_authorized": False,
            "current_profile_resolution_allowed": False,
            "runtime_activation_allowed": False,
            "opponent_private_discards_allowed": False,
        },
    }
    if set(payload) != _CLAIM_KEYS:
        raise AssertionError("rearm1 claim schema implementation changed")
    return payload


def _after_claim_persisted(_claim_path: Path) -> None:
    """Test hook: raising here models a crash that consumes the rearm1 claim."""


def open_performance_lock_rearm1(
    inputs: PerformanceLockRearm1Inputs,
) -> dict[str, Any]:
    """Persist the rearm1 claim without touching its new root output path."""

    claim = _claim_payload(inputs, opened_unix_ns=time.time_ns())
    claim_path = Path(inputs.global_claim_path)
    _write_once_durable(claim_path, claim)
    _after_claim_persisted(claim_path)
    return claim


def validate_global_claim(
    inputs: PerformanceLockRearm1Inputs,
) -> dict[str, Any]:
    claim = _read_canonical(inputs.global_claim_path, "rearm1 global claim")
    if set(claim) != _CLAIM_KEYS:
        raise ValueError("rearm1 global claim fields changed")
    opened = claim.get("opened_unix_ns")
    if isinstance(opened, bool) or not isinstance(opened, int) or opened <= 0:
        raise ValueError("rearm1 global claim timestamp changed")
    expected = _claim_payload(inputs, opened_unix_ns=opened)
    if claim != expected:
        raise ValueError("rearm1 global claim identity changed")
    return claim


def validate_open_claim(
    inputs: PerformanceLockRearm1Inputs,
) -> dict[str, Any]:
    return validate_global_claim(inputs)


def _lock_output_after_claim(inputs: PerformanceLockRearm1Inputs) -> Path:
    """The sole adapter allowed to construct the new output Path."""

    return Path(inputs.lock_output_directory)


def _root_hashes(roots: Sequence[Mapping[str, Any]]) -> list[str]:
    return [canonical_sha256(root) for root in roots]


def _materialization_payload(
    claim: Mapping[str, Any], roots: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if len(roots) != 100:
        raise ValueError("rearm1 materialization is incomplete")
    hashes = _root_hashes(roots)
    value = {
        "schema": MATERIALIZATION_SCHEMA,
        "status": MATERIALIZATION_STATUS,
        "global_claim_sha256": canonical_sha256(claim),
        "plan_sha256": claim["precontent_plan"]["sha256"],
        "run_contract_digest": claim["lock_run_contract_digest"],
        "hand_indices": list(runner.CONTRACT_HAND_INDICES),
        "root_count": 100,
        "root_artifact_sha256": hashes,
        "aggregate_root_sha256": canonical_sha256(hashes),
        "same_identity_resume_only": True,
        "fresh_recovery_seed_schedule": True,
        "old_v1_attempt1_reused": False,
        "old_v1_root_reused": False,
        # Core field retained for the audited shared package/startup validator.
        # It means no reseed occurred after the fresh rearm1 plan/claim.
        "reseeded": False,
        "training_eligible": False,
        "current_profile_changed": False,
    }
    if set(value) != _MATERIALIZATION_KEYS:
        raise AssertionError("rearm1 materialization schema changed")
    return value


def _load_roots(
    output: Path,
    contract: Mapping[str, Any],
    *,
    label: str,
) -> tuple[list[dict[str, Any]], list[tuple[Any, Any]]]:
    root_dir = output / "roots"
    expected = [f"hand_{index:03d}.json" for index in runner.CONTRACT_HAND_INDICES]
    observed = sorted(path.name for path in root_dir.glob("hand_*.json"))
    if observed != expected:
        raise ValueError(f"{label} root set must be exactly 000..099")
    roots: list[dict[str, Any]] = []
    observations: list[tuple[Any, Any]] = []
    for index, name in zip(runner.CONTRACT_HAND_INDICES, expected, strict=True):
        value = _read_canonical(root_dir / name, f"{label} root {index}")
        parsed = runner._validate_root_artifact(contract, value, index=index)
        roots.append(value)
        observations.append(parsed)
    return roots, observations


def materialize_performance_lock_rearm1(
    inputs: PerformanceLockRearm1Inputs,
) -> dict[str, Any]:
    """Create or resume only the exact claimed recovery-v2 root identity."""

    claim = validate_global_claim(inputs)
    # No new-output Path is constructed above this line.
    output = _lock_output_after_claim(inputs)
    output.mkdir(parents=True, exist_ok=True)
    contract = runner.validate_run_contract(claim["lock_run_contract"])
    roots = runner._materialize_roots(
        contract=contract,
        repository_root=Path(inputs.repository_root).resolve(),
        output_dir=output,
        indices=runner.CONTRACT_HAND_INDICES,
    )
    canonical_roots, _ = _load_roots(output, contract, label="rearm1")
    if roots != canonical_roots:
        raise ValueError("rearm1 runner returned a non-canonical root set")
    value = _materialization_payload(claim, canonical_roots)
    _write_or_validate_durable(
        output / "materialization.json", value, "rearm1 materialization"
    )
    return value


def _root_sets(
    roots: Sequence[Mapping[str, Any]],
) -> tuple[set[str], set[str], set[int]]:
    hashes = set(_root_hashes(roots))
    fingerprints = {
        str(observation["observation_fingerprint"])
        for root in roots
        for observation in root["observations"]
    }
    seeds = {
        int(seed)
        for root in roots
        for seed in root["seeds"].values()
    }
    return hashes, fingerprints, seeds


def _load_old_v1_roots(
    inputs: PerformanceLockRearm1Inputs,
) -> tuple[list[dict[str, Any]], list[tuple[Any, Any]]]:
    contract = runner.build_run_contract(
        candidate_library_sha256=CANDIDATE_LIBRARY_SHA256,
        reference_library_sha256=REFERENCE_LIBRARY_SHA256,
        variant=runner.CANDIDATE02_PERFORMANCE_LOCK_VARIANT,
    )
    if runner.canonical_sha256(contract) != legacy_open.LOCK_RUN_CONTRACT_DIGEST:
        raise ValueError("old v1 runner contract changed")
    roots, observations = _load_roots(
        Path(inputs.old_v1_root_directory),
        contract,
        label="old v1",
    )
    seal = _read_canonical(
        Path(inputs.old_v1_root_directory) / "seal.json", "old v1 seal"
    )
    if (
        _root_hashes(roots) != seal.get("root_artifact_sha256")
        or canonical_sha256(_root_hashes(roots))
        != seal.get("aggregate_root_sha256")
    ):
        raise ValueError("old v1 root set differs from its sealed identity")
    return roots, observations


def _build_root_seal(inputs: PerformanceLockRearm1Inputs) -> dict[str, Any]:
    claim = validate_global_claim(inputs)
    output = _lock_output_after_claim(inputs)
    materialization = _read_canonical(
        output / "materialization.json", "rearm1 materialization"
    )
    contract = runner.validate_run_contract(claim["lock_run_contract"])
    roots, observations = _load_roots(output, contract, label="rearm1")
    if materialization != _materialization_payload(claim, roots):
        raise ValueError("rearm1 materialization identity changed")

    hashes = _root_hashes(roots)
    profile_counts = Counter(str(root["profile"]) for root in roots)
    expected_profiles = {profile: 20 for profile in M31_T3_BEHAVIOR_PROFILES}
    if dict(profile_counts) != expected_profiles:
        raise ValueError("rearm1 profile balance changed")
    fingerprints = [
        observation.fingerprint() for pair in observations for observation in pair
    ]
    if len(fingerprints) != 200 or len(set(fingerprints)) != 200:
        raise ValueError("rearm1 observation fingerprints collide")
    if len(hashes) != 100 or len(set(hashes)) != 100:
        raise ValueError("rearm1 root artifact hashes collide")
    seats = Counter(
        str(raw["seat"]) for root in roots for raw in root["observations"]
    )
    if dict(seats) != {"first": 100, "second": 100}:
        raise ValueError("rearm1 seat balance changed")

    development = development_roots.load_frozen_roots(
        inputs.development_root_directory
    )
    old_v1_roots, _old_observations = _load_old_v1_roots(inputs)
    new_sets = _root_sets(roots)
    development_sets = _root_sets(development)
    old_v1_sets = _root_sets(old_v1_roots)
    development_overlap = tuple(
        len(current & prior)
        for current, prior in zip(new_sets, development_sets, strict=True)
    )
    old_v1_overlap = tuple(
        len(current & prior)
        for current, prior in zip(new_sets, old_v1_sets, strict=True)
    )
    if any(development_overlap) or any(old_v1_overlap):
        raise ValueError("rearm1 roots overlap development or old v1 evidence")

    topology = legacy_open._topology_rows(roots, observations)
    value = {
        "schema": SEAL_SCHEMA,
        "status": SEAL_STATUS,
        "global_claim_sha256": canonical_sha256(claim),
        "materialization_sha256": canonical_sha256(materialization),
        "plan_sha256": claim["precontent_plan"]["sha256"],
        "run_contract_digest": claim["lock_run_contract_digest"],
        "hand_indices": list(runner.CONTRACT_HAND_INDICES),
        "root_count": 100,
        "observation_count": 200,
        "profile_counts": expected_profiles,
        "seat_counts": {"first": 100, "second": 100},
        "root_artifact_sha256": hashes,
        "aggregate_root_sha256": canonical_sha256(hashes),
        "root_topology_sha256": canonical_sha256(topology),
        "observation_fingerprint_sha256": canonical_sha256(fingerprints),
        "root_artifact_unique": True,
        "observation_fingerprint_unique": True,
        "development_comparison": {
            "development_all100_root_sha256": development_roots.ALL100_ROOT_SHA256,
            "development_root_count": 100,
            "lock_fingerprint_overlap_count": development_overlap[1],
            "lock_root_hash_overlap_count": development_overlap[0],
            "lock_seed_overlap_count": development_overlap[2],
        },
        "old_performance_lock_comparison": {
            "old_v1_global_claim_sha256": OLD_V1_GLOBAL_CLAIM_SHA256,
            "old_v1_seal_sha256": OLD_V1_SEAL_SHA256,
            "old_v1_root_count": 100,
            "rearm1_fingerprint_overlap_count": old_v1_overlap[1],
            "rearm1_root_hash_overlap_count": old_v1_overlap[0],
            "rearm1_seed_overlap_count": old_v1_overlap[2],
            "old_v1_attempt1_reused": False,
            "old_v1_root_reused": False,
        },
        "visibility": {
            "runner_validator_replayed_all_roots": True,
            "first_observation_count": 100,
            "second_observation_count": 100,
            "opponent_private_discards_used": False,
            "current_profile_resolved": False,
        },
        "selection_inputs": {
            "timing_used": False,
            "q_used": False,
            "ev_used": False,
            "all_100_preregistered_hands_used": True,
        },
        "rearm_guards": {
            "incident_receipt_sha256": STARTUP_FAILURE_RECEIPT_SHA256,
            "fresh_700_series_seed_schedule": True,
            "same_identity_resume_only": True,
            "old_v1_attempt1_reused": False,
            "old_v1_root_reused": False,
            "post_claim_reseeded": False,
        },
        "training_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
    }
    if set(value) != _SEAL_KEYS:
        raise AssertionError("rearm1 seal schema implementation changed")
    return value


def seal_performance_lock_rearm1(
    inputs: PerformanceLockRearm1Inputs,
) -> dict[str, Any]:
    value = _build_root_seal(inputs)
    output = _lock_output_after_claim(inputs)
    _write_or_validate_durable(output / "seal.json", value, "rearm1 seal")
    return value


def validate_root_seal(
    inputs: PerformanceLockRearm1Inputs,
) -> dict[str, Any]:
    validate_global_claim(inputs)
    output = _lock_output_after_claim(inputs)
    stored = _read_canonical(output / "seal.json", "rearm1 seal")
    if set(stored) != _SEAL_KEYS:
        raise ValueError("rearm1 seal fields changed")
    expected = _build_root_seal(inputs)
    if stored != expected:
        raise ValueError("rearm1 seal replay changed")
    return stored


def _inputs_from_args(args: argparse.Namespace) -> PerformanceLockRearm1Inputs:
    return PerformanceLockRearm1Inputs(
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("open", "materialize", "seal"))
    parser.add_argument("--repository-root", type=Path, default=DEFAULT_REPOSITORY_ROOT)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PRECONTENT_PLAN_PATH)
    parser.add_argument("--lock-output", type=Path, required=True)
    parser.add_argument("--candidate-library", type=Path, required=True)
    parser.add_argument("--reference-library", type=Path, required=True)
    parser.add_argument("--feature-encoder", type=Path, required=True)
    parser.add_argument("--startup-source", type=Path, required=True)
    parser.add_argument("--incident-receipt", type=Path, required=True)
    parser.add_argument("--old-v1-roots", type=Path, required=True)
    parser.add_argument(
        "--development-summary",
        type=Path,
        default=DEFAULT_DEVELOPMENT_MERGE_DIR / "summary.json",
    )
    parser.add_argument(
        "--development-validation",
        type=Path,
        default=DEFAULT_DEVELOPMENT_MERGE_DIR / "validation.json",
    )
    parser.add_argument(
        "--development-roots",
        type=Path,
        default=DEFAULT_DEVELOPMENT_ROOT_DIR,
    )
    parser.add_argument(
        "--old-v1-global-claim",
        type=Path,
        default=DEFAULT_OLD_V1_GLOBAL_CLAIM_PATH,
    )
    parser.add_argument(
        "--global-claim",
        type=Path,
        default=DEFAULT_GLOBAL_CLAIM_PATH,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inputs = _inputs_from_args(args)
    if args.command == "open":
        value = open_performance_lock_rearm1(inputs)
    elif args.command == "materialize":
        value = materialize_performance_lock_rearm1(inputs)
    else:
        value = seal_performance_lock_rearm1(inputs)
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ALLOCATION",
    "CLAIM_SCHEMA",
    "DEFAULT_GLOBAL_CLAIM_PATH",
    "DEFAULT_PRECONTENT_PLAN_PATH",
    "DEFAULT_REPOSITORY_ROOT",
    "IMAGE",
    "MATERIALIZATION_SCHEMA",
    "LOCK_RUN_CONTRACT_DIGEST",
    "PRECONTENT_PLAN_SHA256",
    "PerformanceLockInputs",
    "PerformanceLockRearm1Inputs",
    "SEAL_SCHEMA",
    "STARTUP_FAILURE_RECEIPT_SHA256",
    "canonical_bytes",
    "canonical_sha256",
    "main",
    "materialize_performance_lock_rearm1",
    "open_performance_lock_rearm1",
    "seal_performance_lock_rearm1",
    "validate_global_claim",
    "validate_open_claim",
    "validate_root_seal",
]
