"""One-shot open/materialize/seal lifecycle for performance-lock rearm2.

The global claim is durably written before the new output path is constructed.
The lifecycle rejects reuse of either prior root tree, all rearm1 identities,
and authorization that is not preceded by the separately write-once exhaustive
actual-package startup smoke receipt.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

from . import (
    close_hu_m31_t3_step6d_performance_lock_rearm1_startup_failure as closeout,
)
from . import hu_m31_t3_step6d_performance_lock_rearm1_open as rearm1_open
from . import run_hu_m31_t3_step6d_performance_v2 as runner
from . import select_hu_m31_t3_step6d_candidate02_tail_v2 as development_roots
from .hu_m31_t3_behavior_roots import M31_T3_BEHAVIOR_PROFILES


CLAIM_SCHEMA = "hu_m31_t3_step6d_performance_lock_rearm2_global_claim_v1"
MATERIALIZATION_SCHEMA = (
    "hu_m31_t3_step6d_performance_lock_rearm2_root_materialization_v1"
)
SEAL_SCHEMA = "hu_m31_t3_step6d_performance_lock_rearm2_root_seal_v1"
CLAIM_STATUS = (
    "global_rearm2_claim_persisted_before_new_root_touch_crash_consumes_claim"
)
MATERIALIZATION_STATUS = (
    "all_100_rearm2_roots_materialized_exact_claimed_identity"
)
SEAL_STATUS = (
    "sealed_100_fresh_disjoint_hidden_safe_performance_lock_rearm2_roots"
)
PREFLIGHT_SCHEMA = (
    "hu_m31_t3_step6d_performance_lock_rearm2_preopen_preflight_v1"
)
PREFLIGHT_STATUS = (
    "all_rearm2_open_prerequisites_validated_without_persistent_write"
)
_PREFLIGHT_CLAIM_TEMPLATE_OPENED_UNIX_NS = 1

PRECONTENT_PLAN_SHA256 = (
    "8e67f3443208b5fddea20eb574f6ea760d8e9d7f518180f906944103796569d5"
)
LOCK_RUN_CONTRACT_DIGEST = (
    "39bc01820c4dd690f3e971112dca181b45028f67a45893ea0708391ca06280a5"
)
REARM1_STARTUP_FAILURE_RECEIPT_SHA256 = (
    "fa9041b064a11db2b24e9b2051aee0bc72661b384fd4ff84b89c62331eeb4214"
)
OLD_V1_GLOBAL_CLAIM_SHA256 = rearm1_open.OLD_V1_GLOBAL_CLAIM_SHA256
OLD_V1_MATERIALIZATION_SHA256 = rearm1_open.OLD_V1_MATERIALIZATION_SHA256
OLD_V1_SEAL_SHA256 = rearm1_open.OLD_V1_SEAL_SHA256
REARM1_GLOBAL_CLAIM_SHA256 = closeout.EXPECTED_HASHES[
    closeout.GLOBAL_ROOT_CLAIM_NAME
]
REARM1_MATERIALIZATION_SHA256 = closeout.EXPECTED_HASHES[
    closeout.MATERIALIZATION_NAME
]
REARM1_SEAL_SHA256 = closeout.EXPECTED_HASHES[closeout.SEAL_NAME]

CANDIDATE_LIBRARY_SHA256 = rearm1_open.CANDIDATE_LIBRARY_SHA256
REFERENCE_LIBRARY_SHA256 = rearm1_open.REFERENCE_LIBRARY_SHA256
FEATURE_ENCODER_SHA256 = rearm1_open.FEATURE_ENCODER_SHA256
STEP6D_CONTRACT_BYTE_SHA256 = rearm1_open.STEP6D_CONTRACT_BYTE_SHA256
AI_PROFILES_CURRENT_SHA256 = rearm1_open.AI_PROFILES_CURRENT_SHA256
IMAGE = dict(rearm1_open.IMAGE)
ALLOCATION = dict(rearm1_open.ALLOCATION)
MODEL_INPUT_PATHS = rearm1_open.MODEL_INPUT_PATHS
ROOT_GENERATOR_INPUT_PATHS = rearm1_open.ROOT_GENERATOR_INPUT_PATHS

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPOSITORY_ROOT = _REPO_ROOT
DEFAULT_PRECONTENT_PLAN_PATH = (
    _REPO_ROOT
    / "outputs/hu_joint_policy/m31_t3_step6d/performance_lock_rearm2/"
    "precontent_plan_v1.json"
)
DEFAULT_GLOBAL_CLAIM_PATH = (
    _REPO_ROOT
    / "outputs/hu_joint_policy/m31_t3_step6d/"
    "GLOBAL_PERFORMANCE_LOCK_REARM2_CLAIM.json"
)
DEFAULT_OLD_V1_GLOBAL_CLAIM_PATH = rearm1_open.DEFAULT_OLD_V1_GLOBAL_CLAIM_PATH
DEFAULT_REARM1_GLOBAL_CLAIM_PATH = rearm1_open.DEFAULT_GLOBAL_CLAIM_PATH
DEFAULT_DEVELOPMENT_MERGE_DIR = rearm1_open.DEFAULT_DEVELOPMENT_MERGE_DIR
DEFAULT_DEVELOPMENT_ROOT_DIR = rearm1_open.DEFAULT_DEVELOPMENT_ROOT_DIR

_PLAN_MODULE = ".hu_m31_t3_step6d_candidate02_performance_lock_rearm2_plan"

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
        "prior_lock_evidence",
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
        "actual_package_smoke_requirement",
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
        "fresh_recovery_v3_seed_schedule",
        "old_v1_root_reused",
        "rearm1_attempt1_reused",
        "rearm1_package_reused",
        "rearm1_root_reused",
        "rearm1_seed_reused",
        "rearm1_claim_reused",
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
        "prior_lock_comparison",
        "visibility",
        "selection_inputs",
        "actual_package_smoke_requirement",
        "rearm_guards",
        "training_eligible",
        "quality_evidence",
        "promotion_evidence",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
    }
)
_PREFLIGHT_KEYS = frozenset(
    {
        "schema",
        "status",
        "scope",
        "proposed_global_claim_path",
        "proposed_lock_output_directory",
        "precontent_plan_sha256",
        "startup_failure_closeout_sha256",
        "step6d_contract_sha256",
        "lock_run_contract_digest",
        "claim_template_opened_unix_ns",
        "claim_template_sha256",
        "accepted_binary_sha256",
        "startup_sha256",
        "ai_profiles_current_sha256",
        "seed_set_sha256",
        "prior_lock_evidence_sha256",
        "development_go_sha256",
        "model_inputs_sha256",
        "root_generator_inputs_sha256",
        "proposed_global_claim_absent",
        "proposed_lock_output_absent",
        "persistent_write_executed",
        "new_root_content_opened",
        "cloud_mutation_executed",
        "current_profile_changed",
    }
)


@dataclass(frozen=True)
class PerformanceLockRearm2Inputs:
    repository_root: Path
    plan_path: Path
    lock_output_directory: Path
    candidate_library: Path
    reference_library: Path
    feature_encoder: Path
    startup_source: Path
    incident_receipt_path: Path
    old_v1_root_directory: Path
    rearm1_root_directory: Path
    development_summary_path: Path = DEFAULT_DEVELOPMENT_MERGE_DIR / "summary.json"
    development_validation_path: Path = (
        DEFAULT_DEVELOPMENT_MERGE_DIR / "validation.json"
    )
    development_root_directory: Path = DEFAULT_DEVELOPMENT_ROOT_DIR
    old_v1_global_claim_path: Path = DEFAULT_OLD_V1_GLOBAL_CLAIM_PATH
    rearm1_global_claim_path: Path = DEFAULT_REARM1_GLOBAL_CLAIM_PATH
    global_claim_path: Path = DEFAULT_GLOBAL_CLAIM_PATH


PerformanceLockInputs = PerformanceLockRearm2Inputs
canonical_bytes = rearm1_open.canonical_bytes
canonical_sha256 = rearm1_open.canonical_sha256
sha256_file = rearm1_open.sha256_file
_lexical_absolute = rearm1_open._lexical_absolute
_read_canonical = rearm1_open._read_canonical
_write_once_durable = rearm1_open._write_once_durable
_write_or_validate_durable = rearm1_open._write_or_validate_durable
_file_record = rearm1_open._file_record
_input_records = rearm1_open._input_records


def _load_plan_module() -> ModuleType:
    return importlib.import_module(_PLAN_MODULE, package=__package__)


def _load_and_validate_plan(path: Path) -> tuple[dict[str, Any], ModuleType]:
    value = _read_canonical(path, "rearm2 precontent plan")
    module = _load_plan_module()
    validator = getattr(module, "validate_precontent_plan", None)
    if not callable(validator) or validator(value) != value:
        raise ValueError("rearm2 plan validator changed the plan")
    expected = getattr(module, "PRECONTENT_PLAN_SHA256", None)
    if expected != PRECONTENT_PLAN_SHA256 or sha256_file(path) != expected:
        raise ValueError("rearm2 precontent plan SHA changed")
    return value, module


def _validate_incident(path: Path) -> dict[str, Any]:
    if sha256_file(path) != REARM1_STARTUP_FAILURE_RECEIPT_SHA256:
        raise ValueError("rearm1 startup-failure receipt SHA changed")
    value = _read_canonical(path, "rearm1 startup-failure closeout")
    module = _load_plan_module()
    validator = getattr(module, "validate_startup_failure_closeout", None)
    if not callable(validator) or validator(value) != value:
        raise ValueError("rearm1 startup-failure closeout changed")
    return value


def _expected_smoke_requirement() -> dict[str, Any]:
    module = _load_plan_module()
    producer = getattr(module, "_smoke_requirement", None)
    if not callable(producer):
        raise RuntimeError("rearm2 smoke contract producer is absent")
    return dict(producer())


def _validate_prior_lock_controls(
    inputs: PerformanceLockRearm2Inputs,
) -> dict[str, Any]:
    old_v1 = rearm1_open._validate_old_v1_controls(inputs)
    claim_path = Path(inputs.rearm1_global_claim_path)
    root = Path(inputs.rearm1_root_directory)
    materialization_path = root / "materialization.json"
    seal_path = root / "seal.json"
    for path, expected, label in (
        (claim_path, REARM1_GLOBAL_CLAIM_SHA256, "rearm1 global claim"),
        (
            materialization_path,
            REARM1_MATERIALIZATION_SHA256,
            "rearm1 materialization",
        ),
        (seal_path, REARM1_SEAL_SHA256, "rearm1 seal"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"{label} SHA changed")
    claim = _read_canonical(claim_path, "rearm1 global claim")
    materialization = _read_canonical(materialization_path, "rearm1 materialization")
    seal = _read_canonical(seal_path, "rearm1 seal")
    if (
        claim.get("schema") != rearm1_open.CLAIM_SCHEMA
        or claim.get("status") != rearm1_open.CLAIM_STATUS
        or claim.get("lock_run_contract_digest")
        != rearm1_open.LOCK_RUN_CONTRACT_DIGEST
        or claim.get("ai_profiles_current", {}).get("sha256")
        != AI_PROFILES_CURRENT_SHA256
        or claim.get("seed_contract", {}).get("seed_set_sha256")
        != runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SEED_SET_SHA256
        or claim.get("restrictions", {}).get("reseed_allowed") is not False
        or claim.get("restrictions", {}).get("cloud_authorized") is not False
        or materialization.get("schema") != rearm1_open.MATERIALIZATION_SCHEMA
        or materialization.get("root_count") != 100
        or materialization.get("reseeded") is not False
        or seal.get("schema") != rearm1_open.SEAL_SCHEMA
        or seal.get("root_count") != 100
        or seal.get("observation_count") != 200
        or seal.get("visibility", {}).get("opponent_private_discards_used")
        is not False
    ):
        raise ValueError("rearm1 lock evidence changed")
    if _lexical_absolute(root) != claim.get("lock_output_directory"):
        raise ValueError("rearm1 root directory differs from its claim")
    return {
        "old_v1": old_v1,
        "rearm1": {
            "global_claim": _file_record(claim_path),
            "materialization": _file_record(materialization_path),
            "seal": _file_record(seal_path),
            "global_claim_sha256": REARM1_GLOBAL_CLAIM_SHA256,
            "materialization_sha256": REARM1_MATERIALIZATION_SHA256,
            "seal_sha256": REARM1_SEAL_SHA256,
            "run_contract_digest": rearm1_open.LOCK_RUN_CONTRACT_DIGEST,
            "seed_set_sha256": (
                runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SEED_SET_SHA256
            ),
            "attempt1_used": False,
            "package_reuse_authorized": False,
            "root_reuse_authorized": False,
            "seed_reuse_authorized": False,
            "claim_reuse_authorized": False,
        },
        "all_prior_reuse_authorized": False,
    }


def _plan_contract(
    plan: Mapping[str, Any], module: ModuleType
) -> tuple[dict[str, Any], str]:
    contract = runner.build_run_contract(
        candidate_library_sha256=CANDIDATE_LIBRARY_SHA256,
        reference_library_sha256=REFERENCE_LIBRARY_SHA256,
        variant=runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_VARIANT,
    )
    digest = runner.canonical_sha256(contract)
    if (
        digest != LOCK_RUN_CONTRACT_DIGEST
        or getattr(module, "RECOVERY_V3_RUN_CONTRACT_DIGEST", None) != digest
        or plan.get("run_contract") != contract
        or plan.get("run_contract_digest") != digest
    ):
        raise ValueError("rearm2 recovery-v3 runner contract changed")
    return contract, digest


def _validate_plan_boundaries(
    plan: Mapping[str, Any],
    receipt: Mapping[str, Any],
    module: ModuleType,
) -> dict[str, Any]:
    closeout_record = plan.get("startup_failure_closeout")
    disposition = plan.get("rearm1_disposition")
    authority = plan.get("fresh_lock_authority")
    smoke = plan.get("actual_package_smoke_requirement")
    if (
        not isinstance(closeout_record, Mapping)
        or closeout_record.get("sha256")
        != REARM1_STARTUP_FAILURE_RECEIPT_SHA256
        or closeout_record.get("receipt") != receipt
        or not isinstance(disposition, Mapping)
        or not isinstance(authority, Mapping)
        or not isinstance(smoke, Mapping)
    ):
        raise ValueError("rearm2 plan evidence boundary is absent")
    for name, actual in (
        ("_rearm1_disposition", disposition),
        ("_fresh_lock_authority", authority),
        ("_smoke_requirement", smoke),
    ):
        producer = getattr(module, name, None)
        if not callable(producer) or dict(actual) != producer():
            raise ValueError("rearm2 plan authority producer changed")
    if (
        disposition.get("rearm1_attempt1_authorized") is not False
        or disposition.get("rearm1_package_reuse_authorized") is not False
        or disposition.get("rearm1_root_reuse_authorized") is not False
        or disposition.get("rearm1_seed_reuse_authorized") is not False
        or disposition.get("rearm1_claim_reuse_authorized") is not False
        or authority.get("fresh_lock_ordinal") != "rearm2"
        or authority.get("fresh_root_set_required") is not True
        or authority.get("fresh_seed_set_required") is not True
        or authority.get("fresh_global_claim_required_before_root_content")
        is not True
        or authority.get("fresh_root_content_authorized_before_global_claim")
        is not False
        or authority.get("rearm1_attempt1_authorized") is not False
        or authority.get("rearm1_package_authorized") is not False
        or authority.get("rearm1_roots_authorized") is not False
        or authority.get("rearm1_seeds_authorized") is not False
        or authority.get("authorization_before_actual_package_smoke_allowed")
        is not False
        or smoke != _expected_smoke_requirement()
        or smoke.get("authorization_requires_exhaustive_actual_package_smoke")
        is not True
        or smoke.get("verified_job_count") != 20
        or smoke.get("startup_invocation_count") != 20
        or smoke.get("root_read") is not False
    ):
        raise ValueError("rearm2 plan permits prior reuse or weak smoke")
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
        if plan.get(field) is not False:
            raise ValueError(f"rearm2 plan widened authorization: {field}")
    return dict(smoke)


def _disjoint_paths(*values: str) -> bool:
    for index, left in enumerate(values):
        for right in values[index + 1 :]:
            try:
                common = os.path.commonpath((left, right))
            except ValueError:
                continue
            if common in (left, right):
                return False
    return True


def _claim_payload(
    inputs: PerformanceLockRearm2Inputs, *, opened_unix_ns: int
) -> dict[str, Any]:
    root = Path(inputs.repository_root).resolve()
    plan_path = Path(inputs.plan_path).resolve()
    plan, module = _load_and_validate_plan(plan_path)
    receipt = _validate_incident(Path(inputs.incident_receipt_path).resolve())
    smoke = _validate_plan_boundaries(plan, receipt, module)
    contract, digest = _plan_contract(plan, module)
    development_go = rearm1_open.legacy_open._validate_official_development_go(
        Path(inputs.development_summary_path).resolve(),
        Path(inputs.development_validation_path).resolve(),
    )
    prior = _validate_prior_lock_controls(inputs)

    step6d_path = root / "configs/hu_joint_policy_m31_t3_step6d_contract.json"
    step6d = json.loads(step6d_path.read_text(encoding="utf-8"))
    if (
        sha256_file(step6d_path) != STEP6D_CONTRACT_BYTE_SHA256
        or step6d["anchors"]["policy_registry"]["byte_sha256"]
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
        name: value["sha256"] for name, value in accepted.items()
    } != {
        "candidate": CANDIDATE_LIBRARY_SHA256,
        "reference": REFERENCE_LIBRARY_SHA256,
        "feature_encoder": FEATURE_ENCODER_SHA256,
    }:
        raise ValueError("accepted rearm2 binary identity changed")
    seeds = runner.candidate02_performance_lock_recovery_v3_seed_contract()
    if (
        seeds.get("seed_set_sha256")
        != runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_V3_SEED_SET_SHA256
        or seeds.get("seed_min") != 710_108_071_901
        or seeds.get("seed_max") != 715_207_072_198
        or seeds.get("seed_count") != 600
        or seeds.get("all_values_unique") is not True
        or seeds.get("performance_lock_v1_overlap_count") != 0
        or seeds.get("performance_lock_rearm1_overlap_count") != 0
        or seeds.get("existing_step6d_union_overlap_count") != 0
    ):
        raise ValueError("rearm2 seed contract changed")

    claim_identity = _lexical_absolute(inputs.global_claim_path)
    output_identity = _lexical_absolute(inputs.lock_output_directory)
    old_v1_identity = _lexical_absolute(inputs.old_v1_root_directory)
    rearm1_identity = _lexical_absolute(inputs.rearm1_root_directory)
    if claim_identity != _lexical_absolute(DEFAULT_GLOBAL_CLAIM_PATH):
        raise ValueError("rearm2 claim path is not the fixed global path")
    if not _disjoint_paths(output_identity, old_v1_identity, rearm1_identity):
        raise ValueError("rearm2 output and prior root trees must be disjoint")
    for unsafe in (output_identity, old_v1_identity, rearm1_identity):
        try:
            common = os.path.commonpath((claim_identity, unsafe))
        except ValueError:
            continue
        if common == unsafe:
            raise ValueError("rearm2 global claim must be outside root trees")

    value = {
        "schema": CLAIM_SCHEMA,
        "status": CLAIM_STATUS,
        "scope": "candidate02_performance_lock_rearm2_fresh_roots_only",
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
            "schema": receipt.get("schema"),
            "status": receipt.get("status"),
            "canonical_sha256": canonical_sha256(receipt),
        },
        "prior_lock_evidence": prior,
        "development_go": development_go,
        "step6d_contract": {
            **_file_record(step6d_path, root=root),
            "canonical_sha256": canonical_sha256(step6d),
        },
        "lock_run_contract": contract,
        "lock_run_contract_digest": digest,
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
        "seed_contract": seeds,
        "startup_source": _file_record(inputs.startup_source),
        "actual_package_smoke_requirement": smoke,
        "rearm_guards": {
            "new_global_claim": True,
            "claim_before_new_root_path_touch": True,
            "crash_consumes_claim": True,
            "deterministic_exact_identity_resume_only": True,
            "fresh_710_series_seed_schedule": True,
            "old_v1_root_reused": False,
            "rearm1_attempt1_reused": False,
            "rearm1_package_reused": False,
            "rearm1_root_reused": False,
            "rearm1_seed_reused": False,
            "rearm1_claim_reused": False,
        },
        "restrictions": {
            "timing_used_for_root_selection": False,
            "q_used_for_root_selection": False,
            "ev_used_for_root_selection": False,
            "old_v1_root_reuse_allowed": False,
            "rearm1_attempt1_allowed": False,
            "rearm1_package_reuse_allowed": False,
            "rearm1_root_reuse_allowed": False,
            "rearm1_seed_reuse_allowed": False,
            "rearm1_claim_reuse_allowed": False,
            "alternate_seed_allowed": False,
            "reseed_allowed": False,
            "post_claim_reseed_allowed": False,
            "authorization_requires_exhaustive_actual_package_smoke": True,
            "authorization_before_actual_package_smoke_allowed": False,
            "actual_package_smoke_receipt_sha_bound_at_open": False,
            "actual_package_smoke_receipt_sha_must_bind_spot_claim": True,
            "actual_package_smoke_receipt_sha_must_bind_authorization": True,
            "cloud_authorized": False,
            "training_authorized": False,
            "quality_authorized": False,
            "promotion_authorized": False,
            "current_profile_resolution_allowed": False,
            "runtime_activation_allowed": False,
            "opponent_private_discards_allowed": False,
        },
    }
    if set(value) != _CLAIM_KEYS:
        raise AssertionError("rearm2 claim schema implementation changed")
    return value


def _after_claim_persisted(_path: Path) -> None:
    """Test hook for a crash after the one-shot claim is durable."""


def _path_lexists(path: str | Path) -> bool:
    """Return true for every occupied filesystem identity, including symlinks."""

    return os.path.lexists(os.fspath(path))


def _require_unused_preflight_targets(
    inputs: PerformanceLockRearm2Inputs,
) -> None:
    occupied: list[str] = []
    for label, path in (
        ("global claim", inputs.global_claim_path),
        ("lock output", inputs.lock_output_directory),
    ):
        if _path_lexists(path):
            occupied.append(f"{label}: {Path(path)}")
    if occupied:
        raise FileExistsError(
            "rearm2 no-write preflight requires unused targets: "
            + ", ".join(occupied)
        )


def preflight_performance_lock_rearm2(
    inputs: PerformanceLockRearm2Inputs,
) -> dict[str, Any]:
    """Validate the complete open identity without claiming or opening roots.

    The deterministic claim template exercises the same prerequisite checks as
    ``open_performance_lock_rearm2``.  Its fixed positive timestamp is used
    only to make the returned evidence replayable; the actual one-shot claim
    still receives its real durable open timestamp.
    """

    _require_unused_preflight_targets(inputs)
    claim = _claim_payload(
        inputs,
        opened_unix_ns=_PREFLIGHT_CLAIM_TEMPLATE_OPENED_UNIX_NS,
    )
    _require_unused_preflight_targets(inputs)
    value = {
        "schema": PREFLIGHT_SCHEMA,
        "status": PREFLIGHT_STATUS,
        "scope": claim["scope"],
        "proposed_global_claim_path": claim["global_claim_path"],
        "proposed_lock_output_directory": claim["lock_output_directory"],
        "precontent_plan_sha256": claim["precontent_plan"]["sha256"],
        "startup_failure_closeout_sha256": claim[
            "startup_failure_closeout"
        ]["sha256"],
        "step6d_contract_sha256": claim["step6d_contract"]["sha256"],
        "lock_run_contract_digest": claim["lock_run_contract_digest"],
        "claim_template_opened_unix_ns": (
            _PREFLIGHT_CLAIM_TEMPLATE_OPENED_UNIX_NS
        ),
        "claim_template_sha256": canonical_sha256(claim),
        "accepted_binary_sha256": {
            role: claim["accepted_binaries"][role]["sha256"]
            for role in ("candidate", "reference", "feature_encoder")
        },
        "startup_sha256": claim["startup_source"]["sha256"],
        "ai_profiles_current_sha256": claim["ai_profiles_current"]["sha256"],
        "seed_set_sha256": claim["seed_contract"]["seed_set_sha256"],
        "prior_lock_evidence_sha256": canonical_sha256(
            claim["prior_lock_evidence"]
        ),
        "development_go_sha256": canonical_sha256(claim["development_go"]),
        "model_inputs_sha256": canonical_sha256(claim["model_inputs"]),
        "root_generator_inputs_sha256": canonical_sha256(
            claim["root_generator_inputs"]
        ),
        "proposed_global_claim_absent": True,
        "proposed_lock_output_absent": True,
        "persistent_write_executed": False,
        "new_root_content_opened": False,
        "cloud_mutation_executed": False,
        "current_profile_changed": False,
    }
    if set(value) != _PREFLIGHT_KEYS:
        raise AssertionError("rearm2 preopen preflight schema changed")
    return value


def open_performance_lock_rearm2(
    inputs: PerformanceLockRearm2Inputs,
) -> dict[str, Any]:
    value = _claim_payload(inputs, opened_unix_ns=time.time_ns())
    path = Path(inputs.global_claim_path)
    _write_once_durable(path, value)
    _after_claim_persisted(path)
    return value


def validate_global_claim(
    inputs: PerformanceLockRearm2Inputs,
) -> dict[str, Any]:
    value = _read_canonical(inputs.global_claim_path, "rearm2 global claim")
    if set(value) != _CLAIM_KEYS:
        raise ValueError("rearm2 global claim fields changed")
    opened = value.get("opened_unix_ns")
    if isinstance(opened, bool) or not isinstance(opened, int) or opened <= 0:
        raise ValueError("rearm2 global claim timestamp changed")
    if value != _claim_payload(inputs, opened_unix_ns=opened):
        raise ValueError("rearm2 global claim identity changed")
    return value


validate_open_claim = validate_global_claim


def _lock_output_after_claim(inputs: PerformanceLockRearm2Inputs) -> Path:
    return Path(inputs.lock_output_directory)


def _root_hashes(roots: Sequence[Mapping[str, Any]]) -> list[str]:
    return [canonical_sha256(value) for value in roots]


def _materialization_payload(
    claim: Mapping[str, Any], roots: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if len(roots) != 100:
        raise ValueError("rearm2 materialization is incomplete")
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
        "fresh_recovery_v3_seed_schedule": True,
        "old_v1_root_reused": False,
        "rearm1_attempt1_reused": False,
        "rearm1_package_reused": False,
        "rearm1_root_reused": False,
        "rearm1_seed_reused": False,
        "rearm1_claim_reused": False,
        "reseeded": False,
        "training_eligible": False,
        "current_profile_changed": False,
    }
    if set(value) != _MATERIALIZATION_KEYS:
        raise AssertionError("rearm2 materialization schema changed")
    return value


def materialize_performance_lock_rearm2(
    inputs: PerformanceLockRearm2Inputs,
) -> dict[str, Any]:
    claim = validate_global_claim(inputs)
    output = _lock_output_after_claim(inputs)
    output.mkdir(parents=True, exist_ok=True)
    contract = runner.validate_run_contract(claim["lock_run_contract"])
    roots = runner._materialize_roots(
        contract=contract,
        repository_root=Path(inputs.repository_root).resolve(),
        output_dir=output,
        indices=runner.CONTRACT_HAND_INDICES,
    )
    canonical_roots, _ = rearm1_open._load_roots(
        output, contract, label="rearm2"
    )
    if roots != canonical_roots:
        raise ValueError("rearm2 runner returned non-canonical roots")
    value = _materialization_payload(claim, canonical_roots)
    _write_or_validate_durable(
        output / "materialization.json", value, "rearm2 materialization"
    )
    return value


def _root_sets(
    roots: Sequence[Mapping[str, Any]],
) -> tuple[set[str], set[str], set[int]]:
    return rearm1_open._root_sets(roots)


def _load_rearm1_roots(
    inputs: PerformanceLockRearm2Inputs,
) -> tuple[list[dict[str, Any]], list[tuple[Any, Any]]]:
    contract = runner.build_run_contract(
        candidate_library_sha256=CANDIDATE_LIBRARY_SHA256,
        reference_library_sha256=REFERENCE_LIBRARY_SHA256,
        variant=runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT,
    )
    roots, observations = rearm1_open._load_roots(
        Path(inputs.rearm1_root_directory), contract, label="rearm1"
    )
    seal = _read_canonical(
        Path(inputs.rearm1_root_directory) / "seal.json", "rearm1 seal"
    )
    hashes = _root_hashes(roots)
    if (
        hashes != seal.get("root_artifact_sha256")
        or canonical_sha256(hashes) != seal.get("aggregate_root_sha256")
    ):
        raise ValueError("rearm1 roots differ from their seal")
    return roots, observations


def _build_root_seal(inputs: PerformanceLockRearm2Inputs) -> dict[str, Any]:
    claim = validate_global_claim(inputs)
    output = _lock_output_after_claim(inputs)
    materialization = _read_canonical(
        output / "materialization.json", "rearm2 materialization"
    )
    contract = runner.validate_run_contract(claim["lock_run_contract"])
    roots, observations = rearm1_open._load_roots(output, contract, label="rearm2")
    if materialization != _materialization_payload(claim, roots):
        raise ValueError("rearm2 materialization identity changed")
    hashes = _root_hashes(roots)
    profile_counts = Counter(str(root["profile"]) for root in roots)
    expected_profiles = {profile: 20 for profile in M31_T3_BEHAVIOR_PROFILES}
    fingerprints = [
        observation.fingerprint()
        for pair in observations
        for observation in pair
    ]
    seats = Counter(
        str(raw["seat"]) for root in roots for raw in root["observations"]
    )
    if (
        dict(profile_counts) != expected_profiles
        or len(hashes) != 100
        or len(set(hashes)) != 100
        or len(fingerprints) != 200
        or len(set(fingerprints)) != 200
        or dict(seats) != {"first": 100, "second": 100}
    ):
        raise ValueError("rearm2 root balance or uniqueness changed")

    development = development_roots.load_frozen_roots(
        inputs.development_root_directory
    )
    old_v1_roots, _ = rearm1_open._load_old_v1_roots(inputs)
    rearm1_roots, _ = _load_rearm1_roots(inputs)
    current_sets = _root_sets(roots)
    development_sets = _root_sets(development)
    old_v1_sets = _root_sets(old_v1_roots)
    rearm1_sets = _root_sets(rearm1_roots)
    development_overlap = tuple(
        len(left & right)
        for left, right in zip(current_sets, development_sets, strict=True)
    )
    old_v1_overlap = tuple(
        len(left & right)
        for left, right in zip(current_sets, old_v1_sets, strict=True)
    )
    rearm1_overlap = tuple(
        len(left & right)
        for left, right in zip(current_sets, rearm1_sets, strict=True)
    )
    if any((*development_overlap, *old_v1_overlap, *rearm1_overlap)):
        raise ValueError("rearm2 roots overlap development or prior locks")
    topology = rearm1_open.legacy_open._topology_rows(roots, observations)
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
        "prior_lock_comparison": {
            "old_v1_global_claim_sha256": OLD_V1_GLOBAL_CLAIM_SHA256,
            "old_v1_seal_sha256": OLD_V1_SEAL_SHA256,
            "old_v1_root_hash_overlap_count": old_v1_overlap[0],
            "old_v1_fingerprint_overlap_count": old_v1_overlap[1],
            "old_v1_seed_overlap_count": old_v1_overlap[2],
            "rearm1_global_claim_sha256": REARM1_GLOBAL_CLAIM_SHA256,
            "rearm1_seal_sha256": REARM1_SEAL_SHA256,
            "rearm1_root_hash_overlap_count": rearm1_overlap[0],
            "rearm1_fingerprint_overlap_count": rearm1_overlap[1],
            "rearm1_seed_overlap_count": rearm1_overlap[2],
            "rearm1_attempt1_reused": False,
            "rearm1_package_reused": False,
            "rearm1_root_reused": False,
            "rearm1_seed_reused": False,
            "rearm1_claim_reused": False,
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
        "actual_package_smoke_requirement": claim[
            "actual_package_smoke_requirement"
        ],
        "rearm_guards": {
            "incident_receipt_sha256": (
                REARM1_STARTUP_FAILURE_RECEIPT_SHA256
            ),
            "fresh_710_series_seed_schedule": True,
            "same_identity_resume_only": True,
            "old_v1_root_reused": False,
            "rearm1_attempt1_reused": False,
            "rearm1_package_reused": False,
            "rearm1_root_reused": False,
            "rearm1_seed_reused": False,
            "rearm1_claim_reused": False,
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
        raise AssertionError("rearm2 seal schema changed")
    return value


def seal_performance_lock_rearm2(
    inputs: PerformanceLockRearm2Inputs,
) -> dict[str, Any]:
    value = _build_root_seal(inputs)
    output = _lock_output_after_claim(inputs)
    _write_or_validate_durable(output / "seal.json", value, "rearm2 seal")
    return value


def validate_root_seal(
    inputs: PerformanceLockRearm2Inputs,
) -> dict[str, Any]:
    stored = _read_canonical(
        _lock_output_after_claim(inputs) / "seal.json", "rearm2 seal"
    )
    if set(stored) != _SEAL_KEYS or stored != _build_root_seal(inputs):
        raise ValueError("rearm2 seal replay changed")
    return stored


def _inputs_from_args(args: argparse.Namespace) -> PerformanceLockRearm2Inputs:
    return PerformanceLockRearm2Inputs(
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("preflight", "open", "materialize", "seal"),
    )
    parser.add_argument("--repository-root", type=Path, default=DEFAULT_REPOSITORY_ROOT)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PRECONTENT_PLAN_PATH)
    parser.add_argument("--lock-output", type=Path, required=True)
    parser.add_argument("--candidate-library", type=Path, required=True)
    parser.add_argument("--reference-library", type=Path, required=True)
    parser.add_argument("--feature-encoder", type=Path, required=True)
    parser.add_argument("--startup-source", type=Path, required=True)
    parser.add_argument("--incident-receipt", type=Path, required=True)
    parser.add_argument("--old-v1-roots", type=Path, required=True)
    parser.add_argument("--rearm1-roots", type=Path, required=True)
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
        "--development-roots", type=Path, default=DEFAULT_DEVELOPMENT_ROOT_DIR
    )
    parser.add_argument(
        "--old-v1-global-claim",
        type=Path,
        default=DEFAULT_OLD_V1_GLOBAL_CLAIM_PATH,
    )
    parser.add_argument(
        "--rearm1-global-claim",
        type=Path,
        default=DEFAULT_REARM1_GLOBAL_CLAIM_PATH,
    )
    parser.add_argument(
        "--global-claim", type=Path, default=DEFAULT_GLOBAL_CLAIM_PATH
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inputs = _inputs_from_args(args)
    if args.command == "preflight":
        value = preflight_performance_lock_rearm2(inputs)
    elif args.command == "open":
        value = open_performance_lock_rearm2(inputs)
    elif args.command == "materialize":
        value = materialize_performance_lock_rearm2(inputs)
    else:
        value = seal_performance_lock_rearm2(inputs)
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ALLOCATION",
    "CLAIM_SCHEMA",
    "CLAIM_STATUS",
    "DEFAULT_GLOBAL_CLAIM_PATH",
    "DEFAULT_OLD_V1_GLOBAL_CLAIM_PATH",
    "DEFAULT_PRECONTENT_PLAN_PATH",
    "DEFAULT_REARM1_GLOBAL_CLAIM_PATH",
    "IMAGE",
    "LOCK_RUN_CONTRACT_DIGEST",
    "MATERIALIZATION_SCHEMA",
    "MATERIALIZATION_STATUS",
    "OLD_V1_GLOBAL_CLAIM_SHA256",
    "OLD_V1_MATERIALIZATION_SHA256",
    "OLD_V1_SEAL_SHA256",
    "PerformanceLockInputs",
    "PerformanceLockRearm2Inputs",
    "PREFLIGHT_SCHEMA",
    "PREFLIGHT_STATUS",
    "PRECONTENT_PLAN_SHA256",
    "REARM1_GLOBAL_CLAIM_SHA256",
    "REARM1_MATERIALIZATION_SHA256",
    "REARM1_SEAL_SHA256",
    "REARM1_STARTUP_FAILURE_RECEIPT_SHA256",
    "SEAL_SCHEMA",
    "SEAL_STATUS",
    "materialize_performance_lock_rearm2",
    "open_performance_lock_rearm2",
    "preflight_performance_lock_rearm2",
    "seal_performance_lock_rearm2",
    "validate_global_claim",
    "validate_open_claim",
    "validate_root_seal",
    "_CLAIM_KEYS",
    "_MATERIALIZATION_KEYS",
    "_SEAL_KEYS",
]
