"""Freeze, materialize, and seal the run009-authorized T3 performance lock.

This is a local scientific lifecycle only.  It consumes the immutable run009
scientific-gate receipt, creates a fresh 100-paired-hand lock contract, and
keeps root generation behind a write-once claim.  It cannot launch cloud
resources, train, promote, activate a profile, or resolve ``current``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import hu_m31_t3_step6d_full100_wave_scientific_bridge_v2 as run009_bridge
from . import run_hu_m31_t3_step6d_performance_v2 as runner
from .hu_m31_t3_behavior_roots import M31_T3_BEHAVIOR_PROFILES


PLAN_SCHEMA = "hu_m31_t3_step6d_candidate02_performance_lock_plan_v4"
PLAN_STATUS = "frozen_preregistered_plan_roots_unopened"
PLAN_DECISION = (
    "run009_performance_go_freezes_fresh_one_shot_performance_lock_v4_only"
)
PLAN_SCOPE = "one_shot_performance_lock_v4"
ASSIGNMENT_METHOD = "consecutive_ten_hand_arithmetic_v4"

CLAIM_SCHEMA = "hu_m31_t3_step6d_candidate02_performance_lock_claim_v4"
CLAIM_STATUS = "write_once_claim_persisted_roots_may_resume"
MATERIALIZATION_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_lock_materialization_v4"
)
MATERIALIZATION_STATUS = "exactly_100_valid_paired_roots_materialized"
SEAL_SCHEMA = "hu_m31_t3_step6d_candidate02_performance_lock_root_seal_v4"
SEAL_STATUS = "immutable_100_paired_root_set_sealed"

RUN009_GATE_RECEIPT_FILE_SHA256 = (
    "664f86262436d41b62a8324cf1ba8df52e0f6af09f80f2071af825862b39e6b2"
)
RUN009_GATE_RECEIPT_SHA256 = (
    "40bbad13cfc4a0680f7bf88027fbf173b848e86350cc5895c116708c6ad2c04d"
)
RUN009_RUN_NAME = "regular-hu-m31-c02-f100wv2-20260723-009"
RUN009_EXECUTION_IDENTITY_SHA256 = (
    "dd88e7361639f3ada186b0cfa4661681d4c715465dc9b9b7562523221bc902ba"
)
RUN009_WAVE_PLAN_SHA256 = (
    "4db219a6f2b48d8828823a8772571b7ce695a6fd92cd0a20ebb6d27d84b19b22"
)
RUN009_RUN_CONTRACT_DIGEST = (
    "92a87f1544a73104d5256db39b022094c8c7f7b11f77bd19dcf1ab1442581ebd"
)
RUN009_SCIENTIFIC_MERGE_SHA256 = (
    "badfe05d3cd9df0fb7ebac09ac4ff289dfa0003864db627b761a2982f7d2792c"
)
RUN009_RECEIVER_RECEIPT_SHA256 = (
    "8d8202f4f9803b3cb4ca155efdc9f7e6ae6214e7e01d58145fd374ef78c467ab"
)
RUN009_RNG_NAMESPACE_CONTRACT_SHA256 = (
    "f31e771d642ee6f0e36ed7adf48f156a8fc4ea65655d8648590258397c86ce2c"
)
RUN009_PAIR_LAUNCH_LINEAGE_SHA256 = (
    "1315489ea1767c4a21acde6049f839671f215cd58c2cc5cfc2b5f1aacf2c20e8"
)
RUN009_PERFORMANCE_GATE_SHA256 = (
    "fcc28ec22f37857b92ce62b8203901930c56bcfa719ceddd489b1316f7fb48cd"
)
RUN009_ACCEPTED_SNAPSHOT_SHA256 = (
    "78304b2d90559e86727aa074c679ad89ccc980afbde4404d3140fd13f3474da9"
)
RUN009_EXPECTED_INVENTORY_SHA256 = (
    "e707e8611630ec9e2bfa8c617eb59598b4cac712763870a40f2620937ebb6a5c"
)
RUN009_OBSERVED_INVENTORY_SHA256 = (
    "eaaf9e97fe32046dacb3e79c35136bf21a96c73567aa2e1c79953cbc7c1f6573"
)
RUN009_PACKAGE_SHA256 = (
    "771b77a276e4b63070643db2ed3b4a05ad488b5ffd93ca43e5865ea18637df91"
)
RUN009_IMAGE_DIGEST = (
    "sha256:9dd85299f559ea3b143b1a764a9c69e0e535672036c2b45bf1cff25b88da3c0d"
)
RUN009_ALLOCATION_DIGEST = (
    "eadb033bb2ba8c95d3994dd0acc42af47452efb42d9e00e7ae6aa5d7226837a5"
)

CANDIDATE_LIBRARY_SHA256 = (
    "4050e04b22d7943da9e1691402a2b34cf3d3781041a44557f35e2dfcf366d55d"
)
REFERENCE_LIBRARY_SHA256 = (
    "3f831615d9e751b8e1f266f14dd2009903b3ac2a4094f7e47616e0aa372a01b0"
)
CURRENT_PROFILE_REGISTRY_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)
RUN_CONTRACT_DIGEST = (
    "669c1efa1afeebe41fcc531c6458c9d72fffdd5df2cca751c99988a872f3e2b6"
)
PLAN_SHA256 = "2ad08116835a58f5b5927e4de986f2717915d0dd128e7a5f3fa288e0cac6e5be"

SOURCE_ROLES = tuple(runner.SOURCE_ROLES)
SHARD_COUNT_PER_ROLE = 10
HANDS_PER_SHARD = 10
HANDS_PER_PROFILE_PER_SHARD = 2
LOGICAL_JOB_COUNT = len(SOURCE_ROLES) * SHARD_COUNT_PER_ROLE

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN009_SCIENTIFIC_GATE_RECEIPT_PATH = Path(
    r"D:\ofc-gcp-runs\regular-hu-m31-c02-f100wv2-20260723-009"
) / "scientific-gate-receipt.json"
DEFAULT_CURRENT_PROFILE_REGISTRY_PATH = _REPO_ROOT / "src/ofc_regular/ai_profiles.py"
DEFAULT_PLAN_PATH = (
    _REPO_ROOT
    / "outputs/hu_joint_policy/m31_t3_step6d/performance_lock_v4/"
    "performance_lock_v4_plan.json"
)
DEFAULT_ROOT_OUTPUT_DIR = (
    _REPO_ROOT
    / "outputs/hu_joint_policy/m31_t3_step6d/performance_lock_v4/execution"
)
DEFAULT_MATERIALIZATION_RECEIPT_PATH = (
    DEFAULT_ROOT_OUTPUT_DIR / "MATERIALIZATION_RECEIPT.json"
)
DEFAULT_ROOT_SEAL_PATH = DEFAULT_ROOT_OUTPUT_DIR / "ROOT_SEAL.json"
CLAIM_NAME = "MATERIALIZATION_CLAIM.json"

_EXPECTED_BUDGET = {
    "candidate_samples": 8,
    "evaluation_samples": 32,
    "downstream_t3_samples": 4,
    "downstream_t4_samples": 0,
}
_EXPECTED_ALLOCATION = {"workers": 1, "rayon_threads_per_worker": 16}
_EXPECTED_GATE_NAMES = frozenset(
    {
        "exactly_100_first_and_100_second",
        "exactly_100_paired_hands",
        "exactly_200_roots",
        "exactly_20_hands_per_profile",
        "first_max_within_240_seconds",
        "first_p95_within_150_seconds",
        "first_p99_within_240_seconds",
        "missing_or_censored_roots_zero",
        "peak_rss_within_858993459_bytes",
        "portable_semantic_parity_fraction_one",
        "second_p95_within_5_seconds",
    }
)

_PLAN_KEYS = frozenset(
    {
        "schema",
        "status",
        "decision",
        "scope",
        "run009_scientific_gate",
        "candidate_variant",
        "run_contract",
        "run_contract_digest",
        "source_identity",
        "allocation",
        "seed_contract",
        "current_profile_registry",
        "root_set",
        "assignment",
        "source_roles",
        "shard_count_per_role",
        "hands_per_shard",
        "logical_job_count",
        "shards",
        "jobs",
        "open_claim_required_before_root_content",
        "root_content_opened",
        "cloud_started",
        "quality_pilot_authorized",
        "artifact_fanout_authorized",
        "training_authorized",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
        "m31_complete",
    }
)
_GATE_BINDING_KEYS = frozenset(
    {
        "receipt_file_sha256",
        "receipt_sha256",
        "schema",
        "status",
        "decision",
        "run_name",
        "execution_identity_sha256",
        "wave_plan_sha256",
        "run_contract_digest",
        "scientific_merge_sha256",
        "receiver_receipt_sha256",
        "rng_namespace_contract_sha256",
        "pair_launch_lineage_sha256",
        "performance_gate_sha256",
        "accepted_snapshot_sha256",
        "expected_inventory_sha256",
        "observed_inventory_sha256",
        "candidate_library_sha256",
        "reference_library_sha256",
        "budget",
        "allocation",
        "package_sha256",
        "image_digest",
        "allocation_digest",
        "performance_gate_names",
        "all_gates_passed",
        "performance_candidate_frozen",
        "performance_lock_authorized",
        "quality_pilot_authorized",
        "artifact_fanout_authorized",
        "training_authorized",
        "current_profile_changed",
    }
)
_SHARD_KEYS = frozenset({"shard_index", "work_hand_indices", "profile_counts"})
_JOB_KEYS = frozenset(
    {
        "job_id",
        "source_role",
        "shard_index",
        "work_hand_indices",
        "shard_manifest_sha256",
    }
)


def canonical_bytes(value: Any) -> bytes:
    return runner.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return runner.canonical_sha256(value)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_canonical(path: str | Path, label: str) -> dict[str, Any]:
    target = Path(path).resolve()
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    raw = target.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _write_once(path: str | Path, value: Mapping[str, Any], label: str) -> None:
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(f"{label} already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    except FileExistsError as exc:
        raise FileExistsError(f"{label} already exists: {target}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _write_once_or_validate(
    path: str | Path, value: Mapping[str, Any], label: str
) -> None:
    target = Path(path).resolve()
    if target.exists():
        if _read_canonical(target, label) != dict(value):
            raise FileExistsError(f"{label} exists with different content: {target}")
        return
    _write_once(target, value, label)


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys changed")


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _run_contract() -> dict[str, Any]:
    contract = runner.build_run_contract(
        candidate_library_sha256=CANDIDATE_LIBRARY_SHA256,
        reference_library_sha256=REFERENCE_LIBRARY_SHA256,
        workers=1,
        rayon_threads_per_worker=16,
        variant=runner.CANDIDATE02_PERFORMANCE_LOCK_V4_VARIANT,
    )
    if canonical_sha256(contract) != RUN_CONTRACT_DIGEST:
        raise RuntimeError("performance-lock-v4 run contract digest changed")
    return contract


def load_run009_scientific_gate_receipt(
    path: str | Path = DEFAULT_RUN009_SCIENTIFIC_GATE_RECEIPT_PATH,
) -> dict[str, Any]:
    """Validate the exact immutable run009 receipt and return its compact binding."""

    target = Path(path).resolve()
    if sha256_file(target) != RUN009_GATE_RECEIPT_FILE_SHA256:
        raise ValueError("run009 scientific gate receipt file hash changed")
    receipt = run009_bridge.validate_scientific_gate_receipt_value(
        _read_canonical(target, "run009 scientific gate receipt")
    )
    merge = receipt.get("scientific_merge")
    wave_plan = receipt.get("wave_plan")
    performance_gate = receipt.get("performance_gate")
    if not all(
        isinstance(value, Mapping)
        for value in (merge, wave_plan, performance_gate)
    ):
        raise ValueError("run009 scientific gate evidence is incomplete")
    runtime_binding = wave_plan.get("runtime_binding")
    gates = performance_gate.get("gates")
    if not isinstance(runtime_binding, Mapping) or not isinstance(gates, Mapping):
        raise ValueError("run009 runtime or performance gate binding is missing")
    binding = {
        "receipt_file_sha256": RUN009_GATE_RECEIPT_FILE_SHA256,
        "receipt_sha256": receipt.get("receipt_sha256"),
        "schema": receipt.get("schema"),
        "status": receipt.get("status"),
        "decision": receipt.get("decision"),
        "run_name": receipt.get("run_name"),
        "execution_identity_sha256": receipt.get("execution_identity_sha256"),
        "wave_plan_sha256": receipt.get("wave_plan_sha256"),
        "run_contract_digest": receipt.get("run_contract_digest"),
        "scientific_merge_sha256": receipt.get("scientific_merge_sha256"),
        "receiver_receipt_sha256": receipt.get("receiver_receipt_sha256"),
        "rng_namespace_contract_sha256": receipt.get(
            "rng_namespace_contract_sha256"
        ),
        "pair_launch_lineage_sha256": receipt.get("pair_launch_lineage_sha256"),
        "performance_gate_sha256": receipt.get("performance_gate_sha256"),
        "accepted_snapshot_sha256": receipt.get("accepted_snapshot_sha256"),
        "expected_inventory_sha256": receipt.get("expected_inventory_sha256"),
        "observed_inventory_sha256": receipt.get("observed_inventory_sha256"),
        "candidate_library_sha256": merge.get("candidate_library_sha256"),
        "reference_library_sha256": merge.get("reference_library_sha256"),
        "budget": merge.get("budget"),
        "allocation": merge.get("allocation"),
        "package_sha256": runtime_binding.get("package_sha256"),
        "image_digest": runtime_binding.get("image_digest"),
        "allocation_digest": runtime_binding.get("allocation_digest"),
        "performance_gate_names": sorted(gates),
        "all_gates_passed": receipt.get("all_gates_passed"),
        "performance_candidate_frozen": receipt.get(
            "performance_candidate_frozen"
        ),
        "performance_lock_authorized": receipt.get("performance_lock_authorized"),
        "quality_pilot_authorized": receipt.get("quality_pilot_authorized"),
        "artifact_fanout_authorized": receipt.get("artifact_fanout_authorized"),
        "training_authorized": receipt.get("training_authorized"),
        "current_profile_changed": receipt.get("current_profile_changed"),
    }
    _validate_gate_binding(binding)
    return binding


def _validate_gate_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    binding = dict(value)
    _exact_keys(binding, _GATE_BINDING_KEYS, "run009 gate binding")
    expected = {
        "receipt_file_sha256": RUN009_GATE_RECEIPT_FILE_SHA256,
        "receipt_sha256": RUN009_GATE_RECEIPT_SHA256,
        "schema": "hu_m31_t3_step6d_full100_wave_scientific_gate_receipt_v2",
        "status": "pass",
        "decision": "full100_wave_v2_go_open_one_shot_performance_lock_only",
        "run_name": RUN009_RUN_NAME,
        "execution_identity_sha256": RUN009_EXECUTION_IDENTITY_SHA256,
        "wave_plan_sha256": RUN009_WAVE_PLAN_SHA256,
        "run_contract_digest": RUN009_RUN_CONTRACT_DIGEST,
        "scientific_merge_sha256": RUN009_SCIENTIFIC_MERGE_SHA256,
        "receiver_receipt_sha256": RUN009_RECEIVER_RECEIPT_SHA256,
        "rng_namespace_contract_sha256": RUN009_RNG_NAMESPACE_CONTRACT_SHA256,
        "pair_launch_lineage_sha256": RUN009_PAIR_LAUNCH_LINEAGE_SHA256,
        "performance_gate_sha256": RUN009_PERFORMANCE_GATE_SHA256,
        "accepted_snapshot_sha256": RUN009_ACCEPTED_SNAPSHOT_SHA256,
        "expected_inventory_sha256": RUN009_EXPECTED_INVENTORY_SHA256,
        "observed_inventory_sha256": RUN009_OBSERVED_INVENTORY_SHA256,
        "candidate_library_sha256": CANDIDATE_LIBRARY_SHA256,
        "reference_library_sha256": REFERENCE_LIBRARY_SHA256,
        "budget": _EXPECTED_BUDGET,
        "allocation": _EXPECTED_ALLOCATION,
        "package_sha256": RUN009_PACKAGE_SHA256,
        "image_digest": RUN009_IMAGE_DIGEST,
        "allocation_digest": RUN009_ALLOCATION_DIGEST,
        "performance_gate_names": sorted(_EXPECTED_GATE_NAMES),
        "all_gates_passed": True,
        "performance_candidate_frozen": True,
        "performance_lock_authorized": True,
        "quality_pilot_authorized": False,
        "artifact_fanout_authorized": False,
        "training_authorized": False,
        "current_profile_changed": False,
    }
    if binding != expected:
        raise ValueError("run009 scientific gate binding changed")
    return binding


def _arithmetic_shards() -> list[dict[str, Any]]:
    shards = []
    profiles = tuple(M31_T3_BEHAVIOR_PROFILES)
    for shard_index in range(SHARD_COUNT_PER_ROLE):
        indices = list(
            range(shard_index * HANDS_PER_SHARD, (shard_index + 1) * HANDS_PER_SHARD)
        )
        counts = Counter(runner.v1.behavior_profile_for_index(index) for index in indices)
        if set(counts) != set(profiles) or any(
            counts[profile] != HANDS_PER_PROFILE_PER_SHARD for profile in profiles
        ):
            raise RuntimeError("performance-lock-v4 shard profile balance changed")
        shards.append(
            {
                "shard_index": shard_index,
                "work_hand_indices": indices,
                "profile_counts": {profile: counts[profile] for profile in profiles},
            }
        )
    return shards


def _jobs(
    contract: Mapping[str, Any], shards: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    jobs = []
    for role in SOURCE_ROLES:
        for shard in shards:
            manifest = runner.build_shard_manifest(
                run_contract=contract,
                source_role=role,
                work_hand_indices=shard["work_hand_indices"],
            )
            jobs.append(
                {
                    "job_id": f"{role}-shard-{int(shard['shard_index']):02d}",
                    "source_role": role,
                    "shard_index": shard["shard_index"],
                    "work_hand_indices": list(shard["work_hand_indices"]),
                    "shard_manifest_sha256": canonical_sha256(manifest),
                }
            )
    return jobs


def build_performance_lock_v4_plan(
    *,
    run009_scientific_gate_receipt_path: str
    | Path = DEFAULT_RUN009_SCIENTIFIC_GATE_RECEIPT_PATH,
    current_profile_registry_path: str | Path = DEFAULT_CURRENT_PROFILE_REGISTRY_PATH,
) -> dict[str, Any]:
    """Build the deterministic pre-root plan from the exact run009 Go receipt."""

    gate = load_run009_scientific_gate_receipt(
        run009_scientific_gate_receipt_path
    )
    registry = Path(current_profile_registry_path).resolve()
    if registry.is_symlink() or not registry.is_file():
        raise ValueError("current profile registry is missing or unsafe")
    if sha256_file(registry) != CURRENT_PROFILE_REGISTRY_SHA256:
        raise ValueError("current profile registry hash changed")
    contract = _run_contract()
    shards = _arithmetic_shards()
    value = {
        "schema": PLAN_SCHEMA,
        "status": PLAN_STATUS,
        "decision": PLAN_DECISION,
        "scope": PLAN_SCOPE,
        "run009_scientific_gate": gate,
        "candidate_variant": runner.CANDIDATE02_PERFORMANCE_LOCK_V4_VARIANT,
        "run_contract": contract,
        "run_contract_digest": RUN_CONTRACT_DIGEST,
        "source_identity": {
            "candidate_library_sha256": gate["candidate_library_sha256"],
            "reference_library_sha256": gate["reference_library_sha256"],
            "package_sha256": gate["package_sha256"],
            "image_digest": gate["image_digest"],
            "allocation_digest": gate["allocation_digest"],
        },
        "allocation": dict(gate["allocation"]),
        "seed_contract": runner.candidate02_performance_lock_v4_seed_contract(),
        "current_profile_registry": {
            "path": "src/ofc_regular/ai_profiles.py",
            "sha256": CURRENT_PROFILE_REGISTRY_SHA256,
            "must_remain_unchanged": True,
        },
        "root_set": {
            "schema": runner.CANDIDATE02_PERFORMANCE_LOCK_V4_ROOT_SCHEMA,
            "hand_indices": list(runner.CONTRACT_HAND_INDICES),
            "paired_hand_count": 100,
            "root_file_count": 100,
            "observation_count": 200,
            "first_seat_count": 100,
            "second_seat_count": 100,
            "claim_name": CLAIM_NAME,
            "materialization_receipt_name": (
                DEFAULT_MATERIALIZATION_RECEIPT_PATH.name
            ),
            "root_seal_name": DEFAULT_ROOT_SEAL_PATH.name,
            "write_once": True,
            "resume_requires_identical_claim": True,
            "old_root_reuse_allowed": False,
            "root_content_opened": False,
        },
        "assignment": {
            "method": ASSIGNMENT_METHOD,
            "input_scope": "hand_index_and_frozen_behavior_profile_only",
            "timing_used": False,
            "memory_used": False,
            "teacher_values_used": False,
            "runtime_results_used": False,
        },
        "source_roles": list(SOURCE_ROLES),
        "shard_count_per_role": SHARD_COUNT_PER_ROLE,
        "hands_per_shard": HANDS_PER_SHARD,
        "logical_job_count": LOGICAL_JOB_COUNT,
        "shards": shards,
        "jobs": _jobs(contract, shards),
        "open_claim_required_before_root_content": True,
        "root_content_opened": False,
        "cloud_started": False,
        "quality_pilot_authorized": False,
        "artifact_fanout_authorized": False,
        "training_authorized": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
        "m31_complete": False,
    }
    return validate_performance_lock_v4_plan(value)


def validate_performance_lock_v4_plan(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    plan = dict(value)
    _exact_keys(plan, _PLAN_KEYS, "performance-lock-v4 plan")
    gate_raw = plan.get("run009_scientific_gate")
    contract_raw = plan.get("run_contract")
    if not isinstance(gate_raw, Mapping) or not isinstance(contract_raw, Mapping):
        raise ValueError("performance-lock-v4 plan evidence is missing")
    gate = _validate_gate_binding(gate_raw)
    contract = runner.validate_run_contract(contract_raw)
    shards = plan.get("shards")
    jobs = plan.get("jobs")
    expected_shards = _arithmetic_shards()
    expected_jobs = _jobs(contract, expected_shards)
    if not isinstance(shards, list) or not isinstance(jobs, list):
        raise ValueError("performance-lock-v4 shards or jobs are missing")
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise ValueError("performance-lock-v4 shard is not an object")
        _exact_keys(shard, _SHARD_KEYS, "performance-lock-v4 shard")
    for job in jobs:
        if not isinstance(job, Mapping):
            raise ValueError("performance-lock-v4 job is not an object")
        _exact_keys(job, _JOB_KEYS, "performance-lock-v4 job")
    seed_contract = runner.candidate02_performance_lock_v4_seed_contract()
    expected_root_set = {
        "schema": runner.CANDIDATE02_PERFORMANCE_LOCK_V4_ROOT_SCHEMA,
        "hand_indices": list(runner.CONTRACT_HAND_INDICES),
        "paired_hand_count": 100,
        "root_file_count": 100,
        "observation_count": 200,
        "first_seat_count": 100,
        "second_seat_count": 100,
        "claim_name": CLAIM_NAME,
        "materialization_receipt_name": DEFAULT_MATERIALIZATION_RECEIPT_PATH.name,
        "root_seal_name": DEFAULT_ROOT_SEAL_PATH.name,
        "write_once": True,
        "resume_requires_identical_claim": True,
        "old_root_reuse_allowed": False,
        "root_content_opened": False,
    }
    expected_source_identity = {
        "candidate_library_sha256": CANDIDATE_LIBRARY_SHA256,
        "reference_library_sha256": REFERENCE_LIBRARY_SHA256,
        "package_sha256": RUN009_PACKAGE_SHA256,
        "image_digest": RUN009_IMAGE_DIGEST,
        "allocation_digest": RUN009_ALLOCATION_DIGEST,
    }
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("status") != PLAN_STATUS
        or plan.get("decision") != PLAN_DECISION
        or plan.get("scope") != PLAN_SCOPE
        or gate != plan.get("run009_scientific_gate")
        or plan.get("candidate_variant")
        != runner.CANDIDATE02_PERFORMANCE_LOCK_V4_VARIANT
        or contract != plan.get("run_contract")
        or canonical_sha256(contract) != RUN_CONTRACT_DIGEST
        or plan.get("run_contract_digest") != RUN_CONTRACT_DIGEST
        or plan.get("source_identity") != expected_source_identity
        or plan.get("allocation") != _EXPECTED_ALLOCATION
        or plan.get("seed_contract") != seed_contract
        or plan.get("current_profile_registry")
        != {
            "path": "src/ofc_regular/ai_profiles.py",
            "sha256": CURRENT_PROFILE_REGISTRY_SHA256,
            "must_remain_unchanged": True,
        }
        or plan.get("root_set") != expected_root_set
        or plan.get("assignment")
        != {
            "method": ASSIGNMENT_METHOD,
            "input_scope": "hand_index_and_frozen_behavior_profile_only",
            "timing_used": False,
            "memory_used": False,
            "teacher_values_used": False,
            "runtime_results_used": False,
        }
        or plan.get("source_roles") != list(SOURCE_ROLES)
        or plan.get("shard_count_per_role") != SHARD_COUNT_PER_ROLE
        or plan.get("hands_per_shard") != HANDS_PER_SHARD
        or plan.get("logical_job_count") != LOGICAL_JOB_COUNT
        or shards != expected_shards
        or jobs != expected_jobs
        or plan.get("open_claim_required_before_root_content") is not True
        or any(
            plan.get(field) is not False
            for field in (
                "root_content_opened",
                "cloud_started",
                "quality_pilot_authorized",
                "artifact_fanout_authorized",
                "training_authorized",
                "current_profile_changed",
                "named_profile_added",
                "runtime_policy_activated",
                "m31_complete",
            )
        )
    ):
        raise ValueError("performance-lock-v4 plan boundary changed")
    if PLAN_SHA256 and canonical_sha256(plan) != PLAN_SHA256:
        raise ValueError("performance-lock-v4 plan digest changed")
    plan["run009_scientific_gate"] = gate
    plan["run_contract"] = contract
    return plan


def write_performance_lock_v4_plan(
    *,
    output_path: str | Path = DEFAULT_PLAN_PATH,
    run009_scientific_gate_receipt_path: str
    | Path = DEFAULT_RUN009_SCIENTIFIC_GATE_RECEIPT_PATH,
    current_profile_registry_path: str | Path = DEFAULT_CURRENT_PROFILE_REGISTRY_PATH,
) -> dict[str, Any]:
    value = build_performance_lock_v4_plan(
        run009_scientific_gate_receipt_path=run009_scientific_gate_receipt_path,
        current_profile_registry_path=current_profile_registry_path,
    )
    _write_once(output_path, value, "performance-lock-v4 plan")
    return value


def _load_plan_and_authorization(
    *,
    plan_path: str | Path,
    run009_scientific_gate_receipt_path: str | Path,
    current_profile_registry_path: str | Path,
) -> tuple[dict[str, Any], str]:
    plan = validate_performance_lock_v4_plan(
        _read_canonical(plan_path, "performance-lock-v4 plan")
    )
    if canonical_sha256(plan) != PLAN_SHA256:
        raise ValueError("stored performance-lock-v4 plan hash changed")
    gate = load_run009_scientific_gate_receipt(
        run009_scientific_gate_receipt_path
    )
    if gate != plan["run009_scientific_gate"]:
        raise ValueError("run009 gate receipt differs from frozen v4 plan")
    registry_sha = sha256_file(current_profile_registry_path)
    if registry_sha != CURRENT_PROFILE_REGISTRY_SHA256:
        raise ValueError("current profile registry changed")
    return plan, registry_sha


_AUDIT_KEYS = frozenset(
    {
        "root_file_count",
        "paired_hand_count",
        "observation_count",
        "root_hashes",
        "root_hash_aggregate_sha256",
        "profile_counts",
        "seat_counts",
        "seed_set_sha256",
        "seed_overlap_counts",
        "observation_fingerprint_count",
        "observation_fingerprint_aggregate_sha256",
        "all_root_artifacts_valid",
        "current_profile_unchanged",
        "old_root_or_seed_reuse",
    }
)
_CLAIM_KEYS = frozenset(
    {
        "schema",
        "status",
        "plan_sha256",
        "run_contract_digest",
        "authorizing_gate_receipt_file_sha256",
        "authorizing_gate_receipt_sha256",
        "root_output_directory",
        "current_profile_registry_sha256",
        "root_content_opened_at_claim",
        "cloud_started",
        "training_authorized",
        "current_profile_changed",
    }
)
_MATERIALIZATION_KEYS = frozenset(
    {
        "schema",
        "status",
        "plan_sha256",
        "run_contract_digest",
        "claim_sha256",
        "authorizing_gate_receipt_file_sha256",
        "authorizing_gate_receipt_sha256",
        "root_output_directory",
        *_AUDIT_KEYS,
        "cloud_started",
        "quality_pilot_authorized",
        "training_authorized",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
    }
)
_SEAL_KEYS = frozenset(
    {
        "schema",
        "status",
        "plan_sha256",
        "run_contract_digest",
        "claim_sha256",
        "materialization_receipt_sha256",
        "authorizing_gate_receipt_file_sha256",
        "authorizing_gate_receipt_sha256",
        "root_output_directory",
        *_AUDIT_KEYS,
        "sealed",
        "cloud_started",
        "quality_pilot_authorized",
        "training_authorized",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
    }
)


def _claim_value(output_dir: Path) -> dict[str, Any]:
    return {
        "schema": CLAIM_SCHEMA,
        "status": CLAIM_STATUS,
        "plan_sha256": PLAN_SHA256,
        "run_contract_digest": RUN_CONTRACT_DIGEST,
        "authorizing_gate_receipt_file_sha256": (
            RUN009_GATE_RECEIPT_FILE_SHA256
        ),
        "authorizing_gate_receipt_sha256": RUN009_GATE_RECEIPT_SHA256,
        "root_output_directory": str(output_dir.resolve()),
        "current_profile_registry_sha256": CURRENT_PROFILE_REGISTRY_SHA256,
        "root_content_opened_at_claim": False,
        "cloud_started": False,
        "training_authorized": False,
        "current_profile_changed": False,
    }


def _validate_claim(value: Mapping[str, Any], *, output_dir: Path) -> dict[str, Any]:
    claim = dict(value)
    _exact_keys(claim, _CLAIM_KEYS, "performance-lock-v4 claim")
    if claim != _claim_value(output_dir):
        raise ValueError("performance-lock-v4 claim changed")
    return claim


def _seed_overlap_counts() -> dict[str, int]:
    seed_contract = runner.candidate02_performance_lock_v4_seed_contract()
    return {
        "existing_step6d_union": seed_contract[
            "existing_step6d_union_overlap_count"
        ],
        "candidate02_development_run009": seed_contract[
            "candidate02_development_overlap_count"
        ],
        "performance_lock_v1": seed_contract["performance_lock_v1_overlap_count"],
        "performance_lock_rearm1": seed_contract[
            "performance_lock_rearm1_overlap_count"
        ],
        "performance_lock_rearm2": seed_contract[
            "performance_lock_rearm2_overlap_count"
        ],
    }


def _audit_roots(
    *,
    roots: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    current_profile_unchanged: bool,
    validate_artifacts: bool = True,
) -> dict[str, Any]:
    if len(roots) != 100:
        raise ValueError("performance-lock-v4 requires exactly 100 root files")
    root_hashes: list[str] = []
    fingerprints: list[str] = []
    profiles: Counter[str] = Counter()
    seats: Counter[str] = Counter()
    seed_values: list[int] = []
    seen_indices: set[int] = set()
    for expected_index, raw in enumerate(roots):
        if not isinstance(raw, Mapping):
            raise ValueError("performance-lock-v4 root is not an object")
        index = raw.get("hand_index")
        if index != expected_index or index in seen_indices:
            raise ValueError("performance-lock-v4 root index order changed")
        seen_indices.add(index)
        if validate_artifacts:
            runner._validate_root_artifact(contract, raw, index=expected_index)
        root_hashes.append(canonical_sha256(raw))
        profiles[str(raw["profile"])] += 1
        seed_values.extend(int(seed) for seed in raw["seeds"].values())
        observations = raw.get("observations")
        if not isinstance(observations, list) or len(observations) != 2:
            raise ValueError("performance-lock-v4 root observation count changed")
        for observation in observations:
            seats[str(observation["seat"])] += 1
            fingerprints.append(str(observation["observation_fingerprint"]))
    overlap_counts = _seed_overlap_counts()
    expected_profiles = {profile: 20 for profile in M31_T3_BEHAVIOR_PROFILES}
    expected_seats = {"first": 100, "second": 100}
    seed_digest = runner.contract_canonical_sha256(sorted(seed_values))
    if (
        len(root_hashes) != len(set(root_hashes))
        or len(fingerprints) != 200
        or len(fingerprints) != len(set(fingerprints))
        or dict(profiles) != expected_profiles
        or dict(seats) != expected_seats
        or len(seed_values) != 600
        or len(seed_values) != len(set(seed_values))
        or seed_digest != runner.CANDIDATE02_PERFORMANCE_LOCK_V4_SEED_SET_SHA256
        or any(overlap_counts.values())
        or current_profile_unchanged is not True
    ):
        raise ValueError("performance-lock-v4 root-set audit failed")
    return {
        "root_file_count": 100,
        "paired_hand_count": 100,
        "observation_count": 200,
        "root_hashes": root_hashes,
        "root_hash_aggregate_sha256": canonical_sha256(root_hashes),
        "profile_counts": expected_profiles,
        "seat_counts": expected_seats,
        "seed_set_sha256": seed_digest,
        "seed_overlap_counts": overlap_counts,
        "observation_fingerprint_count": 200,
        "observation_fingerprint_aggregate_sha256": canonical_sha256(
            sorted(fingerprints)
        ),
        "all_root_artifacts_valid": True,
        "current_profile_unchanged": True,
        "old_root_or_seed_reuse": False,
    }


def _validate_audit(value: Mapping[str, Any]) -> dict[str, Any]:
    audit = dict(value)
    _exact_keys(audit, _AUDIT_KEYS, "performance-lock-v4 root audit")
    root_hashes = audit.get("root_hashes")
    if (
        audit.get("root_file_count") != 100
        or audit.get("paired_hand_count") != 100
        or audit.get("observation_count") != 200
        or not isinstance(root_hashes, list)
        or len(root_hashes) != 100
        or len(root_hashes) != len(set(root_hashes))
        or not all(_is_sha256(value) for value in root_hashes)
        or audit.get("root_hash_aggregate_sha256")
        != canonical_sha256(root_hashes)
        or audit.get("profile_counts")
        != {profile: 20 for profile in M31_T3_BEHAVIOR_PROFILES}
        or audit.get("seat_counts") != {"first": 100, "second": 100}
        or audit.get("seed_set_sha256")
        != runner.CANDIDATE02_PERFORMANCE_LOCK_V4_SEED_SET_SHA256
        or audit.get("seed_overlap_counts") != _seed_overlap_counts()
        or audit.get("observation_fingerprint_count") != 200
        or not _is_sha256(
            audit.get("observation_fingerprint_aggregate_sha256")
        )
        or audit.get("all_root_artifacts_valid") is not True
        or audit.get("current_profile_unchanged") is not True
        or audit.get("old_root_or_seed_reuse") is not False
    ):
        raise ValueError("performance-lock-v4 root audit changed")
    return audit


def _load_root_files(
    *, output_dir: Path, contract: Mapping[str, Any]
) -> list[dict[str, Any]]:
    root_dir = output_dir / "roots"
    if root_dir.is_symlink() or not root_dir.is_dir():
        raise ValueError("performance-lock-v4 root directory is missing or unsafe")
    expected_names = {f"hand_{index:03d}.json" for index in range(100)}
    actual = {path.name for path in root_dir.iterdir()}
    if actual != expected_names:
        raise ValueError("performance-lock-v4 root file inventory changed")
    roots = []
    for index in range(100):
        path = root_dir / f"hand_{index:03d}.json"
        value = _read_canonical(path, f"performance-lock-v4 root {index}")
        runner._validate_root_artifact(contract, value, index=index)
        roots.append(value)
    return roots


def _materialization_value(
    *,
    output_dir: Path,
    claim: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": MATERIALIZATION_SCHEMA,
        "status": MATERIALIZATION_STATUS,
        "plan_sha256": PLAN_SHA256,
        "run_contract_digest": RUN_CONTRACT_DIGEST,
        "claim_sha256": canonical_sha256(claim),
        "authorizing_gate_receipt_file_sha256": (
            RUN009_GATE_RECEIPT_FILE_SHA256
        ),
        "authorizing_gate_receipt_sha256": RUN009_GATE_RECEIPT_SHA256,
        "root_output_directory": str(output_dir.resolve()),
        **dict(audit),
        "cloud_started": False,
        "quality_pilot_authorized": False,
        "training_authorized": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
    }


def validate_materialization_receipt(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = dict(value)
    _exact_keys(receipt, _MATERIALIZATION_KEYS, "performance-lock-v4 materialization")
    audit = _validate_audit(
        {key: receipt[key] for key in _AUDIT_KEYS if key in receipt}
    )
    output = receipt.get("root_output_directory")
    if (
        receipt.get("schema") != MATERIALIZATION_SCHEMA
        or receipt.get("status") != MATERIALIZATION_STATUS
        or receipt.get("plan_sha256") != PLAN_SHA256
        or receipt.get("run_contract_digest") != RUN_CONTRACT_DIGEST
        or not _is_sha256(receipt.get("claim_sha256"))
        or receipt.get("authorizing_gate_receipt_file_sha256")
        != RUN009_GATE_RECEIPT_FILE_SHA256
        or receipt.get("authorizing_gate_receipt_sha256")
        != RUN009_GATE_RECEIPT_SHA256
        or not isinstance(output, str)
        or not output
        or any(
            receipt.get(field) is not False
            for field in (
                "cloud_started",
                "quality_pilot_authorized",
                "training_authorized",
                "current_profile_changed",
                "named_profile_added",
                "runtime_policy_activated",
            )
        )
    ):
        raise ValueError("performance-lock-v4 materialization boundary changed")
    receipt.update(audit)
    return receipt


def materialize_performance_lock_v4_roots(
    *,
    repository_root: str | Path = _REPO_ROOT,
    plan_path: str | Path = DEFAULT_PLAN_PATH,
    run009_scientific_gate_receipt_path: str
    | Path = DEFAULT_RUN009_SCIENTIFIC_GATE_RECEIPT_PATH,
    current_profile_registry_path: str | Path = DEFAULT_CURRENT_PROFILE_REGISTRY_PATH,
    output_dir: str | Path = DEFAULT_ROOT_OUTPUT_DIR,
    materialization_receipt_path: str
    | Path = DEFAULT_MATERIALIZATION_RECEIPT_PATH,
) -> dict[str, Any]:
    plan, profile_before = _load_plan_and_authorization(
        plan_path=plan_path,
        run009_scientific_gate_receipt_path=run009_scientific_gate_receipt_path,
        current_profile_registry_path=current_profile_registry_path,
    )
    output = Path(output_dir).resolve()
    claim_path = output / CLAIM_NAME
    root_dir = output / "roots"
    if not claim_path.exists():
        if (
            (root_dir.exists() and any(root_dir.iterdir()))
            or Path(materialization_receipt_path).exists()
            or (output / DEFAULT_ROOT_SEAL_PATH.name).exists()
        ):
            raise ValueError("root content exists before performance-lock-v4 claim")
        output.mkdir(parents=True, exist_ok=True)
    claim = _claim_value(output)
    _write_once_or_validate(claim_path, claim, "performance-lock-v4 claim")
    stored_claim = _validate_claim(
        _read_canonical(claim_path, "performance-lock-v4 claim"),
        output_dir=output,
    )
    roots = runner._materialize_roots(
        contract=plan["run_contract"],
        repository_root=Path(repository_root).resolve(),
        output_dir=output,
        indices=runner.CONTRACT_HAND_INDICES,
    )
    profile_after = sha256_file(current_profile_registry_path)
    audit = _audit_roots(
        roots=roots,
        contract=plan["run_contract"],
        current_profile_unchanged=(
            profile_before == profile_after == CURRENT_PROFILE_REGISTRY_SHA256
        ),
        validate_artifacts=False,
    )
    receipt = validate_materialization_receipt(
        _materialization_value(
            output_dir=output,
            claim=stored_claim,
            audit=audit,
        )
    )
    _write_once_or_validate(
        materialization_receipt_path,
        receipt,
        "performance-lock-v4 materialization receipt",
    )
    return receipt


def _seal_value(
    *,
    output_dir: Path,
    claim: Mapping[str, Any],
    materialization: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SEAL_SCHEMA,
        "status": SEAL_STATUS,
        "plan_sha256": PLAN_SHA256,
        "run_contract_digest": RUN_CONTRACT_DIGEST,
        "claim_sha256": canonical_sha256(claim),
        "materialization_receipt_sha256": canonical_sha256(materialization),
        "authorizing_gate_receipt_file_sha256": (
            RUN009_GATE_RECEIPT_FILE_SHA256
        ),
        "authorizing_gate_receipt_sha256": RUN009_GATE_RECEIPT_SHA256,
        "root_output_directory": str(output_dir.resolve()),
        **dict(audit),
        "sealed": True,
        "cloud_started": False,
        "quality_pilot_authorized": False,
        "training_authorized": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
    }


def validate_root_seal(value: Mapping[str, Any]) -> dict[str, Any]:
    seal = dict(value)
    _exact_keys(seal, _SEAL_KEYS, "performance-lock-v4 root seal")
    audit = _validate_audit({key: seal[key] for key in _AUDIT_KEYS if key in seal})
    output = seal.get("root_output_directory")
    if (
        seal.get("schema") != SEAL_SCHEMA
        or seal.get("status") != SEAL_STATUS
        or seal.get("plan_sha256") != PLAN_SHA256
        or seal.get("run_contract_digest") != RUN_CONTRACT_DIGEST
        or not _is_sha256(seal.get("claim_sha256"))
        or not _is_sha256(seal.get("materialization_receipt_sha256"))
        or seal.get("authorizing_gate_receipt_file_sha256")
        != RUN009_GATE_RECEIPT_FILE_SHA256
        or seal.get("authorizing_gate_receipt_sha256")
        != RUN009_GATE_RECEIPT_SHA256
        or not isinstance(output, str)
        or not output
        or seal.get("sealed") is not True
        or any(
            seal.get(field) is not False
            for field in (
                "cloud_started",
                "quality_pilot_authorized",
                "training_authorized",
                "current_profile_changed",
                "named_profile_added",
                "runtime_policy_activated",
            )
        )
    ):
        raise ValueError("performance-lock-v4 root seal boundary changed")
    seal.update(audit)
    return seal


def seal_performance_lock_v4_roots(
    *,
    plan_path: str | Path = DEFAULT_PLAN_PATH,
    run009_scientific_gate_receipt_path: str
    | Path = DEFAULT_RUN009_SCIENTIFIC_GATE_RECEIPT_PATH,
    current_profile_registry_path: str | Path = DEFAULT_CURRENT_PROFILE_REGISTRY_PATH,
    output_dir: str | Path = DEFAULT_ROOT_OUTPUT_DIR,
    materialization_receipt_path: str
    | Path = DEFAULT_MATERIALIZATION_RECEIPT_PATH,
    root_seal_path: str | Path = DEFAULT_ROOT_SEAL_PATH,
) -> dict[str, Any]:
    plan, profile_before = _load_plan_and_authorization(
        plan_path=plan_path,
        run009_scientific_gate_receipt_path=run009_scientific_gate_receipt_path,
        current_profile_registry_path=current_profile_registry_path,
    )
    output = Path(output_dir).resolve()
    claim = _validate_claim(
        _read_canonical(output / CLAIM_NAME, "performance-lock-v4 claim"),
        output_dir=output,
    )
    materialization = validate_materialization_receipt(
        _read_canonical(
            materialization_receipt_path,
            "performance-lock-v4 materialization receipt",
        )
    )
    if (
        materialization["claim_sha256"] != canonical_sha256(claim)
        or materialization["root_output_directory"] != str(output)
    ):
        raise ValueError("materialization receipt is not bound to this root target")
    roots = _load_root_files(output_dir=output, contract=plan["run_contract"])
    profile_after = sha256_file(current_profile_registry_path)
    audit = _audit_roots(
        roots=roots,
        contract=plan["run_contract"],
        current_profile_unchanged=(
            profile_before == profile_after == CURRENT_PROFILE_REGISTRY_SHA256
        ),
        validate_artifacts=False,
    )
    materialization_audit = {
        key: materialization[key] for key in _AUDIT_KEYS
    }
    if audit != materialization_audit:
        raise ValueError("sealed roots differ from materialization receipt")
    seal = validate_root_seal(
        _seal_value(
            output_dir=output,
            claim=claim,
            materialization=materialization,
            audit=audit,
        )
    )
    _write_once_or_validate(
        root_seal_path,
        seal,
        "performance-lock-v4 root seal",
    )
    return seal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze/materialize/seal the run009-authorized lock-v4 roots"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--output", type=Path, default=DEFAULT_PLAN_PATH)
    plan_parser.add_argument(
        "--run009-receipt",
        type=Path,
        default=DEFAULT_RUN009_SCIENTIFIC_GATE_RECEIPT_PATH,
    )
    plan_parser.add_argument(
        "--current-profile-registry",
        type=Path,
        default=DEFAULT_CURRENT_PROFILE_REGISTRY_PATH,
    )

    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument(
        "--repository-root", type=Path, default=_REPO_ROOT
    )
    materialize_parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    materialize_parser.add_argument(
        "--run009-receipt",
        type=Path,
        default=DEFAULT_RUN009_SCIENTIFIC_GATE_RECEIPT_PATH,
    )
    materialize_parser.add_argument(
        "--current-profile-registry",
        type=Path,
        default=DEFAULT_CURRENT_PROFILE_REGISTRY_PATH,
    )
    materialize_parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_ROOT_OUTPUT_DIR
    )
    materialize_parser.add_argument(
        "--materialization-receipt",
        type=Path,
        default=DEFAULT_MATERIALIZATION_RECEIPT_PATH,
    )

    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    seal_parser.add_argument(
        "--run009-receipt",
        type=Path,
        default=DEFAULT_RUN009_SCIENTIFIC_GATE_RECEIPT_PATH,
    )
    seal_parser.add_argument(
        "--current-profile-registry",
        type=Path,
        default=DEFAULT_CURRENT_PROFILE_REGISTRY_PATH,
    )
    seal_parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_ROOT_OUTPUT_DIR
    )
    seal_parser.add_argument(
        "--materialization-receipt",
        type=Path,
        default=DEFAULT_MATERIALIZATION_RECEIPT_PATH,
    )
    seal_parser.add_argument(
        "--root-seal", type=Path, default=DEFAULT_ROOT_SEAL_PATH
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        value = write_performance_lock_v4_plan(
            output_path=args.output,
            run009_scientific_gate_receipt_path=args.run009_receipt,
            current_profile_registry_path=args.current_profile_registry,
        )
    elif args.command == "materialize":
        value = materialize_performance_lock_v4_roots(
            repository_root=args.repository_root,
            plan_path=args.plan,
            run009_scientific_gate_receipt_path=args.run009_receipt,
            current_profile_registry_path=args.current_profile_registry,
            output_dir=args.output_dir,
            materialization_receipt_path=args.materialization_receipt,
        )
    else:
        value = seal_performance_lock_v4_roots(
            plan_path=args.plan,
            run009_scientific_gate_receipt_path=args.run009_receipt,
            current_profile_registry_path=args.current_profile_registry,
            output_dir=args.output_dir,
            materialization_receipt_path=args.materialization_receipt,
            root_seal_path=args.root_seal,
        )
    print(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CLAIM_NAME",
    "DEFAULT_CURRENT_PROFILE_REGISTRY_PATH",
    "DEFAULT_MATERIALIZATION_RECEIPT_PATH",
    "DEFAULT_PLAN_PATH",
    "DEFAULT_ROOT_OUTPUT_DIR",
    "DEFAULT_ROOT_SEAL_PATH",
    "DEFAULT_RUN009_SCIENTIFIC_GATE_RECEIPT_PATH",
    "MATERIALIZATION_SCHEMA",
    "PLAN_DECISION",
    "PLAN_SCHEMA",
    "PLAN_SCOPE",
    "PLAN_SHA256",
    "PLAN_STATUS",
    "RUN_CONTRACT_DIGEST",
    "SEAL_SCHEMA",
    "build_performance_lock_v4_plan",
    "canonical_bytes",
    "canonical_sha256",
    "load_run009_scientific_gate_receipt",
    "main",
    "materialize_performance_lock_v4_roots",
    "seal_performance_lock_v4_roots",
    "validate_materialization_receipt",
    "validate_performance_lock_v4_plan",
    "validate_root_seal",
    "write_performance_lock_v4_plan",
]
