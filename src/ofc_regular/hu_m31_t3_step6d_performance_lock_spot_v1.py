"""Bounded Spot lifecycle for the one-shot Candidate02 performance lock.

This module is intentionally separate from the repeatable full100
performance-development lifecycle.  It packages only a globally claimed and
sealed 100-hand lock root set, freezes twenty source-isolated jobs, and makes
Spot authorization an explicit immutable step.

The scientific root-open claim is created by
``hu_m31_t3_step6d_performance_lock_open`` before this module may package
anything.  Package/authorization/launch do not read teacher result content,
train, promote, activate runtime policy, or resolve ``current``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from . import hu_m31_t3_step6d_candidate02_performance_lock_plan as lock_plan
from . import hu_m31_t3_step6d_full100_spot_v1 as transport
from . import hu_m31_t3_step6d_performance_lock_open as lock_open
from . import hu_m31_t3_step6d_spot_v2 as tail_spot
from . import run_hu_m31_t3_step6d_performance_v2 as runner


PACKAGE_SCHEMA = "hu_m31_t3_step6d_performance_lock_spot_package_v1"
PACKAGE_READY_SCHEMA = "hu_m31_t3_step6d_performance_lock_package_ready_v1"
AUTHORIZATION_SCHEMA = "hu_m31_t3_step6d_performance_lock_launch_authorization_v1"
LAUNCH_CLAIM_SCHEMA = "hu_m31_t3_step6d_performance_lock_launch_claim_v1"
LAUNCH_RESULT_SCHEMA = "hu_m31_t3_step6d_performance_lock_launch_result_v1"
STATUS_SCHEMA = "hu_m31_t3_step6d_performance_lock_cloud_status_v1"
RESUME_PREFLIGHT_SCHEMA = "hu_m31_t3_step6d_performance_lock_resume_preflight_v1"
RESUME_CLAIM_SCHEMA = "hu_m31_t3_step6d_performance_lock_resume_claim_v1"
RESUME_RESULT_SCHEMA = "hu_m31_t3_step6d_performance_lock_resume_result_v1"
RESULT_OPEN_CLAIM_NAME = "result_open_claim.json"
RESULT_OPEN_CLAIM_SCHEMA = "hu_m31_t3_step6d_performance_lock_result_open_claim_v1"
RECEIVE_SCHEMA = "hu_m31_t3_step6d_performance_lock_receive_v1"
GLOBAL_SPOT_CLAIM_SCHEMA = "hu_m31_t3_step6d_performance_lock_global_spot_claim_v1"
GLOBAL_SPOT_CLAIM_NAME = "GLOBAL_PERFORMANCE_LOCK_SPOT_CLAIM.json"

SOURCE_NAME = transport.SOURCE_NAME
STARTUP_NAME = transport.STARTUP_NAME
MANIFEST_NAME = transport.MANIFEST_NAME
READY_NAME = transport.READY_NAME
AUTHORIZATION_NAME = transport.AUTHORIZATION_NAME
LAUNCH_CLAIM_NAME = transport.LAUNCH_CLAIM_NAME
LAUNCH_RESULT_NAME = transport.LAUNCH_RESULT_NAME
RESUME_CLAIM_NAME = transport.RESUME_CLAIM_NAME
RESUME_RESULT_NAME = transport.RESUME_RESULT_NAME
PLAN_PACKAGE_PATH = transport.PLAN_PACKAGE_PATH
ROOT_PACKAGE_DIR = transport.ROOT_PACKAGE_DIR
OPEN_CLAIM_PACKAGE_PATH = "frozen/performance_lock_open_claim.json"
ROOT_SEAL_PACKAGE_PATH = "frozen/performance_lock_root_seal.json"
MATERIALIZATION_PACKAGE_PATH = "frozen/performance_lock_materialization.json"
MATERIALIZER_WINDOWS_PROBE_PACKAGE_PATH = "frozen/materializer_windows_probe.json"
MATERIALIZER_LINUX_PROBE_PACKAGE_PATH = "frozen/materializer_linux_probe.json"
MATERIALIZER_PARITY_RECEIPT_PACKAGE_PATH = "frozen/materializer_platform_parity.json"
MATERIALIZER_PARITY_SCRIPT_PACKAGE_PATH = (
    "frozen/materializer_platform_parity_script.py"
)
MATERIALIZER_RUST_SOURCE_PACKAGE_PATH = "frozen/materializer_stage3_feature_encoder.rs"
MATERIALIZER_CARGO_LOCK_PACKAGE_PATH = "frozen/materializer_Cargo.lock"
MATERIALIZER_WINDOWS_PROBE_NAME = "materializer_windows_probe.json"
MATERIALIZER_LINUX_PROBE_NAME = "materializer_linux_probe.json"
MATERIALIZER_PARITY_RECEIPT_NAME = "materializer_platform_parity.json"
MATERIALIZER_PARITY_SCRIPT_RELATIVE = (
    "scripts/verify_hu_m31_t3_feature_encoder_platform_parity.py"
)
MATERIALIZER_RUST_SOURCE_RELATIVE = "rust/ofc_stage3_feature_encoder/src/lib.rs"
EXPECTED_WINDOWS_FEATURE_ENCODER_SHA256 = (
    "24d156f455867ddb5915b07d440f72c790f893e310ef95d273b74d7b4d017b6b"
)
EXPECTED_MATERIALIZER_PARITY_SCRIPT_SHA256 = (
    "a1905bc951ac54680e4993a5f63cc85a2d2579f0f962529fd3a0c8968bfcd921"
)
EXPECTED_MATERIALIZER_PARITY_OUTPUT_SHA256 = (
    "a1cea3acc7b6fc5cf79dff6e5a61d3970f3ac2589f79623b4a2fd22ab83ea5e3"
)

DEFAULT_PROJECT = transport.DEFAULT_PROJECT
DEFAULT_BUCKET = transport.DEFAULT_BUCKET
DEFAULT_REGION = transport.DEFAULT_REGION
DEFAULT_ZONES = transport.DEFAULT_ZONES
EXPECTED_MACHINE_TYPE = transport.EXPECTED_MACHINE_TYPE
EXPECTED_IMAGE_NAME = transport.EXPECTED_IMAGE_NAME
EXPECTED_IMAGE_ID = transport.EXPECTED_IMAGE_ID
EXPECTED_IMAGE_SELF_LINK = transport.EXPECTED_IMAGE_SELF_LINK
REFERENCE_PACKAGE_PATH = transport.REFERENCE_PACKAGE_PATH
CANDIDATE_PACKAGE_PATH = transport.CANDIDATE_PACKAGE_PATH
FEATURE_PACKAGE_PATH = transport.FEATURE_PACKAGE_PATH
EXPECTED_REFERENCE_RELATIVE = transport.EXPECTED_REFERENCE_RELATIVE
EXPECTED_FEATURE_RELATIVE = transport.EXPECTED_FEATURE_RELATIVE
EXPECTED_FEATURE_SHA256 = transport.EXPECTED_FEATURE_SHA256

SOURCE_ROLES = tuple(runner.SOURCE_ROLES)
SHARDS_PER_ROLE = 10
HANDS_PER_SHARD = 10
MAX_LOGICAL_JOBS = 20
MAX_CONCURRENT_VMS = 20

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN_PATH = (
    _REPO_ROOT / "outputs/hu_joint_policy/m31_t3_step6d/performance_lock/"
    "precontent_plan_v1.json"
)

_PACKAGE_KEYS = transport._PACKAGE_KEYS
_READY_KEYS = transport._READY_KEYS
_AUTHORIZATION_KEYS = frozenset(
    set(transport._AUTHORIZATION_KEYS) | {"global_spot_claim"}
)
_JOB_RECORD_KEYS = transport._JOB_RECORD_KEYS
_LAUNCH_CLAIM_KEYS = transport._LAUNCH_CLAIM_KEYS
_LAUNCH_RESULT_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "selected_job_ids",
        "created",
        "failures",
        "cleanup",
        "cleanup_compute_stopped_or_absent",
        "logical_job_count",
        "max_logical_jobs",
        "max_concurrent_vms",
        "machine_type",
        "process_count",
        "rayon_threads_per_process",
        "run_contract_digest",
        "launch_target",
        "cost_guard",
        "preflight",
        "launch_claim_sha256",
        "production_fanout_authorized",
        "performance_lock_authorized",
        "quality_pilot_authorized",
        "training_eligible",
        "current_profile_changed",
        "runtime_policy_activated",
    }
)
_RESUME_CLAIM_KEYS = transport._RESUME_CLAIM_KEYS
_RESUME_RESULT_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "attempt_index",
        "selected_job_ids",
        "created",
        "failures",
        "cleanup",
        "cleanup_compute_stopped_or_absent",
        "logical_job_count",
        "max_resume_jobs",
        "max_cumulative_vm_jobs",
        "run_contract_digest",
        "launch_target",
        "cost_guard",
        "preflight",
        "resume_claim_sha256",
        "initial_launch_claim_sha256",
        "initial_launch_result_sha256",
        "third_attempt_authorized",
        "performance_lock_authorized",
        "quality_pilot_authorized",
        "training_eligible",
        "current_profile_changed",
        "runtime_policy_activated",
    }
)
_RESULT_OPEN_CLAIM_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "package_manifest_sha256",
        "launch_authorization_sha256",
        "launch_claim_sha256",
        "launch_result_sha256",
        "resume_claim_sha256",
        "resume_result_sha256",
        "precontent_plan_sha256",
        "root_open_claim_sha256",
        "root_seal_sha256",
        "run_contract_digest",
        "authorized_job_ids",
        "done_markers",
        "done_marker_count",
        "all_done_markers_present",
        "active_compute_instances",
        "result_hand_content_addressed_before_claim",
        "compute_retry_after_claim_allowed",
        "reseed_after_claim_allowed",
        "quality_pilot_authorized",
        "training_eligible",
        "current_profile_changed",
        "runtime_policy_activated",
        "claimed_unix_seconds",
    }
)
_RECEIVE_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "package_manifest_sha256",
        "launch_authorization_sha256",
        "launch_claim_sha256",
        "launch_result_sha256",
        "resume_claim_sha256",
        "resume_result_sha256",
        "result_open_claim_sha256",
        "precontent_plan_sha256",
        "root_open_claim_sha256",
        "root_seal_sha256",
        "run_contract_digest",
        "source_roles",
        "logical_job_count",
        "paired_hand_count",
        "root_count",
        "jobs",
        "candidate_done_paths",
        "reference_done_paths",
        "source_isolation_validated",
        "root_pairing_validated",
        "sealed_root_set_validated",
        "merge_executed",
        "performance_lock_finalized",
        "quality_pilot_authorized",
        "training_eligible",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
    }
)
_RECEIVE_JOB_KEYS = frozenset(
    {
        "job_id",
        "source_role",
        "shard_index",
        "work_hand_indices",
        "done_path",
        "done_sha256",
        "root_artifacts",
        "source_hand_artifacts",
        "run_contract_digest",
    }
)
_LOCK_QUALIFICATION_KEYS = frozenset(
    {
        "summary_sha256",
        "validation_sha256",
        "scientific_merge_sha256",
        "receive_receipt_sha256",
        "development_run_contract_digest",
        "all_gates_passed",
        "performance_candidate_frozen",
        "performance_lock_authorized",
        "open_claim_package_path",
        "open_claim_sha256",
        "root_seal_package_path",
        "root_seal_sha256",
        "materialization_package_path",
        "materialization_sha256",
        "lock_root_aggregate_sha256",
        "lock_root_topology_sha256",
        "lock_observation_fingerprint_sha256",
        "lock_development_overlap_count",
    }
)
_GLOBAL_SPOT_CLAIM_KEYS = frozenset(
    {
        "schema",
        "status",
        "global_root_claim_path",
        "global_root_claim_sha256",
        "lock_output_directory",
        "package_run_directory",
        "run_name",
        "package_manifest_sha256",
        "source_sha256",
        "startup_sha256",
        "precontent_plan_sha256",
        "root_seal_sha256",
        "run_contract_digest",
        "authorized_job_ids",
        "max_initial_jobs",
        "max_resume_attempts",
        "alternate_package_authorization_allowed",
        "quality_pilot_authorized",
        "training_eligible",
        "current_profile_changed",
        "runtime_policy_activated",
        "claimed_unix_ns",
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


def _read_packaged_plan(
    run_dir: str | Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    source = Path(run_dir).resolve() / SOURCE_NAME
    with zipfile.ZipFile(source) as archive:
        raw = archive.read(PLAN_PACKAGE_PATH)
    value = json.loads(raw.decode("utf-8"))
    if (
        not isinstance(value, dict)
        or raw != canonical_bytes(value)
        or hashlib.sha256(raw).hexdigest() != lock_plan.PRECONTENT_PLAN_SHA256
        or manifest.get("plan_sha256") != lock_plan.PRECONTENT_PLAN_SHA256
    ):
        raise ValueError("packaged performance-lock plan changed")
    return lock_plan.validate_precontent_plan(value)


def _read_packaged_open_claim(
    run_dir: str | Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    source = Path(run_dir).resolve() / SOURCE_NAME
    with zipfile.ZipFile(source) as archive:
        raw = archive.read(OPEN_CLAIM_PACKAGE_PATH)
    value = json.loads(raw.decode("utf-8"))
    qualification = manifest.get("tail_qualification")
    if (
        not isinstance(value, dict)
        or raw != canonical_bytes(value)
        or not isinstance(qualification, Mapping)
        or hashlib.sha256(raw).hexdigest() != qualification.get("open_claim_sha256")
        or value.get("schema") != lock_open.CLAIM_SCHEMA
    ):
        raise ValueError("packaged performance-lock open claim changed")
    return value


def _write_once(path: str | Path, value: Mapping[str, Any]) -> None:
    transport._write_once(path, value)


@contextmanager
def _lifecycle_mutex(run_dir: str | Path, operation: str) -> Iterator[None]:
    """Serialize launch/resume/result-open without a crash-stale sentinel."""

    target = Path(run_dir).resolve()
    lock_path = target / ".performance_lock_lifecycle.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(
                    f"performance-lock lifecycle is busy: {operation}"
                ) from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(
                    f"performance-lock lifecycle is busy: {operation}"
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def authorized_job_ids() -> tuple[str, ...]:
    return tuple(
        f"{role}-shard-{index:02d}"
        for role in SOURCE_ROLES
        for index in range(SHARDS_PER_ROLE)
    )


def _bounded_jobs(values: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(values)
    allowed = authorized_job_ids()
    if (
        not selected
        or len(selected) > MAX_LOGICAL_JOBS
        or len(selected) != len(set(selected))
        or any(value not in allowed for value in selected)
        or tuple(value for value in allowed if value in set(selected)) != selected
    ):
        raise ValueError("lock resume jobs must be unique and in frozen order")
    return selected


def _lock_qualification(
    *,
    plan: Mapping[str, Any],
    claim: Mapping[str, Any],
    seal: Mapping[str, Any],
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    development = claim["development_go"]
    plan_qualification = plan["development_qualification"]
    comparison = seal["development_comparison"]
    overlap = (
        int(comparison["lock_fingerprint_overlap_count"])
        + int(comparison["lock_root_hash_overlap_count"])
        + int(comparison["lock_seed_overlap_count"])
    )
    value = {
        "summary_sha256": plan_qualification["summary_sha256"],
        "validation_sha256": plan_qualification["validation_sha256"],
        "scientific_merge_sha256": plan_qualification["scientific_merge_sha256"],
        "receive_receipt_sha256": plan_qualification["receive_receipt_sha256"],
        "development_run_contract_digest": plan_qualification["run_contract_digest"],
        "all_gates_passed": development["all_gates_passed"],
        "performance_candidate_frozen": True,
        "performance_lock_authorized": development["performance_lock_authorized"],
        "open_claim_package_path": OPEN_CLAIM_PACKAGE_PATH,
        "open_claim_sha256": canonical_sha256(claim),
        "root_seal_package_path": ROOT_SEAL_PACKAGE_PATH,
        "root_seal_sha256": canonical_sha256(seal),
        "materialization_package_path": MATERIALIZATION_PACKAGE_PATH,
        "materialization_sha256": canonical_sha256(materialization),
        "lock_root_aggregate_sha256": seal["aggregate_root_sha256"],
        "lock_root_topology_sha256": seal["root_topology_sha256"],
        "lock_observation_fingerprint_sha256": seal["observation_fingerprint_sha256"],
        "lock_development_overlap_count": overlap,
    }
    if set(value) != _LOCK_QUALIFICATION_KEYS:
        raise AssertionError("performance-lock qualification schema changed")
    return value


def _materializer_file_record_matches(
    value: Any, path: Path, *, canonical: bool = False
) -> bool:
    if not isinstance(value, Mapping):
        return False
    expected_keys = {"path", "bytes", "sha256"}
    if canonical:
        expected_keys.add("canonical_sha256")
    digest = sha256_file(path)
    return (
        set(value) == expected_keys
        and isinstance(value.get("path"), str)
        and bool(value["path"])
        and value.get("bytes") == path.stat().st_size
        and value.get("sha256") == digest
        and (
            not canonical
            or (
                value.get("canonical_sha256") == digest
                and canonical_sha256(
                    _read_canonical(path, "materializer parity artifact")
                )
                == digest
            )
        )
    )


def _validate_materializer_platform_parity(
    *,
    windows_probe_path: Path,
    linux_probe_path: Path,
    parity_receipt_path: Path,
    parity_script_path: Path,
    rust_source_path: Path,
    cargo_lock_path: Path,
    global_claim_path: Path,
    materialization_path: Path,
    seal_path: Path,
    ai_profiles_path: Path,
) -> dict[str, Any]:
    windows = _read_canonical(windows_probe_path, "Windows materializer parity probe")
    linux = _read_canonical(linux_probe_path, "Linux materializer parity probe")
    receipt = _read_canonical(
        parity_receipt_path, "materializer platform parity receipt"
    )
    claim = _read_canonical(global_claim_path, "materializer global claim")
    materialization = _read_canonical(
        materialization_path, "materializer root materialization"
    )
    seal = _read_canonical(seal_path, "materializer root seal")
    script_sha = sha256_file(parity_script_path)
    rust_sha = sha256_file(rust_source_path)
    cargo_sha = sha256_file(cargo_lock_path)
    current_sha = sha256_file(ai_profiles_path)
    claim_sha = canonical_sha256(claim)
    materialization_sha = canonical_sha256(materialization)
    seal_sha = canonical_sha256(seal)

    def content_record_matches(record: Any, path: Path, expected_sha: str) -> bool:
        return (
            isinstance(record, Mapping)
            and set(record) == {"path", "bytes", "sha256"}
            and isinstance(record.get("path"), str)
            and bool(record["path"])
            and record.get("bytes") == path.stat().st_size
            and record.get("sha256") == expected_sha
        )

    for value, label, expected_library_sha in (
        (windows, "windows", EXPECTED_WINDOWS_FEATURE_ENCODER_SHA256),
        (linux, "linux", lock_plan.FEATURE_ENCODER_SHA256),
    ):
        generator = value.get("generator")
        build_inputs = value.get("build_inputs")
        library = value.get("library")
        if (
            value.get("schema") != "hu_m31_t3_feature_encoder_platform_probe_v1"
            or value.get("status") != "fixed_512_row_feature_encoder_probe_complete"
            or value.get("platform_label") != label
            or not isinstance(generator, Mapping)
            or not content_record_matches(
                generator.get("script"),
                parity_script_path,
                script_sha,
            )
            or not isinstance(build_inputs, Mapping)
            or not content_record_matches(
                build_inputs.get("rust_source"),
                rust_source_path,
                rust_sha,
            )
            or not content_record_matches(
                build_inputs.get("cargo_lock"),
                cargo_lock_path,
                cargo_sha,
            )
            or not isinstance(library, Mapping)
            or library.get("sha256") != expected_library_sha
            or value.get("encoder_status") != 0
            or value.get("feature_dim") != 1076
            or value.get("row_count") != 512
            or value.get("output_bytes") != 2_206_720
            or value.get("output_sha256") != EXPECTED_MATERIALIZER_PARITY_OUTPUT_SHA256
        ):
            raise ValueError(f"{label} materializer parity probe changed")
    if (
        windows.get("input_bytes") != linux.get("input_bytes")
        or windows.get("input_sha256") != linux.get("input_sha256")
        or windows.get("output_bytes") != linux.get("output_bytes")
        or windows.get("output_sha256") != linux.get("output_sha256")
        or windows.get("generator", {}).get("contract_sha256")
        != linux.get("generator", {}).get("contract_sha256")
    ):
        raise ValueError("materializer Windows/Linux probes are not bit-exact")

    artifacts = receipt.get("artifacts")
    parity = receipt.get("platform_parity")
    lock_chain = receipt.get("lock_chain")
    mutation = receipt.get("mutation_scope")
    build_inputs = receipt.get("build_inputs")
    libraries = receipt.get("libraries")
    generator = receipt.get("generator")
    expected_mutation = {
        "only_new_receipt_written": True,
        "seed_changed": False,
        "model_changed": False,
        "root_changed": False,
        "current_profile_changed": False,
        "profile_resolved": False,
        "opponent_private_discards_used": False,
    }
    if (
        receipt.get("schema") != "hu_m31_t3_feature_encoder_platform_parity_receipt_v1"
        or receipt.get("status")
        != (
            "verified_fixed_512_row_windows_linux_bit_exact_"
            "and_performance_lock_chain_intact"
        )
        or not isinstance(generator, Mapping)
        or generator.get("script", {}).get("sha256") != script_sha
        or script_sha != EXPECTED_MATERIALIZER_PARITY_SCRIPT_SHA256
        or not isinstance(parity, Mapping)
        or parity.get("input_bit_exact") is not True
        or parity.get("output_bit_exact") is not True
        or parity.get("row_mapping_checked_by_each_probe") is not True
        or parity.get("row_count") != 512
        or parity.get("input_bytes") != windows["input_bytes"]
        or parity.get("input_sha256") != windows["input_sha256"]
        or parity.get("output_bytes") != windows["output_bytes"]
        or parity.get("output_sha256") != windows["output_sha256"]
        or parity.get("accepted_linux_library_bound_to_global_claim") is not True
        or not isinstance(libraries, Mapping)
        or libraries.get("windows") != windows["library"]
        or libraries.get("linux") != linux["library"]
        or not isinstance(build_inputs, Mapping)
        or build_inputs.get("same_rust_source_content_in_both_probes") is not True
        or build_inputs.get("same_cargo_lock_content_in_both_probes") is not True
        or build_inputs.get("rust_source", {}).get("sha256") != rust_sha
        or build_inputs.get("cargo_lock", {}).get("sha256") != cargo_sha
        or not isinstance(lock_chain, Mapping)
        or lock_chain.get("global_claim_sha256") != claim_sha
        or lock_chain.get("materialization_sha256") != materialization_sha
        or lock_chain.get("seal_sha256") != seal_sha
        or lock_chain.get("accepted_linux_feature_encoder_sha256")
        != lock_plan.FEATURE_ENCODER_SHA256
        or lock_chain.get("ai_profiles_sha256") != current_sha
        or lock_chain.get("root_count") != 100
        or lock_chain.get("reseeded") is not False
        or lock_chain.get("current_profile_changed") is not False
        or mutation != expected_mutation
        or any(
            receipt.get(field) is not False
            for field in (
                "quality_evidence",
                "promotion_evidence",
                "training_eligible",
            )
        )
        or not isinstance(artifacts, Mapping)
        or not _materializer_file_record_matches(
            artifacts.get("windows_probe"),
            windows_probe_path,
            canonical=True,
        )
        or not _materializer_file_record_matches(
            artifacts.get("linux_probe"),
            linux_probe_path,
            canonical=True,
        )
        or not _materializer_file_record_matches(
            artifacts.get("global_claim"),
            global_claim_path,
            canonical=True,
        )
        or not _materializer_file_record_matches(
            artifacts.get("materialization"),
            materialization_path,
            canonical=True,
        )
        or not _materializer_file_record_matches(
            artifacts.get("seal"),
            seal_path,
            canonical=True,
        )
        or not _materializer_file_record_matches(
            artifacts.get("ai_profiles"),
            ai_profiles_path,
            canonical=False,
        )
    ):
        raise ValueError("materializer platform parity receipt changed")
    if (
        claim_sha != materialization.get("global_claim_sha256")
        or claim_sha != seal.get("global_claim_sha256")
        or materialization_sha != seal.get("materialization_sha256")
        or materialization.get("root_count") != 100
        or materialization.get("reseeded") is not False
        or seal.get("root_count") != 100
        or seal.get("observation_count") != 200
        or current_sha != lock_plan.CURRENT_PROFILE_REGISTRY_SHA256
    ):
        raise ValueError("materializer parity lock chain changed")
    return receipt


def _copy_package_sources(
    *,
    repository_root: Path,
    package_root: Path,
    reference_library: Path,
    candidate_library: Path,
    feature_encoder: Path,
    plan_path: Path,
    root_dir: Path,
    claim_path: Path,
    seal_path: Path,
    materialization_path: Path,
    windows_probe_path: Path,
    linux_probe_path: Path,
    parity_receipt_path: Path,
) -> dict[str, dict[str, Any]]:
    entries = tail_spot._copy_source_tree(
        repository_root=repository_root,
        package_root=package_root,
        reference_library=reference_library,
        candidate_library=candidate_library,
        feature_encoder=feature_encoder,
    )

    def copy(relative: str, source: Path) -> None:
        entries[relative] = tail_spot._copy_file(source, package_root / relative)

    copy(PLAN_PACKAGE_PATH, plan_path)
    copy(OPEN_CLAIM_PACKAGE_PATH, claim_path)
    copy(ROOT_SEAL_PACKAGE_PATH, seal_path)
    copy(MATERIALIZATION_PACKAGE_PATH, materialization_path)
    copy(MATERIALIZER_WINDOWS_PROBE_PACKAGE_PATH, windows_probe_path)
    copy(MATERIALIZER_LINUX_PROBE_PACKAGE_PATH, linux_probe_path)
    copy(MATERIALIZER_PARITY_RECEIPT_PACKAGE_PATH, parity_receipt_path)
    copy(
        MATERIALIZER_PARITY_SCRIPT_PACKAGE_PATH,
        repository_root / MATERIALIZER_PARITY_SCRIPT_RELATIVE,
    )
    copy(
        MATERIALIZER_RUST_SOURCE_PACKAGE_PATH,
        repository_root / MATERIALIZER_RUST_SOURCE_RELATIVE,
    )
    copy(MATERIALIZER_CARGO_LOCK_PACKAGE_PATH, repository_root / "Cargo.lock")
    for index in runner.CONTRACT_HAND_INDICES:
        copy(
            f"{ROOT_PACKAGE_DIR}/hand_{index:03d}.json",
            root_dir / f"hand_{index:03d}.json",
        )
    return dict(sorted(entries.items()))


def _job_records(plan: Mapping[str, Any], stage: Path) -> list[dict[str, Any]]:
    records = transport._job_records(plan, stage)
    if [row["job_id"] for row in records] != list(authorized_job_ids()):
        raise ValueError("performance-lock job order changed")
    return records


def package_performance_lock(
    *,
    output_dir: str | Path,
    run_name: str,
    lock_inputs: lock_open.PerformanceLockInputs,
    startup_script: str | Path | None = None,
) -> dict[str, Any]:
    """Build an immutable local package after replaying claim and root seal."""

    if transport._SAFE_RUN.fullmatch(run_name) is None:
        raise ValueError("performance-lock run name is not a safe identity")
    root = Path(lock_inputs.repository_root).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError("performance-lock package destination is immutable")
    plan_path = Path(lock_inputs.plan_path).resolve()
    plan = lock_plan.validate_precontent_plan(
        _read_canonical(plan_path, "performance-lock plan")
    )
    if sha256_file(plan_path) != lock_plan.PRECONTENT_PLAN_SHA256:
        raise ValueError("performance-lock frozen plan file hash changed")
    claim = lock_open.validate_open_claim(lock_inputs)
    seal = lock_open.validate_root_seal(lock_inputs)
    output = Path(lock_inputs.lock_output_directory).resolve()
    materialization_path = output / "materialization.json"
    materialization = _read_canonical(
        materialization_path, "performance-lock materialization"
    )
    windows_probe_path = output / MATERIALIZER_WINDOWS_PROBE_NAME
    linux_probe_path = output / MATERIALIZER_LINUX_PROBE_NAME
    parity_receipt_path = output / MATERIALIZER_PARITY_RECEIPT_NAME
    parity_script_path = root / MATERIALIZER_PARITY_SCRIPT_RELATIVE
    rust_source_path = root / MATERIALIZER_RUST_SOURCE_RELATIVE
    cargo_lock_path = root / "Cargo.lock"
    ai_profiles_path = root / "src/ofc_regular/ai_profiles.py"
    _validate_materializer_platform_parity(
        windows_probe_path=windows_probe_path,
        linux_probe_path=linux_probe_path,
        parity_receipt_path=parity_receipt_path,
        parity_script_path=parity_script_path,
        rust_source_path=rust_source_path,
        cargo_lock_path=cargo_lock_path,
        global_claim_path=Path(lock_inputs.global_claim_path).resolve(),
        materialization_path=materialization_path,
        seal_path=output / "seal.json",
        ai_profiles_path=ai_profiles_path,
    )
    if (
        seal["global_claim_sha256"] != canonical_sha256(claim)
        or seal["materialization_sha256"] != canonical_sha256(materialization)
        or seal["plan_sha256"] != lock_plan.PRECONTENT_PLAN_SHA256
        or seal["run_contract_digest"] != lock_plan.LOCK_RUN_CONTRACT_DIGEST
    ):
        raise ValueError("performance-lock open/seal lineage changed")
    candidate = tail_spot._validate_binary(
        lock_inputs.candidate_library,
        lock_plan.CANDIDATE_LIBRARY_SHA256,
        "performance-lock candidate",
    )
    reference = tail_spot._validate_binary(
        lock_inputs.reference_library,
        lock_plan.REFERENCE_LIBRARY_SHA256,
        "performance-lock reference",
    )
    feature = tail_spot._validate_binary(
        lock_inputs.feature_encoder,
        lock_plan.FEATURE_ENCODER_SHA256,
        "performance-lock feature encoder",
    )
    startup = Path(startup_script or lock_inputs.startup_source).resolve()
    if not startup.is_file() or startup.is_symlink():
        raise ValueError("performance-lock startup source is missing or unsafe")
    if sha256_file(startup) != claim["startup_source"]["sha256"]:
        raise ValueError("performance-lock startup changed after global claim")

    stage = destination.with_name(f".{destination.name}.{os.getpid()}.staging")
    if stage.exists():
        raise FileExistsError("stale performance-lock package staging exists")
    stage.mkdir(parents=True)
    try:
        package_root = stage / "package_src"
        entries = _copy_package_sources(
            repository_root=root,
            package_root=package_root,
            reference_library=reference,
            candidate_library=candidate,
            feature_encoder=feature,
            plan_path=plan_path,
            root_dir=output / "roots",
            claim_path=Path(lock_inputs.global_claim_path).resolve(),
            seal_path=output / "seal.json",
            materialization_path=materialization_path,
            windows_probe_path=windows_probe_path,
            linux_probe_path=linux_probe_path,
            parity_receipt_path=parity_receipt_path,
        )
        source = stage / SOURCE_NAME
        tail_spot._zip_tree(package_root, source)
        shutil.copy2(startup, stage / STARTUP_NAME)
        jobs = _job_records(plan, stage)
        qualification = _lock_qualification(
            plan=plan,
            claim=claim,
            seal=seal,
            materialization=materialization,
        )
        manifest = {
            "schema": PACKAGE_SCHEMA,
            "status": "immutable_performance_lock_package_ready_not_authorized",
            "run_name": run_name,
            "source_name": SOURCE_NAME,
            "source_sha256": sha256_file(source),
            "source_bytes": source.stat().st_size,
            "startup_name": STARTUP_NAME,
            "startup_sha256": sha256_file(stage / STARTUP_NAME),
            "plan_package_path": PLAN_PACKAGE_PATH,
            "plan_sha256": lock_plan.PRECONTENT_PLAN_SHA256,
            "tail_qualification": qualification,
            "run_contract": plan["run_contract"],
            "run_contract_digest": lock_plan.LOCK_RUN_CONTRACT_DIGEST,
            "accepted_reference": {
                "package_path": REFERENCE_PACKAGE_PATH,
                "sha256": lock_plan.REFERENCE_LIBRARY_SHA256,
            },
            "accepted_candidate": {
                "package_path": CANDIDATE_PACKAGE_PATH,
                "sha256": lock_plan.CANDIDATE_LIBRARY_SHA256,
            },
            "feature_encoder": {
                "package_path": FEATURE_PACKAGE_PATH,
                "sha256": lock_plan.FEATURE_ENCODER_SHA256,
            },
            "image": dict(lock_open.IMAGE),
            "allocation": dict(lock_open.ALLOCATION),
            "launch_target": transport.build_launch_target(),
            "cost_guard": transport.build_cost_guard(),
            "schedule": {
                "scope": lock_plan.PLAN_SCOPE,
                "contract_hand_indices": list(runner.CONTRACT_HAND_INDICES),
                "source_roles": list(SOURCE_ROLES),
                "shards_per_role": SHARDS_PER_ROLE,
                "hands_per_shard": HANDS_PER_SHARD,
                "logical_job_count": MAX_LOGICAL_JOBS,
                "mapping": "one_source_ten_hand_arithmetic_shard_per_vm",
                "max_concurrent_vms": MAX_CONCURRENT_VMS,
            },
            "job_manifests": jobs,
            "source_entries": entries,
            "source_entry_count": len(entries),
            "checkpoint": {
                "unit": "completed_source_hand",
                "upload_after_each_hand": True,
                "immutable_write_once": True,
                "resume_verifies_all_artifacts": True,
                "done_uploaded_last": True,
            },
            "heartbeat": {
                "schema": transport.HEARTBEAT_SCHEMA,
                "required": True,
                "interval_seconds": transport.HEARTBEAT_INTERVAL_SECONDS,
                "remote_scope": "progress_only",
            },
            "spot_execution_authorized": False,
            "performance_lock_authorized": False,
            "quality_pilot_authorized": False,
            "training_eligible": False,
            "current_profile_changed": False,
            "named_profile_added": False,
            "runtime_policy_activated": False,
            "m31_complete": False,
            "gcloud_invoked": False,
        }
        _write_once(stage / MANIFEST_NAME, manifest)
        ready = {
            "schema": PACKAGE_READY_SCHEMA,
            "status": "immutable_local_performance_lock_package_complete",
            "run_name": run_name,
            "package_manifest_sha256": sha256_file(stage / MANIFEST_NAME),
            "source_sha256": manifest["source_sha256"],
            "startup_sha256": manifest["startup_sha256"],
            "plan_sha256": lock_plan.PRECONTENT_PLAN_SHA256,
            "run_contract_digest": lock_plan.LOCK_RUN_CONTRACT_DIGEST,
            "logical_job_count": MAX_LOGICAL_JOBS,
            "gcloud_invoked": False,
            "spot_vm_started": False,
            "current_profile_changed": False,
        }
        _write_once(stage / READY_NAME, ready)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return validate_package(destination)


def _validate_archive(
    source: Path, entries: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    tail_spot._validate_zip_entries(source, entries)
    with zipfile.ZipFile(source) as archive:

        def read(relative: str, label: str) -> dict[str, Any]:
            raw = archive.read(relative)
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict) or raw != canonical_bytes(value):
                raise ValueError(f"{label} is not canonical")
            return value

        plan = lock_plan.validate_precontent_plan(
            read(PLAN_PACKAGE_PATH, "packaged lock plan")
        )
        if hashlib.sha256(archive.read(PLAN_PACKAGE_PATH)).hexdigest() != (
            lock_plan.PRECONTENT_PLAN_SHA256
        ):
            raise ValueError("packaged lock plan hash changed")
        claim = read(OPEN_CLAIM_PACKAGE_PATH, "packaged open claim")
        seal = read(ROOT_SEAL_PACKAGE_PATH, "packaged root seal")
        materialization = read(MATERIALIZATION_PACKAGE_PATH, "packaged materialization")
        if (
            claim.get("schema") != lock_open.CLAIM_SCHEMA
            or claim.get("precontent_plan", {}).get("sha256")
            != lock_plan.PRECONTENT_PLAN_SHA256
            or claim.get("lock_run_contract_digest")
            != lock_plan.LOCK_RUN_CONTRACT_DIGEST
            or claim.get("ai_profiles_current", {}).get("sha256")
            != lock_plan.CURRENT_PROFILE_REGISTRY_SHA256
            or claim.get("restrictions", {}).get("alternate_seed_allowed") is not False
            or claim.get("restrictions", {}).get("reseed_allowed") is not False
            or claim.get("restrictions", {}).get("training_authorized") is not False
            or claim.get("restrictions", {}).get("runtime_activation_allowed")
            is not False
            or seal.get("schema") != lock_open.SEAL_SCHEMA
            or seal.get("global_claim_sha256") != canonical_sha256(claim)
            or seal.get("materialization_sha256") != canonical_sha256(materialization)
            or seal.get("plan_sha256") != lock_plan.PRECONTENT_PLAN_SHA256
            or seal.get("run_contract_digest") != lock_plan.LOCK_RUN_CONTRACT_DIGEST
            or seal.get("root_count") != 100
            or seal.get("observation_count") != 200
            or seal.get("development_comparison", {}).get(
                "lock_fingerprint_overlap_count"
            )
            != 0
            or seal.get("development_comparison", {}).get(
                "lock_root_hash_overlap_count"
            )
            != 0
            or seal.get("development_comparison", {}).get("lock_seed_overlap_count")
            != 0
        ):
            raise ValueError("packaged performance-lock claim/seal changed")
        root_hashes: list[str] = []
        contract = runner.validate_run_contract(plan["run_contract"])
        for index in runner.CONTRACT_HAND_INDICES:
            relative = f"{ROOT_PACKAGE_DIR}/hand_{index:03d}.json"
            value = read(relative, f"packaged lock root {index}")
            runner._validate_root_artifact(contract, value, index=index)
            root_hashes.append(canonical_sha256(value))
        if (
            root_hashes != seal["root_artifact_sha256"]
            or canonical_sha256(root_hashes) != seal["aggregate_root_sha256"]
        ):
            raise ValueError("packaged performance-lock root set changed")
        with tempfile.TemporaryDirectory(
            prefix="lock-materializer-parity-"
        ) as temporary:
            extracted = Path(temporary)
            relative_paths = {
                "windows": MATERIALIZER_WINDOWS_PROBE_PACKAGE_PATH,
                "linux": MATERIALIZER_LINUX_PROBE_PACKAGE_PATH,
                "receipt": MATERIALIZER_PARITY_RECEIPT_PACKAGE_PATH,
                "script": MATERIALIZER_PARITY_SCRIPT_PACKAGE_PATH,
                "rust": MATERIALIZER_RUST_SOURCE_PACKAGE_PATH,
                "cargo": MATERIALIZER_CARGO_LOCK_PACKAGE_PATH,
                "claim": OPEN_CLAIM_PACKAGE_PATH,
                "materialization": MATERIALIZATION_PACKAGE_PATH,
                "seal": ROOT_SEAL_PACKAGE_PATH,
                "ai_profiles": "src/ofc_regular/ai_profiles.py",
            }
            paths: dict[str, Path] = {}
            for label, relative in relative_paths.items():
                path = extracted / label
                path.write_bytes(archive.read(relative))
                paths[label] = path
            _validate_materializer_platform_parity(
                windows_probe_path=paths["windows"],
                linux_probe_path=paths["linux"],
                parity_receipt_path=paths["receipt"],
                parity_script_path=paths["script"],
                rust_source_path=paths["rust"],
                cargo_lock_path=paths["cargo"],
                global_claim_path=paths["claim"],
                materialization_path=paths["materialization"],
                seal_path=paths["seal"],
                ai_profiles_path=paths["ai_profiles"],
            )
    return plan, claim, seal, materialization


def validate_package(run_dir: str | Path) -> dict[str, Any]:
    target = Path(run_dir).resolve()
    manifest = _read_canonical(target / MANIFEST_NAME, "lock package manifest")
    ready = _read_canonical(target / READY_NAME, "lock package ready")
    transport._exact_keys(manifest, _PACKAGE_KEYS, "lock package manifest")
    transport._exact_keys(ready, _READY_KEYS, "lock package ready")
    source = target / SOURCE_NAME
    startup = target / STARTUP_NAME
    entries = manifest.get("source_entries")
    if (
        transport._SAFE_RUN.fullmatch(str(manifest.get("run_name", ""))) is None
        or not source.is_file()
        or source.is_symlink()
        or not startup.is_file()
        or startup.is_symlink()
        or not isinstance(entries, Mapping)
    ):
        raise ValueError("performance-lock package files changed")
    plan, claim, seal, materialization = _validate_archive(source, entries)
    contract = runner.validate_run_contract(manifest["run_contract"])
    jobs = manifest.get("job_manifests")
    if not isinstance(jobs, list) or len(jobs) != MAX_LOGICAL_JOBS:
        raise ValueError("performance-lock package job count changed")
    plan_jobs = plan["jobs"]
    for raw, plan_job, identifier in zip(
        jobs, plan_jobs, authorized_job_ids(), strict=True
    ):
        if not isinstance(raw, Mapping):
            raise ValueError("performance-lock job record changed")
        transport._exact_keys(raw, _JOB_RECORD_KEYS, "lock job record")
        job_path = target / str(raw["path"])
        expected_manifest = runner.build_shard_manifest(
            run_contract=contract,
            source_role=plan_job["source_role"],
            work_hand_indices=plan_job["work_hand_indices"],
        )
        if (
            raw.get("job_id") != identifier
            or raw.get("source_role") != plan_job["source_role"]
            or raw.get("shard_index") != plan_job["shard_index"]
            or raw.get("work_hand_indices") != plan_job["work_hand_indices"]
            or raw.get("sha256") != sha256_file(job_path)
            or raw.get("bytes") != job_path.stat().st_size
            or _read_canonical(job_path, f"lock job {identifier}") != expected_manifest
        ):
            raise ValueError("performance-lock job binding changed")
    expected_qualification = _lock_qualification(
        plan=plan,
        claim=claim,
        seal=seal,
        materialization=materialization,
    )
    expected_false = (
        "spot_execution_authorized",
        "performance_lock_authorized",
        "quality_pilot_authorized",
        "training_eligible",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
        "m31_complete",
        "gcloud_invoked",
    )
    if (
        manifest.get("schema") != PACKAGE_SCHEMA
        or manifest.get("status")
        != "immutable_performance_lock_package_ready_not_authorized"
        or manifest.get("source_name") != SOURCE_NAME
        or manifest.get("source_sha256") != sha256_file(source)
        or manifest.get("source_bytes") != source.stat().st_size
        or manifest.get("startup_name") != STARTUP_NAME
        or manifest.get("startup_sha256") != sha256_file(startup)
        or manifest.get("plan_package_path") != PLAN_PACKAGE_PATH
        or manifest.get("plan_sha256") != lock_plan.PRECONTENT_PLAN_SHA256
        or manifest.get("tail_qualification") != expected_qualification
        or contract != plan["run_contract"]
        or runner.contract_variant(contract)
        != getattr(
            lock_plan,
            "CANDIDATE_VARIANT",
            runner.CANDIDATE02_PERFORMANCE_LOCK_VARIANT,
        )
        or manifest.get("run_contract_digest") != lock_plan.LOCK_RUN_CONTRACT_DIGEST
        or manifest.get("accepted_reference")
        != {
            "package_path": REFERENCE_PACKAGE_PATH,
            "sha256": lock_plan.REFERENCE_LIBRARY_SHA256,
        }
        or manifest.get("accepted_candidate")
        != {
            "package_path": CANDIDATE_PACKAGE_PATH,
            "sha256": lock_plan.CANDIDATE_LIBRARY_SHA256,
        }
        or manifest.get("feature_encoder")
        != {
            "package_path": FEATURE_PACKAGE_PATH,
            "sha256": lock_plan.FEATURE_ENCODER_SHA256,
        }
        or manifest.get("image") != lock_open.IMAGE
        or manifest.get("allocation") != lock_open.ALLOCATION
        or manifest.get("launch_target") != transport.build_launch_target()
        or transport.validate_cost_guard(manifest.get("cost_guard"))
        != transport.build_cost_guard()
        or manifest.get("schedule")
        != {
            "scope": lock_plan.PLAN_SCOPE,
            "contract_hand_indices": list(runner.CONTRACT_HAND_INDICES),
            "source_roles": list(SOURCE_ROLES),
            "shards_per_role": SHARDS_PER_ROLE,
            "hands_per_shard": HANDS_PER_SHARD,
            "logical_job_count": MAX_LOGICAL_JOBS,
            "mapping": "one_source_ten_hand_arithmetic_shard_per_vm",
            "max_concurrent_vms": MAX_CONCURRENT_VMS,
        }
        or manifest.get("source_entry_count") != len(entries)
        or any(manifest.get(field) is not False for field in expected_false)
    ):
        raise ValueError("performance-lock package boundary changed")
    expected_ready = {
        "schema": PACKAGE_READY_SCHEMA,
        "status": "immutable_local_performance_lock_package_complete",
        "run_name": manifest["run_name"],
        "package_manifest_sha256": sha256_file(target / MANIFEST_NAME),
        "source_sha256": manifest["source_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "plan_sha256": lock_plan.PRECONTENT_PLAN_SHA256,
        "run_contract_digest": lock_plan.LOCK_RUN_CONTRACT_DIGEST,
        "logical_job_count": MAX_LOGICAL_JOBS,
        "gcloud_invoked": False,
        "spot_vm_started": False,
        "current_profile_changed": False,
    }
    if ready != expected_ready:
        raise ValueError("performance-lock package ready chain changed")
    return manifest


def _global_spot_claim_path(root_claim: Mapping[str, Any]) -> Path:
    root_claim_path = Path(str(root_claim["global_claim_path"])).resolve()
    if root_claim_path != Path(lock_open.DEFAULT_GLOBAL_CLAIM_PATH).resolve():
        raise ValueError("performance-lock global root claim path changed")
    return root_claim_path.with_name(GLOBAL_SPOT_CLAIM_NAME)


def _global_spot_claim_payload(
    *,
    target: Path,
    manifest: Mapping[str, Any],
    root_claim: Mapping[str, Any],
    claimed_unix_ns: int,
) -> dict[str, Any]:
    payload = {
        "schema": GLOBAL_SPOT_CLAIM_SCHEMA,
        "status": "global_one_shot_spot_identity_claimed_before_authorization",
        # These are signed lexical identities from the immutable root claim.
        # Re-resolving them on Windows can restore on-disk casing after the
        # root claim deliberately normalized it with normcase().  Linux
        # startup validation is byte/case-sensitive, so preserve the exact
        # claimed strings here.
        "global_root_claim_path": str(root_claim["global_claim_path"]),
        "global_root_claim_sha256": canonical_sha256(root_claim),
        "lock_output_directory": str(root_claim["lock_output_directory"]),
        "package_run_directory": str(target.resolve()),
        "run_name": manifest["run_name"],
        "package_manifest_sha256": sha256_file(target / MANIFEST_NAME),
        "source_sha256": manifest["source_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "precontent_plan_sha256": lock_plan.PRECONTENT_PLAN_SHA256,
        "root_seal_sha256": manifest["tail_qualification"]["root_seal_sha256"],
        "run_contract_digest": lock_plan.LOCK_RUN_CONTRACT_DIGEST,
        "authorized_job_ids": list(authorized_job_ids()),
        "max_initial_jobs": MAX_LOGICAL_JOBS,
        "max_resume_attempts": 1,
        "alternate_package_authorization_allowed": False,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "runtime_policy_activated": False,
        "claimed_unix_ns": claimed_unix_ns,
    }
    if set(payload) != _GLOBAL_SPOT_CLAIM_KEYS:
        raise AssertionError("performance-lock global Spot claim schema changed")
    return payload


def validate_global_spot_claim(
    run_dir: str | Path, manifest: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    target = Path(run_dir).resolve()
    current_manifest = validate_package(target) if manifest is None else dict(manifest)
    packaged_claim = _read_packaged_open_claim(target, current_manifest)
    root_claim_path = Path(str(packaged_claim["global_claim_path"])).resolve()
    actual_root_claim = _read_canonical(
        root_claim_path, "global performance-lock root claim"
    )
    if (
        actual_root_claim != packaged_claim
        or canonical_sha256(actual_root_claim)
        != current_manifest["tail_qualification"]["open_claim_sha256"]
    ):
        raise ValueError("global performance-lock root claim changed")
    claim_path = _global_spot_claim_path(actual_root_claim)
    claim = _read_canonical(claim_path, "global performance-lock Spot claim")
    transport._exact_keys(
        claim, _GLOBAL_SPOT_CLAIM_KEYS, "global performance-lock Spot claim"
    )
    opened = claim.get("claimed_unix_ns")
    if (
        not isinstance(opened, int)
        or isinstance(opened, bool)
        or opened <= 0
        or claim
        != _global_spot_claim_payload(
            target=target,
            manifest=current_manifest,
            root_claim=actual_root_claim,
            claimed_unix_ns=opened,
        )
    ):
        raise ValueError("global performance-lock Spot claim changed")
    return claim


def _acquire_global_spot_claim(
    *, target: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    root_claim = _read_packaged_open_claim(target, manifest)
    root_claim_path = Path(str(root_claim["global_claim_path"])).resolve()
    actual_root_claim = _read_canonical(
        root_claim_path, "global performance-lock root claim"
    )
    if (
        actual_root_claim != root_claim
        or canonical_sha256(actual_root_claim)
        != manifest["tail_qualification"]["open_claim_sha256"]
    ):
        raise ValueError("global performance-lock root claim changed")
    claim_path = _global_spot_claim_path(actual_root_claim)
    if claim_path.exists():
        return validate_global_spot_claim(target, manifest)
    payload = _global_spot_claim_payload(
        target=target,
        manifest=manifest,
        root_claim=actual_root_claim,
        claimed_unix_ns=time.time_ns(),
    )
    lock_open._write_once_durable(claim_path, payload)
    return validate_global_spot_claim(target, manifest)


def authorize_launch(run_dir: str | Path) -> dict[str, Any]:
    target = Path(run_dir).resolve()
    manifest = validate_package(target)
    global_spot_claim = _acquire_global_spot_claim(target=target, manifest=manifest)
    qualification = manifest["tail_qualification"]
    authorization = {
        "schema": AUTHORIZATION_SCHEMA,
        "status": "explicit_one_shot_performance_lock_spot_authorization",
        "run_name": manifest["run_name"],
        "package_manifest_sha256": sha256_file(target / MANIFEST_NAME),
        "source_sha256": manifest["source_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "plan_sha256": manifest["plan_sha256"],
        "tail_summary_sha256": qualification["summary_sha256"],
        "tail_validation_sha256": qualification["validation_sha256"],
        "run_contract_digest": manifest["run_contract_digest"],
        "launch_target": manifest["launch_target"],
        "cost_guard": manifest["cost_guard"],
        "authorized_job_ids": list(authorized_job_ids()),
        "logical_job_count": MAX_LOGICAL_JOBS,
        "global_spot_claim": global_spot_claim,
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
    _write_once(target / AUTHORIZATION_NAME, authorization)
    validate_launch_authorization(target)
    return authorization


def validate_launch_authorization(
    run_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = Path(run_dir).resolve()
    manifest = validate_package(target)
    global_spot_claim = validate_global_spot_claim(target, manifest)
    authorization = _read_canonical(
        target / AUTHORIZATION_NAME, "lock launch authorization"
    )
    transport._exact_keys(
        authorization, _AUTHORIZATION_KEYS, "lock launch authorization"
    )
    qualification = manifest["tail_qualification"]
    if (
        authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("status")
        != "explicit_one_shot_performance_lock_spot_authorization"
        or authorization.get("run_name") != manifest["run_name"]
        or authorization.get("package_manifest_sha256")
        != sha256_file(target / MANIFEST_NAME)
        or authorization.get("source_sha256") != manifest["source_sha256"]
        or authorization.get("startup_sha256") != manifest["startup_sha256"]
        or authorization.get("plan_sha256") != lock_plan.PRECONTENT_PLAN_SHA256
        or authorization.get("tail_summary_sha256") != qualification["summary_sha256"]
        or authorization.get("tail_validation_sha256")
        != qualification["validation_sha256"]
        or authorization.get("run_contract_digest")
        != lock_plan.LOCK_RUN_CONTRACT_DIGEST
        or authorization.get("launch_target") != transport.build_launch_target()
        or transport.validate_cost_guard(authorization.get("cost_guard"))
        != transport.build_cost_guard()
        or authorization.get("authorized_job_ids") != list(authorized_job_ids())
        or authorization.get("logical_job_count") != MAX_LOGICAL_JOBS
        or authorization.get("global_spot_claim") != global_spot_claim
        or authorization.get("performance_development_only") is not False
        or authorization.get("spot_execution_authorized") is not True
        or authorization.get("performance_lock_authorized") is not True
        or authorization.get("quality_pilot_authorized") is not False
        or any(
            authorization.get(field) is not False
            for field in (
                "training_eligible",
                "current_profile_changed",
                "named_profile_added",
                "runtime_policy_activated",
            )
        )
        or not transport._valid_timestamp(authorization.get("authorized_unix_seconds"))
    ):
        raise ValueError("performance-lock launch authorization changed")
    return manifest, authorization


def _acquire_launch_claim(
    *,
    target: Path,
    manifest: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    claim = {
        "schema": LAUNCH_CLAIM_SCHEMA,
        "status": "exclusive_performance_lock_launch_claim_before_remote_mutation",
        "run_name": manifest["run_name"],
        "selected_job_ids": list(authorized_job_ids()),
        "package_manifest_sha256": sha256_file(target / MANIFEST_NAME),
        "launch_authorization_sha256": sha256_file(target / AUTHORIZATION_NAME),
        "plan_sha256": lock_plan.PRECONTENT_PLAN_SHA256,
        "run_contract_digest": lock_plan.LOCK_RUN_CONTRACT_DIGEST,
        "launch_target": transport.build_launch_target(),
        "cost_guard_sha256": canonical_sha256(manifest["cost_guard"]),
        "preflight_sha256": canonical_sha256(preflight),
        "claimed_unix_seconds": time.time(),
        "crash_reuse_authorized": False,
    }
    transport._exact_keys(claim, _LAUNCH_CLAIM_KEYS, "lock launch claim")
    _write_once(target / LAUNCH_CLAIM_NAME, claim)
    return claim


def _launch_jobs_unlocked(
    *,
    run_dir: str | Path,
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    target = Path(run_dir).resolve()
    manifest, authorization = validate_launch_authorization(target)
    if (target / LAUNCH_CLAIM_NAME).exists() or (target / LAUNCH_RESULT_NAME).exists():
        raise FileExistsError("performance-lock initial launch is already claimed")
    selected = authorized_job_ids()
    preflight = transport.preflight_launch(
        manifest=manifest,
        selected=selected,
        project=project,
        bucket=bucket,
    )
    preflight = transport._validate_launch_preflight(preflight, manifest=manifest)
    claim = _acquire_launch_claim(target=target, manifest=manifest, preflight=preflight)
    prefix = transport._publish_package(
        target=target, manifest=manifest, project=project, bucket=bucket
    )
    claim_uri = f"{prefix}/control/{LAUNCH_CLAIM_NAME}"
    tail_spot._publish_once(target / LAUNCH_CLAIM_NAME, claim_uri, project=project)
    created, failures = transport._launch_selected(
        target=target,
        manifest=manifest,
        authorization=authorization,
        selected=selected,
        prefix=prefix,
        project=project,
        attempt_index=0,
        claim_uri=claim_uri,
        claim_sha256=sha256_file(target / LAUNCH_CLAIM_NAME),
    )
    cleanup = (
        transport._cleanup_instances(
            manifest=manifest,
            selected=selected,
            project=project,
            attempt_index=0,
        )
        if failures
        else []
    )
    result = {
        "schema": LAUNCH_RESULT_SCHEMA,
        "status": (
            "performance_lock_jobs_created"
            if not failures
            else "performance_lock_create_failed_cleanup_attempted"
        ),
        "run_name": manifest["run_name"],
        "selected_job_ids": list(selected),
        "created": created,
        "failures": failures,
        "cleanup": cleanup,
        "cleanup_compute_stopped_or_absent": (
            True
            if not failures
            else all(row["compute_stopped_or_absent"] for row in cleanup)
        ),
        "logical_job_count": len(created),
        "max_logical_jobs": MAX_LOGICAL_JOBS,
        "max_concurrent_vms": MAX_CONCURRENT_VMS,
        "machine_type": EXPECTED_MACHINE_TYPE,
        "process_count": transport.PROCESS_COUNT,
        "rayon_threads_per_process": transport.RAYON_THREADS,
        "run_contract_digest": manifest["run_contract_digest"],
        "launch_target": manifest["launch_target"],
        "cost_guard": manifest["cost_guard"],
        "preflight": preflight,
        "launch_claim_sha256": sha256_file(target / LAUNCH_CLAIM_NAME),
        "production_fanout_authorized": False,
        "performance_lock_authorized": True,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "runtime_policy_activated": False,
    }
    _write_once(target / LAUNCH_RESULT_NAME, result)
    tail_spot._publish_once(
        target / LAUNCH_RESULT_NAME,
        f"{prefix}/control/{LAUNCH_RESULT_NAME}",
        project=project,
    )
    if failures:
        raise RuntimeError(
            "performance-lock instance creation failed; cleanup persisted"
        )
    return result


def launch_jobs(
    *,
    run_dir: str | Path,
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    with _lifecycle_mutex(run_dir, "initial-launch"):
        return _launch_jobs_unlocked(
            run_dir=run_dir,
            project=project,
            bucket=bucket,
        )


def validate_launch_chain(
    run_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    target = Path(run_dir).resolve()
    manifest, _authorization = validate_launch_authorization(target)
    claim = _read_canonical(target / LAUNCH_CLAIM_NAME, "lock launch claim")
    result = _read_canonical(target / LAUNCH_RESULT_NAME, "lock launch result")
    transport._exact_keys(claim, _LAUNCH_CLAIM_KEYS, "lock launch claim")
    transport._exact_keys(result, _LAUNCH_RESULT_KEYS, "lock launch result")
    selected = list(authorized_job_ids())
    expected_names = [
        transport._instance_name(manifest, identifier, 0) for identifier in selected
    ]
    created = result.get("created")
    created_valid = (
        isinstance(created, list)
        and len(created) == MAX_LOGICAL_JOBS
        and [row.get("job_id") for row in created] == selected
        and [row.get("instance") for row in created] == expected_names
        and all(
            isinstance(row, Mapping)
            and row.get("attempt_index") == 0
            and row.get("status") == "created"
            for row in created
        )
    )
    preflight = result.get("preflight")
    if (
        claim.get("schema") != LAUNCH_CLAIM_SCHEMA
        or claim.get("status")
        != "exclusive_performance_lock_launch_claim_before_remote_mutation"
        or claim.get("run_name") != manifest["run_name"]
        or claim.get("selected_job_ids") != selected
        or claim.get("package_manifest_sha256") != sha256_file(target / MANIFEST_NAME)
        or claim.get("launch_authorization_sha256")
        != sha256_file(target / AUTHORIZATION_NAME)
        or claim.get("plan_sha256") != lock_plan.PRECONTENT_PLAN_SHA256
        or claim.get("run_contract_digest") != lock_plan.LOCK_RUN_CONTRACT_DIGEST
        or claim.get("launch_target") != transport.build_launch_target()
        or claim.get("cost_guard_sha256") != canonical_sha256(manifest["cost_guard"])
        or not isinstance(preflight, Mapping)
        or claim.get("preflight_sha256") != canonical_sha256(preflight)
        or not transport._valid_timestamp(claim.get("claimed_unix_seconds"))
        or claim.get("crash_reuse_authorized") is not False
        or result.get("schema") != LAUNCH_RESULT_SCHEMA
        or result.get("status") != "performance_lock_jobs_created"
        or result.get("run_name") != manifest["run_name"]
        or result.get("selected_job_ids") != selected
        or not created_valid
        or result.get("failures") != []
        or result.get("cleanup") != []
        or result.get("cleanup_compute_stopped_or_absent") is not True
        or result.get("logical_job_count") != MAX_LOGICAL_JOBS
        or result.get("max_logical_jobs") != MAX_LOGICAL_JOBS
        or result.get("max_concurrent_vms") != MAX_CONCURRENT_VMS
        or result.get("machine_type") != EXPECTED_MACHINE_TYPE
        or result.get("process_count") != transport.PROCESS_COUNT
        or result.get("rayon_threads_per_process") != transport.RAYON_THREADS
        or result.get("run_contract_digest") != lock_plan.LOCK_RUN_CONTRACT_DIGEST
        or result.get("launch_target") != manifest["launch_target"]
        or result.get("cost_guard") != manifest["cost_guard"]
        or result.get("launch_claim_sha256") != sha256_file(target / LAUNCH_CLAIM_NAME)
        or result.get("production_fanout_authorized") is not False
        or result.get("performance_lock_authorized") is not True
        or result.get("quality_pilot_authorized") is not False
        or any(
            result.get(field) is not False
            for field in (
                "training_eligible",
                "current_profile_changed",
                "runtime_policy_activated",
            )
        )
    ):
        raise ValueError("performance-lock launch hash chain changed")
    transport._validate_launch_preflight(preflight, manifest=manifest)
    return manifest, claim, result


def preflight_resume(
    *,
    run_dir: str | Path,
    selected: Sequence[str],
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    """Check exact incomplete jobs without opening any hand result content."""

    target = Path(run_dir).resolve()
    manifest, _claim, _result = validate_launch_chain(target)
    if (target / RESULT_OPEN_CLAIM_NAME).exists():
        raise FileExistsError(
            "result content is already consumed; compute resume forbidden"
        )
    if (target / RESUME_CLAIM_NAME).exists() or (target / RESUME_RESULT_NAME).exists():
        raise FileExistsError("performance-lock attempt-1 resume already claimed")
    selected_jobs = _bounded_jobs(selected)
    prefix = transport._run_prefix(manifest, project=project, bucket=bucket)
    instances = transport._instance_rows(manifest=manifest, project=project)
    relevant = {
        transport._instance_name(manifest, job, attempt)
        for job in authorized_job_ids()
        for attempt in (0, 1)
    }
    active_initial = [
        str(row.get("name"))
        for row in instances
        if row.get("name") in relevant
        and not str(row.get("name")).endswith("-a01")
        and row.get("status") != "TERMINATED"
    ]
    existing_resume = [
        str(row.get("name"))
        for row in instances
        if row.get("name") in relevant and str(row.get("name")).endswith("-a01")
    ]
    if active_initial or existing_resume:
        raise FileExistsError("performance-lock resume instance eligibility failed")
    completed: list[str] = []
    incomplete: list[str] = []
    done_uris: dict[str, str] = {}
    for identifier in authorized_job_ids():
        uri = f"{prefix}/results/jobs/{identifier}/DONE.json"
        done_uris[identifier] = uri
        if transport._object_exists(uri, project=project):
            completed.append(identifier)
        else:
            incomplete.append(identifier)
    if not incomplete:
        raise ValueError("performance-lock resume has no incomplete jobs")
    if tuple(incomplete) != selected_jobs:
        raise ValueError("resume must select exact DONE-absent job set")
    quota = transport._quota_snapshot(
        project=project,
        required_vcpus=len(selected_jobs) * transport.VCPUS_PER_VM,
    )
    return {
        "schema": RESUME_PREFLIGHT_SCHEMA,
        "status": "performance_lock_attempt1_metadata_only_checks_passed",
        "checked_unix_seconds": time.time(),
        "region": DEFAULT_REGION,
        "attempt_index": 1,
        "selected_job_ids": list(selected_jobs),
        "selected_initial_instance_names": [
            transport._instance_name(manifest, job, 0) for job in selected_jobs
        ],
        "selected_resume_instance_names": [
            transport._instance_name(manifest, job, 1) for job in selected_jobs
        ],
        "required_vcpus": len(selected_jobs) * transport.VCPUS_PER_VM,
        "quota": quota,
        "no_active_initial_instances": True,
        "resume_instance_names_absent": True,
        "completed_job_ids_from_done_existence_only": completed,
        "incomplete_job_ids": incomplete,
        "done_uris": done_uris,
        "result_content_read": False,
        "all_incomplete_jobs_selected": True,
        "cost_guard_sha256": canonical_sha256(manifest["cost_guard"]),
        "max_cumulative_vm_jobs": transport.MAX_CUMULATIVE_VM_JOBS,
        "all_attempts_estimated_max_compute_usd": (
            transport.ALL_ATTEMPTS_ESTIMATED_MAX_COMPUTE_USD
        ),
        "phase_compute_cap_usd": transport.PHASE_COMPUTE_CAP_USD,
    }


def _validate_resume_preflight(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    selected: Sequence[str],
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    payload = dict(value)
    selected_jobs = _bounded_jobs(selected)
    completed = [job for job in authorized_job_ids() if job not in set(selected_jobs)]
    prefix = transport._run_prefix(manifest, project=project, bucket=bucket)
    expected_uris = {
        job: f"{prefix}/results/jobs/{job}/DONE.json" for job in authorized_job_ids()
    }
    quota = payload.get("quota")
    quota_ok = isinstance(quota, Mapping) and set(quota) == {
        "CPUS",
        "PREEMPTIBLE_CPUS",
    }
    if quota_ok:
        quota_ok = all(
            isinstance(row, Mapping)
            and row.get("available") == row.get("limit") - row.get("usage")
            and row.get("available") >= len(selected_jobs) * transport.VCPUS_PER_VM
            for row in quota.values()
        )
    if (
        payload.get("schema") != RESUME_PREFLIGHT_SCHEMA
        or payload.get("status")
        != "performance_lock_attempt1_metadata_only_checks_passed"
        or not transport._valid_timestamp(payload.get("checked_unix_seconds"))
        or payload.get("region") != DEFAULT_REGION
        or payload.get("attempt_index") != 1
        or payload.get("selected_job_ids") != list(selected_jobs)
        or payload.get("selected_initial_instance_names")
        != [transport._instance_name(manifest, job, 0) for job in selected_jobs]
        or payload.get("selected_resume_instance_names")
        != [transport._instance_name(manifest, job, 1) for job in selected_jobs]
        or payload.get("required_vcpus") != len(selected_jobs) * transport.VCPUS_PER_VM
        or not quota_ok
        or payload.get("no_active_initial_instances") is not True
        or payload.get("resume_instance_names_absent") is not True
        or payload.get("completed_job_ids_from_done_existence_only") != completed
        or payload.get("incomplete_job_ids") != list(selected_jobs)
        or payload.get("done_uris") != expected_uris
        or payload.get("result_content_read") is not False
        or payload.get("all_incomplete_jobs_selected") is not True
        or payload.get("cost_guard_sha256") != canonical_sha256(manifest["cost_guard"])
        or payload.get("max_cumulative_vm_jobs") != transport.MAX_CUMULATIVE_VM_JOBS
        or payload.get("all_attempts_estimated_max_compute_usd")
        != transport.ALL_ATTEMPTS_ESTIMATED_MAX_COMPUTE_USD
        or payload.get("phase_compute_cap_usd") != transport.PHASE_COMPUTE_CAP_USD
    ):
        raise ValueError("performance-lock resume preflight changed")
    return payload


def _resume_jobs_unlocked(
    *,
    run_dir: str | Path,
    selected: Sequence[str],
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    target = Path(run_dir).resolve()
    manifest, authorization = validate_launch_authorization(target)
    validate_launch_chain(target)
    preflight = preflight_resume(
        run_dir=target,
        selected=selected,
        project=project,
        bucket=bucket,
    )
    selected_jobs = _bounded_jobs(selected)
    preflight = _validate_resume_preflight(
        preflight,
        manifest=manifest,
        selected=selected_jobs,
        project=project,
        bucket=bucket,
    )
    claim = {
        "schema": RESUME_CLAIM_SCHEMA,
        "status": ("exclusive_performance_lock_attempt1_claim_before_remote_mutation"),
        "run_name": manifest["run_name"],
        "attempt_index": 1,
        "selected_job_ids": list(selected_jobs),
        "initial_launch_claim_sha256": sha256_file(target / LAUNCH_CLAIM_NAME),
        "initial_launch_result_sha256": sha256_file(target / LAUNCH_RESULT_NAME),
        "package_manifest_sha256": sha256_file(target / MANIFEST_NAME),
        "launch_authorization_sha256": sha256_file(target / AUTHORIZATION_NAME),
        "plan_sha256": lock_plan.PRECONTENT_PLAN_SHA256,
        "run_contract_digest": lock_plan.LOCK_RUN_CONTRACT_DIGEST,
        "launch_target": transport.build_launch_target(),
        "cost_guard_sha256": canonical_sha256(manifest["cost_guard"]),
        "preflight_sha256": canonical_sha256(preflight),
        "claimed_unix_seconds": time.time(),
        "third_attempt_authorized": False,
    }
    transport._exact_keys(claim, _RESUME_CLAIM_KEYS, "lock resume claim")
    _write_once(target / RESUME_CLAIM_NAME, claim)
    prefix = transport._run_prefix(manifest, project=project, bucket=bucket)
    claim_uri = f"{prefix}/resume/{RESUME_CLAIM_NAME}"
    tail_spot._publish_once(target / RESUME_CLAIM_NAME, claim_uri, project=project)
    created, failures = transport._launch_selected(
        target=target,
        manifest=manifest,
        authorization=authorization,
        selected=selected_jobs,
        prefix=prefix,
        project=project,
        attempt_index=1,
        claim_uri=claim_uri,
        claim_sha256=sha256_file(target / RESUME_CLAIM_NAME),
    )
    cleanup = (
        transport._cleanup_instances(
            manifest=manifest,
            selected=selected_jobs,
            project=project,
            attempt_index=1,
        )
        if failures
        else []
    )
    result = {
        "schema": RESUME_RESULT_SCHEMA,
        "status": (
            "performance_lock_resume_jobs_created"
            if not failures
            else "performance_lock_resume_create_failed_cleanup_attempted"
        ),
        "run_name": manifest["run_name"],
        "attempt_index": 1,
        "selected_job_ids": list(selected_jobs),
        "created": created,
        "failures": failures,
        "cleanup": cleanup,
        "cleanup_compute_stopped_or_absent": (
            True
            if not failures
            else all(row["compute_stopped_or_absent"] for row in cleanup)
        ),
        "logical_job_count": len(created),
        "max_resume_jobs": MAX_LOGICAL_JOBS,
        "max_cumulative_vm_jobs": transport.MAX_CUMULATIVE_VM_JOBS,
        "run_contract_digest": lock_plan.LOCK_RUN_CONTRACT_DIGEST,
        "launch_target": manifest["launch_target"],
        "cost_guard": manifest["cost_guard"],
        "preflight": preflight,
        "resume_claim_sha256": sha256_file(target / RESUME_CLAIM_NAME),
        "initial_launch_claim_sha256": sha256_file(target / LAUNCH_CLAIM_NAME),
        "initial_launch_result_sha256": sha256_file(target / LAUNCH_RESULT_NAME),
        "third_attempt_authorized": False,
        "performance_lock_authorized": True,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "runtime_policy_activated": False,
    }
    _write_once(target / RESUME_RESULT_NAME, result)
    tail_spot._publish_once(
        target / RESUME_RESULT_NAME,
        f"{prefix}/resume/{RESUME_RESULT_NAME}",
        project=project,
    )
    if failures:
        raise RuntimeError(
            "performance-lock attempt-1 creation failed; cleanup persisted"
        )
    return result


def resume_jobs(
    *,
    run_dir: str | Path,
    selected: Sequence[str],
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    with _lifecycle_mutex(run_dir, "attempt1-resume"):
        return _resume_jobs_unlocked(
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
    """Replay the only allowed attempt-1 resume without reading hand results."""

    target = Path(run_dir).resolve()
    current_manifest = (
        validate_launch_chain(target)[0] if manifest is None else dict(manifest)
    )
    claim_path = target / RESUME_CLAIM_NAME
    result_path = target / RESUME_RESULT_NAME
    if not claim_path.exists() and not result_path.exists():
        return None
    if not claim_path.is_file() or not result_path.is_file():
        raise ValueError("performance-lock resume hash chain is incomplete")
    claim = _read_canonical(claim_path, "lock resume claim")
    result = _read_canonical(result_path, "lock resume result")
    transport._exact_keys(claim, _RESUME_CLAIM_KEYS, "lock resume claim")
    transport._exact_keys(result, _RESUME_RESULT_KEYS, "lock resume result")
    selected_raw = claim.get("selected_job_ids")
    if not isinstance(selected_raw, list):
        raise ValueError("performance-lock resume selected jobs changed")
    selected = _bounded_jobs(selected_raw)
    preflight_raw = result.get("preflight")
    if not isinstance(preflight_raw, Mapping):
        raise ValueError("performance-lock resume preflight changed")
    preflight = _validate_resume_preflight(
        preflight_raw,
        manifest=current_manifest,
        selected=selected,
        project=project,
        bucket=bucket,
    )
    records = {str(row["job_id"]): row for row in current_manifest["job_manifests"]}
    expected_created = []
    for identifier in selected:
        record = records[identifier]
        ordinal = authorized_job_ids().index(identifier)
        expected_created.append(
            {
                "job_id": identifier,
                "source_role": record["source_role"],
                "shard_index": record["shard_index"],
                "work_hand_indices": record["work_hand_indices"],
                "instance": transport._instance_name(current_manifest, identifier, 1),
                "zone": DEFAULT_ZONES[ordinal % len(DEFAULT_ZONES)],
                "attempt_index": 1,
                "status": "created",
            }
        )
    if (
        claim.get("schema") != RESUME_CLAIM_SCHEMA
        or claim.get("status")
        != "exclusive_performance_lock_attempt1_claim_before_remote_mutation"
        or claim.get("run_name") != current_manifest["run_name"]
        or claim.get("attempt_index") != 1
        or claim.get("selected_job_ids") != list(selected)
        or claim.get("initial_launch_claim_sha256")
        != sha256_file(target / LAUNCH_CLAIM_NAME)
        or claim.get("initial_launch_result_sha256")
        != sha256_file(target / LAUNCH_RESULT_NAME)
        or claim.get("package_manifest_sha256") != sha256_file(target / MANIFEST_NAME)
        or claim.get("launch_authorization_sha256")
        != sha256_file(target / AUTHORIZATION_NAME)
        or claim.get("plan_sha256") != lock_plan.PRECONTENT_PLAN_SHA256
        or claim.get("run_contract_digest") != lock_plan.LOCK_RUN_CONTRACT_DIGEST
        or claim.get("launch_target") != transport.build_launch_target()
        or claim.get("cost_guard_sha256")
        != canonical_sha256(current_manifest["cost_guard"])
        or claim.get("preflight_sha256") != canonical_sha256(preflight)
        or not transport._valid_timestamp(claim.get("claimed_unix_seconds"))
        or claim.get("third_attempt_authorized") is not False
        or result.get("schema") != RESUME_RESULT_SCHEMA
        or result.get("status") != "performance_lock_resume_jobs_created"
        or result.get("run_name") != current_manifest["run_name"]
        or result.get("attempt_index") != 1
        or result.get("selected_job_ids") != list(selected)
        or result.get("created") != expected_created
        or result.get("failures") != []
        or result.get("cleanup") != []
        or result.get("cleanup_compute_stopped_or_absent") is not True
        or result.get("logical_job_count") != len(selected)
        or result.get("max_resume_jobs") != MAX_LOGICAL_JOBS
        or result.get("max_cumulative_vm_jobs") != transport.MAX_CUMULATIVE_VM_JOBS
        or result.get("run_contract_digest") != lock_plan.LOCK_RUN_CONTRACT_DIGEST
        or result.get("launch_target") != current_manifest["launch_target"]
        or result.get("cost_guard") != current_manifest["cost_guard"]
        or result.get("preflight") != preflight
        or result.get("resume_claim_sha256") != sha256_file(claim_path)
        or result.get("initial_launch_claim_sha256")
        != sha256_file(target / LAUNCH_CLAIM_NAME)
        or result.get("initial_launch_result_sha256")
        != sha256_file(target / LAUNCH_RESULT_NAME)
        or result.get("third_attempt_authorized") is not False
        or result.get("performance_lock_authorized") is not True
        or any(
            result.get(field) is not False
            for field in (
                "quality_pilot_authorized",
                "training_eligible",
                "current_profile_changed",
                "runtime_policy_activated",
            )
        )
    ):
        raise ValueError("performance-lock resume hash chain changed")
    return claim, result


def cloud_status(
    *,
    run_dir: str | Path,
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    manifest, _claim, _result = validate_launch_chain(run_dir)
    prefix = transport._run_prefix(manifest, project=project, bucket=bucket)
    listing = tail_spot._subprocess_run(
        ["gcloud", "storage", "ls", "--recursive", prefix, "--project", project],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    if listing.returncode != 0:
        raise RuntimeError("failed to inspect performance-lock GCS prefix")
    objects = listing.stdout.splitlines()
    instances = transport._instance_rows(manifest=manifest, project=project)
    expected_names = {
        transport._instance_name(manifest, job, attempt)
        for job in authorized_job_ids()
        for attempt in (0, 1)
    }
    selected_instances = [row for row in instances if row.get("name") in expected_names]
    if len({row.get("name") for row in selected_instances}) != len(selected_instances):
        raise ValueError("performance-lock instance name duplicated across zones")
    by_name = {row.get("name"): row for row in selected_instances}
    jobs: dict[str, Any] = {}
    for identifier in authorized_job_ids():
        result_fragment = f"/results/jobs/{identifier}/"
        progress_fragment = f"/progress/jobs/{identifier}/"
        initial = by_name.get(transport._instance_name(manifest, identifier, 0))
        resumed = by_name.get(transport._instance_name(manifest, identifier, 1))
        jobs[identifier] = {
            "done_present_unvalidated": any(
                uri.endswith(f"/results/jobs/{identifier}/DONE.json") for uri in objects
            ),
            "heartbeat_present": any(
                uri.endswith(f"/progress/jobs/{identifier}/heartbeat.json")
                for uri in objects
            ),
            "progress_object_count": sum(progress_fragment in uri for uri in objects),
            "result_object_count": sum(result_fragment in uri for uri in objects),
            "instance": (
                None
                if initial is None
                else {
                    "name": initial.get("name"),
                    "status": initial.get("status"),
                    "zone": str(initial.get("zone", "")).rsplit("/", 1)[-1],
                }
            ),
            "resume_instance": (
                None
                if resumed is None
                else {
                    "name": resumed.get("name"),
                    "status": resumed.get("status"),
                    "zone": str(resumed.get("zone", "")).rsplit("/", 1)[-1],
                }
            ),
        }
    done_count = sum(row["done_present_unvalidated"] for row in jobs.values())
    return {
        "schema": STATUS_SCHEMA,
        "status": (
            "all_performance_lock_done_present_claim_results_before_receive"
            if done_count == MAX_LOGICAL_JOBS
            else "incomplete"
        ),
        "run_name": manifest["run_name"],
        "done_present_count": done_count,
        "all_done_present_unvalidated": done_count == MAX_LOGICAL_JOBS,
        "jobs": jobs,
        "logical_job_count": MAX_LOGICAL_JOBS,
        "run_contract_digest": lock_plan.LOCK_RUN_CONTRACT_DIGEST,
        "result_content_claimed": Path(run_dir, RESULT_OPEN_CLAIM_NAME).is_file(),
        "performance_lock_authorized": True,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
    }


def _plan_job_by_id(plan: Mapping[str, Any], identifier: str) -> dict[str, Any]:
    rows = [row for row in plan["jobs"] if row["job_id"] == identifier]
    if len(rows) != 1:
        raise ValueError(f"performance-lock plan job is not unique: {identifier}")
    return dict(rows[0])


def _validate_done_marker_metadata(
    *,
    value: Mapping[str, Any],
    plan: Mapping[str, Any],
    identifier: str,
) -> dict[str, Any]:
    done = dict(value)
    runner._require_exact_keys(done, runner._DONE_KEYS, "lock DONE metadata")
    job = _plan_job_by_id(plan, identifier)
    role = job["source_role"]
    work = job["work_hand_indices"]
    contract = plan["run_contract"]
    artifacts = done.get("artifact_manifest")
    expected_paths = [
        path
        for index in work
        for path in (
            f"roots/hand_{index:03d}.json",
            f"hands/{role}/hand_{index:03d}.json",
        )
    ]
    if (
        done.get("schema")
        != getattr(
            lock_plan,
            "DONE_SCHEMA",
            runner.CANDIDATE02_PERFORMANCE_LOCK_DONE_SCHEMA,
        )
        or done.get("status") != "complete_source_isolated_shard"
        or done.get("contract_canonical_sha256")
        != contract["contract_canonical_sha256"]
        or done.get("run_contract_digest") != lock_plan.LOCK_RUN_CONTRACT_DIGEST
        or done.get("shard_manifest_sha256") != job["shard_manifest_sha256"]
        or done.get("source_role") != role
        or done.get("contract_hand_indices") != list(runner.CONTRACT_HAND_INDICES)
        or done.get("tail_hand_indices") != []
        or done.get("work_hand_indices") != work
        or done.get("completed_hand_indices") != work
        or done.get("budget") != contract["budget"]
        or done.get("allocation") != contract["allocation"]
        or done.get("reference_library_sha256") != lock_plan.REFERENCE_LIBRARY_SHA256
        or done.get("candidate_library_sha256") != lock_plan.CANDIDATE_LIBRARY_SHA256
        or done.get("native_library_sha256") != contract[f"{role}_library_sha256"]
        or done.get("artifact_count") != 2 * HANDS_PER_SHARD
        or not isinstance(artifacts, list)
        or len(artifacts) != 2 * HANDS_PER_SHARD
        or done.get("artifact_manifest_sha256") != canonical_sha256(artifacts)
        or [record.get("path") for record in artifacts] != expected_paths
        or any(
            not isinstance(record, Mapping)
            or set(record) != runner._ARTIFACT_RECORD_KEYS
            or record.get("source_role") != role
            or record.get("hand_index") not in work
            or not isinstance(record.get("bytes"), int)
            or isinstance(record.get("bytes"), bool)
            or record.get("bytes") <= 0
            or not transport._is_sha256(record.get("sha256"))
            for record in artifacts
        )
        or any(
            done.get(field) is not False
            for field in (
                "training_eligible",
                "quality_evidence",
                "promotion_evidence",
                "current_profile_changed",
            )
        )
    ):
        raise ValueError(f"performance-lock DONE metadata changed: {identifier}")
    return done


def _active_compute_instances(
    manifest: Mapping[str, Any], *, project: str
) -> list[str]:
    expected = {
        transport._instance_name(manifest, job, attempt)
        for job in authorized_job_ids()
        for attempt in (0, 1)
    }
    rows = transport._instance_rows(manifest=manifest, project=project)
    return sorted(
        str(row.get("name"))
        for row in rows
        if row.get("name") in expected and row.get("status") != "TERMINATED"
    )


def _claimed_done_marker_records(
    *,
    done_dir: Path,
    plan: Mapping[str, Any],
    prefix: str,
) -> list[dict[str, Any]]:
    expected_files = {f"{identifier}.json" for identifier in authorized_job_ids()}
    actual_files = {
        path.relative_to(done_dir).as_posix()
        for path in done_dir.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("performance-lock claimed DONE file set changed")
    records: list[dict[str, Any]] = []
    for identifier in authorized_job_ids():
        path = done_dir / f"{identifier}.json"
        value = _validate_done_marker_metadata(
            value=_read_canonical(path, f"claimed lock DONE {identifier}"),
            plan=plan,
            identifier=identifier,
        )
        records.append(
            {
                "job_id": identifier,
                "uri": f"{prefix}/results/jobs/{identifier}/DONE.json",
                "file_sha256": sha256_file(path),
                "canonical_sha256": canonical_sha256(value),
                "source_role": value["source_role"],
                "work_hand_indices": value["work_hand_indices"],
                "artifact_manifest_sha256": value["artifact_manifest_sha256"],
            }
        )
    return records


def _claim_results_unlocked(
    *,
    run_dir: str | Path,
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    """Claim the complete DONE set before downloading any hand result."""

    target = Path(run_dir).resolve()
    manifest, _launch_claim, _launch_result = validate_launch_chain(target)
    validate_resume_chain(
        run_dir=target,
        manifest=manifest,
        project=project,
        bucket=bucket,
    )
    prefix = transport._run_prefix(manifest, project=project, bucket=bucket)
    if (target / RESULT_OPEN_CLAIM_NAME).exists():
        claim = validate_result_open_claim(
            run_dir=target,
            project=project,
            bucket=bucket,
            verify_remote=False,
        )
        tail_spot._publish_once(
            target / RESULT_OPEN_CLAIM_NAME,
            f"{prefix}/control/{RESULT_OPEN_CLAIM_NAME}",
            project=project,
        )
        return validate_result_open_claim(
            run_dir=target, project=project, bucket=bucket
        )
    active = _active_compute_instances(manifest, project=project)
    if active:
        raise RuntimeError(
            "performance-lock compute is still active: " + ",".join(active)
        )
    source = target / "result_open_done"
    stage = target / ".result_open_done.staging"
    if stage.exists():
        raise FileExistsError("stale performance-lock DONE staging exists")
    plan = _read_packaged_plan(target, manifest)
    if source.exists():
        if source.is_symlink() or not source.is_dir():
            raise ValueError("performance-lock DONE claim material is unsafe")
        done_markers = _claimed_done_marker_records(
            done_dir=source,
            plan=plan,
            prefix=prefix,
        )
    else:
        stage.mkdir()
        try:
            for identifier in authorized_job_ids():
                uri = f"{prefix}/results/jobs/{identifier}/DONE.json"
                path = stage / f"{identifier}.json"
                tail_spot._run(
                    [
                        "gcloud",
                        "storage",
                        "cp",
                        uri,
                        str(path),
                        "--project",
                        project,
                    ],
                    timeout=300,
                )
            done_markers = _claimed_done_marker_records(
                done_dir=stage,
                plan=plan,
                prefix=prefix,
            )
            os.replace(stage, source)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
    resume_claim = target / RESUME_CLAIM_NAME
    resume_result = target / RESUME_RESULT_NAME
    qualification = manifest["tail_qualification"]
    claim = {
        "schema": RESULT_OPEN_CLAIM_SCHEMA,
        "status": "claimed_all_done_before_any_hand_result_content_read",
        "run_name": manifest["run_name"],
        "package_manifest_sha256": sha256_file(target / MANIFEST_NAME),
        "launch_authorization_sha256": sha256_file(target / AUTHORIZATION_NAME),
        "launch_claim_sha256": sha256_file(target / LAUNCH_CLAIM_NAME),
        "launch_result_sha256": sha256_file(target / LAUNCH_RESULT_NAME),
        "resume_claim_sha256": (
            sha256_file(resume_claim) if resume_claim.is_file() else None
        ),
        "resume_result_sha256": (
            sha256_file(resume_result) if resume_result.is_file() else None
        ),
        "precontent_plan_sha256": lock_plan.PRECONTENT_PLAN_SHA256,
        "root_open_claim_sha256": qualification["open_claim_sha256"],
        "root_seal_sha256": qualification["root_seal_sha256"],
        "run_contract_digest": lock_plan.LOCK_RUN_CONTRACT_DIGEST,
        "authorized_job_ids": list(authorized_job_ids()),
        "done_markers": done_markers,
        "done_marker_count": MAX_LOGICAL_JOBS,
        "all_done_markers_present": True,
        "active_compute_instances": [],
        "result_hand_content_addressed_before_claim": False,
        "compute_retry_after_claim_allowed": False,
        "reseed_after_claim_allowed": False,
        "quality_pilot_authorized": False,
        "training_eligible": False,
        "current_profile_changed": False,
        "runtime_policy_activated": False,
        "claimed_unix_seconds": time.time(),
    }
    transport._exact_keys(claim, _RESULT_OPEN_CLAIM_KEYS, "lock result-open claim")
    _write_once(target / RESULT_OPEN_CLAIM_NAME, claim)
    remote = f"{prefix}/control/{RESULT_OPEN_CLAIM_NAME}"
    tail_spot._publish_once(target / RESULT_OPEN_CLAIM_NAME, remote, project=project)
    return validate_result_open_claim(run_dir=target, project=project, bucket=bucket)


def claim_results(
    *,
    run_dir: str | Path,
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    with _lifecycle_mutex(run_dir, "result-open"):
        return _claim_results_unlocked(
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
    target = Path(run_dir).resolve()
    manifest, _launch_claim, _launch_result = validate_launch_chain(target)
    validate_resume_chain(
        run_dir=target,
        manifest=manifest,
        project=project,
        bucket=bucket,
    )
    claim = _read_canonical(target / RESULT_OPEN_CLAIM_NAME, "lock result-open claim")
    transport._exact_keys(claim, _RESULT_OPEN_CLAIM_KEYS, "lock result-open claim")
    done_dir = target / "result_open_done"
    if not done_dir.is_dir():
        raise ValueError("performance-lock claimed DONE directory is missing")
    markers = claim.get("done_markers")
    if not isinstance(markers, list) or len(markers) != MAX_LOGICAL_JOBS:
        raise ValueError("performance-lock claimed DONE records changed")
    plan = _read_packaged_plan(target, manifest)
    prefix = transport._run_prefix(manifest, project=project, bucket=bucket)
    expected_markers = _claimed_done_marker_records(
        done_dir=done_dir,
        plan=plan,
        prefix=prefix,
    )
    resume_claim = target / RESUME_CLAIM_NAME
    resume_result = target / RESUME_RESULT_NAME
    qualification = manifest["tail_qualification"]
    if (
        claim.get("schema") != RESULT_OPEN_CLAIM_SCHEMA
        or claim.get("status") != "claimed_all_done_before_any_hand_result_content_read"
        or claim.get("run_name") != manifest["run_name"]
        or claim.get("package_manifest_sha256") != sha256_file(target / MANIFEST_NAME)
        or claim.get("launch_authorization_sha256")
        != sha256_file(target / AUTHORIZATION_NAME)
        or claim.get("launch_claim_sha256") != sha256_file(target / LAUNCH_CLAIM_NAME)
        or claim.get("launch_result_sha256") != sha256_file(target / LAUNCH_RESULT_NAME)
        or claim.get("resume_claim_sha256")
        != (sha256_file(resume_claim) if resume_claim.is_file() else None)
        or claim.get("resume_result_sha256")
        != (sha256_file(resume_result) if resume_result.is_file() else None)
        or claim.get("precontent_plan_sha256") != lock_plan.PRECONTENT_PLAN_SHA256
        or claim.get("root_open_claim_sha256") != qualification["open_claim_sha256"]
        or claim.get("root_seal_sha256") != qualification["root_seal_sha256"]
        or claim.get("run_contract_digest") != lock_plan.LOCK_RUN_CONTRACT_DIGEST
        or claim.get("authorized_job_ids") != list(authorized_job_ids())
        or markers != expected_markers
        or claim.get("done_marker_count") != MAX_LOGICAL_JOBS
        or claim.get("all_done_markers_present") is not True
        or claim.get("active_compute_instances") != []
        or claim.get("result_hand_content_addressed_before_claim") is not False
        or claim.get("compute_retry_after_claim_allowed") is not False
        or claim.get("reseed_after_claim_allowed") is not False
        or any(
            claim.get(field) is not False
            for field in (
                "quality_pilot_authorized",
                "training_eligible",
                "current_profile_changed",
                "runtime_policy_activated",
            )
        )
        or not transport._valid_timestamp(claim.get("claimed_unix_seconds"))
    ):
        raise ValueError("performance-lock result-open claim changed")
    if verify_remote:
        remote_uri = f"{prefix}/control/{RESULT_OPEN_CLAIM_NAME}"
        with tempfile.TemporaryDirectory(prefix="lock-result-claim-") as temporary:
            remote = Path(temporary) / RESULT_OPEN_CLAIM_NAME
            tail_spot._run(
                [
                    "gcloud",
                    "storage",
                    "cp",
                    remote_uri,
                    str(remote),
                    "--project",
                    project,
                ],
                timeout=300,
            )
            if remote.read_bytes() != (target / RESULT_OPEN_CLAIM_NAME).read_bytes():
                raise ValueError("local and remote result-open claims differ")
    return claim


def _expected_received_relatives(
    source_role: str, work_hand_indices: Sequence[int]
) -> tuple[str, ...]:
    values = ["DONE.json", "run_contract.json", "shard_manifest.json"]
    values.extend(f"roots/hand_{index:03d}.json" for index in work_hand_indices)
    values.extend(
        f"hands/{source_role}/hand_{index:03d}.json" for index in work_hand_indices
    )
    return tuple(sorted(values))


def _validate_received_job(
    *,
    job_dir: Path,
    package_dir: Path,
    record: Mapping[str, Any],
    plan: Mapping[str, Any],
    seal: Mapping[str, Any],
) -> dict[str, Any]:
    identifier = str(record["job_id"])
    plan_job = _plan_job_by_id(plan, identifier)
    packaged_manifest = runner.validate_shard_manifest(
        _read_canonical(
            package_dir / str(record["path"]),
            f"packaged lock job {identifier}",
        )
    )
    role = packaged_manifest["source_role"]
    work = packaged_manifest["work_hand_indices"]
    actual = tuple(
        sorted(
            path.relative_to(job_dir).as_posix()
            for path in job_dir.rglob("*")
            if path.is_file()
        )
    )
    if actual != _expected_received_relatives(role, work):
        raise ValueError(f"performance-lock received file set changed: {identifier}")
    if (
        role != plan_job["source_role"]
        or work != plan_job["work_hand_indices"]
        or canonical_sha256(packaged_manifest) != plan_job["shard_manifest_sha256"]
        or _read_canonical(job_dir / "run_contract.json", "received lock contract")
        != plan["run_contract"]
        or _read_canonical(
            job_dir / "shard_manifest.json", "received lock shard manifest"
        )
        != packaged_manifest
    ):
        raise ValueError(
            f"performance-lock received manifest chain changed: {identifier}"
        )
    done = runner.validate_completed_output(job_dir)
    if (
        done.get("schema")
        != getattr(
            lock_plan,
            "DONE_SCHEMA",
            runner.CANDIDATE02_PERFORMANCE_LOCK_DONE_SCHEMA,
        )
        or done.get("run_contract_digest") != lock_plan.LOCK_RUN_CONTRACT_DIGEST
        or done.get("source_role") != role
        or done.get("work_hand_indices") != work
        or done.get("completed_hand_indices") != work
        or done.get("artifact_count") != 2 * HANDS_PER_SHARD
        or any(
            done.get(field) is not False
            for field in (
                "training_eligible",
                "quality_evidence",
                "promotion_evidence",
                "current_profile_changed",
            )
        )
    ):
        raise ValueError(f"performance-lock received DONE changed: {identifier}")
    sealed_hashes = seal["root_artifact_sha256"]
    roots: list[dict[str, Any]] = []
    hands: list[dict[str, Any]] = []
    for index in work:
        root_path = job_dir / "roots" / f"hand_{index:03d}.json"
        hand_path = job_dir / "hands" / role / f"hand_{index:03d}.json"
        root_value = _read_canonical(root_path, f"received lock root {index}")
        if (
            canonical_sha256(root_value) != sealed_hashes[index]
            or sha256_file(root_path) != sealed_hashes[index]
        ):
            raise ValueError(
                f"performance-lock received root differs from seal: {index}"
            )
        roots.append(
            {
                "hand_index": index,
                "path": (f"jobs/{identifier}/roots/hand_{index:03d}.json"),
                "sha256": sha256_file(root_path),
            }
        )
        hands.append(
            {
                "hand_index": index,
                "path": (f"jobs/{identifier}/hands/{role}/hand_{index:03d}.json"),
                "sha256": sha256_file(hand_path),
            }
        )
    return {
        "job_id": identifier,
        "source_role": role,
        "shard_index": record["shard_index"],
        "work_hand_indices": work,
        "done_path": f"jobs/{identifier}/DONE.json",
        "done_sha256": sha256_file(job_dir / "DONE.json"),
        "root_artifacts": roots,
        "source_hand_artifacts": hands,
        "run_contract_digest": lock_plan.LOCK_RUN_CONTRACT_DIGEST,
    }


def _validate_received_done_against_claim(
    *,
    jobs: Sequence[Mapping[str, Any]],
    result_claim: Mapping[str, Any],
) -> None:
    markers = result_claim.get("done_markers")
    if not isinstance(markers, list):
        raise ValueError("performance-lock result-open DONE records changed")
    by_id = {
        str(marker.get("job_id")): marker
        for marker in markers
        if isinstance(marker, Mapping)
    }
    if set(by_id) != set(authorized_job_ids()):
        raise ValueError("performance-lock result-open DONE identities changed")
    for job in jobs:
        identifier = str(job["job_id"])
        marker = by_id[identifier]
        if (
            job.get("done_sha256") != marker.get("file_sha256")
            or job.get("source_role") != marker.get("source_role")
            or job.get("work_hand_indices") != marker.get("work_hand_indices")
        ):
            raise ValueError(
                f"performance-lock received DONE differs from claim: {identifier}"
            )


def _validate_receive_file_set(
    *, receive_dir: Path, jobs: Sequence[Mapping[str, Any]]
) -> None:
    expected = {
        "receive_receipt.json",
        RESULT_OPEN_CLAIM_NAME,
        MANIFEST_NAME,
        AUTHORIZATION_NAME,
    }
    for job in jobs:
        identifier = str(job["job_id"])
        expected.update(
            f"jobs/{identifier}/{relative}"
            for relative in _expected_received_relatives(
                str(job["source_role"]),
                job["work_hand_indices"],
            )
        )
    actual = {
        path.relative_to(receive_dir).as_posix()
        for path in receive_dir.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise ValueError("performance-lock receive file set changed")


def receive_jobs(
    *,
    run_dir: str | Path,
    destination: str | Path,
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    source = Path(run_dir).resolve()
    target = Path(destination).resolve()
    manifest = validate_package(source)
    validate_launch_chain(source)
    validate_resume_chain(
        run_dir=source,
        manifest=manifest,
        project=project,
        bucket=bucket,
    )
    result_claim = validate_result_open_claim(
        run_dir=source, project=project, bucket=bucket
    )
    if target.exists():
        raise FileExistsError("performance-lock receive destination is immutable")
    stage = target.with_name(f".{target.name}.{os.getpid()}.staging")
    if stage.exists():
        raise FileExistsError("stale performance-lock receive staging exists")
    stage.mkdir(parents=True)
    plan, _root_claim, seal, _materialization = _validate_archive(
        source / SOURCE_NAME, manifest["source_entries"]
    )
    prefix = transport._run_prefix(manifest, project=project, bucket=bucket)
    jobs: list[dict[str, Any]] = []
    try:
        for record in manifest["job_manifests"]:
            identifier = record["job_id"]
            job_dir = stage / "jobs" / identifier
            job_dir.mkdir(parents=True)
            tail_spot._run(
                [
                    "gcloud",
                    "storage",
                    "rsync",
                    "--recursive",
                    f"{prefix}/results/jobs/{identifier}",
                    str(job_dir),
                    "--project",
                    project,
                ],
                timeout=7200,
            )
            jobs.append(
                _validate_received_job(
                    job_dir=job_dir,
                    package_dir=source,
                    record=record,
                    plan=plan,
                    seal=seal,
                )
            )
        if [row["job_id"] for row in jobs] != list(authorized_job_ids()):
            raise ValueError("performance-lock received job order changed")
        _validate_received_done_against_claim(
            jobs=jobs,
            result_claim=result_claim,
        )
        by_role_hand: dict[tuple[str, int], str] = {}
        for row in jobs:
            for root_record in row["root_artifacts"]:
                key = (row["source_role"], root_record["hand_index"])
                if key in by_role_hand:
                    raise ValueError("performance-lock received root duplicated")
                by_role_hand[key] = root_record["sha256"]
        for index, expected in enumerate(seal["root_artifact_sha256"]):
            if (
                by_role_hand.get(("candidate", index)) != expected
                or by_role_hand.get(("reference", index)) != expected
            ):
                raise ValueError(
                    f"performance-lock candidate/reference root mismatch: {index}"
                )
        resume_claim = source / RESUME_CLAIM_NAME
        resume_result = source / RESUME_RESULT_NAME
        candidate_done = [
            row["done_path"] for row in jobs if row["source_role"] == "candidate"
        ]
        reference_done = [
            row["done_path"] for row in jobs if row["source_role"] == "reference"
        ]
        receipt = {
            "schema": RECEIVE_SCHEMA,
            "status": "exact_performance_lock_source_shards_received_and_validated",
            "run_name": manifest["run_name"],
            "package_manifest_sha256": sha256_file(source / MANIFEST_NAME),
            "launch_authorization_sha256": sha256_file(source / AUTHORIZATION_NAME),
            "launch_claim_sha256": sha256_file(source / LAUNCH_CLAIM_NAME),
            "launch_result_sha256": sha256_file(source / LAUNCH_RESULT_NAME),
            "resume_claim_sha256": (
                sha256_file(resume_claim) if resume_claim.is_file() else None
            ),
            "resume_result_sha256": (
                sha256_file(resume_result) if resume_result.is_file() else None
            ),
            "result_open_claim_sha256": sha256_file(source / RESULT_OPEN_CLAIM_NAME),
            "precontent_plan_sha256": lock_plan.PRECONTENT_PLAN_SHA256,
            "root_open_claim_sha256": manifest["tail_qualification"][
                "open_claim_sha256"
            ],
            "root_seal_sha256": manifest["tail_qualification"]["root_seal_sha256"],
            "run_contract_digest": lock_plan.LOCK_RUN_CONTRACT_DIGEST,
            "source_roles": list(SOURCE_ROLES),
            "logical_job_count": MAX_LOGICAL_JOBS,
            "paired_hand_count": 100,
            "root_count": 200,
            "jobs": jobs,
            "candidate_done_paths": candidate_done,
            "reference_done_paths": reference_done,
            "source_isolation_validated": True,
            "root_pairing_validated": True,
            "sealed_root_set_validated": True,
            "merge_executed": False,
            "performance_lock_finalized": False,
            "quality_pilot_authorized": False,
            "training_eligible": False,
            "current_profile_changed": False,
            "named_profile_added": False,
            "runtime_policy_activated": False,
        }
        transport._exact_keys(receipt, _RECEIVE_KEYS, "lock receive receipt")
        _write_once(stage / "receive_receipt.json", receipt)
        shutil.copy2(source / RESULT_OPEN_CLAIM_NAME, stage / RESULT_OPEN_CLAIM_NAME)
        shutil.copy2(source / MANIFEST_NAME, stage / MANIFEST_NAME)
        shutil.copy2(source / AUTHORIZATION_NAME, stage / AUTHORIZATION_NAME)
        _validate_receive_file_set(receive_dir=stage, jobs=jobs)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return validate_received_directory(
        receive_dir=target,
        run_dir=source,
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
    target = Path(receive_dir).resolve()
    source = Path(run_dir).resolve()
    manifest = validate_package(source)
    validate_resume_chain(
        run_dir=source,
        manifest=manifest,
        project=project,
        bucket=bucket,
    )
    result_claim = validate_result_open_claim(
        run_dir=source, project=project, bucket=bucket
    )
    receipt = _read_canonical(target / "receive_receipt.json", "lock receive receipt")
    transport._exact_keys(receipt, _RECEIVE_KEYS, "lock receive receipt")
    plan, _root_claim, seal, _materialization = _validate_archive(
        source / SOURCE_NAME, manifest["source_entries"]
    )
    expected_jobs = [
        _validate_received_job(
            job_dir=target / "jobs" / record["job_id"],
            package_dir=source,
            record=record,
            plan=plan,
            seal=seal,
        )
        for record in manifest["job_manifests"]
    ]
    _validate_received_done_against_claim(
        jobs=expected_jobs,
        result_claim=result_claim,
    )
    _validate_receive_file_set(receive_dir=target, jobs=expected_jobs)
    resume_claim = source / RESUME_CLAIM_NAME
    resume_result = source / RESUME_RESULT_NAME
    if (
        receipt.get("schema") != RECEIVE_SCHEMA
        or receipt.get("status")
        != "exact_performance_lock_source_shards_received_and_validated"
        or receipt.get("run_name") != manifest["run_name"]
        or receipt.get("package_manifest_sha256") != sha256_file(source / MANIFEST_NAME)
        or receipt.get("launch_authorization_sha256")
        != sha256_file(source / AUTHORIZATION_NAME)
        or receipt.get("launch_claim_sha256") != sha256_file(source / LAUNCH_CLAIM_NAME)
        or receipt.get("launch_result_sha256")
        != sha256_file(source / LAUNCH_RESULT_NAME)
        or receipt.get("resume_claim_sha256")
        != (sha256_file(resume_claim) if resume_claim.is_file() else None)
        or receipt.get("resume_result_sha256")
        != (sha256_file(resume_result) if resume_result.is_file() else None)
        or receipt.get("result_open_claim_sha256")
        != sha256_file(source / RESULT_OPEN_CLAIM_NAME)
        or receipt.get("precontent_plan_sha256") != lock_plan.PRECONTENT_PLAN_SHA256
        or receipt.get("root_open_claim_sha256")
        != manifest["tail_qualification"]["open_claim_sha256"]
        or receipt.get("root_seal_sha256")
        != manifest["tail_qualification"]["root_seal_sha256"]
        or receipt.get("run_contract_digest") != lock_plan.LOCK_RUN_CONTRACT_DIGEST
        or receipt.get("source_roles") != list(SOURCE_ROLES)
        or receipt.get("logical_job_count") != MAX_LOGICAL_JOBS
        or receipt.get("paired_hand_count") != 100
        or receipt.get("root_count") != 200
        or receipt.get("jobs") != expected_jobs
        or receipt.get("candidate_done_paths")
        != [
            row["done_path"]
            for row in expected_jobs
            if row["source_role"] == "candidate"
        ]
        or receipt.get("reference_done_paths")
        != [
            row["done_path"]
            for row in expected_jobs
            if row["source_role"] == "reference"
        ]
        or any(
            receipt.get(field) is not True
            for field in (
                "source_isolation_validated",
                "root_pairing_validated",
                "sealed_root_set_validated",
            )
        )
        or any(
            receipt.get(field) is not False
            for field in (
                "merge_executed",
                "performance_lock_finalized",
                "quality_pilot_authorized",
                "training_eligible",
                "current_profile_changed",
                "named_profile_added",
                "runtime_policy_activated",
            )
        )
    ):
        raise ValueError("performance-lock receive receipt changed")
    return receipt


def _parser() -> argparse.ArgumentParser:
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
    validate = commands.add_parser("validate-package")
    validate.add_argument("--run-dir", type=Path, required=True)
    authorize = commands.add_parser("authorize")
    authorize.add_argument("--run-dir", type=Path, required=True)
    launch = commands.add_parser("launch")
    cloud_arguments(launch)
    launch_chain = commands.add_parser("validate-launch-chain")
    launch_chain.add_argument("--run-dir", type=Path, required=True)
    resume = commands.add_parser("resume")
    cloud_arguments(resume)
    resume.add_argument("--job", action="append", required=True)
    status = commands.add_parser("status")
    cloud_arguments(status)
    claim = commands.add_parser("claim-results")
    cloud_arguments(claim)
    validate_claim = commands.add_parser("validate-result-claim")
    cloud_arguments(validate_claim)
    receive = commands.add_parser("receive")
    cloud_arguments(receive)
    receive.add_argument("--destination", type=Path, required=True)
    validate_receive = commands.add_parser("validate-received")
    cloud_arguments(validate_receive)
    validate_receive.add_argument("--receive-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "package":
        result: Any = package_performance_lock(
            output_dir=args.output_dir,
            run_name=args.run_name,
            lock_inputs=lock_open.PerformanceLockInputs(
                repository_root=args.repository_root,
                plan_path=args.plan,
                lock_output_directory=args.lock_output,
                candidate_library=args.candidate_library,
                reference_library=args.reference_library,
                feature_encoder=args.feature_encoder,
                startup_source=args.startup_source,
                development_summary_path=args.development_summary,
                development_validation_path=args.development_validation,
                development_root_directory=args.development_roots,
                global_claim_path=args.global_claim,
            ),
        )
    elif args.command == "validate-package":
        result: Any = validate_package(args.run_dir)
    elif args.command == "authorize":
        result = authorize_launch(args.run_dir)
    elif args.command == "launch":
        result = launch_jobs(
            run_dir=args.run_dir,
            project=args.project,
            bucket=args.bucket,
        )
    elif args.command == "validate-launch-chain":
        result = validate_launch_chain(args.run_dir)
    elif args.command == "resume":
        result = resume_jobs(
            run_dir=args.run_dir,
            selected=args.job,
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
    "AUTHORIZATION_SCHEMA",
    "LAUNCH_CLAIM_SCHEMA",
    "LAUNCH_RESULT_SCHEMA",
    "PACKAGE_SCHEMA",
    "RECEIVE_SCHEMA",
    "RESULT_OPEN_CLAIM_SCHEMA",
    "RESUME_CLAIM_SCHEMA",
    "RESUME_RESULT_SCHEMA",
    "authorized_job_ids",
    "authorize_launch",
    "claim_results",
    "cloud_status",
    "launch_jobs",
    "package_performance_lock",
    "preflight_resume",
    "receive_jobs",
    "resume_jobs",
    "validate_launch_authorization",
    "validate_launch_chain",
    "validate_package",
    "validate_received_directory",
    "validate_result_open_claim",
    "validate_resume_chain",
]
