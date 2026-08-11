"""Immutable Spot-package helpers for the Attempt06 search-quality audit.

This module deliberately keeps package construction on the *closed* side of
the fresh-data boundary.  It validates and copies code, models, native
libraries, the frozen seed schedule, and the candidate ranker, but it never
turns an audit hand seed into cards or an :class:`ActorObservation`.

The shell startup worker owns the irreversible boundary: it validates this
package, claims the global and per-root markers, and only then invokes the
separate root-input builder followed by ``hu_m43_attempt06_teacher shard``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .hu_m43_attempt06_contract import (
    AI_PROFILES_SHA256,
    M43_ATTEMPT06_LAMBDA_SHA256,
    M43_ATTEMPT06_PLAN_SHA256,
    M43_ATTEMPT06_PROFILES,
    enumerate_attempt06_seed_schedules,
    load_and_validate_attempt06_plan,
    load_and_validate_attempt06_status,
)
from .hu_m43_attempt06_teacher import (
    ATTEMPT06_BASELINE_PROFILE,
    ATTEMPT06_CANDIDATE_SAMPLES,
    ATTEMPT06_CANDIDATE_SEED_START,
    ATTEMPT06_CHECKPOINT_SCHEMA,
    ATTEMPT06_CHILD_SEED_START,
    ATTEMPT06_EVALUATION_SAMPLES,
    ATTEMPT06_EVALUATION_SEED_START,
    ATTEMPT06_HEARTBEAT_SCHEMA,
    ATTEMPT06_HAND_SEED_START,
    ATTEMPT06_NATIVE_BATCH_THREADS,
    ATTEMPT06_ROOT_REMATERIALIZATION_MODE,
    ATTEMPT06_ROOT_SCHEMA,
    ATTEMPT06_SEED_STRIDE,
    ATTEMPT06_SHARD_ROW_SCHEMA,
    ATTEMPT06_SHARD_SUMMARY_SCHEMA,
    ATTEMPT06_TEACHER_SCHEMA,
    ATTEMPT06_T2_POLICY_ID,
    ATTEMPT06_TOP_K,
    attempt06_fixed_contract_sha256,
    load_attempt06_roots,
)


PACKAGE_MANIFEST_SCHEMA = "hu_m43_attempt06_spot_package_manifest_v1"
SOURCE_CLOSURE_SCHEMA = "hu_m43_attempt06_spot_source_closure_v1"
SHARD_SCHEMA = "hu_m43_attempt06_spot_shard_v1"
PACKAGE_RESULT_SCHEMA = "hu_m43_attempt06_spot_package_result_v1"
DONE_SCHEMA = "hu_m43_attempt06_spot_done_v1"
ROOT_BUILD_RESULT_SCHEMA = "hu_m43_attempt06_spot_root_build_result_v1"
ROOT_PROVENANCE_SCHEMA = "hu_m43_attempt06_spot_root_provenance_v1"
RECEIVE_AUDIT_SCHEMA = "hu_m43_attempt06_spot_received_shard_audit_v1"
RECEIVE_MERGE_SCHEMA = "hu_m43_attempt06_spot_receive_merge_v1"
AUDIT_OUTPUT_CONSUMPTION_SCHEMA = "hu_m43_attempt06_audit_output_consumption_v1"
ROOT_REMATERIALIZATION_MODE = ATTEMPT06_ROOT_REMATERIALIZATION_MODE

PINNED_TEMPLATE_RUN = (
    "regular-hu-m43-attempt03-c2e64-pilot700-r2-20260714-0228"
)
PINNED_MODEL_MANIFEST_SHA256 = (
    "e9b9d9748bb2b2f31a4baa0d928b60c05f254ab5d7915a7461fb8402e1e7ddb8"
)
PINNED_NATIVE_MANIFEST_SHA256 = (
    "ec295d070ac7bb21a82aeb2f432b8f6f191e61b63027bc446e8bd5f39f51569f"
)
EXPECTED_MODEL_SCHEMA = "hu_m43_attempt02_source_model_manifest_v1"
EXPECTED_NATIVE_SCHEMA = "hu_m43_attempt02_source_native_manifest_v1"
EXPECTED_MODEL_COUNT = 11
EXPECTED_NATIVE_COUNT = 2
EXPECTED_SHARDS = 50
EXPECTED_ROOTS_PER_SHARD = 1
RUN_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,126}[a-z0-9]$")

REQUIREMENTS = (
    "numpy==2.2.6\n"
    "scikit-learn==1.8.0\n"
    "lightgbm==4.6.0\n"
    "torch==2.6.0\n"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_lower_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable file: {path}")
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"immutable file concurrently created: {path}"
            ) from exc
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(path, canonical_json_bytes(payload))


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a mapping")
    return payload


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 changed: expected={expected} actual={actual}")


def build_attempt06_spot_schedule(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    transfer = plan["audit_schedule_transfer"]
    if (
        transfer.get("shards") != EXPECTED_SHARDS
        or transfer.get("roots_per_shard") != EXPECTED_ROOTS_PER_SHARD
        or transfer.get("profile_assignment")
        != "root_index_mod_5_in_frozen_profile_order"
    ):
        raise ValueError("Attempt06 is not frozen to fifty one-root Spot shards")
    schedules = enumerate_attempt06_seed_schedules(plan)
    rows: list[dict[str, Any]] = []
    for root_index in range(EXPECTED_SHARDS):
        profile = M43_ATTEMPT06_PROFILES[root_index % len(M43_ATTEMPT06_PROFILES)]
        rows.append(
            {
                "schema": SHARD_SCHEMA,
                "shard": root_index,
                "root_index": root_index,
                "roots": 1,
                "root_profile": profile,
                "hand_seed": schedules["hand"][root_index],
                "candidate_seed": schedules["candidate"][root_index],
                "evaluation_seed": schedules["evaluation"][root_index],
                "child_policy_seed": schedules["child"][root_index],
                "learned_nonbaseline_top_k": 8,
                "candidate_samples": 8,
                "evaluation_samples": 128,
                "native_batch_threads": ATTEMPT06_NATIVE_BATCH_THREADS,
                "baseline_profile": "stage18_p1",
                "t2_profile": "stage9f_p2",
                "hypothetical_t4_mode": "counter_mc_1",
                "output_prefix": f"shard_{root_index:03d}",
            }
        )
    if len({row["hand_seed"] for row in rows}) != EXPECTED_SHARDS:
        raise AssertionError("Attempt06 Spot schedule duplicated hand seeds")
    counts = {profile: 0 for profile in M43_ATTEMPT06_PROFILES}
    for row in rows:
        counts[str(row["root_profile"])] += 1
    if set(counts.values()) != {10}:
        raise AssertionError("Attempt06 Spot schedule lost balanced profiles")
    return rows


def _schedule_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def _verify_pinned_runtime_closure(template_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    model_manifest_path = template_root / "source_model_manifest.json"
    native_manifest_path = template_root / "source_native_manifest.json"
    _require_hash(
        model_manifest_path,
        PINNED_MODEL_MANIFEST_SHA256,
        "pinned source-model manifest",
    )
    _require_hash(
        native_manifest_path,
        PINNED_NATIVE_MANIFEST_SHA256,
        "pinned source-native manifest",
    )
    model_manifest = _load_mapping(model_manifest_path, "source-model manifest")
    native_manifest = _load_mapping(native_manifest_path, "source-native manifest")
    if (
        model_manifest.get("schema") != EXPECTED_MODEL_SCHEMA
        or model_manifest.get("model_count") != EXPECTED_MODEL_COUNT
        or len(model_manifest.get("models", ())) != EXPECTED_MODEL_COUNT
    ):
        raise ValueError("pinned source-model manifest semantics changed")
    if (
        native_manifest.get("schema") != EXPECTED_NATIVE_SCHEMA
        or native_manifest.get("binary_count") != EXPECTED_NATIVE_COUNT
        or len(native_manifest.get("binaries", ())) != EXPECTED_NATIVE_COUNT
    ):
        raise ValueError("pinned source-native manifest semantics changed")
    for label, rows in (
        ("model", model_manifest["models"]),
        ("native binary", native_manifest["binaries"]),
    ):
        for row in rows:
            relative = Path(str(row["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"pinned {label} path is unsafe")
            source = template_root / relative
            if source.stat().st_size != int(row["bytes"]):
                raise ValueError(f"pinned {label} size changed: {relative.as_posix()}")
            _require_hash(source, str(row["sha256"]), f"pinned {label}")
    return model_manifest, native_manifest


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copyfile(source, destination)


def _copy_source_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )


def _copy_manifest_files(
    template_root: Path,
    package_root: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    for row in rows:
        relative = Path(str(row["path"]))
        _copy_file(template_root / relative, package_root / relative)


def _closure_rows(package_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(
        (item for item in package_root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(package_root).as_posix(),
    ):
        relative = path.relative_to(package_root).as_posix()
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _write_deterministic_zip(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in sorted(
            (item for item in source.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(source).as_posix(),
        ):
            relative = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2026, 7, 14, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with path.open("rb") as handle:
                archive.writestr(info, handle.read(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)


def _validate_existing_package(run_dir: Path, *, run_name: str) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    manifest = _load_mapping(manifest_path, "Attempt06 package manifest")
    if (
        manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA
        or manifest.get("status") != "frozen_package_only_no_fresh_content"
        or manifest.get("run_name") != run_name
        or manifest.get("plan_sha256") != M43_ATTEMPT06_PLAN_SHA256
        or manifest.get("model_sha256") != M43_ATTEMPT06_LAMBDA_SHA256
        or manifest.get("total_shards") != EXPECTED_SHARDS
        or manifest.get("roots_per_shard") != EXPECTED_ROOTS_PER_SHARD
        or manifest.get("native_batch_threads") != ATTEMPT06_NATIVE_BATCH_THREADS
        or manifest.get("fresh_seed_content_opened") is not False
        or manifest.get("teacher_executed") is not False
    ):
        raise ValueError("existing Attempt06 package manifest changed")
    bindings = {
        "shards_manifest.jsonl": "schedule_sha256",
        "source_closure_manifest.json": "source_closure_sha256",
        "ofc_regular_hu_m43_attempt06_teacher_source.zip": "source_zip_sha256",
        "startup_hu_m43_attempt06_teacher.sh": "startup_sha256",
        "hu_joint_policy_m43_attempt06.json": "plan_sha256",
        "hu_joint_policy_m43_attempt06_status.json": "status_sha256",
    }
    for name, field in bindings.items():
        _require_hash(run_dir / name, str(manifest[field]), f"existing {name}")
    return manifest


def package_attempt06_spot(
    *,
    repo_root: str | Path,
    run_dir: str | Path,
    run_name: str,
    plan_path: str | Path,
    status_path: str | Path,
    model_path: str | Path,
    startup_path: str | Path,
    resume_existing: bool = False,
) -> dict[str, Any]:
    """Build the immutable package without opening any frozen audit hand."""

    if RUN_NAME_RE.fullmatch(run_name) is None:
        raise ValueError("RunName is not a safe GCP identity")
    root = Path(repo_root).resolve()
    destination = Path(run_dir).resolve()
    expected_destination = (root / "outputs" / "gcp_runs" / run_name).resolve()
    if os.path.normcase(str(destination)) != os.path.normcase(
        str(expected_destination)
    ):
        raise ValueError(
            "Attempt06 run directory must be exactly "
            "repo/outputs/gcp_runs/<run_name>"
        )
    if destination.exists():
        if not resume_existing:
            raise FileExistsError(f"Attempt06 run directory already exists: {destination}")
        manifest = _validate_existing_package(destination, run_name=run_name)
        return _package_result(destination, manifest, resumed=True)

    plan_file = Path(plan_path).resolve()
    status_file = Path(status_path).resolve()
    model_file = Path(model_path).resolve()
    startup_file = Path(startup_path).resolve()
    plan = load_and_validate_attempt06_plan(plan_file)
    status = load_and_validate_attempt06_status(status_file)
    _require_hash(plan_file, M43_ATTEMPT06_PLAN_SHA256, "Attempt06 plan")
    _require_hash(model_file, M43_ATTEMPT06_LAMBDA_SHA256, "Attempt06 lambda ranker")
    _require_hash(
        root / "src/ofc_regular/ai_profiles.py",
        AI_PROFILES_SHA256,
        "AI profile registry",
    )
    if status["spot_execution"]["authorized"] is not False:
        raise ValueError("Attempt06 package builder requires Spot to remain unauthorized")
    if status["one_shot_search_quality_audit"]["fresh_content_opened"] is not False:
        raise ValueError("Attempt06 package builder found an opened fresh audit")

    schedule = build_attempt06_spot_schedule(plan)
    template_root = (
        root / "outputs/gcp_runs" / PINNED_TEMPLATE_RUN / "package_src"
    )
    model_manifest, native_manifest = _verify_pinned_runtime_closure(template_root)

    staging = destination.with_name(destination.name + ".building")
    if staging.exists():
        raise FileExistsError(f"stale Attempt06 package staging directory: {staging}")
    staging.mkdir(parents=True)
    try:
        package_root = staging / "package_src"
        package_root.mkdir()
        _copy_source_tree(root / "src/ofc_regular", package_root / "src/ofc_regular")
        _copy_manifest_files(
            template_root, package_root, list(model_manifest["models"])
        )
        _copy_manifest_files(
            template_root, package_root, list(native_manifest["binaries"])
        )
        _copy_file(
            template_root / "source_model_manifest.json",
            package_root / "source_model_manifest.json",
        )
        _copy_file(
            template_root / "source_native_manifest.json",
            package_root / "source_native_manifest.json",
        )
        _copy_file(
            model_file, package_root / "artifacts/lambda_rank_candidate.pkl"
        )
        _copy_file(plan_file, package_root / "configs/hu_joint_policy_m43_attempt06.json")
        _copy_file(
            status_file,
            package_root / "configs/hu_joint_policy_m43_attempt06_status.json",
        )
        _atomic_write(package_root / "shards_manifest.jsonl", _schedule_bytes(schedule))
        _atomic_write(package_root / "requirements-attempt06.txt", REQUIREMENTS.encode("ascii"))

        closure = {
            "schema": SOURCE_CLOSURE_SCHEMA,
            "status": "closed_no_fresh_seed_materialized",
            "run_name": run_name,
            "plan_sha256": sha256_file(plan_file),
            "status_sha256": sha256_file(status_file),
            "model_sha256": sha256_file(model_file),
            "ai_profiles_sha256": AI_PROFILES_SHA256,
            "files": _closure_rows(package_root),
            "fresh_seed_content_opened": False,
            "teacher_executed": False,
        }
        _write_json(package_root / "source_closure_manifest.json", closure)

        schedule_path = staging / "shards_manifest.jsonl"
        startup_copy = staging / "startup_hu_m43_attempt06_teacher.sh"
        closure_copy = staging / "source_closure_manifest.json"
        plan_copy = staging / "hu_joint_policy_m43_attempt06.json"
        status_copy = staging / "hu_joint_policy_m43_attempt06_status.json"
        _copy_file(package_root / "shards_manifest.jsonl", schedule_path)
        _copy_file(startup_file, startup_copy)
        _copy_file(package_root / "source_closure_manifest.json", closure_copy)
        _copy_file(plan_file, plan_copy)
        _copy_file(status_file, status_copy)
        source_zip = staging / "ofc_regular_hu_m43_attempt06_teacher_source.zip"
        _write_deterministic_zip(package_root, source_zip)

        manifest = {
            "schema": PACKAGE_MANIFEST_SCHEMA,
            "status": "frozen_package_only_no_fresh_content",
            "run_name": run_name,
            "plan_sha256": sha256_file(plan_file),
            "status_sha256": sha256_file(status_file),
            "model_sha256": sha256_file(model_file),
            "ai_profiles_sha256": AI_PROFILES_SHA256,
            "schedule_sha256": sha256_file(schedule_path),
            "source_closure_sha256": sha256_file(closure_copy),
            "source_zip_sha256": sha256_file(source_zip),
            "startup_sha256": sha256_file(startup_copy),
            "source_model_manifest_sha256": PINNED_MODEL_MANIFEST_SHA256,
            "source_native_manifest_sha256": PINNED_NATIVE_MANIFEST_SHA256,
            "total_roots": EXPECTED_SHARDS,
            "total_shards": EXPECTED_SHARDS,
            "roots_per_shard": EXPECTED_ROOTS_PER_SHARD,
            "root_profile_assignment": "root_index_mod_5_in_frozen_profile_order",
            "candidate_samples": 8,
            "evaluation_samples": 128,
            "native_batch_threads": ATTEMPT06_NATIVE_BATCH_THREADS,
            "learned_nonbaseline_top_k": 8,
            "fresh_seed_content_opened": False,
            "teacher_executed": False,
            "gcloud_invoked": False,
            "instances_created": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        }
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return _package_result(destination, manifest, resumed=False)


def _package_result(
    run_dir: Path, manifest: Mapping[str, Any], *, resumed: bool
) -> dict[str, Any]:
    return {
        "schema": PACKAGE_RESULT_SCHEMA,
        "status": "packaged_without_fresh_content",
        "run_name": manifest["run_name"],
        "run_dir": str(run_dir),
        "manifest": str(run_dir / "manifest.json"),
        "manifest_sha256": sha256_file(run_dir / "manifest.json"),
        "source_zip": str(
            run_dir / "ofc_regular_hu_m43_attempt06_teacher_source.zip"
        ),
        "total_shards": EXPECTED_SHARDS,
        "roots_per_shard": EXPECTED_ROOTS_PER_SHARD,
        "resumed_existing_package": resumed,
        "fresh_seed_content_opened": False,
        "teacher_executed": False,
        "gcloud_invoked": False,
        "instances_created": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }


def build_attempt06_root_input(
    *,
    output: str | Path,
    root_index: int,
    hand_seed: int,
    root_profile: str,
    plan_sha256: str,
    schedule_sha256: str,
    model_sha256: str,
    manifest_sha256: str,
    source_sha256: str,
    startup_sha256: str,
    status_sha256: str,
    source_closure_sha256: str,
    global_marker: str | Path,
    root_claim: str | Path,
    run_name: str,
    run_id: str,
) -> dict[str, Any]:
    """Open exactly one frozen root after the shell has claimed its marker.

    This function must never be called by ``package``.  Its deliberately
    separate CLI subcommand lets the startup worker put the irreversible GCS
    claim immediately before the first card-bearing operation.
    """

    if isinstance(root_index, bool) or not isinstance(root_index, int):
        raise TypeError("Attempt06 root_index must be an integer")
    if not 0 <= root_index < EXPECTED_SHARDS:
        raise ValueError("Attempt06 root_index is outside 0..49")
    expected_hand_seed = ATTEMPT06_HAND_SEED_START + ATTEMPT06_SEED_STRIDE * root_index
    if hand_seed != expected_hand_seed:
        raise ValueError("Attempt06 hand seed disagrees with frozen root_index")
    expected_profile = M43_ATTEMPT06_PROFILES[root_index % len(M43_ATTEMPT06_PROFILES)]
    if root_profile != expected_profile:
        raise ValueError("Attempt06 root profile disagrees with root_index mod 5")
    for value, expected, label in (
        (plan_sha256, M43_ATTEMPT06_PLAN_SHA256, "plan"),
        (model_sha256, M43_ATTEMPT06_LAMBDA_SHA256, "model"),
    ):
        if value != expected:
            raise ValueError(f"Attempt06 {label} hash changed")
    if len(schedule_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in schedule_sha256
    ):
        raise ValueError("Attempt06 schedule hash must be lowercase SHA-256")
    for value, label in (
        (manifest_sha256, "manifest"),
        (source_sha256, "source"),
        (startup_sha256, "startup"),
        (status_sha256, "status"),
        (source_closure_sha256, "source closure"),
    ):
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"Attempt06 {label} hash must be lowercase SHA-256")
    if not run_name or run_id != f"{run_name}:shard={root_index}":
        raise ValueError("Attempt06 run_name/run_id binding changed")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"Attempt06 root input already exists: {destination}")

    # Claim proof is validated before importing or invoking any policy/deal
    # helper.  The caller provides the exact bytes it atomically published to
    # GCS; missing or mismatched proof therefore fails before the fresh seed is
    # turned into cards.
    global_payload = _load_mapping(Path(global_marker), "Attempt06 global marker")
    claim_payload = _load_mapping(Path(root_claim), "Attempt06 root claim")
    expected_common = {
        "run_name": run_name,
        "manifest_sha256": manifest_sha256,
        "schedule_sha256": schedule_sha256,
        "plan_sha256": plan_sha256,
        "model_sha256": model_sha256,
        "source_sha256": source_sha256,
        "startup_sha256": startup_sha256,
        "status_sha256": status_sha256,
        "source_closure_sha256": source_closure_sha256,
        "source_model_manifest_sha256": PINNED_MODEL_MANIFEST_SHA256,
        "source_native_manifest_sha256": PINNED_NATIVE_MANIFEST_SHA256,
        "native_batch_threads": ATTEMPT06_NATIVE_BATCH_THREADS,
    }
    if (
        global_payload.get("schema")
        != "hu_m43_attempt06_global_consumption_marker_v1"
        or global_payload.get("status")
        != "consumed_before_any_fresh_root_content_read"
        or global_payload.get("roots") != EXPECTED_SHARDS
        or global_payload.get("shards") != EXPECTED_SHARDS
        or global_payload.get("roots_per_shard") != EXPECTED_ROOTS_PER_SHARD
        or any(global_payload.get(key) != value for key, value in expected_common.items())
    ):
        raise ValueError("Attempt06 global marker proof changed")
    if (
        claim_payload.get("schema") != "hu_m43_attempt06_root_consumption_claim_v1"
        or claim_payload.get("status")
        != "claimed_before_materializing_policy_observation"
        or claim_payload.get("run_id") != run_id
        or claim_payload.get("root_index") != root_index
        or claim_payload.get("hand_seed") != hand_seed
        or claim_payload.get("root_profile") != root_profile
        or claim_payload.get("retry_same_seed_after_open_allowed") is not False
        or claim_payload.get("fresh_audit_retry_or_alternate_sample_allowed")
        is not False
        or claim_payload.get("deterministic_claim_recovery_allowed") is not True
        or claim_payload.get("deterministic_claim_recovery_mode")
        != ROOT_REMATERIALIZATION_MODE
        or any(claim_payload.get(key) != value for key, value in expected_common.items())
    ):
        raise ValueError("Attempt06 per-root claim proof changed")

    from .action_key import action_key
    from .ai_profiles import ModelPaths, build_policy, load_model_bundle
    from .generate_hu_m4_t1_data import (
        ROOT_GENERATION_POLICY,
        _profile_policy_seed,
        generate_t1_second_root,
    )
    from .play_ai import _choose_from_observation, _hand_decision_seed

    # The seed base is the global audit start, matching a monolithic 50-root
    # generator.  Splitting into one-root Spot shards therefore cannot perturb
    # any prefix-policy decision.
    explicit_profiles = {root_profile, "stage18_p1"}
    bundle = load_model_bundle(ModelPaths(), profiles=explicit_profiles)
    root_policies = {
        seat: build_policy(
            root_profile,
            bundle,
            seed=_profile_policy_seed(
                ATTEMPT06_HAND_SEED_START, root_profile, seat
            ),
            seat=seat,
            opening_lookahead_samples=0,
        )
        for seat in ("first", "second")
    }
    baseline_policy = build_policy(
        "stage18_p1",
        bundle,
        seed=ATTEMPT06_HAND_SEED_START + 101,
        seat="second",
        opening_lookahead_samples=0,
    )
    observation = generate_t1_second_root(
        hand_seed, root_policies=root_policies
    )
    baseline_action = _choose_from_observation(
        baseline_policy,
        observation,
        hand_id=hand_seed,
        game_id=hand_seed,
        decision_seed=_hand_decision_seed(
            base_seed=hand_seed, observation=observation
        ),
    )
    payload = {
        "schema": ATTEMPT06_ROOT_SCHEMA,
        "root_index": root_index,
        "hand_seed": hand_seed,
        "root_profile": root_profile,
        "policy_observation": observation.to_dict(),
        "baseline_action_key": action_key(baseline_action).to_token(),
        "provenance": {
            "schema": ROOT_PROVENANCE_SCHEMA,
            "run_name": run_name,
            "run_id": run_id,
            "root_index": root_index,
            "root_profile": root_profile,
            "root_generation_policy": ROOT_GENERATION_POLICY,
            "root_policy_seed_base": ATTEMPT06_HAND_SEED_START,
            "baseline_profile": "stage18_p1",
            "plan_sha256": plan_sha256,
            "schedule_sha256": schedule_sha256,
            "candidate_model_sha256": model_sha256,
            "source_model_manifest_sha256": PINNED_MODEL_MANIFEST_SHA256,
            "source_native_manifest_sha256": PINNED_NATIVE_MANIFEST_SHA256,
            "source_sha256": source_sha256,
            "startup_sha256": startup_sha256,
            "status_sha256": status_sha256,
            "source_closure_sha256": source_closure_sha256,
            "native_batch_threads": ATTEMPT06_NATIVE_BATCH_THREADS,
            "package_manifest_sha256": manifest_sha256,
            "global_consumption_marker_sha256": sha256_file(global_marker),
            "root_consumption_claim_sha256": sha256_file(root_claim),
            "profile_assignment": "root_index_mod_5_in_frozen_profile_order",
            "current_profile_resolved": False,
            "opponent_private_discard_input_allowed": False,
            "teacher_value_status": "diagnostic_not_match_EV",
            "fresh_audit_retry_or_alternate_sample_allowed": False,
            "deterministic_claim_recovery_allowed": True,
            "deterministic_claim_recovery_mode": ROOT_REMATERIALIZATION_MODE,
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("xb") as handle:
            encoded = canonical_json_bytes(payload)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        loaded = load_attempt06_roots(temporary)
        if loaded[0].root_index != root_index:
            raise AssertionError("Attempt06 root input validation changed identity")
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Attempt06 root input concurrently materialized: {destination}"
            ) from exc
        temporary.unlink()
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return {
        "schema": ROOT_BUILD_RESULT_SCHEMA,
        "status": "fresh_root_materialized_after_external_claim",
        "root_index": root_index,
        "root_profile": root_profile,
        "input_sha256": sha256_file(destination),
        "observation_fingerprint": observation.fingerprint(),
        "current_profile_resolved": False,
        "opponent_private_discard_input_allowed": False,
    }


def _load_consumption_marker(path: str | Path) -> dict[str, Any]:
    marker = _load_mapping(Path(path), "Attempt06 audit-output consumption marker")
    if (
        marker.get("schema") != AUDIT_OUTPUT_CONSUMPTION_SCHEMA
        or marker.get("status")
        != "consumed_before_any_result_teacher_or_root_read"
        or marker.get("current_profile_mutated") is not False
        or marker.get("runtime_policy_activated") is not False
    ):
        raise ValueError("Attempt06 audit-output consumption marker changed")
    return marker


def _read_one_json_line(path: Path, label: str) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"{label} must contain exactly one JSON row")
    payload = json.loads(lines[0])
    if not isinstance(payload, dict):
        raise ValueError(f"{label} row must be a mapping")
    return payload


def _load_schedule(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(rows) != EXPECTED_SHARDS:
        raise ValueError("Attempt06 receive schedule must contain fifty rows")
    for index, row in enumerate(rows):
        expected = {
            "schema": SHARD_SCHEMA,
            "shard": index,
            "root_index": index,
            "roots": EXPECTED_ROOTS_PER_SHARD,
            "root_profile": M43_ATTEMPT06_PROFILES[
                index % len(M43_ATTEMPT06_PROFILES)
            ],
            "hand_seed": ATTEMPT06_HAND_SEED_START
            + ATTEMPT06_SEED_STRIDE * index,
            "candidate_seed": ATTEMPT06_CANDIDATE_SEED_START
            + ATTEMPT06_SEED_STRIDE * index,
            "evaluation_seed": ATTEMPT06_EVALUATION_SEED_START
            + ATTEMPT06_SEED_STRIDE * index,
            "child_policy_seed": ATTEMPT06_CHILD_SEED_START
            + ATTEMPT06_SEED_STRIDE * index,
            "learned_nonbaseline_top_k": ATTEMPT06_TOP_K,
            "candidate_samples": ATTEMPT06_CANDIDATE_SAMPLES,
            "evaluation_samples": ATTEMPT06_EVALUATION_SAMPLES,
            "native_batch_threads": ATTEMPT06_NATIVE_BATCH_THREADS,
            "baseline_profile": ATTEMPT06_BASELINE_PROFILE,
            "t2_profile": ATTEMPT06_T2_POLICY_ID,
            "hypothetical_t4_mode": "counter_mc_1",
            "output_prefix": f"shard_{index:03d}",
        }
        if (
            not isinstance(row, dict)
            or any(row.get(key) != value for key, value in expected.items())
        ):
            raise ValueError("Attempt06 receive schedule identity changed")
    return rows


def validate_received_attempt06_shard(
    *,
    shard_dir: str | Path,
    schedule_path: str | Path,
    shard: int,
    consumption_marker: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Validate one downloaded shard, only after the local one-shot claim."""

    # This read is intentionally first.  No root/teacher path is even formed
    # until the irreversible local marker has been validated.
    marker = _load_consumption_marker(consumption_marker)
    schedule_file = Path(schedule_path)
    schedule = _load_schedule(schedule_file)
    schedule_sha256 = sha256_file(schedule_file)
    marker_run_name = marker.get("run_name")
    if (
        not isinstance(marker_run_name, str)
        or RUN_NAME_RE.fullmatch(marker_run_name) is None
        or marker.get("schedule_sha256") != schedule_sha256
        or marker.get("expected_shards") != EXPECTED_SHARDS
        or marker.get("expected_roots") != EXPECTED_SHARDS
        or marker.get("result_objects_addressed_when_claimed") is not False
        or marker.get("fit_performed") is not False
        or marker.get("threshold_selected") is not False
        or not all(
            _is_lower_sha256(marker.get(key))
            for key in (
                "authorization_file_sha256",
                "manifest_sha256",
                "schedule_sha256",
                "global_consumption_marker_sha256",
            )
        )
    ):
        raise ValueError("Attempt06 local audit-output marker identity changed")
    if not 0 <= shard < EXPECTED_SHARDS:
        raise ValueError("Attempt06 received shard is outside 0..49")
    spec = schedule[shard]
    directory = Path(shard_dir)
    done = _load_mapping(directory / "DONE.json", "Attempt06 DONE")
    fixed_done = {
        "schema": DONE_SCHEMA,
        "status": "complete",
        "run_name": marker_run_name,
        "run_id": f"{marker_run_name}:shard={shard}",
        "shard": shard,
        "root_index": shard,
        "hand_seed": spec["hand_seed"],
        "root_profile": spec["root_profile"],
        "roots": 1,
        "output_prefix": spec["output_prefix"],
        "manifest_sha256": marker["manifest_sha256"],
        "schedule_sha256": schedule_sha256,
        "plan_sha256": M43_ATTEMPT06_PLAN_SHA256,
        "model_sha256": M43_ATTEMPT06_LAMBDA_SHA256,
        "source_model_manifest_sha256": PINNED_MODEL_MANIFEST_SHA256,
        "source_native_manifest_sha256": PINNED_NATIVE_MANIFEST_SHA256,
        "native_batch_threads": ATTEMPT06_NATIVE_BATCH_THREADS,
    }
    if any(done.get(key) != value for key, value in fixed_done.items()):
        raise ValueError("Attempt06 received DONE identity changed")
    if (
        done.get("teacher_values_are_realized_match_ev") is not False
        or done.get("current_profile_mutated") is not False
        or done.get("runtime_policy_activated") is not False
    ):
        raise ValueError("Attempt06 received DONE overstates activation or EV")
    for key in (
        "source_sha256",
        "startup_sha256",
        "status_sha256",
        "source_closure_sha256",
        "config_sha256",
    ):
        if not _is_lower_sha256(done.get(key)):
            raise ValueError(f"Attempt06 received DONE lacks valid {key}")
    file_bindings = {
        "root.jsonl": "input_sha256",
        "teacher.jsonl": "output_sha256",
        "checkpoint.json": "checkpoint_sha256",
        "heartbeat.json": "heartbeat_sha256",
        "generator_summary.json": "generator_summary_sha256",
        "run.log": "run_log_sha256",
        "global_consumption_marker.json": "global_consumption_marker_sha256",
        "root_claim.json": "root_consumption_claim_sha256",
    }
    for name, field in file_bindings.items():
        expected = str(done.get(field, ""))
        if not _is_lower_sha256(expected):
            raise ValueError(f"Attempt06 DONE lacks {field}")
        _require_hash(directory / name, expected, f"received {name}")

    global_marker = _load_mapping(
        directory / "global_consumption_marker.json", "Attempt06 global marker"
    )
    root_claim = _load_mapping(directory / "root_claim.json", "Attempt06 root claim")
    run_id = f"{marker_run_name}:shard={shard}"
    closure = {
        "run_name": marker_run_name,
        "manifest_sha256": marker["manifest_sha256"],
        "source_sha256": done["source_sha256"],
        "startup_sha256": done["startup_sha256"],
        "schedule_sha256": schedule_sha256,
        "plan_sha256": M43_ATTEMPT06_PLAN_SHA256,
        "status_sha256": done["status_sha256"],
        "model_sha256": M43_ATTEMPT06_LAMBDA_SHA256,
        "source_closure_sha256": done["source_closure_sha256"],
        "source_model_manifest_sha256": PINNED_MODEL_MANIFEST_SHA256,
        "source_native_manifest_sha256": PINNED_NATIVE_MANIFEST_SHA256,
        "native_batch_threads": ATTEMPT06_NATIVE_BATCH_THREADS,
    }
    if (
        marker.get("global_consumption_marker_sha256")
        != done["global_consumption_marker_sha256"]
        or global_marker.get("schema")
        != "hu_m43_attempt06_global_consumption_marker_v1"
        or global_marker.get("status")
        != "consumed_before_any_fresh_root_content_read"
        or global_marker.get("roots") != EXPECTED_SHARDS
        or global_marker.get("shards") != EXPECTED_SHARDS
        or global_marker.get("roots_per_shard") != EXPECTED_ROOTS_PER_SHARD
        or any(global_marker.get(key) != value for key, value in closure.items())
    ):
        raise ValueError("Attempt06 received global marker closure changed")
    if (
        root_claim.get("schema") != "hu_m43_attempt06_root_consumption_claim_v1"
        or root_claim.get("status")
        != "claimed_before_materializing_policy_observation"
        or root_claim.get("run_id") != run_id
        or root_claim.get("root_index") != shard
        or root_claim.get("hand_seed") != spec["hand_seed"]
        or root_claim.get("root_profile") != spec["root_profile"]
        or root_claim.get("retry_same_seed_after_open_allowed") is not False
        or root_claim.get("fresh_audit_retry_or_alternate_sample_allowed")
        is not False
        or root_claim.get("deterministic_claim_recovery_allowed") is not True
        or root_claim.get("deterministic_claim_recovery_mode")
        != ROOT_REMATERIALIZATION_MODE
        or any(root_claim.get(key) != value for key, value in closure.items())
    ):
        raise ValueError("Attempt06 received root claim closure changed")

    root = load_attempt06_roots(directory / "root.jsonl")[0]
    root_provenance_expected = {
        "run_name": marker_run_name,
        "run_id": run_id,
        "root_index": shard,
        "root_profile": spec["root_profile"],
        "plan_sha256": M43_ATTEMPT06_PLAN_SHA256,
        "schedule_sha256": schedule_sha256,
        "candidate_model_sha256": M43_ATTEMPT06_LAMBDA_SHA256,
        "package_manifest_sha256": marker["manifest_sha256"],
        "global_consumption_marker_sha256": done[
            "global_consumption_marker_sha256"
        ],
        "root_consumption_claim_sha256": done[
            "root_consumption_claim_sha256"
        ],
        "source_sha256": done["source_sha256"],
        "startup_sha256": done["startup_sha256"],
        "status_sha256": done["status_sha256"],
        "source_closure_sha256": done["source_closure_sha256"],
        "native_batch_threads": ATTEMPT06_NATIVE_BATCH_THREADS,
        "fresh_audit_retry_or_alternate_sample_allowed": False,
        "deterministic_claim_recovery_allowed": True,
        "deterministic_claim_recovery_mode": ROOT_REMATERIALIZATION_MODE,
    }
    if any(
        root.provenance.get(key) != value
        for key, value in root_provenance_expected.items()
    ):
        raise ValueError("Attempt06 received root provenance closure changed")
    expected_config_sha256 = attempt06_fixed_contract_sha256(
        root_index=shard,
        input_sha256=done["input_sha256"],
        model_sha256=M43_ATTEMPT06_LAMBDA_SHA256,
        source_model_manifest_sha256=PINNED_MODEL_MANIFEST_SHA256,
        source_native_manifest_sha256=PINNED_NATIVE_MANIFEST_SHA256,
        run_id=run_id,
        batch_child_selectors=True,
        native_batch_threads=ATTEMPT06_NATIVE_BATCH_THREADS,
    )
    if done.get("config_sha256") != expected_config_sha256:
        raise ValueError("Attempt06 received DONE config SHA-256 changed")
    teacher = _read_one_json_line(directory / "teacher.jsonl", "Attempt06 teacher")
    if (
        teacher.get("schema") != ATTEMPT06_SHARD_ROW_SCHEMA
        or teacher.get("root_index") != shard
        or teacher.get("hand_seed") != spec["hand_seed"]
        or teacher.get("root_profile") != spec["root_profile"]
        or teacher.get("baseline_action_key") != root.baseline_action_key
        or teacher.get("policy_observation") != root.observation.to_dict()
    ):
        raise ValueError("Attempt06 received teacher/root identity changed")
    result = teacher.get("teacher")
    if (
        not isinstance(result, Mapping)
        or result.get("schema") != ATTEMPT06_TEACHER_SCHEMA
        or result.get("teacher_value_status") != "diagnostic_not_match_EV"
        or result.get("runtime_gate_allowed") is not False
    ):
        raise ValueError("Attempt06 received teacher science boundary changed")
    provenance = teacher.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("input_sha256") != done["input_sha256"]
        or provenance.get("config_sha256") != expected_config_sha256
        or provenance.get("model_sha256") != M43_ATTEMPT06_LAMBDA_SHA256
        or provenance.get("source_model_manifest_sha256")
        != PINNED_MODEL_MANIFEST_SHA256
        or provenance.get("source_native_manifest_sha256")
        != PINNED_NATIVE_MANIFEST_SHA256
        or provenance.get("current_profile_resolved") is not False
    ):
        raise ValueError("Attempt06 received teacher provenance changed")

    # Import lazily so the package helper remains the lower-level source of
    # receive schemas.  The public validator reconstructs the ActorObservation
    # legal ActionKeys, top8/c8 lock, locked+baseline-only e128 summaries, RNG
    # domains, concrete stage9f_p2 continuation, and hidden-information rules.
    from .audit_hu_m43_attempt06_search_quality import (
        validate_attempt06_teacher_row,
    )

    validate_attempt06_teacher_row(
        teacher,
        root_index=shard,
        expected_run_name=marker_run_name,
        expected_schedule_sha256=schedule_sha256,
        expected_package_manifest_sha256=marker["manifest_sha256"],
        expected_global_consumption_marker_sha256=done[
            "global_consumption_marker_sha256"
        ],
        expected_root_consumption_claim_sha256=done[
            "root_consumption_claim_sha256"
        ],
        expected_input_sha256=done["input_sha256"],
        expected_source_sha256=done["source_sha256"],
        expected_startup_sha256=done["startup_sha256"],
        expected_status_sha256=done["status_sha256"],
        expected_source_closure_sha256=done["source_closure_sha256"],
    )

    checkpoint = _load_mapping(directory / "checkpoint.json", "Attempt06 checkpoint")
    heartbeat = _load_mapping(directory / "heartbeat.json", "Attempt06 heartbeat")
    summary = _read_one_json_line(
        directory / "generator_summary.json", "Attempt06 generator summary"
    )
    if (
        checkpoint.get("schema") != ATTEMPT06_CHECKPOINT_SCHEMA
        or checkpoint.get("config_sha256") != expected_config_sha256
        or checkpoint.get("input_sha256") != done["input_sha256"]
        or checkpoint.get("model_sha256") != M43_ATTEMPT06_LAMBDA_SHA256
        or checkpoint.get("completed_roots") != 1
        or checkpoint.get("target_roots") != 1
        or checkpoint.get("root_index") != shard
        or checkpoint.get("source_model_manifest_sha256")
        != PINNED_MODEL_MANIFEST_SHA256
        or checkpoint.get("source_native_manifest_sha256")
        != PINNED_NATIVE_MANIFEST_SHA256
    ):
        raise ValueError("Attempt06 received checkpoint changed")
    if (
        heartbeat.get("schema") != ATTEMPT06_HEARTBEAT_SCHEMA
        or heartbeat.get("config_sha256") != expected_config_sha256
        or heartbeat.get("input_sha256") != done["input_sha256"]
        or heartbeat.get("model_sha256") != M43_ATTEMPT06_LAMBDA_SHA256
        or heartbeat.get("status") != "complete"
        or heartbeat.get("root_index") != shard
        or heartbeat.get("output_sha256") != done["output_sha256"]
        or heartbeat.get("source_model_manifest_sha256")
        != PINNED_MODEL_MANIFEST_SHA256
        or heartbeat.get("source_native_manifest_sha256")
        != PINNED_NATIVE_MANIFEST_SHA256
    ):
        raise ValueError("Attempt06 received heartbeat changed")
    if (
        summary.get("schema") != ATTEMPT06_SHARD_SUMMARY_SCHEMA
        or summary.get("config_sha256") != expected_config_sha256
        or summary.get("status") != "complete"
        or summary.get("root_index") != shard
        or summary.get("input_sha256") != done["input_sha256"]
        or summary.get("output_sha256") != done["output_sha256"]
        or summary.get("model_sha256") != M43_ATTEMPT06_LAMBDA_SHA256
        or summary.get("source_model_manifest_sha256")
        != PINNED_MODEL_MANIFEST_SHA256
        or summary.get("source_native_manifest_sha256")
        != PINNED_NATIVE_MANIFEST_SHA256
    ):
        raise ValueError("Attempt06 received generator summary changed")
    audit = {
        "schema": RECEIVE_AUDIT_SCHEMA,
        "status": "pass_after_local_consumption_claim",
        "shard": shard,
        "root_index": shard,
        "root_profile": spec["root_profile"],
        "input_sha256": done["input_sha256"],
        "teacher_sha256": done["output_sha256"],
        "model_sha256": M43_ATTEMPT06_LAMBDA_SHA256,
        "config_sha256": expected_config_sha256,
        "consumption_marker_file_sha256": sha256_file(consumption_marker),
        "consumption_marker_run_name": marker.get("run_name"),
        "teacher_values_are_realized_match_ev": False,
        "fit_performed": False,
        "threshold_selected": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    _write_json(Path(output), audit)
    return audit


def merge_received_attempt06_shards(
    *,
    shards_root: str | Path,
    schedule_path: str | Path,
    consumption_marker: str | Path,
    output: str | Path,
    receipt: str | Path,
) -> dict[str, Any]:
    """Merge exactly fifty already-audited rows without fitting or tuning."""

    marker = _load_consumption_marker(consumption_marker)
    schedule = _load_schedule(Path(schedule_path))
    root = Path(shards_root)
    records: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for spec in schedule:
        directory = root / str(spec["output_prefix"])
        audit = _load_mapping(directory / "received_audit.json", "received shard audit")
        teacher_path = directory / "teacher.jsonl"
        if (
            audit.get("schema") != RECEIVE_AUDIT_SCHEMA
            or audit.get("status") != "pass_after_local_consumption_claim"
            or audit.get("root_index") != spec["root_index"]
            or audit.get("teacher_sha256") != sha256_file(teacher_path)
            or audit.get("consumption_marker_file_sha256")
            != sha256_file(consumption_marker)
        ):
            raise ValueError("Attempt06 merge found an unaudited shard")
        record = _read_one_json_line(teacher_path, "Attempt06 teacher")
        if record.get("root_index") != spec["root_index"]:
            raise ValueError("Attempt06 merge root ordering changed")
        audits.append(audit)
        records.append(record)
    records.sort(key=lambda row: int(row["root_index"]))
    if [int(row["root_index"]) for row in records] != list(range(EXPECTED_SHARDS)):
        raise ValueError("Attempt06 merge roots are not exactly 0..49")
    profile_counts = {profile: 0 for profile in M43_ATTEMPT06_PROFILES}
    for row in records:
        profile_counts[str(row["root_profile"])] += 1
    if set(profile_counts.values()) != {10}:
        raise ValueError("Attempt06 merge lost balanced root profiles")
    output_path = Path(output)
    _atomic_write(
        output_path,
        b"".join(canonical_json_bytes(record) for record in records),
    )
    payload = {
        "schema": RECEIVE_MERGE_SCHEMA,
        "status": "merged_fifty_fresh_rows_without_fit_or_threshold_selection",
        "roots": EXPECTED_SHARDS,
        "shards": EXPECTED_SHARDS,
        "roots_per_shard": EXPECTED_ROOTS_PER_SHARD,
        "profile_counts": profile_counts,
        "merged_teacher_sha256": sha256_file(output_path),
        "consumption_marker_file_sha256": sha256_file(consumption_marker),
        "consumption_marker_run_name": marker.get("run_name"),
        "shard_audit_sha256": [
            sha256_file(root / str(spec["output_prefix"]) / "received_audit.json")
            for spec in schedule
        ],
        "teacher_values_are_realized_match_ev": False,
        "fit_performed": False,
        "threshold_selected": False,
        "go_no_go_computed": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    _write_json(Path(receipt), payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    package = subparsers.add_parser("package")
    package.add_argument("--repo-root", required=True)
    package.add_argument("--run-dir", required=True)
    package.add_argument("--run-name", required=True)
    package.add_argument("--plan", required=True)
    package.add_argument("--status", required=True)
    package.add_argument("--model", required=True)
    package.add_argument("--startup", required=True)
    package.add_argument("--resume-existing", action="store_true")
    root = subparsers.add_parser(
        "build-root-input",
        help="materialize one frozen root; caller must first claim the audit marker",
    )
    root.add_argument("--output", required=True)
    root.add_argument("--root-index", type=int, required=True)
    root.add_argument("--hand-seed", type=int, required=True)
    root.add_argument("--root-profile", required=True)
    root.add_argument("--plan-sha256", required=True)
    root.add_argument("--schedule-sha256", required=True)
    root.add_argument("--model-sha256", required=True)
    root.add_argument("--manifest-sha256", required=True)
    root.add_argument("--source-sha256", required=True)
    root.add_argument("--startup-sha256", required=True)
    root.add_argument("--status-sha256", required=True)
    root.add_argument("--source-closure-sha256", required=True)
    root.add_argument("--global-marker", required=True)
    root.add_argument("--root-claim", required=True)
    root.add_argument("--run-name", required=True)
    root.add_argument("--run-id", required=True)
    validate = subparsers.add_parser("validate-received-shard")
    validate.add_argument("--shard-dir", required=True)
    validate.add_argument("--schedule", required=True)
    validate.add_argument("--shard", type=int, required=True)
    validate.add_argument("--consumption-marker", required=True)
    validate.add_argument("--output", required=True)
    merge = subparsers.add_parser("merge-received")
    merge.add_argument("--shards-root", required=True)
    merge.add_argument("--schedule", required=True)
    merge.add_argument("--consumption-marker", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "package":
        result = package_attempt06_spot(
            repo_root=args.repo_root,
            run_dir=args.run_dir,
            run_name=args.run_name,
            plan_path=args.plan,
            status_path=args.status,
            model_path=args.model,
            startup_path=args.startup,
            resume_existing=args.resume_existing,
        )
    elif args.command == "build-root-input":
        result = build_attempt06_root_input(
            output=args.output,
            root_index=args.root_index,
            hand_seed=args.hand_seed,
            root_profile=args.root_profile,
            plan_sha256=args.plan_sha256,
            schedule_sha256=args.schedule_sha256,
            model_sha256=args.model_sha256,
            manifest_sha256=args.manifest_sha256,
            source_sha256=args.source_sha256,
            startup_sha256=args.startup_sha256,
            status_sha256=args.status_sha256,
            source_closure_sha256=args.source_closure_sha256,
            global_marker=args.global_marker,
            root_claim=args.root_claim,
            run_name=args.run_name,
            run_id=args.run_id,
        )
    elif args.command == "validate-received-shard":
        result = validate_received_attempt06_shard(
            shard_dir=args.shard_dir,
            schedule_path=args.schedule,
            shard=args.shard,
            consumption_marker=args.consumption_marker,
            output=args.output,
        )
    elif args.command == "merge-received":
        result = merge_received_attempt06_shards(
            shards_root=args.shards_root,
            schedule_path=args.schedule,
            consumption_marker=args.consumption_marker,
            output=args.output,
            receipt=args.receipt,
        )
    else:  # pragma: no cover - argparse owns the command domain.
        raise AssertionError(args.command)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DONE_SCHEMA",
    "EXPECTED_ROOTS_PER_SHARD",
    "EXPECTED_SHARDS",
    "PACKAGE_MANIFEST_SCHEMA",
    "PACKAGE_RESULT_SCHEMA",
    "ROOT_BUILD_RESULT_SCHEMA",
    "ROOT_PROVENANCE_SCHEMA",
    "SHARD_SCHEMA",
    "SOURCE_CLOSURE_SCHEMA",
    "build_attempt06_spot_schedule",
    "build_attempt06_root_input",
    "canonical_json_bytes",
    "main",
    "package_attempt06_spot",
    "merge_received_attempt06_shards",
    "sha256_file",
    "validate_received_attempt06_shard",
]
