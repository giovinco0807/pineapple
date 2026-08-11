"""Fail-closed closeout for the consumed M3.1 performance-lock startup failure.

This module is deliberately a pure, local evidence validator.  It does not
query or mutate cloud state and it never opens packaged hand roots or result
content.  The source archive is only hashed as an opaque file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_performance_lock_startup_failure_closeout_v1"
)
RECEIPT_STATUS = (
    "attempt0_consumed_by_deterministic_precontent_path_case_failure_"
    "current_run_irrecoverable"
)
METADATA_SNAPSHOT_SCHEMA = (
    "hu_m31_t3_step6d_performance_lock_failure_metadata_snapshot_v1"
)
METADATA_SNAPSHOT_STATUS = "external_read_only_control_plane_snapshot_complete"

RUN_NAME = "regular-hu-m31-c02-performance-lock-20260717-001"
PROJECT = "ofc-solver-485418"
BUCKET = "pokerhu-ofc-solver-485418-training"
REMOTE_PREFIX = f"gs://{BUCKET}/runs/{RUN_NAME}/full100"

SOURCE_NAME = "ofc_regular_hu_m31_t3_step6d_full100_v1_source.zip"
STARTUP_NAME = "startup_hu_m31_t3_step6d_full100_v1.sh"
MANIFEST_NAME = "manifest.json"
READY_NAME = "PACKAGE_READY.json"
AUTHORIZATION_NAME = "launch_authorization.json"
LAUNCH_CLAIM_NAME = "launch_claim.json"
LAUNCH_RESULT_NAME = "launch_result.json"
GLOBAL_ROOT_CLAIM_NAME = "GLOBAL_PERFORMANCE_LOCK_CLAIM.json"
GLOBAL_SPOT_CLAIM_NAME = "GLOBAL_PERFORMANCE_LOCK_SPOT_CLAIM.json"

PLAN_SHA256 = "d2c9985b574839cf7e1515c12ebffc813ac6b4fb4e6494ed8d1288ef582a4b6b"
RUN_CONTRACT_DIGEST = (
    "e73c2b06279c1f1e91c38b2887ee465acf85f8f1072afef636512283252c34a4"
)
CURRENT_PROFILE_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)

# These anchors identify the already-claimed, already-launched run.  A
# replacement package cannot satisfy this closeout contract.
EXPECTED_HASHES: dict[str, str] = {
    MANIFEST_NAME: "aed60bb67c1178236c09cc6c0d7d42b28c1214d758e907104425f84224ea6344",
    READY_NAME: "8c5121c540e1d66f88744dd85d2bab8e829cafa95be545931cb1dcfb24579dda",
    AUTHORIZATION_NAME: "9402500a407c289cc8a997c15d4b2a4c432fd8859f0e3d289f443de8a91d69b4",
    LAUNCH_CLAIM_NAME: "5730805e531eb217b1c218558e4f2134dcda0db2eda91f51fa7db52f434c7bf9",
    LAUNCH_RESULT_NAME: "f34a5177d2e4fbff8b106d51bc8fa9fc38b481355498a92ed64c5f232b64d77f",
    SOURCE_NAME: "b245a2d55cb0adfbd5c40506a64b38ce17a426fc9e8ddd3510402bbebcf4aca5",
    STARTUP_NAME: "d953baff034998211a203fa4ee8328a33274931fa2ef0062651bca1451c0604c",
    GLOBAL_ROOT_CLAIM_NAME: (
        "8b5db48a71d39eb8b85cc9fab43cf4ee43a9350c7b1fa49b3d004816ec397f31"
    ),
    GLOBAL_SPOT_CLAIM_NAME: (
        "e65dbf8d03837573566d6133c0ff8f7daf1541e09b8f797f537257bb1edf7b57"
    ),
}

JOB_IDS = tuple(
    f"{role}-shard-{index:02d}"
    for role in ("candidate", "reference")
    for index in range(10)
)
COUNT_KEYS = ("live", "done", "heartbeat", "result", "resume", "attempt1")
FORBIDDEN_LOCAL_CONTROLS = (
    "resume_claim.json",
    "resume_result.json",
    "result_open_claim.json",
)

ERROR_LINE = "performance-lock global root/Spot claim chain changed"
FIRST_FAILURE_PREFIX = (
    "full100 startup failure: exit=1 line=746 command=python3 - "
    "/tmp/source.zip /tmp/manifest.json /tmp/authorization.json <<'PY'"
)
SECOND_FAILURE_PREFIX = (
    'full100 startup failure: exit=1 line=746 command=PACKAGE_PHASE="$(python3 '
    "- /tmp/source.zip /tmp/manifest.json /tmp/authorization.json <<'PY'"
)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")


def _read_canonical(path: Path, label: str) -> dict[str, Any]:
    _regular_file(path, label)
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _require_hash(path: Path, anchor_name: str, label: str) -> str:
    _regular_file(path, label)
    actual = sha256_file(path)
    if actual != EXPECTED_HASHES[anchor_name]:
        raise ValueError(f"{label} hash is not the claimed run anchor")
    return actual


def _safe_package_member(package_dir: Path, relative: str, label: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"{label} path is unsafe")
    path = package_dir.joinpath(*pure.parts)
    try:
        path.resolve(strict=False).relative_to(package_dir)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the package") from exc
    return path


def _validate_package_controls(
    package_dir: Path,
    global_root_claim_path: Path,
    global_spot_claim_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    package_dir = package_dir.resolve()
    if not package_dir.is_dir() or package_dir.is_symlink():
        raise ValueError("package directory must be a non-symlink directory")

    paths = {
        name: package_dir / name
        for name in (
            MANIFEST_NAME,
            READY_NAME,
            AUTHORIZATION_NAME,
            LAUNCH_CLAIM_NAME,
            LAUNCH_RESULT_NAME,
            SOURCE_NAME,
            STARTUP_NAME,
        )
    }
    for name, path in paths.items():
        _require_hash(path, name, f"package {name}")
    _require_hash(
        global_root_claim_path, GLOBAL_ROOT_CLAIM_NAME, "global root claim"
    )
    _require_hash(
        global_spot_claim_path, GLOBAL_SPOT_CLAIM_NAME, "global Spot claim"
    )
    for name in FORBIDDEN_LOCAL_CONTROLS:
        if (package_dir / name).exists():
            raise ValueError(f"forbidden post-attempt0 local control exists: {name}")

    manifest = _read_canonical(paths[MANIFEST_NAME], "package manifest")
    ready = _read_canonical(paths[READY_NAME], "package ready")
    authorization = _read_canonical(
        paths[AUTHORIZATION_NAME], "launch authorization"
    )
    launch_claim = _read_canonical(paths[LAUNCH_CLAIM_NAME], "launch claim")
    launch_result = _read_canonical(paths[LAUNCH_RESULT_NAME], "launch result")
    root_claim = _read_canonical(global_root_claim_path, "global root claim")
    spot_claim = _read_canonical(global_spot_claim_path, "global Spot claim")

    source = paths[SOURCE_NAME]
    startup = paths[STARTUP_NAME]
    manifest_sha = EXPECTED_HASHES[MANIFEST_NAME]
    authorization_sha = EXPECTED_HASHES[AUTHORIZATION_NAME]
    launch_claim_sha = EXPECTED_HASHES[LAUNCH_CLAIM_NAME]
    false_manifest_fields = (
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
        manifest.get("schema")
        != "hu_m31_t3_step6d_performance_lock_spot_package_v1"
        or manifest.get("status")
        != "immutable_performance_lock_package_ready_not_authorized"
        or manifest.get("run_name") != RUN_NAME
        or manifest.get("source_name") != SOURCE_NAME
        or manifest.get("source_sha256") != EXPECTED_HASHES[SOURCE_NAME]
        or manifest.get("source_bytes") != source.stat().st_size
        or manifest.get("startup_name") != STARTUP_NAME
        or manifest.get("startup_sha256") != EXPECTED_HASHES[STARTUP_NAME]
        or manifest.get("plan_sha256") != PLAN_SHA256
        or manifest.get("run_contract_digest") != RUN_CONTRACT_DIGEST
        or any(manifest.get(field) is not False for field in false_manifest_fields)
    ):
        raise ValueError("package manifest control boundary changed")
    if (
        ready.get("schema")
        != "hu_m31_t3_step6d_performance_lock_package_ready_v1"
        or ready.get("status")
        != "immutable_local_performance_lock_package_complete"
        or ready.get("run_name") != RUN_NAME
        or ready.get("package_manifest_sha256") != manifest_sha
        or ready.get("source_sha256") != EXPECTED_HASHES[SOURCE_NAME]
        or ready.get("startup_sha256") != EXPECTED_HASHES[STARTUP_NAME]
        or ready.get("plan_sha256") != PLAN_SHA256
        or ready.get("run_contract_digest") != RUN_CONTRACT_DIGEST
        or ready.get("logical_job_count") != 20
        or ready.get("gcloud_invoked") is not False
        or ready.get("spot_vm_started") is not False
        or ready.get("current_profile_changed") is not False
    ):
        raise ValueError("package ready control boundary changed")

    job_records = manifest.get("job_manifests")
    if not isinstance(job_records, list) or len(job_records) != len(JOB_IDS):
        raise ValueError("package job control set changed")
    observed_job_ids: list[str] = []
    for expected_id, record in zip(JOB_IDS, job_records, strict=True):
        if not isinstance(record, Mapping) or record.get("job_id") != expected_id:
            raise ValueError("package job control order changed")
        relative = record.get("path")
        if not isinstance(relative, str):
            raise ValueError("package job control path changed")
        job_path = _safe_package_member(package_dir, relative, "package job control")
        value = _read_canonical(job_path, f"package job control {expected_id}")
        if (
            value.get("schema")
            != "hu_m31_t3_step6d_performance_shard_manifest_v2"
            or value.get("run_contract_digest") != RUN_CONTRACT_DIGEST
            or value.get("source_role") != record.get("source_role")
            or value.get("work_hand_indices") != record.get("work_hand_indices")
            or record.get("sha256") != sha256_file(job_path)
            or record.get("bytes") != job_path.stat().st_size
        ):
            raise ValueError(f"package job control binding changed: {expected_id}")
        observed_job_ids.append(expected_id)
    if observed_job_ids != list(JOB_IDS):
        raise ValueError("package job controls are incomplete")

    if (
        spot_claim.get("schema")
        != "hu_m31_t3_step6d_performance_lock_global_spot_claim_v1"
        or spot_claim.get("status")
        != "global_one_shot_spot_identity_claimed_before_authorization"
        or spot_claim.get("run_name") != RUN_NAME
        or spot_claim.get("global_root_claim_sha256")
        != EXPECTED_HASHES[GLOBAL_ROOT_CLAIM_NAME]
        or spot_claim.get("package_manifest_sha256") != manifest_sha
        or spot_claim.get("source_sha256") != EXPECTED_HASHES[SOURCE_NAME]
        or spot_claim.get("startup_sha256") != EXPECTED_HASHES[STARTUP_NAME]
        or spot_claim.get("precontent_plan_sha256") != PLAN_SHA256
        or spot_claim.get("run_contract_digest") != RUN_CONTRACT_DIGEST
        or spot_claim.get("authorized_job_ids") != list(JOB_IDS)
        or spot_claim.get("max_initial_jobs") != 20
        or spot_claim.get("max_resume_attempts") != 1
        or any(
            spot_claim.get(field) is not False
            for field in (
                "alternate_package_authorization_allowed",
                "quality_pilot_authorized",
                "training_eligible",
                "current_profile_changed",
                "runtime_policy_activated",
            )
        )
    ):
        raise ValueError("global Spot claim control boundary changed")
    if authorization.get("global_spot_claim") != spot_claim:
        raise ValueError("authorization/global Spot claim binding changed")
    if (
        authorization.get("schema")
        != "hu_m31_t3_step6d_performance_lock_launch_authorization_v1"
        or authorization.get("status")
        != "explicit_one_shot_performance_lock_spot_authorization"
        or authorization.get("run_name") != RUN_NAME
        or authorization.get("package_manifest_sha256") != manifest_sha
        or authorization.get("source_sha256") != EXPECTED_HASHES[SOURCE_NAME]
        or authorization.get("startup_sha256") != EXPECTED_HASHES[STARTUP_NAME]
        or authorization.get("plan_sha256") != PLAN_SHA256
        or authorization.get("run_contract_digest") != RUN_CONTRACT_DIGEST
        or authorization.get("authorized_job_ids") != list(JOB_IDS)
        or authorization.get("logical_job_count") != 20
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
    ):
        raise ValueError("launch authorization control boundary changed")
    if (
        launch_claim.get("schema")
        != "hu_m31_t3_step6d_performance_lock_launch_claim_v1"
        or launch_claim.get("status")
        != "exclusive_performance_lock_launch_claim_before_remote_mutation"
        or launch_claim.get("run_name") != RUN_NAME
        or launch_claim.get("selected_job_ids") != list(JOB_IDS)
        or launch_claim.get("package_manifest_sha256") != manifest_sha
        or launch_claim.get("launch_authorization_sha256") != authorization_sha
        or launch_claim.get("plan_sha256") != PLAN_SHA256
        or launch_claim.get("run_contract_digest") != RUN_CONTRACT_DIGEST
        or launch_claim.get("crash_reuse_authorized") is not False
    ):
        raise ValueError("launch claim control boundary changed")

    created = launch_result.get("created")
    if not isinstance(created, list) or len(created) != 20:
        raise ValueError("launch result created set changed")
    for ordinal, (job_id, row) in enumerate(zip(JOB_IDS, created, strict=True)):
        if (
            not isinstance(row, Mapping)
            or row.get("job_id") != job_id
            or row.get("attempt_index") != 0
            or row.get("instance") != f"{RUN_NAME}-j{ordinal:02d}"
            or row.get("status") != "created"
        ):
            raise ValueError(f"launch result job changed: {job_id}")
    if (
        launch_result.get("schema")
        != "hu_m31_t3_step6d_performance_lock_launch_result_v1"
        or launch_result.get("status") != "performance_lock_jobs_created"
        or launch_result.get("run_name") != RUN_NAME
        or launch_result.get("selected_job_ids") != list(JOB_IDS)
        or launch_result.get("failures") != []
        or launch_result.get("cleanup") != []
        or launch_result.get("logical_job_count") != 20
        or launch_result.get("launch_claim_sha256") != launch_claim_sha
        or launch_result.get("performance_lock_authorized") is not True
        or launch_result.get("quality_pilot_authorized") is not False
        or any(
            launch_result.get(field) is not False
            for field in (
                "training_eligible",
                "current_profile_changed",
                "runtime_policy_activated",
            )
        )
    ):
        raise ValueError("launch result control boundary changed")

    root_global_path = root_claim.get("global_claim_path")
    spot_global_path = spot_claim.get("global_root_claim_path")
    root_output_path = root_claim.get("lock_output_directory")
    spot_output_path = spot_claim.get("lock_output_directory")
    for left, right, label in (
        (root_global_path, spot_global_path, "global claim path"),
        (root_output_path, spot_output_path, "lock output path"),
    ):
        if (
            not isinstance(left, str)
            or not isinstance(right, str)
            or left == right
            or left.casefold() != right.casefold()
        ):
            raise ValueError(f"expected deterministic path-case mismatch is absent: {label}")
    if canonical_sha256(root_claim) != EXPECTED_HASHES[GLOBAL_ROOT_CLAIM_NAME]:
        raise ValueError("global root claim canonical hash changed")

    startup_text = startup.read_text(encoding="utf-8")
    for needle in (
        ERROR_LINE,
        'value.get("global_root_claim_path") == root_claim.get("global_claim_path")',
        'value.get("lock_output_directory")',
        '== root_claim.get("lock_output_directory")',
    ):
        if needle not in startup_text:
            raise ValueError("claimed startup path comparison contract changed")

    path_mismatch = {
        "global_claim_path": {
            "root_claim": root_global_path,
            "spot_claim": spot_global_path,
            "exact_equal": False,
            "casefold_equal": True,
        },
        "lock_output_directory": {
            "root_claim": root_output_path,
            "spot_claim": spot_output_path,
            "exact_equal": False,
            "casefold_equal": True,
        },
    }
    anchors = {
        "manifest_sha256": manifest_sha,
        "package_ready_sha256": EXPECTED_HASHES[READY_NAME],
        "launch_authorization_sha256": authorization_sha,
        "launch_claim_sha256": launch_claim_sha,
        "launch_result_sha256": EXPECTED_HASHES[LAUNCH_RESULT_NAME],
        "source_sha256": EXPECTED_HASHES[SOURCE_NAME],
        "startup_sha256": EXPECTED_HASHES[STARTUP_NAME],
        "global_root_claim_sha256": EXPECTED_HASHES[GLOBAL_ROOT_CLAIM_NAME],
        "global_spot_claim_sha256": EXPECTED_HASHES[GLOBAL_SPOT_CLAIM_NAME],
        "plan_sha256": PLAN_SHA256,
        "run_contract_digest": RUN_CONTRACT_DIGEST,
        "current_profile_sha256": CURRENT_PROFILE_SHA256,
    }
    return anchors, path_mismatch


def _validate_counts(counts: Mapping[str, Any]) -> dict[str, int]:
    if set(counts) != set(COUNT_KEYS):
        raise ValueError("control-plane counts have wrong fields")
    normalized: dict[str, int] = {}
    for key in COUNT_KEYS:
        value = counts[key]
        if not isinstance(value, int) or isinstance(value, bool) or value != 0:
            raise ValueError(f"control-plane {key} count must be exactly zero")
        normalized[key] = value
    return normalized


def _load_count_evidence(
    *,
    metadata_snapshot_path: Path | None,
    direct_counts: Mapping[str, Any] | None,
) -> tuple[dict[str, int], dict[str, Any]]:
    if (metadata_snapshot_path is None) == (direct_counts is None):
        raise ValueError("supply exactly one of metadata snapshot or direct counts")
    if direct_counts is not None:
        counts = _validate_counts(direct_counts)
        evidence = {
            "mode": "direct_supplied_counts_no_cloud_query",
            "sha256": canonical_sha256(counts),
        }
        return counts, evidence

    assert metadata_snapshot_path is not None
    snapshot = _read_canonical(metadata_snapshot_path, "metadata snapshot")
    snapshot_sha = sha256_file(metadata_snapshot_path)
    if set(snapshot) == set(COUNT_KEYS):
        counts = _validate_counts(snapshot)
        return counts, {
            "mode": "supplied_counts_json_no_cloud_query",
            "sha256": snapshot_sha,
        }
    expected_keys = {
        "schema",
        "status",
        "run_name",
        "counts",
        "metadata_only",
        "hand_content_opened",
        "root_content_opened",
        "result_content_opened",
        "cloud_mutated",
    }
    if (
        set(snapshot) != expected_keys
        or snapshot.get("schema") != METADATA_SNAPSHOT_SCHEMA
        or snapshot.get("status") != METADATA_SNAPSHOT_STATUS
        or snapshot.get("run_name") != RUN_NAME
        or snapshot.get("metadata_only") is not True
        or snapshot.get("hand_content_opened") is not False
        or snapshot.get("root_content_opened") is not False
        or snapshot.get("result_content_opened") is not False
        or snapshot.get("cloud_mutated") is not False
        or not isinstance(snapshot.get("counts"), Mapping)
    ):
        raise ValueError("metadata snapshot boundary changed")
    return _validate_counts(snapshot["counts"]), {
        "mode": "supplied_metadata_snapshot_no_cloud_query",
        "sha256": snapshot_sha,
    }


def _validate_startup_logs(startup_log_tree: Path) -> dict[str, Any]:
    root = startup_log_tree.resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("startup log tree must be a non-symlink directory")
    expected_relatives = {
        f"jobs/{job_id}/attempt-0/startup.log" for job_id in JOB_IDS
    }
    actual_relatives: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("startup log tree contains a symlink")
        if path.is_file():
            actual_relatives.add(path.relative_to(root).as_posix())
    if actual_relatives != expected_relatives:
        raise ValueError("startup log tree is not the exact 20/20 attempt0 set")

    records: list[dict[str, Any]] = []
    log_hashes: list[str] = []
    for job_id in JOB_IDS:
        relative = f"jobs/{job_id}/attempt-0/startup.log"
        path = root.joinpath(*PurePosixPath(relative).parts)
        _regular_file(path, f"startup log {job_id}")
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"startup log is not UTF-8: {job_id}") from exc
        if "\x00" in text:
            raise ValueError(f"startup log contains NUL: {job_id}")
        lines = text.splitlines()
        error_positions = [i for i, line in enumerate(lines) if line == ERROR_LINE]
        first_positions = [
            i for i, line in enumerate(lines) if line == FIRST_FAILURE_PREFIX
        ]
        second_positions = [
            i for i, line in enumerate(lines) if line == SECOND_FAILURE_PREFIX
        ]
        if (
            len(error_positions) != 1
            or len(first_positions) != 1
            or len(second_positions) != 1
            or not (
                error_positions[0] < first_positions[0] < second_positions[0]
            )
            or not lines
            or lines[-1] != ')"'
        ):
            raise ValueError(f"startup log lacks the exact pre-content failure: {job_id}")
        required_copies = (
            f"{REMOTE_PREFIX}/source/{SOURCE_NAME}",
            f"{REMOTE_PREFIX}/{MANIFEST_NAME}",
            f"{REMOTE_PREFIX}/source/jobs/{job_id}.json",
            f"{REMOTE_PREFIX}/source/{AUTHORIZATION_NAME}",
            f"{REMOTE_PREFIX}/control/{LAUNCH_CLAIM_NAME}",
        )
        if any(value not in text for value in required_copies):
            raise ValueError(f"startup log control copy chain changed: {job_id}")
        if any(
            forbidden in text
            for forbidden in (
                "/attempt-1/",
                "resume_claim.json",
                f"{REMOTE_PREFIX}/results/jobs/",
            )
        ):
            raise ValueError(f"startup log crossed the pre-content boundary: {job_id}")
        digest = hashlib.sha256(raw).hexdigest()
        log_hashes.append(digest)
        records.append(
            {
                "job_id": job_id,
                "attempt_index": 0,
                "relative_path": relative,
                "bytes": len(raw),
                "sha256": digest,
                "exact_error_count": 1,
            }
        )
    return {
        "count": len(records),
        "job_ids": list(JOB_IDS),
        "aggregate_sha256": canonical_sha256(log_hashes),
        "records": records,
    }


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(destination, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        raise


def generate_closeout(
    *,
    package_dir: str | Path,
    global_root_claim_path: str | Path,
    global_spot_claim_path: str | Path,
    startup_log_tree: str | Path,
    output_path: str | Path,
    metadata_snapshot_path: str | Path | None = None,
    direct_counts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate immutable local evidence and write the incident receipt once."""

    output = Path(output_path)
    if output.exists():
        raise FileExistsError("startup-failure closeout receipt is write-once")
    anchors, path_mismatch = _validate_package_controls(
        Path(package_dir),
        Path(global_root_claim_path),
        Path(global_spot_claim_path),
    )
    counts, metadata_evidence = _load_count_evidence(
        metadata_snapshot_path=(
            None if metadata_snapshot_path is None else Path(metadata_snapshot_path)
        ),
        direct_counts=direct_counts,
    )
    logs = _validate_startup_logs(Path(startup_log_tree))
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": RECEIPT_STATUS,
        "run_name": RUN_NAME,
        "incident": {
            "attempt_index": 0,
            "failed_jobs": 20,
            "phase": "pre_content_package_claim_validation",
            "startup_line": 746,
            "error": ERROR_LINE,
            "cause": (
                "windows_normcase_root_claim_paths_were_resolved_to_different_"
                "casing_in_the_spot_claim_then_compared_case_sensitively_on_linux"
            ),
            "deterministic_with_claimed_bytes": True,
        },
        "anchors": anchors,
        "path_case_mismatch": path_mismatch,
        "startup_logs": logs,
        "control_plane_counts": counts,
        "metadata_evidence": metadata_evidence,
        "content_access": {
            "hand_content_opened": False,
            "root_content_opened": False,
            "result_content_opened": False,
            "source_archive_opened": False,
            "source_archive_opaque_hash_verified": True,
            "startup_logs_opened": True,
            "cloud_query_performed": False,
            "cloud_mutated": False,
        },
        "disposition": {
            "current_run_irrecoverable": True,
            "attempt1_authorized": False,
            "attempt1_forbidden_reason": (
                "same_claimed_package_source_startup_and_claim_bytes_"
                "deterministically_repeat_the_precontent_failure"
            ),
            "alternate_seed_authorized": False,
            "reseed_authorized": False,
            "alternate_package_authorized": False,
            "replacement_package_authorized": False,
            "cloud_execution_authorized": False,
            "performance_lock_passed": False,
            "quality_pilot_authorized": False,
            "training_eligible": False,
            "current_profile_changed": False,
            "named_profile_added": False,
            "runtime_policy_activated": False,
            "m31_complete": False,
        },
    }
    _write_once(output, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--global-root-claim", required=True, type=Path)
    parser.add_argument("--global-spot-claim", required=True, type=Path)
    parser.add_argument("--startup-log-tree", required=True, type=Path)
    parser.add_argument(
        "--counts-json",
        "--metadata-snapshot",
        dest="metadata_snapshot",
        type=Path,
    )
    for key in COUNT_KEYS:
        parser.add_argument(f"--{key}-count", type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    supplied = {key: getattr(args, f"{key}_count") for key in COUNT_KEYS}
    any_direct = any(value is not None for value in supplied.values())
    all_direct = all(value is not None for value in supplied.values())
    if args.metadata_snapshot is not None and any_direct:
        raise ValueError("counts JSON and direct count arguments are mutually exclusive")
    if args.metadata_snapshot is None and not all_direct:
        raise ValueError(
            "supply --counts-json or all six direct zero count arguments"
        )
    direct = supplied if all_direct else None
    receipt = generate_closeout(
        package_dir=args.package_dir,
        global_root_claim_path=args.global_root_claim,
        global_spot_claim_path=args.global_spot_claim,
        startup_log_tree=args.startup_log_tree,
        output_path=args.output,
        metadata_snapshot_path=args.metadata_snapshot,
        direct_counts=direct,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "receipt_sha256": canonical_sha256(receipt),
                "status": receipt["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
