"""Fail-closed closeout for the consumed performance-lock rearm1 attempt.

Only immutable local control files, sealed root metadata, and the twenty
attempt-0 startup logs are opened.  Hand roots and result content are never
read, and this module has no cloud mutation path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from . import run_hu_m31_t3_step6d_performance_v2 as runner


RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_performance_lock_rearm1_startup_failure_closeout_v1"
)
RECEIPT_STATUS = (
    "rearm1_attempt0_consumed_by_deterministic_job_schema_mismatch_"
    "current_run_irrecoverable"
)
RUN_NAME = "regular-hu-m31-c02-lock-r1-20260717-001"
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
GLOBAL_ROOT_CLAIM_NAME = "GLOBAL_PERFORMANCE_LOCK_REARM1_CLAIM.json"
GLOBAL_SPOT_CLAIM_NAME = "GLOBAL_PERFORMANCE_LOCK_REARM1_SPOT_CLAIM.json"
MATERIALIZATION_NAME = "materialization.json"
SEAL_NAME = "seal.json"

PLAN_SHA256 = "3b8a4230531f0d81c5320b2b0d051878113ac57a38ad97fdb0b1d89e0f17a886"
RUN_CONTRACT_DIGEST = (
    "f02e8845401eef92ba617f2c832204bdc74c20aa9e91810c52bf99687c4886b5"
)
CURRENT_PROFILE_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)
REARM1_ROOT_CLAIM_SCHEMA = (
    "hu_m31_t3_step6d_performance_lock_rearm1_global_claim_v2"
)
REARM1_ROOT_CLAIM_STATUS = (
    "global_rearm1_claim_persisted_before_new_root_touch_crash_consumes_claim"
)
REARM1_MATERIALIZATION_SCHEMA = (
    "hu_m31_t3_step6d_performance_lock_rearm1_root_materialization_v2"
)
REARM1_SEAL_SCHEMA = "hu_m31_t3_step6d_performance_lock_rearm1_root_seal_v2"

EXPECTED_HASHES: dict[str, str] = {
    MANIFEST_NAME: "911cc6460451c34f4205f1841dc238d0e015b93c0989767bb09e9824845fe76a",
    READY_NAME: "afae9ff082879fb5d05504ce107a6f00e427168ccd9c34d70da9b5a47a4fcb9e",
    AUTHORIZATION_NAME: (
        "e2e4a5aac89aa4608b09c70e39e6a18791e8165109139b386f48f9b2c1eecde8"
    ),
    LAUNCH_CLAIM_NAME: (
        "c6d33e39f34e20647a18e5cf4a33ff8b2edb8244dfb2f46ad1e7168f413e9837"
    ),
    LAUNCH_RESULT_NAME: (
        "9b2a74f6cbf3c4543e4140c3aa93e80db9968bd929efda4952f2523a1c821977"
    ),
    SOURCE_NAME: "b8d34bfd2b46b3a7deaa378c624c474a76ae15066c11029a3bd6585339fc80c6",
    STARTUP_NAME: "feb942e39b52569004721e114baf1e839ab1003ebff63c68c794dd14a903ae94",
    GLOBAL_ROOT_CLAIM_NAME: (
        "461ed8c687c08dd1d9d171694d4ae1704ac745c81b1883befde8973eeb437b58"
    ),
    GLOBAL_SPOT_CLAIM_NAME: (
        "b1a6d897122aa6a2e6cb1004fd7d1c7c0e49dacff9f96987643c2326cafe284f"
    ),
    MATERIALIZATION_NAME: (
        "cd9032c8cce5a44e12fb76edc80f345f40937ab8c4f4b6f57884269b721adfe3"
    ),
    SEAL_NAME: "a804d6a25040ccacf043a11aa8a80b473fed2938ef756b8e9332709363eb39b8",
}

JOB_IDS = tuple(
    f"{role}-shard-{index:02d}"
    for role in ("candidate", "reference")
    for index in range(10)
)
EXPECTED_LOG_SHA256 = {
    "candidate-shard-00": "ec9ce3ceb1948e72ec820ca35b430c99d37398703dd95d392010410572dffee1",
    "candidate-shard-01": "58cef6fe579b477486aeea2be11591b684e2dc8d3813204a31ef8e3bddc6c179",
    "candidate-shard-02": "a52f417590c4b0a6a7d32b39fdc473876028120d71d39d8fe1b5086effaca95f",
    "candidate-shard-03": "d03c16fbc8a1f07ef8cc8cbb25ccf390da7c6a622a75a5a0cbf81b7e85b3f96e",
    "candidate-shard-04": "7c1035096fbddc0ebe84fce96620d6c867da0bdba2b5da808f71cda78a028707",
    "candidate-shard-05": "57580d153f8f251a28571e2f9043823004821fa0630d4cacc961929f1570086f",
    "candidate-shard-06": "d26722771a3cf2265a47d13722036fc5e46f171afdb12610c5fde20fcd6b2e6e",
    "candidate-shard-07": "66cb13e962b2c103c2b6530fb8de080d0ae4cf9ae1b1bc65810f53b6abd6aae8",
    "candidate-shard-08": "823dafe56bcbe360313287f190b7081ce83c5c5508e0f75fb94d2314131d1e0c",
    "candidate-shard-09": "a5284362d239becca3049ed840d77109ae991fd225a2b2f712ced2120b4f40f3",
    "reference-shard-00": "2630495d46934b1eb0afcd26fdc67519456ffd05784ec59db0cd2e8284ae5394",
    "reference-shard-01": "75bb8d19d8c93007bddd4c56c3fb32e4dd1956d4e66fbf00f0c581f430b24611",
    "reference-shard-02": "89bafc6548521f89a1a43ecece494bf06160df39155076d602acc9421de43b7a",
    "reference-shard-03": "be9185506da7e2ac435fdc2cf2873d949f352b892ff30428ecadaad4dcd051d5",
    "reference-shard-04": "3023acec0fab544c3315cca9a354d8352ee729790e1e4992ca29145817d12a55",
    "reference-shard-05": "780aacbf0bf2c3b30dd053c08d84ae4c0a20d978917508895194c075502e6f49",
    "reference-shard-06": "e6259399eebd7f772f74a506d894d3e155ef772fff1d3b1a40b5f6cd89eb15b9",
    "reference-shard-07": "6a6c002f481523138a41328488a71567cf0bd0a39d56e9f818597de9db8de981",
    "reference-shard-08": "8851505eb28e2fa23fab43aedeb4aa53c72c872025b885fb542d17c137b498ea",
    "reference-shard-09": "45a287d4293ea9feb89e17629f5ee9b2d8bb6beef3454ac0c9d06aa0b549242c",
}
COUNT_KEYS = (
    "live",
    "done",
    "heartbeat",
    "progress",
    "result",
    "runner_hand",
    "resume",
    "attempt1",
)
FORBIDDEN_ATTEMPT1_CONTROLS = (
    "resume_claim.json",
    "resume_result.json",
    "result_open_claim.json",
)
ERROR_LINE = "runner job schema changed"
PYTHON_FAILURE_LINE = "full100 startup failure: exit=1 line=816 command=python3 -"
SHELL_FAILURE_LINE = (
    'full100 startup failure: exit=1 line=1484 command=[[ "${#JOB_FIELDS[@]}" -eq 5 ]]'
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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")


def _read_canonical(path: Path, label: str) -> dict[str, Any]:
    _regular(path, label)
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _anchored(path: Path, name: str, label: str) -> str:
    _regular(path, label)
    digest = sha256_file(path)
    if digest != EXPECTED_HASHES[name]:
        raise ValueError(f"{label} is not the claimed run anchor")
    return digest


def _safe_member(package: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError("package job path is unsafe")
    path = package.joinpath(*pure.parts)
    try:
        path.resolve(strict=False).relative_to(package.resolve())
    except ValueError as exc:
        raise ValueError("package job path escapes package") from exc
    return path


def _validate_controls(
    *,
    package_dir: Path,
    global_root_claim_path: Path,
    global_spot_claim_path: Path,
    rearm1_root_directory: Path,
) -> dict[str, Any]:
    package = package_dir.resolve()
    if package.is_symlink() or not package.is_dir():
        raise ValueError("package directory is missing or unsafe")
    for name in FORBIDDEN_ATTEMPT1_CONTROLS:
        if (package / name).exists():
            raise ValueError(f"rearm1 attempt1 control is forbidden: {name}")
    paths = {
        name: package / name
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
    paths[GLOBAL_ROOT_CLAIM_NAME] = global_root_claim_path
    paths[GLOBAL_SPOT_CLAIM_NAME] = global_spot_claim_path
    paths[MATERIALIZATION_NAME] = rearm1_root_directory / MATERIALIZATION_NAME
    paths[SEAL_NAME] = rearm1_root_directory / SEAL_NAME
    for name, path in paths.items():
        _anchored(path, name, name)

    manifest = _read_canonical(paths[MANIFEST_NAME], "manifest")
    root_claim = _read_canonical(paths[GLOBAL_ROOT_CLAIM_NAME], "root claim")
    spot_claim = _read_canonical(paths[GLOBAL_SPOT_CLAIM_NAME], "Spot claim")
    ready = _read_canonical(paths[READY_NAME], "package ready")
    authorization = _read_canonical(paths[AUTHORIZATION_NAME], "authorization")
    launch_claim = _read_canonical(paths[LAUNCH_CLAIM_NAME], "launch claim")
    launch_result = _read_canonical(paths[LAUNCH_RESULT_NAME], "launch result")
    materialization = _read_canonical(
        paths[MATERIALIZATION_NAME], "materialization"
    )
    seal = _read_canonical(paths[SEAL_NAME], "seal")
    contract = manifest.get("run_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("rearm1 package run contract is absent")
    contract = runner.validate_run_contract(contract)
    false_manifest = (
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
        or manifest.get("source_bytes") != paths[SOURCE_NAME].stat().st_size
        or manifest.get("startup_name") != STARTUP_NAME
        or manifest.get("startup_sha256") != EXPECTED_HASHES[STARTUP_NAME]
        or manifest.get("plan_sha256") != PLAN_SHA256
        or manifest.get("run_contract_digest") != RUN_CONTRACT_DIGEST
        or runner.canonical_sha256(contract) != RUN_CONTRACT_DIGEST
        or runner.contract_variant(contract)
        != runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT
        or any(manifest.get(field) is not False for field in false_manifest)
    ):
        raise ValueError("rearm1 package control boundary changed")

    records = manifest.get("job_manifests")
    if not isinstance(records, list) or len(records) != 20:
        raise ValueError("rearm1 package job set changed")
    job_hashes: list[dict[str, Any]] = []
    for expected_id, record in zip(JOB_IDS, records, strict=True):
        if not isinstance(record, Mapping) or record.get("job_id") != expected_id:
            raise ValueError("rearm1 package job order changed")
        relative = record.get("path")
        if not isinstance(relative, str):
            raise ValueError("rearm1 package job path changed")
        job_path = _safe_member(package, relative)
        job = runner.validate_shard_manifest(_read_canonical(job_path, expected_id))
        if (
            job.get("schema")
            != runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SHARD_MANIFEST_SCHEMA
            or job.get("run_contract_digest") != RUN_CONTRACT_DIGEST
            or record.get("sha256") != sha256_file(job_path)
            or record.get("bytes") != job_path.stat().st_size
            or record.get("source_role") != job.get("source_role")
            or record.get("work_hand_indices") != job.get("work_hand_indices")
        ):
            raise ValueError("rearm1 package job anchor changed")
        job_hashes.append(
            {
                "job_id": expected_id,
                "sha256": sha256_file(job_path),
                "bytes": job_path.stat().st_size,
                "schema": job["schema"],
            }
        )

    if (
        root_claim.get("schema") != REARM1_ROOT_CLAIM_SCHEMA
        or root_claim.get("status") != REARM1_ROOT_CLAIM_STATUS
        or root_claim.get("lock_run_contract_digest") != RUN_CONTRACT_DIGEST
        or root_claim.get("ai_profiles_current", {}).get("sha256")
        != CURRENT_PROFILE_SHA256
        or root_claim.get("seed_contract", {}).get("seed_set_sha256")
        != runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SEED_SET_SHA256
        or root_claim.get("restrictions", {}).get("reseed_allowed") is not False
        or root_claim.get("restrictions", {}).get("cloud_authorized") is not False
        or materialization.get("schema") != REARM1_MATERIALIZATION_SCHEMA
        or materialization.get("root_count") != 100
        or materialization.get("reseeded") is not False
        or materialization.get("current_profile_changed") is not False
        or seal.get("schema") != REARM1_SEAL_SCHEMA
        or seal.get("root_count") != 100
        or seal.get("observation_count") != 200
        or seal.get("visibility", {}).get("opponent_private_discards_used")
        is not False
        or seal.get("current_profile_changed") is not False
    ):
        raise ValueError("rearm1 root control boundary changed")

    manifest_sha = EXPECTED_HASHES[MANIFEST_NAME]
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
        or spot_claim.get("alternate_package_authorization_allowed") is not False
    ):
        raise ValueError("rearm1 global Spot claim boundary changed")
    if (
        ready.get("package_manifest_sha256") != manifest_sha
        or ready.get("run_name") != RUN_NAME
        or authorization.get("package_manifest_sha256") != manifest_sha
        or authorization.get("global_spot_claim") != spot_claim
        or authorization.get("authorized_job_ids") != list(JOB_IDS)
        or authorization.get("spot_execution_authorized") is not True
        or authorization.get("performance_lock_authorized") is not True
        or launch_claim.get("selected_job_ids") != list(JOB_IDS)
        or launch_claim.get("crash_reuse_authorized") is not False
        or launch_result.get("selected_job_ids") != list(JOB_IDS)
        or launch_result.get("logical_job_count") != 20
        or launch_result.get("failures") != []
        or launch_result.get("launch_claim_sha256")
        != EXPECTED_HASHES[LAUNCH_CLAIM_NAME]
    ):
        raise ValueError("rearm1 launch control boundary changed")
    created = launch_result.get("created")
    if (
        not isinstance(created, list)
        or len(created) != 20
        or [row.get("job_id") for row in created if isinstance(row, Mapping)]
        != list(JOB_IDS)
        or any(
            not isinstance(row, Mapping)
            or row.get("attempt_index") != 0
            or row.get("status") != "created"
            for row in created
        )
    ):
        raise ValueError("rearm1 attempt0 creation evidence changed")

    return {
        "package_manifest": {
            "sha256": manifest_sha,
            "canonical_sha256": canonical_sha256(manifest),
        },
        "source": {
            "sha256": EXPECTED_HASHES[SOURCE_NAME],
            "bytes": paths[SOURCE_NAME].stat().st_size,
            "opened": False,
        },
        "startup": {
            "sha256": EXPECTED_HASHES[STARTUP_NAME],
            "bytes": paths[STARTUP_NAME].stat().st_size,
        },
        "package_ready_sha256": EXPECTED_HASHES[READY_NAME],
        "launch_authorization_sha256": EXPECTED_HASHES[AUTHORIZATION_NAME],
        "launch_claim_sha256": EXPECTED_HASHES[LAUNCH_CLAIM_NAME],
        "launch_result_sha256": EXPECTED_HASHES[LAUNCH_RESULT_NAME],
        "global_root_claim_sha256": EXPECTED_HASHES[GLOBAL_ROOT_CLAIM_NAME],
        "global_spot_claim_sha256": EXPECTED_HASHES[GLOBAL_SPOT_CLAIM_NAME],
        "materialization_sha256": EXPECTED_HASHES[MATERIALIZATION_NAME],
        "seal_sha256": EXPECTED_HASHES[SEAL_NAME],
        "job_manifest_records": job_hashes,
        "job_manifest_aggregate_sha256": canonical_sha256(job_hashes),
    }


def _validate_counts(value: Mapping[str, Any]) -> dict[str, int]:
    if set(value) != set(COUNT_KEYS):
        raise ValueError("control-plane count fields changed")
    counts: dict[str, int] = {}
    for key in COUNT_KEYS:
        count = value.get(key)
        if isinstance(count, bool) or not isinstance(count, int) or count != 0:
            raise ValueError(f"{key} count must be exactly zero")
        counts[key] = count
    return counts


def _validate_logs(tree: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for job_id in JOB_IDS:
        path = tree / "jobs" / job_id / "attempt-0" / "startup.log"
        _regular(path, f"startup log {job_id}")
        raw = path.read_bytes()
        observed_sha256 = hashlib.sha256(raw).hexdigest()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("startup log is not UTF-8") from exc
        expected_job_uri = f"{REMOTE_PREFIX}/source/jobs/{job_id}.json"
        if (
            text.count(ERROR_LINE) != 2
            or text.count(PYTHON_FAILURE_LINE) != 1
            or text.count(SHELL_FAILURE_LINE) != 1
            or expected_job_uri not in text
            or "unexpected result content" in text
            or "DONE.json" in text
            or observed_sha256 != EXPECTED_LOG_SHA256.get(job_id)
        ):
            raise ValueError(f"startup log {job_id} is not the exact schema failure")
        records.append(
            {
                "job_id": job_id,
                "relative_path": f"jobs/{job_id}/attempt-0/startup.log",
                "sha256": observed_sha256,
                "bytes": len(raw),
                "error_count": 2,
                "python_failure_line": 816,
                "shell_failure_line": 1484,
            }
        )
    observed = sorted(
        path.relative_to(tree).as_posix()
        for path in tree.rglob("*")
        if path.is_file()
    )
    expected = [record["relative_path"] for record in records]
    if observed != expected:
        raise ValueError("startup evidence tree is not exact 20/20 logs only")
    return {
        "count": 20,
        "job_ids": list(JOB_IDS),
        "records": records,
        "aggregate_sha256": canonical_sha256(records),
        "all_exact_schema_failures": True,
    }


def build_closeout(
    *,
    package_dir: str | Path,
    global_root_claim_path: str | Path,
    global_spot_claim_path: str | Path,
    rearm1_root_directory: str | Path,
    startup_log_tree: str | Path,
    direct_counts: Mapping[str, Any],
) -> dict[str, Any]:
    controls = _validate_controls(
        package_dir=Path(package_dir),
        global_root_claim_path=Path(global_root_claim_path),
        global_spot_claim_path=Path(global_spot_claim_path),
        rearm1_root_directory=Path(rearm1_root_directory),
    )
    counts = _validate_counts(direct_counts)
    logs = _validate_logs(Path(startup_log_tree))
    value = {
        "schema": RECEIPT_SCHEMA,
        "status": RECEIPT_STATUS,
        "run_name": RUN_NAME,
        "project": PROJECT,
        "bucket": BUCKET,
        "run_contract_digest": RUN_CONTRACT_DIGEST,
        "precontent_plan_sha256": PLAN_SHA256,
        "controls": controls,
        "startup_logs": logs,
        "control_plane_counts": counts,
        "incident": {
            "attempt_index": 0,
            "failed_jobs": 20,
            "phase": "pre_content_actual_job_contract_validation",
            "cause": (
                "packaged_recovery_v2_job_manifest_schema_was_rejected_by_"
                "startup_verifier_hardcoded_to_legacy_v2_shard_schema"
            ),
            "error": ERROR_LINE,
            "actual_job_schema": (
                runner.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SHARD_MANIFEST_SCHEMA
            ),
            "startup_expected_job_schema": runner.SHARD_MANIFEST_SCHEMA,
            "python_verifier_failure_line": 816,
            "shell_job_fields_failure_line": 1484,
            "deterministic_with_claimed_bytes": True,
        },
        "content_access": {
            "package_controls_opened": True,
            "startup_logs_opened": True,
            "source_archive_opaque_hash_verified": True,
            "source_archive_opened": False,
            "root_metadata_opened": True,
            "root_content_opened": False,
            "hand_content_opened": False,
            "result_content_opened": False,
            "cloud_query_performed": False,
            "cloud_mutated": False,
        },
        "disposition": {
            "attempt0_consumed": True,
            "current_run_irrecoverable": True,
            "rearm1_attempt1_authorized": False,
            "rearm1_package_reuse_authorized": False,
            "rearm1_root_reuse_authorized": False,
            "rearm1_seed_reuse_authorized": False,
            "rearm1_claim_reuse_authorized": False,
            "rearm1_result_reuse_authorized": False,
            "rearm1_cloud_execution_authorized": False,
            "fresh_rearm2_plan_authorized": False,
            "quality_pilot_authorized": False,
            "training_eligible": False,
            "current_profile_changed": False,
            "named_profile_added": False,
            "runtime_policy_activated": False,
            "m31_complete": False,
        },
    }
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"closeout receipt is write-once: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise FileExistsError(f"closeout receipt is write-once: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def generate_closeout(
    *,
    package_dir: str | Path,
    global_root_claim_path: str | Path,
    global_spot_claim_path: str | Path,
    rearm1_root_directory: str | Path,
    startup_log_tree: str | Path,
    output_path: str | Path,
    direct_counts: Mapping[str, Any],
) -> dict[str, Any]:
    value = build_closeout(
        package_dir=package_dir,
        global_root_claim_path=global_root_claim_path,
        global_spot_claim_path=global_spot_claim_path,
        rearm1_root_directory=rearm1_root_directory,
        startup_log_tree=startup_log_tree,
        direct_counts=direct_counts,
    )
    _write_once(Path(output_path), value)
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--global-root-claim", type=Path, required=True)
    parser.add_argument("--global-spot-claim", type=Path, required=True)
    parser.add_argument("--rearm1-roots", type=Path, required=True)
    parser.add_argument("--startup-log-tree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for key in COUNT_KEYS:
        parser.add_argument(f"--{key}-count", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    value = generate_closeout(
        package_dir=args.package_dir,
        global_root_claim_path=args.global_root_claim,
        global_spot_claim_path=args.global_spot_claim,
        rearm1_root_directory=args.rearm1_roots,
        startup_log_tree=args.startup_log_tree,
        output_path=args.output,
        direct_counts={
            key: getattr(args, f"{key}_count") for key in COUNT_KEYS
        },
    )
    print(canonical_sha256(value))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
