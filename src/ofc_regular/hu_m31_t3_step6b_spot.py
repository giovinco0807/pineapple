"""Immutable bounded Spot lifecycle for M3.1 Step 6b canary shards 1-9."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hu_m31_t3_step6a_spot import (
    DEFAULT_BUCKET,
    DEFAULT_PROJECT,
    DEFAULT_ZONE,
    EXPECTED_IMAGE_ID,
    EXPECTED_IMAGE_NAME,
    EXPECTED_IMAGE_SELF_LINK,
    EXPECTED_MACHINE_TYPE,
    _MODEL_PATHS,
    _SAFE_RUN,
    _copy_file,
    _parity_golden,
    _publish_once,
    _run,
    _subprocess_run,
    _zip_tree,
    canonical_bytes,
    sha256_file,
)
from .run_hu_m31_t3_step6a_shard import (
    PAIRED_HANDS_PER_SHARD,
    ROOTS_PER_SHARD,
    STEP5_CONTRACT_CANONICAL_SHA256,
)
from .run_hu_m31_t3_step6b_shard import (
    AUTHORIZED_SHARDS,
    EXPECTED_FEATURE_ENCODER_SHA256,
    EXPECTED_MODELS,
    EXPECTED_NATIVE_LIBRARY_SHA256,
    STEP6B_PACKAGE_SCHEMA,
    STEP6B_PARITY_SCHEMA,
    STEP6B_SHARD_SCHEMA,
    STEP6B_SUMMARY_SCHEMA,
)
from .validate_hu_m31_t3_step5_contract import validate_contract_file


STEP6B_AUTHORIZATION_SCHEMA = "hu_m31_t3_step6b_launch_authorization_v1"
STEP6B_DRY_RUN_SCHEMA = "hu_m31_t3_step6b_local_dry_run_v1"
STEP6B_LAUNCH_SCHEMA = "hu_m31_t3_step6b_launch_v1"
STEP6B_STATUS_SCHEMA = "hu_m31_t3_step6b_cloud_status_v1"
STEP6B_DONE_SCHEMA = "hu_m31_t3_step6b_done_v1"
STEP6B_RECEIVE_SCHEMA = "hu_m31_t3_step6b_receive_v1"
SOURCE_NAME = "ofc_regular_hu_m31_t3_step6b_source.zip"
SCHEDULE_NAME = "shards_manifest.jsonl"
STARTUP_NAME = "startup_hu_m31_t3_step6b.sh"
MAX_LAUNCH_BATCH = 3
DEFAULT_ZONES = ("asia-northeast1-b", "asia-northeast1-c")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATHS = (
    "configs/hu_joint_policy_m31_t3_step5_contract.json",
    "configs/hu_joint_policy_m31_t3_step5_status.json",
    "configs/hu_joint_policy_m31_t3_step6a_status.json",
    "configs/hu_m43_attempt08_runtime_requirements.txt",
)


def _write_once(path: Path, value: Any, *, raw: bool = False) -> None:
    if path.exists():
        raise FileExistsError(f"immutable Step 6b artifact exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value if raw else canonical_bytes(value)
    if not isinstance(payload, bytes):
        raise TypeError("raw Step 6b artifact must be bytes")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_canonical(path: Path, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_bytes(dict(row)) for row in rows)


def build_schedule(run_name: str) -> list[dict[str, Any]]:
    if not _SAFE_RUN.fullmatch(run_name):
        raise ValueError("Step 6b run name is not a safe bounded GCP identity")
    return [
        {
            "schema": STEP6B_SHARD_SCHEMA,
            "run_name": run_name,
            "shard": shard,
            "global_hand_start": shard * PAIRED_HANDS_PER_SHARD,
            "hand_count": PAIRED_HANDS_PER_SHARD,
            "root_count": ROOTS_PER_SHARD,
            "output_prefix": f"shard-{shard:03d}",
        }
        for shard in range(10)
    ]


def _step6a_evidence(
    *, status_path: Path, run_dir: Path, received_dir: Path
) -> dict[str, Any]:
    status = json.loads(status_path.read_text(encoding="utf-8"))
    accepted = status.get("accepted_run")
    if (
        status.get("schema") != "hu_joint_policy_m31_t3_step6a_status_v1"
        or status.get("status") != "step6a_spot_shard0_complete"
        or status.get("spot_canary_shard_0_complete") is not True
        or status.get("spot_instance_deleted") is not True
        or status.get("spot_remaining_canary_shards_authorized") is not False
        or status.get("spot_production_fanout_authorized") is not False
        or status.get("canary_rows_eligible_for_training") is not False
        or status.get("current_profile_changed") is not False
        or not isinstance(accepted, Mapping)
    ):
        raise ValueError("accepted Step 6a status boundary changed")
    paths = {
        "source": run_dir / "ofc_regular_hu_m31_t3_step6a_source.zip",
        "manifest": run_dir / "manifest.json",
        "authorization": run_dir / "launch_authorization.json",
        "done": received_dir / "DONE.json",
        "summary": received_dir / "summary.json",
        "receive_receipt": received_dir / "receive_receipt.json",
    }
    expected = {
        "source": accepted.get("source_sha256"),
        "manifest": accepted.get("manifest_sha256"),
        "authorization": accepted.get("authorization_sha256"),
        "done": accepted.get("done_sha256"),
        "summary": accepted.get("summary_sha256"),
        "receive_receipt": accepted.get("receive_receipt_sha256"),
    }
    if any(not path.is_file() for path in paths.values()) or any(
        sha256_file(paths[name]) != digest for name, digest in expected.items()
    ):
        raise ValueError("accepted Step 6a evidence hash changed")
    done = _load_canonical(paths["done"], "accepted Step 6a DONE")
    summary = _load_canonical(paths["summary"], "accepted Step 6a summary")
    receipt = _load_canonical(
        paths["receive_receipt"], "accepted Step 6a receive receipt"
    )
    if (
        done.get("status") != "complete"
        or done.get("shard") != 0
        or summary.get("all_gates_passed") is not True
        or summary.get("resumed_task_count", 0) < 1
        or receipt.get("status") != "pass"
        or receipt.get("task_count") != 25
        or receipt.get("root_task_count") != 25
        or any(
            value is not False
            for value in (
                summary.get("training_eligible"),
                summary.get("production_fanout_authorized"),
                summary.get("current_profile_changed"),
            )
        )
    ):
        raise ValueError("accepted Step 6a result is not a passed shard 0")
    return {
        "run_name": accepted["run_name"],
        "source_sha256": expected["source"],
        "manifest_sha256": expected["manifest"],
        "schedule_sha256": done["schedule_sha256"],
        "authorization_sha256": expected["authorization"],
        "done_sha256": expected["done"],
        "summary_sha256": expected["summary"],
        "receive_receipt_sha256": expected["receive_receipt"],
        "status_sha256": sha256_file(status_path),
        "native_library_sha256": accepted["native_library_sha256"],
        "feature_encoder_sha256": accepted["feature_encoder_sha256"],
    }


def package_step6b(
    *,
    run_name: str,
    run_dir: str | Path,
    linux_library: str | Path,
    linux_feature_encoder: str | Path,
    step5_validation: str | Path,
    step6a_run_dir: str | Path,
    step6a_received_dir: str | Path,
    repository_root: str | Path = _REPO_ROOT,
    startup: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    destination = Path(run_dir).resolve()
    if destination.exists():
        raise FileExistsError("Step 6b package run directory is immutable")
    if not _SAFE_RUN.fullmatch(run_name):
        raise ValueError("Step 6b run name is invalid")
    contract_path = root / "configs/hu_joint_policy_m31_t3_step5_contract.json"
    validation = validate_contract_file(contract_path, repo_root=root)
    if (
        validation.get("status") != "pass"
        or validation.get("spot_canary_authorized") is not True
        or validation.get("production_fanout_authorized") is not False
    ):
        raise ValueError("Step 5 contract does not allow canary continuation")
    validation_path = Path(step5_validation).resolve()
    step5_status_path = root / "configs/hu_joint_policy_m31_t3_step5_status.json"
    step5_status = json.loads(step5_status_path.read_text(encoding="utf-8"))
    if (
        step5_status.get("status") != "step5_contract_complete"
        or step5_status.get("contract", {}).get("validation_artifact_sha256")
        != sha256_file(validation_path)
        or step5_status.get("spot_production_fanout_authorized") is not False
        or step5_status.get("current_profile_changed") is not False
    ):
        raise ValueError("Step 5 status or validation artifact changed")
    step6a_status_path = root / "configs/hu_joint_policy_m31_t3_step6a_status.json"
    accepted = _step6a_evidence(
        status_path=step6a_status_path,
        run_dir=Path(step6a_run_dir).resolve(),
        received_dir=Path(step6a_received_dir).resolve(),
    )
    library = Path(linux_library).resolve()
    feature = Path(linux_feature_encoder).resolve()
    if (
        not library.is_file()
        or library.name != "libofc_hu_m3_engine.so"
        or sha256_file(library) != EXPECTED_NATIVE_LIBRARY_SHA256
        or not feature.is_file()
        or feature.name != "libofc_stage3_feature_encoder.so"
        or sha256_file(feature) != EXPECTED_FEATURE_ENCODER_SHA256
    ):
        raise ValueError("Step 6b frozen Linux binary identity changed")
    startup_path = Path(startup or root / "scripts" / STARTUP_NAME).resolve()
    if not startup_path.is_file():
        raise ValueError("Step 6b startup script is missing")

    staging = destination.with_name(f".{destination.name}.{os.getpid()}.staging")
    if staging.exists():
        raise FileExistsError("Step 6b package staging already exists")
    package_root = staging / "package_src"
    package_root.mkdir(parents=True)
    try:
        entries: dict[str, dict[str, Any]] = {}
        for path in sorted((root / "src" / "ofc_regular").rglob("*.py")):
            if "__pycache__" in path.parts or path.is_symlink():
                continue
            relative = "src/" + path.relative_to(root / "src").as_posix()
            entries[relative] = _copy_file(path, package_root / relative)
        for relative in (*_CONFIG_PATHS, *_MODEL_PATHS):
            entries[relative] = _copy_file(root / relative, package_root / relative)
        if {
            relative: entries[relative]["sha256"] for relative in _MODEL_PATHS
        } != EXPECTED_MODELS:
            raise ValueError("Step 6b frozen model identity changed")
        native_relative = "native/release/libofc_hu_m3_engine.so"
        feature_relative = "target/release/libofc_stage3_feature_encoder.so"
        entries[native_relative] = _copy_file(library, package_root / native_relative)
        entries[feature_relative] = _copy_file(feature, package_root / feature_relative)
        evidence_sources = {
            "artifacts/step5/contract_validation.json": validation_path,
            "artifacts/step6a/manifest.json": Path(step6a_run_dir).resolve()
            / "manifest.json",
            "artifacts/step6a/launch_authorization.json": Path(step6a_run_dir).resolve()
            / "launch_authorization.json",
            "artifacts/step6a/DONE.json": Path(step6a_received_dir).resolve()
            / "DONE.json",
            "artifacts/step6a/summary.json": Path(step6a_received_dir).resolve()
            / "summary.json",
            "artifacts/step6a/receive_receipt.json": Path(step6a_received_dir).resolve()
            / "receive_receipt.json",
        }
        for relative, source in evidence_sources.items():
            entries[relative] = _copy_file(source, package_root / relative)
        golden_relative = "artifacts/step6b/parity_golden.json"
        _write_once(package_root / golden_relative, _parity_golden(root))
        entries[golden_relative] = {
            "sha256": sha256_file(package_root / golden_relative),
            "bytes": (package_root / golden_relative).stat().st_size,
        }
        source_path = staging / SOURCE_NAME
        _zip_tree(package_root, source_path)
        schedule_path = staging / SCHEDULE_NAME
        _write_once(schedule_path, _canonical_jsonl(build_schedule(run_name)), raw=True)
        shutil.copy2(startup_path, staging / STARTUP_NAME)
        manifest = {
            "schema": STEP6B_PACKAGE_SCHEMA,
            "status": "packaged_local_no_gcloud",
            "run_name": run_name,
            "total_shards": 10,
            "authorized_shards": list(AUTHORIZED_SHARDS),
            "paired_hands_per_shard": PAIRED_HANDS_PER_SHARD,
            "roots_per_shard": ROOTS_PER_SHARD,
            "source_name": SOURCE_NAME,
            "source_sha256": sha256_file(source_path),
            "source_bytes": source_path.stat().st_size,
            "schedule_name": SCHEDULE_NAME,
            "schedule_sha256": sha256_file(schedule_path),
            "startup_name": STARTUP_NAME,
            "startup_sha256": sha256_file(staging / STARTUP_NAME),
            "step5_contract_byte_sha256": sha256_file(contract_path),
            "step5_contract_canonical_sha256": STEP5_CONTRACT_CANONICAL_SHA256,
            "step5_status_sha256": sha256_file(step5_status_path),
            "step5_validation_sha256": sha256_file(validation_path),
            "step6a_accepted": accepted,
            "native_library": {
                "path": native_relative,
                "sha256": entries[native_relative]["sha256"],
                "bytes": entries[native_relative]["bytes"],
                "engine_version": "ofc_hu_m3_engine/0.1.0",
                "rust_toolchain": "1.93.0",
                "docker_image": "rust:1.93.0-bookworm",
                "docker_image_digest": (
                    "sha256:d0a4aa3ca2e1088ac0c81690914a0d810f2eee188197034edf366ed010a2b382"
                ),
            },
            "feature_encoder_library": {
                "path": feature_relative,
                "sha256": entries[feature_relative]["sha256"],
                "bytes": entries[feature_relative]["bytes"],
                "rust_toolchain": "1.93.0",
                "docker_image": "rust:1.93.0-bookworm",
                "docker_image_digest": (
                    "sha256:d0a4aa3ca2e1088ac0c81690914a0d810f2eee188197034edf366ed010a2b382"
                ),
            },
            "models": EXPECTED_MODELS,
            "parity_golden_path": golden_relative,
            "parity_golden_sha256": entries[golden_relative]["sha256"],
            "source_entries": entries,
            "source_entry_count": len(entries),
            "machine_type": EXPECTED_MACHINE_TYPE,
            "image_name": EXPECTED_IMAGE_NAME,
            "image_id": EXPECTED_IMAGE_ID,
            "checkpoint_unit": "completed_paired_hand",
            "heartbeat_interval_seconds": 60,
            "resume_drill_required_each_shard": True,
            "canary_rows_training_eligible": False,
            "remaining_canary_shards_authorized": True,
            "production_fanout_authorized": False,
            "gcloud_invoked": False,
            "spot_vm_started": False,
            "current_profile_changed": False,
            "m31_complete": False,
        }
        _write_once(staging / "manifest.json", manifest)
        os.replace(staging, destination)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_package(run_dir: str | Path) -> dict[str, Any]:
    target = Path(run_dir).resolve()
    manifest = _load_canonical(target / "manifest.json", "Step 6b manifest")
    source = target / SOURCE_NAME
    schedule = target / SCHEDULE_NAME
    startup = target / STARTUP_NAME
    if (
        manifest.get("schema") != STEP6B_PACKAGE_SCHEMA
        or manifest.get("status") != "packaged_local_no_gcloud"
        or manifest.get("total_shards") != 10
        or manifest.get("authorized_shards") != list(AUTHORIZED_SHARDS)
        or manifest.get("source_sha256") != sha256_file(source)
        or manifest.get("schedule_sha256") != sha256_file(schedule)
        or manifest.get("startup_sha256") != sha256_file(startup)
        or manifest.get("native_library", {}).get("sha256")
        != EXPECTED_NATIVE_LIBRARY_SHA256
        or manifest.get("feature_encoder_library", {}).get("sha256")
        != EXPECTED_FEATURE_ENCODER_SHA256
        or manifest.get("models") != EXPECTED_MODELS
        or manifest.get("remaining_canary_shards_authorized") is not True
        or manifest.get("canary_rows_training_eligible") is not False
        or manifest.get("production_fanout_authorized") is not False
        or manifest.get("current_profile_changed") is not False
        or manifest.get("gcloud_invoked") is not False
        or manifest.get("spot_vm_started") is not False
    ):
        raise ValueError("Step 6b package manifest changed")
    rows = [
        json.loads(line) for line in schedule.read_text(encoding="utf-8").splitlines()
    ]
    if rows != build_schedule(str(manifest["run_name"])):
        raise ValueError("Step 6b schedule changed")
    entries = manifest.get("source_entries")
    if not isinstance(entries, Mapping) or len(entries) != manifest.get(
        "source_entry_count"
    ):
        raise ValueError("Step 6b source entry manifest missing")
    package_root = target / "package_src"
    with zipfile.ZipFile(source) as archive:
        names = tuple(sorted(archive.namelist()))
        if names != tuple(sorted(entries)) or any("\\" in name for name in names):
            raise ValueError("Step 6b zip entry set changed")
        for name, record in entries.items():
            data = archive.read(name)
            digest = hashlib.sha256(data).hexdigest()
            if (
                not isinstance(record, Mapping)
                or digest != record.get("sha256")
                or len(data) != record.get("bytes")
                or sha256_file(package_root / name) != digest
            ):
                raise ValueError(f"Step 6b source content changed: {name}")
    accepted = manifest.get("step6a_accepted")
    if (
        not isinstance(accepted, Mapping)
        or accepted.get("native_library_sha256") != EXPECTED_NATIVE_LIBRARY_SHA256
        or accepted.get("feature_encoder_sha256") != EXPECTED_FEATURE_ENCODER_SHA256
    ):
        raise ValueError("Step 6b accepted Step 6a binding changed")
    return manifest


def record_local_dry_run(
    *, run_dir: str | Path, parity_report: str | Path
) -> dict[str, Any]:
    target = Path(run_dir).resolve()
    manifest = validate_package(target)
    parity_path = Path(parity_report).resolve()
    parity = _load_canonical(parity_path, "Step 6b local parity")
    if (
        parity.get("schema") != STEP6B_PARITY_SCHEMA
        or parity.get("all_gates_passed") is not True
        or parity.get("source_package_sha256") != manifest["source_sha256"]
        or parity.get("manifest_sha256") != sha256_file(target / "manifest.json")
        or parity.get("parity_golden_sha256") != manifest["parity_golden_sha256"]
        or parity.get("native_library_sha256") != EXPECTED_NATIVE_LIBRARY_SHA256
        or parity.get("teacher_generation_started") is not False
        or parity.get("current_profile_changed") is not False
        or parity.get("production_fanout_authorized") is not False
    ):
        raise ValueError("Step 6b local Linux dry-run parity failed")
    receipt = {
        "schema": STEP6B_DRY_RUN_SCHEMA,
        "status": "pass",
        "run_name": manifest["run_name"],
        "manifest_sha256": sha256_file(target / "manifest.json"),
        "source_sha256": manifest["source_sha256"],
        "schedule_sha256": manifest["schedule_sha256"],
        "native_library_sha256": EXPECTED_NATIVE_LIBRARY_SHA256,
        "feature_encoder_sha256": EXPECTED_FEATURE_ENCODER_SHA256,
        "parity_report_sha256": sha256_file(parity_path),
        "linux_portable_parity": True,
        "package_validation": True,
        "teacher_generation_started": False,
        "gcloud_invoked": False,
        "current_profile_changed": False,
        "production_fanout_authorized": False,
    }
    _write_once(target / "local_dry_run_receipt.json", receipt)
    return receipt


def authorize_launch(run_dir: str | Path) -> dict[str, Any]:
    target = Path(run_dir).resolve()
    manifest = validate_package(target)
    receipt = _load_canonical(
        target / "local_dry_run_receipt.json", "Step 6b dry-run receipt"
    )
    if (
        receipt.get("schema") != STEP6B_DRY_RUN_SCHEMA
        or receipt.get("status") != "pass"
        or receipt.get("manifest_sha256") != sha256_file(target / "manifest.json")
        or receipt.get("source_sha256") != manifest["source_sha256"]
        or receipt.get("teacher_generation_started") is not False
        or receipt.get("gcloud_invoked") is not False
    ):
        raise ValueError("Step 6b dry-run receipt changed")
    authorization = {
        "schema": STEP6B_AUTHORIZATION_SCHEMA,
        "status": "authorized",
        "run_name": manifest["run_name"],
        "manifest_sha256": sha256_file(target / "manifest.json"),
        "source_sha256": manifest["source_sha256"],
        "schedule_sha256": manifest["schedule_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "dry_run_receipt_sha256": sha256_file(target / "local_dry_run_receipt.json"),
        "step6a_accepted": manifest["step6a_accepted"],
        "authorized_shards": list(AUTHORIZED_SHARDS),
        "spot_authorized": True,
        "remaining_canary_shards_authorized": True,
        "production_fanout_authorized": False,
        "root_execution_started": False,
        "current_profile_changed": False,
        "authorized_unix_seconds": time.time(),
    }
    _write_once(target / "launch_authorization.json", authorization)
    return authorization


def validate_launch(run_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    target = Path(run_dir).resolve()
    manifest = validate_package(target)
    auth = _load_canonical(
        target / "launch_authorization.json", "Step 6b launch authorization"
    )
    timestamp = auth.get("authorized_unix_seconds")
    if (
        auth.get("schema") != STEP6B_AUTHORIZATION_SCHEMA
        or auth.get("status") != "authorized"
        or auth.get("run_name") != manifest["run_name"]
        or auth.get("manifest_sha256") != sha256_file(target / "manifest.json")
        or auth.get("source_sha256") != manifest["source_sha256"]
        or auth.get("schedule_sha256") != manifest["schedule_sha256"]
        or auth.get("startup_sha256") != manifest["startup_sha256"]
        or auth.get("step6a_accepted") != manifest["step6a_accepted"]
        or auth.get("authorized_shards") != list(AUTHORIZED_SHARDS)
        or auth.get("spot_authorized") is not True
        or auth.get("remaining_canary_shards_authorized") is not True
        or auth.get("production_fanout_authorized") is not False
        or auth.get("root_execution_started") is not False
        or auth.get("current_profile_changed") is not False
        or isinstance(timestamp, bool)
        or not isinstance(timestamp, (int, float))
        or not math.isfinite(float(timestamp))
    ):
        raise ValueError("Step 6b launch authorization changed")
    return manifest, auth


def _bounded_shards(shards: Sequence[int]) -> tuple[int, ...]:
    selected = tuple(shards)
    if (
        not selected
        or len(selected) > MAX_LAUNCH_BATCH
        or len(set(selected)) != len(selected)
        or any(
            isinstance(shard, bool) or shard not in AUTHORIZED_SHARDS
            for shard in selected
        )
    ):
        raise ValueError("Step 6b launch requires 1-3 unique shards from 1..9")
    return selected


def _parse_shards(value: str) -> tuple[int, ...]:
    try:
        return _bounded_shards(tuple(int(item) for item in value.split(",")))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _publish_package(
    *, target: Path, manifest: Mapping[str, Any], project: str, bucket: str
) -> str:
    prefix = f"gs://{bucket}/runs/{manifest['run_name']}"
    for source, uri in (
        (target / "manifest.json", f"{prefix}/manifest.json"),
        (target / SOURCE_NAME, f"{prefix}/source/{SOURCE_NAME}"),
        (target / SCHEDULE_NAME, f"{prefix}/source/{SCHEDULE_NAME}"),
        (
            target / "launch_authorization.json",
            f"{prefix}/source/launch_authorization.json",
        ),
    ):
        _publish_once(source, uri, project=project)
    return prefix


def _create_instance(
    *,
    target: Path,
    manifest: Mapping[str, Any],
    auth: Mapping[str, Any],
    prefix: str,
    shard: int,
    project: str,
    zone: str,
) -> dict[str, Any]:
    padded = f"{shard:03d}"
    done_uri = f"{prefix}/results/shard-{padded}/DONE.json"
    done = _subprocess_run(
        ["gcloud", "storage", "objects", "describe", done_uri, "--project", project],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    if done.returncode == 0:
        raise FileExistsError(f"Step 6b shard {shard} DONE already exists")
    instance = f"{manifest['run_name']}-s{padded}"
    existing = _subprocess_run(
        [
            "gcloud",
            "compute",
            "instances",
            "list",
            "--project",
            project,
            f"--filter=name={instance}",
            "--format=value(name)",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    if existing.returncode != 0:
        raise RuntimeError(f"failed to inspect Step 6b instance {instance}")
    if existing.stdout.strip():
        raise FileExistsError(f"Step 6b instance already exists: {instance}")
    metadata = ",".join(
        (
            f"PROJECT_ID={project}",
            f"BUCKET={prefix.split('/')[2]}",
            f"RUN_NAME={manifest['run_name']}",
            f"SHARD={shard}",
            f"SOURCE_URI={prefix}/source/{SOURCE_NAME}",
            f"SOURCE_SHA256={manifest['source_sha256']}",
            f"MANIFEST_SHA256={auth['manifest_sha256']}",
            f"SCHEDULE_SHA256={manifest['schedule_sha256']}",
            f"AUTHORIZATION_SHA256={sha256_file(target / 'launch_authorization.json')}",
            "SELF_DELETE=1",
        )
    )
    _run(
        [
            "gcloud",
            "compute",
            "instances",
            "create",
            instance,
            "--project",
            project,
            "--zone",
            zone,
            "--machine-type",
            EXPECTED_MACHINE_TYPE,
            "--provisioning-model=SPOT",
            "--instance-termination-action=DELETE",
            "--image-project=debian-cloud",
            f"--image={EXPECTED_IMAGE_NAME}",
            "--boot-disk-size=50GB",
            "--boot-disk-type=hyperdisk-balanced",
            "--scopes=https://www.googleapis.com/auth/cloud-platform",
            f"--metadata={metadata}",
            f"--metadata-from-file=startup-script={target / STARTUP_NAME}",
            "--quiet",
        ],
        timeout=600,
    )
    return {"shard": shard, "instance": instance, "zone": zone, "status": "created"}


def launch_shards(
    *,
    run_dir: str | Path,
    shards: Sequence[int],
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
    zones: Sequence[str] = DEFAULT_ZONES,
) -> dict[str, Any]:
    selected = _bounded_shards(shards)
    if not zones or any(zone not in DEFAULT_ZONES for zone in zones):
        raise ValueError("Step 6b launch zones must be asia-northeast1-b/c")
    target = Path(run_dir).resolve()
    manifest, auth = validate_launch(target)
    image = json.loads(
        _run(
            [
                "gcloud",
                "compute",
                "images",
                "describe",
                EXPECTED_IMAGE_NAME,
                "--project",
                "debian-cloud",
                "--format=json",
            ],
            timeout=120,
        ).stdout
    )
    if (
        str(image.get("id")) != EXPECTED_IMAGE_ID
        or image.get("selfLink") != EXPECTED_IMAGE_SELF_LINK
    ):
        raise ValueError("Step 6b immutable Debian image changed")
    prefix = _publish_package(
        target=target, manifest=manifest, project=project, bucket=bucket
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected)) as pool:
        futures = [
            pool.submit(
                _create_instance,
                target=target,
                manifest=manifest,
                auth=auth,
                prefix=prefix,
                shard=shard,
                project=project,
                zone=zones[(shard - 1) % len(zones)],
            )
            for shard in selected
        ]
        created = [future.result() for future in futures]
    return {
        "schema": STEP6B_LAUNCH_SCHEMA,
        "status": "created",
        "run_name": manifest["run_name"],
        "created": sorted(created, key=lambda row: row["shard"]),
        "authorized_shards": list(AUTHORIZED_SHARDS),
        "remaining_canary_shards_authorized": True,
        "production_fanout_authorized": False,
        "current_profile_changed": False,
    }


def cloud_status(
    *,
    run_dir: str | Path,
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    manifest, _auth = validate_launch(run_dir)
    prefix = f"gs://{bucket}/runs/{manifest['run_name']}"
    listing = _subprocess_run(
        ["gcloud", "storage", "ls", "--recursive", prefix, "--project", project],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    objects = listing.stdout.splitlines() if listing.returncode == 0 else []
    instances_result = _subprocess_run(
        [
            "gcloud",
            "compute",
            "instances",
            "list",
            "--project",
            project,
            f"--filter=name~'^{manifest['run_name']}-s00[1-9]$'",
            "--format=json",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    instances = (
        json.loads(instances_result.stdout)
        if instances_result.returncode == 0 and instances_result.stdout.strip()
        else []
    )
    by_name = {row.get("name"): row for row in instances}
    shards = {}
    for shard in AUTHORIZED_SHARDS:
        padded = f"{shard:03d}"
        result_prefix = f"/results/shard-{padded}/"
        progress_prefix = f"/progress/shard-{padded}/"
        instance_name = f"{manifest['run_name']}-s{padded}"
        vm = by_name.get(instance_name)
        shards[padded] = {
            "done": any(
                uri.endswith(f"/results/shard-{padded}/DONE.json") for uri in objects
            ),
            "heartbeat_present": any(
                uri.endswith(f"/progress/shard-{padded}/heartbeat.json")
                for uri in objects
            ),
            "progress_object_count": sum(progress_prefix in uri for uri in objects),
            "result_object_count": sum(result_prefix in uri for uri in objects),
            "instance": (
                None
                if vm is None
                else {
                    "name": vm.get("name"),
                    "status": vm.get("status"),
                    "zone": str(vm.get("zone", "")).rsplit("/", 1)[-1],
                }
            ),
        }
    return {
        "schema": STEP6B_STATUS_SCHEMA,
        "run_name": manifest["run_name"],
        "done_count": sum(row["done"] for row in shards.values()),
        "all_done": all(row["done"] for row in shards.values()),
        "shards": shards,
        "remaining_canary_shards_authorized": True,
        "production_fanout_authorized": False,
        "current_profile_changed": False,
    }


def _validate_received_shard(
    *,
    shard_dir: Path,
    shard: int,
    manifest: Mapping[str, Any],
    auth_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    done = _load_canonical(shard_dir / "DONE.json", f"Step 6b shard {shard} DONE")
    summary = _load_canonical(
        shard_dir / "summary.json", f"Step 6b shard {shard} summary"
    )
    padded = f"{shard:03d}"
    if (
        done.get("schema") != STEP6B_DONE_SCHEMA
        or done.get("status") != "complete"
        or done.get("run_name") != manifest["run_name"]
        or done.get("shard") != shard
        or done.get("source_sha256") != manifest["source_sha256"]
        or done.get("manifest_sha256") != manifest_sha256
        or done.get("schedule_sha256") != manifest["schedule_sha256"]
        or done.get("authorization_sha256") != auth_sha256
        or done.get("native_library_sha256") != EXPECTED_NATIVE_LIBRARY_SHA256
        or done.get("resume_drill_passed") is not True
        or done.get("authorized_shards") != list(AUTHORIZED_SHARDS)
        or done.get("training_eligible") is not False
        or done.get("production_fanout_authorized") is not False
        or done.get("current_profile_changed") is not False
        or summary.get("schema") != STEP6B_SUMMARY_SCHEMA
        or summary.get("status") != "pass"
        or summary.get("run_name") != manifest["run_name"]
        or summary.get("shard") != shard
        or summary.get("global_hand_start") != shard * PAIRED_HANDS_PER_SHARD
        or summary.get("source_package_sha256") != manifest["source_sha256"]
        or summary.get("manifest_sha256") != manifest_sha256
        or summary.get("all_gates_passed") is not True
        or summary.get("resumed_task_count", 0) < 1
        or summary.get("teacher_value_status") != "diagnostic_not_match_EV"
        or summary.get("training_eligible") is not False
        or summary.get("production_fanout_authorized") is not False
        or summary.get("current_profile_changed") is not False
    ):
        raise ValueError(f"Step 6b received boundary failed: shard {padded}")
    files = done.get("files")
    if not isinstance(files, Mapping):
        raise ValueError(f"Step 6b DONE file manifest missing: shard {padded}")
    actual = {
        path.relative_to(shard_dir).as_posix()
        for path in shard_dir.rglob("*")
        if path.is_file()
    }
    if actual != set(files) | {"DONE.json"} or any(
        path.is_symlink() for path in shard_dir.rglob("*")
    ):
        raise ValueError(f"Step 6b received file set changed: shard {padded}")
    for relative, record in files.items():
        path = shard_dir / str(relative)
        if (
            not isinstance(record, Mapping)
            or sha256_file(path) != record.get("sha256")
            or path.stat().st_size != record.get("bytes")
        ):
            raise ValueError(
                f"Step 6b received file changed: shard {padded}/{relative}"
            )
    expected_indices = set(
        range(shard * PAIRED_HANDS_PER_SHARD, (shard + 1) * PAIRED_HANDS_PER_SHARD)
    )
    task_indices = {
        int(path.stem.split("_")[-1])
        for path in (shard_dir / "tasks").glob("hand_*.json")
    }
    root_indices = {
        int(path.stem.split("_")[-1])
        for path in (shard_dir / "roots").glob("hand_*.json")
    }
    heartbeat = _load_canonical(
        shard_dir / "heartbeat.json", f"Step 6b shard {shard} heartbeat"
    )
    if (
        task_indices != expected_indices
        or root_indices != expected_indices
        or heartbeat.get("status") != "pass"
        or heartbeat.get("completed_tasks") != PAIRED_HANDS_PER_SHARD
        or heartbeat.get("pending_tasks") != 0
        or heartbeat.get("resumed_task_count", 0) < 1
    ):
        raise ValueError(f"Step 6b received task/heartbeat set failed: shard {padded}")
    return {
        "done_sha256": sha256_file(shard_dir / "DONE.json"),
        "summary_sha256": sha256_file(shard_dir / "summary.json"),
        "heartbeat_sha256": sha256_file(shard_dir / "heartbeat.json"),
        "task_count": len(task_indices),
        "root_task_count": len(root_indices),
        "resumed_task_count": summary["resumed_task_count"],
    }


def receive_shards(
    *,
    run_dir: str | Path,
    output_dir: str | Path,
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    source = Path(run_dir).resolve()
    manifest, _auth = validate_launch(source)
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError("Step 6b receive destination is immutable")
    stage = destination.with_name(f".{destination.name}.{os.getpid()}.staging")
    shards_root = stage / "shards"
    shards_root.mkdir(parents=True)
    manifest_sha = sha256_file(source / "manifest.json")
    auth_sha = sha256_file(source / "launch_authorization.json")

    def download(shard: int) -> tuple[int, dict[str, Any]]:
        padded = f"{shard:03d}"
        shard_dir = shards_root / f"shard-{padded}"
        shard_dir.mkdir()
        prefix = f"gs://{bucket}/runs/{manifest['run_name']}/results/shard-{padded}"
        _run(
            [
                "gcloud",
                "storage",
                "rsync",
                "--recursive",
                prefix,
                str(shard_dir),
                "--project",
                project,
            ],
            timeout=7200,
        )
        return shard, _validate_received_shard(
            shard_dir=shard_dir,
            shard=shard,
            manifest=manifest,
            auth_sha256=auth_sha,
            manifest_sha256=manifest_sha,
        )

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            results = dict(pool.map(download, AUTHORIZED_SHARDS))
        accepted = manifest["step6a_accepted"]
        receipt = {
            "schema": STEP6B_RECEIVE_SCHEMA,
            "status": "pass",
            "run_name": manifest["run_name"],
            "source_sha256": manifest["source_sha256"],
            "manifest_sha256": manifest_sha,
            "schedule_sha256": manifest["schedule_sha256"],
            "authorization_sha256": auth_sha,
            "native_library_sha256": EXPECTED_NATIVE_LIBRARY_SHA256,
            "feature_encoder_sha256": EXPECTED_FEATURE_ENCODER_SHA256,
            "step6a_done_sha256": accepted["done_sha256"],
            "step6a_summary_sha256": accepted["summary_sha256"],
            "per_shard": {
                f"{shard:03d}": results[shard] for shard in AUTHORIZED_SHARDS
            },
            "all_shards_received": True,
            "training_eligible": False,
            "remaining_canary_shards_authorized": True,
            "production_fanout_authorized": False,
            "current_profile_changed": False,
            "m31_complete": False,
        }
        _write_once(stage / "receive_receipt.json", receipt)
        os.replace(stage, destination)
        return receipt
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    package = commands.add_parser("package")
    package.add_argument("--run-name", required=True)
    package.add_argument("--run-dir", type=Path, required=True)
    package.add_argument("--repository-root", type=Path, default=_REPO_ROOT)
    package.add_argument("--linux-library", type=Path, required=True)
    package.add_argument("--linux-feature-encoder", type=Path, required=True)
    package.add_argument("--step5-validation", type=Path, required=True)
    package.add_argument("--step6a-run-dir", type=Path, required=True)
    package.add_argument("--step6a-received-dir", type=Path, required=True)
    package.add_argument("--startup", type=Path)
    validate = commands.add_parser("validate")
    validate.add_argument("--run-dir", type=Path, required=True)
    dry = commands.add_parser("record-dry-run")
    dry.add_argument("--run-dir", type=Path, required=True)
    dry.add_argument("--parity-report", type=Path, required=True)
    authorize = commands.add_parser("authorize")
    authorize.add_argument("--run-dir", type=Path, required=True)
    launch = commands.add_parser("launch")
    launch.add_argument("--run-dir", type=Path, required=True)
    launch.add_argument("--shards", type=_parse_shards, required=True)
    launch.add_argument("--project", default=DEFAULT_PROJECT)
    launch.add_argument("--bucket", default=DEFAULT_BUCKET)
    launch.add_argument("--zones", default=",".join(DEFAULT_ZONES))
    status = commands.add_parser("status")
    status.add_argument("--run-dir", type=Path, required=True)
    status.add_argument("--project", default=DEFAULT_PROJECT)
    status.add_argument("--bucket", default=DEFAULT_BUCKET)
    receive = commands.add_parser("receive")
    receive.add_argument("--run-dir", type=Path, required=True)
    receive.add_argument("--output-dir", type=Path, required=True)
    receive.add_argument("--project", default=DEFAULT_PROJECT)
    receive.add_argument("--bucket", default=DEFAULT_BUCKET)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "package":
        result = package_step6b(
            run_name=args.run_name,
            run_dir=args.run_dir,
            repository_root=args.repository_root,
            linux_library=args.linux_library,
            linux_feature_encoder=args.linux_feature_encoder,
            step5_validation=args.step5_validation,
            step6a_run_dir=args.step6a_run_dir,
            step6a_received_dir=args.step6a_received_dir,
            startup=args.startup,
        )
    elif args.command == "validate":
        result = validate_package(args.run_dir)
    elif args.command == "record-dry-run":
        result = record_local_dry_run(
            run_dir=args.run_dir, parity_report=args.parity_report
        )
    elif args.command == "authorize":
        result = authorize_launch(args.run_dir)
    elif args.command == "launch":
        result = launch_shards(
            run_dir=args.run_dir,
            shards=args.shards,
            project=args.project,
            bucket=args.bucket,
            zones=tuple(item for item in args.zones.split(",") if item),
        )
    elif args.command == "status":
        result = cloud_status(
            run_dir=args.run_dir, project=args.project, bucket=args.bucket
        )
    elif args.command == "receive":
        result = receive_shards(
            run_dir=args.run_dir,
            output_dir=args.output_dir,
            project=args.project,
            bucket=args.bucket,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "STEP6B_AUTHORIZATION_SCHEMA",
    "STEP6B_DONE_SCHEMA",
    "STEP6B_DRY_RUN_SCHEMA",
    "STEP6B_RECEIVE_SCHEMA",
    "authorize_launch",
    "build_schedule",
    "cloud_status",
    "launch_shards",
    "package_step6b",
    "receive_shards",
    "record_local_dry_run",
    "validate_launch",
    "validate_package",
]
