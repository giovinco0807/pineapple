"""Immutable package and bounded Spot lifecycle for M3.1 Step 6a shard 0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .run_hu_m31_t3_step6a_shard import (
    PAIRED_HANDS_PER_SHARD,
    ROOTS_PER_SHARD,
    STEP6A_PACKAGE_SCHEMA,
    STEP6A_PARITY_SCHEMA,
    STEP6A_SHARD_SCHEMA,
    STEP6A_SUMMARY_SCHEMA,
    portable_decision_sha256,
)
from .validate_hu_m31_t3_step5_contract import (
    EXPECTED_CANONICAL_CONTRACT_SHA256,
    validate_contract_file,
)


STEP6A_AUTHORIZATION_SCHEMA = "hu_m31_t3_step6a_launch_authorization_v1"
STEP6A_DRY_RUN_SCHEMA = "hu_m31_t3_step6a_local_dry_run_v1"
STEP6A_LAUNCH_SCHEMA = "hu_m31_t3_step6a_launch_v1"
STEP6A_STATUS_SCHEMA = "hu_m31_t3_step6a_cloud_status_v1"
STEP6A_DONE_SCHEMA = "hu_m31_t3_step6a_done_v1"
STEP6A_RECEIVE_SCHEMA = "hu_m31_t3_step6a_receive_v1"
SOURCE_NAME = "ofc_regular_hu_m31_t3_step6a_source.zip"
SCHEDULE_NAME = "shards_manifest.jsonl"
STARTUP_NAME = "startup_hu_m31_t3_step6a.sh"
EXPECTED_IMAGE_NAME = "debian-12-bookworm-v20260609"
EXPECTED_IMAGE_ID = "1449487925682397051"
EXPECTED_IMAGE_SELF_LINK = (
    "https://www.googleapis.com/compute/v1/projects/debian-cloud/global/images/"
    "debian-12-bookworm-v20260609"
)
EXPECTED_MACHINE_TYPE = "c4-standard-16"
DEFAULT_PROJECT = "ofc-solver-485418"
DEFAULT_BUCKET = "pokerhu-ofc-solver-485418-training"
DEFAULT_ZONE = "asia-northeast1-b"
_SAFE_RUN = re.compile(r"^[a-z0-9][a-z0-9-]{2,50}[a-z0-9]$")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_PATHS = (
    "models/opening_stage7_torch_wide.pt",
    "models/turn1_stage6_torch_wide.pt",
    "models/turn2_stage8.pkl",
    "models/turn3_stage6.pkl",
    "models/hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt",
    "models/hu_turn1_stage18_first1801_mc32_hgb_abs_aug3.pkl",
    "models/hu_turn1_stage18_highmc_safe_selector_a_meta_only.pkl",
    "models/hu_turn0_stage3_top60_mc32_1000_hgb_baseline-delta_sew1_aug3.pkl",
    "models/hu_turn0_stage19_safe_selector_highmc_candidate_delta_plus_meta.pkl",
    "models/hu_turn3_stage7_reference_override_cached_rank_wide.pt",
    "models/hu_turn3_stage3_mc32_500k_plus_m8_12_f128_100k_w2_cached_rank_wide.pt",
)
_CONFIG_PATHS = (
    "configs/hu_joint_policy_m31_t3_step5_contract.json",
    "configs/hu_joint_policy_m31_t3_step5_status.json",
    "configs/hu_m43_attempt08_runtime_requirements.txt",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    ).encode()


def _write_once(path: Path, value: Any, *, raw: bool = False) -> None:
    if path.exists():
        raise FileExistsError(f"immutable Step 6a artifact exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value if raw else canonical_bytes(value)
    if not isinstance(payload, bytes):
        raise TypeError("raw Step 6a artifact must be bytes")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_canonical(path: Path, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict) or raw != canonical_bytes(payload):
        raise ValueError(f"{label} is not canonical JSON")
    return payload


def build_schedule(run_name: str) -> list[dict[str, Any]]:
    if not _SAFE_RUN.fullmatch(run_name):
        raise ValueError("Step 6a run name is not a safe bounded GCP identity")
    return [
        {
            "schema": STEP6A_SHARD_SCHEMA,
            "run_name": run_name,
            "shard": shard,
            "global_hand_start": shard * PAIRED_HANDS_PER_SHARD,
            "hand_count": PAIRED_HANDS_PER_SHARD,
            "root_count": ROOTS_PER_SHARD,
            "output_prefix": f"shard-{shard:03d}",
        }
        for shard in range(10)
    ]


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_bytes(dict(row)) for row in rows)


def _copy_file(source: Path, destination: Path) -> dict[str, Any]:
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"unsafe or missing Step 6a package file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
    }


def _zip_tree(root: Path, output: Path) -> None:
    with zipfile.ZipFile(
        output,
        "x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root).as_posix()
            if (
                "\\" in relative
                or "__pycache__" in relative
                or relative.endswith(".pyc")
            ):
                raise ValueError(f"forbidden Step 6a zip entry: {relative}")
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


def _parity_golden(repo_root: Path) -> dict[str, Any]:
    task_path = (
        repo_root
        / "outputs/hu_joint_policy/m31_t3_step4/local1000_v1/tasks/primary_hand_000.json"
    )
    summary_path = (
        repo_root / "outputs/hu_joint_policy/m31_t3_step4/local1000_v1/summary.json"
    )
    task = json.loads(task_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        task.get("all_gates_passed") is not True
        or summary.get("all_gates_passed") is not True
        or len(task.get("rows", [])) != 2
        or [row.get("seat") for row in task["rows"]] != ["first", "second"]
    ):
        raise ValueError("Step 4 parity task changed")
    config = summary["config"]
    return {
        "schema": "hu_m31_t3_step6a_parity_golden_v1",
        "source_task_sha256": sha256_file(task_path),
        "source_summary_sha256": sha256_file(summary_path),
        "source_windows_library_sha256": task["engine"]["library_sha256"],
        "runtime_config": {
            "run_id": config["run_id"],
            "candidate_samples": config["budget"]["candidate_samples"],
            "evaluation_samples": config["budget"]["evaluation_samples"],
            "downstream_t3_samples": config["budget"]["downstream_t3_samples"],
            "downstream_t4_samples": config["budget"]["downstream_t4_samples"],
            "continuation_seed": config["continuation_seed"],
            "candidate_seed": config["candidate_seed"],
            "evaluation_seed": config["evaluation_seed"],
        },
        "rows": [
            {
                "seat": row["seat"],
                "observation": row["observation"],
                "observation_fingerprint": row["observation_fingerprint"],
                "portable_decision_sha256": portable_decision_sha256(row["decision"]),
            }
            for row in task["rows"]
        ],
    }


def package_step6a(
    *,
    run_name: str,
    run_dir: str | Path,
    repository_root: str | Path = _REPO_ROOT,
    linux_library: str | Path,
    linux_feature_encoder: str | Path,
    step5_validation: str | Path,
    startup: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    destination = Path(run_dir).resolve()
    if destination.exists():
        raise FileExistsError("Step 6a package run directory is immutable")
    if not _SAFE_RUN.fullmatch(run_name):
        raise ValueError("Step 6a run name is invalid")
    contract_path = root / "configs/hu_joint_policy_m31_t3_step5_contract.json"
    validation = validate_contract_file(contract_path, repo_root=root)
    if (
        validation.get("status") != "pass"
        or validation.get("spot_canary_authorized") is not True
        or validation.get("production_fanout_authorized") is not False
    ):
        raise ValueError("Step 5 contract does not authorize Step 6a shard 0")
    status_path = root / "configs/hu_joint_policy_m31_t3_step5_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    validation_path = Path(step5_validation).resolve()
    if (
        status.get("status") != "step5_contract_complete"
        or status.get("spot_canary_shard_0_authorized") is not True
        or status.get("spot_remaining_canary_shards_authorized") is not False
        or status.get("spot_production_fanout_authorized") is not False
        or status.get("current_profile_changed") is not False
        or status.get("contract", {}).get("validation_artifact_sha256")
        != sha256_file(validation_path)
    ):
        raise ValueError("Step 5 status or validation artifact changed")
    library = Path(linux_library).resolve()
    if not library.is_file() or library.name != "libofc_hu_m3_engine.so":
        raise ValueError("Step 6a requires the frozen Linux native library")
    feature_encoder = Path(linux_feature_encoder).resolve()
    if (
        not feature_encoder.is_file()
        or feature_encoder.name != "libofc_stage3_feature_encoder.so"
    ):
        raise ValueError("Step 6a requires the frozen Linux Stage3 feature encoder")
    startup_path = Path(startup or root / "scripts" / STARTUP_NAME).resolve()
    if not startup_path.is_file():
        raise ValueError("Step 6a startup script is missing")

    staging = destination.with_name(f".{destination.name}.{os.getpid()}.staging")
    if staging.exists():
        raise FileExistsError("Step 6a package staging already exists")
    package_root = staging / "package_src"
    package_root.mkdir(parents=True)
    try:
        entries: dict[str, dict[str, Any]] = {}
        source_root = root / "src" / "ofc_regular"
        for path in sorted(source_root.rglob("*.py")):
            if "__pycache__" in path.parts or path.is_symlink():
                continue
            relative = "src/" + path.relative_to(root / "src").as_posix()
            entries[relative] = _copy_file(path, package_root / relative)
        for relative in (*_CONFIG_PATHS, *_MODEL_PATHS):
            entries[relative] = _copy_file(root / relative, package_root / relative)
        native_relative = "native/release/libofc_hu_m3_engine.so"
        entries[native_relative] = _copy_file(library, package_root / native_relative)
        feature_relative = "target/release/libofc_stage3_feature_encoder.so"
        entries[feature_relative] = _copy_file(
            feature_encoder, package_root / feature_relative
        )
        validation_relative = "artifacts/step5/contract_validation.json"
        entries[validation_relative] = _copy_file(
            validation_path, package_root / validation_relative
        )
        golden = _parity_golden(root)
        golden_relative = "artifacts/step6a/parity_golden.json"
        _write_once(package_root / golden_relative, golden)
        entries[golden_relative] = {
            "sha256": sha256_file(package_root / golden_relative),
            "bytes": (package_root / golden_relative).stat().st_size,
        }
        source_path = staging / SOURCE_NAME
        _zip_tree(package_root, source_path)
        schedule = build_schedule(run_name)
        schedule_path = staging / SCHEDULE_NAME
        _write_once(schedule_path, _canonical_jsonl(schedule), raw=True)
        shutil.copy2(startup_path, staging / STARTUP_NAME)
        model_hashes = {
            relative: entries[relative]["sha256"] for relative in _MODEL_PATHS
        }
        manifest = {
            "schema": STEP6A_PACKAGE_SCHEMA,
            "status": "packaged_local_no_gcloud",
            "run_name": run_name,
            "total_shards": 10,
            "authorized_shards": [0],
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
            "step5_contract_canonical_sha256": EXPECTED_CANONICAL_CONTRACT_SHA256,
            "step5_status_sha256": sha256_file(status_path),
            "step5_validation_sha256": sha256_file(validation_path),
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
            "models": model_hashes,
            "parity_golden_path": golden_relative,
            "parity_golden_sha256": entries[golden_relative]["sha256"],
            "source_entries": entries,
            "source_entry_count": len(entries),
            "machine_type": EXPECTED_MACHINE_TYPE,
            "image_name": EXPECTED_IMAGE_NAME,
            "image_id": EXPECTED_IMAGE_ID,
            "checkpoint_unit": "completed_paired_hand",
            "heartbeat_interval_seconds": 60,
            "resume_drill_required": True,
            "canary_rows_training_eligible": False,
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
    manifest = _load_canonical(target / "manifest.json", "Step 6a manifest")
    source = target / SOURCE_NAME
    schedule = target / SCHEDULE_NAME
    startup = target / STARTUP_NAME
    if (
        manifest.get("schema") != STEP6A_PACKAGE_SCHEMA
        or manifest.get("status") != "packaged_local_no_gcloud"
        or manifest.get("total_shards") != 10
        or manifest.get("authorized_shards") != [0]
        or manifest.get("source_sha256") != sha256_file(source)
        or manifest.get("schedule_sha256") != sha256_file(schedule)
        or manifest.get("startup_sha256") != sha256_file(startup)
        or manifest.get("machine_type") != EXPECTED_MACHINE_TYPE
        or manifest.get("image_id") != EXPECTED_IMAGE_ID
        or manifest.get("gcloud_invoked") is not False
        or manifest.get("spot_vm_started") is not False
        or manifest.get("production_fanout_authorized") is not False
        or manifest.get("current_profile_changed") is not False
    ):
        raise ValueError("Step 6a package manifest changed")
    rows = [
        json.loads(line) for line in schedule.read_text(encoding="utf-8").splitlines()
    ]
    if rows != build_schedule(str(manifest["run_name"])):
        raise ValueError("Step 6a schedule changed")
    entries = manifest.get("source_entries")
    if not isinstance(entries, Mapping):
        raise ValueError("Step 6a source entry manifest missing")
    package_root = target / "package_src"
    with zipfile.ZipFile(source) as archive:
        names = tuple(sorted(archive.namelist()))
        if names != tuple(sorted(entries)) or any("\\" in name for name in names):
            raise ValueError("Step 6a zip entry set changed")
        for name, record in entries.items():
            if not isinstance(record, Mapping):
                raise ValueError("Step 6a source entry record changed")
            data = archive.read(name)
            digest = hashlib.sha256(data).hexdigest()
            if (
                digest != record.get("sha256")
                or len(data) != record.get("bytes")
                or sha256_file(package_root / name) != digest
            ):
                raise ValueError(f"Step 6a source content changed: {name}")
    return manifest


def record_local_dry_run(
    *, run_dir: str | Path, parity_report: str | Path, output: str | Path | None = None
) -> dict[str, Any]:
    target = Path(run_dir).resolve()
    manifest = validate_package(target)
    parity_path = Path(parity_report).resolve()
    parity = _load_canonical(parity_path, "Step 6a local parity")
    if (
        parity.get("schema") != STEP6A_PARITY_SCHEMA
        or parity.get("all_gates_passed") is not True
        or parity.get("native_library_sha256") != manifest["native_library"]["sha256"]
        or parity.get("teacher_generation_started") is not False
    ):
        raise ValueError("Step 6a local Linux dry-run parity failed")
    receipt = {
        "schema": STEP6A_DRY_RUN_SCHEMA,
        "status": "pass",
        "run_name": manifest["run_name"],
        "manifest_sha256": sha256_file(target / "manifest.json"),
        "source_sha256": manifest["source_sha256"],
        "native_library_sha256": manifest["native_library"]["sha256"],
        "parity_report_sha256": sha256_file(parity_path),
        "linux_portable_parity": True,
        "package_validation": True,
        "teacher_generation_started": False,
        "gcloud_invoked": False,
        "current_profile_changed": False,
    }
    _write_once(Path(output or target / "local_dry_run_receipt.json"), receipt)
    return receipt


def authorize_launch(run_dir: str | Path) -> dict[str, Any]:
    target = Path(run_dir).resolve()
    manifest = validate_package(target)
    receipt = _load_canonical(
        target / "local_dry_run_receipt.json", "Step 6a dry-run receipt"
    )
    if (
        receipt.get("schema") != STEP6A_DRY_RUN_SCHEMA
        or receipt.get("status") != "pass"
        or receipt.get("manifest_sha256") != sha256_file(target / "manifest.json")
        or receipt.get("source_sha256") != manifest["source_sha256"]
        or receipt.get("teacher_generation_started") is not False
        or receipt.get("gcloud_invoked") is not False
    ):
        raise ValueError("Step 6a dry-run receipt changed")
    authorization = {
        "schema": STEP6A_AUTHORIZATION_SCHEMA,
        "status": "authorized",
        "run_name": manifest["run_name"],
        "manifest_sha256": sha256_file(target / "manifest.json"),
        "source_sha256": manifest["source_sha256"],
        "schedule_sha256": manifest["schedule_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "dry_run_receipt_sha256": sha256_file(target / "local_dry_run_receipt.json"),
        "authorized_shards": [0],
        "spot_authorized": True,
        "remaining_canary_shards_authorized": False,
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
        target / "launch_authorization.json", "Step 6a authorization"
    )
    timestamp = auth.get("authorized_unix_seconds")
    if (
        auth.get("schema") != STEP6A_AUTHORIZATION_SCHEMA
        or auth.get("status") != "authorized"
        or auth.get("run_name") != manifest["run_name"]
        or auth.get("manifest_sha256") != sha256_file(target / "manifest.json")
        or auth.get("source_sha256") != manifest["source_sha256"]
        or auth.get("authorized_shards") != [0]
        or auth.get("spot_authorized") is not True
        or auth.get("remaining_canary_shards_authorized") is not False
        or auth.get("production_fanout_authorized") is not False
        or auth.get("root_execution_started") is not False
        or auth.get("current_profile_changed") is not False
        or isinstance(timestamp, bool)
        or not isinstance(timestamp, (int, float))
        or not math.isfinite(float(timestamp))
    ):
        raise ValueError("Step 6a launch authorization changed")
    return manifest, auth


def _subprocess_run(
    command: Sequence[str], **kwargs: Any
) -> subprocess.CompletedProcess[str]:
    arguments = list(command)
    try:
        return subprocess.run(arguments, **kwargs)
    except FileNotFoundError:
        if not arguments or arguments[0] != "gcloud":
            raise
        executable = shutil.which("gcloud")
        if executable is None:
            raise
        arguments[0] = executable
        return subprocess.run(arguments, **kwargs)


def _run(
    command: Sequence[str], *, timeout: int = 300
) -> subprocess.CompletedProcess[str]:
    result = _subprocess_run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return result


def _publish_once(source: Path, uri: str, *, project: str) -> None:
    result = _subprocess_run(
        [
            "gcloud",
            "storage",
            "cp",
            str(source),
            uri,
            "--project",
            project,
            "--if-generation-match=0",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1200,
        check=False,
    )
    if result.returncode == 0:
        return
    with tempfile.TemporaryDirectory() as temporary:
        copy = Path(temporary) / source.name
        _run(
            ["gcloud", "storage", "cp", uri, str(copy), "--project", project],
            timeout=1200,
        )
        if sha256_file(copy) != sha256_file(source):
            raise FileExistsError(f"immutable GCS object differs: {uri}")


def launch_shard0(
    *,
    run_dir: str | Path,
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
    zone: str = DEFAULT_ZONE,
) -> dict[str, Any]:
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
        raise ValueError("Step 6a immutable Debian image changed")
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
    done_uri = f"{prefix}/results/shard-000/DONE.json"
    exists = _subprocess_run(
        ["gcloud", "storage", "objects", "describe", done_uri, "--project", project],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    if exists.returncode == 0:
        raise FileExistsError("Step 6a shard 0 DONE already exists")
    instance = f"{manifest['run_name']}-s000"
    absent = _subprocess_run(
        [
            "gcloud",
            "compute",
            "instances",
            "describe",
            instance,
            "--project",
            project,
            "--zone",
            zone,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    if absent.returncode == 0:
        raise FileExistsError(f"Step 6a instance already exists: {instance}")
    metadata = ",".join(
        (
            f"PROJECT_ID={project}",
            f"BUCKET={bucket}",
            f"RUN_NAME={manifest['run_name']}",
            "SHARD=0",
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
    return {
        "schema": STEP6A_LAUNCH_SCHEMA,
        "status": "created",
        "run_name": manifest["run_name"],
        "shard": 0,
        "instance": instance,
        "machine_type": EXPECTED_MACHINE_TYPE,
        "provisioning_model": "SPOT",
        "remaining_canary_shards_authorized": False,
        "production_fanout_authorized": False,
        "current_profile_changed": False,
    }


def cloud_status(
    *,
    run_dir: str | Path,
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
    zone: str = DEFAULT_ZONE,
) -> dict[str, Any]:
    manifest = validate_package(run_dir)
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
    instance = f"{manifest['run_name']}-s000"
    instance_result = _subprocess_run(
        [
            "gcloud",
            "compute",
            "instances",
            "describe",
            instance,
            "--project",
            project,
            "--zone",
            zone,
            "--format=json",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    vm = json.loads(instance_result.stdout) if instance_result.returncode == 0 else None
    return {
        "schema": STEP6A_STATUS_SCHEMA,
        "run_name": manifest["run_name"],
        "done": any(uri.endswith("/results/shard-000/DONE.json") for uri in objects),
        "heartbeat_present": any(
            "/progress/shard-000/heartbeat.json" in uri for uri in objects
        ),
        "progress_object_count": sum("/progress/shard-000/" in uri for uri in objects),
        "result_object_count": sum("/results/shard-000/" in uri for uri in objects),
        "instance": (
            None if vm is None else {"name": vm.get("name"), "status": vm.get("status")}
        ),
        "production_fanout_authorized": False,
        "current_profile_changed": False,
    }


def receive_shard0(
    *,
    run_dir: str | Path,
    output_dir: str | Path,
    project: str = DEFAULT_PROJECT,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any]:
    source = Path(run_dir).resolve()
    manifest, auth = validate_launch(source)
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError("Step 6a receive destination is immutable")
    stage = destination.with_name(f".{destination.name}.{os.getpid()}.staging")
    stage.mkdir(parents=True)
    try:
        prefix = f"gs://{bucket}/runs/{manifest['run_name']}/results/shard-000"
        _run(
            [
                "gcloud",
                "storage",
                "rsync",
                "--recursive",
                prefix,
                str(stage),
                "--project",
                project,
            ],
            timeout=7200,
        )
        done = _load_canonical(stage / "DONE.json", "Step 6a DONE")
        summary = _load_canonical(stage / "summary.json", "Step 6a summary")
        if (
            done.get("schema") != STEP6A_DONE_SCHEMA
            or done.get("status") != "complete"
            or done.get("run_name") != manifest["run_name"]
            or done.get("source_sha256") != manifest["source_sha256"]
            or done.get("manifest_sha256") != auth["manifest_sha256"]
            or done.get("authorization_sha256")
            != sha256_file(source / "launch_authorization.json")
            or summary.get("schema") != STEP6A_SUMMARY_SCHEMA
            or summary.get("all_gates_passed") is not True
            or summary.get("resumed_task_count", 0) < 1
            or summary.get("training_eligible") is not False
            or summary.get("production_fanout_authorized") is not False
            or summary.get("current_profile_changed") is not False
        ):
            raise ValueError("Step 6a received boundary failed")
        files = done.get("files")
        if not isinstance(files, Mapping):
            raise ValueError("Step 6a DONE file manifest missing")
        for relative, record in files.items():
            path = stage / str(relative)
            if (
                not path.is_file()
                or not isinstance(record, Mapping)
                or sha256_file(path) != record.get("sha256")
                or path.stat().st_size != record.get("bytes")
            ):
                raise ValueError(f"Step 6a received file changed: {relative}")
        task_files = list((stage / "tasks").glob("hand_*.json"))
        root_files = list((stage / "roots").glob("hand_*.json"))
        if len(task_files) != 25 or len(root_files) != 25:
            raise ValueError("Step 6a received task set is incomplete")
        receipt = {
            "schema": STEP6A_RECEIVE_SCHEMA,
            "status": "pass",
            "run_name": manifest["run_name"],
            "source_sha256": manifest["source_sha256"],
            "manifest_sha256": auth["manifest_sha256"],
            "authorization_sha256": sha256_file(source / "launch_authorization.json"),
            "done_sha256": sha256_file(stage / "DONE.json"),
            "summary_sha256": sha256_file(stage / "summary.json"),
            "task_count": len(task_files),
            "root_task_count": len(root_files),
            "all_gates_passed": True,
            "training_eligible": False,
            "remaining_canary_shards_authorized": False,
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
    launch.add_argument("--project", default=DEFAULT_PROJECT)
    launch.add_argument("--bucket", default=DEFAULT_BUCKET)
    launch.add_argument("--zone", default=DEFAULT_ZONE)
    status = commands.add_parser("status")
    status.add_argument("--run-dir", type=Path, required=True)
    status.add_argument("--project", default=DEFAULT_PROJECT)
    status.add_argument("--bucket", default=DEFAULT_BUCKET)
    status.add_argument("--zone", default=DEFAULT_ZONE)
    receive = commands.add_parser("receive")
    receive.add_argument("--run-dir", type=Path, required=True)
    receive.add_argument("--output-dir", type=Path, required=True)
    receive.add_argument("--project", default=DEFAULT_PROJECT)
    receive.add_argument("--bucket", default=DEFAULT_BUCKET)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "package":
        result = package_step6a(
            run_name=args.run_name,
            run_dir=args.run_dir,
            repository_root=args.repository_root,
            linux_library=args.linux_library,
            linux_feature_encoder=args.linux_feature_encoder,
            step5_validation=args.step5_validation,
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
        result = launch_shard0(
            run_dir=args.run_dir,
            project=args.project,
            bucket=args.bucket,
            zone=args.zone,
        )
    elif args.command == "status":
        result = cloud_status(
            run_dir=args.run_dir,
            project=args.project,
            bucket=args.bucket,
            zone=args.zone,
        )
    elif args.command == "receive":
        result = receive_shard0(
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
    "STEP6A_AUTHORIZATION_SCHEMA",
    "STEP6A_DONE_SCHEMA",
    "STEP6A_DRY_RUN_SCHEMA",
    "STEP6A_RECEIVE_SCHEMA",
    "authorize_launch",
    "build_schedule",
    "cloud_status",
    "launch_shard0",
    "package_step6a",
    "receive_shard0",
    "record_local_dry_run",
    "validate_launch",
    "validate_package",
]
