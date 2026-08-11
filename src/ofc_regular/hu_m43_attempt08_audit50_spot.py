"""Strict Spot/receive lifecycle for the Attempt08 audit50 overlay.

The development package is an immutable dependency, never an editable base.
PackageOnly freezes a small, separately hashed overlay.  Audit roots remain
closed until a development-GO freeze, an audit-open authorization, and a
separate Spot launch authorization have all been validated.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .hu_m43_attempt08_audit50_contract import (
        AI_PROFILES_SHA256,
        AUDIT50_PLAN_SHA256,
        DEVELOPMENT_FUTURE_RUNNER_SHA256,
        GCP_IMAGE_ID,
        GCP_IMAGE_NAME,
        GCP_IMAGE_SELF_LINK,
        LAMBDA_MODEL_SHA256,
        MACHINE_TYPE,
        MAX_WAVE_SHARDS,
        NATIVE_BATCH_THREADS,
        PROFILES,
        ROOT_FIRST,
        ROOT_LAST,
        RUN_NAME_RE,
        SOURCE_PLAN_SHA256,
        TOTAL_SHARDS,
        atomic_create,
        build_audit50_schedule,
        canonical_json_bytes,
        canonical_sha256,
        load_and_validate_audit50_plan,
        load_json_mapping,
        require_sha256,
        schedule_bytes,
        sha256_file,
        validate_schedule_bytes,
        write_canonical_json,
    )
except ImportError:
    from hu_m43_attempt08_audit50_contract import (  # type: ignore
        AI_PROFILES_SHA256,
        AUDIT50_PLAN_SHA256,
        DEVELOPMENT_FUTURE_RUNNER_SHA256,
        GCP_IMAGE_ID,
        GCP_IMAGE_NAME,
        GCP_IMAGE_SELF_LINK,
        LAMBDA_MODEL_SHA256,
        MACHINE_TYPE,
        MAX_WAVE_SHARDS,
        NATIVE_BATCH_THREADS,
        PROFILES,
        ROOT_FIRST,
        ROOT_LAST,
        RUN_NAME_RE,
        SOURCE_PLAN_SHA256,
        TOTAL_SHARDS,
        atomic_create,
        build_audit50_schedule,
        canonical_json_bytes,
        canonical_sha256,
        load_and_validate_audit50_plan,
        load_json_mapping,
        require_sha256,
        schedule_bytes,
        sha256_file,
        validate_schedule_bytes,
        write_canonical_json,
    )


PACKAGE_SCHEMA = "hu_m43_attempt08_audit50_spot_package_v1"
PACKAGE_RESULT_SCHEMA = "hu_m43_attempt08_audit50_spot_package_result_v1"
SOURCE_CLOSURE_SCHEMA = "hu_m43_attempt08_audit50_source_closure_v1"
OUTER_AUTH_SCHEMA = "hu_m43_attempt08_audit50_open_authorization_v1"
CORE_AUTH_SCHEMA = "hu_m43_attempt08_future_audit_open_authorization_v1"
LAUNCH_AUTH_SCHEMA = "hu_m43_attempt08_audit50_launch_authorization_v1"
GLOBAL_CLAIM_SCHEMA = "hu_m43_attempt08_audit50_global_claim_v1"
ROOT_CLAIM_SCHEMA = "hu_m43_attempt08_audit50_root_claim_v1"
CHECKPOINT_SCHEMA = "hu_m43_attempt08_audit50_checkpoint_v1"
HEARTBEAT_SCHEMA = "hu_m43_attempt08_audit50_heartbeat_v1"
SUMMARY_SCHEMA = "hu_m43_attempt08_audit50_summary_v1"
RESUME_COMMIT_SCHEMA = "hu_m43_attempt08_audit50_resume_commit_v1"
DONE_SCHEMA = "hu_m43_attempt08_audit50_spot_done_v1"
CONSUMPTION_SCHEMA = "hu_m43_attempt08_audit50_output_consumption_v1"
RECEIVED_SCHEMA = "hu_m43_attempt08_audit50_received_shard_v1"
MERGE_SCHEMA = "hu_m43_attempt08_audit50_receive_merge_v1"
SELECTOR_CLAIM_SCHEMA = "hu_m43_attempt08_audit50_selector_claim_v1"
SELECTOR_EXECUTION_SCHEMA = "hu_m43_attempt08_audit50_selector_execution_v1"
SELECTOR_RECEIPT_SCHEMA = "hu_m43_attempt08_audit50_selector_receipt_v1"
_SEARCH_CORE = (
    "same_attempt08_t1_second_root_generation_lambda_candidate_and_teacher_core"
)
_GLOBAL_PRECLAIM_OPERATIONS = {
    "development_and_audit_package_hashes_validated": True,
    "launch_authorization_validated": True,
    "audit_shard_schedule_validated": True,
    "candidate_model_artifact_sha256_validated": True,
    "candidate_model_deserialized": False,
    "candidate_model_inference_performed": False,
    "audit_observation_generated": False,
    "audit_teacher_executed": False,
    "audit_result_content_read": False,
}
_ROOT_PRECLAIM_OPERATIONS = {
    **_GLOBAL_PRECLAIM_OPERATIONS,
    "root_spec_derived_from_frozen_schedule": True,
}

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_PLAN = _REPO_ROOT / "configs/hu_joint_policy_m43_attempt08_audit50.json"
OVERLAY_FILES = (
    "src/ofc_regular/hu_m43_attempt08_audit50_contract.py",
    "src/ofc_regular/hu_m43_attempt08_audit50_spot.py",
    "src/ofc_regular/select_hu_m43_attempt08_audit50.py",
    "configs/hu_joint_policy_m43_attempt08_audit50.json",
    "scripts/HuM43Attempt08Audit50Spot.Common.ps1",
    "scripts/Start-GcpHuM43Attempt08Audit50Run.ps1",
    "scripts/Get-GcpHuM43Attempt08Audit50RunStatus.ps1",
    "scripts/Receive-GcpHuM43Attempt08Audit50Run.ps1",
    "scripts/startup_hu_m43_attempt08_audit50.sh",
)
_CORE_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "development_pass_freeze_sha256",
        "package_manifest_sha256",
        "launch_authorization_sha256",
        "development_decision_sha256",
        "selector_receipt_sha256",
        "development_open_authorization_sha256",
        "plan_sha256",
        "model_sha256",
        "ai_profiles_sha256",
        "source_closure_sha256",
        "source_zip_sha256",
        "runtime_semantic_anchor_sha256",
        "population",
        "root_index_first",
        "root_index_last",
        "total_roots",
        "search_core",
        "development_passed",
        "future_audit_authorized",
        "fit_performed",
        "threshold_selected",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
)


def _run_frozen_python(
    development_run_dir: Path,
    arguments: Sequence[str],
    *,
    timeout: int = 3600,
) -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(development_run_dir / "package_src/src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-B", *arguments],
        cwd=development_run_dir / "package_src",
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            "frozen Attempt08 command failed: "
            + (completed.stderr or completed.stdout).strip()
        )
    return completed.stdout


def _validate_development_package(development_run_dir: Path) -> dict[str, Any]:
    manifest = load_json_mapping(
        development_run_dir / "manifest.json", "frozen development manifest"
    )
    launch = development_run_dir / "launch_authorization.json"
    _run_frozen_python(
        development_run_dir,
        (
            "-m",
            "ofc_regular.hu_m43_attempt08_spot",
            "validate-launch",
            "--run-dir",
            str(development_run_dir),
            "--authorization",
            str(launch),
        ),
    )
    runner = (
        development_run_dir
        / "package_src/src/ofc_regular/run_hu_m43_attempt08_future_audit.py"
    )
    if sha256_file(runner) != DEVELOPMENT_FUTURE_RUNNER_SHA256:
        raise ValueError("frozen Attempt08 future-audit runner changed")
    if (
        manifest.get("schema") != "hu_m43_attempt08_development_spot_package_v1"
        or manifest.get("plan_sha256") != SOURCE_PLAN_SHA256
        or manifest.get("model_sha256") != LAMBDA_MODEL_SHA256
        or manifest.get("ai_profiles_sha256") != AI_PROFILES_SHA256
        or manifest.get("gcp_image_name") != GCP_IMAGE_NAME
        or str(manifest.get("gcp_image_id")) != GCP_IMAGE_ID
        or manifest.get("gcp_image_self_link") != GCP_IMAGE_SELF_LINK
        or manifest.get("current_profile_mutated") is not False
        or manifest.get("runtime_policy_activated") is not False
    ):
        raise ValueError("frozen Attempt08 development package identity changed")
    return manifest


def _tree_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.as_posix()):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _tree_sha256(root: Path) -> str:
    return canonical_sha256({"files": _tree_rows(root)})


def _write_deterministic_zip(root: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.as_posix()):
            name = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


def _validate_zip(root: Path, archive_path: Path) -> None:
    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
    }
    with zipfile.ZipFile(archive_path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(actual):
            raise ValueError("Attempt08 audit50 overlay archive file set changed")
        for name in names:
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError("Attempt08 audit50 archive contains unsafe path")
            if archive.read(name) != actual[name].read_bytes():
                raise ValueError(f"Attempt08 audit50 archive bytes changed: {name}")


def _copy_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite overlay file: {destination}")
    shutil.copyfile(source, destination)


def package_audit50(
    *,
    repo_root: str | Path,
    run_dir: str | Path,
    run_name: str,
    development_run_dir: str | Path,
    audit_plan_path: str | Path,
    resume_existing: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    destination = Path(run_dir).resolve()
    development = Path(development_run_dir).resolve()
    if RUN_NAME_RE.fullmatch(run_name) is None:
        raise ValueError("Attempt08 audit50 RunName is not a safe GCP identity")
    expected = (root / "outputs/gcp_runs" / run_name).resolve()
    if os.path.normcase(str(destination)) != os.path.normcase(str(expected)):
        raise ValueError("Attempt08 audit50 run directory must be outputs/gcp_runs/<run>")
    if destination.exists():
        if not resume_existing:
            raise FileExistsError(f"Attempt08 audit50 run directory exists: {destination}")
        manifest = validate_package(destination, development_run_dir=development)
        return _package_result(destination, manifest, resumed=True)
    audit_plan = Path(audit_plan_path).resolve()
    load_and_validate_audit50_plan(audit_plan)
    dev_manifest = _validate_development_package(development)
    dev_manifest_path = development / "manifest.json"
    dev_launch_path = development / "launch_authorization.json"
    staging = destination.with_name(destination.name + ".building")
    if staging.exists():
        raise FileExistsError(f"stale Attempt08 audit50 staging directory: {staging}")
    staging.mkdir(parents=True)
    try:
        overlay = staging / "overlay_src"
        for relative in OVERLAY_FILES:
            source = root / relative
            if not source.is_file():
                raise ValueError(f"Attempt08 audit50 overlay file is missing: {relative}")
            _copy_exact(source, overlay / relative)
        schedule_path = staging / "shards_manifest.jsonl"
        atomic_create(schedule_path, schedule_bytes(build_audit50_schedule()))
        _copy_exact(audit_plan, staging / "hu_joint_policy_m43_attempt08_audit50.json")
        startup = root / "scripts/startup_hu_m43_attempt08_audit50.sh"
        _copy_exact(startup, staging / startup.name)
        closure = {
            "schema": SOURCE_CLOSURE_SCHEMA,
            "status": "overlay_closed_without_audit_authorization_or_root",
            "run_name": run_name,
            "development_run_name": dev_manifest["run_name"],
            "development_manifest_sha256": sha256_file(dev_manifest_path),
            "development_launch_authorization_sha256": sha256_file(dev_launch_path),
            "development_source_zip_sha256": dev_manifest["source_zip_sha256"],
            "development_package_tree_sha256": _tree_sha256(
                development / "package_src"
            ),
            "development_future_runner_sha256": DEVELOPMENT_FUTURE_RUNNER_SHA256,
            "audit_plan_sha256": AUDIT50_PLAN_SHA256,
            "files": _tree_rows(overlay),
            "audit_authorized": False,
            "audit_started": False,
            "teacher_executed": False,
            "selector_executed": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        }
        write_canonical_json(staging / "source_closure_manifest.json", closure)
        source_zip = staging / "ofc_regular_hu_m43_attempt08_audit50_overlay.zip"
        _write_deterministic_zip(overlay, source_zip)
        manifest = {
            "schema": PACKAGE_SCHEMA,
            "status": "frozen_overlay_package_only_no_audit_root_opened",
            "run_name": run_name,
            "development_run_name": dev_manifest["run_name"],
            "development_manifest_sha256": sha256_file(dev_manifest_path),
            "development_launch_authorization_sha256": sha256_file(dev_launch_path),
            "development_schedule_sha256": dev_manifest["schedule_sha256"],
            "development_source_closure_sha256": dev_manifest[
                "source_closure_sha256"
            ],
            "development_source_zip_sha256": dev_manifest["source_zip_sha256"],
            "development_startup_sha256": dev_manifest["startup_sha256"],
            "development_package_tree_sha256": closure[
                "development_package_tree_sha256"
            ],
            "development_future_runner_sha256": DEVELOPMENT_FUTURE_RUNNER_SHA256,
            "development_plan_sha256": SOURCE_PLAN_SHA256,
            "model_sha256": LAMBDA_MODEL_SHA256,
            "ai_profiles_sha256": AI_PROFILES_SHA256,
            "runtime_semantic_anchor_sha256": dev_manifest[
                "runtime_semantic_anchor_sha256"
            ],
            "runtime_source_closure_sha256": dev_manifest[
                "runtime_source_closure_sha256"
            ],
            "runtime_fingerprint_sha256": dev_manifest[
                "runtime_fingerprint_sha256"
            ],
            "runtime_requirements_sha256": dev_manifest[
                "runtime_requirements_sha256"
            ],
            "audit_plan_sha256": AUDIT50_PLAN_SHA256,
            "schedule_sha256": sha256_file(schedule_path),
            "overlay_source_closure_sha256": sha256_file(
                staging / "source_closure_manifest.json"
            ),
            "overlay_source_zip_sha256": sha256_file(source_zip),
            "startup_sha256": sha256_file(staging / startup.name),
            "gcp_image_name": GCP_IMAGE_NAME,
            "gcp_image_id": GCP_IMAGE_ID,
            "gcp_image_self_link": GCP_IMAGE_SELF_LINK,
            "machine_type": MACHINE_TYPE,
            "native_batch_threads": NATIVE_BATCH_THREADS,
            "max_wave_shards": MAX_WAVE_SHARDS,
            "root_index_first": ROOT_FIRST,
            "root_index_last": ROOT_LAST,
            "total_roots": TOTAL_SHARDS,
            "total_shards": TOTAL_SHARDS,
            "roots_per_shard": 1,
            "roots_per_profile": 10,
            "audit_authorized": False,
            "audit_started": False,
            "seed_material_opened": False,
            "teacher_executed": False,
            "gcloud_invoked": False,
            "instances_created": False,
            "selector_executed": False,
            "fit_performed": False,
            "threshold_selected": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        }
        write_canonical_json(staging / "manifest.json", manifest)
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
        "status": "packaged_without_audit_authorization_root_or_gcloud",
        "run_name": manifest["run_name"],
        "run_dir": str(run_dir),
        "manifest_sha256": sha256_file(run_dir / "manifest.json"),
        "total_shards": TOTAL_SHARDS,
        "resumed_existing_package": resumed,
        "audit_authorized": False,
        "fresh_root_opened": False,
        "teacher_executed": False,
        "gcloud_invoked": False,
        "instances_created": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }


def validate_package(
    run_dir: str | Path, *, development_run_dir: str | Path | None = None
) -> dict[str, Any]:
    root = Path(run_dir)
    manifest = load_json_mapping(root / "manifest.json", "audit50 manifest")
    if (
        manifest.get("schema") != PACKAGE_SCHEMA
        or manifest.get("status")
        != "frozen_overlay_package_only_no_audit_root_opened"
        or manifest.get("audit_plan_sha256") != AUDIT50_PLAN_SHA256
        or manifest.get("development_plan_sha256") != SOURCE_PLAN_SHA256
        or manifest.get("model_sha256") != LAMBDA_MODEL_SHA256
        or manifest.get("ai_profiles_sha256") != AI_PROFILES_SHA256
        or manifest.get("development_future_runner_sha256")
        != DEVELOPMENT_FUTURE_RUNNER_SHA256
        or manifest.get("gcp_image_name") != GCP_IMAGE_NAME
        or str(manifest.get("gcp_image_id")) != GCP_IMAGE_ID
        or manifest.get("gcp_image_self_link") != GCP_IMAGE_SELF_LINK
        or manifest.get("machine_type") != MACHINE_TYPE
        or manifest.get("native_batch_threads") != NATIVE_BATCH_THREADS
        or manifest.get("max_wave_shards") != MAX_WAVE_SHARDS
        or manifest.get("root_index_first") != ROOT_FIRST
        or manifest.get("root_index_last") != ROOT_LAST
        or manifest.get("total_roots") != TOTAL_SHARDS
        or manifest.get("total_shards") != TOTAL_SHARDS
        or manifest.get("roots_per_shard") != 1
        or manifest.get("roots_per_profile") != 10
        or any(
            manifest.get(field) is not False
            for field in (
                "audit_authorized",
                "audit_started",
                "seed_material_opened",
                "teacher_executed",
                "gcloud_invoked",
                "instances_created",
                "selector_executed",
                "fit_performed",
                "threshold_selected",
                "current_profile_mutated",
                "runtime_policy_activated",
            )
        )
    ):
        raise ValueError("Attempt08 audit50 package manifest changed")
    for name, field in (
        ("hu_joint_policy_m43_attempt08_audit50.json", "audit_plan_sha256"),
        ("shards_manifest.jsonl", "schedule_sha256"),
        ("source_closure_manifest.json", "overlay_source_closure_sha256"),
        ("ofc_regular_hu_m43_attempt08_audit50_overlay.zip", "overlay_source_zip_sha256"),
        ("startup_hu_m43_attempt08_audit50.sh", "startup_sha256"),
    ):
        if sha256_file(root / name) != manifest[field]:
            raise ValueError(f"Attempt08 audit50 package binding changed: {name}")
    load_and_validate_audit50_plan(root / "hu_joint_policy_m43_attempt08_audit50.json")
    validate_schedule_bytes((root / "shards_manifest.jsonl").read_bytes())
    closure = load_json_mapping(
        root / "source_closure_manifest.json", "audit50 source closure"
    )
    overlay = root / "overlay_src"
    if (
        closure.get("schema") != SOURCE_CLOSURE_SCHEMA
        or closure.get("run_name") != manifest.get("run_name")
        or closure.get("development_run_name")
        != manifest.get("development_run_name")
        or closure.get("audit_plan_sha256") != AUDIT50_PLAN_SHA256
        or closure.get("files") != _tree_rows(overlay)
        or any(
            closure.get(field) is not False
            for field in (
                "audit_authorized",
                "audit_started",
                "teacher_executed",
                "selector_executed",
                "current_profile_mutated",
                "runtime_policy_activated",
            )
        )
    ):
        raise ValueError("Attempt08 audit50 overlay closure changed")
    _validate_zip(
        overlay, root / "ofc_regular_hu_m43_attempt08_audit50_overlay.zip"
    )
    development = (
        Path(development_run_dir).resolve()
        if development_run_dir is not None
        else root.parent / str(manifest["development_run_name"])
    )
    dev_manifest = _validate_development_package(development)
    if (
        dev_manifest.get("run_name") != manifest.get("development_run_name")
        or sha256_file(development / "manifest.json")
        != manifest.get("development_manifest_sha256")
        or sha256_file(development / "launch_authorization.json")
        != manifest.get("development_launch_authorization_sha256")
        or dev_manifest.get("schedule_sha256")
        != manifest.get("development_schedule_sha256")
        or dev_manifest.get("source_closure_sha256")
        != manifest.get("development_source_closure_sha256")
        or dev_manifest.get("source_zip_sha256")
        != manifest.get("development_source_zip_sha256")
        or dev_manifest.get("startup_sha256")
        != manifest.get("development_startup_sha256")
        or _tree_sha256(development / "package_src")
        != manifest.get("development_package_tree_sha256")
    ):
        raise ValueError("Attempt08 frozen development base changed")
    return manifest


def _copy_immutable_input(source: Path, destination: Path) -> None:
    atomic_create(destination, source.read_bytes(), allow_identical=True)


def _validate_core_with_frozen_runner(
    *,
    development_run_dir: Path,
    core_authorization: Path,
    freeze: Path,
    decision: Path,
    receipt: Path,
) -> None:
    code = (
        "import sys;"
        "from ofc_regular.run_hu_m43_attempt08_future_audit import "
        "load_and_validate_future_audit_open_authorization as v;"
        "v(sys.argv[1],development_pass_freeze_path=sys.argv[2],"
        "run_dir=sys.argv[3],launch_authorization_path=sys.argv[4],"
        "development_decision_path=sys.argv[5],selector_receipt_path=sys.argv[6])"
    )
    _run_frozen_python(
        development_run_dir,
        (
            "-c",
            code,
            str(core_authorization),
            str(freeze),
            str(development_run_dir),
            str(development_run_dir / "launch_authorization.json"),
            str(decision),
            str(receipt),
        ),
    )


def authorize_audit50(
    *,
    run_dir: str | Path,
    development_run_dir: str | Path,
    development_pass_freeze_path: str | Path,
    development_decision_path: str | Path,
    selector_receipt_path: str | Path,
    core_output: str | Path,
    outer_output: str | Path,
) -> dict[str, Any]:
    root = Path(run_dir)
    development = Path(development_run_dir).resolve()
    manifest = validate_package(root, development_run_dir=development)
    freeze_target = root / "development_pass_freeze.json"
    decision_target = root / "development_decision.json"
    receipt_target = root / "development_selector_receipt.json"
    freeze_source = Path(development_pass_freeze_path)
    decision_source = Path(development_decision_path)
    receipt_source = Path(selector_receipt_path)
    # Validate the complete GO chain before copying any immutable input into
    # the audit run.  A bad path must not poison an otherwise unopened run.
    freeze = load_json_mapping(freeze_source, "development-pass freeze")
    decision = load_json_mapping(decision_source, "development decision")
    receipt = load_json_mapping(receipt_source, "development selector receipt")
    dev_manifest = load_json_mapping(
        development / "manifest.json", "development manifest"
    )
    if (
        freeze.get("schema") != "hu_m43_attempt08_development_pass_freeze_v1"
        or freeze.get("status")
        != "development_go_frozen_without_future_audit_authorization"
        or freeze.get("run_name") != manifest["development_run_name"]
        or freeze.get("package_manifest_sha256")
        != manifest["development_manifest_sha256"]
        or freeze.get("launch_authorization_sha256")
        != manifest["development_launch_authorization_sha256"]
        or freeze.get("development_decision_sha256") != sha256_file(decision_source)
        or freeze.get("selector_receipt_sha256") != sha256_file(receipt_source)
        or freeze.get("plan_sha256") != SOURCE_PLAN_SHA256
        or freeze.get("model_sha256") != LAMBDA_MODEL_SHA256
        or freeze.get("ai_profiles_sha256") != AI_PROFILES_SHA256
        or freeze.get("search_freeze_authorized") is not True
        or freeze.get("future_audit_authorized") is not False
        or decision.get("decision") != "go"
        or decision.get("status") != "go_write_separate_search_freeze_only"
        or receipt.get("decision") != "go"
        or receipt.get("status") != "single_frozen_gate_evaluation_complete"
        or receipt.get("selector_executed") is not True
        or any(
            payload.get(field) is not False
            for payload in (freeze, decision, receipt)
            for field in (
                "fit_performed",
                "threshold_selected",
                "current_profile_mutated",
                "runtime_policy_activated",
            )
            if field in payload
        )
    ):
        raise ValueError("Attempt08 audit50 requires the exact development GO chain")
    _copy_immutable_input(freeze_source, freeze_target)
    _copy_immutable_input(decision_source, decision_target)
    _copy_immutable_input(receipt_source, receipt_target)
    core = {
        "schema": CORE_AUTH_SCHEMA,
        "status": "separately_authorized_after_immutable_development_pass_freeze",
        "run_name": manifest["development_run_name"],
        "development_pass_freeze_sha256": sha256_file(freeze_target),
        "package_manifest_sha256": manifest["development_manifest_sha256"],
        "launch_authorization_sha256": manifest[
            "development_launch_authorization_sha256"
        ],
        "development_decision_sha256": sha256_file(decision_target),
        "selector_receipt_sha256": sha256_file(receipt_target),
        "development_open_authorization_sha256": dev_manifest[
            "development_open_authorization_sha256"
        ],
        "plan_sha256": SOURCE_PLAN_SHA256,
        "model_sha256": LAMBDA_MODEL_SHA256,
        "ai_profiles_sha256": AI_PROFILES_SHA256,
        "source_closure_sha256": manifest["development_source_closure_sha256"],
        "source_zip_sha256": manifest["development_source_zip_sha256"],
        "runtime_semantic_anchor_sha256": manifest[
            "runtime_semantic_anchor_sha256"
        ],
        "population": "future_audit",
        "root_index_first": ROOT_FIRST,
        "root_index_last": ROOT_LAST,
        "total_roots": TOTAL_SHARDS,
        "search_core": _SEARCH_CORE,
        "development_passed": True,
        "future_audit_authorized": True,
        "fit_performed": False,
        "threshold_selected": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    if set(core) != _CORE_KEYS:
        raise AssertionError("Attempt08 core authorization fields changed")
    core_path = Path(core_output)
    with tempfile.TemporaryDirectory(prefix="attempt08-audit50-core-") as temporary:
        candidate = Path(temporary) / "core_audit_authorization.json"
        write_canonical_json(candidate, core)
        _validate_core_with_frozen_runner(
            development_run_dir=development,
            core_authorization=candidate,
            freeze=freeze_target,
            decision=decision_target,
            receipt=receipt_target,
        )
    write_canonical_json(core_path, core, allow_identical=True)
    for key in (
        "merge_receipt_sha256",
        "merged_sha256",
        "selector_claim_sha256",
        "remote_selector_claim_sha256",
    ):
        require_sha256(receipt.get(key), f"development selector receipt {key}")
    outer = {
        "schema": OUTER_AUTH_SCHEMA,
        "status": "audit50_opened_once_after_immutable_development_go",
        "run_name": manifest["run_name"],
        "development_run_name": manifest["development_run_name"],
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "audit_plan_sha256": AUDIT50_PLAN_SHA256,
        "schedule_sha256": manifest["schedule_sha256"],
        "overlay_source_closure_sha256": manifest[
            "overlay_source_closure_sha256"
        ],
        "overlay_source_zip_sha256": manifest["overlay_source_zip_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "core_authorization_sha256": sha256_file(core_path),
        "development_pass_freeze_sha256": sha256_file(freeze_target),
        "development_decision_sha256": sha256_file(decision_target),
        "development_selector_receipt_sha256": sha256_file(receipt_target),
        "development_merge_receipt_sha256": receipt["merge_receipt_sha256"],
        "development_merged_sha256": receipt["merged_sha256"],
        "development_manifest_sha256": manifest["development_manifest_sha256"],
        "development_launch_authorization_sha256": manifest[
            "development_launch_authorization_sha256"
        ],
        "development_source_closure_sha256": manifest[
            "development_source_closure_sha256"
        ],
        "development_source_zip_sha256": manifest[
            "development_source_zip_sha256"
        ],
        "development_package_tree_sha256": manifest[
            "development_package_tree_sha256"
        ],
        "development_future_runner_sha256": DEVELOPMENT_FUTURE_RUNNER_SHA256,
        "development_plan_sha256": SOURCE_PLAN_SHA256,
        "model_sha256": LAMBDA_MODEL_SHA256,
        "ai_profiles_sha256": AI_PROFILES_SHA256,
        "runtime_semantic_anchor_sha256": manifest[
            "runtime_semantic_anchor_sha256"
        ],
        "runtime_source_closure_sha256": manifest[
            "runtime_source_closure_sha256"
        ],
        "runtime_fingerprint_sha256": manifest["runtime_fingerprint_sha256"],
        "runtime_requirements_sha256": manifest["runtime_requirements_sha256"],
        "gcp_image_name": GCP_IMAGE_NAME,
        "gcp_image_id": GCP_IMAGE_ID,
        "gcp_image_self_link": GCP_IMAGE_SELF_LINK,
        "machine_type": MACHINE_TYPE,
        "root_index_first": ROOT_FIRST,
        "root_index_last": ROOT_LAST,
        "total_roots": TOTAL_SHARDS,
        "total_shards": TOTAL_SHARDS,
        "roots_per_profile": 10,
        "development_passed": True,
        "audit_authorized": True,
        "audit_started": False,
        "fit_performed": False,
        "threshold_selected": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    write_canonical_json(outer_output, outer, allow_identical=True)
    validate_audit_authorization(
        run_dir=root, development_run_dir=development, outer_path=outer_output
    )
    return outer


def validate_audit_authorization(
    *,
    run_dir: str | Path,
    development_run_dir: str | Path,
    outer_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    development = Path(development_run_dir).resolve()
    manifest = validate_package(root, development_run_dir=development)
    outer_file = Path(outer_path) if outer_path else root / "audit_open_authorization.json"
    core_file = root / "core_audit_authorization.json"
    freeze = root / "development_pass_freeze.json"
    decision = root / "development_decision.json"
    receipt = root / "development_selector_receipt.json"
    outer = load_json_mapping(outer_file, "audit50 outer authorization")
    core = load_json_mapping(core_file, "audit50 core authorization")
    selector_receipt = load_json_mapping(receipt, "development selector receipt")
    if set(core) != _CORE_KEYS:
        raise ValueError("Attempt08 audit50 core authorization fields changed")
    _validate_core_with_frozen_runner(
        development_run_dir=development,
        core_authorization=core_file,
        freeze=freeze,
        decision=decision,
        receipt=receipt,
    )
    expected = {
        "schema": OUTER_AUTH_SCHEMA,
        "status": "audit50_opened_once_after_immutable_development_go",
        "run_name": manifest["run_name"],
        "development_run_name": manifest["development_run_name"],
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "audit_plan_sha256": AUDIT50_PLAN_SHA256,
        "schedule_sha256": manifest["schedule_sha256"],
        "overlay_source_closure_sha256": manifest["overlay_source_closure_sha256"],
        "overlay_source_zip_sha256": manifest["overlay_source_zip_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "core_authorization_sha256": sha256_file(core_file),
        "development_pass_freeze_sha256": sha256_file(freeze),
        "development_decision_sha256": sha256_file(decision),
        "development_selector_receipt_sha256": sha256_file(receipt),
        "development_merge_receipt_sha256": selector_receipt["merge_receipt_sha256"],
        "development_merged_sha256": selector_receipt["merged_sha256"],
        "development_manifest_sha256": manifest["development_manifest_sha256"],
        "development_launch_authorization_sha256": manifest[
            "development_launch_authorization_sha256"
        ],
        "development_source_closure_sha256": manifest[
            "development_source_closure_sha256"
        ],
        "development_source_zip_sha256": manifest["development_source_zip_sha256"],
        "development_package_tree_sha256": manifest[
            "development_package_tree_sha256"
        ],
        "development_future_runner_sha256": DEVELOPMENT_FUTURE_RUNNER_SHA256,
        "development_plan_sha256": SOURCE_PLAN_SHA256,
        "model_sha256": LAMBDA_MODEL_SHA256,
        "ai_profiles_sha256": AI_PROFILES_SHA256,
        "runtime_semantic_anchor_sha256": manifest[
            "runtime_semantic_anchor_sha256"
        ],
        "runtime_source_closure_sha256": manifest[
            "runtime_source_closure_sha256"
        ],
        "runtime_fingerprint_sha256": manifest["runtime_fingerprint_sha256"],
        "runtime_requirements_sha256": manifest["runtime_requirements_sha256"],
        "gcp_image_name": GCP_IMAGE_NAME,
        "gcp_image_id": GCP_IMAGE_ID,
        "gcp_image_self_link": GCP_IMAGE_SELF_LINK,
        "machine_type": MACHINE_TYPE,
        "root_index_first": ROOT_FIRST,
        "root_index_last": ROOT_LAST,
        "total_roots": TOTAL_SHARDS,
        "total_shards": TOTAL_SHARDS,
        "roots_per_profile": 10,
        "development_passed": True,
        "audit_authorized": True,
        "audit_started": False,
        "fit_performed": False,
        "threshold_selected": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    if outer != expected:
        raise ValueError("Attempt08 audit50 outer authorization changed")
    return outer


def authorize_launch(
    *, run_dir: str | Path, development_run_dir: str | Path, output: str | Path
) -> dict[str, Any]:
    root = Path(run_dir)
    manifest = validate_package(root, development_run_dir=development_run_dir)
    outer = validate_audit_authorization(
        run_dir=root, development_run_dir=development_run_dir
    )
    payload = {
        "schema": LAUNCH_AUTH_SCHEMA,
        "status": "spot_authorized_after_audit50_package_and_open_freeze",
        "run_name": manifest["run_name"],
        "development_run_name": manifest["development_run_name"],
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "audit_open_authorization_sha256": sha256_file(
            root / "audit_open_authorization.json"
        ),
        "core_authorization_sha256": outer["core_authorization_sha256"],
        "development_pass_freeze_sha256": outer[
            "development_pass_freeze_sha256"
        ],
        "audit_plan_sha256": AUDIT50_PLAN_SHA256,
        "schedule_sha256": manifest["schedule_sha256"],
        "overlay_source_closure_sha256": manifest[
            "overlay_source_closure_sha256"
        ],
        "overlay_source_zip_sha256": manifest["overlay_source_zip_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "development_manifest_sha256": manifest["development_manifest_sha256"],
        "development_launch_authorization_sha256": manifest[
            "development_launch_authorization_sha256"
        ],
        "development_source_zip_sha256": manifest[
            "development_source_zip_sha256"
        ],
        "development_plan_sha256": SOURCE_PLAN_SHA256,
        "model_sha256": LAMBDA_MODEL_SHA256,
        "ai_profiles_sha256": AI_PROFILES_SHA256,
        "gcp_image_name": GCP_IMAGE_NAME,
        "gcp_image_id": GCP_IMAGE_ID,
        "gcp_image_self_link": GCP_IMAGE_SELF_LINK,
        "machine_type": MACHINE_TYPE,
        "native_batch_threads": NATIVE_BATCH_THREADS,
        "max_wave_shards": MAX_WAVE_SHARDS,
        "total_shards": TOTAL_SHARDS,
        "spot_authorized": True,
        "audit_started": False,
        "fresh_root_opened": False,
        "selector_executed": False,
        "fit_performed": False,
        "threshold_selected": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    write_canonical_json(output, payload)
    return payload


def validate_launch(
    *,
    run_dir: str | Path,
    development_run_dir: str | Path,
    authorization_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    path = Path(authorization_path) if authorization_path else root / "launch_authorization.json"
    authorization = load_json_mapping(path, "audit50 launch authorization")
    # Recreate the expected object in an isolated temporary path without
    # rewriting the immutable authorization.
    manifest = validate_package(root, development_run_dir=development_run_dir)
    outer = validate_audit_authorization(
        run_dir=root, development_run_dir=development_run_dir
    )
    expected = {
        "schema": LAUNCH_AUTH_SCHEMA,
        "status": "spot_authorized_after_audit50_package_and_open_freeze",
        "run_name": manifest["run_name"],
        "development_run_name": manifest["development_run_name"],
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "audit_open_authorization_sha256": sha256_file(root / "audit_open_authorization.json"),
        "core_authorization_sha256": outer["core_authorization_sha256"],
        "development_pass_freeze_sha256": outer["development_pass_freeze_sha256"],
        "audit_plan_sha256": AUDIT50_PLAN_SHA256,
        "schedule_sha256": manifest["schedule_sha256"],
        "overlay_source_closure_sha256": manifest["overlay_source_closure_sha256"],
        "overlay_source_zip_sha256": manifest["overlay_source_zip_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "development_manifest_sha256": manifest["development_manifest_sha256"],
        "development_launch_authorization_sha256": manifest["development_launch_authorization_sha256"],
        "development_source_zip_sha256": manifest["development_source_zip_sha256"],
        "development_plan_sha256": SOURCE_PLAN_SHA256,
        "model_sha256": LAMBDA_MODEL_SHA256,
        "ai_profiles_sha256": AI_PROFILES_SHA256,
        "gcp_image_name": GCP_IMAGE_NAME,
        "gcp_image_id": GCP_IMAGE_ID,
        "gcp_image_self_link": GCP_IMAGE_SELF_LINK,
        "machine_type": MACHINE_TYPE,
        "native_batch_threads": NATIVE_BATCH_THREADS,
        "max_wave_shards": MAX_WAVE_SHARDS,
        "total_shards": TOTAL_SHARDS,
        "spot_authorized": True,
        "audit_started": False,
        "fresh_root_opened": False,
        "selector_executed": False,
        "fit_performed": False,
        "threshold_selected": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    if authorization != expected:
        raise ValueError("Attempt08 audit50 launch authorization changed")
    return authorization


def _schedule(run_dir: Path) -> list[dict[str, Any]]:
    return validate_schedule_bytes((run_dir / "shards_manifest.jsonl").read_bytes())


def _spec(run_dir: Path, shard: int) -> dict[str, Any]:
    if type(shard) is not int or not 0 <= shard < TOTAL_SHARDS:
        raise ValueError("Attempt08 audit50 shard is outside 0..49")
    spec = _schedule(run_dir)[shard]
    if spec["shard"] != shard or spec["root_index"] != ROOT_FIRST + shard:
        raise ValueError("Attempt08 audit50 shard mapping changed")
    return spec


def build_global_claim(
    *, run_dir: str | Path, development_run_dir: str | Path, output: str | Path
) -> dict[str, Any]:
    root = Path(run_dir)
    launch = validate_launch(
        run_dir=root, development_run_dir=development_run_dir
    )
    payload = {
        "schema": GLOBAL_CLAIM_SCHEMA,
        "status": (
            "claimed_after_hash_and_schedule_validation_before_model_"
            "deserialization_inference_observation_teacher_or_result_read"
        ),
        "run_name": launch["run_name"],
        "manifest_sha256": launch["manifest_sha256"],
        "launch_authorization_sha256": sha256_file(root / "launch_authorization.json"),
        "audit_open_authorization_sha256": launch[
            "audit_open_authorization_sha256"
        ],
        "core_authorization_sha256": launch["core_authorization_sha256"],
        "schedule_sha256": launch["schedule_sha256"],
        "audit_plan_sha256": AUDIT50_PLAN_SHA256,
        "total_shards": TOTAL_SHARDS,
        "preclaim_operations": dict(_GLOBAL_PRECLAIM_OPERATIONS),
        "observation_generated": False,
        "teacher_executed": False,
        "selector_executed": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    write_canonical_json(output, payload, allow_identical=True)
    return payload


def build_root_claim(
    *,
    run_dir: str | Path,
    development_run_dir: str | Path,
    shard: int,
    global_claim_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    root = Path(run_dir)
    launch = validate_launch(
        run_dir=root, development_run_dir=development_run_dir
    )
    global_claim = load_json_mapping(global_claim_path, "audit50 global claim")
    if (
        global_claim.get("schema") != GLOBAL_CLAIM_SCHEMA
        or global_claim.get("status")
        != (
            "claimed_after_hash_and_schedule_validation_before_model_"
            "deserialization_inference_observation_teacher_or_result_read"
        )
        or global_claim.get("run_name") != launch["run_name"]
        or global_claim.get("launch_authorization_sha256")
        != sha256_file(root / "launch_authorization.json")
        or global_claim.get("teacher_executed") is not False
        or global_claim.get("preclaim_operations") != _GLOBAL_PRECLAIM_OPERATIONS
    ):
        raise ValueError("Attempt08 audit50 global claim changed")
    spec = _spec(root, shard)
    payload = {
        "schema": ROOT_CLAIM_SCHEMA,
        "status": (
            "root_claimed_after_frozen_spec_derivation_before_model_"
            "deserialization_inference_observation_teacher_or_result_read"
        ),
        "run_name": launch["run_name"],
        "shard": shard,
        "root_index": spec["root_index"],
        "root_profile": spec["root_profile"],
        "spec_sha256": canonical_sha256(spec),
        "global_claim_sha256": sha256_file(global_claim_path),
        "manifest_sha256": launch["manifest_sha256"],
        "launch_authorization_sha256": sha256_file(root / "launch_authorization.json"),
        "audit_open_authorization_sha256": launch[
            "audit_open_authorization_sha256"
        ],
        "core_authorization_sha256": launch["core_authorization_sha256"],
        "preclaim_operations": dict(_ROOT_PRECLAIM_OPERATIONS),
        "observation_generated": False,
        "teacher_executed": False,
        "alternate_root_seed_allowed": False,
        "current_profile_resolved": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    write_canonical_json(output, payload, allow_identical=True)
    return payload


def _validate_claims_before_root(
    *, run_dir: Path, shard: int, directory: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    global_path = directory / "global_claim.json"
    root_path = directory / "root_claim.json"
    global_claim = load_json_mapping(global_path, "audit50 global claim")
    root_claim = load_json_mapping(root_path, "audit50 root claim")
    launch_sha = sha256_file(run_dir / "launch_authorization.json")
    spec = _spec(run_dir, shard)
    if (
        global_claim.get("schema") != GLOBAL_CLAIM_SCHEMA
        or global_claim.get("status")
        != (
            "claimed_after_hash_and_schedule_validation_before_model_"
            "deserialization_inference_observation_teacher_or_result_read"
        )
        or global_claim.get("run_name")
        != load_json_mapping(run_dir / "manifest.json", "manifest")["run_name"]
        or global_claim.get("launch_authorization_sha256") != launch_sha
        or global_claim.get("preclaim_operations") != _GLOBAL_PRECLAIM_OPERATIONS
        or root_claim.get("schema") != ROOT_CLAIM_SCHEMA
        or root_claim.get("status")
        != (
            "root_claimed_after_frozen_spec_derivation_before_model_"
            "deserialization_inference_observation_teacher_or_result_read"
        )
        or root_claim.get("shard") != shard
        or root_claim.get("root_index") != spec["root_index"]
        or root_claim.get("root_profile") != spec["root_profile"]
        or root_claim.get("spec_sha256") != canonical_sha256(spec)
        or root_claim.get("global_claim_sha256") != sha256_file(global_path)
        or root_claim.get("launch_authorization_sha256") != launch_sha
        or root_claim.get("preclaim_operations") != _ROOT_PRECLAIM_OPERATIONS
        or any(
            claim.get(field) is not False
            for claim in (global_claim, root_claim)
            for field in (
                "observation_generated",
                "teacher_executed",
                "current_profile_mutated",
                "runtime_policy_activated",
            )
            if field in claim
        )
    ):
        raise ValueError("Attempt08 audit50 claim chain changed")
    return global_claim, root_claim


def run_shard(
    *,
    run_dir: str | Path,
    development_run_dir: str | Path,
    shard: int,
    directory: str | Path,
) -> dict[str, Any]:
    root = Path(run_dir)
    development = Path(development_run_dir).resolve()
    output = Path(directory)
    validate_launch(run_dir=root, development_run_dir=development)
    _validate_claims_before_root(run_dir=root, shard=shard, directory=output)
    spec = _spec(root, shard)
    teacher = output / "teacher.jsonl"
    if teacher.exists():
        raise FileExistsError("Attempt08 audit50 teacher output already exists")
    started = time.perf_counter()
    command = (
        "-m",
        "ofc_regular.run_hu_m43_attempt08_future_audit",
        "run",
        "--root-index",
        str(spec["root_index"]),
        "--output",
        str(teacher.resolve()),
        "--run-id",
        f"{load_json_mapping(root / 'manifest.json', 'manifest')['run_name']}:shard={shard}",
        "--audit-open-authorization",
        str((root / "core_audit_authorization.json").resolve()),
        "--development-pass-freeze",
        str((root / "development_pass_freeze.json").resolve()),
        "--run-dir",
        str(development),
        "--launch-authorization",
        str(development / "launch_authorization.json"),
        "--development-decision",
        str((root / "development_decision.json").resolve()),
        "--selector-receipt",
        str((root / "development_selector_receipt.json").resolve()),
        "--model",
        str(development / "package_src/artifacts/lambda_rank_candidate.pkl"),
        "--model-sha256",
        LAMBDA_MODEL_SHA256,
        "--plan",
        str(development / "hu_joint_policy_m43_attempt08.json"),
        "--ai-profiles",
        str(development / "package_src/src/ofc_regular/ai_profiles.py"),
    )
    try:
        raw = _run_frozen_python(development, command, timeout=24 * 3600)
        runner_summary = json.loads(raw.strip())
    except BaseException as exc:
        atomic_create(output / "run.log", (f"failed: {exc}\n").encode("utf-8"))
        raise
    elapsed = time.perf_counter() - started
    if (
        not isinstance(runner_summary, dict)
        or runner_summary.get("status") != "complete"
        or runner_summary.get("root_index") != spec["root_index"]
        or runner_summary.get("output_sha256") != sha256_file(teacher)
        or runner_summary.get("teacher_values_are_realized_match_ev") is not False
    ):
        raise ValueError("Attempt08 audit50 frozen runner summary changed")
    manifest = load_json_mapping(root / "manifest.json", "manifest")
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "status": "one_root_complete",
        "run_name": manifest["run_name"],
        "shard": shard,
        "root_index": spec["root_index"],
        "completed_roots": 1,
        "target_roots": 1,
        "teacher_sha256": sha256_file(teacher),
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "launch_authorization_sha256": sha256_file(root / "launch_authorization.json"),
        "audit_open_authorization_sha256": sha256_file(root / "audit_open_authorization.json"),
        "deterministic_same_root_recompute_allowed": True,
        "alternate_root_seed_allowed": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    heartbeat = {
        "schema": HEARTBEAT_SCHEMA,
        "status": "root_complete",
        "run_name": manifest["run_name"],
        "shard": shard,
        "root_index": spec["root_index"],
        "completed_roots": 1,
        "target_roots": 1,
        "teacher_sha256": sha256_file(teacher),
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "complete",
        "run_name": manifest["run_name"],
        "shard": shard,
        "root_index": spec["root_index"],
        "root_profile": spec["root_profile"],
        "teacher_sha256": sha256_file(teacher),
        "runner_summary": runner_summary,
        "elapsed_seconds": elapsed,
        "teacher_values_are_realized_match_ev": False,
        "fit_performed": False,
        "threshold_selected": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    write_canonical_json(output / "checkpoint.json", checkpoint)
    write_canonical_json(output / "heartbeat.json", heartbeat)
    write_canonical_json(output / "generator_summary.json", summary)
    atomic_create(output / "run.log", (raw.strip() + "\n").encode("utf-8"))
    atomic_create(output / "time.txt", f"elapsed_seconds={elapsed:.9f}\n".encode("ascii"))
    return summary


_COMMIT_FILES = (
    "teacher.jsonl",
    "checkpoint.json",
    "heartbeat.json",
    "generator_summary.json",
    "run.log",
    "time.txt",
    "global_claim.json",
    "root_claim.json",
    "boot_image_evidence.json",
)

_CHECKPOINT_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "shard",
        "root_index",
        "completed_roots",
        "target_roots",
        "teacher_sha256",
        "manifest_sha256",
        "launch_authorization_sha256",
        "audit_open_authorization_sha256",
        "deterministic_same_root_recompute_allowed",
        "alternate_root_seed_allowed",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
)
_HEARTBEAT_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "shard",
        "root_index",
        "completed_roots",
        "target_roots",
        "teacher_sha256",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
)
_SUMMARY_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "shard",
        "root_index",
        "root_profile",
        "teacher_sha256",
        "runner_summary",
        "elapsed_seconds",
        "teacher_values_are_realized_match_ev",
        "fit_performed",
        "threshold_selected",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
)
_RUNNER_SUMMARY_KEYS = frozenset(
    {
        "schema",
        "status",
        "population",
        "root_index",
        "output_sha256",
        "generator_elapsed_seconds",
        "generator_peak_rss_bytes",
        "teacher_values_are_realized_match_ev",
        "fit_performed",
        "threshold_selected",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
)
_BOOT_EVIDENCE_KEYS = frozenset(
    {
        "schema",
        "run_name",
        "shard",
        "instance_name",
        "disk_name",
        "source_image",
        "source_image_id",
    }
)
_RESUME_COMMIT_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "shard",
        "root_index",
        "files",
        "manifest_sha256",
        "launch_authorization_sha256",
        "audit_open_authorization_sha256",
        "root_claim_sha256",
        "teacher_values_are_realized_match_ev",
        "fit_performed",
        "threshold_selected",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
)


def _load_canonical_mapping(path: Path, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    value = load_json_mapping(path, label)
    if raw != canonical_json_bytes(value):
        raise ValueError(f"Attempt08 audit50 {label} is not canonical JSON")
    return value


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Attempt08 audit50 {label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Attempt08 audit50 {label} must be numeric") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"Attempt08 audit50 {label} must be finite and nonnegative")
    return result


def _read_elapsed_seconds(path: Path) -> float:
    raw = path.read_bytes()
    prefix = b"elapsed_seconds="
    if not raw.startswith(prefix) or not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise ValueError("Attempt08 audit50 time.txt format changed")
    try:
        elapsed = _finite_nonnegative(raw[len(prefix) : -1].decode("ascii"), "time")
    except UnicodeDecodeError as exc:
        raise ValueError("Attempt08 audit50 time.txt is not ASCII") from exc
    if raw != f"elapsed_seconds={elapsed:.9f}\n".encode("ascii"):
        raise ValueError("Attempt08 audit50 time.txt is not normalized")
    return elapsed


def _validate_teacher_row_semantics(
    *,
    run_dir: Path,
    development_run_dir: Path,
    shard: int,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    development_source = str(development_run_dir / "package_src/src")
    inserted = not sys.path or sys.path[0] != development_source
    if inserted:
        sys.path.insert(0, development_source)
    try:
        if __package__:
            from . import select_hu_m43_attempt08_audit50 as selector
        else:
            import select_hu_m43_attempt08_audit50 as selector  # type: ignore

        from ofc_regular.hu_m43_attempt08_contract import (
            enumerate_attempt08_seed_schedules,
            load_and_validate_attempt08_plan,
        )

        source_plan = load_and_validate_attempt08_plan(
            development_run_dir / "hu_joint_policy_m43_attempt08.json"
        )
        schedules = enumerate_attempt08_seed_schedules(
            source_plan, population="future_audit"
        )
        expected_seeds = {
            domain: int(schedules[domain][shard]) for domain in schedules
        }
        core = load_json_mapping(
            run_dir / "core_audit_authorization.json", "core authorization"
        )
        manifest = load_json_mapping(run_dir / "manifest.json", "manifest")
        return selector._validate_row(
            row,
            root_index=ROOT_FIRST + shard,
            expected_seeds=expected_seeds,
            audit_run_name=manifest["run_name"],
            core_authorization=core,
            core_authorization_sha256=sha256_file(
                run_dir / "core_audit_authorization.json"
            ),
        )
    finally:
        if inserted and sys.path and sys.path[0] == development_source:
            sys.path.pop(0)


def _validate_shard_material(
    *,
    run_dir: Path,
    development_run_dir: Path,
    shard: int,
    directory: Path,
    launch: Mapping[str, Any],
) -> dict[str, Any]:
    for name in _COMMIT_FILES:
        if not (directory / name).is_file():
            raise ValueError(f"Attempt08 audit50 completion file missing: {name}")
    _, root_claim = _validate_claims_before_root(
        run_dir=run_dir, shard=shard, directory=directory
    )
    spec = _spec(run_dir, shard)
    manifest_sha = sha256_file(run_dir / "manifest.json")
    launch_sha = sha256_file(run_dir / "launch_authorization.json")
    audit_open_sha = sha256_file(run_dir / "audit_open_authorization.json")
    teacher_path = directory / "teacher.jsonl"
    teacher_sha = sha256_file(teacher_path)
    row = _load_canonical_mapping(teacher_path, "teacher row")
    _validate_teacher_row_semantics(
        run_dir=run_dir,
        development_run_dir=development_run_dir,
        shard=shard,
        row=row,
    )

    checkpoint = _load_canonical_mapping(directory / "checkpoint.json", "checkpoint")
    expected_checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "status": "one_root_complete",
        "run_name": launch["run_name"],
        "shard": shard,
        "root_index": spec["root_index"],
        "completed_roots": 1,
        "target_roots": 1,
        "teacher_sha256": teacher_sha,
        "manifest_sha256": manifest_sha,
        "launch_authorization_sha256": launch_sha,
        "audit_open_authorization_sha256": audit_open_sha,
        "deterministic_same_root_recompute_allowed": True,
        "alternate_root_seed_allowed": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    if set(checkpoint) != _CHECKPOINT_KEYS or checkpoint != expected_checkpoint:
        raise ValueError("Attempt08 audit50 checkpoint changed")

    heartbeat = _load_canonical_mapping(directory / "heartbeat.json", "heartbeat")
    expected_heartbeat = {
        "schema": HEARTBEAT_SCHEMA,
        "status": "root_complete",
        "run_name": launch["run_name"],
        "shard": shard,
        "root_index": spec["root_index"],
        "completed_roots": 1,
        "target_roots": 1,
        "teacher_sha256": teacher_sha,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    if set(heartbeat) != _HEARTBEAT_KEYS or heartbeat != expected_heartbeat:
        raise ValueError("Attempt08 audit50 heartbeat changed")

    summary = _load_canonical_mapping(
        directory / "generator_summary.json", "generator summary"
    )
    runner = summary.get("runner_summary")
    elapsed = _finite_nonnegative(summary.get("elapsed_seconds"), "summary elapsed")
    if (
        set(summary) != _SUMMARY_KEYS
        or not isinstance(runner, Mapping)
        or set(runner) != _RUNNER_SUMMARY_KEYS
        or summary.get("schema") != SUMMARY_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("run_name") != launch["run_name"]
        or summary.get("shard") != shard
        or summary.get("root_index") != spec["root_index"]
        or summary.get("root_profile") != spec["root_profile"]
        or summary.get("teacher_sha256") != teacher_sha
        or runner.get("schema") != "hu_m43_attempt08_future_audit_summary_v1"
        or runner.get("status") != "complete"
        or runner.get("population") != "future_audit"
        or runner.get("root_index") != spec["root_index"]
        or runner.get("output_sha256") != teacher_sha
        or any(
            value.get(field) is not False
            for value in (summary, runner)
            for field in (
                "teacher_values_are_realized_match_ev",
                "fit_performed",
                "threshold_selected",
                "current_profile_mutated",
                "runtime_policy_activated",
            )
        )
    ):
        raise ValueError("Attempt08 audit50 generator summary changed")
    generator_elapsed = _finite_nonnegative(
        runner.get("generator_elapsed_seconds"), "runner elapsed"
    )
    peak_rss = runner.get("generator_peak_rss_bytes")
    if (
        isinstance(peak_rss, bool)
        or not isinstance(peak_rss, int)
        or peak_rss < 0
        or generator_elapsed > elapsed + 1.0e-9
    ):
        raise ValueError("Attempt08 audit50 runner timing/resource summary changed")
    time_elapsed = _read_elapsed_seconds(directory / "time.txt")
    if not math.isclose(time_elapsed, elapsed, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("Attempt08 audit50 time.txt disagrees with summary")
    run_log = (directory / "run.log").read_bytes()
    if not run_log.endswith(b"\n") or run_log.count(b"\n") != 1:
        raise ValueError("Attempt08 audit50 run.log format changed")
    try:
        logged_runner = json.loads(run_log.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Attempt08 audit50 run.log is invalid") from exc
    if logged_runner != runner:
        raise ValueError("Attempt08 audit50 run.log disagrees with runner summary")

    boot = _load_canonical_mapping(
        directory / "boot_image_evidence.json", "boot evidence"
    )
    if (
        set(boot) != _BOOT_EVIDENCE_KEYS
        or boot.get("schema")
        != "hu_m43_attempt08_audit50_boot_image_evidence_v1"
        or boot.get("run_name") != launch["run_name"]
        or boot.get("shard") != shard
        or not isinstance(boot.get("instance_name"), str)
        or not boot["instance_name"]
        or not isinstance(boot.get("disk_name"), str)
        or not boot["disk_name"]
        or str(boot.get("source_image_id")) != GCP_IMAGE_ID
        or not str(boot.get("source_image", "")).endswith(GCP_IMAGE_SELF_LINK)
    ):
        raise ValueError("Attempt08 audit50 boot image evidence changed")
    if root_claim.get("spec_sha256") != canonical_sha256(spec):
        raise ValueError("Attempt08 audit50 root claim/spec binding changed")
    return {"row": row, "summary": summary, "teacher_sha256": teacher_sha}


def _resume_commit_payload(
    *,
    run_dir: Path,
    shard: int,
    launch: Mapping[str, Any],
    files: Mapping[str, str],
    root_claim_sha256: str,
) -> dict[str, Any]:
    spec = _spec(run_dir, shard)
    return {
        "schema": RESUME_COMMIT_SCHEMA,
        "status": "content_addressed_root_commit_complete",
        "run_name": launch["run_name"],
        "shard": shard,
        "root_index": spec["root_index"],
        "files": dict(files),
        "manifest_sha256": launch["manifest_sha256"],
        "launch_authorization_sha256": sha256_file(
            run_dir / "launch_authorization.json"
        ),
        "audit_open_authorization_sha256": launch[
            "audit_open_authorization_sha256"
        ],
        "root_claim_sha256": root_claim_sha256,
        "teacher_values_are_realized_match_ev": False,
        "fit_performed": False,
        "threshold_selected": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }


def validate_resume_commit_file(
    *,
    run_dir: str | Path,
    development_run_dir: str | Path,
    shard: int,
    commit_path: str | Path,
) -> dict[str, Any]:
    root = Path(run_dir)
    launch = validate_launch(
        run_dir=root, development_run_dir=development_run_dir
    )
    commit = _load_canonical_mapping(Path(commit_path), "resume commit")
    spec = _spec(root, shard)
    files = commit.get("files")
    if (
        set(commit) != _RESUME_COMMIT_KEYS
        or commit.get("schema") != RESUME_COMMIT_SCHEMA
        or commit.get("status") != "content_addressed_root_commit_complete"
        or commit.get("run_name") != launch["run_name"]
        or commit.get("shard") != shard
        or commit.get("root_index") != spec["root_index"]
        or not isinstance(files, Mapping)
        or set(files) != set(_COMMIT_FILES)
        or commit.get("manifest_sha256") != launch["manifest_sha256"]
        or commit.get("launch_authorization_sha256")
        != sha256_file(root / "launch_authorization.json")
        or commit.get("audit_open_authorization_sha256")
        != launch["audit_open_authorization_sha256"]
        or any(
            commit.get(field) is not False
            for field in (
                "teacher_values_are_realized_match_ev",
                "fit_performed",
                "threshold_selected",
                "current_profile_mutated",
                "runtime_policy_activated",
            )
        )
    ):
        raise ValueError("Attempt08 audit50 resume commit changed")
    for name, digest in files.items():
        require_sha256(digest, f"resume object {name}")
    require_sha256(commit.get("root_claim_sha256"), "resume root claim")
    return commit


def complete_shard(
    *,
    run_dir: str | Path,
    development_run_dir: str | Path,
    shard: int,
    directory: str | Path,
) -> dict[str, Any]:
    root = Path(run_dir)
    output = Path(directory)
    launch = validate_launch(run_dir=root, development_run_dir=development_run_dir)
    _, root_claim = _validate_claims_before_root(
        run_dir=root, shard=shard, directory=output
    )
    spec = _spec(root, shard)
    if (output / "DONE.json").exists():
        return validate_completed_bundle(
            run_dir=root,
            development_run_dir=development_run_dir,
            shard=shard,
            directory=output,
        )
    # This is the pre-DONE science gate for both a fresh root and a restored
    # content-addressed resume.  No commit/DONE bytes are created until every
    # result object has passed full semantic and cross-file validation.
    _validate_shard_material(
        run_dir=root,
        development_run_dir=Path(development_run_dir),
        shard=shard,
        directory=output,
        launch=launch,
    )
    files = {name: sha256_file(output / name) for name in _COMMIT_FILES}
    commit = _resume_commit_payload(
        run_dir=root,
        shard=shard,
        launch=launch,
        files=files,
        root_claim_sha256=sha256_file(output / "root_claim.json"),
    )
    if (output / "resume_commit.json").exists():
        restored = validate_resume_commit_file(
            run_dir=root,
            development_run_dir=development_run_dir,
            shard=shard,
            commit_path=output / "resume_commit.json",
        )
        if restored != commit:
            raise ValueError("Attempt08 audit50 restored resume commit/content changed")
    write_canonical_json(output / "resume_commit.json", commit, allow_identical=True)
    done = {
        "schema": DONE_SCHEMA,
        "status": "complete_done_published_last",
        "run_name": launch["run_name"],
        "shard": shard,
        "root_index": spec["root_index"],
        "root_profile": spec["root_profile"],
        "manifest_sha256": launch["manifest_sha256"],
        "launch_authorization_sha256": sha256_file(root / "launch_authorization.json"),
        "audit_open_authorization_sha256": launch[
            "audit_open_authorization_sha256"
        ],
        "core_authorization_sha256": launch["core_authorization_sha256"],
        "audit_plan_sha256": AUDIT50_PLAN_SHA256,
        "spec_sha256": root_claim["spec_sha256"],
        "resume_commit_sha256": sha256_file(output / "resume_commit.json"),
        "files": files,
        "teacher_values_are_realized_match_ev": False,
        "selector_executed": False,
        "fit_performed": False,
        "threshold_selected": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    write_canonical_json(output / "DONE.json", done)
    return done


def validate_done_file(
    *, run_dir: str | Path, authorization_path: str | Path, shard: int, done: str | Path
) -> dict[str, Any]:
    root = Path(run_dir)
    manifest = load_json_mapping(root / "manifest.json", "audit50 manifest")
    launch = load_json_mapping(authorization_path, "audit50 launch authorization")
    spec = _spec(root, shard)
    payload = load_json_mapping(done, "audit50 DONE")
    if (
        payload.get("schema") != DONE_SCHEMA
        or payload.get("status") != "complete_done_published_last"
        or payload.get("run_name") != manifest.get("run_name")
        or payload.get("shard") != shard
        or payload.get("root_index") != spec["root_index"]
        or payload.get("root_profile") != spec["root_profile"]
        or payload.get("manifest_sha256") != sha256_file(root / "manifest.json")
        or payload.get("launch_authorization_sha256")
        != sha256_file(authorization_path)
        or payload.get("audit_open_authorization_sha256")
        != launch.get("audit_open_authorization_sha256")
        or payload.get("core_authorization_sha256")
        != launch.get("core_authorization_sha256")
        or payload.get("audit_plan_sha256") != AUDIT50_PLAN_SHA256
        or payload.get("spec_sha256") != canonical_sha256(spec)
        or not isinstance(payload.get("files"), Mapping)
        or set(payload["files"]) != set(_COMMIT_FILES)
        or any(not isinstance(value, str) or len(value) != 64 for value in payload["files"].values())
        or any(
            payload.get(field) is not False
            for field in (
                "teacher_values_are_realized_match_ev",
                "selector_executed",
                "fit_performed",
                "threshold_selected",
                "current_profile_mutated",
                "runtime_policy_activated",
            )
        )
    ):
        raise ValueError(f"Attempt08 audit50 DONE changed at shard {shard}")
    require_sha256(payload.get("resume_commit_sha256"), "resume commit SHA")
    return payload


def validate_done_set(
    *, run_dir: str | Path, authorization_path: str | Path, done_root: str | Path
) -> list[dict[str, Any]]:
    directory = Path(done_root)
    expected_names = {f"DONE-{shard:03d}.json" for shard in range(TOTAL_SHARDS)}
    actual_names = {path.name for path in directory.glob("DONE-*.json")}
    if actual_names != expected_names:
        raise ValueError("Attempt08 audit50 DONE set must be exactly 50 shards")
    return [
        validate_done_file(
            run_dir=run_dir,
            authorization_path=authorization_path,
            shard=shard,
            done=directory / f"DONE-{shard:03d}.json",
        )
        for shard in range(TOTAL_SHARDS)
    ]


def validate_completed_bundle(
    *,
    run_dir: str | Path,
    development_run_dir: str | Path,
    shard: int,
    directory: str | Path,
) -> dict[str, Any]:
    root = Path(run_dir)
    output = Path(directory)
    validate_launch(run_dir=root, development_run_dir=development_run_dir)
    done = validate_done_file(
        run_dir=root,
        authorization_path=root / "launch_authorization.json",
        shard=shard,
        done=output / "DONE.json",
    )
    commit = load_json_mapping(output / "resume_commit.json", "resume commit")
    if (
        commit.get("schema") != RESUME_COMMIT_SCHEMA
        or commit.get("shard") != shard
        or commit.get("files") != done["files"]
        or sha256_file(output / "resume_commit.json")
        != done["resume_commit_sha256"]
    ):
        raise ValueError("Attempt08 audit50 resume commit changed")
    for name, digest in done["files"].items():
        if sha256_file(output / name) != digest:
            raise ValueError(f"Attempt08 audit50 completed file changed: {name}")
    material = _validate_shard_material(
        run_dir=root,
        development_run_dir=Path(development_run_dir),
        shard=shard,
        directory=output,
        launch=load_json_mapping(
            root / "launch_authorization.json", "launch authorization"
        ),
    )
    if material["teacher_sha256"] != done["files"]["teacher.jsonl"]:
        raise ValueError("Attempt08 audit50 completed teacher binding changed")
    return done


def claim_complete_output(
    *,
    run_dir: str | Path,
    authorization_path: str | Path,
    done_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    root = Path(run_dir)
    done = validate_done_set(
        run_dir=root, authorization_path=authorization_path, done_root=done_root
    )
    manifest = load_json_mapping(root / "manifest.json", "manifest")
    payload = {
        "schema": CONSUMPTION_SCHEMA,
        "status": "claimed_after_all_done_before_any_audit_content_read",
        "run_name": manifest["run_name"],
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "launch_authorization_sha256": sha256_file(authorization_path),
        "audit_open_authorization_sha256": sha256_file(
            root / "audit_open_authorization.json"
        ),
        "done_sha256": {
            f"{shard:03d}": sha256_file(Path(done_root) / f"DONE-{shard:03d}.json")
            for shard in range(TOTAL_SHARDS)
        },
        "done_payload_sha256": {
            f"{shard:03d}": canonical_sha256(done[shard])
            for shard in range(TOTAL_SHARDS)
        },
        "expected_shards": TOTAL_SHARDS,
        "all_done_markers_verified": True,
        "result_objects_addressed_when_claimed": False,
        "remote_claim_required_before_content_read": True,
        "selector_executed": False,
        "fit_performed": False,
        "threshold_selected": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    write_canonical_json(output, payload)
    return payload


def validate_claim_pair(
    *,
    run_dir: str | Path,
    authorization_path: str | Path,
    local_claim: str | Path,
    remote_claim: str | Path,
) -> tuple[dict[str, Any], str]:
    local_path, remote_path = Path(local_claim), Path(remote_claim)
    if local_path.read_bytes() != remote_path.read_bytes():
        raise ValueError("Attempt08 audit50 local and remote claims differ")
    claim = load_json_mapping(local_path, "audit50 consumption claim")
    if (
        claim.get("schema") != CONSUMPTION_SCHEMA
        or claim.get("status")
        != "claimed_after_all_done_before_any_audit_content_read"
        or claim.get("manifest_sha256") != sha256_file(Path(run_dir) / "manifest.json")
        or claim.get("launch_authorization_sha256") != sha256_file(authorization_path)
        or claim.get("expected_shards") != TOTAL_SHARDS
        or claim.get("all_done_markers_verified") is not True
        or claim.get("result_objects_addressed_when_claimed") is not False
        or claim.get("remote_claim_required_before_content_read") is not True
        or claim.get("selector_executed") is not False
        or not isinstance(claim.get("done_sha256"), Mapping)
        or set(claim["done_sha256"]) != {f"{i:03d}" for i in range(TOTAL_SHARDS)}
    ):
        raise ValueError("Attempt08 audit50 consumption claim changed")
    return claim, sha256_file(local_path)


def audit_received_shard(
    *,
    directory: str | Path,
    shard: int,
    run_dir: str | Path,
    development_run_dir: str | Path,
    authorization_path: str | Path,
    local_claim: str | Path,
    remote_claim: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    claim, claim_sha = validate_claim_pair(
        run_dir=run_dir,
        authorization_path=authorization_path,
        local_claim=local_claim,
        remote_claim=remote_claim,
    )
    done = validate_completed_bundle(
        run_dir=run_dir,
        development_run_dir=development_run_dir,
        shard=shard,
        directory=directory,
    )
    done_path = Path(directory) / "DONE.json"
    if claim["done_sha256"][f"{shard:03d}"] != sha256_file(done_path):
        raise ValueError("Attempt08 audit50 received DONE differs from claimed DONE")
    spec = _spec(Path(run_dir), shard)
    payload = {
        "schema": RECEIVED_SCHEMA,
        "status": "verified_after_local_and_remote_consumption_claims",
        "run_name": done["run_name"],
        "shard": shard,
        "root_index": spec["root_index"],
        "root_profile": spec["root_profile"],
        "done_sha256": sha256_file(done_path),
        "teacher_sha256": sha256_file(Path(directory) / "teacher.jsonl"),
        "consumption_claim_sha256": claim_sha,
        "remote_consumption_claim_sha256": sha256_file(remote_claim),
        "audit_plan_sha256": AUDIT50_PLAN_SHA256,
        "selector_executed": False,
        "fit_performed": False,
        "threshold_selected": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    write_canonical_json(output, payload)
    return payload


def merge_received(
    *,
    received_root: str | Path,
    audit_root: str | Path,
    run_dir: str | Path,
    authorization_path: str | Path,
    local_claim: str | Path,
    remote_claim: str | Path,
    output: str | Path,
    receipt: str | Path,
) -> dict[str, Any]:
    claim, claim_sha = validate_claim_pair(
        run_dir=run_dir,
        authorization_path=authorization_path,
        local_claim=local_claim,
        remote_claim=remote_claim,
    )
    rows: list[bytes] = []
    teacher_hashes: dict[str, str] = {}
    audit_hashes: dict[str, str] = {}
    for shard in range(TOTAL_SHARDS):
        directory = Path(received_root) / f"shard_{shard:03d}"
        audit_path = Path(audit_root) / f"audit_{shard:03d}.json"
        audit = load_json_mapping(audit_path, "received audit")
        teacher = directory / "teacher.jsonl"
        raw = teacher.read_bytes()
        row = load_json_mapping(teacher, "audit teacher row")
        if (
            audit.get("schema") != RECEIVED_SCHEMA
            or audit.get("shard") != shard
            or audit.get("root_index") != ROOT_FIRST + shard
            or audit.get("teacher_sha256") != sha256_file(teacher)
            or row.get("root_index") != ROOT_FIRST + shard
            or raw != canonical_json_bytes(row)
        ):
            raise ValueError(f"Attempt08 audit50 received shard changed: {shard}")
        rows.append(raw)
        teacher_hashes[f"{shard:03d}"] = sha256_file(teacher)
        audit_hashes[f"{shard:03d}"] = sha256_file(audit_path)
    merged_path = Path(output)
    atomic_create(merged_path, b"".join(rows))
    manifest = load_json_mapping(Path(run_dir) / "manifest.json", "manifest")
    payload = {
        "schema": MERGE_SCHEMA,
        "status": "exact_50_roots_merged_without_selection",
        "run_name": manifest["run_name"],
        "roots": TOTAL_SHARDS,
        "root_index_first": ROOT_FIRST,
        "root_index_last": ROOT_LAST,
        "merged_sha256": sha256_file(merged_path),
        "teacher_sha256": teacher_hashes,
        "received_audit_sha256": audit_hashes,
        "consumption_claim_sha256": claim_sha,
        "remote_consumption_claim_sha256": sha256_file(remote_claim),
        "manifest_sha256": sha256_file(Path(run_dir) / "manifest.json"),
        "launch_authorization_sha256": sha256_file(authorization_path),
        "audit_plan_sha256": AUDIT50_PLAN_SHA256,
        "selector_must_execute_exactly_once": True,
        "selector_executed": False,
        "fit_performed": False,
        "threshold_selected": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    write_canonical_json(receipt, payload)
    return payload


def validate_merged(
    *,
    run_dir: str | Path,
    authorization_path: str | Path,
    local_claim: str | Path,
    remote_claim: str | Path,
    merged: str | Path,
    receipt: str | Path,
) -> dict[str, Any]:
    _, claim_sha = validate_claim_pair(
        run_dir=run_dir,
        authorization_path=authorization_path,
        local_claim=local_claim,
        remote_claim=remote_claim,
    )
    value = load_json_mapping(receipt, "audit50 merge receipt")
    raw = Path(merged).read_bytes()
    teacher_hashes = value.get("teacher_sha256")
    audit_hashes = value.get("received_audit_sha256")
    expected_keys = {f"{shard:03d}" for shard in range(TOTAL_SHARDS)}
    if (
        value.get("schema") != MERGE_SCHEMA
        or value.get("status") != "exact_50_roots_merged_without_selection"
        or value.get("run_name")
        != load_json_mapping(Path(run_dir) / "manifest.json", "manifest")["run_name"]
        or value.get("roots") != TOTAL_SHARDS
        or value.get("root_index_first") != ROOT_FIRST
        or value.get("root_index_last") != ROOT_LAST
        or value.get("merged_sha256") != sha256_file(merged)
        or value.get("consumption_claim_sha256") != claim_sha
        or value.get("remote_consumption_claim_sha256") != sha256_file(remote_claim)
        or value.get("manifest_sha256")
        != sha256_file(Path(run_dir) / "manifest.json")
        or value.get("launch_authorization_sha256")
        != sha256_file(authorization_path)
        or value.get("audit_plan_sha256") != AUDIT50_PLAN_SHA256
        or not isinstance(teacher_hashes, Mapping)
        or set(teacher_hashes) != expected_keys
        or not isinstance(audit_hashes, Mapping)
        or set(audit_hashes) != expected_keys
        or value.get("selector_must_execute_exactly_once") is not True
        or value.get("selector_executed") is not False
        or any(
            value.get(field) is not False
            for field in (
                "fit_performed",
                "threshold_selected",
                "current_profile_mutated",
                "runtime_policy_activated",
            )
        )
        or len(raw.splitlines()) != TOTAL_SHARDS
    ):
        raise ValueError("Attempt08 audit50 merged receive changed")
    receive_root = Path(merged).resolve().parent.parent
    for shard, line in enumerate(raw.splitlines()):
        row = json.loads(line)
        key = f"{shard:03d}"
        teacher_path = receive_root / f"received/shard_{key}/teacher.jsonl"
        audit_path = receive_root / f"audits/audit_{key}.json"
        audit = load_json_mapping(audit_path, "received audit")
        if (
            row.get("root_index") != ROOT_FIRST + shard
            or line + b"\n" != canonical_json_bytes(row)
            or teacher_path.read_bytes() != line + b"\n"
            or teacher_hashes[key] != sha256_file(teacher_path)
            or audit_hashes[key] != sha256_file(audit_path)
            or audit.get("schema") != RECEIVED_SCHEMA
            or audit.get("shard") != shard
            or audit.get("root_index") != ROOT_FIRST + shard
            or audit.get("teacher_sha256") != teacher_hashes[key]
            or audit.get("consumption_claim_sha256") != claim_sha
            or audit.get("remote_consumption_claim_sha256")
            != sha256_file(remote_claim)
        ):
            raise ValueError("Attempt08 audit50 merged root ordering changed")
    return value


def _selector_paths(run_dir: Path) -> dict[str, Path]:
    root = run_dir / "selector"
    return {
        "root": root,
        "claim": root / "CLAIM.json",
        "remote_claim": root / "REMOTE_CLAIM.json",
        "execution": root / "EXECUTION_STARTED.json",
        "decision": root / "decision.json",
        "receipt": root / "decision_receipt.json",
    }


def _validate_selector_bound_inputs(
    *, selector_run_dir: Path, claim: Mapping[str, Any]
) -> dict[str, Any]:
    merged_value = claim.get("canonical_merged_path")
    receipt_value = claim.get("canonical_merge_receipt_path")
    if not isinstance(merged_value, str) or not isinstance(receipt_value, str):
        raise ValueError("Attempt08 audit50 selector canonical input paths changed")
    expected_merged = (selector_run_dir / "merged/teacher.jsonl").resolve()
    expected_receipt = (selector_run_dir / "merged/merge_receipt.json").resolve()
    merged = Path(merged_value).resolve()
    merge_receipt_path = Path(receipt_value).resolve()
    if merged != expected_merged or merge_receipt_path != expected_receipt:
        raise ValueError("Attempt08 audit50 selector input path escaped canonical receive")
    actual_merged_sha = sha256_file(merged)
    actual_receipt_sha = sha256_file(merge_receipt_path)
    if (
        claim.get("merged_sha256") != actual_merged_sha
        or claim.get("merge_receipt_sha256") != actual_receipt_sha
    ):
        raise ValueError("Attempt08 audit50 selector input changed after claim")
    merge_receipt = _load_canonical_mapping(
        merge_receipt_path, "selector merge receipt"
    )
    raw = merged.read_bytes()
    if (
        merge_receipt.get("schema") != MERGE_SCHEMA
        or merge_receipt.get("status")
        != "exact_50_roots_merged_without_selection"
        or merge_receipt.get("roots") != TOTAL_SHARDS
        or merge_receipt.get("root_index_first") != ROOT_FIRST
        or merge_receipt.get("root_index_last") != ROOT_LAST
        or merge_receipt.get("merged_sha256") != actual_merged_sha
        or len(raw.splitlines()) != TOTAL_SHARDS
        or not raw.endswith(b"\n")
        or b"\r" in raw
    ):
        raise ValueError("Attempt08 audit50 selector merge binding changed")
    for shard, line in enumerate(raw.splitlines()):
        try:
            row = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Attempt08 audit50 selector merged row is invalid") from exc
        if (
            not isinstance(row, Mapping)
            or row.get("root_index") != ROOT_FIRST + shard
            or line + b"\n" != canonical_json_bytes(row)
        ):
            raise ValueError("Attempt08 audit50 selector merged root order changed")
    if (
        sha256_file(merged) != actual_merged_sha
        or sha256_file(merge_receipt_path) != actual_receipt_sha
    ):
        raise ValueError("Attempt08 audit50 selector inputs changed during validation")
    return {
        "merged_path": merged,
        "merge_receipt_path": merge_receipt_path,
        "merged_sha256": actual_merged_sha,
        "merge_receipt_sha256": actual_receipt_sha,
    }


def build_selector_claim(
    *,
    run_dir: str | Path,
    authorization_path: str | Path,
    local_consumption_claim: str | Path,
    remote_consumption_claim: str | Path,
    merged: str | Path,
    merge_receipt: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    root = Path(run_dir)
    merge = validate_merged(
        run_dir=root,
        authorization_path=authorization_path,
        local_claim=local_consumption_claim,
        remote_claim=remote_consumption_claim,
        merged=merged,
        receipt=merge_receipt,
    )
    selector_source = (
        root
        / "overlay_src/src/ofc_regular/select_hu_m43_attempt08_audit50.py"
    )
    paths = _selector_paths(Path(output).parent.parent)
    payload = {
        "schema": SELECTOR_CLAIM_SCHEMA,
        "status": "claimed_before_single_frozen_audit_gate_evaluation",
        "run_name": load_json_mapping(root / "manifest.json", "manifest")["run_name"],
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "launch_authorization_sha256": sha256_file(authorization_path),
        "audit_open_authorization_sha256": sha256_file(
            root / "audit_open_authorization.json"
        ),
        "core_authorization_sha256": sha256_file(
            root / "core_audit_authorization.json"
        ),
        "consumption_claim_sha256": sha256_file(local_consumption_claim),
        "remote_consumption_claim_sha256": sha256_file(
            remote_consumption_claim
        ),
        "merged_sha256": merge["merged_sha256"],
        "merge_receipt_sha256": sha256_file(merge_receipt),
        "selector_source_sha256": sha256_file(selector_source),
        "canonical_merged_path": str(Path(merged).resolve()),
        "canonical_merge_receipt_path": str(Path(merge_receipt).resolve()),
        "canonical_decision_path": str(paths["decision"]),
        "canonical_receipt_path": str(paths["receipt"]),
        "gate_evaluation_count_before_claim": 0,
        "selector_executed": False,
        "fit_performed": False,
        "threshold_selected": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    write_canonical_json(output, payload)
    return payload


def execute_selector_once(
    *,
    run_dir: str | Path,
    development_run_dir: str | Path,
    selector_claim_path: str | Path,
    remote_selector_claim_path: str | Path,
) -> dict[str, Any]:
    root = Path(run_dir)
    selector_root = Path(selector_claim_path).parent
    paths = _selector_paths(selector_root.parent)
    if Path(selector_claim_path).read_bytes() != Path(remote_selector_claim_path).read_bytes():
        raise ValueError("Attempt08 audit50 selector claims differ")
    claim = load_json_mapping(selector_claim_path, "audit50 selector claim")
    if (
        claim.get("schema") != SELECTOR_CLAIM_SCHEMA
        or claim.get("status")
        != "claimed_before_single_frozen_audit_gate_evaluation"
        or claim.get("run_name")
        != load_json_mapping(root / "manifest.json", "manifest")["run_name"]
        or claim.get("selector_source_sha256")
        != sha256_file(
            root / "overlay_src/src/ofc_regular/select_hu_m43_attempt08_audit50.py"
        )
        or claim.get("gate_evaluation_count_before_claim") != 0
        or claim.get("selector_executed") is not False
    ):
        raise ValueError("Attempt08 audit50 selector claim changed")
    bound_inputs = _validate_selector_bound_inputs(
        selector_run_dir=selector_root.parent, claim=claim
    )
    if paths["receipt"].exists():
        return validate_selector_completion(run_dir=selector_root.parent)
    if paths["execution"].exists() or paths["decision"].exists():
        raise ValueError(
            "Attempt08 audit50 selector began without atomic completion; automatic reevaluation is forbidden"
        )
    execution = {
        "schema": SELECTOR_EXECUTION_SCHEMA,
        "status": "single_frozen_gate_evaluation_started",
        "run_name": claim["run_name"],
        "selector_claim_sha256": sha256_file(selector_claim_path),
        "remote_selector_claim_sha256": sha256_file(remote_selector_claim_path),
        "merge_receipt_sha256": claim["merge_receipt_sha256"],
        "merged_sha256": claim["merged_sha256"],
        "selector_source_sha256": claim["selector_source_sha256"],
        "gate_evaluation_count_before_start": 0,
        "selector_executed": False,
        "fit_performed": False,
        "threshold_selected": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    write_canonical_json(paths["execution"], execution)
    development = Path(development_run_dir)
    sys.path.insert(0, str(development / "package_src/src"))
    try:
        if __package__:
            from . import select_hu_m43_attempt08_audit50 as selector
        else:
            import select_hu_m43_attempt08_audit50 as selector  # type: ignore
        core = load_json_mapping(root / "core_audit_authorization.json", "core auth")
        decision = selector._write_from_lifecycle(
            token=selector._LIFECYCLE_TOKEN,
            input_jsonl=Path(claim["canonical_merged_path"]),
            output=paths["decision"],
            audit_plan_path=root / "hu_joint_policy_m43_attempt08_audit50.json",
            source_plan_path=development / "hu_joint_policy_m43_attempt08.json",
            audit_run_name=claim["run_name"],
            core_authorization=core,
            core_authorization_sha256=claim["core_authorization_sha256"],
            outer_authorization_sha256=claim[
                "audit_open_authorization_sha256"
            ],
            merge_receipt_sha256=claim["merge_receipt_sha256"],
        )
    finally:
        if sys.path and sys.path[0] == str(development / "package_src/src"):
            sys.path.pop(0)
    bound_inputs_after = _validate_selector_bound_inputs(
        selector_run_dir=selector_root.parent, claim=claim
    )
    if bound_inputs_after != bound_inputs:
        raise ValueError("Attempt08 audit50 selector inputs changed during execution")
    decision_source = decision.get("source")
    if (
        not isinstance(decision_source, Mapping)
        or decision_source.get("input_jsonl_sha256")
        != bound_inputs["merged_sha256"]
        or decision_source.get("merge_receipt_sha256")
        != bound_inputs["merge_receipt_sha256"]
    ):
        raise ValueError("Attempt08 audit50 decision source/input binding changed")
    receipt = {
        "schema": SELECTOR_RECEIPT_SCHEMA,
        "status": "single_frozen_audit_gate_evaluation_complete",
        "run_name": claim["run_name"],
        "selector_claim_sha256": sha256_file(selector_claim_path),
        "remote_selector_claim_sha256": sha256_file(remote_selector_claim_path),
        "execution_marker_sha256": sha256_file(paths["execution"]),
        "merge_receipt_sha256": claim["merge_receipt_sha256"],
        "merged_sha256": claim["merged_sha256"],
        "selector_source_sha256": claim["selector_source_sha256"],
        "decision_sha256": sha256_file(paths["decision"]),
        "decision": decision["decision"],
        "go_action": (
            "authorize_distillation_from_development200_only"
            if decision["decision"] == "go"
            else None
        ),
        "gate_evaluation_count": 1,
        "selector_executed": True,
        "audit_rows_used_for_fit": False,
        "fit_performed": False,
        "threshold_selected": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    write_canonical_json(paths["receipt"], receipt)
    return receipt


def validate_selector_completion(*, run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    paths = _selector_paths(root)
    claim = load_json_mapping(paths["claim"], "selector claim")
    remote = load_json_mapping(paths["remote_claim"], "remote selector claim")
    execution = load_json_mapping(paths["execution"], "selector execution")
    decision = load_json_mapping(paths["decision"], "selector decision")
    receipt = load_json_mapping(paths["receipt"], "selector receipt")
    bound_inputs = _validate_selector_bound_inputs(
        selector_run_dir=root, claim=claim
    )
    decision_source = decision.get("source")
    if (
        claim != remote
        or paths["claim"].read_bytes() != paths["remote_claim"].read_bytes()
        or receipt.get("schema") != SELECTOR_RECEIPT_SCHEMA
        or receipt.get("status")
        != "single_frozen_audit_gate_evaluation_complete"
        or receipt.get("selector_claim_sha256") != sha256_file(paths["claim"])
        or receipt.get("remote_selector_claim_sha256")
        != sha256_file(paths["remote_claim"])
        or receipt.get("execution_marker_sha256") != sha256_file(paths["execution"])
        or receipt.get("merge_receipt_sha256")
        != bound_inputs["merge_receipt_sha256"]
        or receipt.get("merged_sha256") != bound_inputs["merged_sha256"]
        or receipt.get("decision_sha256") != sha256_file(paths["decision"])
        or receipt.get("decision") != decision.get("decision")
        or receipt.get("gate_evaluation_count") != 1
        or receipt.get("selector_executed") is not True
        or execution.get("selector_executed") is not False
        or execution.get("schema") != SELECTOR_EXECUTION_SCHEMA
        or execution.get("status") != "single_frozen_gate_evaluation_started"
        or execution.get("selector_claim_sha256") != sha256_file(paths["claim"])
        or execution.get("remote_selector_claim_sha256")
        != sha256_file(paths["remote_claim"])
        or execution.get("merge_receipt_sha256")
        != bound_inputs["merge_receipt_sha256"]
        or execution.get("merged_sha256") != bound_inputs["merged_sha256"]
        or not isinstance(decision_source, Mapping)
        or decision_source.get("input_jsonl_sha256")
        != bound_inputs["merged_sha256"]
        or decision_source.get("merge_receipt_sha256")
        != bound_inputs["merge_receipt_sha256"]
        or decision.get("decision_contract", {}).get("gate_evaluation_count") != 1
        or decision.get("decision_contract", {}).get("audit_rows_used_for_fit")
        is not False
        or any(
            receipt.get(field) is not False
            for field in (
                "audit_rows_used_for_fit",
                "fit_performed",
                "threshold_selected",
                "current_profile_mutated",
                "runtime_policy_activated",
            )
        )
    ):
        raise ValueError("Attempt08 audit50 completed selector lifecycle changed")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    package = sub.add_parser("package")
    package.add_argument("--repo-root", required=True, type=Path)
    package.add_argument("--run-dir", required=True, type=Path)
    package.add_argument("--run-name", required=True)
    package.add_argument("--development-run-dir", required=True, type=Path)
    package.add_argument("--audit-plan", required=True, type=Path)
    package.add_argument("--resume-existing", action="store_true")
    validate_package_parser = sub.add_parser("validate-package")
    validate_package_parser.add_argument("--run-dir", required=True, type=Path)
    validate_package_parser.add_argument("--development-run-dir", required=True, type=Path)
    authorize = sub.add_parser("authorize-audit")
    authorize.add_argument("--run-dir", required=True, type=Path)
    authorize.add_argument("--development-run-dir", required=True, type=Path)
    authorize.add_argument("--development-pass-freeze", required=True, type=Path)
    authorize.add_argument("--development-decision", required=True, type=Path)
    authorize.add_argument("--selector-receipt", required=True, type=Path)
    authorize.add_argument("--core-output", required=True, type=Path)
    authorize.add_argument("--outer-output", required=True, type=Path)
    validate_audit = sub.add_parser("validate-audit")
    validate_audit.add_argument("--run-dir", required=True, type=Path)
    validate_audit.add_argument("--development-run-dir", required=True, type=Path)
    launch = sub.add_parser("authorize-launch")
    launch.add_argument("--run-dir", required=True, type=Path)
    launch.add_argument("--development-run-dir", required=True, type=Path)
    launch.add_argument("--output", required=True, type=Path)
    validate_launch_parser = sub.add_parser("validate-launch")
    validate_launch_parser.add_argument("--run-dir", required=True, type=Path)
    validate_launch_parser.add_argument("--development-run-dir", required=True, type=Path)
    validate_launch_parser.add_argument("--authorization", required=True, type=Path)
    global_claim = sub.add_parser("global-claim")
    global_claim.add_argument("--run-dir", required=True, type=Path)
    global_claim.add_argument("--development-run-dir", required=True, type=Path)
    global_claim.add_argument("--output", required=True, type=Path)
    root_claim = sub.add_parser("root-claim")
    root_claim.add_argument("--run-dir", required=True, type=Path)
    root_claim.add_argument("--development-run-dir", required=True, type=Path)
    root_claim.add_argument("--shard", required=True, type=int)
    root_claim.add_argument("--global-claim", required=True, type=Path)
    root_claim.add_argument("--output", required=True, type=Path)
    run = sub.add_parser("run-shard")
    run.add_argument("--run-dir", required=True, type=Path)
    run.add_argument("--development-run-dir", required=True, type=Path)
    run.add_argument("--shard", required=True, type=int)
    run.add_argument("--directory", required=True, type=Path)
    complete = sub.add_parser("complete-shard")
    complete.add_argument("--run-dir", required=True, type=Path)
    complete.add_argument("--development-run-dir", required=True, type=Path)
    complete.add_argument("--shard", required=True, type=int)
    complete.add_argument("--directory", required=True, type=Path)
    resume = sub.add_parser("validate-resume-commit")
    resume.add_argument("--run-dir", required=True, type=Path)
    resume.add_argument("--development-run-dir", required=True, type=Path)
    resume.add_argument("--shard", required=True, type=int)
    resume.add_argument("--commit", required=True, type=Path)
    completed = sub.add_parser("validate-completed")
    completed.add_argument("--run-dir", required=True, type=Path)
    completed.add_argument("--development-run-dir", required=True, type=Path)
    completed.add_argument("--shard", required=True, type=int)
    completed.add_argument("--directory", required=True, type=Path)
    done = sub.add_parser("validate-done")
    done.add_argument("--run-dir", required=True, type=Path)
    done.add_argument("--authorization", required=True, type=Path)
    done.add_argument("--shard", required=True, type=int)
    done.add_argument("--done", required=True, type=Path)
    done_set = sub.add_parser("validate-done-set")
    done_set.add_argument("--run-dir", required=True, type=Path)
    done_set.add_argument("--authorization", required=True, type=Path)
    done_set.add_argument("--done-root", required=True, type=Path)
    claim = sub.add_parser("claim")
    claim.add_argument("--run-dir", required=True, type=Path)
    claim.add_argument("--authorization", required=True, type=Path)
    claim.add_argument("--done-root", required=True, type=Path)
    claim.add_argument("--output", required=True, type=Path)
    claims = sub.add_parser("validate-claims")
    claims.add_argument("--run-dir", required=True, type=Path)
    claims.add_argument("--authorization", required=True, type=Path)
    claims.add_argument("--local-claim", required=True, type=Path)
    claims.add_argument("--remote-claim", required=True, type=Path)
    received = sub.add_parser("audit-received")
    received.add_argument("--directory", required=True, type=Path)
    received.add_argument("--shard", required=True, type=int)
    received.add_argument("--run-dir", required=True, type=Path)
    received.add_argument("--development-run-dir", required=True, type=Path)
    received.add_argument("--authorization", required=True, type=Path)
    received.add_argument("--local-claim", required=True, type=Path)
    received.add_argument("--remote-claim", required=True, type=Path)
    received.add_argument("--output", required=True, type=Path)
    merge = sub.add_parser("merge-received")
    merge.add_argument("--received-root", required=True, type=Path)
    merge.add_argument("--audit-root", required=True, type=Path)
    merge.add_argument("--run-dir", required=True, type=Path)
    merge.add_argument("--authorization", required=True, type=Path)
    merge.add_argument("--local-claim", required=True, type=Path)
    merge.add_argument("--remote-claim", required=True, type=Path)
    merge.add_argument("--output", required=True, type=Path)
    merge.add_argument("--receipt", required=True, type=Path)
    validate_merge = sub.add_parser("validate-merged")
    validate_merge.add_argument("--run-dir", required=True, type=Path)
    validate_merge.add_argument("--authorization", required=True, type=Path)
    validate_merge.add_argument("--local-claim", required=True, type=Path)
    validate_merge.add_argument("--remote-claim", required=True, type=Path)
    validate_merge.add_argument("--merged", required=True, type=Path)
    validate_merge.add_argument("--receipt", required=True, type=Path)
    selector_claim = sub.add_parser("selector-claim")
    selector_claim.add_argument("--run-dir", required=True, type=Path)
    selector_claim.add_argument("--authorization", required=True, type=Path)
    selector_claim.add_argument("--local-consumption-claim", required=True, type=Path)
    selector_claim.add_argument("--remote-consumption-claim", required=True, type=Path)
    selector_claim.add_argument("--merged", required=True, type=Path)
    selector_claim.add_argument("--merge-receipt", required=True, type=Path)
    selector_claim.add_argument("--output", required=True, type=Path)
    select = sub.add_parser("select-once")
    select.add_argument("--run-dir", required=True, type=Path)
    select.add_argument("--development-run-dir", required=True, type=Path)
    select.add_argument("--selector-claim", required=True, type=Path)
    select.add_argument("--remote-selector-claim", required=True, type=Path)
    validate_selector = sub.add_parser("validate-selector")
    validate_selector.add_argument("--run-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = args.command
    if command == "package":
        result = package_audit50(
            repo_root=args.repo_root,
            run_dir=args.run_dir,
            run_name=args.run_name,
            development_run_dir=args.development_run_dir,
            audit_plan_path=args.audit_plan,
            resume_existing=args.resume_existing,
        )
    elif command == "validate-package":
        result = validate_package(
            args.run_dir, development_run_dir=args.development_run_dir
        )
    elif command == "authorize-audit":
        result = authorize_audit50(
            run_dir=args.run_dir,
            development_run_dir=args.development_run_dir,
            development_pass_freeze_path=args.development_pass_freeze,
            development_decision_path=args.development_decision,
            selector_receipt_path=args.selector_receipt,
            core_output=args.core_output,
            outer_output=args.outer_output,
        )
    elif command == "validate-audit":
        result = validate_audit_authorization(
            run_dir=args.run_dir, development_run_dir=args.development_run_dir
        )
    elif command == "authorize-launch":
        result = authorize_launch(
            run_dir=args.run_dir,
            development_run_dir=args.development_run_dir,
            output=args.output,
        )
    elif command == "validate-launch":
        result = validate_launch(
            run_dir=args.run_dir,
            development_run_dir=args.development_run_dir,
            authorization_path=args.authorization,
        )
    elif command == "global-claim":
        result = build_global_claim(
            run_dir=args.run_dir,
            development_run_dir=args.development_run_dir,
            output=args.output,
        )
    elif command == "root-claim":
        result = build_root_claim(
            run_dir=args.run_dir,
            development_run_dir=args.development_run_dir,
            shard=args.shard,
            global_claim_path=args.global_claim,
            output=args.output,
        )
    elif command == "run-shard":
        result = run_shard(
            run_dir=args.run_dir,
            development_run_dir=args.development_run_dir,
            shard=args.shard,
            directory=args.directory,
        )
    elif command == "complete-shard":
        result = complete_shard(
            run_dir=args.run_dir,
            development_run_dir=args.development_run_dir,
            shard=args.shard,
            directory=args.directory,
        )
    elif command == "validate-resume-commit":
        result = validate_resume_commit_file(
            run_dir=args.run_dir,
            development_run_dir=args.development_run_dir,
            shard=args.shard,
            commit_path=args.commit,
        )
    elif command == "validate-completed":
        result = validate_completed_bundle(
            run_dir=args.run_dir,
            development_run_dir=args.development_run_dir,
            shard=args.shard,
            directory=args.directory,
        )
    elif command == "validate-done":
        result = validate_done_file(
            run_dir=args.run_dir,
            authorization_path=args.authorization,
            shard=args.shard,
            done=args.done,
        )
    elif command == "validate-done-set":
        result = {
            "schema": "hu_m43_attempt08_audit50_done_set_validation_v1",
            "status": "exact_50_done_markers_valid",
            "done": len(
                validate_done_set(
                    run_dir=args.run_dir,
                    authorization_path=args.authorization,
                    done_root=args.done_root,
                )
            ),
            "content_opened": False,
        }
    elif command == "claim":
        result = claim_complete_output(
            run_dir=args.run_dir,
            authorization_path=args.authorization,
            done_root=args.done_root,
            output=args.output,
        )
    elif command == "validate-claims":
        claim, digest = validate_claim_pair(
            run_dir=args.run_dir,
            authorization_path=args.authorization,
            local_claim=args.local_claim,
            remote_claim=args.remote_claim,
        )
        result = {
            "schema": "hu_m43_attempt08_audit50_claim_pair_validation_v1",
            "status": "local_and_remote_consumption_claims_byte_identical",
            "run_name": claim["run_name"],
            "consumption_claim_sha256": digest,
            "selector_executed": False,
        }
    elif command == "audit-received":
        result = audit_received_shard(
            directory=args.directory,
            shard=args.shard,
            run_dir=args.run_dir,
            development_run_dir=args.development_run_dir,
            authorization_path=args.authorization,
            local_claim=args.local_claim,
            remote_claim=args.remote_claim,
            output=args.output,
        )
    elif command == "merge-received":
        result = merge_received(
            received_root=args.received_root,
            audit_root=args.audit_root,
            run_dir=args.run_dir,
            authorization_path=args.authorization,
            local_claim=args.local_claim,
            remote_claim=args.remote_claim,
            output=args.output,
            receipt=args.receipt,
        )
    elif command == "validate-merged":
        result = validate_merged(
            run_dir=args.run_dir,
            authorization_path=args.authorization,
            local_claim=args.local_claim,
            remote_claim=args.remote_claim,
            merged=args.merged,
            receipt=args.receipt,
        )
    elif command == "selector-claim":
        result = build_selector_claim(
            run_dir=args.run_dir,
            authorization_path=args.authorization,
            local_consumption_claim=args.local_consumption_claim,
            remote_consumption_claim=args.remote_consumption_claim,
            merged=args.merged,
            merge_receipt=args.merge_receipt,
            output=args.output,
        )
    elif command == "select-once":
        result = execute_selector_once(
            run_dir=args.run_dir,
            development_run_dir=args.development_run_dir,
            selector_claim_path=args.selector_claim,
            remote_selector_claim_path=args.remote_selector_claim,
        )
    elif command == "validate-selector":
        result = validate_selector_completion(run_dir=args.run_dir)
    else:  # pragma: no cover
        raise AssertionError(command)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIT50_PLAN_SHA256",
    "DONE_SCHEMA",
    "LAUNCH_AUTH_SCHEMA",
    "OUTER_AUTH_SCHEMA",
    "PACKAGE_SCHEMA",
    "authorize_audit50",
    "authorize_launch",
    "build_global_claim",
    "build_root_claim",
    "build_selector_claim",
    "claim_complete_output",
    "complete_shard",
    "execute_selector_once",
    "merge_received",
    "package_audit50",
    "run_shard",
    "validate_audit_authorization",
    "validate_claim_pair",
    "validate_completed_bundle",
    "validate_done_file",
    "validate_done_set",
    "validate_launch",
    "validate_merged",
    "validate_package",
    "validate_resume_commit_file",
    "validate_selector_completion",
]
