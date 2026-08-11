"""Small, immutable Spot lifecycle for Attempt09 search roots.

Heavy rollout work remains in the Python/Rust teacher stack.  This module only
freezes a known-good runtime package, creates a deterministic shard schedule,
authorizes a single mode, launches bounded Spot waves, and receives every shard
with one O(N) boundary validation pass.
"""

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

from .hu_m43_attempt09_contract import (
    AI_PROFILES_SHA256,
    ATTEMPT09_LAMBDA_MODEL_SHA256,
    M43_ATTEMPT09_PLAN_SHA256,
    M43_ATTEMPT09_PROFILES,
    enumerate_attempt09_seed_schedules,
    load_and_validate_attempt09_plan,
    validate_attempt09_artifact_bindings,
)
from .hu_infoset import ActorObservation
from .hu_m43_attempt09_teacher import (
    ATTEMPT09_TEACHER_SCHEMA,
    Attempt09TeacherConfig,
    validate_attempt09_teacher_output,
)
from .run_hu_m43_attempt09 import (
    ATTEMPT09_AUTHORIZATION_SCHEMA,
    ATTEMPT09_ROW_SCHEMA,
)


PACKAGE_SCHEMA = "hu_m43_attempt09_spot_package_v1"
LAUNCH_AUTHORIZATION_SCHEMA = "hu_m43_attempt09_spot_launch_authorization_v1"
DONE_SCHEMA = "hu_m43_attempt09_done_v1"
RECEIVE_SCHEMA = "hu_m43_attempt09_receive_v1"
SHARD_SCHEMA = "hu_m43_attempt09_spot_shard_v1"
LAUNCH_WAVE_SCHEMA = "hu_m43_attempt09_launch_wave_v1"
STATUS_SCHEMA = "hu_m43_attempt09_status_v1"
RECEIVED_SHARD_AUDIT_SCHEMA = "hu_m43_attempt09_received_shard_audit_v1"
PREFLIGHT_RESULT_SCHEMA = "hu_m43_attempt09_preflight_result_v1"
SOURCE_NAME = "ofc_regular_hu_m43_attempt09_source.zip"
SCHEDULE_NAME = "shards_manifest.jsonl"
STARTUP_NAME = "startup_hu_m43_attempt09.sh"
EXPECTED_IMAGE_NAME = "debian-12-bookworm-v20260609"
EXPECTED_IMAGE_ID = "1449487925682397051"
EXPECTED_IMAGE_SELF_LINK = (
    "https://www.googleapis.com/compute/v1/projects/debian-cloud/global/images/"
    "debian-12-bookworm-v20260609"
)
EXPECTED_MACHINE_TYPE = "c4-highmem-4"

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = _REPO_ROOT / "configs" / "hu_joint_policy_m43_attempt09.json"
DEFAULT_TEMPLATE_PACKAGE = (
    _REPO_ROOT
    / "outputs"
    / "gcp_runs"
    / "regular-hu-m43-attempt08-development200-finalprop-20260714-215952"
    / "package_src"
)
DEFAULT_STARTUP = _REPO_ROOT / "scripts" / STARTUP_NAME
PLAN_RELATIVE = "configs/hu_joint_policy_m43_attempt09.json"
GATE_RELATIVE = "artifacts/attempt09/preceding_gate.json"
OVERLAY_RELATIVES = (
    PLAN_RELATIVE,
    "src/ofc_regular/hu_m43_attempt09_contract.py",
    "src/ofc_regular/hu_m43_attempt09_teacher.py",
    "src/ofc_regular/run_hu_m43_attempt09.py",
    "outputs/hu_joint_policy/m43_attempt08_development/"
    "regular-hu-m43-attempt08-development200-finalprop-20260714-215952/"
    "selector/decision.json",
    "outputs/hu_joint_policy/m43_attempt08_development/"
    "regular-hu-m43-attempt08-development200-finalprop-20260714-215952/"
    "selector/decision_receipt.json",
)
EXPECTED_GATES = {
    "preflight": ("pass_local_correctness", "authorize_preflight_only"),
    "development": (
        "pass_correctness_preflight",
        "authorize_development200_package_only",
    ),
    "future_audit": ("go_freeze_attempt09_development", "go"),
}
_SAFE_RUN = re.compile(r"^[a-z0-9][a-z0-9-]{2,126}[a-z0-9]$")
_HEX = frozenset("0123456789abcdef")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def _write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable Attempt09 artifact exists: {path}")
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        temporary.unlink(missing_ok=True)
        raise
    temporary.unlink()


def _load_canonical(path: Path, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid UTF-8 JSON") from exc
    if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
        raise ValueError(f"{label} is not canonical JSON")
    return payload


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or not set(value) <= _HEX:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def build_schedule(mode: str, run_name: str) -> list[dict[str, Any]]:
    if not _SAFE_RUN.fullmatch(run_name):
        raise ValueError("Attempt09 run_name is not a safe GCP identity")
    specs: list[tuple[int, bool, str]]
    if mode == "preflight":
        specs = [
            (0, True, "root0_batch_a"),
            (0, True, "root0_batch_b"),
            (0, False, "root0_scalar"),
            (1, True, "root1_batch"),
            (2, True, "root2_batch"),
        ]
    elif mode == "development":
        specs = [(index, True, f"root{index:03d}") for index in range(200)]
    elif mode == "future_audit":
        specs = [(index, True, f"root{index:03d}") for index in range(200, 250)]
    else:
        raise ValueError("mode must be preflight, development, or future_audit")
    rows = []
    for shard, (root_index, batch, slot) in enumerate(specs):
        profile = M43_ATTEMPT09_PROFILES[
            root_index % len(M43_ATTEMPT09_PROFILES)
        ]
        # Identical root proofs must share the complete RNG identity.  The
        # scalar/batch implementation choice is recorded separately and must
        # never perturb sampled beliefs or downstream policy RNG.
        run_id_suffix = "parity" if mode == "preflight" else "search"
        rows.append(
            {
                "schema": SHARD_SCHEMA,
                "run_name": run_name,
                "mode": mode,
                "shard": shard,
                "root_index": root_index,
                "root_profile": profile,
                "batch_child_selectors": batch,
                "native_batch_threads": 4,
                "run_id": (
                    f"{run_name}:{mode}:root={root_index}:{run_id_suffix}"
                ),
                "output_prefix": f"shard-{shard:03d}-{slot}",
            }
        )
    return rows


def _copy_overlay(source: Path, package_root: Path, relative: str) -> dict[str, Any]:
    source_path = source / relative
    if not source_path.is_file() or source_path.is_symlink():
        raise ValueError(f"Attempt09 overlay is missing or unsafe: {relative}")
    destination = package_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    return {
        "path": relative.replace("\\", "/"),
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
    }


def _zip_tree(root: Path, output: Path) -> None:
    temporary = output.with_name(output.name + ".tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
            if path.is_symlink():
                raise ValueError(f"Attempt09 package contains a symlink: {path}")
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2026, 7, 14, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)
    os.replace(temporary, output)


def package_attempt09(
    *,
    mode: str,
    run_name: str,
    run_dir: str | Path,
    repository_root: str | Path = _REPO_ROOT,
    template_package: str | Path = DEFAULT_TEMPLATE_PACKAGE,
    plan: str | Path = DEFAULT_PLAN,
    startup: str | Path = DEFAULT_STARTUP,
    preceding_gate: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze a source package without opening a teacher root or invoking gcloud."""

    root = Path(repository_root).resolve()
    target = Path(run_dir).resolve()
    if target.exists():
        raise FileExistsError(f"Attempt09 run directory already exists: {target}")
    if target.parent != (root / "outputs" / "gcp_runs").resolve():
        raise ValueError("Attempt09 run_dir must be outputs/gcp_runs/<run_name>")
    if target.name != run_name:
        raise ValueError("Attempt09 run_dir basename must equal run_name")
    plan_path = Path(plan).resolve()
    plan_payload = load_and_validate_attempt09_plan(plan_path)
    validate_attempt09_artifact_bindings(plan_payload, repository_root=root)
    template = Path(template_package).resolve()
    if not template.is_dir() or template.is_symlink():
        raise ValueError("Attempt09 immutable runtime template is missing")
    startup_path = Path(startup).resolve()
    if not startup_path.is_file() or startup_path.is_symlink():
        raise ValueError("Attempt09 startup script is missing")
    if preceding_gate is None:
        raise ValueError(f"Attempt09 {mode} package requires its preceding gate")

    target.mkdir(parents=True)
    package_root = target / "package_src"
    try:
        shutil.copytree(template, package_root, symlinks=False)
        overlay_manifest = [
            _copy_overlay(root, package_root, name) for name in OVERLAY_RELATIVES
        ]
        gate_record = None
        if preceding_gate is not None:
            gate_source = Path(preceding_gate).resolve()
            gate_payload = _load_canonical(gate_source, "Attempt09 preceding gate")
            expected_gate = EXPECTED_GATES[mode]
            if (
                gate_payload.get("status") != expected_gate[0]
                or gate_payload.get("decision") != expected_gate[1]
                or gate_payload.get("current_profile_mutated") is not False
                or gate_payload.get("runtime_policy_activated") is not False
            ):
                raise ValueError(f"Attempt09 {mode} preceding gate did not pass")
            gate_destination = package_root / Path(GATE_RELATIVE)
            gate_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(gate_source, gate_destination)
            gate_record = {
                "path": GATE_RELATIVE,
                "sha256": sha256_file(gate_destination),
                "status": gate_payload.get("status"),
                "decision": gate_payload.get("decision"),
            }
        schedule = build_schedule(mode, run_name)
        schedule_path = target / SCHEDULE_NAME
        _write_once(schedule_path, _canonical_jsonl(schedule))
        frozen_startup = target / STARTUP_NAME
        shutil.copy2(startup_path, frozen_startup)
        source_path = target / SOURCE_NAME
        _zip_tree(package_root, source_path)
        template_manifest = template / "source_closure_manifest.json"
        manifest = {
            "schema": PACKAGE_SCHEMA,
            "status": "packaged_without_root_or_gcloud",
            "run_name": run_name,
            "mode": mode,
            "total_shards": len(schedule),
            "source_name": SOURCE_NAME,
            "source_sha256": sha256_file(source_path),
            "source_bytes": source_path.stat().st_size,
            "schedule_name": SCHEDULE_NAME,
            "schedule_sha256": sha256_file(schedule_path),
            "startup_name": STARTUP_NAME,
            "startup_sha256": sha256_file(frozen_startup),
            "plan_sha256": M43_ATTEMPT09_PLAN_SHA256,
            "ai_profiles_sha256": AI_PROFILES_SHA256,
            "model_sha256": ATTEMPT09_LAMBDA_MODEL_SHA256,
            "template_package": str(template),
            "template_source_closure_sha256": (
                sha256_file(template_manifest) if template_manifest.is_file() else None
            ),
            "overlays": overlay_manifest,
            "preceding_gate": gate_record,
            "image": {
                "project": "debian-cloud",
                "name": EXPECTED_IMAGE_NAME,
                "id": EXPECTED_IMAGE_ID,
                "self_link": EXPECTED_IMAGE_SELF_LINK,
            },
            "machine_type": EXPECTED_MACHINE_TYPE,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
            "teacher_executed": False,
            "gcloud_invoked": False,
            "created_unix_seconds": time.time(),
        }
        _write_once(target / "manifest.json", canonical_json_bytes(manifest))
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return manifest


def validate_package(run_dir: str | Path) -> dict[str, Any]:
    target = Path(run_dir).resolve()
    manifest = _load_canonical(target / "manifest.json", "Attempt09 manifest")
    expected_manifest_keys = {
        "schema", "status", "run_name", "mode", "total_shards",
        "source_name", "source_sha256", "source_bytes", "schedule_name",
        "schedule_sha256", "startup_name", "startup_sha256", "plan_sha256",
        "ai_profiles_sha256", "model_sha256", "template_package",
        "template_source_closure_sha256", "overlays", "preceding_gate", "image",
        "machine_type", "current_profile_mutated", "runtime_policy_activated",
        "teacher_executed", "gcloud_invoked", "created_unix_seconds",
    }
    if set(manifest) != expected_manifest_keys or manifest.get("schema") != PACKAGE_SCHEMA:
        raise ValueError("Attempt09 package schema changed")
    if target.name != manifest.get("run_name"):
        raise ValueError("Attempt09 package run name changed")
    mode = manifest.get("mode")
    if mode not in ("preflight", "development", "future_audit"):
        raise ValueError("Attempt09 package mode changed")
    if (
        manifest.get("status") != "packaged_without_root_or_gcloud"
        or manifest.get("source_name") != SOURCE_NAME
        or manifest.get("schedule_name") != SCHEDULE_NAME
        or manifest.get("startup_name") != STARTUP_NAME
        or manifest.get("plan_sha256") != M43_ATTEMPT09_PLAN_SHA256
        or manifest.get("ai_profiles_sha256") != AI_PROFILES_SHA256
        or manifest.get("model_sha256") != ATTEMPT09_LAMBDA_MODEL_SHA256
        or manifest.get("machine_type") != EXPECTED_MACHINE_TYPE
        or manifest.get("current_profile_mutated") is not False
        or manifest.get("runtime_policy_activated") is not False
        or manifest.get("teacher_executed") is not False
        or manifest.get("gcloud_invoked") is not False
    ):
        raise ValueError("Attempt09 frozen package boundary changed")
    created = manifest.get("created_unix_seconds")
    if (
        isinstance(created, bool)
        or not isinstance(created, (int, float))
        or not math.isfinite(float(created))
        or float(created) < 0.0
    ):
        raise ValueError("Attempt09 package timestamp changed")
    image = manifest.get("image")
    if image != {
        "project": "debian-cloud",
        "name": EXPECTED_IMAGE_NAME,
        "id": EXPECTED_IMAGE_ID,
        "self_link": EXPECTED_IMAGE_SELF_LINK,
    }:
        raise ValueError("Attempt09 frozen image boundary changed")
    expected = {
        target / SOURCE_NAME: manifest.get("source_sha256"),
        target / SCHEDULE_NAME: manifest.get("schedule_sha256"),
        target / STARTUP_NAME: manifest.get("startup_sha256"),
    }
    for path, digest in expected.items():
        if not path.is_file() or path.is_symlink() or sha256_file(path) != _require_sha256(digest, path.name):
            raise ValueError(f"Attempt09 frozen package changed: {path.name}")
    source_bytes = manifest.get("source_bytes")
    if type(source_bytes) is not int or source_bytes != (target / SOURCE_NAME).stat().st_size:
        raise ValueError("Attempt09 frozen source size changed")
    schedule_raw = (target / SCHEDULE_NAME).read_bytes()
    try:
        rows = [json.loads(line) for line in schedule_raw.decode("utf-8").splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Attempt09 schedule is invalid") from exc
    if schedule_raw != _canonical_jsonl(rows):
        raise ValueError("Attempt09 schedule is not canonical JSONL")
    if rows != build_schedule(str(manifest["mode"]), str(manifest["run_name"])):
        raise ValueError("Attempt09 shard schedule changed")
    if len(rows) != manifest.get("total_shards"):
        raise ValueError("Attempt09 shard count changed")
    overlays = manifest.get("overlays")
    required_overlays = set(OVERLAY_RELATIVES)
    if not isinstance(overlays, list) or len(overlays) != len(required_overlays):
        raise ValueError("Attempt09 overlay closure changed")
    seen: set[str] = set()
    package_root = target / "package_src"
    for record in overlays:
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256", "bytes"}:
            raise ValueError("Attempt09 overlay record changed")
        relative = record.get("path")
        if not isinstance(relative, str) or relative in seen or relative not in required_overlays:
            raise ValueError("Attempt09 overlay path changed")
        seen.add(relative)
        overlay_path = package_root / Path(relative)
        if (
            not overlay_path.is_file()
            or overlay_path.is_symlink()
            or sha256_file(overlay_path) != _require_sha256(record.get("sha256"), relative)
            or overlay_path.stat().st_size != record.get("bytes")
        ):
            raise ValueError(f"Attempt09 overlay bytes changed: {relative}")
    if seen != required_overlays:
        raise ValueError("Attempt09 overlay exact set changed")
    frozen_plan = load_and_validate_attempt09_plan(
        package_root / Path(PLAN_RELATIVE)
    )
    validate_attempt09_artifact_bindings(
        frozen_plan, repository_root=package_root
    )
    closure_hash = manifest.get("template_source_closure_sha256")
    if closure_hash is not None:
        _require_sha256(closure_hash, "template source closure")
    gate = manifest.get("preceding_gate")
    if not isinstance(gate, Mapping) or set(gate) != {
        "path", "sha256", "status", "decision"
    }:
        raise ValueError("Attempt09 package gate record changed")
    gate_path = package_root / Path(GATE_RELATIVE)
    if (
        gate.get("path") != GATE_RELATIVE
        or not gate_path.is_file()
        or gate_path.is_symlink()
        or sha256_file(gate_path)
        != _require_sha256(gate.get("sha256"), "preceding gate")
    ):
        raise ValueError("Attempt09 preceding gate bytes changed")
    gate_payload = _load_canonical(gate_path, "Attempt09 packaged preceding gate")
    expected_gate = EXPECTED_GATES[mode]
    if (
        gate.get("status") != gate_payload.get("status")
        or gate.get("decision") != gate_payload.get("decision")
        or (gate_payload.get("status"), gate_payload.get("decision"))
        != expected_gate
        or gate_payload.get("current_profile_mutated") is not False
        or gate_payload.get("runtime_policy_activated") is not False
    ):
        raise ValueError("Attempt09 preceding gate summary changed")
    return manifest


def authorize_launch(
    *, run_dir: str | Path, output: str | Path | None = None
) -> dict[str, Any]:
    target = Path(run_dir).resolve()
    manifest = validate_package(target)
    destination = Path(output) if output is not None else target / "launch_authorization.json"
    if destination.resolve() != (target / "launch_authorization.json").resolve():
        raise ValueError("Attempt09 launch authorization path changed")
    mode = str(manifest["mode"])
    gate = manifest.get("preceding_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("Attempt09 phase lost its preceding gate")
    authorization = {
        "schema": LAUNCH_AUTHORIZATION_SCHEMA,
        "status": "authorized",
        "run_name": manifest["run_name"],
        "mode": mode,
        "manifest_sha256": sha256_file(target / "manifest.json"),
        "source_sha256": manifest["source_sha256"],
        "schedule_sha256": manifest["schedule_sha256"],
        "startup_sha256": manifest["startup_sha256"],
        "preceding_gate_sha256": gate.get("sha256") if isinstance(gate, Mapping) else None,
        "total_shards": manifest["total_shards"],
        "spot_authorized": True,
        "root_execution_started": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "authorized_unix_seconds": time.time(),
    }
    _write_once(destination, canonical_json_bytes(authorization))
    if mode != "preflight":
        first, last = (0, 199) if mode == "development" else (200, 249)
        execution = {
            "schema": ATTEMPT09_AUTHORIZATION_SCHEMA,
            "status": "authorized",
            "mode": mode,
            "plan_sha256": M43_ATTEMPT09_PLAN_SHA256,
            "source_package_sha256": manifest["source_sha256"],
            "preceding_gate_artifact": GATE_RELATIVE,
            "preceding_gate_sha256": gate["sha256"],
            "root_index_first": first,
            "root_index_last": last,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        }
        _write_once(
            target / "execution_authorization.json", canonical_json_bytes(execution)
        )
    return authorization


def validate_launch(run_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    target = Path(run_dir).resolve()
    manifest = validate_package(target)
    auth = _load_canonical(
        target / "launch_authorization.json", "Attempt09 launch authorization"
    )
    expected_auth_keys = {
        "schema", "status", "run_name", "mode", "manifest_sha256",
        "source_sha256", "schedule_sha256", "startup_sha256",
        "preceding_gate_sha256", "total_shards", "spot_authorized",
        "root_execution_started", "current_profile_mutated",
        "runtime_policy_activated", "authorized_unix_seconds",
    }
    authorized = auth.get("authorized_unix_seconds")
    gate = manifest.get("preceding_gate")
    if (
        set(auth) != expected_auth_keys
        or auth.get("schema") != LAUNCH_AUTHORIZATION_SCHEMA
        or auth.get("status") != "authorized"
        or auth.get("run_name") != manifest.get("run_name")
        or auth.get("mode") != manifest.get("mode")
        or auth.get("manifest_sha256") != sha256_file(target / "manifest.json")
        or auth.get("source_sha256") != manifest.get("source_sha256")
        or auth.get("schedule_sha256") != manifest.get("schedule_sha256")
        or auth.get("startup_sha256") != manifest.get("startup_sha256")
        or auth.get("preceding_gate_sha256")
        != (gate.get("sha256") if isinstance(gate, Mapping) else None)
        or auth.get("total_shards") != manifest.get("total_shards")
        or auth.get("spot_authorized") is not True
        or auth.get("root_execution_started") is not False
        or auth.get("current_profile_mutated") is not False
        or auth.get("runtime_policy_activated") is not False
        or isinstance(authorized, bool)
        or not isinstance(authorized, (int, float))
        or not math.isfinite(float(authorized))
        or float(authorized) < 0.0
    ):
        raise ValueError("Attempt09 launch authorization changed")
    execution_path = target / "execution_authorization.json"
    if manifest.get("mode") == "preflight":
        if execution_path.exists():
            raise ValueError("Attempt09 preflight unexpectedly has execution authorization")
    else:
        execution = _load_canonical(
            execution_path, "Attempt09 execution authorization"
        )
        first, last = (
            (0, 199)
            if manifest.get("mode") == "development"
            else (200, 249)
        )
        expected_execution = {
            "schema": ATTEMPT09_AUTHORIZATION_SCHEMA,
            "status": "authorized",
            "mode": manifest["mode"],
            "plan_sha256": M43_ATTEMPT09_PLAN_SHA256,
            "source_package_sha256": manifest["source_sha256"],
            "preceding_gate_artifact": GATE_RELATIVE,
            "preceding_gate_sha256": gate["sha256"],
            "root_index_first": first,
            "root_index_last": last,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        }
        if execution != expected_execution:
            raise ValueError("Attempt09 execution authorization changed")
    return manifest, auth


def _subprocess_run(
    command: Sequence[str], **kwargs: Any
) -> subprocess.CompletedProcess[str]:
    """Run a command, resolving the Windows gcloud.CMD shim when necessary."""

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


def _run(command: Sequence[str], *, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    completed = _subprocess_run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    return completed


def _parse_shards(values: Sequence[str], maximum: int) -> list[int]:
    selected: set[int] = set()
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                left, right = token.split("-", 1)
                start, stop = int(left), int(right)
                if start > stop:
                    raise ValueError("Attempt09 shard range is reversed")
                expanded = range(start, stop + 1)
            else:
                expanded = (int(token),)
            for shard in expanded:
                if shard in selected:
                    raise ValueError(f"Attempt09 duplicate shard selection: {shard}")
                selected.add(shard)
    result = sorted(selected)
    if not result or result[0] < 0 or result[-1] >= maximum:
        raise ValueError("Attempt09 shard selector is outside the frozen schedule")
    if len(result) > 25:
        raise ValueError("Attempt09 launches at most 25 shards per wave")
    return result


def launch_wave(
    *,
    run_dir: str | Path,
    project: str,
    bucket: str,
    zone: str,
    shards: Sequence[str],
    no_self_delete: bool = False,
) -> dict[str, Any]:
    """Publish immutable inputs and create at most 25 Spot instances."""

    target = Path(run_dir).resolve()
    manifest, authorization = validate_launch(target)
    schedule_raw = (target / SCHEDULE_NAME).read_bytes()
    schedule = [json.loads(line) for line in schedule_raw.decode("utf-8").splitlines()]
    selection = _parse_shards(shards, len(schedule))
    image = json.loads(
        _run(
            [
                "gcloud", "compute", "images", "describe", EXPECTED_IMAGE_NAME,
                "--project", "debian-cloud", "--format=json",
            ],
            timeout=120,
        ).stdout
    )
    if (
        str(image.get("id")) != EXPECTED_IMAGE_ID
        or image.get("selfLink") != EXPECTED_IMAGE_SELF_LINK
    ):
        raise ValueError("Attempt09 immutable Debian image identity changed")
    prefix = f"gs://{bucket}/runs/{manifest['run_name']}"
    publishes = [
        (target / "manifest.json", f"{prefix}/manifest.json"),
        (target / SOURCE_NAME, f"{prefix}/source/{SOURCE_NAME}"),
        (target / SCHEDULE_NAME, f"{prefix}/source/{SCHEDULE_NAME}"),
        (target / "launch_authorization.json", f"{prefix}/source/launch_authorization.json"),
    ]
    execution = target / "execution_authorization.json"
    if execution.is_file():
        publishes.append((execution, f"{prefix}/source/execution_authorization.json"))
    for source, uri in publishes:
        result = _subprocess_run(
            [
                "gcloud", "storage", "cp", str(source), uri, "--project", project,
                "--if-generation-match=0",
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
        )
        if result.returncode != 0:
            with tempfile.TemporaryDirectory() as directory:
                copy = Path(directory) / source.name
                _run(["gcloud", "storage", "cp", uri, str(copy), "--project", project], timeout=600)
                if sha256_file(copy) != sha256_file(source):
                    raise RuntimeError(f"Attempt09 immutable GCS object differs: {uri}")
    vm_prefix = re.sub(r"[^a-z0-9-]", "-", str(manifest["run_name"]))[-45:].strip("-")
    created = []
    for shard in selection:
        spec = schedule[shard]
        done_uri = f"{prefix}/results/{spec['output_prefix']}/DONE.json"
        exists = _subprocess_run(
            ["gcloud", "storage", "objects", "describe", done_uri, "--project", project],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        )
        if exists.returncode == 0:
            raise FileExistsError(f"Attempt09 DONE already exists for shard {shard}")
        instance = f"{vm_prefix}-s{shard:03d}"[-63:]
        absent = _subprocess_run(
            [
                "gcloud", "compute", "instances", "describe", instance,
                "--zone", zone, "--project", project, "--format=json",
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        )
        if absent.returncode == 0:
            raise FileExistsError(f"Attempt09 instance already exists: {instance}")
        auth_hash = (
            sha256_file(execution) if execution.is_file() else "none"
        )
        metadata = ",".join(
            [
                f"PROJECT_ID={project}", f"BUCKET={bucket}",
                f"RUN_NAME={manifest['run_name']}", f"SHARD={shard}",
                f"SOURCE_URI={prefix}/source/{SOURCE_NAME}",
                f"SOURCE_SHA256={manifest['source_sha256']}",
                f"MANIFEST_SHA256={authorization['manifest_sha256']}",
                f"SCHEDULE_SHA256={manifest['schedule_sha256']}",
                f"AUTHORIZATION_SHA256={auth_hash}",
                f"SELF_DELETE={0 if no_self_delete else 1}",
            ]
        )
        _run(
            [
                "gcloud", "compute", "instances", "create", instance,
                "--project", project, "--zone", zone,
                "--machine-type", EXPECTED_MACHINE_TYPE,
                "--provisioning-model=SPOT", "--instance-termination-action=DELETE",
                "--image-project=debian-cloud", f"--image={EXPECTED_IMAGE_NAME}",
                "--boot-disk-size=50GB", "--boot-disk-type=hyperdisk-balanced",
                "--scopes=https://www.googleapis.com/auth/cloud-platform",
                f"--metadata={metadata}",
                f"--metadata-from-file=startup-script={target / STARTUP_NAME}",
                "--quiet",
            ],
            timeout=300,
        )
        created.append({"shard": shard, "instance": instance})
    return {
        "schema": LAUNCH_WAVE_SCHEMA,
        "status": "created",
        "run_name": manifest["run_name"],
        "mode": manifest["mode"],
        "created": created,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }


def run_status(
    *, run_dir: str | Path, project: str, bucket: str, zone: str
) -> dict[str, Any]:
    manifest = validate_package(run_dir)
    prefix = f"gs://{bucket}/runs/{manifest['run_name']}"
    listing = _subprocess_run(
        ["gcloud", "storage", "ls", "--recursive", f"{prefix}/results", "--project", project],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
    )
    objects = listing.stdout.splitlines() if listing.returncode == 0 else []
    done = sorted(uri for uri in objects if uri.endswith("/DONE.json"))
    vm_prefix = re.sub(r"[^a-z0-9-]", "-", str(manifest["run_name"]))[-45:].strip("-")
    instances_raw = _run(
        [
            "gcloud", "compute", "instances", "list", "--project", project,
            "--filter", f"zone:({zone}) AND name~'{vm_prefix}'", "--format=json",
        ],
        timeout=120,
    ).stdout
    instances = json.loads(instances_raw or "[]")
    return {
        "schema": STATUS_SCHEMA,
        "run_name": manifest["run_name"],
        "mode": manifest["mode"],
        "done": len(done),
        "total": manifest["total_shards"],
        "remaining": int(manifest["total_shards"]) - len(done),
        "instances": [
            {"name": item.get("name"), "status": item.get("status")} for item in instances
        ],
        "complete": len(done) == manifest["total_shards"],
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }


def _reject_hidden(value: Any, path: str = "row") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized == "opponent_private_discard_input_allowed":
                if child is not False:
                    raise ValueError(f"Attempt09 hidden information enabled at {path}.{key}")
            elif (
                "opponent" in normalized
                and "discard" in normalized
                and ("private" in normalized or "hidden" in normalized)
            ):
                raise ValueError(f"Attempt09 hidden information at {path}.{key}")
            _reject_hidden(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_hidden(child, f"{path}[{index}]")


def _validate_received_shard(
    directory: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    schedule_sha256: str,
    authorization_sha256: str,
    spec: Mapping[str, Any],
    expected_seeds: Mapping[str, int],
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    expected_names = {
        "DONE.json", "teacher.jsonl", "checkpoint.json", "heartbeat.json",
        "generator_summary.json", "run.log", "time.txt",
    }
    entries = list(directory.iterdir())
    actual_names = {path.name for path in entries}
    if (
        actual_names != expected_names
        or any(not path.is_file() or path.is_symlink() for path in entries)
    ):
        raise ValueError(
            f"Attempt09 shard {spec['shard']} files changed: "
            f"missing={sorted(expected_names-actual_names)},extra={sorted(actual_names-expected_names)}"
        )
    done = _load_canonical(directory / "DONE.json", "Attempt09 DONE")
    expected_done_keys = {
        "schema", "status", "run_name", "mode", "shard", "root_index",
        "output_prefix", "source_sha256", "manifest_sha256", "schedule_sha256",
        "authorization_sha256", "files", "current_profile_mutated",
        "runtime_policy_activated", "completed_unix_seconds",
    }
    if set(done) != expected_done_keys:
        raise ValueError(f"Attempt09 DONE fields changed at shard {spec['shard']}")
    if (
        done.get("schema") != DONE_SCHEMA
        or done.get("status") != "complete"
        or done.get("run_name") != manifest["run_name"]
        or done.get("mode") != manifest["mode"]
        or done.get("shard") != spec["shard"]
        or done.get("root_index") != spec["root_index"]
        or done.get("output_prefix") != spec["output_prefix"]
        or done.get("source_sha256") != manifest["source_sha256"]
        or done.get("manifest_sha256") != manifest_sha256
        or done.get("schedule_sha256") != schedule_sha256
        or done.get("authorization_sha256") != authorization_sha256
        or done.get("current_profile_mutated") is not False
        or done.get("runtime_policy_activated") is not False
    ):
        raise ValueError(f"Attempt09 DONE identity changed at shard {spec['shard']}")
    completed = done.get("completed_unix_seconds")
    if (
        isinstance(completed, bool)
        or not isinstance(completed, (int, float))
        or not math.isfinite(float(completed))
        or float(completed) < 0.0
    ):
        raise ValueError(f"Attempt09 DONE timestamp changed at shard {spec['shard']}")
    files = done.get("files")
    if not isinstance(files, Mapping) or set(files) != expected_names - {"DONE.json"}:
        raise ValueError(f"Attempt09 DONE content manifest changed at shard {spec['shard']}")
    for name, record in files.items():
        if not isinstance(record, Mapping) or set(record) != {"sha256", "bytes"}:
            raise ValueError(f"Attempt09 DONE file record changed: {name}")
        path = directory / name
        size = record.get("bytes")
        if (
            sha256_file(path) != _require_sha256(record.get("sha256"), name)
            or type(size) is not int
            or size < 0
            or path.stat().st_size != size
        ):
            raise ValueError(f"Attempt09 received content changed: {name}")
    raw = (directory / "teacher.jsonl").read_bytes()
    lines = raw.splitlines()
    if len(lines) != 1 or raw != lines[0] + b"\n" or b"\r" in raw:
        raise ValueError(f"Attempt09 shard {spec['shard']} is not one LF JSONL row")
    try:
        row = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Attempt09 shard {spec['shard']} row is invalid") from exc
    if not isinstance(row, dict) or raw != canonical_json_bytes(row):
        raise ValueError(f"Attempt09 shard {spec['shard']} row is not canonical")
    _reject_hidden(row)
    provenance = row.get("provenance")
    teacher = row.get("teacher")
    expected_authorization = None if authorization_sha256 == "none" else authorization_sha256
    observation_payload = row.get("policy_observation")
    if (
        row.get("schema") != ATTEMPT09_ROW_SCHEMA
        or row.get("root_index") != spec["root_index"]
        or row.get("hand_seed") != expected_seeds["hand"]
        or row.get("root_profile") != spec["root_profile"]
        or not isinstance(observation_payload, Mapping)
        or not isinstance(row.get("baseline_action_key"), str)
        or not isinstance(provenance, Mapping)
        or provenance.get("mode") != manifest["mode"]
        or provenance.get("run_id") != spec["run_id"]
        or provenance.get("root_index") != spec["root_index"]
        or provenance.get("root_profile") != spec["root_profile"]
        or provenance.get("seeds") != dict(expected_seeds)
        or provenance.get("plan_sha256") != M43_ATTEMPT09_PLAN_SHA256
        or provenance.get("ai_profiles_sha256") != AI_PROFILES_SHA256
        or provenance.get("model_sha256") != ATTEMPT09_LAMBDA_MODEL_SHA256
        or provenance.get("batch_child_selectors") is not spec["batch_child_selectors"]
        or provenance.get("native_batch_threads") != 4
        or provenance.get("authorization_sha256") != expected_authorization
        or provenance.get("source_package_sha256")
        != (None if authorization_sha256 == "none" else manifest["source_sha256"])
        or provenance.get("current_profile_resolved") is not False
        or provenance.get("opponent_private_discard_input_allowed") is not False
        or provenance.get("runtime_activation_allowed") is not False
        or not isinstance(teacher, Mapping)
        or teacher.get("schema") != ATTEMPT09_TEACHER_SCHEMA
        or teacher.get("policy_observation") != observation_payload
        or teacher.get("baseline_action_key") != row.get("baseline_action_key")
        or teacher.get("current_profile_resolved") is not False
        or teacher.get("runtime_gate_allowed") is not False
    ):
        raise ValueError(f"Attempt09 shard {spec['shard']} row boundary changed")
    observation = ActorObservation.from_dict(observation_payload)
    if observation.to_dict() != dict(observation_payload):
        raise ValueError(f"Attempt09 shard {spec['shard']} observation changed")
    teacher_config = Attempt09TeacherConfig(
        frozen_model_sha256=ATTEMPT09_LAMBDA_MODEL_SHA256,
        hand_seed=expected_seeds["hand"],
        rerank_seed=expected_seeds["rerank"],
        veto_seed=expected_seeds["veto"],
        stress_seed=expected_seeds["stress"],
        confirmation_seed=expected_seeds["confirmation"],
        evaluation_seed=expected_seeds["evaluation"],
        child_policy_seed=expected_seeds["child"],
        run_id=(
            f"{spec['run_id']}:root={spec['root_index']}:"
            f"seed={expected_seeds['hand']}:obs={observation.fingerprint()}"
        ),
        batch_child_selectors=bool(spec["batch_child_selectors"]),
    )
    validate_attempt09_teacher_output(
        observation,
        baseline_action_key=str(row["baseline_action_key"]),
        payload=teacher,
        config=teacher_config,
    )
    summary = _load_canonical(directory / "generator_summary.json", "Attempt09 summary")
    if (
        summary.get("status") != "complete"
        or summary.get("root_index") != spec["root_index"]
        or summary.get("current_profile_mutated") is not False
        or summary.get("runtime_policy_activated") is not False
    ):
        raise ValueError(f"Attempt09 shard {spec['shard']} summary changed")
    audit = {
        "schema": RECEIVED_SHARD_AUDIT_SCHEMA,
        "status": "pass",
        "shard": spec["shard"],
        "root_index": spec["root_index"],
        "root_profile": spec["root_profile"],
        "output_prefix": spec["output_prefix"],
        "done_sha256": sha256_file(directory / "DONE.json"),
        "teacher_sha256": sha256_file(directory / "teacher.jsonl"),
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    return row, raw, audit


def receive_run(
    *,
    run_dir: str | Path,
    project: str,
    bucket: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Download once, validate every shard in one pass, and atomically merge."""

    source = Path(run_dir).resolve()
    manifest, _launch = validate_launch(source)
    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"Attempt09 receive target already exists: {target}")
    stage = target.with_name(target.name + f".staging-{os.getpid()}")
    if stage.exists():
        raise FileExistsError(f"Attempt09 receive staging already exists: {stage}")
    stage.mkdir(parents=True)
    try:
        shards_root = stage / "shards"
        shards_root.mkdir()
        prefix = f"gs://{bucket}/runs/{manifest['run_name']}/results"
        _run(
            [
                "gcloud", "storage", "rsync", "--recursive", prefix,
                str(shards_root), "--project", project,
            ],
            timeout=7200,
        )
        schedule = [
            json.loads(line)
            for line in (source / SCHEDULE_NAME).read_text(encoding="utf-8").splitlines()
        ]
        root_entries = list(shards_root.iterdir())
        actual_directories = {path.name for path in root_entries}
        expected_directories = {str(spec["output_prefix"]) for spec in schedule}
        if (
            actual_directories != expected_directories
            or any(not path.is_dir() or path.is_symlink() for path in root_entries)
        ):
            raise ValueError(
                "Attempt09 received shard set changed: "
                f"missing={sorted(expected_directories-actual_directories)},"
                f"extra={sorted(actual_directories-expected_directories)}"
            )
        manifest_hash = sha256_file(source / "manifest.json")
        schedule_hash = sha256_file(source / SCHEDULE_NAME)
        execution = source / "execution_authorization.json"
        authorization_hash = sha256_file(execution) if execution.is_file() else "none"
        frozen_repository = source / "package_src"
        plan_payload = load_and_validate_attempt09_plan(
            frozen_repository / Path(PLAN_RELATIVE)
        )
        validate_attempt09_artifact_bindings(
            plan_payload, repository_root=frozen_repository
        )
        seed_schedules = enumerate_attempt09_seed_schedules(
            plan_payload, population=str(manifest["mode"])
        )
        if tuple(seed_schedules) != (
            "hand", "rerank", "veto", "stress", "confirmation", "evaluation", "child"
        ):
            raise ValueError("Attempt09 received seed-domain order changed")
        preflight_indices = plan_payload["preflight_seed_contract"][
            "source_root_indices"
        ]

        def seed_offset(spec: Mapping[str, Any]) -> int:
            root_index = int(spec["root_index"])
            if manifest["mode"] == "preflight":
                return list(preflight_indices).index(root_index)
            if manifest["mode"] == "development":
                return root_index
            return root_index - 200

        rows: list[dict[str, Any]] = []
        raw_rows: list[bytes] = []
        audits: list[dict[str, Any]] = []
        audit_root = stage / "audits"
        audit_root.mkdir()
        # Boundary invariants above are built once.  This loop is strictly O(N).
        for spec in schedule:
            offset = seed_offset(spec)
            expected_seeds = {
                domain: int(values[offset])
                for domain, values in seed_schedules.items()
            }
            row, raw, audit = _validate_received_shard(
                shards_root / str(spec["output_prefix"]),
                manifest=manifest,
                manifest_sha256=manifest_hash,
                schedule_sha256=schedule_hash,
                authorization_sha256=authorization_hash,
                spec=spec,
                expected_seeds=expected_seeds,
            )
            rows.append(row)
            raw_rows.append(raw)
            audits.append(audit)
            _write_once(
                audit_root / f"shard-{int(spec['shard']):03d}.json",
                canonical_json_bytes(audit),
            )
        merged = b"".join(raw_rows)
        merged_root = stage / "merged"
        merged_root.mkdir()
        _write_once(merged_root / "teacher.jsonl", merged)
        receipt = {
            "schema": RECEIVE_SCHEMA,
            "status": "complete",
            "run_name": manifest["run_name"],
            "mode": manifest["mode"],
            "roots": len(rows),
            "root_indices": [row["root_index"] for row in rows],
            "profiles": [row["root_profile"] for row in rows],
            "manifest_sha256": manifest_hash,
            "schedule_sha256": schedule_hash,
            "source_sha256": manifest["source_sha256"],
            "authorization_sha256": authorization_hash,
            "merged_sha256": hashlib.sha256(merged).hexdigest(),
            "audit_sha256": hashlib.sha256(_canonical_jsonl(audits)).hexdigest(),
            "batch_boundary_validation_count": 1,
            "per_shard_boundary_revalidation_count": 0,
            "selector_executed": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        }
        _write_once(merged_root / "receive_receipt.json", canonical_json_bytes(receipt))
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return receipt


def finalize_preflight(
    *, received_dir: str | Path, output: str | Path
) -> dict[str, Any]:
    root = Path(received_dir).resolve()
    receipt = _load_canonical(root / "merged" / "receive_receipt.json", "preflight receipt")
    expected_root_indices = [0, 0, 0, 1, 2]
    expected_profiles = [
        M43_ATTEMPT09_PROFILES[index % len(M43_ATTEMPT09_PROFILES)]
        for index in expected_root_indices
    ]
    if (
        receipt.get("schema") != RECEIVE_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("mode") != "preflight"
        or receipt.get("roots") != 5
        or receipt.get("root_indices") != expected_root_indices
        or receipt.get("profiles") != expected_profiles
        or receipt.get("batch_boundary_validation_count") != 1
        or receipt.get("per_shard_boundary_revalidation_count") != 0
        or receipt.get("selector_executed") is not False
        or receipt.get("current_profile_mutated") is not False
        or receipt.get("runtime_policy_activated") is not False
    ):
        raise ValueError("Attempt09 preflight requires the exact five proof slots")
    merged_path = root / "merged" / "teacher.jsonl"
    if sha256_file(merged_path) != _require_sha256(
        receipt.get("merged_sha256"), "preflight merged"
    ):
        raise ValueError("Attempt09 preflight merged bytes changed")
    root_entries = {path.name for path in root.iterdir()}
    if root_entries != {"shards", "audits", "merged"}:
        raise ValueError("Attempt09 preflight receive root changed")
    audit_paths = sorted((root / "audits").glob("shard-*.json"))
    if [path.name for path in audit_paths] != [
        f"shard-{index:03d}.json" for index in range(5)
    ]:
        raise ValueError("Attempt09 preflight audit exact set changed")
    audit_payloads = [
        _load_canonical(path, f"Attempt09 preflight audit {index}")
        for index, path in enumerate(audit_paths)
    ]
    if hashlib.sha256(_canonical_jsonl(audit_payloads)).hexdigest() != _require_sha256(
        receipt.get("audit_sha256"), "preflight audits"
    ):
        raise ValueError("Attempt09 preflight audit bytes changed")
    shard_rows = []
    for shard in range(5):
        directories = sorted((root / "shards").glob(f"shard-{shard:03d}-*"))
        if len(directories) != 1:
            raise ValueError(f"Attempt09 preflight shard {shard} is missing")
        raw = (directories[0] / "teacher.jsonl").read_bytes()
        shard_rows.append(json.loads(raw))
    teacher_a = shard_rows[0]["teacher"]
    teacher_b = shard_rows[1]["teacher"]
    deterministic = canonical_json_bytes(teacher_a) == canonical_json_bytes(teacher_b)
    if not deterministic:
        raise ValueError("Attempt09 repeated batch root is not byte-identical")
    import copy
    batch = copy.deepcopy(teacher_a)
    scalar = copy.deepcopy(shard_rows[2]["teacher"])
    batch["search_config"]["batch_child_selectors"] = False
    scalar["search_config"]["batch_child_selectors"] = False
    scalar_batch = canonical_json_bytes(batch) == canonical_json_bytes(scalar)
    if not scalar_batch:
        raise ValueError("Attempt09 scalar/batch teacher payloads differ")
    inherited = (
        _REPO_ROOT
        / "outputs" / "hu_joint_policy" / "m43_attempt08_preflight"
        / "regular-hu-m43-attempt08-preflight-finalprop-20260714-213558"
        / "finalization.json"
    )
    inherited_hash = "681854a8cd37bec1abf6f0ff72e69a8e09e2ad257fe2ecfbea513d3f127dd1b3"
    if not inherited.is_file() or sha256_file(inherited) != inherited_hash:
        raise ValueError("Attempt09 inherited exact-engine parity evidence changed")
    inherited_payload = _load_canonical(inherited, "Attempt08 exact parity finalization")
    if inherited_payload.get("status") != "pass_correctness_preflight_and_authorize_development200_only":
        raise ValueError("Attempt09 inherited exact-engine parity did not pass")
    elapsed = [float(row["provenance"]["elapsed_seconds"]) for row in shard_rows]
    result = {
        "schema": PREFLIGHT_RESULT_SCHEMA,
        "status": "pass_correctness_preflight",
        "decision": "authorize_development200_package_only",
        "received_receipt_sha256": sha256_file(root / "merged" / "receive_receipt.json"),
        "deterministic_batch_repeat": deterministic,
        "scalar_batch_teacher_parity": scalar_batch,
        "action_mapping_validated": True,
        "hidden_information_violations": 0,
        "rng_domain_overlap_violations": 0,
        "inherited_t3_t4_exact_parity_sha256": inherited_hash,
        "latency_seconds": elapsed,
        "latency_max_seconds": max(elapsed),
        "development_started": False,
        "future_audit_authorized": False,
        "fit_started": False,
        "runtime_policy_activated": False,
        "current_profile_mutated": False,
    }
    _write_once(Path(output).resolve(), canonical_json_bytes(result))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    package = commands.add_parser("package")
    package.add_argument("--mode", choices=("preflight", "development", "future_audit"), required=True)
    package.add_argument("--run-name", required=True)
    package.add_argument("--run-dir", type=Path, required=True)
    package.add_argument("--repository-root", type=Path, default=_REPO_ROOT)
    package.add_argument("--template-package", type=Path, default=DEFAULT_TEMPLATE_PACKAGE)
    package.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    package.add_argument("--startup", type=Path, default=DEFAULT_STARTUP)
    package.add_argument("--preceding-gate", type=Path)
    authorize = commands.add_parser("authorize")
    authorize.add_argument("--run-dir", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--run-dir", type=Path, required=True)
    launch = commands.add_parser("launch")
    launch.add_argument("--run-dir", type=Path, required=True)
    launch.add_argument("--project", default="ofc-solver-485418")
    launch.add_argument("--bucket", default="pokerhu-ofc-solver-485418-training")
    launch.add_argument("--zone", default="asia-northeast1-b")
    launch.add_argument("--shards", action="append", required=True)
    launch.add_argument("--no-self-delete", action="store_true")
    status = commands.add_parser("status")
    status.add_argument("--run-dir", type=Path, required=True)
    status.add_argument("--project", default="ofc-solver-485418")
    status.add_argument("--bucket", default="pokerhu-ofc-solver-485418-training")
    status.add_argument("--zone", default="asia-northeast1-b")
    receive = commands.add_parser("receive")
    receive.add_argument("--run-dir", type=Path, required=True)
    receive.add_argument("--output-dir", type=Path, required=True)
    receive.add_argument("--project", default="ofc-solver-485418")
    receive.add_argument("--bucket", default="pokerhu-ofc-solver-485418-training")
    final = commands.add_parser("finalize-preflight")
    final.add_argument("--received-dir", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "package":
        result = package_attempt09(
            mode=args.mode, run_name=args.run_name, run_dir=args.run_dir,
            repository_root=args.repository_root, template_package=args.template_package,
            plan=args.plan, startup=args.startup, preceding_gate=args.preceding_gate,
        )
    elif args.command == "authorize":
        result = authorize_launch(run_dir=args.run_dir)
    elif args.command == "validate":
        manifest, authorization = validate_launch(args.run_dir)
        result = {"status": "valid", "manifest": manifest, "authorization": authorization}
    elif args.command == "launch":
        result = launch_wave(
            run_dir=args.run_dir, project=args.project, bucket=args.bucket,
            zone=args.zone, shards=args.shards, no_self_delete=args.no_self_delete,
        )
    elif args.command == "status":
        result = run_status(
            run_dir=args.run_dir, project=args.project, bucket=args.bucket, zone=args.zone
        )
    elif args.command == "receive":
        result = receive_run(
            run_dir=args.run_dir, project=args.project, bucket=args.bucket,
            output_dir=args.output_dir,
        )
    elif args.command == "finalize-preflight":
        result = finalize_preflight(received_dir=args.received_dir, output=args.output)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DONE_SCHEMA", "LAUNCH_AUTHORIZATION_SCHEMA", "PACKAGE_SCHEMA", "RECEIVE_SCHEMA",
    "authorize_launch", "build_schedule", "finalize_preflight", "launch_wave",
    "package_attempt09", "receive_run", "run_status", "validate_launch",
    "validate_package",
]
