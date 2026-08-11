"""Immutable package and receive helpers for the Attempt07 Spot preflight.

The package builder copies the already-validated Attempt06 ``package_src``
tree and applies a small Attempt07 overlay in a staging directory.  It never
modifies the Attempt06 package, generates a root, runs a teacher, invokes
``gcloud``, or resolves ``current``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .hu_m43_attempt06_contract import M43_ATTEMPT06_PLAN_SHA256
from .hu_m43_attempt06_spot import (
    EXPECTED_MODEL_COUNT as ATTEMPT06_EXPECTED_MODEL_COUNT,
    EXPECTED_MODEL_SCHEMA as ATTEMPT06_EXPECTED_MODEL_SCHEMA,
    EXPECTED_NATIVE_COUNT as ATTEMPT06_EXPECTED_NATIVE_COUNT,
    EXPECTED_NATIVE_SCHEMA as ATTEMPT06_EXPECTED_NATIVE_SCHEMA,
    PACKAGE_MANIFEST_SCHEMA as ATTEMPT06_PACKAGE_MANIFEST_SCHEMA,
    PINNED_MODEL_MANIFEST_SHA256 as ATTEMPT06_MODEL_MANIFEST_SHA256,
    PINNED_NATIVE_MANIFEST_SHA256 as ATTEMPT06_NATIVE_MANIFEST_SHA256,
    SOURCE_CLOSURE_SCHEMA as ATTEMPT06_SOURCE_CLOSURE_SCHEMA,
)
from .hu_m43_attempt06_teacher import ATTEMPT06_FROZEN_MODEL_SHA256
from .hu_m43_attempt07_contract import AI_PROFILES_SHA256, M43_ATTEMPT07_PLAN_SHA256
from .run_hu_m43_attempt07_preflight import (
    ATTEMPT07_PREFLIGHT_NATIVE_BATCH_THREADS,
    ATTEMPT07_PREFLIGHT_PLAN_SCHEMA,
    ATTEMPT07_PREFLIGHT_ROW_SCHEMA,
    ATTEMPT07_PREFLIGHT_SOURCE_ROOTS,
    ATTEMPT07_PREFLIGHT_SOURCE_SHA256,
    canonical_json_bytes,
    load_preflight_plan,
)


PACKAGE_MANIFEST_SCHEMA = "hu_m43_attempt07_preflight_spot_package_manifest_v1"
PACKAGE_RESULT_SCHEMA = "hu_m43_attempt07_preflight_spot_package_result_v1"
SCHEDULE_SCHEMA = "hu_m43_attempt07_preflight_spot_job_v1"
OVERLAY_CLOSURE_SCHEMA = "hu_m43_attempt07_preflight_overlay_closure_v1"
DONE_SCHEMA = "hu_m43_attempt07_preflight_spot_done_v1"
RECEIVE_AUDIT_SCHEMA = "hu_m43_attempt07_preflight_receive_audit_v1"
RECEIVE_MERGE_SCHEMA = "hu_m43_attempt07_preflight_receive_merge_v1"
LAUNCH_AUTHORIZATION_SCHEMA = (
    "hu_m43_attempt07_preflight_spot_launch_authorization_v1"
)

ATTEMPT06_BASE_RUN_NAME = (
    "regular-hu-m43-attempt06-preflight-final-20260714-1154"
)
ATTEMPT06_BASE_MANIFEST_SHA256 = (
    "88b05e0510242602f1618ca73306eb1e129f60e5e807dec4b1fac7de7e9114bc"
)
ATTEMPT07_MACHINE_TYPE = "c4-standard-4"
ATTEMPT07_JOB_COUNT = 5
ATTEMPT07_PACKAGE_TESTS_PASSED = 16

_ATTEMPT06_BASE_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "plan_sha256",
        "status_sha256",
        "model_sha256",
        "ai_profiles_sha256",
        "schedule_sha256",
        "source_closure_sha256",
        "source_zip_sha256",
        "startup_sha256",
        "source_model_manifest_sha256",
        "source_native_manifest_sha256",
        "total_roots",
        "total_shards",
        "roots_per_shard",
        "root_profile_assignment",
        "candidate_samples",
        "evaluation_samples",
        "native_batch_threads",
        "learned_nonbaseline_top_k",
        "fresh_seed_content_opened",
        "teacher_executed",
        "gcloud_invoked",
        "instances_created",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
)
_ATTEMPT06_BASE_CLOSURE_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "plan_sha256",
        "status_sha256",
        "model_sha256",
        "ai_profiles_sha256",
        "files",
        "fresh_seed_content_opened",
        "teacher_executed",
    }
)
_PACKAGE_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "jobs",
        "source_roots",
        "machine_type",
        "native_batch_threads",
        "base_attempt06_manifest_sha256",
        "base_attempt06_package_tree_sha256",
        "base_attempt06_source_zip_sha256",
        "package_tree_sha256",
        "overlay_closure_sha256",
        "source_zip_sha256",
        "source_zip_bytes",
        "startup_sha256",
        "schedule_sha256",
        "preflight_plan_sha256",
        "attempt07_plan_sha256",
        "source_merged_sha256",
        "model_sha256",
        "ai_profiles_sha256",
        "new_root_generated",
        "teacher_executed",
        "gcloud_invoked",
        "instances_created",
        "arm_selection_performed",
        "current_profile_resolved",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
)
_LAUNCH_AUTHORIZATION_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "manifest_sha256",
        "preflight_plan_sha256",
        "attempt07_plan_sha256",
        "source_merged_sha256",
        "model_sha256",
        "ai_profiles_sha256",
        "machine_type",
        "native_batch_threads",
        "jobs",
        "source_roots",
        "local_gates",
        "local_evidence",
        "actual_scalar_batch_result",
        "actual_operational_go_no_go",
        "spot_authorized",
        "new_root_generation_allowed",
        "arm_selection_allowed",
        "current_profile_resolved",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
)
_LOCAL_EVIDENCE_SUITES: dict[str, tuple[str, int, str]] = {
    "attempt07_pytest": (
        "hu_m43_attempt07_local_pytest_receipt_v1",
        99,
        "python -m pytest tests/test_hu_m43_attempt07_contract.py "
        "tests/test_hu_m43_attempt07_teacher.py "
        "tests/test_run_hu_m43_attempt07_development.py "
        "tests/test_run_hu_m43_attempt07_preflight.py "
        "tests/test_aggregate_hu_m43_attempt07_preflight.py "
        "tests/test_select_hu_m43_attempt07_development_arm.py -q",
    ),
    "rust_parity": (
        "hu_m3_rust_parity_test_receipt_v1",
        9,
        "python -m pytest tests/test_hu_m3_rust.py -q",
    ),
    "package_tests": (
        "hu_m43_attempt07_preflight_package_test_receipt_v1",
        ATTEMPT07_PACKAGE_TESTS_PASSED,
        "python -m pytest tests/test_hu_m43_attempt07_preflight_spot.py -q",
    ),
}

_DONE_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "job_index",
        "job_id",
        "source_root_index",
        "batch_child_selectors",
        "native_batch_threads",
        "output_prefix",
        "output_sha256",
        "checkpoint_sha256",
        "heartbeat_sha256",
        "summary_sha256",
        "run_log_sha256",
        "manifest_sha256",
        "authorization_sha256",
        "schedule_sha256",
        "attempt07_plan_sha256",
        "source_merged_sha256",
        "model_sha256",
        "ai_profiles_sha256",
        "elapsed_seconds",
        "peak_rss_bytes",
        "teacher_values_exported",
        "arm_selection_performed",
        "new_root_generated",
        "current_profile_resolved",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_RUN_DIR = _REPO_ROOT / "outputs" / "gcp_runs" / ATTEMPT06_BASE_RUN_NAME
DEFAULT_PREFLIGHT_PLAN = (
    _REPO_ROOT / "configs" / "hu_joint_policy_m43_attempt07_preflight.json"
)
DEFAULT_ATTEMPT07_PLAN = (
    _REPO_ROOT / "configs" / "hu_joint_policy_m43_attempt07.json"
)
DEFAULT_SOURCE = _REPO_ROOT / (
    "outputs/hu_joint_policy/m43_attempt06_search_quality/"
    "regular-hu-m43-attempt06-preflight-final-20260714-1154/"
    "merged/teacher.jsonl"
)
DEFAULT_STARTUP = _REPO_ROOT / "scripts" / "startup_hu_m43_attempt07_preflight.sh"

_OVERLAY_SOURCES = {
    "src/ofc_regular/hu_m43_attempt07_contract.py": (
        _REPO_ROOT / "src" / "ofc_regular" / "hu_m43_attempt07_contract.py"
    ),
    "src/ofc_regular/hu_m43_attempt07_teacher.py": (
        _REPO_ROOT / "src" / "ofc_regular" / "hu_m43_attempt07_teacher.py"
    ),
    "src/ofc_regular/run_hu_m43_attempt07_preflight.py": (
        _REPO_ROOT / "src" / "ofc_regular" / "run_hu_m43_attempt07_preflight.py"
    ),
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_atomic_no_clobber(path: Path, payload: bytes) -> None:
    """Publish once, accepting only a byte-identical existing artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise FileExistsError(f"immutable artifact already differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a mapping")
    return payload


def _tree_rows(root: Path, *, exclude: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        if relative in exclude:
            continue
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _tree_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(canonical_json_bytes(list(rows)))


def build_preflight_schedule() -> tuple[dict[str, Any], ...]:
    specs = (
        ("root0_batch_a", 0, True, "a"),
        ("root0_batch_b", 0, True, "b"),
        ("root0_scalar", 0, False, "scalar"),
        ("root1_batch", 1, True, "single"),
        ("root2_batch", 2, True, "single"),
    )
    return tuple(
        {
            "schema": SCHEDULE_SCHEMA,
            "job_index": index,
            "job_id": job_id,
            "source_root_index": source_root,
            "mode": "batch" if batch else "scalar",
            "batch_child_selectors": batch,
            "replicate": replicate,
            "native_batch_threads": ATTEMPT07_PREFLIGHT_NATIVE_BATCH_THREADS,
            "machine_type": ATTEMPT07_MACHINE_TYPE,
            "output_prefix": f"job_{index:03d}_{job_id}",
            "new_root_generation_allowed": False,
            "arm_selection_allowed": False,
        }
        for index, (job_id, source_root, batch, replicate) in enumerate(specs)
    )


def validate_preflight_schedule(rows: Sequence[Mapping[str, Any]]) -> None:
    expected = build_preflight_schedule()
    if tuple(dict(row) for row in rows) != expected:
        raise ValueError("Attempt07 Spot preflight schedule changed")
    if len(rows) != ATTEMPT07_JOB_COUNT:
        raise ValueError("Attempt07 Spot preflight must contain exactly five jobs")
    if [row["source_root_index"] for row in rows] != [0, 0, 0, 1, 2]:
        raise ValueError("Attempt07 Spot preflight source-root assignment changed")
    if [row["batch_child_selectors"] for row in rows] != [True, True, False, True, True]:
        raise ValueError("Attempt07 Spot preflight scalar/batch assignment changed")


def _schedule_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) for row in rows)


def _deterministic_zip(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(
        destination, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in sorted(
            (item for item in source.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(source).as_posix(),
        ):
            relative = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)


def _is_lower_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_package_relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{label} path is unsafe")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or ":" in relative.parts[0]
        or relative.as_posix() != value
    ):
        raise ValueError(f"{label} path is unsafe")
    return value


def _validate_file_rows(
    value: Any,
    *,
    label: str,
    row_keys: frozenset[str] = frozenset({"path", "bytes", "sha256"}),
    require_sorted: bool = False,
) -> dict[str, dict[str, Any]]:
    if type(value) is not list or not value:
        raise ValueError(f"{label} rows changed")
    result: dict[str, dict[str, Any]] = {}
    ordered_paths: list[str] = []
    for row in value:
        if type(row) is not dict or set(row) != row_keys:
            raise ValueError(f"{label} row schema changed")
        relative = _safe_package_relative_path(row.get("path"), label=label)
        if relative in result:
            raise ValueError(f"{label} contains a duplicate path")
        if type(row.get("bytes")) is not int or row["bytes"] < 0:
            raise ValueError(f"{label} byte count changed")
        if not _is_lower_sha256(row.get("sha256")):
            raise ValueError(f"{label} SHA-256 changed")
        if "platform" in row and row["platform"] != "linux-x86_64":
            raise ValueError(f"{label} platform changed")
        result[relative] = dict(row)
        ordered_paths.append(relative)
    if require_sorted and ordered_paths != sorted(ordered_paths):
        raise ValueError(f"{label} path order changed")
    return result


def _validate_runtime_submanifest(
    *,
    package_src: Path,
    closure_rows: Mapping[str, Mapping[str, Any]],
    name: str,
    expected_sha256: str,
    schema: str,
    count_field: str,
    rows_field: str,
    expected_count: int,
    native: bool,
) -> None:
    path = package_src / name
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError(f"Attempt06 {name} SHA-256 changed")
    payload = _load_mapping(path, f"Attempt06 {name}")
    expected_keys = {"schema", count_field, rows_field}
    if (
        set(payload) != expected_keys
        or payload.get("schema") != schema
        or type(payload.get(count_field)) is not int
        or payload[count_field] != expected_count
    ):
        raise ValueError(f"Attempt06 {name} schema changed")
    row_keys = frozenset({"path", "bytes", "sha256", "platform"}) if native else frozenset(
        {"path", "bytes", "sha256"}
    )
    rows = _validate_file_rows(
        payload.get(rows_field), label=f"Attempt06 {name}", row_keys=row_keys
    )
    if len(rows) != expected_count:
        raise ValueError(f"Attempt06 {name} count changed")
    for relative, row in rows.items():
        expected_prefix = "target/release/" if native else "models/"
        if not relative.startswith(expected_prefix):
            raise ValueError(f"Attempt06 {name} path scope changed")
        source = package_src.joinpath(*PurePosixPath(relative).parts)
        closure_row = closure_rows.get(relative)
        if (
            not source.is_file()
            or closure_row is None
            or source.stat().st_size != row["bytes"]
            or sha256_file(source) != row["sha256"]
            or closure_row.get("bytes") != row["bytes"]
            or closure_row.get("sha256") != row["sha256"]
        ):
            raise ValueError(f"Attempt06 {name} file binding changed: {relative}")


def _validate_source_zip(
    *,
    source_zip: Path,
    expected_sha256: str,
    package_rows: Mapping[str, Mapping[str, Any]],
) -> None:
    if not source_zip.is_file() or sha256_file(source_zip) != expected_sha256:
        raise ValueError("Attempt06 base source ZIP SHA-256 changed")
    try:
        with zipfile.ZipFile(source_zip, "r") as archive:
            infos = archive.infolist()
            if len(infos) != len(package_rows):
                raise ValueError("Attempt06 base source ZIP member set changed")
            names: set[str] = set()
            for info in infos:
                relative = _safe_package_relative_path(
                    info.filename, label="Attempt06 base source ZIP"
                )
                member_type = (info.external_attr >> 16) & 0o170000
                if (
                    info.is_dir()
                    or member_type == 0o120000
                    or relative in names
                    or info.flag_bits & 0x1
                ):
                    raise ValueError("Attempt06 base source ZIP member changed")
                names.add(relative)
                expected = package_rows.get(relative)
                if expected is None or info.file_size != expected.get("bytes"):
                    raise ValueError("Attempt06 base source ZIP member binding changed")
                digest = hashlib.sha256()
                with archive.open(info, "r") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != expected.get("sha256"):
                    raise ValueError("Attempt06 base source ZIP member SHA-256 changed")
            if names != set(package_rows):
                raise ValueError("Attempt06 base source ZIP member set changed")
    except zipfile.BadZipFile as exc:
        raise ValueError("Attempt06 base source ZIP is invalid") from exc


def _validate_base_package(base_run_dir: Path) -> tuple[dict[str, Any], str, str]:
    manifest_path = base_run_dir / "manifest.json"
    package_src = base_run_dir / "package_src"
    if not manifest_path.is_file() or not package_src.is_dir():
        raise ValueError("Attempt06 base package closure is missing")
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != ATTEMPT06_BASE_MANIFEST_SHA256:
        raise ValueError("Attempt06 base package manifest SHA-256 changed")
    manifest = _load_mapping(manifest_path, "Attempt06 base manifest")
    if (
        set(manifest) != _ATTEMPT06_BASE_MANIFEST_KEYS
        or manifest_path.read_bytes() != canonical_json_bytes(manifest)
    ):
        raise ValueError("Attempt06 base package manifest schema changed")
    if (
        manifest.get("schema") != ATTEMPT06_PACKAGE_MANIFEST_SCHEMA
        or manifest.get("status") != "frozen_package_only_no_fresh_content"
        or manifest.get("run_name") != ATTEMPT06_BASE_RUN_NAME
        or manifest.get("plan_sha256") != M43_ATTEMPT06_PLAN_SHA256
        or manifest.get("model_sha256") != ATTEMPT06_FROZEN_MODEL_SHA256
        or manifest.get("ai_profiles_sha256") != AI_PROFILES_SHA256
        or manifest.get("source_model_manifest_sha256")
        != ATTEMPT06_MODEL_MANIFEST_SHA256
        or manifest.get("source_native_manifest_sha256")
        != ATTEMPT06_NATIVE_MANIFEST_SHA256
        or type(manifest.get("total_roots")) is not int
        or manifest.get("total_roots") != 50
        or type(manifest.get("total_shards")) is not int
        or manifest.get("total_shards") != 50
        or type(manifest.get("roots_per_shard")) is not int
        or manifest.get("roots_per_shard") != 1
        or manifest.get("root_profile_assignment")
        != "root_index_mod_5_in_frozen_profile_order"
        or type(manifest.get("candidate_samples")) is not int
        or manifest.get("candidate_samples") != 8
        or type(manifest.get("evaluation_samples")) is not int
        or manifest.get("evaluation_samples") != 128
        or type(manifest.get("native_batch_threads")) is not int
        or manifest.get("native_batch_threads") != 4
        or type(manifest.get("learned_nonbaseline_top_k")) is not int
        or manifest.get("learned_nonbaseline_top_k") != 8
        or manifest.get("fresh_seed_content_opened") is not False
        or manifest.get("teacher_executed") is not False
        or manifest.get("gcloud_invoked") is not False
        or manifest.get("instances_created") is not False
        or manifest.get("current_profile_mutated") is not False
        or manifest.get("runtime_policy_activated") is not False
    ):
        raise ValueError("Attempt06 base package boundary changed")
    for field in (
        "plan_sha256",
        "status_sha256",
        "model_sha256",
        "ai_profiles_sha256",
        "schedule_sha256",
        "source_closure_sha256",
        "source_zip_sha256",
        "startup_sha256",
        "source_model_manifest_sha256",
        "source_native_manifest_sha256",
    ):
        if not _is_lower_sha256(manifest.get(field)):
            raise ValueError(f"Attempt06 base package {field} changed")

    closure_root = base_run_dir / "source_closure_manifest.json"
    closure_in_package = package_src / "source_closure_manifest.json"
    if (
        not closure_root.is_file()
        or not closure_in_package.is_file()
        or closure_root.read_bytes() != closure_in_package.read_bytes()
        or sha256_file(closure_root) != manifest["source_closure_sha256"]
    ):
        raise ValueError("Attempt06 base source closure binding changed")
    closure = _load_mapping(closure_root, "Attempt06 base source closure")
    if (
        set(closure) != _ATTEMPT06_BASE_CLOSURE_KEYS
        or closure_root.read_bytes() != canonical_json_bytes(closure)
        or closure.get("schema") != ATTEMPT06_SOURCE_CLOSURE_SCHEMA
        or closure.get("status") != "closed_no_fresh_seed_materialized"
        or closure.get("run_name") != ATTEMPT06_BASE_RUN_NAME
        or closure.get("plan_sha256") != manifest["plan_sha256"]
        or closure.get("status_sha256") != manifest["status_sha256"]
        or closure.get("model_sha256") != manifest["model_sha256"]
        or closure.get("ai_profiles_sha256") != manifest["ai_profiles_sha256"]
        or closure.get("fresh_seed_content_opened") is not False
        or closure.get("teacher_executed") is not False
    ):
        raise ValueError("Attempt06 base source closure semantics changed")
    closure_rows = _validate_file_rows(
        closure.get("files"),
        label="Attempt06 base source closure",
        require_sorted=True,
    )
    if "source_closure_manifest.json" in closure_rows:
        raise ValueError("Attempt06 base source closure contains itself")
    all_items = tuple(package_src.rglob("*"))
    if any(item.is_symlink() for item in all_items):
        raise ValueError("Attempt06 base package contains a symbolic link")
    actual_paths = {
        item.relative_to(package_src).as_posix() for item in all_items if item.is_file()
    }
    if actual_paths != set(closure_rows) | {"source_closure_manifest.json"}:
        raise ValueError("Attempt06 base package tree differs from source closure")
    for relative, row in closure_rows.items():
        source = package_src.joinpath(*PurePosixPath(relative).parts)
        if (
            not source.is_file()
            or source.stat().st_size != row["bytes"]
            or sha256_file(source) != row["sha256"]
        ):
            raise ValueError(f"Attempt06 base source closure file changed: {relative}")

    root_bindings = (
        ("shards_manifest.jsonl", "shards_manifest.jsonl", "schedule_sha256"),
        (
            "hu_joint_policy_m43_attempt06.json",
            "configs/hu_joint_policy_m43_attempt06.json",
            "plan_sha256",
        ),
        (
            "hu_joint_policy_m43_attempt06_status.json",
            "configs/hu_joint_policy_m43_attempt06_status.json",
            "status_sha256",
        ),
        ("startup_hu_m43_attempt06_teacher.sh", None, "startup_sha256"),
    )
    for root_name, package_name, field in root_bindings:
        root_artifact = base_run_dir / root_name
        if not root_artifact.is_file() or sha256_file(root_artifact) != manifest[field]:
            raise ValueError(f"Attempt06 base {root_name} binding changed")
        if package_name is not None:
            package_artifact = package_src.joinpath(*PurePosixPath(package_name).parts)
            if root_artifact.read_bytes() != package_artifact.read_bytes():
                raise ValueError(f"Attempt06 base {root_name} package copy changed")

    _validate_runtime_submanifest(
        package_src=package_src,
        closure_rows=closure_rows,
        name="source_model_manifest.json",
        expected_sha256=manifest["source_model_manifest_sha256"],
        schema=ATTEMPT06_EXPECTED_MODEL_SCHEMA,
        count_field="model_count",
        rows_field="models",
        expected_count=ATTEMPT06_EXPECTED_MODEL_COUNT,
        native=False,
    )
    _validate_runtime_submanifest(
        package_src=package_src,
        closure_rows=closure_rows,
        name="source_native_manifest.json",
        expected_sha256=manifest["source_native_manifest_sha256"],
        schema=ATTEMPT06_EXPECTED_NATIVE_SCHEMA,
        count_field="binary_count",
        rows_field="binaries",
        expected_count=ATTEMPT06_EXPECTED_NATIVE_COUNT,
        native=True,
    )

    package_rows = dict(closure_rows)
    package_rows["source_closure_manifest.json"] = {
        "path": "source_closure_manifest.json",
        "bytes": closure_in_package.stat().st_size,
        "sha256": manifest["source_closure_sha256"],
    }
    _validate_source_zip(
        source_zip=(
            base_run_dir / "ofc_regular_hu_m43_attempt06_teacher_source.zip"
        ),
        expected_sha256=manifest["source_zip_sha256"],
        package_rows=package_rows,
    )
    tree_hash = _tree_digest(_tree_rows(package_src))
    return manifest, manifest_hash, tree_hash


def package_preflight_run(
    *,
    repo_root: str | Path,
    run_dir: str | Path,
    run_name: str,
    base_run_dir: str | Path = DEFAULT_BASE_RUN_DIR,
    preflight_plan: str | Path = DEFAULT_PREFLIGHT_PLAN,
    attempt07_plan: str | Path = DEFAULT_ATTEMPT07_PLAN,
    source: str | Path = DEFAULT_SOURCE,
    startup: str | Path = DEFAULT_STARTUP,
    resume_existing: bool = False,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    destination = Path(run_dir).resolve()
    expected_destination = (repo / "outputs" / "gcp_runs" / run_name).resolve()
    if destination != expected_destination:
        raise ValueError("Attempt07 preflight run_dir must equal outputs/gcp_runs/<run_name>")
    if not run_name or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in run_name):
        raise ValueError("Attempt07 preflight run_name is unsafe")

    base = Path(base_run_dir).resolve()
    base_manifest, base_manifest_hash, base_tree_hash = _validate_base_package(base)
    frozen_preflight = load_preflight_plan(preflight_plan)
    if frozen_preflight.get("schema") != ATTEMPT07_PREFLIGHT_PLAN_SCHEMA:
        raise ValueError("Attempt07 preflight plan schema changed")
    if sha256_file(attempt07_plan) != M43_ATTEMPT07_PLAN_SHA256:
        raise ValueError("Attempt07 plan SHA-256 changed before packaging")
    if sha256_file(source) != ATTEMPT07_PREFLIGHT_SOURCE_SHA256:
        raise ValueError("Attempt06 merged preflight source SHA-256 changed")
    if sha256_file(startup) == _sha256_bytes(b""):
        raise ValueError("Attempt07 startup worker is empty")

    if destination.exists():
        if not resume_existing:
            raise FileExistsError(f"Attempt07 preflight run already exists: {destination}")
        manifest_path = destination / "manifest.json"
        existing = _load_mapping(manifest_path, "existing manifest")
        _validate_package_manifest(existing, manifest_path)
        expected_schedule = _schedule_bytes(build_preflight_schedule())
        if (
            existing.get("run_name") != run_name
            or existing.get("attempt07_plan_sha256") != M43_ATTEMPT07_PLAN_SHA256
            or existing.get("source_merged_sha256") != ATTEMPT07_PREFLIGHT_SOURCE_SHA256
            or existing.get("base_attempt06_manifest_sha256") != base_manifest_hash
            or existing.get("base_attempt06_package_tree_sha256") != base_tree_hash
            or sha256_file(preflight_plan) != existing.get("preflight_plan_sha256")
            or sha256_file(attempt07_plan) != existing.get("attempt07_plan_sha256")
            or sha256_file(startup) != existing.get("startup_sha256")
            or (destination / "shards_manifest.jsonl").read_bytes()
            != expected_schedule
            or sha256_file(destination / "shards_manifest.jsonl")
            != existing.get("schedule_sha256")
            or sha256_file(destination / "startup_hu_m43_attempt07_preflight.sh")
            != existing.get("startup_sha256")
            or sha256_file(destination / "hu_joint_policy_m43_attempt07_preflight.json")
            != existing.get("preflight_plan_sha256")
            or sha256_file(destination / "hu_joint_policy_m43_attempt07.json")
            != existing.get("attempt07_plan_sha256")
            or sha256_file(destination / "source.zip") != existing.get("source_zip_sha256")
            or sha256_file(destination / "package_src" / "preflight_overlay_manifest.json")
            != existing.get("overlay_closure_sha256")
            or _tree_digest(_tree_rows(destination / "package_src"))
            != existing.get("package_tree_sha256")
        ):
            raise ValueError("existing Attempt07 preflight package differs")
        return {
            "schema": PACKAGE_RESULT_SCHEMA,
            "status": "verified_existing_package_without_execution",
            "run_name": run_name,
            "manifest_sha256": sha256_file(destination / "manifest.json"),
            "jobs": ATTEMPT07_JOB_COUNT,
            "gcloud_invoked": False,
            "teacher_executed": False,
            "new_root_generated": False,
            "current_profile_mutated": False,
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + ".staging-" + uuid.uuid4().hex)
    if staging.exists():
        raise FileExistsError(f"Attempt07 staging directory already exists: {staging}")
    try:
        package_src = staging / "package_src"
        shutil.copytree(base / "package_src", package_src, copy_function=shutil.copy2)
        if (
            _tree_digest(_tree_rows(base / "package_src")) != base_tree_hash
            or _tree_digest(_tree_rows(package_src)) != base_tree_hash
        ):
            raise ValueError("Attempt06 base package changed while copying")

        schedule = build_preflight_schedule()
        validate_preflight_schedule(schedule)
        schedule_bytes = _schedule_bytes(schedule)
        _write_atomic(package_src / "shards_manifest.jsonl", schedule_bytes)
        _write_atomic(staging / "shards_manifest.jsonl", schedule_bytes)

        overlays = dict(_OVERLAY_SOURCES)
        overlays.update(
            {
                "configs/hu_joint_policy_m43_attempt07_preflight.json": Path(preflight_plan),
                "configs/hu_joint_policy_m43_attempt07.json": Path(attempt07_plan),
                "preflight_source/teacher.jsonl": Path(source),
            }
        )
        for relative, origin in overlays.items():
            if not Path(origin).is_file():
                raise ValueError(f"Attempt07 overlay source is missing: {origin}")
            target = package_src / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, target)

        if sha256_file(package_src / "artifacts/lambda_rank_candidate.pkl") != ATTEMPT06_FROZEN_MODEL_SHA256:
            raise ValueError("copied frozen Lambda artifact changed")
        if sha256_file(package_src / "src/ofc_regular/ai_profiles.py") != AI_PROFILES_SHA256:
            raise ValueError("copied ai_profiles.py changed")
        if sha256_file(package_src / "configs/hu_joint_policy_m43_attempt07.json") != M43_ATTEMPT07_PLAN_SHA256:
            raise ValueError("overlaid Attempt07 plan changed")
        if sha256_file(package_src / "preflight_source/teacher.jsonl") != ATTEMPT07_PREFLIGHT_SOURCE_SHA256:
            raise ValueError("overlaid Attempt06 source changed")

        closure_relative = "preflight_overlay_manifest.json"
        closure_rows = _tree_rows(package_src, exclude=frozenset({closure_relative}))
        closure = {
            "schema": OVERLAY_CLOSURE_SCHEMA,
            "status": "attempt06_package_copy_plus_attempt07_overlay_no_execution",
            "base_attempt06_manifest_sha256": base_manifest_hash,
            "base_attempt06_package_tree_sha256": base_tree_hash,
            "files": closure_rows,
            "file_count": len(closure_rows),
            "attempt07_plan_sha256": M43_ATTEMPT07_PLAN_SHA256,
            "source_merged_sha256": ATTEMPT07_PREFLIGHT_SOURCE_SHA256,
            "model_sha256": ATTEMPT06_FROZEN_MODEL_SHA256,
            "ai_profiles_sha256": AI_PROFILES_SHA256,
            "new_root_generated": False,
            "teacher_executed": False,
            "current_profile_mutated": False,
        }
        _write_atomic(package_src / closure_relative, canonical_json_bytes(closure))
        closure_sha = sha256_file(package_src / closure_relative)
        final_tree_rows = _tree_rows(package_src)
        final_tree_sha = _tree_digest(final_tree_rows)

        source_zip = staging / "source.zip"
        _deterministic_zip(package_src, source_zip)
        startup_target = staging / "startup_hu_m43_attempt07_preflight.sh"
        shutil.copy2(startup, startup_target)
        preflight_plan_target = staging / "hu_joint_policy_m43_attempt07_preflight.json"
        attempt07_plan_target = staging / "hu_joint_policy_m43_attempt07.json"
        shutil.copy2(preflight_plan, preflight_plan_target)
        shutil.copy2(attempt07_plan, attempt07_plan_target)

        manifest = {
            "schema": PACKAGE_MANIFEST_SCHEMA,
            "status": "packaged_attempt06_copy_plus_overlay_without_execution",
            "run_name": run_name,
            "jobs": ATTEMPT07_JOB_COUNT,
            "source_roots": list(ATTEMPT07_PREFLIGHT_SOURCE_ROOTS),
            "machine_type": ATTEMPT07_MACHINE_TYPE,
            "native_batch_threads": ATTEMPT07_PREFLIGHT_NATIVE_BATCH_THREADS,
            "base_attempt06_manifest_sha256": base_manifest_hash,
            "base_attempt06_package_tree_sha256": base_tree_hash,
            "base_attempt06_source_zip_sha256": base_manifest.get("source_zip_sha256"),
            "package_tree_sha256": final_tree_sha,
            "overlay_closure_sha256": closure_sha,
            "source_zip_sha256": sha256_file(source_zip),
            "source_zip_bytes": source_zip.stat().st_size,
            "startup_sha256": sha256_file(startup_target),
            "schedule_sha256": sha256_file(staging / "shards_manifest.jsonl"),
            "preflight_plan_sha256": sha256_file(preflight_plan_target),
            "attempt07_plan_sha256": M43_ATTEMPT07_PLAN_SHA256,
            "source_merged_sha256": ATTEMPT07_PREFLIGHT_SOURCE_SHA256,
            "model_sha256": ATTEMPT06_FROZEN_MODEL_SHA256,
            "ai_profiles_sha256": AI_PROFILES_SHA256,
            "new_root_generated": False,
            "teacher_executed": False,
            "gcloud_invoked": False,
            "instances_created": False,
            "arm_selection_performed": False,
            "current_profile_resolved": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        }
        _write_atomic(staging / "manifest.json", canonical_json_bytes(manifest))
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "schema": PACKAGE_RESULT_SCHEMA,
        "status": "packaged_without_execution_or_new_root",
        "run_name": run_name,
        "manifest_sha256": sha256_file(destination / "manifest.json"),
        "jobs": ATTEMPT07_JOB_COUNT,
        "gcloud_invoked": False,
        "teacher_executed": False,
        "new_root_generated": False,
        "current_profile_mutated": False,
    }


def _read_schedule(path: Path) -> tuple[dict[str, Any], ...]:
    rows = tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    )
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("Attempt07 preflight schedule contains a non-mapping")
    validate_preflight_schedule(rows)
    return rows


def _require_hash(path: Path, expected: Any, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label} expected SHA-256 is invalid")
    if sha256_file(path) != expected:
        raise ValueError(f"{label} SHA-256 changed")


def _validate_package_manifest(manifest: Mapping[str, Any], manifest_path: Path) -> None:
    if not manifest_path.is_file():
        raise ValueError("Attempt07 preflight package manifest is missing")
    if (
        set(manifest) != _PACKAGE_MANIFEST_KEYS
        or manifest_path.read_bytes() != canonical_json_bytes(dict(manifest))
    ):
        raise ValueError("Attempt07 preflight package manifest schema changed")
    if (
        manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA
        or manifest.get("status")
        != "packaged_attempt06_copy_plus_overlay_without_execution"
        or not isinstance(manifest.get("run_name"), str)
        or not manifest["run_name"]
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
            for character in manifest["run_name"]
        )
        or type(manifest.get("jobs")) is not int
        or manifest.get("jobs") != ATTEMPT07_JOB_COUNT
        or type(manifest.get("source_roots")) is not list
        or any(type(value) is not int for value in manifest["source_roots"])
        or manifest.get("source_roots") != list(ATTEMPT07_PREFLIGHT_SOURCE_ROOTS)
        or manifest.get("machine_type") != ATTEMPT07_MACHINE_TYPE
        or type(manifest.get("native_batch_threads")) is not int
        or manifest.get("native_batch_threads")
        != ATTEMPT07_PREFLIGHT_NATIVE_BATCH_THREADS
        or manifest.get("base_attempt06_manifest_sha256")
        != ATTEMPT06_BASE_MANIFEST_SHA256
        or type(manifest.get("source_zip_bytes")) is not int
        or manifest.get("source_zip_bytes", 0) < 1
        or manifest.get("attempt07_plan_sha256") != M43_ATTEMPT07_PLAN_SHA256
        or manifest.get("source_merged_sha256")
        != ATTEMPT07_PREFLIGHT_SOURCE_SHA256
        or manifest.get("model_sha256") != ATTEMPT06_FROZEN_MODEL_SHA256
        or manifest.get("ai_profiles_sha256") != AI_PROFILES_SHA256
        or manifest.get("new_root_generated") is not False
        or manifest.get("teacher_executed") is not False
        or manifest.get("arm_selection_performed") is not False
        or manifest.get("current_profile_resolved") is not False
        or manifest.get("current_profile_mutated") is not False
        or manifest.get("runtime_policy_activated") is not False
    ):
        raise ValueError("Attempt07 preflight package manifest changed")
    for field in (
        "base_attempt06_manifest_sha256",
        "base_attempt06_package_tree_sha256",
        "base_attempt06_source_zip_sha256",
        "package_tree_sha256",
        "overlay_closure_sha256",
        "source_zip_sha256",
        "startup_sha256",
        "schedule_sha256",
        "preflight_plan_sha256",
        "attempt07_plan_sha256",
        "source_merged_sha256",
        "model_sha256",
        "ai_profiles_sha256",
    ):
        if not _is_lower_sha256(manifest.get(field)):
            raise ValueError(f"Attempt07 preflight package {field} changed")


def _validate_evidence_receipt(
    path: Path,
    *,
    suite: str,
    manifest_sha256: str | None = None,
) -> dict[str, Any]:
    if suite not in _LOCAL_EVIDENCE_SUITES:
        raise ValueError("unknown Attempt07 local evidence suite")
    schema, expected_passed, command = _LOCAL_EVIDENCE_SUITES[suite]
    receipt = _load_mapping(path, "Attempt07 launch evidence receipt")
    if path.read_bytes() != canonical_json_bytes(receipt):
        raise ValueError(f"Attempt07 launch evidence is not canonical: {path}")
    expected_keys = {
        "schema",
        "status",
        "suite",
        "command",
        "passed",
        "failed",
        "spot_result_observed",
        "current_profile_mutated",
    }
    if manifest_sha256 is not None:
        expected_keys.add("manifest_sha256")
    if (
        set(receipt) != expected_keys
        or receipt.get("schema") != schema
        or receipt.get("status") != "pass"
        or receipt.get("suite") != suite
        or receipt.get("command") != command
        or type(receipt.get("passed")) is not int
        or receipt.get("passed") != expected_passed
        or type(receipt.get("failed")) is not int
        or receipt.get("failed") != 0
        or receipt.get("spot_result_observed") is not False
        or receipt.get("current_profile_mutated") is not False
    ):
        raise ValueError(f"Attempt07 launch evidence changed: {path}")
    if manifest_sha256 is not None and receipt.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Attempt07 package-test evidence manifest binding changed")
    return receipt


def create_launch_authorization(
    *,
    manifest_path: str | Path,
    attempt07_tests_receipt: str | Path,
    rust_parity_receipt: str | Path,
    package_tests_receipt: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Create the run-specific launch authorization after package-only validation."""

    manifest_file = Path(manifest_path)
    manifest = _load_mapping(manifest_file, "Attempt07 preflight manifest")
    _validate_package_manifest(manifest, manifest_file)
    manifest_sha = sha256_file(manifest_file)
    evidence_specs = (
        ("attempt07_pytest", Path(attempt07_tests_receipt), None),
        ("rust_parity", Path(rust_parity_receipt), None),
        ("package_tests", Path(package_tests_receipt), manifest_sha),
    )
    evidence: dict[str, Any] = {}
    for name, path, expected_manifest in evidence_specs:
        receipt = _validate_evidence_receipt(
            path,
            suite=name,
            manifest_sha256=expected_manifest,
        )
        evidence[name] = {
            "receipt_sha256": sha256_file(path),
            "passed": receipt["passed"],
            "failed": 0,
        }
    payload = {
        "schema": LAUNCH_AUTHORIZATION_SCHEMA,
        "status": "authorized_for_bounded_spot_preflight",
        "run_name": manifest["run_name"],
        "manifest_sha256": manifest_sha,
        "preflight_plan_sha256": manifest["preflight_plan_sha256"],
        "attempt07_plan_sha256": M43_ATTEMPT07_PLAN_SHA256,
        "source_merged_sha256": ATTEMPT07_PREFLIGHT_SOURCE_SHA256,
        "model_sha256": ATTEMPT06_FROZEN_MODEL_SHA256,
        "ai_profiles_sha256": AI_PROFILES_SHA256,
        "machine_type": ATTEMPT07_MACHINE_TYPE,
        "native_batch_threads": ATTEMPT07_PREFLIGHT_NATIVE_BATCH_THREADS,
        "jobs": ATTEMPT07_JOB_COUNT,
        "source_roots": list(ATTEMPT07_PREFLIGHT_SOURCE_ROOTS),
        "local_gates": {
            "correctness_smoke": "pass",
            "determinism": "pass",
            "scalar_batch_parity_test_harness": "pass",
            "package_closure": "pass",
        },
        "local_evidence": evidence,
        "actual_scalar_batch_result": "pending_spot_preflight_receive_and_aggregate",
        "actual_operational_go_no_go": "pending_spot_preflight_receive_and_aggregate",
        "spot_authorized": True,
        "new_root_generation_allowed": False,
        "arm_selection_allowed": False,
        "current_profile_resolved": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    _write_atomic_no_clobber(Path(output), canonical_json_bytes(payload))
    return {
        "schema": LAUNCH_AUTHORIZATION_SCHEMA,
        "status": "authorized_for_bounded_spot_preflight",
        "run_name": manifest["run_name"],
        "manifest_sha256": manifest_sha,
        "authorization_sha256": sha256_file(output),
        "actual_scalar_batch_result": "pending_spot_preflight_receive_and_aggregate",
        "current_profile_mutated": False,
    }


def create_local_evidence_receipt(
    *,
    suite: str,
    passed: int,
    failed: int,
    output: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Record one already-completed local command as immutable evidence."""

    if suite not in _LOCAL_EVIDENCE_SUITES:
        raise ValueError("unknown Attempt07 local evidence suite")
    schema, expected_passed, command = _LOCAL_EVIDENCE_SUITES[suite]
    if (
        type(passed) is not int
        or passed != expected_passed
        or type(failed) is not int
        or failed != 0
    ):
        raise ValueError("Attempt07 local evidence counts changed")
    payload: dict[str, Any] = {
        "schema": schema,
        "status": "pass",
        "suite": suite,
        "command": command,
        "passed": passed,
        "failed": failed,
        "spot_result_observed": False,
        "current_profile_mutated": False,
    }
    if suite == "package_tests":
        if manifest_path is None:
            raise ValueError("package_tests evidence requires the package manifest")
        manifest = Path(manifest_path)
        package = _load_mapping(manifest, "Attempt07 package-test manifest")
        _validate_package_manifest(package, manifest)
        payload["manifest_sha256"] = sha256_file(manifest)
    elif manifest_path is not None:
        raise ValueError("only package_tests evidence may bind a package manifest")
    _write_atomic_no_clobber(Path(output), canonical_json_bytes(payload))
    return payload


def _validate_authorization(
    authorization_path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
) -> str:
    authorization = _load_mapping(authorization_path, "Attempt07 launch authorization")
    if authorization_path.read_bytes() != canonical_json_bytes(authorization):
        raise ValueError("Attempt07 launch authorization is not canonical")
    if (
        set(authorization) != _LAUNCH_AUTHORIZATION_KEYS
        or authorization.get("schema") != LAUNCH_AUTHORIZATION_SCHEMA
        or authorization.get("status") != "authorized_for_bounded_spot_preflight"
        or authorization.get("run_name") != manifest.get("run_name")
        or authorization.get("manifest_sha256") != manifest_sha256
        or authorization.get("preflight_plan_sha256")
        != manifest.get("preflight_plan_sha256")
        or authorization.get("attempt07_plan_sha256") != M43_ATTEMPT07_PLAN_SHA256
        or authorization.get("source_merged_sha256")
        != ATTEMPT07_PREFLIGHT_SOURCE_SHA256
        or authorization.get("model_sha256") != ATTEMPT06_FROZEN_MODEL_SHA256
        or authorization.get("ai_profiles_sha256") != AI_PROFILES_SHA256
        or authorization.get("machine_type") != ATTEMPT07_MACHINE_TYPE
        or type(authorization.get("native_batch_threads")) is not int
        or authorization.get("native_batch_threads")
        != ATTEMPT07_PREFLIGHT_NATIVE_BATCH_THREADS
        or type(authorization.get("jobs")) is not int
        or authorization.get("jobs") != ATTEMPT07_JOB_COUNT
        or type(authorization.get("source_roots")) is not list
        or any(type(value) is not int for value in authorization["source_roots"])
        or authorization.get("source_roots")
        != list(ATTEMPT07_PREFLIGHT_SOURCE_ROOTS)
        or not isinstance(authorization.get("local_evidence"), Mapping)
        or authorization.get("spot_authorized") is not True
        or authorization.get("actual_scalar_batch_result")
        != "pending_spot_preflight_receive_and_aggregate"
        or authorization.get("actual_operational_go_no_go")
        != "pending_spot_preflight_receive_and_aggregate"
        or authorization.get("new_root_generation_allowed") is not False
        or authorization.get("arm_selection_allowed") is not False
        or authorization.get("current_profile_resolved") is not False
        or authorization.get("current_profile_mutated") is not False
        or authorization.get("runtime_policy_activated") is not False
    ):
        raise ValueError("Attempt07 launch authorization fields changed")
    gates = authorization.get("local_gates")
    if not isinstance(gates, Mapping) or gates != {
        "correctness_smoke": "pass",
        "determinism": "pass",
        "scalar_batch_parity_test_harness": "pass",
        "package_closure": "pass",
    }:
        raise ValueError("Attempt07 launch authorization local gates changed")
    evidence = authorization.get("local_evidence")
    expected_passed = {
        name: specification[1]
        for name, specification in _LOCAL_EVIDENCE_SUITES.items()
    }
    if not isinstance(evidence, Mapping) or set(evidence) != {
        "attempt07_pytest",
        "rust_parity",
        "package_tests",
    }:
        raise ValueError("Attempt07 launch authorization evidence changed")
    for name, row in evidence.items():
        if (
            not isinstance(row, Mapping)
            or set(row) != {"receipt_sha256", "passed", "failed"}
            or not isinstance(row.get("receipt_sha256"), str)
            or len(row["receipt_sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in row["receipt_sha256"])
            or type(row.get("passed")) is not int
            or row["passed"] != expected_passed[name]
            or type(row.get("failed")) is not int
            or row["failed"] != 0
        ):
            raise ValueError("Attempt07 launch authorization evidence changed")
    return sha256_file(authorization_path)


def _contains_forbidden_result_value(value: Any, path: str = "row") -> str | None:
    forbidden = {
        "arms",
        "selected_action_key",
        "override_fired",
        "mean",
        "p01",
        "p05",
        "min",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in forbidden:
                return f"{path}.{key}"
            if str(key) in {"screen", "rerank", "veto", "assessment"} and isinstance(
                child, (Mapping, list, tuple)
            ):
                return f"{path}.{key}"
            found = _contains_forbidden_result_value(child, f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            found = _contains_forbidden_result_value(child, f"{path}[{index}]")
            if found:
                return found
    return None


def validate_received_job(
    *,
    job_dir: str | Path,
    schedule_path: str | Path,
    manifest_path: str | Path,
    authorization_path: str | Path,
    job_index: int,
    output: str | Path,
) -> dict[str, Any]:
    manifest = _load_mapping(Path(manifest_path), "Attempt07 preflight manifest")
    manifest_file = Path(manifest_path)
    _validate_package_manifest(manifest, manifest_file)
    manifest_sha = sha256_file(manifest_file)
    authorization_sha = _validate_authorization(
        Path(authorization_path), manifest=manifest, manifest_sha256=manifest_sha
    )
    schedule = _read_schedule(Path(schedule_path))
    if not 0 <= job_index < ATTEMPT07_JOB_COUNT:
        raise ValueError("Attempt07 preflight job_index is outside 0..4")
    spec = schedule[job_index]
    directory = Path(job_dir)
    done = _load_mapping(directory / "DONE.json", "Attempt07 preflight DONE")
    fixed_done = {
        "schema": DONE_SCHEMA,
        "status": "complete",
        "run_name": manifest.get("run_name"),
        "job_index": job_index,
        "job_id": spec["job_id"],
        "source_root_index": spec["source_root_index"],
        "batch_child_selectors": spec["batch_child_selectors"],
        "native_batch_threads": ATTEMPT07_PREFLIGHT_NATIVE_BATCH_THREADS,
        "output_prefix": spec["output_prefix"],
        "manifest_sha256": manifest_sha,
        "authorization_sha256": authorization_sha,
        "schedule_sha256": manifest.get("schedule_sha256"),
        "attempt07_plan_sha256": M43_ATTEMPT07_PLAN_SHA256,
        "source_merged_sha256": ATTEMPT07_PREFLIGHT_SOURCE_SHA256,
        "model_sha256": ATTEMPT06_FROZEN_MODEL_SHA256,
        "ai_profiles_sha256": AI_PROFILES_SHA256,
        "current_profile_resolved": False,
        "current_profile_mutated": False,
        "arm_selection_performed": False,
        "teacher_values_exported": False,
        "new_root_generated": False,
        "runtime_policy_activated": False,
    }
    if (
        set(done) != _DONE_KEYS
        or type(done.get("job_index")) is not int
        or type(done.get("source_root_index")) is not int
        or type(done.get("native_batch_threads")) is not int
        or type(done.get("batch_child_selectors")) is not bool
        or type(done.get("current_profile_resolved")) is not bool
        or type(done.get("current_profile_mutated")) is not bool
        or type(done.get("arm_selection_performed")) is not bool
        or type(done.get("teacher_values_exported")) is not bool
        or type(done.get("new_root_generated")) is not bool
        or type(done.get("runtime_policy_activated")) is not bool
        or any(done.get(key) != value for key, value in fixed_done.items())
    ):
        raise ValueError("Attempt07 preflight DONE identity changed")
    elapsed = done.get("elapsed_seconds")
    peak_rss = done.get("peak_rss_bytes")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not 0.0 <= float(elapsed) < float("inf")
        or type(peak_rss) is not int
        or peak_rss < 0
    ):
        raise ValueError("Attempt07 preflight DONE resource metrics changed")
    bindings = {
        "preflight.json": "output_sha256",
        "checkpoint.json": "checkpoint_sha256",
        "heartbeat.json": "heartbeat_sha256",
        "summary.json": "summary_sha256",
        "run.log": "run_log_sha256",
    }
    for name, field in bindings.items():
        _require_hash(directory / name, done.get(field), f"received {name}")
    row = _load_mapping(directory / "preflight.json", "Attempt07 preflight output")
    if (
        row.get("schema") != ATTEMPT07_PREFLIGHT_ROW_SCHEMA
        or row.get("status") != "pass_preflight_only_no_arm_selection"
        or row.get("source", {}).get("source_root_index") != spec["source_root_index"]
        or row.get("execution", {}).get("batch_child_selectors")
        is not spec["batch_child_selectors"]
        or row.get("contract", {}).get("plan_sha256") != M43_ATTEMPT07_PLAN_SHA256
        or row.get("science_boundary", {}).get("arm_selection_allowed") is not False
        or row.get("science_boundary", {}).get("current_profile_resolved") is not False
    ):
        raise ValueError("Attempt07 preflight redacted output identity changed")
    forbidden = _contains_forbidden_result_value(row)
    if forbidden:
        raise ValueError(f"Attempt07 preflight output exposes arm value at {forbidden}")
    audit = {
        "schema": RECEIVE_AUDIT_SCHEMA,
        "status": "validated_value_redacted_preflight_proof",
        "job_index": job_index,
        "job_id": spec["job_id"],
        "source_root_index": spec["source_root_index"],
        "batch_child_selectors": spec["batch_child_selectors"],
        "output_sha256": done["output_sha256"],
        "manifest_sha256": sha256_file(manifest_path),
        "authorization_sha256": authorization_sha,
        "teacher_values_opened": False,
        "arm_selection_performed": False,
        "current_profile_mutated": False,
    }
    _write_atomic_no_clobber(Path(output), canonical_json_bytes(audit))
    return audit


def merge_received_jobs(
    *,
    jobs_root: str | Path,
    schedule_path: str | Path,
    manifest_path: str | Path,
    output: str | Path,
    receipt: str | Path,
) -> dict[str, Any]:
    schedule = _read_schedule(Path(schedule_path))
    root = Path(jobs_root)
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for spec in schedule:
        directory = root / spec["output_prefix"]
        audit = _load_mapping(directory / "received_audit.json", "receive audit")
        proof = _load_mapping(directory / "preflight.json", "preflight proof")
        if (
            audit.get("schema") != RECEIVE_AUDIT_SCHEMA
            or audit.get("job_index") != spec["job_index"]
            or audit.get("output_sha256") != sha256_file(directory / "preflight.json")
            or _contains_forbidden_result_value(proof)
        ):
            raise ValueError("Attempt07 preflight merge found an invalid job")
        audits.append(audit)
        rows.append(proof)
    _write_atomic_no_clobber(
        Path(output), b"".join(canonical_json_bytes(row) for row in rows)
    )
    authorization_hashes = {audit.get("authorization_sha256") for audit in audits}
    if len(authorization_hashes) != 1 or not all(
        isinstance(value, str) and len(value) == 64 for value in authorization_hashes
    ):
        raise ValueError("Attempt07 preflight merge authorization binding changed")
    payload = {
        "schema": RECEIVE_MERGE_SCHEMA,
        "status": "merged_five_value_redacted_preflight_proofs",
        "jobs": ATTEMPT07_JOB_COUNT,
        "merged_sha256": sha256_file(output),
        "manifest_sha256": sha256_file(manifest_path),
        "authorization_sha256": next(iter(authorization_hashes)),
        "audit_sha256": [
            sha256_file(root / spec["output_prefix"] / "received_audit.json")
            for spec in schedule
        ],
        "teacher_values_opened": False,
        "arm_selection_performed": False,
        "current_profile_mutated": False,
    }
    _write_atomic_no_clobber(Path(receipt), canonical_json_bytes(payload))
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    package = commands.add_parser("package")
    package.add_argument("--repo-root", required=True)
    package.add_argument("--run-dir", required=True)
    package.add_argument("--run-name", required=True)
    package.add_argument("--base-run-dir", default=str(DEFAULT_BASE_RUN_DIR))
    package.add_argument("--preflight-plan", default=str(DEFAULT_PREFLIGHT_PLAN))
    package.add_argument("--attempt07-plan", default=str(DEFAULT_ATTEMPT07_PLAN))
    package.add_argument("--source", default=str(DEFAULT_SOURCE))
    package.add_argument("--startup", default=str(DEFAULT_STARTUP))
    package.add_argument("--resume-existing", action="store_true")
    authorization = commands.add_parser("create-launch-authorization")
    authorization.add_argument("--manifest", required=True)
    authorization.add_argument("--attempt07-tests-receipt", required=True)
    authorization.add_argument("--rust-parity-receipt", required=True)
    authorization.add_argument("--package-tests-receipt", required=True)
    authorization.add_argument("--output", required=True)
    evidence = commands.add_parser("create-local-evidence-receipt")
    evidence.add_argument(
        "--suite",
        required=True,
        choices=("attempt07_pytest", "rust_parity", "package_tests"),
    )
    evidence.add_argument("--passed", required=True, type=int)
    evidence.add_argument("--failed", required=True, type=int)
    evidence.add_argument("--manifest")
    evidence.add_argument("--output", required=True)
    validate = commands.add_parser("validate-received-job")
    validate.add_argument("--job-dir", required=True)
    validate.add_argument("--schedule", required=True)
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--authorization", required=True)
    validate.add_argument("--job-index", required=True, type=int)
    validate.add_argument("--output", required=True)
    merge = commands.add_parser("merge-received-jobs")
    merge.add_argument("--jobs-root", required=True)
    merge.add_argument("--schedule", required=True)
    merge.add_argument("--manifest", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "package":
        result = package_preflight_run(
            repo_root=args.repo_root,
            run_dir=args.run_dir,
            run_name=args.run_name,
            base_run_dir=args.base_run_dir,
            preflight_plan=args.preflight_plan,
            attempt07_plan=args.attempt07_plan,
            source=args.source,
            startup=args.startup,
            resume_existing=args.resume_existing,
        )
    elif args.command == "create-launch-authorization":
        result = create_launch_authorization(
            manifest_path=args.manifest,
            attempt07_tests_receipt=args.attempt07_tests_receipt,
            rust_parity_receipt=args.rust_parity_receipt,
            package_tests_receipt=args.package_tests_receipt,
            output=args.output,
        )
    elif args.command == "create-local-evidence-receipt":
        result = create_local_evidence_receipt(
            suite=args.suite,
            passed=args.passed,
            failed=args.failed,
            manifest_path=args.manifest,
            output=args.output,
        )
    elif args.command == "validate-received-job":
        result = validate_received_job(
            job_dir=args.job_dir,
            schedule_path=args.schedule,
            manifest_path=args.manifest,
            authorization_path=args.authorization,
            job_index=args.job_index,
            output=args.output,
        )
    else:
        result = merge_received_jobs(
            jobs_root=args.jobs_root,
            schedule_path=args.schedule,
            manifest_path=args.manifest,
            output=args.output,
            receipt=args.receipt,
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ATTEMPT06_BASE_MANIFEST_SHA256",
    "ATTEMPT07_JOB_COUNT",
    "ATTEMPT07_MACHINE_TYPE",
    "DONE_SCHEMA",
    "LAUNCH_AUTHORIZATION_SCHEMA",
    "PACKAGE_MANIFEST_SCHEMA",
    "SCHEDULE_SCHEMA",
    "build_preflight_schedule",
    "create_launch_authorization",
    "create_local_evidence_receipt",
    "merge_received_jobs",
    "package_preflight_run",
    "sha256_file",
    "validate_preflight_schedule",
    "validate_received_job",
]
