"""Canonical full-Python-tree runtime anchor builder for Attempt08.

The full ``src/ofc_regular/**/*.py`` tree is frozen, including this validator,
``__init__.py``, wrappers, and modules reached by dynamic imports.  Only the
small constant-only anchor contract is excluded to avoid self-reference.
Pinned manifests are not merely hashed: every model/native artifact row is
opened and verified against the package tree.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .hu_m43_attempt08_runtime_identity import (
    ATTEMPT08_GCP_IMAGE_ID,
    ATTEMPT08_GCP_IMAGE_NAME,
    ATTEMPT08_GCP_IMAGE_SELF_LINK,
)


ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SCHEMA = (
    "hu_m43_attempt08_runtime_semantic_anchor_v3"
)
ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SCHEMA = (
    "hu_m43_attempt08_runtime_source_closure_v2"
)
ATTEMPT08_RUNTIME_CONTRACT_RELATIVE = (
    "src/ofc_regular/hu_m43_attempt08_runtime_anchor_contract.py"
)
ATTEMPT08_RUNTIME_TREE_GLOB = (
    "src/ofc_regular/**/*.py+frozen-config-orchestration-and-gate-files"
)
ATTEMPT08_RUNTIME_SEMANTIC_FILES = (
    "configs/fl_ev_regular_2k.json",
    "scripts/startup_hu_m43_attempt08_preflight.sh",
    "scripts/Start-GcpHuM43Attempt08PreflightRun.ps1",
    "scripts/Get-GcpHuM43Attempt08PreflightRunStatus.ps1",
    "scripts/Receive-GcpHuM43Attempt08PreflightRun.ps1",
    # Attempt08's development orchestration dot-sources Attempt08 Common,
    # which in turn dot-sources Attempt04 Common for the bounded-process,
    # exact-GCS-copy, hashing, and path-safety trust boundary.  Freeze both
    # transitive scripts and copy both into the immutable package.
    "scripts/HuM43Attempt04Spot.Common.ps1",
    "scripts/HuM43Attempt08Spot.Common.ps1",
    "scripts/startup_hu_m43_attempt08_development.sh",
    "scripts/Start-GcpHuM43Attempt08DevelopmentRun.ps1",
    "scripts/Get-GcpHuM43Attempt08DevelopmentRunStatus.ps1",
    "scripts/Receive-GcpHuM43Attempt08DevelopmentRun.ps1",
    "tests/test_run_hu_m43_attempt08_preflight.py",
    "tests/test_aggregate_hu_m43_attempt08_preflight.py",
    "tests/test_finalize_hu_m43_attempt08_preflight.py",
    "tests/test_hu_m43_attempt08_preflight_spot.py",
    "tests/test_hu_m43_attempt08_runtime_anchor.py",
    "tests/test_hu_m43_attempt08_spot.py",
    "tests/test_run_hu_m43_attempt08_development.py",
    "tests/test_select_hu_m43_attempt08_development.py",
    "tests/test_run_hu_m43_attempt08_future_audit.py",
    "tests/test_hu_m43_attempt08_development_spot_scripts.py",
)

_MODEL_RELATIVE = (
    "outputs/hu_joint_policy/m43_attempt05_old_dev900_architecture_once/"
    "lambda_rank_candidate.pkl"
)
_MODEL_SHA256 = "e7fa728308c8973e2e202baf79309e30b9b6f58c37431d8ea55850390abc28c3"
_MODEL_MANIFEST_SHA256 = (
    "e9b9d9748bb2b2f31a4baa0d928b60c05f254ab5d7915a7461fb8402e1e7ddb8"
)
_NATIVE_MANIFEST_SHA256 = (
    "ec295d070ac7bb21a82aeb2f432b8f6f191e61b63027bc446e8bd5f39f51569f"
)
_MODEL_MANIFEST_SCHEMA = "hu_m43_attempt02_source_model_manifest_v1"
_NATIVE_MANIFEST_SCHEMA = "hu_m43_attempt02_source_native_manifest_v1"
ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256 = (
    "a0ce16d1ab481528ac53f2e94fd9037af7f94004a38b8cf8116537bbf4277c68"
)
_REQUIREMENTS_RELATIVE = "configs/hu_m43_attempt08_runtime_requirements.txt"

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_ARTIFACT_ROOT = _REPO_ROOT / (
    "outputs/gcp_runs/regular-hu-m43-attempt06-preflight-final-20260714-1154/"
    "package_src"
)
DEFAULT_MODEL_MANIFEST_PATH = DEFAULT_RUNTIME_ARTIFACT_ROOT / (
    "source_model_manifest.json"
)
DEFAULT_NATIVE_MANIFEST_PATH = DEFAULT_RUNTIME_ARTIFACT_ROOT / (
    "source_native_manifest.json"
)
DEFAULT_REQUIREMENTS_PATH = _REPO_ROOT / _REQUIREMENTS_RELATIVE


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_source_closure_rows(repository_root: str | Path) -> list[dict[str, Any]]:
    """Hash every Python file in the producer tree except the self contract."""

    root = Path(repository_root).resolve()
    source_root = root / "src" / "ofc_regular"
    if not source_root.is_dir():
        raise ValueError("Attempt08 runtime source tree is missing")
    paths = sorted(
        (
            path
            for path in source_root.rglob("*.py")
            if path.is_file()
            and path.relative_to(root).as_posix()
            != ATTEMPT08_RUNTIME_CONTRACT_RELATIVE
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not paths or any(path.is_symlink() for path in paths):
        raise ValueError("Attempt08 runtime source tree is empty or has a symlink")
    paths.extend(root / relative for relative in ATTEMPT08_RUNTIME_SEMANTIC_FILES)
    paths = sorted(paths, key=lambda path: path.relative_to(root).as_posix())
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise ValueError("Attempt08 runtime semantic file is missing or is a symlink")
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in paths
    ]


def runtime_source_closure_sha256(repository_root: str | Path) -> str:
    return hashlib.sha256(
        canonical_json_bytes(runtime_source_closure_rows(repository_root))
    ).hexdigest()


def normalized_anchor_contract_sha256(repository_root: str | Path) -> str:
    """Hash the contract while replacing only its three digest/count literals."""

    path = Path(repository_root).resolve() / ATTEMPT08_RUNTIME_CONTRACT_RELATIVE
    raw = path.read_text(encoding="utf-8-sig")
    names = (
        "ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256",
        "ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256",
        "ATTEMPT08_RUNTIME_SOURCE_FILE_COUNT",
    )
    normalized = raw
    replacements = {
        names[0]: '"<SOURCE_CLOSURE_SHA256>"',
        names[1]: '"<SEMANTIC_ANCHOR_SHA256>"',
        names[2]: "<SOURCE_FILE_COUNT>",
    }
    for name in names:
        pattern = rf"(?m)^{name}\s*=\s*.+$"
        matches = re.findall(pattern, normalized)
        if len(matches) != 1:
            raise ValueError("Attempt08 runtime anchor contract shape changed")
        normalized = re.sub(
            pattern, f"{name} = {replacements[name]}", normalized, count=1
        )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def runtime_semantic_anchor_payload(repository_root: str | Path) -> dict[str, Any]:
    rows = runtime_source_closure_rows(repository_root)
    source_sha = hashlib.sha256(canonical_json_bytes(rows)).hexdigest()
    return {
        "schema": ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SCHEMA,
        "source_closure": {
            "schema": ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SCHEMA,
            "tree_glob": ATTEMPT08_RUNTIME_TREE_GLOB,
            "excluded_self_contract": ATTEMPT08_RUNTIME_CONTRACT_RELATIVE,
            "excluded_self_contract_normalized_sha256": (
                normalized_anchor_contract_sha256(repository_root)
            ),
            "file_count": len(rows),
            "files": rows,
            "sha256": source_sha,
        },
        "artifacts": {
            "lambda_model": {"path": _MODEL_RELATIVE, "sha256": _MODEL_SHA256},
            "model_manifest": {
                "logical_path": "runtime/source_model_manifest.json",
                "sha256": _MODEL_MANIFEST_SHA256,
            },
            "native_manifest": {
                "logical_path": "runtime/source_native_manifest.json",
                "sha256": _NATIVE_MANIFEST_SHA256,
            },
        },
        "environment": {
            "gcp_image_name": ATTEMPT08_GCP_IMAGE_NAME,
            "gcp_image_id": ATTEMPT08_GCP_IMAGE_ID,
            "gcp_image_self_link": ATTEMPT08_GCP_IMAGE_SELF_LINK,
            "requirements_path": _REQUIREMENTS_RELATIVE,
            "requirements_sha256": ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
        },
    }


def _safe_artifact_path(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("Attempt08 runtime manifest path is unsafe")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != value
    ):
        raise ValueError("Attempt08 runtime manifest path is unsafe")
    target = root.joinpath(*relative.parts).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("Attempt08 runtime manifest path escapes artifact root") from exc
    return target


def _validate_artifact_manifest(
    *,
    path: Path,
    artifact_root: Path,
    expected_sha256: str,
    expected_schema: str,
    count_key: str,
    rows_key: str,
    expected_count: int,
    native: bool,
) -> None:
    if not path.is_file() or _sha256_file(path) != expected_sha256:
        raise ValueError(f"Attempt08 runtime manifest drifted: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", count_key, rows_key}
        or payload.get("schema") != expected_schema
        or type(payload.get(count_key)) is not int
        or payload[count_key] != expected_count
        or not isinstance(payload.get(rows_key), list)
        or len(payload[rows_key]) != expected_count
    ):
        raise ValueError("Attempt08 runtime artifact manifest schema changed")
    expected_row_keys = {"path", "bytes", "sha256", "platform"} if native else {
        "path",
        "bytes",
        "sha256",
    }
    seen: set[str] = set()
    for row in payload[rows_key]:
        if not isinstance(row, dict) or set(row) != expected_row_keys:
            raise ValueError("Attempt08 runtime artifact manifest row changed")
        relative = row.get("path")
        target = _safe_artifact_path(artifact_root, relative)
        if not isinstance(relative, str) or relative in seen:
            raise ValueError("Attempt08 runtime artifact manifest has duplicate path")
        seen.add(relative)
        if (
            type(row.get("bytes")) is not int
            or row["bytes"] < 1
            or not isinstance(row.get("sha256"), str)
            or len(row["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in row["sha256"])
            or not target.is_file()
            or target.stat().st_size != row["bytes"]
            or _sha256_file(target) != row["sha256"]
            or (native and row.get("platform") != "linux-x86_64")
        ):
            raise ValueError(f"Attempt08 runtime artifact drifted: {relative}")


def validate_runtime_semantic_anchor_payload(
    payload: Mapping[str, Any],
    *,
    expected_source_closure_sha256: str,
    expected_anchor_sha256: str,
    expected_source_file_count: int,
) -> None:
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema",
        "source_closure",
        "artifacts",
        "environment",
    }:
        raise ValueError("Attempt08 runtime semantic anchor fields changed")
    source = payload.get("source_closure")
    artifacts = payload.get("artifacts")
    environment = payload.get("environment")
    if (
        payload.get("schema") != ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SCHEMA
        or not isinstance(source, Mapping)
        or set(source)
        != {
            "schema",
            "tree_glob",
            "excluded_self_contract",
            "excluded_self_contract_normalized_sha256",
            "file_count",
            "files",
            "sha256",
        }
        or source.get("schema") != ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SCHEMA
        or source.get("tree_glob") != ATTEMPT08_RUNTIME_TREE_GLOB
        or source.get("excluded_self_contract")
        != ATTEMPT08_RUNTIME_CONTRACT_RELATIVE
        or not isinstance(source.get("excluded_self_contract_normalized_sha256"), str)
        or len(source["excluded_self_contract_normalized_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in source["excluded_self_contract_normalized_sha256"]
        )
        or source.get("file_count") != expected_source_file_count
        or not isinstance(source.get("files"), list)
        or len(source["files"]) != expected_source_file_count
        or source.get("sha256") != expected_source_closure_sha256
        or hashlib.sha256(canonical_json_bytes(source["files"])).hexdigest()
        != expected_source_closure_sha256
        or artifacts
        != {
            "lambda_model": {"path": _MODEL_RELATIVE, "sha256": _MODEL_SHA256},
            "model_manifest": {
                "logical_path": "runtime/source_model_manifest.json",
                "sha256": _MODEL_MANIFEST_SHA256,
            },
            "native_manifest": {
                "logical_path": "runtime/source_native_manifest.json",
                "sha256": _NATIVE_MANIFEST_SHA256,
            },
        }
        or environment
        != {
            "gcp_image_name": ATTEMPT08_GCP_IMAGE_NAME,
            "gcp_image_id": ATTEMPT08_GCP_IMAGE_ID,
            "gcp_image_self_link": ATTEMPT08_GCP_IMAGE_SELF_LINK,
            "requirements_path": _REQUIREMENTS_RELATIVE,
            "requirements_sha256": ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
        }
        or hashlib.sha256(canonical_json_bytes(dict(payload))).hexdigest()
        != expected_anchor_sha256
    ):
        raise ValueError("Attempt08 runtime semantic anchor payload changed")


def validate_runtime_semantic_anchor(
    *,
    repository_root: str | Path,
    expected_source_closure_sha256: str,
    expected_anchor_sha256: str,
    expected_source_file_count: int,
    runtime_artifact_root: str | Path = DEFAULT_RUNTIME_ARTIFACT_ROOT,
    model_manifest_path: str | Path = DEFAULT_MODEL_MANIFEST_PATH,
    native_manifest_path: str | Path = DEFAULT_NATIVE_MANIFEST_PATH,
    requirements_path: str | Path = DEFAULT_REQUIREMENTS_PATH,
) -> dict[str, Any]:
    """Recompute the full source tree and verify every pinned artifact row."""

    root = Path(repository_root).resolve()
    artifact_root = Path(runtime_artifact_root).resolve()
    payload = runtime_semantic_anchor_payload(root)
    validate_runtime_semantic_anchor_payload(
        payload,
        expected_source_closure_sha256=expected_source_closure_sha256,
        expected_anchor_sha256=expected_anchor_sha256,
        expected_source_file_count=expected_source_file_count,
    )
    model = root / _MODEL_RELATIVE
    if not model.is_file() or _sha256_file(model) != _MODEL_SHA256:
        raise ValueError("Attempt08 runtime Lambda model drifted")
    requirements = Path(requirements_path).resolve()
    if (
        not requirements.is_file()
        or _sha256_file(requirements) != ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256
    ):
        raise ValueError("Attempt08 runtime requirements drifted")
    _validate_artifact_manifest(
        path=Path(model_manifest_path).resolve(),
        artifact_root=artifact_root,
        expected_sha256=_MODEL_MANIFEST_SHA256,
        expected_schema=_MODEL_MANIFEST_SCHEMA,
        count_key="model_count",
        rows_key="models",
        expected_count=11,
        native=False,
    )
    _validate_artifact_manifest(
        path=Path(native_manifest_path).resolve(),
        artifact_root=artifact_root,
        expected_sha256=_NATIVE_MANIFEST_SHA256,
        expected_schema=_NATIVE_MANIFEST_SCHEMA,
        count_key="binary_count",
        rows_key="binaries",
        expected_count=2,
        native=True,
    )
    return payload


__all__ = [
    "ATTEMPT08_RUNTIME_CONTRACT_RELATIVE",
    "ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SCHEMA",
    "ATTEMPT08_RUNTIME_SEMANTIC_FILES",
    "ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SCHEMA",
    "ATTEMPT08_RUNTIME_TREE_GLOB",
    "ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256",
    "DEFAULT_MODEL_MANIFEST_PATH",
    "DEFAULT_NATIVE_MANIFEST_PATH",
    "DEFAULT_RUNTIME_ARTIFACT_ROOT",
    "DEFAULT_REQUIREMENTS_PATH",
    "canonical_json_bytes",
    "runtime_semantic_anchor_payload",
    "normalized_anchor_contract_sha256",
    "runtime_source_closure_rows",
    "runtime_source_closure_sha256",
    "validate_runtime_semantic_anchor",
    "validate_runtime_semantic_anchor_payload",
]
