"""Immutable source freeze for the Attempt12 distilled runtime.

The development teacher already has a separately frozen semantic identity.  A
distilled policy adds new inference and acceptance code, so it needs a new
source closure without rewriting that teacher identity.  This module builds a
byte-deterministic ZIP from an explicit source root and binds the new closure
to the existing teacher lineage and pinned external runtime.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import stat
import uuid
import zipfile
from collections.abc import Iterator, Mapping as MappingABC
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .hu_m43_attempt12_contract import (
    ATTEMPT12_LAMBDA_MODEL_SHA256,
    M43_ATTEMPT12_PLAN_SHA256,
)
from .hu_m43_attempt08_runtime_identity import (
    ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256 as ATTEMPT12_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
    ATTEMPT08_GCP_IMAGE_ID as ATTEMPT12_GCP_IMAGE_ID,
    ATTEMPT08_GCP_IMAGE_NAME as ATTEMPT12_GCP_IMAGE_NAME,
    ATTEMPT08_GCP_IMAGE_SELF_LINK as ATTEMPT12_GCP_IMAGE_SELF_LINK,
    ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256 as ATTEMPT12_RUNTIME_REQUIREMENTS_SHA256,
)


ATTEMPT12_DISTILLED_RUNTIME_MANIFEST_SCHEMA = (
    "hu_m43_attempt12_distilled_runtime_source_manifest_v1"
)
ATTEMPT12_DISTILLED_RUNTIME_SOURCE_CLOSURE_SCHEMA = (
    "hu_m43_attempt12_distilled_runtime_source_closure_v1"
)
ATTEMPT12_DISTILLED_RUNTIME_SEMANTIC_CLOSURE_SCHEMA = (
    "hu_m43_attempt12_distilled_runtime_semantic_closure_v1"
)
_ARCHIVE_SCHEMA = "hu_m43_attempt12_distilled_runtime_source_zip_v1"
_STATUS = "frozen_before_distilled_fit_or_population_execution"
ATTEMPT12_RUNTIME_ACTIVATED = False
# Structural dispatch is available only through the complete immutable binding
# checked by the Attempt12 acceptance loader.  This does not activate a profile.
ATTEMPT12_SHARED_RUNTIME_DISPATCH_ENABLED = True
_REQUIREMENTS_PATH = "configs/hu_m43_attempt08_runtime_requirements.txt"
ATTEMPT12_DISTILLED_POPULATION_PLAN_PATH = (
    "configs/hu_joint_policy_m43_attempt12_population.json"
)
ATTEMPT12_DISTILLED_RUNTIME_REGISTRY_PATHS = (
    "configs/hu_joint_policy_m2_teacher.json",
    "configs/hu_joint_policy_m3_status.json",
    "configs/hu_joint_policy_m42_status.json",
    "configs/hu_joint_policy_m43_attempt02.json",
    "configs/hu_joint_policy_m43_attempt03.json",
    "configs/hu_joint_policy_m43_attempt03_model_freeze.json",
    "configs/hu_joint_policy_m43_attempt03_model_freeze_original_ce47.json",
    "configs/hu_joint_policy_m43_population.json",
    "configs/hu_joint_policy_m43_attempt03_population.json",
    "configs/hu_joint_policy_m43_attempt05.json",
    "configs/hu_joint_policy_m43_attempt06.json",
    "configs/hu_joint_policy_m43_attempt06_status.json",
    "configs/hu_joint_policy_m43_attempt07.json",
    "configs/hu_joint_policy_m43_attempt07_preflight.json",
    "configs/hu_joint_policy_m43_attempt08.json",
    "configs/hu_joint_policy_m43_attempt08_preflight.json",
    "configs/hu_joint_policy_m43_attempt08_population.json",
    "configs/hu_joint_policy_m43_attempt09.json",
    "configs/hu_joint_policy_m43_attempt10.json",
    "configs/hu_joint_policy_m43_attempt10_population.json",
    "configs/hu_joint_policy_m43_attempt11.json",
    "configs/hu_joint_policy_m43_attempt11_population.json",
    "configs/hu_joint_policy_m43_attempt12.json",
    "configs/hu_joint_policy_m43_pilot.json",
    "configs/hu_turn3_stage1_cycle10_margin8_candidate.json",
    "configs/hu_turn3_stage1_cycle11_margin8_support_cycle10_4_experiment.json",
    "configs/hu_turn3_stage2_mc32_500k_margin10_candidate.json",
    "configs/hu_turn3_stage3_mc32_500k_plus_m8_12_f128_100k_w2_margin10_candidate.json",
    "configs/hu_turn3_stage4ft_stage3init_mc2048_w4_lr1e4_rejected.json",
    "configs/hu_turn3_stage5_mistake_mining_mc128_1k_plan.json",
    "configs/hu_turn3_stage5_stage3init_boundary_mc2048_mistake_lr5e5_rejected.json",
    "configs/hu_turn3_stage6_stage3init_s3vss5_mc2048_w6_lr5e5_candidate.json",
    "configs/hu_turn3_stage7_reference_override_plan.json",
    "configs/hu_turn3_stage9d_s10_r02_score10_first_tail_veto_candidate.json",
    "configs/hu_turn3_stage9d_s10_r05_score10_first_tail_veto_candidate.json",
    "configs/hu_turn3_stage9d_s10_score10_first_tail_veto_candidate.json",
    "configs/hu_turn3_stage9d_s15_r02_score10_first_tail_veto_candidate.json",
)
ATTEMPT12_BOUND_EXECUTION_MODULES = (
    "ofc_regular.hu_m43_attempt12_distilled_runtime",
    "ofc_regular.hu_m43_attempt12_distilled_model",
    "ofc_regular.train_hu_m43_attempt12_distilled",
    "ofc_regular.evaluate_hu_m4_population",
    "ofc_regular.hu_m43_joint_model_loader",
    "ofc_regular.hu_m4_t1_policy",
    "ofc_regular.validate_hu_m43_attempt12_acceptance",
)
ATTEMPT12_DISTILLED_SOURCE_MODEL_MANIFEST_SHA256 = (
    "e9b9d9748bb2b2f31a4baa0d928b60c05f254ab5d7915a7461fb8402e1e7ddb8"
)
ATTEMPT12_DISTILLED_SOURCE_NATIVE_MANIFEST_SHA256 = (
    "ec295d070ac7bb21a82aeb2f432b8f6f191e61b63027bc446e8bd5f39f51569f"
)
ATTEMPT12_DISTILLED_RUNTIME_DEPENDENCY_SCHEMA = (
    "hu_m43_attempt12_distilled_runtime_dependencies_v1"
)
_MODEL_MANIFEST_SCHEMA = "hu_m43_attempt02_source_model_manifest_v1"
_NATIVE_MANIFEST_SCHEMA = "hu_m43_attempt02_source_native_manifest_v1"
_MODEL_MANIFEST_PATH = "source_model_manifest.json"
_NATIVE_MANIFEST_PATH = "source_native_manifest.json"
_EXPECTED_MODEL_COUNT = 11
_EXPECTED_NATIVE_COUNT = 2
_SELECTION = {
    "pyproject": "pyproject.toml",
    "python_sources": "src/**/*.py",
    "runtime_requirements": _REQUIREMENTS_PATH,
    "population_plan": ATTEMPT12_DISTILLED_POPULATION_PLAN_PATH,
    "seed_registry_configs": list(ATTEMPT12_DISTILLED_RUNTIME_REGISTRY_PATHS),
}
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_ZIP_MODE = stat.S_IFREG | 0o644
_ZIP_METADATA = {
    "compression": "stored",
    "create_system": 3,
    "timestamp": list(_ZIP_TIMESTAMP),
    "unix_mode": "0100644",
    "entry_comment": "",
    "entry_extra": "",
    "archive_comment": "",
}


def canonical_json_bytes(value: Any) -> bytes:
    """Return the sole accepted byte representation for freeze manifests."""

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_distilled_runtime_source_freeze(
    *,
    source_root: str | Path,
    output_archive: str | Path,
    output_manifest: str | Path,
) -> dict[str, Any]:
    """Freeze the exact distilled source tree without consulting ambient cwd."""

    requested_root = Path(source_root)
    if requested_root.is_symlink():
        raise ValueError("Attempt12 distilled source root must not be a symlink")
    root = requested_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Attempt12 distilled source root must be a real directory")
    archive_path = Path(output_archive)
    manifest_path = Path(output_manifest)
    if archive_path.resolve() == manifest_path.resolve():
        raise ValueError("Attempt12 archive and manifest paths must be distinct")
    if archive_path.exists() or manifest_path.exists():
        raise FileExistsError("Attempt12 distilled source freeze is immutable")

    rows = _source_rows(root)
    file_set_payload = {
        "schema": ATTEMPT12_DISTILLED_RUNTIME_SOURCE_CLOSURE_SCHEMA,
        "selection": dict(_SELECTION),
        "files": rows,
    }
    file_set_sha = _sha256_bytes(canonical_json_bytes(file_set_payload))
    semantic_payload = _semantic_payload(source_closure_sha256=file_set_sha)
    semantic_sha = _sha256_bytes(canonical_json_bytes(semantic_payload))

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive = archive_path.with_name(
        f".{archive_path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        _write_deterministic_zip(root, rows, temporary_archive)
        archive_sha = sha256_file(temporary_archive)
        archive_bytes = temporary_archive.stat().st_size
        os.link(temporary_archive, archive_path)
    finally:
        temporary_archive.unlink(missing_ok=True)

    manifest = {
        "schema": ATTEMPT12_DISTILLED_RUNTIME_MANIFEST_SCHEMA,
        "status": _STATUS,
        "file_set": {
            **file_set_payload,
            "file_count": len(rows),
            "sha256": file_set_sha,
        },
        "semantic_closure": {**semantic_payload, "sha256": semantic_sha},
        "archive": {
            "schema": _ARCHIVE_SCHEMA,
            "sha256": archive_sha,
            "bytes": archive_bytes,
            "entries": len(rows),
            "fixed_metadata": dict(_ZIP_METADATA),
        },
    }
    try:
        _write_new_bytes(manifest_path, canonical_json_bytes(manifest))
    except BaseException:
        archive_path.unlink(missing_ok=True)
        raise
    validate_distilled_runtime_source_archive(
        archive_path=archive_path, manifest=manifest_path
    )
    return manifest


def load_and_validate_distilled_runtime_manifest(
    source: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Load and strictly validate a distilled-runtime source manifest."""

    if isinstance(source, Mapping):
        manifest = dict(source)
    else:
        raw = Path(source).read_bytes()
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Attempt12 distilled runtime manifest is invalid") from error
        if not isinstance(value, dict):
            raise ValueError("Attempt12 distilled runtime manifest must be a mapping")
        manifest = value
        if raw != canonical_json_bytes(manifest):
            raise ValueError("Attempt12 distilled runtime manifest is not canonical")
    _validate_manifest(manifest)
    return manifest


def validate_distilled_runtime_source_archive(
    *,
    archive_path: str | Path,
    manifest: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Validate archive bytes, names, metadata, and every source-file digest."""

    validated = load_and_validate_distilled_runtime_manifest(manifest)
    archive = Path(archive_path)
    if not archive.is_file() or archive.is_symlink():
        raise ValueError("Attempt12 distilled runtime archive is missing or unsafe")
    archive_contract = _mapping(validated["archive"], "archive")
    if (
        archive.stat().st_size != archive_contract["bytes"]
        or sha256_file(archive) != archive_contract["sha256"]
    ):
        raise ValueError("Attempt12 distilled runtime archive bytes changed")

    expected_rows = {
        str(row["path"]): row
        for row in _sequence(validated["file_set"]["files"], "source files")
    }
    try:
        with zipfile.ZipFile(archive, "r") as bundle:
            if bundle.comment != b"":
                raise ValueError("Attempt12 distilled runtime ZIP comment changed")
            infos = bundle.infolist()
            names = [info.filename for info in infos]
            for name in names:
                _safe_relative_path(name)
            if len(names) != len(set(names)):
                raise ValueError("Attempt12 distilled runtime ZIP has duplicate entries")
            if names != sorted(expected_rows) or set(names) != set(expected_rows):
                raise ValueError("Attempt12 distilled runtime ZIP file set changed")
            for info in infos:
                _validate_zip_metadata(info)
                row = expected_rows[info.filename]
                if info.file_size != row["bytes"]:
                    raise ValueError("Attempt12 distilled runtime ZIP size changed")
                encoded = bundle.read(info)
                if (
                    len(encoded) != row["bytes"]
                    or _sha256_bytes(encoded) != row["sha256"]
                ):
                    raise ValueError("Attempt12 distilled runtime ZIP content changed")
    except zipfile.BadZipFile as error:
        raise ValueError("Attempt12 distilled runtime archive is not a ZIP") from error
    return validated


def validate_distilled_runtime_extracted_tree(
    *,
    extracted_root: str | Path,
    manifest: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an extracted tree against the frozen manifest, with no extras."""

    validated = load_and_validate_distilled_runtime_manifest(manifest)
    root = Path(extracted_root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Attempt12 extracted runtime root is missing or unsafe")
    expected_rows = {
        str(row["path"]): row
        for row in _sequence(validated["file_set"]["files"], "source files")
    }
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Attempt12 extracted runtime tree contains a symlink")
        relative = path.relative_to(root).as_posix()
        _safe_relative_path(relative)
        if path.is_dir():
            actual_directories.add(relative)
        elif path.is_file():
            actual_files.add(relative)
        else:
            raise ValueError("Attempt12 extracted runtime tree has a special file")
    expected_directories = {
        PurePosixPath(name).parent.as_posix()
        for name in expected_rows
        if PurePosixPath(name).parent.as_posix() != "."
    }
    expected_directories |= {
        parent.as_posix()
        for name in expected_rows
        for parent in PurePosixPath(name).parents
        if parent.as_posix() != "."
    }
    if actual_files != set(expected_rows) or actual_directories != expected_directories:
        raise ValueError("Attempt12 extracted runtime tree file set changed")
    for name, row in expected_rows.items():
        target = root.joinpath(*PurePosixPath(name).parts)
        if target.stat().st_size != row["bytes"] or sha256_file(target) != row["sha256"]:
            raise ValueError("Attempt12 extracted runtime source bytes changed")
    requirements = root.joinpath(*PurePosixPath(_REQUIREMENTS_PATH).parts)
    if sha256_file(requirements) != ATTEMPT12_RUNTIME_REQUIREMENTS_SHA256:
        raise ValueError("Attempt12 extracted runtime requirements changed")
    return validated


def validate_distilled_runtime_dependencies(
    dependency_root: str | Path,
) -> dict[str, Any]:
    """Verify the exact historical model/native closure used by development.

    The root must contain the two pinned manifests and every file they name.
    No model or native binary is accepted merely because its live-repository
    path happens to have the expected basename.
    """

    root = Path(dependency_root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Attempt12 runtime dependency root is missing or unsafe")
    root = root.resolve(strict=True)
    model_path = root / _MODEL_MANIFEST_PATH
    native_path = root / _NATIVE_MANIFEST_PATH
    for path, expected, label in (
        (
            model_path,
            ATTEMPT12_DISTILLED_SOURCE_MODEL_MANIFEST_SHA256,
            "source-model manifest",
        ),
        (
            native_path,
            ATTEMPT12_DISTILLED_SOURCE_NATIVE_MANIFEST_SHA256,
            "source-native manifest",
        ),
    ):
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
            raise ValueError(f"Attempt12 pinned {label} bytes changed")
    model = _json_mapping(model_path, "source-model manifest")
    native = _json_mapping(native_path, "source-native manifest")
    if (
        model.get("schema") != _MODEL_MANIFEST_SCHEMA
        or model.get("model_count") != _EXPECTED_MODEL_COUNT
        or len(_sequence(model.get("models"), "models")) != _EXPECTED_MODEL_COUNT
    ):
        raise ValueError("Attempt12 pinned source-model manifest semantics changed")
    if (
        native.get("schema") != _NATIVE_MANIFEST_SCHEMA
        or native.get("binary_count") != _EXPECTED_NATIVE_COUNT
        or len(_sequence(native.get("binaries"), "native binaries"))
        != _EXPECTED_NATIVE_COUNT
    ):
        raise ValueError("Attempt12 pinned source-native manifest semantics changed")
    normalized_models = _validate_dependency_rows(
        root, model["models"], label="model", expect_platform=False
    )
    normalized_binaries = _validate_dependency_rows(
        root, native["binaries"], label="native binary", expect_platform=True
    )
    closure_payload = {
        "schema": ATTEMPT12_DISTILLED_RUNTIME_DEPENDENCY_SCHEMA,
        "source_model_manifest_sha256": (
            ATTEMPT12_DISTILLED_SOURCE_MODEL_MANIFEST_SHA256
        ),
        "source_native_manifest_sha256": (
            ATTEMPT12_DISTILLED_SOURCE_NATIVE_MANIFEST_SHA256
        ),
        "models": normalized_models,
        "binaries": normalized_binaries,
    }
    return {
        **closure_payload,
        "model_count": len(normalized_models),
        "binary_count": len(normalized_binaries),
        "sha256": _sha256_bytes(canonical_json_bytes(closure_payload)),
    }


def _validate_frozen_execution_module_paths(
    *,
    extracted_root: str | Path,
    manifest: str | Path | Mapping[str, Any],
    module_names: tuple[str, ...] | list[str],
) -> tuple[Path, dict[str, str]]:
    validated = validate_distilled_runtime_extracted_tree(
        extracted_root=extracted_root, manifest=manifest
    )
    root = Path(extracted_root).resolve(strict=True)
    expected_rows = {
        str(row["path"]): row for row in validated["file_set"]["files"]
    }
    result: dict[str, str] = {}
    for name in module_names:
        if not isinstance(name, str) or not name:
            raise ValueError("Attempt12 frozen execution module name is invalid")
        module = importlib.import_module(name)
        raw_file = getattr(module, "__file__", None)
        if not isinstance(raw_file, str):
            raise ValueError(f"Attempt12 execution module has no file: {name}")
        module_file = Path(raw_file).resolve(strict=True)
        try:
            relative = module_file.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(
                f"Attempt12 execution module was imported outside frozen tree: {name}"
            ) from error
        row = expected_rows.get(relative)
        if row is None or sha256_file(module_file) != row["sha256"]:
            raise ValueError(
                f"Attempt12 execution module is not in the frozen manifest: {name}"
            )
        result[name] = relative
    return root, result


@dataclass(frozen=True, init=False)
class FrozenExecutionModulesAttestation(MappingABC[str, str]):
    """Immutable proof created only by validating active imported modules."""

    extracted_root: str
    module_paths: tuple[tuple[str, str], ...]

    def __init__(
        self,
        *,
        extracted_root: str | Path,
        manifest: str | Path | Mapping[str, Any],
        module_names: tuple[str, ...] | list[str],
    ) -> None:
        root, paths = _validate_frozen_execution_module_paths(
            extracted_root=extracted_root,
            manifest=manifest,
            module_names=module_names,
        )
        object.__setattr__(self, "extracted_root", str(root))
        object.__setattr__(self, "module_paths", tuple(paths.items()))

    def __getitem__(self, key: str) -> str:
        return dict(self.module_paths)[key]

    def __iter__(self) -> Iterator[str]:
        return (name for name, _path in self.module_paths)

    def __len__(self) -> int:
        return len(self.module_paths)

    def covers(self, module_names: tuple[str, ...] | list[str]) -> bool:
        required = tuple(module_names)
        return (
            all(isinstance(name, str) and bool(name) for name in required)
            and set(required).issubset(dict(self.module_paths))
        )


def validate_frozen_execution_modules(
    *,
    extracted_root: str | Path,
    manifest: str | Path | Mapping[str, Any],
    module_names: tuple[str, ...] | list[str],
) -> FrozenExecutionModulesAttestation:
    """Prove imports came from the tree and return a non-forgeable-by-API receipt."""

    return FrozenExecutionModulesAttestation(
        extracted_root=extracted_root,
        manifest=manifest,
        module_names=module_names,
    )


def extract_distilled_runtime_source_archive(
    *,
    archive_path: str | Path,
    manifest: str | Path | Mapping[str, Any],
    output_root: str | Path,
) -> dict[str, Any]:
    """Validate first, then extract only frozen entries into a new directory."""

    validated = validate_distilled_runtime_source_archive(
        archive_path=archive_path, manifest=manifest
    )
    destination = Path(output_root)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Attempt12 extracted runtime destination must be new")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    staging.mkdir()
    try:
        with zipfile.ZipFile(archive_path, "r") as bundle:
            for row in validated["file_set"]["files"]:
                name = str(row["path"])
                relative = _safe_relative_path(name)
                target = staging.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as handle:
                    handle.write(bundle.read(name))
                    handle.flush()
                    os.fsync(handle.fileno())
        validate_distilled_runtime_extracted_tree(
            extracted_root=staging, manifest=validated
        )
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return validated


def _source_rows(root: Path) -> list[dict[str, Any]]:
    pyproject = root / "pyproject.toml"
    requirements = root.joinpath(*PurePosixPath(_REQUIREMENTS_PATH).parts)
    source = root / "src"
    configs = root / "configs"
    if not pyproject.is_file() or not requirements.is_file() or not source.is_dir():
        raise ValueError("Attempt12 distilled runtime source boundary is incomplete")
    if (
        pyproject.is_symlink()
        or requirements.is_symlink()
        or source.is_symlink()
        or configs.is_symlink()
    ):
        raise ValueError("Attempt12 distilled runtime source boundary has a symlink")
    if sha256_file(requirements) != ATTEMPT12_RUNTIME_REQUIREMENTS_SHA256:
        raise ValueError("Attempt12 distilled runtime requirements drifted")
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError("Attempt12 distilled runtime Python tree has a symlink")
    python_files = sorted(
        (path for path in source.rglob("*.py") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not python_files:
        raise ValueError("Attempt12 distilled runtime Python source tree is empty")
    registry_files = [
        root.joinpath(*PurePosixPath(relative).parts)
        for relative in ATTEMPT12_DISTILLED_RUNTIME_REGISTRY_PATHS
    ]
    if any(not path.is_file() or path.is_symlink() for path in registry_files):
        raise ValueError("Attempt12 distilled seed-registry source boundary is incomplete")
    population_plan = root.joinpath(
        *PurePosixPath(ATTEMPT12_DISTILLED_POPULATION_PLAN_PATH).parts
    )
    if not population_plan.is_file() or population_plan.is_symlink():
        raise ValueError("Attempt12 distilled population plan source is missing")
    selected = [
        pyproject,
        requirements,
        population_plan,
        *registry_files,
        *python_files,
    ]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in selected:
        relative = path.relative_to(root).as_posix()
        _safe_relative_path(relative)
        if relative in seen:
            raise ValueError("Attempt12 distilled runtime source selection is duplicated")
        seen.add(relative)
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    rows.sort(key=lambda row: str(row["path"]))
    return rows


def _semantic_payload(*, source_closure_sha256: str) -> dict[str, Any]:
    return {
        "schema": ATTEMPT12_DISTILLED_RUNTIME_SEMANTIC_CLOSURE_SCHEMA,
        "distilled_source_closure_sha256": _require_sha256(
            source_closure_sha256, "distilled source closure"
        ),
        "teacher_contract": {
            "schema": "hu_m43_attempt12_teacher_lineage_v1",
            "plan_sha256": M43_ATTEMPT12_PLAN_SHA256,
            "candidate_model_sha256": ATTEMPT12_LAMBDA_MODEL_SHA256,
        },
        "activation": {
            "runtime_activated": ATTEMPT12_RUNTIME_ACTIVATED,
            "shared_dispatch_enabled": ATTEMPT12_SHARED_RUNTIME_DISPATCH_ENABLED,
            "current_profile_mutated": False,
        },
        "external_runtime": {
            "requirements_path": _REQUIREMENTS_PATH,
            "requirements_sha256": ATTEMPT12_RUNTIME_REQUIREMENTS_SHA256,
            "runtime_fingerprint_sha256": (
                ATTEMPT12_EXPECTED_RUNTIME_FINGERPRINT_SHA256
            ),
            "gcp_image": {
                "name": ATTEMPT12_GCP_IMAGE_NAME,
                "id": ATTEMPT12_GCP_IMAGE_ID,
                "self_link": ATTEMPT12_GCP_IMAGE_SELF_LINK,
            },
        },
        "runtime_dependencies": {
            "schema": ATTEMPT12_DISTILLED_RUNTIME_DEPENDENCY_SCHEMA,
            "source_model_manifest_sha256": (
                ATTEMPT12_DISTILLED_SOURCE_MODEL_MANIFEST_SHA256
            ),
            "source_native_manifest_sha256": (
                ATTEMPT12_DISTILLED_SOURCE_NATIVE_MANIFEST_SHA256
            ),
            "model_count": _EXPECTED_MODEL_COUNT,
            "binary_count": _EXPECTED_NATIVE_COUNT,
        },
    }


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if set(manifest) != {"schema", "status", "file_set", "semantic_closure", "archive"}:
        raise ValueError("Attempt12 distilled runtime manifest keys changed")
    if (
        manifest.get("schema") != ATTEMPT12_DISTILLED_RUNTIME_MANIFEST_SCHEMA
        or manifest.get("status") != _STATUS
    ):
        raise ValueError("Attempt12 distilled runtime manifest identity changed")
    file_set = _mapping(manifest.get("file_set"), "file_set")
    if set(file_set) != {"schema", "selection", "files", "file_count", "sha256"}:
        raise ValueError("Attempt12 distilled runtime file-set keys changed")
    if (
        file_set.get("schema")
        != ATTEMPT12_DISTILLED_RUNTIME_SOURCE_CLOSURE_SCHEMA
        or file_set.get("selection") != _SELECTION
    ):
        raise ValueError("Attempt12 distilled runtime source selection changed")
    rows = _sequence(file_set.get("files"), "source files")
    normalized_rows: list[dict[str, Any]] = []
    names: list[str] = []
    for raw in rows:
        row = _mapping(raw, "source row")
        if set(row) != {"path", "bytes", "sha256"}:
            raise ValueError("Attempt12 distilled runtime source row keys changed")
        if not isinstance(row.get("path"), str):
            raise ValueError("Attempt12 distilled runtime source path is invalid")
        name = row["path"]
        _safe_relative_path(name)
        size = row.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("Attempt12 distilled runtime source size is invalid")
        normalized_rows.append(
            {
                "path": name,
                "bytes": size,
                "sha256": _require_sha256(row.get("sha256"), "source file"),
            }
        )
        names.append(name)
    if (
        not rows
        or names != sorted(names)
        or len(names) != len(set(names))
        or "pyproject.toml" not in names
        or _REQUIREMENTS_PATH not in names
        or ATTEMPT12_DISTILLED_POPULATION_PLAN_PATH not in names
        or any(name not in names for name in ATTEMPT12_DISTILLED_RUNTIME_REGISTRY_PATHS)
        or not any(name.startswith("src/") and name.endswith(".py") for name in names)
        or file_set.get("file_count") != len(rows)
    ):
        raise ValueError("Attempt12 distilled runtime source file set is invalid")
    closure_payload = {
        "schema": ATTEMPT12_DISTILLED_RUNTIME_SOURCE_CLOSURE_SCHEMA,
        "selection": dict(_SELECTION),
        "files": normalized_rows,
    }
    closure_sha = _sha256_bytes(canonical_json_bytes(closure_payload))
    if _require_sha256(file_set.get("sha256"), "source closure") != closure_sha:
        raise ValueError("Attempt12 distilled runtime source closure changed")

    semantic = _mapping(manifest.get("semantic_closure"), "semantic closure")
    expected_semantic = _semantic_payload(source_closure_sha256=closure_sha)
    if set(semantic) != set(expected_semantic) | {"sha256"} or any(
        semantic.get(key) != value for key, value in expected_semantic.items()
    ):
        raise ValueError("Attempt12 distilled runtime semantic binding changed")
    semantic_sha = _sha256_bytes(canonical_json_bytes(expected_semantic))
    if _require_sha256(semantic.get("sha256"), "semantic closure") != semantic_sha:
        raise ValueError("Attempt12 distilled runtime semantic closure changed")

    archive = _mapping(manifest.get("archive"), "archive")
    if set(archive) != {
        "schema",
        "sha256",
        "bytes",
        "entries",
        "fixed_metadata",
    }:
        raise ValueError("Attempt12 distilled runtime archive keys changed")
    if (
        archive.get("schema") != _ARCHIVE_SCHEMA
        or archive.get("fixed_metadata") != _ZIP_METADATA
        or archive.get("entries") != len(rows)
        or isinstance(archive.get("bytes"), bool)
        or not isinstance(archive.get("bytes"), int)
        or archive["bytes"] <= 0
    ):
        raise ValueError("Attempt12 distilled runtime archive contract changed")
    _require_sha256(archive.get("sha256"), "source archive")


def _write_deterministic_zip(
    root: Path, rows: list[dict[str, Any]], destination: Path
) -> None:
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_STORED) as bundle:
        bundle.comment = b""
        for row in rows:
            name = str(row["path"])
            info = zipfile.ZipInfo(name, date_time=_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = _ZIP_MODE << 16
            info.internal_attr = 0
            info.extra = b""
            info.comment = b""
            bundle.writestr(info, root.joinpath(*PurePosixPath(name).parts).read_bytes())


def _validate_zip_metadata(info: zipfile.ZipInfo) -> None:
    mode = info.external_attr >> 16
    if (
        info.is_dir()
        or info.date_time != _ZIP_TIMESTAMP
        or info.compress_type != zipfile.ZIP_STORED
        or info.create_system != 3
        or mode != _ZIP_MODE
        or info.internal_attr != 0
        or info.extra != b""
        or info.comment != b""
    ):
        raise ValueError("Attempt12 distilled runtime ZIP metadata changed")


def _safe_relative_path(value: str) -> PurePosixPath:
    if (
        not value
        or "\\" in value
        or "\x00" in value
        or ":" in value
        or value.startswith("/")
        or value.endswith("/")
    ):
        raise ValueError("Attempt12 distilled runtime path is unsafe")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or path.as_posix() != value
    ):
        raise ValueError("Attempt12 distilled runtime path is unsafe")
    return path


def _write_new_bytes(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_bytes(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"Attempt12 {label} SHA-256 is invalid")
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Attempt12 {label} must be a mapping")
    return dict(value)


def _sequence(value: Any, label: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        raise ValueError(f"Attempt12 {label} must be a list")
    return value


def _json_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Attempt12 {label} is invalid") from error
    return _mapping(value, label)


def _validate_dependency_rows(
    root: Path,
    rows: list[Any],
    *,
    label: str,
    expect_platform: bool,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    expected_keys = {"path", "bytes", "sha256"} | (
        {"platform"} if expect_platform else set()
    )
    for raw in rows:
        row = _mapping(raw, label)
        if set(row) != expected_keys or not isinstance(row.get("path"), str):
            raise ValueError(f"Attempt12 pinned {label} row changed")
        relative = _safe_relative_path(row["path"])
        if row["path"] in names:
            raise ValueError(f"Attempt12 pinned {label} path is duplicated")
        names.add(row["path"])
        size = row.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"Attempt12 pinned {label} size is invalid")
        digest = _require_sha256(row.get("sha256"), label)
        target = root.joinpath(*relative.parts)
        try:
            resolved_target = target.resolve(strict=True)
            resolved_target.relative_to(root)
        except (FileNotFoundError, ValueError) as error:
            raise ValueError(
                f"Attempt12 pinned {label} path escapes its root: {row['path']}"
            ) from error
        if any(
            parent.is_symlink()
            for parent in (target, *target.parents)
            if parent != root and root in parent.parents
        ):
            raise ValueError(f"Attempt12 pinned {label} path contains a symlink")
        if (
            not target.is_file()
            or target.is_symlink()
            or target.stat().st_size != size
            or sha256_file(target) != digest
        ):
            raise ValueError(f"Attempt12 pinned {label} bytes changed: {row['path']}")
        normalized_row = {"path": row["path"], "bytes": size, "sha256": digest}
        if expect_platform:
            if row.get("platform") != "linux-x86_64":
                raise ValueError("Attempt12 pinned native binary platform changed")
            normalized_row["platform"] = "linux-x86_64"
        normalized.append(normalized_row)
    return normalized


__all__ = [
    "ATTEMPT12_BOUND_EXECUTION_MODULES",
    "ATTEMPT12_EXPECTED_RUNTIME_FINGERPRINT_SHA256",
    "ATTEMPT12_GCP_IMAGE_ID",
    "ATTEMPT12_GCP_IMAGE_NAME",
    "ATTEMPT12_GCP_IMAGE_SELF_LINK",
    "ATTEMPT12_DISTILLED_RUNTIME_DEPENDENCY_SCHEMA",
    "ATTEMPT12_DISTILLED_RUNTIME_MANIFEST_SCHEMA",
    "ATTEMPT12_DISTILLED_POPULATION_PLAN_PATH",
    "ATTEMPT12_DISTILLED_RUNTIME_REGISTRY_PATHS",
    "ATTEMPT12_DISTILLED_RUNTIME_SEMANTIC_CLOSURE_SCHEMA",
    "ATTEMPT12_DISTILLED_RUNTIME_SOURCE_CLOSURE_SCHEMA",
    "ATTEMPT12_DISTILLED_SOURCE_MODEL_MANIFEST_SHA256",
    "ATTEMPT12_DISTILLED_SOURCE_NATIVE_MANIFEST_SHA256",
    "ATTEMPT12_RUNTIME_REQUIREMENTS_SHA256",
    "ATTEMPT12_RUNTIME_ACTIVATED",
    "ATTEMPT12_SHARED_RUNTIME_DISPATCH_ENABLED",
    "FrozenExecutionModulesAttestation",
    "build_distilled_runtime_source_freeze",
    "canonical_json_bytes",
    "extract_distilled_runtime_source_archive",
    "load_and_validate_distilled_runtime_manifest",
    "sha256_file",
    "validate_distilled_runtime_extracted_tree",
    "validate_distilled_runtime_dependencies",
    "validate_distilled_runtime_source_archive",
    "validate_frozen_execution_modules",
]
