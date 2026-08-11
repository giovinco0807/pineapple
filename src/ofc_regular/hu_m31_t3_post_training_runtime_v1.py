"""Immutable real-runtime bundle for the M3.1 post-training ABR grid.

The ABR worker is only scientifically meaningful when the Python source,
all behavior-policy models, both native engines, and the offline wheelhouse
are frozen together.  This module creates that closure once, validates every
member again from the stored archives, and writes ``READY`` last.

It never launches cloud resources and never resolves or changes ``current``.
The production packager has no placeholder-model or network-install mode.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import tarfile
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .hu_m31_t3_behavior_roots import M31_T3_BEHAVIOR_PROFILES
from . import hu_m31_t3_step6d_fresh_quality_transport_v1 as fresh_transport


BUNDLE_SCHEMA = "hu_m31_t3_post_training_abr_runtime_bundle_v1"
READY_SCHEMA = "hu_m31_t3_post_training_abr_runtime_ready_v1"
MANIFEST_NAME = "ABR_RUNTIME_BUNDLE_MANIFEST.json"
READY_NAME = "ABR_RUNTIME_BUNDLE_READY.json"
RUNTIME_ARCHIVE_RELATIVE_PATH = "runtime/runtime.tar.gz"
WHEELHOUSE_ARCHIVE_RELATIVE_PATH = "wheelhouse/wheelhouse.zip"
WHEELHOUSE_SOURCE_MANIFEST_RELATIVE_PATH = (
    "wheelhouse/wheelhouse_source_manifest.json"
)
ACCEPTED_LIBRARY_RELATIVE_PATH = (
    "native/accepted/libofc_hu_m3_engine.so"
)
DIAGNOSTIC_LIBRARY_RELATIVE_PATH = (
    "native/diagnostic/libofc_hu_m3_engine.so"
)

PROFILE_REGISTRY_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)
ABR_TEACHER_SOURCE_SHA256 = (
    "61c6d932a89adb0f6d00b87a6c90251a09d5965be77a3d68b26005e3c40caa19"
)
ACCEPTED_SEARCH_SOURCE_SHA256 = (
    "b12d19eaf194baf4b5eb09c968972d3e206ce84b1135f8c3980b64d9b09d2384"
)
ACCEPTED_CANDIDATE_LIBRARY_SHA256 = (
    "4050e04b22d7943da9e1691402a2b34cf3d3781041a44557f35e2dfcf366d55d"
)
ACCEPTED_CANDIDATE_LIBRARY_BYTES = 1_230_800

RAW_WHEELHOUSE_ARCHIVE_SHA256 = (
    "8704107c8e63f2947128f7f5a0c3ba3b45c0aace3636115d0720b77a74b40405"
)
RAW_WHEELHOUSE_ARCHIVE_BYTES = 259_523_538
RAW_WHEELHOUSE_MANIFEST_SHA256 = (
    "f8c1b3fc02d075ff44e6823c3121cf379360cef296def89ec07365cceeb6fb3d"
)
RAW_WHEELHOUSE_MANIFEST_BYTES = 5_612
RAW_WHEELHOUSE_ENTRY_COUNT = 23
RAW_WHEELHOUSE_ENTRIES_SHA256 = (
    "4a60df57508f2b29664e9e5907a2c60ff9688677e31c7f8d479924430db530c3"
)

REQUIRED_SOURCE_PATHS = frozenset(
    {
        "pyproject.toml",
        "src/ofc_regular/__init__.py",
        "src/ofc_regular/ai_profiles.py",
        "src/ofc_regular/hu_m31_t3_abr_teacher_v1.py",
        "src/ofc_regular/hu_m31_t3_behavior_roots.py",
        "src/ofc_regular/hu_m31_t3_post_training_gcp_v1.py",
        "src/ofc_regular/hu_m31_t3_post_training_runtime_v1.py",
        "rust/hu_m3_engine/src/search.rs",
    }
)

_SHA256_LENGTH = 64
_ALLOWED_FILESYSTEMS = frozenset({"ext4", "xfs"})


@dataclass(frozen=True)
class ModelSpec:
    """One mandatory behavior-policy model and its ModelPaths binding."""

    model_paths_field: str
    bundle_field: str
    filename: str
    bytes: int
    sha256: str

    def record(self) -> dict[str, Any]:
        return {
            "model_paths_field": self.model_paths_field,
            "bundle_field": self.bundle_field,
            "path": f"models/{self.filename}",
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        "opening",
        "opening",
        "opening_stage7_torch_wide.pt",
        16_788_660,
        "4cfec60e3035323d10348f4f28938c24428880edc557a5e824723d32838fe939",
    ),
    ModelSpec(
        "turn1",
        "turn1",
        "turn1_stage6_torch_wide.pt",
        16_788_952,
        "ecffa86f0ccbf0bea1d697ae98bf4b9de316006e5aa4539eda4476a29f3f8b4a",
    ),
    ModelSpec(
        "turn2",
        "turn2",
        "turn2_stage8.pkl",
        12_903_515,
        "4be8e1c3647306a18b74676bf4835fcac9e03f3dde65fba391e9aaf5ac3929fe",
    ),
    ModelSpec(
        "turn3",
        "turn3",
        "turn3_stage6.pkl",
        7_183_883,
        "5996204bf904b258451097042f38bbd87b3f570fbcbe9bf5f4a1d2bdaf376737",
    ),
    ModelSpec(
        "hu_turn1_stage18_p1",
        "hu_turn1_stage18_p1",
        "hu_turn1_stage18_first1801_mc32_hgb_abs_aug3.pkl",
        1_468_838,
        "e910131715efbec7a116e411a8dc23dcfc9a904f2aef9e0eee9d4913a02dfee7",
    ),
    ModelSpec(
        "hu_turn1_stage18_p1_safe_selector",
        "hu_turn1_stage18_p1_safe_selector",
        "hu_turn1_stage18_highmc_safe_selector_a_meta_only.pkl",
        44_836,
        "838ff3421c2393648aaabeb7316c9e6c3fe97197485b1363be4e8b5dd33ae086",
    ),
    ModelSpec(
        "hu_turn0_stage19_p0",
        "hu_turn0_stage19_p0",
        "hu_turn0_stage3_top60_mc32_1000_hgb_baseline-delta_sew1_aug3.pkl",
        3_612_847,
        "8f360bc4c326f0d8d20710a4efc7c42c62a80cca11341b34a41a72a4c6aadf71",
    ),
    ModelSpec(
        "hu_turn0_stage19_p0_safe_selector",
        "hu_turn0_stage19_p0_safe_selector",
        "hu_turn0_stage19_safe_selector_highmc_candidate_delta_plus_meta.pkl",
        173_024,
        "7cd85b3a824816223ceb0d84870feb62e43d827452031e6f51de0ee452514abe",
    ),
    ModelSpec(
        "hu_turn2_stage8b",
        "hu_turn2_stage8b",
        "hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt",
        7_058_012,
        "2a6a52ee09e329852d00197e686509b0747ff86ef627a7b6ed9a21311ec25b3b",
    ),
    ModelSpec(
        "hu_turn3_stage7",
        "hu_turn3_stage7",
        "hu_turn3_stage7_reference_override_cached_rank_wide.pt",
        7_052_944,
        "727fb766b940f17c6c6d00a373b47d68bd07aa1095fe86633ba0fcab6c6e8f20",
    ),
    ModelSpec(
        "hu_turn3_stage7_reference",
        "hu_turn3_stage7_reference",
        "hu_turn3_stage3_mc32_500k_plus_m8_12_f128_100k_w2_cached_rank_wide.pt",
        7_053_380,
        "36f6ac00ca308b92aa4a11a7d6b3c65c4b5f65f6fb5fa2e71585d7f87d8dd112",
    ),
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _self_digest(value: Mapping[str, Any], field: str) -> str:
    copied = deepcopy(dict(value))
    copied.pop(field, None)
    return canonical_sha256(copied)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _regular_file(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    resolved = candidate.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{label} must be a regular file")
    return resolved


def _directory(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    resolved = candidate.resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise ValueError(f"{label} must be a directory")
    return resolved


def _safe_relative(value: str, label: str) -> str:
    pure = PurePosixPath(value)
    if (
        not value
        or pure.is_absolute()
        or ".." in pure.parts
        or "\\" in value
        or pure.as_posix() != value
    ):
        raise ValueError(f"{label} is unsafe")
    return value


def _record(path: Path, relative_path: str) -> dict[str, Any]:
    source = _regular_file(path, relative_path)
    return {
        "path": _safe_relative(relative_path, "bundle relative path"),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


def _read_canonical(path: str | Path, label: str) -> dict[str, Any]:
    source = _regular_file(path, label)
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw not in (
        canonical_bytes(value),
        canonical_bytes(value) + b"\n",
    ):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    raw = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _copy_new(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_stream:
        with destination.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, 1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())


def _decode_mount_path(value: str) -> str:
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _filesystem_type(path: str | Path) -> str:
    """Return the longest matching Linux mount's filesystem type."""

    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        resolved = probe.resolve()
        rows = Path("/proc/self/mountinfo").read_text(
            encoding="utf-8"
        ).splitlines()
    except (OSError, RuntimeError):
        return "unknown"
    matches: list[tuple[int, str]] = []
    for row in rows:
        before, separator, after = row.partition(" - ")
        if not separator:
            continue
        fields = before.split()
        tail = after.split()
        if len(fields) < 5 or not tail:
            continue
        mount = Path(_decode_mount_path(fields[4]))
        try:
            resolved.relative_to(mount)
        except ValueError:
            continue
        matches.append((len(str(mount)), tail[0]))
    return max(matches)[1] if matches else "unknown"


def _source_inventory(repository_root: Path) -> list[dict[str, Any]]:
    source_root = _directory(
        repository_root / "src" / "ofc_regular", "ofc_regular source directory"
    )
    for item in source_root.rglob("*"):
        if item.is_symlink():
            raise ValueError("runtime Python source tree contains a symlink")
    sources = [
        _regular_file(path, "runtime Python source")
        for path in source_root.rglob("*.py")
    ]
    pyproject = _regular_file(repository_root / "pyproject.toml", "pyproject")
    search = _regular_file(
        repository_root / "rust" / "hu_m3_engine" / "src" / "search.rs",
        "accepted Rust search source",
    )
    rows = [
        _record(
            path,
            path.relative_to(repository_root).as_posix(),
        )
        for path in sources
    ]
    rows.extend(
        (
            _record(pyproject, "pyproject.toml"),
            _record(search, "rust/hu_m3_engine/src/search.rs"),
        )
    )
    rows.sort(key=lambda row: str(row["path"]))
    paths = {str(row["path"]) for row in rows}
    by_path = {str(row["path"]): row for row in rows}
    if (
        not REQUIRED_SOURCE_PATHS.issubset(paths)
        or by_path["src/ofc_regular/ai_profiles.py"]["sha256"]
        != PROFILE_REGISTRY_SHA256
        or by_path["src/ofc_regular/hu_m31_t3_abr_teacher_v1.py"]["sha256"]
        != ABR_TEACHER_SOURCE_SHA256
        or by_path["rust/hu_m3_engine/src/search.rs"]["sha256"]
        != ACCEPTED_SEARCH_SOURCE_SHA256
    ):
        raise ValueError("ABR runtime source/profile/search pin changed")
    return rows


def _model_inventory(models_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if len(MODEL_SPECS) != 11:
        raise ValueError("ABR behavior model cardinality changed")
    if len({spec.filename for spec in MODEL_SPECS}) != len(MODEL_SPECS):
        raise ValueError("ABR behavior model inventory is not unique")
    for spec in MODEL_SPECS:
        path = _regular_file(models_root / spec.filename, spec.filename)
        row = spec.record()
        if path.stat().st_size != spec.bytes or sha256_file(path) != spec.sha256:
            raise ValueError(f"ABR behavior model changed: {spec.filename}")
        rows.append(row)
    return rows


def _validate_model_deserialization(
    models_root: Path, model_inventory_sha256: str
) -> dict[str, Any]:
    """Load the exact profile closure; optional-loader fallbacks are forbidden."""

    from .ai_profiles import ModelPaths, load_model_bundle

    path_arguments = {
        spec.model_paths_field: models_root / spec.filename
        for spec in MODEL_SPECS
    }
    bundle = load_model_bundle(
        ModelPaths(**path_arguments),
        profiles=set(M31_T3_BEHAVIOR_PROFILES),
    )
    required = [spec.bundle_field for spec in MODEL_SPECS]
    loaded = [
        field for field in required if getattr(bundle, field, None) is not None
    ]
    if loaded != required:
        missing = sorted(set(required) - set(loaded))
        raise ValueError(f"real ABR behavior model load is incomplete: {missing}")
    return {
        "behavior_profiles": list(M31_T3_BEHAVIOR_PROFILES),
        "required_bundle_fields": required,
        "loaded_bundle_fields": loaded,
        "model_inventory_sha256": model_inventory_sha256,
        "all_required_models_deserialized": True,
        "optional_loader_fallback_used": False,
        "placeholder_or_synthetic_model_used": False,
    }


def _raw_wheelhouse(
    archive_path: Path, manifest_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    archive = _regular_file(archive_path, "raw offline wheelhouse")
    manifest_file = _regular_file(
        manifest_path, "raw offline wheelhouse manifest"
    )
    if (
        sha256_file(archive) != RAW_WHEELHOUSE_ARCHIVE_SHA256
        or archive.stat().st_size != RAW_WHEELHOUSE_ARCHIVE_BYTES
        or sha256_file(manifest_file) != RAW_WHEELHOUSE_MANIFEST_SHA256
        or manifest_file.stat().st_size != RAW_WHEELHOUSE_MANIFEST_BYTES
    ):
        raise ValueError("raw offline wheelhouse source pin changed")
    manifest, summary = fresh_transport._validate_wheelhouse(  # type: ignore[attr-defined]
        archive_path=archive,
        manifest_path=manifest_file,
    )
    if (
        summary["entry_count"] != RAW_WHEELHOUSE_ENTRY_COUNT
        or summary["entries_sha256"] != RAW_WHEELHOUSE_ENTRIES_SHA256
    ):
        raise ValueError("raw offline wheelhouse inventory pin changed")
    entries = [
        {
            "path": str(row["filename"]),
            "sha256": str(row["sha256"]),
            "bytes": int(row["bytes"]),
        }
        for row in manifest["entries"]
    ]
    if [row["path"] for row in entries] != sorted(row["path"] for row in entries):
        raise ValueError("raw offline wheelhouse manifest ordering changed")
    return manifest, entries


def _runtime_files(
    repository_root: Path,
    models_root: Path,
    source_rows: Sequence[Mapping[str, Any]],
) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for row in source_rows:
        relative = str(row["path"])
        files.append((relative, repository_root.joinpath(*PurePosixPath(relative).parts)))
    for spec in MODEL_SPECS:
        files.append((f"models/{spec.filename}", models_root / spec.filename))
    files.sort(key=lambda item: item[0])
    return files


def _write_runtime_tar(path: Path, files: Sequence[tuple[str, Path]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(
                fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT
            ) as archive:
                for relative, source_path in files:
                    relative = _safe_relative(relative, "runtime tar member")
                    source = _regular_file(source_path, relative)
                    info = tarfile.TarInfo(f"runtime/{relative}")
                    info.mode = 0o644
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    info.size = source.stat().st_size
                    with source.open("rb") as stream:
                        archive.addfile(info, stream)
        raw.flush()
        os.fsync(raw.fileno())


def _write_prefixed_wheelhouse(
    destination: Path,
    source_archive: Path,
    entries: Sequence[Mapping[str, Any]],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_names = [str(row["path"]) for row in entries]
    with zipfile.ZipFile(source_archive, "r") as source:
        by_name = {info.filename: info for info in source.infolist()}
        if set(by_name) != set(expected_names):
            raise ValueError("raw wheelhouse changed before transport write")
        with destination.open("xb") as raw:
            with zipfile.ZipFile(
                raw,
                "w",
                compression=zipfile.ZIP_STORED,
                strict_timestamps=True,
            ) as output:
                for name in sorted(expected_names):
                    info = zipfile.ZipInfo(
                        f"wheelhouse/{name}",
                        date_time=(1980, 1, 1, 0, 0, 0),
                    )
                    info.compress_type = zipfile.ZIP_STORED
                    info.create_system = 3
                    info.external_attr = (0o100644 & 0xFFFF) << 16
                    with source.open(by_name[name], "r") as input_stream:
                        with output.open(info, "w") as output_stream:
                            shutil.copyfileobj(
                                input_stream, output_stream, 1024 * 1024
                            )
            raw.flush()
            os.fsync(raw.fileno())


def _tar_inventory(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with tarfile.open(_regular_file(path, "runtime archive"), "r:*") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if not names or names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("runtime archive ordering/uniqueness changed")
        for member in members:
            pure = PurePosixPath(member.name)
            if (
                not member.isfile()
                or pure.is_absolute()
                or ".." in pure.parts
                or len(pure.parts) < 2
                or pure.parts[0] != "runtime"
                or member.mode != 0o644
                or member.uid != 0
                or member.gid != 0
                or member.mtime != 0
                or member.uname != ""
                or member.gname != ""
            ):
                raise ValueError("runtime archive contains an unsafe member")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("runtime archive member cannot be read")
            digest = hashlib.sha256()
            size = 0
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
            rows.append(
                {
                    "path": PurePosixPath(*pure.parts[1:]).as_posix(),
                    "sha256": digest.hexdigest(),
                    "bytes": size,
                }
            )
    return rows


def _wheelhouse_inventory(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(_regular_file(path, "transport wheelhouse"), "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if not names or names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("transport wheelhouse ordering/uniqueness changed")
        for info in infos:
            pure = PurePosixPath(info.filename)
            mode = (info.external_attr >> 16) & 0xFFFF
            if (
                info.is_dir()
                or pure.is_absolute()
                or ".." in pure.parts
                or len(pure.parts) != 2
                or pure.parts[0] != "wheelhouse"
                or info.date_time != (1980, 1, 1, 0, 0, 0)
                or info.compress_type != zipfile.ZIP_STORED
                or mode & 0o170000 not in (0, 0o100000)
            ):
                raise ValueError("transport wheelhouse contains an unsafe member")
            digest = hashlib.sha256()
            size = 0
            with archive.open(info, "r") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
            rows.append(
                {
                    "path": pure.parts[1],
                    "sha256": digest.hexdigest(),
                    "bytes": size,
                }
            )
    return rows


def package_runtime_bundle(
    *,
    repository_root: str | Path,
    models_root: str | Path,
    raw_wheelhouse_path: str | Path,
    raw_wheelhouse_manifest_path: str | Path,
    accepted_library_path: str | Path,
    diagnostic_library_path: str | Path,
    expected_diagnostic_library_sha256: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Create one real, deterministic, create-only ABR runtime closure."""

    repository = _directory(repository_root, "repository root")
    models = _directory(models_root, "real behavior models root")
    raw_wheelhouse = _regular_file(
        raw_wheelhouse_path, "raw offline wheelhouse"
    )
    raw_manifest = _regular_file(
        raw_wheelhouse_manifest_path, "raw offline wheelhouse manifest"
    )
    accepted = _regular_file(accepted_library_path, "accepted native engine")
    diagnostic = _regular_file(
        diagnostic_library_path, "diagnostic native engine"
    )
    output = Path(output_root)
    if output.exists() or output.is_symlink():
        raise FileExistsError("ABR runtime bundle output is create-only")
    output.parent.mkdir(parents=True, exist_ok=True)
    filesystem_type = _filesystem_type(output.parent)
    if filesystem_type not in _ALLOWED_FILESYSTEMS:
        raise PermissionError("ABR runtime bundle output must be ext4 or xfs")
    if (
        not _is_sha256(expected_diagnostic_library_sha256)
        or sha256_file(accepted) != ACCEPTED_CANDIDATE_LIBRARY_SHA256
        or accepted.stat().st_size != ACCEPTED_CANDIDATE_LIBRARY_BYTES
        or sha256_file(diagnostic) != expected_diagnostic_library_sha256
    ):
        raise ValueError("accepted or diagnostic native engine pin changed")

    source_rows = _source_inventory(repository)
    model_rows = _model_inventory(models)
    model_inventory_sha256 = canonical_sha256(model_rows)
    model_load_receipt = _validate_model_deserialization(
        models, model_inventory_sha256
    )
    _raw_manifest_value, wheel_rows = _raw_wheelhouse(
        raw_wheelhouse, raw_manifest
    )

    # No package path exists before every source, model, native, and wheel pin
    # has passed.  Any later partial directory is intentionally retained.
    output.mkdir()
    runtime_path = output.joinpath(*PurePosixPath(RUNTIME_ARCHIVE_RELATIVE_PATH).parts)
    wheelhouse_path = output.joinpath(
        *PurePosixPath(WHEELHOUSE_ARCHIVE_RELATIVE_PATH).parts
    )
    wheelhouse_manifest_path = output.joinpath(
        *PurePosixPath(WHEELHOUSE_SOURCE_MANIFEST_RELATIVE_PATH).parts
    )
    accepted_path = output.joinpath(
        *PurePosixPath(ACCEPTED_LIBRARY_RELATIVE_PATH).parts
    )
    diagnostic_path = output.joinpath(
        *PurePosixPath(DIAGNOSTIC_LIBRARY_RELATIVE_PATH).parts
    )
    _write_runtime_tar(
        runtime_path,
        _runtime_files(repository, models, source_rows),
    )
    _write_prefixed_wheelhouse(wheelhouse_path, raw_wheelhouse, wheel_rows)
    _copy_new(raw_manifest, wheelhouse_manifest_path)
    _copy_new(accepted, accepted_path)
    _copy_new(diagnostic, diagnostic_path)

    runtime_inventory = _tar_inventory(runtime_path)
    expected_runtime = sorted(
        [
            *[dict(row) for row in source_rows],
            *[
                {
                    "path": f"models/{spec.filename}",
                    "sha256": spec.sha256,
                    "bytes": spec.bytes,
                }
                for spec in MODEL_SPECS
            ],
        ],
        key=lambda row: str(row["path"]),
    )
    transport_wheel_rows = _wheelhouse_inventory(wheelhouse_path)
    if runtime_inventory != expected_runtime or transport_wheel_rows != wheel_rows:
        raise ValueError("stored ABR runtime archive replay changed")

    artifacts = [
        _record(runtime_path, RUNTIME_ARCHIVE_RELATIVE_PATH),
        _record(wheelhouse_path, WHEELHOUSE_ARCHIVE_RELATIVE_PATH),
        _record(
            wheelhouse_manifest_path,
            WHEELHOUSE_SOURCE_MANIFEST_RELATIVE_PATH,
        ),
        _record(accepted_path, ACCEPTED_LIBRARY_RELATIVE_PATH),
        _record(diagnostic_path, DIAGNOSTIC_LIBRARY_RELATIVE_PATH),
    ]
    manifest_core = {
        "schema": BUNDLE_SCHEMA,
        "status": "complete_real_runtime_ready_for_local_smoke",
        "artifacts": artifacts,
        "artifact_inventory_sha256": canonical_sha256(artifacts),
        "runtime_inventory": runtime_inventory,
        "runtime_inventory_sha256": canonical_sha256(runtime_inventory),
        "source_inventory": source_rows,
        "source_inventory_sha256": canonical_sha256(source_rows),
        "model_inventory": model_rows,
        "model_inventory_sha256": model_inventory_sha256,
        "model_load_receipt": model_load_receipt,
        "wheelhouse": {
            "source_archive_sha256": RAW_WHEELHOUSE_ARCHIVE_SHA256,
            "source_archive_bytes": RAW_WHEELHOUSE_ARCHIVE_BYTES,
            "source_manifest_sha256": RAW_WHEELHOUSE_MANIFEST_SHA256,
            "source_manifest_bytes": RAW_WHEELHOUSE_MANIFEST_BYTES,
            "entry_count": RAW_WHEELHOUSE_ENTRY_COUNT,
            "entries_sha256": RAW_WHEELHOUSE_ENTRIES_SHA256,
            "transport_inventory": transport_wheel_rows,
            "transport_inventory_sha256": canonical_sha256(
                transport_wheel_rows
            ),
            "network_install_allowed": False,
        },
        "native": {
            "accepted_library_sha256": ACCEPTED_CANDIDATE_LIBRARY_SHA256,
            "accepted_library_bytes": ACCEPTED_CANDIDATE_LIBRARY_BYTES,
            "diagnostic_library_sha256": expected_diagnostic_library_sha256,
            "diagnostic_library_bytes": diagnostic.stat().st_size,
            "dual_engine_search_required": True,
        },
        "profile_registry_sha256": PROFILE_REGISTRY_SHA256,
        "abr_teacher_source_sha256": ABR_TEACHER_SOURCE_SHA256,
        "accepted_search_source_sha256": ACCEPTED_SEARCH_SOURCE_SHA256,
        "behavior_profiles": list(M31_T3_BEHAVIOR_PROFILES),
        "output_filesystem_type": filesystem_type,
        "real_models_only": True,
        "placeholder_or_synthetic_model_allowed": False,
        "hidden_opponent_discard_exposed": False,
        "cloud_execution_started": False,
        "current_profile_changed": False,
        "create_only": True,
        "ready_written_last": True,
    }
    manifest = {
        **manifest_core,
        "manifest_sha256": canonical_sha256(manifest_core),
    }
    manifest_path = output / MANIFEST_NAME
    _write_json_new(manifest_path, manifest)
    ready_artifacts = [
        *artifacts,
        _record(manifest_path, MANIFEST_NAME),
    ]
    ready_core = {
        "schema": READY_SCHEMA,
        "status": "ready_for_real_one_pair_local_smoke",
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": sha256_file(manifest_path),
        "files": ready_artifacts,
        "file_inventory_sha256": canonical_sha256(ready_artifacts),
        "file_count": len(ready_artifacts),
        "ready_published_after_all_payloads": True,
        "real_models_only": True,
        "cloud_execution_started": False,
        "current_profile_changed": False,
    }
    ready = {**ready_core, "ready_sha256": canonical_sha256(ready_core)}
    _write_json_new(output / READY_NAME, ready)
    return validate_runtime_bundle(manifest_path)


def _expected_model_rows() -> list[dict[str, Any]]:
    return [spec.record() for spec in MODEL_SPECS]


def validate_runtime_bundle(
    manifest_path: str | Path,
    *,
    expected_diagnostic_library_sha256: str | None = None,
) -> dict[str, Any]:
    """Replay every stored file and archive member, failing closed on drift."""

    path = _regular_file(manifest_path, "ABR runtime bundle manifest")
    if path.name != MANIFEST_NAME:
        raise ValueError("ABR runtime bundle manifest filename changed")
    root = path.parent
    manifest = _read_canonical(path, "ABR runtime bundle manifest")
    ready_path = _regular_file(root / READY_NAME, "ABR runtime bundle READY")
    ready = _read_canonical(ready_path, "ABR runtime bundle READY")
    artifacts = manifest.get("artifacts")
    source_rows = manifest.get("source_inventory")
    model_rows = manifest.get("model_inventory")
    runtime_rows = manifest.get("runtime_inventory")
    wheelhouse = manifest.get("wheelhouse")
    native = manifest.get("native")
    load_receipt = manifest.get("model_load_receipt")
    if (
        manifest.get("schema") != BUNDLE_SCHEMA
        or manifest.get("status")
        != "complete_real_runtime_ready_for_local_smoke"
        or manifest.get("manifest_sha256")
        != _self_digest(manifest, "manifest_sha256")
        or manifest.get("output_filesystem_type") not in _ALLOWED_FILESYSTEMS
        or manifest.get("profile_registry_sha256")
        != PROFILE_REGISTRY_SHA256
        or manifest.get("abr_teacher_source_sha256")
        != ABR_TEACHER_SOURCE_SHA256
        or manifest.get("accepted_search_source_sha256")
        != ACCEPTED_SEARCH_SOURCE_SHA256
        or manifest.get("behavior_profiles")
        != list(M31_T3_BEHAVIOR_PROFILES)
        or manifest.get("real_models_only") is not True
        or manifest.get("placeholder_or_synthetic_model_allowed") is not False
        or manifest.get("hidden_opponent_discard_exposed") is not False
        or manifest.get("cloud_execution_started") is not False
        or manifest.get("current_profile_changed") is not False
        or manifest.get("create_only") is not True
        or manifest.get("ready_written_last") is not True
        or not isinstance(artifacts, list)
        or not isinstance(source_rows, list)
        or not isinstance(model_rows, list)
        or not isinstance(runtime_rows, list)
        or not isinstance(wheelhouse, Mapping)
        or not isinstance(native, Mapping)
        or not isinstance(load_receipt, Mapping)
    ):
        raise ValueError("ABR runtime bundle manifest boundary changed")

    expected_paths = [
        RUNTIME_ARCHIVE_RELATIVE_PATH,
        WHEELHOUSE_ARCHIVE_RELATIVE_PATH,
        WHEELHOUSE_SOURCE_MANIFEST_RELATIVE_PATH,
        ACCEPTED_LIBRARY_RELATIVE_PATH,
        DIAGNOSTIC_LIBRARY_RELATIVE_PATH,
    ]
    if (
        [row.get("path") for row in artifacts] != expected_paths
        or manifest.get("artifact_inventory_sha256")
        != canonical_sha256(artifacts)
        or manifest.get("source_inventory_sha256")
        != canonical_sha256(source_rows)
        or manifest.get("model_inventory_sha256")
        != canonical_sha256(model_rows)
        or manifest.get("runtime_inventory_sha256")
        != canonical_sha256(runtime_rows)
        or model_rows != _expected_model_rows()
    ):
        raise ValueError("ABR runtime artifact/source/model inventory changed")

    artifact_by_path = {str(row["path"]): row for row in artifacts}
    if len(artifact_by_path) != len(artifacts):
        raise ValueError("ABR runtime artifact inventory is not unique")
    for relative, record in artifact_by_path.items():
        relative = _safe_relative(relative, "ABR runtime artifact")
        stored = _regular_file(
            root.joinpath(*PurePosixPath(relative).parts),
            relative,
        )
        if (
            sha256_file(stored) != record.get("sha256")
            or stored.stat().st_size != record.get("bytes")
        ):
            raise ValueError("ABR runtime stored artifact changed")

    runtime_path = root.joinpath(
        *PurePosixPath(RUNTIME_ARCHIVE_RELATIVE_PATH).parts
    )
    expected_runtime = sorted(
        [
            *[
                {
                    "path": str(row["path"]),
                    "sha256": str(row["sha256"]),
                    "bytes": int(row["bytes"]),
                }
                for row in source_rows
            ],
            *[
                {
                    "path": f"models/{spec.filename}",
                    "sha256": spec.sha256,
                    "bytes": spec.bytes,
                }
                for spec in MODEL_SPECS
            ],
        ],
        key=lambda row: row["path"],
    )
    if runtime_rows != expected_runtime or _tar_inventory(runtime_path) != runtime_rows:
        raise ValueError("ABR runtime tar member inventory changed")
    source_by_path = {str(row["path"]): row for row in source_rows}
    if (
        len(source_by_path) != len(source_rows)
        or not REQUIRED_SOURCE_PATHS.issubset(source_by_path)
        or source_by_path["src/ofc_regular/ai_profiles.py"]["sha256"]
        != PROFILE_REGISTRY_SHA256
        or source_by_path["src/ofc_regular/hu_m31_t3_abr_teacher_v1.py"][
            "sha256"
        ]
        != ABR_TEACHER_SOURCE_SHA256
        or source_by_path["rust/hu_m3_engine/src/search.rs"]["sha256"]
        != ACCEPTED_SEARCH_SOURCE_SHA256
    ):
        raise ValueError("ABR runtime source closure changed")

    wheel_rows = wheelhouse.get("transport_inventory")
    wheelhouse_path = root.joinpath(
        *PurePosixPath(WHEELHOUSE_ARCHIVE_RELATIVE_PATH).parts
    )
    raw_manifest_path = root.joinpath(
        *PurePosixPath(WHEELHOUSE_SOURCE_MANIFEST_RELATIVE_PATH).parts
    )
    raw_manifest = _read_canonical(
        raw_manifest_path, "stored raw wheelhouse manifest"
    )
    raw_entries = [
        {
            "path": str(row["filename"]),
            "sha256": str(row["sha256"]),
            "bytes": int(row["bytes"]),
        }
        for row in raw_manifest.get("entries", [])
    ]
    if (
        wheelhouse.get("source_archive_sha256")
        != RAW_WHEELHOUSE_ARCHIVE_SHA256
        or wheelhouse.get("source_archive_bytes")
        != RAW_WHEELHOUSE_ARCHIVE_BYTES
        or wheelhouse.get("source_manifest_sha256")
        != RAW_WHEELHOUSE_MANIFEST_SHA256
        or wheelhouse.get("source_manifest_bytes")
        != RAW_WHEELHOUSE_MANIFEST_BYTES
        or sha256_file(raw_manifest_path) != RAW_WHEELHOUSE_MANIFEST_SHA256
        or raw_manifest_path.stat().st_size != RAW_WHEELHOUSE_MANIFEST_BYTES
        or wheelhouse.get("entry_count") != RAW_WHEELHOUSE_ENTRY_COUNT
        or wheelhouse.get("entries_sha256")
        != RAW_WHEELHOUSE_ENTRIES_SHA256
        or raw_manifest.get("entries_sha256")
        != RAW_WHEELHOUSE_ENTRIES_SHA256
        or raw_manifest.get("entry_count") != RAW_WHEELHOUSE_ENTRY_COUNT
        or wheelhouse.get("network_install_allowed") is not False
        or not isinstance(wheel_rows, list)
        or wheelhouse.get("transport_inventory_sha256")
        != canonical_sha256(wheel_rows)
        or wheel_rows != raw_entries
        or _wheelhouse_inventory(wheelhouse_path) != wheel_rows
    ):
        raise ValueError("ABR runtime wheelhouse closure changed")

    accepted_path = root.joinpath(
        *PurePosixPath(ACCEPTED_LIBRARY_RELATIVE_PATH).parts
    )
    diagnostic_path = root.joinpath(
        *PurePosixPath(DIAGNOSTIC_LIBRARY_RELATIVE_PATH).parts
    )
    diagnostic_sha = str(native.get("diagnostic_library_sha256", ""))
    if (
        native.get("accepted_library_sha256")
        != ACCEPTED_CANDIDATE_LIBRARY_SHA256
        or native.get("accepted_library_bytes")
        != ACCEPTED_CANDIDATE_LIBRARY_BYTES
        or sha256_file(accepted_path) != ACCEPTED_CANDIDATE_LIBRARY_SHA256
        or accepted_path.stat().st_size != ACCEPTED_CANDIDATE_LIBRARY_BYTES
        or not _is_sha256(diagnostic_sha)
        or sha256_file(diagnostic_path) != diagnostic_sha
        or diagnostic_path.stat().st_size
        != native.get("diagnostic_library_bytes")
        or native.get("dual_engine_search_required") is not True
        or (
            expected_diagnostic_library_sha256 is not None
            and diagnostic_sha != expected_diagnostic_library_sha256
        )
    ):
        raise ValueError("ABR runtime native engine closure changed")

    required_fields = [spec.bundle_field for spec in MODEL_SPECS]
    if (
        load_receipt.get("behavior_profiles")
        != list(M31_T3_BEHAVIOR_PROFILES)
        or load_receipt.get("required_bundle_fields") != required_fields
        or load_receipt.get("loaded_bundle_fields") != required_fields
        or load_receipt.get("model_inventory_sha256")
        != manifest["model_inventory_sha256"]
        or load_receipt.get("all_required_models_deserialized") is not True
        or load_receipt.get("optional_loader_fallback_used") is not False
        or load_receipt.get("placeholder_or_synthetic_model_used") is not False
    ):
        raise ValueError("ABR runtime real-model load receipt changed")

    ready_files = ready.get("files")
    expected_ready_files = [
        *artifacts,
        _record(path, MANIFEST_NAME),
    ]
    if (
        ready.get("schema") != READY_SCHEMA
        or ready.get("status") != "ready_for_real_one_pair_local_smoke"
        or ready.get("ready_sha256") != _self_digest(ready, "ready_sha256")
        or ready.get("manifest_sha256") != manifest["manifest_sha256"]
        or ready.get("manifest_file_sha256") != sha256_file(path)
        or ready_files != expected_ready_files
        or ready.get("file_inventory_sha256")
        != canonical_sha256(expected_ready_files)
        or ready.get("file_count") != len(expected_ready_files)
        or ready.get("ready_published_after_all_payloads") is not True
        or ready.get("real_models_only") is not True
        or ready.get("cloud_execution_started") is not False
        or ready.get("current_profile_changed") is not False
    ):
        raise ValueError("ABR runtime READY boundary changed")

    actual_files = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file()
    }
    if actual_files != {*expected_paths, MANIFEST_NAME, READY_NAME}:
        raise ValueError("ABR runtime bundle contains an unbound file")
    return {
        "manifest": manifest,
        "ready": ready,
        "manifest_path": path,
        "ready_path": ready_path,
        "runtime_archive_path": runtime_path,
        "wheelhouse_archive_path": wheelhouse_path,
        "wheelhouse_source_manifest_path": raw_manifest_path,
        "accepted_library_path": accepted_path,
        "diagnostic_library_path": diagnostic_path,
    }


def resolve_bundle(
    manifest_path: str | Path,
    *,
    expected_diagnostic_library_sha256: str | None = None,
) -> dict[str, Any]:
    """Public name used by the post-training workload contract."""

    return validate_runtime_bundle(
        manifest_path,
        expected_diagnostic_library_sha256=(
            expected_diagnostic_library_sha256
        ),
    )


__all__ = [
    "ACCEPTED_CANDIDATE_LIBRARY_SHA256",
    "BUNDLE_SCHEMA",
    "MANIFEST_NAME",
    "MODEL_SPECS",
    "READY_NAME",
    "READY_SCHEMA",
    "package_runtime_bundle",
    "resolve_bundle",
    "validate_runtime_bundle",
]
