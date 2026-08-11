"""Build a deterministic Attempt03 model-worker package without fit data.

This is the package-only Spot dry-run boundary.  It accepts the executable
training freeze and source tree, but has no argument through which a fit,
pre-calibration, calibration, or locked-holdout row can be supplied.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Sequence

from .hu_m43_attempt03_training import (
    M43_ATTEMPT03_SCIENCE_CORRECTION_PATH,
    M43_ATTEMPT03_WORKER_SOURCE_PATHS,
    load_attempt03_training_freeze,
)
from .hu_m43_pilot_contract import canonical_manifest_sha256


M43_ATTEMPT03_MODEL_SPOT_PACKAGE_SCHEMA = (
    "hu_m43_attempt03_v5_model_spot_package_only_v1"
)
_ROOT_MODULES = (
    "ofc_regular",
    "ofc_regular.train_hu_m43_attempt03_fold_job",
)


def build_attempt03_model_spot_package(
    *,
    repo_root: str | Path,
    training_freeze_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    freeze_path = Path(training_freeze_path).resolve()
    freeze = load_attempt03_training_freeze(freeze_path, repo_root=root)
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError("Attempt03 package-only output is immutable")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{os.getpid()}.package"
    if staging.exists():
        raise FileExistsError("Attempt03 package-only staging exists")
    staging.mkdir()
    try:
        selected = _discover_local_closure(root)
        archive_path = staging / "source.zip"
        module_hashes = _write_deterministic_zip(
            archive_path, selected=selected
        )
        _audit_archive(archive_path, expected_hashes=module_hashes)
        frozen_files = freeze["executable_sources"]["files"]
        for relative in M43_ATTEMPT03_WORKER_SOURCE_PATHS:
            if module_hashes.get(relative) != frozen_files[relative]:
                raise ValueError(
                    f"Attempt03 package worker source is not frozen: {relative}"
                )
        entries = tuple(sorted(module_hashes))
        manifest = {
            "schema": M43_ATTEMPT03_MODEL_SPOT_PACKAGE_SCHEMA,
            "status": "package_only_dry_run_no_fit_inputs",
            "roots": list(_ROOT_MODULES),
            "source_archive": {
                "path": "source.zip",
                "file_sha256": _file_sha256(archive_path),
                "bytes": archive_path.stat().st_size,
            },
            "source_package_policy": {
                "mode": "ast_recursive_local_import_closure_v1",
                "module_count": len(entries),
                "entries": list(entries),
                "entries_sha256": canonical_manifest_sha256(list(entries)),
                "module_sha256": module_hashes,
                "portable_entry_separators": True,
                "deterministic_zip_metadata": True,
                "post_archive_source_audit": "pass",
            },
            "training_freeze": {
                "path": str(freeze_path),
                "file_sha256": _file_sha256(freeze_path),
                "canonical_sha256": freeze["freeze_sha256"],
                "worker_files_sha256": freeze["executable_sources"][
                    "worker_files_sha256"
                ],
            },
            "parent_model_freeze_file_sha256": freeze[
                "parent_model_freeze"
            ]["file_sha256"],
            "science_correction_file_sha256": _file_sha256(
                root / M43_ATTEMPT03_SCIENCE_CORRECTION_PATH
            ),
            "frozen_worker_source_matches": True,
            "fit_input_count": 0,
            "holdout_input_count": 0,
            "row_valued_metric_used": False,
            "cloud_upload_performed": False,
            "instance_launch_performed": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
            "full_replacement": False,
        }
        manifest["manifest_sha256"] = canonical_manifest_sha256(manifest)
        _write_json_exclusive(staging / "package_manifest.json", manifest)
        os.replace(staging, destination)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _discover_local_closure(root: Path) -> dict[str, Path]:
    package_root = root / "src" / "ofc_regular"
    queue = list(_ROOT_MODULES)
    seen: set[str] = set()
    selected: dict[str, Path] = {}

    def module_path(name: str) -> Path | None:
        parts = name.split(".")
        if not parts or parts[0] != "ofc_regular":
            return None
        candidate = package_root.joinpath(*parts[1:])
        path = (
            candidate / "__init__.py"
            if candidate.is_dir()
            else candidate.with_suffix(".py")
        )
        return path if path.is_file() else None

    def add(name: str) -> None:
        if (
            name.startswith("ofc_regular")
            and name not in seen
            and name not in queue
            and module_path(name) is not None
        ):
            queue.append(name)

    while queue:
        name = queue.pop(0)
        if name in seen:
            continue
        path = module_path(name)
        if path is None:
            raise ValueError(f"Attempt03 package local module is missing: {name}")
        seen.add(name)
        relative = "src/" + path.relative_to(root / "src").as_posix()
        selected[relative] = path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        package = name if path.name == "__init__.py" else name.rsplit(".", 1)[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package.split(".")
                    if node.level > 1:
                        base = base[: -(node.level - 1)]
                    target = ".".join(
                        base + ([node.module] if node.module else [])
                    )
                    add(target)
                    if node.module is None:
                        for alias in node.names:
                            add(target + "." + alias.name)
                elif node.module:
                    add(node.module)
    if "src/ofc_regular/train_hu_m43_attempt03_fold_job.py" not in selected:
        raise AssertionError("Attempt03 package closure omitted its worker")
    return selected


def _write_deterministic_zip(
    path: Path, *, selected: dict[str, Path]
) -> dict[str, str]:
    module_hashes: dict[str, str] = {}
    with zipfile.ZipFile(
        path,
        "x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for relative in sorted(selected):
            if "\\" in relative or _forbidden_entry(relative):
                raise ValueError(
                    f"Attempt03 package contains a forbidden entry: {relative}"
                )
            payload = selected[relative].read_bytes()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED)
            module_hashes[relative] = hashlib.sha256(payload).hexdigest()
    return module_hashes


def _audit_archive(path: Path, *, expected_hashes: dict[str, str]) -> None:
    with zipfile.ZipFile(path) as archive:
        names = tuple(sorted(archive.namelist()))
        if names != tuple(sorted(expected_hashes)) or any("\\" in name for name in names):
            raise ValueError("Attempt03 package archive entry set changed")
        for name in names:
            actual = hashlib.sha256(archive.read(name)).hexdigest()
            if actual != expected_hashes[name]:
                raise ValueError(
                    f"Attempt03 package post-archive source SHA changed: {name}"
                )


def _forbidden_entry(relative: str) -> bool:
    path = Path(relative)
    token = relative.lower()
    return (
        "locked" in token
        or token.endswith((".json", ".jsonl"))
        or any(
            part.lower() in {"output", "outputs", "config", "configs", "data"}
            for part in path.parts
        )
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--training-freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_attempt03_model_spot_package(
        repo_root=args.repo_root,
        training_freeze_path=args.training_freeze,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "M43_ATTEMPT03_MODEL_SPOT_PACKAGE_SCHEMA",
    "build_attempt03_model_spot_package",
    "main",
    "parse_args",
]
