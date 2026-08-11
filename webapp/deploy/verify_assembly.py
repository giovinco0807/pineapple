#!/usr/bin/env python3
"""Seal build-produced artifacts and verify the immutable runtime assembly."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

BUILD_TIME = "BUILD_TIME"


class AssemblyError(RuntimeError):
    """Raised when the runtime assembly is malformed or fails integrity checks."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_assembly_sha(payload: dict[str, Any]) -> str:
    canonical_payload = dict(payload)
    canonical_payload["assembly_sha256"] = BUILD_TIME
    canonical = json.dumps(
        canonical_payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def artifact_bindings(value: Any, pointer: str = "$") -> Iterator[tuple[str, dict[str, Any]]]:
    if isinstance(value, dict):
        if "path" in value or "sha256" in value:
            if not isinstance(value.get("path"), str) or not isinstance(
                value.get("sha256"), str
            ):
                raise AssemblyError(f"{pointer} must contain string path and sha256")
            yield pointer, value
        for key, child in value.items():
            yield from artifact_bindings(child, f"{pointer}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from artifact_bindings(child, f"{pointer}[{index}]")


def resolve_artifact(root: Path, relative_path: str, pointer: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise AssemblyError(f"{pointer}.path must be relative to the assembly root")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise AssemblyError(f"{pointer}.path escapes the assembly root") from exc
    if not resolved.is_file():
        raise AssemblyError(f"{pointer}.path is missing: {resolved}")
    return resolved


def load_assembly(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssemblyError(f"cannot read assembly {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AssemblyError("assembly root must be an object")
    if payload.get("schema") != "ofc_webapp_assembly_v1":
        raise AssemblyError("unsupported assembly schema")
    return payload


def seal(assembly_path: Path, root: Path) -> str:
    payload = load_assembly(assembly_path)
    target_mode = (
        assembly_path.stat().st_mode & 0o777 if assembly_path.exists() else 0o644
    )
    for pointer, binding in artifact_bindings(payload):
        if binding["sha256"] != BUILD_TIME:
            continue
        artifact = resolve_artifact(root, binding["path"], pointer)
        binding["sha256"] = sha256_file(artifact)
    unresolved = [
        pointer
        for pointer, binding in artifact_bindings(payload)
        if binding["sha256"] == BUILD_TIME
    ]
    if unresolved:
        raise AssemblyError(
            "unresolved build-time digests: " + ", ".join(sorted(unresolved))
        )
    payload["assembly_sha256"] = canonical_assembly_sha(payload)
    assembly_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{assembly_path.name}.", suffix=".tmp", dir=assembly_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
            handle.write("\n")
        os.chmod(temporary_name, target_mode)
        os.replace(temporary_name, assembly_path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return payload["assembly_sha256"]


def verify(assembly_path: Path, root: Path) -> str:
    payload = load_assembly(assembly_path)
    expected_assembly_sha = payload.get("assembly_sha256")
    if not isinstance(expected_assembly_sha, str) or expected_assembly_sha == BUILD_TIME:
        raise AssemblyError("assembly_sha256 was not sealed at image build time")
    actual_assembly_sha = canonical_assembly_sha(payload)
    if actual_assembly_sha != expected_assembly_sha:
        raise AssemblyError(
            "assembly digest mismatch: "
            f"expected {expected_assembly_sha}, got {actual_assembly_sha}"
        )
    failures: list[str] = []
    for pointer, binding in artifact_bindings(payload):
        expected = binding["sha256"].lower()
        if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
            failures.append(f"{pointer}.sha256 is not a lowercase SHA-256 digest")
            continue
        try:
            artifact = resolve_artifact(root, binding["path"], pointer)
        except AssemblyError as exc:
            failures.append(str(exc))
            continue
        actual = sha256_file(artifact)
        if actual != expected:
            failures.append(
                f"{pointer} digest mismatch for {binding['path']}: "
                f"expected {expected}, got {actual}"
            )
    if failures:
        raise AssemblyError("\n".join(failures))
    return expected_assembly_sha


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assembly",
        type=Path,
        default=Path(os.environ.get("OFC_ASSEMBLY_PATH", "/app/assembly.json")),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("OFC_ASSEMBLY_ROOT", "/app")),
    )
    parser.add_argument("--seal-build-artifacts", action="store_true")
    args = parser.parse_args()

    assembly_path = args.assembly.resolve()
    root = args.root.resolve()
    try:
        digest = (
            seal(assembly_path, root)
            if args.seal_build_artifacts
            else verify(assembly_path, root)
        )
    except AssemblyError as exc:
        print(f"assembly verification failed: {exc}", file=os.sys.stderr)
        return 1
    action = "sealed" if args.seal_build_artifacts else "verified"
    print(f"assembly {action}: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
