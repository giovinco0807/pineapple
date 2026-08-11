"""Build reproducibility manifests for HU search, training, and evaluation runs.

The manifest deliberately records only profiles and artifacts supplied by the
caller.  It never reads or changes the runtime ``current`` profile.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


MANIFEST_SCHEMA = "ofc_regular_run_manifest_v1"
_HASH_CHUNK_BYTES = 1024 * 1024
_REPRODUCIBILITY_PACKAGES = (
    "numpy",
    "scikit-learn",
    "torch",
    "xgboost",
)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of *path* without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def discover_repo_root(start: str | Path = ".") -> Path:
    completed = _run_git(Path(start).resolve(), "rev-parse", "--show-toplevel")
    return Path(_decode_line(completed.stdout)).resolve()


def collect_git_state(repo_root: str | Path) -> dict[str, Any]:
    """Fingerprint the exact tracked diff and every non-ignored untracked file."""
    root = Path(repo_root).resolve()
    head = _decode_line(_run_git(root, "rev-parse", "HEAD").stdout)
    branch_result = _run_git(
        root, "symbolic-ref", "--quiet", "--short", "HEAD", check=False
    )
    branch = (
        _decode_line(branch_result.stdout) if branch_result.returncode == 0 else None
    )
    upstream_result = _run_git(
        root,
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
        check=False,
    )
    upstream = (
        _decode_line(upstream_result.stdout)
        if upstream_result.returncode == 0
        else None
    )
    upstream_commit = None
    if upstream is not None:
        upstream_commit = _decode_line(
            _run_git(root, "rev-parse", upstream).stdout
        )

    status = _run_git(
        root, "status", "--porcelain=v1", "-z", "--untracked-files=all"
    ).stdout
    tracked_diff = _run_git(
        root, "diff", "--no-ext-diff", "--binary", "HEAD", "--"
    ).stdout
    untracked_raw = _run_git(
        root, "ls-files", "--others", "--exclude-standard", "-z"
    ).stdout
    untracked_paths = sorted(
        entry.decode("utf-8", errors="surrogateescape")
        for entry in untracked_raw.split(b"\0")
        if entry
    )
    untracked_files = [_file_record(root / path, root) for path in untracked_paths]

    return {
        "head_commit": head,
        "branch": branch,
        "upstream": upstream,
        "upstream_commit": upstream_commit,
        "dirty": bool(status),
        "status_porcelain_sha256": _sha256_bytes(status),
        "tracked_diff_sha256": _sha256_bytes(tracked_diff),
        "untracked_file_count": len(untracked_files),
        "untracked_files": untracked_files,
    }


def build_run_manifest(
    *,
    repo_root: str | Path,
    run_id: str,
    phase: str,
    profiles: Sequence[str] = (),
    artifacts: Sequence[tuple[str, str | Path]] = (),
    seeds: dict[str, int] | None = None,
    seed_stride: int | None = None,
    command: str | None = None,
    metadata: dict[str, str] | None = None,
    notes: Sequence[str] = (),
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build a manifest without mutating repository or profile state."""
    root = Path(repo_root).resolve()
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if not phase.strip():
        raise ValueError("phase must not be empty")
    if seed_stride is not None and seed_stride <= 0:
        raise ValueError("seed_stride must be positive")
    if len(set(profiles)) != len(profiles):
        raise ValueError("profiles must be unique and ordered")

    artifact_records = [
        {"role": role, **_file_record(_resolve_path(path, root), root)}
        for role, path in artifacts
    ]
    if len({(row["role"], row["path"]) for row in artifact_records}) != len(
        artifact_records
    ):
        raise ValueError("duplicate artifact role/path pair")

    timestamp = created_at_utc or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    return {
        "schema": MANIFEST_SCHEMA,
        "created_at_utc": timestamp,
        "run": {
            "run_id": run_id,
            "phase": phase,
            "command": command,
            "profiles": list(profiles),
            "profile_selection": "explicit_only",
            "seeds": dict(sorted((seeds or {}).items())),
            "seed_stride": seed_stride,
            "metadata": dict(sorted((metadata or {}).items())),
            "notes": list(notes),
        },
        "repository": {
            "root": str(root),
            **collect_git_state(root),
        },
        "artifacts": artifact_records,
        "environment": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
    }


def write_run_manifest(
    path: str | Path, manifest: dict[str, Any], *, overwrite: bool = False
) -> None:
    """Atomically write *manifest*, refusing accidental replacement by default."""
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"manifest already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help="Artifact to hash; repeat for configs, models, inputs, and binaries.",
    )
    parser.add_argument("--seed", action="append", default=[], metavar="NAME=INT")
    parser.add_argument("--seed-stride", type=int)
    parser.add_argument("--command")
    parser.add_argument("--meta", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.repo_root.resolve() if args.repo_root else discover_repo_root()
    manifest = build_run_manifest(
        repo_root=root,
        run_id=args.run_id,
        phase=args.phase,
        profiles=args.profile,
        artifacts=[_parse_pair(value, "artifact") for value in args.artifact],
        seeds=_pairs_to_dict(
            (
                (key, _parse_int(value, f"seed {key}"))
                for key, value in (
                    _parse_pair(raw, "seed") for raw in args.seed
                )
            ),
            "seed",
        ),
        seed_stride=args.seed_stride,
        command=args.command,
        metadata=_pairs_to_dict(
            (_parse_pair(raw, "meta") for raw in args.meta), "meta"
        ),
        notes=args.note,
    )
    write_run_manifest(args.output, manifest, overwrite=args.force)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "schema": MANIFEST_SCHEMA,
                "head_commit": manifest["repository"]["head_commit"],
                "dirty": manifest["repository"]["dirty"],
                "artifacts": len(manifest["artifacts"]),
            },
            sort_keys=True,
        )
    )


def _file_record(path: Path, repo_root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"artifact file does not exist: {path}")
    resolved = path.resolve()
    try:
        display_path = resolved.relative_to(repo_root).as_posix()
        repository_relative = True
    except ValueError:
        display_path = str(resolved)
        repository_relative = False
    return {
        "path": display_path,
        "repository_relative": repository_relative,
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _resolve_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def _parse_pair(raw: str, label: str) -> tuple[str, str]:
    key, separator, value = raw.partition("=")
    if not separator or not key.strip() or not value:
        raise ValueError(f"--{label} must use NAME=VALUE format: {raw!r}")
    return key.strip(), value


def _parse_int(raw: str, label: str) -> int:
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer: {raw!r}") from exc


def _pairs_to_dict(
    pairs: Iterable[tuple[str, Any]], label: str
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate --{label} name: {key}")
        result[key] = value
    return result


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in _REPRODUCIBILITY_PACKAGES:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _run_git(
    repo_root: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and completed.returncode != 0:
        error = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {error}")
    return completed


def _decode_line(payload: bytes) -> str:
    return payload.decode("utf-8", errors="replace").strip()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    main()
