"""Build the immutable R0a-v2 allowlist from the frozen v1 baseline.

The v2 closure mirrors the real rearm2 package boundary: every Python file
copied by ``hu_m31_t3_step6d_spot_v2._copy_source_tree``, its fixed configs,
models and binaries, the parity/startup inputs, canonical lifecycle controls,
all three prior root trees, and focused lifecycle tests.  It deliberately
excludes the non-executable roadmap and never opens a rearm2 claim or root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_SPEC = REPO_ROOT / "configs/hu_joint_policy_r0_snapshot_v1.json"
DEFAULT_OUTPUT = REPO_ROOT / "configs/hu_joint_policy_r0_snapshot_v2.json"
ROADMAP_SOURCE = "docs/hu_joint_policy_full_hand_rl_milestones_20260718.md"
MISSING_R0B_ROLE = "m30_windows_exact_runtime"

REPO_CONTROL_SOURCES = (
    "configs/hu_m43_attempt08_runtime_requirements.txt",
    "scripts/startup_hu_m31_t3_step6d_full100_v1.sh",
    "scripts/verify_hu_m31_t3_feature_encoder_platform_parity.py",
    "scripts/build_hu_joint_policy_r0_v2_spec.py",
    "rust/ofc_stage3_feature_encoder/src/lib.rs",
    "outputs/hu_joint_policy/m31_t3_step6d/GLOBAL_PERFORMANCE_LOCK_CLAIM.json",
    (
        "outputs/hu_joint_policy/m31_t3_step6d/"
        "GLOBAL_PERFORMANCE_LOCK_REARM1_CLAIM.json"
    ),
    (
        "outputs/hu_joint_policy/m31_t3_step6d/"
        "performance_lock/precontent_plan_v1.json"
    ),
    (
        "outputs/hu_joint_policy/m31_t3_step6d/"
        "performance_lock_rearm1/performance_lock_v1_startup_failure_closeout.json"
    ),
    (
        "outputs/hu_joint_policy/m31_t3_step6d/"
        "performance_lock_rearm1/precontent_plan_v1.json"
    ),
    (
        "outputs/hu_joint_policy/m31_t3_step6d/"
        "performance_lock_rearm2/"
        "performance_lock_rearm1_startup_failure_closeout.json"
    ),
    (
        "outputs/hu_joint_policy/m31_t3_step6d/"
        "performance_lock_rearm2/precontent_plan_v1.json"
    ),
)
DEVELOPMENT_ROOT = (
    REPO_ROOT
    / "outputs/hu_joint_policy/m31_t3_step6d/"
    "candidate02_development/tail_reselection_v2/roots"
)
OLD_V1_ROOT = Path(
    "D:/ofc-gcp-runs/"
    "regular-hu-m31-c02-performance-lock-20260717-001/roots-open"
)
REARM1_ROOT = Path(
    "D:/ofc-gcp-runs/"
    "regular-hu-m31-c02-performance-lock-rearm1-20260717-001/roots-open"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_pretty(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _role(kind: str, scope: str, source: str) -> str:
    identity = f"{scope}\0{source}".encode("utf-8")
    return f"r0v2_{kind}_{hashlib.sha256(identity).hexdigest()[:20]}"


def _kind_and_archive(source: str) -> tuple[str, str]:
    if source.startswith("src/") or source.startswith("rust/"):
        return "source", f"source/{source}"
    if source.startswith("tests/"):
        return "test", f"test/{source}"
    if source.startswith("scripts/"):
        return "source", f"source/{source}"
    if source.startswith("configs/"):
        return "config", f"config/{source}"
    if source.startswith("models/"):
        return "model", f"model/{source}"
    if source.startswith("outputs/"):
        relative = source.removeprefix("outputs/hu_joint_policy/")
        return "artifact", f"artifact/hu_joint_policy/{relative}"
    raise ValueError(f"unclassified R0a-v2 source: {source}")


def _iter_lifecycle_tests() -> Iterable[Path]:
    tests = REPO_ROOT / "tests"
    for path in sorted(tests.glob("test_hu_m31_t3_step6d*.py")):
        yield path
    for path in sorted(
        tests.glob("test_verify_hu_m31_t3_feature_encoder_platform_parity*.py")
    ):
        yield path
    for path in sorted(
        tests.glob("test_close_hu_m31_t3_step6d_performance_lock*.py")
    ):
        yield path
    yield tests / "test_freeze_hu_joint_policy_r0_v2.py"


def _tree_files(root: Path) -> list[Path]:
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"R0a-v2 root tree is missing or unsafe: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if any(path.is_symlink() for path in files):
        raise ValueError(f"R0a-v2 root tree contains a symlink: {root}")
    return files


def build_spec(base_spec: Path = DEFAULT_BASE_SPEC) -> dict[str, Any]:
    value = json.loads(base_spec.read_bytes())
    entries = [
        dict(entry)
        for entry in value["entries"]
        if entry["source"] != ROADMAP_SOURCE
    ]
    source_keys = {
        (entry["source_scope"], os.path.normcase(entry["source"]))
        for entry in entries
    }
    archive_paths = {entry["archive_path"] for entry in entries}
    roles = {entry["role"] for entry in entries}

    def add_repo(source: str) -> None:
        key = ("repo", os.path.normcase(source))
        if key in source_keys:
            return
        path = REPO_ROOT.joinpath(*source.split("/"))
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"R0a-v2 repo input is missing: {source}")
        kind, archive = _kind_and_archive(source)
        role = _role(kind, "repo", source)
        if role in roles or archive in archive_paths:
            raise ValueError(f"R0a-v2 repo input collides: {source}")
        entries.append(
            {
                "role": role,
                "kind": kind,
                "source_scope": "repo",
                "source": source,
                "archive_path": archive,
                "sha256": _sha256(path),
            }
        )
        source_keys.add(key)
        archive_paths.add(archive)
        roles.add(role)

    def add_external(path: Path, *, lineage: str, relative: str) -> None:
        source = path.resolve().as_posix()
        key = ("external", os.path.normcase(source))
        if key in source_keys:
            return
        archive = f"artifact/roots/{lineage}/{relative}"
        role = _role("artifact", "external", source)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"R0a-v2 external input is missing: {path}")
        if role in roles or archive in archive_paths:
            raise ValueError(f"R0a-v2 external input collides: {path}")
        entries.append(
            {
                "role": role,
                "kind": "artifact",
                "source_scope": "external",
                "source": source,
                "archive_path": archive,
                "sha256": _sha256(path),
            }
        )
        source_keys.add(key)
        archive_paths.add(archive)
        roles.add(role)

    for path in sorted((REPO_ROOT / "src/ofc_regular").rglob("*.py")):
        add_repo(path.relative_to(REPO_ROOT).as_posix())
    for source in REPO_CONTROL_SOURCES:
        add_repo(source)
    for path in _iter_lifecycle_tests():
        add_repo(path.relative_to(REPO_ROOT).as_posix())
    for path in _tree_files(DEVELOPMENT_ROOT):
        add_repo(path.relative_to(REPO_ROOT).as_posix())
    for root, lineage in ((OLD_V1_ROOT, "old_v1"), (REARM1_ROOT, "rearm1")):
        for path in _tree_files(root):
            add_external(
                path,
                lineage=lineage,
                relative=path.relative_to(root).as_posix(),
            )

    value["snapshot_id"] = "hu-joint-policy-r0-snapshot-v2"
    value["gates"] = {
        "default_gate_id": "r0a_t3_research_snapshot_v2",
        "definitions": [
            {
                "gate_id": "r0a_t3_research_snapshot_v2",
                "purpose": "reproducible_rearm2_preclaim_research_snapshot",
                "excluded_roles": [MISSING_R0B_ROLE],
            },
            {
                "gate_id": "r0b_exact_t4_release_runtime",
                "purpose": "exact_t4_release_runtime_recovery",
                "excluded_roles": [],
            },
        ],
    }
    value["entries"] = entries
    value["notes"] = [
        (
            "R0a-v2 is provenance only and does not authorize a rearm2 claim, "
            "root materialization, package, cloud, current, training, quality, "
            "promotion, or runtime activation."
        ),
        (
            "The allowlist covers every src/ofc_regular Python byte copied by "
            "the real package builder, fixed configs/models/binaries, startup "
            "and parity inputs, canonical lifecycle controls, focused tests, "
            "and all development, old-v1, and rearm1 root-tree files."
        ),
        (
            "The non-executable milestone roadmap is intentionally excluded "
            "to avoid circular status pinning."
        ),
        (
            "R0b remains deferred because the exact accepted historical M3.0 "
            "Windows release DLL is absent; the available rebuild failed its "
            "locked second-seat p99 latency gate and is not a substitute."
        ),
    ]
    return value


def write_spec(output: Path, value: dict[str, Any]) -> None:
    target = output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_pretty(value)
    with target.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-spec", type=Path, default=DEFAULT_BASE_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    value = build_spec(args.base_spec)
    write_spec(args.output, value)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "entry_count": len(value["entries"]),
                "spec_sha256": hashlib.sha256(
                    _canonical_pretty(value)
                ).hexdigest(),
                "roadmap_included": any(
                    entry["source"] == ROADMAP_SOURCE
                    for entry in value["entries"]
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
