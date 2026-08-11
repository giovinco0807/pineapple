import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from ofc_regular.artifact_manifest import (
    MANIFEST_SCHEMA,
    build_run_manifest,
    sha256_file,
    write_run_manifest,
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _init_repo(path: Path) -> None:
    path.mkdir()
    _git(path, "init")
    tracked = path / "tracked.txt"
    tracked.write_text("baseline\n", encoding="utf-8")
    _git(path, "add", "tracked.txt")
    _git(
        path,
        "-c",
        "user.name=Manifest Test",
        "-c",
        "user.email=manifest@example.invalid",
        "commit",
        "-m",
        "baseline",
    )


def test_sha256_file_streams_exact_bytes(tmp_path: Path):
    artifact = tmp_path / "artifact.bin"
    payload = b"ofc\x00pineapple\r\n"
    artifact.write_bytes(payload)

    assert sha256_file(artifact) == hashlib.sha256(payload).hexdigest()


def test_manifest_fingerprints_dirty_tree_and_explicit_artifacts(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    config = repo / "config.json"
    config.write_text('{"profile":"stage18_p1"}\n', encoding="utf-8")

    manifest = build_run_manifest(
        repo_root=repo,
        run_id="m0-smoke",
        phase="correctness_smoke",
        profiles=("stage19_p0", "stage18_p1", "stage9f_p2", "stage7_m5_r10"),
        artifacts=(("config", Path("config.json")),),
        seeds={"candidate": 101, "evaluation": 202},
        seed_stride=1_000_003,
        command="python -m ofc_regular.evaluate_matchups --games 10",
        metadata={"seat_mode": "paired_swap"},
        created_at_utc="2026-07-12T00:00:00Z",
    )

    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["created_at_utc"] == "2026-07-12T00:00:00Z"
    assert manifest["run"]["profile_selection"] == "explicit_only"
    assert manifest["run"]["profiles"] == [
        "stage19_p0",
        "stage18_p1",
        "stage9f_p2",
        "stage7_m5_r10",
    ]
    assert manifest["run"]["seeds"] == {"candidate": 101, "evaluation": 202}
    assert manifest["run"]["seed_stride"] == 1_000_003
    assert manifest["repository"]["dirty"] is True
    assert len(manifest["repository"]["tracked_diff_sha256"]) == 64
    assert manifest["repository"]["untracked_file_count"] == 1
    assert manifest["repository"]["untracked_files"][0]["path"] == "config.json"
    assert manifest["artifacts"] == [
        {
            "role": "config",
            "path": "config.json",
            "repository_relative": True,
            "size_bytes": config.stat().st_size,
            "sha256": sha256_file(config),
        }
    ]
    assert "numpy" in manifest["environment"]["packages"]


def test_manifest_validation_rejects_ambiguous_inputs(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_repo(repo)

    with pytest.raises(ValueError, match="profiles must be unique"):
        build_run_manifest(
            repo_root=repo,
            run_id="duplicate-profile",
            phase="smoke",
            profiles=("stage18_p1", "stage18_p1"),
        )
    with pytest.raises(ValueError, match="seed_stride must be positive"):
        build_run_manifest(
            repo_root=repo,
            run_id="bad-stride",
            phase="smoke",
            seed_stride=0,
        )


def test_manifest_write_is_atomic_and_refuses_overwrite(tmp_path: Path):
    output = tmp_path / "nested" / "manifest.json"
    manifest = {"schema": MANIFEST_SCHEMA, "value": 1}

    write_run_manifest(output, manifest)

    assert json.loads(output.read_text(encoding="utf-8")) == manifest
    with pytest.raises(FileExistsError, match="manifest already exists"):
        write_run_manifest(output, {"schema": MANIFEST_SCHEMA, "value": 2})
    write_run_manifest(
        output, {"schema": MANIFEST_SCHEMA, "value": 2}, overwrite=True
    )
    assert json.loads(output.read_text(encoding="utf-8"))["value"] == 2
