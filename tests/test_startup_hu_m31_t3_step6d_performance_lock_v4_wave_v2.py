from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from copy import deepcopy
from pathlib import Path, PurePosixPath

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2,
)
from ofc_regular import (
    hu_m31_t3_step6d_full100_wave_science_registry_v2 as registry,
)
from ofc_regular import (
    hu_m31_t3_step6d_full100_wave_package_v2 as wave_package,
)
from ofc_regular import (
    hu_m31_t3_step6d_performance_lock_v4_spot_package as package,
)


ROOT = Path(__file__).resolve().parents[1]
STARTUP = (
    ROOT
    / "scripts/startup_hu_m31_t3_step6d_performance_lock_v4_wave_v2.sh"
)
DEV_STARTUP = ROOT / "scripts/startup_hu_m31_t3_step6d_full100_wave_v2.sh"
PACKAGE_ROOT = package.DEFAULT_PACKAGE_DIR


def _source() -> str:
    return STARTUP.read_text(encoding="utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"


def _heredocs() -> list[str]:
    return re.findall(r"<<'PY'\n(.*?)\nPY", _source(), flags=re.DOTALL)


def _phase1_block() -> str:
    return next(
        block
        for block in _heredocs()
        if "scientific archive missing/extra member" in block
    )


def _phase2_block() -> str:
    return next(
        block
        for block in _heredocs()
        if "all 20 v4 job manifest records changed" in block
    )


def _build_wave(manifest: dict[str, object]) -> dict[str, object]:
    plan = json.loads(package.v4.DEFAULT_PLAN_PATH.read_text("utf-8"))
    return wave_v2.build_wave_plan(
        run_name="regular-hu-m31-c02-f100wv2-20260723-v4smoke",
        identity_salt="1234567890abcdef1234567890abcdef",
        package_sha256=str(manifest["source_sha256"]),
        image_digest=str(plan["source_identity"]["image_digest"]),
        full100_plan=plan,
    )


def _materialize_worker_root(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    content = root / "content"
    (root / "science").mkdir(parents=True)
    (root / "wheelhouse").mkdir()
    content.mkdir()
    manifest_raw = (PACKAGE_ROOT / package.MANIFEST_NAME).read_bytes()
    manifest = json.loads(manifest_raw.decode("ascii"))
    source = PACKAGE_ROOT / package.SOURCE_NAME
    shutil.copyfile(source, content / "source.zip")
    (content / "scientific_manifest.json").write_bytes(manifest_raw)
    wave = _build_wave(manifest)
    (content / "wave_plan.json").write_bytes(wave_v2.canonical_bytes(wave))
    first = manifest["job_manifests"][0]
    shutil.copyfile(
        PACKAGE_ROOT / first["path"], content / "job_manifest.json"
    )

    wheel_name = "offline_smoke-1.0-py3-none-any.whl"
    wheel_raw = b"offline-wheel-smoke"
    with zipfile.ZipFile(
        content / "wheelhouse.zip", "w", compression=zipfile.ZIP_STORED
    ) as archive:
        archive.writestr(wheel_name, wheel_raw)
    wheel_entries = [
        {
            "filename": wheel_name,
            "sha256": _sha(wheel_raw),
            "bytes": len(wheel_raw),
            "distribution": "offline-smoke",
            "version": "1.0",
            "tags": ["py3-none-any"],
        }
    ]
    wheel_manifest = {
        "schema": "hu_m31_t3_step6d_perfdev_v2_wheelhouse_v1",
        "status": "complete_hash_pinned_offline_wheelhouse",
        "requirements_sha256": "0" * 64,
        "python_abi": "cp311",
        "target_os": "linux",
        "target_architecture": "x86_64",
        "network_install_allowed": False,
        "entries": wheel_entries,
        "entry_count": 1,
        "entries_sha256": _sha(_canonical(wheel_entries)),
    }
    (content / "wheelhouse_manifest.json").write_bytes(
        _canonical(wheel_manifest)
    )
    args = [
        str(root),
        str(manifest["source_sha256"]),
        str(manifest["source_bytes"]),
        _sha(manifest_raw),
        _sha((content / "wave_plan.json").read_bytes()),
        _sha((content / "job_manifest.json").read_bytes()),
        _sha((content / "wheelhouse.zip").read_bytes()),
        _sha((content / "wheelhouse_manifest.json").read_bytes()),
    ]
    completed = subprocess.run(
        [sys.executable, "-c", _phase1_block(), *args],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return manifest, wave


def _run_phase2(
    root: Path,
    manifest: dict[str, object],
    *,
    record_index: int,
    bootstrap_role: str | None = None,
    selected_sha: str | None = None,
) -> subprocess.CompletedProcess[str]:
    record = manifest["job_manifests"][record_index]
    job_id = str(record["job_id"])
    role = str(record["source_role"])
    job_path = PACKAGE_ROOT / str(record["path"])
    shutil.copyfile(job_path, root / "content/job_manifest.json")
    bootstrap = {
        "job_id": job_id,
        "source_role": role if bootstrap_role is None else bootstrap_role,
    }
    bootstrap_path = root / "bootstrap.json"
    bootstrap_path.write_bytes(_canonical(bootstrap))
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "science/src")
    return subprocess.run(
        [
            sys.executable,
            "-c",
            _phase2_block(),
            str(root),
            str(bootstrap_path),
            str(record["sha256"] if selected_sha is None else selected_sha),
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
        env=environment,
    )


def test_dev_bytes_unchanged_and_v4_startup_is_frozen_valid_bash() -> None:
    assert DEV_STARTUP.stat().st_size == 37217
    assert _sha(DEV_STARTUP.read_bytes()) == registry.DEVELOPMENT_STARTUP_SHA256
    assert STARTUP.stat().st_size == 35205
    assert _sha(STARTUP.read_bytes()) == (
        registry.PERFORMANCE_LOCK_V4_STARTUP_SHA256
    )
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is unavailable")
    syntax = subprocess.run(
        [
            bash,
            "-n",
            "scripts/startup_hu_m31_t3_step6d_performance_lock_v4_wave_v2.sh",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        cwd=ROOT,
    )
    assert syntax.returncode == 0, syntax.stderr
    for block in _heredocs():
        compile(block, str(STARTUP), "exec")
    source = _source()
    assert "RAYON_NUM_THREADS=16" in source
    assert "OFC_HU_M3_BATCH_THREADS=1" in source
    assert source.count(
        "python -m ofc_regular.run_hu_m31_t3_step6d_performance_v2"
    ) == 2
    assert "ifGenerationMatch=0" in source
    assert "complete_validated_single_job_attempt" in source
    assert (
        "startup_hu_m31_t3_step6d_performance_lock_v4_wave_v2.sh"
        in source
    )


def test_worker_replays_v4_archive_and_all_20_job_manifests(
    tmp_path: Path,
) -> None:
    root = tmp_path / "worker"
    manifest, _ = _materialize_worker_root(root)
    # One real worker replay covers package schema, plan, claim,
    # materialization, seal, all 100 roots, all 20 job records and binaries.
    completed = _run_phase2(root, manifest, record_index=0)
    assert completed.returncode == 0, completed.stderr
    assert len(completed.stdout.splitlines()) == 8

    # Exercise the selected-job binding for all 20 manifests without
    # repeating the intentionally expensive 100-root replay 20 times.
    plan = package.v4.validate_performance_lock_v4_plan(
        json.loads(package.v4.DEFAULT_PLAN_PATH.read_text("utf-8"))
    )
    expected_payloads, expected_records = package._job_payloads(plan)
    assert manifest["job_manifests"] == expected_records
    assert len(expected_records) == 20
    for record in expected_records:
        raw = (PACKAGE_ROOT / record["path"]).read_bytes()
        frozen = {
            row["job_id"]: row for row in plan["jobs"]
        }[record["job_id"]]
        assert raw == expected_payloads[record["path"]]
        assert _sha(raw) == record["sha256"]
        assert record["sha256"] == frozen["shard_manifest_sha256"]
        assert record["source_role"] == frozen["source_role"]
        assert record["work_hand_indices"] == frozen["work_hand_indices"]


def test_worker_rejects_mixed_schema_and_selected_hash_tamper(
    tmp_path: Path,
) -> None:
    root = tmp_path / "worker"
    manifest, _ = _materialize_worker_root(root)
    manifest_path = root / "content/scientific_manifest.json"
    mixed = deepcopy(manifest)
    mixed["schema"] = registry.DEVELOPMENT_PACKAGE_SCHEMA
    manifest_path.write_bytes(_canonical(mixed))
    rejected_schema = _run_phase2(root, mixed, record_index=0)
    assert rejected_schema.returncode != 0
    manifest_path.write_bytes(
        (PACKAGE_ROOT / package.MANIFEST_NAME).read_bytes()
    )
    rejected_hash = _run_phase2(
        root, manifest, record_index=0, selected_sha="f" * 64
    )
    assert rejected_hash.returncode != 0
    assert "selected v4 job/wave binding changed" in rejected_hash.stderr


_STARTUP_LINEAGES = sorted(
    registry.startup_relative_paths_by_sha256().items(), key=lambda row: row[1]
)


def test_every_registered_startup_lineage_is_covered() -> None:
    """Guard the parametrisation below against an empty or shrunken registry."""

    relatives = {relative for _, relative in _STARTUP_LINEAGES}
    assert len(_STARTUP_LINEAGES) >= 2
    assert len(relatives) == len(_STARTUP_LINEAGES)
    assert str(STARTUP.relative_to(ROOT)).replace("\\", "/") in relatives
    assert str(DEV_STARTUP.relative_to(ROOT)).replace("\\", "/") in relatives


@pytest.mark.parametrize(
    ("startup_sha256", "relative"),
    _STARTUP_LINEAGES,
    ids=[PurePosixPath(relative).stem for _, relative in _STARTUP_LINEAGES],
)
def test_staged_startup_name_matches_the_name_the_script_verifies(
    startup_sha256: str, relative: str
) -> None:
    """The stager must place a startup script under the name it checks for.

    Each startup script re-derives its own staged object name and aborts with
    "bootstrap object topology changed" if it differs, so a lineage staged
    under another lineage's name fails on every VM before doing any work.
    Driving this from the registry means a newly registered lineage is covered
    without anyone remembering to extend a list here.
    """

    script = ROOT / relative
    assert _sha(script.read_bytes()) == startup_sha256
    staged = wave_package.startup_content_path(startup_sha256)
    assert staged == f"content/startup/{script.name}"
    assert f'"/{staged}"' in _collapse_literals(
        script.read_text(encoding="utf-8")
    )


def _collapse_literals(source: str) -> str:
    """Join adjacent Python string literals so split suffixes are comparable."""

    return re.sub(r'"\s*\n\s*"', "", source)
